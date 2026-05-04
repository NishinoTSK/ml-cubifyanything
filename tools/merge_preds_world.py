import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dhash64(img: Image.Image) -> int:
    """
    Difference hash (dHash) 64-bit.
    Good enough for matching the *same* object crop across views.
    """
    g = img.convert("L").resize((9, 8), resample=Image.BILINEAR)
    px = np.asarray(g, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]  # 8x8 bool
    bits = diff.flatten().astype(np.uint8)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def clamp_bbox_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(float(x1), w - 1.0))
    y1 = max(0.0, min(float(y1), h - 1.0))
    x2 = max(0.0, min(float(x2), w - 1.0))
    y2 = max(0.0, min(float(y2), h - 1.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def crop_from_det(image: Image.Image, det: dict) -> Optional[Image.Image]:
    bbox = det.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    w, h = image.size
    x1, y1, x2, y2 = clamp_bbox_xyxy(x1, y1, x2, y2, w, h)
    # reject too small crops
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None
    return image.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))


@dataclass
class FramePred:
    json_path: Path
    image_path: Path
    image: Image.Image
    detections: List[dict]
    centers_cam: np.ndarray  # Nx3
    dims_cam: np.ndarray  # Nx3 (l,h,w)
    R_cam: np.ndarray  # Nx3x3
    scores: np.ndarray  # N


def resolve_image_path(json_path: Path, src_img: str) -> Path:
    """
    infer_image.py saves source_image as passed to --image (e.g. 'teste/13.jpeg').
    If the JSON lives in teste/13_inf.json, joining parent + 'teste/13.jpeg' wrongly
    becomes teste/teste/13.jpeg. Try several sensible locations.
    """
    p = Path(src_img)
    if p.is_absolute():
        return p.resolve()

    candidates = [
        json_path.parent / p.name,
        json_path.parent / p,
        Path.cwd() / p,
        p,
    ]
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except OSError:
            continue
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Image referenced by JSON not found. Tried: {tried}")


def read_frame(json_path: Path, score_thresh: float) -> FramePred:
    js = _load_json(json_path)
    src_img = js.get("source_image")
    if not src_img:
        raise ValueError(f"{json_path} missing 'source_image'")
    image_path = resolve_image_path(json_path, src_img)

    image = Image.open(str(image_path)).convert("RGB")
    dets_all = js.get("detections") or []
    # filter by score, keeping original index mapping
    keep = []
    for idx, d in enumerate(dets_all):
        s = float(d.get("score", 0.0))
        if s >= score_thresh:
            keep.append((idx, d, s))

    if "boxes_3d" not in js:
        raise ValueError(f"{json_path} missing 'boxes_3d' (need 3D boxes for merge)")
    b3 = js["boxes_3d"]
    centers_all = np.asarray(b3["gravity_center_xyz"], dtype=np.float32)
    dims_all = np.asarray(b3["dims_lhw"], dtype=np.float32)
    R_all = np.asarray(b3["R_3x3"], dtype=np.float32)

    idxs = [i for i, _, _ in keep]
    dets = [d for _, d, _ in keep]
    scores = np.asarray([s for _, _, s in keep], dtype=np.float32)

    centers = centers_all[idxs]
    dims = dims_all[idxs]
    Rmats = R_all[idxs]

    return FramePred(
        json_path=json_path,
        image_path=image_path,
        image=image,
        detections=dets,
        centers_cam=centers,
        dims_cam=dims,
        R_cam=Rmats,
        scores=scores,
    )


def build_hashes(frame: FramePred) -> List[Optional[int]]:
    hashes: List[Optional[int]] = []
    for det in frame.detections:
        crop = crop_from_det(frame.image, det)
        if crop is None:
            hashes.append(None)
            continue
        hashes.append(dhash64(crop))
    return hashes


def mutual_best_matches(hA: List[Optional[int]], hB: List[Optional[int]], max_hamming: int) -> List[Tuple[int, int, int]]:
    # returns list of (iA, iB, dist)
    best_B_for_A: Dict[int, Tuple[int, int]] = {}
    for i, ha in enumerate(hA):
        if ha is None:
            continue
        best = None
        for j, hb in enumerate(hB):
            if hb is None:
                continue
            d = _hamming(ha, hb)
            if best is None or d < best[1]:
                best = (j, d)
        if best is not None and best[1] <= max_hamming:
            best_B_for_A[i] = best

    best_A_for_B: Dict[int, Tuple[int, int]] = {}
    for j, hb in enumerate(hB):
        if hb is None:
            continue
        best = None
        for i, ha in enumerate(hA):
            if ha is None:
                continue
            d = _hamming(hb, ha)
            if best is None or d < best[1]:
                best = (i, d)
        if best is not None and best[1] <= max_hamming:
            best_A_for_B[j] = best

    matches = []
    for iA, (jB, d1) in best_B_for_A.items():
        if jB in best_A_for_B and best_A_for_B[jB][0] == iA:
            matches.append((iA, jB, d1))
    matches.sort(key=lambda x: x[2])
    return matches


def _bbox_center(det: dict) -> Optional[Tuple[float, float]]:
    b = det.get("bbox_xyxy")
    if not b or len(b) != 4:
        return None
    x1, y1, x2, y2 = map(float, b)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def orb_homography_matches(
    frameA: FramePred,
    frameB: FramePred,
    max_px: float = 80.0,
    min_matches: int = 4,
) -> List[Tuple[int, int, float]]:
    """
    Fallback matcher when crop-hash fails:
    - Estimate homography between full images using ORB.
    - Warp bbox centers A -> B and match nearest bbox centers in B.
    Returns (iA, iB, pixel_dist).
    """
    if cv2 is None:
        return []

    imgA = np.asarray(frameA.image.convert("L"))
    imgB = np.asarray(frameB.image.convert("L"))

    # ORB features
    orb = cv2.ORB_create(2000)
    kA, dA = orb.detectAndCompute(imgA, None)
    kB, dB = orb.detectAndCompute(imgB, None)
    if dA is None or dB is None or len(kA) < 10 or len(kB) < 10:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    raw = bf.match(dA, dB)
    if len(raw) < 20:
        return []
    raw.sort(key=lambda m: m.distance)
    raw = raw[:500]

    ptsA = np.float32([kA[m.queryIdx].pt for m in raw])
    ptsB = np.float32([kB[m.trainIdx].pt for m in raw])

    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
    if H is None:
        return []

    centersA = []
    idxA = []
    for i, det in enumerate(frameA.detections):
        c = _bbox_center(det)
        if c is None:
            continue
        centersA.append(c)
        idxA.append(i)
    centersB = []
    idxB = []
    for j, det in enumerate(frameB.detections):
        c = _bbox_center(det)
        if c is None:
            continue
        centersB.append(c)
        idxB.append(j)
    if len(centersA) == 0 or len(centersB) == 0:
        return []

    centersA = np.asarray(centersA, dtype=np.float32)
    centersB = np.asarray(centersB, dtype=np.float32)

    # Warp A centers to B
    ones = np.ones((centersA.shape[0], 1), dtype=np.float32)
    p = np.concatenate([centersA, ones], axis=1)  # Nx3
    pw = (p @ H.T)
    pw = pw[:, :2] / (pw[:, 2:3] + 1e-6)

    # Nearest neighbor match with mutual best constraint.
    best_for_A: Dict[int, Tuple[int, float]] = {}
    for a_i in range(pw.shape[0]):
        dists = np.linalg.norm(centersB - pw[a_i][None, :], axis=1)
        b_j = int(dists.argmin())
        dist = float(dists[b_j])
        if dist <= max_px:
            best_for_A[a_i] = (b_j, dist)

    best_for_B: Dict[int, Tuple[int, float]] = {}
    for b_j in range(centersB.shape[0]):
        dists = np.linalg.norm(pw - centersB[b_j][None, :], axis=1)
        a_i = int(dists.argmin())
        dist = float(dists[a_i])
        if dist <= max_px:
            best_for_B[b_j] = (a_i, dist)

    out = []
    for a_i, (b_j, dist) in best_for_A.items():
        if b_j in best_for_B and best_for_B[b_j][0] == a_i:
            out.append((idxA[a_i], idxB[b_j], dist))
    out.sort(key=lambda x: x[2])
    if len(out) < min_matches:
        return []
    return out


@dataclass
class Similarity:
    s: float
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3


def umeyama_similarity(X: np.ndarray, Y: np.ndarray, allow_scale: bool = True) -> Similarity:
    """
    Find similarity transform mapping X -> Y:
        y ~= s * R * x + t
    X, Y: Nx3
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    cov = (Yc.T @ Xc) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float32)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = (U @ S @ Vt).astype(np.float32)
    if allow_scale:
        var_x = (Xc * Xc).sum() / float(n)
        scale = float((D * np.diag(S)).sum() / (var_x + 1e-8))
    else:
        scale = 1.0
    t = (mu_y - scale * (R @ mu_x)).astype(np.float32)
    return Similarity(s=scale, R=R, t=t)


def apply_sim(sim: Similarity, pts: np.ndarray) -> np.ndarray:
    return (sim.s * (pts @ sim.R.T) + sim.t[None]).astype(np.float32)


def compose(a: Similarity, b: Similarity) -> Similarity:
    """
    Compose transforms: first b, then a.
    x' = b(x)
    x'' = a(x')
    """
    s = a.s * b.s
    R = (a.R @ b.R).astype(np.float32)
    t = (a.s * (a.R @ b.t) + a.t).astype(np.float32)
    return Similarity(s=s, R=R, t=t)


def invert(sim: Similarity) -> Similarity:
    R_inv = sim.R.T.astype(np.float32)
    s_inv = 1.0 / float(sim.s)
    t_inv = (-s_inv * (R_inv @ sim.t)).astype(np.float32)
    return Similarity(s=s_inv, R=R_inv, t=t_inv)


def ransac_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    iters: int = 300,
    inlier_thresh: float = 0.5,
    min_inliers: int = 4,
) -> Optional[Similarity]:
    n = X.shape[0]
    if n < 3:
        return None

    best = None
    best_inliers = None
    idxs = list(range(n))
    for _ in range(iters):
        sample = random.sample(idxs, 3)
        sim = umeyama_similarity(X[sample], Y[sample], allow_scale=True)
        pred = apply_sim(sim, X)
        err = np.linalg.norm(pred - Y, axis=1)
        inliers = err < inlier_thresh
        cnt = int(inliers.sum())
        if best is None or cnt > best:
            best = cnt
            best_inliers = inliers
        if best is not None and best >= max(min_inliers, int(0.8 * n)):
            break

    if best is None or best_inliers is None or best < min_inliers:
        return None
    # refine on inliers
    return umeyama_similarity(X[best_inliers], Y[best_inliers], allow_scale=True)


def dedup_boxes(
    centers: np.ndarray,
    dims: np.ndarray,
    Rmats: np.ndarray,
    scores: np.ndarray,
    dist_thresh: float = 0.6,
    dim_rel_thresh: float = 0.35,
) -> dict:
    order = np.argsort(-scores)
    kept = []
    for idx in order:
        c = centers[idx]
        d = dims[idx]
        s = scores[idx]
        matched = False
        for k in kept:
            ck = k["center"]
            dk = k["dims"]
            if np.linalg.norm(c - ck) > dist_thresh:
                continue
            # relative dim diff
            rel = np.abs(d - dk) / (np.maximum(dk, 1e-6))
            if float(rel.max()) > dim_rel_thresh:
                continue
            # merge by score-weighted average center/dims, keep rotation of higher score
            w0 = float(k["score"])
            w1 = float(s)
            wsum = w0 + w1
            k["center"] = (ck * w0 + c * w1) / wsum
            k["dims"] = (dk * w0 + d * w1) / wsum
            if s > k["score"]:
                k["R"] = Rmats[idx]
            k["score"] = max(k["score"], float(s))
            matched = True
            break
        if not matched:
            kept.append(
                {
                    "center": c.copy(),
                    "dims": d.copy(),
                    "R": Rmats[idx].copy(),
                    "score": float(s),
                }
            )

    out_centers = np.stack([k["center"] for k in kept], axis=0).astype(np.float32) if kept else np.zeros((0, 3), dtype=np.float32)
    out_dims = np.stack([k["dims"] for k in kept], axis=0).astype(np.float32) if kept else np.zeros((0, 3), dtype=np.float32)
    out_R = np.stack([k["R"] for k in kept], axis=0).astype(np.float32) if kept else np.zeros((0, 3, 3), dtype=np.float32)
    out_scores = np.asarray([k["score"] for k in kept], dtype=np.float32) if kept else np.zeros((0,), dtype=np.float32)
    return {
        "gravity_center_xyz": out_centers.tolist(),
        "dims_lhw": out_dims.tolist(),
        "R_3x3": out_R.tolist(),
        "scores": out_scores.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Merge multiple *_inf.json into a single world-aligned set of 3D boxes.")
    ap.add_argument("--inputs", nargs="+", required=True, help="List of per-image *_inf.json files.")
    ap.add_argument("--out", required=True, help="Output merged JSON path.")
    ap.add_argument("--score-thresh", type=float, default=0.25, help="Only keep detections above this score.")
    ap.add_argument("--max-hamming", type=int, default=10, help="Max dHash Hamming distance to match object crops.")
    ap.add_argument("--orb-max-px", type=float, default=80.0, help="Fallback ORB+homography bbox-center match threshold (pixels).")
    ap.add_argument("--ransac-iters", type=int, default=300, help="RANSAC iterations for pose estimation.")
    ap.add_argument("--inlier-thresh", type=float, default=0.6, help="Inlier threshold in meters for 3D center alignment.")
    ap.add_argument("--dedup-dist", type=float, default=0.6, help="World dedup center distance threshold (m).")
    args = ap.parse_args()

    random.seed(0)

    json_paths = [Path(p) for p in args.inputs]
    frames = [read_frame(p, score_thresh=float(args.score_thresh)) for p in json_paths]
    hashes = [build_hashes(f) for f in frames]

    # Build pairwise transforms using mutual-best hash matches.
    # We'll pick frame 0 as world and estimate transforms to connect others to it (directly or via chain).
    n = len(frames)
    edges: Dict[Tuple[int, int], Similarity] = {}
    match_counts: Dict[Tuple[int, int], int] = {}

    for i in range(n):
        for j in range(i + 1, n):
            matches = mutual_best_matches(hashes[i], hashes[j], max_hamming=int(args.max_hamming))
            if len(matches) >= 4:
                pairs = [(a, b) for a, b, _ in matches]
            else:
                # Fallback: ORB+homography on full images, then match bbox centers.
                orb_pairs = orb_homography_matches(frames[i], frames[j], max_px=float(args.orb_max_px), min_matches=4)
                pairs = [(a, b) for a, b, _ in orb_pairs]

            if len(pairs) < 4:
                continue

            Xi = np.stack([frames[i].centers_cam[a] for a, b in pairs], axis=0)
            Yj = np.stack([frames[j].centers_cam[b] for a, b in pairs], axis=0)
            sim_ij = ransac_similarity(
                Xi,
                Yj,
                iters=int(args.ransac_iters),
                inlier_thresh=float(args.inlier_thresh),
                min_inliers=4,
            )
            if sim_ij is None:
                continue
            edges[(i, j)] = sim_ij
            edges[(j, i)] = invert(sim_ij)
            match_counts[(i, j)] = len(pairs)
            match_counts[(j, i)] = len(pairs)

    # BFS from 0 over the strongest connections first.
    world_from: List[Optional[Similarity]] = [None] * n
    world_from[0] = Similarity(s=1.0, R=np.eye(3, dtype=np.float32), t=np.zeros((3,), dtype=np.float32))

    frontier = [0]
    while frontier:
        cur = frontier.pop(0)
        # neighbors sorted by match count (more overlap first)
        nbrs = []
        for j in range(n):
            if j == cur:
                continue
            if (cur, j) in edges and world_from[j] is None:
                nbrs.append((j, match_counts.get((cur, j), 0)))
        nbrs.sort(key=lambda x: -x[1])
        for j, _cnt in nbrs:
            # world_from[cur] maps cur->world
            # edges[(cur,j)] maps cur->j, so invert gives j->cur; easier: world = world_from[cur] ∘ (j->cur)
            j_from_cur = edges[(cur, j)]  # cur -> j
            cur_from_j = invert(j_from_cur)  # j -> cur
            world_from[j] = compose(world_from[cur], cur_from_j)
            frontier.append(j)

    if any(w is None for w in world_from):
        missing = [i for i, w in enumerate(world_from) if w is None]
        raise RuntimeError(
            f"Could not connect frames {missing} into a single world. "
            f"Need more overlapping objects between photos or relax matching thresholds."
        )

    # Transform all boxes into world and aggregate.
    all_centers = []
    all_dims = []
    all_R = []
    all_scores = []
    sources = []

    for i, fr in enumerate(frames):
        sim_w = world_from[i]
        assert sim_w is not None
        centers_w = apply_sim(sim_w, fr.centers_cam)
        dims_w = (fr.dims_cam * float(sim_w.s)).astype(np.float32)
        R_w = (sim_w.R @ fr.R_cam).astype(np.float32)
        for k in range(len(fr.detections)):
            all_centers.append(centers_w[k])
            all_dims.append(dims_w[k])
            all_R.append(R_w[k])
            all_scores.append(float(fr.scores[k]))
            sources.append(
                {
                    "frame_index": i,
                    "json": str(fr.json_path),
                    "image": str(fr.image_path),
                    "det_index_filtered": k,
                }
            )

    all_centers = np.stack(all_centers, axis=0).astype(np.float32) if all_centers else np.zeros((0, 3), dtype=np.float32)
    all_dims = np.stack(all_dims, axis=0).astype(np.float32) if all_dims else np.zeros((0, 3), dtype=np.float32)
    all_R = np.stack(all_R, axis=0).astype(np.float32) if all_R else np.zeros((0, 3, 3), dtype=np.float32)
    all_scores = np.asarray(all_scores, dtype=np.float32) if all_scores else np.zeros((0,), dtype=np.float32)

    merged_boxes = dedup_boxes(
        centers=all_centers,
        dims=all_dims,
        Rmats=all_R,
        scores=all_scores,
        dist_thresh=float(args.dedup_dist),
    )

    out = {
        "inputs": [str(p) for p in json_paths],
        "world_from_frames": [
            {
                "frame_index": i,
                "s": float(world_from[i].s),  # type: ignore[union-attr]
                "R_3x3": world_from[i].R.tolist(),  # type: ignore[union-attr]
                "t_xyz": world_from[i].t.tolist(),  # type: ignore[union-attr]
                "json": str(frames[i].json_path),
                "image": str(frames[i].image_path),
            }
            for i in range(n)
        ],
        "merged_boxes_3d": merged_boxes,
        "notes": {
            "method": "dHash mutual-best matching on 2D crops (fallback: ORB+homography bbox-center matching) + RANSAC Umeyama similarity on 3D centers, then world dedup",
            "score_thresh": float(args.score_thresh),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved merged world JSON: {out_path}")


if __name__ == "__main__":
    main()

