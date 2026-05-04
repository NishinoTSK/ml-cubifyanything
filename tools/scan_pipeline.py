"""
Build a persistent room map (`room.json`) from a directory of Quest 3
passthrough captures.

Each capture is expected to be a pair:

    <stem>.png   (or .jpg)
    <stem>.json  with at least:
        {
            "fx": ..., "fy": ..., "cx": ..., "cy": ...,
            "width": ..., "height": ...,
            "timestamp": "...",
            "pose_R_wc": [[r00..r02],[r10..r12],[r20..r22]],
            "pose_t_wc": [tx, ty, tz],
            "anchor_uuid": "..."          (optional)
        }

Optional sibling `<stem>_depth.png` (UInt16, millimeters) is picked up
automatically when the chosen checkpoint is RGB-D.

The pipeline:

    image -> CuTR -> 3D boxes in CAMERA frame (CuTR/OpenCV convention)
                  -> change-of-basis to Unity world via M=diag(1,-1,1)
                  -> apply Unity pose (R_wc, t_wc)
                  -> dedup across frames
                  -> rooms/<room_id>/room.json

The output schema is documented in the README.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cutr_runtime import CutrRunner, load_depth_image_mm_png, make_default_intrinsics  # noqa: E402


# Change-of-basis: OpenCV (x right, y down, z forward, RH) <-> Unity (x right, y up, z forward, LH).
# Same matrix in both directions (involutive).
M_UNITY_FROM_CV = np.diag([1.0, -1.0, 1.0]).astype(np.float32)


def find_captures(captures_dir: Path) -> List[Path]:
    """Return JSON paths that have a sibling image."""
    out: List[Path] = []
    for js in sorted(captures_dir.glob("*.json")):
        if js.name.endswith("_inf.json"):
            continue
        if js.name.endswith("_depth.json"):
            continue
        png = js.with_suffix(".png")
        jpg = js.with_suffix(".jpg")
        jpeg = js.with_suffix(".jpeg")
        if any(p.exists() for p in (png, jpg, jpeg)):
            out.append(js)
    return out


def sibling_image(json_path: Path) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        cand = json_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def sibling_depth(json_path: Path) -> Optional[Path]:
    base = json_path.with_suffix("")
    for cand in (
        base.with_name(base.name + "_depth.png"),
        base.with_name(base.name + "_depth.tif"),
    ):
        if cand.exists():
            return cand
    return None


def parse_pose(meta: dict) -> Optional[tuple[np.ndarray, np.ndarray]]:
    R = meta.get("pose_R_wc")
    t = meta.get("pose_t_wc")
    if R is None or t is None:
        return None
    R = np.asarray(R, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    if R.shape != (3, 3) or t.shape != (3,):
        return None
    return R, t


def parse_intrinsics(meta: dict) -> Optional[torch.Tensor]:
    keys = ("fx", "fy", "cx", "cy")
    if not all(k in meta for k in keys):
        return None
    K = make_default_intrinsics(int(meta["width"]), int(meta["height"]))
    K[0, 0] = float(meta["fx"])
    K[1, 1] = float(meta["fy"])
    K[0, 2] = float(meta["cx"])
    K[1, 2] = float(meta["cy"])
    return K


def transform_boxes(
    centers_cv: np.ndarray,
    R_boxes_cv: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform CuTR boxes from camera frame into world frame.

    convention="unity"  : pose is Unity (LH, Y-up). Apply M change-of-basis.
    convention="opencv" : pose is already in CV (RH, Y-down). Skip M.
    """
    if convention == "unity":
        M = M_UNITY_FROM_CV
        centers_w = ((R_wc @ M) @ centers_cv.T).T + t_wc[None, :]
        # Use M on both sides of R_box to stay a proper rotation in Unity.
        R_w = R_wc @ M @ R_boxes_cv @ M
    elif convention == "opencv":
        centers_w = (R_wc @ centers_cv.T).T + t_wc[None, :]
        R_w = R_wc @ R_boxes_cv
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return centers_w.astype(np.float32), R_w.astype(np.float32)


_OBB_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ],
    dtype=np.float32,
)


def aabb_of_obb(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-axis-aligned bounding box that encloses an oriented box."""
    half = dims.astype(np.float32) * 0.5
    local = _OBB_CORNER_SIGNS * half  # (8, 3) in box-local frame
    world = local @ R.T + center.astype(np.float32)[None, :]  # (8, 3)
    return world.min(axis=0), world.max(axis=0)


def aabb_iou(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    """Volume-IoU between two axis-aligned boxes."""
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter))
    if inter_vol <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
    vol_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
    union = vol_a + vol_b - inter_vol
    if union <= 0.0:
        return 0.0
    return float(inter_vol / union)


def aabb_containment(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    """min(intersection / vol_a, intersection / vol_b).

    This catches the case where a small box is fully inside a large box: IoU
    is small (because the bigger box dominates union) but containment is 1.0.
    Useful when CuTR detects the same chair as a tight box from one angle and
    a loose bounding region from another.
    """
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter))
    if inter_vol <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
    vol_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
    smaller = min(vol_a, vol_b)
    if smaller <= 0.0:
        return 0.0
    return float(inter_vol / smaller)


def _center_distance(a: dict, b: dict) -> float:
    return float(np.linalg.norm(a["center"] - b["center"]))


def _diag(dims: np.ndarray) -> float:
    return float(np.sqrt(float(np.sum(dims.astype(np.float32) ** 2))))


def dedup_boxes(
    centers: np.ndarray,
    dims: np.ndarray,
    Rmats: np.ndarray,
    scores: np.ndarray,
    sources: List[str],
    bboxes: Optional[List[Optional[List[float]]]] = None,
    iou_thresh: float = 0.2,
    containment_thresh: float = 0.5,
    center_fuse_ratio: float = 0.5,
    max_passes: int = 8,
) -> List[dict]:
    """3D dedup combining IoU, containment, and adaptive center-distance.

    Two boxes are merged when *any* of these hold:
      - AABB-IoU >= ``iou_thresh`` (clear spatial overlap), or
      - one is mostly inside the other (containment >= ``containment_thresh``), or
      - centers closer than ``center_fuse_ratio * min(diag_a, diag_b)``
        — handles the common case where CuTR places the same object at
        slightly different metric depths from different views, so the boxes
        are similar in size but offset by 10-30 cm.

    The pass repeats until no merges happen or ``max_passes`` is reached.
    """
    n = len(centers)
    if n == 0:
        return []

    clusters: List[dict] = []
    for i in range(n):
        d = dims[i].astype(np.float32)
        amin, amax = aabb_of_obb(centers[i], d, Rmats[i])
        bb = None
        if bboxes is not None and i < len(bboxes) and bboxes[i] is not None:
            bb = [float(v) for v in bboxes[i]]
        clusters.append(
            {
                "center": centers[i].astype(np.float32).copy(),
                "dims": d.copy(),
                "R": Rmats[i].astype(np.float32).copy(),
                "score": float(scores[i]),
                "frames": {sources[i]},
                "best_frame": sources[i],
                "best_bbox": bb,
                "aabb_min": amin,
                "aabb_max": amax,
                "diag": _diag(d),
            }
        )

    def _merge(into: dict, src: dict):
        w0 = into["score"]
        w1 = src["score"]
        wsum = w0 + w1 if (w0 + w1) > 0 else 1.0
        into["center"] = (into["center"] * w0 + src["center"] * w1) / wsum
        into["dims"] = (into["dims"] * w0 + src["dims"] * w1) / wsum
        if src["score"] > into["score"]:
            into["R"] = src["R"]
            into["best_frame"] = src["best_frame"]
            if src.get("best_bbox") is not None:
                into["best_bbox"] = src["best_bbox"]
        into["score"] = max(into["score"], src["score"])
        into["frames"].update(src["frames"])
        into["aabb_min"], into["aabb_max"] = aabb_of_obb(into["center"], into["dims"], into["R"])
        into["diag"] = _diag(into["dims"])

    for _pass in range(max_passes):
        clusters.sort(key=lambda c: -c["score"])
        merged_any = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                a = clusters[i]
                b = clusters[j]
                cd = _center_distance(a, b)
                fuse_radius = center_fuse_ratio * min(a["diag"], b["diag"])

                if cd <= fuse_radius:
                    _merge(a, b)
                    clusters.pop(j)
                    merged_any = True
                    continue

                if np.any(a["aabb_max"] < b["aabb_min"]) or np.any(b["aabb_max"] < a["aabb_min"]):
                    j += 1
                    continue

                iou = aabb_iou(a["aabb_min"], a["aabb_max"], b["aabb_min"], b["aabb_max"])
                contain = aabb_containment(
                    a["aabb_min"], a["aabb_max"], b["aabb_min"], b["aabb_max"]
                )
                if iou >= iou_thresh or contain >= containment_thresh:
                    _merge(a, b)
                    clusters.pop(j)
                    merged_any = True
                else:
                    j += 1
            i += 1
        if not merged_any:
            break

    return clusters


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(
    captures_dir: Path,
    out_path: Path,
    model_path: Path,
    device: str,
    score_thresh: float,
    convention: str,
    iou_thresh: float,
    containment_thresh: float,
    center_fuse_ratio: float,
    max_edge: int,
    min_evidence: int = 1,
    min_volume: float = 0.0,
    room_id: Optional[str] = None,
):
    captures_dir = captures_dir.resolve()
    if not captures_dir.exists():
        raise FileNotFoundError(captures_dir)

    json_paths = find_captures(captures_dir)
    if not json_paths:
        raise RuntimeError(f"No captures found in {captures_dir}")

    print(f"Loading CuTR model: {model_path}")
    runner = CutrRunner(model_path=model_path, device=device)

    all_centers: list[np.ndarray] = []
    all_dims: list[np.ndarray] = []
    all_R: list[np.ndarray] = []
    all_scores: list[float] = []
    all_sources: list[str] = []
    all_bboxes: list[Optional[List[float]]] = []

    used = 0
    skipped = 0
    skipped_reasons: dict[str, int] = {}
    anchor_uuid: Optional[str] = None

    def _skip(reason: str, name: str):
        nonlocal skipped
        skipped += 1
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        print(f"  SKIP {name}: {reason}")

    for js_path in json_paths:
        name = js_path.stem
        try:
            meta = json.loads(js_path.read_text(encoding="utf-8"))
        except Exception as e:
            _skip(f"json parse error: {e}", name)
            continue

        pose = parse_pose(meta)
        if pose is None:
            _skip("missing pose_R_wc / pose_t_wc", name)
            continue
        R_wc, t_wc = pose

        K = parse_intrinsics(meta)
        if K is None:
            _skip("missing intrinsics (fx, fy, cx, cy)", name)
            continue

        if anchor_uuid is None:
            anchor_uuid = meta.get("anchor_uuid")

        img_path = sibling_image(js_path)
        if img_path is None:
            _skip("no sibling image", name)
            continue
        img = Image.open(str(img_path)).convert("RGB")

        depth_m = None
        if runner.is_depth_model:
            dpath = sibling_depth(js_path)
            if dpath is None:
                _skip("RGB-D model but no sibling _depth.png", name)
                continue
            depth_m = load_depth_image_mm_png(dpath)

        try:
            pred = runner.infer(
                image=img,
                K=K,
                depth_m=depth_m,
                score_thresh=score_thresh,
                max_edge=(None if max_edge <= 0 else max_edge),
            )
        except Exception as e:
            _skip(f"infer error: {e}", name)
            continue

        b3 = pred.get("boxes_3d")
        dets = pred.get("detections") or []
        if not b3 or not dets:
            print(f"  ZERO {name}: no detections above score_thresh={score_thresh}")
            used += 1
            continue

        centers_cv = np.asarray(b3["gravity_center_xyz"], dtype=np.float32)
        dims_cv = np.asarray(b3["dims_lhw"], dtype=np.float32)
        R_boxes_cv = np.asarray(b3["R_3x3"], dtype=np.float32)
        scores = np.asarray([float(d.get("score", 0.0)) for d in dets], dtype=np.float32)
        bboxes_xyxy: list[Optional[List[float]]] = []
        for d in dets:
            bb = d.get("bbox_xyxy")
            if bb and len(bb) == 4:
                bboxes_xyxy.append([float(v) for v in bb])
            else:
                bboxes_xyxy.append(None)

        centers_w, R_w = transform_boxes(centers_cv, R_boxes_cv, R_wc, t_wc, convention=convention)

        all_centers.append(centers_w)
        all_dims.append(dims_cv)
        all_R.append(R_w)
        all_scores.extend(scores.tolist())
        all_sources.extend([img_path.name] * len(scores))
        all_bboxes.extend(bboxes_xyxy)
        used += 1
        print(f"  OK   {name}: {len(scores)} detections")

    if used == 0 or not all_centers:
        raise RuntimeError(
            f"No usable captures in {captures_dir}. "
            f"Skipped={skipped}, reasons={skipped_reasons}"
        )

    centers = np.concatenate(all_centers, axis=0)
    dims = np.concatenate(all_dims, axis=0)
    Rmats = np.concatenate(all_R, axis=0)
    scores = np.asarray(all_scores, dtype=np.float32)

    print(f"Aggregating {len(centers)} detections from {used} captures...")
    clusters = dedup_boxes(
        centers=centers,
        dims=dims,
        Rmats=Rmats,
        scores=scores,
        sources=all_sources,
        bboxes=all_bboxes,
        iou_thresh=iou_thresh,
        containment_thresh=containment_thresh,
        center_fuse_ratio=center_fuse_ratio,
    )
    print(f"After dedup: {len(clusters)} clusters")

    if min_volume > 0.0:
        before = len(clusters)
        clusters = [c for c in clusters if float(np.prod(c["dims"])) >= min_volume]
        print(f"After min_volume={min_volume:.4f}m^3: {len(clusters)} (dropped {before - len(clusters)} tiny boxes)")

    if min_evidence > 1:
        before = len(clusters)
        clusters = [c for c in clusters if len(c["frames"]) >= min_evidence]
        print(f"After min_evidence={min_evidence}: {len(clusters)} (dropped {before - len(clusters)} singletons)")

    clusters.sort(key=lambda c: -c["score"])

    objects = []
    for i, k in enumerate(clusters):
        evidence: dict = {
            "n_frames": len(k["frames"]),
            "best_frame": k["best_frame"],
            "frames": sorted(k["frames"]),
        }
        if k.get("best_bbox") is not None:
            evidence["best_bbox"] = list(k["best_bbox"])
        objects.append(
            {
                "id": f"obj_{i:04d}",
                "label": None,
                "category": None,
                "category_score": None,
                "score": float(k["score"]),
                "center_xyz": k["center"].astype(np.float32).tolist(),
                "dims_lhw": k["dims"].astype(np.float32).tolist(),
                "R_3x3": k["R"].astype(np.float32).tolist(),
                "evidence": evidence,
            }
        )

    out = {
        "room_id": room_id or captures_dir.parent.name,
        "world_frame": {
            "anchor_uuid": anchor_uuid,
            "units": "meters",
            "convention": (
                "unity_left_handed_y_up" if convention == "unity" else "opencv_right_handed_y_down"
            ),
        },
        "created_at": now_iso(),
        "n_captures_used": used,
        "n_captures_skipped": skipped,
        "skipped_reasons": skipped_reasons,
        "objects": objects,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved room map: {out_path}")
    print(f"  objects:  {len(objects)}")
    print(f"  used:     {used}")
    print(f"  skipped:  {skipped} {skipped_reasons if skipped_reasons else ''}")


def main():
    ap = argparse.ArgumentParser(description="Build room.json from a directory of Quest captures.")
    ap.add_argument(
        "--captures",
        required=True,
        help="Directory of <stem>.png + <stem>.json captures (with pose_R_wc / pose_t_wc).",
    )
    ap.add_argument("--out", required=True, help="Output room.json path.")
    ap.add_argument("--model-path", required=True, help="Path to cutr_rgb.pth or cutr_rgbd.pth.")
    ap.add_argument("--device", default="cuda", help="cpu|cuda|mps")
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--max-edge", type=int, default=1024, help="0 = no resize")
    ap.add_argument(
        "--convention",
        default="unity",
        choices=("unity", "opencv"),
        help="Pose frame convention. 'unity' (default) applies M=diag(1,-1,1) "
             "between CuTR camera frame and the user-supplied world pose.",
    )
    ap.add_argument(
        "--iou-thresh",
        type=float,
        default=0.2,
        help="3D AABB-IoU threshold for fusing boxes (lower = more aggressive merging). Default 0.2.",
    )
    ap.add_argument(
        "--containment-thresh",
        type=float,
        default=0.5,
        help="Fuse if smaller box is at least this fraction inside the larger one. Default 0.5.",
    )
    ap.add_argument(
        "--center-fuse-ratio",
        type=float,
        default=0.5,
        help="Fuse boxes whose centers are within ratio*min(diag) of each other. Default 0.5.",
    )
    ap.add_argument(
        "--min-evidence",
        type=int,
        default=1,
        help="Drop clusters from fewer than N frames. Set to 2 to remove one-shot detections.",
    )
    ap.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Drop clusters with volume below this (m^3). 0.001 ~= 10cm cube. Default off.",
    )
    ap.add_argument("--room-id", default=None, help="Optional room_id field; defaults to parent folder name.")
    args = ap.parse_args()

    run(
        captures_dir=Path(args.captures),
        out_path=Path(args.out),
        model_path=Path(args.model_path),
        device=args.device,
        score_thresh=float(args.score_thresh),
        convention=args.convention,
        iou_thresh=float(args.iou_thresh),
        containment_thresh=float(args.containment_thresh),
        center_fuse_ratio=float(args.center_fuse_ratio),
        min_evidence=int(args.min_evidence),
        min_volume=float(args.min_volume),
        max_edge=int(args.max_edge),
        room_id=args.room_id,
    )


if __name__ == "__main__":
    main()
