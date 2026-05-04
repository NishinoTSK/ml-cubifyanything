import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# COLMAP parsing utilities
# -----------------------------


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray  # (4,) qw qx qy qz
    tvec: np.ndarray  # (3,)
    camera_id: int
    name: str


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    # COLMAP uses qw, qx, qy, qz
    qw, qx, qy, qz = [float(x) for x in qvec]
    # normalize
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-12
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def parse_images_txt(path: Path) -> List[ColmapImage]:
    """
    Parse COLMAP sparse model images.txt (text format).
    Note: line format:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] (next line) - ignored
    """
    images: List[ColmapImage] = []
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln or ln.startswith("#"):
            i += 1
            continue
        parts = ln.split()
        if len(parts) < 10:
            i += 1
            continue
        image_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float32)
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images.append(ColmapImage(image_id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name))
        i += 2  # skip points2D line
    return images


# -----------------------------
# Pred JSON handling
# -----------------------------


def load_pred_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def apply_world_from_cam(R_wc: np.ndarray, t_wc: np.ndarray, centers_c: np.ndarray, R_boxes_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # centers: Nx3 (cam)
    centers_w = (centers_c @ R_wc.T + t_wc[None, :]).astype(np.float32)
    R_boxes_w = (R_wc @ R_boxes_c).astype(np.float32)
    return centers_w, R_boxes_w


def dedup_boxes(
    centers: np.ndarray,
    dims: np.ndarray,
    Rmats: np.ndarray,
    scores: np.ndarray,
    dist_thresh: float,
    dim_rel_thresh: float = 0.35,
) -> dict:
    order = np.argsort(-scores)
    kept = []
    for idx in order:
        c = centers[idx]
        d = dims[idx]
        s = float(scores[idx])
        matched = False
        for k in kept:
            if np.linalg.norm(c - k["center"]) > dist_thresh:
                continue
            rel = np.abs(d - k["dims"]) / (np.maximum(k["dims"], 1e-6))
            if float(rel.max()) > dim_rel_thresh:
                continue
            w0 = float(k["score"])
            w1 = float(s)
            wsum = w0 + w1
            k["center"] = (k["center"] * w0 + c * w1) / wsum
            k["dims"] = (k["dims"] * w0 + d * w1) / wsum
            if s > k["score"]:
                k["R"] = Rmats[idx]
            k["score"] = max(k["score"], s)
            matched = True
            break
        if not matched:
            kept.append({"center": c.copy(), "dims": d.copy(), "R": Rmats[idx].copy(), "score": s})

    if not kept:
        return {"gravity_center_xyz": [], "dims_lhw": [], "R_3x3": [], "scores": []}
    return {
        "gravity_center_xyz": np.stack([k["center"] for k in kept], axis=0).astype(np.float32).tolist(),
        "dims_lhw": np.stack([k["dims"] for k in kept], axis=0).astype(np.float32).tolist(),
        "R_3x3": np.stack([k["R"] for k in kept], axis=0).astype(np.float32).tolist(),
        "scores": np.asarray([k["score"] for k in kept], dtype=np.float32).tolist(),
    }


# -----------------------------
# External commands
# -----------------------------


def run(cmd: List[str], cwd: Optional[Path] = None):
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main():
    ap = argparse.ArgumentParser(description="RGB-only room reconstruction: video -> frames -> COLMAP poses -> CuTR per frame -> merge world boxes.")
    ap.add_argument("--video", required=True, help="Input video path (perspective).")
    ap.add_argument("--model-path", required=True, help="Path to cutr_rgb.pth")
    ap.add_argument("--workdir", required=True, help="Working directory (will create subfolders).")
    ap.add_argument("--device", default="cuda", help="cpu|cuda|mps")
    ap.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate (fps).")
    ap.add_argument("--max-frames", type=int, default=0, help="Optional cap on extracted frames (0=unlimited).")
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--infer-max-edge", type=int, default=1024, help="Passed to infer_image.py --max-edge")
    ap.add_argument("--dedup-dist", type=float, default=0.7, help="Dedup center distance threshold in COLMAP world units.")
    ap.add_argument("--out", required=True, help="Output merged world json path.")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(video)

    workdir = Path(args.workdir)
    frames_dir = workdir / "frames"
    db_path = workdir / "colmap.db"
    sparse_dir = workdir / "sparse"
    sparse0_dir = sparse_dir / "0"
    preds_dir = workdir / "preds"

    frames_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract frames
    # Use ffmpeg. The user can install it in WSL: sudo apt install ffmpeg
    out_pattern = str(frames_dir / "frame_%06d.jpg")
    run(["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={args.fps}", out_pattern])

    # Optional cap
    if args.max_frames and args.max_frames > 0:
        # delete extra frames
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        for p in frames[args.max_frames :]:
            p.unlink(missing_ok=True)

    # 2) COLMAP SfM (monocular)
    # Requires COLMAP installed in WSL.
    if db_path.exists():
        db_path.unlink()

    run(["colmap", "feature_extractor", "--database_path", str(db_path), "--image_path", str(frames_dir)])
    run(["colmap", "exhaustive_matcher", "--database_path", str(db_path)])
    run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(frames_dir),
            "--output_path",
            str(sparse_dir),
        ]
    )

    images_txt = sparse0_dir / "images.txt"
    if not images_txt.exists():
        raise RuntimeError(f"COLMAP did not produce {images_txt}. Check COLMAP output.")

    col_images = parse_images_txt(images_txt)
    by_name: Dict[str, ColmapImage] = {im.name: im for im in col_images}

    # 3) Run CuTR per frame using infer_image.py (keeps boxes_3d in camera coords)
    # We call python3 tools/infer_image.py ...
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    for fr in frames:
        out_json = preds_dir / (fr.stem + "_inf.json")
        if out_json.exists():
            continue
        run(
            [
                "python3",
                "tools/infer_image.py",
                "--image",
                str(fr),
                "--model-path",
                str(Path(args.model_path)),
                "--device",
                str(args.device),
                "--score-thresh",
                str(args.score_thresh),
                "--max-edge",
                str(args.infer_max_edge),
                "--out-json",
                str(out_json),
            ],
            cwd=Path.cwd(),
        )

    # 4) Transform boxes into COLMAP world and merge
    all_centers = []
    all_dims = []
    all_R = []
    all_scores = []
    used = 0
    skipped = 0

    for fr in frames:
        pred_path = preds_dir / (fr.stem + "_inf.json")
        if not pred_path.exists():
            continue
        name = fr.name
        if name not in by_name:
            skipped += 1
            continue
        cim = by_name[name]
        R_cw = qvec_to_rotmat(cim.qvec)  # world -> cam
        t_cw = cim.tvec
        # invert to get cam -> world
        R_wc = R_cw.T
        t_wc = (-R_wc @ t_cw).astype(np.float32)

        js = load_pred_json(pred_path)
        b3 = js.get("boxes_3d", None)
        if not b3:
            continue
        centers_c = np.asarray(b3["gravity_center_xyz"], dtype=np.float32)
        dims_c = np.asarray(b3["dims_lhw"], dtype=np.float32)
        R_boxes_c = np.asarray(b3["R_3x3"], dtype=np.float32)
        scores = np.asarray([float(d.get("score", 0.0)) for d in js.get("detections", [])], dtype=np.float32)

        if len(centers_c) == 0:
            continue
        centers_w, R_boxes_w = apply_world_from_cam(R_wc, t_wc, centers_c, R_boxes_c)

        all_centers.append(centers_w)
        all_dims.append(dims_c)  # note: scale is arbitrary in monocular SfM + monocular depth
        all_R.append(R_boxes_w)
        all_scores.append(scores)
        used += 1

    if not all_centers:
        raise RuntimeError("No predictions were merged. Check that infer_image produced boxes_3d and COLMAP registered frames.")

    centers = np.concatenate(all_centers, axis=0)
    dims = np.concatenate(all_dims, axis=0)
    Rmats = np.concatenate(all_R, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    merged = dedup_boxes(centers, dims, Rmats, scores, dist_thresh=float(args.dedup_dist))

    out = {
        "video": str(video),
        "workdir": str(workdir),
        "colmap_model": str(sparse0_dir),
        "frames_used": used,
        "frames_skipped_not_registered": skipped,
        "merged_boxes_3d": merged,
        "notes": {
            "scale": "COLMAP world scale is arbitrary; CuTR(RGB) metric scale is approximate. Use for layout/dedup, not exact meters.",
            "dedup_dist_units": "COLMAP world units (arbitrary scale). You may need to tune --dedup-dist.",
        },
    }

    save_json(Path(args.out), out)
    print(f"Saved merged world JSON: {args.out}")


if __name__ == "__main__":
    main()

