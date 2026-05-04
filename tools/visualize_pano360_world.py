import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _color_for_index(i: int) -> Tuple[int, int, int]:
    palette = [
        (255, 59, 48),
        (255, 149, 0),
        (255, 204, 0),
        (52, 199, 89),
        (0, 199, 190),
        (0, 122, 255),
        (88, 86, 214),
        (175, 82, 222),
        (255, 45, 85),
        (142, 142, 147),
    ]
    return palette[i % len(palette)]


def _build_box_corners(center: np.ndarray, dims_lhw: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Returns 8 corners (8x3) in world.
    dims_lhw assumed to correspond to local X,Y,Z half-extents (l,h,w) in this visualization.
    """
    l, h, w = float(dims_lhw[0]), float(dims_lhw[1]), float(dims_lhw[2])
    hx, hy, hz = l * 0.5, h * 0.5, w * 0.5
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    return (local @ R.T + center[None, :]).astype(np.float32)


def _world_to_equirect_uv(points: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    points: Nx3 in world, camera at origin.
    Returns Nx2 pixel coords (u,v) on equirectangular panorama.
    """
    p = points.astype(np.float32)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    lon = np.arctan2(p[:, 0], p[:, 2])  # [-pi,pi]
    lat = np.arcsin(np.clip(p[:, 1], -1.0, 1.0))  # [-pi/2,pi/2]
    u = (lon + math.pi) / (2.0 * math.pi) * W
    v = (math.pi / 2.0 - lat) / math.pi * H
    return np.stack([u, v], axis=1).astype(np.float32)


def _unwrap_u(uv: np.ndarray, W: int) -> np.ndarray:
    """
    Handle seam crossing by shifting u values so that the set has minimal span.
    """
    u = uv[:, 0].copy()
    v = uv[:, 1].copy()
    # Try three representations: u, u+W, u-W; pick one with minimal range.
    candidates = [
        u,
        u + W,
        u - W,
    ]
    best = None
    best_span = None
    for c in candidates:
        span = float(c.max() - c.min())
        if best is None or span < best_span:
            best = c
            best_span = span
    out = np.stack([best, v], axis=1)
    return out


def draw_boxes_on_pano(
    pano: Image.Image,
    centers: np.ndarray,
    dims: np.ndarray,
    Rmats: np.ndarray,
    scores: np.ndarray | None,
    score_thresh: float,
    line_width: int,
) -> Image.Image:
    img = pano.convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # edges between corner indices
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    n = centers.shape[0]
    for i in range(n):
        s = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        if s < score_thresh:
            continue

        corners = _build_box_corners(centers[i], dims[i], Rmats[i])
        uv = _world_to_equirect_uv(corners, W=W, H=H)
        uv_unwrap = _unwrap_u(uv, W=W)

        color = _color_for_index(i)
        for a, b in edges:
            x1, y1 = float(uv_unwrap[a, 0]), float(uv_unwrap[a, 1])
            x2, y2 = float(uv_unwrap[b, 0]), float(uv_unwrap[b, 1])
            # draw potentially twice if unwrapped goes outside [0,W)
            for shift in (0.0, -W, W):
                xa1, xa2 = x1 + shift, x2 + shift
                if (xa1 < -W) and (xa2 < -W):
                    continue
                if (xa1 > 2 * W) and (xa2 > 2 * W):
                    continue
                draw.line([(xa1, y1), (xa2, y2)], fill=color, width=line_width)

    return img


def main():
    ap = argparse.ArgumentParser(description="Overlay merged world 3D boxes onto an equirectangular 360 pano.")
    ap.add_argument("--pano", required=True, help="Path to pano image (equirectangular).")
    ap.add_argument("--merged-json", required=True, help="Path to pano_merged_world.json (from tools/infer_pano360_rgb.py).")
    ap.add_argument("--out", default=None, help="Output image path. Default: <pano>_inf.png")
    ap.add_argument("--score-thresh", type=float, default=0.25, help="Only draw boxes with score >= threshold.")
    ap.add_argument("--line-width", type=int, default=3)
    args = ap.parse_args()

    pano_path = Path(args.pano)
    merged_path = Path(args.merged_json)
    if not pano_path.exists():
        raise FileNotFoundError(pano_path)
    if not merged_path.exists():
        raise FileNotFoundError(merged_path)

    pano = Image.open(str(pano_path)).convert("RGB")
    js = _load_json(merged_path)
    mb = js.get("merged_boxes_3d", None)
    if mb is None:
        raise ValueError("Missing 'merged_boxes_3d' in merged json")

    centers = np.asarray(mb.get("gravity_center_xyz", []), dtype=np.float32)
    dims = np.asarray(mb.get("dims_lhw", []), dtype=np.float32)
    Rmats = np.asarray(mb.get("R_3x3", []), dtype=np.float32)
    scores = mb.get("scores", None)
    scores_np = np.asarray(scores, dtype=np.float32) if scores is not None else None

    out_img = draw_boxes_on_pano(
        pano=pano,
        centers=centers,
        dims=dims,
        Rmats=Rmats,
        scores=scores_np,
        score_thresh=float(args.score_thresh),
        line_width=int(args.line_width),
    )

    out_path = Path(args.out) if args.out else pano_path.with_name(pano_path.stem + "_inf.png")
    out_img.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

