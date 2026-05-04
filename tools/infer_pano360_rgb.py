import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from cubifyanything.cubify_transformer import make_cubify_transformer
from cubifyanything.measurement import ImageMeasurementInfo
from cubifyanything.preprocessor import Augmentor, Preprocessor, move_input_to_current_device
from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo


def make_default_intrinsics(w: int, h: int, fov_deg: float) -> torch.Tensor:
    # Pinhole with principal point at center.
    fov = math.radians(fov_deg)
    fx = (w * 0.5) / math.tan(fov * 0.5)
    fy = fx
    cx = w * 0.5
    cy = h * 0.5
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def rot_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # Yaw around +Y, pitch around +X (camera convention here).
    R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    return (R_yaw @ R_pitch).astype(np.float32)


def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    img: HxWxC (uint8)
    u, v: float arrays in pixel coords (x, y)
    """
    H, W, C = img.shape

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # wrap horizontally, clamp vertically
    u0 = np.mod(u0, W)
    u1 = np.mod(u1, W)
    v0 = np.clip(v0, 0, H - 1)
    v1 = np.clip(v1, 0, H - 1)

    du = (u - np.floor(u)).astype(np.float32)[..., None]
    dv = (v - np.floor(v)).astype(np.float32)[..., None]

    p00 = img[v0, u0]
    p10 = img[v0, u1]
    p01 = img[v1, u0]
    p11 = img[v1, u1]

    p0 = p00 * (1.0 - du) + p10 * du
    p1 = p01 * (1.0 - du) + p11 * du
    p = p0 * (1.0 - dv) + p1 * dv
    return p


def equirect_to_perspective(
    pano: Image.Image,
    out_w: int,
    out_h: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
) -> Image.Image:
    pano_np = np.asarray(pano.convert("RGB"))
    H, W, _ = pano_np.shape

    # pixel grid in perspective image
    xs = (np.arange(out_w, dtype=np.float32) + 0.5)
    ys = (np.arange(out_h, dtype=np.float32) + 0.5)
    xx, yy = np.meshgrid(xs, ys)

    # normalized camera coords
    fov = math.radians(fov_deg)
    fx = (out_w * 0.5) / math.tan(fov * 0.5)
    fy = fx
    cx = out_w * 0.5
    cy = out_h * 0.5

    x = (xx - cx) / fx
    y = (yy - cy) / fy
    z = np.ones_like(x)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8

    R = rot_yaw_pitch(yaw_deg, pitch_deg)
    dirs_w = dirs @ R.T

    # spherical coords: lon [-pi,pi], lat [-pi/2,pi/2]
    lon = np.arctan2(dirs_w[..., 0], dirs_w[..., 2])
    lat = np.arcsin(np.clip(dirs_w[..., 1], -1.0, 1.0))

    # map to pano pixel coords
    u = (lon + math.pi) / (2.0 * math.pi) * W
    v = (math.pi / 2.0 - lat) / math.pi * H

    out = _bilinear_sample(pano_np, u, v).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@dataclass
class BoxSet:
    centers: np.ndarray  # Nx3
    dims: np.ndarray  # Nx3
    R: np.ndarray  # Nx3x3
    scores: np.ndarray  # N


def dedup_boxes(
    centers: np.ndarray,
    dims: np.ndarray,
    Rmats: np.ndarray,
    scores: np.ndarray,
    dist_thresh: float = 0.6,
    dim_rel_thresh: float = 0.35,
) -> BoxSet:
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
        return BoxSet(
            centers=np.zeros((0, 3), dtype=np.float32),
            dims=np.zeros((0, 3), dtype=np.float32),
            R=np.zeros((0, 3, 3), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
        )

    return BoxSet(
        centers=np.stack([k["center"] for k in kept], axis=0).astype(np.float32),
        dims=np.stack([k["dims"] for k in kept], axis=0).astype(np.float32),
        R=np.stack([k["R"] for k in kept], axis=0).astype(np.float32),
        scores=np.asarray([k["score"] for k in kept], dtype=np.float32),
    )


def run_model_on_view(model, device_tensor, view_img: Image.Image, fov_deg: float, score_thresh: float):
    rgb = torch.from_numpy(np.asarray(view_img).copy()).permute(2, 0, 1).contiguous()
    h, w = int(rgb.shape[1]), int(rgb.shape[2])

    K = make_default_intrinsics(w, h, fov_deg=fov_deg)

    wide = PosedSensorInfo()
    wide.RT = torch.eye(4, dtype=torch.float32)[None]
    wide.T_gravity = torch.eye(3, dtype=torch.float32)[None]
    wide.image = ImageMeasurementInfo(size=(w, h), K=K[None])

    sample = {
        "sensor_info": SensorArrayInfo(wide=wide),
        "wide": {"image": rgb[None]},
        "meta": {"video_id": None, "timestamp": None},
    }

    augmentor = Augmentor(("wide/image",))
    size_div = 32
    square_pad = int(math.ceil(max(w, h) / size_div) * size_div)
    preprocessor = Preprocessor(square_pad=square_pad, size_divisibility=size_div)

    packaged = augmentor.package(sample)
    packaged = move_input_to_current_device(packaged, device_tensor)
    packaged = preprocessor.preprocess([packaged])

    with torch.no_grad():
        pred = model(packaged)[0]
    pred = pred[pred.scores >= float(score_thresh)]
    if not pred.has("pred_boxes_3d"):
        raise RuntimeError("Model output did not include pred_boxes_3d")

    centers = pred.pred_boxes_3d.gravity_center.detach().cpu().numpy().astype(np.float32)
    dims = pred.pred_boxes_3d.dims.detach().cpu().numpy().astype(np.float32)
    Rmats = pred.pred_boxes_3d.R.detach().cpu().numpy().astype(np.float32)
    scores = pred.scores.detach().cpu().numpy().astype(np.float32)
    return centers, dims, Rmats, scores


def main():
    ap = argparse.ArgumentParser(description="Run CuTR (RGB) on an equirectangular 360 pano by slicing into perspective views.")
    ap.add_argument("--image360", required=True, help="Path to equirectangular panorama (2:1) image.")
    ap.add_argument("--model-path", required=True, help="Path to cutr_rgb.pth checkpoint.")
    ap.add_argument("--device", default="cuda", help="cpu|cuda|mps")
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--out", required=True, help="Output merged world JSON path.")
    ap.add_argument("--out-dir-views", default=None, help="Optional: save perspective view images to this dir.")
    ap.add_argument("--view-width", type=int, default=1024)
    ap.add_argument("--view-height", type=int, default=768)
    ap.add_argument("--fov-deg", type=float, default=90.0)
    ap.add_argument("--pitch-deg", type=float, default=0.0)
    ap.add_argument("--num-yaw", type=int, default=12, help="Number of yaw views around 360.")
    ap.add_argument("--dedup-dist", type=float, default=0.7, help="Dedup center distance threshold (m).")
    args = ap.parse_args()

    pano_path = Path(args.image360)
    model_path = Path(args.model_path)
    if not pano_path.exists():
        raise FileNotFoundError(pano_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    pano = Image.open(str(pano_path)).convert("RGB")

    checkpoint = torch.load(str(model_path), map_location=args.device or "cpu")["model"]
    backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
    is_depth_model = any(k.startswith("backbone.0.patch_embed_depth.") for k in checkpoint.keys())
    if is_depth_model:
        raise ValueError("This script is for RGB checkpoints only. Use cutr_rgb.pth.")

    model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=False).eval()
    model.load_state_dict(checkpoint)
    model = model.to(args.device)

    device_tensor = model.pixel_mean  # for move_input_to_current_device

    yaw_step = 360.0 / float(args.num_yaw)
    yaws = [i * yaw_step for i in range(int(args.num_yaw))]

    if args.out_dir_views:
        out_views_dir = Path(args.out_dir_views)
        out_views_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_views_dir = None

    all_centers_w = []
    all_dims_w = []
    all_R_w = []
    all_scores = []
    sources = []

    for idx, yaw in enumerate(yaws):
        view = equirect_to_perspective(
            pano,
            out_w=int(args.view_width),
            out_h=int(args.view_height),
            fov_deg=float(args.fov_deg),
            yaw_deg=float(yaw),
            pitch_deg=float(args.pitch_deg),
        )
        if out_views_dir is not None:
            view_path = out_views_dir / f"view_{idx:02d}_yaw{int(round(yaw))}.png"
            view.save(str(view_path))
        else:
            view_path = None

        centers_c, dims_c, R_c, scores = run_model_on_view(
            model=model,
            device_tensor=device_tensor,
            view_img=view,
            fov_deg=float(args.fov_deg),
            score_thresh=float(args.score_thresh),
        )

        # World: same camera center for all views; only rotation differs.
        R_view = rot_yaw_pitch(float(yaw), float(args.pitch_deg))  # cam -> world
        centers_w = (centers_c @ R_view.T).astype(np.float32)
        R_w = (R_view @ R_c).astype(np.float32)

        all_centers_w.append(centers_w)
        all_dims_w.append(dims_c)
        all_R_w.append(R_w)
        all_scores.append(scores)
        sources.append(
            {
                "index": idx,
                "yaw_deg": float(yaw),
                "pitch_deg": float(args.pitch_deg),
                "view_image": str(view_path) if view_path is not None else None,
            }
        )

    centers = np.concatenate(all_centers_w, axis=0) if all_centers_w else np.zeros((0, 3), dtype=np.float32)
    dims = np.concatenate(all_dims_w, axis=0) if all_dims_w else np.zeros((0, 3), dtype=np.float32)
    Rmats = np.concatenate(all_R_w, axis=0) if all_R_w else np.zeros((0, 3, 3), dtype=np.float32)
    scores = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float32)

    merged = dedup_boxes(
        centers=centers,
        dims=dims,
        Rmats=Rmats,
        scores=scores,
        dist_thresh=float(args.dedup_dist),
    )

    out = {
        "source_pano": str(pano_path),
        "views": sources,
        "merged_boxes_3d": {
            "gravity_center_xyz": merged.centers.tolist(),
            "dims_lhw": merged.dims.tolist(),
            "R_3x3": merged.R.tolist(),
            "scores": merged.scores.tolist(),
        },
        "notes": {
            "assumptions": [
                "Equirectangular pano sliced into perspective pinhole views.",
                "All views share the same camera center; only yaw/pitch rotation differs.",
                "RGB-only CuTR provides approximate metric scale; world is up to global scale drift.",
            ],
            "fov_deg": float(args.fov_deg),
            "view_size": [int(args.view_width), int(args.view_height)],
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved merged pano world JSON: {out_path}")


if __name__ == "__main__":
    main()

