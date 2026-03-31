import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
from cubifyanything.preprocessor import Augmentor, Preprocessor, move_input_to_current_device
from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo
from cubifyanything.cubify_transformer import make_cubify_transformer


def _to_builtin(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _safe_stem(p: Path):
    # Windows-friendly filename stem.
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in p.stem)


def save_pred_bboxes_json(pred_instances, image_path: Path, out_path: Path):
    image_h, image_w = pred_instances.image_size

    dets = []
    for i in range(len(pred_instances)):
        det = {}
        if pred_instances.has("pred_boxes"):
            det["bbox_xyxy"] = _to_builtin(pred_instances.pred_boxes[i])
        if pred_instances.has("scores"):
            det["score"] = float(_to_builtin(pred_instances.scores[i]))
        if pred_instances.has("pred_classes"):
            det["class_id"] = int(_to_builtin(pred_instances.pred_classes[i]))
        dets.append(det)

    payload = {
        "source_image": str(image_path),
        "timestamp": None,
        "video_id": None,
        "image_size_hw": [int(image_h), int(image_w)],
        "detections": dets,
    }

    if pred_instances.has("pred_boxes_3d"):
        boxes3d = pred_instances.pred_boxes_3d
        payload["boxes_3d"] = {
            "gravity_center_xyz": _to_builtin(boxes3d.gravity_center),
            "dims_lhw": _to_builtin(boxes3d.dims),
            "R_3x3": _to_builtin(boxes3d.R),
        }

    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_rgb_image(path: Path, max_edge: int | None = None) -> tuple[torch.Tensor, float]:
    img = Image.open(str(path)).convert("RGB")
    scale = 1.0
    if max_edge is not None:
        w, h = img.size
        longest = max(w, h)
        if longest > max_edge:
            scale = float(max_edge) / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # np.asarray(PIL) can be read-only; copy to avoid PyTorch warning.
    arr = np.asarray(img).copy()  # HWC, uint8
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return t, scale


def load_depth_image_mm_png(path: Path) -> torch.Tensor:
    # Expect a single-channel PNG where values are millimeters (UInt16), like CA-1M.
    img = Image.open(str(path))
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel depth image, got shape={arr.shape}")
    if arr.dtype != np.uint16 and arr.dtype != np.int32 and arr.dtype != np.int64:
        # PIL may give uint16, but guard.
        arr = arr.astype(np.uint16)
    depth_m = torch.from_numpy(arr.astype(np.float32)) / 1000.0
    return depth_m


def make_default_intrinsics(w: int, h: int) -> torch.Tensor:
    # A reasonable default when intrinsics are unknown.
    # This is not "correct" physically, but lets the model run.
    fx = float(max(w, h))
    fy = float(max(w, h))
    cx = float(w) * 0.5
    cy = float(h) * 0.5
    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return K


def main():
    ap = argparse.ArgumentParser(description="Run CubifyAnything on an arbitrary image.")
    ap.add_argument("--image", required=True, help="Path to an RGB image (png/jpg).")
    ap.add_argument("--model-path", required=True, help="Path to CuTR checkpoint (.pth).")
    ap.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    ap.add_argument("--score-thresh", type=float, default=0.25, help="Filter detections by score.")
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path. Default: <image>_inf.json in same folder.",
    )
    ap.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Optional camera fx in pixels (if omitted, uses a default based on image size).",
    )
    ap.add_argument("--fy", type=float, default=None, help="Optional camera fy in pixels.")
    ap.add_argument("--cx", type=float, default=None, help="Optional camera cx in pixels.")
    ap.add_argument("--cy", type=float, default=None, help="Optional camera cy in pixels.")
    ap.add_argument(
        "--max-edge",
        type=int,
        default=1024,
        help="If the image is larger than this on its longest edge, it will be resized down (keeps aspect ratio). Use 0 to disable.",
    )
    ap.add_argument(
        "--depth",
        default=None,
        help="Optional depth image path (UInt16 PNG in millimeters). Required if using an RGB-D model.",
    )
    args = ap.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load checkpoint and detect backbone size + depth modality.
    checkpoint = torch.load(str(model_path), map_location=args.device or "cpu")["model"]
    backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
    is_depth_model = any(k.startswith("backbone.0.patch_embed_depth.") for k in checkpoint.keys())

    model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
    model.load_state_dict(checkpoint)
    model = model.to(args.device)

    max_edge = None if (args.max_edge is None or int(args.max_edge) <= 0) else int(args.max_edge)
    rgb, resize_scale = load_rgb_image(image_path, max_edge=max_edge)
    h, w = int(rgb.shape[1]), int(rgb.shape[2])

    K = make_default_intrinsics(w, h)
    if args.fx is not None:
        K[0, 0] = float(args.fx)
    if args.fy is not None:
        K[1, 1] = float(args.fy)
    if args.cx is not None:
        K[0, 2] = float(args.cx)
    if args.cy is not None:
        K[1, 2] = float(args.cy)

    # If we resized the image and the user provided intrinsics, assume they corresponded
    # to the original image and scale them down accordingly.
    if resize_scale != 1.0 and any(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        K[:2, :] *= float(resize_scale)

    wide = PosedSensorInfo()
    wide.RT = torch.eye(4, dtype=torch.float32)[None]
    wide.T_gravity = torch.eye(3, dtype=torch.float32)[None]
    wide.image = ImageMeasurementInfo(size=(w, h), K=K[None])

    sample = {
        "sensor_info": SensorArrayInfo(wide=wide),
        "wide": {
            "image": rgb[None],  # (1, C, H, W)
        },
        "meta": {
            "video_id": None,
            "timestamp": None,
        },
    }

    if is_depth_model:
        if args.depth is None:
            raise ValueError("This checkpoint is an RGB-D model; pass --depth <path_to_depth_png>.")
        depth_path = Path(args.depth)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth not found: {depth_path}")
        depth_m = load_depth_image_mm_png(depth_path)

        # If depth resolution differs, just resize to RGB for convenience.
        if depth_m.shape[0] != h or depth_m.shape[1] != w:
            depth_img = Image.fromarray(depth_m.numpy())
            depth_img = depth_img.resize((w, h), resample=Image.NEAREST)
            depth_m = torch.from_numpy(np.asarray(depth_img).astype(np.float32))

        wide.depth = DepthMeasurementInfo(size=(w, h), K=K[None])
        sample["wide"]["depth"] = depth_m[None]

    augmentor = Augmentor(("wide/image", "wide/depth") if is_depth_model else ("wide/image",))
    # The default Preprocessor only supports square_pad up to 1024, so for larger images
    # choose a square pad that fits the longest edge and respects size_divisibility.
    longest_edge = max(w, h)
    size_div = 32
    square_pad = int(np.ceil(longest_edge / size_div) * size_div)
    preprocessor = Preprocessor(square_pad=square_pad, size_divisibility=size_div)

    # Match tools/demo.py: move to device BEFORE preprocessing/batching, while we still
    # have per-measurement objects that implement `.to(...)`.
    packaged = augmentor.package(sample)
    packaged = move_input_to_current_device(packaged, model.pixel_mean)
    packaged = preprocessor.preprocess([packaged])

    with torch.no_grad():
        pred_instances = model(packaged)[0]

    pred_instances = pred_instances[pred_instances.scores >= float(args.score_thresh)]

    out_json = Path(args.out_json) if args.out_json else image_path.with_name(_safe_stem(image_path) + "_inf.json")
    save_pred_bboxes_json(pred_instances, image_path=image_path, out_path=out_json)
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()

