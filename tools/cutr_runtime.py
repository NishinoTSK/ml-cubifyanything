"""
CutrRunner: load CubifyAnything CuTR once and run it on arbitrary images.

This refactors the inference pipeline of tools/infer_image.py into a
reusable class so that long-running processes (servers, batch pipelines)
do not pay the model-load cost on every image.

The output schema (`detections`, `boxes_3d`, `image_size_hw`) matches
what tools/infer_image.py writes to its `*_inf.json`, so existing
visualizers keep working.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from cubifyanything.cubify_transformer import make_cubify_transformer
from cubifyanything.measurement import DepthMeasurementInfo, ImageMeasurementInfo
from cubifyanything.preprocessor import (
    Augmentor,
    Preprocessor,
    move_input_to_current_device,
)
from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo


def _to_builtin(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def make_default_intrinsics(w: int, h: int) -> torch.Tensor:
    """Reasonable fallback intrinsics when none are provided."""
    fx = float(max(w, h))
    fy = float(max(w, h))
    cx = float(w) * 0.5
    cy = float(h) * 0.5
    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def load_rgb_image(path: Path, max_edge: Optional[int] = None) -> tuple[torch.Tensor, float]:
    img = Image.open(str(path)).convert("RGB")
    return load_rgb_pil(img, max_edge=max_edge)


def load_rgb_pil(img: Image.Image, max_edge: Optional[int] = None) -> tuple[torch.Tensor, float]:
    img = img.convert("RGB")
    scale = 1.0
    if max_edge is not None and max_edge > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_edge:
            scale = float(max_edge) / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    arr = np.asarray(img).copy()  # HWC uint8 (PIL arrays may be read-only)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return t, scale


def load_depth_image_mm_png(path: Path) -> torch.Tensor:
    img = Image.open(str(path))
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel depth image, got shape={arr.shape}")
    if arr.dtype not in (np.uint16, np.int32, np.int64):
        arr = arr.astype(np.uint16)
    return torch.from_numpy(arr.astype(np.float32)) / 1000.0


class CutrRunner:
    """Holds a loaded CuTR checkpoint and runs inference on demand.

    Parameters
    ----------
    model_path : str | Path
        Path to a CuTR `.pth` checkpoint (RGB or RGB-D).
    device : str
        ``cpu`` | ``cuda`` | ``mps``.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(str(model_path), map_location=device or "cpu")["model"]
        backbone_dim = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
        is_depth_model = any(
            k.startswith("backbone.0.patch_embed_depth.") for k in checkpoint.keys()
        )

        model = make_cubify_transformer(dimension=backbone_dim, depth_model=is_depth_model).eval()
        model.load_state_dict(checkpoint)
        model = model.to(device)

        self.model = model
        self.device = device
        self.is_depth_model = is_depth_model
        self.model_path = model_path

    @torch.no_grad()
    def infer(
        self,
        image: Image.Image | torch.Tensor,
        K: Optional[torch.Tensor | np.ndarray] = None,
        depth_m: Optional[torch.Tensor | np.ndarray] = None,
        score_thresh: float = 0.25,
        max_edge: Optional[int] = 1024,
    ) -> dict:
        """Run inference and return a dict in the same schema as
        ``tools/infer_image.py``'s output JSON (minus ``source_image``)."""

        if isinstance(image, torch.Tensor):
            if image.dtype != torch.uint8 or image.dim() != 3:
                raise ValueError("Tensor image must be CHW uint8.")
            rgb = image
            scale = 1.0
        else:
            rgb, scale = load_rgb_pil(image, max_edge=max_edge)

        h, w = int(rgb.shape[1]), int(rgb.shape[2])

        if K is None:
            K_t = make_default_intrinsics(w, h)
        else:
            if isinstance(K, np.ndarray):
                K_t = torch.from_numpy(K.astype(np.float32))
            else:
                K_t = K.to(torch.float32)
            if scale != 1.0:
                K_t = K_t.clone()
                K_t[:2, :] *= float(scale)

        wide = PosedSensorInfo()
        wide.RT = torch.eye(4, dtype=torch.float32)[None]
        wide.T_gravity = torch.eye(3, dtype=torch.float32)[None]
        wide.image = ImageMeasurementInfo(size=(w, h), K=K_t[None])

        sample = {
            "sensor_info": SensorArrayInfo(wide=wide),
            "wide": {"image": rgb[None]},
            "meta": {"video_id": None, "timestamp": None},
        }

        if self.is_depth_model:
            if depth_m is None:
                raise ValueError("This checkpoint is RGB-D; pass depth_m (in meters).")
            if isinstance(depth_m, np.ndarray):
                depth_t = torch.from_numpy(depth_m.astype(np.float32))
            else:
                depth_t = depth_m.to(torch.float32)
            if depth_t.shape[0] != h or depth_t.shape[1] != w:
                arr = depth_t.numpy() if depth_t.device.type == "cpu" else depth_t.cpu().numpy()
                d_img = Image.fromarray(arr)
                d_img = d_img.resize((w, h), resample=Image.NEAREST)
                depth_t = torch.from_numpy(np.asarray(d_img).astype(np.float32))
            wide.depth = DepthMeasurementInfo(size=(w, h), K=K_t[None])
            sample["wide"]["depth"] = depth_t[None]

        keys = ("wide/image", "wide/depth") if self.is_depth_model else ("wide/image",)
        augmentor = Augmentor(keys)

        longest_edge = max(w, h)
        size_div = 32
        square_pad = int(np.ceil(longest_edge / size_div) * size_div)
        preprocessor = Preprocessor(square_pad=square_pad, size_divisibility=size_div)

        packaged = augmentor.package(sample)
        packaged = move_input_to_current_device(packaged, self.model.pixel_mean)
        packaged = preprocessor.preprocess([packaged])

        pred_instances = self.model(packaged)[0]
        pred_instances = pred_instances[pred_instances.scores >= float(score_thresh)]

        return self._to_dict(pred_instances, image_h=h, image_w=w)

    @staticmethod
    def _to_dict(pred_instances, image_h: int, image_w: int) -> dict:
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

        out: dict = {
            "image_size_hw": [int(image_h), int(image_w)],
            "detections": dets,
        }

        if pred_instances.has("pred_boxes_3d"):
            b3 = pred_instances.pred_boxes_3d
            out["boxes_3d"] = {
                "gravity_center_xyz": _to_builtin(b3.gravity_center),
                "dims_lhw": _to_builtin(b3.dims),
                "R_3x3": _to_builtin(b3.R),
            }

        return out
