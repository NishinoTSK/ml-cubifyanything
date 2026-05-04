"""CLI wrapper around CutrRunner for one-shot inference on a single image."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cutr_runtime import (  # noqa: E402
    CutrRunner,
    load_depth_image_mm_png,
    make_default_intrinsics,
)

_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")


def _safe_stem(p: Path):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in p.stem)


def load_meta_json(meta_path: Path) -> dict:
    """Read a Quest/Unity passthrough JSON, tolerating pt-BR comma decimals."""
    raw = meta_path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = _DECIMAL_COMMA.sub(".", raw)
        meta = json.loads(fixed)
        print(
            f"WARN: {meta_path.name} had comma decimals; auto-fixed in memory. "
            f"Run tools/fix_decimal_commas.py on the source folder for a permanent fix."
        )
        return meta


def _meta_intrinsic(meta: Optional[dict], key: str) -> Optional[float]:
    if not meta or key not in meta:
        return None
    try:
        return float(meta[key])
    except (TypeError, ValueError):
        return None


def save_pred_json(payload: dict, image_path: Path, out_path: Path):
    payload = {
        "source_image": str(image_path),
        "timestamp": None,
        "video_id": None,
        **payload,
    }
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
    ap.add_argument("--fx", type=float, default=None, help="Optional camera fx in pixels.")
    ap.add_argument("--fy", type=float, default=None, help="Optional camera fy in pixels.")
    ap.add_argument("--cx", type=float, default=None, help="Optional camera cx in pixels.")
    ap.add_argument("--cy", type=float, default=None, help="Optional camera cy in pixels.")
    ap.add_argument(
        "--meta-json",
        default=None,
        help="Optional path to a passthrough JSON with fx/fy/cx/cy (Unity/Quest "
             "capture). Explicit --fx/--fy/--cx/--cy still take precedence.",
    )
    ap.add_argument(
        "--max-edge",
        type=int,
        default=1024,
        help="Resize so the longest edge fits this. Use 0 to disable.",
    )
    ap.add_argument(
        "--depth",
        default=None,
        help="Optional depth image path (UInt16 PNG in millimeters). Required for RGB-D models.",
    )
    ap.add_argument(
        "--label",
        action="store_true",
        help="After CuTR, run labeling backend to enrich each detection with label/category.",
    )
    ap.add_argument(
        "--label-backend",
        default="both",
        choices=(
            "blip", "dino", "owlv2", "yolo",
            "both", "both_owl", "both_yolo", "all", "none",
        ),
        help="Which labeling models to load. Default: both (BLIP+DINO).",
    )
    ap.add_argument(
        "--vocab",
        default=None,
        help="Path to vocab file (one class per line) for DINO/OWL-ViT. "
        "Default: tools/labeling_vocab_default.txt",
    )
    ap.add_argument(
        "--blip-model",
        default="Salesforce/blip-image-captioning-base",
        help="HF model id for BLIP captioning.",
    )
    ap.add_argument(
        "--dino-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="HF model id for Grounding-DINO.",
    )
    ap.add_argument(
        "--owlv2-model",
        default="google/owlv2-base-patch16-ensemble",
        help="HF model id for OWL-ViT v2.",
    )
    ap.add_argument(
        "--yolo-model",
        default="yolov8l-worldv2.pt",
        help="Ultralytics checkpoint for YOLO-World (e.g. yolov8l-worldv2.pt).",
    )
    ap.add_argument(
        "--iou-min",
        type=float,
        default=0.3,
        help="Min IoU for matching CuTR bbox to a detector output (DINO/OWL-ViT).",
    )
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    meta: Optional[dict] = None
    if args.meta_json:
        meta_path = Path(args.meta_json)
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta JSON not found: {meta_path}")
        meta = load_meta_json(meta_path)

    runner = CutrRunner(model_path=args.model_path, device=args.device)

    img = Image.open(str(image_path)).convert("RGB")
    w_orig, h_orig = img.size

    fx = args.fx if args.fx is not None else _meta_intrinsic(meta, "fx")
    fy = args.fy if args.fy is not None else _meta_intrinsic(meta, "fy")
    cx = args.cx if args.cx is not None else _meta_intrinsic(meta, "cx")
    cy = args.cy if args.cy is not None else _meta_intrinsic(meta, "cy")

    K_user = None
    if any(v is not None for v in (fx, fy, cx, cy)):
        K_user = make_default_intrinsics(w_orig, h_orig)
        if fx is not None:
            K_user[0, 0] = float(fx)
        if fy is not None:
            K_user[1, 1] = float(fy)
        if cx is not None:
            K_user[0, 2] = float(cx)
        if cy is not None:
            K_user[1, 2] = float(cy)
        src = "cli" if any(v is not None for v in (args.fx, args.fy, args.cx, args.cy)) and meta is None else \
              "meta" if all(v is None for v in (args.fx, args.fy, args.cx, args.cy)) and meta is not None else \
              "cli+meta"
        print(
            f"Intrinsics ({src}): fx={fx}, fy={fy}, cx={cx}, cy={cy} "
            f"(image {w_orig}x{h_orig})"
        )

    depth_m = None
    if runner.is_depth_model:
        if args.depth is None:
            raise ValueError("This checkpoint is RGB-D; pass --depth <path_to_depth_png>.")
        depth_path = Path(args.depth)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth not found: {depth_path}")
        depth_m = load_depth_image_mm_png(depth_path)

    max_edge = None if (args.max_edge is None or int(args.max_edge) <= 0) else int(args.max_edge)

    pred = runner.infer(
        image=img,
        K=K_user,
        depth_m=depth_m,
        score_thresh=float(args.score_thresh),
        max_edge=max_edge,
    )

    if args.label and pred.get("detections"):
        from labeler import Labeler  # noqa: E402

        ih, iw = pred["image_size_hw"]
        # bbox_xyxy is in image_size_hw pixel space; resize the original image to
        # match before captioning so BLIP/DINO see the same coordinates as CuTR.
        if (img.size[1], img.size[0]) != (int(ih), int(iw)):
            img_for_label = img.resize((int(iw), int(ih)), resample=Image.BILINEAR)
        else:
            img_for_label = img

        vocab_path = Path(args.vocab) if args.vocab else None
        labeler = Labeler(
            backend=args.label_backend,
            vocab_path=vocab_path,
            device=args.device,
            blip_model=args.blip_model,
            dino_model=args.dino_model,
            owlv2_model=args.owlv2_model,
            yolo_model=args.yolo_model,
            iou_min=float(args.iou_min),
        )
        labeler.label_detections(img_for_label, pred["detections"])
        n_cat = sum(1 for d in pred["detections"] if d.get("category"))
        n_cap = sum(1 for d in pred["detections"] if d.get("label"))
        print(
            f"Labeling done ({args.label_backend}): {n_cat}/{len(pred['detections'])} with category, "
            f"{n_cap}/{len(pred['detections'])} with caption."
        )

    out_json = (
        Path(args.out_json) if args.out_json
        else image_path.with_name(_safe_stem(image_path) + "_inf.json")
    )
    save_pred_json(pred, image_path=image_path, out_path=out_json)
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
