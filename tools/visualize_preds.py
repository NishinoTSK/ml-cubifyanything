import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _default_font():
    # Use a basic font fallback that works cross-platform.
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _color_for_class(class_id: int):
    # Deterministic palette.
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
    return palette[class_id % len(palette)]


def draw_predictions(
    image: Image.Image,
    preds: dict,
    score_thresh: float = 0.0,
    show_labels: bool = True,
    line_width: int = 3,
):
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _default_font()

    w, h = img.size
    dets = preds.get("detections", []) or []

    for det in dets:
        score = det.get("score", None)
        if score is not None and float(score) < float(score_thresh):
            continue

        bbox = det.get("bbox_xyxy", None)
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        # Clamp to image bounds.
        x1 = _clamp(x1, 0.0, w - 1.0)
        y1 = _clamp(y1, 0.0, h - 1.0)
        x2 = _clamp(x2, 0.0, w - 1.0)
        y2 = _clamp(y2, 0.0, h - 1.0)

        class_id = det.get("class_id", 0)
        try:
            class_id = int(class_id)
        except Exception:
            class_id = 0

        color = _color_for_class(class_id)

        # Rectangle (draw multiple times for thickness for broad PIL compat).
        for i in range(max(1, int(line_width))):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        if show_labels:
            caption = det.get("label") or det.get("category") or f"cls={class_id}"
            label = str(caption)
            if score is not None:
                label += f" {float(score):.2f}"

            # Text background
            if font is not None:
                # textbbox exists in newer PIL; fallback if unavailable
                try:
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = draw.textsize(label, font=font)
            else:
                tw, th = (len(label) * 6, 11)

            tx = x1
            ty = max(0.0, y1 - th - 4)
            pad = 2
            draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=color)
            draw.text((tx + pad, ty + pad), label, fill=(0, 0, 0), font=font)

    return img


def main():
    ap = argparse.ArgumentParser(description="Visualize CuTR predicted 2D boxes saved by tools/demo.py --save-preds-dir")
    ap.add_argument("--image", required=True, help="Path to an image file (png/jpg).")
    ap.add_argument("--pred-json", required=True, help="Path to the per-frame JSON saved by --save-preds-dir.")
    ap.add_argument("--out", default=None, help="Output image path. Default: <image>_inf.png")
    ap.add_argument("--score-thresh", type=float, default=0.0, help="Filter detections below this score.")
    ap.add_argument("--no-labels", action="store_true", help="Do not render labels.")
    ap.add_argument("--line-width", type=int, default=3, help="Box line width in pixels.")
    args = ap.parse_args()

    img_path = Path(args.image)
    pred_path = Path(args.pred_json)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred JSON not found: {pred_path}")

    img = Image.open(str(img_path))
    preds = _load_json(str(pred_path))

    out_img = draw_predictions(
        img,
        preds,
        score_thresh=args.score_thresh,
        show_labels=not args.no_labels,
        line_width=args.line_width,
    )

    out_path = Path(args.out) if args.out else img_path.with_name(img_path.stem + "_inf.png")
    os.makedirs(out_path.parent, exist_ok=True)
    out_img.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

