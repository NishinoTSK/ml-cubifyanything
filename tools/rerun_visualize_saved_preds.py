import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import rerun
from PIL import Image
from scipy.spatial.transform import Rotation


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_np(x, dtype=np.float32):
    if x is None:
        return None
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _color_for_index(i: int):
    # Deterministic palette (RGB uint8).
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


def main():
    ap = argparse.ArgumentParser(
        description="Offline Rerun visualization for predictions saved by tools/demo.py --save-preds-dir"
    )
    ap.add_argument("--image", required=True, help="Path to image.png for the frame.")
    ap.add_argument("--pred-json", required=True, help="Path to the saved per-frame prediction JSON.")
    ap.add_argument(
        "--rrd-out",
        default=None,
        help="Optional path to save an .rrd recording. Default: <image>_inf.rrd",
    )
    ap.add_argument(
        "--application-id",
        default=None,
        help="Optional Rerun application id (defaults to <video_id>).",
    )
    ap.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show text on 2D strips and 3D boxes (captions, categories, or cls/score). "
        "Use --no-labels to draw only the geometry.",
    )
    ap.add_argument(
        "--category-from",
        default="category",
        choices=("category", "category_dino", "category_owlv2", "category_yolo", "label"),
        help="Which field to display. Default: category (legacy canonical). "
             "Use 'label' to show the BLIP free-form caption.",
    )
    args = ap.parse_args()

    img_path = Path(args.image)
    pred_path = Path(args.pred_json)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred JSON not found: {pred_path}")

    preds = _load_json(pred_path)
    video_id = preds.get("video_id", "preds")
    timestamp = float(preds.get("timestamp", 0.0) or 0.0)
    app_id = args.application_id or str(video_id)

    recording_id = uuid.uuid4()
    recording = rerun.new_recording(
        application_id=app_id,
        recording_id=recording_id,
        make_default=True,
    )
    rerun.spawn()

    # Keep consistent with tools/demo.py
    rerun.log("/world", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rerun.set_time_seconds("pts", timestamp, recording=recording)

    # Log image.
    image = np.array(Image.open(str(img_path)).convert("RGB"))
    rerun.log("/device/wide/image", rerun.Image(image).compress(), recording=recording)

    # Log 2D boxes (XYXY).
    dets = preds.get("detections", []) or []
    boxes2d = []
    labels = []
    labels_3d = []
    colors = []
    any_text_label = False
    cat_key = f"category_{args.category_from}" if args.category_from in ("dino", "owlv2", "yolo") else "category"
    for i, det in enumerate(dets):
        bbox = det.get("bbox_xyxy", None)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        boxes2d.append([x1, y1, x2, y2])
        score = det.get("score", None)
        class_id = det.get("class_id", None)
        if args.category_from == "label":
            text_label = det.get("label")
        else:
            text_label = det.get(cat_key) or det.get("category")
        if text_label:
            any_text_label = True
            score_str = f" {float(score):.2f}" if score is not None else ""
            labels.append(f"{text_label}{score_str}")
            labels_3d.append(str(text_label))
        elif score is None:
            labels.append("" if class_id is None else f"{class_id}")
            labels_3d.append("" if class_id is None else f"cls={class_id}")
        else:
            labels.append(
                f"{class_id} {float(score):.3f}"
                if class_id is not None
                else f"{float(score):.3f}"
            )
            labels_3d.append(
                f"cls={class_id} {float(score):.2f}"
                if class_id is not None
                else f"{float(score):.2f}"
            )

        colors.append(list(_color_for_index(i)))

    if not args.labels:
        labels = [""] * len(labels)
        labels_3d = [""] * len(labels_3d)
        any_text_label = False

    if boxes2d:
        # Rerun SDK APIs differ across versions; LineStrips2D works reliably on older versions (e.g. 0.19.x).
        rects = []
        for x1, y1, x2, y2 in boxes2d:
            rects.append(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                    [x1, y1],
                ]
            )

        rerun.log(
            # Log under the image entity so it overlays in the 2D view.
            "/device/wide/image/pred_boxes_2d",
            rerun.LineStrips2D(
                _as_np(rects, dtype=np.float32),
                colors=colors,
                labels=labels,
                radii=1.5,
            ),
            recording=recording,
        )

    # Log 3D boxes if available.
    boxes3d = preds.get("boxes_3d", None)
    if boxes3d:
        centers = _as_np(boxes3d.get("gravity_center_xyz"), dtype=np.float32)
        sizes = _as_np(boxes3d.get("dims_lhw"), dtype=np.float32)
        Rmats = _as_np(boxes3d.get("R_3x3"), dtype=np.float32)

        if centers is not None and sizes is not None and Rmats is not None and len(centers) == len(sizes) == len(Rmats):
            quats_xyzw = Rotation.from_matrix(Rmats).as_quat().astype(np.float32)
            box3d_kwargs = dict(
                centers=centers,
                sizes=sizes,
                quaternions=[rerun.Quaternion(xyzw=q) for q in quats_xyzw],
            )
            if any_text_label and len(labels_3d) == len(centers):
                box3d_kwargs["labels"] = labels_3d
                box3d_kwargs["show_labels"] = True
            else:
                box3d_kwargs["show_labels"] = False
            rerun.log(
                "/device/wide/pred_instances",
                rerun.Boxes3D(**box3d_kwargs),
                recording=recording,
            )

    # Always save an .rrd as well (useful for WSL/Windows workflows).
    out_path = Path(args.rrd_out) if args.rrd_out else img_path.with_name(img_path.stem + "_inf.rrd")
    if out_path.suffix.lower() != ".rrd":
        out_path = out_path.with_suffix(".rrd")

    # Rerun APIs vary by version; try the common variants.
    out = str(out_path)
    try:
        if hasattr(rerun, "save"):
            rerun.save(out)  # type: ignore[attr-defined]
        elif hasattr(recording, "save"):
            recording.save(out)  # type: ignore[attr-defined]
        else:
            raise RuntimeError("This rerun version does not support saving .rrd via this script.")
    finally:
        print(f"Saved RRD: {out}")


if __name__ == "__main__":
    main()

