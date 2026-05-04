"""Enrich an existing ``room.json`` with semantic labels.

For each object in ``room.json`` we open ``<captures>/<evidence.best_frame>``,
synthesize a single fake CuTR detection from ``evidence.best_bbox``, and run
the shared :class:`tools.labeler.Labeler` to fill in:

- ``label``           : free-form BLIP caption.
- ``category``        : Grounding-DINO closed-set match.
- ``category_score``  : confidence score for ``category``.

Grounding-DINO is run once per *unique image* (so labeling N objects sharing
the same best_frame is much cheaper than N runs). BLIP runs once per object,
on the crop defined by ``best_bbox``.

Usage:
    python tools/label_room.py \
        --room teste/recontruct2/room.json \
        --captures teste/recontruct2 \
        --label-backend both \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeler import Labeler, _bbox_iou  # noqa: E402


def _load_image(captures_dir: Path, frame_name: str) -> Optional[Image.Image]:
    cand = captures_dir / frame_name
    if cand.exists():
        return Image.open(str(cand)).convert("RGB")
    base = Path(frame_name).stem
    for ext in (".png", ".jpg", ".jpeg"):
        p = captures_dir / (base + ext)
        if p.exists():
            return Image.open(str(p)).convert("RGB")
    return None


def label_room(
    room_path: Path,
    captures_dir: Path,
    out_path: Optional[Path],
    backend: str,
    vocab_path: Optional[Path],
    device: str,
    blip_model: str,
    dino_model: str,
    iou_min: float,
    score_thresh_dino: float,
    text_thresh_dino: float,
):
    if not room_path.exists():
        raise FileNotFoundError(room_path)
    if not captures_dir.exists():
        raise FileNotFoundError(captures_dir)

    room = json.loads(room_path.read_text(encoding="utf-8"))
    objects: List[Dict] = room.get("objects", []) or []
    if not objects:
        print(f"No objects in {room_path}, nothing to label.")
        return

    print(
        f"Loading labeler (backend={backend}, blip={blip_model if backend in ('blip','both') else '-'}, "
        f"dino={dino_model if backend in ('dino','both') else '-'}, device={device})..."
    )
    labeler = Labeler(
        backend=backend,
        vocab_path=vocab_path,
        device=device,
        blip_model=blip_model,
        dino_model=dino_model,
        iou_min=float(iou_min),
        score_thresh_dino=float(score_thresh_dino),
        text_thresh_dino=float(text_thresh_dino),
    )

    by_frame: Dict[str, List[int]] = {}
    for idx, obj in enumerate(objects):
        ev = obj.get("evidence") or {}
        frame = ev.get("best_frame")
        bbox = ev.get("best_bbox")
        if not frame or not bbox or len(bbox) != 4:
            obj.setdefault("label", None)
            obj.setdefault("category", None)
            obj.setdefault("category_score", None)
            continue
        by_frame.setdefault(frame, []).append(idx)

    n_labeled = 0
    n_categorized = 0
    for frame, idxs in by_frame.items():
        img = _load_image(captures_dir, frame)
        if img is None:
            print(f"  WARN: image not found for frame={frame}, skipping {len(idxs)} object(s).")
            for idx in idxs:
                objects[idx].setdefault("label", None)
                objects[idx].setdefault("category", None)
                objects[idx].setdefault("category_score", None)
            continue

        # Run DINO once per frame, then attribute by IoU to each object's bbox.
        dino_out = labeler._dino_full_image(img) if labeler.use_dino else None

        for idx in idxs:
            obj = objects[idx]
            bbox = obj["evidence"]["best_bbox"]

            if labeler.use_blip:
                from labeler import _crop_with_padding  # noqa: WPS433
                crop = _crop_with_padding(img, bbox, pad=4)
                cap = labeler._caption_crop(crop) if crop is not None else None
                obj["label"] = cap
                if cap:
                    n_labeled += 1
            else:
                obj.setdefault("label", None)

            if labeler.use_dino and dino_out and dino_out["boxes"]:
                best_iou = -1.0
                best_idx = -1
                for k, b in enumerate(dino_out["boxes"]):
                    iou = _bbox_iou(bbox, b)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = k
                if best_idx >= 0 and best_iou >= labeler.iou_min:
                    obj["category"] = dino_out["labels"][best_idx] or None
                    obj["category_score"] = float(dino_out["scores"][best_idx])
                    if obj["category"]:
                        n_categorized += 1
                else:
                    obj["category"] = None
                    obj["category_score"] = None
            else:
                obj.setdefault("category", None)
                obj.setdefault("category_score", None)

        print(f"  {frame}: labeled {len(idxs)} object(s)")

    out_path = out_path or room_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(room, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\nWrote: {out_path}\n  caption set : {n_labeled}/{len(objects)}\n"
        f"  category set: {n_categorized}/{len(objects)}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Add BLIP/Grounding-DINO labels to room.json objects."
    )
    ap.add_argument("--room", required=True, help="Path to room.json (modified in place by default).")
    ap.add_argument(
        "--captures",
        required=True,
        help="Directory holding the original best_frame images.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path. Default: overwrite --room.",
    )
    ap.add_argument(
        "--label-backend",
        default="both",
        choices=("blip", "dino", "both", "none"),
        help="Which labeling models to load.",
    )
    ap.add_argument("--vocab", default=None, help="Vocab file for Grounding-DINO.")
    ap.add_argument("--device", default="cuda", help="cpu|cuda|mps")
    ap.add_argument("--blip-model", default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--iou-min", type=float, default=0.3)
    ap.add_argument("--score-thresh-dino", type=float, default=0.25)
    ap.add_argument("--text-thresh-dino", type=float, default=0.20)
    args = ap.parse_args()

    label_room(
        room_path=Path(args.room),
        captures_dir=Path(args.captures),
        out_path=Path(args.out) if args.out else None,
        backend=args.label_backend,
        vocab_path=Path(args.vocab) if args.vocab else None,
        device=args.device,
        blip_model=args.blip_model,
        dino_model=args.dino_model,
        iou_min=float(args.iou_min),
        score_thresh_dino=float(args.score_thresh_dino),
        text_thresh_dino=float(args.text_thresh_dino),
    )


if __name__ == "__main__":
    main()
