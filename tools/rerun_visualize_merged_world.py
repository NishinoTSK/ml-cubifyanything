"""Visualize a world-merged 3D boxes JSON in Rerun.

Auto-detects between two schemas:

1. Legacy: tools/merge_preds_world.py output, with key
   ``merged_boxes_3d.{gravity_center_xyz, dims_lhw, R_3x3, scores}``.
2. New room map: tools/scan_pipeline.py output, with key
   ``objects[*]`` and ``world_frame.convention``.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rerun
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


def parse_room_json(js: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], str]:
    """Return centers, sizes, Rmats, scores, labels, convention for the new schema.

    `labels` is None when no object has a meaningful `label` field — caller
    decides whether to display them.
    """
    objects = js.get("objects") or []
    if not objects:
        raise ValueError("room.json has empty 'objects'")
    centers = np.asarray([o["center_xyz"] for o in objects], dtype=np.float32)
    sizes = np.asarray([o["dims_lhw"] for o in objects], dtype=np.float32)
    Rmats = np.asarray([o["R_3x3"] for o in objects], dtype=np.float32)
    scores = np.asarray([float(o.get("score", 0.0)) for o in objects], dtype=np.float32)

    has_real_labels = any(o.get("label") for o in objects)
    if has_real_labels:
        labels: Optional[List[str]] = []
        for o in objects:
            lab = o.get("label")
            s = float(o.get("score", 0.0))
            labels.append(f"{lab} ({s:.2f})" if lab else f"({s:.2f})")
    else:
        labels = None

    convention = (js.get("world_frame") or {}).get("convention", "unity_left_handed_y_up")
    return centers, sizes, Rmats, scores, labels, convention


def parse_legacy_json(js: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], str]:
    """Return tuple for the legacy merge_preds_world.py schema."""
    mb = js.get("merged_boxes_3d")
    if mb is None:
        raise ValueError("Missing both 'objects' (room map) and 'merged_boxes_3d' (legacy)")
    centers = _as_np(mb.get("gravity_center_xyz"), dtype=np.float32)
    sizes = _as_np(mb.get("dims_lhw"), dtype=np.float32)
    Rmats = _as_np(mb.get("R_3x3"), dtype=np.float32)
    scores = _as_np(mb.get("scores"), dtype=np.float32)
    if centers is None or sizes is None or Rmats is None:
        raise ValueError("merged_boxes_3d must contain gravity_center_xyz, dims_lhw, R_3x3")
    labels = None
    if scores is not None and len(scores) == len(centers):
        labels = [f"{float(s):.3f}" for s in scores]
    return centers, sizes, Rmats, scores if scores is not None else np.zeros(len(centers)), labels, "opencv_right_handed_y_down"


def view_coords_for(convention: str):
    """Pick a reasonable Rerun view-coords preset for the world frame."""
    if convention.startswith("unity"):
        return rerun.ViewCoordinates.LEFT_HAND_Y_UP
    return rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN


def main():
    ap = argparse.ArgumentParser(description="Visualize merged-world or room-map JSON in Rerun.")
    ap.add_argument(
        "--merged-json",
        default=None,
        help="Legacy merge_preds_world.py JSON. Same as --room-json — kept for back-compat.",
    )
    ap.add_argument(
        "--room-json",
        default=None,
        help="Path to a room.json (scan_pipeline.py output) OR legacy merged JSON.",
    )
    ap.add_argument("--application-id", default="room_map", help="Rerun application id")
    ap.add_argument("--rrd-out", default=None, help="Optional .rrd output path. Default: <input>.rrd")
    label_grp = ap.add_mutually_exclusive_group()
    label_grp.add_argument(
        "--show-labels",
        action="store_true",
        help="Force labels on. Default shows labels only when objects have a non-null 'label' field.",
    )
    label_grp.add_argument(
        "--hide-labels",
        action="store_true",
        help="Force labels off (default behavior when 'label' fields are null).",
    )
    args = ap.parse_args()

    in_path = args.room_json or args.merged_json
    if not in_path:
        raise SystemExit("Pass --room-json <path> (or legacy --merged-json <path>).")

    src_path = Path(in_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {src_path}")

    js = _load_json(src_path)

    if "objects" in js:
        centers, sizes, Rmats, scores, labels, convention = parse_room_json(js)
        print(f"Detected schema: room.json ({len(centers)} objects, convention={convention})")
    else:
        centers, sizes, Rmats, scores, labels, convention = parse_legacy_json(js)
        print(f"Detected schema: legacy merged_boxes_3d ({len(centers)} boxes)")

    if len(centers) != len(sizes) or len(centers) != len(Rmats):
        raise ValueError("Inconsistent lengths between centers / sizes / Rmats.")

    recording = rerun.new_recording(
        application_id=str(args.application_id),
        recording_id=uuid.uuid4(),
        make_default=True,
    )
    rerun.spawn()

    rerun.log("/world", view_coords_for(convention), static=True)

    if args.hide_labels:
        show_labels = False
    elif args.show_labels:
        show_labels = labels is not None
    else:
        show_labels = labels is not None

    quats = Rotation.from_matrix(Rmats).as_quat().astype(np.float32)  # xyzw
    rerun.log(
        "/world/objects",
        rerun.Boxes3D(
            centers=centers,
            sizes=sizes,
            quaternions=[rerun.Quaternion(xyzw=q) for q in quats],
            labels=labels if show_labels else None,
            show_labels=show_labels,
        ),
        recording=recording,
    )

    out_path = Path(args.rrd_out) if args.rrd_out else src_path.with_suffix(".rrd")
    if out_path.suffix.lower() != ".rrd":
        out_path = out_path.with_suffix(".rrd")

    if hasattr(rerun, "save"):
        rerun.save(str(out_path))  # type: ignore[attr-defined]
    elif hasattr(recording, "save"):
        recording.save(str(out_path))  # type: ignore[attr-defined]
    else:
        raise RuntimeError("This rerun version does not support saving .rrd via this script.")

    print(f"Saved RRD: {out_path}")


if __name__ == "__main__":
    main()
