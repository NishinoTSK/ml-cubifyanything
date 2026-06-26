"""Run CuTR inference + visualization on every image in a folder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cutr_runtime import CutrRunner  # noqa: E402
from infer_image import _safe_stem, save_pred_json  # noqa: E402
from visualize_preds import draw_predictions  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_images(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    images: list[Path] = []
    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.endswith("_inf"):
            continue
        images.append(path)
    return images


def pred_json_path(image_path: Path) -> Path:
    return image_path.with_name(_safe_stem(image_path) + "_inf.json")


def pred_png_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + "_inf.png")


def main():
    ap = argparse.ArgumentParser(
        description="Run infer_image + visualize_preds on all images in a folder."
    )
    ap.add_argument(
        "--folder",
        required=True,
        help="Directory containing images (png/jpg/jpeg/...).",
    )
    ap.add_argument(
        "--model-path",
        required=True,
        help="Path to CuTR checkpoint (.pth).",
    )
    ap.add_argument("--device", default="cuda", help="cpu|cuda|mps")
    ap.add_argument(
        "--max-edge",
        type=int,
        default=0,
        help="Resize so the longest edge fits this. Use 0 to disable.",
    )
    ap.add_argument(
        "--score-thresh",
        type=float,
        default=0.35,
        help="Filter detections by score (inference + visualization).",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Also process images in subfolders.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose *_inf.json already exists.",
    )
    ap.add_argument(
        "--infer-only",
        action="store_true",
        help="Only write *_inf.json, skip visualization.",
    )
    ap.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only render *_inf.png from existing *_inf.json files.",
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = find_images(folder, recursive=args.recursive)
    if not images:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(images)} image(s) in {folder}")

    runner = None
    max_edge = None if int(args.max_edge) <= 0 else int(args.max_edge)

    if not args.visualize_only:
        runner = CutrRunner(model_path=args.model_path, device=args.device)

    ok = 0
    skipped = 0
    failed = 0

    for image_path in images:
        out_json = pred_json_path(image_path)
        out_png = pred_png_path(image_path)

        try:
            if args.visualize_only:
                if not out_json.exists():
                    print(f"SKIP (no JSON): {image_path.name}")
                    skipped += 1
                    continue
            elif args.skip_existing and out_json.exists():
                print(f"SKIP (exists): {image_path.name}")
                skipped += 1
                continue

            if not args.visualize_only:
                img = Image.open(str(image_path)).convert("RGB")
                pred = runner.infer(
                    image=img,
                    K=None,
                    depth_m=None,
                    score_thresh=float(args.score_thresh),
                    max_edge=max_edge,
                )
                save_pred_json(pred, image_path=image_path, out_path=out_json)
                print(f"Inferred: {image_path.name} -> {out_json.name}")

            if not args.infer_only:
                if args.visualize_only:
                    img = Image.open(str(image_path)).convert("RGB")
                    preds = json.loads(out_json.read_text(encoding="utf-8"))
                else:
                    preds = pred
                out_img = draw_predictions(
                    img, preds, score_thresh=float(args.score_thresh)
                )
                out_img.save(str(out_png))
                print(f"Visualized: {image_path.name} -> {out_png.name}")

            ok += 1
        except Exception as exc:
            failed += 1
            print(f"ERROR {image_path.name}: {exc}", file=sys.stderr)

    print(f"Done: {ok} ok, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
