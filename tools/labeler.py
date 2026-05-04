"""Shared labeling backend for CuTR detections.

Loads BLIP (image captioning) and/or Grounding-DINO (open-vocabulary
detection) once, then enriches CuTR detections in-place with:

- ``label`` : free-form caption from BLIP per crop (e.g. "a wooden chair").
- ``category`` + ``category_score`` : closed-set category from Grounding-DINO,
  matched to the CuTR bbox by 2D IoU (e.g. "chair", 0.78).

Both models are loaded once in ``Labeler.__init__`` and reused across calls,
so this is suitable for long-lived processes (servers, batch pipelines) and
for one-shot CLI scripts alike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent / "labeling_vocab_default.txt"


def load_vocab(path: Optional[Path] = None) -> List[str]:
    """Load a vocabulary file; one class per line, blanks/`#` comments ignored."""
    p = Path(path) if path else DEFAULT_VOCAB_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Vocab file not found: {p}. Pass --vocab explicitly or create it."
        )
    seen = set()
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s_norm = s.lower()
        if s_norm in seen:
            continue
        seen.add(s_norm)
        out.append(s_norm)
    if not out:
        raise ValueError(f"Vocab file {p} is empty after stripping comments.")
    return out


def vocab_to_prompt(vocab: Sequence[str]) -> str:
    """Build the Grounding-DINO prompt: 'class1. class2. class3.'."""
    return ". ".join(vocab) + "."


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _crop_with_padding(img: Image.Image, bbox: Sequence[float], pad: int = 4) -> Optional[Image.Image]:
    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1 = max(0, int(round(float(x1))) - pad)
    y1 = max(0, int(round(float(y1))) - pad)
    x2 = min(w, int(round(float(x2))) + pad)
    y2 = min(h, int(round(float(y2))) + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img.crop((x1, y1, x2, y2))


class Labeler:
    """Open-vocab labeler combining BLIP captions and Grounding-DINO categories.

    Parameters
    ----------
    backend : ``"blip" | "dino" | "both" | "none"``
        Which models to load and run. ``"none"`` disables labeling.
    vocab_path : optional path
        File with the closed-set vocabulary for Grounding-DINO. Defaults to
        ``tools/labeling_vocab_default.txt``.
    device : ``"cpu" | "cuda" | "mps"``.
    blip_model, dino_model : HuggingFace model ids.
    iou_min : float
        Minimum IoU between a CuTR bbox and a Grounding-DINO output for the
        DINO label to be accepted as the CuTR detection's category.
    score_thresh_dino, text_thresh_dino : float
        Grounding-DINO post-processing thresholds.
    """

    def __init__(
        self,
        backend: str = "both",
        vocab_path: Optional[Path] = None,
        device: str = "cuda",
        blip_model: str = "Salesforce/blip-image-captioning-base",
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        iou_min: float = 0.3,
        score_thresh_dino: float = 0.25,
        text_thresh_dino: float = 0.20,
    ):
        if backend not in ("blip", "dino", "both", "none"):
            raise ValueError(
                f"backend must be one of 'blip'|'dino'|'both'|'none', got {backend!r}"
            )
        self.backend = backend
        self.use_blip = backend in ("blip", "both")
        self.use_dino = backend in ("dino", "both")
        self.device = device
        self.iou_min = float(iou_min)
        self.score_thresh_dino = float(score_thresh_dino)
        self.text_thresh_dino = float(text_thresh_dino)

        self._blip_proc = None
        self._blip_model = None
        self._dino_proc = None
        self._dino_model = None

        self.vocab: List[str] = []
        self.vocab_prompt: str = ""

        if self.use_dino:
            self.vocab = load_vocab(vocab_path)
            self.vocab_prompt = vocab_to_prompt(self.vocab)
            self._load_dino(dino_model)

        if self.use_blip:
            self._load_blip(blip_model)

    def _load_blip(self, model_id: str):
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for the BLIP backend. "
                "Install with: pip install transformers accelerate"
            ) from e
        self._blip_proc = BlipProcessor.from_pretrained(model_id)
        m = BlipForConditionalGeneration.from_pretrained(model_id)
        self._blip_model = m.to(self.device).eval()

    def _load_dino(self, model_id: str):
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for the Grounding-DINO backend. "
                "Install with: pip install transformers accelerate"
            ) from e
        self._dino_proc = AutoProcessor.from_pretrained(model_id)
        m = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._dino_model = m.to(self.device).eval()

    @torch.no_grad()
    def _caption_crop(self, crop: Image.Image) -> Optional[str]:
        if self._blip_model is None or self._blip_proc is None:
            return None
        inputs = self._blip_proc(images=crop, return_tensors="pt").to(self.device)
        out = self._blip_model.generate(**inputs, max_new_tokens=24)
        text = self._blip_proc.decode(out[0], skip_special_tokens=True)
        text = text.strip()
        return text or None

    def _chunk_vocab(self, max_tokens: int = 240) -> List[List[str]]:
        """Split ``self.vocab`` so each chunk's prompt fits within Grounding-DINO's
        text-token budget (default 256).

        We greedily add classes until the tokenized prompt would exceed
        ``max_tokens`` and then start a new chunk. The default 240 leaves a
        small safety margin under the 256-token limit imposed by the model.
        """
        if not self.vocab:
            return []
        if self._dino_proc is None:
            return [list(self.vocab)]

        tokenizer = getattr(self._dino_proc, "tokenizer", None)
        if tokenizer is None:
            return [list(self.vocab)]

        chunks: List[List[str]] = []
        cur: List[str] = []
        for cls in self.vocab:
            trial = cur + [cls]
            prompt = vocab_to_prompt(trial)
            ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
            if int(ids.shape[1]) > max_tokens and cur:
                chunks.append(cur)
                cur = [cls]
            else:
                cur = trial
        if cur:
            chunks.append(cur)
        return chunks

    @torch.no_grad()
    def _dino_chunk(self, image: Image.Image, chunk_vocab: List[str]) -> Dict[str, list]:
        """Run Grounding-DINO on the full image with one vocab chunk."""
        empty: Dict[str, list] = {"scores": [], "labels": [], "boxes": []}
        if not chunk_vocab or self._dino_model is None or self._dino_proc is None:
            return empty

        w, h = image.size
        prompt = vocab_to_prompt(chunk_vocab)
        inputs = self._dino_proc(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        outputs = self._dino_model(**inputs)

        kw = dict(
            outputs=outputs,
            input_ids=inputs.input_ids,
            text_threshold=self.text_thresh_dino,
            target_sizes=[(h, w)],
        )
        try:
            results = self._dino_proc.post_process_grounded_object_detection(
                **kw, threshold=self.score_thresh_dino
            )[0]
        except TypeError:
            results = self._dino_proc.post_process_grounded_object_detection(
                **kw, box_threshold=self.score_thresh_dino
            )[0]

        labels_key = "text_labels" if "text_labels" in results else "labels"
        raw_labels = results.get(labels_key, [])

        out_labels: List[str] = []
        for lab in raw_labels:
            if isinstance(lab, str):
                out_labels.append(lab)
            else:
                try:
                    out_labels.append(str(lab))
                except Exception:
                    out_labels.append("")

        scores = results.get("scores")
        boxes = results.get("boxes")
        out_scores = (
            [float(s) for s in scores.detach().cpu().tolist()]
            if hasattr(scores, "detach")
            else [float(s) for s in (scores or [])]
        )
        out_boxes = (
            [list(map(float, b)) for b in boxes.detach().cpu().tolist()]
            if hasattr(boxes, "detach")
            else [list(map(float, b)) for b in (boxes or [])]
        )

        return {"scores": out_scores, "labels": out_labels, "boxes": out_boxes}

    def _dino_full_image(self, image: Image.Image) -> Dict[str, list]:
        """Run Grounding-DINO on the full image, splitting the vocab across
        multiple calls when it would exceed the model's text-token budget.
        Boxes are in original-image pixel space (xyxy).
        """
        empty: Dict[str, list] = {"scores": [], "labels": [], "boxes": []}
        if self._dino_model is None or self._dino_proc is None or not self.vocab:
            return empty

        chunks = self._chunk_vocab()
        if not chunks:
            return empty

        merged: Dict[str, list] = {"scores": [], "labels": [], "boxes": []}
        for chunk in chunks:
            out = self._dino_chunk(image, chunk)
            merged["scores"].extend(out["scores"])
            merged["labels"].extend(out["labels"])
            merged["boxes"].extend(out["boxes"])
        return merged

    def label_detections(
        self,
        image: Image.Image,
        detections: List[Dict],
    ) -> List[Dict]:
        """Enrich each detection in place with label / category / category_score.

        Returns the same list (also modified in place) for chaining.
        """
        if not detections or self.backend == "none":
            for det in detections or []:
                det.setdefault("label", None)
                det.setdefault("category", None)
                det.setdefault("category_score", None)
            return detections

        rgb = image.convert("RGB")

        dino_out = self._dino_full_image(rgb) if self.use_dino else None

        for det in detections:
            bbox = det.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                det.setdefault("label", None)
                det.setdefault("category", None)
                det.setdefault("category_score", None)
                continue

            if self.use_blip:
                crop = _crop_with_padding(rgb, bbox, pad=4)
                det["label"] = self._caption_crop(crop) if crop is not None else None
            else:
                det.setdefault("label", None)

            if self.use_dino and dino_out and dino_out["boxes"]:
                best_iou = -1.0
                best_idx = -1
                for k, b in enumerate(dino_out["boxes"]):
                    iou = _bbox_iou(bbox, b)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = k
                if best_idx >= 0 and best_iou >= self.iou_min:
                    det["category"] = dino_out["labels"][best_idx] or None
                    det["category_score"] = float(dino_out["scores"][best_idx])
                else:
                    det["category"] = None
                    det["category_score"] = None
            else:
                det.setdefault("category", None)
                det.setdefault("category_score", None)

        return detections
