"""
Mask generation for real photographs.

Two jobs:

  1. TARGET MASK from a user box. The box resolves which instance is meant --
     the whole reason the box input exists -- and SAM turns it into an object
     mask instead of a rectangle.

  2. OTHER-INSTANCE MASKS for the leakage metric. Neither PIE-Bench nor
     MAG-Bench ships masks for the same-category objects the user did NOT
     select, so the metric that measures this project's motivating failure
     cannot run without generating them.

Nothing here has been executed -- no SAM weights or GPU in the environment
where it was written. `validate_against_gt` exists because of that: run it
against the 171 real PIE-Bench++ masks before trusting any generated mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .masking import clean_mask, select_instance

SAM_CHECKPOINTS = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


class SAMSegmenter:
    """Box -> candidate masks. Satisfies the Segmenter protocol in masking.py.

    Returns ALL candidates rather than SAM's top pick, because `select_instance`
    scores them against the box: SAM's own ranking optimises mask quality, not
    "is this the object the user boxed", and it will happily return a part
    (the apple's highlight) or a whole (the entire fruit bowl).
    """

    def __init__(self, checkpoint: str | Path, model_type: str = "vit_h", device="cuda"):
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device)
        self.predictor = SamPredictor(sam)
        self.sam = sam
        self.device = device
        self._image_id = None

    def set_image(self, image: np.ndarray):
        """Embedding is the expensive part; cache it across several boxes."""
        key = id(image)
        if key != self._image_id:
            self.predictor.set_image(np.asarray(image, np.uint8))
            self._image_id = key

    def __call__(self, image: np.ndarray, box: tuple) -> np.ndarray:
        self.set_image(image)
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(box, np.float32)[None], multimask_output=True
        )
        return masks.astype(bool)

    def mask_from_box(self, image, box, keep_largest: bool = True) -> np.ndarray:
        cands = self(image, box)
        picked, _ = select_instance(cands, box)
        return clean_mask(picked, keep_largest)

    def mask_from_points(self, image, points, labels=None) -> np.ndarray:
        """points: [(x, y), ...]; labels: 1 foreground, 0 background."""
        self.set_image(image)
        pts = np.asarray(points, np.float32)
        lbl = np.ones(len(pts), int) if labels is None else np.asarray(labels, int)
        masks, scores, _ = self.predictor.predict(
            point_coords=pts, point_labels=lbl, multimask_output=True
        )
        return clean_mask(masks[int(np.argmax(scores))])


@dataclass
class InstanceCandidate:
    mask: np.ndarray
    area: float
    score: float          # CLIP similarity to the category word
    bbox: tuple


def find_similar_instances(
    image: np.ndarray,
    target_mask: np.ndarray,
    category: str,
    sam_model,
    clip_scorer=None,
    min_area_ratio: float = 0.3,
    max_area_ratio: float = 3.0,
    min_clip: float = 0.20,
    max_instances: int = 6,
    device="cuda",
) -> list[np.ndarray]:
    """Find OTHER objects of the same category as the target.

    Runs SAM's automatic generator, drops anything overlapping the target,
    keeps candidates of comparable area, then -- if a CLIP scorer is given --
    keeps only those whose crop matches the category word.

    The area filter alone is not enough: a plate under an apple has similar
    area and would be counted as an apple, inflating the leakage denominator
    with objects that were never at risk. Pass a scorer.

    `category` comes from PIE-Bench's blended_words -- the SOURCE word of the
    pair, e.g. "bicycle" from "bicycle,motorcycle".
    """
    from segment_anything import SamAutomaticMaskGenerator

    gen = SamAutomaticMaskGenerator(sam_model, points_per_side=24)
    anns = gen.generate(np.asarray(image, np.uint8))

    tgt = np.asarray(target_mask) > 0
    tgt_area = float(tgt.sum())
    if tgt_area == 0:
        return []

    cands: list[InstanceCandidate] = []
    for a in anns:
        m = a["segmentation"].astype(bool)
        if iou(m, tgt) > 0.2 or (m & tgt).sum() / max(m.sum(), 1) > 0.5:
            continue                                   # this IS the target
        ratio = m.sum() / tgt_area
        if not (min_area_ratio <= ratio <= max_area_ratio):
            continue

        score = 1.0
        if clip_scorer is not None:
            ys, xs = np.where(m)
            crop = np.asarray(image)[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
            score = float(clip_scorer(crop, f"a photo of a {category}"))
            if score < min_clip:
                continue
        cands.append(
            InstanceCandidate(m, float(m.sum()), score, tuple(a["bbox"]))
        )

    cands.sort(key=lambda c: -c.score)
    return [c.mask for c in cands[:max_instances]]


def make_clip_scorer(device="cuda"):
    """CLIP similarity between an image crop and a text string."""
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    tok = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    def score(crop: np.ndarray, text: str) -> float:
        with torch.no_grad():
            im = preprocess(Image.fromarray(np.asarray(crop, np.uint8))).unsqueeze(0)
            f_i = model.encode_image(im.to(device))
            f_t = model.encode_text(tok([text]).to(device))
            f_i = f_i / f_i.norm(dim=-1, keepdim=True)
            f_t = f_t / f_t.norm(dim=-1, keepdim=True)
            return float((f_i @ f_t.T).item())

    return score


def validate_against_gt(
    cases, segmenter: SAMSegmenter, use_gt_bbox: bool = True
) -> dict:
    """IoU of generated masks against ground truth. Run this FIRST.

    The 171 PIE-Bench++ entries that carry real masks are the validation set.
    If IoU is high there, generated masks for the remaining cases are credible
    and you can report a number rather than an assurance.

    use_gt_bbox derives the prompt box from the ground-truth mask, which
    isolates SAM's segmentation quality from box-placement error. Set False to
    measure the realistic end-to-end pipeline with user boxes.
    """
    scores, failed = [], 0
    for case in cases:
        gt = np.asarray(case.target_mask) > 0
        if not gt.any():
            continue
        if use_gt_bbox:
            ys, xs = np.where(gt)
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        elif case.box:
            box = case.box
        else:
            continue
        try:
            pred = segmenter.mask_from_box(case.image, box)
            scores.append(iou(pred, gt))
        except Exception:
            failed += 1

    s = np.array(scores, float)
    return {
        "n": len(s),
        "failed": failed,
        "mean_iou": float(s.mean()) if s.size else float("nan"),
        "median_iou": float(np.median(s)) if s.size else float("nan"),
        "frac_above_0.7": float((s > 0.7).mean()) if s.size else float("nan"),
        "frac_below_0.5": float((s < 0.5).mean()) if s.size else float("nan"),
        "scores": s,
    }
