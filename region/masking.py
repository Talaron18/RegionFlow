"""
From user intent to a clean binary target mask.

The instance-disambiguation claim lives here. Methods that derive their mask
from cross-attention of the object word cannot distinguish one of several
similar objects -- the attention map fires on all of them. A user box resolves
it. That is the whole argument for keeping the box input, so this module treats
the box as the primary signal and attention/segmentation as the refiner.

Segmentation backend is injected rather than imported, so the sweep harness can
run on ground-truth masks (MAG-Bench and PIE-Bench both ship them) without
pulling SAM in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.ndimage import binary_fill_holes, label

from .lambda_field import DEFAULT_PIXELS_PER_TOKEN, target_token_area


class Segmenter(Protocol):
    """Anything that turns an image + box into candidate masks.

    SAM's predictor satisfies this with a thin adapter. `multimask_output=True`
    is what we want: we do the picking ourselves.
    """

    def __call__(self, image: np.ndarray, box: tuple) -> np.ndarray:
        """Returns (K, H, W) bool candidate masks, best-first is not assumed."""


# Below this many tokens there is no room for a graded field: the band swamps
# the target and every tier collapses. Route to crop-and-zoom instead.
MIN_TOKENS_FOR_GRADED_FIELD = 4.0


@dataclass
class TargetRegion:
    mask: np.ndarray            # (H, W) bool
    box: tuple | None           # (x0, y0, x1, y1) if user-supplied
    token_area: float
    needs_crop_path: bool
    source: str                 # 'ground_truth' | 'box+seg' | 'box_only'
    n_candidates: int = 1

    def report(self) -> str:
        route = "CROP" if self.needs_crop_path else "graded"
        return (
            f"{self.source}: {self.token_area:.1f} tokens, {route} path, "
            f"{self.n_candidates} candidate(s)"
        )


def _box_containment(mask: np.ndarray, box: tuple) -> float:
    """Fraction of the mask's area that falls inside the box.

    Containment, not IoU. A correct mask for a thin object inside a loose box
    has poor IoU but perfect containment; IoU would reject it and pick a
    box-filling blob instead.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    total = mask.sum()
    if total == 0:
        return 0.0
    return float(mask[y0:y1, x0:x1].sum()) / float(total)


def clean_mask(mask: np.ndarray, keep_largest: bool = True) -> np.ndarray:
    """Fill holes and optionally drop all but the largest component.

    keep_largest=False matters for edits whose target is genuinely disjoint --
    'recolour both chairs'. Leaving it True there silently edits one chair.
    """
    m = binary_fill_holes(np.asarray(mask) > 0)
    if keep_largest:
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = lab == sizes.argmax()
    return m


def select_instance(candidates: np.ndarray, box: tuple) -> tuple[np.ndarray, int]:
    """Pick the candidate the user meant.

    Scores containment against the box and penalises masks that spill well
    beyond it. Returns (mask, index).
    """
    best, best_score = 0, -np.inf
    for k, cand in enumerate(candidates):
        contain = _box_containment(cand, box)
        x0, y0, x1, y1 = (int(v) for v in box)
        box_area = max((x1 - x0) * (y1 - y0), 1)
        fill = float(cand[y0:y1, x0:x1].sum()) / box_area
        # want high containment, and a fill fraction that is neither tiny
        # (fragment) nor ~1.0 (the box itself, not the object)
        score = contain - 0.5 * abs(fill - 0.6)
        if score > best_score:
            best, best_score = k, score
    return candidates[best], best


def resolve_target(
    image: np.ndarray | None = None,
    box: tuple | None = None,
    ground_truth_mask: np.ndarray | None = None,
    segmenter: Segmenter | None = None,
    keep_largest: bool = True,
    pixels_per_token: int = DEFAULT_PIXELS_PER_TOKEN,
) -> TargetRegion:
    """Produce a clean target mask by whichever route is available.

    Precedence: ground truth > box + segmenter > box only.

    There is deliberately no attention-derived fallback for the no-box case in
    v0. Adding one would let the pipeline run without a box and quietly hand
    back an all-instances mask, which is precisely the failure the box exists
    to prevent -- and it would contaminate the instance claim.
    """
    n_cand = 1

    if ground_truth_mask is not None:
        mask = clean_mask(ground_truth_mask, keep_largest)
        source = "ground_truth"

    elif box is not None and segmenter is not None:
        cands = np.asarray(segmenter(image, box))
        if cands.ndim == 2:
            cands = cands[None]
        n_cand = len(cands)
        picked, _ = select_instance(cands, box)
        mask = clean_mask(picked, keep_largest)
        source = "box+seg"

    elif box is not None:
        x0, y0, x1, y1 = (int(v) for v in box)
        h, w = (image.shape[:2] if image is not None else (y1, x1))
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        source = "box_only"

    else:
        raise ValueError(
            "no box, no ground-truth mask. v0 requires one of them: an "
            "attention-derived mask cannot disambiguate instances."
        )

    area = target_token_area(mask, pixels_per_token)
    return TargetRegion(
        mask=mask,
        box=box,
        token_area=area,
        needs_crop_path=area < MIN_TOKENS_FOR_GRADED_FIELD,
        source=source,
        n_candidates=n_cand,
    )
