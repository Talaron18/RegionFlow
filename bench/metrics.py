"""
Metrics for the Stage-0 sweep.

Standard editing metrics answer "is the background preserved" and "does the
result match the prompt". Neither measures the two failures that motivate this
project, so those get dedicated metrics:

    instance leakage -- did non-target instances change?
    feature bleed    -- did the replacement retain the original's attributes?

Both are reported as rates over a split, not as aggregate quality. The point is
never that a LoRA'd 2B model beats a frontier system; it is that a specific
failure mode goes away.

Heavy dependencies (LPIPS, CLIP) are injected so the numeric core stays
runnable and testable without them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def _as_float(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def masked_mse(a, b, mask, inside: bool = False) -> float:
    """MSE over the region outside (default) or inside the mask."""
    a, b = _as_float(a), _as_float(b)
    m = np.asarray(mask) > 0
    sel = m if inside else ~m
    if sel.sum() == 0:
        return float("nan")
    d = (a - b) ** 2
    if d.ndim == 3:
        sel = np.repeat(sel[:, :, None], d.shape[2], axis=2)
    return float(d[sel].mean())


def masked_psnr(a, b, mask, inside: bool = False) -> float:
    mse = masked_mse(a, b, mask, inside)
    if not np.isfinite(mse) or mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def background_change(source, edited, target_mask) -> float:
    """Mean absolute change outside the target. Lower is better."""
    s, e = _as_float(source), _as_float(edited)
    m = np.asarray(target_mask) > 0
    sel = ~m
    if sel.sum() == 0:
        return float("nan")
    d = np.abs(s - e)
    if d.ndim == 3:
        d = d.mean(axis=2)
    return float(d[sel].mean())


def instance_leakage(
    source,
    edited,
    target_mask,
    other_instance_masks,
    threshold: float = 0.02,
) -> dict:
    """Did instances the user did NOT select get modified?

    This is the metric the project exists for. `other_instance_masks` is a list
    of masks for the same-category objects that were not selected -- the other
    two tulips, the other chair.

    Returns per-instance change and a boolean leak flag. `threshold` is in
    mean-absolute-difference units on [0,1] pixels; calibrate it once against
    a no-op edit to absorb VAE round-trip noise, otherwise you will measure
    reconstruction error and call it leakage.
    """
    s, e = _as_float(source), _as_float(edited)
    d = np.abs(s - e)
    if d.ndim == 3:
        d = d.mean(axis=2)

    tgt = np.asarray(target_mask) > 0
    per = []
    for om in other_instance_masks:
        om = (np.asarray(om) > 0) & ~tgt
        per.append(float(d[om].mean()) if om.sum() else float("nan"))

    target_change = float(d[tgt].mean()) if tgt.sum() else float("nan")
    leaked = [bool(np.isfinite(c) and c > threshold) for c in per]
    return {
        "target_change": target_change,
        "other_change": per,
        "leaked": leaked,
        "leak_rate": float(np.mean(leaked)) if leaked else float("nan"),
        # Bounded in [0, 1]: 1.0 = only the selected instance changed, 0.5 =
        # non-target instances changed as much as the target. A bare ratio
        # target/other is unbounded and explodes when the denominator is near
        # zero, which makes it impossible to average across a split.
        "selectivity": _selectivity(target_change, per),
    }


def _selectivity(target_change: float, other_changes) -> float:
    other = [c for c in other_changes if np.isfinite(c)]
    if not other or not np.isfinite(target_change):
        return float("nan")
    o = float(np.mean(other))
    denom = target_change + o
    if denom <= 1e-9:
        return float("nan")  # nothing changed anywhere; the case is uninformative
    return float(target_change / denom)


def colour_bleed(source, edited, target_mask, bins: int = 16) -> float:
    """Histogram-based feature-bleed proxy for colour edits.

    Overlap between the edited target's colour histogram and the source
    target's. High overlap after a colour edit means the original colour
    survived, i.e. bleed. Pure numpy, so this is the metric to develop against
    before wiring CLIP in; only valid for the colour subset.

    Returns intersection in [0, 1]. Lower is better.
    """
    s, e = _as_float(source), _as_float(edited)
    m = np.asarray(target_mask) > 0
    if m.sum() == 0:
        return float("nan")

    def hist(img):
        px = img[m].reshape(-1, img.shape[2] if img.ndim == 3 else 1)
        h, _ = np.histogramdd(
            px, bins=[bins] * px.shape[1], range=[(0, 1)] * px.shape[1]
        )
        return h / max(h.sum(), 1e-9)

    hs, he = hist(s), hist(e)
    return float(np.minimum(hs, he).sum())


def clip_bleed(
    edited_crop,
    source_attribute: str,
    target_attribute: str,
    clip_score,
) -> float:
    """Directional feature-bleed metric. Requires a CLIP scorer.

    clip_score(image, text) -> float similarity.

    Positive means the crop still looks more like the ORIGINAL attribute than
    the requested one: bleed. Signed, so it is readable as a margin rather
    than an absolute.
    """
    return float(
        clip_score(edited_crop, source_attribute)
        - clip_score(edited_crop, target_attribute)
    )


@dataclass
class CaseResult:
    case_id: str
    r: float
    lam_in: float
    lam_out: float
    mask_bucket: str          # 'small' | 'medium' | 'large'
    n_instances: int
    bg_change: float
    bg_psnr: float
    target_change: float
    leak_rate: float
    selectivity: float
    bleed: float
    clip_edited: float = float("nan")
    clip_whole: float = float("nan")

    def row(self):
        return asdict(self)


def aggregate(results, by=("r",)) -> dict:
    """Group CaseResults and average the numeric fields.

    Always inspect counts per group. MAG-Bench's small-mask bucket is where
    the graded field has no room to work, and averaging it into the total
    hides both the failure and the effect.
    """
    groups: dict = {}
    for res in results:
        key = tuple(getattr(res, k) for k in by)
        groups.setdefault(key, []).append(res)

    out = {}
    numeric = [
        "bg_change", "bg_psnr", "target_change", "leak_rate",
        "selectivity", "bleed", "clip_edited", "clip_whole",
    ]
    for key, rows in groups.items():
        summary = {"n": len(rows)}
        for f in numeric:
            vals = [getattr(x, f) for x in rows]
            vals = [v for v in vals if np.isfinite(v)]
            summary[f] = float(np.mean(vals)) if vals else float("nan")
        out[key] = summary
    return out


# ---------------------------------------------------------------------------
# Boundary quality
# ---------------------------------------------------------------------------
# The graded band exists to fix boundary discordance -- the pasted-on seam
# where an edited region meets untouched background. Region-averaged
# differences cannot see this: a perfectly harmonised edit and a hard paste
# score identically on `background_change` and `selectivity`, because both
# statistics average over areas and a seam is a one-dimensional defect.
#
# These three measure it directly. `transition_width` is the one that should
# respond to r, since r literally sets the width of the transition.

from scipy.ndimage import distance_transform_edt as _edt


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Negative inside the mask, positive outside, in pixels."""
    m = np.asarray(mask) > 0
    if not m.any() or m.all():
        return np.zeros(m.shape, np.float32)
    return (_edt(~m) - _edt(m)).astype(np.float32)


def boundary_profile(
    source, edited, mask, max_dist: int = 24, bin_px: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Mean |change| as a function of signed distance from the mask boundary.

    Returns (distances, mean_change). Negative distance is inside the target.
    This is the raw curve; the scalars below summarise it.
    """
    s, e = _as_float(source), _as_float(edited)
    d = np.abs(s - e)
    if d.ndim == 3:
        d = d.mean(2)
    sd = signed_distance(mask)

    edges = np.arange(-max_dist, max_dist + bin_px, bin_px)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sd >= lo) & (sd < hi)
        if sel.sum() >= 8:
            centres.append((lo + hi) / 2.0)
            means.append(float(d[sel].mean()))
    return np.array(centres), np.array(means)


def transition_width(source, edited, mask, max_dist: int = 24) -> float:
    """Pixels over which the change falls from 90% to 10% of its inside level.

    A hard paste transitions within a pixel or two. A properly feathered edit
    spreads over the band. This is the direct read-out of what r controls, and
    the only metric here that should vary monotonically with it.

    Returns NaN when the profile is too flat to define a transition, which
    happens when the edit barely changed anything.
    """
    x, y = boundary_profile(source, edited, mask, max_dist)
    if len(x) < 5:
        return float("nan")

    inside = y[x < -2]
    outside = y[x > max_dist * 0.6]
    if inside.size == 0 or outside.size == 0:
        return float("nan")
    hi, lo = float(inside.mean()), float(outside.mean())
    if hi - lo < 1e-4:
        return float("nan")

    hi_t, lo_t = lo + 0.9 * (hi - lo), lo + 0.1 * (hi - lo)

    def crossing(level):
        for i in range(len(x) - 1):
            a, b = y[i], y[i + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                f = (level - a) / (b - a)
                return float(x[i] + f * (x[i + 1] - x[i]))
        return None

    x90, x10 = crossing(hi_t), crossing(lo_t)
    if x90 is None or x10 is None:
        return float("nan")
    return abs(x10 - x90)


def seam_gradient_excess(source, edited, mask, ring: int = 2) -> float:
    """Gradient energy the edit ADDED in a thin ring on the boundary.

    Measured relative to the source, because a real object boundary already
    carries gradient; only the excess is a seam. Normalised by the source's
    own boundary gradient so it is comparable across images.

    0 means the edit added no edge. Large positive means a visible seam.
    """
    s, e = _as_float(source), _as_float(edited)
    if s.ndim == 3:
        s, e = s.mean(2), e.mean(2)
    sd = signed_distance(mask)
    band = np.abs(sd) <= ring
    if band.sum() < 8:
        return float("nan")

    def gmag(a):
        gy, gx = np.gradient(a)
        return np.hypot(gy, gx)

    gs, ge = gmag(s), gmag(e)
    base = float(gs[band].mean())
    return float((ge[band].mean() - base) / max(base, 1e-4))


def seam_normal_alignment(source, edited, mask, ring: int = 2) -> float:
    """How much of the added gradient runs perpendicular to the mask contour.

    A pasted region produces edges that follow the mask outline exactly, so the
    change-gradient aligns with the distance transform's gradient. Natural
    content does not respect the mask contour.

    Returned as excess over the source, so 0 means the edit introduced no new
    contour-following edge. Only meaningful when the mask boundary does NOT
    coincide with a real object edge -- if it does, the source already has an
    edge there and there is nothing to add.
    """
    s, e = _as_float(source), _as_float(edited)
    if s.ndim == 3:
        s, e = s.mean(2), e.mean(2)
    sd = signed_distance(mask)
    band = np.abs(sd) <= ring
    if band.sum() < 8:
        return float("nan")

    ny, nx = np.gradient(sd)                     # mask contour normal
    nn = np.hypot(ny, nx)

    def aligned_energy(a):
        """Gradient magnitude projected onto the contour normal."""
        gy, gx = np.gradient(a)
        proj = np.abs(gy * ny + gx * nx) / np.maximum(nn, 1e-6)
        return float(proj[band].mean())

    # The EDITED image's contour-aligned edge, minus the source's. Measuring
    # the gradient of the change field instead (the first version of this)
    # does not work: the change field follows the mask by construction, so it
    # scores ~1 for a feathered blend and a hard paste alike.
    base = aligned_energy(s)
    return float((aligned_energy(e) - base) / max(base, 1e-4))


def boundary_metrics(source, edited, mask, ring: int = 2) -> dict:
    """All three, for a sweep row."""
    return {
        "transition_width": transition_width(source, edited, mask),
        "seam_excess": seam_gradient_excess(source, edited, mask, ring),
        "seam_alignment": seam_normal_alignment(source, edited, mask, ring),
    }
