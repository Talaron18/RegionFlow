"""
Structure metrics that survive a real colour edit.

WHY THIS FILE EXISTS. Stage 0 measured structure preservation as the Pearson
correlation of LUMINANCE EDGE ENERGY inside the mask (tau_sweep.struct_corr).
The oracle probe showed what that costs: the GROUND-TRUTH MagicBrush edits --
real photographs of a black sheep, a red phone booth -- score

    sheep 0.10,  phone_booth 0.29,  ball 0.43     (mean struct@dE25 = 0.365)

while the learned model scores ~0.41. The model beats the ground truth. That is
not possible if the metric measures structure preservation, so it does not.

The mechanism is simple. Darkening wool flattens its luminance contrast;
turning a green ball blue changes its lightness. Edge ENERGY falls for any
realistic recolour, by construction, so `struct` is anti-correlated with `dE`
before any model is involved. Stage 0's corr(dE, struct) = -0.958 is therefore
partly a property of the metric pair rather than of the editing method, and the
Gate A threshold of 0.70 was never attainable.

The Gate 0 synthetic control did not catch this because it used a
luminance-PRESERVING L*a*b* chroma shift -- the one case where edge energy
survives. It validated the metric on an edit that does not occur in practice.

WHAT REPLACES IT. Three candidates, all invariant to local contrast scaling:

  grad_orient   cosine similarity of gradient DIRECTION fields, magnitude-
                weighted. Where an edge points is preserved by recolouring;
                how strong it is, is not. This is the primary proposal.

  lcn_corr      edge correlation after local contrast normalisation: divide
                the luma gradient magnitude by its local RMS before
                correlating. Keeps magnitude information but removes the
                global scaling a colour change induces.

  oracle_rel    struct(edited) / struct(ground-truth edit) at matched dE. Not
                a metric so much as a calibration: the oracle defines what is
                achievable for THAT edit, so 1.0 means "as structure-preserving
                as the real thing". Works with ANY underlying metric, including
                the Stage 0 one, and is the most defensible number to report
                because it needs no argument about which metric is right.

Report oracle_rel as primary and grad_orient alongside. Keep the Stage 0
struct_corr in the table too -- the comparison with Stage 0 is the point, and
hiding the metric that produced it would be worse than explaining it.
"""

from __future__ import annotations

import numpy as np


def luma(img) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _grads(y):
    gy, gx = np.gradient(y.astype(np.float32))
    return gx, gy


def grad_orient(src, edit, mask, eps=1e-6, weight=True) -> float:
    """Cosine similarity of gradient DIRECTION inside the mask.

    Invariant to any monotone local rescaling of contrast, so a recolour that
    darkens or lightens a region keeps its score as long as the edges stay
    where they were. Magnitude-weighted by default (using the SOURCE gradient
    magnitude) so flat areas, whose orientation is noise, do not dominate.

    Range is [-1, 1]; 1.0 means every edge points the same way it did.
    """
    m = np.asarray(mask) > 0
    if m.sum() < 16:
        return float("nan")
    gxs, gys = _grads(luma(src))
    gxe, gye = _grads(luma(edit))

    ns = np.sqrt(gxs ** 2 + gys ** 2) + eps
    ne = np.sqrt(gxe ** 2 + gye ** 2) + eps
    cos = (gxs * gxe + gys * gye) / (ns * ne)

    w = ns[m] if weight else np.ones(m.sum(), dtype=np.float32)
    return float(np.average(cos[m], weights=w))


def lcn_corr(src, edit, mask, sigma_px: int = 8, eps=1e-6) -> float:
    """Edge-magnitude correlation after local contrast normalisation.

    Divides gradient magnitude by its local RMS (box average, sigma_px), which
    removes the global contrast change a recolour causes while keeping
    relative edge structure. Closer in spirit to Stage 0's struct_corr than
    grad_orient is, so it makes the cleaner apples-to-apples comparison.
    """
    m = np.asarray(mask) > 0
    if m.sum() < 16:
        return float("nan")

    def norm_mag(y):
        gx, gy = _grads(y)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        k = max(1, int(sigma_px))
        n = 2 * k + 1
        pad = np.pad(mag, k, mode="edge")
        # Integral image with a zero row and column prepended, so the box sum
        # comes out at the original shape. Without the zero border the four
        # corner lookups are off by one and the result is (H-1, W-1).
        c = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), dtype=np.float64)
        c[1:, 1:] = np.cumsum(np.cumsum(pad, 0), 1)
        loc = (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)
        return mag / (loc + eps)

    a, b = norm_mag(luma(src))[m], norm_mag(luma(edit))[m]
    if a.std() < eps or b.std() < eps:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def oracle_relative(struct_edit: float, struct_oracle: float,
                    eps=1e-6) -> float:
    """struct(edited) / struct(ground-truth edit), clipped at 0.

    1.0 means the model preserved structure as well as the real edit did.
    Above 1.0 means it changed LESS than the real edit -- which is not
    necessarily good, so always report it next to dE.

    This is the number to lead with. It requires no argument about which
    structure metric is correct, because whatever bias the metric has applies
    to numerator and denominator alike.
    """
    if struct_oracle is None or abs(struct_oracle) < eps:
        return float("nan")
    return float(max(struct_edit, 0.0) / max(struct_oracle, eps))


METRICS = {"grad_orient": grad_orient, "lcn_corr": lcn_corr}
