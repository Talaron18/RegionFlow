"""
Graded region field for localized few-step editing.

Builds a per-token attention-bias field from a binary target mask:

    lambda(p) = lam_in                      inside target
                interp over smoothstep      within r tokens of boundary
                lam_out                     beyond

At r = 0 this collapses exactly to a binary in-mask / out-mask field, which is
the MAG-Edit spatial-ratio constraint expressed as a prior instead of a
gradient objective. That makes r = 0 the null case of the ablation.

Distances are computed at IMAGE resolution and then area-pooled to the latent
token grid. Downsampling the mask first and doing the transform in token space
throws away sub-token structure, which matters because small targets occupy
only a handful of tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


# SD3 / MMDiT: 8x VAE downsample, 2x patchify -> one token spans 16 image px.
DEFAULT_PIXELS_PER_TOKEN = 16


@dataclass
class RegionFieldConfig:
    """Parameters of the graded region field.

    band_tokens (r):
        Width of the transition ring, measured OUTWARD from the target
        boundary, in latent tokens. r=0 gives a hard binary mask.
        Rough guide from the edit-type taxonomy:
            colour / texture        r ~ 1     (seam only)
            object replacement      r ~ 3     (shape headroom)
            relighting / shadow     r ~ 4+    (physical spillover)
    lam_in:
        Bias added to (image_token -> edit_token) attention logits inside the
        target. Positive.
    lam_out:
        Bias applied far outside. Negative, so edit tokens are actively
        suppressed in the background rather than merely not encouraged. This
        is the "minimise outside" half of the spatial-ratio constraint.
    falloff:
        'smoothstep' (C1 continuous, no visible ring artefact) or 'linear'.
    dilate_tokens:
        Optional hard dilation applied to the target itself before the band is
        grown. Use when the edit needs the silhouette to change and the source
        mask is a tight fit around the old object.
    """

    band_tokens: float = 2.0
    lam_in: float = 4.0
    lam_out: float = -2.0
    falloff: str = "smoothstep"
    dilate_tokens: float = 0.0


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """1 -> 0 as x goes 0 -> 1, with zero derivative at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - (3.0 * x**2 - 2.0 * x**3)


def _falloff(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "smoothstep":
        return _smoothstep(x)
    if kind == "linear":
        return 1.0 - np.clip(x, 0.0, 1.0)
    raise ValueError(f"unknown falloff: {kind!r}")


def _area_pool(field: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool an image-resolution field down to the token grid.

    Area pooling (not nearest / max) is deliberate: a target that covers half a
    token should get half weight, otherwise thin structures either vanish or
    inflate to a full token.
    """
    h, w = field.shape
    if h % factor or w % factor:
        pad_h = (-h) % factor
        pad_w = (-w) % factor
        field = np.pad(field, ((0, pad_h), (0, pad_w)), mode="edge")
        h, w = field.shape
    return field.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def build_region_field(
    mask: np.ndarray,
    cfg: RegionFieldConfig | None = None,
    pixels_per_token: int = DEFAULT_PIXELS_PER_TOKEN,
) -> np.ndarray:
    """Binary image-resolution mask -> float lambda field on the token grid.

    Args:
        mask: (H, W) array, non-zero inside the edit target. H, W in image px.
        cfg:  RegionFieldConfig.
        pixels_per_token: VAE downsample * patch size.

    Returns:
        (H // pixels_per_token, W // pixels_per_token) float32 array with
        values in [lam_out, lam_in].
    """
    cfg = cfg or RegionFieldConfig()
    m = np.asarray(mask) > 0

    if not m.any():
        grid = _area_pool(np.zeros_like(m, dtype=np.float32), pixels_per_token)
        return np.full_like(grid, cfg.lam_out, dtype=np.float32)

    band_px = cfg.band_tokens * pixels_per_token
    dilate_px = cfg.dilate_tokens * pixels_per_token

    # Distance from every background pixel to the nearest target pixel.
    outside_dist = distance_transform_edt(~m).astype(np.float32)

    if dilate_px > 0:
        m = outside_dist <= dilate_px
        outside_dist = distance_transform_edt(~m).astype(np.float32)

    # Weight in [0, 1]: 1 on the target, decaying to 0 at the band edge.
    weight = np.ones_like(outside_dist, dtype=np.float32)
    if band_px > 0:
        ring = outside_dist > 0
        weight[ring] = _falloff(outside_dist[ring] / band_px, cfg.falloff)
    else:
        weight[outside_dist > 0] = 0.0

    # Pool the CONTINUOUS weight, not the binary mask, so partial coverage of a
    # token survives the downsample.
    weight_tok = _area_pool(weight, pixels_per_token)

    field = cfg.lam_out + (cfg.lam_in - cfg.lam_out) * weight_tok
    return field.astype(np.float32)


def target_token_area(
    mask: np.ndarray, pixels_per_token: int = DEFAULT_PIXELS_PER_TOKEN
) -> float:
    """Effective size of the target in tokens.

    Below ~4 tokens there is no room for a graded field: the band swamps the
    target and every tier collapses into the same handful of tokens. Use this
    to route small targets to the crop-and-zoom path instead, and to report the
    small-mask bucket of MAG-Bench separately.
    """
    m = (np.asarray(mask) > 0).astype(np.float32)
    return float(_area_pool(m, pixels_per_token).sum())


def schedule_scale(t: float, mode: str = "front") -> float:
    """Time-dependent multiplier on the whole field.

    MAG-Edit found colour edits settle in early denoising steps while texture
    and shape need later ones, and that over-optimising late produces
    structural artefacts. With a 1-4 step budget there is barely a schedule to
    speak of, so this is near-flat by default and only matters if you decompose
    the interval into more steps.

    t is normalised progress through generation, 0 = noise, 1 = image.
    """
    if mode == "flat":
        return 1.0
    if mode == "front":
        # Strong early (layout / semantics), relaxed late (blending).
        return float(np.clip(1.0 - 0.5 * t, 0.0, 1.0))
    if mode == "back":
        return float(np.clip(0.5 + 0.5 * t, 0.0, 1.0))
    raise ValueError(f"unknown schedule mode: {mode!r}")
