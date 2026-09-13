"""
Spatially-varying inversion depth -- the second field.

`lambda_field` fixes instance leakage: which tokens the instruction drives.
This module fixes feature bleed: how much of the original survives.

Boosting edit-token attention inside the mask does not remove the source
signal competing with it, which is why changing a black shirt to white leaves
black elements behind. The fix is to invert the target region deeply (little
of the original left to compete) while leaving the background shallow.

    tau(p) in [tau_min, 1]   =  how far this token is inverted
        1.0        target region -- full noise, fully regenerated
        tau_min    background   -- barely perturbed, preserved
        graded     band

Implementation is a graded generalisation of blended latent diffusion. Keep the
inversion trajectory {z_t}. During generation, tokens whose tau is below the
current time are pinned to the stored inverted latent; tokens above it are
free. Hard blending at a mask boundary is exactly what produces the seam
artefacts that mask-based baselines are criticised for, so the pin is soft.

Both fields derive from the same distance transform, so the band radius stays
one knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lambda_field import (
    DEFAULT_PIXELS_PER_TOKEN,
    RegionFieldConfig,
    _area_pool,
    _falloff,
)
from scipy.ndimage import distance_transform_edt


@dataclass
class RetentionConfig:
    """Parameters of the inversion-depth field.

    band_tokens:
        Shared with RegionFieldConfig. Keep them equal unless ablating.
    tau_min:
        Inversion depth in the background. 0.0 pins the background to the
        source for the entire trajectory (maximum preservation, zero global
        adaptation). Small positive values let lighting and colour adapt
        slightly to the edit, which is usually what you want and is the
        cheap version of the "global harmonization" box in the original
        design. Start at 0.1.
    tau_target:
        Depth in the target region. 1.0 = full regeneration. Lower it for
        attribute edits where the silhouette should be retained.
    softness:
        Width, in units of t, of the soft transition of the pin. 0 gives hard
        blended-latent behaviour and reintroduces seams.
    """

    band_tokens: float = 2.0
    tau_min: float = 0.1
    tau_target: float = 1.0
    falloff: str = "smoothstep"
    softness: float = 0.08


def build_retention_field(
    mask: np.ndarray,
    cfg: RetentionConfig | None = None,
    pixels_per_token: int = DEFAULT_PIXELS_PER_TOKEN,
) -> np.ndarray:
    """Binary mask -> (h_tok, w_tok) inversion-depth field tau in [tau_min, 1]."""
    cfg = cfg or RetentionConfig()
    m = np.asarray(mask) > 0

    if not m.any():
        grid = _area_pool(np.zeros_like(m, dtype=np.float32), pixels_per_token)
        return np.full_like(grid, cfg.tau_min, dtype=np.float32)

    band_px = cfg.band_tokens * pixels_per_token
    outside = distance_transform_edt(~m).astype(np.float32)

    weight = np.ones_like(outside, dtype=np.float32)
    if band_px > 0:
        ring = outside > 0
        weight[ring] = _falloff(outside[ring] / band_px, cfg.falloff)
    else:
        weight[outside > 0] = 0.0

    weight_tok = _area_pool(weight, pixels_per_token)
    tau = cfg.tau_min + (cfg.tau_target - cfg.tau_min) * weight_tok
    return tau.astype(np.float32)


def pin_weights(tau: np.ndarray, t: float, softness: float = 0.08) -> np.ndarray:
    """How strongly each token is pinned to the inverted trajectory at time t.

    t is the generation time, 1 = noise, 0 = image. Token p begins free
    generation at t = tau(p): it is PINNED while t > tau(p) and free below.

    A background token with tau = 0.1 was never carried to noise, so it must
    track the source for almost the whole trajectory and only gains freedom at
    the very end -- which is where the mild global adaptation comes from. A
    target token with tau = 1.0 is free immediately.

    Returns values in [0, 1], same shape as tau.
        1 -> take the stored inverted latent (preserve)
        0 -> take the freshly generated latent (regenerate)
    """
    if softness <= 0:
        return (t > tau).astype(np.float32)
    x = (t - tau) / softness
    return np.clip(0.5 * (1.0 + np.tanh(x)), 0.0, 1.0).astype(np.float32)


def blend_latents(z_gen, z_inv, tau, t: float, softness: float = 0.08):
    """z <- w * z_inv + (1 - w) * z_gen, broadcasting over channels.

    Works for numpy or torch given matching shapes:
        z_*: (..., N_tok, C)   tau: (N_tok,)
    """
    w = pin_weights(np.asarray(tau), t, softness)
    if hasattr(z_gen, "device"):  # torch
        import torch

        w = torch.as_tensor(w, device=z_gen.device, dtype=z_gen.dtype)
    w = w[..., None]
    return w * z_inv + (1.0 - w) * z_gen


def paired_configs(
    band_tokens: float,
    lam_in: float = 4.0,
    lam_out: float = -2.0,
    tau_min: float = 0.1,
) -> tuple[RegionFieldConfig, RetentionConfig]:
    """Build both fields from one radius, so the sweep has a single knob."""
    return (
        RegionFieldConfig(band_tokens=band_tokens, lam_in=lam_in, lam_out=lam_out),
        RetentionConfig(band_tokens=band_tokens, tau_min=tau_min),
    )
