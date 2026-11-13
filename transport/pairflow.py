"""
Gate A: conditional transport directly between source and target latents.

This is the cheap, decisive test of the Stage 1 hypothesis, and it needs
neither MeanFlow nor inversion nor a noise schedule.

Stage 0 formulated editing as a mixing ratio between the source and noise. The
finding was that every control collapsed onto one axis: corr(dE, struct) =
-0.958, -0.726, -0.955 across three objects. The structural reading was that a
mixing ratio can only say HOW MUCH SOURCE TO KEEP, never WHERE TO MOVE, so
attribute change and source destruction cannot come apart.

The minimal formulation with a "where to move" is a straight-line flow between
the two latents:

    z_s = (1 - s) * z_src + s * z_tgt        s in [0, 1]
    v   = z_tgt - z_src                       (constant along the path)

train u_theta(z_s, s, c, M) ~ v, and read the edit out in one step at s = 0:

    z_edit = z_src + u_theta(z_src, 0, c, M)

which is literally z + Delta z. Noise never enters. The source is never
destroyed to make room for the edit -- it is displaced.

DIRECTION A (GATE_A_REOPENING_PLAN.md SS7-8), sigma0 > 0 in PairFlowConfig:
this is Gate A's own reopening after the above was diagnosed as multimodal-
target collapse (MSE learns E[v|z,c,M], and at s=0 every plausible edit
mode shares the identical query point, so nothing disambiguates them). Adds
mask-gated noise AT the query point specifically -- see sample_path's own
docstring for why sigma(0)=0 (the conventional stochastic-interpolant
choice) would fix nothing here. sigma0=0.0 (default) reproduces the
paragraph above exactly; noise stays opt-in.

WHAT MAKES THIS A FALSIFICATION AND NOT A DEMO. The comparison is not "does it
produce nice images". It is corr(dE, struct) on the SAME three objects, the
SAME two colour directions and the SAME strength sweep as Stage 0, with
strength now swept by scaling the learned displacement:

    z_edit(a) = z_src + a * u_theta(z_src, 0, c, M),   a in [0, 1.5]

If that sweep also gives corr(dE, struct) ~ -0.95, the coupling is a property
of the latent space rather than of the mixing formulation, the Stage 0
conclusion is wrong in its stated cause, and MeanFlow will not save it. If it
comes back near zero, the hypothesis survives and Gate B is worth its risk.

Pre-register the threshold before running: the report already commits to
"substantially closer to 0 than -0.95". Fix it at |corr| < 0.5 and do not move
it afterwards.

SCOPE. `a` scaling is a linear extrapolation of a field that was only trained
at a = 1. It is a probe of the direction, not a claim that the model is
calibrated off-manifold. Report it as a sweep of displacement magnitude and
say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class PairFlowConfig:
    #: Weight on the region OUTSIDE the mask. The target latent already equals
    #: the source there (in a well-built triplet), so v = 0 outside and this
    #: term teaches the model to leave the background alone. Keeping it
    #: separate from the in-mask term means the loss can be reported split,
    #: which is the only way to see a model that is winning by moving
    #: everything a little.
    outside_weight: float = 1.0
    inside_weight: float = 1.0
    #: Fraction of samples trained with the mask blanked, to keep an
    #: unmasked-instruction path alive as a control arm.
    mask_dropout: float = 0.1
    #: Fraction with the text condition blanked. Needed if you ever want CFG at
    #: inference; also a clean ablation of how much of the displacement is
    #: coming from the instruction versus from the mask.
    cond_dropout: float = 0.1
    #: Direction A (GATE_A_REOPENING_PLAN.md SS8): sigma(0) in the stochastic
    #: interpolant z_s = (1-s)*z_src + s*z_tgt + sigma(s)*eps, sigma(s) =
    #: sigma0*(1-s). 0.0 (default) is byte-identical to the original
    #: deterministic pairflow -- see sample_path's own docstring for why
    #: sigma0 has to be > 0 at s=0 specifically, not just somewhere on the
    #: path, for this to do anything at the one-step readout.
    sigma0: float = 0.0
    #: Interval conditioning: train the velocity over a random sub-interval
    #: [s, r] instead of always predicting the full displacement to s=1.
    #: This keeps inference-time multi-step trajectories closer to the
    #: distribution seen during training.
    #: NOT the objective behind the documented Gate A results: those (sigma0
    #: sweep, diagnose tables) were trained at 2508cee with r = s. Since
    #: 9c67baa the plain branch passes r = 1 and this branch passes r > s;
    #: with AdapterConfig.use_interval=True, r reaches the model through the
    #: interval embedding, so neither current setting reproduces r = s.
    #: Default stays True (the setting the perceptual-loss work was written
    #: under). sigma0 > 0 is rejected with it: the interval path never calls
    #: sample_path, so sigma0 would be dropped silently.
    interval_conditioning: bool = True
    interval_min: float = 0.1
    perceptual_weight: float = 0.0
    #: Which feature extractor to use.
    #:   "dinov2"  -- facebook/dinov2-base, patch 14, 16x16 grid at 224x224.
    #:   "lpips"   -- AlexNet LPIPS, kept only as the failed baseline.
    perceptual_feature: str = "dinov2"
    #: Erode the patch mask by this many DINO patches before the loss.
    #: 1 avoids boundary patches whose receptive field crosses the mask
    #: edge (the discrimination check found oracle distance 0.21-0.23 on
    #: some samples purely from that bleed).
    perceptual_mask_erode: int = 1
# ---------------------------------------------------- perceptual features --

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _dino_normalize(img: torch.Tensor) -> torch.Tensor:
    """img: (B, 3, H, W) in [-1, 1]. Returns ImageNet-normalized, 224x224."""
    img = (img + 1.0) / 2.0
    mean = torch.tensor(_IMAGENET_MEAN, device=img.device,
                        dtype=img.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=img.device,
                       dtype=img.dtype).view(1, 3, 1, 1)
    img = (img - mean) / std
    img = F.interpolate(img, size=(224, 224), mode="bilinear",
                        align_corners=False)
    return img


def dino_patches(dino, img_neg1_1: torch.Tensor) -> torch.Tensor:
    """img_neg1_1: (B, 3, H, W) in [-1, 1].
    Returns L2-normalized patch features (B, N, D), N=256."""
    x = _dino_normalize(img_neg1_1)
    out = dino(pixel_values=x)
    patches = out.last_hidden_state[:, 1:, :]  # drop CLS
    patches = patches / patches.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return patches


def mask_to_patch_grid(mask: torch.Tensor, grid: int = 16,
                       erode: int = 1) -> torch.Tensor:
    """mask: (B, 1, h, w) in {0, 1}. Returns (B, N) bool, N = grid*grid.

    `erode` shrinks the mask by that many patches. Boundary patches have
    receptive fields that cross the mask edge, so their feature distance
    is nonzero even for the oracle -- eroding removes that.
    """
    m = F.adaptive_avg_pool2d(mask, (grid, grid))
    m = (m > 0.5).view(mask.shape[0], -1)
    if erode > 0:
        k = 2 * erode + 1
        m = F.max_pool2d(
            (~m).float().view(mask.shape[0], 1, grid, grid),
            kernel_size=k, stride=1, padding=erode,
        ).view(mask.shape[0], -1) < 0.5
    return m


def dino_patch_loss(dino, img_pred: torch.Tensor, img_tgt: torch.Tensor,
                    mask: torch.Tensor, erode: int = 1) -> torch.Tensor:
    """Mean (1 - cos) over mask patches, in DINOv2 feature space."""
    p_pred = dino_patches(dino, img_pred)
    with torch.no_grad():
        p_tgt = dino_patches(dino, img_tgt)
    m = mask_to_patch_grid(mask, grid=16, erode=erode)
    cos = (p_pred * p_tgt).sum(dim=-1)
    d = (1.0 - cos) * m.float()
    return d.sum() / m.float().sum().clamp(min=1.0)
    
def sample_path(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    mask: torch.Tensor | None = None,
    sigma0: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform s along the straight path. Returns (z_s, s, v).

    s is uniform, NOT logit-normal. The logit-normal schedule in meanflow.py
    exists to match SD3's noise-level training distribution; there is no noise
    here and no reason to prefer any point on the path. Copying that choice
    across would over-sample s ~ 0.5 and starve s = 0, which is the only point
    the one-step readout ever evaluates.

    sigma0 > 0 -- DIRECTION A (GATE_A_REOPENING_PLAN.md SS7-8). The module
    docstring's "noise never enters" was true of the ORIGINAL deterministic
    formulation and is exactly what Gate A's own reopening plan diagnoses as
    the bug: the one-step readout queries u(z_src, 0, c, M), and at s=0 the
    conditioning is IDENTICAL for every plausible edit mode -- there is
    nothing at the query point to select which one. Standard stochastic
    interpolants take sigma(0)=sigma(1)=0 so the endpoints stay exact; doing
    that here would change nothing at s=0, the only point ever read out.
    This uses sigma(s) = sigma0*(1-s) instead: sigma(0)=sigma0>0 (mode
    selection AT the query point), sigma(1)=0 (z_1=z_tgt stays exact, same
    as before).

    Folds into a NOISED EFFECTIVE SOURCE rather than a separate noise term:

        z_src' = z_src + sigma0*eps*mask
        z_s    = (1-s)*z_src' + s*z_tgt        (this function's existing
        v      = z_tgt - z_src'                 formula, z_src -> z_src')

    which is why this is a substitution into the existing code path rather
    than new machinery. MASK-GATED so the noise never touches the
    background: outside the mask z_src'=z_src exactly and v=0 exactly as
    before, preserving PairFlowConfig's "leave the background alone"
    property and this module's own inside/outside loss split unchanged.
    sigma0=0.0 (the default) makes z_src'=z_src identically -- BYTE-IDENTICAL
    to the pre-Direction-A behaviour, not merely close to it (see self_test).
    """
    b = z_src.shape[0]
    s = torch.rand(b, device=z_src.device, generator=generator)
    s_ = s.reshape(-1, 1, 1, 1)

    z_src_eff = z_src
    if sigma0 > 0:
        if mask is None:
            raise ValueError(
                "sigma0 > 0 requires mask (the noise is mask-gated so it "
                "never touches the background -- see docstring)")
        eps = torch.randn(z_src.shape, device=z_src.device,
                          generator=generator, dtype=z_src.dtype)
        z_src_eff = z_src + sigma0 * eps * mask

    z_s = (1 - s_) * z_src_eff + s_ * z_tgt
    v = z_tgt - z_src_eff
    return z_s, s, v

def sample_interval_path(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    min_interval: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an interval-conditioned straight-line transport target.

    Instead of always training:
        u(z_s, s) = z_tgt - z_src

    this trains:
        u(z_s, s, r) = (z_r - z_s) / (r-s)

    where r > s.  The state z_s is still sampled from the original
    source-target line, but the model learns local transport over arbitrary
    intervals.
    """
    b = z_src.shape[0]
    device = z_src.device

    s = torch.rand(b, device=device, generator=generator)
    remaining = 1.0 - s

    # ensure r > s and avoid tiny intervals
    delta = min_interval + (remaining - min_interval).clamp(min=0) * torch.rand(
        b, device=device, generator=generator
    )
    delta = torch.minimum(delta, remaining.clamp(min=1e-4))
    r = s + delta

    s_ = s.reshape(-1, 1, 1, 1)
    r_ = r.reshape(-1, 1, 1, 1)

    z_s = (1 - s_) * z_src + s_ * z_tgt
    z_r = (1 - r_) * z_src + r_ * z_tgt

    v = (z_r - z_s) / (r_ - s_)

    return z_s, s, r, v



def pairflow_loss(
    model,
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    mask: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    cfg: PairFlowConfig,
    generator: torch.Generator | None = None,
    vae=None,
    dino=None,
    lpips_fn=None,
) -> tuple[torch.Tensor, dict]:
    """Masked rectified-flow regression from source to target.

    model is a TransportAdapter in mode="pair". r is passed equal to s only to
    satisfy the signature; in pair mode the gate ignores it (gate = 1), which
    is the whole reason that mode exists.
    """
    if cfg.interval_conditioning and cfg.sigma0 > 0:
        raise ValueError(
            "interval_conditioning=True ignores sigma0 (sample_interval_path "
            f"has no noise term); got sigma0={cfg.sigma0}. Set "
            "interval_conditioning=False or sigma0=0.")
    if cfg.interval_conditioning:
        z_s, s, r, v = sample_interval_path(
            z_src, z_tgt, min_interval=cfg.interval_min, generator=generator
        )
    else:
        z_s, s, v = sample_path(
            z_src, z_tgt, mask=mask, sigma0=cfg.sigma0, generator=generator
        )
        r = torch.ones_like(s)

    m = mask
    if cfg.mask_dropout > 0:
        drop = (torch.rand(z_src.shape[0], device=z_src.device,
                           generator=generator) < cfg.mask_dropout)
        m = mask * (~drop).to(mask.dtype).reshape(-1, 1, 1, 1)

    pe, po = prompt_embeds, pooled
    if cfg.cond_dropout > 0:
        drop_c = (torch.rand(z_src.shape[0], device=z_src.device,
                             generator=generator) < cfg.cond_dropout)
        # Cast the mask to the embedding dtype BEFORE multiplying. A float32
        # keep-mask times a bf16 embedding promotes the result to float32,
        # which then hits the frozen bf16 context_embedder and raises
        # "mat1 and mat2 must have the same dtype". The cast in the caller is
        # undone here otherwise, silently.
        keep = (~drop_c).to(pe.dtype)
        pe = pe * keep.reshape(-1, 1, 1)
        po = po * keep.to(po.dtype).reshape(-1, 1)

    u = model(z_s, s, r, pe, po, mask=m)

    err = (u - v).pow(2)
    inside = mask
    outside = 1.0 - mask

    n_in = inside.sum().clamp(min=1.0)
    n_out = outside.sum().clamp(min=1.0)
    l_in = (err * inside).sum() / (n_in * err.shape[1])
    l_out = (err * outside).sum() / (n_out * err.shape[1])

    loss = cfg.inside_weight * l_in + cfg.outside_weight * l_out

    # ---- perceptual loss -------------------------------------------
    # Discrimination check (scripts/feature_discrimination.py) on the
    # sigma0=0.0 checkpoint: in-mask DINOv2 patch distance pred/oracle
    # = 30.1, vs LPIPS = 3.8. LPIPS training did not sharpen images
    # because in LPIPS's feature space the blurry MSE optimum already
    # sits close to the sharp target; DINO's does not. So DINO.
    l_percept = None
    if cfg.perceptual_weight > 0:
        if vae is None:
            raise RuntimeError(
                "perceptual_weight > 0 but vae=None; caller must pass it.")
        vae_dtype = next(vae.parameters()).dtype
        scaling = float(vae.config.scaling_factor)
        shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)

        z_pred = (z_src + u * mask) / scaling + shift
        img_pred = vae.decode(z_pred.to(vae_dtype)).sample.float().clamp(-1, 1)

        with torch.no_grad():
            z_ref = z_tgt / scaling + shift
            img_tgt = vae.decode(z_ref.to(vae_dtype)).sample.float().clamp(-1, 1)

        if cfg.perceptual_feature == "dinov2":
            if dino is None:
                raise RuntimeError(
                    "perceptual_feature='dinov2' but dino=None.")
            l_percept = dino_patch_loss(
                dino, img_pred, img_tgt, mask,
                erode=cfg.perceptual_mask_erode)
        elif cfg.perceptual_feature == "lpips":
            if lpips_fn is None:
                raise RuntimeError(
                    "perceptual_feature='lpips' but lpips_fn=None.")
            l_percept = lpips_fn(img_pred, img_tgt).mean()
        else:
            raise ValueError(
                f"perceptual_feature must be 'dinov2' or 'lpips', got "
                f"{cfg.perceptual_feature!r}")

        loss = loss + cfg.perceptual_weight * l_percept

    with torch.no_grad():
        # Per-element means throughout. ...
        C = err.shape[1]
        v2_in = float((v.pow(2) * inside).sum() / (n_in * C))
        u2_in = float((u.pow(2) * inside).sum() / (n_in * C))
        v2_out = float((v.pow(2) * outside).sum() / (n_out * C))

        stats = {
            "loss": float(loss),
            "loss_in": float(l_in),
            "loss_out": float(l_out),
            "loss_percept": (float(l_percept) if l_percept is not None
                             else float("nan")),
            # THE NUMBER TO WATCH. loss_in divided by the error a model that
            # predicts u=0 would incur. Above 1.0 the model is worse than
            # doing nothing; below 1.0 it has learned something. A run whose
            # rel_in never crosses 1.0 has collapsed to the trivial solution,
            # and a decoupling verdict taken from it would be meaningless --
            # bench/decoupling.classify_sweep would call it "null", but only
            # after the GPU time had been spent.
            "rel_in": float(l_in) / max(v2_in, 1e-8),
            "rel_out": float(l_out) / max(v2_out, 1e-8),
            "v_rms_in": v2_in ** 0.5,
            "u_rms_in": u2_in ** 0.5,
            # |u|/|v| inside the mask. Near 0 means under-moving: a later null
            # result would be about optimisation, not about the hypothesis.
            "move_ratio": (u2_in ** 0.5) / max(v2_in ** 0.5, 1e-8),
            # DIRECTIONAL ALIGNMENT -- the quantity that actually limits this
            # model. Under MSE the optimal amplitude for a direction with
            # alignment rho is exactly r* = rho, and the best achievable
            # rel_in is 1 - rho^2. So:
            #
            #   move_ratio / cos ~ 1   the amplitude is already MSE-optimal;
            #                          it will not grow unless cos grows.
            #                          Shrinkage is correct hedging, not
            #                          timidity, and more steps will not fix it.
            #   cos well below 1       different target transports are being
            #                          averaged: the conditional mean of a
            #                          multimodal target. A regression loss
            #                          cannot represent that, by construction.
            #
            # Measured on the 2000-step run: cos plateaued near 0.49 with
            # move_ratio/cos = 0.95 and rel_in = 1 - cos^2 to three decimals.
            "cos": float(((u * v * inside).sum() / (n_in * C)))
                   / max((u2_in * v2_in) ** 0.5, 1e-8),
            "rel_in_floor": None,   # filled below; = 1 - cos^2
        }
        stats["rel_in_floor"] = 1.0 - stats["cos"] ** 2
        # Undefined when the direction is near-random: r/cos explodes and
        # poisons any average taken over the run (it hit 8.6e11 at step 225 of
        # the 4511-triplet run). Report it only where it means something.
        stats["amp_vs_optimal"] = (
            stats["move_ratio"] / stats["cos"] if stats["cos"] > 0.2
            else float("nan")
        )

        # cos_clean -- cos(u, v) against the TRUE, sigma0-INDEPENDENT
        # displacement z_tgt - z_src, not the noised v = z_tgt - z_src'
        # sample_path returned above. Diagnostic only; never fed into the
        # loss (that stays the interpolant's own noised target, which is
        # what Direction A is supposed to learn against).
        #
        # WHY THIS EXISTS. "cos" above is NOT comparable across sigma0
        # values -- it measures alignment against a DIFFERENT, noisier
        # target at every sigma0 (v_rms_in ~1.0-1.1, i.e. the same scale
        # as eps's own unit variance, so by sigma0=1.0 that target is
        # roughly half true-edit-signal, half injected noise). z_s
        # CONTAINS sigma0*eps*mask near s=0 -- almost exactly the point
        # the one-step readout queries -- so a model can partially learn
        # to echo that noise straight back out of its own input rather
        # than learning the true edit direction, and that echo gets
        # EASIER to fit as sigma0 grows (linearly present in the input,
        # unlike real edit semantics). A rising "cos" at higher sigma0 is
        # consistent with genuine mode-selection improving; it is EQUALLY
        # consistent with the model gaming an increasingly-noise-dominated
        # target. cos_clean can't be gamed this way: z_tgt - z_src has no
        # eps in it regardless of what sigma0 or what eps generated z_s,
        # so echoing eps back does not help this number.
        #
        # sigma0=0: v already equals z_tgt - z_src, so cos_clean == cos
        # exactly (not approximately) -- asserted in self_test.
        v_clean = z_tgt - z_src
        v2c_in = float((v_clean.pow(2) * inside).sum() / (n_in * C))
        stats["cos_clean"] = (
            float(((u * v_clean * inside).sum() / (n_in * C)))
            / max((u2_in * v2c_in) ** 0.5, 1e-8)
        )
    return loss, stats


@torch.no_grad()
def ablate_text_condition(
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove or mismatch the instruction for a diagnostic forward.

    "zero"     both tensors zeroed -- exactly what cond_dropout does to 10%
               of training samples, so the model has seen this input.
               Measures what the instruction buys over image + mask alone.
    "shuffle"  each item gets its batch neighbour's instruction (roll by
               one). The text is still a real, in-distribution instruction,
               just the wrong one. Measures instruction SPECIFICITY, without
               the "no text at all" input.

    Scoring is unchanged by either: the caller still compares against the
    item's own true displacement.
    """
    if mode == "zero":
        return torch.zeros_like(prompt_embeds), torch.zeros_like(pooled)
    if mode == "shuffle":
        if prompt_embeds.shape[0] < 2:
            raise ValueError("ablate_text 'shuffle' needs a batch of >= 2")
        return prompt_embeds.roll(1, dims=0), pooled.roll(1, dims=0)
    raise ValueError(f"ablate_text mode must be 'zero' or 'shuffle', got {mode!r}")


def direction_position_diagnostics(
    u: torch.Tensor,
    v_clean: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-example cos(u, v_clean) under three treatments -- the WHERE/WHAT
    split (GATE_A_REOPENING_PLAN.md SS2) applied to a trained checkpoint's
    own predictions, not to raw data. u is the model's predicted velocity
    at s=0 (the actual one-step readout query point); v_clean = z_tgt -
    z_src, sigma0-independent, same as pairflow_loss's own cos_clean.

    raw       cosine over the FULL (C,H,W) tensor, unmasked. The model has
              to get both WHERE (which pixels move) and WHAT (which
              direction) right here -- comparable to
              ambiguity_probe.py's raw-v ceiling (0.420 on the real
              cache).
    pooled    u and v_clean each mean-pooled over the mask into one value
              per channel FIRST (same construction as
              ambiguity_probe.mean_pool_in_mask), then cosine. Removes the
              positional burden entirely -- comparable to its
              mask-restricted ceiling (0.621).
    centered  pooled, then a fixed per-channel `bias` (the shared,
              non-instruction-specific direction most edits move in --
              fit externally on held-out ground truth via
              ambiguity_probe.held_out_centered_gap, NOT estimated from
              this batch) is subtracted from both u and v_clean before
              cosine. Omitted (bias=None, the default) if no bias is
              available. Comparable to the mean-centered ceiling (0.458 /
              held-out distribution).

    Why this triple exists: the raw-vs-pooled gap tells you how much of
    the model's shortfall is POSITIONAL (getting the right pixels) versus
    DIRECTIONAL (getting the right per-channel shift); pooled-vs-centered
    tells you how much of what looks directional is actually the model
    reproducing a generic bias every edit shares, versus genuine
    instruction-specific direction.

    Returns per-example tensors of shape (B,) (NaN where a denominator is
    ~0), for the caller to aggregate (mean/median/percentiles) across a
    whole eval set the same way ambiguity_probe.report() does.
    """
    def _cos(a: torch.Tensor, b: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
        num = (a * b).sum(dim=dim)
        den = (a.pow(2).sum(dim=dim).clamp(min=1e-16).sqrt()
              * b.pow(2).sum(dim=dim).clamp(min=1e-16).sqrt())
        return torch.where(den > 1e-8, num / den, torch.full_like(num, float("nan")))

    raw = _cos(u, v_clean, dim=(1, 2, 3))

    denom = mask.sum(dim=(2, 3)).clamp(min=1e-8)          # (B, 1)
    u_pooled = (u * mask).sum(dim=(2, 3)) / denom          # (B, C)
    v_pooled = (v_clean * mask).sum(dim=(2, 3)) / denom    # (B, C)
    pooled = _cos(u_pooled, v_pooled, dim=(1,))

    out = {"raw": raw, "pooled": pooled}
    if bias is not None:
        C = u_pooled.shape[1]
        b = bias.reshape(1, C).to(dtype=u_pooled.dtype, device=u_pooled.device)
        out["centered"] = _cos(u_pooled - b, v_pooled - b, dim=(1,))
    return out


@torch.no_grad()
def apply_transport(
    model,
    z_src: torch.Tensor,
    mask: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    strength: float = 1.0,
    steps: int = 1,
    pin_background: bool = True,
    sigma0: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """z_src -> z_edit. steps=1 is the headline; steps>1 integrates the path.

    sigma0 > 0 -- DIRECTION A sampling (GATE_A_REOPENING_PLAN.md SS8). The
    query starts at z_src + sigma0*eps*mask (matching training's z_src',
    mask-gated so the background is never perturbed), not at z_src itself.
    Sample fresh eps each call for a genuinely different mode each time;
    pass `generator` for a reproducible draw (same generator + same inputs
    -> same edit). A model TRAINED with sigma0>0 and then EVALUATED at
    sigma0=0 queries it at z_src exactly, which training never actually saw
    as a fixed point (only as eps's average) -- mildly off-distribution.
    Use sigma0=0 at eval only for continuity with Stage 0's existing
    deterministic sweep methodology; Direction A's primary evaluation
    samples WITH sigma0>0 (see the reopening plan's evaluation track).

    pin_background=True applies the displacement ONLY inside the mask:

        z <- z + dt * u * M

    This is the same background pin Stage 0 used (copy source latents outside
    the region), and it is required for the Gate A numbers to be comparable
    with Stage 0's. Without it the eval adds u everywhere, and the model's
    small out-of-mask residual -- correct to within its training loss, but not
    exactly zero -- gets multiplied by the strength and smears the whole
    frame. Measured on the phone_booth case, the entire scene slid brown by
    a=2.0 for that reason alone.

    pin_background=False measures whether the model localises ON ITS OWN,
    which is a separate and genuinely interesting result: on the sheep case
    the background survived to a=2.5 unpinned, on phone_booth it did not.
    Report it as self-localisation, not as edit quality.

    steps > 1 is the honest check on whether one step was enough: if the
    multi-step result differs materially, the path is curved and the one-step
    readout is an approximation that should be reported as such.
    """
    z = z_src.clone()
    if sigma0 > 0:
        eps = torch.randn(z_src.shape, device=z_src.device,
                          generator=generator, dtype=z_src.dtype)
        z = z + sigma0 * eps * mask
    b = z.shape[0]
    m = mask if pin_background else torch.ones_like(mask)
    grid = torch.linspace(0.0, strength, steps + 1, device=z.device)
    for i in range(steps):
        s = torch.full((b,), float(grid[i]), device=z.device)
        r = torch.full((b,), float(grid[i + 1]), device=z.device)
        u = model(z, s, r, prompt_embeds, pooled, mask=mask)
        z = z + float(grid[i + 1] - grid[i]) * u * m
    return z


# --------------------------------------------------------------- self-test --

def _fake_pair_model(u_value: torch.Tensor):
    """A model stub that always returns u_value, ignoring its inputs -- for
    exercising apply_transport's integration/masking logic without a real
    TransportAdapter (which needs a loaded SD3 transformer)."""
    def fn(z, s, r, prompt_embeds, pooled, mask=None):
        return u_value
    return fn


def self_test() -> None:
    """sample_path's sigma0 math and apply_transport's sigma0 sampling,
    CPU-only, no model weights needed. Run BEFORE trusting Direction A's
    training run, same discipline as bench/off_instance.py and
    bench/edit_quality_vlm.py -- this module had no self-test before
    Direction A added non-trivial new logic to it.
    """
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 8, 8
    z_src = torch.randn(B, C, H, W)
    z_tgt = torch.randn(B, C, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 2:6, 2:6] = 1.0  # a block in the middle, rest is background

    # sigma0=0.0 must be BYTE-IDENTICAL to the pre-Direction-A formula --
    # this is the backward-compatibility guarantee every existing Gate A
    # run and checkpoint depends on.
    gen = torch.Generator().manual_seed(1)
    z_s0, s0, v0 = sample_path(z_src, z_tgt, mask=mask, sigma0=0.0, generator=gen)
    gen2 = torch.Generator().manual_seed(1)
    s_ref = torch.rand(B, generator=gen2).reshape(-1, 1, 1, 1)
    assert torch.allclose(z_s0, (1 - s_ref) * z_src + s_ref * z_tgt)
    assert torch.equal(v0, z_tgt - z_src)

    # sigma0 > 0 requires a mask -- noise with nothing to gate it is exactly
    # the "noise enters everywhere" failure mode this design explicitly
    # avoids; refuse rather than silently un-gate it.
    try:
        sample_path(z_src, z_tgt, mask=None, sigma0=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("sigma0 > 0 without a mask must raise")

    # sigma0 > 0: outside the mask, z_s and v must be UNCHANGED from the
    # sigma0=0 case -- the noise must never touch the background, or
    # PairFlowConfig's "leave the background alone" property breaks.
    gen = torch.Generator().manual_seed(2)
    z_s1, s1, v1 = sample_path(z_src, z_tgt, mask=mask, sigma0=0.8, generator=gen)
    out = (mask < 0.5).expand_as(v1)
    gen = torch.Generator().manual_seed(2)
    z_s0b, _, v0b = sample_path(z_src, z_tgt, mask=mask, sigma0=0.0, generator=gen)
    assert torch.allclose(v1[out], v0b[out]), "background v must be untouched"
    # inside the mask, v must actually have moved (the whole point of sigma0).
    ins = (mask >= 0.5).expand_as(v1)
    assert not torch.allclose(v1[ins], v0b[ins]), \
        "sigma0>0 did not change anything inside the mask"

    # v is constant across s (same invariant as the sigma0=0 case) -- the
    # noise depends on the fixed per-example eps, not on s itself.
    gen = torch.Generator().manual_seed(3)
    _, _, v_a = sample_path(z_src, z_tgt, mask=mask, sigma0=0.5, generator=gen)
    gen = torch.Generator().manual_seed(3)
    # same generator seed -> same eps -> same v, independent of how many
    # times s itself is redrawn from it beforehand doesn't apply here since
    # s and eps are drawn in a fixed order inside sample_path; re-deriving
    # the call must reproduce identically.
    _, _, v_b = sample_path(z_src, z_tgt, mask=mask, sigma0=0.5, generator=gen)
    assert torch.equal(v_a, v_b)

    # apply_transport: at strength=0, sigma0=0 must return z_src EXACTLY
    # (the "do nothing" baseline every eval sweep's a=0 point relies on).
    u_stub = torch.ones(B, C, H, W)
    z_out = apply_transport(_fake_pair_model(u_stub), z_src, mask,
                            prompt_embeds=None, pooled=None,
                            strength=0.0, steps=4, sigma0=0.0)
    assert torch.equal(z_out, z_src)

    # apply_transport: at strength=0, sigma0>0 must perturb ONLY inside the
    # mask -- the background must still read back exactly z_src, even
    # though the query point itself was noised.
    gen = torch.Generator().manual_seed(4)
    z_out2 = apply_transport(_fake_pair_model(u_stub), z_src, mask,
                             prompt_embeds=None, pooled=None,
                             strength=0.0, steps=4, sigma0=0.6, generator=gen)
    assert torch.allclose(z_out2[out], z_src[out]), \
        "sigma0>0 leaked noise into the background"
    assert not torch.allclose(z_out2[ins], z_src[ins]), \
        "sigma0>0 did not perturb the masked query point"

    # Same generator -> same sampled edit (reproducibility); different
    # generator -> a different one (genuine stochasticity, the whole point).
    gen_a = torch.Generator().manual_seed(5)
    z_a = apply_transport(_fake_pair_model(u_stub), z_src, mask,
                          prompt_embeds=None, pooled=None,
                          strength=1.0, steps=1, sigma0=0.6, generator=gen_a)
    gen_b = torch.Generator().manual_seed(5)
    z_b = apply_transport(_fake_pair_model(u_stub), z_src, mask,
                          prompt_embeds=None, pooled=None,
                          strength=1.0, steps=1, sigma0=0.6, generator=gen_b)
    assert torch.equal(z_a, z_b)
    gen_c = torch.Generator().manual_seed(6)
    z_c = apply_transport(_fake_pair_model(u_stub), z_src, mask,
                          prompt_embeds=None, pooled=None,
                          strength=1.0, steps=1, sigma0=0.6, generator=gen_c)
    assert not torch.equal(z_a, z_c)

    # pairflow_loss / cos_clean: was never exercised by this self-test
    # before Direction A added it. A fixed-output stub model isolates the
    # stats computation from anything the model itself does.
    pe = torch.randn(B, 4, 8)
    po = torch.randn(B, 8)
    u_fixed = torch.randn(B, C, H, W)

    cfg0 = PairFlowConfig(mask_dropout=0.0, cond_dropout=0.0, sigma0=0.0,
                          interval_conditioning=False)
    _, stats0 = pairflow_loss(_fake_pair_model(u_fixed), z_src, z_tgt, mask,
                              pe, po, cfg0)
    # sigma0=0: v IS the true displacement already (sample_path's own
    # byte-identical guarantee), so cos and cos_clean must match exactly,
    # not approximately.
    assert abs(stats0["cos"] - stats0["cos_clean"]) < 1e-6, stats0

    cfg1 = PairFlowConfig(mask_dropout=0.0, cond_dropout=0.0, sigma0=0.7,
                          interval_conditioning=False)
    gen = torch.Generator().manual_seed(7)
    _, stats1 = pairflow_loss(_fake_pair_model(u_fixed), z_src, z_tgt, mask,
                              pe, po, cfg1, generator=gen)
    # cos_clean must be INDEPENDENT of sigma0/eps -- same u, same
    # z_tgt - z_src regardless of which noised path produced z_s. This is
    # the property that makes it comparable across a sigma0 sweep, which
    # "cos" (against the noised target) is not.
    assert abs(stats1["cos_clean"] - stats0["cos_clean"]) < 1e-6, \
        (stats1, stats0)
    # "cos" (against the noised target) DOES move once sigma0>0 -- this is
    # exactly the divergence a sweep comparison plot would otherwise
    # conflate with a real improvement.
    assert abs(stats1["cos"] - stats1["cos_clean"]) > 1e-3, stats1

    # the silent-drop guard: interval path + sigma0 > 0 must refuse
    try:
        pairflow_loss(_fake_pair_model(u_fixed), z_src, z_tgt, mask, pe, po,
                      PairFlowConfig(sigma0=0.3, interval_conditioning=True))
    except ValueError:
        pass
    else:
        raise AssertionError("interval_conditioning + sigma0>0 did not raise")

    # ablate_text_condition: zero matches cond_dropout's own zeroing;
    # shuffle moves every row and is a pure permutation.
    pe_t = torch.randn(3, 5, 8)
    po_t = torch.randn(3, 6)
    pe_z, po_z = ablate_text_condition(pe_t, po_t, "zero")
    assert not pe_z.any() and not po_z.any()
    assert pe_z.shape == pe_t.shape and po_z.shape == po_t.shape
    pe_s, po_s = ablate_text_condition(pe_t, po_t, "shuffle")
    for i in range(3):
        assert torch.equal(pe_s[i], pe_t[i - 1]) and torch.equal(po_s[i], po_t[i - 1])
        assert not torch.equal(pe_s[i], pe_t[i])
    for bad in (lambda: ablate_text_condition(pe_t[:1], po_t[:1], "shuffle"),
                lambda: ablate_text_condition(pe_t, po_t, "blank")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("ablate_text_condition accepted a bad call")

    # direction_position_diagnostics: a model stub that gets direction right
    # INSIDE the mask but predicts pure noise everywhere else. raw (whole
    # frame) must be dragged down by the background noise; pooled (masked
    # region only) must recover near-perfect alignment since it never sees
    # the background at all.
    torch.manual_seed(8)
    Bd, Cd, Hd, Wd = 3, 4, 8, 8
    mask_d = torch.zeros(Bd, 1, Hd, Wd)
    mask_d[:, :, 2:6, 2:6] = 1.0
    v_clean_d = torch.zeros(Bd, Cd, Hd, Wd)
    v_clean_d[:, :, 2:6, 2:6] = torch.randn(Bd, Cd, 1, 1)  # constant per-example inside mask
    u_d = v_clean_d.clone()
    u_d[:, :, mask_d[0, 0] < 0.5] = torch.randn(Bd, Cd, (mask_d[0, 0] < 0.5).sum())
    diag = direction_position_diagnostics(u_d, v_clean_d, mask_d)
    assert torch.all(diag["pooled"] > 0.99), diag["pooled"]
    assert torch.all(diag["raw"] < diag["pooled"]), (diag["raw"], diag["pooled"])

    # centered: subtracting the TRUE shared bias from both u and v_clean,
    # when u and v_clean are otherwise identical (perfect prediction), must
    # still give cosine ~1 -- centering a perfect match stays a perfect
    # match.
    bias_d = v_clean_d[:, :, 2, 2].mean(dim=0)  # a (C,) vector, not exactly
    # the per-channel mean used elsewhere but any fixed (C,) vector
    # exercises the subtraction path identically.
    diag_c = direction_position_diagnostics(v_clean_d, v_clean_d, mask_d, bias=bias_d)
    assert torch.all(diag_c["centered"] > 0.99) or torch.all(torch.isnan(diag_c["centered"])), \
        diag_c["centered"]

    # centered actually changes the answer when u and v_clean diverge only
    # in their shared-bias component (a case where pooled would say "aligned"
    # but centered reveals there is no genuine per-instruction signal beyond
    # the bias).
    direction = torch.randn(Cd)
    v_bias_only = torch.zeros(Bd, Cd, Hd, Wd)
    v_bias_only[:, :, 2:6, 2:6] = direction.reshape(1, Cd, 1, 1)
    u_opposite_signal = torch.zeros(Bd, Cd, Hd, Wd)
    # u matches v_clean's bias component but has the OPPOSITE genuine signal
    # on top -- pooled cosine (dominated by the large shared bias) should
    # still read positive, while centered (bias removed) should reveal the
    # true disagreement.
    big_bias = direction * 5.0
    small_signal = torch.randn(Cd)
    v_pool_case = torch.zeros(Bd, Cd, Hd, Wd)
    v_pool_case[:, :, 2:6, 2:6] = (big_bias + small_signal).reshape(1, Cd, 1, 1)
    u_pool_case = torch.zeros(Bd, Cd, Hd, Wd)
    u_pool_case[:, :, 2:6, 2:6] = (big_bias - small_signal).reshape(1, Cd, 1, 1)
    diag_case = direction_position_diagnostics(u_pool_case, v_pool_case, mask_d, bias=big_bias)
    assert torch.all(diag_case["pooled"] > 0.9), diag_case["pooled"]
    assert torch.all(diag_case["centered"] < -0.9), diag_case["centered"]

    print("pairflow self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
