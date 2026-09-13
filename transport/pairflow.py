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


def sample_path(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform s along the straight path. Returns (z_s, s, v).

    s is uniform, NOT logit-normal. The logit-normal schedule in meanflow.py
    exists to match SD3's noise-level training distribution; there is no noise
    here and no reason to prefer any point on the path. Copying that choice
    across would over-sample s ~ 0.5 and starve s = 0, which is the only point
    the one-step readout ever evaluates.
    """
    b = z_src.shape[0]
    s = torch.rand(b, device=z_src.device, generator=generator)
    s_ = s.reshape(-1, 1, 1, 1)
    z_s = (1 - s_) * z_src + s_ * z_tgt
    v = z_tgt - z_src
    return z_s, s, v


def pairflow_loss(
    model,
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    mask: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    cfg: PairFlowConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Masked rectified-flow regression from source to target.

    model is a TransportAdapter in mode="pair". r is passed equal to s only to
    satisfy the signature; in pair mode the gate ignores it (gate = 1), which
    is the whole reason that mode exists.
    """
    z_s, s, v = sample_path(z_src, z_tgt, generator)

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

    u = model(z_s, s, s, pe, po, mask=m)

    err = (u - v).pow(2)
    inside = mask
    outside = 1.0 - mask

    n_in = inside.sum().clamp(min=1.0)
    n_out = outside.sum().clamp(min=1.0)
    l_in = (err * inside).sum() / (n_in * err.shape[1])
    l_out = (err * outside).sum() / (n_out * err.shape[1])

    loss = cfg.inside_weight * l_in + cfg.outside_weight * l_out

    with torch.no_grad():
        # Per-element means throughout. The first version reported u_norm_in
        # and v_norm_in as sqrt(sum/n_in) while summing over all 16 latent
        # channels, making them 4x the per-element RMS -- not comparable with
        # loss_in, which IS a per-element mean. That hid the only question
        # that matters early on: is the model beating a prediction of zero?
        C = err.shape[1]
        v2_in = float((v.pow(2) * inside).sum() / (n_in * C))
        u2_in = float((u.pow(2) * inside).sum() / (n_in * C))
        v2_out = float((v.pow(2) * outside).sum() / (n_out * C))

        stats = {
            "loss": float(loss),
            "loss_in": float(l_in),
            "loss_out": float(l_out),
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
    return loss, stats


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
) -> torch.Tensor:
    """z_src -> z_edit. steps=1 is the headline; steps>1 integrates the path.

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
    b = z.shape[0]
    m = mask if pin_background else torch.ones_like(mask)
    grid = torch.linspace(0.0, strength, steps + 1, device=z.device)
    for i in range(steps):
        s = torch.full((b,), float(grid[i]), device=z.device)
        u = model(z, s, s, prompt_embeds, pooled, mask=mask)
        z = z + float(grid[i + 1] - grid[i]) * u * m
    return z
