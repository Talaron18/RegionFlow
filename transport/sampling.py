"""
Few-step sampling with an average-velocity field.

The whole point of u(z, r, t) is that a jump from t to r is one model call:

    z_r = z_t - (t - r) * u(z_t, r, t)

so a k-step sampler is k calls, and k = 1 is a single call from pure noise.
Compare bench/inversion.py, where heun-50 costs 200 model evaluations.

NFE ACCOUNTING. Report NFE, not steps, and report it next to every quality
number. Stage 0 already established the habit (heun-50 vs euler-100 at matched
NFE) and Stage 1's speed claim is worthless without it. CFG doubles NFE; a
model distilled from a guided teacher runs at guidance 1.0 and does not, which
is where most of the practical speedup actually comes from.

WHAT THIS IS NOT. This sampler is infrastructure. It is not the contribution,
it will not beat a distillation lab's numbers, and the report should not
present it as a result. It exists so that instance-selective editing can be
measured at a usable cost.
"""

from __future__ import annotations

import numpy as np
import torch


def uniform_grid(steps: int) -> list[float]:
    """t = 1 -> 0 in equal steps."""
    return [float(x) for x in np.linspace(1.0, 0.0, steps + 1)]


def shifted_grid(steps: int, shift: float = 3.0) -> list[float]:
    """SD3's resolution-shifted schedule, which spends more steps at high t.

    Matches the spacing the pretrained model was trained under. At small step
    counts the choice of grid matters more than the solver did in Stage 0, so
    sweep it rather than assuming; `uniform_grid` is the control.
    """
    xs = np.linspace(1.0, 0.0, steps + 1)
    return [float(shift * x / (1 + (shift - 1) * x)) for x in xs]


@torch.no_grad()
def sample(
    model,
    noise: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    steps: int = 1,
    mask: torch.Tensor | None = None,
    guidance: float = 1.0,
    negative_embeds: torch.Tensor | None = None,
    negative_pooled: torch.Tensor | None = None,
    grid: list[float] | None = None,
) -> tuple[torch.Tensor, int]:
    """Integrate noise -> data with the average-velocity field. Returns (z0, nfe)."""
    ts = grid if grid is not None else uniform_grid(steps)
    z = noise.float().clone()
    b = z.shape[0]
    nfe = 0

    for i in range(len(ts) - 1):
        t = torch.full((b,), ts[i], device=z.device)
        r = torch.full((b,), ts[i + 1], device=z.device)
        u = model(z, t, r, prompt_embeds, pooled, mask=mask,
                  guidance=guidance, negative_embeds=negative_embeds,
                  negative_pooled=negative_pooled)
        nfe += 2 if guidance > 1.0 and negative_embeds is not None else 1
        z = z - (ts[i] - ts[i + 1]) * u

    return z, nfe


@torch.no_grad()
def reconstruct_from_teacher(
    model,
    z0: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    steps: int = 1,
    seed: int = 0,
) -> dict:
    """Gate B's quality probe, without needing FID.

    Draw a fixed noise, run the frozen teacher at high NFE to get a reference
    image, then run the student at k steps from the SAME noise. Report PSNR and
    LPIPS between them.

    This measures flow-map agreement rather than distributional quality, which
    is what Gate B is actually asking: has the student learned the pretrained
    model's transport, or a different one? It also needs no reference dataset
    and no FID sample budget, both of which are out of reach here. FID belongs
    in the paper; this belongs in the training loop.
    """
    raise NotImplementedError(
        "Wired up in scripts/gate_b_meanflow.py, which owns the teacher "
        "trajectory cache. Kept here as documentation of the intended probe."
    )
