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
    z_src: torch.Tensor | None = None,
    mask_cond: torch.Tensor | None = None,
    pin_background: bool = False,
) -> tuple[torch.Tensor, int]:
    """Integrate noise -> data with the average-velocity field. Returns (z0, nfe).

    TIER B (`EDITOR_PLAN.md` section 2). `z_src`/`mask_cond` are the source
    latent and the mask as INPUT CHANNELS, forwarded to the model. They are
    forged by `transport/source_cond.py::cond_from_tensors`, which is the same
    builder training uses -- so a checkpoint cannot be trained under one mask
    convention and sampled under another.

    THEY ARE FORWARDED ONLY WHEN PROVIDED, and that is deliberate. Forwarding
    them unconditionally would break every existing caller whose model does not
    take them (the frozen teacher, Gate B's stubs) with a TypeError, so the
    source_cond=False path stays byte-identical to every existing run. Whether
    the PAIR is required is the adapter's decision, not this function's: a
    source_cond adapter raises from `_condition_input` when they are missing,
    and raises when they are passed to a source_cond=False model. There is no
    silent fallback in either direction.

    `pin_background` is tier B's sampler recipe: outside the mask, take the
    source latent rather than the integrated one. See `pin_background_exact`.
    Off by default, because it is a hard blend and hard blends seam (see
    `region/alpha_field.py`'s note); it is the exact-copy analogue of
    pairflow's `pin_background`, not a soft one.
    """
    ts = grid if grid is not None else uniform_grid(steps)
    z = noise.float().clone()
    b = z.shape[0]
    nfe = 0

    cond: dict = {}
    if z_src is not None or mask_cond is not None:
        cond = {"z_src": z_src, "mask_cond": mask_cond}

    for i in range(len(ts) - 1):
        t = torch.full((b,), ts[i], device=z.device)
        r = torch.full((b,), ts[i + 1], device=z.device)
        u = model(z, t, r, prompt_embeds, pooled, mask=mask,
                  guidance=guidance, negative_embeds=negative_embeds,
                  negative_pooled=negative_pooled, **cond)
        nfe += 2 if guidance > 1.0 and negative_embeds is not None else 1
        z = z - (ts[i] - ts[i + 1]) * u

    if pin_background:
        z = pin_background_exact(z, z_src, mask_cond)

    return z, nfe


def pin_background_exact(
    z0: torch.Tensor,
    z_src: torch.Tensor | None,
    mask_cond: torch.Tensor | None,
) -> torch.Tensor:
    """Outside the mask, return the SOURCE latent instead of the sampled one.

    Tier B's recipe (`EDITOR_PLAN.md` section 2): "mask 外用源 latent 精确 pin
    (背景免费保住)". This is exact, not a blend: the data endpoints satisfy
    `z_tgt == z_src` outside the mask by construction, so at t=0 the background
    of a correct sample IS `z_src`, and pinning it removes the model's own
    background error for free -- the same thing `pairflow`'s `pin_background`
    does, and the same reason Stage 0 called the background a solved problem.

    IT IS A HARD BLEND, so it can seam at the mask boundary: the model's
    inside-mask content is not guaranteed to match `z_src` at the edge.
    `region/alpha_field.py` documents soft pinning as the answer to that, and
    the honest reading is that this trades background fidelity for a possible
    boundary discontinuity. That is a measurable trade, not a free win, which is
    why it is opt-in and reported separately.

    Mask polarity is the project's: white = editable, so `mask_cond` is the
    region REPLACED and `1 - mask_cond` is the region pinned.
    """
    if z_src is None or mask_cond is None:
        raise ValueError(
            "pin_background needs both z_src and mask_cond; got "
            f"z_src={'set' if z_src is not None else 'None'}, "
            f"mask_cond={'set' if mask_cond is not None else 'None'}. Pinning "
            "without a source would silently leave the sample unpinned.")
    m = mask_cond.to(dtype=z0.dtype)
    if m.ndim == 3:                         # (B, h, w) -> (B, 1, h, w)
        m = m.unsqueeze(1)
    if m.shape[-2:] != z0.shape[-2:]:
        raise ValueError(
            f"mask is {tuple(m.shape[-2:])} but the state is "
            f"{tuple(z0.shape[-2:])}; mask_to_latent_grid() is the thing that "
            "puts it on the latent grid")
    if m.shape[0] != z0.shape[0]:
        raise ValueError(
            f"batch sizes disagree: state {z0.shape[0]}, mask {m.shape[0]}")
    src = z_src.to(dtype=z0.dtype)
    if src.shape != z0.shape:
        raise ValueError(
            f"z_src is {tuple(src.shape)} but the state is "
            f"{tuple(z0.shape)}; the source must be the same latent geometry")
    return z0 * m + src * (1.0 - m)


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


# --------------------------------------------------------------- self-test --

class _StrictModel:
    """A model whose signature does NOT take z_src/mask_cond.

    This is the frozen teacher and every existing Gate B stub. It exists so the
    "forward only when provided" claim is a TEST rather than a hope: if `sample`
    ever forwards the tier B pair unconditionally, this raises TypeError here
    instead of breaking a real run.
    """

    def __init__(self, u):
        self.u = u
        self.calls = 0

    def __call__(self, z, t, r, prompt_embeds, pooled, mask=None, guidance=1.0,
                 negative_embeds=None, negative_pooled=None):
        self.calls += 1
        return self.u.to(z.dtype)


class _RecordingModel:
    """Accepts the tier B pair and records exactly what it was handed."""

    def __init__(self, u):
        self.u = u
        self.seen = []

    def __call__(self, z, t, r, prompt_embeds, pooled, mask=None, guidance=1.0,
                 negative_embeds=None, negative_pooled=None,
                 z_src=None, mask_cond=None):
        self.seen.append((z_src, mask_cond))
        return self.u.to(z.dtype)


def self_test() -> None:
    B, C, h, w = 2, 4, 8, 8
    x0 = torch.randn(B, C, h, w)
    eps = torch.randn(B, C, h, w)
    v = eps - x0                    # the convention: v = eps - x0, t=1 is noise
    pe = torch.randn(B, 5, 8)
    po = torch.randn(B, 8)

    # 1. A CALLER WITH NO TIER B KEEPS WORKING, and the pair is not sent.
    strict = _StrictModel(v)
    z0, nfe = sample(strict, eps, pe, po, steps=4)
    assert strict.calls == 4 and nfe == 4, (strict.calls, nfe)
    assert torch.allclose(z0, x0, atol=1e-5), (z0 - x0).abs().max()

    # 2. THE INTEGRATOR IS EXACT FOR A CONSTANT FIELD, at every step count.
    #    This is the straight-path property LAYER1_NEXT.md section 5.3 leans on:
    #    v does not vary along t, so extra steps buy nothing HERE. If step count
    #    ever starts to matter on a constant field, either the sampler broke or
    #    the path stopped being straight -- and LAYER1_NEXT's whole argument
    #    about the pair leg would have to be redone.
    for k in (1, 2, 3, 4, 8):
        zk, _ = sample(_StrictModel(v), eps, pe, po, steps=k)
        assert torch.allclose(zk, x0, atol=1e-5), (k, (zk - x0).abs().max())
    z1, _ = sample(_StrictModel(v), eps, pe, po, steps=1)
    assert torch.equal(z1, eps - v), "one step must be exactly eps - v"

    # 3. WHEN PROVIDED, THE PAIR IS FORWARDED -- by identity, not by value.
    z_src = torch.randn(B, C, h, w)
    mcond = torch.zeros(B, 1, h, w)
    mcond[:, :, 2:6, 2:6] = 1.0
    rec = _RecordingModel(v)
    sample(rec, eps, pe, po, steps=2, z_src=z_src, mask_cond=mcond)
    assert len(rec.seen) == 2, rec.seen
    assert all(a is z_src and b is mcond for a, b in rec.seen), \
        "the conditioning must reach the model unchanged"

    # 4. ...and it is not invented when absent. Half a pair is forwarded as half
    #    a pair; refusing it is the ADAPTER's job, because the adapter is the
    #    only thing that knows whether source_cond is on. Deciding it here too
    #    would give two places that can disagree.
    rec2 = _RecordingModel(v)
    sample(rec2, eps, pe, po, steps=1)
    assert rec2.seen == [(None, None)], rec2.seen
    rec3 = _RecordingModel(v)
    sample(rec3, eps, pe, po, steps=1, z_src=z_src)
    assert rec3.seen == [(z_src, None)], rec3.seen

    # 5. THE PIN. Exact outside, untouched inside, and it refuses to guess.
    sampled = x0 + torch.randn(B, C, h, w) * 0.1   # a model with bg error
    pinned = pin_background_exact(sampled, z_src, mcond)
    inside = mcond.expand_as(pinned) > 0.5
    assert torch.equal(pinned[inside], sampled[inside]), \
        "the pin must not touch the editable region"
    outside = ~inside
    assert torch.equal(pinned[outside], z_src.expand_as(pinned)[outside]), \
        "outside the mask the pin must return the source EXACTLY"
    assert not torch.allclose(pinned, sampled), "the pin did nothing"

    # a soft mask blends per element, and a (B, h, w) mask is accepted
    soft = torch.full((B, h, w), 0.25)
    got = pin_background_exact(sampled, z_src, soft)
    assert got.shape == sampled.shape
    assert torch.allclose(got, sampled * 0.25 + z_src * 0.75), got[0, 0, 0, 0]

    # ...and it refuses rather than silently leaving the sample unpinned
    for bad in (dict(z_src=None, mask_cond=mcond),
                dict(z_src=z_src, mask_cond=None),
                dict(z_src=None, mask_cond=None)):
        try:
            pin_background_exact(sampled, **bad)
        except ValueError as exc:
            assert "z_src" in str(exc), exc
        else:
            raise AssertionError(f"pin with {bad} must raise")
    # a mask on the wrong grid (the pixel grid) must not broadcast quietly
    try:
        pin_background_exact(sampled, z_src, torch.ones(B, 1, h * 8, w * 8))
    except ValueError as exc:
        assert "mask_to_latent_grid" in str(exc), exc
    else:
        raise AssertionError("a pixel-grid mask must be refused")
    # a source with the wrong geometry must not broadcast quietly either
    try:
        pin_background_exact(sampled, torch.randn(B, C, h, w + 1), mcond)
    except ValueError as exc:
        assert "latent geometry" in str(exc), exc
    else:
        raise AssertionError("a mismatched source must be refused")

    # 6. PINNING IS OFF BY DEFAULT: sample() alone must not pin.
    plain, _ = sample(_RecordingModel(v), eps, pe, po, steps=1,
                      z_src=z_src, mask_cond=mcond)
    assert torch.allclose(plain, x0, atol=1e-5), (plain - x0).abs().max()
    onp, _ = sample(_RecordingModel(v), eps, pe, po, steps=1,
                    z_src=z_src, mask_cond=mcond, pin_background=True)
    # The pinned background is EXACTLY the source, whatever the model produced
    # there -- that is the whole claim, and it is bit-exact by construction so
    # it is asserted with `equal`, not `allclose`.
    m_b = mcond.expand_as(onp)
    assert torch.equal(onp[m_b <= 0.5], z_src.expand_as(onp)[m_b <= 0.5]), \
        "pin_background=True must hand back the source outside the mask"
    # ...and the EDITABLE region is the sampler's own output, untouched by the
    # pin. Compared against `plain` (same call, pin off) rather than against x0,
    # because z0 = eps - v only equals x0 up to float rounding.
    assert torch.equal(onp[m_b > 0.5], plain[m_b > 0.5]), \
        "the pin must leave the editable region exactly as the sampler left it"

    # 7. NFE ACCOUNTING IS UNCHANGED by tier B: it adds no model calls, and the
    #    whole point of this sampler is that NFE is reported honestly next to
    #    every quality number.
    assert sample(_RecordingModel(v), eps, pe, po, steps=3)[1] == 3
    assert sample(_RecordingModel(v), eps, pe, po, steps=3, guidance=5.0,
                  negative_embeds=pe, negative_pooled=po)[1] == 6

    print("sampling self-test OK")
    print("  a model without tier B kwargs still runs (the pair is not sent)")
    print("  constant field: z0 == x0 at 1,2,3,4,8 steps (no interval info)")
    print("  the tier B pair reaches the model UNCHANGED when provided")
    print("  pin_background_exact: exact outside, untouched inside, opt-in")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
