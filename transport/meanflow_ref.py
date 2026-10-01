"""
Reference-faithful MeanFlow, ported from the official JAX implementation.

SOURCE: Gsunshine/meanflow, meanflow.py (MIT), the official code for
"Mean Flows for One-step Generative Modeling" (arXiv 2505.13447).
This module mirrors its `sample_tr`, `guidance_fn`, `cond_drop` and `forward`
as closely as PyTorch allows. It exists as a SEPARATE module from
transport/meanflow.py so the baseline stays frozen and every deviation is
introduced deliberately, one at a time, rather than accumulating.

WHY IT EXISTS. transport/meanflow.py was written from the identity without
consulting the reference, and three differences turned out to matter:

  1. STRUCTURAL. The reference forces r = t on `data_proportion` (0.75) of
     every batch. Those samples reduce to plain flow matching (u* = v) and
     anchor the interval field to the instantaneous one. The Delta-gated
     adapter cannot use them at all: with every trainable path multiplied by
     (t - r), the gradient at r = t is exactly zero. The gate bought an exact
     boundary condition and paid for it with 75% of the training signal.

  2. The Delta curriculum (delta_start/delta_end/delta_schedule) is not part of
     the method. The reference samples both endpoints from the same
     distribution from step one; the r = t population is what provides
     stability. `delta_end = 1.0` was a guess in a custom formulation, and the
     target explosion it produced is therefore not evidence about SD3.

  3. The loss reduction differs by ~65536x. The reference SUMS squared error
     over all non-batch dims, then weights by (loss + 0.01)^1.0. Ours took a
     MEAN and used (loss + 1e-3)^0.75. Same algebra, entirely different
     numerical regime -- with p = 1 the weighting makes the loss
     approximately scale-free, which is not what p = 0.75 on a mean does.

WHAT WAS ALREADY RIGHT and is kept: conditioning on (t, h = t - r) rather than
(t, r); the JVP with tangent (v, 1, 0) in (z, t, r); the target
u* = v - (t - r) du/dt; the stop-gradient on it.

DEVIATIONS STILL PRESENT, deliberately, and to be justified individually:
  - text conditioning instead of class labels, so `cond_drop` blanks the
    prompt embedding rather than swapping in a null class id
  - a LoRA adapter over a frozen pretrained model instead of training a DiT
    from scratch, which is the whole point of Gate B

OPTIONAL DEVIATIONS, all off by default so the reference path is unchanged:
  - `tangent`: which velocity the z-tangent of du/dt uses. "boundary" is iMF's
    u_theta(z_t, t, t) (arXiv 2512.02012, Eq. 12); "frozen" is the pretrained
    field. Motivated by scripts/probe_target_quality.py: at t_center 0.5 the
    reference target (total/cond) had correction cos 0.09-0.13 against the
    integrated average velocity, and swapping the tangent alone (total/marg)
    barely moved it. The tangent only matters combined with:
  - `estimator="fd"` at eps ~0.025-0.05, which averages over the fast
    components of SD3's time conditioning instead of differentiating them
    (fd/marg: cos 0.55-0.88 for Delta <= 0.5, 0.45 at Delta = 0.8).
  - `max_delta` / `max_t`: training-range caps, pending the probe's t-sweep
    that separates interval length from the high-t region.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .meanflow import math_attention


@dataclass
class RefMeanFlowConfig:
    """Defaults are the reference's, not ours. Do not tune before replicating."""

    #: logit-normal endpoint sampling. Reference: P_mean=-0.4, P_std=1.0.
    p_mean: float = -0.4
    p_std: float = 1.0
    #: draw the anchor mask stochastically rather than as a fixed prefix.
    #: Required at small batch: the reference's int(bz * 0.75) gives 1 of 2
    #: at batch 2, i.e. 0.5 anchors every step. Set False to reproduce the
    #: reference's exact indexing at large batch.
    stochastic_anchor: bool = True
    #: fraction of the batch forced to r == t. Reference: 0.75. These are the
    #: flow-matching anchors, and they are the single most important thing the
    #: gated implementation was missing.
    data_proportion: float = 0.75
    #: adaptive weighting, applied to a SUM-reduced per-sample loss.
    #: Reference: norm_p=1.0, norm_eps=0.01.
    norm_p: float = 1.0
    norm_eps: float = 0.01
    #: guidance. Reference run_b4 uses cfg with omega=1.0, kappa=0.5.
    guidance_eq: str = "cfg"
    omega: float = 1.0
    kappa: float = 0.5
    cond_dropout_prob: float = 0.1
    estimator: str = "jvp"
    fd_eps: float = 1e-3
    #: z-TANGENT of du/dt, for both estimators.
    #:   "cond"      v_g (e - x0, guided). The reference. Default.
    #:   "boundary"  u_theta(z_t, t, t) under no_grad. iMF's boundary-condition
    #:               tangent: the network's own instantaneous velocity. Costs
    #:               one no-grad adapter forward.
    #:   "frozen"    the pretrained field v(z_t, t) under no_grad. What the
    #:               target probe's marg rows measured; identical to
    #:               "boundary" at initialisation, but does not inherit the
    #:               adapter's r-independent drift later. Needs base_fn.
    #: The regression target's FIRST term stays v_g either way, as in iMF.
    tangent: str = "cond"
    #: FIRST TERM of the regression target, target = first - Delta * du/dt.
    #:   "cond"    v_g (e - x0, guided). Reference and iMF. Default.
    #:   "frozen"  the pretrained field v(z_t, t). Anchors then DISTIL the
    #:             teacher exactly (u(z,t,t) -> v) instead of re-fitting the
    #:             instantaneous field to this dataset. Tests whether the
    #:             ~900 SSE r-independent excess every run shares (concat 913,
    #:             centred 908, fd 904 at Delta=0.05) comes from that re-fit.
    #:             Removes the flow-matching noise floor from the loss, which
    #:             changes what norm_p does -- see __post_init__.
    first_term: str = "cond"
    #: Training-range caps, applied in sample_tr before the anchor mask.
    #: 1.0 = no cap = the reference. max_delta clamps r >= t - max_delta
    #: (mass piles up AT the cap; the loss logs how much). max_t clamps t.
    max_delta: float = 1.0
    max_t: float = 1.0
    #: H3 (H3_SIGNED_DELTA_DESIGN.md Sec.4). Fraction of rows flipped to the
    #: reverse direction r > t, AFTER the caps, by swapping the endpoints of
    #: the already-capped interval ("option A, cap-then-flip"): a reverse row
    #: covers exactly the interval its forward twin would, queried from the
    #: LOWER endpoint. So |Delta| is an exact mirror, the caps need no reverse
    #: form, and the t marginal of reverse rows is the lower-endpoint marginal
    #: (mean ~0.31 vs ~0.53) -- compare directions conditioned on t.
    #: Drawn from its OWN generator, so (a, b, anchor) are identical to the
    #: forward-only arm at every step and arms stay row-paired.
    #: 0.0 = today, bit for bit (no draw is made). Needs a signed embedding;
    #: gate_b_meanflow.py enforces that.
    reverse_frac: float = 0.0
    #: GRADIENT CHECKPOINTING FORCES A DEVIATION FROM THE REFERENCE.
    #: torch.utils.checkpoint wraps the forward in a custom autograd.Function
    #: (_NoopSaveInputs) with no jvp rule, so forward-mode AD cannot pass
    #: through it:
    #:
    #:     NotImplementedError: You must implement the jvp function for
    #:     custom autograd.Function to use it with forward mode AD
    #:
    #: The reference takes u OUT of the jvp call, where it carries gradient.
    #: With checkpoint_safe=True we instead take du/dt from a no_grad jvp and
    #: compute u with an ordinary forward. That is MATHEMATICALLY IDENTICAL --
    #: same function, same value, same gradient, and the target is detached
    #: either way -- at the cost of one extra forward. It is a deviation in
    #: structure, not in objective, and it is recorded here rather than left
    #: implicit.
    #:
    #: Set False only with checkpointing disabled, to run the reference
    #: structure exactly. Expect much higher memory: fp32 weights are ~8GB
    #: before activations.
    checkpoint_safe: bool = True

    def __post_init__(self) -> None:
        if self.tangent not in TANGENTS:
            raise ValueError(f"tangent must be one of {TANGENTS}, "
                             f"got {self.tangent!r}")
        for name in ("max_delta", "max_t"):
            val = getattr(self, name)
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {val}")
        if not 0.0 <= self.reverse_frac <= 1.0:
            raise ValueError(f"reverse_frac must be in [0, 1], "
                             f"got {self.reverse_frac}")
        if self.fd_eps <= 0:
            raise ValueError(f"fd_eps must be > 0, got {self.fd_eps}")
        if self.first_term not in FIRST_TERMS:
            raise ValueError(f"first_term must be one of {FIRST_TERMS}, "
                             f"got {self.first_term!r}")
        # The frozen first term replaces the guided empirical velocity; a
        # guided frozen first term is a different objective nobody has
        # specified. Refuse rather than silently ignore omega/kappa.
        if self.first_term == "frozen" and (self.omega != 1.0
                                            or self.kappa != 0.0):
            raise ValueError("first_term='frozen' requires omega=1, kappa=0")


TANGENTS = ("cond", "boundary", "frozen")
FIRST_TERMS = ("cond", "frozen")


def orient_reverse(t: torch.Tensor, r: torch.Tensor, reverse_frac: float,
                   rev_generator: torch.Generator | None):
    """H3 option A, cap-then-flip. Swap (t, r) on a Bernoulli(reverse_frac)
    subset. Returns (t, r, reverse_mask).

    Call AFTER every cap. Anchor rows (r == t) are unchanged by a swap, so the
    call may come before or after the anchor mask; the effective reverse
    INTERVAL fraction is reverse_frac * (1 - data_proportion).

    reverse_frac == 0 returns the inputs untouched and draws NOTHING, so a
    forward-only run's RNG streams are byte-identical to before H3.
    """
    if reverse_frac <= 0.0:
        return t, r, torch.zeros_like(t, dtype=torch.bool)
    if rev_generator is None:
        # A shared stream would shift every later (a, b) draw and unpair the
        # arm from its forward-only control. Refuse rather than drift.
        raise ValueError("reverse_frac > 0 needs its own rev_generator")
    rev = (torch.rand(t.shape[0], device=t.device, generator=rev_generator)
           < reverse_frac)
    return torch.where(rev, r, t), torch.where(rev, t, r), rev


def sample_tr(batch: int, cfg: RefMeanFlowConfig, device,
              generator: torch.Generator | None = None,
              rev_generator: torch.Generator | None = None):
    """Reference `sample_tr`: two logit-normal draws, sorted, then r := t on
    the first `data_proportion` of the batch.

    Note what this is NOT: there is no curriculum and no schedule. Delta =
    t - r spans its full range from the first step, and the r == t population
    is what keeps that stable. The optional static caps (cfg.max_delta,
    cfg.max_t) default to 1.0, where they are no-ops; when set they restrict
    the range for the whole run, never as a function of step.
    """
    def draw():
        n = torch.randn(batch, device=device, generator=generator)
        return torch.sigmoid(n * cfg.p_std + cfg.p_mean)

    a, b = draw(), draw()
    t = torch.maximum(a, b)
    r = torch.minimum(a, b)
    # Caps (no-ops at the default 1.0). Before the anchor mask, so anchors
    # stay exactly r == t.
    if cfg.max_t < 1.0:
        t = t.clamp(max=cfg.max_t)
        r = torch.minimum(r, t)
    if cfg.max_delta < 1.0:
        r = torch.maximum(r, t - cfg.max_delta)

    # STOCHASTIC, not a deterministic prefix. The reference uses
    # `jnp.arange(bz) < int(bz * data_proportion)`, which is fine at its batch
    # size (int(256*0.75) = 192) but wrong at ours: int(2*0.75) = 1 gives
    # exactly 0.5 anchors EVERY step, a systematic 33% shortfall rather than
    # noise. A Bernoulli draw preserves the proportion in expectation at any
    # batch size, and matches the reference exactly as batch -> large.
    if cfg.stochastic_anchor:
        anchor = (torch.rand(batch, device=device, generator=generator)
                  < cfg.data_proportion)
    else:
        anchor = (torch.arange(batch, device=device)
                  < int(batch * cfg.data_proportion))
    r = torch.where(anchor, t, r)
    t, r, _ = orient_reverse(t, r, cfg.reverse_frac, rev_generator)
    return t, r, anchor


def guided_velocity(v, z_t, t, base_fn, cond, uncond, cfg: RefMeanFlowConfig):
    """Reference `guidance_fn`, text-conditional.

    v is the EMPIRICAL velocity (eps - x0). The reference mixes it with model
    velocities rather than replacing it:

        kappa == 0:  v_g = v_uncond + omega * (v - v_uncond)
        kappa >  0:  v_g = omega*v + (1 - omega - kappa)*v_uncond + kappa*v_cond

    base_fn(z, t, embeds) -> instantaneous velocity from the FROZEN model.
    Ours substitutes the frozen pretrained field for the reference's
    `v_fn` (the model at h=0), because in Gate B the base model is the thing
    being adapted rather than trained.
    """
    if cfg.guidance_eq != "cfg":
        return v
    # SHORT-CIRCUIT. With omega=1 and kappa=0 the expression collapses to
    #     v_g = v_un + 1*(v - v_un) = v
    # exactly, so the base_fn call is pure waste -- and each one is a full
    # forward through a 2B-parameter model in fp32. The kappa=0.5 default
    # costs TWO forwards per step (uncond and cond) out of roughly seven
    # forward-equivalents, so turning guidance off is worth ~30% of step time.
    if cfg.omega == 1.0 and cfg.kappa == 0.0:
        return v
    with torch.no_grad():
        v_un = base_fn(z_t, t, uncond)
        if cfg.kappa == 0:
            return v_un + cfg.omega * (v - v_un)
        v_c = base_fn(z_t, t, cond)
    return cfg.omega * v + (1 - cfg.omega - cfg.kappa) * v_un + cfg.kappa * v_c


def cond_drop(v, v_g, cond, uncond, cfg: RefMeanFlowConfig, generator=None):
    """Reference `cond_drop`: blank the condition AND revert the guided target
    to the plain velocity for the dropped samples.

    Both halves matter. Dropping the condition without reverting v_g would ask
    the unconditional model to reproduce a guided target.
    """
    b = v.shape[0]
    drop = (torch.rand(b, device=v.device, generator=generator)
            < cfg.cond_dropout_prob)
    pe, po = cond
    npe, npo = uncond
    keep = (~drop).to(pe.dtype)
    pe_in = keep.reshape(-1, 1, 1) * pe + (1 - keep).reshape(-1, 1, 1) * npe
    po_in = keep.reshape(-1, 1) * po + (1 - keep).reshape(-1, 1) * npo
    v_g = torch.where(drop.reshape(-1, 1, 1, 1), v, v_g)
    return (pe_in, po_in), v_g


def tangent_velocity(fn, base_fn, z_t, t, v_g, cfg: RefMeanFlowConfig):
    """The z-tangent for du/dt. Detached in every mode."""
    if cfg.tangent == "cond":
        return v_g
    with torch.no_grad():
        if cfg.tangent == "boundary":
            w = fn(z_t, t, t)
        else:  # "frozen"
            if base_fn is None:
                raise ValueError("tangent='frozen' needs base_fn(z, t)")
            w = base_fn(z_t, t)
    return w.detach().to(v_g.dtype)


def central_difference(fn, z_t, t, r, w, eps: float):
    """(u(z + e w, t + e, r) - u(z - e w, t - e, r)) / 2e, r held fixed.

    The half-widths shrink near the ends of [0, 1] so no evaluation leaves the
    trained time range; the divisor is the actual span. In the interior this
    is exactly the reference's symmetric stencil.

    KNOWN BIAS for interval samples with |Delta| < eps: the stencil straddles
    Delta = 0. With an UNSIGNED embedding the far point reads as r == t (the
    clamp). With a SIGNED embedding it reads the OTHER direction's sheet --
    correct if the field is smooth through Delta = 0, contaminated by any kink
    there (H3 design Sec.7.2). This is why signedness is a flag and not simply
    the new default: unclamping changes the forward TARGET of the ~6.8% of
    non-anchor rows with Delta < 0.025 (13% at fd_eps 0.05) even with
    reverse_frac = 0. The target term is multiplied by Delta, so the weight is
    small but not zero; the loss logs the straddling fraction.
    """
    ep = torch.clamp(1.0 - t, min=0.0, max=eps)
    em = torch.clamp(t, min=0.0, max=eps)
    sh = (-1,) + (1,) * (z_t.dim() - 1)
    with torch.no_grad():
        up = fn(z_t + ep.reshape(sh) * w, t + ep, r)
        um = fn(z_t - em.reshape(sh) * w, t - em, r)
    span = (ep + em).clamp(min=1e-12).reshape(sh)
    return ((up - um) / span).detach()


def ref_meanflow_loss(u_fn, z_t, t, r, v_g, cfg: RefMeanFlowConfig,
                      base_fn=None, anchor_mask=None):
    """Reference `forward`, loss section.

    u_fn(z, t, r) -> u, closing over the conditioning. It must take (t, r) and
    internally condition on (t, h = t - r), matching the reference's
    `u_fn(z_t, t, t - r, y)`.

    base_fn(z, t) -> frozen pretrained velocity, closing over the same
    conditioning as u_fn. Only used by tangent='frozen'.

    The JVP is taken with respect to (z, t, r) with tangents (w, 1, 0), where w
    is v_g in the reference and otherwise per `cfg.tangent`. The target is
    v_g - (t - r) * du/dt in every mode.

    anchor_mask: which samples are anchors, for `sse_anchor` / `sse_interval`
    and `frac_anchor` in the returned stats. Defaults to the structural
    inference `(t - r) < 1e-8`, which is correct for the reference sampler but
    silently wrong under a curriculum whose anchors are not exactly r == t
    (delta_curriculum.py's `anchor_mode="small_delta"`): every such anchor has
    t - r > 0, so the inferred mask calls it an interval sample instead. Pass
    the sampler's own boolean mask to avoid that.
    """
    def fn(z_, t_, r_):
        return u_fn(z_, t_, r_)

    dt = torch.ones_like(t)
    dr = torch.zeros_like(r)
    w = tangent_velocity(fn, base_fn, z_t, t, v_g, cfg)

    if cfg.estimator == "jvp":
        if cfg.checkpoint_safe:
            # du/dt under no_grad: forward-mode AD never meets the checkpoint
            # machinery. u then comes from an ordinary grad-enabled forward.
            with torch.no_grad(), math_attention():
                _, du_dt = torch.func.jvp(fn, (z_t, t, r), (w, dt, dr))
            du_dt = du_dt.detach()
            u = fn(z_t, t, r)
        else:
            with math_attention():
                u, du_dt = torch.func.jvp(fn, (z_t, t, r), (w, dt, dr))
    else:
        du_dt = central_difference(fn, z_t, t, r, w, cfg.fd_eps)
        u = fn(z_t, t, r)

    if cfg.first_term == "cond":
        first = v_g
    elif cfg.tangent == "frozen":
        first = w                      # already the frozen field; no forward
    else:
        if base_fn is None:
            raise ValueError("first_term='frozen' needs base_fn(z, t)")
        with torch.no_grad():
            first = base_fn(z_t, t).detach().to(v_g.dtype)

    # Signed (H3). clamp(-1, 1) is the identity on everything a forward-only
    # sampler can emit (t - r in [0, 1]), so forward rows are bit-identical.
    delta = (t - r).clamp(-1.0, 1.0).reshape(-1, *([1] * (u.dim() - 1)))
    u_tgt = (first - delta * du_dt).detach()

    # SUM over non-batch dims, not mean. This is the 65536x that put our
    # adaptive weighting in a different regime.
    per_sample = (u - u_tgt).pow(2).flatten(1).sum(1)
    adp_wt = (per_sample + cfg.norm_eps) ** cfg.norm_p
    loss = (per_sample / adp_wt.detach()).mean()

    with torch.no_grad():
        # abs(): without it every reverse row (t - r < 0) would be inferred an
        # anchor -- the same re-inference bug as small_delta, other sign.
        anchor = (t - r).abs() < 1e-8 if anchor_mask is None else anchor_mask
        rev = (r - t) > 1e-8
        stats = {
            "loss": float(loss),
            "sse": float(per_sample.mean()),
            "delta_mean": float((t - r).mean()),
            "frac_anchor": float(anchor.float().mean()),
            "u_norm": float(u.flatten(1).norm(dim=1).mean()),
            "dudt_norm": float(du_dt.flatten(1).norm(dim=1).mean()),
            "target_norm": float(u_tgt.flatten(1).norm(dim=1).mean()),
            # `loss` saturates at 1.0 by design: with norm_p=1 the weight is
            # (sse + eps), so loss = sse/(sse+eps) ~ 1 for any sse >> eps.
            # It is a per-sample gradient normaliser, not a progress signal.
            # Read sse_anchor and sse_interval instead.
            "loss_is_saturated": bool(float(loss) > 0.99),
            # 0.0 exactly for tangent='cond'. Nonzero otherwise, and the
            # trainer-side guard checks that it is -- a tangent flag that
            # never reached the loss would otherwise look normal.
            "tangent_rel_vg": float(((w - v_g).flatten(1).norm(dim=1)
                                     / v_g.flatten(1).norm(dim=1)
                                     .clamp(min=1e-12)).mean()),
            "delta_max": float((t - r).max()),
            # 0.0 exactly for first_term='cond'; guarded trainer-side.
            "first_rel_vg": float(((first - v_g).flatten(1).norm(dim=1)
                                   / v_g.flatten(1).norm(dim=1)
                                   .clamp(min=1e-12)).mean()),
            # ADAPTIVE-WEIGHT CONCENTRATION. Per-sample gradient weight is
            # 1/(sse + eps)^p, so one low-sse sample can dominate a step.
            # ess = (sum w)^2 / (N sum w^2): 1.0 = uniform, 1/N = one sample
            # carries the step. sse_min and t_at_sse_min name the culprit.
            "adp_ess": float(((1.0 / adp_wt).sum() ** 2)
                             / (per_sample.numel()
                                * ((1.0 / adp_wt) ** 2).sum())),
            "sse_min": float(per_sample.min()),
            "t_at_sse_min": float(t[per_sample.argmin()]),
            # clamping piles probability mass at the cap; 0.0 with no cap
            "frac_at_delta_cap": float((((t - r).abs() >= cfg.max_delta - 1e-6)
                                        & ~anchor).float().mean()),
            # H3. `delta_max`/`delta_mean` keep their forward-only meaning
            # (identical logs for every existing config); these carry both.
            "frac_reverse": float(rev.float().mean()),
            "delta_abs_max": float((t - r).abs().max()),
            # rows whose FD stencil crosses Delta = 0 (see central_difference)
            "frac_fd_straddle": float((((t - r).abs() < cfg.fd_eps) & ~anchor)
                                      .float().mean())
            if cfg.estimator != "jvp" else 0.0,
        }
        # The anchors are the diagnostic that matters: at r == t the target is
        # exactly v_g, so this is plain flow matching and its loss should fall
        # like one. If it does not move, the adapter is not learning from
        # them -- which is precisely what the Delta-gate caused.
        if anchor.any():
            stats["sse_anchor"] = float(per_sample[anchor].mean())
        # sse_interval = FORWARD interval rows only, so it means today's thing
        # in every run; reverse rows get their own series.
        fwd = ~anchor & ~rev
        if fwd.any():
            stats["sse_interval"] = float(per_sample[fwd].mean())
        if (rev & ~anchor).any():
            stats["sse_reverse"] = float(per_sample[rev & ~anchor].mean())
    return loss, stats
