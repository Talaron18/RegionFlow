"""
The MeanFlow objective, and the derivative it needs.

IDENTITY. For a flow with instantaneous velocity v(z, t), define the average
velocity over [r, t]:

    u(z_t, r, t) = 1/(t - r) * integral_r^t v(z_s, s) ds

Differentiating the integral form in t gives the relation that makes this
trainable without ever computing an integral:

    u(z, r, t) = v(z, t) - (t - r) * d/dt u(z, r, t)

where d/dt is the TOTAL derivative along the trajectory,

    d/dt u = (du/dz) . v  +  du/dt

which is exactly one Jacobian-vector product with tangent (v, 0, 1) in
(z, r, t). Training regresses u_theta onto the right-hand side with the target
stop-gradiented.

At r = t the second term vanishes and the identity collapses to u = v. Our
adapter enforces that architecturally (gating.py), so the objective and the
architecture agree at the boundary instead of fighting over it.

WHY THE JVP IS THE RISK ITEM. It is forward-mode autodiff through the whole
MMDiT. Stage 0's smoke test hit signal 11 on four separate forward-mode probes
and the cause was never isolated past "harness or CUDA forward-AD"
(docker/triage_jvp.py). Stage 1 cannot proceed on `jvp` alone, so this module
offers three interchangeable estimators behind one interface:

    "jvp"     torch.func.jvp, under the SDPA math backend. Exact. Requires
              math_attention() -- the fused attention kernels have no
              forward-AD rule and raise NotImplementedError. Costs memory,
              because the math path materialises the attention matrix.
    "fd"      central finite difference along the tangent. No forward-mode AD
              anywhere. MEASURED ON THIS MODEL: at eps=1e-2 it recovers ~11%
              of the JVP's magnitude and is essentially orthogonal to it
              (rel_err 0.9999), and its sweep is still rising at eps=1e-4.
              Under-resolved by curvature, not by rounding. Do not use it
              without first extending the sweep below 1e-4 and finding a
              plateau that agrees with the JVP.
    "fd1"     one-sided difference. One extra forward. Use only if memory
              forces it; the O(eps) bias is a real bias, not noise, and it
              biases the term the whole method depends on.

GRADIENT CHECKPOINTING. torch.utils.checkpoint replays the forward under
no_grad, which historically does not compose with forward-mode AD. With the
restructure above the JVP runs under no_grad and outside the training graph, so
the two should not meet -- but verify it on a handful of steps before starting
a multi-block run, because the failure mode is a wrong number rather than an
exception.

MEASURED ON THIS PROJECT'S HARDWARE. The first probe run reported a bf16 dtype
error, which masked the real cause; in fp32 it surfaced as
NotImplementedError on _scaled_dot_product_efficient_attention. Forward-mode AD
was never broken -- one fused kernel simply lacks a rule. This also resolves
the Stage 0 signal-11 item, which was never closed.

Probe 2 then passed at 8.7 GB peak (fp32, batch 1, 512px, inference only).
Training adds adapter gradients, optimiser state and activations on top of
that, so 8.7 GB is a floor, not a budget.

The finite-difference fallback did NOT converge in that probe, in fp32, at any
epsilon from 1e-4 to 1e-1: the estimate fell monotonically with no plateau.
Some of that is a bad probe (the tangent was a random direction rather than the
model's own velocity), but until a sweep shows a plateau, "fd" is not
validated on this model and should not be used as if it were. Prefer "jvp"
under math_attention(), and treat "fd" as the thing you fall back to only after
its own convergence sweep passes.

Run scripts/probe_jvp_mmdit.py to pick one. Record the choice in the run
config -- "which derivative estimator" is exactly the kind of silent variable
that Stage 0's withdrawn-claims table is full of.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

import torch


@contextlib.contextmanager
def math_attention():
    """Force SDPA's math backend for the duration of a block.

    PyTorch's fused attention kernels have no forward-mode AD rule:

        NotImplementedError: Trying to use forward AD with
        _scaled_dot_product_efficient_attention that does not support it

    The math backend is built from ordinary differentiable ops and does support
    it. This is the whole fix -- without it torch.func.jvp cannot pass through
    any MMDiT block, with it the JVP is exact.

    It is NOT free. The math path materialises the full attention matrix
    instead of tiling it, so peak memory rises sharply and throughput falls.
    Scope it to the JVP call only; the frozen teacher forward and everything
    else should keep the fused kernels.

    Measure the memory before assuming a batch size: at 512px SD3 sees 1024
    image tokens plus 333 text tokens, so each head materialises a 1357x1357
    matrix.
    """
    try:                                    # torch >= 2.3
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except ImportError:
        pass
    try:                                    # older API
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            yield
        return
    except Exception:                       # noqa: BLE001
        pass
    raise RuntimeError(
        "could not force the SDPA math backend; torch.func.jvp will fail "
        "inside attention. Use estimator='fd' instead."
    )


@dataclass
class MeanFlowConfig:
    estimator: str = "fd"          # "jvp" | "fd" | "fd1"
    fd_eps: float = 1e-3
    #: adaptive loss weight w = 1 / (||err||^2 + c)^p, stop-gradiented.
    #: p=0 disables it. p=0.75 follows the MeanFlow paper and matters: without
    #: it the large-Delta samples dominate and the small-Delta region -- the
    #: one adjacent to the pretrained boundary -- gets no gradient.
    adaptive_p: float = 0.75
    adaptive_c: float = 1e-3
    #: fraction of samples drawn with r == t. These are free supervision for
    #: the boundary and cost nothing extra, but with the gate in place they
    #: also produce exactly zero gradient, so keep it small and use it as a
    #: monitor rather than as training signal.
    boundary_frac: float = 0.0


def total_derivative(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    cfg: MeanFlowConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (u, du/dt), where du/dt is the total derivative along the flow.

    THE DERIVATIVE IS COMPUTED UNDER no_grad, AND u IS COMPUTED SEPARATELY.

    This is not a micro-optimisation. In meanflow_loss the regression target is
    fully detached, so du/dt never needs a gradient with respect to the adapter
    parameters -- only u does. Taking u out of the JVP call and computing it in
    an ordinary forward avoids backpropagating through the forward-mode
    machinery entirely (reverse-over-forward), which would roughly double
    memory and may not compose with gradient checkpointing at all.

    Cost: one extra forward. Saving: the whole reverse-over-forward graph, and
    u's forward can use the FUSED attention kernels while only the derivative
    pays for the math backend.

    fn must be pure in (z, r, t) -- close over prompt embeddings, mask and
    everything else -- and must be in eval mode with no dropout, or "fd"
    silently differentiates noise.
    """
    dr = torch.zeros_like(r)
    dt = torch.ones_like(t)

    if cfg.estimator == "jvp":
        with torch.no_grad(), math_attention():
            _, du = torch.func.jvp(fn, (z, r, t), (v, dr, dt))
        return fn(z, r, t), du.detach()

    eps = cfg.fd_eps
    if cfg.estimator == "fd":
        with torch.no_grad():
            up = fn(z + eps * v, r, t + eps * dt)
            um = fn(z - eps * v, r, t - eps * dt)
        return fn(z, r, t), ((up - um) / (2 * eps)).detach()

    if cfg.estimator == "fd1":
        with torch.no_grad():
            u0 = fn(z, r, t)
            up = fn(z + eps * v, r, t + eps * dt)
        return fn(z, r, t), ((up - u0) / eps).detach()

    raise ValueError(f"unknown estimator {cfg.estimator!r}")


def meanflow_loss(
    fn: Callable,
    z: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    cfg: MeanFlowConfig,
) -> tuple[torch.Tensor, dict]:
    """MeanFlow regression loss.

    v is the instantaneous velocity at (z, t). Two ways to obtain it:

      empirical  v = eps - x0 from the pair actually used to build z_t. Exact,
                 free, and high variance -- it is a single sample of a
                 conditional expectation.
      teacher    v = v_base(z, t, c) from the frozen pretrained SD3, optionally
                 with CFG folded in. One extra forward, far lower variance,
                 and it distils the guided field so the student samples without
                 CFG at inference.

    Use the teacher. On this compute budget the variance reduction is worth
    more than the forward pass, and CFG-free sampling halves inference NFE --
    which is the whole point of the few-step infrastructure.
    """
    u, du_dt = total_derivative(fn, z, r, t, v, cfg)

    delta = (t - r).reshape(-1, *([1] * (u.dim() - 1)))
    target = (v - delta * du_dt).detach()

    err = u - target
    per_sample = err.pow(2).flatten(1).mean(1)

    if cfg.adaptive_p > 0:
        w = 1.0 / (per_sample.detach() + cfg.adaptive_c) ** cfg.adaptive_p
        w = w / w.mean().clamp(min=1e-12)
    else:
        w = torch.ones_like(per_sample)

    loss = (w * per_sample).mean()

    with torch.no_grad():
        stats = {
            "loss": float(loss),
            "mse": float(per_sample.mean()),
            "delta_mean": float((t - r).mean()),
            "u_norm": float(u.flatten(1).norm(dim=1).mean()),
            "dudt_norm": float(du_dt.flatten(1).norm(dim=1).mean()),
            "target_norm": float(target.flatten(1).norm(dim=1).mean()),
            # STIFFNESS. du/dt divided by the model's own output scale. The
            # MeanFlow target is v - Delta*du/dt, so the correction term
            # overtakes the signal once Delta > ||v|| / ||du/dt||. Measured on
            # SD3-Medium at step 50: dudt_norm ~ 5000 against u_norm ~ 340,
            # giving a usable interval around Delta ~ 0.07 -- not the 1.0 the
            # curriculum default assumed.
            "stiffness": float(du_dt.flatten(1).norm(dim=1).mean()
                               / u.flatten(1).norm(dim=1).mean().clamp(min=1e-8)),
            # Delta at which the correction equals the signal. Track it: it is
            # the Gate B answer, and it is measurable long before convergence.
            "delta_usable": float(
                v.flatten(1).norm(dim=1).mean()
                / du_dt.flatten(1).norm(dim=1).mean().clamp(min=1e-8)),
            # >1 means the target is bigger than what the model emits; the
            # regression is chasing a moving quantity it cannot reach.
            # Delta * S. The most directly readable of the four: 1.0 means the
            # first-order correction equals the field, >1 means it dominates.
            "correction_ratio": float(
                ((t - r) * du_dt.flatten(1).norm(dim=1)).mean()
                / u.flatten(1).norm(dim=1).mean().clamp(min=1e-8)),
            "target_over_u": float(
                target.flatten(1).norm(dim=1).mean()
                / u.flatten(1).norm(dim=1).mean().clamp(min=1e-8)),
        }
    return loss, stats


# --------------------------------------------------------------- curriculum --

def sample_interval(
    batch: int,
    device,
    max_delta: float,
    boundary_frac: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) with 0 <= r <= t <= 1 and t - r <= max_delta.

    max_delta is the curriculum knob and the thing Gate B measures. Starting at
    max_delta ~ 0.05 keeps the model in the neighbourhood where it is already
    correct by construction, then widening it asks how far a pretrained
    instantaneous field can be stretched at LoRA rank. The answer -- the
    largest Delta that still converges -- is the Gate B result whether it is
    large or small. A pass/fail gate at Delta = 1 would throw that away and
    would confound capacity, learning rate and representational mismatch into
    one bit.

    t is drawn logit-normal, matching SD3's own training distribution. Drawing
    t uniformly would over-sample the two endpoints the pretrained model has
    barely seen -- the same under-training that forced Stage 0's
    sigma clamp to [0.05, 0.98].
    """
    n = torch.randn(batch, device=device, generator=generator)
    t = torch.sigmoid(n)                       # logit-normal(0, 1)
    t = t.clamp(0.02, 0.99)

    span = torch.rand(batch, device=device, generator=generator) * max_delta
    r = (t - span).clamp(min=0.0)

    if boundary_frac > 0:
        hit = torch.rand(batch, device=device, generator=generator) < boundary_frac
        r = torch.where(hit, t, r)
    return t, r
