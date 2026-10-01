"""
A flow-matching sampler that owns its own loop, so guidance can be injected.

WHY A SECOND SAMPLER EXISTS. The real editor (an SD3 InstructPix2Pix
checkpoint) brings its own denoising loop, and `edit.py` injects guidance into
*that* loop through diffusers' `callback_on_step_end` -- the least invasive
option and the one most likely to survive an unfamiliar fork's internals. This
loop is for everything else: the CPU toy in `demo_toy.py`, any base field that
is just a callable, and the case where you want the VJP form of the guidance
(which needs the velocity to be a differentiable function of the state, and so
needs a loop that calls the model itself).

The two paths are checked against each other on CPU: `guidance.self_test` has
the equivalence between the velocity and direct forms, and `demo_toy.py` runs
this loop end to end.

TIME. Repo convention throughout: `sigma == t`, t=1 is noise, t=0 is data,
z_t = (1-t) x0 + t eps, v = eps - x0. A step is
`z_next = z + (sigma_next - sigma) * v`, which at sigma_next = 0 lands on x0.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Callable, Sequence

import torch

from .guidance import GuidanceTerm, guided_velocity, one_step_clean, renoise


def sigma_grid(steps: int, sigma_max: float = 1.0, sigma_min: float = 0.0,
               shift: float | None = None) -> torch.Tensor:
    """`steps + 1` sigmas, sigma_max -> sigma_min, evenly in the shifted time.

    `shift` applies the flow-match time shift used by SD3,
    `s' = shift*s / (1 + (shift-1)*s)`, which moves resolution toward the noise
    end. None (default) is a uniform grid. Returned highest-sigma first, so
    consecutive entries are (sigma, sigma_next) in sampler order.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    s = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    if shift is not None and shift > 0:
        s = shift * s / (1.0 + (shift - 1.0) * s)
    sig = sigma_max + (sigma_min - sigma_max) * s
    return sig


@dataclass
class Trace:
    """Per-step record. Enough to see *when* guidance acted, which is the
    question a failed run always turns out to be about."""

    sigma: list = dc_field(default_factory=list)
    sigma_next: list = dc_field(default_factory=list)
    move: list = dc_field(default_factory=list)
    guide: list = dc_field(default_factory=list)


def sample(field: Callable[[torch.Tensor, float], torch.Tensor],
           z_init: torch.Tensor,
           sigmas: Sequence[float],
           terms: Sequence[GuidanceTerm] = (),
           *,
           direct_kappa: Callable[[float, GuidanceTerm], float] | None = None,
           pin: tuple[torch.Tensor, torch.Tensor] | None = None,
           differentiable_field: bool = False,
           trace: bool = False) -> tuple[torch.Tensor, Trace | None]:
    """Integrate `field` from `z_init` down the sigma grid, applying guidance.

    field(z, t) -> velocity, plain tensors, repo time.
    terms        guidance terms; empty is a plain sampler.
    direct_kappa (t, term) -> blend weight for the "correct x0, re-noise" form.
                 None uses the paper's velocity form (equation (2)).
    pin          (z_pin, mask): after every step, hard-copy z_pin outside the
                 mask, `z <- mask*z + (1-mask)*z_pin`. This is the same
                 background pin Stage 0 used and the reason its localisation
                 numbers were good; it is orthogonal to guidance.
    differentiable_field
                 re-call field with grad enabled so `jacobian="vjp"` terms work.
                 Costs one forward + one backward per term per step.
    """
    z = z_init
    tr = Trace() if trace else None
    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        if trace:
            tr.sigma.append(sigma)
            tr.sigma_next.append(sigma_next)

        if direct_kappa is None:
            v = field(z, sigma)
            v_guided = guided_velocity(
                z, sigma, v, terms,
                velocity_fn=(field if differentiable_field else None))
            z_new = z + (sigma_next - sigma) * v_guided
            if trace:
                tr.move.append(float((z_new - z).abs().mean()))
                tr.guide.append(float((v_guided - v).abs().mean()))
        else:
            v = field(z, sigma)
            z_new = renoise(z, sigma, v, sigma_next, terms,
                            velocity_fn=(field if differentiable_field
                                         else None),
                            direct_kappa=direct_kappa)
            if trace:
                tr.move.append(float((z_new - z).abs().mean()))
                tr.guide.append(float((z_new - (z + (sigma_next - sigma) * v))
                                      .abs().mean()))

        if pin is not None:
            z_pin, mask = pin
            z_new = mask * z_new + (1.0 - mask) * z_pin
        z = z_new
    return z, tr


def linear_kappa(rate: float, t_lo: float = 0.1, t_hi: float = 1.0
                 ) -> Callable[[float, GuidanceTerm], float]:
    """A usable default schedule for the direct form: no guidance once the
    state is nearly clean (t < t_lo), ramping to `rate` over [t_lo, t_hi].

    WHY IT MUST VANISH NEAR t=0. The direct form recovers eps as
    (z - (1-t)x0)/t, so as t -> 0 the recovered noise and the correction it
    carries are divided by a vanishing number. Guidance near the data end is
    both unnecessary (there is nothing left to steer) and ill-conditioned.

    THE RAMP IS A CONFOUND, WHICH IS WHY flat_kappa EXISTS. This ramps
    `rate * (t - t_lo)/(t_hi - t_lo)`, so a nominal kappa of 1.0 is really 0.41
    at t=0.5 and 0.18 at t=0.3 -- i.e. most of the trajectory runs at a
    fraction of the requested strength. A sweep that plateaus under THIS
    schedule has not shown that the mechanism has a floor; it has shown that
    the ramp has one. Use `flat_kappa` to tell those apart.
    """
    def fn(t: float, term: GuidanceTerm) -> float:
        if t <= t_lo:
            return 0.0
        ramp = min(1.0, (t - t_lo) / max(t_hi - t_lo, 1e-6))
        return rate * ramp
    return fn


def flat_kappa(rate: float, t_lo: float = 0.1
               ) -> Callable[[float, GuidanceTerm], float]:
    """Constant `rate` for t > t_lo, 0 below. The stronger form of the same
    test: if a plateau survives this, it is the mechanism's, not the ramp's."""
    def fn(t: float, term: GuidanceTerm) -> float:
        return 0.0 if t <= t_lo else rate
    return fn


def kappa_schedule(mode: str, rate: float, t_lo: float = 0.15,
                   t_hi: float = 1.0):
    """`mode` is 'ramp' (linear_kappa) or 'flat' (flat_kappa)."""
    if mode == "flat":
        return flat_kappa(rate, t_lo=t_lo)
    if mode == "ramp":
        return linear_kappa(rate, t_lo=t_lo, t_hi=t_hi)
    raise ValueError(f"kappa mode must be 'ramp' or 'flat', got {mode!r}")


def one_step_clean_trace(field, z_init, sigmas, **kw) -> torch.Tensor:
    """Convenience: the final clean estimate rather than the state. At
    sigmas[-1] == 0 these are the same thing; this exists so callers do not
    have to know that."""
    z, _ = sample(field, z_init, sigmas, **kw)
    return one_step_clean(z, float(sigmas[-1]), field(z, float(sigmas[-1])))
