"""
Tier D core: PiGDM-style inference-time guidance for local image editing.

WHAT THIS IS. The paper's term (equations (2)-(4) of the sheet this package was
written from) adds a correction to a base velocity field. Rewritten:

    (3)  A_hat^1 = A^tau + (1 - tau) v        <- one-step "jump to clean" estimate
    (4)  r_tau^2 = (1-tau)^2 / (tau^2 + (1-tau)^2)
    (2)  dv      = min(beta, (1-tau)/(tau r_tau^2)) * (Y - A_hat^1)^T W dA_hat^1/dA^tau

Three readings that decide how it is implemented here:

1.  r_tau^2 IS THE POSTERIOR VARIANCE of the clean data given the current state.
    For A^tau = tau*A^1 + (1-tau)*eps with eps ~ N(0,I) and a unit-Gaussian
    prior on A^1, the precision of the posterior is tau^2/(1-tau)^2 + 1, whose
    inverse is exactly (1-tau)^2/(tau^2 + (1-tau)^2). So (4) is not a fudge
    factor, it is the product-of-Gaussians posterior variance -- hence the
    name. min(beta, .) is a trust region on the gain.

2.  diag(W) IS A PER-COORDINATE WEIGHT. That is where the locality lives: in an
    image editor W is the edit mask. This is the whole reason the scheme is
    interesting for local editing, and it is already in the formula.

3.  THE TERM IS A VJP, NOT A JVP. (Y - A_hat)^T W dA_hat/dA is
    (dA_hat/dA)^T [W (Y - A_hat)] -- a vector-Jacobian product, i.e. one
    REVERSE-mode pass, which torch does natively and cheaply. The paper writes
    it JVP-shaped only because it is a row vector. This matters here
    specifically: this project has forward-mode AD on record as fragile
    (transport/meanflow.py's own warning; forward-mode AD hit signal 11 on four
    separate probes in Stage 0). Reverse mode has no such record.
    `jacobian="vjp"` computes it exactly; `jacobian="identity"` drops the
    Jacobian entirely (dA_hat/dA ~= I), which is the cheap default and is what
    the callback path uses.

SIGN. The correction is applied to the velocity, but the cost is on the clean
estimate, and dx0/dv = -t. Writing the cost as the thing to MINIMISE,

    dv = sign * gain * (dA_hat/dz)^T grad_x0 cost(A_hat)

so `sign=+1` descends the cost (pulls the clean estimate toward the target) and
`sign=-1` ascends it. `self_test` checks both directions rather than trusting
the algebra.

TIME CONVENTION. Everything user-facing in this package is in the REPO's time
`t` (t = 1 noise, t = 0 data, z_t = (1-t) x0 + t eps, v = eps - x0), because
every other file in the repo is. The paper's `tau` (tau = 1 data) appears only
inside the formulas lifted from it, and `tau = 1 - t`. Both are exposed and
`self_test` asserts they agree.

WHAT THIS FILE DOES NOT DO. It does not decide what Y should be. Guidance needs
a target: a reference image, the source image, or a differentiable score. With
none of those there is nothing to steer toward, and the instruction-only local
edit is a *training* problem (see short_term/README.md), not a guidance one.

    python short_term/guidance.py --self-test      # CPU, seconds, no weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


# --------------------------------------------------------------- time maps --

def tau_of_t(t):
    """Repo time (t=1 noise) -> paper time (tau=1 data)."""
    return 1.0 - t


def t_of_tau(tau):
    """Paper time (tau=1 data) -> repo time (t=1 noise)."""
    return 1.0 - tau


# ------------------------------------------------------------------- gain ---

#: Clamp for t in the denominators below. The uncapped gain diverges at both
#: t=0 and t=1, which is exactly what min(beta, .) is for; the clamp only keeps
#: the *uncapped* diagnostic from returning inf.
_T_EPS = 1e-4


def posterior_var_ratio(tau):
    """Equation (4), in paper time: (1-tau)^2 / (tau^2 + (1-tau)^2).

    Verified in `self_test` against the exact linear-Gaussian posterior
    variance, not against the formula restated.
    """
    tau = torch.as_tensor(tau, dtype=torch.float64)
    return (1.0 - tau) ** 2 / (tau ** 2 + (1.0 - tau) ** 2)


def posterior_var_ratio_t(t):
    """Equation (4) in repo time (tau = 1 - t): t^2 / ((1-t)^2 + t^2).

    t = 1 (noise) -> 1: nothing is determined yet.
    t = 0 (data)  -> 0: the state IS the clean sample.
    """
    t = torch.as_tensor(t, dtype=torch.float64)
    return t ** 2 / ((1.0 - t) ** 2 + t ** 2)


def gain_uncapped_t(t):
    """(1-tau)/(tau r_tau^2) in repo time == ((1-t)^2 + t^2) / (t (1-t)).

    NOT a monotone schedule: it is symmetric under t -> 1-t, diverges at both
    ends, and has a MINIMUM OF 2 at t = 0.5. So on (0,1) the gain is never
    below 2, which means a cap of beta <= 2 is active on the ENTIRE trajectory
    and the "schedule" degenerates to a constant. Asserted in `self_test`.
    """
    t = torch.as_tensor(t, dtype=torch.float64).clamp(_T_EPS, 1.0 - _T_EPS)
    return ((1.0 - t) ** 2 + t ** 2) / (t * (1.0 - t))


def picdm_gain(t, beta):
    """Equation (2)'s min(beta, (1-tau)/(tau r_tau^2)), in repo time.

    Practical consequence of the minimum being 2: for beta <= 2 this is
    constant, so the posterior-variance part of (2) only has content for
    beta > 2 and only near the middle of the trajectory. The load-bearing part
    of the scheme is W and A_hat^1, not the gain schedule.
    """
    return torch.minimum(torch.as_tensor(beta, dtype=torch.float64),
                         gain_uncapped_t(t))


# ------------------------------------------------------- clean estimate -----

def one_step_clean(z, t, v):
    """Equation (3), in repo time: x0_hat = z - t*v.

    Exact for the linear path: z = (1-t)x0 + t*eps and v = eps - x0 give
    z - t*v == x0 (asserted in self_test). This is the same quantity the
    project's `ref_meanflow_loss` anchor term already computes.
    """
    return z - _scalar_like(t, z) * v


def _scalar_like(t, like):
    if torch.is_tensor(t):
        return t.reshape(-1, *([1] * (like.dim() - 1))).to(like.dtype)
    return t


def _as_field(x, like):
    """Broadcast a (H,W) / (B,H,W) / (B,1,H,W) / (B,C,H,W) tensor to `like`."""
    if x is None:
        return None
    x = torch.as_tensor(x, dtype=like.dtype, device=like.device)
    while x.dim() < like.dim():
        x = x.unsqueeze(0)
    if x.shape[1] == 1 and like.shape[1] > 1:
        x = x.expand(-1, like.shape[1], *x.shape[2:])
    return x


# --------------------------------------------------------------- the term ---

@dataclass
class GuidanceTerm:
    """One guidance term.

    weight   (B,1,H,W) / (H,W) in [0,1]; 1 = enforce here. THIS IS THE MASK,
             and the convention matches the rest of the project: WHITE IS
             EDITABLE, so `weight=1` inside the region you are editing.
             None = everywhere.
    target   the clean-latent target Y. With `target` set the cost is the
             weighted squared error and grad_x0 cost = weight * (x0_hat - Y).
    score_fn a differentiable cost instead of a target: callable
             (x0_hat) -> scalar tensor, MINIMISED by guidance. `target` is
             exactly the special case score = 0.5 * ||sqrt(W)(x - Y)||^2, which
             `self_test` asserts -- that is what makes the score branch
             verified rather than aspirational.
    beta     the gain cap in equation (2).
    sign     +1 pulls the clean estimate toward lower cost, -1 pushes away.
    jacobian "identity" (cheap default) or "vjp" (exact, needs a differentiable
             velocity function).
    kappa_max only used by `projected_x0`: cap on the direct blend weight, so a
             guidance term can never overshoot past the target.
    """

    beta: float = 1.0
    sign: float = 1.0
    weight: torch.Tensor | None = None
    target: torch.Tensor | None = None
    score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    jacobian: str = "identity"
    kappa_max: float = 1.0
    name: str = "term"

    def __post_init__(self) -> None:
        if self.target is None and self.score_fn is None:
            raise ValueError(
                f"GuidanceTerm({self.name!r}): needs a `target` tensor or a "
                "`score_fn`. Guidance steers toward something; with neither "
                "there is nothing to steer toward.")
        if self.jacobian not in ("identity", "vjp"):
            raise ValueError(
                f"GuidanceTerm({self.name!r}): jacobian must be 'identity' or "
                f"'vjp', got {self.jacobian!r}")


def score_grad(x0_hat: torch.Tensor, term: GuidanceTerm) -> torch.Tensor:
    """grad_{x0} cost(x0_hat), same shape as x0_hat.

    Target form: weight * (x0_hat - Y), so guidance pulls x0_hat toward Y
    wherever the weight is nonzero. Because the correction is applied to the
    velocity and dx0/dv = -t, the resulting motion is *down* this gradient.
    """
    w = _as_field(term.weight, x0_hat)
    if term.score_fn is None:
        y = _as_field(term.target, x0_hat)
        g = x0_hat - y
        return g if w is None else w * g
    if not x0_hat.requires_grad:
        raise ValueError(
            f"GuidanceTerm({term.name!r}): score_fn needs x0_hat to carry grad. "
            "Call guidance with the z that the score is taken through, or use a "
            "target instead.")
    (g,) = torch.autograd.grad(term.score_fn(x0_hat), x0_hat,
                               create_graph=False, retain_graph=False)
    return g if w is None else w * g


def guidance_correction(z, t, v, term: GuidanceTerm,
                        velocity_fn: Callable | None = None):
    """The velocity correction dv of equation (2), repo time.

        dv = sign * gain * (dA_hat/dz)^T grad_x0 cost(A_hat)

    `velocity_fn(z, t) -> v` is only needed when `term.jacobian == "vjp"`; it is
    re-called with grad enabled so the VJP can be taken through the one-step
    map. Cost is one extra forward + one backward per step, no forward-mode AD.
    """
    gain = picdm_gain(t, term.beta).to(z.dtype).to(z.device)
    if term.jacobian == "identity":
        x0 = one_step_clean(z, t, v)
        return term.sign * gain * score_grad(x0, term)

    if velocity_fn is None:
        raise ValueError(
            f"GuidanceTerm({term.name!r}): jacobian='vjp' needs velocity_fn, "
            "so the one-step map can be differentiated through z. Without it "
            "use jacobian='identity' -- that is the callback path's only mode.")

    with torch.enable_grad():
        z_g = z.detach().requires_grad_(True)
        v_g = velocity_fn(z_g, t)
        x0 = one_step_clean(z_g, t, v_g)
        g = score_grad(x0, term)
        jac_t_g = torch.autograd.grad(x0, z_g, grad_outputs=g,
                                      retain_graph=False)[0]
    return term.sign * gain * jac_t_g.detach()


def guided_velocity(z, t, v, terms, velocity_fn: Callable | None = None):
    """v plus every term's correction. `terms` may be empty (returns v)."""
    out = v
    for term in terms:
        out = out + guidance_correction(z, t, v, term, velocity_fn=velocity_fn)
    return out


# ------------------------------------------------- the direct-blend form ----

def equivalent_kappa(sigma, sigma_next, t, beta):
    """The direct blend weight that a velocity correction of gain `beta`
    produces in ONE step.

    The sampler steps z_next = z + (sigma_next - sigma) v, so a velocity
    correction dv shifts z_next by (sigma_next - sigma) dv. The direct form
    (see `projected_x0`) shifts z_next by (1 - sigma_next) * kappa * W (Y - x0).
    Setting those equal:

        kappa_eq = (sigma - sigma_next) * gain(t, beta) / (1 - sigma_next)

    This is exact, and `self_test` asserts the two forms agree when kappa is
    set this way. It is also the number worth looking at before choosing beta:
    on a 50-step schedule (sigma - sigma_next ~ 0.02, 1 - sigma_next ~ 0.5)
    kappa_eq ~ 0.04 * gain, so beta = 10 buys about a 40% blend *per step*,
    applied at every step -- an integral controller, not a single set.
    """
    num = (sigma - sigma_next) * float(picdm_gain(t, beta))
    return num / max(1.0 - float(sigma_next), 1e-6)


def clean_blend(t, beta):
    """How far ONE step's guidance moves the clean estimate toward the target.

        x0_new = z - t(v + dv) = x0 - t*gain*W*(x0 - Y)
        =>  x0_new - Y  =  (x0 - Y) * (1 - t*gain)      (for W = 1)

    so the effective blend is `kappa = t * gain(t, beta)`, and **kappa > 1
    OVERSHOOTS**: the error flips sign and GROWS by |1 - kappa|. Since t -> 1
    at the noise end, a beta above 1 already overshoots at the first step, and
    beta = 30 asks for a 30x overshoot there.

    This is why the direct form (`projected_x0`) is the safer parameterisation:
    its blend is an explicit kappa clipped to `kappa_max`, so it cannot
    overshoot by construction, while the velocity form's blend is beta*t and is
    whatever the cap lets through.
    """
    return (_scalar_float(t) * float(picdm_gain(t, beta)))


def _scalar_float(x):
    return float(torch.as_tensor(x, dtype=torch.float64))


def projected_x0(x0_hat, term: GuidanceTerm, kappa):
    """Direct blend of the clean estimate toward the target, clipped so it
    cannot overshoot:

        x0 <- x0 + clip(kappa, 0, kappa_max) * W * (Y - x0)

    `kappa = 1` with `W = 1` sets the clean estimate exactly to Y inside the
    mask. This is the same idiom as region/alpha_field.py's `blend_latents`
    (z <- w z_inv + (1-w) z_gen), which is why the CLI offers it as the default
    in the callback path: stable, interpretable, and already the project's
    house style for masked blending.
    """
    if term.target is None:
        raise ValueError(
            f"GuidanceTerm({term.name!r}): projected_x0 needs a `target` "
            "tensor; a bare score_fn has no Y to blend toward.")
    w = _as_field(term.weight, x0_hat)
    y = _as_field(term.target, x0_hat)
    k = min(max(float(kappa), 0.0), float(term.kappa_max))
    delta = y - x0_hat
    if w is not None:
        delta = w * delta
    return x0_hat + k * delta


# -------------------------------------------------------- the safe setter ---

def renoise(z_sigma, sigma, v, sigma_next, terms, velocity_fn=None,
            direct_kappa: Callable | None = None):
    """One guided step, written as "correct the clean estimate, re-noise".

    WHY THIS SHAPE. It is exact, it needs only the two sigmas, and it avoids
    having to know anything about someone else's scheduler internals beyond
    `sigmas`. Two ways to use it:

        direct_kappa=None        velocity form, equation (2)
        direct_kappa=fn(t)       direct form with kappa = fn(t) per term

    Derivation for the velocity form: recover the noise, correct x0, put it
    back. z_sigma = (1-sigma) x0 + sigma eps gives eps = (z_sigma - (1-sigma)x0)
    / sigma, and the new z at sigma_next is (1-sigma_next) x0_new + sigma_next
    eps. Substituting the velocity correction reproduces the scheduler's own
    (sigma_next - sigma) dv exactly -- asserted in `self_test`.
    """
    t = float(sigma)
    if direct_kappa is None:
        v_g = guided_velocity(z_sigma, t, v, terms, velocity_fn=velocity_fn)
        return z_sigma + (float(sigma_next) - float(sigma)) * v_g

    x0 = one_step_clean(z_sigma, t, v)
    # Recover the noise from the UNCORRECTED clean estimate. Doing this after
    # the correction (the first version did) silently re-noises with a
    # contaminated eps that no longer satisfies z = (1-s)x0 + s*eps, so the
    # step is not the one the derivation describes -- it is off by
    # (1-sigma_next) * d(x0)/sigma, which at sigma=0.8 is ~1.2x the intended
    # correction. Caught by the direct-vs-velocity equivalence check.
    eps = (z_sigma - (1.0 - t) * x0) / max(t, 1e-6)
    x0_corr = x0
    for term in terms:
        x0_corr = projected_x0(x0_corr, term, direct_kappa(t, term))
    return (1.0 - float(sigma_next)) * x0_corr + float(sigma_next) * eps


# ------------------------------------------------------------- self-test ----

def self_test() -> None:
    """CPU, no weights, seconds. Run before trusting any GPU run."""
    torch.manual_seed(0)
    dt = torch.float64

    # --- (4) is the exact posterior variance, not a restated formula --------
    # For a = tau*x0 + (1-tau)*eps with x0, eps ~ N(0,1) independent, measure
    # Var(x0 | a) DIRECTLY by binning on a, so the check does not assume the
    # linear-Gaussian identity it is testing. (A plug-in 1 - cov^2/var estimate
    # is unusable here: its error is amplified by d/dcov = -2cov/var ~ -2.2, so
    # 4e5 samples leave 3e-3 of noise on a quantity whose answer is 0.012.)
    n = 2000000
    x0 = torch.randn(n, dtype=dt)
    eps = torch.randn(n, dtype=dt)
    n_bin = 400
    per = n // n_bin
    for tau in (0.15, 0.4, 0.6, 0.9):
        tau = torch.tensor(tau, dtype=dt)
        a = tau * x0 + (1 - tau) * eps
        order = torch.argsort(a)
        xb = x0[order].reshape(n_bin, per)
        within = xb.var(dim=1, unbiased=True).mean()   # E[Var(x0 | a)] + O(bin^2)
        formula = float(posterior_var_ratio(tau))
        assert abs(float(within) - formula) < 1e-3, (tau, float(within), formula)
        # and the repo-time version is the same number
        assert abs(float(posterior_var_ratio_t(1 - tau)) - formula) < 1e-12

    # --- the uncapped gain has a MINIMUM of 2 at t = 0.5 -------------------
    grid = torch.linspace(0.01, 0.99, 99, dtype=dt)
    g = gain_uncapped_t(grid)
    assert float(g.min()) > 2.0 - 1e-9, float(g.min())
    assert abs(float(gain_uncapped_t(0.5)) - 2.0) < 1e-12
    assert float(gain_uncapped_t(torch.tensor(0.25, dtype=dt))) == \
        float(gain_uncapped_t(torch.tensor(0.75, dtype=dt)))
    # beta <= 2 therefore clips EVERYWHERE: the schedule has no content there
    for t in (0.05, 0.3, 0.5, 0.7, 0.95):
        assert abs(float(picdm_gain(t, 1.5)) - 1.5) < 1e-12
    # ... and beta > 2 only bites near the middle
    assert float(picdm_gain(0.05, 5.0)) == 5.0
    assert float(picdm_gain(0.5, 5.0)) == 2.0

    # --- (3) recovers x0 exactly on the linear path ------------------------
    B, C, H, W = 2, 3, 4, 4
    x0 = torch.randn(B, C, H, W, dtype=dt)
    eps = torch.randn(B, C, H, W, dtype=dt)
    for t in (0.05, 0.3, 0.5, 0.8, 0.97):
        z = (1 - t) * x0 + t * eps
        v = eps - x0
        assert torch.allclose(one_step_clean(z, t, v), x0, atol=1e-12), t

    # --- target form: the correction moves x0_hat TOWARD Y -----------------
    mask = torch.zeros(B, 1, H, W, dtype=dt)
    mask[:, :, :, :2] = 1.0
    y = torch.randn(B, C, H, W, dtype=dt)
    z = torch.randn(B, C, H, W, dtype=dt)
    v = torch.randn(B, C, H, W, dtype=dt)
    t = 0.4
    base = one_step_clean(z, t, v)

    pull = GuidanceTerm(beta=3.0, sign=+1.0, weight=mask, target=y, name="pull")
    push = GuidanceTerm(beta=3.0, sign=-1.0, weight=mask, target=y, name="push")

    def dist(x):
        return float((mask * (x - y) ** 2).sum())

    for sign, term, tag in ((+1, pull, "toward"), (-1, push, "away")):
        v_g = guided_velocity(z, t, v, [term])
        moved = one_step_clean(z, t, v_g)
        if sign > 0:
            assert dist(moved) < dist(base), f"{tag} did not reduce the cost"
        else:
            assert dist(moved) > dist(base), f"{tag} did not increase the cost"

    # zero weight -> no change at all; target already reached -> no change
    inert = GuidanceTerm(beta=3.0, weight=torch.zeros_like(mask), target=y)
    assert torch.allclose(guided_velocity(z, t, v, [inert]), v, atol=1e-14)
    arrived = GuidanceTerm(beta=3.0, weight=mask, target=base)
    assert torch.allclose(guided_velocity(z, t, v, [arrived]), v, atol=1e-12)
    # beta = 0 -> no change
    off = GuidanceTerm(beta=0.0, weight=mask, target=y)
    assert torch.allclose(guided_velocity(z, t, v, [off]), v, atol=1e-14)

    # --- score_fn IS the target form, not a parallel implementation --------
    yy = y
    mm = mask

    def score(x):
        return 0.5 * (mm * (x - yy) ** 2).sum()

    via_score = GuidanceTerm(beta=3.0, weight=mask, score_fn=score, name="score")
    z_g = z.clone().requires_grad_(True)
    v_g = v.clone()
    from_score = guidance_correction(z_g, t, v_g, via_score)
    from_target = guidance_correction(z, t, v, pull)
    assert torch.allclose(from_score, from_target, atol=1e-10), \
        (from_score - from_target).abs().max()

    # --- VJP form: equals identity for a velocity that ignores z, and
    #     differs once the velocity depends on z (so the test is not vacuous)
    def const_vel(z_, t_):
        return v

    a = guidance_correction(z, t, v, GuidanceTerm(
        beta=3.0, weight=mask, target=y, jacobian="vjp"), velocity_fn=const_vel)
    assert torch.allclose(a, from_target, atol=1e-9)

    # a velocity that DOES depend on z: v(z) = eps - x0(z) with x0(z) = z/2
    # (an analytic one-step map, so the exact VJP is known in closed form)
    eps_c = torch.randn(B, C, H, W, dtype=dt)

    def half_vel(z_, t_):
        return eps_c - 0.5 * z_

    # x0_hat = z - t(eps_c - z/2) = z(1 + t/2) - t eps_c
    # dA_hat/dz = (1 + t/2) I  -> VJP = (1 + t/2) g
    x0_h = one_step_clean(z, t, half_vel(z, t))
    g = score_grad(x0_h, pull)
    exact = (1 + t / 2) * g * pull.sign * picdm_gain(t, pull.beta).to(dt)
    got = guidance_correction(z, t, half_vel(z, t),
                              GuidanceTerm(beta=3.0, weight=mask, target=y,
                                           sign=+1.0, jacobian="vjp"),
                              velocity_fn=half_vel)
    assert torch.allclose(got, exact, atol=1e-9), (got - exact).abs().max()
    assert not torch.allclose(got, from_target, atol=1e-6), \
        "VJP form collapsed onto the identity form -- test is vacuous"

    # --- the direct form with kappa_eq reproduces the velocity form --------
    sigma, sigma_next = 0.8, 0.76
    kappa = equivalent_kappa(sigma, sigma_next, sigma, pull.beta)
    by_velocity = renoise(z, sigma, v, sigma_next, [pull])
    by_direct = renoise(z, sigma, v, sigma_next, [pull],
                        direct_kappa=lambda t_, term_: kappa)
    assert torch.allclose(by_velocity, by_direct, atol=1e-9), \
        (by_velocity - by_direct).abs().max()

    # kappa = 1 with W = 1 SETS the clean estimate to Y inside the mask
    full = GuidanceTerm(beta=1.0, weight=torch.ones_like(mask), target=y)
    set_to_y = projected_x0(base, full, kappa=1.0)
    assert torch.allclose(set_to_y, y, atol=1e-12)
    # and kappa is clipped, so it can never overshoot past the target
    over = projected_x0(base, full, kappa=5.0)
    assert torch.allclose(over, set_to_y, atol=1e-12)

    # --- the effective per-step blend is t*gain, and >1 overshoots ---------
    # This is not a curiosity: it is why beta > 1 is not a free knob at the
    # noise end, and why the direct form (explicit clipped kappa) is safer.
    assert abs(clean_blend(1.0, 2.0) - 2.0) < 1e-9      # beta*t = 2: exactly
    assert abs(clean_blend(0.5, 2.0) - 1.0) < 1e-9      # a full blend, no more
    assert abs(clean_blend(0.25, 2.0) - 0.5) < 1e-9
    assert clean_blend(1.0, 30.0) > 1.0                 # 30x overshoot at t=1
    # and the identity the direct form relies on: kappa_eq * (1 - sigma_next)
    # equals the state-space displacement a velocity correction produces
    sig, sig_n = 0.8, 0.76
    k = equivalent_kappa(sig, sig_n, sig, 3.0)
    assert abs(k - (sig - sig_n) * float(picdm_gain(sig, 3.0))
               / (1.0 - sig_n)) < 1e-12

    # --- a tiny end-to-end: guidance on a 1-D two-mode field picks the mode -
    class TwoMode:
        """Exact marginal field of x0 = +/-1 with equal weight. Deterministic
        endpoint given the drawn z_init, which is the point: the distribution
        lives in z_init (the "noise" side), not in a regression head."""

        def __call__(self, z_, t_):
            t_c = max(float(t_), 1e-3)
            lp = []
            for m in (1.0, -1.0):
                d = z_ - (1 - t_c) * m
                lp.append(-(d ** 2).sum() / (2 * t_c ** 2))
            post = torch.softmax(torch.stack(lp), dim=0)
            mean = post[0] * 1.0 + post[1] * (-1.0)
            return (z_ - mean) / t_c

    field = TwoMode()
    z_pos = torch.full((1, 1, 2, 2), 0.9, dtype=dt)
    z_neg = torch.full((1, 1, 2, 2), -0.9, dtype=dt)
    # unguided: each draw keeps its own sign -> a distribution, not a mean
    for z_init, want in ((z_pos, +1.0), (z_neg, -1.0)):
        z = z_init.clone()
        for s in torch.linspace(1.0, 0.0, 21, dtype=dt)[:-1]:
            z = z - (1.0 / 20) * field(z, s)
        assert (z.mean() - want).abs() < 0.2, (float(z.mean()), want)
    # guided toward +1 with a HALF-region mask: the mode flips, and only the
    # masked half carries the evidence
    half = torch.zeros(1, 1, 2, 2, dtype=dt)
    half[:, :, :, :1] = 1.0
    tgt = torch.full_like(z_neg, 1.0)
    term = GuidanceTerm(beta=1.0, weight=half, target=tgt, name="mode-pick")
    z = z_neg.clone()
    for s in torch.linspace(1.0, 0.0, 21, dtype=dt)[:-1]:
        s = float(s)
        v = field(z, s)
        nxt = max(s - 1.0 / 20, 0.0)
        z = renoise(z, s, v, nxt, [term],
                    direct_kappa=lambda t_, _term: min(1.0, 3.0 * (1.0 - t_)))
    assert z.mean() > 0.5, float(z.mean())

    print("guidance self-test OK")
    print("  (4) checked against the exact posterior variance, not restated")
    print("  uncapped gain: minimum 2 at t=0.5 -> beta<=2 is active everywhere")
    print("  (2) is a VJP; identity and vjp forms both checked, vjp against a")
    print("      closed-form Jacobian; score_fn == target form to 1e-10")
    print("  direct form with kappa_eq reproduces the velocity form to 1e-9")
    print("  toy two-mode field: guidance picks the mode from a half mask")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
