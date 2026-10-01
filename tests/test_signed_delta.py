"""H3 V0 -- signed Delta. CPU, seconds. Run from the repo root:

    PYTORCH_JIT=0 python -m pytest tests/test_signed_delta.py -q

What this pins down (H3_SIGNED_DELTA_DESIGN.md Sec.3/Sec.6 V0):
  * r <= t: signed and unsigned embeddings are torch.equal (not allclose),
    for both interval kinds, with NON-zero weights (a zero-initialised last
    layer would make every comparison trivially equal).
  * r >  t: unsigned reads EXACTLY as r == t (the recorded "before");
    signed does not.
  * signedness adds no state_dict keys (eval refuses missing interval.* keys).
  * the MeanFlow identity holds with signed Delta on a closed-form field, and
    the pre-H3 clamp demonstrably breaks it on reverse rows.
  * reverse_frac = 0 consumes no RNG and changes no draw; reverse_frac > 0 is
    an exact endpoint swap of the forward arm's rows, anchors paired.
  * meanflow_ref.sample_tr and delta_curriculum.sample_tr_scheduled agree row
    for row with reverse rows on (the inline restatement cannot drift).
  * true_average_velocity: finite in reverse, v at Delta == 0, and one
    trajectory's average agrees from either end (round trip).

Extends, never edits, tests/test_meanflow_target.py (design Sec.3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transport.conditioning import (  # noqa: E402
    CenteredIntervalEmbedding, IntervalEmbedding, interval_delta,
    make_interval_embedding)
from transport.meanflow_ref import (  # noqa: E402
    RefMeanFlowConfig, orient_reverse, ref_meanflow_loss, sample_tr)

KINDS = [IntervalEmbedding, CenteredIntervalEmbedding]


def _randomise(m: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def _pair(kind, seed=0):
    a, b = kind(pooled_dim=64, signed=False), kind(pooled_dim=64, signed=True)
    _randomise(a, seed)
    b.load_state_dict(a.state_dict())
    return a, b


def _tr(n=256, seed=0, anchor_frac=0.25):
    g = torch.Generator().manual_seed(seed)
    a, b = torch.rand(n, generator=g), torch.rand(n, generator=g)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    anc = torch.rand(n, generator=g) < anchor_frac
    return t, torch.where(anc, t, r)


# ------------------------------------------------------------ embeddings --

def test_interval_delta_forward_identity():
    t, r = _tr()
    assert torch.equal(interval_delta(t, r, False), (t - r).clamp(min=0.0))
    assert torch.equal(interval_delta(t, r, True), interval_delta(t, r, False))


@pytest.mark.parametrize("kind", KINDS)
def test_forward_bit_identical(kind):
    uns, sgn = _pair(kind)
    t, r = _tr()
    out = uns(t, r)
    assert out.abs().max() > 0, "weights not randomised -- test is vacuous"
    assert torch.equal(out, sgn(t, r))


@pytest.mark.parametrize("kind", KINDS)
def test_reverse_before_and_after(kind):
    uns, sgn = _pair(kind)
    t, r = _tr(anchor_frac=0.0)
    # r > t, strictly
    tr_, rr_ = r, t
    assert bool((rr_ > tr_).all())
    # BEFORE (today): r > t reads exactly as r == t
    assert torch.equal(uns(tr_, rr_), uns(tr_, tr_.clone()))
    # AFTER: it does not
    assert not torch.allclose(sgn(tr_, rr_), sgn(tr_, tr_.clone()))


def test_centered_boundary_two_sided():
    # The bracket must vanish at r == t from BOTH sides -- that is what keeps
    # u(z,t,t) = v a two-sided boundary. It is NOT bit-exact and cannot be:
    # delta_part evaluates f and f0 in ONE batched matmul, whose accumulation
    # order differs from a per-half call. MEASURED here:
    #     batched   max|f - f0| = 8.345e-07
    #     separate  max|f - f0| = 0.0
    # and signed=False gives the SAME 8.345e-07, so this is a float property of
    # the formulation, not of H3. A per-half call would be exact and cost 2x the
    # branch; not worth it for an ~1e-6 offset on a pooled-projection vector.
    # Asserted with that tolerance, PLUS a non-vacuity check -- without the
    # second half a dead delta_branch would pass.
    _, sgn = _pair(CenteredIntervalEmbedding)
    t, _ = _tr()
    with torch.no_grad():
        at_boundary = sgn.delta_part(t, t.clone())
        off = sgn.delta_part(t, t - 0.25)
    b = float(at_boundary.abs().max())
    o = float(off.abs().max())
    assert b < 1e-6, f"boundary bracket must be ~0, got {b:.3e}"
    assert o > 100 * b, (
        f"the bracket must be MUCH larger off the diagonal or this test is "
        f"vacuous: off {o:.3e} vs at-boundary {b:.3e}")


@pytest.mark.parametrize("kind", ["concat", "centered"])
def test_signed_is_not_state(kind):
    a = make_interval_embedding(kind, 64, signed=False)
    b = make_interval_embedding(kind, 64, signed=True)
    assert b.signed and not a.signed
    assert list(a.state_dict()) == list(b.state_dict())


# ------------------------------------------------ the identity, signed --

def test_meanflow_identity_signed():
    """v(s) = c*s (z-free) => u(t, r) = c*(t + r)/2 in BOTH directions, and
    u = v(t) - (t - r) * c/2 is the MeanFlow identity with signed Delta."""
    torch.manual_seed(0)
    n, c = 64, 3.0
    t, r = _tr(n, seed=1, anchor_frac=0.25)
    flip = torch.rand(n) < 0.5
    t, r = torch.where(flip, r, t), torch.where(flip, t, r)
    assert bool((r > t).any())
    z = torch.randn(n, 4, 2, 2)

    def u_fn(z_, t_, r_):
        return (c * (t_ + r_) / 2).reshape(-1, 1, 1, 1).expand_as(z_) + 0 * z_

    v_g = (c * t).reshape(-1, 1, 1, 1).expand_as(z).contiguous()
    cfg = RefMeanFlowConfig(estimator="fd", fd_eps=0.025, norm_p=0.0)
    _, st = ref_meanflow_loss(u_fn, z, t, r, v_g, cfg)
    assert st["sse"] < 1e-8, st["sse"]
    assert st["frac_reverse"] > 0 and "sse_reverse" in st
    # anchors inferred with abs(): reverse rows must NOT count as anchors
    assert abs(st["frac_anchor"] - float(((t - r).abs() < 1e-8).float().mean())) < 1e-9

    # the pre-H3 clamp on the target's Delta breaks it on reverse rows only
    d_old = (t - r).clamp(0.0, 1.0)
    tgt_old = c * t - d_old * (c / 2)
    err_old = (c * (t + r) / 2 - tgt_old).abs()
    assert float(err_old[r > t].min()) > 0
    assert float(err_old[r <= t].max()) < 1e-6


# --------------------------------------------------------------- sampler --

def test_reverse_frac_zero_draws_nothing():
    cfg = RefMeanFlowConfig(max_delta=0.5)
    g = torch.Generator().manual_seed(5)
    t, r, a = sample_tr(512, cfg, "cpu", generator=g)
    state_after = g.get_state()
    g2 = torch.Generator().manual_seed(5)
    t2, r2, a2 = sample_tr(512, cfg, "cpu", generator=g2, rev_generator=None)
    assert torch.equal(t, t2) and torch.equal(r, r2) and torch.equal(a, a2)
    assert torch.equal(state_after, g2.get_state())
    assert bool((t >= r).all())
    tt, rr, rev = orient_reverse(t, r, 0.0, None)
    assert tt is t and rr is r and not bool(rev.any())


def test_reverse_rows_are_swaps_of_forward_twins():
    fwd = RefMeanFlowConfig(max_delta=0.5)
    rev = RefMeanFlowConfig(max_delta=0.5, reverse_frac=0.5)
    tF, rF, aF = sample_tr(4096, fwd, "cpu", torch.Generator().manual_seed(9))
    tR, rR, aR = sample_tr(4096, rev, "cpu", torch.Generator().manual_seed(9),
                           rev_generator=torch.Generator().manual_seed(10))
    assert torch.equal(aF, aR)
    same = (tR == tF) & (rR == rF)
    swap = (tR == rF) & (rR == tF)
    assert bool((same | swap).all())
    assert torch.equal((tR - rR).abs(), (tF - rF).abs())
    frac = float(((rR - tR) > 1e-8).float().mean())
    assert abs(frac - 0.5 * 0.25) < 0.02, frac
    with pytest.raises(ValueError):
        sample_tr(8, rev, "cpu", torch.Generator().manual_seed(1))


def test_curriculum_sampler_matches_reference_with_reverse():
    from delta_curriculum import DeltaCurriculumConfig, sample_tr_scheduled
    ref = RefMeanFlowConfig(max_delta=0.5, reverse_frac=0.5)
    off = DeltaCurriculumConfig(enabled=False, hard_max=0.5)
    t1, r1, a1 = sample_tr(2048, ref, "cpu", torch.Generator().manual_seed(3),
                           rev_generator=torch.Generator().manual_seed(4))
    t2, r2, a2 = sample_tr_scheduled(
        2048, ref, "cpu", 0, 1000, off, torch.Generator().manual_seed(3),
        rev_generator=torch.Generator().manual_seed(4))
    assert torch.equal(t1, t2) and torch.equal(r1, r2) and torch.equal(a1, a2)


# ---------------------------------------------------------- ground truth --

def _toy_v(z, s):
    s = s.reshape(-1, *([1] * (z.dim() - 1)))
    return torch.sin(3 * z) * (1 + s) - 2 * s * z + torch.cos(5 * s)


def test_true_average_velocity_signed():
    eif = pytest.importorskip("eval_interval_field")
    tav = eif.true_average_velocity
    z = torch.linspace(-2, 2, 7, dtype=torch.float64).reshape(1, 7, 1, 1)
    one = lambda x: torch.full((1,), x, dtype=torch.float64)  # noqa: E731

    # Delta == 0 returns v, not 0
    assert torch.allclose(tav(_toy_v, z, one(.5), one(.5), 16), _toy_v(z, one(.5)))
    # reverse is finite and the same order of magnitude as forward
    ur = tav(_toy_v, z, one(.4), one(.65), 16)
    assert float(ur.abs().max()) < 10
    # round trip: one trajectory, both ends, error falls with substeps
    errs = []
    for sub in (8, 16, 32):
        uf, zlo = tav(_toy_v, z, one(.75), one(.25), sub, return_end=True)
        ub = tav(_toy_v, zlo, one(.25), one(.75), sub)
        errs.append(float((ub - uf).norm() / uf.norm()))
    assert errs[-1] < 1e-4 and errs[0] / errs[1] > 4 and errs[1] / errs[2] > 4, errs
