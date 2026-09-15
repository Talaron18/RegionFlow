"""
Delta-ORDER curriculum for the reference MeanFlow path.

WHAT THIS IS NOT. It is not a way to reach larger Delta. Delta = 1/NFE is the
operating point, not a free hyperparameter: Delta 0.5 IS two-step editing,
Delta 0.25 IS four-step. The cap must END at the largest Delta the sampler will
actually query at inference, and `cap_end` defaults to 0.5 for that reason.
What the curriculum changes is the ORDER in which those intervals are learned,
not the set of them.

THE HYPOTHESIS IT TESTS. Stage 1C recorded that anchor SSE climbs 0.14 -> 339
by step 1000 and decays to ~170, unclipped, with the clipping explanation
withdrawn and nothing put in its place (STAGE1C_REPORT Sec.4). Kim et al.,
"Understanding, Accelerating, and Improving MeanFlow Training" (arXiv
2511.19065), report that u-supervision at large temporal gaps degrades an
already-pretrained instantaneous velocity, while small gaps build and refine
it. Our adapter sits on a FROZEN pretrained v, so their O1 ("establish v
early") is satisfied for free -- but their O2 predicts that our wide-Delta
samples are actively corrupting the effective v the adapter induces, and the
anchor SSE climb is what that looks like from inside our training loop.

    PREDICTION: cap Delta small and the anchor-SSE climb flattens.
    If it climbs identically, the mechanism is something else and one 2 h run
    has closed a loose end that has been open since Stage 1B.

Their acceleration half -- p_acc(t) timestep sampling and alpha(t) weighting on
L_v -- does NOT transfer. We are not learning v. Implementing it would be half
a method for no reason. Only the progressive Delta weighting is ported here.

WHAT IS PRESERVED EXACTLY. `sample_tr_scheduled` draws (a, b) exactly as
`meanflow_ref.sample_tr` does and takes t = max(a, b) unchanged. Only r moves,
and only ever UP (toward t). So the marginal of t is bit-identical to the
reference at every step, and at cap >= 1.0 the whole function reduces to the
reference. That is asserted in `self_test`, not assumed: a curriculum that
quietly reshapes the t marginal would confound the very comparison it exists
to make.

THE ANCHOR QUESTION, REOPENED. The reference forces r == t on
`data_proportion` (0.75) of the batch to supply the flow-matching signal. Kim
et al. find small-gap u-supervision is a better proxy for that signal than pure
v-loss -- higher task affinity for the later, wider intervals. So the move is
not to LOWER the anchor fraction (which Gate B tried) but to REPLACE anchors
with small-Delta samples. `anchor_mode="small_delta"` does that.

    Gate B's "P10" run never ran at 10%: `data_proportion` was never threaded
    from the CLI and `frac_anchor` came out 0.762. `assert_flag_arrived` below
    is there so that cannot happen twice. Call it. Check `vars(args)` in the
    output config before reading a single number.

    python delta_curriculum.py --self-test
    python delta_curriculum.py --plot       # print the cap schedule
"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------- schedule --

@dataclass
class DeltaCurriculumConfig:
    """Add these to RefMeanFlowConfig, or hold alongside it.

    Defaults reproduce the REFERENCE exactly (`enabled=False`). Turning the
    curriculum on is an explicit act that shows up in `vars(args)`.
    """

    enabled: bool = False

    #: Delta cap at step 0. 0.05 is the smallest interval the eval grid
    #: reports, so the first phase trains on intervals we actually measure.
    cap_start: float = 0.05

    #: Delta cap once warmup ends. MUST equal the widest interval inference
    #: will query: 0.5 for a two-step sampler, 0.25 for four-step. This is an
    #: operating point, not a tuning knob -- see the module docstring.
    cap_end: float = 0.5

    #: Fraction of training spent widening. The last (1 - frac) holds at
    #: cap_end so the final checkpoint has CONVERGED at the interval it will
    #: be evaluated on, rather than having just arrived there.
    warmup_frac: float = 0.5

    #: "exact"       r := t on the anchor samples (the reference).
    #: "small_delta" r := t - w*U(0,1) instead. Tests whether small-gap
    #:               u-supervision substitutes for the r == t anchors.
    anchor_mode: str = "exact"
    anchor_width: float = 0.05


def delta_cap(step: int, total_steps: int, cfg: DeltaCurriculumConfig) -> float:
    """Geometric widening, then hold.

    Geometric rather than linear for the same reason train.delta_schedule is:
    what matters is the interval's ratio to the region where the frozen field
    is locally accurate, and that region does not grow linearly. SD3-Medium is
    stiff -- ||du/dt|| is ~15x ||u||, so the correction term (t-r)*du/dt
    overtakes v at Delta ~ 0.07 -- which is exactly why the early phase has to
    sit below that and the widening has to be multiplicative.
    """
    if not cfg.enabled:
        return 1.0
    n = max(1, int(total_steps * cfg.warmup_frac))
    if step >= n:
        return cfg.cap_end
    frac = step / n
    return float(cfg.cap_start * (cfg.cap_end / cfg.cap_start) ** frac)


# ---------------------------------------------------------------- sampler --

def sample_tr_scheduled(batch, cfg, device, step, total_steps,
                        cur: DeltaCurriculumConfig,
                        generator=None):
    """Drop-in for `meanflow_ref.sample_tr`, with the Delta cap applied.

    Returns (t, r, anchor), same contract as the reference.

    Deviation from the reference, stated rather than buried: after the
    reference's two logit-normal draws and sort, r is raised to t - cap when
    the drawn interval exceeds the cap. t is never touched. At cap >= 1.0 no r
    is raised and the function IS the reference.
    """
    import torch  # noqa: PLC0415  -- keeps this module importable for tests

    def draw():
        n = torch.randn(batch, device=device, generator=generator)
        return torch.sigmoid(n * cfg.p_std + cfg.p_mean)

    a, b = draw(), draw()
    t = torch.maximum(a, b)
    r = torch.minimum(a, b)

    cap = delta_cap(step, total_steps, cur)
    if cap < 1.0:
        r = torch.maximum(r, t - cap)

    if cfg.stochastic_anchor:
        anchor = (torch.rand(batch, device=device, generator=generator)
                  < cfg.data_proportion)
    else:
        anchor = (torch.arange(batch, device=device)
                  < int(batch * cfg.data_proportion))

    if cur.enabled and cur.anchor_mode == "small_delta":
        # Replace the r == t anchors with small-Delta samples. The anchors'
        # JOB is to supply the flow-matching signal; small-gap u-supervision
        # is reported to do that job better, and unlike r == t it produces a
        # non-degenerate interval target.
        w = torch.rand(batch, device=device, generator=generator) * cur.anchor_width
        r = torch.where(anchor, (t - w).clamp_min(0.0), r)
    else:
        r = torch.where(anchor, t, r)

    # DOMAIN GUARD. `true_average_velocity` integrates from r with no clamp,
    # and Stage 1C had to drop two eval cells for querying r < 0 / t > 1. The
    # guard belongs on the training side too, as an assertion rather than a
    # silent clamp -- a clamp would hide a schedule bug.
    assert float(r.min()) >= -1e-6, f"r < 0 at step {step}"
    assert float(t.max()) <= 1.0 + 1e-6, f"t > 1 at step {step}"
    assert float((t - r).min()) >= -1e-6, f"negative interval at step {step}"
    return t, r, anchor


def curriculum_stats(t, r, anchor, step, total_steps,
                     cur: DeltaCurriculumConfig) -> dict:
    """Log these every step. `delta_cap_applied` vs `delta_max` is what tells
    you the schedule is actually biting rather than being a no-op, which is
    the failure mode that made the P10 run worthless."""
    d = (t - r)
    return {
        "delta_cap_applied": delta_cap(step, total_steps, cur),
        "delta_max": float(d.max()),
        "delta_mean": float(d.mean()),
        "frac_anchor": float(anchor.float().mean()),
        "frac_at_cap": float(
            (d >= delta_cap(step, total_steps, cur) - 1e-6).float().mean()),
    }


def assert_flag_arrived(args, cfg, pairs) -> None:
    """Habit 6, made mandatory.

    Gate B's anchor-competition hypothesis is STILL untested because
    `data_proportion` was never threaded from the CLI to the loss config:
    `frac_anchor` came out 0.762 on a run labelled 10%, and the write-up
    compared two identical runs for a week. Call this immediately after
    building the config.

        assert_flag_arrived(args, cfg, [("data_proportion", "data_proportion")])
    """
    bad = []
    for arg_name, cfg_name in pairs:
        want = getattr(args, arg_name, None)
        got = getattr(cfg, cfg_name, None)
        if want is None:
            bad.append(f"--{arg_name.replace('_', '-')} is not on the parser")
        elif abs(float(want) - float(got)) > 1e-9:
            bad.append(f"{arg_name}={want} never reached cfg.{cfg_name}={got}")
    if bad:
        raise SystemExit("FLAG DID NOT ARRIVE:\n  " + "\n  ".join(bad))


# ------------------------------------------------------------- self-tests --

def self_test() -> None:
    import torch  # noqa: PLC0415

    cfg = _RefStub()
    off = DeltaCurriculumConfig(enabled=False)
    on = DeltaCurriculumConfig(enabled=True, cap_start=0.05, cap_end=0.5,
                               warmup_frac=0.5)
    dev = "cpu"
    N, TOT = 4096, 4000

    # 1. DISABLED IS THE REFERENCE, BIT FOR BIT. Same generator seed must give
    #    the identical (t, r) as the unscheduled path.
    g1 = torch.Generator(device=dev).manual_seed(7)
    t0, r0, a0 = sample_tr_scheduled(N, cfg, dev, 0, TOT, off, g1)
    g2 = torch.Generator(device=dev).manual_seed(7)
    t1, r1, a1 = _reference_sample_tr(N, cfg, dev, g2)
    assert torch.equal(t0, t1) and torch.equal(r0, r1) and torch.equal(a0, a1)

    # 2. t's MARGINAL IS UNTOUCHED by the curriculum. Only r moves.
    g1 = torch.Generator(device=dev).manual_seed(11)
    tA, rA, _ = sample_tr_scheduled(N, cfg, dev, 0, TOT, on, g1)
    g2 = torch.Generator(device=dev).manual_seed(11)
    tB, rB, _ = _reference_sample_tr(N, cfg, dev, g2)
    assert torch.equal(tA, tB), "the curriculum must not reshape t"
    assert bool((rA >= rB - 1e-6).all()), "r may only move UP, toward t"

    # 3. THE CAP BITES, and only where it should.
    d = (tA - rA)[~_anchors(N, cfg, dev, 11)]
    assert float(d.max()) <= 0.05 + 1e-6, float(d.max())

    # 4. THE SCHEDULE IS MONOTONE and lands exactly on the operating point.
    caps = [delta_cap(s, TOT, on) for s in range(0, TOT + 1, 50)]
    assert all(x <= y + 1e-9 for x, y in zip(caps, caps[1:])), "cap must not shrink"
    assert abs(delta_cap(TOT, TOT, on) - on.cap_end) < 1e-9
    assert abs(delta_cap(0, TOT, on) - on.cap_start) < 1e-9
    # and it holds at cap_end for the whole back half, not just the last step
    assert abs(delta_cap(int(TOT * 0.5), TOT, on) - on.cap_end) < 1e-9

    # 5. ANCHOR FRACTION IS PRESERVED in both anchor modes -- replacing the
    #    anchors must change WHAT they supervise, not HOW MANY there are.
    sd = DeltaCurriculumConfig(enabled=True, anchor_mode="small_delta",
                               anchor_width=0.05)
    g = torch.Generator(device=dev).manual_seed(3)
    t2, r2, a2 = sample_tr_scheduled(N, cfg, dev, 0, TOT, sd, g)
    assert abs(float(a2.float().mean()) - cfg.data_proportion) < 0.02
    # exact anchors are gone; small ones took their place
    assert float((t2 - r2)[a2].max()) <= 0.05 + 1e-6
    assert float((t2 - r2)[a2].mean()) > 0.0, "small_delta anchors must not be r == t"

    # 6. DOMAIN GUARD holds at every phase of the schedule.
    for s in (0, TOT // 4, TOT // 2, TOT):
        g = torch.Generator(device=dev).manual_seed(s + 1)
        t3, r3, _ = sample_tr_scheduled(N, cfg, dev, s, TOT, on, g)
        assert float(r3.min()) >= 0.0 and float(t3.max()) <= 1.0

    # 7. THE STAT THAT PROVES THE FLAG ARRIVED is nonzero early, ~0 late.
    g = torch.Generator(device=dev).manual_seed(5)
    st = curriculum_stats(*sample_tr_scheduled(N, cfg, dev, 0, TOT, on, g),
                          0, TOT, on)
    assert st["delta_cap_applied"] == on.cap_start
    assert st["delta_max"] <= on.cap_start + 1e-6

    # 8. assert_flag_arrived CATCHES the exact P10 bug.
    class Args:  # the CLI said 0.10 ...
        data_proportion = 0.10

    class BadCfg:  # ... and the config still holds the default
        data_proportion = 0.75
    try:
        assert_flag_arrived(Args(), BadCfg(), [("data_proportion",
                                                "data_proportion")])
    except SystemExit as exc:
        assert "never reached" in str(exc)
    else:
        raise AssertionError("assert_flag_arrived failed to catch the P10 bug")

    print("delta_curriculum self-test OK")


@dataclass
class _RefStub:
    """The fields sample_tr_scheduled reads from RefMeanFlowConfig."""
    p_mean: float = -0.4
    p_std: float = 1.0
    stochastic_anchor: bool = True
    data_proportion: float = 0.75


def _reference_sample_tr(batch, cfg, device, generator):
    """meanflow_ref.sample_tr, inlined ONLY so the self-test can prove
    equivalence without importing the project. Not for production use --
    production imports the real one."""
    import torch  # noqa: PLC0415

    def draw():
        n = torch.randn(batch, device=device, generator=generator)
        return torch.sigmoid(n * cfg.p_std + cfg.p_mean)
    a, b = draw(), draw()
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    if cfg.stochastic_anchor:
        anchor = (torch.rand(batch, device=device, generator=generator)
                  < cfg.data_proportion)
    else:
        anchor = torch.arange(batch, device=device) < int(batch * cfg.data_proportion)
    return t, torch.where(anchor, t, r), anchor


def _anchors(batch, cfg, device, seed):
    import torch  # noqa: PLC0415
    g = torch.Generator(device=device).manual_seed(seed)
    _ = torch.randn(batch, device=device, generator=g)
    _ = torch.randn(batch, device=device, generator=g)
    return torch.rand(batch, device=device, generator=g) < cfg.data_proportion


def print_schedule(total_steps: int = 4000) -> None:
    cur = DeltaCurriculumConfig(enabled=True)
    print(f"  {'step':>6} {'cap':>8}   phase")
    n = int(total_steps * cur.warmup_frac)
    for s in range(0, total_steps + 1, total_steps // 20):
        print(f"  {s:>6} {delta_cap(s, total_steps, cur):>8.3f}   "
              f"{'widening' if s < n else 'hold at operating point'}")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    elif "--plot" in sys.argv:
        print_schedule()
    else:
        print(__doc__)
