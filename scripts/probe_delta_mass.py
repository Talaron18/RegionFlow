"""Where does the interval supervision MASS live in Delta? CPU only, seconds.

THE QUESTION (H0_H0PRIME_SPEC.md Sec.1.6). A2 replaces the `r == t` anchors with
Delta in (0, `anchor_width`], i.e. 75% of its batch sits at Delta <= 0.05 while
the remaining 25% comes from the same capped base draw as A0's interval samples.
A2 is measurably broken at Delta ~ 0.1 (per-cell `cos` NEGATIVE, -0.559/-0.717,
where A0/A1 read +0.67..+0.80) and fine at Delta = 0.25/0.50. The hypothesis under
test is that this creates a HOLE in the Delta band around 0.1.

WHY SIMULATE RATHER THAN REASON. This calls `delta_curriculum.sample_tr_scheduled`
-- the same function the training loop calls, with the same dataclasses -- so the
distribution is measured, not asserted. The base draw is two iid
`sigmoid(N(p_mean, p_std))` samples sorted into (t, r); the cap is applied as
`r := max(r, t - cap)`; the anchor mask is `rand(batch) < data_proportion`.

SELF-CHECKS against values that exist in the logs (run with --check):
  * `frac_anchor` ~= 0.75 in every arm
  * an A0-like config's INTERVAL mean Delta ~= 0.22 (logged 0.222 / 0.220)
  * A1 at step 0 (cap = cap_start = 0.05) puts nearly every interval sample AT the
    cap. The first logged record read frac_at_cap = 0.3125 == 5/16 EXACTLY, which
    is impossible as an ensemble mean (the non-anchor fraction is 0.25, so even
    every non-anchor clamping caps it at 0.25) but is exactly one batch of 16
    with 5 non-anchors. Hence this also prints the PER-BATCH spread, and the
    number stops looking like a contradiction.

    python scripts/probe_delta_mass.py
    python scripts/probe_delta_mass.py --check
    python scripts/probe_delta_mass.py --anchor-widths 0.02 0.05 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from delta_curriculum import (  # noqa: E402
    DeltaCurriculumConfig, delta_cap, sample_tr_scheduled,
)
from transport.meanflow_ref import RefMeanFlowConfig  # noqa: E402

# (lo, hi] bands; the first is the exact-zero (r == t) anchor rows and the last
# is the pile-up at the cap, which is a point mass, not a density.
BANDS = [(0.0, 0.0), (0.0, 0.025), (0.025, 0.05), (0.05, 0.075), (0.075, 0.10),
         (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.35), (0.35, 0.45),
         (0.45, 0.4999), (0.4999, 0.5001)]
BAND_LABEL = ["= 0 (exact anchor)", "(0, 0.025]", "(0.025, 0.05]",
              "(0.05, 0.075]", "(0.075, 0.10]", "(0.10, 0.15]", "(0.15, 0.20]",
              "(0.20, 0.25]", "(0.25, 0.35]", "(0.35, 0.45]", "(0.45, 0.4999]",
              "= 0.5 (cap)"]


def draw(cfg, cur, step, total, samples, batch):
    """(t, r, anchor) for `samples` training draws, in batches of `batch`."""
    ts, rs, an = [], [], []
    n = 0
    g = torch.Generator().manual_seed(1234)      # the trainer's tr_gen seed + 1
    while n < samples:
        t, r, a = sample_tr_scheduled(batch, cfg, "cpu", step, total, cur,
                                      generator=g)
        ts.append(t)
        rs.append(r)
        an.append(a)
        n += batch
    return torch.cat(ts), torch.cat(rs), torch.cat(an)


def band_masses(delta, sel, tol=1e-6):
    """Fraction of `sel` rows falling in each band -> (list, total_selected).

    The tolerance matters: the cap is applied as `r = max(r, t - cap)`, so a
    capped row has Delta = t - (t - cap), which in float32 lands a few 1e-8
    ABOVE cap often enough to be visible. Without `tol` the whole cap pile-up
    jumps into the next band up -- at A1's step-0 cap of 0.05 that moved 80% of
    the interval mass into "(0.05, 0.075]", which is impossible when the cap IS
    0.05.
    """
    d = delta[sel]
    out = []
    for lo, hi in BANDS:
        if lo == hi:
            m = (d <= 1e-9) if lo == 0.0 else (d >= 0.4999)
        else:
            m = ((d > lo + tol if lo > 0.0 else d > 0.0) & (d <= hi + tol))
        out.append(float(m.float().mean()) if d.numel() else float("nan"))
    return out, int(d.numel())


def report(name, delta, anchor, cur, cfg, batch, cap, per_batch_cap=None):
    total = delta.numel()
    interval = delta > 1e-9
    n_int = int(interval.sum())
    print(f"\n=== {name} ===")
    print(f"  batch {batch}  total draws {total}  frac_anchor "
          f"{float(anchor.float().mean()):.4f}   "
          f"interval-carrying {n_int / total:.4f} of the batch")
    # cap must be passed in: hardcoding 0.4999 here reported 0.0000 for A1 at
    # step 0, whose cap is 0.05 and where ~22% of the batch sits AT it.
    print(f"  interval mean Delta {float(delta[interval].mean()):.4f}   "
          f"frac_at_cap(Delta >= cap - 1e-6) "
          f"{float((delta >= cap - 1e-6).float().mean()):.4f}   (cap {cap:.4f})")
    if per_batch_cap is not None:
        print(f"  per-batch frac_at_cap over batches of {batch}: "
              f"mean {per_batch_cap.mean():.4f}  sd {per_batch_cap.std():.4f}  "
              f"max {per_batch_cap.max():.4f}   (k/{batch}: " + ", ".join(
                  f"{k}/{batch}={k / batch:.4f}" for k in
                  sorted({int(round(x * batch)) for x in
                          per_batch_cap.unique().tolist()})) + ")")

    m_all, _ = band_masses(delta, torch.ones_like(delta, dtype=torch.bool))
    m_int, _ = band_masses(delta, interval)
    print(f"  {'band':<20}{'of batch':>10}{'of interval':>13}{'density':>10}")
    for label, (lo, hi), a, b in zip(BAND_LABEL, BANDS, m_all, m_int):
        w = max(hi - lo, 1e-9)
        dens = (b / w) if hi > lo and b == b else float("nan")
        bar = "#" * int(round(40 * a / 0.8)) if a == a else ""
        print(f"  {label:<20}{100 * a:>9.2f}%{100 * b:>12.2f}%{dens:>10.2f}"
              f"   {bar}")
    # THE MECHANISM NUMBER: how much denser is the 0.05 band than the bulk?
    d_small = sum(m_int[i] for i, (lo, hi) in enumerate(BANDS)
                  if 0.0 <= lo and hi <= 0.05) / 0.05
    d_bulk = sum(m_int[i] for i, (lo, hi) in enumerate(BANDS)
                 if lo >= 0.10) / 0.40
    # MASS ratio, not density: this is the quantity `anchor_width` CANNOT change
    # (the small-Delta mass is set by data_proportion), so it is the one to
    # quote when comparing arms. The density ratio above moves with anchor_width
    # for free, which is exactly why narrowing the band must not be sold as a fix.
    mass_small = sum(m_int[i] for i, (lo, hi) in enumerate(BANDS)
                     if hi <= 0.05)
    mass_bulk = sum(m_int[i] for i, (lo, hi) in enumerate(BANDS)
                    if lo >= 0.10)
    if mass_bulk < 1e-4:
        print(f"  mass(Delta<=0.05) {100 * mass_small:6.2f}%   "
              f"mass([0.10,0.50]) ~0%   ratio n/a "
              f"(nothing above Delta 0.10 at this cap)")
    else:
        print(f"  mass(Delta<=0.05) {100 * mass_small:6.2f}%   "
              f"mass([0.10,0.50]) {100 * mass_bulk:6.2f}%   "
              f"ratio {mass_small / mass_bulk:5.2f}x   "
              f"<-- invariant to anchor_width")
    share = next(100 * b for i, b in enumerate(m_int)
                 if BANDS[i][0] == 0.10)
    print(f"  share of interval supervision in (0.10, 0.15]: {share:.2f}%")
    # The band table stops at Delta=0.50. At --max-delta 0.75 the cap pile-up
    # moves OUT of it, and without this line the printed masses simply do not
    # sum to 100% -- which is the whole point of the arm that tests whether the
    # pile-up at the cap is what poisons the Delta=0.50 cells.
    above = float((delta[interval] > 0.50).float().mean())
    if above > 1e-4:
        print(f"  interval mass ABOVE Delta 0.50: {100 * above:.2f}%  "
              f"(outside the band table; the cap pile-up moved here)")
    if d_bulk <= 0:
        # GENUINE at a narrow cap: at A1's step-0 cap of 0.05 the model never
        # sees Delta > 0.05 at all, so the ratio is undefined, not huge.
        print("  density(0, 0.05] / density[0.10, 0.50] = n/a "
              "(no interval mass above Delta 0.10 at this cap)")
        return float("nan")
    print(f"  density(0, 0.05] / density[0.10, 0.50] = "
          f"{d_small / d_bulk:.2f}x")
    return d_small / d_bulk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=16,
                    help="the arms trained at 16x1; fd_distill at 2x8")
    ap.add_argument("--max-delta", type=float, default=0.5)
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--anchor-widths", type=float, nargs="+", default=[0.05])
    ap.add_argument("--data-proportions", type=float, nargs="+", default=[0.75],
                    help="the fraction forced into the small-Delta band. 0.75 is "
                         "what every arm ran; this is the ONLY knob the sim "
                         "shows can move the mass (anchor_width cannot).")
    ap.add_argument("--check", action="store_true",
                    help="assert the self-checks against logged values")
    args = ap.parse_args()

    ref = RefMeanFlowConfig(max_delta=args.max_delta, data_proportion=0.75)
    fails = []

    # A0 control: curriculum OFF. delta_cap returns hard_max, so the cap is still
    # applied -- `hard_max` must be wired to --max-delta ALWAYS (see the field's
    # own comment), which is what makes this a reproduction of the baseline.
    cur0 = DeltaCurriculumConfig(enabled=False, hard_max=args.max_delta)
    t0, r0, a0 = draw(ref, cur0, 0, args.total_steps, args.samples, args.batch)
    d0 = t0 - r0
    cap0 = delta_cap(0, args.total_steps, cur0)
    report(f"A0 control (curriculum off, static cap {args.max_delta})", d0, a0,
           cur0, ref, args.batch, cap0)
    if args.check:
        fa = float(a0.float().mean())
        if abs(fa - 0.75) > 0.005:
            fails.append(f"A0 frac_anchor {fa:.4f} != 0.75")
        mi = float(d0[d0 > 1e-9].mean())
        if abs(mi - 0.22) > 0.02:
            fails.append(f"A0 interval mean Delta {mi:.4f} not ~0.22")
        print(f"  [check] interval mean Delta {mi:.4f} vs logged 0.220/0.222"
              f"   {'PASS' if abs(mi - 0.22) <= 0.02 else 'FAIL'}")

    # A1 cap curriculum, at several steps: the cap widens geometrically.
    for step in (0, 500, 1000, 2500, args.total_steps):
        cur1 = DeltaCurriculumConfig(enabled=True, cap_start=0.05,
                                     cap_end=0.5, warmup_frac=0.5,
                                     hard_max=args.max_delta)
        t1, r1, a1 = draw(ref, cur1, step, args.total_steps, args.samples,
                          args.batch)
        d1 = t1 - r1
        cap = delta_cap(step, args.total_steps, cur1)
        pb = None
        if step == 0:
            # per-batch frac_at_cap, to explain the logged 0.3125 = 5/16
            g = torch.Generator().manual_seed(7)
            vals = []
            for _ in range(2000):
                tt, rr, _ = sample_tr_scheduled(args.batch, ref, "cpu", step,
                                                args.total_steps, cur1,
                                                generator=g)
                vals.append(float(((tt - rr) >= cap - 1e-6).float().mean()))
            pb = torch.tensor(vals)
        report(f"A1 cap curriculum, step {step} (cap {cap:.4f})", d1, a1, cur1,
               ref, args.batch, cap, per_batch_cap=pb)
        if args.check and step == 0:
            print(f"  [check] logged first-record frac_at_cap 0.3125 == 5/{args.batch}"
                  f" = {5 / args.batch:.4f}; per-batch max here "
                  f"{float(pb.max()):.4f}, ensemble "
                  f"{float((d1 >= cap - 1e-6).float().mean()):.4f}")
            if float(pb.max()) < 0.30:
                fails.append("A1 step-0 per-batch frac_at_cap never reaches 0.30, "
                             "so 0.3125 is not a single-batch value")

    # A2 anchor replacement, plus the proposed fixes. `anchor_width` is included
    # only to be REFUTED (it cannot move the mass); `data_proportion` is the one
    # knob that can, and it is an existing CLI flag -- so a mechanism arm that
    # targets the measured imbalance needs NO code change.
    for dp in args.data_proportions:
        ref_dp = RefMeanFlowConfig(max_delta=args.max_delta, data_proportion=dp)
        for w in args.anchor_widths:
            cur2 = DeltaCurriculumConfig(
                enabled=True, cap_start=0.5, cap_end=0.5, warmup_frac=0.5,
                anchor_mode="small_delta", anchor_width=w,
                hard_max=args.max_delta)
            t2, r2, a2 = draw(ref_dp, cur2, 0, args.total_steps, args.samples,
                              args.batch)
            d2 = t2 - r2
            report(f"A2-like small_delta: data_proportion={dp} "
                   f"anchor_width={w}", d2, a2, cur2, ref_dp, args.batch,
                   delta_cap(0, args.total_steps, cur2))

    if args.check:
        print("\n" + "=" * 78)
        if fails:
            print("SELF-CHECK FAILURES:")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("self-checks PASS: the sampler's own code reproduces the logged "
              "frac_anchor, the logged interval mean Delta, and the logged "
              "step-0 frac_at_cap is a single-batch value, not a mean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())