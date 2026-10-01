"""Is there a step change between two checkpoint ladder points?

WHY THIS EXISTS. `scripts/eval_interval_field.py` only has checkpoints every
`ckpt_every * 5` steps (transport/train.py:415), so a change seen between, say,
11k and 12k cannot be resolved from snapshots. The training log is the only
finer-grained record. This asks the only question the log can answer: did the
windowed statistics SHIFT between the two checkpoints, or was the intervening
stretch stationary?

WHAT IT CANNOT DO. It cannot measure `D_r_ratio` or `cos` -- those are forward
probe quantities, not training stats. A null result here means "no detectable
optimisation event", NOT "the eval change was an artifact". Those are different
claims and the second does not follow.

WHY WINDOW KEYS. The log's non-`win_` keys are a SNAPSHOT OF ONE STEP in every
`log_every` (transport/train.py:375-392); `win_*` are means over every step
since the last log. `delta_mean` as a snapshot swings 0.00 -> 0.089 between
adjacent 100-step samples, so any test built on snapshots is dominated by
single-step heavy tails and has no power. This script reads `win_*` by default
and refuses to run on snapshot keys without --allow-raw.

MISSING KEYS ARE NOT ZEROS. `sse_interval` is only emitted when a batch has a
non-anchor row, so an all-anchor batch (~1% of steps) leaves the key ABSENT.
That is why a row can show a blank there. Indexing it as 0 would bias every mean.

USAGE
    python scripts/h1b_step_check.py --log runs/h1b/log.jsonl \
        --split 11500 --half 3000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: windowed statistics worth testing for a shift. Chosen because each is a
#: different axis: anchor (drift), interval (r-dependence error), the three
#: norms that say how big the field and its tangent are, the tangent ratio
#: (did the target move relative to the frozen field), the gradient, and lr
#: (the control: it MUST be smooth, or the log itself is broken).
DEFAULT_KEYS = [
    "win_sse_anchor", "win_sse_interval", "win_loss",
    "win_u_norm", "win_dudt_norm", "win_target_norm",
    "win_tangent_rel_vg", "grad_norm_win_mean",
    "win_delta_mean", "win_frac_anchor", "lr",
]


def _fit(pts):
    """Least-squares (intercept, slope) of y on x. Used to REMOVE the trend
    before testing for a change point."""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


def welch(a: list[float], b: list[float]):
    """(mean_a, mean_b, se, z) on whatever it is given."""
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    return ma, mb, se, (mb - ma) / se if se > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--split", type=int, required=True,
                    help="the step to test as a change point, e.g. 11500 for "
                         "'between the 11k and 12k checkpoints'")
    ap.add_argument("--half", type=int, default=3000,
                    help="window half-width in steps on each side of --split")
    ap.add_argument("--keys", nargs="+", default=None,
                    help=f"default: {DEFAULT_KEYS}")
    ap.add_argument("--allow-raw", action="store_true",
                    help="permit non-win_ keys (they are single-step snapshots "
                         "and the test loses its power -- see the docstring)")
    args = ap.parse_args()

    rows = []
    for line in Path(args.log).read_text(encoding="utf-8",
                                         errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    print(f"{args.log}: {len(rows)} records, "
          f"steps {min(r['step'] for r in rows)}..{max(r['step'] for r in rows)}")

    keys = args.keys or [k for k in DEFAULT_KEYS
                         if any(k in r for r in rows)]
    if not args.allow_raw:
        raw = [k for k in keys if not (k.startswith("win_") or k == "lr"
                                        or k.startswith("grad_norm_win"))]
        if raw:
            print(f"REFUSING: {raw} are single-step snapshots, not window "
                  f"means; the test would have no power. Pass --allow-raw to "
                  f"override, or use the win_* form.")
            return 2

    lo, hi = args.split - args.half, args.split + args.half
    if max(r["step"] for r in rows) < args.split:
        print(f"the log ends at {max(r['step'] for r in rows)} < --split "
              f"{args.split}: nothing to test. This is the check that needs a "
              f"log reaching PAST the second checkpoint.")
        return 3

    print(f"\nchange-point test at step {args.split}  (window {lo}..{hi})")
    print("  SHIFT = two-window mean difference, Z = Welch standard errors "
          "apart.")
    print("  Z_detr = the same test on the RESIDUALS of a linear fit over the "
          "whole window.")
    print("  READ Z_detr. A monotone statistic ALWAYS 'shifts' under a raw "
          "two-window test -- `lr` is a cosine, so Z_raw for it is ~-25 by "
          "construction. That is the trend, not a change point.")
    print(f"  {'key':<22}{'n':>5}{'before':>12}{'after':>12}"
          f"{'trend/1k':>11}{'shift':>11}{'z_raw':>8}{'z_detr':>8}")
    for k in keys:
        pts = [(r["step"], r[k]) for r in rows
               if k in r and r[k] is not None and lo <= r["step"] <= hi]
        if len(pts) < 12:
            print(f"  {k:<22}{len(pts):>5}   (too few points)")
            continue
        got = welch([v for s, v in pts if s < args.split],
                    [v for s, v in pts if s >= args.split])
        if got is None:
            print(f"  {k:<22}{len(pts):>5}   (too few points either side)")
            continue
        ma, mb, se, z_raw = got
        a0, slope = _fit(pts)
        resid = [(s, v - (a0 + slope * s)) for s, v in pts]
        gotr = welch([v for s, v in resid if s < args.split],
                     [v for s, v in resid if s >= args.split])
        z_d = gotr[3] if gotr else float("nan")
        flag = "  <-- CHANGE" if abs(z_d) >= 3 else ""
        print(f"  {k:<22}{len(pts):>5}{ma:>12.6g}{mb:>12.6g}"
              f"{slope * 1000:>11.4g}{mb - ma:>11.4g}{z_raw:>8.2f}"
              f"{z_d:>8.2f}{flag}")

    print("\nCONTROL: after detrending, `lr` must come out near zero -- it is a "
          "smooth cosine, so any |Z_detr| there means the log or the window is "
          "broken and nothing else is readable.")
    print("A null (|Z_detr| < 3 everywhere) means NO DETECTABLE OPTIMISATION "
          "EVENT between the two checkpoints. It does NOT mean an eval change "
          "was an artifact: this script cannot see D_r_ratio or cos at all.")
    print("NOTE: the eval's inputs are checkpoint-independent "
          "(eval_interval_field.py:209/229/319: x0 from the cache, eps from a "
          "seeded generator), so a checkpoint-to-checkpoint move in D_r_ratio "
          "is a real change in the weights' outputs, not a sampling artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
