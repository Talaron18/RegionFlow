"""
Does the `person` concentration distort Gate C, and should it be capped?

THE CONCERN. `person` is ~52% of the benchmark (89 of 171 before filtering).
A headline "28% mis-targeting" half composed of people is partly a statement
about people, and COCO people overlap constantly, which interacts with the
well-separated-siblings scope limit already on record.

WHY NOT JUST CAP. Dropping cases to force a category balance throws away real
data, costs power on a result that is already thin (the ablation has 90 paired
cases at best), and the cap fraction is arbitrary -- 25% has no more claim than
30% or 20%. Capping also does not answer the question it appears to: it
produces ONE number from ONE random subset, when the actual question is whether
person and non-person cases behave differently at all.

WHAT THIS DOES INSTEAD.

  1. Tests whether person differs from non-person on the headline outcome
     (mis-targeting rate), with Fisher's exact test. If they do not differ,
     the pooled number is not distorted and capping is cosmetic.

  2. Reports a CATEGORY-BALANCED estimate alongside the pooled one -- the
     macro-average over strata, each weighted equally. This answers "what
     would the rate be if categories were balanced?" using ALL the data,
     with no dropped cases and no arbitrary threshold. It is strictly more
     informative than a cap.

  3. Shows what capping WOULD cost, so the choice is made against a number:
     cases retained and the width of the resulting interval.

  4. Bootstraps CIs for both estimates, because 151 cases split by category
     leaves small cells and point estimates alone would mislead.

READ IT AS. If Fisher is non-significant and pooled ≈ macro, report pooled with
the stratified breakdown beside it and do NOT cap -- state the composition as a
scope limit instead. If they diverge, report the macro-average as primary,
since it is the composition-independent estimate.

USAGE
    python bench/stratify_analysis.py --results runs/gate_c/refedit_sd3.jsonl
    python bench/stratify_analysis.py --results ... --self-test
"""

from __future__ import annotations

import argparse
import collections
import json
from math import comb
from pathlib import Path

import numpy as np


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p. No scipy dependency."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x):
        return (comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
                if 0 <= x <= r1 and 0 <= c1 - x <= n - r1 else 0.0)

    obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(0, min(r1, c1) + 1)
                        if prob(x) <= obs + 1e-12))


def wilson(k, n, z=1.96):
    """Wilson interval — honest at small n and near 0/1, unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_rate(flags, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    f = np.asarray(flags, float)
    if not len(f):
        return (0.0, 0.0)
    draws = rng.choice(f, size=(n_boot, len(f)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def bootstrap_macro(by_stratum, n_boot=5000, seed=0):
    """Resample WITHIN each stratum, then average strata equally."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(v, float) for v in by_stratum.values() if len(v)]
    if not arrs:
        return (0.0, 0.0)
    draws = np.stack([rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
                      for a in arrs]).mean(axis=0)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def analyse(recs, kind="full", valid_k=None):
    rs = [r for r in recs if r["prompt_kind"] == kind]
    if valid_k is not None:
        rs = [r for r in rs
              if r["target_change"] > valid_k * (r["background_change"] + 1e-9)]
    return rs


def mis_flags(rs):
    return [1.0 if r["outcome"] == "mis_targeted" else 0.0 for r in rs]


def report(recs, args):
    for kind in ("full", "single"):
        rs = analyse(recs, kind)
        if not rs:
            continue
        person = [r for r in rs if r["category"] == "person"]
        other = [r for r in rs if r["category"] != "person"]

        print("\n" + "=" * 76)
        print(f"  prompt = {kind}   ({len(rs)} cases, person "
              f"{len(person)} = {len(person) / len(rs):.0%})")
        print("=" * 76)

        pk, pn = int(sum(mis_flags(person))), len(person)
        ok_, on = int(sum(mis_flags(other))), len(other)
        tk, tn = pk + ok_, pn + on

        print("\n  MIS-TARGETING RATE (the headline; threshold-free)")
        for name, k, n in (("person", pk, pn), ("non-person", ok_, on),
                           ("pooled", tk, tn)):
            lo, hi = wilson(k, n)
            print(f"    {name:<12} {k:>3}/{n:<4} = {k / max(n, 1):>6.1%}  "
                  f"95% CI [{lo:.1%}, {hi:.1%}]")

        p = fisher_exact_2x2(pk, pn - pk, ok_, on - ok_)
        print(f"\n    person vs non-person: Fisher exact p = {p:.3f}"
              f"   {'-> no detectable difference' if p > 0.05 else '-> THEY DIFFER'}")

        strata = {"person": mis_flags(person), "other": mis_flags(other)}
        macro = np.mean([np.mean(v) for v in strata.values() if len(v)])
        mlo, mhi = bootstrap_macro(strata, seed=args.seed)
        plo, phi = bootstrap_rate(mis_flags(rs), seed=args.seed)
        print(f"\n    pooled estimate          {tk / max(tn, 1):.1%}  "
              f"boot CI [{plo:.1%}, {phi:.1%}]")
        print(f"    category-balanced (macro) {macro:.1%}  "
              f"boot CI [{mlo:.1%}, {mhi:.1%}]   <- uses ALL cases, no cap")

        print("\n  per-category (n >= 3)")
        by = collections.defaultdict(list)
        for r in rs:
            by[r["category"]].append(r["outcome"] == "mis_targeted")
        for cat, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
            if len(v) >= 3:
                lo, hi = wilson(sum(v), len(v))
                print(f"    {cat:<14} {sum(v):>3}/{len(v):<3} = {np.mean(v):>6.1%}"
                      f"  [{lo:.0%}, {hi:.0%}]")

        if kind == "full":
            print("\n  what CAPPING person would cost")
            rng = np.random.default_rng(args.seed)
            for frac in (0.50, 0.40, 0.30, 0.25):
                keep_p = min(len(person), int(frac * len(other) / (1 - frac)))
                rates = []
                for _ in range(400):
                    idx = rng.choice(len(person), size=keep_p, replace=False)
                    sub = [person[i] for i in idx] + other
                    rates.append(np.mean(mis_flags(sub)))
                n_kept = keep_p + len(other)
                lo, hi = np.percentile(rates, [2.5, 97.5])
                print(f"    cap {frac:.0%}: keep {keep_p}/{len(person)} person, "
                      f"n={n_kept} (-{len(rs) - n_kept})  rate "
                      f"{np.mean(rates):.1%} [{lo:.1%}, {hi:.1%}]")
            print("    -> a cap narrows nothing and discards cases; the macro "
                  "estimate above\n       answers the same question using all "
                  "of them.")

        print("\n  LEAKAGE by category, across validity thresholds")
        print(f"    {'K':>4}  {'person n':>9} {'median':>8}   "
              f"{'other n':>8} {'median':>8}")
        for K in (10.0, 5.0, 3.0, 2.0):
            vs = analyse(recs, kind, valid_k=K)
            pv = [r["leakage_ratio"] for r in vs if r["category"] == "person"]
            ov = [r["leakage_ratio"] for r in vs if r["category"] != "person"]
            print(f"    {K:>4.0f}  {len(pv):>9} "
                  f"{np.median(pv) if pv else float('nan'):>8.3f}   "
                  f"{len(ov):>8} {np.median(ov) if ov else float('nan'):>8.3f}")


def self_test():
    # Fisher against known values
    assert abs(fisher_exact_2x2(1, 9, 11, 3) - 0.0027) < 0.002, fisher_exact_2x2(1, 9, 11, 3)
    assert fisher_exact_2x2(5, 5, 5, 5) == 1.0

    lo, hi = wilson(28, 100)
    assert lo < 0.28 < hi and 0.19 < lo < 0.21 and 0.37 < hi < 0.39, (lo, hi)
    assert wilson(0, 10)[0] == 0.0

    # macro differs from pooled when strata are imbalanced AND differ
    strata = {"a": [1.0] * 10 + [0.0] * 90, "b": [1.0] * 5 + [0.0] * 5}
    macro = np.mean([np.mean(v) for v in strata.values()])
    pooled = np.mean(strata["a"] + strata["b"])
    assert abs(macro - 0.30) < 1e-9 and abs(pooled - 0.136) < 0.01, (macro, pooled)

    recs = []
    for i in range(60):
        recs.append({"id": f"r-{i}", "prompt_kind": "full",
                     "category": "person" if i < 30 else "car",
                     "outcome": "mis_targeted" if i % 3 == 0 else "measurable",
                     "leakage_ratio": 0.1, "target_change": 0.2,
                     "background_change": 0.01, "valid": True})
    assert len(mis_flags(recs)) == 60
    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="runs/gate_c/refedit_sd3.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    recs = [json.loads(l) for l in Path(args.results).read_text().splitlines()
            if l.strip()]
    # Must match run_gate_c.outcome_class EXACTLY, including the order of the
    # tests: no_tracked_change (formerly no_change) is checked FIRST.
    # Classifying mis_targeted first moves cases where everything is tiny AND
    # sib > tgt into the wrong bucket, and the two scripts then disagree on
    # the headline rate -- the exact 47-vs-43 failure mode compare_arms.py's
    # docstring records. This fallback is the second copy that failure mode
    # warns about; it exists only for a cwd where the import fails, and it
    # has to be kept current by hand for exactly that reason.
    try:
        from run_gate_c import outcome_class  # noqa: PLC0415
    except ImportError:
        from off_instance import OFF_INSTANCE_EDIT_OVER_TARGET  # noqa: PLC0415

        def outcome_class(r):
            tgt, sib = r["target_change"], r["sibling_change"]
            bg = r["background_change"]
            if max(tgt, sib) <= bg:
                if r.get("off_instance_over_target", 0.0) > OFF_INSTANCE_EDIT_OVER_TARGET:
                    return "off_instance_edit"
                return "no_tracked_change"
            return "mis_targeted" if sib > tgt else "measurable"
    for r in recs:
        r.setdefault("outcome", outcome_class(r))
    report(recs, args)


if __name__ == "__main__":
    main()
