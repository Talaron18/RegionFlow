"""
Compare Gate C arms, paired case by case.

Every arm shares `case_seed(id, 0)`, so the same case under two mask modes
differs ONLY in the spatial condition. That makes the comparison paired, and
paired is not optional here: the completed run already showed that unpaired
medians move the opposite way to the paired test, consistently, across four
thresholds (gate_c_note.md Sec.15). Anything this script reports across arms is
paired; anything it cannot pair, it says so.

    python compare_arms.py \\
        --arm none:runs/gate_c/refedit_sd3.jsonl \\
        --arm white:runs/gate_c/ue_white.jsonl \\
        --arm oracle:runs/gate_c/ue_oracle.jsonl \\
        --arm sibling:runs/gate_c/ue_sibling.jsonl \\
        --reference none --prompt-kind full

`outcome_class` is IMPORTED from run_gate_c, not restated. A duplicated
classifier in `stratify_analysis` already produced a 47-vs-43 disagreement
because it checked `mis_targeted` before `no_change` while the original checks
`no_change` first.
"""

from __future__ import annotations

import argparse
import collections
import json
from math import comb
from pathlib import Path

import numpy as np

from run_gate_c import outcome_class


def load(path: str, kind: str) -> dict:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {r["id"]: r for r in recs if r["prompt_kind"] == kind}


def mcnemar(a: dict, b: dict, ids, predicate) -> tuple:
    """Exact paired binary test. Returns (b_count, c_count, p)."""
    nb = sum(1 for i in ids if not predicate(a[i]) and predicate(b[i]))
    nc = sum(1 for i in ids if predicate(a[i]) and not predicate(b[i]))
    n = nb + nc
    if n == 0:
        return nb, nc, 1.0
    p = sum(comb(n, k) for k in range(min(nb, nc) + 1)) / 2 ** n * 2
    return nb, nc, min(p, 1.0)


def mis(r: dict) -> bool:
    return r.get("outcome", outcome_class(r)) == "mis_targeted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME:PATH", help="repeatable; e.g. oracle:runs/x.jsonl")
    ap.add_argument("--reference", default="none",
                    help="arm every other arm is tested against")
    ap.add_argument("--prompt-kind", default="full", choices=("full", "single"))
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="PRE-REGISTERED. 0.01, not 0.05: several arms are "
                         "tested against one reference")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition(":")
        arms[name] = load(path, args.prompt_kind)
    if args.reference not in arms:
        raise SystemExit(f"reference arm {args.reference!r} not among {list(arms)}")

    ids = sorted(set.intersection(*(set(a) for a in arms.values())))
    print("=" * 74)
    print(f"GATE C ARMS -- prompt={args.prompt_kind}, paired on {len(ids)} cases")
    for n, a in arms.items():
        print(f"  {n:<10} {len(a)} cases in file")
    if not ids:
        raise SystemExit("no cases common to all arms")
    print("=" * 74)

    # ---------------------------------------------------------- premise P1 --
    # Does mask conditioning HARD-COMPOSITE? If it does, sibling_change is zero
    # by construction, `mis_targeted` is identically 0, and the arm cannot
    # speak to selectivity at all -- the collapse would be arithmetic, not a
    # finding. Detect it from the data: a compositing pipeline leaves
    # background_change at ~0 outside the mask.
    print("\nP1  does the mask arm composite, or condition?")
    ref_bg = np.median([arms[args.reference][i]["background_change"] for i in ids])
    for n, a in arms.items():
        bg = np.array([a[i]["background_change"] for i in ids])
        verdict = ("COMPOSITES -- mis-targeting is 0 by construction here; do "
                   "NOT read it" if np.median(bg) < 0.1 * ref_bg
                   else "conditions softly; mis-targeting is readable")
        print(f"  {n:<10} background_change median {np.median(bg):.4f}   {verdict}")

    # ---------------------------------------------------------- premise P2 --
    # Does the model USE the mask? Pointing it at a sibling must drive
    # mis-targeting up sharply. If it does not, the mask is being ignored and
    # BOTH branches of the oracle reading key are void -- Case A and Case B
    # would then be the same null.
    print("\nP2  does the model use the mask? (sibling arm = positive control)")
    if "sibling" in arms:
        ref_rate = np.mean([mis(arms[args.reference][i]) for i in ids])
        sib_rate = np.mean([mis(arms["sibling"][i]) for i in ids])
        nb, nc, p = mcnemar(arms[args.reference], arms["sibling"], ids, mis)
        ok = sib_rate > ref_rate and p < args.alpha
        print(f"  mis-targeting {ref_rate:.1%} -> {sib_rate:.1%}   "
              f"discordant {nb}:{nc}  p={p:.4f}")
        print("  " + ("PASS: the mask moves the edit. Oracle arm is readable."
                      if ok else
                      "FAIL: the mask is being ignored. STOP -- the oracle "
                      "result means nothing\n  either way, and neither branch "
                      "of the reading key may be applied."))
    else:
        print("  sibling arm absent. Without it you cannot tell a real oracle "
              "effect\n  from a model that ignores masks. Run it.")

    # ------------------------------------------------------------ outcomes --
    print("\nOUTCOMES (all cases; `valid` discards the worst failures)")
    # off_inst is the count `outcome_class` now carves out of what used to be
    # `no_change`: a real edit, landing entirely outside every tracked mask.
    # It is disjoint from no_tracked -- outcome_class returns exactly one
    # string per case -- so the five columns still sum to len(ids).
    hdr = (f"  {'arm':<10} {'mis':>12} {'no_tracked':>12} {'off_inst':>12}"
          f" {'measurable':>12} {'valid':>10}")
    print(hdr)
    for n, a in arms.items():
        oc = collections.Counter(a[i].get("outcome", outcome_class(a[i])) for i in ids)
        nv = sum(1 for i in ids if a[i]["valid"])
        print(f"  {n:<10} {oc['mis_targeted']:>5} ({oc['mis_targeted']/len(ids):>4.0%})"
              f" {oc['no_tracked_change']:>5} ({oc['no_tracked_change']/len(ids):>4.0%})"
              f" {oc['off_instance_edit']:>5} ({oc['off_instance_edit']/len(ids):>4.0%})"
              f" {oc['measurable']:>5} ({oc['measurable']/len(ids):>4.0%})"
              f" {nv:>4} ({nv/len(ids):>4.0%})")

    print(f"\nMIS-TARGETING vs '{args.reference}' (McNemar, exact, "
          f"alpha={args.alpha})")
    for n, a in arms.items():
        if n == args.reference:
            continue
        nb, nc, p = mcnemar(arms[args.reference], a, ids, mis)
        star = "*" if p < args.alpha else " "
        print(f"  {n:<10} {np.mean([mis(a[i]) for i in ids]):>6.1%}   "
              f"became mis-targeted {nb}, recovered {nc}   p={p:.4f} {star}")

    # -------------------------------------------------------- off-instance --
    print("\nOFF-INSTANCE (third failure mode; computed over ALL cases)")
    for n, a in arms.items():
        oi = [a[i] for i in ids if a[i].get("off_instance_bbox") is not None]
        if not oi:
            print(f"  {n:<10} not recorded -- rerun with the patched scorer")
            continue
        over = np.array([r["off_instance_over_target"] for r in oi])
        ratio = np.array([r["off_instance_ratio"] for r in oi])
        print(f"  {n:<10} ratio median {np.median(ratio):>5.2f}   "
              f"over-target median {np.median(over):>5.2f}   "
              f"louder than the target in {(over > 1).sum()}/{len(oi)}")

    # -------------------------------------------------- leakage, with its K --
    # No leakage figure may be quoted without its K: the median moved
    # 0.082 -> 0.168 as K fell 10 -> 2 on the completed run.
    print("\nLEAKAGE across the validity threshold K (valid: tgt > K x bg)")
    print(f"  {'K':>4}  {'arm':<10} {'n_valid':>8} {'ratio med':>10} "
          f"{'excess med':>11}")
    for K in (10.0, 5.0, 3.0, 2.0):
        for n, a in arms.items():
            vs = [a[i] for i in ids
                  if a[i]["target_change"] > K * (a[i]["background_change"] + 1e-9)]
            if not vs:
                print(f"  {K:>4.0f}  {n:<10} {0:>8}")
                continue
            lr = np.array([r["leakage_ratio"] for r in vs])
            ex = np.array([r["sibling_excess_over_bg"] for r in vs])
            print(f"  {K:>4.0f}  {n:<10} {len(vs):>8} {np.median(lr):>10.3f} "
                  f"{np.median(ex):>+11.4f}")
    print("\n  Mis-targeting does NOT depend on K. Leakage does. Lead with "
          "mis-targeting.")

    # -------------------------------------------------------- reading key --
    print("\n" + "=" * 74)
    print("READING KEY -- pre-registered, apply only if P1=conditions and P2=PASS")
    print("=" * 74)
    print("""
  A  oracle mis-targeting collapses AND the blind clean-edit rate stays ~18%
     -> localisation is solved by conditioning; transformation is the whole
        residual. RegionFlow's contribution must be DIRECTION, not region.

  B  oracle mis-targeting stays ~28%
     -> text conditioning overrides spatial conditioning inside the field.
        Stronger than A, and it makes the mask-conditioned ablation row cheap.

  C  mis-targeting collapses AND clean-edit rate rises well above 18%
     -> mask conditioning is most of the win; the margin over an UltraEdit-
        style baseline is narrow and the thesis needs rewriting.
        COMMITTED TO IN ADVANCE. Report it if it happens.

  The clean-edit rate is NOT in this script and cannot be. The metric is blind
  to edit quality by construction (gate_c_note.md Sec.17). It comes from the
  blind audit, on the SAME case ids as the original 40, with all arms shuffled
  together and model identity hidden.
""")


if __name__ == "__main__":
    main()
