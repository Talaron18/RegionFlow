#!/usr/bin/env bash
# WHERE does A2 lose? -- a t-region decomposition, on existing checkpoints.
#
# ---------------------------------------------------------------------------
# WHY THIS AND NOT A Delta=0.05 DRIFT TEST (the earlier proposal, withdrawn).
#
# The withdrawn hypothesis was "A2 replaces the r == t anchors, so it loses the
# Delta=0 identity u(z,t,t) = v and drifts away from the instantaneous field".
# DELTA_CURRICULUM_REPORT.md:83-97 refutes it from the sampler code:
# sample_tr_scheduled applies the Delta cap BEFORE the anchor branch, and under
# anchor_mode="small_delta" the 75% "anchor" rows sit at nonzero Delta <= 0.05
# and DO feed f (unlike anchor_mode="exact", where delta_part(t,0) = 0
# identically and those rows give f no gradient at all). So A2 has MORE small-
# Delta supervision, not less, and the measured mechanism is gradient IMBALANCE:
# f is dominated by Delta <= 0.05 and goes 10x too flat in r at Delta ~ 0.2.
# That predicts A2 is GOOD at Delta=0.05, the opposite of the withdrawn test.
#
# LIVE TENSION, after pass3 settled the seed question: A2's Delta=0.5 composition
# lead REPLICATES in both of its seeds (0.241 / 0.225 vs A0's 0.195 / 0.176, all
# four summaries agreeing on that cell), while the per-cell eval at
# t_center=0.5 calls A2 vs A1 a TIE (vs_base_at_a_opt 0.4313 vs 0.4303).
#
# Instrument B compares one jump against THE SAME FIELD'S OWN fine trajectory, so
# it scores SELF-CONSISTENCY -- the property a coarse sampler relies on. The
# per-cell eval scores ACCURACY against the true interval average u_true at a
# fixed t_center. A2 can lead the first while tying on the second for two
# different reasons, and separating them is this script's whole job:
#
#   H-acc     A2's field is a better interval average somewhere on the grid.
#             -> A2 must be BETTER than A0 on vs_base at some t_center.
#   H-smooth  A2's flow has smaller curvature ||Dg/Dt||, and E_comp converges to
#             the Euler floor (Delta^2/2)||Dg/Dt||. A self-consistent but no more
#             accurate field therefore scores better on instrument B.
#             -> A2 must be tied or WORSE than A0 at EVERY t_center, while
#                instrument B keeps its Delta=0.5 lead.
#
# PRE-REGISTERED READ (before looking at the numbers):
#   * primary  the (A2 - A0) vs_base and D_r_ratio matrices over t_center x
#              Delta. A negative entry means A2 is better on that cell (a
#              SMALLER sse_model / sse_base ratio).
#   * if NO t_center shows A2 ahead at Delta=0.5, report H-smooth: the
#              composition lead is self-consistency, not accuracy, and section 4
#              of H0_H0PRIME_SPEC.md must carry that caveat wherever R_comp is
#              quoted.
#   * a single t_center where A2 wins at Delta=0.5 is enough for H-acc, but
#              check it against the t_center=0.5 tie quoted above before
#              claiming it.
#   * either way this is NOT a verdict of the form "A2 is better/worse". The
#              question is what instrument B measures.
#   * Delta=0.5 at t_center=0.25 queries r=0.00,t=0.50 and at 0.75 queries
#              r=0.50,t=1.00: both inside [0,1], no domain skip. (STAGE1C's
#              t_center=0.20 problem was r=-0.05.)
#
#   bash scripts/a2_anchor_probe.sh          # ~3 model loads, a few minutes
#   FORCE=1 bash scripts/a2_anchor_probe.sh  # re-run even if the JSONs exist
# ---------------------------------------------------------------------------
set -u

cd "$(dirname "$0")/.." || exit 1
# The stdout redirect is opened by the SHELL, before python can create its
# --out parent (Path(...).parent.mkdir happens inside the eval). Without this,
# a missing runs/ turns every invocation into a shell error instead of a run.
mkdir -p runs

CACHE="${CACHE:-/data/cache/13061d51aee4}"
S1="${S1:-/data/runs}"
PY="${PY:-python}"
ESEED="${ESEED:-0}"
TC_LIST="${TC_LIST:-0.25 0.5 0.75}"
DELTAS="${DELTAS:-0.05 0.1 0.2 0.5}"
ARMS="${ARMS:-curr_a0_control_s1 curr_a1_cap_s1 curr_a2_anchor_s1}"

CKPTS=""
for a in $ARMS; do
    if [ ! -f "$S1/$a/ckpt_final.pt" ]; then
        echo "[MISS] $S1/$a/ckpt_final.pt -- set S1=/path/to/seed1/runs"
        exit 1
    fi
    CKPTS="$CKPTS $S1/$a/ckpt_final.pt"
done
# fd_distill_5k is the fourth reference the H1a sweep used; optional.
if [ -f runs/fd_distill_5k/ckpt_step5000.pt ]; then
    CKPTS="$CKPTS runs/fd_distill_5k/ckpt_step5000.pt"
fi

# One invocation per t_center, ALL checkpoints inside it: eval_interval_field.py
# draws eps once (line 229) before the checkpoint loop, so every checkpoint in
# one invocation is scored on identical (x0, eps, prompts). Comparing across
# invocations is still fine because the sample draw is seeded by --seed.
for tc in $TC_LIST; do
    tag="tc$(echo "$tc" | tr -d '.')"
    out="runs/a2_${tag}.json"
    if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] $out"
        continue
    fi
    echo "[run ] t_center=$tc -> $out"
    # shellcheck disable=SC2086
    $PY scripts/eval_interval_field.py --cache "$CACHE" --ckpt $CKPTS \
        --deltas $DELTAS --t-center "$tc" --samples 32 --substeps 16 \
        --rank 32 --seed "$ESEED" --out "$out" \
        > "runs/a2_${tag}.stdout" 2>&1 \
        && echo "[ ok ] $out" \
        || echo "[FAIL] $out -- see runs/a2_${tag}.stdout"
done

$PY - <<'PY'
import glob, json, os, sys

# Sort by the t_center RECORDED IN THE FILE, not by the filename: "tc05" is
# t_center 0.5, and int("05")/100 would sort it as 0.05.
blobs = []
for path in glob.glob("runs/a2_tc*.json"):
    try:
        blobs.append((float(json.load(open(path))["config"]["t_center"]),
                      path))
    except Exception:                            # noqa: BLE001
        print(f"[skip] unreadable {path}")
blobs.sort()
if not blobs:
    print("no runs/a2_tc*.json yet -- nothing to tabulate")
    sys.exit(0)

SHORT = {"curr_a0_control_s1": "A0", "curr_a1_cap_s1": "A1",
         "curr_a2_anchor_s1": "A2", "fd_distill_5k": "fd5k"}
def short(name):
    run, stem = name.split("/", 1)
    return SHORT.get(run, run)

for tc, path in blobs:
    blob = json.load(open(path))
    res = blob["results"]
    deltas = [r["delta"] for r in next(iter(res.values()))["rows"]]
    print(f"\n=== t_center {tc}  ({os.path.basename(path)}) ===")
    for key in ("vs_base", "D_r_ratio"):
        label = ("vs_base = sse_model/sse_base   (< 1 beats reusing the "
                 "instantaneous velocity)" if key == "vs_base" else
                 "D_r_model / D_r_true   (1.0 = right magnitude of "
                 "r-dependence)")
        print(f"  {label}")
        print(f"    {'ckpt':<6}" + "".join(f"{'d=' + str(d):>10}" for d in deltas))
        for name, r in sorted(res.items(), key=lambda kv: short(kv[0])):
            cells = {row["delta"]: row[key] for row in r["rows"]}
            print(f"    {short(name):<6}"
                  + "".join(f"{cells.get(d, float('nan')):>10.3f}"
                            for d in deltas))
    # the difference matrix: the actual test
    a0 = next((r for n, r in res.items() if short(n) == "A0"), None)
    a2 = next((r for n, r in res.items() if short(n) == "A2"), None)
    if a0 and a2:
        r0 = {row["delta"]: row for row in a0["rows"]}
        r2 = {row["delta"]: row for row in a2["rows"]}
        print("  A2 - A0 (negative = A2 better on that cell):")
        print(f"    {'metric':<10}" + "".join(f"{'d=' + str(d):>10}"
                                              for d in deltas))
        for key in ("vs_base", "D_r_ratio"):
            print(f"    {key:<10}"
                  + "".join(f"{r2[d][key] - r0[d][key]:>10.3f}"
                            if d in r0 and d in r2 else f"{'-':>10}"
                            for d in deltas))
        for d in deltas:
            if d in r0 and d in r2:
                who = "A2 better" if r2[d]["vs_base"] < r0[d]["vs_base"] \
                    else "A2 worse/equal"
                print(f"    d={d:<5} vs_base A0 {r0[d]['vs_base']:.3f} vs "
                      f"A2 {r2[d]['vs_base']:.3f}   -> {who}")
print("\nR1 vs R2: see the header. t_center=0.5 is the cell the per-cell "
      "report measured;\ncomposition (instrument B) averages over the whole "
      "grid. pass3/seed 0 gates any A2 claim.")
PY