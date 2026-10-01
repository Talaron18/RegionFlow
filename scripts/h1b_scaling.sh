#!/usr/bin/env bash
# H1b -- the training-scaling arm. One run, 20k steps, a full snapshot ladder.
#
#   bash scripts/h1b_scaling.sh --bench     # pin s/step first (2 min) -- see WHY below
#   bash scripts/h1b_scaling.sh --run       # launch (put it in tmux)
#   bash scripts/h1b_scaling.sh --eval      # after it finishes: per-cell ladder
#   bash scripts/h1b_scaling.sh --table     # reprint from the existing JSON
#
# WHY THIS CONFIG AND NOT A1's. The question is "does more training keep raising
# composition?", so the thing that must stay FIXED across the ladder while `step`
# moves is the training distribution. A0's is step-invariant (curriculum off, a
# static cap of 0.5); A1's is not by construction (its cap climbs to 0.5 over the
# first half of `--steps`), so a 20k A1 run would answer a different question with
# a different distribution at every snapshot. Exact anchors and cap 0.5 are also
# the only configuration whose distillation pin converges (spec Sec.1.7): the
# `small_delta` route is closed and `--max-delta 0.75` is refuted (Sec.1.9).
#
# THREE DETAILS THAT WOULD SILENTLY DESTROY THIS RUN:
#   1. keep_last PRUNES. `transport/train.py:316-323` keeps only the LAST
#      `keep_last` numbered snapshots. A 20k run takes 20 snapshots at
#      ckpt_every*5 = 1000 steps; with the arms' keep_last=10 the pruning would
#      delete steps 1000..10000 and leave 11000..20000 -- the entire 1k-5k part of
#      the ladder, which is the part we already have to compare against. Hence
#      --keep-last 20. Disk: 20 x ~334 MB = ~6.7 GB.
#   2. ckpt_every IS ALSO THE CRASH BOUND. `ckpt_latest.pt` is rewritten every
#      ckpt_every steps, and a resume restarts from it. At 200 that is ~5 minutes
#      of loss (measured 1.49 s/step); at 1000 it is ~25.
#   3. THE STEP TIME IS UNRESOLVED BY 2x. The bench said 2.98 s/step (2x8) and a
#      real arm measured 1.49 s/step (2025 steps / 1.69 h). 20k steps is therefore
#      8.3 h or 16.6 h depending on which is right. Run --bench before booking.
#
# PRE-REGISTERED PREDICTION (written before the run; read `cos`, it is the only
# cross-run comparand -- same-recipe scatter 0.011 against 0.16-0.40 for the
# magnitude metrics, spec Sec.6.9):
#   * `cos` at Delta = 0.10/0.25 rises monotonically from 5k to 20k, and the
#     per-cell `m/b` falls with it.
#   * the composition ladder rises above the fd_s5000 baseline (R_comp 0.235 /
#     0.209 / 0.192 at Delta = 0.1/0.25/0.5) on the snapshots measured.
#   * CONTROL, and it must pass or nothing else is interpretable: this run's
#     step-5000 checkpoint has to reproduce the existing 5k runs' `cos` at the
#     same cells. If it does not, the config differs somewhere and the ladder
#     cannot be placed on the same axis as Sec.1.5/Sec.1.8.
#   * FALSIFICATION: if `cos` is flat from 5k to 20k, the training axis is closed
#     and the ~80% of the Euler headroom that R_comp does not explain is NOT an
#     undertraining effect. That is the input to the stop condition (Sec.6.5).
set -u

cd "$(dirname "$0")/.." || exit 1
mkdir -p runs

CACHE="${CACHE:-/data/cache/13061d51aee4}"
PY="${PY:-python}"
OUT="${OUT:-runs/h1b}"
STEPS="${STEPS:-20000}"
MODE="${1:---run}"

# Offline: `--model` defaults to a Hub id and an HTTPS probe on a box with no
# route retries for MINUTES before failing (that cost one 4.5-minute bench).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Shape matched to the arm checkpoints (scripts/diff_run_config.py on
# curr_a1/curr_a2: batch 16, accum 1, seed 1, grad_clip 0, ckpt_every 200) and
# the flags that made the runs start at all (--omega 1 --kappa 0 is REQUIRED by
# --first-term frozen, transport/meanflow_ref.py:156-158; --grad-clip 0 is the
# documented pair for --norm-p 0).
CFG="--cache $CACHE --estimator fd --fd-eps 0.025 --tangent frozen \
--first-term frozen --omega 1 --kappa 0 --norm-p 0 --grad-clip 0 \
--interval-kind centered --no-grad-checkpointing --batch 16 --accum 1 \
--rank 32 --lr 5e-5 --steps $STEPS --seed 1 --ckpt-every 200 --keep-last 20 \
--val-every 500 --max-delta 0.5 --num-workers 4 --out $OUT --device cuda"

case "$MODE" in
--bench)
    echo "[cmd ] $PY scripts/gate_b_meanflow.py train $CFG --bench-steps 5"
    echo "[host] $(date '+%H:%M:%S')" \
         "$(free -g 2>/dev/null | awk '/^Mem:/ {printf "RAM %dG used / %dG total", $3, $2}')"
    # shellcheck disable=SC2086
    $PY -u scripts/gate_b_meanflow.py train $CFG --bench-steps 5 2>&1 \
        | grep -E "^\[bench\]" || echo "[FAIL] no [bench] line -- rerun without the grep"
    echo "[note] if the BENCH number is the truth (2.98 s/step): ~$((STEPS * 298 / 360000)) h"
    echo "       if the measured real-run rate holds (1.49 s/step): ~$((STEPS * 149 / 360000)) h"
    echo "       (the two differ by 2x and the discrepancy is unexplained -- see the header)"
    ;;
--run)
    if [ -f "$OUT/ckpt_final.pt" ]; then
        echo "[skip] $OUT/ckpt_final.pt already exists"; exit 0
    fi
    # The SHELL opens the redirect below, before python can create anything.
    # `mkdir -p runs` is NOT enough: the target is runs/h1b/train.stdout, so a
    # missing $OUT kills the whole launch with "No such file or directory".
    # (Same class as the h1a_sweep.sh bug; the structural checker missed it
    # because the redirect pointed into a SUBdirectory -- now it checks that.)
    mkdir -p "$OUT"
    if [ -f "$OUT/ckpt_latest.pt" ]; then
        echo "[note] resuming from $OUT/ckpt_latest.pt (Trainer.load, gate_b_meanflow.py:640-642)"
    fi
    echo "[cmd ] $PY -u scripts/gate_b_meanflow.py train $CFG"
    echo "[run ] log -> $OUT/train.stdout   (live: tail -f that file)"
    # shellcheck disable=SC2086
    $PY -u scripts/gate_b_meanflow.py train $CFG > "$OUT/train.stdout" 2>&1 \
        && echo "[ ok ] done" || echo "[FAIL] see $OUT/train.stdout"
    ;;
--eval)
    CKPTS=$(ls -1 "$OUT"/ckpt_step*.pt 2>/dev/null | sort -V)
    [ -n "$CKPTS" ] || { echo "[MISS] no $OUT/ckpt_step*.pt"; exit 1; }
    list="$CKPTS $OUT/ckpt_final.pt"
    echo "[run ] per-cell eval over $(echo $list | wc -w) checkpoints, ONE model load"
    # THE DELTAS MUST MATCH THE BASELINES or the control cannot be read: Sec.1.8's
    # paired table is at 0.05 / 0.1 / 0.2 / 0.5 with t_center=0.5.
    # shellcheck disable=SC2086
    $PY -u scripts/eval_interval_field.py --cache "$CACHE" --ckpt $list \
        --deltas 0.05 0.1 0.2 0.5 --t-center 0.5 --samples 32 --substeps 16 \
        --rank 32 --seed 0 --out runs/h1b_ladder.json > runs/h1b_ladder.stdout 2>&1 \
        && echo "[ ok ] runs/h1b_ladder.json" || echo "[FAIL] see runs/h1b_ladder.stdout"
    "$0" --table
    ;;
--table) : ;;
*) echo "usage: $0 [--bench|--run|--eval|--table]"; exit 2 ;;
esac

# --bench and --run produce nothing for the ladder block.
[ "$MODE" = "--run" ] && exit 0
[ "$MODE" = "--bench" ] && exit 0

$PY - <<'PY'
import json, os, collections
path = "runs/h1b_ladder.json"
if not os.path.exists(path):
    print("no runs/h1b_ladder.json yet -- run --eval first"); raise SystemExit(0)
blob = json.load(open(path))
by = collections.defaultdict(dict)          # delta -> {step: (cos, m/b, Dr)}
for name, res in blob["results"].items():
    step = res.get("step")
    for r in res["rows"]:
        by[r["delta"]][step] = (r.get("cos_correction"), r.get("vs_base"),
                                r.get("D_r_ratio"))
steps = sorted({s for d in by.values() for s in d})
if not steps:
    print("no rows"); raise SystemExit(0)
print(f"\nH1b ladder -- `cos` per cell (the reliable comparand, spec Sec.6.9), "
      f"cells at t_center=0.5")
print(f"{'Delta':>6}" + "".join(f"{s:>10}" for s in steps))
for d in sorted(by):
    row = "".join((f"{by[d][s][0]:>10.3f}" if s in by[d] and by[d][s][0] is not None
                   else f"{'-':>10}") for s in steps)
    print(f"{d:>6.2f}" + row)
print(f"\n`m/b` (< 1 beats reusing v)")
print(f"{'Delta':>6}" + "".join(f"{s:>10}" for s in steps))
for d in sorted(by):
    row = "".join((f"{by[d][s][1]:>10.3f}" if s in by[d] and by[d][s][1] is not None
                   else f"{'-':>10}") for s in steps)
    print(f"{d:>6.2f}" + row)
print("\nTREND (first vs last checkpoint, by cos):")
for d in sorted(by):
    have = [s for s in steps if s in by[d] and by[d][s][0] is not None]
    if len(have) >= 2:
        a, b = by[d][have[0]][0], by[d][have[-1]][0]
        print(f"  Delta={d:<5} cos {a:+.3f} (step {have[0]}) -> {b:+.3f} "
              f"(step {have[-1]})   {b - a:+.3f}")
print("\nCONTROL: the step-5000 row must match the existing 5k runs at the same")
print("cells (A0 s1: cos +0.773 / +0.792 at Delta=0.1 / 0.2, t_center=0.5).")
print("If it does not, the ladder is not on the same axis as Sec.1.5/Sec.1.8.")
PY
