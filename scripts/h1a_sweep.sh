#!/usr/bin/env bash
# Arms 1 and 2 for the Delta-distribution mechanism question. ONE GPU, so the
# first question is whether two runs can share it.
#
#   bash scripts/run_arms_1_2.sh --bench   # ~minutes: s/step and peak VRAM for
#                                          # (A alone, B alone, A+B together)
#   bash scripts/run_arms_1_2.sh --run     # both, concurrently, per-run logs
#   bash scripts/run_arms_1_2.sh --run --serial   # one after the other
#
# PRE-REGISTERED PREDICTIONS (H0_H0PRIME_SPEC.md Sec.1.6; written before running)
#
#   arm 1  A2's imbalance removed: small_delta anchors kept, but the fraction
#          forced into the small-Delta band cut 0.75 -> 0.25. The sampler
#          simulation (scripts/probe_delta_mass.py) says the mass ratio
#          mass(d<=0.05)/mass([0.10,0.50]) goes 4.24x -> 0.63x and the share of
#          interval supervision in (0.10,0.15] goes 3.05% -> 9.02%.
#          PREDICT:  A2's per-cell cos at Delta~0.1 turns POSITIVE (it was
#          -0.559 at t=0.75 and -0.717 at t=0.50), and its m/b at Delta=0.1
#          gets WORSE, because the previous "accuracy" there was r-independent
#          drift, not interval learning. D_r overshoot at Delta=0.05 shrinks.
#
#   arm 2  ONE variable vs A0: --max-delta 0.5 -> 0.75, everything else equal.
#          At cap 0.5 the sampler puts a 2.09% POINT MASS exactly at Delta=0.5
#          (~100x the smooth density there), and our weakest evaluation cell
#          (t=1.0, Delta=0.5) sits on that spike. At cap 0.75 the spike drops to
#          0.08% and Delta=0.5 becomes an interior point.
#          PREDICT:  the Delta=0.5 cells improve, most at t=1.0 -- which is
#          where a 4-step sampler takes its first and largest step.
#
# NEITHER ARM IS A STEP TOWARDS THE PROJECT GOAL ON ITS OWN. They are mechanism
# experiments inside layer 3 (the few-step base); the goal is blocked by layers
# 1/2/4 (edit objective, inversion, mask path). See GATE_A_REOPENING_PLAN.md.
#
# IDEMPOTENT: a run whose <out>/ckpt_final.pt exists is skipped.
#
# RUN THE TWO COMMANDS ONE AT A TIME, AND THE BENCH ALONE.
#   `python scripts/diff_run_config.py ...` is CPU-only by construction, so
#   GPU 0% is EXPECTED for it -- but it should print within a second and use
#   well under 1 GB. The bench loads the fp32 pipeline (~30 GB of host RAM) and
#   spawns --num-workers dataloader processes. Started together they contend for
#   host memory and BOTH sit at GPU 0% with RAM climbing, which is not a GPU
#   problem at all. To isolate the dataloader: `NW=0` (no worker processes).
set -u

cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/arms12 runs

CACHE="${CACHE:-/data/cache/13061d51aee4}"
PY="${PY:-python}"
MODE="${1:---bench}"
NW="${NW:-4}"                 # dataloader workers; loading only, never the math
BENCH_STEPS="${BENCH_STEPS:-5}"   # use 1 to make the bench itself fast

# OFFLINE BY DEFAULT, and this is not a nicety. `--model` defaults to a Hub id
# (scripts/gate_b_meanflow.py:753), so `from_pretrained` makes an HTTPS request
# to the Hub unless offline mode is set. On a box with no route that request
# retries with backoff for MINUTES and then raises
#     httpx.ConnectError: [Errno 101] Network is unreachable
# which is indistinguishable from a hang: a bench sat for 4.5 minutes at 733 MB
# RSS (imports only) and 0 MiB of GPU before dying. The model is already in the
# local HF cache. Offline mode resolves it there, and if it is ever NOT cached it
# fails in one second with a cache-miss message instead of hanging.
# Override with HF_HUB_OFFLINE=0 if the cache really must be refreshed.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

preflight () {
    echo "[host] $(date '+%H:%M:%S')" \
         "$(free -g 2>/dev/null | awk '/^Mem:/ {printf "RAM %dG used / %dG total", $3, $2}')"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null | sed 's/^/    GPU /'
    fi
}
preflight

# \/\/\/ MATCH YOUR A0 LAUNCH LINE \/\/\/
# These are the values STAGE1B and DELTA_CURRICULUM_REPORT.md record for the
# arms (fd estimator, fp32, frozen tangent/first term, plain MSE weighting, no
# clipping because norm_p=0 gradients run ~1e2-1e4, grad checkpointing off --
# which is the configuration the 4.1 h / 5000-step throughput was measured in).
# If A0 differed anywhere, change it here too or the comparison is not paired.
#
# --kappa 0 --omega 1 IS NOT OPTIONAL with --first-term frozen. The CLI default
# is kappa=0.5, and transport/meanflow_ref.py:156-158 refuses the combination
# ("first_term='frozen' requires omega=1, kappa=0"), deliberately, rather than
# silently ignoring the guidance parameters the frozen first term replaces.
# Do not "clean up" these two flags.
BASE="--cache $CACHE --estimator fd --fd-eps 0.025 --tangent frozen \
--first-term frozen --omega 1 --kappa 0 --norm-p 0 --grad-clip 0 \
--interval-kind centered --no-grad-checkpointing --batch 2 --accum 8 \
--rank 32 --lr 5e-5 --steps 5000 --seed 0 --ckpt-every 1000 --keep-last 3 \
--val-every 500 --num-workers $NW --device cuda"

ARM1="--max-delta 0.5 --curriculum --cap-start 0.5 --cap-end 0.5 \
--anchor-mode small_delta --data-proportion 0.25 --out runs/a3_dp25"
ARM2="--max-delta 0.75 --out runs/a4_md75"

launch () {   # launch <tag> <extra> [<bench-flag>]
    local tag="$1" extra="$2" bench="${3:-}" pre=""
    if [ "${DRY:-0}" = "1" ]; then
        echo "  [dry] $PY scripts/gate_b_meanflow.py train $BASE $extra $bench"
        return 0
    fi
    mkdir -p "runs/arms12"
    # Optional hard stop, so a hang fails loudly instead of sitting there. The
    # bench path sets TIMEOUT (see below); real runs leave it at 0.
    if [ "${TIMEOUT:-0}" != "0" ] && command -v timeout >/dev/null 2>&1; then
        pre="timeout ${TIMEOUT}"
    fi
    # Say something BEFORE the run. The child's output is redirected to the file
    # on purpose (a crash must not be lost), which left the terminal silent for
    # the whole run and made "is it working or hung?" unanswerable from the
    # console -- that ambiguity already cost an hour once. `python -u` makes the
    # log live and keeps it readable if the process is killed.
    echo "[run ] $tag -> runs/arms12/$tag.stdout   (live: tail -f that file)"
    # shellcheck disable=SC2086
    $pre $PY -u scripts/gate_b_meanflow.py train $BASE $extra $bench \
        > "runs/arms12/${tag}.stdout" 2>&1
}

# Print what actually ran, once, so a bad flag is visible instead of inferred.
echo "[cmd ] $PY scripts/gate_b_meanflow.py train $BASE <arm> --bench-steps 5"
echo "[dry ] set DRY=1 to print the per-arm command without running it"

show_fail () {   # never swallow a failure: rc and the tail, right here
    local tag="$1" rc="$2" why="$3"
    echo "  [FAIL] $tag rc=$rc ($why)"
    echo "  --- tail of runs/arms12/$tag.stdout ---"
    tail -n 15 "runs/arms12/$tag.stdout" 2>/dev/null | sed 's/^/    | /'
    echo "  ----------------------------------------"
}

bench_one () {   # bench_one <tag> <extra>
    if [ "${DRY:-0}" = "1" ]; then
        echo "  [dry] $PY scripts/gate_b_meanflow.py train $BASE $2 --bench-steps 5"
        return 0
    fi
    local rc
    launch "$1" "$2" "--bench-steps $BENCH_STEPS"; rc=$?
    if [ "$rc" -ne 0 ]; then show_fail "$1" "$rc" "nonzero exit"; return 1; fi
    if ! grep -q "\[bench\]" "runs/arms12/$1.stdout"; then
        show_fail "$1" 0 "rc=0 but no [bench] line"
        return 1
    fi
    grep -h "\[bench\]" "runs/arms12/$1.stdout" | head -2
}

if [ "$MODE" = "--bench" ]; then
    # A hung bench must not sit there: fail at 15 min with the log tail shown.
    TIMEOUT="${TIMEOUT:-900}"
    echo "=== A alone ==="
    bench_one bench_a "$ARM1"
    echo "=== B alone ==="
    bench_one bench_b "$ARM2"
    echo "=== A and B together ==="
    if [ "${DRY:-0}" = "1" ]; then
        echo "  [dry] both concurrently"
    else
        launch bench_ab_a "$ARM1" "--bench-steps $BENCH_STEPS" &
        pa=$!
        launch bench_ab_b "$ARM2" "--bench-steps $BENCH_STEPS" &
        pb=$!
        wait "$pa"; ra=$?
        wait "$pb"; rb=$?
        [ "$ra" -eq 0 ] && grep -h "\[bench\]" runs/arms12/bench_ab_a.stdout \
            | head -1 | sed 's/^/  A concurrent: /' \
            || show_fail bench_ab_a "$ra" "concurrent"
        [ "$rb" -eq 0 ] && grep -h "\[bench\]" runs/arms12/bench_ab_b.stdout \
            | head -1 | sed 's/^/  B concurrent: /' \
            || show_fail bench_ab_b "$rb" "concurrent"
    fi
    cat <<'EOF'

READ IT LIKE THIS. If the concurrent s/step is ~1x the solo number, the GPU has
spare capacity at batch 2 and running both costs one arm's wall time. If it is
~2x, there is nothing to gain and --run --serial is equivalent. Do not skip this
step: batch 2 x accum 8 is likely latency-bound, but only the measurement knows.
EOF
    exit 0
fi

if [ "$MODE" != "--run" ]; then
    echo "usage: $0 [--bench|--run] [--serial]"; exit 2
fi

skip () { [ -f "$1/ckpt_final.pt" ]; }

if [ "${2:-}" = "--serial" ]; then
    echo "=== serial ==="
    for pair in "a3_dp25|$ARM1" "a4_md75|$ARM2"; do
        tag="${pair%%|*}"; extra="${pair#*|}"
        out="runs/$tag"
        if skip "$out"; then echo "[skip] $tag"; continue; fi
        launch "$tag" "$extra" && echo "[ ok ] $tag" \
            || echo "[FAIL] $tag -- see runs/arms12/$tag.stdout"
    done
else
    echo "=== concurrent (logs in runs/arms12/*.stdout) ==="
    started=""
    for pair in "a3_dp25|$ARM1" "a4_md75|$ARM2"; do
        tag="${pair%%|*}"; extra="${pair#*|}"
        if skip "runs/$tag"; then echo "[skip] $tag"; continue; fi
        launch "$tag" "$extra" &
        started="$started $!"
    done
    rc=0
    for pid in $started; do wait "$pid" || rc=1; done
    echo "[done] rc=$rc -- check each runs/<tag>/ for ckpt_final.pt"
fi

cat <<'EOF'

AFTER BOTH FINISH (evaluation is serial -- the eval loads the model too):

  bash scripts/h1a_sweep.sh --table          # only if these arms are added to it

and on each new checkpoint, the two instruments that produced the predictions:

  python scripts/eval_interval_field.py --cache $CACHE \
      --ckpt runs/a3_dp25/ckpt_final.pt runs/a4_md75/ckpt_final.pt \
      --deltas 0.1 0.25 0.5 --t-center 0.5 --samples 32 --substeps 16 \
      --rank 32 --seed 0 --out runs/arms12_eval.json

  bash scripts/a2_anchor_probe.sh            # t_center cut, if A0/A2 are in S1

Predictions to check first, in this order: arm 1's cos at Delta=0.1 (must turn
positive from -0.559/-0.717) and arm 2's t=1.0 / Delta=0.5 vs_base.
EOF