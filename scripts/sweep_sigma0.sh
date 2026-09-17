#!/usr/bin/env bash
# Launch a sigma0 sweep: several concurrent Gate A Direction A training
# runs, one process per sigma0 value, on the SAME GPU. Answers
# GATE_A_REOPENING_PLAN.md SS8's open question ("pick a first sigma0 --
# not decided here") by comparing several at once instead of guessing one
# and training sequentially.
#
# Each run is a fully independent process -- own optimiser state, own
# dataloader, own --out directory -- nothing coordinates between them.
# The only thing that must differ is --out, which this script handles by
# construction (runs/gate_a_sigma_<value>/).
#
#   ./scripts/sweep_sigma0.sh /data/cache/<hash>
#   ./scripts/sweep_sigma0.sh /data/cache/<hash> 0.0 0.1 0.2 0.3
#
# VRAM: check your own headroom before widening the sweep. At ~13.8GB per
# run (single-run baseline reported against this codebase), leave real
# margin for activation-memory spikes and per-process CUDA context
# overhead rather than dividing total VRAM by the steady-state number --
# 4-5 concurrent runs on a 96GB card is a reasonable starting point, not
# 6-7.
#
# Watch all of them at once:
#   python scripts/gate_a_transport.py watch-sweep \
#       --outs runs/gate_a_sigma_0.0 runs/gate_a_sigma_0.1 ... --every 100
set -euo pipefail

CACHE="${1:?usage: sweep_sigma0.sh <cache> [sigma0_values...]}"
shift || true
SIGMAS=("$@")
if [[ ${#SIGMAS[@]} -eq 0 ]]; then
  SIGMAS=(0.0 0.1 0.2 0.3)
fi

STEPS="${STEPS:-8000}"
# --no-grad-checkpointing works around this diffusers install's
# checkpointed-branch tuple bug (see gate_a_transport.py cmd_train's own
# comment) -- drop it here if your install doesn't need the workaround,
# or if VRAM headroom forces checkpointing back on for a wider sweep.
EXTRA_ARGS=(${EXTRA_ARGS:---no-grad-checkpointing})

echo "Launching ${#SIGMAS[@]} concurrent run(s): ${SIGMAS[*]}"
echo "cache=$CACHE steps=$STEPS extra=${EXTRA_ARGS[*]}"
echo

OUT_DIRS=()
for s in "${SIGMAS[@]}"; do
  out="runs/gate_a_sigma_${s}"
  OUT_DIRS+=("$out")
  mkdir -p "$out"
  log="${out}.stdout.log"
  echo "  sigma0=$s -> $out  (stdout -> $log)"
  python scripts/gate_a_transport.py train --cache "$CACHE" \
    --out "$out" --steps "$STEPS" --sigma0 "$s" "${EXTRA_ARGS[@]}" \
    > "$log" 2>&1 &
done

echo
echo "All launched. PIDs: $(jobs -p | tr '\n' ' ')"
echo
echo "Watch combined progress:"
echo "  python scripts/gate_a_transport.py watch-sweep --outs ${OUT_DIRS[*]} --every 100"
echo
echo "Run this from inside tmux/screen if you're going to disconnect --"
echo "whether background jobs survive a closed terminal/Ctrl-C here depends"
echo "on your shell's job-control settings, not guaranteed either way."
echo "To stop everything explicitly: kill \$(jobs -p) from this shell, or"
echo "pkill -f 'gate_a_transport.py train' from any shell on this box."
echo
echo "Waiting on all runs to finish..."
wait
