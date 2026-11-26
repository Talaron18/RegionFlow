#!/usr/bin/env bash
# Score white/oracle/sibling/freeform with the same VLM judge, now validated
# on `none` (GATE_C_REPORT.md SS5: rho=0.717, n=39). This is what turns SS9's
# quality claim from "40 audited none/oracle cases" into a full-151
# VLM-judged comparison across every arm.
#
# Reuses edit_quality_vlm.py's own --build/--score CLI unchanged -- it
# already does everything needed per arm (resumable, same judge, same crop
# fix, same prompts). This script is just the loop over arms so you don't
# retype the common flags four times and don't typo a path between runs.
#
# RUN FROM THE REPO ROOT (~/RegionFlow), same as run_gate_c.py always is --
# `python bench/edit_quality_vlm.py`, with runs/gate_c/... relative to the
# repo root, not to bench/. (First version of this script cd'd into bench/
# first, which broke every path -- fixed.)
#
# Paths below confirmed against your actual runs/gate_c/ (2026-09-17):
# white/oracle/sibling all used --save-images, freeform did not, so its
# edited images were never written to disk (run_gate_c.py's resume logic
# keys on the results file, not on whether images exist -- adding
# --save-images now without --overwrite would just skip every case as
# already-done and save nothing). freeform is skipped here until you
# decide whether to spend the GPU time regenerating it:
#
#   PYTHONPATH=/data/repos/UltraEdit/diffusers/src \
#   HF_HOME=/root/autodl-tmp/regionflow-data/hf HF_HUB_DISABLE_XET=1 \
#   python bench/run_gate_c.py --model sd3-ultraedit-freeform --mask-mode none \
#       --out runs/gate_c/ultraedit_freeform.jsonl --overwrite \
#       --save-images runs/gate_c/img_freeform
set -euo pipefail

if [[ ! -f bench/edit_quality_vlm.py ]]; then
  echo "run this from the repo root (~/RegionFlow), not from bench/" >&2
  exit 1
fi

declare -A RESULTS=(
  [white]="runs/gate_c/ue_white.jsonl"
  [oracle]="runs/gate_c/ue_oracle.jsonl"
  [sibling]="runs/gate_c/ue_sibling.jsonl"
)
declare -A EDITS=(
  [white]="runs/gate_c/img_white"
  [oracle]="runs/gate_c/img_oracle"
  [sibling]="runs/gate_c/img_sibling"
)

echo "SKIP freeform: runs/gate_c/img_freeform doesn't exist -- edited images" >&2
echo "  were never saved for that run. See the comment at the top of this" >&2
echo "  script for the regeneration command, once you decide it's worth it." >&2
echo >&2

for arm in white oracle sibling; do
  results="${RESULTS[$arm]}"
  edits="${EDITS[$arm]}"
  out="runs/gate_c/vlm_quality_${arm}.jsonl"

  echo "=== $arm ==="
  # --prompt-kinds full only, matching SS3's headline table. Add "single"
  # too if you want the ablation comparison at full VLM-judged scale.
  python bench/edit_quality_vlm.py --build \
    --judge-module edit_quality_vlm:siliconflow_judge \
    --results "$results" --edits "$edits" \
    --prompt-kinds full \
    --out "$out"
  python bench/edit_quality_vlm.py --score "$out"
  echo
done
