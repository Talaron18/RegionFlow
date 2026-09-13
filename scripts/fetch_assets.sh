#!/usr/bin/env bash
# Fetch models, benchmarks and baseline repos into $DATA_DIR.
#
# Run INSIDE the container (paths assume /data is mounted):
#   bash scripts/fetch_assets.sh [stage0|stage2|all]
#
# Idempotent: re-running skips anything already present.

set -euo pipefail
DATA="${DATA_ROOT:-/data}"
STAGE="${1:-stage0}"

# --- preflight -------------------------------------------------------------
# The image chowns /data, but a bind mount shadows that layer and the HOST
# directory's ownership wins. This is the single most common first-run failure.
if [ ! -w "$DATA" ]; then
  cat <<EOF
!! $DATA is not writable by $(id -un) (uid $(id -u)).

   The bind mount overrides the image's ownership, so fix it ON THE HOST:

       sudo mkdir -p <your DATA_DIR>
       sudo chown -R \$(id -u):\$(id -g) <your DATA_DIR>

   Then check compose actually resolved DATA_DIR — compose v2 reads .env from
   the compose file's directory, so a .env at the repo root is ignored when
   you pass -f docker/compose.yml:

       docker compose -f docker/compose.yml config | grep -A3 volumes

   If that prints the literal default /mnt/data, either move .env into
   docker/ or pass --env-file .env explicitly.
EOF
  exit 1
fi

mkdir -p "$DATA"/{hf,torch,datasets,outputs,repos}

say() { printf '\n=== %s ===\n' "$1"; }
have() { [ -e "$1" ]; }

# The CLI was renamed: `huggingface-cli` is deprecated and now refuses to run.
# Prefer `hf`, fall back to the old name for older hub versions.
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "!! neither 'hf' nor 'huggingface-cli' found — pip install -U huggingface-hub"
  exit 1
fi
echo "using $HF_CLI"

# --- models ---------------------------------------------------------------
say "SD3-Medium (gated — accept the licence and set HF_TOKEN first)"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! HF_TOKEN unset. Accept the licence at"
  echo "   https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers"
  echo "   then create a read token and put it in docker/.env"
  exit 1
fi

# Check auth separately so a genuine licence/token problem is not confused
# with a CLI or network failure.
if ! $HF_CLI auth whoami >/dev/null 2>&1; then
  echo "!! token rejected by the hub. Check it is a valid READ token."
  exit 1
fi

# ~25 GB with all three text encoders. This is the bulk of the download.
# Resumable: re-run after an interruption and it picks up where it stopped.
#
# NOTE on syntax: the hf CLI treats bare positional args after the repo id as
# FILENAMES to fetch. Passing two patterns to one --exclude makes the second
# one a requested file, and you get "File not found in repository" for a glob.
# Each pattern needs its own flag.
#
# The exclusions drop the single-file checkpoints (sd3_medium.safetensors and
# the *_incl_clips* bundles), which are for ComfyUI and duplicate what the
# diffusers layout already provides. Without them the download roughly
# doubles.
$HF_CLI download stabilityai/stable-diffusion-3-medium-diffusers \
  --exclude "*.onnx" \
  --exclude "sd3_medium*" \
  --exclude "*incl_clips*" \
  || {
    echo "!! SD3 download failed."
    echo "   If the error mentions a 401/403, the licence has not been"
    echo "   accepted for this account: $($HF_CLI auth whoami 2>/dev/null | head -1)"
    echo "   If it mentions 'File not found', it is a CLI syntax problem here,"
    echo "   not a permissions problem."
    exit 1
  }

say "SAM ViT-H (2.4 GB) — only needed for the instance annotation pass"
# -c so an interrupted download resumes instead of restarting.
if ! have "$DATA/torch/sam_vit_h_4b8939.pth"; then
  wget -c -q --show-progress -O "$DATA/torch/sam_vit_h_4b8939.pth" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# ------------------------------------------------------------ benchmarks ----
say "PIE-Bench (700 images, 10 edit types, RLE masks)"
if ! have "$DATA/repos/PnPInversion"; then
  git clone --depth 1 https://github.com/cure-lab/PnPInversion.git \
    "$DATA/repos/PnPInversion"
fi
# The images may ship in-repo or via a separate link in that README. Check:
if ! have "$DATA/repos/PnPInversion/data/annotation_images"; then
  echo "!! annotation_images/ not present after clone."
  echo "   Read $DATA/repos/PnPInversion/README.md for the data download link."
  echo "   mapping_file.json alone is not enough — you need the images."
fi

say "MAG-Bench (200 images, stratified by mask size)"
if ! have "$DATA/repos/MAG-Edit"; then
  git clone --depth 1 https://github.com/HelenMao/MAG-Edit.git \
    "$DATA/repos/MAG-Edit"
fi
echo "NOTE: the paper describes MAG-Bench but a public release is not"
echo "      confirmed. If it is absent, PIE-Bench alone is enough for Stage 0"
echo "      — it has more cases and better annotations. Email the authors."

if [ "$STAGE" = "stage2" ] || [ "$STAGE" = "all" ]; then
  say "MagicBrush (10K editing triplets, for stages 2-3 only)"
  $HF_CLI download osunlp/MagicBrush --repo-type dataset \
    --local-dir "$DATA/datasets/MagicBrush" \
    || echo "!! MagicBrush is gated on HF; accept its terms first"
fi

# ------------------------------------------------------- metric checkpoints -
say "Metric weights (LPIPS + CLIP), pre-cached so the sweep does not stall"
python - <<'PY'
import lpips, open_clip, torch
lpips.LPIPS(net='alex')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
print("metric checkpoints cached")
PY

say "Done"
du -sh "$DATA"/* 2>/dev/null || true
