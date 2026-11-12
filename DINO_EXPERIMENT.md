# DINO perceptual-loss experiment — commands and kill criterion

Written **before** the run. Do not change the thresholds after seeing a number.

**Question.** Does adding a DINOv2 patch loss to Gate A's displacement regression
turn the blurry one-step edit into a clean one?

**Why a control arm is required.** The documented checkpoints (σ0 sweep, 0.541 /
0.769 / 0.664) were trained with `r = s`. Current code trains with interval
conditioning (`r > s`). A DINO run compared against the old checkpoints would
change two things at once. So train two arms that differ only in
`--perceptual-weight`.

## 0. Setup

Data root on the host is `/mnt/e/regionflow-data`. The scripts default to `/data`.

```bash
# Docker (docker/compose.yml mounts ${DATA_DIR} at /data and sets HF_HOME=/data/hf)
export DATA_DIR=/mnt/e/regionflow-data
docker compose -f docker/compose.yml run --rm dev bash

# Without Docker: symlink once, or replace /data with /mnt/e/regionflow-data below
sudo ln -s /mnt/e/regionflow-data /data
export HF_HOME=/data/hf TORCH_HOME=/data/torch
```

DINOv2 (`facebook/dinov2-base`) downloads into `HF_HOME` on first use.

```bash
ulimit -n 8192
git log -1 --format=%h                                   # record the commit
python transport/pairflow.py --self-test                 # must print OK
python scripts/compare_judged_arms.py --self-test        # must print OK
ls /data/cache/13061d51aee4 | head -3                    # the 4,511-item cache
```

## 1. Train both arms (same seed, same flags, one difference)

```bash
COMMON="--cache /data/cache/13061d51aee4 --steps 8000 --sigma0 0.0 \
  --mask-encoder-kind single --interval-conditioning \
  --batch 2 --accum 8 --lr 1e-4 --rank 16 --seed 0"

# control: MSE only
python scripts/gate_a_transport.py train $COMMON \
  --perceptual-weight 0.0 --out runs/gate_a_ctrl

# treatment: MSE + 0.1 * DINOv2 patch loss
python scripts/gate_a_transport.py train $COMMON \
  --perceptual-weight 0.1 --perceptual-feature dinov2 --perceptual-mask-erode 1 \
  --out runs/gate_a_dino
```

Notes:
- `--batch 2 --accum 8` keeps the effective batch at 16. The DINO arm decodes
  through the VAE in fp32 with gradients, which the default batch 4 may not fit
  in 24 GB. This is untested: if it still runs out of memory use
  `--batch 1 --accum 16` **for both arms**.
- If the first step dies with `layer_norm(): argument 'input' ... must be
  Tensor, not tuple`, add `--no-grad-checkpointing` to both arms (known bug in
  the vendored diffusers).
- Add `--max-hours 5.5` on a time-limited box; rerunning the same command resumes.
- Early abort (not a verdict): `loss_percept` is `nan`, or not lower at step
  1000 than at step 100 in `runs/gate_a_dino/log.jsonl`. That means the loss is
  not wired or the weight is too small; fix and restart, do not evaluate.

## 2. Evaluate

### K1 — the synthetic unambiguous edit (minutes)

```bash
for arm in ctrl dino; do
  python scripts/clear_edit.py --ckpt runs/gate_a_$arm/ckpt_final.pt \
    --r 1.0 --out runs/gate_a_$arm/clear_edit
done
```

`--r 1.0` because both arms are interval-trained and Gate C reads out with
`r = 1`. Each run writes `synthetic_result.png` and `clear_edit.json`.
Calibration on the committed panel: target and oracle give `blue_ok: true`
(oracle mean RGB 32/81/216); the current σ0=0 model gives `blue_ok: false`
(164/152/128).

### K2 — Gate C, oracle mask, 151 cases, calibrated judge

```bash
for arm in ctrl dino; do
  python bench/run_gate_c.py --model gate-a-transport \
    --gate-a-ckpt runs/gate_a_$arm/ckpt_final.pt --gate-a-sigma0 0.0 \
    --gate-a-mask-encoder-kind single --mask-mode oracle \
    --out runs/gate_c/gate_a_${arm}_oracle.jsonl \
    --save-images runs/gate_c/img_gate_a_$arm --overwrite

  SILICONFLOW_API_KEY=... python bench/edit_quality_vlm.py --build \
    --judge-module edit_quality_vlm:siliconflow_judge \
    --results runs/gate_c/gate_a_${arm}_oracle.jsonl \
    --edits runs/gate_c/img_gate_a_$arm --prompt-kinds full \
    --out runs/gate_c/vlm_quality_gate_a_$arm.jsonl
  python bench/edit_quality_vlm.py --score runs/gate_c/vlm_quality_gate_a_$arm.jsonl
done

python scripts/compare_judged_arms.py \
  --treat runs/gate_c/vlm_quality_gate_a_dino.jsonl \
  --control runs/gate_c/vlm_quality_gate_a_ctrl.jsonl
```

### Look at the images (required before recording any number)

Open `runs/gate_a_dino/clear_edit/synthetic_result.png` and at least 20 panels
from `runs/gate_c/img_gate_a_dino` next to the same ids in `img_gate_a_ctrl`.

## 3. Kill criterion

The DINO arm **passes only if all three hold**:

| | check | pass |
|---|---|---|
| K1 | `runs/gate_a_dino/clear_edit/clear_edit.json` → `model.blue_ok` | `true` (in-mask mean colour is blue-dominant and nearer the target fill than the source fill) |
| K2 | `compare_judged_arms.py` on all paired full-prompt cases | `pass: true`: more wins than losses, exact two-sided sign test p < 0.05, **and** median `overall` gain ≥ 1.0 (0–10 scale) |
| K3 | the 20 panels, by eye | DINO outputs are not visibly worse than control (no new texture hallucination, no destroyed objects) |

**If it fails:** the displacement-regression line (one-step `z_src + u·M` on
frozen SD3) is closed. No further loss variants, weights, extractors,
σ0 values or diagnostics on it. Next experiment is the noise-start,
source+mask-conditioned control in `STATUS_REVIEW.md` §4.3.

**If it passes:** repeat the DINO arm with `--seed 1` and re-check K2 before
quoting anything, then run the 40-case blind audit against UltraEdit-oracle
(22% clean; judge median 4.87 among its valid+measurable cases) as the bar
for a publishable claim.

**Not part of the verdict** (record, do not decide on them):
- `diagnose` cosines. A perceptual loss can lower latent cosine while improving
  the picture, and the reverse.
- `scripts/feature_discrimination.py` DINO pred/oracle ratio (was 30.1). The
  DINO arm is trained on this quantity, so it will improve by construction.
- `control` vs the old σ0=0 checkpoint. Useful to learn what interval
  conditioning did, but it is a different question.

**Known limits of this design:** one seed per arm; the judge was calibrated on
`none`-arm outputs only (ρ = 0.717); K1 is a single out-of-distribution image.
