# UltraEdit-base experiment — scope

Status: **scope only.** Stage 0 uses code that already exists. Stage 1 needs new
training code that is not written yet. Thresholds below are fixed before any run.

## Why this base

Gate A asked a LoRA to turn SD3 (a noise→image generator) into a one-step editor.
Both arms ended up ignoring the instruction (`STATUS_REVIEW.md` §4b) and failed the
synthetic colour test (`DINO_EXPERIMENT.md`). `BleachNick/SD3_UltraEdit_w_mask` is
the same SD3 MMDiT already trained to edit from noise with a source image and a
mask as input (`in_channels=48`). On Gate C with the oracle mask it scores 0%
mis-targeted, 22% clean by blind audit (9/40), and a judged median of 4.87 among its
valid+measurable cases. So the open gap is edit quality, not localisation or
text-following from scratch.

**Question:** can a small LoRA on a base that already edits raise edit quality on
instance-level cases, where the same LoRA budget on plain SD3 could not produce an
edit at all?

## Prerequisites on the training machine

- UltraEdit's diffusers fork (provides `StableDiffusion3InstructPix2PixPipeline` and
  the `mask_img` argument). `bench/run_gate_c.py` expects it on `PYTHONPATH`, e.g.
  `/data/repos/UltraEdit/diffusers/src`. It is not in the current `/data/repos`
  listing (only `MAG-Edit`, `PnPInversion`).
- The checkpoint `BleachNick/SD3_UltraEdit_w_mask` in `HF_HOME`.
- `SILICONFLOW_API_KEY` for the judge.

## Stage 0 — no training (decides whether Stage 1 is worth building)

**0a. Reproduce the baseline on this machine.** Different boxes disagreed by more
than measured effects before (STAGE1C §5), so the baseline must be local.

```bash
export PYTHONPATH=/data/repos/UltraEdit/diffusers/src
python bench/run_gate_c.py --model sd3-ultraedit-mask --mask-mode oracle \
  --out runs/gate_c/ue_oracle_local.jsonl \
  --save-images runs/gate_c/img_ue_oracle_local --overwrite
SILICONFLOW_API_KEY=... python bench/edit_quality_vlm.py --build \
  --judge-module edit_quality_vlm:siliconflow_judge \
  --results runs/gate_c/ue_oracle_local.jsonl \
  --edits runs/gate_c/img_ue_oracle_local --prompt-kinds full \
  --out runs/gate_c/vlm_quality_ue_oracle_local.jsonl
python bench/edit_quality_vlm.py --score runs/gate_c/vlm_quality_ue_oracle_local.jsonl
```

Pass: 0–2 of 151 mis-targeted (recorded: 0). If not, stop and find the difference.

**0b. Premise checks on the base** (to build, small; see "To build"):
- Synthetic red→blue circle through UltraEdit with the circle mask: `blue_ok` true.
- Instruction swap on ~50 dev cases, same seed: the output must change with the
  instruction (the check Gate A failed).

If either fails, the base does not have the property this experiment relies on. Stop.

**0c. Is the gap an inference-setting problem?** The Gate C settings (50 steps,
guidance 7.5, image guidance 1.5, dilate 8) were frozen from a 4-case smoke run, and
25% of audited oracle outputs were "no edit". Sweep guidance ∈ {5, 7.5, 10} ×
image guidance ∈ {1.0, 1.5, 2.0} on a **dev set only** (`bench/cases_dev.json`,
MagicBrush dev × COCO), never on the 151. If the best dev setting beats the frozen
one by judged median ≥ 1.0, re-score the 151 once at that setting; that becomes the
baseline Stage 1 must beat, and it may make Stage 1 unnecessary.

## Stage 1 — LoRA on the UltraEdit base (not yet built)

- **Objective:** the base's own training objective, conditional flow matching from
  noise: `z_t = (1−t)·z_tgt + t·ε`, predict `ε − z_tgt`, with source latent and mask
  fed through the base's existing 48-channel input. No displacement regression, no
  one-step readout.
- **Trainable:** rank-16 LoRA on attention projections (reuse `transport/gating.py`
  with gate ≡ 1). Nothing else.
- **Data:** the existing MagicBrush turn-1 cache (4,511), **minus any item whose COCO
  image appears in the 151 Gate C cases** (both draw from COCO train2014; the overlap
  has not been counted yet).
- **Arms:** base with no LoRA (control, from Stage 0) vs base + LoRA. Same seeds,
  same inference settings.
- **Budget:** 4,000 steps, effective batch 16, one seed first.

### Kill criterion (all must hold)

| | check | pass |
|---|---|---|
| K1 | synthetic red→blue, LoRA arm | `blue_ok` true (the base passes in 0b; the LoRA must not break it) |
| K2 | 151 cases, oracle mask, judged, LoRA vs base paired (`compare_judged_arms.py`) | more wins than losses, sign test p < 0.05, median `overall` gain ≥ 1.0 |
| K3 | mis-targeting, LoRA arm | ≤ 2 of 151 (localisation must not regress) |
| K4 | instruction swap, LoRA arm | output still changes with the instruction |

Fail → a LoRA on 4.5k MagicBrush pairs does not move instance-level edit quality on
this base. Do not respond with more data variants; write up Gate C plus the negative
results. Pass → second seed, then the 40-case blind audit against the base's 22%.

## To build (none of this exists yet)

1. `scripts/ultraedit_probe.py`: synthetic colour test and instruction-swap check
   through the UltraEdit pipeline (Stage 0b). Small.
2. Inference-setting sweep on the dev cases (Stage 0c): a loop over
   `run_gate_c.py --cases bench/cases_dev.json`. Small; needs the dev cases checked
   for the fields `run_gate_c.py` expects.
3. `scripts/ultraedit_lora.py`: training loop. Medium. **Read the fork's pipeline
   first** to copy exactly how the mask becomes 16 channels and how the three blocks
   are ordered and scaled; a mismatch there trains on an input the base never saw.
4. COCO-id overlap filter between the MagicBrush cache and the 151 cases. Small.
5. A `run_gate_c.py` model wrapper that loads the base plus LoRA weights. Small.

## Known limits

One seed per arm at first; the judge was calibrated on `none`-arm outputs only
(ρ = 0.717); the 22% clean figure has a wide interval ([12%, 38%], n = 40), so only
the paired judged comparison can detect a moderate gain.
