# Modern-base evaluation on Gate C — scope

Status: **scope only; inference only; nothing run.** Thresholds are fixed here first.

**Question.** Does instance-level mis-targeting survive in a 2026 open editor, and
does such a base avoid the anatomy artifacts seen with SD3-UltraEdit? The answer
decides whether RegionFlow needs a method at all, or is a benchmark paper.

## Candidates for a 24 GB RTX 3090 Ti

Verified from the Hugging Face repos (file sizes, configs, cards) on 2026-11-13
unless marked otherwise.

| | repo | denoiser | text encoder | licence | mask input | 24 GB |
|---|---|---|---|---|---|---|
| **first** | `black-forest-labs/FLUX.2-klein-4B` | 4B, 7.8 GB; 5 double + 20 single-stream blocks; 32-channel VAE | Qwen3 LLM, ~8 GB | Apache 2.0, ungated | none documented (image + text) | yes: card states ~13 GB and names the RTX 3090; 4 steps |
| **second** | `Qwen/Qwen-Image-2.1` | 7B single-stream, 14.2 GB | ~17.5 GB | Qwen Research License (non-commercial), ungated | card: local edits "via circles, painted annotations, or separate masks" | by size only: encode text, free the encoder, then denoise. Not confirmed. |
| does not fit | `black-forest-labs/flux-3-image-klein` | 5.6B | Qwen3-VL-4B-Instruct, plus a required 9B prompt upsampler | FLUX Kommunity (non-commercial; commercial outputs under US$5M revenue), gated | bounding boxes written in the prompt; card: "not a hard mask" | **no** as documented: card gives peak 31 / 26 / 25 GB (BF16 / FP8 / NVFP4) at 1024² with the upsampler off, and the upsampler adds ~28 GB. Lower resolution untested. |
| too large | `Qwen/Qwen-Image-Edit-2511` | ~40 GB | ~16.6 GB | Apache 2.0 | none official | only with heavy quantisation |

Ranking evidence is weak: third-party leaderboard snapshots (not opened directly)
put Qwen-Image-2.1 top among open models in Sept 2026 and FLUX.2 klein 9B ≈
Qwen-Image-Edit-2511 in March 2026. klein 4B has no leaderboard number found. No
controlled hands/faces comparison was found for any of them.

## Prerequisites

- A **separate Python environment** with current `diffusers` (the klein card says
  install from git; configs are `0.37.0.dev0`). The project image pins the UltraEdit
  fork at `0.30.0.dev0`; do not upgrade that one.
- ~16 GB download for klein 4B into `HF_HOME`.
- `SILICONFLOW_API_KEY` for the judge.

## To build

1. ~~A `run_gate_c.py` model wrapper `flux2-klein-4b`~~ **written**: `bench/flux2_klein_model.py`,
   registered as `--model flux2-klein-4b` (`--klein-steps 4`, `--klein-guidance 1.0`,
   `--klein-offload`). Checked against a stub pipeline only; the real pipeline call is
   untested until the 4-case smoke run.
2. Optional second arm: a `qwen-image-2.1` wrapper with `--mask-mode none` and
   `oracle` (its card documents mask input; the exact argument must be read from
   its pipeline first).
3. `scripts/clear_edit.py` equivalent for these pipelines (red→blue circle).

## Run (once the wrapper exists)

```bash
python bench/run_gate_c.py --model flux2-klein-4b --mask-mode none --limit 4 \
  --out runs/gate_c/klein4b_smoke.jsonl --save-images runs/gate_c/img_klein4b_smoke
# look at the four images, then:
python bench/run_gate_c.py --model flux2-klein-4b --mask-mode none \
  --out runs/gate_c/klein4b_none.jsonl --save-images runs/gate_c/img_klein4b_none
SILICONFLOW_API_KEY=... python bench/edit_quality_vlm.py --build \
  --judge-module edit_quality_vlm:siliconflow_judge \
  --results runs/gate_c/klein4b_none.jsonl --edits runs/gate_c/img_klein4b_none \
  --prompt-kinds full --out runs/gate_c/vlm_quality_klein4b_none.jsonl
python bench/edit_quality_vlm.py --score runs/gate_c/vlm_quality_klein4b_none.jsonl
```

Then a 40-case blind audit on the same ids as the existing audit
(`bench/blind_audit.py`), with one added tally: outputs showing anatomy or
"AI-look" artifacts.

## How the result will be read (fixed in advance)

Reference points, same 151 cases, no mask: UltraEdit-freeform 40% mis-targeted,
RefEdit-SD3 28%. With oracle mask: 0% mis-targeted, 22% clean (9/40).

| mis-targeted, no mask | clean rate (blind audit, 40) | reading |
|---|---|---|
| ≤ 10% | ≥ 50% | Modern bases largely solve it without a mask. No method needed; the project is the benchmark plus this measurement. |
| ≤ 10% | < 50% | Targeting is solved by the base, edit quality is not. Any method work is about quality, not instances. |
| > 10% | any | Instance mis-targeting survives in a 2026 base. A mask/region mechanism on this base is a live method question (Qwen-Image-2.1's mask arm first, then a LoRA on klein 4B). |

Artifact tally: report as a count out of 40 next to UltraEdit-oracle's
`destroyed` (3/40) and `degraded` (12/40). No threshold; it is the first
measurement of this for the benchmark.

## Limits

One seed; the judge was calibrated on RefEdit-SD3 outputs only (ρ = 0.717) and may
score a different model family differently, so the blind audit carries the quality
claim; klein 4B is the smallest of its family, so a poor result does not speak for
larger models.

---

## Result — FLUX.2 klein 4B, no mask (2026-11-14, user's GPU run)

151 cases, full prompt, 4 steps, guidance 1.0, seed base 0. Judge: GLM via
SiliconFlow, 140 of 151 scored (11 empty API replies, to be retried).

| | klein 4B, no mask | RefEdit-SD3, no mask | UltraEdit freeform | UltraEdit oracle mask |
|---|---|---|---|---|
| mis-targeted | **34 (23%)** | 43 (28%) | 60 (40%) | 0 (0%) |
| no tracked change | **0** | 10 (7%) | 12 (8%) | 1 |
| off-instance edit | 9 (6%) | 0 | 0 | 1 |
| measurable | 108 (72%) | 98 (65%) | 79 (52%) | 149 (99%) |
| valid (K=10) | 21 (14%) | 38 (25%) | 1 | 16 (11%) |
| leakage ratio median (K=10) | 0.155 | 0.082 | 1.121 | 0.154 |
| sibling excess median (K=10) | +0.0166 | +0.0083 | +0.1634 | +0.0097 |
| judged `overall` median, all judged | 9.00 (n=140) | — | — | — |
| judged `overall` median, valid+measurable | 9.22 (n=18; 3 below 5) | — | — | 4.87 (n=16; 8 below 5) |

Reference floors: ground truth 0.011 / −0.005; category-level 1.001 / +0.157.

**Reading, by the rule fixed above: mis-targeted > 10%, so instance mis-targeting
survives in a 2026 base.** It is slightly below RefEdit-SD3's rate and well below
UltraEdit-freeform's, but nowhere near the 0% a mask gives.

Also in the numbers:
- The model always acts: zero no-change cases, against 7–8% for the old models.
- Judged quality is far higher than UltraEdit-oracle on the comparable subset
  (9.22 vs 4.87). The judge was calibrated on RefEdit-SD3 outputs only, so the
  blind audit must confirm this.
- Mis-targeted cases still score a judged median of 7.09 on the target region
  (it was 0.00 for RefEdit-SD3). Unverified inference: klein often edits the
  target *and* a sibling, i.e. category-level leakage, rather than only the wrong
  instance. Needs a look at the images.
- Sibling excess (+0.0166) is about twice RefEdit-SD3's and UltraEdit-oracle's.
- Only 14% of cases are valid at K=10, and 49 of 151 have an off-instance hot spot
  louder than the target: klein changes the rest of the frame more than a
  mask-conditioned editor does. Background preservation is a second open issue.

Not yet done: retry the 11 judge failures; the 40-case blind audit.
