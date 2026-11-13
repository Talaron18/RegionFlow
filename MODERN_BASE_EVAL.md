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
| unverified | `black-forest-labs/flux-3-image-klein` | 11.1 GB (6.3 / 5.3 GB quantised variants in repo) | ~8.9 GB | FLUX community licence, **gated** | unknown (card needs login) | likely by size |
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

1. A `run_gate_c.py` model wrapper `flux2-klein-4b` calling
   `Flux2KleinPipeline(prompt=instruction, image=source, num_inference_steps=4,
   guidance_scale=1.0)` at the benchmark's resolution, seeded with `case_seed`.
   Small. Cannot be tested without a GPU; first run it with `--limit 4` and look
   at the four images.
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
