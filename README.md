# RegionFlow — graded region control for few-step image editing

## Staging

Stage 0 needs **no training**. Both fields are priors, so the whole tiering
claim is testable on SD3 with off-the-shelf inversion at 50 steps. Only after
it validates is the BiFM reimplementation worth the compute.

| stage | what | training |
|---|---|---|
| 0 | λ field + α field, 50-step RF inversion, r-sweep | none |
| 1 | feature-bleed split, α ablation | none |
| 2 | BiFM reimplementation → 1–4 steps | 32×H100-scale |
| 3 | LoRA with both fields present at train time | LoRA |

BiFM has no public code as of this writing (CVPR 2026 accepted, arXiv +
CVF only), so Stage 2 assumes reimplementation.

## Claim

Spatial control in image editing is currently implemented as inference-time
gradient optimisation, which costs minutes per image and cannot survive a
few-step sampling budget. Making it a **graded architectural prior** instead of
an iterative objective gives comparable localisation at 1–4 NFE.

Region tiering and few-step are not two contributions. Tiering is *what makes*
few-step localised editing possible: gradient guidance discovers spatial
structure over many iterations, a graded field injects it in one pass.

## Pipeline (v0)

```
source image ─┐
              ├─► BiFM inversion (no bias) ──► z
source prompt ┘

target mask ──► distance transform ──► area-pool to token grid ──► λ field
                                                                      │
                                                                      ▼
z ──► BiFM generation, λ-biased joint attention ──► edited image
      ▲
target prompt (edit tokens identified vs source prompt)
```

Three things deliberately cut from the original design:

- **MLLM edit planner** — returns in v1 as a single scalar (edit type → band
  radius), not a conditioning pathway.
- **Object parser geometry/scene heads** — unused by the mechanism.
- **Global harmonization stage** — MMDiT attention is global, so the reverse
  pass already harmonises. A separate stage would only undo local fidelity.

The reference-image branch (adding a subject) is **v2**. SD3 is text-only;
it needs encoder-based identity injection, and per-image fine-tuning of the
S²Edit kind is incompatible with a few-step budget.

## The field

`λ(p) = λ_out + (λ_in − λ_out) · f(d(p)/r)`

- `d` = outward distance from target boundary, at **image** resolution
- `f` = smoothstep, 1 → 0 across the band
- pooled to the token grid by **area** averaging, so partial coverage survives
- `r` in **tokens**, not pixels — one token is 16 image px at SD3's 8× VAE +
  2× patch, so a "few pixel" ring rounds to nothing

`r = 0` reduces to the binary in-mask/out-mask field, i.e. the MAG-Edit spatial
ratio constraint as a prior. Prior work is the null case of the sweep.

`λ_out < 0` matters: background edit-token attention is *suppressed*, not
merely un-boosted.

## Band semantics

The ring does three jobs that want different radii. Do not conflate them.

| job | radius | why |
|---|---|---|
| seam | r≈1 | edit shouldn't look pasted |
| physical spillover | r≈4, often asymmetric | shadow, contact, bounced light |
| shape headroom | r≈3 | new silhouette doesn't fit the old mask |

Only shape headroom is where we expect to *beat* rather than merely match the
optimisation baseline — it is the case (pose change) that attention-injection
methods report as a failure.

## Experiments

1. **r-sweep** — r ∈ {0, 1, 2, 4} on MAG-Bench and PIE-Bench. r=0 is the
   MAG-Edit-equivalent control. This sweep *is* the tiering claim.
2. **Speed/quality Pareto** — vs MAG-Edit at 1/4/50 NFE. Their stated first
   limitation is 1–5 min per image on an A100.
3. **Small-target bucket** — MAG-Bench is stratified by mask size. Targets
   under ~4 tokens have no room for a graded field; report separately and
   route to crop-and-zoom.
4. **Ablations** — preserve-token inversion on/off; bias on inversion pass
   on/off (expect: harmful); smoothstep vs linear vs hard.

Metrics come free from PIE-Bench: background preservation (LPIPS/SSIM/PSNR/MSE)
and CLIP semantics are reported separately, and that is exactly the axis λ
trades along.

## Training

Two routes, do both:

- **Architectural** — LoRA on MMDiT blocks + the time-interval embedding,
  MagicBrush triplets (which ship edit-region annotations).
- **Distillation** — run MAG-Edit as a slow teacher over MagicBrush/PIE-Bench
  sources to manufacture localised-edit targets; train the few-step student to
  reproduce them. Solves the data bottleneck and the speed claim in one move.

## Status

- [x] `region/lambda_field.py` — field construction, verified numerically
- [x] `region/attention.py` — joint-attention bias, indexing verified
- [x] `region/prompts.py` — positional token diff, repeated-word case verified
- [x] `region/masking.py` — instance selection, small-target routing verified
- [x] `region/pipeline.py` — dry-runs end to end without torch/SAM
- [ ] BiFM checkpoint / reimplementation
- [ ] raster-order check on `field` vs patch embedder  ← **first bug to hit**
- [ ] context-order check on token masks vs SD3 encoder concat ← **second**
- [x] `region/alpha_field.py` — inversion-depth field (pin logic sign-checked)
- [x] `bench/metrics.py` — leakage / bleed rates, selectivity bounded to [0,1]
- [x] `bench/sweep.py` — two-stage sweep + confound guard, smoke-tested
- [x] `bench/data.py` — PIE-Bench loader verified on fixtures; MAG-Bench tolerant + `inspect_schema`
- [ ] `bench/inversion.py` — SD3 RF inversion wrapper
- [ ] multi-instance annotations (needed for the leakage metric)

## Use PIE-Bench's own annotations, not our heuristics

Two loader findings supersede code written earlier:

* **Brackets over diffing.** PIE-Bench marks changed spans in the target
  prompt (`a slanted [rusty] mountain bicycle...`). That is a human annotation
  of the edit tokens. `region.prompts.build_token_masks` (SequenceMatcher) is
  now the fallback for datasets without brackets.
* **`editing_type_id` over keywords.** Ten annotated categories replace
  `infer_edit_type`'s wordlist, which already misclassified
  "black hair"→"blond hair" as `replace` in testing.

Style and background edits target the whole image, so the hierarchy degenerates.
`load_pie_bench(exclude_global=True)` drops them by default; report separately.

## Open decisions

1. **λ_in / λ_out magnitudes** are guesses (4.0 / −2.0). They are logit-space
   and interact with CFG scale. Needs a coarse 2D sweep before the r-sweep,
   or r will be confounded.
2. **Mask source.** SAM, or binarised cross-attention of the object word
   (annotation-free, S²Edit's trick)? The second is cheaper and makes the
   instance-disambiguation story harder to tell. Recommend SAM + user box for
   v0 so the instance claim stays clean.
3. **Which blocks get the bias.** All, or a depth range? Attention-control
   methods typically find early/mid blocks carry layout. Untested here.

## Known measurement caveat

Increasing `r` slightly grows the count of tokens at full `lam_in`, because
smoothstep is near 1 just outside the boundary. The core is therefore not held
exactly constant across the sweep. Report total field mass alongside r, or
threshold at 0.999, so the tiering effect is not confounded with a mild
dilation effect.
