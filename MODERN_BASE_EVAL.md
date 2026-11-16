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

**Judge, complete (151/151, `runs/gate_c/vlm_quality_klein4b_none.jsonl`):** `overall`
median 9.00, mean 7.00; semantic-consistency score ≥ 8 in 121 of 151; `overall` < 5 in
36. By outcome: measurable 9.22 (n=108), mis-targeted 6.20 (n=34), off-instance 3.00
(n=9). Valid+measurable: 9.22 (n=20, 3 below 5).

**Blind audit (40 klein cases, single arm, ids NOT the earlier audit's 40):**

| arm | clean | degraded | destroyed | wrong_edit | no_edit | clean % |
|---|---|---|---|---|---|---|
| klein 4B, no mask | 9 | 1 | 0 | **28** | 2 | 22% |
| UltraEdit oracle (earlier audit, other ids) | 9 | 12 | 3 | 6 | 10 | 22% |

- Artifacts ("AI-look"): degraded + destroyed = **1 of 40**, against 15 of 40 for
  UltraEdit-oracle. Different ids, so indicative, not paired.
- Clean rate is 22%, the same as UltraEdit-oracle's and under the 50% line.
- The failure mode moved: 28 of 40 are `wrong_edit`.
- **The judge and the human disagree sharply.** The judge gives a high semantic
  score to 80% of cases; the audit calls 22% clean. Until the two are cross-tabulated
  per case, the judged 9.0 must not be quoted as klein's quality. Open: what the 28
  `wrong_edit` cases are (leakage onto siblings, which the target-crop judge cannot
  see, or a genuinely different edit), and whether the audit sample over-represents
  mis-targeted cases (it is stratified by outcome).


### Cross-tabulation of the audit (ledger + key pushed by the user, 2026-11-14)

The audit sample is stratified, not random: 16 measurable / 15 mis-targeted / 9
off-instance, against 108 / 34 / 9 in the full run.

| human verdict | measurable | mis-targeted | off-instance | judge `overall` median | judge sc ≥ 8 |
|---|---|---|---|---|---|
| clean (9) | 6 | 2 | 1 | 8.05 | 7 of 9 |
| degraded (1) | 1 | 0 | 0 | 5.48 | 1 of 1 |
| wrong_edit (28) | 9 | 13 | 6 | 8.29 | 18 of 28 |
| no_edit (2) | 0 | 0 | 2 | 4.87 | 1 of 2 |

- **The judge does not separate clean from wrong_edit** (8.05 vs 8.29). It scores the
  target crop only, so it cannot see an edit that also landed on siblings. For klein it
  measures "was the target edited well", not "was only the target edited".
- **Rater's description of the 28 wrong edits:** almost all also edited untargeted
  siblings; 4 were a different edit on the target itself (whole cake removed instead of
  half; a cat added beside the dog instead of replacing it; two parrots instead of one).
  This split is the rater's summary, not a per-case label.
- **The metric's `mis_targeted` undercounts leakage.** It fires only when a sibling
  changed MORE than the target. In the `measurable` stratum, where the target changed
  most, 9 of 16 were still judged wrong_edit.
- **Weighted to the full 151:** clean ≈ 30% (6/16, 2/15, 1/9 by stratum); wrong_edit
  ≈ 64%. If about 24 of 28 wrong edits are sibling leakage, roughly half of all klein
  edits on this benchmark leak onto a sibling. One rater, small strata: an estimate,
  not a measurement.

Next, before any method work: (1) tag each wrong_edit in the ledger's `notes` as `leak`
or `true` so the split is countable; (2) score the trivial baseline, klein's output
composited into the source through the oracle mask, which removes sibling leakage by
construction and shows what a mask alone buys on this base.

## Composite baseline — scope (written before any run)

`--klein-composite` edits with no mask, exactly as the plain arm (same call, same
seed), then pastes the result into the source through the oracle mask
(`--mask-dilate 8`, `--klein-feather 4`, both frozen here). Outside the mask the
output is the source, so sibling leakage is zero by construction.

```bash
python bench/run_gate_c.py --model flux2-klein-4b --klein-composite \
  --mask-mode oracle --out runs/gate_c/klein4b_comp_oracle.jsonl \
  --save-images runs/gate_c/img_klein4b_comp_oracle
```

Audit the composite arm on the **same 40 ids** as the klein audit. `--carry-over`
reads any JSON list of records with an `id`, so the klein key file serves directly:

```bash
python bench/blind_audit.py --build \
  --arm comp:runs/gate_c/klein4b_comp_oracle.jsonl:runs/gate_c/img_klein4b_comp_oracle \
  --reference comp --carry-over runs/gate_c/blind_klein_key.json --extra 0 \
  --out-dir runs/gate_c/blind_comp \
  --ledger runs/gate_c/blind_comp_ledger.json --key runs/gate_c/blind_comp_key.json
python bench/blind_audit.py --score runs/gate_c/blind_comp_ledger.json \
  --key runs/gate_c/blind_comp_key.json
```

One arm, 40 panels. It is not blind to the arm (the rater knows these are paste-backs),
but it is paired with the klein verdicts case by case. Not run here: `--extra 0` with
a carry-over is untested.

How the composite arm's clean rate (of 40) will be read:

| composite clean | reading |
|---|---|
| ≥ 70% | A mask plus paste-back is enough on this base. No learned region mechanism is needed; what remains is obtaining the mask (referring segmentation) and the cases below. |
| 40–70% | The mask removes leakage but something else fails often. Classify the failures before choosing a method. |
| < 40% | Paste-back is not enough; conditioning the model on the region is a live method question. |

Record separately, as they are what paste-back cannot fix: visible seams or lighting
mismatch at the mask edge; edits that need to extend past the mask (added objects,
shadows, pose changes); removals, where the pasted region still shows the model's
whole-image re-render; and the 4 true wrong edits, which a mask does not address.

### Composite baseline — result (2026-11-15, user's run and audit, same 40 ids)

| arm | clean | degraded | destroyed | wrong_edit | no_edit | clean % |
|---|---|---|---|---|---|---|
| klein 4B, no mask | 9 | 1 | 0 | 28 | 2 | 22% |
| klein 4B + paste-back through the oracle mask | **19** | 4 | 0 | 4 | **13** | **48%** |

Metric side: 39 of 40 audited cases are now valid+measurable (was 5), as expected when
everything outside the mask is the source.

**Reading, by the rule fixed above: 48% is in the 40–70% band.** The mask removes
leakage (wrong_edit 28 → 4) and more than doubles the clean rate, but paste-back
alone is not enough.

What is left, 21 of 40 (totals only; per-case pairing needs the comp ledger and key):
- 13 `no_edit`. Expected mechanism, not yet checked per case: where klein edited
  only a sibling, the target region it pastes back is unchanged. A mask applied
  AFTER generation cannot redirect the edit; the model has to be told where.
- 4 `wrong_edit`, the same count as the rater's "true wrong edits" in the plain arm.
- 4 `degraded`, the paste-back's own cost (seams or mismatch at the edge), up from 1.

The sample is stratified (16 measurable / 15 mis-targeted / 9 off-instance), so 48% is
not a population rate; the weighted figure needs the per-case files.

### Composite baseline — paired with the plain arm (ledgers pushed 2026-11-15)

Plain klein verdict (rows) → paste-back verdict (columns), same 40 ids:

| plain \ paste-back | clean | degraded | wrong_edit | no_edit |
|---|---|---|---|---|
| clean (9) | 6 | 1 | 1 | 1 |
| degraded (1) | 1 | 0 | 0 | 0 |
| wrong_edit (28) | 12 | 3 | 3 | 10 |
| no_edit (2) | 0 | 0 | 0 | 2 |

13 cases became clean, 3 stopped being clean (exact two-sided sign test p = 0.021).

Paste-back clean rate by the plain run's metric class:

| plain run's class | share of all 151 | paste-back clean | plain clean |
|---|---|---|---|
| measurable (target changed most) | 108 (72%) | 13 / 16 | 6 / 16 |
| mis-targeted | 34 (23%) | 6 / 15 | 2 / 15 |
| off-instance | 9 (6%) | 0 / 9 | 1 / 9 |

Weighted to all 151: paste-back ≈ **67%** clean, plain ≈ 30%. Still inside the
40–70% band, just under its upper edge; small strata, one rater.

- **Where the model already edits the target, a mask plus paste-back is nearly
  enough** (13 of 16).
- **All 13 `no_edit` cases come from the two classes where the plain edit landed
  elsewhere** (6 mis-targeted, 7 off-instance). Confirmed: a mask applied after
  generation cannot move the edit.
- **Some of those are a mask-coverage limit, not a model failure.** Several are
  additions whose correct location is beside the referenced instance ("add a wine
  glass in front of the woman", "add a child standing beside the rear of a car",
  "add a bird to the bench next to..."). The oracle mask is the referenced instance,
  so paste-back deletes a correctly placed addition. By instruction text about 3–4 of
  the 13; not verified against the images.
- The 4 `wrong_edit` cases are 3 carried over (cake, dog→cat, laptop wallpaper) plus
  one that was clean in the plain arm (bird → birdhouse, noted "leak and false").

**What a method now has to beat:** ≈ 67% clean overall, 81% where the base already
targets correctly. The residual is the ≈ 28% of cases where the base edits the wrong
place, plus additions that belong outside the instance mask.

**Next baseline before any training — crop, edit, paste.** Crop to the target's
box with a margin so siblings are out of view, edit the crop with the benchmark's
single-instance instruction (`edit_instruction_single`), paste back. It tells the
model where without changing the model. If it clears the residual, the result is a
pipeline (referring segmentation → crop → edit → paste), not a learned mechanism.

## Crop-edit-paste baseline — scope (written before any run)

`--klein-crop`: crop the source to the oracle mask's box plus a 25% margin (min 16 px),
upsample the crop to about 512² for the model, edit it, paste it back through the mask
(`--klein-crop-paste mask`, feather 4). Margin, feather and dilation are frozen here.
Run with `--ablation` so both the full referring instruction and the benchmark's
single-instance instruction are produced; **the single instruction is the primary
arm**, because the full one refers to context the crop no longer shows.

```bash
python bench/run_gate_c.py --model flux2-klein-4b --klein-crop --ablation \
  --mask-mode oracle --out runs/gate_c/klein4b_crop_oracle.jsonl \
  --save-images runs/gate_c/img_klein4b_crop_oracle

python bench/blind_audit.py --build \
  --arm crop:runs/gate_c/klein4b_crop_oracle.jsonl:runs/gate_c/img_klein4b_crop_oracle \
  --reference crop --prompt-kind single \
  --carry-over runs/gate_c/blind_klein_key.json --extra 0 \
  --out-dir runs/gate_c/blind_crop \
  --ledger runs/gate_c/blind_crop_ledger.json --key runs/gate_c/blind_crop_key.json
```

Same 40 ids as the two earlier audits. Not checked here: that `--save-images` keeps
the `single` and `full` outputs apart, and that the audit finds the `single` images.
Look at the first few panels against the instruction before judging all 40.

Reading of the crop arm's clean count (of 40; paste-back alone was 19):

| crop clean | reading |
|---|---|
| ≥ 30 | Cropping tells the model where well enough. The practical answer is a pipeline (find the object → crop → edit → paste). A learned region mechanism would need a different justification (speed, edits needing scene context, additions outside the box). |
| 24–29 | A real gain over paste-back, with a residual. Classify it before deciding. |
| ≤ 23 | No meaningful gain over paste-back. Telling the model where needs to happen inside the model: the case for a region-conditioned adapter. |

Record separately: cases that fail *because* context was cropped away (lighting or
scale mismatch after paste, instructions that need the scene), and additions that
belong outside the mask (re-check those with `--klein-crop-paste box`).

### Crop baseline — automated result (2026-11-16, user's run, before the dilation fix)

`flux2-klein-4b+crop-mask`, oracle mask, 151 cases × {full, single} prompt:

| | full | single |
|---|---|---|
| mis-targeted | 1 (refedit-27) | 1 (refedit-27) |
| measurable | 150 | 150 |
| valid (K=10) | 146 (97%) | 147 (97%) |
| leakage ratio median | 0.002 | 0.002 |
| sibling excess median | −0.0009 | −0.0008 |

These are at the ground-truth floor (0.011 / −0.005) **by construction**: everything
outside the mask is the source. As with UltraEdit's oracle arm, the where-metrics are
saturated and say nothing about whether the edit is right. Full vs single prompt shows
no difference on them (sign test p = 0.74). Only the blind audit can score this arm.

**Audit panels built (2026-11-16).** `runs/gate_c/blind_crop/` has 40
panels on the same 40 ids; the key's metrics match the `single` records in 40 of 40 and
the `single` and `full` images differ in 38 of 40 (the other two, refedit-14 and -156,
have identical prompts). `blind_audit.py` now filters each arm to `--prompt-kind`;
before, a two-prompt file was keyed by id and the last line won, which happened to be
`single`. The `READING` footer that `--score` prints is about RefEdit-SD3 and does not
apply; read the count against the table above.

### Crop baseline — audit result (2026-11-16, user's audit, same 40 ids, single prompt)

> **Read the correction further down first.** In 12 of these 40 cases the crop box was
> enlarged by a dilation bug, in 4 of them to the whole image. The counts below are as
> observed; the reading is on hold until the rerun.

| arm | clean | degraded | destroyed | wrong_edit | no_edit | clean % |
|---|---|---|---|---|---|---|
| klein 4B, no mask | 9 | 1 | 0 | 28 | 2 | 22% |
| + paste-back through the oracle mask | 19 | 4 | 0 | 4 | 13 | 48% |
| + crop, edit, paste | **24** | 3 | **4** | 3 | 6 | **60%** |

**Reading, by the rule fixed above: 24 is in the 24–29 band, on its lower edge.** One
verdict fewer would be the "no meaningful gain" band. The band stands as the reading: a
gain over paste-back, with a residual to classify. Context for how large the gain is:

- Paired with paste-back on the same ids: 7 became clean, 2 stopped being clean
  (exact two-sided sign test p = 0.18). Against the plain arm: 17 and 2 (p = 0.001).
- Weighted to all 151 by the plain run's class, crop ≈ **65%** clean and paste-back
  ≈ 67%. This near-tie turns on two cases (refedit-154 and -84, the whole difference
  in the `measurable` row below, each worth about 4.5 points); it does not override
  the count.
- The two arms win on different cases:

| plain run's class | share of all 151 | plain clean | paste-back clean | crop clean |
|---|---|---|---|---|
| measurable (target changed most) | 108 (72%) | 6 / 16 | 13 / 16 | 11 / 16 |
| mis-targeted | 34 (23%) | 2 / 15 | 6 / 15 | 9 / 15 |
| off-instance | 9 (6%) | 1 / 9 | 0 / 9 | 4 / 9 |

Where the base edits the wrong place, cropping helps (13 of 24 against 6 of 24). Where
the base already aims correctly, the two are not distinguishable (11 against 13 of 16,
two cases). Crop has 4 `destroyed` verdicts where the two earlier arms had none.

**The 13 cases paste-back left unedited** (the ones a mask after generation cannot
fix): crop made 4 clean (refedit-190, -14, -130, -176), 6 are still `no_edit`, and 3
became wrong_edit / destroyed / degraded. Telling the model where by cropping fixes
under a third of them.

**Residual, 16 of 40.** Failure type from the rater's notes where given (quoted), and
otherwise from the panels, opened after judging was finished (seven panels: the four
unnoted destroyed/degraded and three `no_edit`). Type only; no verdict was changed.

- **The instance mask clips the edit, 6 cases.** Three `no_edit` additions whose place
  is beside the instance (refedit-153 child beside the car, -39 wine glass in front of
  the woman, -172 bird on the bench; by instruction text, the same three flagged under
  paste-back). Three seen in the panels: refedit-29, the added parrot sits on the
  shoulder outside the chef's mask and only a sliver survives the paste; -154, the
  bear is re-dressed inside the mask while the veil outside it stays and the cape is
  cut at the edge; -57, the jug is removed but the blender's base, outside the mask,
  remains. All six are candidates for the `--klein-crop-paste box` re-check.
- **Old and new object mixed, 2.** refedit-183 (half backpack, half basketball) and
  -72 ("this is a cat-dog").
- **Scale or placement, 2.** refedit-113 ("this cat is big as a cow") and -151
  ("floating birdhouse"). These are the only two that look like lost scene context.
- **Edit not made, 3 `no_edit`.** refedit-184 (hair bow; small target, no visible
  change), -156 (remove the apple; the object fills the crop and is still there),
  -199 (goatee). On -199 the goatee looks absent in the edited half of `panel_0015`
  at panel resolution; worth a second look by the rater.
- **A different edit, 3 `wrong_edit`.** refedit-117; -84 (suit recoloured along with
  the shirt); -150 (half the cake, wrong in all three arms).

Clean in at least one of the two mask baselines: 26 of 40. Clean in neither: 14.

**What this means for the method question (WITHDRAWN 2026-11-16, see the correction
below; kept for the record).** A fixed pipeline (mask, then paste-back
or crop) reaches roughly two thirds clean on this benchmark either way. The largest
single residual is not the model but the paste: an instance mask is the wrong shape
for additions and for edits that change the object's outline. Lost scene context
accounts for two cases, not most of them.

Limits: one rater; 40 stratified cases with strata as small as 9, so per-class rows are
estimates. This audit was not blind to the arm and was the rater's third pass over the
same 40 ids; `destroyed` appears four times here after none in the two earlier
sittings, so a shift in how the rubric was applied cannot be excluded.

### Correction (2026-11-16): the crop was not a crop in 54 of 151 cases

Found while checking the box-paste run against its images. `run_gate_c.dilate` grew the
mask with `np.roll`, which wraps round the image. A target within 8 px of a border
reappeared as a strip on the opposite border; the crop takes the mask's bounding box,
so the box stretched across the image.

Evidence, all from files on disk:
- Predicting each case's crop box with the wrap explains every changed pixel of the
  box-paste images in 151 of 151 cases; predicting it without the wrap leaves changes
  outside the box in 52.
- In the 22 cases where the wrapped box is the whole image, the crop arm's `full`
  output is pixel-identical to the paste-back arm's: same input, prompt and seed.

| | of 151 | of the 40 audited |
|---|---|---|
| mask reaches a border (a strip of up to 8 px on the opposite side, every dilated arm) | 67 | 16 |
| crop box enlarged | 54 | 12 |
| crop box = the whole image (intended in 3) | 22 | 4 |

In the enlarged cases the box covered a median 81% of the image where 38% was
intended. Ids and box sizes: `runs/gate_c/crop_wrap_affected.json`.

**What this does to the results above.**
- **Crop audit, 24 of 40.** The count stands as observed, but in 12 of the 40 the arm
  as run was not the arm as specified, so its reading against the 24–29 band is **on
  hold** until the rerun. Not known: which way the count moves.
- **Residual classification.** Five of the 16 non-clean cases are among the 12
  (refedit-184, -153, -39 `no_edit`; -183 `destroyed`; -113 `degraded`). For -184 and
  -39 the model was given the whole image. Their failure types are unexplained until
  rerun.
- **Withdrawn:** "the largest single residual is not the model but the paste" and
  "lost scene context accounts for two cases". Both leaned on affected cases.
- **Crop automated table.** Where-metrics under mask paste are saturated by
  construction, so those numbers barely depend on the box; they will be re-recorded
  with the rerun.
- **Paste-back and UltraEdit-oracle arms.** Same dilation, so in the 67 border cases
  their masks carried the strip of up to 8 px on the opposite border. Small next to
  what those arms measure; not re-run. The frozen metric's own `off_instance._dilate`
  wraps the same way and has been left untouched: it removes that strip from the
  off-instance area. Changing it is a separate decision.
- **Not affected:** the plain arm (no mask), and the grounding probe, its join with
  the audits and the feature check, which use the undilated masks.

**Fix.** `dilate` now pads before rolling. Its self-test checks that an interior mask
is bit-identical to the old function and that refedit-184's box no longer reaches the
far borders (bounding box rows 332–511, columns 0–118; before, the whole image).
`python bench/run_gate_c.py --self-test` passes on CPU. Not run: any model.

### Crop with box paste — automated result (2026-11-16, user's run, BEFORE the fix)

`flux2-klein-4b+crop-box`, 151 cases, single prompt, split by whether the wrap enlarged
the crop box:

| | box enlarged (54) | box as intended (97) |
|---|---|---|
| mis-targeted | 11 | 2 |
| valid (K=10) | 12 | 62 |
| sibling excess > 0.02 (cut-off picked here) | 19 | 25 |

Most of the run's 13 mis-targeted cases come from the enlarged boxes, where siblings
were inside the crop. In the 97 cases run as intended, box paste still shows sibling
change in about a quarter; that mixes real edits with the crop's resample round trip,
and the split between the two is not known.

Seen in the images, for the record and not as verdicts: with box paste a whole parrot
appears on the chef's shoulder in refedit-29 (mask paste left a sliver), the cape in
-154 is no longer cut, and in -57 the neighbouring glass loses its straw. -29, -154
and -57 had the intended box. The additions in -153 and -39 and the hair bow in -184
(which landed on the wrong man) came from enlarged or whole-image boxes and say
nothing about cropping.

**Untested baseline worth one line:** a text rule (instructions that add an object get
box paste, the rest mask paste) is still a fixed pipeline. 27 of the 151 instructions
start with add / put / place / insert, 10 of them among the audited 40.

### Rerun after the fix (commands; nothing below has run)

```bash
python bench/run_gate_c.py --self-test
python bench/run_gate_c.py --model flux2-klein-4b --klein-crop --ablation \
  --mask-mode oracle --out runs/gate_c/klein4b_crop2_oracle.jsonl \
  --save-images runs/gate_c/img_klein4b_crop2_oracle
python bench/run_gate_c.py --model flux2-klein-4b --klein-crop --klein-crop-paste box \
  --ablation --mask-mode oracle --out runs/gate_c/klein4b_cropbox2_oracle.jsonl \
  --save-images runs/gate_c/img_klein4b_cropbox2_oracle
```

New file names, so the first runs stay on disk for comparison. Before trusting the
rerun: the 97 cases whose box did not change should reproduce their old images within
a small tolerance (same input, prompt and seed). If they do, only the 12 enlarged ids
among the 40 need new verdicts for the crop arm. That mixes two sittings, so whether
to re-judge 12 or all 40 is the rater's call. The band (≥ 30, 24–29, ≤ 23) is
unchanged.

### Rerun after the fix — result (2026-11-16, user's runs, checked from the saved images)

- **The fix took effect.** Box-paste changes lie inside the intended crop box in 151 of
  151 cases.
- **Unaffected cases reproduce.** The 84 cases whose mask never reached a border are
  pixel-identical to the first run in both arms; the 13 that only carried a strip
  differ in at most 0.3% of pixels; all 54 enlarged cases changed.

| single prompt, 151 cases | mask paste, first run | mask paste, rerun | box paste, first run | box paste, rerun |
|---|---|---|---|---|
| mis-targeted | 1 | 1 | 13 | 4 |
| valid (K=10) | 147 | 148 | 74 | 91 |
| leakage ratio median | 0.002 | 0.001 | 0.060 | 0.024 |
| sibling excess > 0.02 (cut-off picked here) | 4 | 5 | 44 | 33 |
| hot spot louder than the target | 0 | 0 | 34 | 27 |

Mask paste stays saturated, as it must. Box paste with the correct box: mis-targeting
falls from 13 to 4, and about a fifth of cases (33) still show sibling change, which
mixes real edits with the crop's resample round trip.

**Audit still owed.** Of the 40 audited ids, 12 have a new crop output; the other 28
are unchanged (24 pixel-identical, 4 within the strip). `runs/gate_c/crop2_rejudge_ids.json`
lists those 12 plus 8 of the 24 identical ones (seed 1116) as hidden repeats, so the
same sitting also measures how consistently the rubric is applied:

```bash
python bench/blind_audit.py --build \
  --arm crop:runs/gate_c/klein4b_crop2_oracle.jsonl:runs/gate_c/img_klein4b_crop2_oracle \
  --reference crop --prompt-kind single \
  --carry-over runs/gate_c/crop2_rejudge_ids.json --extra 0 \
  --out-dir runs/gate_c/blind_crop2 \
  --ledger runs/gate_c/blind_crop2_ledger.json --key runs/gate_c/blind_crop2_key.json
```

Fixed before judging: the corrected crop count is the 17 clean among the 28 unchanged
ids plus the clean verdicts among the 12, read against the same band (≥ 30, 24–29,
≤ 23). On the 8 repeats, the first-pass verdicts were 5 clean, 1 degraded, 1 wrong_edit,
1 no_edit; fewer than 6 of 8 matching means the rubric has drifted enough that the
three arms' counts should not be compared to within a few cases.

## Predicted-mask pipeline — scope (written before any run, 2026-11-17)

Every mask result above uses the benchmark's own mask. A mask-free method competes
with a pipeline that has to FIND the instance first: referring segmentation, then
paste-back. This measures that pipeline's first step on the same 151 cases.

**Model.** One backend, a sentence-level referring-segmentation model (Sa2VA family;
the exact checkpoint is recorded in every output row). The benchmark has only the full
editing instruction, not the referring phrase, so the model must take a sentence. Two
caveats fixed here: Sa2VA is trained on RefCOCO-family data and RefEdit-Bench's images
come from RefCOCO (the instructions were written by hand), so its score here may be
inflated; and it runs the model repository's own code.

**Prompt, frozen (no variants after a run):**
`Please segment the existing object in the image that this editing instruction refers to: {instruction}`
For "add X beside Y" the benchmark's target is Y, which is what this asks for. The
4-case smoke run is for mechanics only, not for tuning this sentence.

**Primary statistic: which instance the predicted mask lands on,** not IoU with the
whole instance. Many instructions name a part (shirt, screen, goatee) while the ground
truth is the whole person or laptop; a correct part mask would fail an IoU test. With
`in_T` = share of predicted pixels inside the target and `in_S` = the largest share
inside any one sibling (target pixels removed):

| class | rule (frozen) |
|---|---|
| `target` | `in_T` ≥ 0.5 and `in_S` < 0.2 |
| `sibling` | `in_S` ≥ 0.5 and `in_T` < 0.2 |
| `both` | `in_T` ≥ 0.2 and `in_S` ≥ 0.2 (paste-back would leak through this mask) |
| `empty` | fewer than 0.05% of the image's pixels |
| `other` | anything else |

Reported with it: a sensitivity table over the two thresholds (0.5 / 0.7 / 0.9 ×
0.1 / 0.2 / 0.3), IoU with the target as a secondary number, and the model's raw text
answer per case.

**Controls, run through the same save, load and score path:** the ground-truth target
mask must score 151 of 151 `target`; the largest sibling's mask must score 0. Then the
single-instance instruction, on the 141 cases whose two prompts differ: it names only
the category, so its `target` rate shows how much comes from something other than the
referring expression. It is not a clean null (its edit words carry some referent
information, as the attention probe found).

**Reading (amends the rule proposed in chat, because of the contamination):**

| `target` class, full instruction | reading |
|---|---|
| ≤ 113 of 151 (75%) | Strong evidence: same-kind referents are hard for a pipeline too, even with a model that may have seen these images. The mask-free method direction is alive, and the benchmark claim extends beyond editors. |
| ≥ 136 of 151 (90%) | Finding the instance is solved ON THIS BENCHMARK, possibly inflated. A method cannot claim aiming as its contribution here without a set the segmenter has not seen. Not, by itself, a decision about the paper. |
| 114–135 | Mixed. Report by class and move to the end-to-end arm. |

Reported separately, because it is where a pipeline would have to beat the editor: the
43 cases where plain klein edited off target, against the 108 where it did not; and
for comparison klein's own rates (edit on target 108 of 151; attention points at the
target 118 of 151).

**Then, end to end:** paste-back with the predicted masks
(`--klein-composite --mask-mode predicted --pred-masks DIR`), compared with oracle
paste-back case by case. A predicted mask goes through the same 8 px dilation; an
empty mask yields the source unchanged. Whether to audit it is decided after the
automatic numbers.

**Built (2026-11-17), not yet run on a model.** `bench/predict_masks.py` and
`--mask-mode predicted` in `bench/run_gate_c.py`; both self-tests pass on CPU.
Checkpoint: `ByteDance/Sa2VA-Qwen2_5-VL-7B` (Apache-2.0, about 34 GB on disk, greedy
decoding per its own code; revision recorded per row). Its card's example omits a
`processor` argument that this checkpoint's code requires; the backend follows the
official demo script instead. It needs a recent `transformers` and `qwen_vl_utils`:
**run it in a separate environment, and do not upgrade the one klein runs in**, or
klein's outputs may stop being pixel-comparable with the earlier arms.

Run in this order; each step gates the next.

```bash
# 1. controls: no model. Must print target 151 of 151, then 0 of 151.
python bench/predict_masks.py --backend oracle
python bench/predict_masks.py --backend sibling

# 2. runner path: paste-back through the ORACLE masks read from disk.
python bench/run_gate_c.py --model flux2-klein-4b --klein-composite \
  --mask-mode predicted --pred-masks runs/gate_c/pred_masks_oracle_full \
  --out runs/gate_c/klein4b_comp_predoracle.jsonl \
  --save-images runs/gate_c/img_klein4b_comp_predoracle

# 3. smoke: 4 cases, mechanics only. Look at the overlays and raw answers.
python bench/predict_masks.py --backend sa2va --limit 4 \
  --out runs/gate_c/pred_masks_smoke.jsonl --out-dir runs/gate_c/pred_masks_smoke \
  --save-overlays runs/gate_c/pred_overlays_smoke

# 4. full runs, then the report
python bench/predict_masks.py --backend sa2va --prompt-kind full \
  --save-overlays runs/gate_c/pred_overlays_sa2va_full
python bench/predict_masks.py --backend sa2va --prompt-kind single
python bench/predict_masks.py --report runs/gate_c/pred_masks_sa2va_full.jsonl \
  --report-single runs/gate_c/pred_masks_sa2va_single.jsonl \
  --outcomes runs/gate_c/klein4b_none.jsonl

# 5. end to end: paste-back through the predicted masks (full instruction only)
python bench/run_gate_c.py --model flux2-klein-4b --klein-composite \
  --mask-mode predicted --pred-masks runs/gate_c/pred_masks_sa2va_full \
  --out runs/gate_c/klein4b_comp_pred.jsonl \
  --save-images runs/gate_c/img_klein4b_comp_pred
```

Checks fixed here. Step 1: anything other than 151 and 0 stops the line; list the ids
(a sibling lying wholly inside the target would score as `target`, a benchmark fact
rather than a bug). Step 2: on the 84 cases whose mask never reached a border, the
images must be pixel-identical to the existing paste-back arm; that validates the
predicted-mask path of the runner, which has never executed, and gives a post-fix
oracle baseline for step 5. Step 5 uses the full instruction only: the masks come from
it, so a single-prompt edit through them would mix two conditions. After step 4, count
masks the scorer calls `empty` (under 0.05% of the image) that are not all-zero, since
the runner only skips all-zero masks.

Noted while reading the diffusers source (0.43.0): the library ships a
`Flux2KleinInpaintPipeline` that takes `mask_image`. That is a third mask-given
baseline (masking inside the sampler rather than after it) and should be scored
alongside paste-back and crop before a learned mechanism is claimed to beat "a mask".
