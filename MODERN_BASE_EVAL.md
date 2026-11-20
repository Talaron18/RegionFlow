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

### Crop baseline — corrected audit result (2026-11-17, user's re-judge of 20 panels)

**Rubric consistency.** On the 8 hidden repeats (panels pixel-identical to the first
crop audit) 7 verdicts match; the one change is degraded → destroyed (refedit-154),
and clean / not-clean agrees in 8 of 8. That clears the 6-of-8 bar fixed above. The
rater may have recognised some panels, so this is an upper bound on consistency.

**The 12 re-run cases** (first verdict → verdict with the correct crop box): 6 unchanged
clean (refedit-14, -19, -74, -69, -15 and, new, -184 no_edit → clean); 2 lost clean
(-139 and -126 → degraded); -39 no_edit → degraded; -183 destroyed → no_edit;
-113 degraded → wrong_edit ("cat big as a cow", so that scale error is real with the
intended box too); -153 still no_edit.

| arm, same 40 ids | clean | degraded | destroyed | wrong_edit | no_edit |
|---|---|---|---|---|---|
| klein 4B, no mask | 9 | 1 | 0 | 28 | 2 |
| + paste-back through the oracle mask | 19 | 4 | 0 | 4 | 13 |
| + crop, edit, paste — first run (box wrong in 12) | 24 | 3 | 4 | 3 | 6 |
| + crop, edit, paste — **corrected** | **23** | 5 | 3 | 4 | 5 |

**Reading, by the rule fixed before the first run: 23 is in the "≤ 23" band — no
meaningful gain over paste-back.** It sits on the band's edge exactly as 24 sat on the
other side, but this time the other evidence agrees with the band:

- Paired with paste-back: 6 became clean, 2 stopped being clean (sign test p = 0.29).
  Against the plain arm: 17 and 3 (p = 0.003).
- Weighted to all 151 by the plain run's class: crop ≈ 66%, paste-back ≈ 67%.

| plain run's class | share of all 151 | plain clean | paste-back clean | crop clean (corrected) |
|---|---|---|---|---|
| measurable (target changed most) | 108 (72%) | 6 / 16 | 13 / 16 | 11 / 16 |
| mis-targeted | 34 (23%) | 2 / 15 | 6 / 15 | 10 / 15 |
| off-instance | 9 (6%) | 1 / 9 | 0 / 9 | 2 / 9 |

The pattern from the first run survives the fix: where the base edits the wrong place,
cropping helps (12 of 24 against 6 of 24); where it already aims correctly, the two
are within two cases of each other. Of the 13 cases paste-back left unedited, crop
makes 5 clean and leaves 5 unedited. Clean in at least one of the two baselines: 25
of 40.

**What stands.** With the benchmark's own mask, both fixed pipelines reach about two
thirds clean and neither dominates. The rule's stated consequence for "≤ 23" was that
telling the model where has to happen inside the model; read that as "cropping does
not close the gap paste-back leaves", not as evidence that a learned mechanism would.
Failure types for the six re-run cases that are not clean have not been re-classified,
so the withdrawn sentences about the paste shape stay withdrawn. One rater, 40
stratified cases, two sittings mixed for this arm.

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

**Steps 1 and 2 — passed (2026-11-17, user's runs, checked from the saved files).**
- Controls: the ground-truth mask scores `target` in 151 of 151; the largest sibling
  scores `sibling` in 151 of 151, `target` in 0.
- Runner path: paste-back through the oracle masks read from disk is pixel-identical
  to the existing paste-back arm in all 84 cases whose mask never reached a border,
  with identical target-change values. The other 67 differ in at most 3.8% of pixels,
  which is the wrapped strip the dilation fix removed. `klein4b_comp_predoracle.jsonl`
  is therefore the post-fix oracle paste-back baseline for step 5.

Environment note: `qwen_vl_utils` was installed into the container that runs klein.
Installed there: torch 2.6.0+cu124, diffusers 0.44.0.dev0, transformers 5.23.0. Whether
that install changed anything klein uses is not known from the package list; an 8-case
plain rerun compared against `img_klein4b_none` settles it. The checkpoint's config
names transformers 4.56.1, so its code may not load under 5.x. Sa2VA therefore runs in a
separate venv under WSL (transformers 4.56.1, torch 2.6.0+cu124), not in the klein
container; the model loads straight to the GPU (`device_map`) because loading through
CPU RAM ran WSL out of memory.

**Klein environment check — passed (2026-11-18).** The container was recreated through
`docker/compose.yml` (the one with `qwen_vl_utils` is gone). An 8-case plain rerun
(`envcheck_none.jsonl`) is pixel-identical to `img_klein4b_none` in 8 of 8, with the same
seeds and target/sibling change values, so step 5 runs on the same klein as every
earlier arm.

**Step 3 — smoke run passed (2026-11-18, 4 cases, overlays viewed).** Every raw answer
is `Sure, [SEG].`; decoding greedy; revision `4cd6709067cf257235e922c201a6f265292f5fc8`.
Classes match the overlays: refedit-2 (lower quilt), -5 (woman with pizza), -8 (pink
case) are on the target with nothing on siblings. refedit-7 is a real miss: "Let the man
with longer curly hair, wearing a loose tie be a woman" gets a mask on the woman at the
left edge, not the man. The instruction names the edit's result ("a woman"), and the
segmenter took it as the referent. Four cases say nothing about the rate; they show the
pipeline runs and that the scorer reads the masks correctly.

**Step 4 — result (2026-11-18, user's runs; misses checked against overlays and masks).**
One mask per case; raw answers are `Sure, [SEG].` (109) or `Sure, it is [SEG].` (42);
one revision throughout.

| prompt | target | sibling | both | other | empty | median IoU |
|---|---|---|---|---|---|---|
| full instruction | **137** | 8 | 0 | 5 | 1 | 0.80 |
| single-instance instruction | 92 | 38 | 2 | 14 | 5 | 0.27 |

- **Reading: 137 of 151 is in the "≥ 136" band — finding the instance is solved on this
  benchmark, possibly inflated by contamination.** It clears the band by one case, and
  the sensitivity table puts it at 136 (t_in 0.7) and 121 (t_in 0.9). Even the strictest
  cell is above the "≤ 113" band, so the line does not support "same-kind referents are
  hard for a pipeline too".
- The referring expression is what does it: on the 141 cases whose prompts differ, the
  full instruction alone finds the target in 47, the single prompt alone in 2
  (p < 1e-4).
- Against the editor's own aiming (plain klein edit on target, 108 of 151): the
  segmenter finds the target in 38 cases where klein missed, and misses in 9 where klein
  hit (p = 2.5e-5). On the 43 cases klein edited off target it finds the target in 38.
  Its misses do not concentrate where klein's do (Fisher p = 0.54).
- Misses (14), from the overlays I viewed: the instruction names the edit's result and
  the segmenter goes there (refedit-7 "be a woman" → the woman; refedit-100 "to a young
  girl" → a girl), or it lands on a non-instance object (refedit-88 "the man in black
  wear a scarf" → his chair). Not every miss was viewed; no count by type.
- Post-step-4 check: one mask the scorer calls `empty` is not all-zero (refedit-71, 63
  pixels, 81% inside the target, "Add a necklace to the woman…"). The runner will paste
  through it, so step 5 edits a tiny region there rather than returning the source.

What this means for the plan, per the reading fixed above: a method cannot claim
**aiming** as its contribution on this benchmark, because an off-the-shelf segmenter
plus the instruction already aims at 91% here. The open question moves to step 5:
whether paste-back through these masks keeps the oracle paste-back's quality (about two
thirds clean on the audit sample), i.e. whether the pipeline is a full answer or only
an answer to the aiming half.

**Step 5 — automatic result (2026-11-18, `klein4b_comp_pred.jsonl`, full instruction).**
Measurable 140, mis-targeted 8, off-instance 3, against the post-fix oracle paste-back's
150 measurable. The outcome follows the mask almost one to one: all 136 cases with a
`target` mask whose oracle run was measurable stay measurable; 6 of 8 `sibling` masks
become mis-targeted; the 3 off-instance cases are `other` masks. Nothing was skipped as
empty; refedit-71's 63-pixel mask gives target change 0.0007 against 0.027, effectively
no edit. **Under paste-back this metric mostly measures where the mask is, not whether
the edit is any good,** so these numbers are not the comparison.

What the metric misses, seen in the images: on some cases Sa2VA's mask lies inside the
right person but covers only a small or scattered part of it, and paste-back through it
cuts off the edit. refedit-29 ("add a parrot on the shoulder of the chef"): the mask is a
few patches on his hair; oracle paste-back shows the parrot, predicted shows none.
refedit-27 ("hold a burger instead of a glass"): the mask is scattered patches on her
jacket, not the glass or hand; oracle shows a burger, predicted keeps a glass with a red
fragment pasted onto it. Among the 137 `target` masks, 11 give a target change under half
the oracle's (all with IoU ≤ 0.16). Two overlays viewed; the other nine are not
classified (part mask versus fragment). Either way, "lands on the right object" is not
"covers the pixels the edit needs".

**Decision: a targeted two-arm blind audit, not a 40-panel rerun.** On the 40 audit ids,
the predicted-mask image differs from the post-fix oracle paste-back image in more than
1% of pixels in 17; the other 23 are near-identical and keep their verdicts. Of the 19
oracle-clean ids, only refedit-160 (sibling mask) and -71 (63-pixel mask) differ
materially, so a full 40-panel audit would land near 17 by construction and could not
reach a low band. Instead, the 17 differing ids (`runs/gate_c/pred_audit_ids.json`) go
into one shuffled build with both arms, predicted and post-fix oracle (34 panels, same
sitting, arm hidden). Bands set 2026-11-18, after viewing refedit-29 and -27 and the
image-difference counts above, before any panel is judged. Statistic: net loss = ids
clean under oracle only minus ids clean under predicted only, among the 17.

| net loss | reading |
|---|---|
| ≤ 2 | Predicted masks cost no more than the known misses: this frozen pipeline matches oracle paste-back on the audit sample. |
| 3–4 | Mixed: report the cases. |
| ≥ 5 | This frozen pipeline (this prompt, pasting through the raw mask) loses edits that oracle paste-back keeps, beyond its aiming misses. It does not show a mask-free method is needed: obvious variants (paste through the mask's box, more dilation, ask for the whole object) are untested, and the pipeline's ceiling is oracle paste-back either way. |

Build in the klein container (it reads `/data`):

```bash
python bench/blind_audit.py --build \
  --arm pred:runs/gate_c/klein4b_comp_pred.jsonl:runs/gate_c/img_klein4b_comp_pred \
  --arm oracle:runs/gate_c/klein4b_comp_predoracle.jsonl:runs/gate_c/img_klein4b_comp_predoracle \
  --reference oracle --carry-over runs/gate_c/pred_audit_ids.json --extra 0 \
  --out-dir runs/gate_c/blind_pred \
  --ledger runs/gate_c/blind_pred_ledger.json --key runs/gate_c/blind_pred_key.json
```

**Predicted-mask audit — result (2026-11-18, user's judging of 34 panels).** All 34
judged. Among the 17 ids, oracle paste-back clean 8, predicted clean 5: **net loss 3, the
"3–4, mixed" band.** No id is clean under predicted only. The three losses:
- refedit-160, clean → wrong_edit ("leaked to sibling"): the mask is on the sibling
  microwave. A known aiming miss.
- refedit-71, clean → no_edit: the 63-pixel necklace mask. A known mask failure.
- refedit-181, clean → degraded ("change the white cat on the right to an orange cat"):
  the mask covers the cat but misses its tail, so the pasted cat is orange with the old
  black tail. The one loss from mask shape alone.

Also: refedit-29 degraded → no_edit (the parrot cut off; not a clean loss). The oracle
arm, re-judged blind in this sitting, matches its earlier verdicts in 16 of 17
(refedit-139 degraded → clean).

On the 40 audit ids (the other 23 near-identical, verdicts carried): oracle paste-back
about 19–20 clean, predicted about 16–17. Reading: on this sample the frozen pipeline
loses about 3 of 40 against oracle paste-back, two from the 9% of masks that miss and one
from an incomplete mask. It sits close to its ceiling, and its ceiling is oracle
paste-back (19–20 of 40 on this stratified sample, about two thirds weighted to 151). The finding to carry forward: **with a
segmenter that aims at 91%, the remaining gap is edit quality (no_edit, degraded,
wrong_edit under a correct mask), not aiming.**

Noted while reading the diffusers source (0.43.0): the library ships a
`Flux2KleinInpaintPipeline` that takes `mask_image`. That is a third mask-given
baseline (masking inside the sampler rather than after it) and should be scored
alongside paste-back and crop before a learned mechanism is claimed to beat "a mask".

## Fresh-case aiming test — scope (written before any case, 2026-11-18)

**Question.** Sa2VA finds the target in 137 of 151 RefEdit-Bench cases, but it was trained
on RefCOCO-family data and every RefEdit-Bench image is a RefCOCO photograph. Is 91% a
property of the pipeline, or of this benchmark? This decides whether aiming is still
open (learned-grounding line) or closed (evaluation finding + edit quality).

**Two arms, written by the user, nothing run until both are frozen.**
- **Arm A (the band):** at least 60 cases (domains below) on images that are not COCO and not from any
  referring-segmentation set (RefCOCO/+/g, gRefCOCO, ReasonSeg, GranD/SA-1B, MeViS,
  Ref-YouTube-VOS). Own photos are best; PIE-Bench++ or ImgEdit images are allowed if
  their source is checked and recorded. MagicBrush is COCO, so arm B only.
- **Arm B (interpretation only):** 20 cases on RefEdit-Bench / COCO images with freshly
  written instructions, preferably aimed at an instance the benchmark did not target.
  It separates "the images are familiar" from "the user's expressions are harder than
  RefEdit's".

**Case rules (enforced by `bench/fresh_cases.py --check`).**
- Each image has at least two instances of the target's category; every same-category
  instance gets a box (target first). Ground truth is boxes, not masks.
- At most 2 cases per image (different targets). Arm A: ≥ 10 cases of each expression
  type: `attribute` ("the woman in red"), `spatial` ("the left cup"), `relation` ("the
  dog next to the bench").
- One editing sentence in RefEdit style, with only the referring words needed to
  disambiguate; mix edit kinds (replace, change colour/attribute, add, remove).
- Choose the image and target before writing the sentence. Write all cases, run
  `--check`, record `sha256sum runs/fresh/cases.json` here, then run. No case is edited
  or dropped after a run.

**Pipeline, unchanged:** same Sa2VA checkpoint, same frozen prompt, images resized to
512×512, same `target / sibling / both / other / empty` scorer with t_in 0.5 and
t_leak 0.2, against boxes (sibling boxes minus the target box). Every overlay is viewed;
a class the viewer disagrees with (boxes are crude where instances overlap) goes into
`overrides.json` with a reason, and the report prints the band with and without them.

**Domains (decided 2026-11-18, user: an ideal model should work on photos, paintings
and illustrations).** Arm A is at least 60 cases: ≥ 25 photos, ≥ 15 illustrations,
≥ 15 paintings, each case tagged `domain`. The band below is read on all of arm A;
separately, any domain at ≤ 70% is reported as "aiming open for that domain" whatever
the total says. At 15 cases a domain's rate is ±25 points, so a domain reading is a
flag, not a measurement.

**Bands on arm A target rate (agreed 2026-11-18):**

| arm A target | reading |
|---|---|
| ≥ 85% (≥ 51 of 60) | Aiming is solved in general for this pipeline. The paper is the evaluation finding plus the edit-quality residual. |
| 71–84% | Mixed: report per expression type; no line reopened or closed. |
| ≤ 70% (≤ 42 of 60) | Aiming is open off RefCOCO; the learned-grounding line returns, and an uncontaminated split becomes part of the contribution. |

Arm B read with arm A when A ≤ 84%: B ≥ 17 of 20 → the drop comes from the images
(familiarity/contamination); B ≤ 14 of 20 → the fresh expressions are harder than
RefEdit's, which reopens aiming for a different reason. 60 cases give about ±9 points
at 95%, so the bands are coarse on purpose.

Commands (arm A and B in one `cases.json`; GUI annotator needs a matplotlib window, e.g.
WSLg; any tool giving pixel boxes `[x0, y0, x1, y1]` works instead):

```bash
python bench/fresh_cases.py --annotate /path/to/images --cases runs/fresh/cases.json
python bench/fresh_cases.py --check --cases runs/fresh/cases.json
sha256sum runs/fresh/cases.json          # record above before running
# in the Sa2VA venv:
python bench/fresh_cases.py --run --cases runs/fresh/cases.json \
  --out runs/fresh/sa2va.jsonl --out-dir runs/fresh/masks --save-overlays runs/fresh/overlays
# view every overlay, write runs/fresh/overrides.json if needed, then:
python bench/fresh_cases.py --report runs/fresh/sa2va.jsonl --cases runs/fresh/cases.json \
  --overrides runs/fresh/overrides.json
```

### Fresh-case smoke run (2026-11-18, user's runs; images viewed; arm S, never counted)

Sa2VA puts the mask on the intended instance in all three cases (`in_T` 1.00,
`in_S` 0.00): the back-right girl in dark blue, not the front-right girl in light blue,
and the left Special Week in both spellings. Three cases say nothing about the rate.

- **Teio (both spellings).** klein changes 0.3% of pixels; the user confirms it does not
  know Tokai Teio. This is missing character knowledge, not instance selectivity, so it
  is out of scope. Named-identity edits stay out of arm A.
- **Ice axe** ("make the girl in the back row in blue hold an ice axe in her hand").
  All three arms fail, each differently:
  - *plain:* a stray blade behind the middle girl's head, attached to no one, plus a
    third arm from the target reaching down past the front-right girl to hold the shaft.
    Her two raised gloves are still there.
  - *paste-back:* keeps the third arm (inside her mask) and cuts the axe off at the mask
    edge. The mask removes the off-target object and leaves the anatomy error.
  - *crop-box:* a plausible axe in front of her, but its shaft is gripped by the red
    glove next to her, while her own navy gloves stay raised. The object lands in the
    region and is attached to the wrong hand.

Reading: a correct mask is not enough for an edit that must attach an object to the
referent's body. Choosing a better region (crop-box) fixes the object being cut off,
but the remaining failures are a limb that should not exist and an attachment to the
wrong hand. The 40-panel audit already contains the same kind: refedit-27 (burger in
hand) and -29 (parrot on shoulder).

Count from `bench/cases_refedit_final.json`, by reading the instructions: about 10 of
151 cases add an object attached to the referent's body (refedit-27, -29, -36, -71,
-73, -85, -88, -141, -162, -184). Most "holding/wearing" matches are in the referring
words, not the edit. So this kind is rare on this benchmark; how badly it fails here
is not yet counted.

## Attachment-edit count — scope (written before any case, 2026-11-18)

**Question.** With aiming correct, how often does the pipeline fail when the edit must
attach an object to the referred person's body (hold, wear, sit on), and is that worse
than an edit inside the same person's outline? The smoke ice-axe case failed three ways
(third arm, axe cut off, axe in the neighbour's hand); one case is not a rate.

**Sets (judged together in one shuffled blind sitting; set and arm hidden).**
- **Fresh (arm T in `fresh_cases.py`), the one the reading uses.** Multi-person scenes,
  the referent among at least one same-category sibling.
  - ≥ 30 `attach` cases: ≥ 10 `hold` ("make the woman on the left hold an umbrella"),
    ≥ 8 `wear` (hat, scarf, glasses, necklace), ≥ 5 `onbody` (a cat on her lap, a
    parrot on his shoulder); ≥ 8 per domain (photo, illustration, painting).
  - ≥ 15 `inbody` controls: a change inside the clothes the person already wears
    (recolour, swap jacket for a rain jacket), each on the same image and the same
    target box as one attach case. This separates "attachment is hard" from "anything
    in these scenes is hard".
  - No named characters (that is knowledge, not attachment). Arm T images are never
    used in arm A. Max 2 cases per image (an attach case and its control). Write all,
    `--check`, record the sha256, then run.
- **RefEdit-Bench (supporting only).** The 10 attachment cases in
  `bench/attach_ids_refedit.json`, from images already on disk: plain, oracle
  paste-back (post-fix), crop-box. Note the paste arm here uses the benchmark's mask,
  the fresh paste arm uses Sa2VA's.

**Arms (fresh):** `fresh_cases.py --edit` writes plain, paste-back through the Sa2VA
mask, and crop-box (crop around the mask, edit, paste the whole box). **Primary arm:
crop-box**, fixed now because it is the arm that does not cut objects off; no best-of
across arms.

**Rubric (`bench/attach_audit.py`), the first that applies:** `no_object` (absent or
cut off past recognition) > `wrong_owner` (held/worn by someone else or by no one) >
`anatomy` (a body part added, missing or malformed, anyone's) > `identity` (the
referent's face or body changed beyond the edit, or merged with someone) >
`off_target` (the edit is right, something else changed) > `degraded` > `clean`.

**Statistics, on fresh cases Sa2VA aimed correctly (class `target`; mis-aimed cases are
reported separately), crop-box arm.** "Edit right" = verdict `clean`, `off_target` or
`degraded`: the edit on the referent is right, whatever else happened (amended
2026-11-18 after the second smoke run, before any arm-T case was written: crop-box
pastes its whole box and recolours siblings inside it by construction, which must not
count as an attachment failure; `off_target` is reported beside it).
A = edit-right rate on all aimed attach cases;
F = share of attach failures that are `wrong_owner` or `anatomy`; Z = share that are
`no_object`; gap = edit-right rate of in-body controls minus that of their attach
partners, **on pairs only** (same image, same person, both aimed), with the discordant
counts (attach-only failures, control-only failures). For the `no_object` failures, the
plain arm's verdict on the same case says whether klein drew the object at all.

| result (rows read top to bottom, first match) | reading |
|---|---|
| A ≥ 70% edit-right | Attachment is not a major residual for this pipeline; no method target here. |
| A ≤ 50% and Z ≥ 50% | The object does not appear. If plain drew it in ≥ half of those cases, the region choice is the problem (cheap to fix); if not, klein does not do these edits. Neither is attachment. |
| A ≤ 50%, F ≥ 50% and paired gap ≥ 25 points | Attachment is a specific residual (worse than in-body edits on the same people, failing by owner or anatomy): candidate method target. |
| A ≤ 50% and paired gap < 25 points | A general quality problem in these scenes, not specific to attachment. |
| anything else | Mixed: report the cases. |

At 30 cases a rate is about ±17 points; these bands only separate large effects.

`refedit_items` (the RefEdit panels) is tested only with fake bench rows, COCO records
and image dirs; the first real `--build` is its smoke test, so look at a few RefEdit
panels (target outlined green, instruction caption) before judging any.

Commands:

```bash
python bench/fresh_cases.py --check --cases runs/fresh/attach.json
sha256sum runs/fresh/attach.json        # record here before running
# Sa2VA venv
python bench/fresh_cases.py --run --cases runs/fresh/attach.json \
  --out runs/fresh/attach_sa2va.jsonl --out-dir runs/fresh/attach_masks \
  --save-overlays runs/fresh/attach_overlays
# klein container (reads /data for the RefEdit panels)
python bench/fresh_cases.py --edit --cases runs/fresh/attach.json \
  --out-dir runs/fresh/attach_masks --edit-dir runs/fresh/attach_edits
python bench/attach_audit.py --build \
  --fresh-cases runs/fresh/attach.json --fresh-edits runs/fresh/attach_edits \
  --refedit-arm plain:runs/gate_c/img_klein4b_none \
  --refedit-arm paste:runs/gate_c/img_klein4b_comp_predoracle \
  --refedit-arm cropbox:runs/gate_c/img_klein4b_cropbox2_oracle \
  --out-dir runs/fresh/attach_blind
# judge every panel in runs/fresh/attach_blind/ledger.json (do not open key.json), then
python bench/attach_audit.py --score runs/fresh/attach_blind --aiming runs/fresh/attach_sa2va.jsonl
```

### Second smoke run (2026-11-18, user's runs; every grid and overlay viewed)

**Box bug, mine:** the classroom file is 2364×1330 and its boxes were drawn on a
2000×1125 copy, so the run scored the two desk cases `both` / `other`. Rescored from the
saved masks with the boxes rescaled, both are `target` (teacher's desk `in_T` 1.00;
left desk 0.82, its mask also covers the chair beside it). Boxes fixed in
`bench/smoke_cases_example.json`; cases now record `image_wh`, and `--run` / `--edit`
refuse a file whose size differs.

| case | Sa2VA | plain klein | paste-back | crop-box |
|---|---|---|---|---|
| rain jacket (back-right girl) | target | target **and middle girl** yellow | only the target yellow | target yellow; middle girl's jacket and the front girl's hood partly yellow (inside the box) |
| jacket to red (same girl) | **empty mask** ("Sure, it is [SEG].") | every girl red or orange, front-right girl's whole outfit red | source unchanged | not run (empty mask) |
| teacher's desk white | target | every desk and chair white | only the teacher's desk white | teacher's desk and the student desks next to it white |
| near-left desk red | target (+ chair) | most desks and chairs on the left red | target desk and the chair beside it red | target desk, the desk behind it and chairs red |

Readings:
- **Plain klein leaks on every familiar edit here.** Each colour or garment change
  spread to same-category instances, in photos and in the illustration. The leakage
  that started this project is strong in these scenes, not only in RefEdit-Bench.
- **Paste-back is right on all three cases with a mask**; its errors are the mask's
  (the chair under the left-desk mask). On these edits the pipeline is close to done.
- **Crop-box leaks inside its box by construction.** It is still the arm that keeps an
  added object whole (the ice axe), so it stays primary for arm T with the "edit right"
  statistic above; it is not a preservation baseline.
- **One segmenter miss:** "Change the jacket of the girl in the back row in blue to red"
  gives an empty mask, while the rain-jacket sentence on the same girl gives a correct
  one. Same instruction-shape as the earlier misses (the sentence names a colour the edit
  should produce). One case, not counted.

## Attribution arm — design and smoke criteria (written before any GPU run, 2026-11-18)

**Idea (referent-counterfactual change attribution).** Neither fixed region works: the
referent's mask cuts off what extends past it (axe, parrot), and the crop box recolours
neighbours inside it. Instead, let the editor's own outputs decide. Run klein again,
same seed, with the same edit aimed at a sibling (`counter_instructions`, written per
case: "make the girl in the back row in pink hold an ice axe…"). Outside the referent's
(dilated) Sa2VA mask, keep a pixel only if it changed vs the source, differs from every
counterfactual output, and connects to the mask through such pixels. Inside the mask
everything is kept, so a leak that also hits the target cannot erase the target's edit.
Paste the edit through that region, feathered as paste-back.

Fixed now: 1/4-resolution grid, 3×3 box blur, both thresholds 30/255, 4-connectivity.
One extra klein run per counterfactual sentence. Code: `attribution_mask` in
`bench/fresh_cases.py`; `--edit` writes `<id>_attrib.png`, `<id>_attrib_mask.png` and
`<id>_counter<k>.png` for cases with `counter_instructions`, and prints
`changed_outside / spillover_dropped / attributed_kept` (grid cells). Tested on synthetic
images and end to end with a fake pipe; not run on a GPU.

**Known limits, stated before the run.** A leak that happens only for the real referent
is kept. If two prompts with the same seed diverge everywhere (klein regenerates the
whole image), every change looks referent-specific and the arm degenerates toward
"plain, connected to the mask"; `spillover_dropped` near zero on a case with a visible
leak is that failure.

**Smoke reading, on the four smoke cases with a mask and counterfactuals** (axe, rain
jacket, teacher's desk, near-left desk; red jacket has no mask):
- *Works:* on at least 2 of the 3 in-body cases the attrib image has no sibling change
  that paste-back does not also have, **and** on the axe it keeps visibly more of the
  object than paste-back. Then it becomes a fourth arm in arm T, scored on edit-right
  and `off_target` against paste-back and crop-box, written into the arm-T scope before
  any arm-T case exists.
- *Fails:* it keeps sibling changes paste-back removes on 2 or more in-body cases, or
  `spillover_dropped` is near zero where the plain edit visibly leaked. Then the
  counterfactual does not isolate spillover with this editor; one diagnosis only
  (view `<id>_counter0.png` against `<id>_plain.png`), no threshold tuning on these
  cases.
- Anything else: report the four cases, decide with the user.

### Attribution arm — smoke result (2026-11-19, user's run in `runs/fresh/smoke_edits_attrib`; all grids viewed)

Changed-pixel fractions (vs source, >30/255) and what the images show:

| case | paste | crop-box | connected (no counter) | **attrib** | plain vs counter0 differ | seen |
|---|---|---|---|---|---|---|
| rain jacket | 0.051 | 0.109 | 0.063 | **0.052** | 0.077 | attrib ≈ paste: target yellow, middle girl not; `connected` keeps part of the middle girl's yellow, the counterfactual removes it |
| teacher's desk | 0.042 | 0.100 | 0.435 | **0.043** | 0.010 | attrib ≈ paste; `connected` keeps every white desk, the counterfactual removes all of it |
| near-left desk | 0.067 | 0.110 | 0.167 | **0.140** | 0.337 | **fails**: keeps red chairs and desks around the target. The counter run ("…on the right") spread red to a *different* set of desks, so the left-side leak looked referent-specific |
| ice axe | 0.007 | 0.017 | 0.012 | **0.012** | 0.041 | keeps the shaft and pick that paste cut, i.e. it rescues **the third arm** holding them; the stray blade behind the middle girl (also in the counterfactual) is dropped |

**Reading against the criteria fixed before the run.** In-body cases: 2 of 3 with no
sibling change beyond paste-back (rain, teacher's desk); 1 fails (near-left desk). Axe:
keeps visibly more of the object than paste-back. By the letter that is the "works" row,
but the substance does not support adopting it as the method:
- On in-body edits it only matches paste-back, which was already right; there is
  nothing for it to win there.
- Its premise, that spillover does not depend on who is referred to, fails when the leak
  spreads *from* the referent: klein recolours the referent's neighbourhood, so moving
  the referent moves the leak (near-left desk, 34% of pixels differ between the two runs).
- On the one attachment case it faithfully keeps what the generator drew outside the
  mask, and what the generator drew is a third arm.

**Decision (with the user, 2026-11-19): attribution is not the method.** It stays an
optional arm-T column, not a contribution. The smoke runs together say region choice is
exhausted: paste-back already handles in-body edits, and for additions every region
rule (mask, box, attribution) can only keep or cut what the generator produced. The
remaining failure is in generation itself: limbs that should not exist and objects bound
to the wrong hand.

## Seed sweep — is the right answer in klein's output at all? (rule written before any run, 2026-11-19)

**Question.** Every region rule can only keep or cut what klein draws. If klein sometimes
draws the axe in the target's own hand with two arms, a training-free "generate several,
check the skeleton, keep a valid one" method is possible. If it never does, rejection
cannot help and the line moves to skeleton conditioning or training.

**Run.** `smoke-axe` only, plain klein (nothing pasted), seeds k = 0…7 with
`case_seed("smoke-axe", k)`; seed 0 reproduces the earlier plain image (third arm, stray
blade). Output: `<id>_seed<k>.png` and `<id>_seeds_sheet.png`.

**Label each seed by eye, first that applies:**
`no_object` (no axe, or unrecognisable) > `wrong_owner` (held by another girl, or by no
one: floating, lying, stuck behind someone) > `anatomy` (anyone has an extra, missing or
malformed limb or hand) > `off_target` (axe correct, but another girl's body or clothes
changed) > `correct` (the back-right girl holds the axe in one of her own two hands, no
body part added or lost anywhere).

**Rule:**

| correct of 8 | reading |
|---|---|
| ≥ 2 | The right answer is within klein's distribution. A skeleton check that picks it is worth building: next, the same sweep on 3–5 photo attachment cases with an automatic pose check. |
| 1 | Marginal: 8 more seeds (k = 8…15); ≥ 3 of 16 reads as "within reach", fewer as "out of reach". |
| 0 | Out of reach for rejection on this case: the line moves to giving klein the skeleton (a pose reference image, then training), not to more sampling. |

One illustrated case decides only whether the idea survives the smoke, not the method.
Seeds are not re-labelled after the sheet is seen, and no seeds are added beyond the
table's own rule. Rejection stays in or out on the photo cases' own result.

```bash
python bench/fresh_cases.py --edit --seeds 8 --only smoke-axe \
  --cases bench/smoke_cases_example.json --edit-dir runs/fresh/smoke_seeds
```

### Seed sweep — result, seeds 0–7 (2026-11-19, user's run in `runs/fresh/smoke_seeds`)

Labels are **mine**, from the sheet and full-resolution crops of the back row; the user
had not sent labels. They stand unless the user disagrees on a specific seed.

| seed | what is drawn | label |
|---|---|---|
| 0 | blade behind the middle girl held by no one; a third arm from the target holds a shaft | wrong_owner (also anatomy) |
| 1 | no axe | no_object |
| 2 | large blade held up between the back-left and middle girls, not by the target | wrong_owner |
| 3 | no axe | no_object |
| 4 | axe raised behind the middle girl's head, in a hand that is not the target's | wrong_owner |
| 5 | axe held between the back-left and middle girls | wrong_owner |
| 6 | **the target holds an ice-axe pick in her own left hand**; her right red mitten stays raised, two hands. Her left glove turned from navy to brown. **The back-left girl's open raised mitten became a fist** | off_target |
| 7 | a large axe held in front of the target's face by the **red mitten**; her navy glove stays raised | **anatomy** (user's label, adopted: the red mitten is the pink girl's, not the target's, whose gloves are navy; the hand is drawn as if it belonged to both, i.e. a hand shared between two bodies). My first label, `correct`, was wrong |

**Reading by the rule, with the user's label for seed 7: 0 of 8 correct, the "out of
reach" row.** No further seeds (the 8–15 run was only allowed under the marginal row).
Rejection sampling does not survive the smoke on this case; the line moves to giving
klein the skeleton. Caveat kept on record, not used to change the reading: seed 6 has the
attachment right (her own hand, with its glove recoloured navy → brown) and its one fault
is another girl's hand outside the target's mask, which paste-back would remove.

What the 8 seeds show: the axe appears in 6 of 8, and **none** is held by a hand that is
unambiguously the target's own and unchanged. The failure is ownership: whose hand, and
hands that belong to two bodies at once (seed 7) or to no body (seed 0).

## Skeleton reference — does klein follow a stick figure? (rule written before any run, 2026-11-19)

**Why.** The seed sweep put the axe case in the "out of reach" row: klein does not
produce a correct owner/anatomy often enough to select. Next step by that rule: give
klein the skeleton. Cheapest form, no training: FLUX.2 klein takes several reference
images (each gets its own position offset in `Flux2KleinPipeline`), so the second
reference is a stick-figure drawing of everyone's pose, with a red mark where the object
is held. The prompt gets a fixed suffix (`POSE_SUFFIX` in `fresh_cases.py`) asking it to
keep each body as drawn and not to draw the lines.

**Case.** `smoke-axe-raised`, on the user's own drawing (same 452×640 canvas as the
poster): "make the girl in the back row with her arm raised hold an ice axe in her raised
hand"; the red mark sits on the back-left girl's raised arm. This target is the back-left
girl, not the earlier back-right one; a raised hand in open sky is probably an easier
attachment, which is why the same seeds also run **without** the drawing as the control.

**Run.** Seeds k = 0…7, each seed twice: plain (`<id>_seed<k>.png`) and with the drawing
(`<id>_pose_seed<k>.png`), same seed, same instruction (the pose run adds only the suffix).

**Labels, first that applies:** `sketch_leak` (blue or red lines of the drawing visible)
> `no_object` > `wrong_owner` > `anatomy` (any extra, missing, shared or malformed limb or
hand, anyone's) > `off_target` (axe right, another girl changed) > `correct`. Labelled by
the user; plain and pose sheets judged in the same sitting.

| result | reading |
|---|---|
| pose correct ≥ 3 of 8 **and** ≥ plain correct + 2 | klein uses a stick figure zero-shot. Next: the photo attachment cases with detected skeletons. |
| pose and plain within 1 | klein ignores the drawing; the reference-image route is closed, the line moves to a trained pose adapter. |
| sketch_leak in ≥ 4 of 8, or pose correct ≤ plain correct − 2 | the drawing hurts; same consequence as "ignores". |
| anything else | report, decide with the user. |

```bash
# save the drawing as runs/fresh/smoke_images/yamanosusume_pose.png (452x640)
python bench/fresh_cases.py --edit --seeds 8 --only smoke-axe-raised \
  --cases bench/smoke_cases_example.json --edit-dir runs/fresh/smoke_pose
```

### Skeleton reference — first run void (2026-11-19, `runs/fresh/smoke_pose`; all 16 viewed)

**Mis-specified, my error; not read against the rule.** The user redrew the stick figure
with the **back-right** girl's arm raised and every joint marked red. The case still said
"the girl in the back row **with her arm raised**" (the back-left girl, who raises her
arm in the source), and the prompt suffix said "held where the red mark is", which now
pointed at every joint. Instruction and drawing named different girls.

What the images show anyway:
- **Plain (no drawing):** the back-left girl holds an axe in her raised hand in nearly
  every seed. Raised hand in open sky is an easy attachment; it needs no help.
- **With the drawing:** in pose seeds 1 and 2 the **back-right** girl raises her right
  arm and holds the axe, with her other glove still at her chest: the drawing's pose, which
  no plain seed shows. Seeds 0, 4, 5 follow the instruction (back-left girl); seeds 3, 6
  put the axe behind someone with no clear owner; seed 7 gives it to the middle girl and
  shifts the whole image's style. No drawn lines leak into any image.

So klein does take something from a stick figure, and it fights the instruction when they
disagree. That is a reason to run the test properly, not a result.

**Re-run, specified consistently (rule unchanged from the section above).** Case
`smoke-axe-pose`: the original hard target (back-right girl in blue), instruction "make
the girl in the back row in blue raise her right hand and hold an ice axe in it", the
user's second drawing as `yamanosusume_pose2.png`. Both arms (plain and with drawing) get
the same instruction, so the drawing is the only difference. The suffix no longer
mentions the red mark (`POSE_SUFFIX`); the grip is stated in the instruction only.

```bash
# save the second drawing (452x640) as runs/fresh/smoke_images/yamanosusume_pose2.png
python bench/fresh_cases.py --edit --seeds 8 --only smoke-axe-pose \
  --cases bench/smoke_cases_example.json --edit-dir runs/fresh/smoke_pose2
```

### Skeleton reference — consistent re-run result (2026-11-19, `runs/fresh/smoke_pose2`)

User's label: **fail on every seed, both sheets.** I viewed all 16 back-row crops and
agree; 0 correct of 8 with the drawing and 0 of 8 without. By the rule (within 1): **klein
does not use the stick figure for this edit; the reference-image route is closed.** No
lines leak. (The drawing file `yamanosusume_pose2.png` was not in the push; the run's
inputs therefore cannot be re-checked from the repo.)

What the failures have in common, plain and with drawing alike: the girl in blue keeps
**both** of her original gloves at her chest, and the raised arm holding the axe is a
**new** limb, grown from her shoulder, from behind her, or from the girl in pink's sleeve
(pink arm in plain seed 5, pose seeds 0–2). In 16 tries klein never moves an existing
arm: it adds one, or gives the job to a neighbour. The edit is drawn as pixels added on
top of an image it otherwise preserves, not as a body part changing position. For a
raised hand that already exists (the back-left girl, void run), the same instruction is
easy.

Consequence: the failure is not missing pose information (the drawing supplied it and
changed nothing) but that this editor cannot *re-pose* a part; it can only add. That is
the user's skeleton point in its sharpest form: a creator moves the arm, the editor
grows a new one.

**Pose seed 6, user's note (2026-11-19).** The user judged it the closest: the girl in
blue raises a navy-sleeved arm holding the axe and keeps her navy glove at her chest. In
the source only one of her hands is visible (the red mitten in front of her belongs to
the girl in pink), so a raised right arm need not be an added limb. But the raised arm
starts too far toward the pink girl, the upper arm is too long, the drawing style
differs, and it covers where the pink girl's red mitten was. Proportion and covering the
original arm, not ownership, are what fail here. Still `fail` by the user's label.

## Erase then redraw — the painter's order (rule written before any run, 2026-11-19)

**Idea.** Klein adds a limb rather than moving one. A painter would first erase what is in
the way, then draw the new pose. Two plain klein steps, same seed: step 1
`pre_instruction` removes the hand in front of the target's chest; step 2 runs the case's
instruction on step 1's output. Optional `pre_mask` (user-drawn, white = may change,
452×640) pastes step 1 into the source only there. The plain control (instruction on the
source, no step 1) runs on the same seeds in the same call.

**Case.** `smoke-axe-erase`: target the girl in blue; step 1 "Remove the gloved hand held
up in front of the chest of the girl in the back row in blue, so both of her arms hang
down at her sides out of view"; step 2 "make the girl in the back row in blue raise her
right hand and hold an ice axe in it" (the pose2 instruction).

**Labels.** Final images: as in the seed sweep (`no_object` > `wrong_owner` > `anatomy` >
`off_target` > `correct`), plus the limb label (`unchanged / moved / added / shared /
removed / other`). Step-1 images: `cleared` (her hand gone, nothing else broken) or
`not cleared`.

| result | reading |
|---|---|
| step 1 `not cleared` in ≥ 6 of 8 | Step 2 was not tested. One re-run with a user-drawn `pre_mask`; no other change. |
| erase correct ≥ 2 of 8 **and** ≥ plain correct + 2 | Erase-then-redraw gets round add-only on this case: it becomes a column in arm T (needs a step-1 sentence per attach case). |
| erase within 1 of plain | Clearing the way does not make klein draw a correct arm; the line goes to B (a trained re-posing adapter), justified by arm T. |
| anything else | report, decide with the user. |

```bash
python bench/fresh_cases.py --edit --seeds 8 --only smoke-axe-erase \
  --cases bench/smoke_cases_example.json --edit-dir runs/fresh/smoke_erase
```
Sheets: `smoke-axe-erase_seeds_sheet.png` (plain control), `_step1_seeds_sheet.png`,
`_erase_seeds_sheet.png`.

## Arm T rubric addition (2026-11-19, before any arm-T case)

Every arm-T panel also gets a **limb** label, independent of the verdict: `unchanged`,
`moved` (an existing limb changed position, nothing added), `added` (a new limb while the
original stays), `shared` (one limb drawn on two bodies), `removed`, `other`.
`attach_audit.py --score` refuses unknown labels, warns on missing ones, and prints a
limb table per set, edit kind and arm. The claim it is for: editors **add** instead of
**move** when an edit needs a pose change.

### Erase then redraw — result (2026-11-19, `runs/fresh/smoke_erase`; every image viewed)

**Step 1 failed: `not cleared` in 7 of 8 (my labels).** Asked to remove the hand in front
of her chest, klein removed far more:
- seeds 0, 1: the girl in blue is deleted entirely, replaced by mountains;
- seeds 2, 3, 4, 7: her body below the head dissolves into fog or cloud;
- seed 5: she is intact and her hands are still at her chest (nothing removed);
- seed 6: her hands are gone and her body is intact, though darker: the only `cleared`.

**Reading by the rule: step 2 was not tested.** With the target missing or dissolved,
step 2 mostly gave the axe to the girl in pink (final seeds 0, 1, 2, 4, 5), which says
nothing about redrawing. One re-run is allowed: same sentences, a user-drawn `pre_mask`.

Observation, not a reading: "remove" works on **whole instances**, not parts. Told to take
away a hand, the editor takes away the person, or her body. It is the same pattern as
add-instead-of-move: the editor's unit is the object, not the limb.

Risk of the re-run, stated now: the mask only limits *where* step 1 is pasted. If klein
still deletes the whole girl, the paste puts mountain or fog on her chest inside the
mask. If that happens in ≥ 6 of 8, the next step 1 is the inpainting pipeline
(`Flux2KleinInpaintPipeline`, the region filled with "her dark navy jacket"), not more
prompt wording.

```bash
# draw the mask on the 452x640 poster canvas: white over her navy glove and the area in
# front of her chest where the hand is, black elsewhere; save as
# runs/fresh/smoke_images/yamanosusume_erase_mask.png, then add to the case
#   "pre_mask": "runs/fresh/smoke_images/yamanosusume_erase_mask.png"
python bench/fresh_cases.py --edit --seeds 8 --only smoke-axe-erase \
  --cases bench/smoke_cases_example.json --edit-dir runs/fresh/smoke_erase_mask
```

### Erase then redraw, masked re-run — result (2026-11-19, `runs/fresh/smoke_erase_mask`)

The user's mask (white over the target's navy glove, in front of her chest) was used:
outside it, step 1 changed 0.0000 of pixels in every seed. Inside it, step 1 replaced the
hand with a patch of fog or haze in 6 of 8 seeds (0, 2, 3, 4, 6, 7), a sky-coloured patch
in seed 1, and left the hand in seed 5. User's verdict: not working; I agree.

**The decisive check: step 2 does not depend on step 1.** Same seeds, final image vs the
plain control (the instruction applied directly to the untouched source):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| pixels differing **outside** the mask | 1.1% | 0.2% | 1.1% | 1.2% | 4.2% | 0.3% | 0.5% | 3.7% |

Apart from the masked patch itself, klein draws the same raised arm and axe, from the same
place, whether or not her hand is still there. Removing the obstacle does not change what
it draws, so a cleaner step 1 (the inpainting pipeline) would not change step 2 either.
**Option A (erase then redraw) is closed** without the inpainting follow-up: its premise,
that the added limb is caused by the old one being in the way, is contradicted.

Summary of the axe line, all on one illustrated case: region rules (paste, crop-box,
attribution) can only keep or cut what klein draws; more seeds give 0 of 8; a stick-figure
reference changes nothing (0 of 8 vs 0 of 8); clearing the old hand changes nothing. What
klein draws for "raise her hand and hold X" is a new arm whose origin does not depend on
the body that is there. Whether that holds beyond one poster is what arm T measures.

## Move then attach — smallest test of sketch-then-detail (rule written before any run, 2026-11-19)

**Why.** Every route that asks klein to change a pose has failed on the axe case, but when
a raised hand already exists the same attachment is easy (void pose run: the back-left
girl). The user's idea separates structure from appearance: a sketch decides the new
pose, a detail model renders it. Its smallest training-free form: **move** the existing arm
with a tool built to move parts (drag editing), then let klein **attach** the axe to the
now-raised hand.

**Step 1 — move (DragDiffusion, separate environment).** Released code, README states
~14 GB, SD-based, LoRA per image (github.com/Yujun-Shi/DragDiffusion). Run it in its own
venv, not in the klein container, and not at the same time as klein. In its Gradio UI, on
`yamanosusume.jpg`:
- editable mask: the girl in blue's left arm, glove and the space above her right
  shoulder (her own side only; not the pink girl);
- handle point: her navy glove at her chest, about (361, 217) in 452×640 pixels;
- target point: beside and above her head, about (395, 110);
- an anime SD1.5 checkpoint may suit the poster better than base SD1.5; record which.
Save the result as `runs/fresh/smoke_move/moved.png`.

Label step 1 once: **moved** (the glove is gone from her chest, one arm is raised from her
own shoulder, no extra limb, nobody else changed), **not moved** (little or no motion), or
**broken** (smeared arm, extra limb, face or neighbour damaged). Up to 3 attempts with
different point placements or checkpoints are allowed; report all of them, use the best.

**Step 2 — attach (klein, seeds 0–7).** Case `smoke-axe-moved`: "make the girl in the back
row in blue hold an ice axe in her raised hand" on `moved.png`. Labels as in the seed
sweep plus the limb label. Control: the same girl, no move step: 0 of 8 in the seed
sweep, the pose runs and the erase runs.

| result | reading |
|---|---|
| no attempt reaches `moved` | Drag tools cannot make this articulated move here; move-then-attach is closed at step 1, and sketch-then-detail needs a trained renderer that obeys a skeleton. |
| `moved`, and step 2 correct ≥ 3 of 8 | Moving first is enough: the smallest form of sketch-then-detail works. Next: automate it on photos (pose detector → planned keypoint → drag → attach) as an arm-T column. |
| `moved`, step 2 correct ≤ 1 of 8 | Even with the pose already right, klein fails to attach: the problem is not only re-posing. |
| anything else | report, decide with the user. |

```bash
# after step 1 (check moved.png's size; set image_wh in the case, rescale boxes if not 452x640)
python bench/fresh_cases.py --edit --seeds 8 --only smoke-axe-moved \
  --cases bench/smoke_cases_example.json --edit-dir runs/fresh/smoke_move
```

### Move then attach — skipped (2026-11-19, user's decision)

DragDiffusion could not be made to run in the user's environment (a dependency chain
pinned to 2023 packages, then UI errors). The test is dropped, not failed.

**What it was for:** to separate "klein cannot re-pose an arm" from "klein cannot attach
an object to the right hand". Existing runs already bear on that, without a drag tool:
- **free hand, no crowding** (back-left girl, raised hand in open sky, void pose run):
  klein attaches the axe correctly in nearly every seed;
- **hand in a crowd, no move strictly needed** (seed sweep: "hold an ice axe in her
  hand", the target's hand at her chest between her body and the pink girl's mitten):
  0 of 8, mostly another girl's or nobody's hand.
So the variable that looks decisive is whether the hand sits in open space or among other
people's hands and bodies, not only whether the arm must move. One poster, so this is a
hypothesis for arm T, not a result.

**Arm T addition (before any arm-T case is run):** attach cases carry an optional
`hand` tag, `free` (the hand that should hold it is in open space), `crowded`
(overlapping or touching someone else) or `hidden` (not visible in the source).
`attach_audit.py --score` prints edit-right by tag beside the reading; it is descriptive
and does not change the pre-registered bands. The hiking-photo umbrella case is tagged
`crowded` (his hand on the pole sits beside the olive-jacket man).

## Arm T — cases written (2026-11-20, `bench/attach_cases.json`)

From the user's images in `runs/fresh/arm-t/` (28 used; `apls1.webp` and `grad1.png` are
duplicates of the `.jpg` files). Instructions and boxes written by me from gridded
previews; every box checked on an overlay before committing.

- **53 cases: 38 attach + 15 in-body controls**, each control on the same image and target
  as one attach case. Attach kinds: hold 24, wear 8, on-body 6. Hand tags on hold cases:
  free 13, crowded 11.
- Domains of attach cases: photo 27, illustration 11, **painting 0**: the one rule still
  failing. `--check` / `--run` refuse until at least 8 painting attach cases exist (the
  domain rule was set for "photos, paintings and illustrations").
- `apec.jpg` is **not used**: making identifiable heads of state hold or wear things is
  political-manipulation territory and could not be shown in a paper. Other images of real
  people (athletes, actors, the Nasdaq photo) are used for internal testing; check
  publication rights before any edited image appears in a paper.
- Max 2 counted cases per image; no arm-T image may be used in arm A.
- Not yet frozen: the sha256 is recorded once the painting cases are added and `--check`
  passes.

**Paintings added (2026-11-20).** 9 paintings, 17 cases (13 attach, 4 in-body controls):
Botticelli, Leonardo's Adoration, van Gogh's Prisoners' Round, two Monets, Werner's
Proclamation, Warhol's Marilyns, The Night Watch, The Anatomy Lesson. Targets avoid the
nude Venus and the cadaver. The Warhol grid is nine copies of one face: a pure sibling
test with no hands (wear and in-body only).

**Arm T total: 70 cases, 51 attach + 19 controls**; attach domains photo 27 / painting 13
/ illustration 11; kinds hold 32 / wear 13 / on-body 6. `--check` passes. Not frozen yet:
the user reviews the instructions, then the sha256 is recorded here before any run.

**Frozen (2026-11-20).** `sha256 bench/attach_cases.json = 2bfd2810cfb553aa3c9dabb97dcf4226b5e5ae5b8b624d0fcdab9289263036fa`
(the user's hash, matched here). Two images the cases use, `nasdaq.jpg` and `tsurugi_group.jpg`,
had been deleted from `runs/fresh/arm-t/` in commit 8254d1c along with the intended removals;
restored from its parent, sizes match `image_wh`.

### Arm T step 1 — Sa2VA aiming (2026-11-20, user's run; all 70 overlays viewed)

One mask per case, raw answers `Sure, [SEG].` / `Sure, it is [SEG].`, revision
`4cd6709…` as before. **Aimed (`target`): 64 of 70, attach 46 of 51, controls 18 of 19.**
By domain, attach: photo 25 of 27, illustration 11 of 11, painting 10 of 13.

The six not aimed, each checked on the overlay:
- T-goph1-scarf, T-goph1-broom (van Gogh): the mask is on the prisoner beside the target.
- T-rambrandt2-cap ("the man on the far left"): the target is cut by the frame edge; the
  mask takes the leftmost whole figure instead.
- T-alps2-cap: scattered patches, mostly on the girl to the left of the target.
- T-hsm2-hat: a few dots on the target's head (0.02% of the image), class `empty`.
- **T-kon1-blazer is a scorer artifact:** the mask is on the right girl, but her box
  overlaps a sibling's, so `in_S` = 0.21 and the class is `both`. Excluded anyway, per
  the frozen rule (statistics use class `target`).

Paired gap therefore uses 17 of 19 pairs (alps2 and kon1 drop out). Several aimed
masks cover only a garment or a small part (controls naturally; among attach cases,
T-grad3-lei, -yama2-cat, -yama3-sunhat, -nemzeti-helmet, -spb1-bird cover under 60% of
the target box). Crop-box crops around the mask, so for these its box is small; that is
part of what the arm measures, not a reason to change anything now.

### Arm T judging — reduced to the primary arm (2026-11-20, decided after 30 panels, before scoring)

The full sitting is 240 panels. The reading uses only the fresh crop-box arm (plus the
plain arm's verdict on crop-box `no_object` cases), so after the user judged panels
0000–0029 (8 of them crop-box) the rest of the sitting is cut to the 62 remaining
fresh crop-box panels (`runs/fresh/attach_blind/todo_round2.txt`), then the plain panels
of the cases crop-box judges `no_object`. Paste-back and the RefEdit panels stay
unjudged beyond the first 30 and are not reported. Cost to blinding: the user now knows
the listed panels share one arm, though not which. The verdicts already given stand;
`limb` was missing on 29 of the 30 and is filled in afterwards.

### Arm T — result (2026-11-20, `attach_audit.py --score runs/fresh/attach_blind --aiming runs/fresh/attach_sa2va.jsonl`)

All 70 fresh crop-box panels judged. **Pre-registered reading: mixed: report the cases.**

| statistic (crop-box, Sa2VA-aimed) | value | band line |
|---|---|---|
| A, attach edit-right (clean+off_target+degraded) | 29/46 = 63% (clean 23, off_target 2, degraded 4) | ≥70% not a residual; ≤50% residual |
| F, owner+anatomy share of the 17 failures | 12/17 = 71% (wrong_owner 8, anatomy 4) | ≥50% |
| Z, no_object share of failures | 2/17 = 12% | ≥50% |
| paired gap, in-body control minus attach (17 pairs) | +47% | ≥25 |

A sits between the two lines, so neither "not a residual" nor "attachment-specific"
applies; F and the gap clear their bars but the rule does not let them override A.
Remaining failures: identity 3.

- **Robustness of the band:** 4 failures re-judged as passes would reach 70%; 6 passes
  re-judged as failures would reach 50%. The 95% Wilson interval on 29/46 is 49–75%, so
  it covers both lines; the band is the pre-registered answer, not a tight estimate.
- **Paired result (descriptive, not pre-registered):** over 17 pairs, 8 attach-only
  failures and 0 control-only failures (exact two-sided sign test p = 0.008). All 18
  aimed in-body controls are edit-right. On the same person in the same image, klein's
  crop-box edit changes clothing reliably and fails to attach an object about half the
  time. This is the strongest thing in the data; it sits beside the band, not over it.
- **Hand split (planned, descriptive):** crowded 7/14, free 11/17, untagged 11/15.
- **Post hoc, not planned:** hold 18/31, wear 6/9, on-body 5/6; photo 17/25,
  illustration 6/11, painting 6/10. Flags fail in 5 of 6 cases (four `wrong_owner`, one
  `anatomy`). These are observations for discussion, not findings.
- **Plain arm on the `no_object` cases:** panels 0101 and 0128 were not judged. The
  scorer's "plain drew the object in 0/2" is its default for missing verdicts, not a
  result. It cannot change the reading: plain_drew matters only when Z ≥ 50%.
- **`limb` was not collected.** Only panel 0028 has a label (`added`; the user wrote
  "3 arms" in the limb field, moved to `notes` by me with the user's sense kept; backup
  at `runs/fresh/attach_blind/ledger.before_0028fix.json`). I had proposed counting
  blanks as `unchanged` and writing that rule down before scoring; I scored first, and
  the rule is dropped: three `anatomy` panels (0068, 0216, 0229) would have been
  recorded as unchanged limbs. No outcome depends on limb.
- Not reported, per the reduced plan: the paste, plain and RefEdit rows the scorer
  prints (a partial, non-random sample from the first 30 panels).

**The 17 failures, viewed (descriptions only; verdicts are the user's):**

| panel | case | verdict | what the panel shows |
|---|---|---|---|
| 0042 | airforce-flag | wrong_owner | Flag hangs low between the target and the cadet to her right; unclear whose hand holds it. Not folded. |
| 0051 | army-flag | wrong_owner | Small flag raised beside the target's shoulder; holding hand ambiguous. Visible paste seam on the road below the box. |
| 0161 | alps4-flag | wrong_owner | A **new person** appears right of the target holding a large red flag; the target holds nothing. |
| 0073 | nemzeti-flag | wrong_owner | Red flag held by a white figure at the frame edge, left of the box; the boxed officer is unchanged. |
| 0129 | optimus-cape | wrong_owner | A **new caped robot** is drawn above the target; the yellow robot has no cape. |
| 0030 | kon1-tea | wrong_owner | The row is redrawn: a long-black-haired girl appears in position 1 holding the tea; the target's own pose changes. |
| 0066 | hsm1-balloon | wrong_owner | Balloon string held by a hand between the target and the blonde girl to her right; not in the target's raised hands. |
| 0053 | tsurugidake-puppy | wrong_owner | Dog sits on the rocks beside the woman's knee, not on it. |
| 0028 | alps6-guitar | anatomy | Target plays the guitar while still hugging both friends: three arms. |
| 0068 | yama4-flag | anatomy | Flag in a raised right hand; the original V-sign arm is gone. |
| 0216 | alps5-mug | anatomy | Steaming mug held; the target's arms are redrawn. |
| 0229 | yama1-umbrella | anatomy | Umbrella held; the hair changes from grey to a silver bob (user notes identity too). |
| 0016 | nba-headband | identity | Headband present; the black Bucks #34 becomes a white #34 jersey (the instruction's "white" is ambiguous). |
| 0149 | davinci1-hat | identity | Red hat on the man; his face and beard are redrawn in a different style. |
| 0183 | yama3-icecream | identity | Cone held; the camera is gone and the face changes (user: "mild, but noticeable"). |
| 0059 | rambrandt1-lantern | no_object | **A lit lantern is visible**, raised in a black-sleeved arm just left of the captain's head. |
| 0124 | davinci1-candle | no_object | **A lit candle is visible**, held by a re-drawn kneeling old man who replaces the original figure. |

Two observations from viewing, not re-judgments:
1. The two `no_object` panels both show the object. I've asked the user to recheck them.
   Neither recheck can move the band: even if both became passes, A = 31/46 = 67%,
   still mixed. Any failing verdict leaves A at 63%. Z would fall from 12% toward 0.
2. Three of the eight `wrong_owner` cases put the object on a **newly drawn figure**:
   0161, 0129 and probably 0073. In 0030 the group is redrawn. The new holder appears
   inside the crop-box's padded crop, not on a sibling who was already there. So part of
   `wrong_owner` here is klein inventing a holder, which is partly a crop-box mechanism.
   Five wrong_owner cases are unpaired (0030, 0051, 0053, 0073, 0129), so the paired
   design does not control for this in them.

What attachment means for the method is a TRAINING_DESIGN question for the user. No
follow-up run is proposed here.

**Judge's impressions (the user, given after scoring; qualitative, not counted):**
1. Quality: no big problems, but sometimes the light and style of the edit don't blend
   with the original. Seen in viewing: a paste seam on the road in 0051; faces and
   figures redrawn in a different style in the paintings (0124, 0149).
2. Spatial: sometimes left and right are confused. The rubric doesn't score which hand
   holds the object. 7 attach instructions name a hand (0009, 0042, 0051, 0059, 0135,
   0178, 0183). Of the 4 I viewed, 3 put the object in the other hand, from the
   person's point of view: 0051, 0059 and 0183. 0042 is correct. The other 3 weren't
   checked for side.
3. Boundary: the crop box recolours siblings inside it (by construction; this is why
   off_target counts as edit-right). Separately, when part of an arm is covered by
   something, the model seems to ignore the hidden rest and modify only the visible
   stem.
4. Invention: the edit creates figures or people from nowhere (0161, 0129, probably 0073
   above), and changes colours without being told to. Panel 0006, plain arm, umbrella
   instruction: another person was recoloured.

None of these change the reading. Items 2 and 4 are failure modes the 7-label rubric
folds into `wrong_owner`/`identity`/`clean`. Any future rubric should score hand side
and invented figures separately.

### Arm T — limb labels (2026-11-20, proposed by Claude, **pending the user's recheck**)

At the user's request I labelled `limb` on the 51 fresh crop-box attach panels; 0028
(`added`) is the user's. Method: zoomed source/edit crops around the target box, not
blind to verdicts (I had seen the 17 failures). In `ledger.json` each of mine carries
`"limb_by": "claude"` and a `limb_note`. Four are marked uncertain in the note: 0024,
0059, 0142, 0216. Backup before writing: `ledger.before_limb.json`. Limb is descriptive and
outside the pre-registered reading, which is unchanged.

All 51: unchanged 24, moved 17, other 7, added 2, removed 1. Against the user's
verdicts, aimed cases only (46):

| limb | edit-right | failures |
|---|---|---|
| unchanged | 16/20 | identity 2, wrong_owner 2 |
| moved | 12/17 | anatomy 2, wrong_owner 1, no_object 1, identity 1 |
| other (new or redrawn figure) | 0/6 | wrong_owner 5, no_object 1 |
| added | 1/2 | anatomy 1 (0028); 0083 judged clean |
| removed | 0/1 | anatomy 1 (0068) |

If the recheck holds, the expected "adds a limb instead of moving one" pattern is rare
here: 2 of 51 panels. When an arm has to change, klein mostly moves it (17), and usually
gets that right (12/17). The failure that shows up is different: klein **draws a new
or redrawn figure to hold the object** (`other`, 6 aimed, all failures, 5 of them
`wrong_owner`). This is post hoc and rests on one labeller's reading.

### Arm T — reconciliation with the parallel write-up (commit e0f200f)

Another session scored the same ledger without `runs/fresh/attach_sa2va.jsonl`, so it
counted all 51 attach cases as aimed: **A 31/51 = 61%, F 70%, Z 10%, gap +53 over 19
pairs, discordant 10 vs 0 (p ≈ 0.002); reading mixed.** The frozen rule counts only
Sa2VA class `target`, so the aimed figures above (A 29/46 = 63%) are the
pre-registered ones. Both readings agree.

That session also recorded an instruction error, from the user: in the airforce
image "the instruction itself is wrong; the female cadet is front right". The pair is
excluded with that reason and not re-run. Aimed, without the airforce pair: **A 29/45 =
64%, F 69%, gap +44 over 16 pairs, discordant 7 vs 0.** Same reading. Its two "owed"
items are now covered: the aiming file is used above, and limb labels are proposed
(pending the user's recheck). Its copy of the user's qualitative notes duplicates the
judge's impressions above and is not repeated.
