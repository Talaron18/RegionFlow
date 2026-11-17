# Referent-grounded editing on FLUX.2 klein — training design (draft)

> **Superseded (2026-11-18).** This design assumed aiming was the open problem. The
> predicted-mask results in `MODERN_BASE_EVAL.md` (steps 4–5: Sa2VA finds the target in
> 137/151; its paste-back pipeline sits about 3/40 below oracle paste-back) put aiming in
> the "solved on this benchmark" band fixed before the run. Do not build from this file
> unless an uncontaminated test reopens aiming. Kept for the record.

Status: **design draft for discussion. Nothing here is built or run, and nothing is
frozen yet.** It becomes a scope (with thresholds fixed) only after the two results it
still waits on, listed at the end.

## What the method is for

A single-pass editor that takes an image and a referring instruction, with no mask at
inference, and edits only the instance referred to. Trained with masks; run without.

Measured on the plain base (`MODERN_BASE_EVAL.md`, `GROUNDING_DIAGNOSTIC.md`):

| failure | how often | signature | what it calls for |
|---|---|---|---|
| mis-aimed: the edit lands on a sibling or elsewhere | about 28% of cases | attention does not favour the target | grounding |
| leak: the target is edited and a sibling changes too | roughly half of cases (estimate) | attention favours the target, by a small margin | preservation, possibly also grounding |

What it has to beat: plain klein (about 30% clean, weighted), and the mask-given
pipelines (paste-back about 67%; crop on hold pending its re-audit). The fair
competitor for a mask-free method is the predicted-mask pipeline, not yet run.

## Model

- Base: `black-forest-labs/FLUX.2-klein-base-4B` (undistilled), frozen.
- Trainable: LoRA on the attention projections of the transformer only. Rank and
  target blocks to be fixed in the scope; start small (rank 16).
- No architecture change, no new input. The mask is supervision, never an input.
- Text encoder and VAE frozen; their outputs cached, as in Gate A.

## Losses

1. **Flow matching, the base's own objective.** Noise-start velocity prediction on
   (source, instruction, target) triplets. No displacement regression, no one-step
   readout. This is the guard against the Gate A blur.
2. **Sibling preservation.** On the region of every same-category non-target instance,
   penalise the difference between the predicted clean image (or velocity) and the
   source. Mask-weighted, zero elsewhere.
3. **Grounding.** In chosen blocks, push the attention from the instruction's content
   tokens to the reference tokens toward the target instance and away from siblings.
   Form to be decided (margin on target-vs-sibling density, as the probe measures it,
   is the direct one). The probe found the signal strongest around the middle
   single-stream blocks, but that choice was exploratory; the scope must either
   supervise all blocks or justify a subset on held-out cases.
4. **Retention (teacher).** On ordinary edits and text-to-image prompts where the base
   is already right, keep the adapted model's velocity close to the frozen base's. One
   extra no-grad forward of the base per such sample. **Applied only to samples
   without same-category siblings**, so it cannot preserve the leakage it would
   otherwise protect.

Weights between the four are unknowns. The scope fixes them from a small pilot on a
dev split, never on the 151 benchmark cases.

## Data

- **Counterfactual referent pairs (the core).** Same image, same edit, different
  referent among same-category instances; the two targets differ. The loss cannot be
  lowered without resolving the reference.
- **Where targets come from: the mask-given pipeline as teacher.** Source images with
  two or more same-category instances and instance masks (COCO); for each instance,
  an edit of that instance alone produced by the pipeline that already works with a
  mask, kept only if automatic checks pass (target changed, siblings unchanged).
  Which pipeline (paste-back or crop) waits on the crop re-audit.
- **Referring expressions** for each instance: RefCOCO-family annotations where they
  exist. To check: licence, and overlap with the benchmark (below).
- **No overlap with the evaluation set.** The 151 cases are RefEdit-Bench × COCO
  train2014. Every COCO image id in the benchmark, and ideally every RefCOCO image
  used by RefEdit-Bench, is excluded from training. Count the overlap before building.
- **Retention samples:** ordinary single-object edits and plain prompts, with the
  base's own prediction as the target (no labels).
- A dev split, disjoint from both training and the 151, for every tuning decision.

Risk to state up front: the teacher's outputs carry the teacher's flaws (paste seams,
clipped additions). The student may learn them. Additions that belong beside the
object are the known weak spot of an instance-mask teacher.

## Guards carried over from the Gate A failure

1. Noise-start objective only (above).
2. **Conditioning-use check at a few hundred steps:** swap the referent within a
   counterfactual pair and confirm the output moves to the other instance. Gate A ran
   8,000 steps before the equivalent check showed the instruction was ignored.
3. **Trivial baselines scored on every metric first:** base with no adapter; "return
   the source unchanged" (which a preservation loss could collapse to); the teacher
   pipeline itself.
4. **A fixed synthetic test:** two same-colour shapes, "make the left one blue", read
   by eye and by the in-mask colour check.
5. **Each auxiliary loss paired with the failure it could cause:** preservation with
   a "did the edit still happen" rate; grounding with image quality on the audit.
6. Kill criteria written before each run; no new diagnostic in answer to a negative
   without a stated stopping rule.

## Evaluation

- **Primary:** blind audit on the benchmark (clean rate), same rubric, against plain
  klein and the predicted-mask pipeline. The automatic judge cannot see sibling
  leakage and is not used for this claim.
- **Where:** mis-targeted rate and sibling excess from the existing metric; the
  attention probe's target preference before and after (does the margin widen, and
  does that track the outcome as the diagnostic predicts).
- **Retention, pass/fail, fixed in the scope:**
  - general editing: a sample of PIE-Bench, base vs adapter-on, same seeds;
  - generation: a fixed prompt and seed list, base vs adapter-on;
  - rule: adapter-on not meaningfully worse than base on either (margin to be fixed),
    plus a direct similarity of adapter-on outputs to base outputs on these sets.
  - If it fails: use the adapter selectively (the probe's weak-margin signal is the
    candidate switch) and report that, not an always-on claim.
- **Transfer to the distilled model:** train on base-4B, test the same LoRA on the
  4-step `FLUX.2-klein-4B`. Reported separately; not assumed.
- **Ablations the paper needs:** each loss removed in turn; pairs vs unpaired data of
  the same size; teacher pipeline choice.

## What a first run would be (to be scoped, not yet)

A pilot, not the result: a few thousand pairs, flow matching + preservation +
retention, grounding added as the second arm, the step-300 conditioning check as the
gate, evaluated on the dev split only. Its purpose is to find loss weights and to
learn whether training fits in 24 GB at all (unverified: LoRA training on base-4B on
a 3090 Ti is reported by third parties, not tested here).

## Waiting on

1. The crop re-audit (20 panels), which decides the teacher pipeline.
2. The predicted-mask pipeline run, which sets the real bar for a mask-free method
   and says whether "finding the instance" is already solved on this benchmark.

## Not yet verified

That klein-base-4B LoRA training fits in 24 GB; how attention can be supervised
through diffusers' fused attention call without a large slowdown (the probe recomputes
weights outside the kernel; training needs them inside the graph); RefCOCO licence and
overlap with RefEdit-Bench; that a base-4B LoRA transfers to the distilled checkpoint;
novelty of the combination beyond the search-level check in `RELATED_WORK.md`.
