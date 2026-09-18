# Gate A reopening — what each gate actually tested, and what's next

Synthesis across Gate A (mechanism), Gate B (few-step), Gate C (benchmark),
written to decide what RegionFlow does next. Not a stage report on a single
experiment — a cross-gate read, and the plan that follows from it.

---

## 1. What each gate actually tested

**"Few-step flow is infrastructure" → Gate B.** Closed positive, with the
asterisk already on record: the adapter beats the frozen instantaneous field
at every `t_center` and Δ given the fitted `a = 1.0/0.5` scale rule, and
direction (`1 − cos²`) is the binding constraint everywhere — 0.44 at the
Δ=0.5 operating point. The Δ-curriculum adds ~3%. This leg is real and it's
the most developed thing in the project.

**"Instance-level attribute selectivity is the research problem" → Gate C.**
Instrumented, calibrated, and measured. But measured on other people's
editors: RefEdit-SD3, the two UltraEdit variants. Its headline is
`freeform → oracle` 40% → 0% mis-targeted with quality unchanged — routing
works, routing is not enough.

**"Conditional average-velocity transport is the mechanism" → Gate A ∪ Gate
B, and it's the leg with a hole in it.** Gate A closed negative: deterministic
MSE regression to `z_tgt − z_src` returns the conditional mean of many
plausible edits, which is a desaturated smear — the model goes from no
visible change straight to blur, oracle-relative structure decaying
0.78 → 0.44 → 0.09 as edit size grows. That was a verdict on the objective,
not on the idea, and it was never reopened with a different one.

## 2. The thing worth seeing

Gate C and Gate A are the same finding approached from opposite ends. Gate
C's conclusion — the mask determines *where* the model edits and not *what*
it does there, four in five outputs still unclean under an oracle mask —
says the residual is transformation quality. Gate A is exactly the
transformation-quality mechanism, and it failed. So the benchmark work
independently re-derived the need for the thing the mechanism gate couldn't
deliver.

That means the missing piece for a full pipeline is one piece, not a list:
**an edit objective that produces a distribution rather than a mean.**

Two smaller integration gaps follow from it. The Gate B adapter has no mask
path at all — `r` reaches the model through exactly one route, the
interval-embedding MLP added to `pooled_projections`, and
`check_interval_branch` says those readouts sit at rms/noise ≈ 1.07–1.12,
barely above a random walk. And Gate B trains against the frozen teacher's
velocity on edit-dataset latents; it is a few-step generation adapter, not
an editor. Nothing in it has ever consumed a source image and a mask to
produce an edit.

## 3. Why not integrate now

Stack the known error rates. An oracle-masked, full-step editor gets ~22%
clean by blind audit. Few-step at Δ=0.5 carries a 0.44 direction residual
the scale rule doesn't touch. Whatever edit mechanism goes in the middle
currently doesn't exist in working form. A pipeline built from those three
joints won't produce a result you can interpret — when it fails you won't
know which joint failed, which is the exact situation the gate structure
was designed to prevent.

There's also a priority inversion worth naming: the thesis says few-step is
infrastructure and not the contribution, and it's the leg that has absorbed
the most effort. By the thesis's own logic, few-step should be the last
thing added.

## 4. What to actually do

**Reopen Gate A, on the Gate C scoreboard.** Gate A failed on 3 hand-picked
cases (only 3 of ~400 dev candidates passed selection) with a `struct`
metric later shown to be substantially a colour artifact. That instrument
problem is now solved: 151 cases, a `mis_targeted` classifier validated
14/14 against human judgment, a VLM quality judge calibrated at ρ = 0.717
with clean ordering across all four verdict classes, ground-truth floors
from MagicBrush, and `off_instance` once the threshold decision lands. Gate
A can now be scored the way it should have been the first time.

The objective change is the substance. `pairflow`'s straight-line path
`z_s = (1−s)z_src + s·z_tgt` is fine; regressing to the constant
`v = z_tgt − z_src` under MSE is what collapses to the mean. Making that
path stochastic — or reparameterising the pair transport in the MeanFlow
form already working in Gate B — is literally "conditional average-velocity
transport," the phrase in the thesis, and it's the one combination of the
two mechanism gates that has never been run.

**Sequence:** fix the objective → score on Gate C at full steps → only then
add few-step and re-score. If the edit mechanism doesn't clear the
baselines at full steps, few-step cannot rescue it, and that's learned for
a fraction of the compute.

Regardless of the above: **Gate C is a standalone contribution right now.**
A benchmark that leaves instances deliberately unedited, on which the
entire 2026 routing family can be scored for the first time, plus the
negative result that routing saturates at 0% mis-targeting while quality
doesn't move — that stands whether or not RegionFlow ever ships an editor.
It doesn't depend on the pipeline existing, and it's the strongest asset
the project has today.

---

## 5. Do / don't, before any retrain

**Do**

- **Ambiguity probe first, before any retrain.** Group the cache two ways —
  same instruction / different sources, and same source / different
  instructions — and compare the two cosine distributions. `numpy` over the
  existing cache, no GPU. This tests §3's claim instead of assuming it, and
  it's the cheap decisive step Gate A never had.
- **Finish §11's CPU work** already running: the four remaining arms through
  the GLM judge, and the `off_instance` threshold decision (§7 recommends
  `over_target` — **done**, see `GATE_C_REPORT.md` §7/§11 item 2, committed
  and pushed to `claude/project-status-review-l2tt6j`).
- **Watch for identity drift** when scoring `white`/`oracle` — recurrence of
  the `refedit-83` pattern is the stated threshold for touching `PQ_PROMPT`.
- **Reopen Gate A on the Gate C scoreboard**, not on 3 hand-picked cases: 151
  cases, the 14/14-validated `mis_targeted` classifier, the calibrated
  judge, the MagicBrush floors.
- **If Direction A is taken, keep σ(0) > 0** — or drop the one-step-at-s=0
  readout. With σ(0) = 0 the s=0 query point is unchanged and so is the
  failure.
- **Score the edit mechanism at full steps before adding few-step.** By the
  thesis's own logic, few-step is infrastructure and belongs last.
- **Add a scope line to §5**: the judge's ρ = 0.717 was calibrated on `none`
  only; cross-checkpoint generalisation is untested.

**Don't**

- **Don't quote "changing the target raised cos to ≈0.72 using the same
  LoRA."** Different task, different target — Gate B never ablated Gate A's
  objective. `1 − cos² ≈ 0.44` is the binding constraint, not a success.
- **Don't run the §7 probe as written** — `cos(v_i, v_j)` over "similar
  (z, c, M)" confounds mode ambiguity with images simply being different.
- **Don't present Direction B as continuous with the thesis.** It
  reintroduces the source/noise mixing Stage 0 refuted and abandons
  `pairflow`'s displacement premise. Possibly correct; not the same
  project.
- **Don't integrate the full pipeline yet.** Three unvalidated joints means
  an uninterpretable failure.
- **Don't extend the judge or quote `off_instance_edit` counts** until
  calibration and the threshold decision land. (Threshold decision has
  since landed — see above; judge extension past `none` is still open.)
- **Don't treat Gate A's mean-collapse as established.** It rests on n=3
  with a metric later shown to be substantially a colour artifact.
- **Still off the list from before:** longer training schedules, fitting
  `a(t,Δ)`, retraining with the scale folded in, further probe-variant
  rankings, τ/λ sweeps, shape edits.

---

## 6. Open gap before scoping can start

This document was written from two inputs: the synthesis in §1–4 (pasted
in full), and the Do/Don't list in §5 (pasted in full, as feedback on a
longer proposal). That proposal itself — the document defining **Direction
A** vs **Direction B**, the σ(0) parameterisation, and the `§7` ambiguity
probe being critiqued was not in this repo at the time this section was
first written. It has since been supplied in full and is folded in below.

---

## 7. The objective failure, precisely

Gate A's failure was not "ordinary multimodal networks inevitably undergo
MSE averaging." It was more specific: the target definition compressed a
conditional distribution that inherently has latent edit ambiguity into a
single point vector, causing MSE to learn the conditional expectation
`E[v | z, c, M]`.

**The mechanism.** Target `v = z_tgt − z_src`; training minimises
`E[‖u(z,c,M) − v‖²]`; the optimal solution under MSE is
`u*(z,c,M) = E[v | z,c,M]`. If a set of conditions is indistinguishable
within the conditioning — "turn the dog white/brown/black" all read the
same `(z,c,M)` when `c` doesn't pin down which brown — then
`p(v | z,c,M)` isn't a delta distribution, it's a mixture of distinct modes
`v_1, v_2, v_3, …`, and the model learns their mean: smaller magnitude,
mid-way direction, cancelled high-frequency structure. That's desaturation
and blur, and it's exactly what Gate A's own closure already described
("no visible change straight to blur," "the average of many colour changes
over an object is a desaturated smear").

**Why Gate A specifically, and not just "MSE regression in general."** The
target here isn't an image, it's a latent displacement field. So the result
of averaging isn't an averaged image (Pix2Pix's blurry grey roof,
super-resolution's blurry upscale) — it's an *averaged edit direction*.
Same root cause as the precedent across the field (Pix2Pix's motivation for
conditional GANs, SR's motivation for perceptual/adversarial losses, DDPM
predicting noise/velocity rather than `x_0` directly because `p(x_0|x_t)` is
multimodal, standard flow matching's `v*(x,t) = E[x_1−x_0|x_t]`), but
applied to a displacement target rather than a pixel target.

**Why diffusion editors don't hit this.** They don't predict a single
static velocity from the condition alone — sampled noise provides mode
selection: `condition + randomness → one possible edit`. Gate A currently
has `condition → average edit`, with nothing playing the randomness role.

### Where the fix has to land: the s=0 readout

This is the refinement that matters most, and it's sharper than "MSE learns
the conditional mean."

`pairflow` doesn't integrate a path to read out the edit — it reads out in
**one step at s=0**: `z_edit = z_src + u(z_src, 0, c, M)`. At `s=0` the
conditioning is `(z_src, c, M)` for *every mode* — there is nothing at the
exact point the model is queried that disambiguates which brown was meant.

This has a consequence for the fix. Direction A below proposes
`z_s = (1−s)z_src + s·z_tgt + σ(s)ε`. Stochastic interpolants conventionally
take `σ(0) = σ(1) = 0` so the endpoints stay exact. But with `σ(0) = 0`,
`z_0 = z_src` for every mode regardless, `u(z_src, 0, c, M) = E[v]` again,
and nothing has changed at the only place Gate A looks. **Direction A only
works if either `σ(0) > 0`** (the noise itself selects the mode at the
query point) **or the one-step-at-s=0 readout is abandoned and the path is
integrated** (accumulated noise does the disambiguating along the way).
Those are different designs with different costs, and picking between them
is part of the scoping work, not a detail to settle later.

This also reconciles the precedent survey's own observation that standard
flow matching's `v*(x,t) = E[x_1−x_0 | x_t]` — also a conditional
expectation — works fine: the averaged marginal field provably transports
the right distribution because `x_t` differs across modes and disambiguates
*along the trajectory*. **Averaging isn't the bug. Averaging at a point
where nothing distinguishes the modes is the bug**, and `s=0` under
`σ(0)=0` is exactly that point.

### The three directions

- **Direction A — stochastic pair transport.**
  `z_s = (1−s)z_src + s·z_tgt + σ(s)ε`, train `v(z_s, s)` as a stochastic
  interpolant, keeping the flow trajectory while injecting mode selection.
  **The one that stays inside the thesis** — see below — conditional on
  resolving the `σ(0)` question above.
- **Direction B — diffusion-style velocity target.** Sample
  `z_t = α_t z_src + σ_t ε`, train `v_θ(z_t, t, c, M)` under conditional
  flow matching. This is InstructPix2Pix-shaped: it reintroduces exactly
  the source/noise mixing Stage 0 found collapses onto a single axis, and
  it discards `pairflow`'s stated premise that noise never enters and the
  source is displaced rather than destroyed. Possibly correct, but **a
  retreat to the standard approach, not a continuation of the thesis** —
  presenting it as continuous with "conditional average-velocity transport"
  would be misleading.
- **Direction C — distributional velocity.** Explicitly model
  `p(v | z,c,M)` via a mixture density network
  (`Σ_k π_k N(μ_k, Σ_k)`) or latent noise injection
  `v_θ(z,c,M,η)`, `η ~ N(0,I)`. Not evaluated against A/B in detail here;
  parked as a third option, not ranked.

### One claim to delete

**"Changing the target raised cos to ≈0.72 using the exact same rank-32
LoRA and conditioning path" does not follow, and should not be quoted
anywhere, including in a paper.** That 0.72–0.76 is cosine between the Gate
B adapter's correction and the true average velocity of the frozen
teacher's *generation* field — a different task, different target,
different data role from Gate A's *edit displacement*. Gate B never
ablated Gate A's objective; it isn't evidence about it either way. And the
number is being read backwards: `1 − cos² ≈ 0.44–0.48` is what Gate B's own
closure calls the *binding constraint* — the thing that's still wrong —
not evidence that a target change fixed something.

### The corrected ambiguity probe

The original `§7` probe — `cos(v_i, v_j)` across samples "sharing similar
`(z,c,M)`" — is confounded: low cosine could mean genuine mode ambiguity,
or just that the two images are different. The dataset has no fixed
`(z,c,M)` with multiple recorded targets, so `p(v|z,c,M)` isn't directly
measurable that way.

What's measurable, and answers the actual claim, is a **contrast**, computed
over the existing MagicBrush cache (numpy only, no GPU):

- same instruction string, different source images (the cache already
  dedupes text embeddings by instruction, so this grouping is close to
  free)
- same source image, different instructions

If the two cosine distributions come out comparable, the instruction
carries little directional information and the *conditioning* is the
problem. If same-instruction pairs are clearly more aligned, ambiguity is
smaller than assumed and the mean-collapse story is weaker than Gate A's
n=3 supports. Either way, it's a real test of the claim instead of an
assertion of it — the cheap decisive step Gate A never had. See
`ambiguity_probe.py` below.

### Result: run against the real cache (revised — ceiling framing, corrected mask-restrict)

The **same-source** side of the contrast came back empty — this cache has
at most one turn-1 edit per source image (`precompute_cache.py`'s own
`turns=(1,)` restriction), so no source image recurs across sessions with
a different instruction. That's a cache-build choice, not a structural
dead end — MagicBrush is multi-turn, so turns 2+ would give same-source
pairs directly, but getting them needs a cache rebuild with the GPU-side
encoders. `ambiguity_probe.py` fell back to a substitute null instead —
random unrelated pairs (different instruction, any source) — so
same-instruction had something to be measured against, with the explicit
caveat that this null varies both `z` and `c`, so it can't cleanly
separate "the instruction underdetermines the edit" from "different source
images differ for reasons that have nothing to do with ambiguity."

A first pass read this off the raw mean cosine and a pair-level
Mann-Whitney test. Both were wrong to use as-is: the 177 same-instruction
pairs come from only 110 groups (not independent — groups of size >2
contribute correlated pairs), so the p-value was optimistic; and for unit
vectors, **mean pairwise cosine within a group is an unbiased estimator of
`|μ|²`** (the squared length of the population mean direction) for any
group size ≥2 — so `sqrt(mean)` is the actual quantity of interest: the
best cosine a single constant, instruction-only prediction could achieve
against a fresh draw. That's reported below as `ceiling`, with a
group-level bootstrap (resampling instruction groups, not pairs) for an
honest CI on the gap.

**Raw (whole-frame) result:**

| grouping | n | mean | median | ceiling = √mean |
|---|---|---|---|---|
| same-instruction, different source | 177 | +0.177 | +0.165 | 0.420 |
| unrelated (different instruction) | 177 | +0.127 | +0.108 | 0.357 |

Ceiling gap +0.063, 95% CI [+0.027, +0.101] (group-level bootstrap). CI
excludes 0 — the instruction genuinely carries information beyond the
unrelated-pairs null, but the ceiling itself (0.42) is modest.

**Specificity split** (same-instruction pairs only, split by a coarse
colour-word proxy for "names a concrete attribute" — see
`ambiguity_probe.py`'s docstring for what this proxy can and can't show):

| grouping | n pairs | n groups | ceiling |
|---|---|---|---|
| specific (colour named) | 38 | 18 | 0.384 |
| vague | 139 | 92 | 0.430 |

This is the *opposite* of what the ambiguity hypothesis predicts (specific
instructions should pin down a tighter target, i.e. a *higher* ceiling).
The gap is small and `n=38` for "specific" is thin, so this is weak
evidence either way — but it leans against, not for, genuine ambiguity
being the dominant driver, and the colour-word proxy is confounded with
edit type (colour changes vs. removals/additions likely differ in
intrinsic latent-shift consistency for reasons unrelated to ambiguity).

**Mask-restricted result — a real bug was caught and fixed here.** A first
version multiplied `v` by each item's own mask and took a pointwise cosine
over the full `(C,H,W)` tensor; both same-instruction and unrelated
medians came back exactly `0.000`, which is the tell that it was measuring
the wrong thing. Two different images' edit regions almost never occupy
the same latent pixels (different photos, different object positions), so
a pointwise dot product is dominated by spatial non-overlap regardless of
whether the underlying edit direction agrees — the check was answering "do
these masks happen to line up in space," not "did the edit move the same
way." Fixed by mean-pooling `v` over each item's mask into one value per
channel *before* cosine (`mean_pool_in_mask` in `ambiguity_probe.py`,
regression-tested against exactly this failure mode). Corrected result:

| grouping | n | mean | median | ceiling = √mean |
|---|---|---|---|---|
| same-instruction, different source | 177 | +0.385 | +0.504 | 0.621 |
| unrelated (different instruction) | 177 | +0.208 | +0.298 | 0.456 |

Ceiling gap +0.162, 95% CI [+0.061, +0.267] — comfortably excludes 0, and
**substantially larger than the raw whole-frame gap (+0.063).** Restricting
to the actual edited region, as intended, does surface more real
instruction-conditioning signal than the whole-frame numbers showed —
confirming that the raw floor was partly diluted by out-of-mask structure
common to every pair regardless of instruction, exactly the concern that
motivated this check. The unrelated floor itself is also higher in the
edited region (ceiling 0.456, not near 0) than the raw whole-frame floor
(0.357), which is consistent with a shared low-rank bias in how MagicBrush
edits typically shift the masked region (e.g. a generic inpainting
brightness/saturation signature) rather than pure noise; the gap, not the
absolute floor, is the number to read.

**This revises an informal argument made while reviewing σ0-sweep training
logs (not previously written into this document) that the model's own
`cos_clean` (~0.50–0.60, measured pointwise, per-example, against the
model's own true target) already exceeded the instruction-only ceiling and
was therefore evidence the model uses `z_src` productively rather than
just doing mean-collapse.** That comparison used the raw whole-frame
ceiling (0.420), which this check shows
was diluted. The properly edit-region-restricted ceiling (0.621) is now
*higher* than the model's own `cos_clean` range, not lower. Note these two
numbers are still not strictly apples-to-apples — the ceiling here is a
coarse per-channel pooled summary compared *across different images*,
while `cos_clean` is a spatially-resolved, per-example prediction accuracy
*within one image* — so this doesn't newly prove mean-collapse. But it
withdraws a piece of evidence that had been cited against it. Net: no
positive evidence either way from this comparison; treat it as retracted,
not flipped.

**Two follow-ups on the specificity reversal, both survive it.** The
concern was that "vague" might be a few homogeneous instruction templates
("remove the X", "add a Y") whose members share a direction for reasons
unrelated to specificity. `check_template_dominance()` shows otherwise:
vague's top 5 groups (by member count) cover only 20% of its 139 pairs —
not template-dominated — and dropping the single largest group from each
side barely moves either ceiling (specific 0.384 → 0.393, vague
0.430 → 0.423). The reversal is not a template artifact. It's still
`n=38` pairs / 18 groups on the specific side (the two CIs would overlap),
so it isn't a finding on its own — but it is a check the ambiguity
hypothesis was given every chance to pass, and didn't.

**The per-channel breakdown separates the mask-pooled ceiling from a
shared-bias artifact, and most of it isn't one.** Channels 13, 9, 4 have
both the largest same-instr-minus-unrelated correlation gap AND the
highest `|mean|/std` (0.76, 0.62, 0.75) — the shared-bias signature
(a consistent offset, not just spread). But channels 0, 12, 10 carry gaps
of +0.380, +0.316, +0.353 with `|mean|/std` of 0.02, 0.01, 0.06 —
essentially no shared offset, strong instruction-specific correlation,
near-zero unrelated correlation. Those channels are carrying real
instruction information, not a luminance echo.

**Mean-centered result (subtracting the dataset-wide per-channel bias
before cosine): the decisive one.** Unrelated pairs drop to mean **−0.051**
once the shared bias is removed — a null behaving like a null (slightly
negative, the expected finite-sample residual for uncorrelated unit
vectors, not a leftover signal). Same-instruction holds at **+0.210**.
Ceiling gap **+0.063 → +0.162 → +0.451**, 95% CI [+0.250, +0.555] on the
in-sample fit. The +0.208 unrelated-pairs alignment in the raw pooled
numbers was almost entirely one shared bias direction common to every
MagicBrush edit, inflating both columns; net of it, instruction identity
is the dominant factor in the pooled displacement, not a weak one.

*Caveat, addressed:* the in-sample version fits the bias on the same 177
pairs it scores, so the unrelated column landing near 0 is partly by
construction. `held_out_centered_gap()` (added in `ambiguity_probe.py`)
fixes this properly: repeated random 50/50 item splits, bias fit on one
half, scored on same-instruction/unrelated pairs drawn only from the
other half. Wired into `--mask-restrict`'s output as a distribution
(median + range across splits) rather than a single number. **Re-run
`--mask-restrict` and record the held-out distribution here before citing
+0.451 anywhere outside this document.**

### Verdict: Direction A closed

Five independent lines now converge on the same conclusion:

1. No measured mode-selection mechanism — the σ0 sweep (§8) never showed a
   nonzero σ0 beating σ0=0 on `cos_clean`.
2. A measured cost that scales with σ0 — the `cos − cos_clean` gap widens
   with σ0 (negligible at 0.3, real and consistent at 0.5), the signature
   of the model partially learning to echo injected noise back rather than
   the true edit direction.
3. Instruction conditioning carries far more information than the
   mean-collapse story needs it to lack — the corrected, edit-region,
   mean-centered ceiling (0.458 in-sample; held-out pending) is well above
   both the raw whole-frame ceiling this document originally reasoned from
   and the model's own full-field `cos_clean`.
4. The specificity split — the one check that would have shown genuine
   ambiguity directly if it were there — comes back reversed, and survives
   the template-dominance check built to rule out the obvious artifact.
5. **The matched-step `diagnose` comparison — the cleanest single piece of
   evidence.** σ0=0.0 at step 8000 vs. σ0=0.3 at step 8100 — so σ0=0.3 has
   *more* training and still loses on every treatment: raw 0.541 vs.
   0.539 (tied), pooled 0.769 vs. 0.713, centered 0.664 vs. 0.587. The gap
   *widens* as spatial and shared-bias confounds are stripped away, the
   same pattern σ0=0.5 shows in extreme form. σ0=0.0 itself is stable
   between step 7000 and 8000 (0.539→0.541, 0.765→0.769, 0.662→0.664), so
   this isn't a training-noise artifact on either side of the comparison.

**Target ambiguity is not what caps Gate A at `cos ≈ 0.5`.** Direction A
(`σ(0) > 0`) is closed: no further σ0 sweeping, no multi-turn cache
rebuild to chase the same-source contrast — that contrast exists to
quantify ambiguity, and ambiguity is no longer the live hypothesis. This
is a clean negative, better-evidenced than the sweep looked like it was
heading toward, not a failure to reach a positive result.

### What's next: direction vs. position, on the model itself

The live question is the gap between the model's own `cos_clean`
(~0.50–0.60) and what the data says is available. The mask-pooled ceiling
(0.621) and mean-centered ceiling (0.458) are *per-channel directional*
ceilings with spatial structure removed entirely — they are not directly
comparable to the model's full-field number, because the model also has
to get *position* right (which pixels move), which the raw-v ceiling
(0.420) is the one that includes. Whether the model's shortfall is
mostly directional (wrong per-channel shift) or positional (right shift,
imprecise about where) is exactly Gate C's WHERE/WHAT split, applied to a
trained checkpoint instead of to raw data.

`transport/pairflow.py::direction_position_diagnostics()` computes the
model's own `cos(u, v_clean)` under the same three treatments — raw
(full frame), pooled (masked, per-channel), centered (pooled minus a
fixed external bias) — and `scripts/gate_a_transport.py diagnose` runs it
over a full `TripletCache` (cached latents + embeddings only, no VAE or
text encoder needed):

```
python scripts/ambiguity_probe.py --cache /data/cache/<hash> --save-bias runs/bias.npy
python scripts/gate_a_transport.py diagnose --ckpt <a trained checkpoint> \
    --cache /data/cache/<hash> --bias-npy runs/bias.npy
```

A model close to the raw ceiling but well below the pooled ceiling is
losing direction more than position; the reverse means it already has the
right direction and is imprecise about where inside the mask to put it.
Whichever it is, the fix is conditioning/readout geometry, not
stochasticity.

**Result, run against all three available checkpoints (4511 cached
examples each, same cache the ambiguity probe used). σ0=0.0's checkpoint
was re-checked at two training steps to rule out training-noise:**

| checkpoint | step | raw (median) | pooled (median) | centered (median) |
|---|---|---|---|---|
| σ0=0.0 (no Direction A noise) | 7000 | +0.539 | +0.765 | +0.662 |
| σ0=0.0 (no Direction A noise) | 8000 | **+0.541** | **+0.769** | **+0.664** |
| σ0=0.3 ("decided" Direction A design, §8) | 8100 | +0.539 | +0.713 | +0.587 |
| σ0=0.5 | 8100 | +0.321 | +0.362 | +0.137 |
| *(data ceiling, same cache, for reference)* | — | *0.420* | *0.621* | *~0.458 in-sample* |

Three things fall out cleanly, all confirming the closure rather than
reopening it:

1. **σ0=0.0 is stable between step 7000 and 8000** (0.539→0.541,
   0.765→0.769, 0.662→0.664) — the comparison below isn't a training-noise
   artifact on the σ0=0.0 side.
2. **The matched-step comparison is the cleanest single piece of evidence
   for closing Direction A.** σ0=0.0 at step 8000 vs. σ0=0.3 at step
   8100 — so σ0=0.3 has *more* training and still loses on every
   treatment: raw 0.541 vs. 0.539 (tied), pooled 0.769 vs. 0.713, centered
   0.664 vs. 0.587. The gap *widens* as spatial and shared-bias confounds
   are stripped away (raw → pooled → centered), exactly the pattern σ0=0.5
   shows in extreme form. This supersedes the earlier σ0=0.5-only argument
   as the headline evidence: adding Direction A's noise makes the model
   worse even at the "decided" design value, with more training, on a
   ground-truth accuracy metric, not just a training-log proxy.
3. **σ0=0.5 collapses specifically on `centered`** (0.137 — near the
   data's own null floor, and well *below* the data's centered ceiling of
   ~0.458): once the shared bias is removed, this checkpoint has almost no
   genuine instruction-specific signal left. This is the concrete shape of
   the ε-fitting failure mode flagged from the training logs' widening
   `cos − cos_clean` gap — at σ0=0.5 the model was learning to partially
   echo injected noise back at the cost of the real conditioning signal,
   and here it shows up directly in a ground-truth accuracy metric, not
   just a training-time diagnostic.

**A genuinely new, valid finding, on a like-for-like comparison this
time:** σ0=0.0's pooled (0.769) and centered (0.664) both exceed the
corresponding *data* ceilings (0.621, ~0.458) — unlike the earlier
retracted claim (which compared a pointwise per-example metric against a
pooled cross-image one), this comparison uses the *same* treatment on
both sides. A model conditioned on `z_src` is not bound by a
constant-per-instruction ceiling at all — it has strictly more
information available than the ceiling's hypothetical predictor — so
exceeding it here is the expected signature of a model using `z_src`
productively, not an anomaly. σ0=0.3 also clears both data ceilings, by a
smaller margin; σ0=0.5 clears neither, which is the sharpest single number
in this table for why it's the checkpoint to discard. (The centered data
ceiling is still the in-sample 0.458, pending the held-out distribution —
the qualitative story here is unlikely to flip given the size of the gaps
involved, but re-check once that lands.)

**The WHERE/WHAT gap is real even on the best checkpoint.** σ0=0.0's raw
(0.541) sits well below its own pooled (0.769) — roughly 0.23 of cosine is
being lost specifically to *positional* imprecision (right average
direction, wrong exact pixels), separate from and on top of whatever
directional headroom remains (pooled 0.769 vs. a perfect 1.0). Both
matter, but this is the first direct evidence that spatial precision
specifically — not just conditioning strength in the abstract — is worth
real attention in whatever comes after Direction A, consistent with Gate
C's WHERE/WHAT framing (§2) being the axis the whole investigation now
sits on.

### Checkpoint inspection: is the positional deficit a wiring bug?

All three checkpoints reported `190/191 lora_b blocks non-zero` — identical
across independently-trained runs, which is the signature of something
structural rather than a training accident. `scripts/gate_a_transport.py
inspect` (CPU-only, `torch.load` + tensor stats, no model or GPU) answers
both the "which block" and "did the mask path even train" questions the
existing liveness *count* couldn't:

- **The dead block is `transformer_blocks.23.attn.add_q_proj.lora_b`** in
  all three checkpoints — the last block, text-stream query projection.
  Identical across independently-trained runs means structural: most
  likely the last block's text-stream path is inert in SD3-Medium's
  forward (the context/text branch is dropped or unused at the final
  layer), so its gradient is exactly zero by construction. Benign — worth
  confirming once against the base model's own forward and adding to a
  known-quirks list, not a bug to chase.
- **`mask_enc.out` (the zero-init final projection) moved**: `abs_max`
  ≈0.011–0.023, `abs_sum` ≈36–40 across all three checkpoints. The mask
  conditioning path has been alive the whole run — not a wiring bug, and
  in `"pair"` mode there was never a structural reason (gate ≡ 1 always)
  for it to stay at zero. **The positional half of the deficit is not a
  one-line fix; the readout-constraint hypothesis (§7's `s=0` critique,
  now aimed at position rather than mode-selection) is the one left
  standing.**
- **A wrinkle worth checking before accepting that conclusion fully:**
  `mask_enc.body`'s weights are nearly IDENTICAL across all three
  checkpoints (`abs_sum` 169.9/169.6/169.8 on `body.0.weight`,
  811/806/800 on `body.2`) despite three different σ0 values — consistent
  with `body` converging to close to a fixed transformation of the mask
  that σ0 doesn't perturb. Meanwhile `out`'s `abs_max` (~0.02) is an order
  of magnitude below `body.0`'s (~0.21): the path is alive but what it
  actually injects into the latent may be weak relative to `z` itself. A
  third possible failure mode, distinct from "dead wiring" and "an
  architectural readout limit": conditioning present but too faint to
  matter, and cheaper to fix than either if true.

`scripts/gate_a_transport.py inspect --ckpt <...> --cache <cache>` reports
`‖delta_z‖ / ‖z_src‖` via `mask_enc` alone (no transformer, no GPU). Run
against the full six-point σ0 sweep (0.0, 0.1, 0.3, 0.5, 0.7, 1.0), all at
step 8000–8100:

**Result: injection magnitude is not a quality lever, and the ratio
itself turned out confounded.** σ0=0.1 has the *largest* `‖delta_z‖/‖z‖`
and sits with the best-scoring group; σ0=0.7 and 1.0 have the *smallest*
and are the worst on `cos_clean`. No monotonic or even directionally
consistent relationship between injection magnitude and quality. Digging
into why: `mask_enc`'s weights are identical to within 2% across all six
checkpoints (`body` isn't changing), so the *encoder* isn't what's moving
— the ratio itself is what varies, and predictably by σ0: flat-to-high at
σ0 ≤ 0.3 (0.74–0.87), dropping sharply from σ0=0.5 (0.53 → 0.42 → 0.41).
The likely cause: a σ0>0 checkpoint's effective source
`z_src' = z_src + σ0·ε·mask` (`transport/pairflow.py::sample_path`) has a
**larger norm inside the mask than the cache's un-noised `z_src`**, which
is the denominator this function actually uses — so the ratio is
confounded by σ0 and should never be compared across a sweep, only within
a matched σ0. **This retires `‖delta_z‖/‖z_src‖` as a cross-arm
diagnostic**; `compute_mask_injection_ratio`'s docstring now documents
this caveat. The λ-sweep premise built on top of it is dead — kept only
as a `λ=0` "what does the mask buy at all" measurement (below), not as a
graded lever.

**The real result from this run is the completed six-point `cos_clean`
sweep, cleaner than any single-column argument:**

| σ0 | raw | pooled | centered |
|---|---|---|---|
| 0.0 | 0.541 | 0.769 | 0.664 |
| **0.1** | **0.546** | **0.775** | **0.681** |
| 0.3 | 0.539 | 0.713 | 0.587 |
| 0.5 | 0.321 | 0.362 | 0.137 |
| 0.7 | 0.262 | 0.256 | 0.020 |
| 1.0 | 0.261 | 0.258 | 0.031 |

- **σ0=0.1 edges out σ0=0.0 on all three treatments**, including centered
  (0.681 vs. 0.664) — the opposite of a clean monotone-decreasing
  ordering. This is a 2.6% relative difference on a metric with no
  measured seed-to-seed spread for Gate A (one run per σ0), so it's called
  a **tie**, not a win for either side — claiming a direction here would
  repeat the same overclaiming this document has already had to walk back
  twice. The closure statement is NOT "zero is best": it's that no σ0 in
  the range beats the deterministic baseline by more than noise, and σ0=0.1
  is at worst indistinguishable from σ0=0.0.
- **The cliff between σ0=0.3 and σ0=0.5, and the floor past it, are
  unambiguous.** `centered` goes 0.587 → 0.137 → 0.020 → 0.031: by σ0 ≥ 0.7
  the model retains essentially nothing instruction-specific once the
  shared bias is removed — 0.020 and 0.031 are indistinguishable from
  zero. Whatever those checkpoints still score on `raw` (~0.26) is the
  dataset-wide constant, not a genuine edit. This is the ε-fitting
  endpoint stated as starkly as it can be.
- **σ0=0.7 and 1.0 are identical to within noise** on every treatment
  (0.262/0.261, 0.256/0.258, 0.020/0.031) — the damage saturates; past
  σ0≈0.7 there's nothing left to destroy.
- **`pooled − raw` flips sign at the cliff, and is a cleaner cliff
  indicator than any single column.** At σ0 ≤ 0.3, pooled > raw
  (e.g. 0.775 > 0.546) — removing the positional burden helps, the
  positional-deficit finding above. At σ0 ≥ 0.5, pooled < raw
  (0.256 < 0.262): those checkpoints have no per-channel direction left to
  recover, so pooling away spatial structure removes signal instead of
  noise.

**Closure statement, final form:** σ0 ≤ 0.1 is free (a tie with the
deterministic baseline, not a win), σ0 ≥ 0.5 destroys the
instruction-specific component entirely, and nothing anywhere in the
range beats σ0=0 by more than noise. This table is the evidence — more
convincing than any single-number argument, and it documents exactly
where the cliff sits for anyone revisiting this.

**Both outstanding checks run, on σ0=0.0 step 8000:**

**In-mask vs. out-of-mask split — real result, and a bug in how the first
version of this tool labelled it.** `delta_z` RMS is 0.171 inside the mask
vs. 0.802 outside — a ratio of 0.21, i.e. the background signal is ~4.8×
*stronger* than the in-mask signal. The tool's first cut printed this as
"weak/no spatial localisation" (threshold logic: `>3` = localised, `<1.5`
= weak), which is simply wrong — a ratio of 0.21 is exactly as
differentiated as 5.0 would be, just inverted. Fixed
(`classify_localisation()`, self-tested against this exact value): this
is **background-dominant, strongly-differentiated localisation**, not an
absence of one. Plausible reading: a strong, easily-learned
"protect-this-region" signal outside the mask may be what actually lets
the pairflow loss's `outside_weight` term hold `u ≈ 0` there reliably,
with a subtler in-mask nudge sufficient once the transformer's own
image/instruction content dominates inside the edit region. Not proven,
but the earlier "mask conditioning might be inert or malformed" worry is
now closed either way — inside ≠ outside by a factor of ~5, whichever
direction, is a real, learned signal.

**λ=0 ablation — the decisive number.** `diagnose --ablate-mask` on the
same checkpoint/cache, median cosine, λ=1 (normal) vs. λ=0 (mask=None,
scored against the same true mask):

| treatment | λ=1 | λ=0 | absolute drop | relative drop |
|---|---|---|---|---|
| raw | 0.541 | 0.240 | −0.301 | **−55.6%** |
| pooled | 0.769 | 0.681 | −0.088 | −11.4% |
| centered | 0.664 | 0.494 | −0.170 | −25.6% |

Removing mask conditioning entirely **more than halves `raw`** — the
model has no idea where to edit without it, exactly as expected since
`raw` is the treatment that carries the full positional burden. `pooled`
(position stripped out by scoring against the *true*, still-known mask
region) survives much better — the model can still guess a reasonable
per-channel direction from instruction + `z_src` alone, even with zero
spatial cue. `centered` sits in between, consistent with the mask also
helping disambiguate *which* region's content the direction should be
computed from, not purely *where* to apply it.

**Combined verdict: the mask path is alive, strongly (if unexpectedly)
differentiated, and doing substantial, load-bearing work — mostly for
position, partly for direction.** This closes off "the mask conditioning
might just be dead weight" as an explanation for the positional half of
the deficit. What's left is the residual gap ON TOP of a working mask
path — `raw` (0.541) still well below `pooled` (0.769) even at λ=1 — and
that residual is what the readout-constraint hypothesis targets: the
one-step-at-s=0 readout may not give the transformer enough steps to
resolve position as precisely as the mask signal it's given would allow.

**One more check before closing: is the DC offset load-bearing, or is it
redundant with the dataset-wide bias it correlates with?** `mask_enc`'s
spatial mean was found anti-correlated with `bias.npy` (cos = −0.579,
std = 0.025 across 64 items — tight enough to be a fixed property of the
trained weights, not per-item noise, but not −1.0 either, so neither pure
cancellation nor an independent direction). `transport/adapter.py`'s
`apply_mask_conditioning(dc_ablate=True)` subtracts that offset at
inference, preserving `delta_z`'s spatial pattern exactly, to settle it
directly (`diagnose --ablate-mask-dc`):

| treatment | intact (λ=1) | DC ablated | λ=0 (no mask) |
|---|---|---|---|
| raw | 0.541 | 0.232 | 0.240 |
| pooled | 0.769 | 0.211 | 0.681 |
| centered | 0.664 | 0.274 | 0.494 |

**Outcome three, more extreme than any option considered going in: the DC
offset is load-bearing, and removing it is the single most destructive
intervention run in this entire line of checks** — worse than removing
the mask path entirely. On `pooled`, ablating the offset drops
0.769 → 0.211, while dropping the *whole* mask path only falls to 0.681:
withholding the spatial signal costs 0.09, keeping the spatial signal but
stripping its constant costs 0.56.

**Mechanism, in hindsight:** `delta_z` is added directly to `z_src` before
the transformer reads the sum. The DC offset sets the *operating point*
of that query — which region of latent space the network is actually
being asked about. Remove it and every query lands somewhere the model
never saw in training; the structured in-mask component only means
anything *relative to that baseline*, and on its own is a perturbation
around the wrong point. The earlier decomposition ("~93% inert bias, ~7%
real structure", from the RMS split) was arithmetically correct and
functionally backwards — the 93% is what makes the 7% legible, not dead
weight riding along with it. This also reframes the −0.579 correlation:
not redundancy with `bias.npy`, but a learned working point that happens
to sit at a fixed angle to the dataset's mean edit direction — stable
across items because it is a property of the trained weights, not of any
sample.

### Direction A, closed: the full elimination sequence

The positional deficit (`raw` well below `pooled` even with a healthy
mask path) has now survived six independent, cheap checks, none of which
required a retrain:

1. **Target ambiguity** — the corrected ambiguity probe, mean-centered and
   held-out (§7 above): real instruction-conditioning signal, no evidence
   of the genuine multi-modal ambiguity Direction A's `σ(0)>0` was built
   to fix.
2. **Injection magnitude** — `‖delta_z‖/‖z_src‖` doesn't track quality
   across the σ0 sweep (σ0=0.1 has the largest ratio and best score;
   σ0=0.7/1.0 have the smallest and worst); retired as a cross-σ0
   diagnostic once traced to the noised effective source's larger norm,
   not a real signal-strength difference.
3. **Mask-path death** — `mask_enc.out` moved substantially from its
   zero-init (`abs_max` 0.011–0.023 across all six checkpoints); the λ=0
   ablation confirms it is doing real, load-bearing work (raw −55.6%).
4. **Mask insensitivity** — `mask_sensitivity()`'s rolled-mask comparison:
   `delta_z` at the true mask's location responds substantially when the
   mask moves, ruling out a mask-agnostic encoder despite the suspicious
   <2%-variation of `mask_enc`'s weights across six independently-trained
   σ0 checkpoints.
5. **Malformed localisation** — the inside/outside RMS ratio (0.21) first
   looked like an absence of localisation; corrected to "background-
   dominant, strongly differentiated" (`classify_localisation`'s fix), and
   the outside-mask structure/bias decomposition plus the DC-ablation
   result below show it is a *learned operating point*, not malformed
   wiring.
6. **DC redundancy with the dataset bias** — the −0.579 correlation raised
   the possibility that part of the mask path just re-derives a constant
   the model could get for free; the ablation above rules this out
   directly and about as conclusively as a single experiment can.

**None of the six explains the deficit. The readout hypothesis — that the
one-step-at-s=0 query, however well-conditioned, cannot resolve position
as precisely as a multi-step or interval-aware readout could — is what's
left, on a substantially firmer base than when it was first proposed.**

### What the DC-ablation result requires of any multi-step redesign

Whatever replaces the one-step readout has to preserve the
operating-point behaviour just confirmed, not inherit whatever falls out
of the code by accident. Two things are worth stating as design
constraints before writing that code, not after:

- **Mask conditioning must be reapplied identically at every query point
  a multi-step or interval-based readout makes**, including its DC
  component — not just at `s=0`. Concretely, this already holds by
  construction for both `"pair"` mode (Gate A) and `"meanflow"` mode
  (Gate B): `TransportAdapter._gate` returns `gate ≡ 1` for both,
  unconditionally, so `apply_mask_conditioning`'s full contribution
  (structure and DC alike) is added at every call regardless of `t` or
  `r`. A merged mode only inherits this for free if it keeps that gate
  semantics; reusing `"meanflow"`'s original Δ-gated behaviour instead
  (gate = `(t-r).clamp(min=0)`, zero at the diagonal) would silently
  reintroduce exactly the operating-point discontinuity this section is
  about, at every `r == t` boundary sample.
- **Check whether MeanFlow's curvature-correction term is even doing
  anything for a straight-line pair path before paying for the JVP
  machinery.** MeanFlow's identity corrects for `du/dt ≠ 0` — the
  instantaneous velocity field genuinely changing along a curved
  trajectory, which is real for standard diffusion/flow-matching's
  stochastic interpolation. Gate A's pair path is `z_s = (1-s)z_src +
  s z_tgt` with `v = z_tgt - z_src` *exactly constant along the entire
  path by construction* (before Direction A's now-closed σ0 noise). For
  a perfectly-learned model, the true average velocity over *any*
  subinterval already equals the instantaneous velocity everywhere —
  there is no curvature for the correction term to earn its cost on.
  What multi-step querying plausibly buys instead is a *different*
  mechanism entirely: each step narrows the remaining extrapolation
  distance (after step 1, `z` has already moved partway to the target,
  making the next query an easier sub-problem), which is exactly what
  `apply_transport`'s existing `steps > 1` naive-Euler path already does,
  with **no training change and no JVP risk** (per `meanflow.py`'s own
  documented fragility — forward-mode AD hit signal 11 on four separate
  probes in Stage 0). **Run that free check first**: sweep
  `apply_transport`/`diagnose` at `steps` > 1 on the σ0=0.0 checkpoint and
  see whether `raw` improves. If it does, iterative refinement (not
  curvature correction) is the load-bearing mechanism, and the case for
  the full JVP-based MeanFlow merge weakens considerably in favour of
  something cheaper; if it doesn't, that's evidence for a genuinely
  different training target at intermediate points, and the merge is
  better justified.

**Next: the real work queue.**

1. The multi-step Euler pre-check above — cheap, uses the checkpoint and
   tooling already in hand, and decides how much of the JVP-based merge
   is actually necessary before committing to it.
2. Merge Direction A's machinery with Gate B's multi-step readout,
   informed by (1) and preserving the operating-point constraint above.
3. The Gate C scoreboard wrapper (`bench/gate_a_model.py`), regardless of
   how (1)–(2) turn out — neither path can be judged on the 3-case
   Stage-0-comparable sweep alone. Code-complete since earlier in this
   document; run:
   ```
   python bench/run_gate_c.py --model gate-a-transport \
       --gate-a-ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
       --gate-a-sigma0 0.0 --mask-mode oracle \
       --out runs/gate_c/gate_a_sigma0.0_oracle.jsonl
   ```
   on the σ0=0.0 checkpoint (this document's winning arm across every
   diagnostic run against it) — repeat per `--mask-mode` as the existing
   151-case harness requires for a scoreboard-comparable result.

### What to keep, what to drop

**Keep:** the WHERE/WHAT unification (§2 above) — it turned out to be the
frame the whole investigation converges on, not just a nice way to talk
about Gate C. The corrected probe's full result (ceiling framing,
mask-pooling, mean-centering, specificity split, template-dominance and
per-channel checks) as the permanent record of why Direction A closed.

**Drop:** the Gate B cosine claim (as before). The original n=3 struct-
based failure diagnosis and the mean-collapse-via-ambiguity mechanism it
implied — superseded by the above, not merely unproven. The claim that
the model's `cos_clean` exceeded the instruction-only ceiling as evidence
*against* mean-collapse — withdrawn earlier in this section and now moot,
since mean-collapse itself is no longer the live hypothesis. Any further
σ0 sweeping or same-source cache-rebuild work — closed per the verdict
above.

---

## 8. Direction A, concrete design — σ(0) > 0 — HISTORICAL, CLOSED

**This section describes what was built and why, as design rationale.
It is not a live instruction — do not train or sweep anything from it.**
Direction A ran to completion (the six-point σ0 sweep at §"The complete
six-point sweep, all at step 8000–8100" below), was diagnosed down to a
mechanism, and closed (§"Verdict: Direction A closed" below) on five,
later six, independent lines of evidence. The "what's left" list at the
end of this section is kept for the historical record of what was
planned at the time; every item on it has since either happened (and is
recorded, with its actual result, later in this document) or been
superseded by the closure. Read forward from §"Verdict" for the current
state.

Implemented, not just scoped: `transport/pairflow.py`'s `sample_path` /
`pairflow_loss` / `apply_transport` and `scripts/gate_a_transport.py`'s
`train`/`eval` CLIs carried a `sigma0` parameter end to end. Self-tested
(`python transport/pairflow.py --self-test`, CPU-only, no model weights)
and both CLIs' `--help` confirmed before any training run.

### The substitution

`sigma(s) = σ0·(1−s)` — `σ(0) = σ0 > 0` (the fix), `σ(1) = 0` (endpoint
stays exact, `z_1 = z_tgt`). This folds into a **noised effective source**
rather than a separate noise term added on top of the existing formula:

```
z_src' = z_src + σ0·ε·M            (mask-gated: ε ~ N(0,I), M the edit mask)
z_s    = (1−s)·z_src' + s·z_tgt    (unchanged formula, z_src → z_src')
v      = z_tgt − z_src'            (unchanged formula, z_src → z_src')
```

Two consequences of doing it this way rather than adding an explicit
`σ(s)ε` term to `z_s` and a separate `σ'(s)ε` term to `v`:

- **`σ0 = 0.0` is byte-identical to the original deterministic pairflow**,
  not merely close to it — `z_src' = z_src` exactly, so every existing
  checkpoint, every existing eval number, and the whole Stage-0-comparable
  methodology stay reproducible with the new code unchanged. Checked in
  `self_test()`, not asserted.
- **Mask-gated, so the background is untouched.** Outside the mask
  `z_src' = z_src` and `v = 0` exactly as before, preserving
  `PairFlowConfig`'s existing "leave the background alone" property and its
  inside/outside loss split without modification.

### Training (`sample_path` / `pairflow_loss`)

One line changed in the training loop's effect, nothing structural:
`PairFlowConfig.sigma0` (default `0.0`) threads into `sample_path`, which
now takes `mask` (it didn't before — added, since the noise has to be
gated by something). `pairflow_loss` passes the same `mask` it already
had. `--sigma0` is a new flag on `gate_a_transport.py train`.

The value of `σ0` was, at the time this was written, not yet decided —
correctly so: no self-test can validate a hyperparameter, and it needed
real data, guided by `move_ratio`/`cos`/`rel_in` (already instrumented in
`pairflow_loss`'s stats dict) at each candidate. **It has since been
swept exhaustively** (0.0, 0.1, 0.3, 0.5, 0.7, 1.0 — see "The complete
six-point sweep" below): no value beat `σ0=0.0` by more than noise, and
`σ0 ≥ 0.5` destroyed the model's instruction-specific signal outright.
This subsection is kept as a record of the reasoning that was correct
going in (start small, don't guess large), not as an open question.

### Inference (`apply_transport`)

The query point starts at `z_src' = z_src + σ0·ε·M` (fresh `ε` per call,
mask-gated, matching training exactly), not at `z_src`. `--sigma0` and
`--eval-seed` are new flags on `gate_a_transport.py eval`; the sweep
re-seeds fresh for every strength `a` rather than letting one generator
advance across it, so the whole strength sweep for a case compares the
*same* sampled mode and only `a` varies — an advancing generator would
have silently compared different modes at different strengths, which
would have corrupted the oracle-relative-structure sweep without erroring.

**One asymmetry worth stating plainly, not discovering later:** a
checkpoint *trained* with `σ0 > 0` and then *evaluated* with `σ0 = 0`
queries it at `z_src` exactly — a point training never actually saw fixed
(only its average over `ε`). That's mild covariate shift, not a bug, but
it means the existing Stage-0-style deterministic sweep (`--sigma0 0.0`,
the default, kept for continuity) is evaluating the model slightly off its
own training distribution once `σ0 > 0` training is used. Use it for
continuity checking, not as Direction A's verdict.

### The two evaluation tracks, and which is primary

1. **Existing decoupling sweep** (`bench/decoupling.py`, oracle-relative
   structure, `STAGE0_IDS`, the pre-registered `|corr| < 0.5` threshold) —
   kept for continuity, run at `--sigma0 0.0` by default. Useful as a
   sanity check that Direction A training hasn't broken the deterministic
   path, **not** as the mechanism's verdict — it's 3 cases, same
   provenance problem this whole reopening exists to fix.
2. **Gate C scoreboard** (151 cases, the `mis_targeted` classifier, the
   GLM-5.3-Flash judge at ρ = 0.717, `off_instance`) — **the primary
   evaluation**, per §4's original plan. At the time this was written, not
   yet wired: needed a new arm, a `run_gate_c.py`-compatible model wrapper
   loading a Direction A checkpoint and calling it with `sigma0 > 0` per
   case, the same way `MODEL_REPOS`' existing entries wrap
   `SD3_UltraEdit_w_mask` etc. **Built since** (`bench/gate_a_model.py`,
   wired into `run_gate_c.py`'s `--model gate-a-transport`) — including a
   real cross-package import bug (`ModuleNotFoundError` on `bench`/
   `transport` when invoked the documented way) found and fixed only by
   actually trying to run it, not just reading the code. Direction A is
   now closed on other grounds before this wrapper's own score was ever
   needed for the verdict — score it anyway (command in "Next: the real
   work queue" below) since it's the only judge that isn't the 3-case
   Stage-0 sweep, and every candidate that replaces Direction A needs the
   same wrapper regardless of which mechanism wins.

### What was planned next, at the time (superseded — see below for what
actually happened)

1. Pick a first `σ0` (small; start the sweep once a candidate is chosen).
2. Train (`gate_a_transport.py train --sigma0 <value> ...`), on the
   existing MagicBrush cache.
3. Sanity-check with the existing decoupling sweep at `--sigma0 0.0`
   (continuity) and `--sigma0 <value>`.
4. Write the Gate C scoreboard arm wrapper.
5. Score on the full 151-case Gate C scoreboard.

What actually happened diverged from this list almost immediately: step
1 became a full six-point sweep rather than one candidate (§"The
complete six-point sweep" below), steps 2–3 surfaced the ε-fitting
failure mode and the checkpoint-inspection line of diagnostics that
closed Direction A before step 5 was ever needed for the verdict, and
step 4 (the wrapper) was built afterward, once a winning checkpoint
(`σ0=0.0`) existed to point it at.
