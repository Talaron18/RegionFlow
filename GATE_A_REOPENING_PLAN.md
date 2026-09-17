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

### Result: run against the real cache

The **same-source** side of the contrast came back empty — this cache has
at most one turn-1 edit per source image (`precompute_cache.py`'s own
`turns=(1,)` restriction), so no source image recurs across sessions with
a different instruction. That's a finding about the cache, not the
hypothesis: the intended contrast can't be run as designed against this
data. `ambiguity_probe.py` fell back to a substitute null — random
unrelated pairs (different instruction, any source) — so same-instruction
had something to be measured against.

| grouping | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| same-instruction, different source | 177 | +0.177 | +0.165 | +0.047 | +0.352 |
| unrelated (different instruction) | 177 | +0.127 | +0.108 | +0.036 | +0.262 |

Median gap +0.056, Mann-Whitney U p = 2.1×10⁻⁵. **Statistically real, but
small, and reads as a genuinely mixed result — not the clean confirmation
either direction was hoping for.**

- The gap is real: same-instruction pairs *are* more aligned than
  unrelated ones, at p < 0.0001. The instruction is not directionally
  inert.
- The gap is small relative to the spread within each group (p10–p90 spans
  ~0.2–0.3 in both), roughly a medium effect size, not a large one — same-
  instruction pairs are nowhere close to aligned in any strong sense.
- **Both medians are low in absolute terms** (0.165 and 0.108, both far
  below 1.0). Even pairs sharing the exact same instruction string, across
  different source images, average close to orthogonal. That's *consistent
  with* substantial latent ambiguity in `v` given `(z,c,M)` — but it isn't
  proof of it, because this substitute null can't separate "the instruction
  underdetermines the edit" from "different source images produce
  different displacement vectors for reasons that have nothing to do with
  ambiguity" (different objects, different masks, different scene
  content). The same-source contrast this probe was actually designed to
  run would have isolated that; it wasn't available here.
- **A new observation the unrelated-pairs baseline surfaced on its own:**
  cosine 0.108 between *fully unrelated* edits is well above the ~0
  a high-dimensional random-vector null would predict (65536-dim latents:
  expected cosine noise floor is on the order of 1/√65536 ≈ 0.004). That
  means most displacement vectors — regardless of instruction — already
  share a fair amount of common structure, plausibly generic (mask
  size/shape statistics, VAE latent-space geometry, "how much a masked
  region typically shifts") rather than semantic. This dilutes both groups
  similarly, but it means the same-instruction signal is smaller relative
  to structure that has nothing to do with meaning than the raw 0.165
  figure suggests on its own, and it's worth accounting for — e.g.
  normalising out a generic/mean displacement component — before reading
  too much into any single-instruction cosine number in future work.

**Read on the hypothesis:** weak-to-moderate support for real ambiguity,
not a decisive confirmation. Enough to proceed toward Direction A (with the
`σ(0) > 0` fix) rather than abandon the reopening, but not enough to
declare the mean-collapse mechanism established — it remains a
well-supported hypothesis, which given Gate A's own n=3 provenance is the
honest place to leave it before the retrain, not after.

### What to keep, what to drop

**Keep:** the WHERE/WHAT unification (§2 above) as a genuinely good frame.
The mode-selection diagnosis, with the `s=0` refinement. The probe's result
— weak-to-moderate, real support for ambiguity, not proof of it — as the
actual current evidence base, in place of the original n=3.

**Drop:** the Gate B cosine claim. Any confidence that the mechanism is
established as fact rather than hypothesis — Gate A's failure rests on 3
cases scored with a metric (`struct`) later shown to be substantially a
colour artifact, and the probe that was meant to settle it came back
statistically real but small, with its most informative half (same-source)
unavailable in this cache. Proceed toward Direction A; don't claim the
mechanism is established.
