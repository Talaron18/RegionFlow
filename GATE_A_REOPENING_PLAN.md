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
A** vs **Direction B**, the σ(0) parameterisation, the `1 − cos²` ≈ 0.44 vs
≈0.72 LoRA ablation, and the exact `§7` ambiguity probe being critiqued —
is not in this repo and was not pasted into this conversation. Grepping the
full tree for "Direction A", "Direction B", "σ(0)"/"sigma(0)", "ambiguity
probe", and "same LoRA" returns nothing.

Scoping the Gate A reopening means designing the stochastic-path objective
concretely enough to run — and "if Direction A is taken, keep σ(0) > 0" is
a real constraint on that design that can't be honored without knowing what
Direction A specifies. Rather than guess at a design that might silently
contradict it, this is the one blocking question before scoping starts:
**share the document (or its relevant section) that defines Direction A and
Direction B**, or restate σ(0), the two directions, and the LoRA ablation
directly. Everything else in this file is ready to act on now.
