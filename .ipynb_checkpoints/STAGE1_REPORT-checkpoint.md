# RegionFlow Stage 1 Report

**Status:** Gate A closed. Gate B negative on its first faithful run. One
methodological result (a metric failure) that reaches back into Stage 0 and is
probably the most portable thing here.

---

## 1. Summary

Stage 0 concluded that a source/noise mixing ratio cannot express an attribute
direction, and that an editor needs a learned conditional transport `z + Δz`.
Stage 1 built that transport, tested it, and found:

1. **Deterministic regression to `z_tgt − z_src` cannot produce clean edits.**
   The model goes from "no visible change" straight to blur, at every strength.
   This is the conditional mean of a multimodal target, and it is a property of
   the objective, not of the latent space or the learner.

2. **Stage 0's headline number is substantially a metric artifact.** The
   `struct` metric correlates luminance edge energy, which any realistic
   recolour reduces by construction. Measured against it, the *ground-truth
   MagicBrush edits* score worse than the model does.

3. **A rank-32 adapter did not acquire the interval dimension.** Under the
   reference-faithful MeanFlow objective, the learned field is worse than
   simply reusing the instantaneous velocity at every Δ from 0.05 to 0.8, and
   its correction is uncorrelated with the correction actually required.

4. **Several confident intermediate conclusions were wrong and were
   withdrawn.** They are documented in §6 because the pattern that produced
   them is more instructive than the conclusions were.

---

## 2. Gate A — conditional transport

**Setup.** Straight-line flow between source and target latents,
`z_s = (1−s)z_src + s·z_tgt`, `v = z_tgt − z_src`, one-step readout at `s = 0`.
Rank-16 LoRA on frozen SD3-Medium, 4511 MagicBrush turn-1 triplets, 8000 steps.

**Result.** Directional alignment `cos(u, v)` reached 0.585–0.656 against a
measured statistical ceiling of ~0.65 (see §2.1). Amplitude sat at 93% of the
MSE optimum for that alignment — `‖u‖ ≈ cos·‖v‖`, which is correct hedging
under directional uncertainty, not timidity. Structure preservation did not
improve as alignment improved: `struct@dE30` stayed at 0.355–0.389 while `cos`
moved 30%.

**Reading.** MSE regression on a multimodal target returns the average of the
plausible edits. The average of many colour changes over an object is a
desaturated smear, which is exactly what the contact sheets show. Amplitude is
not an independent quantity: under MSE it is pinned to alignment, so
"under-moving" was never a separate problem to fix.

### 2.1 The directional ceiling

For a target `v = μ + noise`, pairwise agreement between targets under the same
conditioning and the achievable alignment are related by
`cos_max ≈ sqrt(E[cos(v_i, v_j)])`. Measured on the cache by grouping
conditioning-space neighbours and tightening the similarity floor:

| neighbour similarity | ceiling |
|---|---|
| 0.697 (loose) | 0.508 |
| 0.819 | 0.547 |
| 0.917 | 0.611 |
| 0.959 | 0.630 |
| extrapolated to 1.0 | **0.647** |

A constant-direction predictor that ignores all conditioning scores 0.453. So
the learnable range is 0.453 → 0.647, and the model reached 0.585 — about 68%
of it. Roughly half the target direction is genuinely unpredictable from
`(z, c, M)`; the model captures most of the rest.

---

## 3. The metric failure

This is the result most likely to be useful outside this project.

Stage 0's structure metric is the Pearson correlation of luminance edge energy
inside the mask. Any realistic recolour changes lightness and therefore reduces
edge energy, so `dE` and `struct` are mechanically anti-correlated before any
model is involved.

The check that exposed it: apply the **ground-truth** displacement
`z_tgt − z_src` from the dataset and score it on the same metric.

| case | oracle `struct@dE25` |
|---|---|
| sheep | 0.10 |
| phone booth | 0.29 |
| ball | 0.43 |
| mean | **0.365** |

The learned model scores ~0.41. **The model beats reality.** That is not
possible if the metric measures structure preservation, so it does not.

Consequences:

- Stage 0's `corr(dE, struct) = −0.958` is partly a property of the metric
  pair. It should not be reported as a property of the editing method.
- The pre-registered Gate A threshold (`struct@dE30 ≥ 0.70` to refute) was
  unreachable — it asked the model to beat the ground truth.
- The synthetic control that validated the metric used a *luminance-preserving*
  L\*a\*b\* chroma shift, an edit that does not occur in practice. It validated
  the metric on the one case where it works.

**Replacement.** `bench/structure.py` provides `grad_orient` (cosine similarity
of gradient direction fields, magnitude-weighted) and `lcn_corr` (edge
correlation after local contrast normalisation). Calibration on a real
photograph:

| edit | dE | struct0 | grad_orient | lcn |
|---|---|---|---|---|
| darken 75% (structure intact) | 41.8 | **0.475** | 0.977 | 0.745 |
| hue swap, L preserved | 23.4 | 0.997 | 0.997 | 0.995 |
| blur | 15.0 | 0.051 | 0.297 | 0.195 |
| destroy 60% | 43.8 | 0.137 | 0.441 | 0.220 |

**The general protocol worth keeping:** *the ground-truth edit should score near
the top of your metric. If it doesn't, the metric is wrong.* That check costs
minutes and would have prevented a wrong claim from reaching a draft here.

---

## 4. Gate B — interval-field conversion

### 4.1 The implementation was not faithful at first

The first implementation was written from the MeanFlow identity without
consulting the official code. Three differences mattered:

| | ours (initial) | reference |
|---|---|---|
| anchors at `r = t` | 0%, and **zero gradient** under the Δ-gate | 75%, full gradient |
| Δ schedule | invented curriculum, `delta_end = 1.0` | none; both endpoints sampled from step 1 |
| loss reduction | mean, `(L + 1e-3)^0.75` | **sum**, `(L + 0.01)^1.0` |

The first is structural. Gating every trainable path on `Δ = t − r` gives an
exact boundary condition `u(z,t,t,c) = v_base` for free — and in exchange makes
the gradient at `r = t` identically zero, discarding three quarters of the
reference's training signal. Demonstrated rather than argued:
`|grad| = 1.151` in ungated mode versus `0.000e+00` gated, same batch.

The third matters more than it looks: with `p = 1` on a sum-reduced loss, the
weighting is `sse/(sse + 0.01)`, which saturates at 1.0. It is a per-sample
gradient normaliser, not a progress signal — the reported `loss` is pinned at
1.0 by construction and carries no information.

The instability observed under the custom formulation (MSE 2.6 → 183 in twenty
steps) **did not reappear** under the faithful one: `sse_anchor` held flat at
~2×10⁴ for 725 steps with Δ spanning its full range from step one. That
instability was the formulation, not SD3.

### 4.2 The stiffness detour, and what survived it

A chain of reasoning — "‖du/dt‖/‖u‖ ≈ 15, so Δ_usable ≈ 0.07, so ~14 steps
minimum, so pretrained fields cannot be converted" — collapsed under
measurement:

| measurement | S | Δ_usable |
|---|---|---|
| through the adapter, CFG 5, total derivative | 87.2 | 0.011 |
| frozen model, CFG 1, total derivative | 11.1 | 0.090 |
| **frozen model, CFG 1, spatial part only** | **2.84** | **0.35** |

The temporal part dominates by ~6× and is not trajectory curvature: it is the
sinusoidal timestep embedding, whose components run down to period 0.020 in
normalised `t` and pick up a ~316× gain under `d/dt`. Probing at 0.06 spacing
aliased it, producing a zigzag that looked like structure. A fine sweep
confirmed a dominant period of 0.0203 against embedding component 16 at 0.0199.

**What survives:** trajectory curvature is `S ≈ 2.84`, implying a step floor
around 3 for a constant-velocity approximation. The "SD3's field is too stiff"
hypothesis is unsupported.

**What this suggests but does not establish:** MeanFlow's target uses the
*total* derivative, so the timestep encoding's high-frequency content enters the
regression target directly. That is a plausible mechanism for target noise and
a cheap ablation (low-pass the interval conditioning basis), not a finding.

### 4.3 The result: no interval dependence was learned

Measured against a ground-truth average velocity, integrated from the frozen
teacher with a midpoint rule (verified exact on a linear field):

```
u_true(z_t, r, t) = 1/(t−r) ∫_r^t v_teacher(z_s, s) ds
```

| Δ | model | base (`v` reused) | **excess** | model/base |
|---|---|---|---|---|
| 0.05 | 970 | 78 | **892** | 12.46 |
| 0.10 | 1119 | 247 | **872** | 4.53 |
| 0.20 | 1618 | 739 | **879** | 2.19 |
| 0.30 | 2361 | 1456 | **905** | 1.62 |
| 0.50 | 4771 | 3679 | 1092 | 1.30 |
| 0.80 | 14057 | 11489 | 2568 | 1.22 |

`model/base > 1` everywhere: **the learned field is worse than doing nothing.**
And `sse_model − sse_base` is 887 ± 12 (1.4%) across a 6× range of Δ — a
constant-magnitude perturbation contributing nothing that varies with the
interval. `model/base` only improves at large Δ because the *base* error grows
while the adapter's contribution does not.

Direct r-sensitivity confirms it independently:

| Δ | `D_r` true | `D_r` model | ratio | cos(correction) |
|---|---|---|---|---|
| 0.05 | 3.0% | 1.9% | 0.63 | −0.034 |
| 0.10 | 4.8% | 1.9% | 0.39 | −0.015 |
| 0.20 | 8.0% | 1.6% | 0.20 | +0.022 |
| 0.30 | 11.5% | 1.6% | 0.14 | +0.004 |
| 0.50 | 18.0% | 1.8% | 0.10 | −0.006 |
| 0.80 | 30.8% | 1.8% | 0.06 | −0.004 |

The required r-dependence grows tenfold across Δ; the model's is flat at ~1.7%
and points nowhere (`|cos| < 0.035` everywhere). Of the adapter's ~8.5% total
deviation from the base field, only ~1.7% varies with `r` at all.

**Training makes it worse.** Excess error ~485 SSE at step 200, ~890 at step
1000. The harmful r-independent perturbation nearly doubles while r-dependence
stays flat and uncorrelated.

### 4.4 The anchor-competition hypothesis is untested

The intended P10 run (10% anchors) **did not run at 10%**: `frac_anchor` came
out 0.762 because `data_proportion` was never threaded from the CLI to the loss
config. It was a second P75 run.

The teammates' write-up compares that run's step-200 validation (238k) against
P75's best (304k) as preliminary support for anchor competition. Those are two
P75 runs, so the comparison instead gives something else useful: a **~28%
run-to-run spread** on that metric between identical configurations.

**And the hypothesis is in tension with the reference's own ablation.** MeanFlow
reports 1-NFE FID of 61.06 at 25% non-anchor, 63.14 at 50%, 67.32 at 100%. A
knob that moves quality ~10% in the source setting is not a plausible cause of
a *complete* absence of r-learning. It should be tested — the pretrained setting
genuinely differs, since the backbone already satisfies the boundary condition
the anchors exist to teach — but it should not be the first thing tested.

---

## 5. Revised priorities

Ordered by information per GPU-hour.

**P1 — Is the interval pathway used at all?** (no GPU, minutes)
`r` reaches the model through exactly one path: an `IntervalEmbedding` MLP
whose output is added to `pooled_projections` — a single global 2048-dim vector
into the adaLN modulation, zero-initialised. Load a checkpoint and check
whether those weights moved. If they are still near zero, no sampling ratio
will help and P2 is the answer.

**P2 — Explicit interval conditioning.** Give `r` a wider path: its own
embedding into the token stream, or per-block modulation, rather than one
global additive vector. This is the most likely fix if P1 shows a live but
ineffective pathway.

**P3 — Rank / capacity.** Rank 32 → 128 on the same objective. Cheap to run,
and it bounds how much of the failure is representational.

**P4 — Anchor ratio (the real P10).** Now runnable via `--data-proportion`,
with a runtime guard that aborts if the realised `frac_anchor` disagrees with
the request.

**P5 — Low-pass the interval conditioning basis.** Drop the high-frequency half
of the sinusoidal basis and see whether target noise falls. Motivated by §4.2,
speculative.

Gate C (instance-level selectivity) remains untouched and is still the research
problem. `bench/instance_bench.py` is written; the benchmark build is CPU work
independent of all of the above, and it is the deliverable no metric revision
can invalidate.

---

## 6. Withdrawn claims

Recorded because the pattern matters more than the individual errors.

| claim | why it fell |
|---|---|
| Stage 0's coupling is a property of SD3's latent space | the ground-truth edit is structure-preserving; the metric was measuring luminance contrast |
| approximate transport falls off-manifold, hence the collapse | misalignment sweep was nearly flat (0.528 → 0.423 from ρ=1.0 to 0.585) |
| the model is barely above a constant-direction predictor | measured mid-training; 2000 more steps took skill from 0.29 to 0.60 |
| SD3's field is too stiff for interval conversion (S ≈ 15–87) | CFG amplification plus an aliased timestep embedding; spatial curvature is 2.84 |
| the fd/jvp disagreement is TF32 / kernel nondeterminism / backend mismatch | all three ruled out by measurement; it was the fp32 timestep embedding inside an fp64 model |
| Gate A REFUTED at oracle parity (1.065) | rested on excluding one of three cases, by a rule written after seeing which case failed |

Every one was caught by building a control, not by reasoning harder. The
controls that did the work: the oracle probe, the metric calibration on a real
photograph, the fp64 machinery check, the frozen-model measurement, the
reference audit.

---

## 7. Methodology notes

Tooling failures in this stage clustered into two kinds — a stale file, or a
check that inspected text rather than structure. Both pass silently and fail on
the real model, often after loading a 2B-parameter checkpoint. `tests/` now
holds:

- `check_names.py` — scope-aware undefined-name check (a flat version passed a
  file whose `encode_null` used a `torch` imported only inside a sibling
  function)
- `check_modes.py` — parses the adapter's mode guard rather than grepping it
  (`grep -c meanflow` returned 4 on a file whose guard still rejected the mode)
- `check_resume.py` — save/load round-trip including device placement
- `test_preflight.py` — full stack on a 2-layer model, both dtypes, CUDA and CPU
- `test_checkpoint_jvp.py` — gradient checkpointing against forward-mode AD, on
  the path Gate B actually executes
- `test_reference_audit.py` — eight checks against the official MeanFlow source

Three rules that would have saved the most time, in order:

1. **Parse, don't grep.** Two wrong conclusions came from checks that matched
   strings in docstrings.
2. **Test the executed path.** The checkpoint/JVP test passed while Gate B died
   in checkpointing, because it exercised an adjacent function.
3. **Score the ground truth on your metric before trusting it.** §3.

And one that is specific to this project: **read the reference implementation
before building from the paper.** Five deviations from the official MeanFlow
code turned out to matter, and one of them silently removed 75% of the training
signal.
