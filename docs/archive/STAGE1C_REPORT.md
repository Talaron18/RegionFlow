# RegionFlow Stage 1C — Gate B closed positive, with a stated method

Supersedes STAGE1B_REPORT.md §2.5 (the headline 5k result), §2.7 (the global
correction scale), §3 (verdict), §5 (the drift explanation) and §7.1b (which was
the run this stage executed). Everything else in STAGE1B_REPORT.md stands.

Read this after STAGE1B_REPORT.md and before doing anything.

---

## 0. Where this sits relative to the project's goal

The goal, unchanged: **change one named object's attribute while preserving its
form, leaving the rest of the image untouched, and without affecting other
instances of the same category.** Instance-level selectivity is the research
problem. Everything else is infrastructure.

This stage, like 1B, was entirely infrastructure. It closes Gate B positive.
**Gate C has still never been run and `bench/instance_bench.py` is still
unrun.** That is now the only thing left on the critical path, and §7 is
short because there is only one item on it.

---

## 1. What 1B said, and what this stage changed

STAGE1B_REPORT.md §2.5 reported the 5k run beating the frozen instantaneous
field at Δ ≥ 0.5, with the remaining error being "a systematic ~15–25%
overshoot at large Δ and mid-to-high t, direction unaffected." §7.1b
pre-registered a frozen-first-term ("distill") run to test whether removing the
`e − x0` first term would fix the small-Δ drift.

Three things came out of running it:

1. The distill run is better than the 5k run on **both** axes at **every** Δ,
   and it is the checkpoint to quote.
2. Every number in §2.5 and §2.7 was measured at a single `t_center = 0.5`.
   That was never stated as a limitation because nobody had varied it. The
   result is strongly t-dependent, and the 1B headline is t-local.
3. The "~15–25% overshoot" is not a constant. It is a monotone function of t,
   crossing 1.0 near t ≈ 0.3. But once t is swept properly, a **single
   inference-time constant** covers the whole failing region.

---

## 2. Experiments, in order

### 2.1 The distill 5k run (executing §7.1b)

`--first-term frozen --norm-p 0 --grad-clip 0`, fd 0.025, frozen tangent,
Δ ≤ 0.5, 5000 steps. At `t_center = 0.5`, 64 samples:

| Δ | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| cos | 0.700 | 0.754 | 0.776 | 0.778 | 0.734 |
| excess over base | +77 | −55 | −189 | −790 | −1143 |
| (§2.5 run, cos) | 0.626 | 0.712 | 0.739 | 0.715 | 0.714 |
| (§2.5 run, excess) | +1015 | +905 | +579 | +120 | −293 |

Better on both axes at every Δ. The pre-registered success bar was cos ≥ 0.7 at
every Δ **and** excess < 0 at every Δ including 0.05. It passes the first and
misses the second by +77 — 93% of the way from the §2.5 run's +1015. It is
neither the pre-registered "success" nor the pre-registered "partial" branch,
and the honest sentence is "success except at Δ=0.05."

**The §7.1b watch item resolved the opposite way to how it was framed.**
`corr_err` at Δ=0.05 is 1.250, still > 1, and the key said that would mean
"small-Δ direction is a separate problem." But cos at Δ=0.05 is 0.700, the same
as everywhere else. Solving `corr_err = 1 + ρ² − 2ρ·cos` gives ρ = 1.58: the
correction points the right way and is 58% too large. Small-Δ was a calibration
problem all along. **A reading key whose conditional embeds an assumption can
be falsified by its own premise — check the premise before applying the
branch.**

### 2.2 Checkpoint trajectory — training is done at step 4000

Mean cos over Δ = 0.1–0.5, `t_center = 0.5`:

| step | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|
| mean cos | 0.266 | 0.703 | 0.745 | **0.761** | **0.761** |

Flat from 4000. Longer schedules are not a lever. At the measured throughput
(§5) a 10k run is ~4 h, and it would buy nothing.

### 2.3 The t_center sweep — the 1B headline is t-local

64 samples, `ckpt_final`. `vs_base` (< 1 beats the frozen instantaneous field):

| t_c | Δ.05 | Δ.10 | Δ.20 | Δ.30 | Δ.50 |
|---|---|---|---|---|---|
| 0.20 | 0.46 | 0.24 | 0.24 | 0.34 | 0.38\* |
| 0.25 | 0.70 | 0.26 | **0.20** | 0.26 | 0.37 |
| 0.35 | 1.32 | 0.49 | 0.32 | 0.25 | 0.31 |
| 0.50 | 1.94 | 0.79 | 0.77 | 0.50 | 0.70 |
| 0.575 | — | 1.02 | 0.95 | — | 1.12 |
| 0.65 | 2.48 | 1.07 | 1.18 | 0.70 | 1.23 |

cos falls from 0.86 at t_c 0.25 to 0.62 at 0.65. Sample noise is ~0.01
(32 → 64 samples moved cos by −0.010, −0.012, −0.009, −0.004, −0.007), so the
0.14 swing across t is ~14× the noise.

\* `t_c 0.20 / Δ=0.5` queries `r = −0.05` and is **out of domain** —
`true_average_velocity` integrates from `r` with no clamp. The mirror cell
`t_c 0.80 / Δ=0.5` queries `t = 1.05` and returns cos −0.056, `D_r_true` 68%.
Both must be dropped. **Check `r = t_c − Δ/2 ≥ 0` and `t = t_c + Δ/2 ≤ 1`
before reading any cell.**

### 2.4 Coverage ruled out

`meanflow_config.json` reads back `max_t: 1.0`, `min_t: None`,
`t_sampler: None` — uniform t, nothing capped. Interval-center density under
uniform t with Δ ≤ 0.5 is symmetric about 0.5; performance is sharply
asymmetric. Equal data, unequal results.

A second argument from the data itself: at t_c 0.65 the failures are at Δ 0.05,
0.1, 0.2 and 0.5, but Δ=0.3 passes. A hard t cap near 0.8 would make Δ=0.3
(t = 0.80) the *worst* cell, not the best. Degradation tracks `t_center`, not
the largest t reached.

### 2.5 The target-quality probe measures a proxy, not the loss

Running `probe_target_quality.py` across t appeared to show the training target
becoming *worse than predicting no correction* above t ≈ 0.25
(`tgt_vs_base` for `fd=0.025/marg` at Δ=0.1: 0.83, 1.02, 1.39, 1.94, 2.91,
3.35 at t_c 0.20 → 0.80). The derived target magnitude ratio ρ was ~1.8–2.5 and
at one cell landed on 1.9997 — a clean factor of two, with a clean derivation
to match: substituting `dv/dt` for `∂u/∂t` while keeping the full Δ doubles the
correction term, since `∂u/∂t ≈ v′/2` to first order.

**That reading was wrong, and reading eleven lines of source overturned it.**

- The probe (line 421) builds `c = (-d * du_dt)` with `du_dt` from
  `variants(field, z_t, t, w)` — the derivative of the **frozen field**. The
  Δ/2 factor applies here.
- The training loss (`meanflow_ref.py` line 309) defines `def fn(z_, t_, r_)`
  and line 342 computes `u_tgt = (first - delta * du_dt)` from a jvp/central
  difference **of that three-argument function** — i.e. `∂u/∂t` at fixed `r`,
  the MeanFlow identity done correctly. There is no factor of 2 in training.

So the probe's `tgt_vs_base` rankings — including the original selection of
`fd=0.025/marg` as the training variant — grade a non-bootstrapped proxy, not
the objective. They cannot be read as statements about the training target.

A wider ε sweep also killed the "exactly 2": ρ slides monotonically with ε
(2.54, 2.11, 2.00, 1.91, 1.78, 1.61 at ε = 0.00625 → 0.2) and never settles.
The 1.9997 was one cell of a curve passing through 2. **Two coincidences
reinforcing each other is the shape of a wrong answer that feels right.**

### 2.6 Finite differences are roundoff-limited, not truncation-limited

Shrinking ε makes the probe's target *worse* on both direction and magnitude
(cos at Δ=0.1: 0.691, 0.839, 0.901, 0.924, 0.911, 0.843 for
ε = 0.00625 → 0.2). Roundoff, which goes as δ/ε, dominates the O(ε²)
truncation term in float32 at these magnitudes. `fd_eps = 0.025` sits on the
wrong side of the floor; ε = 0.05–0.1 is better. This is a numerical property
of central differences on this model and does carry to training, even though
§2.5 means the probe's variant *ranking* does not.

### 2.7 The correction-scale hook

Added to `eval_interval_field.py` (§6). For `u(a) = ref + a·(u_hat − ref)` with
`ref = v_inst`, so `a = 0` is exactly the baseline and `a = 1` exactly the
model, it reports `a_opt` (the scale minimising summed SSE, in float64) and
`vs_base_at_a_opt` — the ceiling of the entire a(t) programme, with no fitting.

Measured `a_opt`, 32 samples:

| t_c | 0.20 | 0.25 | 0.30 | 0.35 | 0.425 | 0.50 | 0.575 | 0.65 | 0.80 |
|---|---|---|---|---|---|---|---|---|---|
| Δ=0.1 | 1.11 | 0.84 | 0.76 | 0.65 | 0.55 | 0.53 | 0.50 | 0.50 | 0.49 |
| Δ=0.2 | 1.28 | 1.06 | 0.88 | 0.75 | 0.64 | 0.54 | 0.51 | 0.46 | 0.41 |

No knee — a smooth decay that flattens at ≈0.5 from t_c 0.5 upward. Note
`a_opt = cos/ρ`, so with cos ≈ 0.70 it reflects direction error as well as
overshoot; the norm overshoot alone is ρ ≈ 1.4, not 2.

### 2.8 A single constant recovers the failing region

`--corr-scale 0.5`, 32 samples, against the per-cell ceiling:

| t_c | Δ | `vs_base` | at a=0.5 | ceiling | gap |
|---|---|---|---|---|---|
| 0.50 | 0.1 | 0.857 | 0.393 | 0.391 | 0.6% |
| 0.50 | 0.2 | 0.828 | 0.348 | 0.345 | 0.9% |
| 0.575 | 0.1 | 1.021 | 0.468 | 0.468 | 0.01% |
| 0.575 | 0.2 | 0.949 | 0.410 | 0.409 | 0.07% |
| 0.65 | 0.1 | 1.014 | 0.493 | 0.493 | 0.005% |
| 0.65 | 0.2 | 1.225 | 0.476 | 0.471 | 1.1% |
| 0.80 | 0.1 | 1.030 | 0.627 | 0.627 | 0.02% |
| 0.80 | 0.2 | 1.418 | 0.600 | 0.582 | 3.1% |

97–100% of what per-cell fitting could achieve, from one number.

### 2.9 The two-piece rule, and held-out validation

**a = 1.0 below t_c 0.35, a = 0.5 at and above.** The threshold is not a
separate fitted quantity: a = 0.5 beats a = 1.0 exactly when `a_opt` < 0.75
(the midpoint), which is t_c 0.35. Both numbers fall out of `a_opt`.

In-sample, `vs_base` under the rule:

| t_c | 0.20 | 0.25 | 0.30 | 0.35 | 0.425 | 0.50 | 0.575 | 0.65 | 0.80 |
|---|---|---|---|---|---|---|---|---|---|
| Δ=0.1 | 0.228 | 0.237 | 0.300 | 0.305 | 0.340 | 0.393 | 0.468 | 0.493 | 0.627 |
| Δ=0.2 | 0.229 | 0.190 | 0.214 | 0.308 | 0.332 | 0.348 | 0.410 | 0.476 | 0.600 |

Held out at four unseen t-centers, 64 samples, including Δ=0.5 which the rule
was never fitted on:

| t_c | 0.275 | 0.39 | 0.46 | 0.70 |
|---|---|---|---|---|
| Δ=0.1 | 0.288 | 0.314 | 0.371 | 0.606 |
| Δ=0.2 | 0.204 | 0.327 | 0.334 | 0.518 |
| Δ=0.5 | 0.328 | 0.438 | 0.438 | 0.625 |

Twelve cells, all below 1, worst 0.625, none degraded versus unscaled. The
strongest single point is t_c 0.70, which had no fitted value of its own: raw
`vs_base` 1.255 / 0.939 / 1.256 → 0.606 / 0.518 / 0.625, with `a_opt` there
still on the 0.5 plateau.

The rule's one real loss is **t_c 0.275, Δ=0.5**, where `a_opt` = 1.28 and the
rule applies 1.0 (0.328 against a 0.295 ceiling). `a_opt` is Δ-dependent at low
t (0.78 / 0.99 / 1.28 across Δ at that t_c), so closing it needs a(t, Δ). It is
a 10% loss in a region already winning 3×. Not worth two dimensions of fit.

---

## 3. Verdict on Gate B — PASSED, with the method stated

**The adapter beats the frozen instantaneous field at every t_center from 0.20
to 0.80 and every Δ from 0.1 to 0.5, given a single inference-time constant:
scale the correction by 1.0 below t_center 0.35 and 0.5 at or above.**

That sentence, including its second clause, is the claim. The constant and the
threshold are part of the method and belong in the abstract, not a footnote.
Without them the checkpoint loses to the baseline above t_center ≈ 0.55.

What this is **not**: a better model. `corr_err_best` = 1 − cos² is untouched
by any rescaling, and it is now the binding constraint at every cell (0.26 at
t_c 0.275 rising to 0.67 at 0.70). The scale fix converts a calibration
problem into a pure direction problem. Direction is what a retrain would have
to attack, and nothing in this stage suggests how.

Scope limits to state with it: held out in t and Δ but **not in data** — every
run uses the same hard-coded sample seed (`default_rng(0)` for selection,
`manual_seed(1234)` for noise; the eval has no `--seed` flag). And all of §2.7
onward comes from one machine (§5).

---

## 4. Withdrawn / corrected claims

- **1B §2.5's "beats base at Δ ≥ 0.5"** — true only at t_center 0.5. Restate
  with its t range or with the §3 method.
- **1B §2.7's global scale `a ≈ 0.59`** — a function evaluated at one point.
  The implied optimum runs 1.28 → 0.41 across the t × Δ grid.
- **1B §5's "the anchor-SSE climb was a clipping artifact"** — this run is
  unclipped (`grad_clip_frac` 0.0 throughout) and anchor SSE still climbs
  0.14 → 339 by step 1000, decaying to ~170. The *conclusion* stands (it is
  ~100× smaller than under `e − x0`) but the explanation does not. The climb
  is real, slow, and is what the residual +77 at Δ=0.05 is.
- **"The training target is worse than nothing above t ≈ 0.25"** — withdrawn
  entirely. That measured the probe's frozen-field proxy (§2.5).
- **The factor-of-2 target bug** — raised and withdrawn in the same stage. It
  is real in the probe, absent from the loss.
- **`fd=0.025` as a chosen operating point** — selected on probe rankings that
  §2.5 shows grade a proxy, and §2.6 shows it sits on the roundoff side of the
  FD floor regardless.

Non-signals to stop reading: `loss_is_saturated` is meaningless under
`--norm-p 0` (the weight is 1, so it just reports sse > 0.99); `val_interval`
rose 60k → 118k across this run while the real eval improved, exactly as 1B
§7.6 predicted.

---

## 5. Environment — two boxes disagree by more than the effects measured

`sse_base` at t_c 0.65, Δ=0.1, same checkpoint, same 32 samples, same seeds:
**301.49 on one box, 319.9 on the other — 6.1%.** `sse_base` involves only the
frozen model and the teacher integration, no adapter, so this is purely
environmental: python 3.11 `dist-packages` + one torch build versus python 3.12
miniconda + another.

It flips a verdict. At t_c 0.65, Δ=0.1 one box reports `vs_base` 1.014 (loses)
and the other 0.967 (wins). **No boundary claim near 1.0 is safe across boxes,
and rows from different boxes must not share a table.** Designate one canonical
eval box, and disable TF32 in `eval_interval_field.py` the way
`probe_target_quality.py` already does by default — its own comment notes TF32's
~1e-4 relative precision is the size of a fine finite-difference step.

Throughput, for planning: **1.38 s/step** (5000 steps in 1.92 h) on the Pro
6000, ~8× the 3090 Ti's ~11 s/step. A 5k run is under two hours.

---

## 6. Code added this stage

`scripts/eval_interval_field.py`:

- `--corr-scale EXPR` — python expression in `tc` / `t` / `d` giving the scale
  applied to the model's correction. Evaluated with empty builtins; trusted
  input.
- `--corr-ref {v_inst, model_r0}` — origin the correction is rescaled about.
  `v_inst` makes a=0 the frozen-field baseline, so `vs_base_scaled` is on the
  same axis as `vs_base`.
- `scaling_metrics()` — emits `a_opt`, `sse_model_at_a_opt`,
  `vs_base_at_a_opt`, `a_applied`, `sse_model_scaled`, `vs_base_scaled`,
  `excess_over_base_scaled`, `corr_ref`, `a_opt_alt_ref`. Dot products in
  float64 (these are ~1e6-element reductions). `a_opt` minimises the **summed**
  SSE — the quantity `sse_model` reports — not the mean of per-sample optima,
  which is a different and unachievable number.

Self-test, and it was run: `--corr-scale 1.0` reproduces `vs_base` to every
digit; `--corr-scale 0.0` with `--corr-ref v_inst` gives exactly 1.000.

Still missing and worth adding before the next eval campaign: a `--seed` flag
(both RNGs are hard-coded), a domain guard rejecting `r < 0` or `t > 1`, and
the TF32 lines from §5.

---

## 7. What to do next — Gate C, and only Gate C

Gate B is closed. Gate C has never been run, across three stages. Per
START_HERE and 1B §7.3, it is mostly CPU work and needs no GPU to start:

1. Fetch RefEdit-Bench and MIRA-Bench; record what masks each actually ships.
2. Write the loader; count candidate cases with ≥ 2 same-category instances.
3. Calibrate the leakage metric on ground-truth edits **before** trusting it.
4. Only then the GPU parts: the RefEdit-SD3 baseline, and `instance_bench.py`.

Note from 1B §6: "no benchmark for instance selectivity exists anywhere in the
field" is no longer true, and the framing must be corrected before it goes in a
paper.

Do **not** do: a longer training schedule (§2.2 says it is flat from 4000);
fitting a(t, Δ) (§2.9, 10% of a region already winning 3×); retraining with the
scale folded in (§3 — it buys nothing the post-hoc scale does not, because
direction is binding); further probe-variant rankings until §2.5's proxy issue
is fixed.

---

## 8. Habits that earned their place again

1. **Sweep the axis you never varied.** One `t_center` produced a headline that
   was wrong by a factor of two in reach. Nothing else this stage mattered as
   much.
2. **Check the premise before applying the branch.** §2.1's pre-registered
   conditional assumed `corr_err > 1` implied bad direction. It did not.
3. **Read the source of any instrument you are about to draw a conclusion
   from.** Eleven lines overturned a derivation that had a clean number and a
   clean mechanism behind it.
4. **Widen the sweep before believing a constant.** ε = 0.025 alone gave
   1.9997. Four more points showed a curve.
5. **Verify the hook before reading its output.** The a=1 / a=0 self-test cost
   one run and would have caught a silently wrong scale.
6. **Check `vars(args)` in the output config to confirm a flag arrived.** Seven
   result files were read as new runs before the absent `corr_scale` key showed
   they predated the patch.
7. **Compute the ceiling before fitting.** `a_opt` bounded the entire a(t)
   programme in one run, with no fitting, and showed a constant would do.
8. **Hold out.** The rule was fitted on nine t-points at two Δ and validated at
   four unseen t-centers at three Δ, on doubled samples.
