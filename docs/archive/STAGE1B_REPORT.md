# RegionFlow Stage 1B — Gate B reopened

Supersedes STAGE1_REPORT.md §4 (Gate B) and its "Revised priorities". Everything
else in STAGE1_REPORT.md stands: Gate A is still closed, the Stage 0 metric
failure is still the reason its headline number is withdrawn.

Read this after STAGE1_REPORT.md and before doing anything.

---

## 0. Where this sits relative to the project's goal

The goal, unchanged since START_HERE.md: **change one named object's attribute
while preserving its form, leaving the rest of the image untouched, and without
affecting other instances of the same category.** Instance-level selectivity is
the research problem. Everything else — inversion, few-step sampling, the
interval field — is infrastructure.

This stage was entirely infrastructure. It took Gate B from "closed, negative"
to a positive result, and that matters, but **Gate C has still never been run,
and `bench/instance_bench.py` is still unrun.** START_HERE's own advice ("if you
only have a day, build the instance-selectivity benchmark") was not followed in
this stage. §7 acts on that.

One change to the Gate C claim, from a literature check run this stage (§6):
"no benchmark for instance selectivity exists anywhere in the field" is **no
longer true**. It was true when START_HERE was written. The contribution
survives, but narrower, and the framing must be corrected before it goes in a
paper.

---

## 1. What Stage 1 said, and what was wrong with it

STAGE1_REPORT.md closed Gate B negative: the adapter learned an
r-**independent** perturbation, `cos(correction, true) ≈ 0`, `D_r` flat at ~1.7%
against a required 3–31%.

That measurement was correct. The diagnosis attached to it was not. Stage 1
attributed the failure to the adapter (rank, conditioning width, anchor ratio).
It was **the regression target**. With the target replaced and nothing else of
substance changed, the same rank-32 LoRA through the same one-vector
conditioning path reaches `cos ≈ 0.72`.

The three things Stage 1 proposed to fix it — P2 (wider conditioning), P4 (anchor
ratio), P5 (low-pass the Δ basis) — were all tested or ruled out this stage.
None of them was the cause.

---

## 2. Experiments, in order

Every gate/verdict below was written down before the measurement it judges.

### 2.1 P1 — is the interval pathway used at all? (CPU, minutes)

Loaded the P75 checkpoints and measured the `IntervalEmbedding` output directly.

- The pathway is **live** (`W_out` off its zero init) but carries almost no Δ:
  the Δ-varying part of its output was 23% of the constant part at step 200,
  falling to 10% at step 1000.
- The weights had **random-walked**. Adam's own noise over the actual LR
  schedule predicts an RMS displacement of 1.64e-4 at step 200; the measured
  value was 1.70e-4. At step 1000: predicted 9.42e-4, measured 1.65e-3, so a
  consistent drift of only ~5% of what a coherent gradient would give.
- A same-norm random readout pushed through the *same* hidden features scored
  0.101 against the trained 0.100 — indistinguishable. And it was identical at
  10% and 76% anchors (0.098), which **killed P4**: the anchor fraction does not
  move this number at all.

Two hypotheses were raised and refuted along the way, both worth recording
because both were plausible: Adam's `eps` throttling the layer (refuted: √v was
~1e-5, three orders above eps), and the anchor population shaping where updates
land (refuted as above).

### 2.2 The centred embedding (P2-minimal)

`d_pool = g(t) + [f(t, Δ) − f(t, 0)]`. The Δ branch is exactly zero at anchors,
so it cannot absorb an r-independent offset; that offset has to go to `g` or the
LoRA, which keep their full anchor gradient. **This is not the Δ-gate the
reference audit removed** — that one zeroed every trainable path on anchors.

Result, 1000 steps, pre-registered criteria **failed**:

- `f` random-walked the whole run (rms/noise 0.93 → 1.03), while `g` pulled
  clear (1.93). Given a Δ-free place to put its correction, the gradient went
  there.
- `D_r` flat at 0.3–0.6% against a true 2.5–31.5%; `cos` 0.003–0.019.
- Diagnostic value: the r-independent excess over base was **unchanged** (concat
  913, centred 908 at Δ=0.05) even though the pooled offset fell from 11.5% to
  1.5%. So that excess does not live in the embedding. See §5.

### 2.3 The target-quality probe (`scripts/probe_target_quality.py`)

The decisive experiment of this stage. It grades the target's correction term
`−Δ·du/dt` against the true correction `u_true − v` at initialisation, with no
training, crossing derivative variants against z-tangents.

At t_center 0.5, 32 samples (reproduces `eval_interval_field.py`'s `D_r_true`
exactly, 2.48% → 31.5%, which is how we know it measures the same thing):

| variant | cos @ Δ=0.2 | samples to SNR 1 |
|---|---|---|
| `total/cond` (the MeanFlow target as trained) | 0.12 | ~70 |
| `lp>=0.5/marg` (fast timestep components stop-gradded) | 0.70 | 1.1 |
| `fd=0.025/marg` (central difference, frozen tangent) | 0.76 | 0.73 |
| `spatial/*` (no time derivative) | −0.05 | ∞ |

The training target's Δ-term was **17–36× larger than the true correction** at
the sample level. The adapter was being asked to find a smooth interval
dependence inside a residual dominated by the derivative of SD3's own
fast timestep-embedding components.

The t-sweep (t_center 0.15 → 0.85) then showed cos falling ~0.03–0.05 per 0.1 in
t *and* per 0.1 in Δ. `fd=0.025` held cos ≥ 0.5 everywhere including t=0.95;
`fd=0.05` broke down at the noise end (0.39). That chose ε=0.025 and
`--max-delta 0.5`, both pre-registered before the run.

**Unresolved, and it should stay on the record:** the JVP disagrees with central
differences by 60–100% at every step size from 1e-2 down to 1e-4, TF32 on or
off. The error falls with ε down to 3e-4 then rises (fp32 roundoff), so the JVP
carries content faster than period ~0.006 — faster than any component of the
timestep embedding. Either harmonics generated inside the network, or a
forward-mode AD artifact. Settling it needs float64. **Every `S` figure computed
with that JVP code, including Stage 1's `S ≈ 11.1`, inherits this caveat.**

### 2.4 fd + frozen tangent, 1000 steps

`--estimator fd --fd-eps 0.025 --tangent frozen --max-delta 0.5`, centred
embedding, everything else matched to the previous run.

First pre-registered positive of the project: `cos` 0.18–0.20 at every Δ (vs
≤0.02), `D_r` rising 1.0% → 1.7%, `f` rms/noise 1.51.

But the *magnitude* was near-optimal for that weak direction already
(`corr_err` 0.96–0.98, against a floor of 0.96 at that cos). Scaling up would
have made it worse. Direction was the binding constraint.

### 2.5 The 5000-step run — the headline result

Same configuration, 5000 steps, ~18 h wall clock (13.3 h + a 4.9 h resume after
a machine crash; resume is automatic from `ckpt_latest.pt` and skips warmup
correctly, since warmup is keyed to absolute step).

At step 5000, 32 samples, t_center 0.5:

| Δ | cos | D_r ratio | corr_err | floor (1−cos²) | excess over base |
|---|---|---|---|---|---|
| 0.05 | 0.626 | 0.86 | 0.682 | 0.595 | +1015 |
| 0.10 | 0.712 | 0.88 | 0.551 | 0.489 | +905 |
| 0.20 | 0.739 | 0.97 | 0.538 | 0.451 | +579 |
| 0.30 | 0.715 | 0.84 | 0.523 | 0.485 | +120 |
| 0.50 | 0.714 | 1.22 | 0.831 | 0.487 | **−293** |
| 0.70* | 0.600 | 0.67 | 0.675 | 0.636 | **−1879** |
| 0.80* | 0.553 | 0.62 | 0.737 | 0.687 | **−2284** |

\* outside the trained range (`--max-delta 0.5`), i.e. extrapolation.

Pre-registered criteria (Δ 0.1–0.5: cos ≥ 0.35, corr_err ≤ 0.9): **passed**, and
`corr_err ≤ 0.6` passed at Δ 0.1/0.2/0.3, failed at 0.5.

**Negative excess over base at Δ ≥ 0.5 is the first time anything in this
project has beaten the frozen instantaneous field on its own terms.**

Trajectory: direction improved to step ~3400 and then stopped (mean cos 0.724 →
0.720 over the last 1600 steps); the entire gain in that window was
calibration (mean `corr_err` 0.735 → 0.611). The LR decay is what fixed the
overshoot — the D_r ratio at Δ=0.5 fell 1.44 → 1.22 and the overshoot at Δ
0.2–0.3 resolved completely.

Training was clean throughout: grad norm flat at 0.13 across the decay, the
1.0 clip essentially never binding, ‖du/dt‖ bounded at 330–395 (the JVP run's
blow-up does not recur), `adp_ess` 0.93 (weights near-uniform, so `--norm-p 1`
was behaving like plain MSE).

### 2.6 The zigzag, and two refuted explanations

A dense Δ sweep found the D_r ratio oscillating — 0.98, 0.93, **1.28**, 0.96,
0.98, 0.83, **1.22**, 0.95, 0.86, 0.65, 0.66 across Δ 0.35→0.6 — while cos
declined smoothly. Replicated across two sample sets at correlation **0.999**,
so not sampling noise.

- **Round timestep values: refuted.** The eval sets t = t_center + Δ/2, so Δ and
  t are coupled. A t-center sweep put two different (t_center, Δ) pairs on the
  same t; they disagreed (1.279 vs 1.321 at t=0.700).
- **Aliased Δ basis (P5): refuted.** `scripts/probe_delta_ripple.py` measured the
  trained embedding's Δ amplitude on a dense grid: ripple 5.2–5.8% against a
  same-norm **random-readout null of 5.0–5.3%**. Indistinguishable from
  untrained geometry. (`centered_p75` does show 14% vs an 8.5% null — but that is
  the run whose cos never exceeded 0.02.)

What survives: the model overshoots **everywhere** in that region (ratio > 1 at
all ten t-sweep points), and the overshoot's size depends on t and Δ jointly in
a way ten points do not resolve. The isolated spikes were a slice through a 2-D
surface, not a feature of Δ. **P5 has no supporting evidence and should not be
built.**

### 2.7 The correction-scale probe

Fits the least-squares scale `a = ⟨c, c*⟩/⟨c, c⟩` on a 3×5 (t_center, Δ) grid.
Applying the optimal scale takes `corr_err` to exactly `1 − cos²`, the floor the
eval already reports — this is not a new metric, it reports *which* scalar gets
there and whether one scalar does it everywhere.

- **One global scalar a = 0.592** takes mean `corr_err` 0.650 → **0.508**, against
  0.489 for a per-cell scalar and a 0.472 floor. Per-cell fitting buys 0.019
  more.
- `a_spread` within a cell is 0.17–0.22, well under the 0.4 threshold: the right
  scale is consistent **across images**, so this is not sample-dependent.
- Caveat: two cells (t_c 0.45/0.5 at Δ=0.3, where `a_pool` is 0.97 and 0.82) get
  *worse* under the global scalar than under no correction at all. A global
  scale is a good approximation, not a uniform improvement.
- `a_pool` does drift with t and Δ (corr −0.70, −0.63; linear fit R²=0.49,
  residuals to 0.24), but exploiting that drift buys 0.019, so the global
  scalar is the honest summary.

**Magnitude accounts for 27% of the remaining correction error. The other 73% is
direction, and nothing found in this stage reduces it.**

### 2.8 The distill arm, unclipped — the drift is resolved

`--first-term frozen --norm-p 0 --grad-clip 0`, 1000 steps, everything else
matched. The target's first term becomes the frozen field `v` instead of
`e − x0`, so anchors distil the teacher rather than re-fitting the data.

**Pre-registered outcome: refit-through-`e − x0` confirmed, spillover refuted.**
Excess over base at Δ=0.05 fell **1015 → 36**, a 96% drop, against a threshold
of <300 for "refit" and >700 for "spillover".

At step 1000, against `fd_frozen_p75` at its own step 1000:

| Δ | excess (distill) | excess (5k run @5000) | cos (distill @1000) | cos (fd_frozen @1000) |
|---|---|---|---|---|
| 0.05 | **+36** | +1015 | 0.207 | 0.179 |
| 0.10 | **−151** | +905 | 0.250 | 0.192 |
| 0.20 | **−455** | +579 | 0.296 | 0.202 |
| 0.30 | **−756** | +120 | 0.291 | 0.198 |
| 0.50 | **−1459** | −293 | 0.283 | 0.181 |
| 0.80 | **−3323** | −2284 | 0.271 | 0.185 |

Three things follow:

- **It beats the frozen instantaneous field at every Δ ≥ 0.1 after 1000 steps.**
  The 5k run only managed that at Δ ≥ 0.5 after 5000. `sse_model` at Δ=0.05 is
  113.8 against a base of 77.8 (`vs_base` 1.46), where the 5k run sat at 1092.8
  (`vs_base` 14.0).
- **Interval learning was not sacrificed.** The pre-registered guard (cos ≥ 0.15
  at Δ 0.1–0.5) passed, and at matched step count the distill arm is *ahead* on
  both cos (0.25–0.30 vs 0.18–0.20) and D_r ratio (0.20–0.57 vs 0.10–0.41). Its
  embedding r-signal is also higher (20.7–35.7% vs 16.6–30.8%).
- **Direction is still weak in absolute terms** — cos 0.21–0.30, `corr_err`
  0.92–0.96, and >1 at Δ=0.05, i.e. worse than no correction there. This arm has
  had 1000 steps; the 5k run needed ~3400 to reach cos 0.72.

**Mechanism.** Under `first_term = cond`, the anchor target is `e − x0`, so the
model is pulled toward the dataset's empirical velocity at r = t and `u(z,t,t)`
drifts off the teacher. Under `first_term = frozen` the anchor target *is* `v`,
and the drift largely disappears. The centred embedding's `g(t)` offset tracks
this: 24.97% at t=0.5 in the 5k run against **0.65%** in the distill arm.

Note the offset is not a clean standalone predictor — `centered_p75` had a 1.5%
offset with the full ~908 excess — so the determining variable is the target's
first term, not the embedding.

---

## 3. Verdict on Gate B

**Gate B is open and positive on Δ ∈ [0.1, 0.5], with a stated ceiling.**

A pretrained instantaneous field converts to an interval field via a rank-32
LoRA through a single 2048-d additive conditioning vector, reaching
`cos(correction, true) = 0.71–0.74`, beating the frozen field outright at
Δ ≥ 0.5, and sitting near its directional floor once a single post-hoc scale
a ≈ 0.59 is applied.

What made the difference was **the target, not the adapter**: a central
difference at ε=0.025 along the frozen field's own tangent, in place of the
total-derivative JVP against `e − x0`.

The honest limitation: direction plateaued at cos ≈ 0.72 by step 3400 and did
not move for 1600 further steps at non-trivial LR. Nothing tried this stage
moves it.

**Superseded in part by §2.8:** the `first_term = frozen` arm beats the frozen
field at every Δ ≥ 0.1 after only 1000 steps, where this checkpoint needed 5000
to manage Δ ≥ 0.5. The combined run (§7.1b) is the one that should be quoted, if
it holds.

---

## 4. Withdrawn / corrected claims

1. **"Gate B closed negative; the adapter learned an r-independent
   perturbation."** Correct as a measurement of that run, wrong as a claim about
   the method. It was the target.
2. **"P4, the anchor ratio, is the next knob."** Dead. The anchor fraction does
   not move the embedding's Δ-sensitivity at all (0.098 at 10% vs 0.101 at 76%).
3. **"P5, low-pass the Δ basis."** No supporting evidence; the ripple probe
   returns the random-readout null. Do not build it.
4. **"The ~2e4 anchor SSE is a mismatch that freezing the first term fixes."**
   Still wrong as stated: that SSE is irreducible flow-matching noise,
   ‖(e − x0) − v‖², predicted by the logged tangent ratio (0.39² × ‖e−x0‖² ≈ 2e4),
   and anchor SSE ≈ 0 under a frozen first term holds by construction at step 0.
   What the frozen first term *does* fix is a different quantity — the model's
   drift from the teacher at r = t, measured as excess over base (§2.8).
5. **"The small-Δ excess is interval updates spilling through the shared LoRA."**
   Refuted by the unclipped distill arm (§2.8, §5). Its supporting evidence came
   from a run clipped on every step.
6. **`S ≈ 11.1` and any other JVP-derived stiffness figure.** See §2.3; the JVP
   is unreconciled with finite differences by 60–100%.
7. **"No instance-selectivity benchmark exists."** See §6.
8. **The dense-sweep "ripple" reading.** An artifact of the eval coupling t to Δ.
   I proposed it; it did not survive the t-sweep. Recorded so it is not
   re-derived.

---

## 5. The r-independent drift at small Δ — resolved

**Cause: the `e − x0` first term. Fixed by `--first-term frozen`.** See §2.8.

For the record, since two hypotheses were entertained and one was wrong:

- Excess over base at Δ=0.05 was ~900–1015 in every `first_term = cond` run —
  concat 913, centred 908, fd 904, fd-5k 1015 — across three targets, two
  embeddings, and pooled offsets from 0.4% to 25%.
- The **spillover** hypothesis (interval updates leaking through shared LoRA
  weights and moving `u(z,t,t)` off the teacher) is **refuted**: with the frozen
  first term the drift is 36, not >700.
- The supporting evidence for spillover — anchor SSE climbing 0.35 → 306 in the
  first clipped distill arm — was an artifact of that run being clipped on every
  step (§7.1). It did not survive the unclipped rerun.
- The removed Δ-gate is therefore **not** needed. Do not reintroduce it.

---

## 6. Literature check (run this stage)

Multi-instance editing benchmarks that did not exist when START_HERE was
written:

- **MIRA-Bench** (Apr 2026) — closest. 100 human-validated images, 3–5 similar
  instances each, motivated by exactly our failure (asked to change the leftmost
  of three sculptures, the model changed all three). Differs: all edits applied
  at once, VLM-judge scoring, synthetic images. Check whether it ships per-instance
  masks — if so they are ready-made siblings.
- **BindEdit** (Jun 2026) — names the problem ("Source Dominance Leakage"), 238
  images, but scores with CLIP/LPIPS outside the object mask, so no
  sibling-specific number.
- **RefEdit-Bench** (ICCV 2025) — real photos, built on RefCOCO, so **every image
  has ≥2 same-category objects with human-drawn COCO instance masks**. Also ships
  RefEdit-SD3, an SD3-family comparator.
- **ALE-Bench** (Dec 2024) — measures leakage, but between *different* objects,
  not same-category siblings.

Also relevant as Gate C baselines (training-free attention interventions on
flow models): Instance-Disentangled Attention, Editing Everything Everywhere.

**Revised Gate C framing.** Not "the first benchmark". The contribution is a
**pixel-level sibling-leakage measure on real photographs**, which the existing
VLM-judge / CLIP / background-LPIPS scores on mostly synthetic images do not
provide. That is smaller, and it stands better as Gate C's measuring instrument
than as a standalone result.

---

## 7. What to do next

Ordered. (1) and (2) are cheap and unblock the rest.

### 7.1 DONE — the clipping test and the distill arm

Completed. `--grad-clip 0` fixed the confound (the earlier arm's loss ran ~2e4×
larger under `--norm-p 0`, so the hard-coded 1.0 clip bound on every step,
turning training into Adam on per-step unit-norm gradients). Result in §2.8:
the small-Δ drift is caused by the `e − x0` first term and is removed by
`--first-term frozen`.

### 7.1b **The next long run: frozen first term at 5000 steps** ← run this first

The two changes that each worked separately have never been combined. The 5k
schedule gave direction (cos 0.72); the frozen first term gives calibration at
small Δ (excess 1015 → 36) and, at matched steps, slightly better direction too.

```bash
PYTORCH_JIT=0 python scripts/gate_b_meanflow.py train --cache /data/cache/13061d51aee4 \
    --estimator fd --fd-eps 0.025 --tangent frozen --first-term frozen \
    --norm-p 0 --grad-clip 0 --dtype float32 --kappa 0 --omega 1 \
    --data-proportion 0.75 --interval-kind centered \
    --steps 5000 --val-every 1000 --ckpt-every 200 --max-delta 0.5 --keep-last 10 \
    --out runs/fd_distill_5k
```

Run §7.2 first — this is the run that pays for the throughput work.

**Pre-registered reading** (32 samples, Δ 0.1–0.5, against `fd_frozen_p75_5k`):

- **Success:** cos ≥ 0.7 at every Δ **and** excess over base < 0 at every Δ
  including 0.05. That is a model that beats the frozen instantaneous field
  everywhere in the trained range — a cleaner Gate B result than §3's.
- **Partial:** excess stays negative but cos plateaus below 0.5. Then the frozen
  first term trades direction for calibration, and §3's checkpoint stays the
  headline.
- **Watch `corr_err` at Δ=0.05**, which is >1 in the distill arm at step 1000.
  If it is still >1 at 5000, small-Δ direction is a separate problem from
  large-Δ direction.
- Re-run `probe_correction_scale.py` afterwards: if the overshoot is gone,
  `a_global` should come out near 1 and the §7.5 calibration becomes unnecessary.

### 7.2 Measure throughput before the next long run (§8)

`--bench-steps` now exists. 5 minutes, and it decides the configuration for
every run below.

### 7.3 Gate C — the actual research problem

This is the project's contribution and it has never been touched. All CPU work,
and it can run alongside any GPU job.

1. Fetch **RefEdit-Bench** and **MIRA-Bench** first (both small), check which
   masks they ship, and pull COCO instance annotations for the RefCOCO images.
   Keep ImgEdit for metric calibration — it is the only source here with
   ground-truth edited images.
2. Write the loader. `bench/instance_bench.py` has the filter and the metric but
   **no loader**; `build()` expects `records` and `load_masks` for files it has
   never seen. Record the real schema before writing code.
3. Count candidates before touching images: edits where the target category has
   ≥2 instances and exactly one was edited. That number decides whether 300–500
   stratified cases is realistic.
4. **Calibrate the leakage metric on ground-truth edits before trusting it**
   (habit #1). ImgEdit's edited images come from an automated pipeline; cases
   where the "ground truth" hits siblings cannot serve as positives.
5. Baselines: RefEdit-SD3, plus one training-free attention method.

### 7.4 Direction, if Gate B is worth pushing further

cos has plateaued at 0.72 and 73% of the remaining error is direction. Two
candidates, neither cheap, neither supported by evidence yet:

- **Longer schedule.** The only untested lever that is pure compute. 10k steps,
  ~18 h at current throughput (less after §7.2). Pre-registered: success is cos
  ≥ 0.8 at Δ 0.1–0.5 by step 10k; a flat cos closes the "under-trained"
  hypothesis for good.
- **Distillation onto `u_true` directly** (integrate the teacher during
  training). This is the ceiling experiment: it separates *target noise* from
  *adapter capacity*. Expensive per step (substeps × forwards), so run it at
  small Δ only, as a diagnostic, not as a training recipe.

### 7.5 Reporting the scale calibration

If any result is published from the 5k checkpoint, report `a ≈ 0.59` as what it
is: a **post-hoc calibration fitted on the eval set**, with the two Δ=0.3 cells
where it hurts. It is not part of the model.

### 7.6 Do not do

- P5 (low-pass the Δ basis) — §2.6.
- P4 (anchor ratio) — §2.1.
- Further dense Δ sweeps at fixed `t_center`. The eval couples t to Δ; any
  structure found that way needs a t-sweep before it means anything.
- Treating `loss`, `val_interval`, or `sse_interval` as progress signals. All
  self-referential; Δ-normalised interval SSE *rose* 14%/1k steps while the
  eval's cos improved. `eval_interval_field.py` is the judge.
- Quoting Δ=0.7/0.8 numbers without the "extrapolation beyond `--max-delta 0.5`"
  caveat.

---

## 8. Throughput — what was found and what changed

The 5k run took ~11 s/step at `--batch 2 --accum 8` (16 samples/step) and peaked
at **9.5 of 24 GB**. There is real headroom. Three levers, in expected order of
payoff:

1. **Micro-batch.** `batch × accum` is what the optimiser sees, and the loss is a
   per-sample mean, so `--batch 8 --accum 2` gives the **same gradient** as
   `--batch 2 --accum 8` while cutting per-micro-batch overhead 4×. This is the
   safest win and needs no code change — only a flag.
2. **Gradient checkpointing.** It was unconditionally on. It is *required* for
   `--estimator jvp`, but `fd` uses no forward-mode AD, so for every run in §7 it
   is purely a memory/speed dial, typically worth 30–40% of step time.
   `--no-grad-checkpointing` now exists (refused with `jvp`).
3. **Data loading.** `persistent_workers` is now on — the trainer re-iterates the
   loader on `StopIteration`, which was respawning all 4 workers every epoch, and
   at batch 2 an epoch is short. `--num-workers` and `prefetch_factor=4` added.

**Not recommended without measurement:** bf16 weights. It would halve the 8 GB
of fp32 weights and use the 3090Ti's tensor cores, but the fd estimator takes a
difference of two forward passes ~0.05 apart in t, and bf16's ~0.4% relative
precision would inject roughly 10% noise into `du/dt`. If you try it, re-run
`probe_target_quality.py` in bf16 first and check `cos` for `fd=0.025/marg`
against the fp32 value.

**Measure, do not assume.** `--bench-steps N` runs the real training step
(micro-batches, accumulation, clip, Adam, EMA), after 3 warmup steps, prints
s/step, s/sample and peak VRAM, and exits without writing to the run directory:

```bash
for cfg in "2 8" "4 4" "8 2"; do
  set -- $cfg
  PYTORCH_JIT=0 python scripts/gate_b_meanflow.py train --cache /data/cache/13061d51aee4 \
      --estimator fd --fd-eps 0.025 --tangent frozen --dtype float32 \
      --kappa 0 --omega 1 --interval-kind centered --max-delta 0.5 \
      --batch $1 --accum $2 --bench-steps 10 --out runs/_bench
done
# then the best of those, with checkpointing off:
PYTORCH_JIT=0 python scripts/gate_b_meanflow.py train --cache /data/cache/13061d51aee4 \
    --estimator fd --fd-eps 0.025 --tangent frozen --dtype float32 \
    --kappa 0 --omega 1 --interval-kind centered --max-delta 0.5 \
    --batch 4 --accum 4 --no-grad-checkpointing --bench-steps 10 --out runs/_bench
```

Take the fastest configuration that peaks under ~21 GiB. If `--no-grad-checkpointing`
OOMs, halve `--batch` and double `--accum` — the gradient is unchanged.

---

## 9. Code added this stage

| file | what |
|---|---|
| `transport/conditioning.py` | `CenteredIntervalEmbedding`, `make_interval_embedding`, `interval_kind_from_keys` |
| `transport/meanflow_ref.py` | `tangent` (cond/boundary/frozen), `first_term` (cond/frozen), `central_difference` with an end-safe stencil, static `max_delta`/`max_t` caps, `tangent_rel_vg` / `first_rel_vg` / `adp_ess` / `frac_at_delta_cap` stats |
| `transport/train.py` | window-aggregated logging (`win_*`, `grad_norm_win_mean/max`, `grad_clip_frac`), `clip_threshold` (grad_clip ≤ 0 disables) |
| `scripts/gate_b_meanflow.py` | `--interval-kind --tangent --first-term --norm-p --max-delta --max-t --grad-clip --keep-last --num-workers --no-grad-checkpointing --bench-steps`, startup guards that read every flag back off the object the loss receives, `meanflow_config.json` provenance + resume refusal |
| `scripts/eval_interval_field.py` | per-checkpoint interval-kind detection, refusal on absent interval weights, run-scoped result keys, `corr_err` / `corr_err_best` / `corr_ratio` |
| `scripts/probe_target_quality.py` | **new** — grades target variants against `u_true`, no training |
| `scripts/probe_delta_ripple.py` | **new** — CPU, Δ-ripple in the embedding vs a random-readout null |
| `scripts/probe_correction_scale.py` | **new** — optimal correction scale per (t, Δ) cell, global vs per-cell |
| `scripts/check_interval_branch.py` | **new** — CPU, weight movement vs Adam's own noise walk |
| `tests/test_centered_interval.py`, `tests/test_meanflow_target.py`, `tests/test_grad_clip.py` | **new** — CPU, include a verbatim regression check that the reference path is bit-identical |
| `scripts/test_reference_audit.py` | curriculum ban narrowed to schedule symbols; static default-off caps allowed and checked |

Defaults are unchanged everywhere: an unflagged run trains exactly as it did
before this stage.

---

## 10. Habits that earned their place again

Additions to START_HERE's list, all paid for this stage:

8. **Log window aggregates, not snapshots.** `log_every=25` meant the plots saw
   4% of steps; a single-step grad spike was read as a 200-step regime.
9. **Write the reading key before the measurement, with magnitudes.** Twice a
   criterion was written without one ("cos positive") and a random-walking
   parameter satisfied it.
10. **Score a random control through the same pipeline.** The embedding's
    "learned" Δ-sensitivity was identical to noise pushed through its own
    features. Without the control it read as a finding.
11. **Check what a flag actually reached.** Stage 1's P10 run never ran at 10%.
    Every target flag now has a startup guard that reads it back off the config
    object the loss receives.
12. **When two variables are coupled in the measurement, sweep them apart before
    interpreting structure.** The Δ "ripple" was t.
