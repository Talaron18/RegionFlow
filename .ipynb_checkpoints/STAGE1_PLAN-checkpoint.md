# Stage 1 — conditional interval transport

**Thesis.** Few-step flow is infrastructure; conditional average-velocity
transport is the mechanism; instance-level attribute selectivity is the research
problem.

**Hypothesis.** Replacing source/noise mixing with a learned conditional
displacement `z_edit = z_src + Δz_θ(z, c, M)` decouples attribute change from
source destruction, and that decoupling is what makes instance-level selectivity
possible.

Note the `z` argument. The displacement is conditional on the current state, not
a free-floating direction in latent space — it is `(t−r)·u(z, r, t, c, M)`.
Writing it as `Δz(c, M)` would claim something much stronger and much less
likely.

---

## Run order

Gates are ordered so the cheap science comes before the risky engineering, and
so the parallel track survives if every gate fails.

| # | What | Where | Cost | Blocks |
|---|---|---|---|---|
| 0 | metric control | CPU | minutes | Gate A |
| 1 | JVP probe | 3090Ti | ~1h | Gate B |
| 2 | cache build | 3090Ti | ~2h | Gates A, B |
| A | pair transport + decoupling test | 3090Ti | 1–2 days | Gate B is only worth its risk if A survives |
| B | flow conversion, graded | 8×A100 chained | 2–4 blocks | Gate C at low NFE |
| C | instance selectivity | 3090Ti | — | the thesis |
| ∥ | benchmark build | CPU, parallel | days | nothing |

```bash
# 0 — calibrate the verdict function before trusting it
python scripts/test_decoupling_metric.py --image data/source.png

# 1 — decides Gate B's derivative estimator, and local vs A100
python scripts/probe_jvp_mmdit.py

# 2 — encoders run once; training never loads T5 again
python scripts/precompute_cache.py --dataset magicbrush --root /data/magicbrush \
    --out /data/cache --limit 4000

# A — train, then evaluate against the pre-registered threshold
python scripts/gate_a_transport.py train --cache /data/cache/<hash> --steps 8000
python scripts/gate_a_transport.py eval --ckpt runs/gate_a/ckpt_final.pt \
    --cases captions/captions.json --out runs/gate_a/eval
#   ... look at the contact sheets, then rerun with --i-looked-at-the-images

# B — only if A survives. --max-hours chains under the 6h cap.
python scripts/gate_b_meanflow.py train --cache /data/cache/<hash> \
    --estimator fd --max-hours 5.5 --out runs/gate_b
python scripts/gate_b_meanflow.py sweep --ckpt runs/gate_b/ckpt_final.pt \
    --prompts prompts.json
```

---

## What each gate decides

**Gate 0 — metric control.** Constructs two edits with known ground truth and
checks the verdict function returns the right answer on each. This is not
ceremony. The first version of the Stage 1 metric mirrored Stage 0's headline
statistic, `corr(dE, struct)`, with a threshold at |corr| < 0.5. Run against a
luminance-preserving recolour on a real photograph — an edit that destroys no
structure whatsoever — it returned **corr = −0.92** and would have declared the
coupling confirmed. It would have killed the project on a model that was working
perfectly.

Pearson correlation saturates near −1 for any monotone decline regardless of
magnitude. It cannot see effect size, which is the entire question. The primary
statistic is now **struct at fixed dE**, directly comparable to Stage 0's own
ladder (dE 29.0 → struct 0.350). Correlation is still reported, because Stage 0's
headline is a correlation and the comparison must be available, but it decides
nothing.

Measured separation on the control: struct@dE30 = 0.44 for a destroying edit
versus 0.995 for a preserving one, a 20× difference in slope — while correlation
reads −0.967 and −0.935, effectively identical.

A second confound the control quantifies: **gamut clipping**. A structure-
preserving recolour clips ~18% of in-mask pixels at these strengths, and clipping
compresses local contrast, which `struct` reads as structure loss. Report in-mask
clipped fraction next to struct. Some fraction of Stage 0's −0.958 is this.

**Gate 1 — JVP probe.** MeanFlow's objective is one forward-mode derivative
through the MMDiT. Stage 0's smoke test hit signal 11 on four forward-AD probes
and never closed the cause. The probe escalates from one block to the full model
to a finite-difference fallback, each in a fresh subprocess (an inherited CUDA
context across fork is the leading suspect), and reports peak memory. **Finite
differences passing while JVP crashes is a fine outcome** — two extra forwards
per step, Gate B otherwise unchanged.

**Gate A — does a learned transport decouple?** A straight-line flow between
source and target latents: `z_s = (1−s)z_src + s·z_tgt`, `v = z_tgt − z_src`,
one-step readout at `s = 0`. No noise, no inversion, no MeanFlow. Evaluated by
sweeping displacement magnitude `a ∈ [0, 1.5]` on the Stage 0 cases with the
Stage 0 metric code, loaded by file path rather than reimplemented.

Pre-registered: **REFUTED** (hypothesis lives) if mean struct@dE30 ≥ 0.70,
**CONFIRMED** (Stage 0 stands) if ≤ 0.45. Fixed before the numbers exist.

**Gate B — graded, not pass/fail.** The question is not "does conversion work"
but "what is the largest interval Δ at which the converted model still agrees
with the teacher's flow map". A pass/fail gate at Δ = 1 would collapse LoRA
capacity, learning rate, conditioning design and genuine representational
mismatch into one uninterpretable bit. The curve is the result at either end.

**Gate C — the thesis.** Needs the benchmark, which is why the benchmark starts
now and not after.

---

## Structural guarantees

Two properties hold by construction, not by training, and are asserted in code:

1. **Identity at init.** Every new path is zero-initialised on its output side,
   so the wrapped model is bit-identical to pretrained SD3 at step 0. Every
   Stage 0 number therefore remains a valid control.
2. **Boundary condition.** Every new path is gated on Δ = t − r, so
   `u(z, t, t, c) = v_pretrained(z, t, c)` exactly, for all parameters, at every
   point in training. MeanFlow requires this; learning it would spend capacity on
   a constraint already known and leave it approximate exactly where the
   curriculum starts.

`TransportAdapter.check_identity()` asserts both and also asserts the adapter is
*live* — Stage 0 read byte-identical output as a finding three separate times.
`gate_b_meanflow.py sweep` re-asserts the boundary on every checkpoint and voids
the run if it fails.

---

## Inherited rules, carried forward unchanged

- fp32 for anything that accumulates; bf16 confined to the frozen forward
- retrace the grid; a round trip that does not retrace is not a round trip
- report NFE next to every quality number
- masked and global PSNR reported separately
- contact sheet before verdict — automated verdicts were wrong every time
- clear `__pycache__` and grep for the change after every edit

## Open items

- `apply_transport` extrapolates a field trained only at `a = 1`. It is a probe
  of the direction, not a calibration claim. Say so in the writeup.
- The Δ-curriculum end point in `train.py` is a guess. Gate B's whole job is to
  replace it with a measurement.
- Gate B trains without masks (generative flow only). Mask conditioning has to be
  re-introduced before Gate C; whether that needs a second training phase or can
  be joint is unresolved.
- BiFM (arXiv 2603.24942) and BiFlow (arXiv 2512.10953) are different papers by
  different groups. Do not let them merge in the related-work section.
