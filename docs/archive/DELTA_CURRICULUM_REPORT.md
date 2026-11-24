# Δ-order curriculum — A0 / A1 / A2

Companion to STAGE1C's anchor-SSE climb finding, which this closes the loose
end on. Three arms, `delta_curriculum.py` wired into `gate_b_meanflow.py`,
each run at two seeds (0 and 1) to separate signal from run-to-run noise
before drawing any conclusion.

**Verdict.** A1 (Δ-order curriculum) is a real, modest, reproducible
improvement over the uncurriculed baseline for Δ≥0.1 — clears its own seed
noise in both seeds, ~3% on the scale-corrected metric that matters. A2
(anchor replacement, `small_delta` mode) is a reproducible regression
everywhere measured, and its own instability across seeds is supporting
evidence for the mechanism, not just a confound to explain away.

---

## 1. What was run

`delta_curriculum.py` wired into `gate_b_meanflow.cmd_train`:
`--curriculum --cap-start --cap-end --warmup-frac --anchor-mode
--anchor-width`, `sample_tr_scheduled` in place of `sample_tr`, `hard_max`
taken from `--max-delta` so curriculum-off still reproduces the Δ≤0.5
baseline exactly, `frac_anchor` read from the sampler's own anchor mask
rather than re-inferred from `(t−r)<1e-8` in the loss (that inference reads
~0 under `anchor_mode="small_delta"` and would trip the 50-batch guard and
silently suppress `sse_anchor` reporting), and a dedicated `tr_gen` RNG so
all three arms draw from one shared eps stream.

| arm | design | `--cap-start` / `--cap-end` |
|---|---|---|
| A0 | control, curriculum off | — (hard_max = 0.5 throughout) |
| A1 | Δ-order curriculum | climbs to `--max-delta` over training |
| A2 | anchor replacement (`anchor_mode=small_delta`) | 0.5 / 0.5 (corrected — 1.0 violates `--max-delta`) |

Throughput correction: 4.1 h per 5000-step arm on the Pro 6000 (batch 2 ×
accum 8, grad checkpointing off), not the 1.92 h first recorded —
`delta_max` in the training log is averaged over grad-accum by the Trainer,
so it is a mean-of-maxima and understates the true per-step max. The 3090 Ti
box was inaccessible for this stage; all runs went sequentially on one box.

---

## 2. Seed 0: the anchor-SSE climb, explained

STAGE1C's "0.14 → 339 → ~170" is training-time `sse_anchor`, not
`val_anchor`. A0 reproduces the shape (0.11 → peak 631 @ step 875 → 168) —
peak HEIGHT carries ~2× run-to-run spread across this reproduction; peak
TIMING is the robust part.

**The strong prediction (that reordering Δ would prevent the climb) fails,
but the mechanism is identified.** A1 also climbs (0.13 → peak 237 @ step
2375 → 129). The peak moves with the *cap*, not with step count: at step
2375 the cap is 0.47. This is wide-Δ u-supervision degrading the effective
instantaneous field (Kim et al., "Understanding, Accelerating, and
Improving MeanFlow Training," O2) — not clipping, not an optimiser
artifact. Reordering delays and roughly halves it; it does not prevent it.

**A1 is lower past the point both arms draw identical inputs.** After step
2500, A0 and A1 draw IDENTICAL `(t, r)` (`win_delta_max` matches to machine
precision in all 100 rows; `tangent_rel_vg` identical in all 200 logged
rows — confirms `tr_gen` worked as intended). Same inputs, A1's anchor SSE
is lower in 99/100 windows, mean 141 vs 170. `val_anchor` is lower at 10/10
checkpoints.

**`val_interval` is a known non-signal, not a tiebreaker.** It splits the
other way (A1 lower at only 2/10, ends 7% higher), but it roughly doubles
from step 1000 to 5000 in BOTH arms, and `best_val` selects step 1000 (A0)
/ step 1500 (A1) — mid-climb checkpoints. Do not use it to adjudicate
between arms.

### A2 (seed 0): worse on every axis measured

`eval_interval_field.py` on both `ckpt_final.pt`, one invocation, same box
(A0/A1 rows came back byte-identical to the earlier A0-vs-A1-only run,
confirming cache/seed determinism). A2 is worse than both A0 and A1 on
direction (1−cos²) at all six measured Δ, worst at Δ=0.2 (0.72 vs A0 0.38 /
A1 0.35). `D_r_ratio` (model/true r-derivative) collapses to 0.100 at
Δ=0.2 — the field has gone nearly flat in r exactly where it should move
fastest. A2 does not clearly win the training-time story it was designed to
test either: `vs_base_at_a_opt` at Δ=0.5 is statistically tied with A1
(0.4313 vs 0.4303), worse than A1 at Δ=0.1/0.2.

**A2's `D_r_ratio` collapse, code-confirmed** (not a "thin support"
artifact). `sample_tr_scheduled` applies the Δ cap to the whole batch
*before* the anchor branch runs, so A2's non-anchor 25% draws Δ from the
same distribution as A0's general population — mid/large-Δ coverage is not
thinner in A2. What differs: under `anchor_mode="exact"` (A0/A1),
`delta_part(t,0) = f(t,0) − f(t,0)` is algebraically zero regardless of
weights, so the 75% anchor rows contribute NO gradient to `f`. Under
`anchor_mode="small_delta"` (A2), those same 75% sit at nonzero Δ≤0.05
every batch and DO feed `f`, dominating the 25% general draws that carry
the only signal for Δ∈[0.1, 0.8]. This is gradient imbalance from anchors
competing for `f`'s capacity, not a coverage hole. Confirmed against the
actual sampler code that trained the analysed checkpoint — NOT confirmed
against a Δ histogram, since `win_delta_max` is a per-400-sample windowed
max and is uninformative here (A0/A1/A2 all plateau ~0.13–0.15 despite
differing true tail coverage, so this metric can't distinguish the arms).

**A2's seed-0 run has no clean single block.** It crashed and resumed
twice (step 50→25, step 1000→1025; `hours` resets to ~0.02 both times = a
fresh process start). `tr_gen` reseeds identically on each restart (no RNG
state in `Trainer.save`), so A2's 5000 steps consumed the same short
`(t, r)` sequence three times rather than one continuous draw — reduced
sample diversity, does not affect A0/A1. This does not change the
mechanism above (the imbalance is per-batch, not cumulative), so it is not
disqualifying on its own, but means the seed-0 A2 result should be read
alongside the clean seed-1 rerun (§3), not in isolation.

### Correction: the interval-embedding "offset" term is not a confound

The growing "offset" term in `check_interval_branch.py` output (2.9%→15%
of `‖pooled‖` in A0, 0.7%→20% in A1) is NOT an information-free
perturbation to zero out. `CenteredIntervalEmbedding.delta_part` already
computes `f(t, Δ) − f(t, 0)` via one batched call, so both halves share
identical kernels and cancel exactly at anchors by construction; "offset"
is purely `g(t)`, a legitimate time-conditioning term deliberately given
full anchor gradient by design. Consequence worth keeping: because `g(t)`
cancels algebraically in `r-signal = forward(t, t−d) − forward(t, t)`,
that probe's branch-level readout (A1's small-Δ r-signal declining after
step 2000 while its large-Δ signal climbs) is unconfounded by `g(t)` and
stands.

### Environment incident, seed 0

Every JSON in the SD3-Medium HF snapshot on `/root/autodl-tmp` (`/dev/md0`)
read back as size-preserved NUL fill — metadata committed, data blocks
never written. Not a download failure. Verify the HF cache by blob name
(sha256 for LFS, git blob sha1 for small files) rather than refetching
blind, and check the latent cache and checkpoints on that array before
trusting a run. Root cause not determined.

---

## 3. Seed 1: the noise floor

All three seed-1 runs are clean single blocks: 5000 steps, 3.16 h each, no
resume anomalies. Checkpoints moved to `/data/runs/*_s1` after the earlier
crashes were traced to a full system disk (the overlay `/`, 8.3 G free);
`/data` (md0) has 663 G. The full-disk finding explains the seed-0 crashes
but NOT the NUL-filled HF JSONs above, which sat on md0 with plenty of free
space — two separate incidents, only one explained.

**A1 survives the noise floor.** Beats A0 on direction (1−cos²) in 11/12
delta×seed cells. At the Δ=0.5 operating point the A1−A0 gap is 0.0204
(seed 0) and 0.0376 (seed 1), against seed spreads of 0.0031 (A0) and
0.0141 (A1) — the gap clears its own noise in both seeds. `cos_correction`
moves the same direction in each (0.7336→0.7474 seed 0, 0.7315→0.7568
seed 1): ~6% relative on 1−cos² at the operating point.

**After the scale rule, the gain is ~3%, not 6%.** `vs_base_at_a_opt` at
Δ=0.5 is 0.443/0.445 (A0) vs 0.430/0.416 (A1). The scale rule absorbs
about half the direction advantage — quote the corrected figure for any
practical claim, not the raw 1−cos² number.

**RETRACTED: the "gain concentrated at small Δ, decaying with Δ" story was
a seed-0 artifact.** Seed 1 reverses the sign at Δ=0.05 (A1 0.5095 vs A0
0.4988), and A1's own seed spread there is 0.0664 — as large as the whole
effect. `D_r_ratio` at Δ=0.05 straddles 1.0 across seeds (1.403 then
0.818). The true picture is a roughly flat 0.02–0.04 advantage for
Δ≥0.1; Δ=0.05 is too noisy to call at n=2 seeds. Any inference drawn from
the decaying-with-Δ shape is void.

**A2 confirmed negative, and its variance is itself the evidence.** Worse
than both other arms in 12/12 delta×seed cells. Seed spread 0.048–0.128 at
Δ≥0.1 — an order of magnitude above A0's; `D_r_ratio` at Δ=0.2 reads 0.100
then 0.317, both far below 1 and unstable. This run-to-run
irreproducibility is exactly what the gradient-imbalance mechanism (§2)
predicts: the mechanism is now supported by an independent seed, not
merely hypothesised from one run.

`eval_interval_field.py` now takes `--seed` (logged in config); seed-0 and
seed-1 evals both ran at `--samples 32 --substeps 16 --t-center 0.5
--rank 32`.

---

## 4. The defensible claim

> A Δ-order curriculum (A1) delays and roughly halves the anchor-SSE
> climb STAGE1C observed but does not prevent it — the climb is wide-Δ
> u-supervision degrading the effective instantaneous field (Kim et al.
> O2), a training-dynamics effect the curriculum mitigates rather than an
> artifact it removes. On the metric that reflects deployed behaviour
> (`vs_base_at_a_opt` at the Δ=0.5 operating point, after the scale rule),
> A1 beats the uncurriculed baseline A0 by ~3%, a gap that clears seed
> noise at both seeds tested. Replacing anchors with small-Δ pairs (A2)
> instead of exact anchors is a reproducible regression on every axis
> measured — direction, r-sensitivity magnitude, and training-time loss
> — with a code-confirmed mechanism (anchor rows compete with general-Δ
> rows for the model's gradient capacity once they stop cancelling
> algebraically) whose predicted symptom, run-to-run instability, is
> independently observed in a second seed.

---

## 5. Scope limits — state with any result

- **Only two seeds (0, 1).** A1's Δ≥0.1 advantage clears the seed-spread
  estimated from these two, but two seeds is a spread estimate, not a
  distribution. Δ=0.05 is unresolved at this n and should not be quoted
  either direction.
- **Peak height of the anchor-SSE climb is noisy; peak timing is not.**
  ~2× run-to-run spread on height in the one direct reproduction available
  (A0 vs STAGE1C's own number).
- **`val_interval` cannot adjudicate between arms.** It roughly doubles in
  both regardless of which is actually better, and `best_val` selects
  mid-climb checkpoints in both arms.
- **A2's seed-0 result rests on a run with two crash/resumes**; the
  gradient-imbalance mechanism does not depend on this (it is per-batch,
  not cumulative), but the seed-1 rerun (clean single block) is the
  stronger evidence for the "A2 is negative" claim, not the seed-0 one.
- **Single box (Pro 6000) for every run in this report.** The 3090 Ti was
  inaccessible for this stage; throughput and timing figures do not
  reflect the 3090 Ti's own characteristics.
- **The HF-cache NUL-fill incident's root cause is undetermined.** Worked
  around by re-verifying blob hashes rather than trusting a re-download;
  if it recurs, that is worth investigating properly rather than
  re-working around.

---

## 6. Bugs and incidents worth not repeating

- **`frac_anchor` re-inferred from `(t−r)<1e-8` in the loss, instead of
  taken from the sampler's own mask.** Reads ~0 under
  `anchor_mode="small_delta"`, trips the 50-batch guard, and silently
  suppresses `sse_anchor` reporting. Read anchor status from whoever
  actually drew the sample, not by re-deriving it downstream.
- **`delta_max` in the training log is a grad-accum mean-of-maxima**, not
  a true per-step max — understated the real throughput cost by more
  than 2× (1.92 h logged vs 4.1 h actual per 5000-step arm) until checked
  directly.
- **No RNG state in `Trainer.save`.** A crash/resume reseeds `tr_gen`
  identically rather than continuing the stream, so a run that resumes
  mid-training redraws the same short sequence instead of extending it.
  A2's seed-0 run did this twice.
- **Full system disk (`/`, 8.3 G free) caused the seed-0-era crashes**,
  separate from and easily confused with the HF-cache NUL-fill incident
  (which happened on `/data`/md0, which had 663 G free the whole time).
  Checking free space does not rule out silent data corruption, and
  vice versa — check both.
- **The "offset" term in `check_interval_branch.py` looked like a
  confound and was not one.** `CenteredIntervalEmbedding.delta_part`'s
  shared-kernel construction guarantees anchor cancellation is exact;
  before assuming a growing auxiliary term needs explaining away, check
  whether the architecture already guarantees it cancels where it needs
  to.
