# RegionFlow — Stage 0 report

Status as of the end of the synthetic-image phase. Nothing here has been run on
real photographs or on a benchmark.

---

## 1. Verified working

| claim | evidence |
|---|---|
| `field.reshape(-1)` matches SD3's patch order | 5 probes incl. asymmetric (3,7), (32,21) — all row-major |
| Region bias reaches attention | `assert_bias_applied`, \|bias\| sum 1200 |
| Context length is 333 tokens | measured against real `encode_prompt` output |
| λ field steers attention spatially | 88× attention-mass concentration in the biased quadrant |
| Inversion round trip | 39.16 dB vs 39.19 dB VAE ceiling |
| Boundary metrics discriminate | hard paste 1.60 → 17.68 px transition width across feather widths |
| λ + α produce localised colour edits | selectivity 0.95, background change 0.0035 |

The pipeline works end to end. That is the real Stage 0 achievement.

## 2. Bugs found and fixed

1. **diffusers' `JointAttnProcessor2_0` accepts `attention_mask` and never
   forwards it to SDPA.** The original wrapper approach would have run the
   entire sweep against a model that never saw the field, producing normal
   images and meaningless numbers. Replaced with a custom processor.
2. **SD3 context is 333 tokens, not 410.** CLIP-L and CLIP-G concatenate along
   channels, not sequence. The token masks were the wrong length.
3. **α-pin sign was inverted** — pinned the target to the source and freed the
   background, i.e. edited everything except the selected region.
4. **σ endpoints are degenerate.** At σ≈0 the model cannot know the noise and
   collapses the latent; both endpoints are under-trained under SD3's
   logit-normal timestep sampling. Clamping to [0.05, 0.98] took the round trip
   from 15.83 dB to 39.16 dB and removed a 16px patch-periodic artefact.
5. **`generate()` started at σ=1.0** regardless of where inversion stopped.
6. **`selectivity` was unbounded**, reaching ~700,000 when the denominator
   approached zero. Now bounded to [0,1].
7. **Environment**: `PYTORCH_JIT=0` required (TorchScript segfaults on the first
   forward-mode AD call); pip constraints needed to stop torch-dependent
   packages replacing the cu124 build.

## 3. Negative results

These are the substantive findings, and they run against the original thesis.

**r does not affect selectivity.** 0.94 / 0.95 / 0.95 / 0.93 across r ∈ {0,1,2,4}.

**r does not prevent seams — `softness` does.** With softness=0, seam goes
0.45 → 1.15 → 1.92 as r grows (worse). With softness=0.08 it is flat at
0.10–0.18 at every r. A temporal parameter added as an implementation detail is
doing the work the thesis assigns to the spatial band.

**r increases spill.** On the shape edit, pixels changed outside the dilated
mask went 676 (r=0) → 6105 (r=4), a 9× increase, and the result was blurrier
(gradient 0.0154 → 0.0072).

**r increases transition width, but that is near-tautological.** τ is
constructed graded over r pixels, so the pixel transition tracks it by
construction. It is not evidence of benefit.

**No shape edit has ever succeeded.** "apples → bananas" produced a tinted disc
at every setting tried. The headroom argument — the band's strongest case —
remains untested rather than refuted.

## 4. The σ_max mistake

σ_max is the depth inversion reaches, not a freedom knob. Lowering it noises the
latent *less*, giving weaker edits: target change 0.1158 at σ_max=0.98 vs 0.0278
at 0.7.

It also caps the target region, defeating the α field. τ_target=1.0 asks for the
target to be fully noised, but it can never exceed σ_max. **Background
protection must come from τ_min, not from a global σ_max cap.** Any σ sweep
should hold σ_max near 0.95–0.98 and vary τ_min instead.

## 5. Metric weaknesses

- **`selectivity` is unreliable when the edit is weak.** It is
  target/(target+others); when the target barely changes, it collapses for
  arithmetic reasons. The 0.63–0.73 values in the σ_max=0.7 run are artefacts,
  not leakage.
- **`transition_width` is truncated above 24px** — `boundary_profile` only
  samples to ±24, so r=4 rows reading 30.11 are unreliable. `max_dist` should
  scale with r.
- **Seam metrics read ~0 when the mask boundary sits on a real object edge.**
  Only meaningful with `--dilate-mask` or imprecise masks.
- **Nothing measures perceptual quality.** All metrics are pixel statistics; a
  plausible edit and an implausible one score alike.

## 6. Blocked / not started

- **No real masks.** PIE-Bench is behind a Google Form (submitted, awaiting
  reply). PIE-Bench++ has 529/700 placeholder masks; the 171 real ones use a
  (start, length) pair encoding, now decodable.
- **No real photographs tested.** SAM pipeline written, never executed.
- **Leakage metric cannot run** — no benchmark ships masks for non-selected
  same-category objects.
- **`bench/run.py` does not exist**; the sweep CLI in SETUP.md is unwritten.
- **Stage 2 (BiFM)** unstarted; no public code, reimplementation assumed.

## 7. Where the thesis stands

The original claim was that spatial control implemented as a *graded prior*
would beat gradient optimisation and survive a few-step budget. Stage 0 tests
only the first half — whether grading helps at all — and on the evidence so far
it does not, on one synthetic image, for colour edits.

Two readings, both live:

**(a) The test is too easy.** Flat discs, exact masks, wide separation, colour
edits. Selectivity is 0.94 at r=0, so there is no headroom to improve. The
band's rationale — imprecise masks, shadows, silhouette change — is untested.
Real photographs are the honest next step.

**(b) The temporal parameter is the real finding.** `softness` demonstrably
prevents seams and r demonstrably does not. "When does the handover happen"
may be a better paper than "how wide is the ring". This would be a pivot, not
a failure.

Deciding between them requires real images. Nothing else on the list matters
until that runs.

## 8. Immediate next steps

1. Run on real photographs with `--sam`. Imprecise masks and textured
   backgrounds are where the band's case lives.
2. σ sweep done correctly: hold σ_max ≈ 0.95, vary τ_min.
3. Fix `max_dist` scaling in `boundary_profile`.
4. Add a weak-edit guard to `selectivity` (return NaN below a target-change
   threshold).
5. Validate SAM masks against the 171 real ones before generating any others.
