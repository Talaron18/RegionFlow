# Stage 0 — findings of record

Everything below is measured on this hardware with the code in this repo. Each
claim carries its evidence and its n. Claims that did not survive are kept, not
deleted, because several of them were believed for hours and the reasons they
failed are the useful part.

---

## 1. The headline result

**Edit strength and source destruction are the same axis.**

The method has two nominal controls: τ (inversion-depth pin, temporal) and λ
(attention bias on edit tokens, spatial). On paper they are independent. In the
output they are not — both move the result along a single continuum from
"source retained, no edit" to "source destroyed, noise fill". There is no
setting that preserves structure while changing an attribute.

Measured on a 3×3 grid, τ ∈ {0.5, 0.7, 0.85} × λ_in ∈ {4, 10, 16}, three objects:

| statistic | ball (17320) | sheep (158548) | booth (429011) |
|---|---|---|---|
| **corr(dE, struct)** | **−0.958** | **−0.726** | **−0.955** |
| corr(λ_in, struct) | −0.820 | −0.890 | −0.871 |
| corr(λ_in, sat) | −0.873 | −0.471 | −0.850 |
| corr(struct, sat) | +0.951 | +0.204 | +0.969 |
| corr(λ_in, dE) | — | — | +0.891 |

The three cases differ in nearly every respect that could confound them:

| | ball | sheep | booth |
|---|---|---|---|
| edit | green→blue (chroma) | white→black (lightness) | blue→red (chroma) |
| mask | 15.1% | 0.93% | 6.15% |
| SAM IoU vs GT | 0.822 | 0.418 | 0.856 |
| reconstruction | 31.2 dB | 25.6 dB | 25.8 dB |
| source saturation | 0.619 | 0.164 | 0.828 |

The coupling holds across all of it. dE is CIE76 distance in L\*a\*b\*, which
covers chroma and lightness edits alike; `dhue` is undefined for a pure
lightness move and scored the sheep's edit as hue noise.

**The sheep's outlier is explained, not anomalous.** Its corr(struct, sat) of
+0.204 against +0.951 and +0.969 on the other two is a floor effect: source
saturation 0.164 leaves nothing for chroma to lose. Both chromatic objects agree
to within 0.02. Read struct, not sat, on desaturated targets.

**The mechanism is clearest in the booth's chroma table**, ordered by colour
movement:

    dE  7.3  ->  saturation 96% of source,  struct 0.645
    dE 21.1  ->             80%,            struct 0.471
    dE 29.0  ->             41%,            struct 0.350
    dE 46.3  ->             15%,            struct 0.102
    dE 69.2  ->             26%,            struct 0.074

Colour movement is bought entirely with chroma and structure. No cell makes the
box redder rather than greyer.

**The edits also move in the wrong direction.** Measured dL\* is positive in all
nine ball cells (+0.5 to +18.2) and positive at λ ∈ {10, 16} on both the sheep
and the booth — the regions wash out toward light and neutral rather than
acquiring the requested colour.

**Three automated "window" verdicts were issued and all three were false**, each
refuted by looking at the image within seconds:

- sheep τ=0.85 λ=4 (dL\* −9.01, correct sign): body replaced by a dark maroon
  blob, head deformed.
- booth τ=0.85 λ=4 (dE 21.1, dhue +23.9°, sat 80% of source): still blue, shifted
  toward purple, with door panels, windows and "POLICE" lettering dissolved.
- ball, earlier grid: hue movement at λ=10 was the hue of carpet texture.

dE, dL\* and struct measure movement, direction and edge retention. None of them
measures correctness, and a metric-only protocol would have recorded three
successes here. The contact sheet is not optional.

λ = 4.0 is therefore not a weak edit. It is the only setting that preserves the
object, and it produces no edit.

**What this does not say.** Localisation works, and works well: background
change 0.012 across all nine ball cells, seam excess near zero, SAM IoU 0.82.
The failure is not in finding the region. It is that the mechanism can only
choose how much of the region to discard, never what it should become.

### Consequence for the project

The α-pin computes `z = w·z_source + (1−w)·z_generated`. The only quantity under
control is w. That formulation cannot express "keep the geometry, change the
colour" because it has no term for attribute direction — only a mixing ratio
between the original and a fresh sample.

This is the empirical case for learned conditional transport, and it is stronger
than the one the project started with. BiFM is not a speed optimisation and not
merely a better inverter. It is a mechanism that learns *where to move on the
manifold*, which is the degree of freedom the current design provably lacks.

### Residual caveats

The small-mask worry raised by the sheep (0.93% of image, IoU 0.418) is closed:
the booth sits between the other two at 6.15% with IoU 0.856 and reproduces the
ball to within 0.003 on corr(dE, struct).

What remains: the booth's reconstruction gap is 8.99 dB (25.78 against a 34.77
ceiling), the worst of the three, and even its best-preserving cell has lost the
"POLICE" lettering. Some of what `struct` scores as structure loss at λ=4 is
inversion error rather than the edit mechanism. This does not affect the
correlations — inversion error is constant across a grid — but the λ=4 cells are
not a clean no-op baseline, and should not be described as one.

---

## 2. Solver

**Heun, 50 steps.** Settled; do not sweep again.

n = 12 MagicBrush photographs, paired:

| comparison | NFE | result |
|---|---|---|
| heun-50 vs euler-100 | 200 vs 200 | +3.08 dB, 9/12 wins |
| heun-50 vs euler-200 | 200 vs 400 | +1.04 dB, 6/12 — tie at half the compute |

Heun wins at matched NFE and matches doubled-Euler at half the model calls.
Fixing this as the operating point keeps it out of every downstream comparison.

Unexplained and not worth chasing: the heun/euler correlation across images is
0.21–0.26, while euler-200 vs euler-50 is 0.53. The solvers fail on different
photographs. Noted for the report, not investigated.

---

## 3. Inversion error is structural, not noise

Global roundtrip is 10.73 dB below the VAE ceiling at the best setting (n = 12).
But the character of that error matters more than its size.

Downsampling the reconstruction 32× raises PSNR only 17.27 → 25.09 dB.
Independent noise would gain ≈ 30 dB over the same range; the measured gain is
7.8. The error therefore lives at low spatial frequency. Visually confirmed by
hand annotation of a 512×512 reconstruction: local texture, object identity,
labels and small text all survive, while the global geometry is rearranged — a
tiered shelf reconstructed as diagonal stairs.

**PSNR misranks this error.** Scoring hand-drawn polygons on that reconstruction:

| annotator's label | PSNR |
|---|---|
| "nice reconstruction" | 21.44 |
| "bad reconstruction" | 19.82 |
| "looks alike but different" | 16.18 |
| cans (22 polygons, best-preserved objects) | 16.79 |

The region judged *bad* scores 3.6 dB above the one judged merely *different*,
and the best-preserved objects score lowest of any category. PSNR penalises
displacement of a well-reconstructed object more than blurring of an
unstructured one. Report masked and global PSNR, and do not use either as a
quality proxy without looking.

**Implication for the bottleneck argument.** The α-pin copies source latents
into the preserved region, so the background never travels through the drifting
trajectory. What an edit actually pays is the in-mask reconstruction error, not
the global figure. On 17320_t1: global 31.24 dB, in-mask 35.09 dB. The 10.73 dB
headline overstates the cost to the editing task.

---

## 4. Spatial band radius r

**r does not do what it was introduced to do.** Across every measurement:

- selectivity flat at 0.94/0.95/0.95/0.93 for r ∈ {0,1,2,4} — no dynamic range
- r increases spill: shape edit went 676 → 6105 px outside the dilated mask
- seams are suppressed by `softness`, not by r (synthetic)

r = 0 is used throughout the current experiments.

---

## 5. Softness — synthetic result did NOT transfer

On synthetic colour edits with exact masks, softness cut seam excess roughly
10×, and the temporal-handover explanation was constructed to fit it.

On real photographs with SAM masks (12 runs, 3 cases):

    softness 0.00 → mean seam −0.08
    softness 0.08 → mean seam −0.11

Flat. The synthetic finding does not reproduce under real photographs and
imprecise masks. Whether that is because the effect was an artifact of exact
masks, or because seams are simply not the binding constraint once the region is
being destroyed anyway, is unresolved. It should be reported as not transferred,
not as confirmed.

---

## 6. Data

**MagicBrush is usable.** It is the only surveyed dataset shipping a per-edit
mask on every row.

- `mask_img` is the source photo with the edit region punched out as
  **transparency** (DALL·E 2 inpainting convention). The RGB channels carry
  picture content; converting to L composites the mask away and measures image
  brightness. The mask is `alpha < 128`.
- Verified: 13.7% mean coverage, recall 0.658, **8.76× lift over chance**.
- Rows mix resolutions: source 500×500 RGB, mask and target 1024×1024 RGBA.
- `turn_index > 1` edits are applied to the previous turn's *output*. Turn 1 only.
- Ships instructions, not caption pairs. Captions must be hand-written; a wrong
  caption was measured at **3.65 dB**, larger than the entire solver effect.

**Not usable as-is:** PIE-Bench (behind an unanswered Google Form), PIE-Bench++
(529/700 masks are `'0 262144'` placeholders).

**Still blocked:** the leakage metric. It needs masks for non-selected
same-category instances, which no benchmark ships. `523394_t1` ("make one of the
shoes black") is a natural first annotation target.

---

## 7. Bugs found and fixed

Each of these silently corrupted measurements before it was caught.

1. **diffusers `JointAttnProcessor2_0` accepts `attention_mask` and drops it.**
   Every region-bias number before the custom processor was meaningless.
2. **bf16 timestep quantisation.** Timestep passed as `self.dtype`; ±2.0 error at
   σ×1000 is 11% of a step at 50 steps and 43% at 200 — reconstruction got worse
   with more steps.
3. **bf16 ODE-state accumulation.** `x + dt·v` in bf16 rounds the increment
   against an 8-bit mantissa; as dt shrinks the loss grows. euler-200 went
   12.36 → 21.16 dB. Isolated numerically: 2% relative error at N=50, 84% at 400.
4. **σ endpoint asymmetry.** `generate` had no lower clamp and integrated past
   where inversion started, through the degenerate near-zero region.
5. **Generation rebuilt its own σ grid instead of retracing the inversion's.**
   Endpoints snapped to whatever points existed at that step count: measured a
   discontinuity 0.06690 → 0.11846 between 50 and 51 generation steps, costing
   10.6 dB in flat plateaus with cliffs. `generate` now reverses `inv.sigmas`.
6. Prompt mismatch (3.65 dB), α-pin sign inversion, σ endpoint degeneracy
   (15.83 → 39.16 dB), unbounded selectivity, SD3 context length 333 not 410.

---

## 8. Claims that were made and then withdrawn

Recorded because each was believed on evidence that did not support it.

- **"Inversion is unstable on real photographs, swinging 8–10 dB."** False. The
  diagnostic varied generation steps independently of inversion steps, which no
  real code path does. `roundtrip_error` always passed the same N to both legs.
  The instability was an artifact of the test.
- **"euler-200 has a 6 dB advantage."** n=1 artifact of one photograph. Across
  12 it comes second.
- **"Perturbing the noise latent will show amplification."** It does not. 2%
  perturbation costs 0.1 dB.
- **"SAM masks are polarity-inverted."** The verdict came from measuring the
  wrong channel, scored by a test whose two branches summed to 1.0 and so could
  never both fail.
- **"box_preview and the edit mask are in different coordinate frames."** They
  were stale files from an earlier invocation with different coordinates.
- **"corr(struct, sat) ≈ 0.95 is the replication test."** Chroma-specific. It
  gave 0.204 on the sheep because a white object's saturation is already at the
  floor. The mechanism replicated on other statistics; this criterion should be
  retired.

---

## 9. Open work, in priority order

1. **Calibrate `struct`, or drop its verdict.** Edge-energy correlation with
   invented thresholds. Synthetic pairs gave 0.27 / 0.02 / 0.00 for correct
   recolour / noise fill / flat blob — ordering right, scale unestablished.
   Three false "window" verdicts came from these thresholds. Either calibrate
   against labelled successes, or report the trade-off curve with no automated
   pass/fail.
2. **Annotate same-category instances** on a MagicBrush subset to unblock the
   leakage metric. `523394_t1` ("make one of the shoes black") first.
3. `bench/run.py` — referenced by SETUP.md, never written.
4. `max_dist` in `boundary_profile` truncates above 24 px.
5. Weak-edit guard on `selectivity` (NaN below a target-change threshold).

**Done, do not repeat:** the third grid case (booth) and the ΔL\*/dE metric,
both completed. §1 is settled at n=3.

**Not to do:** further τ or λ sweeps. §1 shows the axis is singular across three
objects and two colour directions; more tuning explores a one-dimensional family
that does not contain the target.

---

## 10. Compute

Stage 0 has produced its result on the 3090Ti. Nothing here justifies renting
A100s yet. The next spend should be on the conditional-transport direction in
§1, not on more sweeps of a mechanism now known to be underdetermined.
