# RegionFlow — Research Report (Stage 0 complete)

**Status:** Stage 0 finished. The core mechanism has been characterised and
found insufficient. Result is negative, reproducible, and n=3.

**Purpose of this document:** to be the only thing a person needs to read to
resume this project cold. It states the goal, the design and why each choice was
made, what was built, what was measured, what was concluded, what was concluded
and then withdrawn, and what to do next.

**How to read the evidence:** every claim carries its n and its measurement.
Where a claim was believed and later withdrawn, both the claim and the reason it
failed are recorded (§10). Several of those cost hours; the reasons are the
useful part.

---

# PART I — PROBLEM AND DESIGN

## 1. Goal

Region-controlled attribute editing on real photographs. Given a photograph, a
region (mask or box), and an instruction of the form "make *this* object *that*
colour", produce an image in which:

1. the named attribute of the named object changes,
2. that object's identity, geometry and texture are preserved,
3. everything outside the region is untouched,
4. the boundary shows no seam,
5. other instances of the same category are **not** affected (instance-level
   selection, not category-level).

Requirement 5 is the differentiator. Existing mask-based methods recolour "the
apple" by attending to the token "apple", which affects every apple in the
frame. The project's original name for this failure is *instance leakage*.

## 2. Prior work

Five papers were read in full at the start of the project.

| paper | arXiv | what it gives | what it lacks |
|---|---|---|---|
| **MAG-Edit** | 2312.11396 | mask-based cross-attention constraints (token ratio + spatial ratio); genuine region control | 1–5 min/image on an A100 |
| **BiFM** | 2603.24942 | bidirectional flow matching, LoRA on SD3-Medium, trained on MagicBrush, evaluated on PIE-Bench; CVPR 2026 | **no public code**; ~8000 A100-hours to train (32×H100 for ~120 h) |
| **BiFlow** | 2512.10953 | bidirectional normalizing flows, 1-NFE; Appendix D shows training-free inpainting via noise-space locality | JAX/TPU, class-conditional ImageNet only |
| **pMF** (Pixel Mean Flows) | 2601.22158 | latent-free one-step generation, 2.22 FID @256 | generation, not editing |
| **S²Edit** | 2507.04584 | identity token + orthogonality constraint + spatial CA masking | identity-preservation focus, not attribute control |

**The gap identified.** MAG-Edit has control but no speed. BiFM has speed but no
region control. The unclaimed intersection is *fast region-controlled editing*.

**Prior-work risk flagged early:** "Recolour What Matters" (arXiv 2603.18466)
does region-aware editing on FLUX-Kontext. Different base model and paradigm, but
it should be re-checked before any writeup.

## 3. Central hypothesis (as originally stated)

> Instance leakage and feature bleed are *separable* failures with *separable*
> fixes, both derivable from one mask and one radius:
>
> - **λ (attention bias field)** biases image→text attention so edit tokens bind
>   only to the selected instance. Fixes instance leakage.
> - **α/τ (inversion-depth field)** pins non-target tokens to the inverted source
>   trajectory so their content cannot drift. Fixes feature bleed.
>
> A graded band of radius *r* around the mask gives the model geometric headroom
> and suppresses seams.

**Verdict after Stage 0:** the localisation half is correct and works well. The
attribute-control half is false — λ and τ are not independent controls (§9.1).

## 4. Paradigm decision: inversion, not concatenation

Two families were available.

- **Concatenative** (FLUX-Kontext, Qwen-Image-Edit): the source image is a
  conditioning input; the model generates the edit directly. Never inverts.
- **Inversion-based** (SD3 + BiFM): the source is inverted to noise along the
  flow ODE, then regenerated with modified conditioning.

**Chose inversion.** Rationale:

1. The unclaimed gap (§2) lives here. MAG-Edit and BiFM are both inversion-side.
2. SD3-Medium is 2B parameters and co-resides with T5 in 24 GB.
3. Concatenative models never invert, which makes BiFM — the intended speed
   component — irrelevant to them.

**Consequence discovered later:** inversion quality on real photographs is the
dominant reconstruction cost, and the error is *structural* rather than noise
(§9.3). This was not anticipated at design time.

## 5. Base model decision: SD3-Medium

Stayed on `stabilityai/stable-diffusion-3-medium-diffusers` rather than SD3.5.

- SD3.5-Large is 8B and will not co-reside with T5 in 24 GB.
- SD3.5-Medium uses MMDiT-X with dual attention, which would break the attention
  hook the whole method depends on.

## 6. Scope cuts made at design time

Removed from v0 and never implemented: the MLLM edit planner, the geometry and
scene heads, and the global harmonisation stage. The original hand-drawn
architecture (`1805.png`) had MLLM → edit planner + object parser → region
selector (nested target/local/global attention tiers) → local edit flow → global
harmonisation → output. v0 is the region selector plus the local edit flow only.

---

# PART II — MODEL AND IMPLEMENTATION

## 7. Model structure

### 7.1 The two fields

Both derive from one binary mask plus one scalar radius `r` (`band_tokens`).

**λ — region attention field** (`region/lambda_field.py`)

A per-token additive bias on the joint-attention logits. Inside the mask the
bias is `lam_in` (default 4.0) on image→edit-token pairs; outside it is
`lam_out` (default −2.0). A distance transform grades the value across a band of
width `r`, and the field is area-pooled to the token grid at 16 px per token.

The flatten order was verified: `field.reshape(-1)` matches SD3's patch order
(row-major), confirmed with 5 probes including asymmetric shapes (3,7) and
(32,21).

**τ — inversion-depth retention field** (`region/alpha_field.py`)

Per-token time threshold. `pin_weights(t)` returns w ∈ [0,1]; `blend_latents`
computes

    z(t) = w(t) · z_inverted(t) + (1 − w(t)) · z_generated(t)

A token with a low τ tracks the inverted source trajectory for most of the path
and is released only near t → 0. `tau_target = 1.0` releases target tokens
immediately (full regeneration); `tau_min` (default 0.1) governs the preserved
region. `softness` controls how abruptly w hands over.

### 7.2 Attention plumbing

`region/sd3_processor.py` implements `RegionJointAttnProcessor`, installed on all
24 blocks.

**This is not optional.** diffusers' stock `JointAttnProcessor2_0` accepts an
`attention_mask` argument and silently drops it before SDPA. Using it would
produce normal-looking images from a model that never saw the field, and every
number would be meaningless. This was the first bug found and the most dangerous.

SD3's context length is **333 tokens**, not 410 — CLIP-L and CLIP-G concatenate
along channels, not sequence. Verified against a real `encode_prompt` call.

### 7.3 Inversion (`bench/inversion.py`)

`SD3EditPipeline` wraps a `StableDiffusion3Pipeline`.

- `invert(latent, prompt, steps, sigma_min, sigma_max, solver)` → `Inversion`
  (noise, per-σ trajectory, `sigma_end`)
- `generate(inv, target_prompt, source_prompt, lam_field, tau_field, guidance,
  softness, solver, retrace)` → image
- `roundtrip_error(...)` → dict with `roundtrip_psnr`, `vae_only_psnr`

Solvers: Euler (1 NFE/step, O(h)) and Heun (2 NFE/step, O(h²)). Heun's order was
verified at 1.99 against a closed-form ODE (Euler measured 1.02).

**Critical invariants, each learned from a bug:**

- The ODE state must be **fp32**. The transformer runs in bf16; `_velocity` casts
  down on entry and returns fp32. `_step` raises if handed a non-fp32 state.
- The timestep must be **fp32**, not `self.dtype`.
- σ is clamped to [0.05, 0.98]. The endpoints are degenerate.
- `generate` **retraces `inv.sigmas` reversed** rather than rebuilding a grid.

## 8. Codebase map

```
region/
  lambda_field.py    graded λ field, distance transform, area-pool to token grid
  alpha_field.py     τ field, pin_weights, blend_latents, paired_configs
  attention.py       RegionAttentionSpec, build_joint_bias
  sd3_processor.py   custom joint-attn processor (REQUIRED, see §7.2)
  masking.py         instance selection, small-target routing
  prompts.py         token diffing, SD3 encoder specs, build_token_masks
  segment.py         SAM wrapper (mask_from_box), find_similar_instances,
                     make_clip_scorer, validate_against_gt  [never executed]
  pipeline.py        orchestration

bench/
  inversion.py       SD3EditPipeline: invert / generate / roundtrip_error
  metrics.py         leakage, selectivity, transition_width,
                     seam_gradient_excess, seam_normal_alignment,
                     boundary_metrics
  data.py            PIE-Bench, PIE-Bench++, MagicBrush loaders; decode_rle,
                     parse_brackets, verify_magicbrush, dump_magicbrush_overlays
  sweep.py           two-stage λ then r sweep with confound guard

scripts/
  stage0_real_edit.py  real photo + SAM + edit, with pre-declared success criteria
  tau_sweep.py         τ × λ grid with Lab metrics  <- the key experiment
  smoke_edit.py        single-image end-to-end check, solver sweep
  check_raster_order.py, check_processor.py, diagnose_*.py, inspect_data.py
  fetch_assets.sh

docker/              Dockerfile, compose.yml, requirements.txt, smoke_test.py,
                     triage_jvp.py, README.md, .env.example
legacy/              stale root-level duplicates of inversion.py / smoke_edit.py
                     kept only for reference; do not import
```

**Not written:** `bench/run.py`, which `SETUP.md` references.

## 9. Environment

- RTX 3090Ti (24 GB, sm_86), Ryzen 9950X, 64 GB RAM, WSL2
- Code at `/mnt/e/RegionFlow` on drvfs — **this caused repeated stale-file
  failures**; move to native ext4. Data at `/data`.
- Container mounts the repo at `/workspace`.

**Environment gotchas, each cost real time:**

- `PYTORCH_JIT=0` is required; TorchScript segfaults on the first forward-mode AD
  call.
- A pip constraints file is needed to stop torch-dependent packages replacing the
  cu124 build.
- `.env` must live in `docker/`, not the repo root.
- `hf` CLI replaces `huggingface-cli`; each `--exclude` needs its own flag.
- `HF_HUB_DISABLE_XET=1` works around CAS transfer errors on large downloads.
- `ulimit -n 8192` — fd exhaustion presents as `CUDA driver error: unknown error`.
- SD3 + T5 + SAM ViT-H will not comfortably co-reside. Segment first, free SAM,
  then load the diffusion model.
- **Clear `__pycache__` and verify with `grep` after every code change.** Three
  separate times in Stage 0, byte-identical results were produced by stale files
  and briefly interpreted as findings.

---

# PART III — METRICS

## 10. Metric definitions and their known failure modes

| metric | definition | trust |
|---|---|---|
| `roundtrip_psnr` | PSNR(source, invert→generate) | misranks structural error (§13.3) |
| `vae_only_psnr` | PSNR(source, encode→decode) | the achievable ceiling |
| masked PSNR | same, restricted to a region | what the edit actually pays |
| `struct` | Pearson correlation of **edge energy** inside mask | thresholds uncalibrated |
| `sat` | mean HSV saturation inside mask | floor effect on desaturated objects |
| `dhue` | saturation-weighted circular hue rotation | **undefined for lightness edits** |
| `dL*` | signed CIE L\* change inside mask | direction, not correctness |
| `dE` | mean CIE76 distance inside mask | movement, not correctness |
| `seam_gradient_excess` | contour-following edge the edit added | reads ~0 when the mask sits on a real object edge |
| `transition_width` | px for change to fall 90%→10% at boundary | `max_dist` truncates above 24 px |
| `selectivity` | target change / non-target change | no dynamic range observed (0.93–0.95) |
| `leakage` | change on non-selected same-category instances | **cannot be computed** — no dataset ships these masks |

### Metric lessons that generalise

**`struct` must use edge energy, not luminance.** The obvious version — correlate
raw luminance inside the mask — scores −1.000 for a *correct* green→blue recolour,
because luminance weights green at 0.587 and blue at 0.114, so an honest hue
change inverts the luminance ordering. Gradient magnitude is sign-invariant and
is what "did the form survive" actually means.

**`dhue` is blind to lightness edits.** Validated: white→black gives
dhue = NaN, dL\* = −84.29. The sheep experiment's ±20° hue readings were noise on
a desaturated object. Use `dE` as the unified movement metric.

**No automated metric measures correctness.** dE, dL\* and struct measure
movement, direction and edge retention. Three "window exists" verdicts were
issued by the script and all three were false (§10 of STAGE0_FINDINGS). Contact
sheets are not optional.

**Test every "A vs not-A" comparison for degeneracy.** The mask-polarity test
compared recall on a mask against recall on its complement. Those sum to 1.0 by
construction, so one side always "wins" and the test can never fail. It reported
a confident wrong answer.

---

# PART IV — EXPERIMENTS CONDUCTED

## 11. E1 — Synthetic validation (3 red spheres)

Colour and shape edits on a synthetic image with exact masks.

| finding | evidence |
|---|---|
| r has no effect on selectivity | 0.94 / 0.95 / 0.95 / 0.93 for r ∈ {0,1,2,4} |
| r increases spill | shape edit: 676 → 6105 px outside dilated mask (9×) |
| softness, not r, suppresses seams | s=0: seam 0.45→1.15→1.92 with r; s=0.08: flat 0.10–0.18 |
| no shape edit ever succeeded | apples→bananas gave tinted discs at every setting |

**Mechanism proposed at the time:** softness=0 holds the background verbatim to
t=0.10 then releases in one step, so abrupt spatial adjacency creates a seam.
softness=0.08 eases out from t≈0.18 (0.92→0.78→0.50→0.38→0.22), giving a
harmonisation window. Framed as *temporal handover*.

**This did not transfer to real photographs** (§13.5).

## 12. E2 — Real photograph (can.jpg, ~20 identical cans)

First real-photo test. Edits localised correctly — one boxed can recoloured,
others intact — but the round trip was 14.94 dB against a 27.22 dB VAE ceiling.

This experiment produced the session's most expensive confound: the prompt was
left at a default describing a different scene. Fixing it to a matching caption
moved reconstruction **15.67 → 19.32 dB (+3.65 dB)** — larger than the entire
solver effect. Caption quality is a first-class experimental variable.

## 13. E3 — Solver comparison

n = 12 MagicBrush photographs, paired, empty prompt (conditioning held fixed).

| config | roundtrip | inversion cost |
|---|---|---|
| euler-50 (100 NFE) | 20.28 ± 3.35 | 14.28 |
| euler-100 (200 NFE) | 20.74 ± 3.17 | 13.81 |
| euler-200 (400 NFE) | 22.78 ± 5.27 | 11.77 |
| **heun-50 (200 NFE)** | **23.82 ± 3.87** | **10.73** |

Paired differences:

| comparison | NFE | mean | wins |
|---|---|---|---|
| heun-50 − euler-100 | 200 v 200 | +3.08 dB | 9/12 |
| heun-50 − euler-200 | 200 v 400 | +1.04 dB | 6/12 (tie) |

**Decision: Heun, 50 steps, fixed for all downstream work.** Beats Euler at
matched NFE and matches doubled Euler at half the compute.

Unexplained, not pursued: heun/euler correlation across images is 0.21–0.26,
while euler-200 vs euler-50 is 0.53. The solvers fail on different photographs.

## 14. E4 — Dataset qualification

| dataset | verdict |
|---|---|
| PIE-Bench | behind an unanswered Google Form; repo ships code, no images |
| PIE-Bench++ | 529/700 masks are `'0 262144'` placeholders |
| **MagicBrush** | **usable** — per-edit mask on every row |

**MagicBrush specifics that will bite anyone who skips them:**

- `mask_img` is the source photo with the edit region punched out as
  **transparency** (DALL·E 2 inpainting convention). RGB channels carry picture
  content; `convert("RGB").convert("L")` composites the mask away and measures
  image brightness. **The mask is `alpha < 128`.**
- Verified: 13.7% mean coverage, recall 0.658, **8.76× lift over chance**.
- Resolutions differ *within a row*: source 500×500 RGB, mask and target
  1024×1024 RGBA.
- `turn_index > 1` edits apply to the previous turn's *output*. Use turn 1 only.
- Ships **instructions, not caption pairs**. Captions must be hand-written from
  the image; paraphrasing the instruction reintroduces the 3.65 dB confound.
- Instruction distribution: in the first 200 turn-1 rows, 30 are colour/material
  edits; the rest are add/remove/replace, which have never succeeded.

## 15. E5 — τ sweep (the phantom diagnosis)

Stage 0's first real-photo + SAM run produced 4/12 formal passes whose images
were unusable — "phantom" objects. Inside the mask, saturation collapsed
(ball 0.619 → 0.087) while gradient *rose* (1.80 → 4.77). Saturation down with
gradient up is a region replaced by noise, not a recolour.

**Cause:** `RetentionConfig.tau_target` defaults to **1.0** — full regeneration —
and had never been swept. τ=1.0 frees target tokens at t=1, i.e. from pure noise.
The pipeline was not recolouring the ball; it was deleting the ball and asking
SD3 to redraw one from a caption.

Sweeping τ ∈ {0.2, 0.35, 0.5, 0.7, 1.0} at λ=4:

    tau    struct   sat     dhue
    0.20   0.552   0.609   -0.6°     object perfect, no edit
    0.35   0.533   0.608   -0.7°
    0.50   0.474   0.609   +0.5°
    0.70   0.384   0.643   +3.5°
    1.00   0.196   0.087   +3.8°     phantom

Perfect preservation up to τ=0.7, phantom at 1.0, **nothing in between**.

## 16. E6 — τ × λ grid (the headline experiment)

3×3 grid, τ ∈ {0.5, 0.7, 0.85} × λ_in ∈ {4, 10, 16}, three objects.
Fixed: heun-50, softness 0.08, guidance 5.0, r=0, SAM masks.

| | ball (17320) | sheep (158548) | booth (429011) |
|---|---|---|---|
| edit | green→blue (chroma) | white→black (lightness) | blue→red (chroma) |
| mask | 15.1% | 0.93% | 6.15% |
| SAM IoU vs GT | 0.822 | 0.418 | 0.856 |
| reconstruction | 31.2 dB | 25.6 dB | 25.8 dB |
| source saturation | 0.619 | 0.164 | 0.828 |
| **corr(dE, struct)** | **−0.958** | **−0.726** | **−0.955** |
| corr(λ, struct) | −0.820 | −0.890 | −0.871 |
| corr(λ, sat) | −0.873 | −0.471 | −0.850 |
| corr(struct, sat) | +0.951 | +0.204 | +0.969 |

Booth chroma ladder, ordered by colour movement:

    dE  7.3  ->  saturation 96% of source,  struct 0.645
    dE 21.1  ->             80%,            struct 0.471
    dE 29.0  ->             41%,            struct 0.350
    dE 46.3  ->             15%,            struct 0.102
    dE 69.2  ->             26%,            struct 0.074

---

# PART V — FINDINGS

## 17. F1 (headline) — Edit strength and source destruction are the same axis

τ and λ are nominally independent: τ is a *time* threshold on the latent pin, λ
is a *spatial* bias on attention. In the output they are not independent. Both
move the result along one continuum:

    source retained, no edit  <------->  source destroyed, noise fill

corr(dE, struct) is −0.958, −0.726 and −0.955 across three objects that differ in
edit type, mask scale (0.93%–15.1%), SAM quality (IoU 0.42–0.86), reconstruction
(25.6–31.2 dB) and source saturation (0.164–0.828). Every unit of colour movement
costs a proportional unit of form.

**The sheep's −0.726 and +0.204 are explained, not anomalous.** It is the one
desaturated object (source sat 0.164), so chroma has nothing left to lose. Both
chromatic objects agree to within 0.02 on corr(struct, sat). Read `struct`, not
`sat`, on desaturated targets.

**The edits also move in the wrong direction.** dL\* is positive in all nine ball
cells and positive at λ ∈ {10,16} on sheep and booth. Regions wash out toward
light and neutral rather than acquiring the requested colour. This is consistent
with "the region is replaced by a generic sample" and inconsistent with "the edit
happens but too weakly".

**λ = 4.0 is not a weak edit.** It is the only setting that preserves the object,
and it produces no edit.

### Why, mechanistically

In flow matching, low-frequency content — colour, layout, global tone — is
decided *early* (high t); high-frequency detail resolves *late* (t → 0). τ
releases target tokens late, so by the time they gain freedom the colour is
already fixed by the pinned source latents. Freedom late buys texture-level
change only; freedom early (τ→1) starts the region from noise.

"Preserve form, change colour" therefore asks for freedom in the low-frequency
band and pinning in the high-frequency band. τ is a *time* threshold, and time
and frequency are coupled in the opposite direction from what the edit needs.

λ fails for a different reason with the same outcome: an attention bias strong
enough to change the region's content is strong enough to overwhelm it.

### The formal statement

The α-pin computes `z = w·z_source + (1−w)·z_generated`. The only controlled
quantity is `w`. This formulation has no term for *attribute direction* — only a
mixing ratio between the original and a fresh sample. Writing E_s for source
destruction and E_a for attribute change, an ideal editor needs ∂E_a/∂E_s ≈ 0;
this design measures ∂E_a/∂E_s > 0, at correlation ≈ 0.95 on chromatic objects.

## 18. F2 — Localisation is solved; transformation is not

This is the other half of F1 and matters as much.

- background change 0.012–0.026 across every cell of every grid
- seam excess near zero throughout
- SAM IoU 0.82–0.86 on well-formed objects
- masked reconstruction *better* than global (ball: 35.09 in-mask vs 31.24 global)

The failure is not in finding the region. Editing decomposes into **localisation**
(where?) — solved by masks, SAM, position — and **transformation** (what should it
become?) — not solvable by masks at all. Stage 0 solved the first and proved the
second needs a different mechanism.

## 19. F3 — Inversion error is structural, not noise

Global roundtrip is 10.73 dB below the VAE ceiling at the best setting (n=12).
The *character* of that error matters more than its size.

Downsampling the reconstruction 32× raises PSNR only 17.27 → 25.09 dB.
Independent noise would gain ≈30 dB over that range; measured gain is 7.8. The
error therefore lives at low spatial frequency. Confirmed by hand annotation:
local texture, object identity, labels and small text survive, while global
geometry is rearranged (a tiered shelf reconstructed as diagonal stairs).

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
unstructured one.

**Implication:** the α-pin copies source latents into the preserved region, so
the background never travels through the drifting trajectory. What an edit pays
is the *in-mask* reconstruction error. The 10.73 dB headline overstates the cost
to the editing task. Report masked and global PSNR separately.

## 20. F4 — r does nothing useful

Selectivity flat at 0.93–0.95 across r ∈ {0,1,2,4}; r increases spill 9× on shape
edits; seams are governed by `softness`, not r. **r = 0 throughout.**

## 21. F5 — the softness result did not transfer

On synthetic colour edits with exact masks, softness cut seam excess ~10×, and
the temporal-handover explanation (§11) was built to fit it.

On real photographs with SAM masks (12 runs, 3 cases):

    softness 0.00 -> mean seam -0.08
    softness 0.08 -> mean seam -0.11

Flat. Report as **not transferred**, not as confirmed. Whether the synthetic
effect was an artifact of exact masks, or whether seams simply stop being the
binding constraint once the region is being destroyed anyway, is unresolved.

## 22. F6 — bugs found (each silently corrupted measurements)

1. **diffusers `JointAttnProcessor2_0` accepts `attention_mask` and drops it.**
   Every region-bias number before the custom processor was meaningless.
2. **bf16 timestep quantisation.** Timestep passed as `self.dtype`; ±2.0 error at
   σ×1000 is 11% of a step at 50 steps and 43% at 200 — reconstruction got
   *worse* with more steps.
3. **bf16 ODE-state accumulation.** `x + dt·v` in bf16 rounds the increment
   against an 8-bit mantissa; as dt shrinks the loss grows. euler-200 went
   12.36 → 21.16 dB. Isolated numerically: 2% relative error at N=50, 84% at 400.
4. **σ endpoint asymmetry.** `generate` had no lower clamp and integrated past
   where inversion started, through the degenerate near-zero region.
5. **Generation rebuilt its own σ grid instead of retracing the inversion's.**
   Endpoints snapped to whatever points existed at that step count — measured
   discontinuity 0.06690 → 0.11846 between 50 and 51 generation steps, costing
   10.6 dB in flat plateaus with cliffs. `generate` now reverses `inv.sigmas`.
6. **Prompt mismatch: 3.65 dB.**
7. α-pin sign inverted (edited everything *except* the target).
8. σ endpoint degeneracy: clamping to [0.05, 0.98] took the synthetic round trip
   15.83 → 39.16 dB and removed 16px patch-periodic striping.
9. `selectivity` unbounded (~700,000); bounded to [0,1].
10. SD3 context length 333, not 410.
11. `generate()` started at σ=1.0 regardless of inversion end.

## 23. Claims made and then withdrawn

Kept because each was believed on evidence that did not support it.

| claim | why it failed |
|---|---|
| "Inversion is unstable on real photos, swinging 8–10 dB" | the diagnostic varied generation steps independently of inversion steps, which no real code path does. Artifact of the test. |
| "euler-200 has a 6 dB advantage" | n=1 artifact of one photograph; across 12 it comes second |
| "Perturbing the noise latent will show amplification" | it does not — 2% perturbation costs 0.1 dB |
| "SAM/MagicBrush masks are polarity-inverted" | measured the wrong channel, scored by a degenerate test whose branches summed to 1.0 |
| "box_preview and the edit mask are in different coordinate frames" | stale files from an earlier invocation |
| "corr(struct, sat) ≈ 0.95 is the replication test" | chroma-specific; gave 0.204 on a desaturated object |
| three separate "WINDOW EXISTS" verdicts | all false; refuted by looking at the image in seconds |

**The pattern:** every one came from a confident automated verdict read off
numbers whose generating process had not been checked. The countermeasure that
worked was cheap and repeatable — look at the image, and test whether the
diagnostic could have produced a different answer.

---

# PART VI — WHAT TO DO NEXT

## 24. Do not do

- **Further τ or λ sweeps.** F1 shows the axis is singular across three objects
  and two colour directions. More tuning explores a one-dimensional family that
  does not contain the target.
- **Shape edits, for now.** No shape edit has succeeded at any setting on any
  image, synthetic or real. Attempting them mixes "region control fails" with
  "shape edits fail generally".
- **Renting A100s.** Nothing measured yet justifies it.

## 25. Small open items

1. **Calibrate `struct`, or drop its automated verdict.** Thresholds are
   invented; synthetic pairs gave 0.27 / 0.02 / 0.00 for correct recolour / noise
   fill / flat blob — ordering right, scale unestablished. Three false verdicts
   came from these thresholds. Reporting the trade-off curve with no pass/fail is
   the honest option.
2. **Annotate same-category instances** on a MagicBrush subset to unblock the
   leakage metric — the metric the whole project was originally motivated by, and
   still uncomputed. `523394_t1` ("make one of the shoes black") is the natural
   first case: same-category multi-instance, already in the colour subset.
3. `bench/run.py` — referenced by `SETUP.md`, never written.
4. `max_dist` in `boundary_profile` truncates above 24 px.
5. Weak-edit guard on `selectivity` (NaN below a target-change threshold).
6. Re-check "Recolour What Matters" (arXiv 2603.18466) for prior-work collision.

## 26. Stage 1 — the actual next direction

F1 says the mechanism needs a control that specifies *where to move on the
manifold*, not *how much source to discard*. Formally, replace

    z_edit = w·z_source + (1−w)·z_generated        (mixing ratio; current)

with

    z_edit = z_source + Δz(c, M)                   (attribute direction; target)

where `c` is the edit condition and `M` the region. The velocity field becomes
`dz/dt = v_θ(z, t, c, M)` — the network learns the direction to move, and the
mask constrains *where* it may act rather than *how much* is kept.

**Two implementation routes:**

| route | pros | cons |
|---|---|---|
| Reimplement BiFM | published, CVPR 2026, evaluated on PIE-Bench | **no public code**; ~8000 A100-h to train from scratch |
| Minimal Δz LoRA on SD3-Medium | fits the 3090Ti for small-scale proof; directly tests F1's prediction | not a published baseline; needs its own validation |

**Recommended: the minimal LoRA first**, as a falsification test of F1 rather
than a product. Train on MagicBrush colour edits (30 in the first 200 turn-1
rows; more available across the full dev split) with source/target image pairs
and the provided masks. The prediction F1 makes is sharp and cheap to check:

> A learned Δz conditioned on (mask, instruction) should achieve
> corr(dE, struct) substantially closer to 0 than −0.95. If it does not, the
> coupling is a property of the base model's latent space rather than of the
> α-pin formulation, which is a different and larger finding.

Reuse from Stage 0 without change: the SAM masking path, the MagicBrush loader,
`struct`/`dE`/`dL*`, the contact-sheet discipline, and heun-50.

## 27. Compute budget

- Local 3090Ti: sufficient for Stages 0, 1, 3.
- Free 8×A100 for <6 h, repeatable — chain with checkpointing.
- Rented 8×A100 up to 50 h ≈ 400 A100-hours, against BiFM's ~8000.

Stage 0 cost roughly 6 GPU-hours of sweeps and produced its result locally.
Spend the next budget on §26, not on more sweeps of a mechanism now known to be
underdetermined.

---

## 28. One-paragraph summary for a reader in a hurry

RegionFlow set out to fix instance leakage in region-controlled image editing
using two fields derived from one mask: an attention bias (λ) to bind edit tokens
to one instance, and an inversion-depth pin (τ) to hold everything else in place.
The localisation half works — background change 0.012, seams near zero, SAM IoU
0.85. The attribute-control half does not. Across three objects and two colour
directions, colour movement and structure preservation are anti-correlated at
−0.958, −0.726 and −0.955: τ and λ turn out to be the same knob, controlling only
how much of the source is discarded, never what it becomes. λ=4 preserves the
object and produces no edit; λ=16 produces movement by destroying the object, in
the wrong colour direction. This is a clean negative result and it points
somewhere specific: an editor needs a learned conditional transport that supplies
an attribute *direction*, which a source/noise mixing ratio structurally cannot
express.
