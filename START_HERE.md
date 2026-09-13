# START HERE

Written to resume this project cold, in a new session, with no memory of how it
got here. Supersedes `START_HERE_stage0.md`.

---

## What this is

RegionFlow: change one named object's attribute while preserving its form,
leaving the rest of the image untouched, and **without affecting other
instances of the same category**. That last clause — instance-level
selectivity — is the research problem. Everything else is infrastructure.

Built on SD3-Medium (`stabilityai/stable-diffusion-3-medium-diffusers`).
Constraint: one 3090Ti (24GB), plus repeatable 6-hour 8×A100 blocks. Cannot win
on compute; the contribution has to be a measurement or a mechanism.

## Reading order

1. **STAGE1_REPORT.md** — current state. Gate A closed, Gate B negative, the
   metric failure, revised priorities, withdrawn claims. Read this first.
2. **RESEARCH_REPORT.md** — Stage 0 in full. Still the best account of the
   design and the region machinery. **Its headline number is now known to be
   substantially a metric artifact** (STAGE1_REPORT §3) — read it with that in
   hand.
3. **STAGE1_PLAN.md** — the plan Stage 1 was run against. Useful for intent
   versus outcome; its Gate B section is superseded.
4. **SETUP.md** — environment.
5. **STAGE0_REPORT.md**, **STAGE0_FINDINGS.md**, **START_HERE_stage0.md** —
   history. The synthetic-phase conclusions about `r` and softness did not
   transfer to real photographs.

## Where things stand

**Works:** localisation. Background change 0.002–0.026, seams near zero, SAM
IoU 0.82–0.86. The region machinery is not the problem and never was.

**Closed, negative:** deterministic regression `z_src → z_tgt`. Returns the
conditional mean of many plausible edits, which renders as a smear. Goes from
"no visible change" straight to blur at every strength. Do not revisit.

**Closed, negative:** rank-32 LoRA conversion of the instantaneous field to a
MeanFlow interval field. Worse than reusing `v` at every Δ from 0.05 to 0.8;
learned an r-**independent** perturbation (`cos(correction, true) ≈ 0`, `D_r`
flat at 1.7% against a required 3–31%). Not closed as a claim about MeanFlow —
closed as a claim about *this adapter, at this rank, through this conditioning
path*.

**Untouched:** Gate C, instance-level selectivity. No benchmark for it exists
anywhere in the field, `bench/instance_bench.py` is written and unrun, and no
metric revision can invalidate that contribution.

## The next four steps

In this order. The first takes minutes and can eliminate two of the others.

```bash
# 1. Is the interval pathway even used? (no GPU)
python -c "
import torch
sd = torch.load('runs/gate_b_p10/ckpt_best.pt', map_location='cpu',
                weights_only=False)['adapter']
for k, v in sd.items():
    if 'interval' in k:
        print(f'{k:<40} |w| = {v.float().norm():.4e}')
"
# near zero -> the conditioning path is the bottleneck: go to (3), skip (4)

# 2. Everything still runs
python tests/check_names.py scripts/*.py transport/*.py bench/*.py
python scripts/test_preflight.py --device cuda
python scripts/test_reference_audit.py

# 3. Widen the interval conditioning  (likely the fix; needs writing)
#    r currently reaches the model through ONE global 2048-dim additive vector
#    on pooled_projections. Give it per-block modulation, or its own tokens.

# 4. The anchor ratio, which never actually ran at 10%
python scripts/gate_b_meanflow.py train --cache /data/cache/<hash> \
    --estimator jvp --dtype float32 --kappa 0 --omega 1 \
    --data-proportion 0.10 --steps 1000 --val-every 200 --out runs/p10_real
```

Evaluate any Gate B checkpoint with `scripts/eval_interval_field.py`, which
scores against a ground-truth average velocity integrated from the frozen
teacher. Read `cos_correction` first: if the correction points nowhere, its
magnitude is irrelevant.

## Environment facts that cost time to learn

- Data lives on the **WSL host**; the container sees it at `/data`. Never pass
  `~` to a container-side script — the tilde expands against the container user.
- `--estimator jvp` **requires `--dtype float32`**. Forward-mode AD promotes
  tangents to fp32 and they cannot meet bf16 weights.
- The JVP must run inside `transport.meanflow.math_attention()`. The fused SDPA
  kernels have no forward-AD rule.
- Gradient checkpointing and forward-mode AD do not compose. `checkpoint_safe`
  (default on) takes `du/dt` from a `no_grad` JVP and `u` from a separate
  forward — mathematically identical, one extra forward.
- MagicBrush's mask is the **alpha channel**, `alpha < 128` = edit region. Not
  the RGB channels.
- Use `turn_index == 1` only. Later turns have a model output as their source.
- Encoders run once into a cache (`scripts/precompute_cache.py`); training never
  loads T5. In fp32 the full pipeline wants ~30GB.

## Non-negotiable habits, all learned by losing time

1. **Score the ground truth on your metric before trusting it.** The
   ground-truth MagicBrush edits scored *worse* than the model on Stage 0's
   structure metric. That check invalidates a headline number and costs two
   minutes.
2. **Parse, don't grep.** `grep -c meanflow adapter.py` returned 4 on a file
   whose mode guard still rejected the mode. `tests/check_modes.py` does it
   properly.
3. **Test the path that actually executes.** The checkpoint/JVP test passed
   while training died inside checkpointing, because it exercised a neighbouring
   function.
4. **Read the reference implementation before building from the paper.** Five
   deviations from the official MeanFlow code mattered; one silently removed
   75% of the training signal.
5. **Check that the file you are running is the file you edited.** Byte-identical
   output from a stale copy has been read as a finding at least five times
   across both stages. Print a hash if in doubt.
6. **Look at the images before recording any number.** Every automated verdict
   that has fired in this project has been wrong at least once.
7. **`ulimit -n 8192`** before anything that touches SAM.

## Things not to do

- Further τ/λ sweeps, or shape edits. Both closed.
- More Gate A eval cases. The diagnosis is coherent, and only 3 of ~400 dev
  candidates passed selection anyway.
- Treating `loss` as a progress signal under the MeanFlow objective. With
  `norm_p = 1` on a sum-reduced loss it is pinned at 1.0 by construction. Read
  `sse_anchor` / `sse_interval`, or better, `eval_interval_field.py`.
- Quoting `S ≈ 15` or `N_min ≈ 94` for SD3's field stiffness. Both were CFG
  amplification plus an aliased timestep embedding. Trajectory curvature is
  **2.84**, implying a step floor near 3.
- Reimplementing BiFM. No public code; ~8000 A100-hours estimated.

## If you only have a day

Build the instance-selectivity benchmark. It is CPU work, it is the one thing
in this project nobody else has, and it is immune to everything that went wrong
in Stages 0 and 1. `bench/instance_bench.py` is written; it needs
`ImgEdit_recap_mask` joined against ImgEdit triplets, filtered to images with
≥2 instances of the target category where only one was edited. 300–500 cases,
stratified, with a 50-case hand-audited subset.
