# RegionFlow — status review: why it stalled, and what to do next

Written from the repo as of commit `09ba658` (reports, plan doc, code, and the committed
`synthetic_*.png`). Nothing was trained or scored: this container has no GPU, no `/data`
cache and no `runs/`. Where a claim below is inference rather than measurement, it says so.

## 1. Where the project actually stands

| leg | verdict | source |
|---|---|---|
| Region machinery (λ/α fields, masks, seams) | works; not the problem | START_HERE, Stage 0/1 |
| Gate C benchmark (151 RefEdit×COCO cases, instance selectivity) | **done, standalone contribution** | GATE_C_REPORT |
| Mask conditioning → wrong-instance edits | solved by *existing* models (freeform 40% → oracle-mask 0% mis-targeted) | GATE_C_REPORT §3 |
| Edit quality under an oracle mask | unsolved: ~22% clean by blind audit for an oracle-masked full-step editor ("valid K=10" in GATE_C_REPORT §3 only counts cases where leakage is measurable, not quality) | PLAN §3 |
| Gate A: one-step transport `z_src → z_tgt` (our mechanism) | **negative**: cos≈0.54 raw / 0.77 pooled, blurry output | PLAN verdict + sweeps |
| Gate B: few-step interval field | adapter no better than reusing `v` (STAGE1_REPORT) vs "closed positive with asterisk" (PLAN §1) — **the docs disagree** | |

`START_HERE.md` is stale (calls Gate C "untouched"). The PLAN doc stops before the last two
commits; the LPIPS→DINOv2 work and the synthetic experiments are recorded only in code
comments and one PNG.

## 2. The newest evidence: `synthetic_result.png`

`scripts/clear_edit.py` was built to separate two causes. Panel = source | target | oracle | model.
On a red→blue circle edit with a mask on the red circle only:

- oracle (`z_src + (z_tgt − z_src)·m`): clean blue circle. So the VAE, the mask and the
  compositing are fine.
- model (`z_src + u·m`): a **grey-tan disc with a visible periodic grid texture**, not blue.

The script's own rule: a smeared blob means the problem is spatial precision/architecture,
not ambiguity. That matches the plan's independent closure of Direction A (ambiguity, σ0 noise,
mask-path death, DC redundancy and naive multi-step Euler were all ruled out).
Caveats: one sample, one checkpoint, and an out-of-distribution prompt/image, so "model cannot
generalise a colour word from ~4.5k MagicBrush pairs at rank 16" is not excluded by this
panel alone.

## 3. Why it failed (my read — evidence strength noted)

1. **The mechanism bypasses the generative prior (inference, consistent with §2).** SD3 is a
   noise→data velocity model. Gate A queries it at a *clean* source latent at s=0 and asks a
   rank-16 LoRA to emit a displacement, then decodes `z_src + u·m`. Nothing keeps that latent
   on the VAE's natural-image manifold; regression residue decodes as muddy colour plus a
   latent-grid ripple. Working editors (InstructPix2Pix, UltraEdit) start from noise, so the
   sampler supplies sharpness. The project removed noise by design ("noise never enters"),
   and Direction B, the one option that puts it back, was ruled out as "a retreat, not a
   continuation of the thesis". The thesis, not the evidence, chose that.
2. **Progress was tracked on proxies.** `cos(u, v)` in latent space moved ~30% while
   structure/image quality did not (STAGE1_REPORT §2); the `struct` metric was itself an
   artifact (the model "beat" ground truth). Each negative result then produced a deeper
   diagnostic (six elimination checks, an injection-ratio metric that had to be retired, a
   localisation metric that was inverted) instead of a decision. Rigorous work, but it
   answered "why is cos ~0.5" and not "does this make a good edit".
3. **Three theses in one project.** Few-step distillation, graded region priors, and
   instance-level selectivity. Gate C showed the region/instance part is solved by
   conditioning that already exists; what remains (transformation quality) is the hard,
   general editing problem on a 3090Ti-scale budget. By the plan's own ordering few-step
   belongs last, yet most effort went to it.
4. **Fragility and hygiene.** Commit `9c67baa` ("Adding LPIP as loss") also changed the training
   objective: `interval_conditioning` defaults to **True**, and the plain branch now passes
   `r = 1` where the documented σ0 checkpoints (trained at `2508cee`) used `r = s`. With
   `AdapterConfig.use_interval=True`, `r` reaches the model through the interval embedding, so
   **neither current setting reproduces the documented baseline**, and a DINO run is not
   comparable to those checkpoints. In the interval branch σ0 was silently ignored, which
   broke `pairflow.py --self-test` (passes at `2508cee`, fails from `9c67baa`).
   `mask_to_patch_grid(erode=0)` returned the wrong shape and would have crashed
   `dino_patch_loss`. The repo also tracks `.ipynb_checkpoints/` copies, `__pycache__`, `.patch`
   files and `STAGE1C_REPORT (1).md`, which makes "which file did I run" (habit #5) hard.
   (An earlier version of this commit set the default to False and claimed that restored the
   documented objective. That was wrong; see §5.)

## 4. What to do next (in order)

1. **Score the baseline first.** Gate C wrapper on the σ0=0.0 checkpoint and on `global_local`
   (PLAN work-queue items 1 and 5; commands there). Neither has ever been scored on Gate C, so a
   DINO run has nothing to be compared against until this exists. Also run the held-out
   centred-gap re-run (item 2).
2. **Then the DINO experiment, with a kill criterion written first.** Because the documented
   checkpoints were trained at `r = s`, decide the objective deliberately, e.g.
   `python scripts/gate_a_transport.py train --cache /data/cache/13061d51aee4 --out runs/gate_a_dino
   --steps 8000 --sigma0 0.0 --perceptual-weight 0.1 --perceptual-feature dinov2` (current default,
   interval conditioning on) and retrain a no-DINO control under the *same* flags. Pass = the DINO
   arm gives a recognisably blue circle in `clear_edit.py` *and* beats the control on the
   calibrated VLM judge / blind-audit clean rate on the 151 cases. Fail = stop the
   displacement-regression line; do not add more diagnostics.
3. **Run the control the thesis avoided.** A noise-start, source+mask-conditioned flow-matching
   fine-tune at full steps (Direction B shape) on the same MagicBrush cache, scored on the same
   151 cases. If it beats Gate A, the premise was the bottleneck; few-step distillation can then
   be added last, as the plan says.
4. **Ship Gate C regardless.** Benchmark + the "routing saturates at 0% mis-targeting but quality
   does not move" result needs no editor and is the strongest asset.
5. **Verify one open risk (cheap).** Old checkpoints were trained with `r = s`;
   `apply_transport` queries `r = grid[i+1]` (=1 for one step); current code trains with `r ≠ s`
   under either default. Run `diagnose` on the σ0=0 checkpoint with r=s vs r=1; if they differ,
   every diagnose/Gate C number so far carries that offset.
6. Refresh `START_HERE.md` and append the last two commits' results to the PLAN doc;
   add `.ipynb_checkpoints/` to `.gitignore` and untrack the copies.

## 4b. Text-ablation check (added after the review)

Question: does the instruction set the edit direction, or do the image and mask?
The mask was ablated earlier (raw 0.541 → 0.240); the instruction never was.

```bash
CK=runs/gate_a_sigma_0.0/ckpt_latest.pt
C="--cache /data/cache/13061d51aee4 --bias-npy runs/bias.npy --batch-size 8"
python scripts/gate_a_transport.py diagnose --ckpt $CK $C                        # baseline
python scripts/gate_a_transport.py diagnose --ckpt $CK $C --ablate-text zero     # no instruction
python scripts/gate_a_transport.py diagnose --ckpt $CK $C --ablate-text shuffle  # wrong instruction
python scripts/gate_a_transport.py diagnose --ckpt $CK $C --readout-r 1.0        # r=s vs r=1 risk (§4 item 5)
```

For the interval-trained arms (`runs/gate_a_ctrl`, `runs/gate_a_dino`) add
`--readout-r 1.0` to all three text runs.

How to read `centered` (the instruction-specific treatment), decided in advance:

| drop from baseline under `shuffle` | reading |
|---|---|
| under 10% relative | the model mostly ignores which instruction it got: a text-binding failure |
| 10–40% | partial use of the instruction |
| over 40% | the instruction is load-bearing; the failure is elsewhere |

`zero` gives the same comparison against "no text at all" (an input the model
saw in 10% of training). `shuffle` is the cleaner one because the text stays a
real instruction. Reference point from the data: `centered` for mismatched
instruction pairs is about −0.05, same-instruction about +0.21 (PLAN §7).
`shuffle` skips a trailing batch of one, so n can be one short.

**Result (user's GPU run, `runs/gate_a_ctrl/ckpt_final.pt`, step 8000, `--readout-r 1.0`, n=4511, medians):**

| run | raw | pooled | centered | centered vs baseline |
|---|---|---|---|---|
| baseline | 0.547 | 0.769 | 0.672 | — |
| `--ablate-text zero` | 0.539 | 0.750 | 0.635 | −5.5% |
| `--ablate-text shuffle` | 0.546 | 0.769 | 0.669 | −0.4% |

Both are under the 10% line: by the rule above this is a **text-binding failure**. The
control arm produces almost the same displacement whichever instruction it is given.
Not verified: that batch neighbours in the cache carry different instructions (if many
were identical, `shuffle` would understate the drop; `zero` does not depend on this).

**Same check on the DINO arm (`runs/gate_a_dino/ckpt_final.pt`, step 8000, same flags):**

| run | raw | pooled | centered | centered vs baseline |
|---|---|---|---|---|
| baseline | 0.544 | 0.771 | 0.673 | — |
| `--ablate-text zero` | 0.539 | 0.754 | 0.648 | −3.7% |
| `--ablate-text shuffle` | 0.544 | 0.770 | 0.677 | +0.6% |

Same text-binding failure; the perceptual loss did not change it. This is not the
DINO verdict (cosines are excluded from it in `DINO_EXPERIMENT.md`); K1 decides.

Untested inference: `v = z_tgt − z_src` contains `−z_src` inside the mask, so a model
that only erases the source content there already correlates with `v`. That would
explain both the cosine level and the grey disc in `synthetic_result.png`. Check: score
an instruction-free "erase" predictor (`u = mean in-mask target latent − z_src` inside
the mask, 0 outside) with the same three treatments. If it lands near 0.5 raw, every
cosine in the σ0 sweep was mostly measuring erasure.

**Erase baseline result (user's run, `scripts/erase_baseline.py`, n=4511, medians):**

| predictor | raw | pooled | centered |
|---|---|---|---|
| erase (no instruction, no target) | 0.420 | 0.602 | 0.502 |
| flatten | 0.391 | — | — |
| constant shift | 0.141 | 0.520 | −0.009 |
| trained control arm, for comparison | 0.547 | 0.769 | 0.672 |

Erasing the source content alone reaches about three quarters of the trained
model's score on every treatment (77% / 78% / 75%). The inference above is
confirmed: Gate A's cosines, including the σ0 sweep, mostly measured removal of
the source, and the part attributable to anything else is about +0.13 raw.

## 5. Changes made (two commits on this branch)
- `transport/pairflow.py`: `interval_conditioning` stays **True** (first commit flipped it to
  False; reverted, because False is a third objective, not the baseline). Added a `ValueError`
  when `interval_conditioning` and `sigma0 > 0`, so the silent σ0 drop is loud; the self-test's
  σ0 block now sets `interval_conditioning=False` explicitly and tests the guard.
  `mask_to_patch_grid` flattens for `erode=0`; duplicate `loss` line removed.
- `scripts/gate_a_transport.py`: `--perceptual-mask-erode` and
  `--interval-conditioning/--no-interval-conditioning` plumbed to `PairFlowConfig`. Note
  `sweep_sigma0.sh`-style runs (σ0>0) now need `--no-interval-conditioning`.
- `.claude/` (settings + `explorer`/`researcher`/`worker` agents) and `CLAUDE.md`.
- Verified on CPU: `python transport/pairflow.py --self-test` → OK (note `_fake_pair_model`
  ignores `r`, so this does not test objective equivalence); `mask_to_patch_grid` and
  `dino_patch_loss` (stub extractor) run for `erode` 0 and 1. `tests/check_names.py` still
  reports `bench/test.py: load_target_mask` undefined (pre-existing).
