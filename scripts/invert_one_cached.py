"""Round-trip on a CACHED sample: use the instruction the model was
actually trained on, from the cache itself. This removes the prompt-OOD
confound.

Usage:
    python scripts/invert_one_cached.py \
        --ckpt runs/fd_distill_5k/ckpt_final.pt \
        --cache /data/cache/13061d51aee4 \
        --idx 0 \
        --which z_tgt \
        --interval-kind centered \
        --steps 20 \
        --out runs/inversion_z_tgt

WHAT cos(z_recon, z_ref) CAN AND CANNOT ATTRIBUTE.

The round trip composes two DIFFERENT functions of one model:

    inversion leg   Euler on the instantaneous slice  u(z, s, s)   (r == t)
    generation leg  the sampler's own interval slices u(z, t, r)   (t > r)

For a model whose two slices are mutually consistent -- u(z,t,r) equals the
average of u(z,.,.) along the trajectory from z_t -- the composition is the
identity up to the inversion leg's discretisation. Everything the number can
be measuring, and the control that separates each, is:

  1. the field itself. A per-step direction error of 1 - cos ~ 0.26-0.67 is
     what eval_interval_field.py reports for this checkpoint (STAGE1C 3), and
     a 20-step trajectory integrates 20 of those. Control: --invert-field
     teacher (invert with the FROZEN SD3 field instead of the adapter's) and
     compare. If teacher and instant agree, the adapter's slices are the
     problem; if they agree but both are bad, the harness or the endpoints
     are.
  2. the inversion leg's discretisation / solver / endpoints. bench/inversion
     .py's docstrings are explicit: inversion must recover ONE trajectory, so
     truncation accumulates where generation forgives it, and the sigma ends
     are degenerate (at sigma ~ 0 SD3 cannot know the noise and returns ~
     -x0; SD3's logit-normal barely trains either end). This script used
     neither a clamp nor a solver choice. Controls: --invert-steps,
     --invert-solver heun, --sigma-min 0.05 --sigma-max 0.98.
  3. the harness. --invert-field fixedpoint inverts the SAMPLER'S OWN step
     (z_t = z_r + h*u(z_t,r,t), iterated) instead of integrating a different
     field. On a consistent field that is exact by construction, so it must
     print cos ~ 1.0 at ANY step count. If it does not, the harness is broken
     (grid not retraced, dtype, checkpoint, interval-kind) and no other number
     from this script means anything. It is a HARNESS CHECK, not a quality
     number: it is tautologically 1.0 and must never be quoted as a result.

The reported cos is also an ill-conditioned scale: it is measured in data-
latent space, where the model's noise->data map has a strongly non-normal
Jacobian. A small error in z_noise can read as 0.999 or as 0.8 depending only
on its direction, so cos(z_recon, z_ref) must not be compared against a
per-step field cosine. z_noise-space error (REPORTED below as cos_B, the
composition invert(generate(n))) and the decoded-image PSNR are the readable
quantities.

READ THE OUTPUT IN THIS ORDER.

  1. cos(z_recon, other endpoint). Above cos(z_recon, z_ref) means the round
     trip landed on the pair's other mode. Stop and check the cache's src/tgt
     ordering.
  2. cos_B and the fixedpoint harness check. If the sampler's own inverse does
     not return z_noise, the harness is broken and nothing below counts.
  3. cos(z_noise_teacher, z_noise). The adapter's r == t slice against the
     frozen field it was distilled from. Low = the anchor slice drifted.
  4. cos(z_recon_teacher, z_ref). Teacher inversion + adapter sampler: the
     trajectory-level field-consistency number with the inversion leg removed.
     This is the one to quote next to Gate B's per-step `cos_correction`.
  5. Only then cos(z_recon, z_ref), read against the step/endpoint sweeps.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache


def _ensure(t, nd):
    while t.dim() < nd:
        t = t.unsqueeze(0)
    return t


def descending_grid(steps: int, sigma_min: float, sigma_max: float,
                    kind: str = "uniform", shift: float = 3.0):
    """t = sigma_max -> sigma_min: the GENERATION leg's points, clamped and
    remapped the way bench/inversion.py's `fixed_endpoints` does it.

    The inversion leg reverses THIS list rather than rebuilding one, so the
    two legs share every endpoint by construction -- "a round trip that does
    not retrace is not a round trip" (STAGE1_PLAN, inherited rules). STAGE0
    found that rebuilding the grid instead cost 10.6 dB on one test image.

    kind="shifted" is transport/sampling.py's resolution-shifted schedule,
    which spends more steps at high t. At few steps the schedule matters more
    than the solver did (STAGE0), and it is also the control for "does the
    collapse come from dwelling at sigma ~ 0": shifted puts FEWER points
    there, uniform puts the same spacing everywhere.
    """
    if kind == "uniform":
        # float32, like the original `torch.linspace(1.0, 0.0, steps + 1)`.
        # The two legs now share these exact points (inversion reverses this
        # list); the old code generated an ascending and a descending
        # linspace independently, which differ by ~5e-8 in t -- negligible,
        # but it is not a retrace, and the rule is the rule.
        xs = torch.linspace(1.0, 0.0, steps + 1).tolist()
    elif kind == "shifted":
        from transport.sampling import shifted_grid
        xs = shifted_grid(steps, shift=shift)
    else:
        raise ValueError(f"grid kind must be uniform|shifted, got {kind!r}")
    return [sigma_min + x * (sigma_max - sigma_min) for x in xs]


def _u(adapter, z, t, r, pe, pooled, field, corr_scale=1.0):
    """One velocity query. (B,) times, (B,C,h,w) state -> (B,C,h,w) fp32.

    field="teacher" deliberately ignores r: it is the FROZEN pretrained
    instantaneous field, i.e. the SD3 round trip Stage 0 measured at 39.16 dB
    against a 39.19 dB VAE ceiling. Using it on the inversion leg asks "is the
    adapter's field the reason this fails", with the harness held fixed.

    corr_scale != 1.0 rescales the interval correction about the model's own
    r == t slice: u_eff = u(z,t,t) + a*(u(z,t,r) - u(z,t,t)). At r == t that
    is the identity, so the scaling is skipped there rather than paying a
    second forward for a no-op.
    """
    if field == "teacher":
        return adapter.base_velocity(z, t, pe, pooled)
    u = adapter(z, t, r, pe, pooled)
    if corr_scale != 1.0 and bool((t - r).abs().max() > 0):
        u0 = adapter(z, t, t, pe, pooled)
        u = u0 + corr_scale * (u - u0)
    return u


@torch.no_grad()
def invert(adapter, z_data, pe, pooled, steps, device, field="instant",
           solver="euler", iters=3, sigma_min=0.0, sigma_max=1.0,
           grid_kind="uniform", shift=3.0, corr_scale=1.0):
    """data (t=sigma_min) -> noise (t=sigma_max).

    instant/teacher: integrate the instantaneous field, ascending.
        euler : z + h*v(z, s)                      1 call,  O(h)
        heun  : z + h/2 * (v(z,s) + v(z+h*v, s+h)) 2 calls, O(h^2)

    fixedpoint: solve the SAMPLER'S OWN step for z_t,
        z_t = z_r + (t-r) * u(z_t, t, r)
    by fixed-point iteration from the euler guess. This is the inverse of the
    map the generating leg applies, so the round trip is exact by construction
    (up to convergence) whatever the field is. Convergence needs the
    iteration's contraction factor ||(t-r) du/dz|| < 1; a step size where it
    diverges is a step size whose forward map is expansive, and the warning
    below is the signal for that -- it is printed rather than raised, because
    the point of the mode is to say whether the harness is sound, not to
    pretend the check ran. The residual after `iters` iterations is ~rho**iters,
    so raise --invert-iters if the warning is quiet but cos_B is not ~1.
    """
    z = z_data.clone()
    b = z.shape[0]
    ts = list(reversed(descending_grid(steps, sigma_min, sigma_max,
                                       grid_kind, shift)))   # ascending
    for i in range(len(ts) - 1):
        a, r_ = float(ts[i]), float(ts[i + 1])
        h = r_ - a
        ta = torch.full((b,), a, device=z.device)
        tr = torch.full((b,), r_, device=z.device)

        if field == "fixedpoint":
            tgt = z.clone()
            x = z + h * _u(adapter, z, tr, ta, pe, pooled, field, corr_scale)
            first = float((x - tgt).norm())
            for _ in range(iters):
                x = tgt + h * _u(adapter, x, tr, ta, pe, pooled, field,
                                 corr_scale)
            last = float((x - tgt).norm())
            # Convergence needs ||I - h du/dz|| < 1, i.e. the step map must be
            # INVERTIBLE BY ITERATION. A factor of 4 growth is divergence, not
            # a converged fixed point's last rounding wobble -- measure that
            # rather than a bare first-vs-last comparison, which fires on
            # every healthy step.
            if first > 0 and last > 4.0 * first:
                print(f"  [fixedpoint] WARNING: iteration is EXPANDING at "
                      f"t={r_:.3f} (update {first:.4g} -> {last:.4g}); this "
                      f"step is not invertible by iteration at this step size")
            z = x
            continue

        if solver == "heun":
            v1 = _u(adapter, z, ta, ta, pe, pooled, field)
            pred = z + h * v1
            v2 = _u(adapter, pred, tr, tr, pe, pooled, field)
            z = z + h * 0.5 * (v1 + v2)
        else:
            z = z + h * _u(adapter, z, ta, ta, pe, pooled, field)
    return z


@torch.no_grad()
def generate(adapter, z_noise, pe, pooled, steps, device,
             sigma_min=0.0, sigma_max=1.0, grid_kind="uniform", shift=3.0,
             corr_scale=1.0):
    """noise (t=sigma_max) -> data (t=sigma_min), retracing invert's grid.

    Matches transport/sampling.py exactly: u is queried at the TOP endpoint
    and the update is z - (t-r) u.

    corr_scale != 1.0 rescales the INTERVAL correction about the model's own
    r == t slice:

        u_eff = u(z,t,t) + a * (u(z,t,r) - u(z,t,t))

    which is eval_interval_field.py's `--corr-ref model_r0` construction and
    the a(t) rule STAGE1C 3 claims as the method (a = 1 below t_c 0.35, 0.5
    above). It costs one extra forward per step. The two ends of the
    bisection are informative on their own:

        a = 1.0   the raw model (the default -- no extra forward)
        a = 0.5   the STAGE1C correction scale, applied to the sampler
        a = 0.0   the sampler degenerates to Euler on the r == t slice, i.e.
                  the best-trained field of this adapter (the anchors), with
                  the interval slices removed entirely.

    If a coarse step count collapses and a = 0 rescues it, the interval slice
    at Delta = 1/K is what the coarse step cannot survive. If a = 0 collapses
    too, the coarse-step integration itself is the problem and no amount of
    interval-field calibration will fix it.
    """
    z = z_noise.clone()
    b = z.shape[0]
    ts = descending_grid(steps, sigma_min, sigma_max, grid_kind, shift)
    for i in range(len(ts) - 1):
        t_val, r_val = float(ts[i]), float(ts[i + 1])
        t = torch.full((b,), t_val, device=z.device)
        r = torch.full((b,), r_val, device=z.device)
        z = z - (t_val - r_val) * _u(adapter, z, t, r, pe, pooled, "instant",
                                     corr_scale)
    return z


def _n(x):
    return float(x.pow(2).mean().sqrt())


def _cos(a, b):
    na, nb = a.norm(), b.norm()
    return float((a * b).sum() / (na * nb + 1e-8))


def _rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def _psnr(a, b):
    a = np.asarray(a, np.float32) / 255.0
    b = np.asarray(b, np.float32) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def _collapse(img, z):
    """Does the decoded latent still look like a picture, as a number.

    Habit 6 is "look at the images before recording any number". This is the
    complement, because the eye is not always in the loop and the cosine is
    blind to it: a 200-step run of this probe returned a recon whose decoded
    image was constant along x (within-row variation 2.7 grey levels against a
    40-level row profile) and saturated over [31, 255]. That is not a worse
    image, it is a latent that collapsed onto a 1-D, off-manifold pattern, and
    cos(z_recon, z_ref) cannot tell the difference between that and a blurry
    near-miss.
    """
    g = np.asarray(img, np.float32).mean(2)
    amp_rows = float(g.mean(1).std())          # variation down the image
    amp_cols = float(g.mean(0).std())          # variation across the image
    within_row = float(g.std(1).mean())
    sat = float(((g <= 2) | (g >= 253)).mean())
    f = z.float()
    return dict(
        x_amp=amp_cols, y_amp=amp_rows, within_row=within_row, saturated=sat,
        z_absmax=float(f.abs().max()),
        z_row_std=float(f.mean(dim=(0, 1, 3)).std()),
        z_col_std=float(f.mean(dim=(0, 1, 2)).std()),
        axis_ratio=amp_rows / max(amp_cols, 1e-6),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--which", default="z_src",
                    choices=["z_src", "z_tgt"],
                    help="z_tgt is IN-distribution for Gate B; z_src is not")
    ap.add_argument("--interval-kind", default="centered",
                    choices=("concat", "centered"))
    # --- the inversion leg, separable from the generation leg ---------------
    ap.add_argument("--invert-steps", type=int, default=0,
                    help="0 = same as --steps. Raise it to test whether the "
                         "round trip is solver-limited.")
    ap.add_argument("--invert-field", default="instant",
                    choices=("instant", "teacher", "fixedpoint"),
                    help="instant: the adapter's own r=t slice (the original "
                         "behaviour). teacher: the FROZEN pretrained field, "
                         "holding the harness fixed. fixedpoint: invert the "
                         "sampler's own step -- exact by construction, a "
                         "HARNESS CHECK, not a quality number.")
    ap.add_argument("--invert-solver", default="euler",
                    choices=("euler", "heun"),
                    help="heun is O(h^2); bench/inversion.py's docstrings "
                         "argue inversion needs it far more than generation.")
    ap.add_argument("--invert-iters", type=int, default=3,
                    help="fixed-point iterations per step (--invert-field "
                         "fixedpoint only)")
    ap.add_argument("--sigma-min", type=float, default=0.0,
                    help="skip the degenerate low-t end (bench/inversion.py "
                         "uses 0.05: at sigma~0 the model cannot know the "
                         "noise and returns ~ -x0, so euler shrinks instead "
                         "of noising)")
    ap.add_argument("--sigma-max", type=float, default=1.0,
                    help="skip the under-trained high-t end (bench/inversion"
                         ".py uses 0.98); generation starts here too, so the "
                         "path is retraced")
    ap.add_argument("--grid", default="uniform",
                    choices=("uniform", "shifted"),
                    help="generation/inversion schedule. 'shifted' is "
                         "transport/sampling.py's resolution-shifted grid "
                         "(more steps at high t, fewer at sigma~0) and is the "
                         "control for a collapse caused by dwelling at the "
                         "degenerate data end.")
    ap.add_argument("--shift", type=float, default=3.0,
                    help="shift for --grid shifted")
    ap.add_argument("--corr-scale", type=float, default=1.0,
                    help="rescale the sampler's interval correction about the"
                         " model's own r == t slice (1.0 = the raw model). "
                         "0.5 = the STAGE1C 3 rule; 0.0 = drop the interval "
                         "slices entirely and Euler the best-trained field. "
                         "Costs one extra forward per generation step.")
    ap.add_argument("--invert-corr-scale", type=float, default=1.0,
                    help="same, for the inversion leg (only meaningful with "
                         "--invert-field fixedpoint; the instant/teacher legs "
                         "query r == t, where the correction is zero)")
    ap.add_argument("--dump", default=None,
                    help="optional .npz of z_ref/z_noise/z_recon/z_noise_b")
    ap.add_argument("--no-teacher-ref", action="store_true",
                    help="skip the frozen-SD3 reference inversion. It costs "
                         "one more inversion leg and is the control that "
                         "separates a drifted r=t slice from a bad interval "
                         "field: if the adapter's inversion lands on the same "
                         "noise the frozen field's does, the anchor slice is "
                         "fine and the sampler's interval slices are what the "
                         "round trip is measuring.")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
    from bench.inversion import SD3EditPipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)

    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer",
        torch_dtype=torch.float32).to(args.device)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="meanflow", rank=args.rank,
                          interval_kind=args.interval_kind)
    ).to(args.device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # WHICH INTERVAL EMBEDDING IS THIS? The checkpoint knows, and it is not
    # always what --interval-kind says: Gate A's pair checkpoint is `concat`
    # (interval.net.*), Gate B's distill checkpoint is `centered`
    # (interval.t_branch/delta_branch). Load one into the other and every
    # interval key is "unexpected"; tolerate that with strict=False alone and
    # the interval path stays zero-initialised, i.e. the r-dependence silently
    # vanishes. Same policy as eval_interval_field.py: read it off the keys.
    from transport.conditioning import (interval_kind_from_keys,
                                        make_interval_embedding)
    kind = interval_kind_from_keys(sd["adapter"].keys())
    if kind is None:
        raise SystemExit(f"{args.ckpt}: no interval weights in the checkpoint; "
                         f"a meanflow adapter without them is not the model "
                         f"that was trained")
    if kind != adapter.cfg.interval_kind:
        print(f"interval embedding rebuilt as {kind!r} (checkpoint keys; "
              f"--interval-kind said {adapter.cfg.interval_kind!r})")
        adapter.interval = make_interval_embedding(
            kind, int(tr.config.pooled_projection_dim)).to(args.device)
        adapter.cfg.interval_kind = kind
    missing, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")
    lost = [k for k in missing if k.startswith("interval.")]
    if lost:
        raise SystemExit(f"interval weights absent from checkpoint: {lost[:4]}")
    adapter.eval()

    cache = TripletCache(args.cache)
    rec = cache[args.idx]
    meta = rec["meta"]
    inv_steps = args.invert_steps or args.steps
    print(f"idx {args.idx}: {meta['id']}  "
          f"instruction={meta['instruction']!r}")
    # Echo the whole integration config, not just the flags someone remembers
    # to report: STAGE1C §8.6 lost seven result files to an absent key that
    # showed they predated the patch.
    keys = ('which', 'interval_kind', 'steps', 'invert_steps', 'invert_field',
            'invert_solver', 'invert_iters', 'sigma_min', 'sigma_max', 'grid',
            'shift', 'corr_scale', 'no_teacher_ref')
    cfg = {k: getattr(args, k) for k in keys}
    print(f"config: step={sd.get('step', '?')}  "
          f"integration={json.dumps(cfg, sort_keys=True)}")

    z_src = _ensure(torch.from_numpy(rec["z_src"]).float(), 4).to(args.device)
    z_tgt = _ensure(torch.from_numpy(rec["z_tgt"]).float(), 4).to(args.device)
    pe = _ensure(torch.from_numpy(rec["prompt_embeds"]), 3).to(args.device)
    pooled = _ensure(torch.from_numpy(rec["pooled"]), 2).to(args.device)

    z_ref = z_tgt if args.which == "z_tgt" else z_src
    print(f"round-trip target: {args.which}")

    z_noise = invert(adapter, z_ref, pe, pooled, inv_steps, args.device,
                     field=args.invert_field, solver=args.invert_solver,
                     iters=args.invert_iters, sigma_min=args.sigma_min,
                     sigma_max=args.sigma_max, grid_kind=args.grid,
                     shift=args.shift, corr_scale=args.invert_corr_scale)
    z_recon = generate(adapter, z_noise, pe, pooled, args.steps, args.device,
                       sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                       grid_kind=args.grid, shift=args.shift,
                       corr_scale=args.corr_scale)
    # COMPOSITION B: invert the recon. If the inversion leg is a genuine
    # inverse of the generation leg this must return z_noise. It is the
    # noise-space statement of the same property, in the better-conditioned
    # space (see the module docstring).
    z_noise_b = invert(adapter, z_recon, pe, pooled, inv_steps, args.device,
                       field=args.invert_field, solver=args.invert_solver,
                       iters=args.invert_iters, sigma_min=args.sigma_min,
                       sigma_max=args.sigma_max, grid_kind=args.grid,
                       shift=args.shift, corr_scale=args.invert_corr_scale)

    img_ref = helper.decode_latent(z_ref)
    img_recon = helper.decode_latent(z_recon)
    img_noise = helper.decode_latent(z_noise)

    nfe_inv = inv_steps * (2 if args.invert_solver == "heun" else 1)
    print(f"\n||z_ref||    = {_n(z_ref):.4f}")
    print(f"||z_noise||  = {_n(z_noise):.4f}")
    print(f"||z_recon||  = {_n(z_recon):.4f}")
    print(f"\ncos(z_recon, z_ref) = {_cos(z_recon, z_ref):+.4f}"
          f"   (data-latent space: ill-conditioned, see docstring)")
    print(f"rel ||z_recon - z_ref|| / ||z_ref|| = {_rel(z_recon, z_ref):.4f}")
    print(f"psnr(recon, ref)    = {_psnr(img_recon, img_ref):.2f} dB"
          f"   (vae-only ceiling on this pair is not measured here)")
    print(f"cos_B invert(recon) vs z_noise = {_cos(z_noise_b, z_noise):+.4f}"
          f"   rel={_rel(z_noise_b, z_noise):.4f}")
    # The OTHER endpoint of the pair is already in hand. A round trip that
    # lands closer to it than to its own target is not a failed inversion, it
    # is a landing on the pair's other mode -- and then the cache's src/tgt
    # ordering is the first thing to check, not the model.
    z_other = z_src if args.which == "z_tgt" else z_tgt
    print(f"cos(z_recon, other endpoint) = {_cos(z_recon, z_other):+.4f}")

    # Latent health. A collapsed recon is not a "worse image" and the cosine
    # does not flag it; report the shape of it instead.
    href, hrec = _collapse(img_ref, z_ref), _collapse(img_recon, z_recon)
    print(f"\nshape of the recon (x_amp = variation ACROSS the image, "
          f"y_amp = down it):")
    print(f"  ref   x_amp={href['x_amp']:7.2f} y_amp={href['y_amp']:7.2f} "
          f"within-row={href['within_row']:6.2f} sat={href['saturated']:.3f} "
          f"|z|max={href['z_absmax']:7.2f}")
    print(f"  recon x_amp={hrec['x_amp']:7.2f} y_amp={hrec['y_amp']:7.2f} "
          f"within-row={hrec['within_row']:6.2f} sat={hrec['saturated']:.3f} "
          f"|z|max={hrec['z_absmax']:7.2f}   "
          f"axis_ratio={hrec['axis_ratio']:.2f}")
    if hrec["axis_ratio"] > 3 or hrec["z_absmax"] > 4 * max(href["z_absmax"], 1e-6):
        print("  WARNING: the recon has collapsed onto one spatial axis, or "
              "left the latent range the reference occupies. It is not a "
              "degraded image; do not read cos as 'close but imperfect'.")

    # --- the frozen-field reference leg ------------------------------------
    # Same harness, same grid, same solver; only the field being integrated
    # changes. Stage 0's SD3 round trip is 39.16 dB against a 39.19 dB VAE
    # ceiling, so this leg has a known-good answer. Two numbers come out:
    #   cos(z_noise_teacher, z_noise) -- did the adapter's r == t slice drift
    #       off the pretrained field it was distilled from?
    #   cos(z_recon_teacher, z_ref)   -- teacher inversion + adapter sampler,
    #       i.e. the trajectory-level "is the adapter's interval field the
    #       teacher's flow map" question, with the inversion leg taken out.
    if not args.no_teacher_ref and args.invert_field != "teacher":
        z_noise_t = invert(adapter, z_ref, pe, pooled, inv_steps, args.device,
                           field="teacher", solver=args.invert_solver,
                           sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                           grid_kind=args.grid, shift=args.shift)
        z_recon_t = generate(adapter, z_noise_t, pe, pooled, args.steps,
                             args.device, sigma_min=args.sigma_min,
                             sigma_max=args.sigma_max, grid_kind=args.grid,
                             shift=args.shift, corr_scale=args.corr_scale)
        img_recon_t = helper.decode_latent(z_recon_t)
        print(f"\n[frozen-SD3 reference inversion, same grid/solver]")
        print(f"cos(z_noise_teacher, z_noise)   = "
              f"{_cos(z_noise_t, z_noise):+.4f}   "
              f"rel={_rel(z_noise_t, z_noise):.4f}")
        print(f"cos(z_recon_teacher, z_ref)     = "
              f"{_cos(z_recon_t, z_ref):+.4f}")
        print(f"psnr(recon_teacher, ref)        = "
              f"{_psnr(img_recon_t, img_ref):.2f} dB")
        print("read: noise cos ~ 1 -> the r==t slice did not drift, and "
              "z_recon_teacher is the interval field's error with the "
              "inversion leg removed. noise cos low -> the anchor slice "
              "(the --first-term frozen distill target) is the drift.")
    elif args.invert_field == "teacher":
        print("\n[--invert-field teacher: the frozen-field reference IS the "
              "inversion leg; no second leg run]")

    print(f"nfe: inversion {nfe_inv} * 1 + generation {args.steps} * 1")
    if args.sigma_min > 0 or args.sigma_max < 1:
        print(f"NOTE: endpoints clamped to [{args.sigma_min}, {args.sigma_max}]"
              f" on BOTH legs and retraced. The latent comparison above is "
              f"then offset from the data endpoint by sigma_min; the PSNR is "
              f"the offset-tolerant reading.")
    if args.invert_field == "fixedpoint":
        print("HARNESS CHECK: --invert-field fixedpoint inverts the sampler's"
              " own step map. cos_B ~ 1.0 and cos(z_recon, z_ref) ~ 1.0 are"
              " tautological here; they validate the harness, nothing else.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_arr = np.asarray(Image.fromarray(img_ref).resize((512, 512)))
    rec_arr = np.asarray(Image.fromarray(img_recon).resize((512, 512)))
    nz_arr = np.asarray(Image.fromarray(img_noise).resize((512, 512)))
    panel = np.concatenate([ref_arr, rec_arr, nz_arr], axis=1)
    Image.fromarray(panel).save(out / "roundtrip.png")
    print(f"saved {out / 'roundtrip.png'}  ({args.which} | recon | noise)")

    if args.dump:
        np.savez_compressed(
            args.dump,
            z_ref=z_ref.float().cpu().numpy(),
            z_noise=z_noise.float().cpu().numpy(),
            z_recon=z_recon.float().cpu().numpy(),
            z_noise_b=z_noise_b.float().cpu().numpy(),
            steps=np.array([args.steps]),
            invert_steps=np.array([inv_steps]),
            sigma_min=np.array([args.sigma_min]),
            sigma_max=np.array([args.sigma_max]),
        )
        print(f"saved {args.dump}")


if __name__ == "__main__":
    main()