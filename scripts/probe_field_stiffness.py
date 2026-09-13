"""
How straight is SD3-Medium's rectified flow, and what step count does that imply?

WHAT THIS MEASURES. For the frozen pretrained model, along real trajectories:

    S(t) = ||dv/dt|| / ||v||        stiffness -- how fast the velocity field
                                    changes relative to its own magnitude

and its reciprocal, the interval over which a first-order (constant-velocity)
approximation stays comparable to the field itself.

WHY IT MATTERS MORE THAN THE SINGLE NUMBER. A perfectly rectified flow has
v constant along its trajectory, so dv/dt = 0 and S = 0: one step suffices,
exactly. S is therefore a direct measure of how far SD3's flow is from
straight. The Gate B timing run saw S ~ 15 through the adapter; if that holds
on the frozen model, SD3-Medium's flow is markedly curved and the entire
few-step programme is fighting that curvature rather than an optimisation
problem.

The integral turns it into a step count:

    N_min = integral_0^1 S(t) dt

is the number of intervals needed for the constant-velocity approximation to
stay within its own scale across the trajectory. If S ~ 15 uniformly, N ~ 15;
but S is very unlikely to be uniform, and where the curvature concentrates
determines the STEP SCHEDULE, not just the count. A field that is stiff only
near t=1 wants fine steps at the noise end and coarse ones near the data --
which is a concrete, actionable design output, and not something any single
scalar can give.

THREE CONFOUNDS THIS AVOIDS, all present in the Gate B training-log estimate:

  - no adapter, so the field is SD3's and not partly a LoRA's
  - no gradient steps, so nothing drifts while being measured
  - measured ON-TRAJECTORY, by integrating from noise, rather than at
    off-manifold points built from an arbitrary latent

CAVEAT KEPT IN VIEW. S is a LOCAL first-order quantity. It bounds where the
constant-velocity approximation holds; it does not prove a learned interval
estimator cannot do better, since such an estimator is not restricted to first
order. That distinction is exactly the one Gate B still has to settle
experimentally.

    python scripts/probe_field_stiffness.py --prompts prompts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None,
                    help="json list of strings; defaults to a small builtin set")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=50,
                    help="integration steps used to walk the trajectory")
    ap.add_argument("--probes", type=int, default=17,
                    help="how many t values to measure S at")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mode", default="interpolate",
                    choices=["interpolate", "trajectory"],
                    help="WHERE to measure. 'interpolate' builds "
                         "z_t=(1-t)x0+t*eps from cached real latents -- the "
                         "same construction the MeanFlow loss uses, so the "
                         "number is the one training actually sees. "
                         "'trajectory' Euler-integrates the guided field, "
                         "which DRIFTS OFF that path; Jacobians are much "
                         "larger off-manifold and the first run of this probe "
                         "reported S~52 that way against S~15 from the "
                         "training log. Same model, different points.")
    ap.add_argument("--cache", default=None,
                    help="triplet cache, required for --mode interpolate")
    ap.add_argument("--samples", type=int, default=32,
                    help="samples per t. dv/dt is heavy-tailed: at 9 samples "
                         "the median swung 20x between adjacent t while |v| "
                         "moved 1%%, which is sampling noise, not structure.")
    ap.add_argument("--decompose", action="store_true",
                    help="split dv/dt into the SPATIAL part (J.v, tangent "
                         "(v,0)) and the TEMPORAL part (dv/dt at fixed z, "
                         "tangent (0,1)). Only the spatial part is trajectory "
                         "curvature; the temporal part is dominated by the "
                         "sinusoidal timestep embedding, whose fastest "
                         "component has period ~0.0063 in normalised t and "
                         "picks up a ~1000x gain under d/dt.")
    ap.add_argument("--fine-sweep", type=float, default=None,
                    help="centre t for a FINE sweep at 0.0005 spacing over "
                         "+-0.02. If dv/dt oscillates with period ~0.006 "
                         "there, the coarse profile was aliasing the timestep "
                         "embedding and its zigzag is an artifact, not "
                         "curvature. Try --fine-sweep 0.32 (the S=21.9 dip).")
    ap.add_argument("--out", default="runs/field_stiffness.json")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline

    from transport.meanflow import math_attention

    if args.mode == "interpolate" and not args.cache:
        raise SystemExit("--mode interpolate needs --cache (it measures at the "
                         "interpolated points training uses)")

    prompts = (json.loads(Path(args.prompts).read_text()) if args.prompts
               else ["a photograph of a red phone booth on a city street",
                     "a green tennis ball on a carpet",
                     "a sheep standing in a grassy field"])

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # TWO-PHASE LOAD. The text encoders run once in bfloat16 and are then
    # freed; only the transformer is loaded in float32, which the JVP needs.
    # Loading the whole pipeline in fp32 would want ~30GB before a single
    # forward -- T5-XXL alone is ~19GB in fp32 -- and this probe originally
    # did exactly that.
    print("[phase 1] encoding prompts (bf16, encoders freed after)")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, transformer=None, vae=None,
    ).to(args.device)
    embeds = []
    with torch.no_grad():
        for prompt in prompts:
            pe, npe, po, npo = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None,
                device=args.device, do_classifier_free_guidance=True)
            embeds.append(tuple(x.float() for x in (pe, npe, po, npo)))
    del pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("[phase 2] loading transformer in float32 for the JVP")
    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.float32
    ).to(args.device).eval().requires_grad_(False)
    print(f"  allocated {torch.cuda.memory_allocated()/2**30:.1f} GB")

    T = 1000.0
    probes = np.linspace(0.02, 0.98, args.probes)
    rows = []

    for pi, (pe, npe, po, npo) in enumerate(embeds):

        def field(z_, tn_):
            """Guided velocity at normalised time tn_. CFG folded in, because
            that is the field a sampler actually integrates."""
            b = z_.shape[0]
            out = tr(hidden_states=torch.cat([z_, z_]),
                     encoder_hidden_states=torch.cat([npe, pe]),
                     pooled_projections=torch.cat([npo, po]),
                     timestep=(tn_.float() * T).repeat(2),
                     return_dict=False)[0]
            un, c = out.chunk(2)
            return un + args.guidance * (c - un)

        if args.mode == "interpolate":
            # Same construction as the training loss: real data latent, noise,
            # linear interpolation. No integration, so no drift.
            from transport.data import TripletCache
            cache = TripletCache(args.cache)
            idx = np.random.default_rng(0).choice(
                len(cache), size=min(args.samples, len(cache)), replace=False)
            for k, j in enumerate(idx):
                x0 = torch.from_numpy(cache[int(j)]["z_tgt"]).to(
                    args.device)[None].float()
                g = torch.Generator(device=args.device).manual_seed(k)
                eps_ = torch.randn(x0.shape, device=args.device, generator=g)
                for tv in probes:
                    tn = torch.full((1,), float(tv), device=args.device)
                    z_t = (1 - tv) * x0 + tv * eps_
                    with torch.no_grad():
                        v0 = field(z_t, tn)
                    with math_attention():
                        v, dv = torch.func.jvp(
                            field, (z_t, tn), (v0, torch.ones_like(tn)))
                    vn, dn = float(v.norm()), float(dv.norm())
                    extra = {}
                    if args.decompose:
                        with math_attention():
                            _, d_space = torch.func.jvp(
                                field, (z_t, tn), (v0, torch.zeros_like(tn)))
                            _, d_time = torch.func.jvp(
                                field, (z_t, tn),
                                (torch.zeros_like(z_t), torch.ones_like(tn)))
                        extra = {"dvdt_spatial": float(d_space.norm()),
                                 "dvdt_temporal": float(d_time.norm())}
                    rows.append({"prompt": pi, "seed": k, "t": float(tv),
                                 "v_norm": vn, "dvdt_norm": dn, **extra,
                                 "stiffness": dn / max(vn, 1e-8),
                                 "delta_usable": vn / max(dn, 1e-8)})
            print(f"  prompt {pi}: {len(idx)} samples x {len(probes)} t values")
            continue

        for seed in range(args.seeds):
            g = torch.Generator(device=args.device).manual_seed(seed)
            z = torch.randn(1, 16, 64, 64, device=args.device, generator=g)

            # Walk the trajectory from noise to data, measuring S on the way.
            grid = np.linspace(1.0, 0.0, args.steps + 1)
            probe_set = set(np.searchsorted(-grid, -probes))
            for i in range(len(grid) - 1):
                tn = torch.full((1,), float(grid[i]), device=args.device)
                if i in probe_set:
                    # The tangent for z is dz/dt along the trajectory, which
                    # IS the field itself: get it with a plain forward first,
                    # then take the JVP along it. Using a random tangent here
                    # would measure curvature in a direction the sampler never
                    # travels.
                    with torch.no_grad():
                        v0 = field(z, tn)
                    with math_attention():
                        v, dv = torch.func.jvp(
                            field, (z, tn), (v0, torch.ones_like(tn)))
                    vn = float(v.norm())
                    dn = float(dv.norm())
                    rows.append({
                        "prompt": pi, "seed": seed, "t": float(grid[i]),
                        "v_norm": vn, "dvdt_norm": dn,
                        "stiffness": dn / max(vn, 1e-8),
                        "delta_usable": vn / max(dn, 1e-8),
                    })
                with torch.no_grad():
                    v_step = field(z, tn)
                z = z - float(grid[i] - grid[i + 1]) * v_step

            print(f"  prompt {pi} seed {seed}: "
                  f"{sum(1 for r in rows if r['seed'] == seed and r['prompt'] == pi)}"
                  " probes")

    if args.decompose and rows and "dvdt_spatial" in rows[0]:
        sp = np.median([r["dvdt_spatial"] for r in rows])
        tm = np.median([r["dvdt_temporal"] for r in rows])
        vv = np.median([r["v_norm"] for r in rows])
        print(f"\n--- decomposition of dv/dt (medians over all probes) ---")
        print(f"  spatial  ||J.v||        {sp:>12.1f}   S_spatial = {sp/vv:>7.2f}")
        print(f"  temporal ||dv/dt|_z     {tm:>12.1f}   S_temporal= {tm/vv:>7.2f}")
        print(f"  ratio temporal/spatial  {tm/max(sp,1e-8):>12.2f}")
        print("  Only the SPATIAL part is trajectory curvature. If the")
        print("  temporal part dominates, the stiffness number is mostly the")
        print("  timestep embedding's fastest component (period ~0.0063 in t,")
        print("  ~1000x gain under d/dt) -- an oscillation no sampler")
        print("  resolves, and NOT a bound on step count.")

    # ---- aggregate -------------------------------------------------------
    if args.fine_sweep is not None:
        c = args.fine_sweep
        fine = np.arange(c - 0.02, c + 0.02, 0.0005)
        from transport.data import TripletCache
        cache = TripletCache(args.cache)
        x0 = torch.from_numpy(cache[0]["z_tgt"]).to(args.device)[None].float()
        g = torch.Generator(device=args.device).manual_seed(0)
        eps_ = torch.randn(x0.shape, device=args.device, generator=g)
        pe, npe, po, npo = embeds[0]

        def field0(z_, tn_):
            out = tr(hidden_states=torch.cat([z_, z_]),
                     encoder_hidden_states=torch.cat([npe, pe]),
                     pooled_projections=torch.cat([npo, po]),
                     timestep=(tn_.float() * T).repeat(2),
                     return_dict=False)[0]
            un, cc = out.chunk(2)
            return un + args.guidance * (cc - un)

        print(f"\n--- fine sweep around t={c}, spacing 0.0005 ---")
        print(f"{'t':>8}{'|dv/dt|':>12}")
        fine_rows = []
        for tv in fine:
            tn = torch.full((1,), float(tv), device=args.device)
            z_t = (1 - tv) * x0 + tv * eps_
            with torch.no_grad():
                v0 = field0(z_t, tn)
            with math_attention():
                _, dv = torch.func.jvp(
                    field0, (z_t, tn), (v0, torch.ones_like(tn)))
            fine_rows.append({"t": float(tv), "dvdt_norm": float(dv.norm())})
            print(f"{tv:>8.4f}{fine_rows[-1]['dvdt_norm']:>12.1f}")
        y = np.array([r["dvdt_norm"] for r in fine_rows])
        y = y - y.mean()
        f = np.fft.rfftfreq(len(y), d=0.0005)
        peak = f[1:][np.argmax(np.abs(np.fft.rfft(y))[1:])]
        print(f"\n  dominant period in t: {1/peak:.5f}")
        print(f"  timestep embedding fastest period: 0.00628")
        if abs(1 / peak - 0.00628) < 0.003:
            print("  MATCH: the coarse profile was ALIASING the timestep\n"
                  "  embedding. Its zigzag is an artifact and N_min from it\n"
                  "  is not a step bound.")

    ts = sorted({r["t"] for r in rows}, reverse=True)
    print(f"\n{'t':>6}{'|v|':>10}{'|dv/dt|':>11}{'S':>8}"
          f"{'S IQR':>18}{'delta_usable':>14}{'n':>5}")
    prof = []
    for t in ts:
        sel = [r for r in rows if abs(r["t"] - t) < 1e-9]
        vals = np.array([r["stiffness"] for r in sel])
        S = float(np.median(vals))
        q1, q3 = (float(np.percentile(vals, 25)), float(np.percentile(vals, 75)))
        du = float(np.median([r["delta_usable"] for r in sel]))
        prof.append({"t": t, "stiffness": S, "s_q1": q1, "s_q3": q3,
                     "n": len(sel), "delta_usable": du,
                     "v_norm": float(np.median([r["v_norm"] for r in sel])),
                     "dvdt_norm": float(np.median([r["dvdt_norm"] for r in sel]))})
        print(f"{t:>6.2f}{prof[-1]['v_norm']:>10.1f}"
              f"{prof[-1]['dvdt_norm']:>11.1f}{S:>8.2f}"
              f"{f'[{q1:.1f}, {q3:.1f}]':>18}{du:>14.4f}{len(sel):>5}")

    tt = np.array([p["t"] for p in prof])
    SS = np.array([p["stiffness"] for p in prof])
    order = np.argsort(tt)
    n_min = float(np.trapezoid(SS[order], tt[order])) if hasattr(np, "trapezoid") \
        else float(np.trapz(SS[order], tt[order]))

    print("\n" + "=" * 60)
    print(f"median stiffness over t : {np.median(SS):.2f}")
    print(f"min / max S             : {SS.min():.2f} at t={tt[SS.argmin()]:.2f}"
          f"  /  {SS.max():.2f} at t={tt[SS.argmax()]:.2f}")
    print(f"N_min = integral S dt   : {n_min:.1f} steps")
    print("=" * 60)
    print("N_min is the step count at which a CONSTANT-VELOCITY approximation\n"
          "stays within the field's own scale across the trajectory. It is a\n"
          "first-order bound on the pretrained field, NOT a bound on what a\n"
          "learned interval estimator can achieve -- that estimator is not\n"
          "restricted to first order, and settling the difference is Gate B's\n"
          "job.")
    # Only claim a SHAPE if the within-t spread is small next to the
    # across-t variation. The first run flagged "concentrated curvature" on a
    # profile whose adjacent points differed 20x purely from sampling noise.
    iqr = np.array([p["s_q3"] - p["s_q1"] for p in prof])
    within = float(np.median(iqr))
    across = float(SS.max() - SS.min())
    print(f"\nspread: median within-t IQR {within:.1f}, across-t range "
          f"{across:.1f}")
    if within > 0.5 * across:
        print("  The within-t spread is comparable to the across-t variation.\n"
              "  DO NOT read a shape from this profile, and do not quote\n"
              "  N_min: both are dominated by sampling noise. Raise --samples\n"
              "  until the IQRs are narrow relative to the range.")
    elif SS.max() / max(SS.min(), 1e-8) > 3:
        print("  S varies more than 3x across t and the within-t spread is\n"
              "  small, so the curvature is genuinely CONCENTRATED: a uniform\n"
              "  step grid is the wrong schedule. Compare uniform against a\n"
              "  grid with equal integral of S per step in gate_b sweep.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"profile": prof, "n_min": n_min, "guidance": args.guidance,
         "raw": rows}, indent=1))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
