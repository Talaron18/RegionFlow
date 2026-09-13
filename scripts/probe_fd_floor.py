"""
Why do fd and jvp disagree on this model?

CONTEXT. Probe 4 in probe_jvp_mmdit.py found fd never comes within 90% of the
JVP at any epsilon from 1e-6 to 1e-2, with fd_norm following a clean power law
(~eps^-0.55) and no plateau anywhere. TF32 was ruled out: the sweep is bitwise
identical with TF32 on and off.

That leaves two possibilities and they have opposite consequences:

  A. The JVP is correct and fd has no valid window on this function -- the
     noise floor at small eps is still above the curvature error at large eps.
     Gate B proceeds on jvp. fd is dead, which it already was.

  B. The JVP is wrong. Gate B would train against a wrong regression target
     for weeks before anything looked odd.

Do not resolve this by argument. Probe 5 settles the machinery question in
fp64, where fd is trustworthy by construction; probe 6 measures where the noise
in the fp32 forward actually comes from.

    python scripts/probe_fd_floor.py

RUN THIS ONLY IF the fd/jvp gap needs to be in the writeup, or if you want the
JVP independently validated before committing A100 blocks to Gate B. The second
reason is a good one.
"""

from __future__ import annotations

import argparse
import json

import torch


def probe5_fp64_block(device="cpu") -> dict:
    """Machinery check: one MMDiT block, fp64, jvp vs fd.

    In float64 the rounding floor is ~1e-16 relative, so a central difference
    at eps=1e-6 has ~1e-10 error and IS the derivative for practical purposes.
    If jvp and fd agree here, torch.func.jvp under the math backend is sound
    and the fp32 disagreement is about conditioning, not about the JVP.

    CPU by default: fp64 on consumer NVIDIA parts runs at 1/64 rate, and one
    block is small enough that CPU is the faster path.
    """
    from diffusers import SD3Transformer2DModel
    from transport.meanflow import math_attention

    torch.manual_seed(0)
    tr = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="transformer", torch_dtype=torch.float32)
    blk = tr.transformer_blocks[0].to(device).double().eval()
    blk.requires_grad_(False)
    del tr

    D = blk.norm1.linear.in_features
    B, N, L = 1, 64, 16
    hs = torch.randn(B, N, D, dtype=torch.float64, device=device)
    ehs = torch.randn(B, L, D, dtype=torch.float64, device=device)
    temb = torch.randn(B, D, dtype=torch.float64, device=device)
    tangent = torch.randn_like(hs)

    def f(h):
        out = blk(hidden_states=h, encoder_hidden_states=ehs, temb=temb)
        return out[1] if isinstance(out, tuple) else out

    with math_attention():
        _, jvp = torch.func.jvp(f, (hs,), (tangent,))

    rows = []
    with torch.no_grad():
        for eps in (1e-4, 1e-5, 1e-6, 1e-7):
            fd = (f(hs + eps * tangent) - f(hs - eps * tangent)) / (2 * eps)
            rows.append({"eps": eps,
                         "rel_err": float((fd - jvp).norm() / jvp.norm())})

    best = min(r["rel_err"] for r in rows)
    return {"probe": "5_fp64_block", "ok": best < 1e-6,
            "best_rel_err": best, "jvp_norm": float(jvp.norm()),
            "sweep": rows,
            "reading": ("jvp machinery is sound; the fp32 gap is conditioning"
                        if best < 1e-6 else
                        "torch.func.jvp disagrees with fd even in fp64 -- the "
                        "JVP itself is suspect and Gate B must not start")}


def probe6_noise_floor(device="cuda") -> dict:
    """Where does the fp32 forward's noise come from, and which tangent is bad?

    Three measurements on the FULL transformer:

    repeatability   full(z, t) called twice on IDENTICAL inputs. Any nonzero
                    difference is pure kernel nondeterminism and sets the floor
                    that fd divides by 2*eps. Predicted fd noise at a given eps
                    is floor/(2*eps); compare against the observed sweep.

    split tangents  fd and jvp separately along z only and along t only. The
                    total derivative mixes them; if one component agrees and
                    the other does not, that localises the problem to the
                    latent path or to the timestep embedding.

    unit tangent    the model's velocity has norm ~560 against a latent of norm
                    ~256, so eps*v is a large perturbation and curvature error
                    is correspondingly large. Repeating with a unit-norm
                    tangent separates "fd has no window" from "this particular
                    direction has no window".
    """
    from diffusers import SD3Transformer2DModel
    from transport.meanflow import math_attention

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    tr = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="transformer", torch_dtype=torch.float32).to(device).eval()
    tr.requires_grad_(False)

    B, C, H, W = 1, 16, 64, 64
    L = 333
    z = torch.randn(B, C, H, W, device=device)
    pe = torch.randn(B, L, int(tr.config.joint_attention_dim), device=device)
    po = torch.randn(B, int(tr.config.pooled_projection_dim), device=device)
    tn = torch.full((B,), 0.5, device=device)

    def full(z_, tn_):
        return tr(hidden_states=z_, encoder_hidden_states=pe,
                  pooled_projections=po, timestep=tn_ * 1000.0,
                  return_dict=False)[0]

    with torch.no_grad():
        a, b = full(z, tn), full(z, tn)
        floor = float((a - b).norm())   # measured 0.0: the forward is
                                        # bitwise deterministic, so there is
                                        # no noise floor to divide by eps
        out_norm = float(a.norm())
        v = a.clone()
    v_unit = v / v.norm()

    def compare(tangent_z, tangent_t, label):
        """fd and jvp, BOTH under the math backend.

        The first version of this ran the JVP inside math_attention() and the
        finite differences under the default (efficient) SDPA kernel -- i.e. it
        compared two derivative estimators computed on two DIFFERENT
        implementations of attention. Measured, those implementations disagree
        by ~2.7e-5 relative, input-dependently, so the disagreement does not
        cancel in (up - um) and gets divided by 2*eps like anything else.
        Comparing estimators across backends is not a comparison of estimators.

        Also reports the per-element input step against fp32 spacing. When
        eps*|tangent| per element falls below the spacing of z, the
        perturbation is silently rounded away and fd cannot converge at any
        smaller eps -- which is what made the unit-tangent row diverge.
        """
        with math_attention():
            _, j = torch.func.jvp(full, (z, tn), (tangent_z, tangent_t))
        spacing = float(torch.finfo(torch.float32).eps * z.abs().median())
        rows = []
        with math_attention(), torch.no_grad():
            for eps in (1e-5, 1e-4, 1e-3, 1e-2):
                fd = (full(z + eps * tangent_z, tn + eps * tangent_t)
                      - full(z - eps * tangent_z, tn - eps * tangent_t)) / (2 * eps)
                step = eps * float(tangent_z.abs().median())
                rows.append({
                    "eps": eps,
                    "rel_err": float((fd - j).norm() / j.norm().clamp(min=1e-12)),
                    "input_step_per_elem": step,
                    "below_fp32_spacing": bool(step < spacing),
                })
        return {"label": label, "jvp_norm": float(j.norm()),
                "fp32_spacing_of_z": spacing,
                "best_rel_err": min(r["rel_err"] for r in rows), "sweep": rows}

    zero_t, zero_z = torch.zeros_like(tn), torch.zeros_like(z)
    parts = [
        compare(v, torch.ones_like(tn), "both (v, 1) -- what MeanFlow uses"),
        compare(v, zero_t, "z only (v, 0)"),
        compare(zero_z, torch.ones_like(tn), "t only (0, 1)"),
        compare(v_unit, zero_t, "z only, UNIT tangent"),
    ]
    return {
        "probe": "6_noise_floor",
        "repeatability_abs": floor,
        "out_norm": out_norm,
        "repeatability_rel": floor / out_norm,
        "note": ("If repeatability_rel is ~1e-7 the forward is clean and the "
                 "fd failure is curvature. If it is ~1e-4 or worse, the "
                 "forward itself is noisy and NO finite difference can work "
                 "here at any epsilon."),
        "parts": parts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-fp64", action="store_true")
    args = ap.parse_args()

    if not args.skip_fp64:
        r5 = probe5_fp64_block()
        print("\n=== 5. fp64 single block: is the JVP machinery sound? ===")
        print(json.dumps(r5, indent=1))
        if not r5["ok"]:
            print("\nSTOP. The JVP disagrees with fd in float64, where fd is "
                  "trustworthy. Do not start Gate B until this is explained.")
            return 1

    r6 = probe6_noise_floor(args.device)
    print("\n=== 6. fp32 full model: noise floor and split tangents ===")
    print(json.dumps(r6, indent=1))

    print("\n--- reading ---")
    print("  Rows flagged below_fp32_spacing were never valid: the input "
          "perturbation rounded away before the model saw it.")
    if r6["repeatability_rel"] > 1e-5:
        print(f"  The forward is not repeatable to better than "
              f"{r6['repeatability_rel']:.2e} relative. Finite differences "
              "divide by 2*eps, so this alone explains the sweep and no "
              "epsilon can work. jvp is the only option, and its agreement "
              "with fd in fp64 is what licenses it.")
    else:
        print("  The forward is clean, so the fd failure is curvature along "
              "this direction. Check whether the UNIT tangent row converges: "
              "if it does, the velocity tangent is simply too large a step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
