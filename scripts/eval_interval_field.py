"""
Evaluate the learned interval field against a GROUND-TRUTH average velocity.

WHY THIS EXISTS. Training reports `sse_interval` and validation reports
`val interval(d=...)`, and both measure

    || u_theta - (v - Delta * du_theta/dt) ||

which is a SELF-CONSISTENCY RESIDUAL: the target contains du_theta/dt, so it
moves with the model. That residual can grow purely because ||du_theta/dt||
grew, with u_theta no worse as an average velocity. It has no fixed reference
and cannot separate "the model got worse" from "its temporal derivative got
bigger". The Gate B run's validation swung 39x on that metric while
oscillating rather than trending -- 0.81, 0.30, 8.58, 11.89, 4.93, 8.38 (M) --
which is not something a self-referential quantity can adjudicate.

The average velocity has a definition that does not involve the model:

    u_true(z_t, r, t) = 1/(t - r) * integral_r^t v_teacher(z_s, s) ds

computed by integrating the FROZEN pretrained field along the trajectory that
starts at z_t. This module measures || u_theta - u_true || against it. Model
independent, directly interpretable, and it is the quantity MeanFlow is
defined to learn.

WHAT IT PRODUCES. Error as a function of Delta, per checkpoint:

    ckpt        d=0.05  d=0.10  d=0.20  d=0.30  d=0.50  d=0.80
    step750      ...     ...     ...     ...     ...     ...
    step1050     ...     ...     ...     ...     ...     ...

with two baselines on every cell, because an error number alone says nothing:

    zero      what u = 0 scores. Above this the model is worse than useless.
    base      what the frozen instantaneous field v(z_t, t) scores when used
              as if it were the average velocity. This is the number to beat:
              it is what you get by doing NOTHING, and any interval field that
              cannot beat it has not learned an interval field.

`base` is the honest bar for Gate B. At small Delta the instantaneous field IS
approximately the average velocity, so beating it is easy and means little;
the interesting question is the largest Delta at which the learned field still
beats it.

    python scripts/eval_interval_field.py \\
        --cache /data/cache/<hash> --ckpt runs/gate_b_ref/ckpt_step750.pt \\
                                          runs/gate_b_ref/ckpt_latest.pt \\
        --out runs/interval_field.json

NOTE ON COST. Each (checkpoint, Delta, sample) costs `--substeps` teacher
forwards plus one student forward. At the defaults that is ~16 forwards per
cell; with 6 Deltas and 32 samples that is ~3000 teacher forwards per
checkpoint. Minutes, not hours, and no training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter  # noqa: E402
from transport.conditioning import (  # noqa: E402
    interval_kind_from_keys, make_interval_embedding,
)
from transport.data import TripletCache  # noqa: E402

DELTAS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)


@torch.no_grad()
def true_average_velocity(base_fn, z_t, t, r, substeps: int):
    """u_true = 1/(t-r) * integral_r^t v(z_s, s) ds, along the trajectory.

    Integrated with the midpoint rule from t DOWN to r, advancing the state
    with the field itself:  dz/ds = v(z_s, s).

    Midpoint rather than Euler because the whole question here is how much the
    field changes over the interval; a first-order quadrature would bake the
    thing being measured into the measurement. With substeps=16 over
    Delta<=0.8 the residual quadrature error is far below the differences
    being reported -- check by re-running with --substeps 32 and confirming
    the numbers move by less than a percent.
    """
    b = z_t.shape[0]
    delta = (t - r)
    h = delta / substeps
    z = z_t.clone()
    acc = torch.zeros_like(z)

    for i in range(substeps):
        s_hi = t - i * h
        s_mid = s_hi - 0.5 * h
        # midpoint state: half an Euler step in from s_hi
        v_hi = base_fn(z, s_hi)
        z_mid = z - (0.5 * h).reshape(-1, 1, 1, 1) * v_hi
        v_mid = base_fn(z_mid, s_mid)
        acc = acc + h.reshape(-1, 1, 1, 1) * v_mid
        z = z - h.reshape(-1, 1, 1, 1) * v_mid

    return acc / delta.clamp(min=1e-8).reshape(-1, 1, 1, 1)


def sse(a, b):
    return (a - b).pow(2).flatten(1).sum(1)


def correction_metrics(c_model: torch.Tensor, c_true: torch.Tensor) -> dict:
    """Per-sample scores of the learned r-dependence, averaged.

    c_model = u_hat(r) - u_hat(r = t), c_true = u_true - v.

      corr_err       ||c_model - c_true||^2 / ||c_true||^2. 1.0 is what
                     predicting NO correction scores; below 1 the learned
                     r-dependence helps. Exact, per sample -- not rebuilt
                     from the D_r ratio and mean cos.
      corr_err_best  1 - cos^2: the best any rescaling of c_model could
                     reach with its current direction. When corr_err sits
                     near this, magnitude is already right for the direction
                     and only better direction can lower it.
      corr_ratio     ||c_model|| / ||c_true||, per-sample mean.
    """
    cm, ct = c_model.flatten(1).double(), c_true.flatten(1).double()
    nt2 = ct.pow(2).sum(1).clamp(min=1e-24)
    cos = torch.nn.functional.cosine_similarity(cm, ct, dim=1)
    return {
        "corr_err": float(((cm - ct).pow(2).sum(1) / nt2).mean()),
        "corr_err_best": float((1 - cos ** 2).mean()),
        "corr_ratio": float((cm.norm(dim=1) / nt2.sqrt()).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--deltas", type=float, nargs="+", default=list(DELTAS))
    ap.add_argument("--samples", type=int, default=32,
                    help="validation used 2-4, which is why a 39x swing could "
                         "not be interpreted. 32 costs one extra minute.")
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--t-center", type=float, default=0.5,
                    help="interval is centred here: t = c + d/2, r = c - d/2")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--out", default="runs/interval_field.json")
    args = ap.parse_args()

    from diffusers import SD3Transformer2DModel

    cache = TripletCache(args.cache)
    n = min(args.samples, len(cache))
    idx = np.random.default_rng(0).choice(len(cache), size=n, replace=False)
    x0 = torch.stack([torch.from_numpy(cache[int(j)]["z_tgt"]) for j in idx])
    pe = torch.stack([torch.from_numpy(cache[int(j)]["prompt_embeds"])
                      for j in idx])
    po = torch.stack([torch.from_numpy(cache[int(j)]["pooled"]) for j in idx])
    print(f"[eval] {n} samples from {args.cache}")

    dt = getattr(torch, args.dtype)
    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dt
    ).to(args.device).eval().requires_grad_(False)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="meanflow", rank=args.rank)).to(args.device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()
    adapter.eval()

    x0 = x0.to(args.device).float()
    pe = pe.to(args.device, adapter.model_dtype)
    po = po.to(args.device, adapter.model_dtype)
    g = torch.Generator(device=args.device).manual_seed(1234)
    eps = torch.randn(x0.shape, device=args.device, generator=g)

    results = {}
    for ckpt_path in args.ckpt:
        # run/ckpt, not just ckpt: two runs' ckpt_final would otherwise share
        # a key and the second would silently overwrite the first.
        name = f"{Path(ckpt_path).parent.name}/{Path(ckpt_path).stem}"
        if name in results:
            raise RuntimeError(f"{name} given twice")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # The embedding's structure is read from the checkpoint, not a flag,
        # so concat and centered runs can be scored in one invocation. Only
        # the interval module is swapped: rebuilding TransportAdapter on the
        # same transformer would inject LoRA a second time.
        kind = interval_kind_from_keys(sd["adapter"].keys())
        if kind is not None and kind != adapter.cfg.interval_kind:
            adapter.interval = make_interval_embedding(
                kind, int(tr.config.pooled_projection_dim)).to(args.device)
            adapter.cfg.interval_kind = kind
        missing, unexpected = adapter.load_state_dict(sd["adapter"],
                                                      strict=False)
        if unexpected:
            raise RuntimeError(f"{name}: unexpected keys {unexpected[:5]}")
        # strict=False has to tolerate the frozen base weights, which are
        # never saved -- but it would equally tolerate absent interval
        # weights, and a zero-initialised interval path scores as "no r
        # dependence learned" without any error. Refuse that case.
        lost = [k for k in missing if k.startswith("interval.")]
        if lost:
            raise RuntimeError(f"{name}: interval weights absent from "
                               f"checkpoint: {lost[:4]}")
        print(f"\n[{name}] step {sd.get('step')}  "
              f"interval={adapter.cfg.interval_kind}")

        rows = []
        for d in args.deltas:
            t = torch.full((n,), args.t_center + d / 2, device=args.device)
            r = torch.full((n,), args.t_center - d / 2, device=args.device)
            tt = t.reshape(-1, 1, 1, 1)
            z_t = (1 - tt) * x0 + tt * eps

            def base_fn(z_, s_):
                return adapter.base_velocity(z_, s_, pe, po)

            u_true = true_average_velocity(base_fn, z_t, t, r, args.substeps)
            with torch.no_grad():
                u_hat = adapter(z_t, t, r, pe, po)
                v_inst = adapter.base_velocity(z_t, t, pe, po)
                # r-sensitivity needs the model at the SAME (z, t) with r = t,
                # i.e. Delta = 0. That is the model's own notion of the
                # instantaneous field, which is not identical to v_inst once
                # the adapter is ungated.
                u_hat_r0 = adapter(z_t, t, t.clone(), pe, po)

            e_model = float(sse(u_hat, u_true).mean())
            e_zero = float(sse(torch.zeros_like(u_true), u_true).mean())
            e_base = float(sse(v_inst, u_true).mean())

            # --- r-sensitivity, D_r ---------------------------------------
            # The TRUE r-dependence is not a free parameter: v = u_true at
            # Delta = 0, so ||u_true(Delta) - v|| / ||u_true|| is exactly the
            # dependence the adapter is supposed to acquire. Measured on the
            # P75 checkpoints it runs 2.5% at Delta=0.05 to 31.7% at
            # Delta=0.8, while the adapter's own deviation sat at ~8.5% at
            # EVERY Delta. So the null for D_r is this curve, NOT zero: a
            # model emitting a fixed 8.5% perturbation would otherwise look
            # like it had "some" r-sensitivity.
            nrm = lambda x: x.flatten(1).norm(dim=1)
            dr_true = float((nrm(u_true - v_inst) / nrm(u_true).clamp(min=1e-8)).mean())
            dr_model = float((nrm(u_hat - u_hat_r0) / nrm(u_hat_r0).clamp(min=1e-8)).mean())
            # Direction, not just magnitude: does the adapter's correction
            # point where the true one does? A model with the right magnitude
            # and a random direction scores ~0 here and is worth nothing.
            corr = torch.nn.functional.cosine_similarity(
                (u_hat - u_hat_r0).flatten(1), (u_true - v_inst).flatten(1),
                dim=1)
            cm = correction_metrics(u_hat - u_hat_r0, u_true - v_inst)
            rows.append({
                "delta": d, "sse_model": e_model, "sse_zero": e_zero,
                "sse_base_instantaneous": e_base,
                "vs_zero": e_model / max(e_zero, 1e-12),
                "vs_base": e_model / max(e_base, 1e-12),
                "u_true_norm": float(u_true.flatten(1).norm(dim=1).mean()),
                "u_hat_norm": float(u_hat.flatten(1).norm(dim=1).mean()),
                "excess_over_base": e_model - e_base,
                "D_r_true": dr_true,
                "D_r_model": dr_model,
                "D_r_ratio": dr_model / max(dr_true, 1e-12),
                "cos_correction": float(corr.mean()),
                **cm,
            })
            rr = rows[-1]
            print(f"  d={d:<5} model {e_model:>10.4g}  base {e_base:>10.4g}"
                  f"  m/b {rr['vs_base']:>6.3f}"
                  f"  excess {rr['excess_over_base']:>9.4g}"
                  f"  D_r {100*rr['D_r_model']:>5.1f}% vs true "
                  f"{100*rr['D_r_true']:>5.1f}%"
                  f"  cos {rr['cos_correction']:>+6.3f}"
                  f"  corr_err {rr['corr_err']:>6.3f}"
                  f" (best@cos {rr['corr_err_best']:>6.3f})")

        results[name] = {"step": sd.get("step"),
                         "interval_kind": adapter.cfg.interval_kind,
                         "rows": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "results": results}, indent=1, default=str))

    print("\n" + "=" * 78)
    print("model/base < 1 means the learned interval field beats simply reusing")
    print("the instantaneous velocity. The largest Delta where that holds is")
    print("the Gate B answer.\n")
    hdr = f"{'ckpt':<30}" + "".join(f"{'d=' + str(d):>10}" for d in args.deltas)
    print(hdr)
    for name, res in results.items():
        print(f"{name:<30}" + "".join(f"{r['vs_base']:>10.3f}"
                                      for r in res["rows"]))
    print("(model/base; < 1.0 beats doing nothing)")

    print("\n--- D_r: has the adapter acquired the interval dimension? ---")
    print(f"{'':<30}" + "".join(f"{'d=' + str(d):>10}" for d in args.deltas))
    first = next(iter(results.values()))
    print(f"{'TRUE (target)':<30}"
          + "".join(f"{100*r['D_r_true']:>9.1f}%" for r in first["rows"]))
    for name, res in results.items():
        print(f"{name:<30}"
              + "".join(f"{100*r['D_r_model']:>9.1f}%" for r in res["rows"]))
    print("\nD_r_model / D_r_true  (1.0 = right magnitude of r-dependence):")
    for name, res in results.items():
        print(f"{name:<30}" + "".join(f"{r['D_r_ratio']:>10.2f}"
                                      for r in res["rows"]))
    print("\ncos(model correction, true correction)  (1.0 = right direction):")
    for name, res in results.items():
        print(f"{name:<30}" + "".join(f"{r['cos_correction']:>10.3f}"
                                      for r in res["rows"]))

    print("\ncorrection error ||c_model - c_true||^2 / ||c_true||^2 "
          "(<1 beats predicting no correction):")
    for name, res in results.items():
        print(f"{name:<30}" + "".join(f"{r['corr_err']:>10.3f}"
                                      for r in res["rows"]))
    print("  ...best achievable at each sample's own cos (magnitude "
          "ratio = cos):")
    for name, res in results.items():
        print(f"{name:<30}" + "".join(f"{r['corr_err_best']:>10.3f}"
                                      for r in res["rows"]))

    print("\n--- excess over base: sse_model - sse_base ---")
    print("A value that is CONSTANT across Delta means the adapter added a")
    print("Delta-INDEPENDENT perturbation and learned nothing about intervals.")
    for name, res in results.items():
        ex = [r["excess_over_base"] for r in res["rows"]]
        sub = [e for e, r in zip(ex, res["rows"]) if r["delta"] <= 0.3]
        spread = (100 * float(np.std(sub) / max(np.mean(sub), 1e-12))
                  if sub else float("nan"))
        print(f"{name:<30}" + "".join(f"{e:>10.4g}" for e in ex)
              + f"   (d<=0.3 spread {spread:.1f}%)")

    print("\nREAD IN THIS ORDER:")
    print("  1. cos_correction. Near 0 means the correction points nowhere")
    print("     useful, and D_r magnitude is then irrelevant.")
    print("  2. D_r ratio across Delta. Flat means no interval dependence was")
    print("     learned, whatever the magnitude.")
    print("  3. excess spread. Small spread confirms 1 and 2 independently.")
    print("  4. corr_err against corr_err_best. Equal means the magnitude is")
    print("     already right for the direction; only better direction helps.")
    print("  5. model/base. Only meaningful once the above show real learning,")
    print("     and at small Delta it is dominated by r-INDEPENDENT drift.")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
