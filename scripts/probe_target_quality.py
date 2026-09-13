"""
Grade the MeanFlow regression target against the TRUE average velocity. No training.

    PYTORCH_JIT=0 python scripts/probe_target_quality.py --selftest        # CPU, seconds
    PYTORCH_JIT=0 python scripts/probe_target_quality.py \\
        --cache /data/cache/<hash> --samples 4 --deltas 0.2                 # smoke
    PYTORCH_JIT=0 python scripts/probe_target_quality.py \\
        --cache /data/cache/<hash> --out runs/target_quality.json           # full

WHY. The centred-embedding run showed the Delta branch random-walking while
the t branch learned. The training log put the target's Delta-term,
Delta * ||du/dt||, at a median 47x the true correction ||u_true - v||.
That points at the TARGET, not the adapter. This probe measures target
quality directly, at initialisation (u = v, the frozen SD3 field), against
the model-independent average velocity eval_interval_field.py integrates.

WHAT IS GRADED. The part of the target that carries interval information:

    c      = -Delta * du/dt            (the target's correction term)
    c_true = u_true - v_marg           (what the correction should be)

for each (derivative variant) x (z tangent):

  derivative
    total        JVP with tangent (w, 1). What training uses.
    lp>=P        same JVP, but the timestep embedding's tangent is kept only
                 for components with period >= P in normalised t. Forward
                 VALUE is exactly the pretrained one (m*e + (1-m)*sg(e)); only
                 the derivative of the fast components is removed. Stage 1
                 found the dominant ||du/dt|| term at period 0.0199 (component
                 16, ~316x gain under d/dt).
    spatial      JVP with tangent (w, 0): no time derivative at all.
    fd=E         central difference over +-E along (w, 1). Averages over the
                 fast components instead of removing them. NOT previously
                 validated on this model.

  z tangent
    cond         w = eps - x0. What training uses (reference MeanFlow).
    marg         w = v_marg, the frozen field at z_t. Since the JVP is linear
                 in w and E[eps - x0 | z_t] = v_marg, the marg row is the
                 conditional EXPECTATION of the cond row: marg shows the BIAS,
                 cond shows bias + variance.

COLUMNS
    cos       mean per-sample cos(c, c_true). The direction. 0 = useless.
    cos_null  cos(c_i, c_true_j) for a different sample j in the same chunk.
              Structure shared across unrelated samples. cos means nothing
              unless it clears this.
    gain      ||c||/||c_true|| * cos: the component along the truth.
    orth      ||c||/||c_true|| * sqrt(1 - cos^2): the component that is noise.
    n_snr1    (orth/gain)^2: rough number of samples averaged before the
              correct direction reaches SNR 1. Order of magnitude only.
    tgt/base  sse(full target, u_true) / sse(v_marg, u_true). The whole
              target, first term included, vs doing nothing.
    S         ||du/dt|| / ||v_marg||. Stage 1 (frozen, CFG 1, marg tangent,
              t over [0.02, 0.98]): total 11.1, spatial 2.84.

A NOTE ON gain. At init u = v, so du/dt is dv/dt, while to first order the
true correction is -(Delta/2) dv/dt. A correct first-iterate target therefore
has gain ~2 at small Delta, falling at larger Delta. gain ~2 is not a bug;
the MeanFlow fixed point corrects the scale as u moves.

PRE-REGISTERED READING (written before any result exists)
  1. total/cond is the training target. Expected: cos near cos_null and
     n_snr1 in the thousands, reproducing the ~47x from the log. If instead
     its cos is clearly positive (say > 0.3, i.e. n_snr1 ~ 10), the diagnosis
     is wrong and the Gate B failure is capacity or optimisation -- next test
     is distillation straight onto u_true.
  2. total/marg ~ 0 but lp/marg or spatial/marg clearly > 0.3: the timestep
     embedding's fast components are the bias. The low-pass target is the
     candidate for the next run.
  3. marg rows fine, matching cond rows collapse: the problem is variance from
     the conditional tangent. Fix is variance reduction (the marginal tangent
     costs one extra forward), not the embedding.
  4. every row near cos_null, spatial/marg included: the first-order target at
     init does not point at the true average velocity on SD3 at these Deltas.
     No target modification rescues it; distillation onto u_true (or much
     smaller Delta) is what remains.

PREFLIGHT, printed before the sweep and stored in the output:
  - time_proj layout matches the frequency formula the mask assumes
  - the low-pass hook leaves the forward value unchanged
  - lp>=inf equals spatial (the hook removes exactly the time tangent)
  - JVP total against central differences at eps 1e-2..1e-4, TF32 off and
    on (the first smoke run found 74% disagreement at 1e-4 with TF32 on)
  - S_total and S_spatial for comparison with Stage 1

COST. Per (sample, Delta): 2*substeps teacher forwards for u_true, plus
(3 + #cutoffs) JVPs and 2*#fd_eps forwards per tangent. At the defaults and 32
samples x 6 Deltas this is on the order of an hour on a 24 GB card. Run the
smoke first. `--samples 32` with the default seeds reproduces exactly the
samples and noise of eval_interval_field.py, so D_r_true should match that
JSON (0.0248, 0.0446, 0.0774, 0.109, 0.176, 0.315).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

T_SCALE = 1000.0            # AdapterConfig.num_train_timesteps
DELTAS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)


# ----------------------------------------------------------------- low-pass --
class TimeDerivativeLowPass:
    """Forward hook on SD3's time_text_embed.time_proj.

    With `cutoff = P`, returns  m * e + (1 - m) * e.detach()  where m selects
    the components whose period in normalised t is >= P. The VALUE is e,
    bit-for-bit; forward-mode AD sees a tangent only through the slow
    components. `cutoff = None` disables the hook; `cutoff = inf` removes the
    whole time tangent.
    """

    def __init__(self, module, t_scale: float = T_SCALE,
                 max_period: float = 10_000.0) -> None:
        self.module = module
        self.cutoff = None
        self.n = int(module.num_channels)
        self.half = self.n // 2
        shift = float(getattr(module, "downscale_freq_shift", 0.0))
        self.scale = float(getattr(module, "scale", 1.0))
        k = torch.arange(self.half, dtype=torch.float64)
        # angular frequency per unit TIMESTEP, as diffusers computes it
        self.freqs = torch.exp(-math.log(max_period) * k / (self.half - shift))
        # period per unit NORMALISED t
        self.period_t = 2 * math.pi / (self.freqs * self.scale * t_scale)
        self._masks: dict = {}
        self.handle = module.register_forward_hook(self._hook)

    def mask(self, cutoff: float, device, dtype) -> torch.Tensor:
        key = (cutoff, str(device), dtype)
        if key not in self._masks:
            keep = (self.period_t >= cutoff).to(dtype)
            m = torch.cat([keep, keep])
            if self.n % 2:
                m = torch.cat([m, torch.zeros(1, dtype=dtype)])
            self._masks[key] = m.to(device)
        return self._masks[key]

    def kept(self, cutoff: float) -> int:
        return int((self.period_t >= cutoff).sum())

    def _hook(self, module, inputs, output):
        if self.cutoff is None:
            return None
        m = self.mask(self.cutoff, output.device, output.dtype)
        return m * output + (1 - m) * output.detach()

    @torch.no_grad()
    def verify(self, timesteps: torch.Tensor) -> str:
        """Confirm the module's output is [cos|sin](k-ascending) x2 with the
        frequencies assumed above. The mask is identical for both half
        orders, but it is NOT identical for a different frequency formula --
        a diffusers change there would silently mask the wrong components."""
        saved, self.cutoff = self.cutoff, None
        out = self.module(timesteps).double()[:, : 2 * self.half]
        self.cutoff = saved
        a = (timesteps.double()[:, None] * self.scale
             * self.freqs.to(timesteps.device)[None])
        for name, ref in (("cos_sin", torch.cat([a.cos(), a.sin()], -1)),
                          ("sin_cos", torch.cat([a.sin(), a.cos()], -1))):
            if torch.allclose(out, ref, atol=1e-3):
                return name
        raise RuntimeError("time_proj output does not match the assumed "
                           "sinusoidal formula; the low-pass mask would be "
                           "wrong. Inspect diffusers' Timesteps before use.")


# ---------------------------------------------------------------- selftest --
def selftest() -> int:
    from diffusers.models.embeddings import Timesteps
    torch.manual_seed(0)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            fails.append(name)

    tp = Timesteps(num_channels=256, flip_sin_to_cos=True,
                   downscale_freq_shift=0)
    lin = torch.nn.Linear(256, 8)
    lp = TimeDerivativeLowPass(tp)
    t = torch.rand(5) * 0.8 + 0.1
    ones = torch.ones_like(t)
    f = lambda tt: lin(tp(tt * T_SCALE))  # noqa: E731

    check("layout matches formula", lp.verify(t * T_SCALE) in
          ("cos_sin", "sin_cos"), lp.verify(t * T_SCALE))
    p16 = float(lp.period_t[16])
    check("component 16 period ~ 0.0199 (Stage 1)", abs(p16 - 0.0199) < 5e-4,
          f"{p16:.4f}")

    with torch.no_grad():
        lp.cutoff = None
        y0 = f(t)
        lp.cutoff = 0.05
        y1 = f(t)
    check("hook leaves value unchanged", torch.equal(y0, y1),
          f"max {(y0 - y1).abs().max().item():.1e}")

    lp.cutoff = None
    _, e_dot = torch.func.jvp(lambda tt: tp(tt * T_SCALE), (t,), (ones,))
    lp.cutoff = 0.05
    _, y_dot = torch.func.jvp(f, (t,), (ones,))
    expected = (e_dot * lp.mask(0.05, t.device, e_dot.dtype)) @ lin.weight.T
    rel = ((y_dot - expected).norm() / expected.norm()).item()
    check("tangent flows only through kept components", rel < 1e-4,
          f"rel {rel:.1e}  kept {lp.kept(0.05)}/{lp.half}")

    lp.cutoff = math.inf
    _, y_inf = torch.func.jvp(f, (t,), (ones,))
    check("cutoff=inf removes the whole time tangent",
          y_inf.abs().max().item() == 0.0,
          f"max {y_inf.abs().max().item():.1e}")
    lp.cutoff = None

    for c in (0.02, 0.05, 0.2, 0.5, 1.0):
        print(f"        cutoff {c:<5} keeps {lp.kept(c):>3}/{lp.half} components")
    print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


# ------------------------------------------------------------------- probe --
def rel_diff(a, b) -> float:
    return float(((a - b).flatten(1).norm(dim=1)
                  / b.flatten(1).norm(dim=1).clamp(min=1e-8)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="CPU check of the low-pass hook; no model, no cache")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4,
                    help="JVP chunk size; math attention materialises full "
                         "attention matrices, so lower this on OOM (>=2 keeps "
                         "the cos_null column)")
    ap.add_argument("--deltas", type=float, nargs="+", default=list(DELTAS))
    ap.add_argument("--t-center", type=float, default=0.5)
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.05, 0.5])
    ap.add_argument("--fd-eps", type=float, nargs="+",
                    default=[0.01, 0.025, 0.05])
    ap.add_argument("--tangents", nargs="+", default=["cond", "marg"],
                    choices=["cond", "marg"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--allow-tf32", action="store_true",
                    help="leave PyTorch's TF32 defaults on. Off by default: "
                         "cuDNN convolutions (SD3's patch embedding) run in "
                         "TF32 on Ampere unless disabled, and TF32's ~1e-4 "
                         "relative precision is the size of a fine "
                         "finite-difference step")
    ap.add_argument("--out", default="runs/target_quality.json")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.cache:
        ap.error("--cache is required unless --selftest")

    from diffusers import SD3Transformer2DModel
    from eval_interval_field import true_average_velocity
    from transport.data import TripletCache
    from transport.meanflow import math_attention

    dev = args.device
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    # ---- samples: identical selection and noise to eval_interval_field ----
    cache = TripletCache(args.cache)
    n = min(args.samples, len(cache))
    idx = np.random.default_rng(0).choice(len(cache), size=n, replace=False)
    x0 = torch.stack([torch.from_numpy(cache[int(j)]["z_tgt"]) for j in idx])
    pe = torch.stack([torch.from_numpy(cache[int(j)]["prompt_embeds"])
                      for j in idx])
    po = torch.stack([torch.from_numpy(cache[int(j)]["pooled"]) for j in idx])
    x0 = x0.to(dev).float()
    pe, po = pe.to(dev).float(), po.to(dev).float()
    g = torch.Generator(device=dev).manual_seed(1234)
    eps = torch.randn(x0.shape, device=dev, generator=g)
    print(f"[probe] {n} samples, batch {args.batch}, deltas {args.deltas}")

    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.float32
    ).to(dev).eval().requires_grad_(False)
    lp = TimeDerivativeLowPass(tr.time_text_embed.time_proj)

    def make_field(pe_, po_):
        def field(z_, t_):
            return tr(hidden_states=z_, encoder_hidden_states=pe_,
                      pooled_projections=po_, timestep=t_ * T_SCALE,
                      return_dict=False)[0]
        return field

    def jvp(field, z, t, w, dt_value, cutoff=None):
        lp.cutoff = cutoff
        try:
            with torch.no_grad(), math_attention():
                _, d = torch.func.jvp(field, (z, t),
                                      (w, torch.full_like(t, dt_value)))
        finally:
            lp.cutoff = None
        return d

    def fd(field, z, t, w, e):
        with torch.no_grad():
            return (field(z + e * w, t + e) - field(z - e * w, t - e)) / (2 * e)

    def variants(field, z, t, w):
        yield "total", jvp(field, z, t, w, 1.0)
        for c in args.cutoffs:
            yield f"lp>={c:g}", jvp(field, z, t, w, 1.0, cutoff=c)
        yield "spatial", jvp(field, z, t, w, 0.0)
        for e in args.fd_eps:
            yield f"fd={e:g}", fd(field, z, t, w, e)

    # ---- preflight ---------------------------------------------------------
    b0 = min(args.batch, n)
    d0 = 0.2 if 0.2 in args.deltas else args.deltas[0]
    t0 = torch.full((b0,), args.t_center + d0 / 2, device=dev)
    tt0 = t0.reshape(-1, 1, 1, 1)
    z0 = (1 - tt0) * x0[:b0] + tt0 * eps[:b0]
    field0 = make_field(pe[:b0], po[:b0])
    layout = lp.verify(t0 * T_SCALE)
    with torch.no_grad():
        v_plain = field0(z0, t0)
        lp.cutoff = args.cutoffs[0]
        v_hooked = field0(z0, t0)
        lp.cutoff = None
    d_total = jvp(field0, z0, t0, v_plain, 1.0)
    d_spatial = jvp(field0, z0, t0, v_plain, 0.0)
    d_inf = jvp(field0, z0, t0, v_plain, 1.0, cutoff=math.inf)
    # FD LADDER, TF32 off and on. The smoke run's single fd(1e-4) disagreed
    # with the JVP by 74%. Two readings predict different shapes:
    #   numerical (roundoff/TF32): error small at 1e-3, rising at 1e-4, and
    #     worse with TF32 on
    #   JVP wrong: error large at every step size, TF32 irrelevant
    # 1e-2 is expected to disagree regardless: a central difference that wide
    # attenuates the period-0.02 component by sin(3.16)/3.16 ~ 0.
    ladder = {}
    for tf32 in (False, True):
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for e in (1e-2, 1e-3, 3e-4, 1e-4):
            ladder[f"tf32={'on' if tf32 else 'off'} eps={e:g}"] = rel_diff(
                fd(field0, z0, t0, v_plain, e), d_total)
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    vn = v_plain.flatten(1).norm(dim=1)
    pre = {
        "time_proj_layout": layout,
        "component16_period": float(lp.period_t[16]),
        "kept_components": {f"{c:g}": lp.kept(c) for c in args.cutoffs},
        "hook_value_rel_diff": rel_diff(v_hooked, v_plain),
        "lp_inf_vs_spatial_rel_diff": rel_diff(d_inf, d_spatial),
        "tf32_during_sweep": args.allow_tf32,
        "fd_vs_jvp_total_rel_diff": ladder,
        "S_total_marg": float((d_total.flatten(1).norm(dim=1) / vn).mean()),
        "S_spatial_marg": float((d_spatial.flatten(1).norm(dim=1) / vn).mean()),
    }
    print("\n--- preflight ---")
    for k, v in pre.items():
        print(f"  {k:<30} {v}")
    print("  expect: hook diff ~0 (<1e-5), lp>=inf vs spatial ~0 (<1e-4); "
          "fd ladder: large at 1e-2 (filters the fast component), a few % at "
          "1e-3 if the JVP is right")
    if pre["hook_value_rel_diff"] > 1e-5 or pre["lp_inf_vs_spatial_rel_diff"] > 1e-4:
        print("  !! a hook check is off -- read the numbers before trusting "
              "the lp rows")
    del v_plain, v_hooked, d_total, d_spatial, d_inf

    # ---- sweep -------------------------------------------------------------
    results = {}
    for d in args.deltas:
        t_start = time.time()
        acc = defaultdict(lambda: defaultdict(list))
        sse_base_all, dr_true_all = [], []
        for s in range(0, n, args.batch):
            sl = slice(s, min(s + args.batch, n))
            b = sl.stop - sl.start
            t = torch.full((b,), args.t_center + d / 2, device=dev)
            r = torch.full((b,), args.t_center - d / 2, device=dev)
            tt = t.reshape(-1, 1, 1, 1)
            x0b, epsb = x0[sl], eps[sl]
            z_t = (1 - tt) * x0b + tt * epsb
            field = make_field(pe[sl], po[sl])

            with torch.no_grad():
                v_marg = field(z_t, t)
            u_true = true_average_velocity(field, z_t, t, r, args.substeps)
            c_true = (u_true - v_marg).flatten(1)
            nt = c_true.norm(dim=1).clamp(min=1e-8)
            sse_base = (v_marg - u_true).pow(2).flatten(1).sum(1)
            sse_base_all += sse_base.tolist()
            dr_true_all += (nt / u_true.flatten(1).norm(dim=1)).tolist()
            vn = v_marg.flatten(1).norm(dim=1)

            tangents = {"cond": epsb - x0b, "marg": v_marg}
            for tname in args.tangents:
                w = tangents[tname]
                for vname, du_dt in variants(field, z_t, t, w):
                    c = (-d * du_dt).flatten(1)
                    ratio = c.norm(dim=1) / nt
                    cos = F.cosine_similarity(c, c_true, dim=1)
                    row = acc[f"{vname}/{tname}"]
                    row["cos"] += cos.tolist()
                    if b > 1:
                        row["cos_null"] += F.cosine_similarity(
                            c, c_true.roll(1, dims=0), dim=1).tolist()
                    row["gain"] += (ratio * cos).tolist()
                    row["orth"] += (ratio * (1 - cos ** 2).clamp(min=0)
                                    .sqrt()).tolist()
                    # first term is the tangent itself: eps - x0 for cond
                    # (as in training), v_marg for marg
                    tgt = w - d * du_dt
                    row["sse_tgt"] += (tgt - u_true).pow(2).flatten(1).sum(1).tolist()
                    row["S"] += (du_dt.flatten(1).norm(dim=1) / vn).tolist()

        base = float(np.mean(sse_base_all))
        summary = {}
        for key, row in acc.items():
            gain, orth = float(np.mean(row["gain"])), float(np.mean(row["orth"]))
            summary[key] = {
                "cos": float(np.mean(row["cos"])),
                "cos_null": (float(np.mean(row["cos_null"]))
                             if row["cos_null"] else float("nan")),
                "gain": gain, "orth": orth,
                "n_snr1": (orth / gain) ** 2 if gain > 0 else float("inf"),
                "tgt_vs_base": float(np.mean(row["sse_tgt"])) / max(base, 1e-12),
                "S": float(np.mean(row["S"])),
            }
        results[f"{d:g}"] = {"D_r_true": float(np.mean(dr_true_all)),
                             "sse_base": base, "variants": summary}
        print(f"\n[delta {d:g}] D_r_true {100 * np.mean(dr_true_all):.2f}%  "
              f"({time.time() - t_start:.0f}s)")
        for key, m in summary.items():
            mark = "   <- training target" if key == "total/cond" else ""
            print(f"  {key:<16} cos {m['cos']:+.3f} (null {m['cos_null']:+.3f})"
                  f"  gain {m['gain']:7.2f}  orth {m['orth']:8.2f}"
                  f"  n_snr1 {m['n_snr1']:9.3g}  tgt/base {m['tgt_vs_base']:8.3g}"
                  f"  S {m['S']:6.2f}{mark}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "preflight": pre, "results": results},
            indent=1, default=str))

    # ---- tables --------------------------------------------------------------
    keys = list(next(iter(results.values()))["variants"].keys())
    cols = list(results.keys())
    for metric, fmt in (("cos", "{:+9.3f}"), ("n_snr1", "{:9.3g}"),
                        ("gain", "{:9.2f}"), ("tgt_vs_base", "{:9.3g}")):
        print(f"\n--- {metric} ---")
        print(f"{'':<16}" + "".join(f"{'d=' + c:>9}" for c in cols))
        if metric == "cos":
            print(f"{'(null, mean)':<16}" + "".join(
                "{:+9.3f}".format(float(np.nanmean(
                    [results[c]['variants'][k]['cos_null'] for k in keys])))
                for c in cols))
        for k in keys:
            print(f"{k:<16}" + "".join(
                fmt.format(results[c]["variants"][k][metric]) for c in cols))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
