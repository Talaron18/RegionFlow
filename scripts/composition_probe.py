"""Instrument B: does the learned interval field COMPOSE?  (H0 / H0' of H0_H0PRIME_SPEC.md)

THE QUESTION. MeanFlow-style training claims that one interval jump

    z_j' = z_i - (t_i - t_j) * u_theta(z_i, t_i, t_j)

reproduces integrating the model's OWN field in many small steps. This script
measures exactly that claim, and nothing else:

    E_comp^field(Delta) = E_{(i,j)} [ || z_j' - z_j || / (||z_j|| + eps) ]

where z_i and z_j are points of that field's own fine-grid trajectory and
t_i - t_j = Delta. No interpolation line is involved anywhere: the reference is
the field's own multi-step composition, so a model that simply does not follow
the training line is not punished for it. (That was the flaw of the earlier
oracle: it compared the marginal flow against the conditional interpolation
line, which is a structural coupling gap, not an error.)

WHY THE FROZEN FIELD IS THE DENOMINATOR, AND WHY NOTHING HERE IS A THRESHOLD.

    E_frozen(Delta)  ->  (Delta^2 / 2) || Dg/Dt ||   as the reference refines

The frozen instantaneous field's one-jump error does NOT go to zero as the
reference converges: it converges to the Euler linearisation error, which is
the HEADROOM interval averaging exists to capture. So

    R_comp(Delta) = 1 - E_adapter(Delta) / E_frozen(Delta)

is "what fraction of the Euler headroom did the learned field explain":
    1  = a perfect flow map (one jump exact for every Delta)
    0  = no better than pretending the field is constant
    <0 = the learned correction actively hurts composition

H0 (INSTRUMENT VALIDATION) is a self-check on the reference, and it is NOT
"E -> 0":
    * E_N(Delta) stabilises as the fine step count N grows   (rho_N);
    * the trajectory endpoint stabilises                      (d_N).
If E_N drifts with N, the reference is not converged and nothing else is
readable. Small-Delta cells have almost no headroom, so R_comp ~ 0 there is not
evidence of failure: the few-step claim lives at mid/large Delta.

HYPOTHESES, NOT THEOREMS (do not read these as proven):
    - that E_a grows like Delta and R_comp therefore rises monotonically with
      Delta. A learned interval field's error is shaped by conditioning error,
      state-distribution shift, (t,r) extrapolation, capacity and trajectory
      curvature too. The only pre-registered statement is that R_comp should
      show a positive, SYSTEMATIC gain at sufficiently large Delta.

READOUT LEVELS
    L1  E_comp(Delta), R_comp(Delta), paired per-item sign   -- main result
    L2  D_traj(t) = ||z^a_t - z^f_t|| / ||z^f_t||            -- did the adapter
        change the trajectory at all? Explanatory, never a success metric.
    L3  shared-state probe (triggered when L2 drifts): fine-integrate EACH
        field from the SAME state, then compare that field's one jump against
        its own reference. Separates trajectory drift from interval error.

H6 METRICS FROM THE START. Every probed jump is also scored in image space
(decoded jump vs decoded reference): MSE and PSNR always, SSIM/LPIPS only when
the heavy dependencies import (bench/metrics.py policy). Masks from the cache
are on the LATENT grid, so they are upsampled 8x for image-space regions.

USAGE

    python scripts/composition_probe.py \
        --ckpt runs/fd_distill_5k/ckpt_final.pt \
        --cache /data/cache/13061d51aee4 \
        --limit 4 --fine-steps 64 128 256 --deltas 0.1 0.25 0.5 \
        --fields adapter teacher --out runs/comp_h0

    NO SUBCOMMAND, unlike project_edit.py (`displace` | `project`). A stray
    leading `project` is ignored with a note rather than rejected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                  # so `import project_edit` works
sys.path.insert(0, str(_HERE.parent))           # repo root

from project_edit import _rel, build_adapter, read_cache   # noqa: E402

#: Stage 0's clamp. The composition question is about the field, so keep the
#: trajectory inside the range SD3 is trained on; the degenerate ends are a
#: separate question, already priced at 5-10% on the old oracle.
SIGMA_MIN, SIGMA_MAX = 0.05, 0.98
#: 512 image px / 64 latent px, for upsampling a cache mask to image space.
MASK_UP = 8


# --------------------------------------------------------------------- core --
def fine_grid(t_from: float, t_to: float, steps: int):
    """Uniform grid from t_from to t_to, endpoints included. Descending when
    t_from > t_to (our sampler's direction: noise -> data)."""
    return [t_from + (t_to - t_from) * i / steps for i in range(steps + 1)]


@torch.no_grad()
def integrate(field, z0, t_from, t_to, steps):
    """Walk the field from t_from to t_to in `steps` equal pieces.

    field(z, t, r) -> u, the sampler's convention
        z_r = z_t - (t - r) * u(z_t, t, r)
    Returns (states, z_end); states[k] is the state after k steps.
    """
    z = z0.clone()
    ts = fine_grid(t_from, t_to, steps)
    states = [z.clone()]
    for k in range(steps):
        t, r = ts[k], ts[k + 1]
        z = z - (t - r) * field(z, t, r)
        states.append(z.clone())
    return states, z


@torch.no_grad()
def jump(field, z_i, t_i: float, t_j: float):
    """One interval jump: the object MeanFlow's claim is about."""
    return z_i - (t_i - t_j) * field(z_i, t_i, t_j)


def comp_error(j_out, z_ref) -> float:
    return float((j_out - z_ref).norm() / (z_ref.norm() + 1e-12))


# ------------------------------------------------- correction-scale (G1) ------
def make_scale_fn(expr: str | None):
    """a(t) from a CLI expression, e.g. "0.5 if t >= 0.35 else 1.0".

    ONLY `t` IS AVAILABLE, and that is deliberate. In `eval_interval_field.py`
    an `a` may be a function of the CELL (tc, d) because there it scales a
    READOUT evaluated at one fixed (t, r). Here the scaled object has to be a
    FIELD: this probe integrates it to build its own reference trajectory, and
    an `a` that depended on the jump width would make that reference
    ill-defined. Gate B's rule is a function of t anyway ("a = 1.0/0.5"), so
    nothing under test is lost. An expression mentioning `tc` or `d` raises
    NameError rather than silently reading something else.
    """
    if expr is None:
        return None

    def _a(t: float) -> float:
        return float(eval(expr, {"__builtins__": {}}, {"t": t}))  # noqa: S307
    return _a


def scaled_field(field, base, a_of_t):
    """u_a = base + a(t) * (field - base) -- the affine blend toward frozen.

    A BLEND TOWARD `base`, NOT A MULTIPLIED CORRECTION, because of what the two
    endpoints mean:

        a = 0  ->  u_a IS base  ->  E_comp(a=0) MUST EQUAL E_frozen
        a = 1  ->  u_a IS field ->  E_comp(a=1) MUST EQUAL E_adapter

    Both endpoints return the SAME OBJECT rather than an arithmetic
    reconstruction, so those two identities are bit-exact instead of
    approximate. That turns the a-curve's endpoints into a wiring self-check
    that costs no extra run to interpret -- and it is also why `a` is applied
    to the trajectory AND to the jump here, while `eval_interval_field.py`'s
    `--corr-scale` only rescales a readout.
    """

    def _f(z, t, r):
        a = a_of_t(float(t))
        if a == 1.0:
            return field(z, t, r)
        v = base(z, t)
        if a == 0.0:
            return v
        return v + a * (field(z, t, r) - v)
    return _f


def const_scale(a: float):
    def _f(_t: float) -> float:
        return float(a)
    return _f


def data_line(t: float, z_tgt, zT):
    """The state `cos` is scored on: (1-t)*x0 + t*eps, with x0 = z_tgt.

    `cos` in eval_interval_field.py is measured on THIS line; E_comp runs on
    the field's OWN trajectory states. This function is what lets the gap be
    read instead of assumed (see the --states-gap readout).
    """
    return (1.0 - t) * z_tgt + t * zT


def make_manifest(zT_ids, deltas, pairs_per_cell, steps, seed,
                  t_from=SIGMA_MAX, t_to=SIGMA_MIN):
    """Jump pairs as index pairs on THIS grid, so z_i and z_j are exact
    trajectory points and no interpolation is involved.

    INDEX CONVENTION: the grid runs from t_from = sigma_max (index 0) DOWN to
    t_to = sigma_min, so a larger index is a SMALLER time. A jump therefore has
    i < j with t_i > t_j and delta = t_i - t_j > 0, matching
    z_j = z_i - (t_i - t_j) u(z_i, t_i, t_j).

    Both fields consume the same list for a given (steps, delta): otherwise a
    field that fails on some trajectory point silently changes the sampling
    distribution and the two E_comp numbers stop being comparable. Pairs are
    sampled from a seeded RNG, so the manifest is reproducible and is written
    to disk next to the results.
    """
    rng = np.random.default_rng(seed)
    h = (t_to - t_from) / steps                     # negative: t descends
    out = []
    for zid in zT_ids:
        for d in deltas:
            k = max(1, int(round(abs(d) / abs(h))))
            if k > steps:
                continue
            for _ in range(pairs_per_cell):
                j = int(rng.integers(k, steps + 1))
                i = j - k                            # i < j, t_i > t_j
                out.append(dict(zT=int(zid), i=int(i), j=int(j),
                                t_i=float(t_from + i * h),
                                t_j=float(t_from + j * h),
                                delta=float(abs(k * h)), delta_req=float(d),
                                steps=int(steps)))
    return out


def summarise(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), max=float(v.max()),
                std=float(v.std(ddof=0)) if v.size > 1 else 0.0)


def ssim_callable():
    """SSIM if scikit-image is importable, else None. Never a hard dependency."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    def _ssim(a, b):
        return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    return _ssim


def lpips_callable(device):
    """LPIPS if the package is installed, else None (bench/metrics.py policy)."""
    try:
        import lpips
    except Exception:
        return None
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def _lpips(a, b):
        ta = torch.from_numpy(a).float().permute(2, 0, 1)[None].to(device)
        tb = torch.from_numpy(b).float().permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            return float(net(ta * 2 - 1, tb * 2 - 1).item())
    return _lpips


# --------------------------------------------------------------------- main --
def _argv_without_stray_subcommand(argv=None) -> list:
    """Tolerate a leading `project`/`displace` token.

    THIS PROBE HAS NO SUBCOMMANDS; project_edit.py does (`displace` | `project`),
    and the two are run in the same shell loop, so the token gets copied. It
    cost one round trip, and argparse's "unrecognized arguments: project" reads
    like a bug in the script rather than a wrong invocation. Say which it is and
    carry on.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("project", "displace"):
        print(f"[cli] ignoring the stray {argv[0]!r}: composition_probe.py "
              f"takes no subcommand (that one belongs to project_edit.py)")
        argv = argv[1:]
    return argv


# --------------------------------------------------------------- self-test --
def self_test() -> None:
    """CPU, no model, no cache, no GPU.

    The invariants asserted here are exactly where a silent wiring error would
    live: the two endpoints of the a-curve, and the claim that a=0 reproduces
    the frozen trajectory rather than merely approximating it.
    """
    torch.manual_seed(0)
    C, h, w = 3, 4, 4

    def base(z, t, r=None):                 # 3-arg like a real field, 2-arg
        return torch.full_like(z, 0.25) * t  # like scaled_field calls it

    def field(z, t, r):
        return torch.full_like(z, 0.5) * t - 0.1 * r

    z = torch.randn(C, 1, h, w)
    t, r = 0.7, 0.4

    # 1. THE ENDPOINTS ARE THE SAME OBJECT, bit-exact. This is what makes
    #    E_comp(a=0) == E_frozen structural rather than hoped for.
    assert torch.equal(scaled_field(field, base, const_scale(0.0))(z, t, r),
                       base(z, t)), "a=0 must be bit-exact to the frozen field"
    assert torch.equal(scaled_field(field, base, const_scale(1.0))(z, t, r),
                       field(z, t, r)), "a=1 must be bit-exact to the field"

    # 2. ...and the frozen TRAJECTORY is reproduced exactly, which is what the
    #    readout actually compares.
    z0 = torch.randn(C, 1, h, w)
    s_frozen, _ = integrate(base, z0, SIGMA_MAX, SIGMA_MIN, 8)
    s_a0, _ = integrate(scaled_field(field, base, const_scale(0.0)),
                        z0, SIGMA_MAX, SIGMA_MIN, 8)
    assert all(torch.equal(a, b) for a, b in zip(s_frozen, s_a0)), \
        "the a=0 trajectory must be bit-identical to the frozen one"

    ti = s_frozen[3]
    e_frozen = comp_error(jump(base, ti, 0.6, 0.3), s_frozen[6])
    e_a0 = comp_error(jump(scaled_field(field, base, const_scale(0.0)),
                           ti, 0.6, 0.3), s_a0[6])
    assert e_frozen == e_a0, (e_frozen, e_a0)

    s_f1, _ = integrate(field, z0, SIGMA_MAX, SIGMA_MIN, 8)
    assert (comp_error(jump(scaled_field(field, base, const_scale(1.0)),
                            ti, 0.6, 0.3), s_f1[6])
            == comp_error(jump(field, ti, 0.6, 0.3), s_f1[6]))

    # 3. the blend is the affine one at a generic a
    got = scaled_field(field, base, const_scale(0.25))(z, t, r)
    want = base(z, t) + 0.25 * (field(z, t, r) - base(z, t))
    assert torch.allclose(got, want), (float(got[0, 0, 0, 0]),
                                       float(want[0, 0, 0, 0]))

    # 4. THE EXPRESSION INTERFACE. Gate B's rule must be expressible, and the
    #    variables it must NOT see really are absent -- a silently-available
    #    `d` would define an "a" that changes with the jump, i.e. no field.
    rule = make_scale_fn("0.5 if t >= 0.35 else 1.0")
    assert [rule(x) for x in (0.5, 0.35, 0.34)] == [0.5, 0.5, 1.0], \
        [rule(x) for x in (0.5, 0.35, 0.34)]
    for bad in ("d", "tc", "__import__('os')"):
        try:
            make_scale_fn(bad)(0.5)
        except (NameError, SyntaxError) as exc:
            assert str(exc), bad
        else:
            raise AssertionError(f"{bad!r} must not be evaluable")

    # 5. the data line's two ends
    z_tgt, zT = torch.randn(C, 1, h, w), torch.randn(C, 1, h, w)
    assert torch.equal(data_line(0.0, z_tgt, zT), z_tgt)
    assert torch.equal(data_line(1.0, z_tgt, zT), zT)

    print("composition_probe self-test OK")
    print("  a=0 is bit-exact to the frozen field => E_comp(a=0) == E_frozen")
    print("  a=1 is bit-exact to the raw field    => E_comp(a=1) == E_adapter")
    print("  the a-curve endpoints are therefore a wiring check, not a result")
    print("  --corr-scale sees `t` only: tc/d raise instead of reading junk")
    print("  data_line(1) = noise, data_line(0) = target, exactly")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=4,
                    help="cache items; each contributes its instruction "
                         "embedding and one noise sample (one zT)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--fields", nargs="+", default=["adapter", "teacher"],
                    choices=("adapter", "teacher"))
    ap.add_argument("--fine-steps", type=int, nargs="+", default=[64, 128, 256],
                    help="the self-check list; E_N must stabilise over it. The "
                         "LARGEST is used for the L1 readout")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--pairs-per-cell", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-states", type=int, default=0,
                    help="L3 shared-state probe count; 0 = auto (on when L2 "
                         "drift exceeds --probe-drift, else off)")
    ap.add_argument("--probe-drift", type=float, default=0.05)
    ap.add_argument("--image-metrics", type=int, default=2,
                    help="decode this many jump/reference pairs per delta for "
                         "MSE/PSNR (+SSIM/LPIPS when available). 0 = skip and "
                         "do not even load the VAE")
    ap.add_argument("--corr-scale", default=None,
                    help="G1: scale the adapter's correction toward the frozen "
                         "field by a(t), e.g. \"0.5 if t >= 0.35 else 1.0\". "
                         "Reported as a curve NEXT TO the unscaled L1; the main "
                         "L1 readout is left untouched so H0' stays comparable.")
    ap.add_argument("--corr-scale-sweep", type=float, nargs="*", default=None,
                    help="constant a values to trace the curve over, e.g. "
                         "0.0 0.5 1.0. a=0 must reproduce E_frozen and a=1 must "
                         "reproduce E_adapter -- those two cells are a wiring "
                         "self-check, so run them in every invocation.")
    ap.add_argument("--states-gap", "--data-line", dest="states_gap",
                    action="store_true",
                    help="G4: query the SAME (t,r) cells from the field's own "
                         "trajectory states AND from the interpolation line "
                         "(1-t)x0 + t*eps -- the states `cos` is measured on -- "
                         "and compare. Needs z_tgt in the cache. NOTE: "
                         "--data-line is the old name of this flag; its old "
                         "D_data readout was wrong and has been replaced, see "
                         "the block in main().")
    ap.add_argument("--states-gap-n", type=int, default=8,
                    help="entries per delta for the G4 probe. Each costs |i-j| "
                         "extra forward passes PER FIELD, so it runs on a "
                         "subset rather than all 160 entries.")
    args = ap.parse_args(_argv_without_stray_subcommand())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from transport.data import TripletCache

    adapter = build_adapter(args.ckpt, "meanflow", args.device, args.rank)
    cache = TripletCache(args.cache)
    items = read_cache(cache, args.limit, args.device)
    by_id = {it["idx"]: it for it in items}
    steps_list = sorted(set(args.fine_steps))
    n_fine = steps_list[-1]

    def field_of(name, zid):
        """One closure per (field, item): the field the trajectory is OF."""
        it = by_id[zid]
        if name == "teacher":
            def _t(z, t, r):
                b = z.shape[0]
                return adapter.base_velocity(
                    z, torch.full((b,), float(t), device=z.device),
                    it["pe"], it["pooled"])
            return _t

        def _a(z, t, r):
            b = z.shape[0]
            return adapter(z, torch.full((b,), float(t), device=z.device),
                           torch.full((b,), float(r), device=z.device),
                           it["pe"], it["pooled"])
        return _a

    def base_of(name, zid):
        """(z, t) -> v_inst: the frozen instantaneous slice, the SAME object the
        `teacher` field is built from. Sharing it is what makes a=0 bit-exact
        to E_frozen instead of merely close to it."""
        it = by_id[zid]

        def _b(z, t):
            b = z.shape[0]
            return adapter.base_velocity(
                z, torch.full((b,), float(t), device=z.device),
                it["pe"], it["pooled"])
        return _b

    fields = {(n, zid): field_of(n, zid) for n in args.fields for zid in by_id}

    def key_of(name: str) -> str:
        """Row-key for a field. `teacher` IS the frozen field, and it is the
        denominator of R_comp by definition; naming the row key `e_frozen`
        keeps that explicit. (First version wrote `e_teacher` while the report
        looked up `e_frozen`, so E_frozen/R_comp/paired-fraction came out nan
        while the numbers were sitting in the self-check table.)"""
        return "frozen" if name == "teacher" else name

    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    zT = {zid: torch.randn(by_id[zid]["z_src"].shape, device=args.device,
                           generator=gen) for zid in by_id}

    manifests = {N: make_manifest(list(by_id), args.deltas,
                                  args.pairs_per_cell, N, args.seed + 1)
                 for N in steps_list}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(
        {str(k): v for k, v in manifests.items()}, indent=1))
    print(f"zT {len(zT)}  fields {args.fields}  fine-steps {steps_list}  "
          f"deltas {args.deltas}  pairs/cell {args.pairs_per_cell}")
    print(f"sigma range [{SIGMA_MIN}, {SIGMA_MAX}]  reference = each field's "
          f"OWN fine trajectory; the frozen one's floor is the Euler headroom")
    print(f"manifest sizes {[len(manifests[N]) for N in steps_list]} -> "
          f"{out / 'manifest.json'}")

    # --- trajectories, per field per N ------------------------------------
    traj = {}
    for zid in by_id:
        for name in args.fields:
            for N in steps_list:
                traj[(name, N, zid)] = integrate(fields[(name, zid)], zT[zid],
                                                 SIGMA_MAX, SIGMA_MIN, N)[0]
        print(f"  fine trajectories: item {zid} done")

    # --- H0 self-check: E_N stability (rho_N) and endpoint (d_N) ----------
    selfcheck = []
    endpoint = []
    for name in args.fields:
        for N2 in steps_list[1:]:
            N1 = steps_list[steps_list.index(N2) - 1]
            for zid in by_id:
                d = _rel(traj[(name, N2, zid)][-1], traj[(name, N1, zid)][-1])
                endpoint.append(dict(field=name, N_from=N1, N_to=N2, zT=zid,
                                     d_endpoint=d))
        for d in args.deltas:
            row = dict(field=name, delta_req=d, cells=[])
            prev = None
            for N in steps_list:
                sel = [e for e in manifests[N] if e["delta_req"] == d]
                if not sel:
                    continue
                errs = [comp_error(
                    jump(fields[(name, e["zT"])],
                         traj[(name, N, e["zT"])][e["i"]], e["t_i"], e["t_j"]),
                    traj[(name, N, e["zT"])][e["j"]]) for e in sel]
                mean = float(np.mean(errs))
                row["cells"].append(dict(
                    N=N, delta=float(sel[0]["delta"]), E=mean,
                    rho=(abs(mean - prev) / max(abs(mean), 1e-9))
                    if prev is not None else None))
                prev = mean
            selfcheck.append(row)
    (out / "selfcheck.json").write_text(json.dumps(
        dict(E_N=selfcheck, endpoint=endpoint), indent=1))

    print("\nH0-a  E_N must STABILISE (rho -> small), it must NOT vanish")
    print(f"  {'field':<9}{'delta':>7}" +
          "".join(f"{'N=' + str(n):>18}" for n in steps_list))
    for row in selfcheck:
        cells = "".join(
            f"{c['E']:>11.4f}" + (f"({c['rho']:.3f})" if c["rho"] is not None
                                  else "  (  -  )") for c in row["cells"])
        print(f"  {row['field']:<9}{row['delta_req']:>7.3f}{cells}")
    print("  (cell = E_N(rho_N) with rho_N = |E_N - E_prev| / E_N)")
    print("\nH0-b  trajectory endpoint convergence  d_N = "
          "||z_end(N2) - z_end(N1)|| / ||z_end(N2)||")
    for name in args.fields:
        for N2 in steps_list[1:]:
            vals = [e["d_endpoint"] for e in endpoint
                    if e["field"] == name and e["N_to"] == N2]
            print(f"  {name:<9} N {steps_list[steps_list.index(N2) - 1]}"
                  f" -> {N2}:  mean {np.mean(vals):.4f}  max {np.max(vals):.4f}")

    # --- L1: composition error and the Euler headroom ---------------------
    N = n_fine
    rows = []
    for e in manifests[N]:
        rec = dict(zT=e["zT"], i=e["i"], j=e["j"], t_i=e["t_i"], t_j=e["t_j"],
                   delta=e["delta"], delta_req=e["delta_req"])
        for name in args.fields:
            st = traj[(name, N, e["zT"])]
            z_i, z_j = st[e["i"]], st[e["j"]]
            j_out = jump(fields[(name, e["zT"])], z_i, e["t_i"], e["t_j"])
            k = key_of(name)
            rec[f"e_{k}"] = comp_error(j_out, z_j)
            rec[f"_ref_{k}"] = z_j
            rec[f"_jmp_{k}"] = j_out
        rows.append(rec)

    print(f"\nH0' L1  composition error on the finest grid (N={N})")
    print(f"  {'delta':>7}{'E_adapter':>12}{'E_frozen':>12}{'R_comp':>9}"
          f"{'n':>4}{'adv<frz':>9}{'med_a':>9}{'med_f':>9}")
    l1 = []
    for d in args.deltas:
        sel = [r for r in rows if r["delta_req"] == d]
        if not sel:
            continue
        ea = summarise([r.get("e_adapter") for r in sel])
        ef = summarise([r.get("e_frozen") for r in sel])
        rc = (1.0 - ea["mean"] / ef["mean"]) if (ea.get("n") and ef.get("n")
                                                 and ef["mean"] > 0) else None
        frac = (float(np.mean([r["e_adapter"] < r["e_frozen"] for r in sel]))
                if ea.get("n") and ef.get("n") else None)
        l1.append(dict(delta_req=d, delta=sel[0]["delta"], E_adapter=ea,
                       E_frozen=ef, R_comp=rc, frac_adapter_better=frac))
        print(f"  {d:>7.3f}{ea.get('mean', float('nan')):>12.4f}"
              f"{ef.get('mean', float('nan')):>12.4f}"
              f"{(rc if rc is not None else float('nan')):>9.3f}"
              f"{ea.get('n', 0):>4}"
              f"{(frac if frac is not None else float('nan')):>9.2f}"
              f"{ea.get('median', float('nan')):>9.4f}"
              f"{ef.get('median', float('nan')):>9.4f}")
    print("  R_comp = fraction of the Euler headroom the learned field explains"
          " (1 perfect flow map, 0 no better than constant, <0 harmful)")
    print("  'adv<frz' is the paired per-item fraction; 'systematic' means a"
          " consistent sign across items AND deltas, not one lucky cell")

    # --- G1: the correction-scale curve ----------------------------------
    # The L1 readout above is deliberately UNSCALED, so the H0' control (does
    # this checkpoint reproduce fd_s5000?) stays interpretable. This is the
    # separate question behind branch A/C: is the composition deficit a
    # MAGNITUDE error that one scalar could remove, or not?
    e_frozen_d = {r["delta_req"]: r["E_frozen"]["mean"] for r in l1}
    e_adapter_d = {r["delta_req"]: r["E_adapter"]["mean"] for r in l1}
    a_specs = [(f"a={a:g}", const_scale(a))
               for a in (args.corr_scale_sweep or [])]
    scale_fn = make_scale_fn(args.corr_scale)
    if scale_fn is not None:
        a_specs.append((f"a(t)={args.corr_scale}", scale_fn))
    a_curve = []
    if a_specs:
        print("\nG1  correction-scale curve   u_a = v_inst + a(t)*(u - v_inst)")
        print(f"  {'a':<28}" + "".join(f"{'d=' + f'{d:g}':>13}"
                                       for d in args.deltas))
        for label, a_of_t in a_specs:
            fld = {zid: scaled_field(fields[("adapter", zid)],
                                     base_of("adapter", zid), a_of_t)
                   for zid in by_id}
            st = {zid: integrate(fld[zid], zT[zid], SIGMA_MAX, SIGMA_MIN,
                                 N)[0] for zid in by_id}
            cell = {}
            for d in args.deltas:
                sel = [e for e in manifests[N] if e["delta_req"] == d]
                if not sel:
                    continue
                errs = [comp_error(
                    jump(fld[e["zT"]], st[e["zT"]][e["i"]], e["t_i"], e["t_j"]),
                    st[e["zT"]][e["j"]]) for e in sel]
                m = float(np.mean(errs))
                cell[d] = dict(
                    E=m,
                    R=(1.0 - m / e_frozen_d[d]) if e_frozen_d.get(d) else None,
                    frac_better_frozen=float(np.mean(
                        [comp_error(jump(fld[e["zT"]], st[e["zT"]][e["i"]],
                                         e["t_i"], e["t_j"]), st[e["zT"]][e["j"]])
                         < comp_error(jump(fields[("teacher", e["zT"])],
                                           traj[("teacher", N, e["zT"])][e["i"]],
                                           e["t_i"], e["t_j"]),
                                      traj[("teacher", N, e["zT"])][e["j"]])
                         for e in sel])) if "teacher" in args.fields else None,
                )
            a_curve.append(dict(label=label, cells=cell))
            print(f"  {label:<28}" + "".join(
                f"{cell[d]['E']:>13.4f}" if d in cell else f"{'-':>13}"
                for d in args.deltas))
        # WIRING SELF-CHECK, computed rather than asserted: a=0 must reproduce
        # E_frozen and a=1 must reproduce the unscaled E_adapter. If either
        # fails the scaling is not attached where it claims to be and the whole
        # curve is void -- the same class of check as the f2/f3 anchor guards.
        dev = []
        for rec in a_curve:
            tgt = (e_frozen_d if rec["label"] == "a=0"
                   else e_adapter_d if rec["label"] == "a=1" else None)
            if tgt:
                dev += [abs(rec["cells"][d]["E"] - tgt[d])
                        for d in rec["cells"] if d in tgt]
        if dev:
            worst = max(dev)
            print(f"  wiring self-check vs E_frozen (a=0) and E_adapter (a=1): "
                  f"max |dE| = {worst:.3e} "
                  f"{'OK' if worst < 1e-9 else '<-- FAILED, CURVE IS VOID'}")
        else:
            print("  NOTE: no a=0 / a=1 in the sweep, so the endpoints could "
                  "not be self-checked. Pass `--corr-scale-sweep 0 0.5 1`.")

    # --- G4: the states gap between instrument A and instrument B --------
    # `cos` is measured on the interpolation line (1-t)x0 + t*eps. E_comp runs
    # on the field's OWN trajectory. This asks whether the FIELD differs
    # between those two STATE SETS at the SAME (t, r) cells, with both arms
    # using an identical reference construction (the field integrated |i-j|
    # steps from the queried state), so the only thing that changes is where
    # the query starts.
    #
    # IT NEEDS NO CORRESPONDENCE between zT and z_tgt. Three earlier versions of
    # this test are recorded here so they are not re-proposed: (1) a
    # FROZEN-field tangent compared against the true one -- the Stage 1C proxy
    # error; (2) a hard-coded field name, so both arms read the adapter and
    # matched to four decimals; (3) `||z_t - line(t)||` with a RANDOM zT, which
    # asks whether a sample equals one particular target -- true by
    # construction, and therefore empty.
    sg = None
    if args.states_gap:
        print("\nG4  states gap: the SAME (t,r) cells queried from the field's "
              "OWN trajectory states vs from the interpolation line "
              f"(first {args.states_gap_n} entries per delta)")
        if not all("z_tgt" in by_id[zid] for zid in by_id):
            print("  SKIPPED: these cache items carry no z_tgt.")
        else:
            # A SEPARATE generator. Drawing from `gen` here would shift zT and
            # silently change every L1/L2/L3 number printed above.
            gsg = torch.Generator(device="cpu").manual_seed(args.seed + 7)
            x0_by, eps_by = {}, {}
            for zid in by_id:
                x0 = by_id[zid]["z_tgt"].to(zT[zid].device).float()
                x0_by[zid] = x0
                eps_by[zid] = torch.randn(x0.shape, generator=gsg).to(
                    x0.device, x0.dtype)
            traj_by = {(r["zT"], r["i"], r["j"]): r for r in rows}
            sg = {}
            for d in args.deltas:
                sel = [e for e in manifests[N]
                       if e["delta_req"] == d][:args.states_gap_n]
                if not sel:
                    continue
                acc, nfe = {}, 0
                for name in args.fields:
                    k = key_of(name)
                    for e in sel:
                        zid = e["zT"]
                        nsub = max(1, abs(e["i"] - e["j"]))
                        z0 = data_line(e["t_i"], x0_by[zid], eps_by[zid])
                        _, z_ref = integrate(fields[(name, zid)], z0,
                                             e["t_i"], e["t_j"], nsub)
                        nfe += nsub
                        val = comp_error(
                            jump(fields[(name, zid)], z0, e["t_i"], e["t_j"]),
                            z_ref)
                        acc.setdefault(k, []).append(
                            (val, traj_by[(zid, e["i"], e["j"])][f"e_{k}"]))
                cell = {}
                for k, pr in acc.items():
                    e_line = float(np.mean([p[0] for p in pr]))
                    e_tr = float(np.mean([p[1] for p in pr]))
                    cell[k] = dict(
                        E_line=e_line, E_traj=e_tr,
                        ratio=(e_line / e_tr if e_tr > 0 else None),
                        line_worse=float(np.mean([a > b for a, b in pr])))
                sg[d] = cell
                txt = "".join(
                    f"  {kk}: traj {c['E_traj']:.4f} line {c['E_line']:.4f} "
                    f"({c['ratio']:.2f}x, worse {c['line_worse']:.2f})"
                    for kk, c in sorted(cell.items()))
                print(f"  d={d:<5}{txt}  [{nfe} extra NFE]")
            print("  READ: ratio ~1 with line_worse ~0.5 means the two state "
                  "sets are interchangeable and gap #4 is empty. ratio << 1 "
                  "means the LINE states are easier, so E_comp (trajectory "
                  "states) UNDERSTATES the field that `cos` describes.")

    # --- L2: trajectory drift --------------------------------------------
    l2 = None
    if "adapter" in args.fields and "teacher" in args.fields:
        print("\nH0' L2  trajectory drift  D_traj(t) (diagnostic, not a metric)")
        per_item = []
        for zid in by_id:
            a = traj[("adapter", N, zid)]
            t = traj[("teacher", N, zid)]
            dmax = max(_rel(a[k], t[k]) for k in range(N + 1))
            per_item.append(dmax)
            print(f"  zT {zid:<3} max_t D_traj = {dmax:.4f}")
        l2 = dict(max_drift=float(np.max(per_item)),
                  mean_drift=float(np.mean(per_item)))

    # --- L3: shared-state probe (triggered) ------------------------------
    l3 = None
    if args.probe_states > 0 or (l2 is not None
                                 and l2["max_drift"] > args.probe_drift):
        k_states = args.probe_states or 2
        why = ("requested" if args.probe_states
               else f"auto: L2 drift {l2['max_drift']:.3f} > {args.probe_drift}")
        print(f"\nH0' L3  shared-state probe ({why}, {k_states} states): "
              f"both fields fine-resolved FROM THE SAME state, then each "
              f"field's own one-jump vs its own reference")
        probe = []
        for e in manifests[N][:k_states]:
            for name in args.fields:
                z_i = traj[(name, N, e["zT"])][e["i"]]
                # The probe's reference must be MUCH FINER than the jump, or it
                # is a different (coarser) reference and contradicts the main
                # measurement -- observed once at max(8, ...): 8 steps for a
                # 28-step-wide jump produced a SMALLER error than the fine
                # grid, i.e. the coarse reference sat accidentally closer.
                n_probe = max(16, 8 * abs(e["i"] - e["j"]))
                _st, z_ref = integrate(fields[(name, e["zT"])], z_i,
                                       e["t_i"], e["t_j"], n_probe)
                val = comp_error(jump(fields[(name, e["zT"])], z_i, e["t_i"],
                                      e["t_j"]), z_ref)
                probe.append(dict(zT=e["zT"], field=name, delta=e["delta"],
                                  E_shared=val))
                print(f"  zT {e['zT']:<3} {name:<8} delta={e['delta']:.3f}"
                      f"  E_shared={val:.4f}")
        l3 = probe

    # --- H6: image-space metrics on a few probed pairs --------------------
    img_rows = []
    if args.image_metrics > 0:
        from diffusers import StableDiffusion3Pipeline

        from bench.inversion import SD3EditPipeline
        from bench.metrics import masked_mse, masked_psnr
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(args.device)
        helper = SD3EditPipeline(pipe, device=args.device)
        ssim, lp = ssim_callable(), lpips_callable(args.device)
        print(f"\nH6  image-space composition error, whole frame and inside the"
              f" (upsampled) edit mask   [MSE/PSNR"
              f"{'+SSIM' if ssim else ''}{'+LPIPS' if lp else ''}]")
        if not ssim:
            print("  (no SSIM: scikit-image absent)")
        if not lp:
            print("  (no LPIPS: `pip install lpips` to enable)")
        for d in args.deltas:
            sel = [r for r in rows if r["delta_req"] == d][:args.image_metrics]
            for r in sel:
                mk = by_id[r["zT"]]["mask"][0, 0].cpu().numpy() > 0.5
                mk = np.repeat(np.repeat(mk, MASK_UP, 0), MASK_UP, 1)
                for name in args.fields:
                    k = key_of(name)
                    a = helper.decode_latent(r[f"_jmp_{k}"]).astype(np.float32) / 255.0
                    b = helper.decode_latent(r[f"_ref_{k}"]).astype(np.float32) / 255.0
                    row = dict(zT=r["zT"], field=name, delta_req=d,
                               mse=float(((a - b) ** 2).mean()),
                               psnr=masked_psnr(a, b, mk, inside=False),
                               mse_in=masked_mse(a, b, mk, inside=True),
                               psnr_in=masked_psnr(a, b, mk, inside=True))
                    if ssim:
                        row["ssim"] = ssim(a, b)
                    if lp:
                        row["lpips"] = lp(a, b)
                    img_rows.append(row)
                    print(f"  d={d:<5} zT={r['zT']:<3} {name:<8}"
                          f" mse={row['mse']:.5f} psnr={row['psnr']:.2f}"
                          f" | in-mask mse={row['mse_in']:.5f}"
                          f" psnr={row['psnr_in']:.2f}"
                          + (f" ssim={row['ssim']:.4f}" if ssim else "")
                          + (f" lpips={row['lpips']:.4f}" if lp else ""))

    (out / "rows.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        indent=1))
    (out / "summary.json").write_text(json.dumps(
        dict(manifest_sizes={str(k): len(v) for k, v in manifests.items()},
             fine_steps=steps_list, E_N=selfcheck, endpoint=endpoint,
             L1=l1, L2=l2, L3=l3, a_curve=a_curve, states_gap=sg,
             image=img_rows), indent=1))

    print("\nREAD (pre-registered; the scaling claims are hypotheses, not"
          " theorems -- see H0_H0PRIME_SPEC.md)")
    print("  1 H0 first: every E_N cell stabilises (rho small) AND the endpoint"
          " d_N is small. If not, the reference is not converged and nothing"
          " below is readable.")
    print("  2 H0' uses R_comp with NO threshold: ~0 or <0 across deltas with"
          " the paired per-item fraction near 0.5 means the interval field"
          " produced no systematic composition advantage over the frozen field"
          " -- a reportable negative, not a tuning signal.")
    print("  3 small deltas have little headroom, so R_comp ~ 0 there is"
          " EXPECTED; the few-step hypothesis is read at mid/large delta.")
    print(f"\nwritten: {out}/manifest.json, selfcheck.json, rows.json,"
          f" summary.json")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        raise SystemExit(0)
    raise SystemExit(main())