"""E0/E1: can Gate B's flow map carry Gate A's edit? The A+B projection test.

WHY THIS IS THE NEXT EXPERIMENT.

Gate A is closed for a reason that is now measured: a deterministic latent
displacement regressor returns the hedged conditional mean -- amplitude at the
MSE optimum (`amp_vs_optimal ~ 1`), direction aligned only 0.54 raw, and the
MISALIGNMENT IS CONCENTRATED IN THE HIGH SPATIAL FREQUENCIES (GateA.md 7:
low-pass sigma=16 raises the mask cosine 0.534 -> 0.778 while cos_hp stays flat
at 0.516). Low frequencies right, high frequencies in the wrong place: that is
what decodes as a smear.

Gate B is a generative flow map over `z_tgt`. A generative flow map cannot
average two modes -- it has to land on one. So the combination to test is not
"train something new", it is:

    A proposes the edit direction (low frequencies: the part it gets right),
    B projects that off-manifold point back onto the data manifold,
    B's prior resupplies the high frequencies A cannot place.

That is the non-linear decode of option C, built out of two checkpoints that
already exist. It needs NO training, and it is the cheapest experiment that can
falsify the whole route.

TWO ARMS, AND THE ORDER MATTERS.

  E0  ONE-CALL CONSISTENCY ORACLE. Project z_tgt itself:
          z_t = (1-t) z_tgt + t*eps,   z_out = z_t - t*u_B(z_t, 0, t)
      A mean-flow model claims that single call IS the flow map over [0, t].
      If z_out does not come back to z_tgt, the flow map is inconsistent over
      Delta = t and NOTHING built on top of it can be read. Run this first; it
      is minutes and it can kill the plan. Gate B's field was only ever scored
      per-cell (`eval_interval_field.py`); its one-call consistency has never
      been measured, and this session's 20-step sampler collapse (Delta = 0.05
      per step) is a warning that the composed map is not free.
      Same call on z_src measures the projection's error on the source side.

  E1  THE COMBINATION. Project z_A = z_src + u_A (Gate A's one-step readout):
          z_out(t) = project(z_A, t)
      and score it against the two oracles above *and* against pure A.

WHAT CAME BACK (first round, 32 items, fd_distill_5k -- record, do not re-run)

  E0 oracle, unclamped, one call: rel_tgt 0.113 at Delta=0.15 rising to 0.282
  at Delta=0.5 (cos 0.993 -> 0.957). The knee sweep is roughly
  rel ~ 0.019 + 0.53*Delta, so a 10% reconstruction bar needs Delta <= 0.1 --
  i.e. ten-plus steps, on the property editing actually needs.

  E1: DEAD. A_raw (Gate A's own readout, measured by the same code at the same
  coordinates) 0.650 -> A_edit 0.646 at t=0.15, 0.656 at t=0.5; cos_hp 0.550
  -> 0.557..0.541; sharpness 0.0494 -> 0.0486..0.0459. The projection is the
  identity plus the model's own error: flat in t, worse at larger t. Editing
  cannot be injected through the starting latent.

  The 2x3 ({adapter, teacher} x substeps {1,2,4}) is the interesting part:
      adapter 0.282 / 0.290 / 0.297        teacher 0.273 / 0.274 / 0.282
  BOTH get worse with finer steps, and the adapter is 3-5% worse at every
  substep count. So the residual is not Euler truncation (subdividing would
  help) and not adapter-specific. But "the field is biased" and "the unclamped
  sigma->0 end dominates, since a finer grid spends more steps down there" fit
  those three numbers equally well, and they imply opposite next moves. The
  clamp is what separates them.

PRE-REGISTERED READING FOR THE CLAMP TEST (write it down before looking):

  A. clamped rel_path falls far below unclamped rel_start (and keeps falling
     as substeps rise) -> the sigma->0 end owned the residual. The field is
     fine; every unclamped measurement in this project -- this probe AND
     invert_one_cached.py's round trip -- has to be re-read, and no retrain is
     justified yet.
  B. clamped is as flat as unclamped -> the residual is a PROPERTY OF THE
     FIELD (a Delta-proportional transport error). Rerun --field teacher: if
     the frozen field is as flat, the retrain target is the field itself, and
     the few-step claim is capped by this curve.
  C. adapter worse than teacher in BOTH regimes -> the interval conditioning
     costs reconstruction fidelity. Gate B's current form is a negative, and a
     retrain has to change the objective, not just the conditioning.

WHAT THIS DELIBERATELY AVOIDS. Inversion. Both arms start from a known latent;
no z_src inversion is involved, so none of the round-trip questions from
`invert_one_cached.py` (OOD source endpoint, step-count sensitivity) can
confound the answer. Inversion belongs to the DirectEdit variant, and it can
come back later if a clamp result revives the route.

USAGE

    # pass 1 (Gate A checkpoint): dump the one-step readout z_A
    python scripts/project_edit.py displace \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 --limit 32 \
        --out runs/ab_disp.npz

    # pass 2 (Gate B checkpoint): E0 oracles alone -- no --disp
    python scripts/project_edit.py project \
        --ckpt runs/fd_distill_5k/ckpt_final.pt \
        --cache /data/cache/13061d51aee4 --limit 32 \
        --t 0.15 0.25 0.35 0.5 --out runs/ab_proj_e0

    # E1: same command plus --disp runs/ab_disp.npz

    # THE CLAMP TEST (now the default: --sigma-min 0.05, 0 = old behaviour).
    # Run it for both fields and every substeps count -- the substeps trend is
    # where the two readings of the 2x3 table part company:
    for f in adapter teacher; do for s in 1 2 4 16; do
      python scripts/project_edit.py project \
        --ckpt runs/fd_distill_5k/ckpt_final.pt \
        --cache /data/cache/13061d51aee4 --limit 32 --field $f \
        --substeps $s --t 0.5 --out runs/oa_${f}_s${s}
    done; done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache

# Gate B's fd_distill_5k was trained with --max-delta 0.5: a single call over
# [0, t] is IN DISTRIBUTION only for t <= 0.5. Beyond that it is extrapolation
# and a failure there says nothing about the model.
MAX_DELTA = 0.5


def _ensure(t, nd):
    while t.dim() < nd:
        t = t.unsqueeze(0)
    return t


def _blur(x, sigma):
    """Separable gaussian, same convention as GateA.md 7 (mask zeroed first)."""
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), padding=(0, r),
                 groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), padding=(r, 0),
                 groups=c)
    return x


def _masked(a, mask):
    return a * mask


def _cos(a, b, mask=None):
    if mask is not None:
        a, b = _masked(a, mask), _masked(b, mask)
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _rel(a, b, mask=None):
    if mask is not None:
        a, b = _masked(a, mask), _masked(b, mask)
    return float((a - b).norm() / (b.norm() + 1e-12))


def _sharpness(img):
    """High-frequency energy of a decoded uint8 image, as a smear detector."""
    g = np.asarray(img, np.float32).mean(2) / 255.0
    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    return float(np.sqrt((gx ** 2).mean() + (gy ** 2).mean()))


def _psnr(a, b):
    a = np.asarray(a, np.float32) / 255.0
    b = np.asarray(b, np.float32) / 255.0
    m = float(((a - b) ** 2).mean())
    return float("inf") if m < 1e-12 else 10 * np.log10(1.0 / m)


@torch.no_grad()
def project(adapter, z_start, t_val, pe, pooled, eps, substeps=1,
            corr_scale=1.0, field="adapter", sigma_min=0.0):
    """(1-t) z_start + t eps  ->  sigma_min, with the mean-flow flow map.

    substeps=1 is the one-call claim this whole route rests on. substeps>1
    walks the same interval in equal pieces (the sampler's own operator), so
    the pair isolates "is the single big jump consistent" from "is the fine
    integration better".

    field="teacher" integrates the FROZEN pretrained instantaneous velocity
    instead. A single Euler call of it over the same Delta is the naive
    "1-NFE SD3" the adapter is supposed to beat; without it, an oracle error
    has no scale.

    sigma_min > 0 STOPS SHORT OF THE DATA END, and the comparison must then be
    against `path_point`, not against z_start -- see both docstrings. At t=0
    the model is asked the one question it cannot answer (v = eps - x0 from a
    clean latent; its best estimate is -x0) and the last Euler step shrinks
    the state instead of denoising it. Stage 0's clamp took the round trip
    15.83 -> 39.16 dB on its own (bench/inversion.py), and every instrument
    built since -- this one included -- silently reintroduced the unclamped
    end.
    """
    b = z_start.shape[0]
    z = (1.0 - t_val) * z_start + t_val * eps
    ts = [sigma_min + (t_val - sigma_min) * (1.0 - i / substeps)
          for i in range(substeps + 1)]
    for i in range(substeps):
        t, r = ts[i], ts[i + 1]
        tt = torch.full((b,), t, device=z.device)
        rr = torch.full((b,), r, device=z.device)
        if field == "teacher":
            u = adapter.base_velocity(z, tt, pe, pooled)
        else:
            u = adapter(z, tt, rr, pe, pooled)
            if corr_scale != 1.0:
                u0 = adapter(z, tt, tt.clone(), pe, pooled)
                u = u0 + corr_scale * (u - u0)
        z = z - (t - r) * u
    return z


def path_point(z_start, eps, s):
    """The exact point on the interpolation line `project` started from.

        z_s = (1-s) z_start + s eps

    This is not a model output and not an assumption: it is the same line the
    projection noised from, so the residual against it is the field's
    transport error over [s, t] and nothing else. That is what makes a clamped
    run readable. Comparing a latent stopped at sigma_min against z_start
    would measure the ENDPOINT OFFSET instead -- the confound Stage 0's clamp
    introduced and had to handle by comparing decoded images.
    """
    return (1.0 - s) * z_start + s * eps


#: default lp/hp split width. GateA.md 7 reports sigma=16 (lp) and the
#: sigma->inf limit (hp); the sign of the effect is what is read here, and the
#: SAME sigma must be used for A_raw and A_edit or the two rows are not
#: comparable -- which is the whole point of keeping A_raw in the table.
LP_SIGMA = 8.0


def _arm_row(k, it, name, t, z_out, lp_sigma=LP_SIGMA, z_start=None,
             z_path=None):
    """One (arm, t) measurement. Kept in one place so A_raw -- the
    UN-PROJECTED readout -- is measured by exactly the code that measures the
    projected arms. Without that row E1 has no control: a projection that does
    nothing produces the same table as a projection that works.

    Three "did it come back" numbers, on purpose:

      rel_tgt    the historical reading: unclamped output vs z_tgt. Only the
                 identity arms have a meaningful target here.
      rel_start  output (stopping at t=0) vs the point the projection STARTED
                 from. Uniform across arms, so adapter/teacher and A_edit are
                 comparable -- and paired with rel_path, it isolates the
                 endpoint.
      rel_path   output (stopping at sigma_min) vs the analytic path point at
                 sigma_min. Same start, same eps, same t, same substeps; only
                 the stopping point moves. This is the clamp test.
    """
    m = it["mask"]
    row = dict(item=k, idx=int(it["idx"]), arm=name, t=float(t),
               cos_tgt=_cos(z_out, it["z_tgt"], m),
               rel_tgt=_rel(z_out, it["z_tgt"], m),
               cos_src=_cos(z_out, it["z_src"], m),
               rel_src=_rel(z_out, it["z_src"], m))
    if z_start is not None:
        row["cos_start"] = _cos(z_out, z_start, m)
        row["rel_start"] = _rel(z_out, z_start, m)
    if z_path is not None:
        row["cos_path"] = _cos(z_out, z_path, m)
        row["rel_path"] = _rel(z_out, z_path, m)
    if name != "tgt_identity":
        d_out = z_out - it["z_src"]
        d_v = it["z_tgt"] - it["z_src"]
        lp_o = _blur(_masked(d_out, m), lp_sigma)
        lp_v = _blur(_masked(d_v, m), lp_sigma)
        row["cos_lp"] = _cos(lp_o, lp_v)
        row["cos_hp"] = _cos(d_out - _blur(d_out, lp_sigma),
                             d_v - _blur(d_v, lp_sigma), m)
        row["norm_ratio"] = float(_masked(d_out, m).norm()
                                  / (_masked(d_v, m).norm() + 1e-12))
    return row


def build_adapter(ckpt, mode, device, rank, interval_kind="centered",
                  mask_encoder_kind="single", max_blocks=None):
    from diffusers import SD3Transformer2DModel

    from transport.conditioning import (interval_kind_from_keys,
                                        make_interval_embedding)
    tr = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="transformer", torch_dtype=torch.float32).to(device)
    cfg = AdapterConfig(mode=mode, rank=rank, interval_kind=interval_kind,
                        mask_encoder_kind=mask_encoder_kind,
                        max_blocks=max_blocks)
    adapter = TransportAdapter(tr, cfg).to(device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    # THE CHECKPOINT SAYS WHICH INTERVAL EMBEDDING IT OWNS, and the two runs
    # disagree: Gate A's pair checkpoint is `concat` (interval.net.*), Gate B's
    # distill checkpoint is `centered` (interval.t_branch/delta_branch). Load
    # one into the other's module and every interval key is "unexpected"; with
    # strict=False alone the interval path would stay zero-initialised, i.e. the
    # r-dependence silently disappears -- the failure eval_interval_field.py
    # already refuses to tolerate. Rebuild the module instead of guessing from
    # a flag.
    kind = interval_kind_from_keys(sd["adapter"].keys())
    if kind is None:
        if mode == "meanflow":
            raise SystemExit(
                f"{ckpt}: no interval weights in the checkpoint; a meanflow "
                f"adapter without them is not the model that was trained")
        adapter.interval = None      # pair mode never needed it
        print("no interval weights in checkpoint: interval path disabled")
    elif kind != adapter.cfg.interval_kind:
        adapter.interval = make_interval_embedding(
            kind, int(tr.config.pooled_projection_dim)).to(device)
        adapter.cfg.interval_kind = kind
        print(f"interval embedding rebuilt as {kind!r} (checkpoint keys)")

    missing, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        hint = ""
        if any("global_bias" in k or ".local." in k for k in unexpected):
            hint = ("\n  the checkpoint carries global_local mask weights: "
                    "re-run with --mask-encoder-kind global_local")
        elif any("mask_enc." in k for k in unexpected):
            hint = "\n  mask_enc shape mismatch: check --mask-encoder-kind"
        raise SystemExit(f"unexpected keys: {unexpected[:5]}{hint}")
    lost = [k for k in missing if k.startswith("interval.")]
    if lost:
        raise SystemExit(f"interval weights absent from checkpoint: {lost[:4]}")
    adapter.eval()
    print(f"loaded {ckpt}  step {sd.get('step', '?')}  mode={mode}")
    return adapter


def read_cache(cache, limit, device):
    n = min(limit, len(cache))
    items = []
    for i in range(n):
        rec = cache[i]
        mask = _ensure(torch.from_numpy(rec["mask"]).float(), 4)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        items.append(dict(
            idx=i, id=rec["meta"].get("id"),
            instruction=rec["meta"].get("instruction"),
            z_src=_ensure(torch.from_numpy(rec["z_src"]).float(), 4).to(device),
            z_tgt=_ensure(torch.from_numpy(rec["z_tgt"]).float(), 4).to(device),
            mask=mask.to(device),
            pe=_ensure(torch.from_numpy(rec["prompt_embeds"]), 3).to(device),
            pooled=_ensure(torch.from_numpy(rec["pooled"]), 2).to(device),
        ))
    return items


# --------------------------------------------------------------- subcommands --

def cmd_displace(args) -> int:
    """Gate A pass: z_A = z_src + u_A(z_src, s=0)."""
    adapter = build_adapter(args.ckpt, "pair", args.device, args.rank,
                            mask_encoder_kind=args.mask_encoder_kind)
    cache = TripletCache(args.cache)
    items = read_cache(cache, args.limit, args.device)
    out = {}
    z_a = []
    with torch.no_grad():
        for it in items:
            s0 = torch.zeros(1, device=args.device)
            u = adapter(it["z_src"], s0, s0, it["pe"], it["pooled"],
                        mask=it["mask"])
            z_a.append((it["z_src"] + u)[0].float().cpu().numpy())
    for key in ("z_src", "z_tgt", "mask"):
        out[key] = np.stack([it[key][0].float().cpu().numpy() for it in items])
    out["z_A"] = np.stack(z_a)
    out["id"] = np.array([str(it["id"]) for it in items])
    out["instruction"] = np.array([str(it["instruction"]) for it in items])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out}  ({len(items)} items)")
    return 0


def report(rows, ts, has_disp, sigma_min=0.0, substeps=1):
    """Print the arm x t table and the pre-registered read.

    A PURE function of the row list, deliberately: the first version printed
    inline and crashed on the E0-only path (no --disp), because the A arms do
    not exist there and the read subtracted None. Every quantity below is
    optional, and "the A arms are absent" is itself a reportable state -- the
    oracle-only run is the cheap kill test, not a misuse.
    """
    keys = ("cos_tgt", "rel_tgt", "cos_src", "rel_src", "cos_start",
            "rel_start", "cos_path", "rel_path", "cos_lp", "cos_hp",
            "norm_ratio", "sharpness", "psnr_tgt")

    def agg(arm, t):
        v = [r for r in rows if r["arm"] == arm and r["t"] == t]
        if not v:
            return None
        return {k: float(np.mean([r[k] for r in v if k in r]))
                for k in keys if any(k in r for r in v)}

    def one(arm, t, key):
        a = agg(arm, t)
        return None if a is None else a.get(key)

    def f(x, spec=">9.3f"):
        return "      n/a" if x is None else format(x, spec)

    print(f"\n{'arm':<14}{'t':>6}{'cos_tgt':>9}{'rel_tgt':>9}{'rel_strt':>10}"
          f"{'rel_path':>10}{'cos_src':>9}{'cos_lp':>8}{'cos_hp':>8}"
          f"{'norm_r':>8}{'sharp':>8}")
    for name in ("tgt_identity", "src_identity", "A_raw", "A_edit"):
        for t in ([-1.0] if name == "A_raw" else ts):
            a = agg(name, t)
            if a is None:
                continue
            tstr = "   raw" if t < 0 else f"{t:>6.2f}"
            print(f"{name:<14}{tstr}{f(a.get('cos_tgt'))}"
                  f"{f(a.get('rel_tgt'))}{f(a.get('rel_start'), '>10.3f')}"
                  f"{f(a.get('rel_path'), '>10.3f')}"
                  f"{f(a.get('cos_src'))}"
                  f"{f(a.get('cos_lp'), '>8.3f')}"
                  f"{f(a.get('cos_hp'), '>8.3f')}"
                  f"{f(a.get('norm_ratio'), '>8.3f')}"
                  f"{f(a.get('sharpness'), '>8.4f')}")

    # ------------------------------------------------- pre-registered read --
    print("\nREAD (pre-registered, in this order):")
    o_hi, o_lo = one("tgt_identity", ts[-1], "rel_tgt"), \
        one("tgt_identity", ts[0], "rel_tgt")
    o_src = one("src_identity", ts[-1], "rel_tgt")
    a_raw, a_ed = one("A_raw", -1.0, "rel_tgt"), one("A_edit", ts[0], "rel_tgt")
    print(f"  1 ORACLE FLOOR: rel_tgt {f(o_lo)} at t={ts[0]:g}  ->  "
          f"{f(o_hi)} at t={ts[-1]:g}   (unclamped: includes the sigma->0 end)")
    if o_src is not None and o_hi is not None:
        print(f"     src->tgt separation {o_src - o_hi:+.3f}: the projection's "
              f"own error must be well below it, or nothing else is readable")

    # THE CLAMP TEST: rel_start (stop at 0) vs rel_path (stop at sigma_min),
    # same start/eps/t/substeps. Paired by construction.
    clamped = [(n, t, one(n, t, "rel_start"), one(n, t, "rel_path"))
               for n in ("tgt_identity", "src_identity", "A_edit")
               for t in ts]
    clamped = [c for c in clamped if c[2] is not None and c[3] is not None]
    if clamped:
        print(f"  2 CLAMP TEST (sigma_min={sigma_min:g}): unclamped "
              f"rel_start -> clamped rel_path, same everything else")
        for n, t, rs, rp in clamped:
            print(f"       {n:<14} t={t:<5g} {rs:8.3f} -> {rp:8.3f}   "
                  f"({(rs - rp) / max(rs, 1e-9) * 100:+6.1f}% )")
        worst = max(clamped, key=lambda c: c[2] - c[3])
        gain = worst[2] - worst[3]
        if gain > 0.3 * worst[2]:
            print("     => the sigma->0 end owns most of the residual. The "
                  "field is fine; every unclamped measurement this session "
                  "(round trip, projection) has to be re-read, and no "
                  "retrain is justified yet.")
        elif gain > 0.05:
            print("     => the clamp removes part of it. Report both numbers; "
                  "the remainder is still the field's.")
        else:
            print("     => the clamp changes nothing. The residual is a "
                  "PROPERTY OF THE FIELD (a Delta-proportional transport "
                  "error), not of the endpoint -- rerun with --field teacher: "
                  "if the frozen field is as flat, the retrain target is the "
                  "field itself.")
        print(f"     (substeps={substeps}: the substeps trend needs the same "
              f"command at 1/2/4/16 -- a clamp that removes the endpoint error "
              f"should now IMPROVE with finer steps, which the unclamped run "
              f"did not.)")

    if a_raw is None and a_ed is None:
        why = ("this run is the E0 oracle only (no --disp)" if not has_disp
               else "A arms missing even though --disp was passed: check that "
                    "the displace dump matches this --cache and --limit")
        print(f"  2/3/4 A arms absent -- {why}. Run `displace` and pass "
              f"--disp to get the A_raw control row and the A_edit arms; the "
              f"oracle above stands on its own.")
        return

    if a_raw is not None:
        print(f"  2 CONTROL (never omit this again): A_raw un-projected "
              f"rel_tgt {f(a_raw)}, cos_hp "
              f"{f(one('A_raw', -1.0, 'cos_hp'), '>8.3f')}, sharpness "
              f"{f(one('A_raw', -1.0, 'sharpness'), '>8.4f')}")
    if a_raw is not None and a_ed is not None:
        print(f"  3 DID THE PROJECTION DO ANYTHING? rel_tgt {f(a_raw)} -> "
              f"{f(a_ed)} ({a_raw - a_ed:+.3f}) at t={ts[0]:g}; cos_hp "
              f"{f(one('A_raw', -1.0, 'cos_hp'), '>8.3f')} -> "
              f"{f(one('A_edit', ts[0], 'cos_hp'), '>8.3f')}; sharpness "
              f"{f(one('A_raw', -1.0, 'sharpness'), '>8.4f')} -> "
              f"{f(one('A_edit', ts[0], 'sharpness'), '>8.4f')}")
        if o_src is not None:
            print(f"     edit signal available = {o_src - a_raw:+.3f} "
                  f"(src_identity {f(o_src)} - A_raw {f(a_raw)})")
    av = [v for v in (one("A_edit", t, "rel_tgt") for t in ts) if v is not None]
    if len(av) > 1:
        print(f"  4 t-INVARIANCE: A_edit rel_tgt spans {min(av):.3f}-"
              f"{max(av):.3f} across t={ts[0]:g}..{ts[-1]:g}. A working "
              f"projector must depend on how much noise you add; flatness "
              f"means it is the identity plus the model's own error")
    print("  5 the lp/hp split is only comparable WITHIN this table (one "
          "--lp-sigma on every arm); GateA.md 7 used sigma=16 / the limit")


def cmd_project(args) -> int:
    adapter = build_adapter(args.ckpt, "meanflow", args.device, args.rank)
    from diffusers import StableDiffusion3Pipeline
    from bench.inversion import SD3EditPipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)

    cache = TripletCache(args.cache)
    items = read_cache(cache, args.limit, args.device)

    disp = dict(np.load(args.disp)) if args.disp else None
    if disp is not None and len(disp["z_A"]) < len(items):
        raise SystemExit(f"--disp has {len(disp['z_A'])} items, cache read "
                         f"{len(items)}; re-run displace with the same --limit")

    ts = [float(x) for x in args.t]
    over = [t for t in ts if t > MAX_DELTA + 1e-9]
    if over:
        print(f"WARNING: t={over} exceed the trained --max-delta "
              f"{MAX_DELTA}; those cells are EXTRAPOLATION and cannot be read "
              f"as evidence about the model")
    print(f"\nitems {len(items)}   t={ts}   substeps={args.substeps}   "
          f"corr_scale={args.corr_scale}   field={args.field}   "
          f"sigma_min={args.sigma_min}")
    if args.sigma_min > 0:
        print("  rel_tgt/cos_tgt = projection stops at t=0 (unclamped, the "
              "historical semantics)\n"
              "  rel_path/cos_path = projection stops at sigma_min, compared "
              "against the analytic path point (the transport over "
              "[sigma_min, t] only)")

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    rows = []
    sheets = []
    for k, it in enumerate(items):
        # ONE noise per item, reused by every arm and every t: the arms then
        # differ only by the starting latent, which is what makes the oracle
        # comparison controlled rather than two independent draws.
        eps = torch.randn(it["z_src"].shape, device=args.device,
                          generator=g)
        variants = {"src_identity": it["z_src"], "tgt_identity": it["z_tgt"]}
        if disp is not None:
            za = torch.from_numpy(disp["z_A"][k]).float()[None].to(args.device)
            variants["A_edit"] = za
            # THE CONTROL ROW. Gate A's own output, measured by the same code
            # and at the same coordinates as the projected arms. t = -1 marks
            # "no projection" (there is nothing to sweep). Without it a
            # projection that does nothing is indistinguishable from one that
            # works.
            rows.append(_arm_row(k, it, "A_raw", -1.0, za, args.lp_sigma))
            if k < args.sheets:
                img = helper.decode_latent(za)
                rows[-1]["sharpness"] = _sharpness(img)
                rows[-1]["psnr_tgt"] = _psnr(img, helper.decode_latent(
                    it["z_tgt"]))
                sheets.append((k, "A_raw", -1.0, img, za))
        for name, start in variants.items():
            for t in ts:
                z_out = project(adapter, start, t, it["pe"], it["pooled"], eps,
                                substeps=args.substeps,
                                corr_scale=args.corr_scale, field=args.field)
                row = _arm_row(k, it, name, t, z_out, args.lp_sigma,
                               z_start=start)
                if args.sigma_min > 0:
                    z_cl = project(adapter, start, t, it["pe"], it["pooled"],
                                   eps, substeps=args.substeps,
                                   corr_scale=args.corr_scale,
                                   field=args.field,
                                   sigma_min=args.sigma_min)
                    # same start / eps / t / substeps, different stop point
                    zp = path_point(start, eps, args.sigma_min)
                    row["cos_path"] = _cos(z_cl, zp, it["mask"])
                    row["rel_path"] = _rel(z_cl, zp, it["mask"])
                if k < args.sheets:
                    img = helper.decode_latent(z_out)
                    row["sharpness"] = _sharpness(img)
                    row["psnr_tgt"] = _psnr(img, helper.decode_latent(
                        it["z_tgt"]))
                    sheets.append((k, name, t, img, z_out))
                rows.append(row)
        if (k + 1) % 8 == 0:
            print(f"  {k + 1}/{len(items)} items")

    # ------------------------------------------------------------- report --
    report(rows, ts, disp is not None, args.sigma_min, args.substeps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(out / "rows.npz",
                        **{k: np.array([r.get(k, np.nan) for r in rows])
                           for k in ("item", "idx", "t", "cos_tgt", "rel_tgt",
                                     "cos_src", "rel_src", "cos_start",
                                     "rel_start", "cos_path", "rel_path",
                                     "cos_lp", "cos_hp", "norm_ratio",
                                     "sharpness", "psnr_tgt")},
                        arm=np.array([r["arm"] for r in rows]))
    _contact_sheets(sheets, out, args.sheets)
    print(f"\nwritten: {out / 'rows.json'}, {out / 'rows.npz'}, sheets in {out}")
    return 0


def _contact_sheets(sheets, out, n_items):
    from PIL import Image
    for k in range(n_items):
        row_imgs = [v for v in sheets if v[0] == k]
        if not row_imgs:
            continue
        ref = np.asarray(Image.fromarray(row_imgs[0][3]))
        panel = [ref]
        labels = ["ref"]
        for _k, name, t, img, _z in row_imgs:
            panel.append(np.asarray(Image.fromarray(img)))
            labels.append(f"{name} t={t:g}" if t >= 0 else f"{name}")
        Image.fromarray(np.concatenate(panel, axis=1)).save(
            out / f"item{k:03d}.png")
    print("  contact sheets: [ref | " +
          " | ".join(f"{n}" + (f" t={t:g}" if t >= 0 else "")
                     for _k, n, t, _i, _z in sheets) + "]")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ckpt", required=True)
        p.add_argument("--cache", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, default=32)
        p.add_argument("--device", default="cuda")
        p.add_argument("--model",
                       default="stabilityai/stable-diffusion-3-medium-diffusers")
        p.add_argument("--rank", type=int, default=16)

    d = sub.add_parser("displace", help="Gate A pass: dump z_A = z_src + u_A")
    common(d)
    d.add_argument("--mask-encoder-kind", default="single",
                   choices=("single", "global_local"))
    d.set_defaults(func=cmd_displace)

    p = sub.add_parser("project", help="Gate B pass: E0 oracles + E1 combination")
    common(p)
    p.set_defaults(rank=32)
    # no --interval-kind: the checkpoint's own keys decide (see build_adapter)
    p.add_argument("--disp", default=None,
                   help="npz from `displace`; without it only the E0 oracles "
                        "run (that is the cheap kill test -- run it first)")
    p.add_argument("--t", type=float, nargs="+",
                   default=[0.15, 0.25, 0.35, 0.5],
                   help=f"projection depths; > {MAX_DELTA} is extrapolation")
    p.add_argument("--substeps", type=int, default=1,
                   help="1 = the one-call flow-map claim; >1 walks the same "
                        "interval in pieces")
    p.add_argument("--corr-scale", type=float, default=1.0,
                   help="Gate B overshoots (STAGE1C 3: rho=1.58 at small "
                        "Delta); 0.5 is the STAGE1C rule")
    p.add_argument("--lp-sigma", type=float, default=LP_SIGMA,
                   help="gaussian width for the lp/hp split; the same value "
                        "is used for A_raw and A_edit (GateA.md 7 used 16)")
    p.add_argument("--sigma-min", type=float, default=0.05,
                   help="second stopping point for the projection (0 = only "
                        "the historical t=0 run). At sigma_min the comparison "
                        "moves to the analytic path point, so rel_path vs "
                        "rel_start is a paired clamp test. Stage 0 used 0.05 "
                        "and it was worth 15.83 -> 39.16 dB on the round trip")
    p.add_argument("--field", default="adapter", choices=("adapter", "teacher"),
                   help="whose field does the projecting: the adapter, or the "
                        "FROZEN pretrained one. A single Euler call of the "
                        "frozen field over the same Delta is the naive 1-NFE "
                        "baseline the adapter is supposed to beat, and without "
                        "it the oracle error has no scale.")
    p.add_argument("--sheets", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_project)

    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())