"""
Does forward-mode AD work through SD3's MMDiT, and what does it cost?

RUN THIS BEFORE ANY GATE B WORK. Stage 0's smoke test hit signal 11 on four
separate forward-mode probes and docker/triage_jvp.py narrowed the cause to
"harness or CUDA forward-AD" without closing it. MeanFlow's entire objective is
one JVP through this model, so the answer decides whether Gate B runs on
torch.func.jvp or on finite differences -- and, if neither fits in 24GB, whether
Gate B is a local job at all or an A100 job.

    python scripts/probe_jvp_mmdit.py --model stabilityai/stable-diffusion-3-medium-diffusers

Each probe runs in a FRESH SUBPROCESS with no prior CUDA use in the parent, for
the reason triage_jvp.py identified: a CUDA context inherited across fork is a
classic source of exactly this crash, and testing in-process would reproduce
the ambiguity rather than resolve it.

The probes escalate:

    1  jvp through one MMDiT block             cheapest possible real test
    2  jvp through the full transformer        the thing Gate B needs
    3  central finite difference, full model   the fallback, must always pass
    4  fd vs jvp agreement                     only if both survived

Probe 3 passing while 1 and 2 crash is a completely acceptable outcome. It
costs two extra forwards per step and Gate B proceeds unchanged; set
MeanFlowConfig(estimator="fd"). Probe 3 failing means something is wrong with
the model load, not with autodiff.

Peak memory is reported for each, because the decision is as likely to be made
on 24GB as on correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

PREAMBLE = '''
import contextlib, json, torch
from diffusers import SD3Transformer2DModel

# TF32 OFF BY DEFAULT IN THIS PROBE. TF32 keeps 10 mantissa bits, so matmuls
# run at roughly bf16 precision even when every tensor says float32. A central
# difference divides by 2*eps, which AMPLIFIES that error by 1/eps: with TF32
# on, the fd sweep followed a clean power law across five decades with no
# plateau and never came within 90% of the JVP -- it was differentiating
# matmul noise. Forward-mode AD does not divide by anything, so the JVP only
# inherits the ~1e-3 relative error and stays usable.
#
# Training may keep TF32 on (transport/train.py does). Numerical probes must
# not.
_TF32 = {tf32}
torch.backends.cuda.matmul.allow_tf32 = _TF32
torch.backends.cudnn.allow_tf32 = _TF32
if not _TF32:
    torch.set_float32_matmul_precision("highest")
MODEL = {model!r}
DEV = "cuda"
DTYPE = getattr(torch, {dtype!r})

# Local files only. The first fp32 run hit a TLS EOF mid-download and retried
# inside the probe, which turns an unrelated network flake into a probe
# failure. Set HF_HUB_OFFLINE=1 (or pass a local path) once the weights are
# cached, so a probe result is never a report on the network.
tr = SD3Transformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", torch_dtype=DTYPE).to(DEV).eval()
tr.requires_grad_(False)

# SEEDED. Each probe is a separate subprocess, so without this every probe
# draws a DIFFERENT z and their out_norms are not comparable -- which is why
# probe 2 reported 584 and probe 3 reported 188 for what looked like the same
# forward. Same seed everywhere; only within-probe comparisons were ever valid
# before this line existed.
torch.manual_seed(0)

B, C, H, W = 1, 16, 64, 64
L, D = 333, int(tr.config.joint_attention_dim)
P = int(tr.config.pooled_projection_dim)

z = torch.randn(B, C, H, W, device=DEV, dtype=DTYPE)
pe = torch.randn(B, L, D, device=DEV, dtype=DTYPE)
po = torch.randn(B, P, device=DEV, dtype=DTYPE)

# NORMALISED time, matching transport/adapter.py. The transformer wants
# sigma*1000, so the wrapper scales here and every epsilon below is in the SAME
# units MeanFlowConfig.fd_eps uses. An earlier version of this probe perturbed
# raw timestep units by 1e-3 -- 1e-6 in sigma, a thousand times smaller than
# the library default and far inside the bf16 noise floor.
T_SCALE = 1000.0
tn = torch.full((B,), 0.5, device=DEV, dtype=torch.float32)


@contextlib.contextmanager
def math_attn():
    """SDPA math backend -- the only one with a forward-AD rule.

    The fused kernels raise NotImplementedError under torch.func.jvp. The math
    path is built from ordinary differentiable ops, at the cost of
    materialising the full attention matrix.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except ImportError:
        with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False,
                enable_math=True):
            yield


CONFIG = {{"dtype": {dtype!r}, "tf32": _TF32}}


def peak():
    return torch.cuda.max_memory_allocated() / 2**30


def full(z_, tn_):
    """z, normalised t -> velocity. Cast confined to the forward, fp32 out."""
    return tr(hidden_states=z_.to(DTYPE), encoder_hidden_states=pe,
              pooled_projections=po, timestep=tn_.float() * T_SCALE,
              return_dict=False)[0].float()


# The tangent is the MODEL'S OWN VELOCITY, not a random direction. MeanFlow's
# total derivative is taken along the trajectory, so dz/dt = v(z, t). An
# earlier version used randn_like(z) -- norm ~256, a direction the model never
# travels -- which is part of why its fd sweep never converged. Defined AFTER
# `full`, which the first patched version got wrong.
with torch.no_grad():
    v = full(z, tn)
'''


PROBES = [
    ("1. jvp through ONE MMDiT block",
     "cheapest real forward-AD test. A dtype error here is NOT a crash -- "
     "forward-AD is running and tripping on bf16 weights; rerun with "
     "--dtype float32.",
     '''
blk = tr.transformer_blocks[0]
D_ = int(tr.config.num_attention_heads * tr.config.attention_head_dim)
temb = torch.randn(B, D_, device=DEV, dtype=DTYPE)
hs = torch.randn(B, (H//2)*(W//2), D_, device=DEV, dtype=DTYPE)
ehs = torch.randn(B, L, D_, device=DEV, dtype=DTYPE)

def f(h):
    out = blk(hidden_states=h, encoder_hidden_states=ehs, temb=temb)
    return out[1] if isinstance(out, tuple) else out

torch.cuda.reset_peak_memory_stats()
with math_attn():
    o, tan = torch.func.jvp(f, (hs,), (torch.randn_like(hs),))
print(json.dumps({"ok": True, "cfg": CONFIG, "peak_gb": peak(),
                  "out_norm": float(o.float().norm()),
                  "tan_norm": float(tan.float().norm())}))
'''),

    ("2. jvp through the FULL transformer",
     "exactly what MeanFlow needs. Peak memory decides local vs A100.",
     '''
torch.cuda.reset_peak_memory_stats()
with math_attn():
    o, tan = torch.func.jvp(full, (z.float(), tn), (v.float(), torch.ones_like(tn)))
print(json.dumps({"ok": True, "cfg": CONFIG, "peak_gb": peak(),
                  "out_norm": float(o.norm()), "tan_norm": float(tan.norm())}))
'''),

    ("3. finite difference EPSILON SWEEP -- the real test of the fallback",
     "A single fd number proves nothing. If the estimate scales as 1/eps it is "
     "rounding noise; a plateau across two decades is the derivative. In bf16 "
     "the model output is quantised at ~0.008 near 1.0, so small eps "
     "differentiates rounding and nothing else.",
     '''
torch.cuda.reset_peak_memory_stats()
out = []
with torch.no_grad():
    base = full(z.float(), tn)
    for eps in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        up = full(z.float() + eps*v.float(), tn + eps)
        um = full(z.float() - eps*v.float(), tn - eps)
        d = (up - um) / (2*eps)
        out.append({"eps": eps, "norm": float(d.norm())})
norms = [o["norm"] for o in out]
# noise regime: halving eps roughly doubles the estimate
ratios = [norms[i]/norms[i+1] for i in range(len(norms)-1)]
noisy = ratios[0] > 1.6
print(json.dumps({"ok": not noisy, "cfg": CONFIG, "peak_gb": peak(),
                  "out_norm": float(base.norm()),
                  "sweep": out, "small_eps_is_noise": noisy}))
'''),

    ("4. does fd CONVERGE to jvp as eps -> 0",
     "the decisive check. A single-eps disagreement means nothing on a curved "
     "function. If rel_err falls monotonically toward 0 as eps shrinks, the "
     "JVP is confirmed and fd was merely under-resolved. If rel_err flattens "
     "well above 0, one of the two is measuring the wrong thing.",
     '''
with math_attn():
    o, tan = torch.func.jvp(full, (z.float(), tn), (v.float(), torch.ones_like(tn)))
j = tan.float()
rows = []
with torch.no_grad():
    for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        up = full(z.float() + eps*v.float(), tn + eps)
        um = full(z.float() - eps*v.float(), tn - eps)
        fd = (up - um) / (2*eps)
        rows.append({"eps": eps,
                     "fd_norm": float(fd.norm()),
                     "rel_err": float((fd - j).norm() / j.norm().clamp(min=1e-12))})
best = min(r["rel_err"] for r in rows)
print(json.dumps({"ok": best < 0.1, "cfg": CONFIG, "jvp_norm": float(j.norm()),
                  "best_rel_err": best, "sweep": rows}))
'''),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--only", type=int, default=None,
                    help="run a single probe by number")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 matmuls. OFF by default: TF32 destroys "
                         "finite differences, which divide by eps. Only turn "
                         "it on to reproduce the misleading result.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="model dtype. Run BOTH bfloat16 and float32: a dtype "
                         "error under bfloat16 is a forward-AD promotion "
                         "issue, not a broken install.")
    args = ap.parse_args()

    # Identify the file that is ACTUALLY running. Stage 0 lost three days to
    # byte-identical output from stale files being read as findings; a probe
    # whose numbers do not move after an edit should be assumed stale until
    # this hash says otherwise.
    _self = Path(__file__).resolve()
    _hash = hashlib.sha1(_self.read_bytes()).hexdigest()[:10]
    print(f"probe {_self}\n  sha1={_hash}  dtype={args.dtype}  "
          f"tf32={bool(args.tf32)}")

    results = {}
    for i, (title, why, body) in enumerate(PROBES, start=1):
        if args.only and i != args.only:
            continue
        print(f"\n=== {title} ===\n    {why}")
        if i in (3, 4) and args.tf32:
            print("    (TF32 ON -- this probe's numbers are not meaningful)")
        src = (PREAMBLE.format(model=args.model, dtype=args.dtype,
                       tf32=bool(args.tf32))
               + textwrap.dedent(body))
        proc = subprocess.run([sys.executable, "-c", src],
                              capture_output=True, text=True)
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        if proc.returncode != 0:
            sig = -proc.returncode if proc.returncode < 0 else proc.returncode
            print(f"    FAIL (exit {proc.returncode}"
                  f"{', signal ' + str(sig) if proc.returncode < 0 else ''})")
            print(textwrap.indent(proc.stderr.strip()[-600:], "      "))
            results[i] = {"ok": False, "returncode": proc.returncode}
        else:
            try:
                payload = json.loads(tail)
            except Exception:                              # noqa: BLE001
                payload = {"ok": True, "raw": tail}
            ok = bool(payload.get("ok", True))
            print(f"    {'PASS' if ok else 'RAN, BUT FAILED ITS OWN CHECK'}  "
                  f"{tail}")
            results[i] = {"ok": ok, "payload": payload}

    print(f"\n--- recommendation (dtype={args.dtype}) ---")
    r3 = results.get(3, {})
    if r3.get("ok") is False and "sweep" in r3.get("payload", {}):
        print("  WARNING: the fd sweep never plateaus. Until it does, 'fd' is "
              "NOT a validated estimator on this model -- do not fall back to "
              "it just because jvp failed.")
    r4 = results.get(4, {})
    if r4 and not r4.get("ok"):
        print("  NOTE: fd never came within 10% of the JVP. Read probe 4's "
              "sweep: falling rel_err means fd was under-resolved and the JVP "
              "stands; a flat rel_err means the two disagree and neither is "
              "trusted until that is explained.")
    if results.get(2, {}).get("ok"):
        print("  MeanFlowConfig(estimator='jvp'), and Gate B MUST run inside "
              "transport.meanflow.math_attention(). Check probe 2's peak_gb "
              "against 24GB: the math backend materialises the attention "
              "matrix, so this is the number that decides local vs A100.")
    elif results.get(3, {}).get("ok"):
        print("  MeanFlowConfig(estimator='fd'). Forward-mode AD is not "
              "available here; finite differences cost two extra forwards per "
              "step and Gate B is otherwise unchanged. Record this in the run "
              "config.")
    else:
        print("  Nothing passed. If probe 2 died on a DTYPE error rather than a "
              "signal, forward-AD is working and only the weight dtype is "
              "wrong: rerun with --dtype float32 before concluding anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
