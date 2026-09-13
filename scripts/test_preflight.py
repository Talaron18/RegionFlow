"""
Pre-flight for the transport stack. CPU, seconds, no weights downloaded.

WHY. Gate A's first launch died after loading a 2B-parameter model, on
`mat1 and mat2 must have the same dtype`, from a float32 keep-mask in
cond_dropout promoting bf16 embeddings back to float32 -- three lines below a
caller that had just cast them correctly. Nothing about that bug needed real
weights, a GPU, or a dataset. It needed one forward through a two-layer model.

This builds a randomly-initialised SD3Transformer2DModel from config (same
class, same forward path, ~1MB), wraps it in TransportAdapter, and exercises
every path the real run uses. Anything that is a shape, dtype, or wiring
problem shows up here; anything that survives is about the data or the
optimisation, which is where debugging effort is actually worth spending.

    python scripts/test_preflight.py

Run it after ANY edit to transport/. It costs less than reading the traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter  # noqa: E402
from transport.meanflow import MeanFlowConfig, meanflow_loss, sample_interval  # noqa: E402
from transport.pairflow import PairFlowConfig, apply_transport, pairflow_loss  # noqa: E402


def build_tiny(dtype, device="cpu"):
    from diffusers import SD3Transformer2DModel

    return SD3Transformer2DModel(
        sample_size=16, patch_size=2, in_channels=16, num_layers=2,
        attention_head_dim=16, num_attention_heads=4, joint_attention_dim=64,
        caption_projection_dim=64, pooled_projection_dim=32, out_channels=16,
        pos_embed_max_size=32,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)


class DtypeSpy(torch.nn.Module):
    """Wrap the transformer and assert every input matches its weight dtype.

    The real failure surfaces deep inside diffusers, in whichever Linear
    happens to be hit first. Checking at the boundary names the offending
    argument instead.
    """

    def __init__(self, inner, expect):
        super().__init__()
        self.inner, self.expect = inner, expect
        self.violations = []

    def forward(self, **kw):
        for name, val in kw.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                # timestep is legitimately fp32: diffusers builds the
                # sinusoidal embedding in float32 regardless of model dtype.
                if name == "timestep":
                    continue
                if val.dtype != self.expect:
                    self.violations.append(f"{name}: {val.dtype} != {self.expect}")
        return self.inner(**kw)


def make_batch(B, dtype, device="cpu"):
    z_src = torch.randn(B, 16, 16, 16, device=device)
    z_tgt = z_src + 0.3 * torch.randn_like(z_src)
    mask = torch.zeros(B, 1, 16, 16, device=device)
    mask[:, :, 4:12, 4:12] = 1.0
    pe = torch.randn(B, 24, 64, device=device, dtype=dtype)
    po = torch.randn(B, 32, device=device, dtype=dtype)
    return z_src, z_tgt, mask, pe, po


#: KNOWN LIMITATION, not a bug to fix here.
#: torch.func.jvp promotes its tangents to float32, so with bfloat16 weights
#: the two meet at the first attention projection and raise
#: "expected mat1 and mat2 to have the same dtype: float != BFloat16".
#: Confirmed on CUDA; CPU takes a different kernel path and does NOT reproduce
#: it, which is why an earlier CPU-only pre-flight passed and Gate B still
#: died on launch.
#:
#: Consequence: Gate B (interval mode, MeanFlow, jvp) MUST run float32.
#: Gate A (pair mode, no forward-mode AD anywhere) runs bfloat16 happily.
#: Testing the combination is still worth doing -- it documents the limit and
#: would flag a torch upgrade that lifted it -- but it must not block a launch
#: of a configuration that works.
XFAIL = {("bfloat16", "interval")}


def run(dtype, B=2, device="cpu") -> list:
    fails, xfails = [], []
    tag = str(dtype).replace("torch.", "")

    for mode in ("pair", "interval"):
        tr = DtypeSpy(build_tiny(dtype, device), dtype)
        # .to(device) is not optional: TransportAdapter owns its own
        # submodules (MaskEncoder, IntervalEmbedding) which are constructed on
        # CPU regardless of where the wrapped transformer lives. Both gate
        # scripts do this; the pre-flight did not, which is why it failed on
        # cuda with "Input type (cuda.FloatTensor) and weight type
        # (FloatTensor)" inside the mask conv.
        ad = TransportAdapter(tr.inner,
                              AdapterConfig(mode=mode, rank=4)).to(device)
        ad.transformer = tr.to(device)           # route through the spy
        for p in ad.trainable_parameters():
            p.data = p.data.float()

        # GRADIENT CHECKPOINTING ON. Gate A's first launch died in
        # checkpoint's recompute pass and this pre-flight passed cleanly
        # beforehand, because checkpointing was not enabled here. Anything
        # the real trainer turns on, this must turn on too.
        tr.inner.enable_gradient_checkpointing()

        z_src, z_tgt, mask, pe, po = make_batch(B, dtype, device)
        t = torch.full((B,), 0.5, device=device)
        r = torch.zeros(B, device=device) if mode == "interval" else t.clone()

        try:
            u = ad(z_src, t, r, pe, po, mask=mask)
            assert u.dtype == torch.float32, f"u is {u.dtype}, expected float32"
            assert u.shape == z_src.shape, f"u {u.shape} != z {z_src.shape}"
        except Exception as exc:                              # noqa: BLE001
            fails.append(f"[{tag}/{mode}] forward: {type(exc).__name__}: {exc}")

        # CFG path: gate must be repeated to cover [uncond, cond]
        try:
            ad(z_src, t, r, pe, po, mask=mask, guidance=3.0,
               negative_embeds=torch.zeros_like(pe),
               negative_pooled=torch.zeros_like(po))
        except Exception as exc:                              # noqa: BLE001
            fails.append(f"[{tag}/{mode}] cfg forward: {type(exc).__name__}: {exc}")

        # Boundary guarantee: interval mode must equal the base at r == t.
        if mode == "interval":
            chk = ad.check_identity(z_src, t, pe, po, mask=mask)
            if not chk["boundary_ok"]:
                fails.append(f"[{tag}/interval] boundary violated: "
                             f"{chk['boundary_max_abs_err']:.3e}")

        # Losses, with dropout ON -- this is where the promotion bug lived.
        try:
            if mode == "pair":
                loss, st = pairflow_loss(
                    ad, z_src, z_tgt, mask, pe, po,
                    PairFlowConfig(mask_dropout=1.0, cond_dropout=1.0))
            else:
                tt, rr = sample_interval(B, z_src.device, max_delta=0.5)
                eps = torch.randn_like(z_src)
                zt = (1 - tt.reshape(-1, 1, 1, 1)) * z_src \
                    + tt.reshape(-1, 1, 1, 1) * eps
                loss, st = meanflow_loss(
                    lambda z_, r_, t_: ad(z_, t_, r_, pe, po),
                    zt, rr, tt, eps - z_src,
                    MeanFlowConfig(estimator="jvp"))
            loss.backward()
            grads = [p.grad for p in ad.trainable_parameters() if p.grad is not None]
            if not grads:
                fails.append(f"[{tag}/{mode}] no gradients reached the adapter")
            elif all(float(g.abs().sum()) == 0 for g in grads):
                fails.append(f"[{tag}/{mode}] all adapter gradients are zero")
            elif not all(torch.isfinite(g).all() for g in grads):
                fails.append(f"[{tag}/{mode}] non-finite adapter gradients")
        except Exception as exc:                              # noqa: BLE001
            msg = f"[{tag}/{mode}] loss/backward: {type(exc).__name__}: {exc}"
            if (tag, mode) in XFAIL and "same dtype" in str(exc):
                xfails.append(msg)
            else:
                fails.append(msg)

        if mode == "pair":
            try:
                out = apply_transport(ad, z_src, mask, pe, po,
                                      strength=1.0, steps=2)
                assert out.shape == z_src.shape
            except Exception as exc:                          # noqa: BLE001
                fails.append(f"[{tag}/pair] apply_transport: "
                             f"{type(exc).__name__}: {exc}")

        if tr.violations and (tag, mode) not in XFAIL:
            fails.append(f"[{tag}/{mode}] wrong dtype into transformer: "
                         + "; ".join(sorted(set(tr.violations))))
    for x in xfails:
        print(f"  XFAIL (known, not a blocker) {x}")
    return fails


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu",
                    help="RUN THIS ON cuda TOO. CPU takes different kernel "
                         "paths: a bfloat16 JVP passes here on CPU and fails "
                         "on CUDA with a float/BFloat16 dtype error, which is "
                         "how Gate B's first launch died after this "
                         "pre-flight came back clean.")
    args = ap.parse_args()

    all_fails = []
    for dtype in (torch.bfloat16, torch.float32):
        print(f"--- {dtype} on {args.device} ---")
        f = run(dtype, device=args.device)
        all_fails += f
        print("  OK" if not f else "\n".join("  FAIL " + x for x in f))

    print()
    if all_fails:
        print(f"{len(all_fails)} failure(s). Do not launch training.")
        return 1
    print("Known limitation, exercised and expected: bfloat16 + interval mode "
          "(MeanFlow/jvp) fails on CUDA.\n  Gate B must run --dtype float32; "
          "Gate A's pair mode uses no forward-mode AD and is fine in bf16.")
    print(f"Pre-flight clean on {args.device}: forward, CFG, boundary, both "
          "losses, backward, and the one-step readout all work in bf16 and "
          "fp32.")
    if args.device == "cpu":
        print("  Now rerun with --device cuda. CPU does not exercise the "
              "kernels that broke Gate B's first launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
