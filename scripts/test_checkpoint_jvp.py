"""
Does gradient checkpointing change anything in the MeanFlow step?

WHY THIS IS NOT PARANOIA. Two things in this stack interact badly with
checkpointing, and both were introduced to fix earlier failures:

  1. The gate. Checkpointing replays the forward during backward, OUTSIDE the
     context manager the original forward ran in. transport/gating.py therefore
     lets the gate value PERSIST past its scope, so the recompute reads the
     last value set. That is correct as long as the recompute belongs to the
     most recent forward -- but nothing enforces it. If an ordering ever
     changes so that another forward intervenes, the recompute silently applies
     the wrong per-sample scaling and the gradients are wrong with no error.

  2. The JVP. transport/meanflow.py runs it under no_grad specifically so
     forward-mode AD never meets the checkpoint machinery. That separation is
     an assumption about how the two compose, not something torch guarantees.

Both failure modes produce WRONG NUMBERS, not exceptions. A Gate B run would
train happily against a corrupted target and only look odd after hours.

WHAT IS COMPARED. One MeanFlow step on a tiny randomly-initialised SD3, with
checkpointing on and off, with identical inputs and RNG:

    du/dt        the JVP output. Should be BIT-IDENTICAL: it is computed under
                 no_grad and checkpointing has no business touching it.
    loss         should be bit-identical for the same reason.
    param grads  the real test. Checkpoint recompute reconstructs the
                 activations these are built from, so any gate mismatch shows
                 up here and nowhere else. Small float differences from a
                 different summation order are expected; anything above ~1e-4
                 relative is not.

    python scripts/test_checkpoint_jvp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter  # noqa: E402
from transport.meanflow import MeanFlowConfig, meanflow_loss  # noqa: E402
from transport.meanflow_ref import (  # noqa: E402
    RefMeanFlowConfig, ref_meanflow_loss,
)


def build(dtype, device, seed=0):
    from diffusers import SD3Transformer2DModel

    torch.manual_seed(seed)
    tr = SD3Transformer2DModel(
        sample_size=16, patch_size=2, in_channels=16, num_layers=2,
        attention_head_dim=16, num_attention_heads=4, joint_attention_dim=64,
        caption_projection_dim=64, pooled_projection_dim=32, out_channels=16,
        pos_embed_max_size=32,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)

    torch.manual_seed(seed + 1)          # same adapter init both times
    ad = TransportAdapter(tr, AdapterConfig(mode="interval", rank=4)).to(device)
    for p in ad.trainable_parameters():
        p.data = p.data.float()
    return ad


def one_step(ad, ckpt: bool, estimator: str, device, dtype, seed=0):
    if ckpt:
        ad.transformer.enable_gradient_checkpointing()
    elif hasattr(ad.transformer, "disable_gradient_checkpointing"):
        ad.transformer.disable_gradient_checkpointing()

    torch.manual_seed(seed + 100)
    B = 2
    x0 = torch.randn(B, 16, 16, 16, device=device)
    pe = torch.randn(B, 24, 64, device=device, dtype=dtype)
    po = torch.randn(B, 32, device=device, dtype=dtype)
    eps = torch.randn_like(x0)
    t = torch.full((B,), 0.7, device=device)
    r = torch.full((B,), 0.2, device=device)
    t_ = t.reshape(-1, 1, 1, 1)
    z_t = (1 - t_) * x0 + t_ * eps
    v = eps - x0

    for p in ad.trainable_parameters():
        if p.grad is not None:
            p.grad = None

    captured = {}

    def fn(z_, r_, t_in):
        return ad(z_, t_in, r_, pe, po)

    from transport.meanflow import total_derivative
    cfg = MeanFlowConfig(estimator=estimator, adaptive_p=0.0)
    u, du = total_derivative(fn, z_t, r, t, v, cfg)
    captured["du"] = du.detach().clone()

    loss, _ = meanflow_loss(fn, z_t, r, t, v, cfg)
    loss.backward()
    grads = torch.cat([p.grad.detach().flatten().clone()
                       for p in ad.trainable_parameters()
                       if p.grad is not None])
    captured["loss"] = float(loss)
    captured["grads"] = grads
    return captured


def rel(a, b):
    d = (a - b).norm()
    n = b.norm().clamp(min=1e-30)
    return float(d / n)


def ref_step(ad, ckpt: bool, estimator: str, device, dtype, seed=0,
             checkpoint_safe=True):
    """One reference-path step, the code Gate B actually executes."""
    if ckpt:
        ad.transformer.enable_gradient_checkpointing()
    elif hasattr(ad.transformer, "disable_gradient_checkpointing"):
        ad.transformer.disable_gradient_checkpointing()

    torch.manual_seed(seed + 100)
    B = 2
    z = torch.randn(B, 16, 16, 16, device=device)
    pe = torch.randn(B, 24, 64, device=device, dtype=dtype)
    po = torch.randn(B, 32, device=device, dtype=dtype)
    t = torch.full((B,), 0.7, device=device)
    r = torch.full((B,), 0.2, device=device)
    v_g = torch.randn_like(z)

    for p in ad.trainable_parameters():
        p.grad = None
    cfg = RefMeanFlowConfig(estimator=estimator,
                            checkpoint_safe=checkpoint_safe)
    loss, st = ref_meanflow_loss(
        lambda z_, t_, r_: ad(z_, t_, r_, pe, po), z, t, r, v_g, cfg)
    loss.backward()
    grads = torch.cat([p.grad.detach().flatten().clone()
                       for p in ad.trainable_parameters()
                       if p.grad is not None])
    return {"loss": float(loss), "grads": grads, "sse": st["sse"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--estimator", default="jvp", choices=["jvp", "fd"])
    ap.add_argument("--path", default="both", choices=["both", "custom", "ref"],
                    help="which loss path to test. 'ref' is the one Gate B "
                         "actually runs; the first version of this test only "
                         "exercised 'custom', passed, and Gate B then died in "
                         "checkpoint's forward-AD machinery.")
    args = ap.parse_args()

    dtype = torch.float32
    print(f"device={args.device}  estimator={args.estimator}\n")

    ad = build(dtype, args.device)
    off = one_step(ad, False, args.estimator, args.device, dtype)

    ad2 = build(dtype, args.device)      # identical init, fresh module
    on = one_step(ad2, True, args.estimator, args.device, dtype)

    du_rel = rel(on["du"], off["du"])
    g_rel = rel(on["grads"], off["grads"])
    l_abs = abs(on["loss"] - off["loss"])
    cos = float(torch.nn.functional.cosine_similarity(
        on["grads"][None], off["grads"][None]).item())

    print(f"{'quantity':<24}{'rel err':>12}   note")
    print(f"{'du/dt (JVP)':<24}{du_rel:>12.3e}   expect 0 exactly (no_grad)")
    print(f"{'loss':<24}{l_abs:>12.3e}   expect 0 exactly")
    print(f"{'param grads':<24}{g_rel:>12.3e}   expect < 1e-4")
    print(f"{'grad cosine':<24}{cos:>12.6f}   expect > 0.9999")
    print(f"\ngrad norms: ckpt {on['grads'].norm():.6e}  "
          f"no-ckpt {off['grads'].norm():.6e}")
    print(f"stale gate reads: ckpt {ad2.holder.stale_reads}, "
          f"no-ckpt {ad.holder.stale_reads}")
    print("  Non-zero WITH checkpointing is expected -- that is the recompute\n"
          "  reading the persisted gate, which is the intended behaviour.\n"
          "  Non-zero WITHOUT checkpointing would mean a forward ran outside\n"
          "  any scope, which is a bug.")

    ref_ok = True
    if args.path in ("both", "ref"):
        print("\n--- reference path (transport/meanflow_ref.py) ---")
        try:
            a1 = build(dtype, args.device)
            r_off = ref_step(a1, False, args.estimator, args.device, dtype)
            a2 = build(dtype, args.device)
            r_on = ref_step(a2, True, args.estimator, args.device, dtype)
            rg = rel(r_on["grads"], r_off["grads"])
            rl = abs(r_on["loss"] - r_off["loss"])
            print(f"{'loss':<24}{rl:>12.3e}")
            print(f"{'param grads':<24}{rg:>12.3e}   expect < 1e-4")
            ref_ok = rg < 1e-4 and rl < 1e-6
        except Exception as exc:                              # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            ref_ok = False

        # And confirm the unsafe structure really is what breaks, so the
        # checkpoint_safe deviation is justified by evidence rather than
        # assumed.
        try:
            a3 = build(dtype, args.device)
            ref_step(a3, True, args.estimator, args.device, dtype,
                     checkpoint_safe=False)
            print("  NOTE: checkpoint_safe=False also worked. The deviation "
                  "may no longer be needed on this torch version.")
        except NotImplementedError:
            print("  confirmed: checkpoint_safe=False raises "
                  "NotImplementedError under checkpointing, as expected "
                  "(checkpoint's autograd.Function has no jvp rule)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  checkpoint_safe=False failed differently: "
                  f"{type(exc).__name__}: {exc}")

    ok = (du_rel < 1e-12 and l_abs < 1e-9 and g_rel < 1e-4 and cos > 0.9999
          and ref_ok)
    bad_stale = ad.holder.stale_reads > 0
    print("\n" + ("PASS: checkpointing does not perturb the MeanFlow step. "
                  "Gate B can run with it enabled."
                  if ok and not bad_stale else
                  "FAIL: do NOT start Gate B with checkpointing.\n"
                  "  If only the param grads differ, the gate is being read "
                  "wrong on recompute -- the JVP is fine.\n"
                  "  Workaround: train without checkpointing at batch 1 and "
                  "more accumulation, or drop the derivative to 256px "
                  "latents, before changing the gating."))
    return 0 if (ok and not bad_stale) else 1


if __name__ == "__main__":
    raise SystemExit(main())
