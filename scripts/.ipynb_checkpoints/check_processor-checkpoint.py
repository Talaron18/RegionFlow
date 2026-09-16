"""
Does the custom processor match diffusers' stock one when no bias is set?

This is the test that should have existed before any inversion run. With
spec=None, RegionJointAttnProcessor must be numerically IDENTICAL to
JointAttnProcessor2_0. Any difference means the reimplementation is wrong and
every downstream number -- reconstruction PSNR, background preservation,
selectivity -- is measuring processor bugs.

The bug that motivated this: QK-norm is per stream. diffusers applies
norm_q/norm_k to image tokens and norm_added_q/norm_added_k to text tokens,
each before concatenation. Normalising the concatenated sequence with the image
norms corrupts the model, and the corruption showed up as periodic striping in
the decoded image -- which is easy to misread as inversion drift.

    python scripts/check_processor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"


def main():
    from diffusers import StableDiffusion3Pipeline

    print(f"loading {MODEL} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    tf = pipe.transformer
    dtype = tf.dtype

    # Report which norms exist -- if SD3-medium has no QK norm, that particular
    # bug would not have fired and something else is at fault.
    a0 = tf.transformer_blocks[0].attn
    print("\nQK-norm modules on block 0:")
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        mod = getattr(a0, name, None)
        print(f"  {name:<16} {type(mod).__name__ if mod is not None else 'None'}")

    side = 64
    with torch.no_grad():
        pe, _, pool, _ = pipe.encode_prompt(
            prompt="a red apple on a wooden table",
            prompt_2=None, prompt_3=None, device="cuda",
        )
    latent = torch.randn(1, tf.config.in_channels, side, side, device="cuda", dtype=dtype)
    ts = torch.tensor([500.0], device="cuda")

    def forward():
        with torch.no_grad():
            return tf(
                hidden_states=latent,
                encoder_hidden_states=pe.to(dtype),
                pooled_projections=pool.to(dtype),
                timestep=ts,
            )[0].float()

    print("\n--- stock processor ---")
    ref = forward()
    print(f"  output  mean {ref.mean():+.5f}  std {ref.std():.5f}")

    print("\n--- custom processor, spec=None ---")
    from region.sd3_processor import install

    proc = install(tf)
    proc.set_region(None, direction="generate")
    got = forward()
    print(f"  output  mean {got.mean():+.5f}  std {got.std():.5f}")

    diff = (ref - got).abs()
    rel = diff.max().item() / max(ref.abs().max().item(), 1e-9)
    print(f"\n  max abs diff {diff.max().item():.3e}")
    print(f"  mean abs diff {diff.mean().item():.3e}")
    print(f"  relative      {rel:.3e}")

    # bf16 round-off across a different op order is ~1e-2 relative at worst;
    # a real logic error is orders of magnitude larger.
    tol = 5e-2
    ok = rel < tol
    print("\n" + "=" * 60)
    if ok:
        print("  PASS: the custom processor matches stock within bf16 noise.")
        print("  Inversion error is therefore genuine, not a processor bug.")
    else:
        print("  FAIL: the custom processor does NOT match stock.")
        print("  Every reconstruction and preservation number so far is invalid.")
        print("  Compare region/sd3_processor.py against your installed")
        print("  diffusers JointAttnProcessor2_0 line by line.")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
