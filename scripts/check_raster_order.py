"""
Does field.reshape(-1) match SD3's token order?

Every field in this project is built on a (h_tok, w_tok) grid and flattened
row-major. If the patch embedder emits tokens in any other order, the bias
lands on the wrong part of the image and NOTHING RAISES -- you get plausible
but wrong edits, and the r-sweep measures noise.

Three tests, cheapest first:

  1. PATCHIFY ORDER. Push a latent with a single marked patch through
     transformer.pos_embed and find which token carries it. Definitive, needs
     no generation, runs in seconds.
  2. BIAS DELIVERY. Confirm the region bias actually reaches
     scaled_dot_product_attention. diffusers' stock joint processor accepts an
     attention_mask and silently drops it, so this is not paranoia.
  3. END-TO-END. Generate twice from one seed, with and without a bias
     confined to one corner, and check the diff concentrates there.

    python scripts/check_raster_order.py
    python scripts/check_raster_order.py --skip-generation   # tests 1-2 only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"


# ---------------------------------------------------------------- test 1 ----
def test_patchify_order(transformer, latent_hw=None, device="cuda"):
    """Mark one latent patch, see which token index it becomes."""
    pos = transformer.pos_embed
    patch = getattr(pos, "patch_size", 2)
    ch = transformer.config.in_channels
    # Default to the model's own sample_size so the token grid matches what
    # the sweep will actually use.
    if latent_hw is None:
        s = transformer.config.sample_size
        latent_hw = (s, s)
    H, W = latent_hw
    h_tok, w_tok = H // patch, W // patch

    print(f"\n--- test 1: patchify order ---")
    print(f"  latent {H}x{W}, patch {patch} -> token grid {h_tok}x{w_tok}")

    ok = True
    dtype = next(pos.parameters()).dtype
    # Probe several positions, including asymmetric ones so a transpose shows.
    probes = [(0, 0), (0, w_tok - 1), (h_tok - 1, 0), (3, 7), (h_tok // 2, w_tok // 3)]
    for (tr, tc) in probes:
        z = torch.zeros(1, ch, H, W, device=device, dtype=dtype)
        z[:, :, tr * patch : (tr + 1) * patch, tc * patch : (tc + 1) * patch] = 100.0

        with torch.no_grad():
            tokens = pos(z)
        if tokens.ndim == 3:
            energy = tokens[0].float().abs().sum(-1)
        else:
            raise RuntimeError(f"unexpected pos_embed output shape {tokens.shape}")

        got = int(energy.argmax().item())
        want = tr * w_tok + tc                      # row-major, == reshape(-1)
        want_col_major = tc * h_tok + tr
        match = "row-major OK" if got == want else (
            "COLUMN-MAJOR!" if got == want_col_major else "UNKNOWN ORDER"
        )
        if got != want:
            ok = False
        print(f"  latent patch (r={tr:>2}, c={tc:>2}) -> token {got:>5} "
              f"(row-major predicts {want:>5})  {match}")

    print(f"  => {'PASS' if ok else 'FAIL'}: "
          f"{'field.reshape(-1) is correct' if ok else 'FIX THE FLATTEN ORDER'}")
    return ok


# ---------------------------------------------------------------- test 2 ----
def test_bias_delivery(pipe, device="cuda"):
    """Does the bias actually reach SDPA, or is it silently dropped?"""
    from region.attention import RegionAttentionSpec
    from region.sd3_processor import assert_bias_applied, install

    print(f"\n--- test 2: bias delivery ---")
    proc = install(pipe.transformer)

    dtype = pipe.transformer.dtype
    h_tok = w_tok = pipe.transformer.config.sample_size // 2
    n_img = h_tok * w_tok
    field = torch.zeros(1, n_img, device=device, dtype=dtype)
    field[0, :50] = 4.0

    # SD3 context length: merged CLIP (77) + T5 (256) = 333. Test 2 also
    # verifies this against the real encoder output.
    with torch.no_grad():
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(
            prompt="a photo of a cat", prompt_2=None, prompt_3=None, device=device
        )
    n_txt = prompt_embeds.shape[1]
    print(f"  real context length: {n_txt} tokens "
          f"({'matches the 333 assumption' if n_txt == 333 else 'DIFFERS from 333'})")

    edit = torch.zeros(1, n_txt, dtype=torch.bool, device=device)
    edit[0, 5:8] = True
    proc.set_region(
        RegionAttentionSpec(field=field, edit_token_mask=edit), direction="generate"
    )

    latent = torch.randn(
        1, pipe.transformer.config.in_channels, h_tok * 2, w_tok * 2, device=device
    ).to(dtype)
    with torch.no_grad():
        pipe.transformer(
            hidden_states=latent,
            encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
            pooled_projections=pooled.to(pipe.transformer.dtype),
            timestep=torch.tensor([500.0], device=device),
        )

    try:
        assert_bias_applied(proc)
        print(f"  bias reached SDPA, |bias| sum = {proc.last_bias_norm:.1f}")
        print("  => PASS")
        return True, n_txt
    except RuntimeError as exc:
        print(f"  => FAIL: {exc}")
        return False, n_txt


# ---------------------------------------------------------------- test 3 ----
def test_attention_localisation(pipe, n_txt, device="cuda", size=512):
    """Do biased image tokens actually attend more to the edit tokens?

    The earlier version of this test compared two GENERATIONS and was
    uninterpretable: with no source image to preserve, biasing attention moves
    content around the whole canvas, so the pixel diff measures displacement
    rather than localisation. It reported a horizontal flip that test 1 had
    already ruled out.

    This measures the mechanism directly instead -- the attention block itself
    -- so it needs no inversion and has no generative semantics to confound it.
    """
    from region.attention import RegionAttentionSpec
    from region.sd3_processor import install

    print("\n--- test 3: attention localisation ---")
    proc = install(pipe.transformer)
    proc.capture = True
    proc.capture_layer = len(pipe.transformer.transformer_blocks) // 2

    latent_side = size // 8
    h_tok = w_tok = latent_side // 2
    n_img = h_tok * w_tok
    dtype = pipe.transformer.dtype
    print(f"  probing at {size}px -> {h_tok}x{w_tok} = {n_img} image tokens")

    grid = np.full((h_tok, w_tok), -2.0, dtype=np.float32)
    grid[: h_tok // 2, : w_tok // 2] = 4.0        # TOP-LEFT only
    field = torch.as_tensor(grid.reshape(-1), device=device, dtype=dtype)[None]

    edit = torch.zeros(1, n_txt, dtype=torch.bool, device=device)
    edit[0, 1:12] = True
    proc.set_region(
        RegionAttentionSpec(field=field, edit_token_mask=edit), direction="generate"
    )

    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt="a red apple on a table", prompt_2=None, prompt_3=None, device=device
        )
        latent = torch.randn(
            1, pipe.transformer.config.in_channels, latent_side, latent_side,
            device=device, dtype=dtype,
        )
        pipe.transformer(
            hidden_states=latent,
            encoder_hidden_states=pe.to(dtype),
            pooled_projections=pooled.to(dtype),
            timestep=torch.tensor([500.0], device=device),
        )

    if proc.captured is None:
        print("  => FAIL: no attention captured")
        return False

    edit_mass = proc.captured[:, edit[0].cpu()].sum(-1).numpy().reshape(h_tok, w_tok)
    q = {
        "top-left": edit_mass[: h_tok // 2, : w_tok // 2].mean(),
        "top-right": edit_mass[: h_tok // 2, w_tok // 2 :].mean(),
        "bottom-left": edit_mass[h_tok // 2 :, : w_tok // 2].mean(),
        "bottom-right": edit_mass[h_tok // 2 :, w_tok // 2 :].mean(),
    }
    print("  attention mass on edit tokens (field was on TOP-LEFT):")
    for k, v in q.items():
        print(f"    {k:<14} {v:.6f}")

    winner = max(q, key=q.get)
    others = np.mean([v for k, v in q.items() if k != "top-left"])
    ratio = q["top-left"] / max(others, 1e-12)
    print(f"  strongest: {winner}   top-left / others = {ratio:.2f}x")

    if winner == "top-left" and ratio > 1.5:
        print("  => PASS: the field steers attention where it says it does")
        return True
    if winner == "top-right":
        print("  => FAIL: horizontal flip in the flatten order")
    elif winner == "bottom-left":
        print("  => FAIL: vertical flip in the flatten order")
    else:
        print("  => FAIL: no localisation -- raise lam_in and retry")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    from diffusers import StableDiffusion3Pipeline

    print(f"loading {MODEL} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16
    ).to("cuda")
    print(f"  transformer: {type(pipe.transformer).__name__}, "
          f"{len(pipe.transformer.transformer_blocks)} blocks, "
          f"sample_size={pipe.transformer.config.sample_size}")

    r1 = test_patchify_order(pipe.transformer)
    r2, n_txt = test_bias_delivery(pipe)
    r3 = None
    if not args.skip_generation:
        if r1 and r2:
            r3 = test_attention_localisation(pipe, n_txt)
        else:
            print("\nskipping test 3 — fix tests 1-2 first, or it is uninterpretable")

    print("\n" + "=" * 70)
    print(f"  patchify order : {'PASS' if r1 else 'FAIL'}")
    print(f"  bias delivery  : {'PASS' if r2 else 'FAIL'}")
    print(f"  attention loc  : {'PASS' if r3 else ('FAIL' if r3 is False else 'skipped')}")
    print("=" * 70)
    sys.exit(0 if (r1 and r2 and r3 is not False) else 1)


if __name__ == "__main__":
    main()
