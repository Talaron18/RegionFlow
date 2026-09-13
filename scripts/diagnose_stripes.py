"""
Where do the 16px stripes come from?

The round-trip image has a purely horizontal artifact with period exactly equal
to the transformer patch (8x VAE * 2x patchify = 16 image px), amplitude ~23
grey levels against a flat source. That is structural, not integration error.

Four generations, each measured by the same FFT metric, isolating one stage at
a time:

  A. pipeline __call__            -- reference. Clean by definition.
  B. our generate(), random noise -- tests _velocity / the stepping loop.
  C. invert then generate         -- tests invert() on top of B.
  D. invert then decode directly  -- tests invert() alone, no generation loop.

Read it as a bisection:
  A clean, B striped   -> the bug is in _velocity or the Euler loop
  A,B clean, C striped -> the bug is in invert()
  D striped            -> inversion produces out-of-distribution latents
  all striped          -> the bug is in decode_latent / VAE scaling

    python scripts/diagnose_stripes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
OUT = Path("/data/outputs/stripes")
PROMPT = "three red apples on a wooden table"


def stripe_score(img: np.ndarray) -> dict:
    """Energy at the 16px patch frequency, in a flat region."""
    g = np.asarray(img.convert("L") if hasattr(img, "convert") else img, np.float32)
    if g.ndim == 3:
        g = g.mean(2)
    h = g.shape[0]
    band = g[h // 8 : h // 4].mean(0)          # upper background strip
    band = band - band.mean()
    sp = np.abs(np.fft.rfft(band))
    sp[0] = 0
    k_patch = len(band) // 16                  # 16px period
    return {
        "amp": float(band.std()),
        "patch_freq_energy": float(sp[k_patch]),
        "dominant_period": float(len(band) / max(int(np.argmax(sp)), 1)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from diffusers import StableDiffusion3Pipeline
    from PIL import Image

    from bench.inversion import SD3EditPipeline
    from scripts.smoke_edit import synthetic

    img, _ = synthetic(512)
    Image.fromarray(img).save(OUT / "source.png")

    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    ed = SD3EditPipeline(pipe)

    results = {}

    # --- A: the pipeline's own sampler --------------------------------
    g = torch.Generator("cuda").manual_seed(0)
    a = np.array(
        pipe(prompt=PROMPT, num_inference_steps=28, guidance_scale=5.0, generator=g).images[0]
    )
    Image.fromarray(a).save(OUT / "A_pipeline.png")
    results["A pipeline __call__"] = stripe_score(a)

    # --- B: our loop, from random noise -------------------------------
    from bench.inversion import Inversion

    torch.manual_seed(0)
    noise = torch.randn(
        1, pipe.transformer.config.in_channels, 64, 64,
        device="cuda", dtype=ed.dtype,
    )
    b = ed.generate(Inversion(noise=noise), PROMPT, steps=28, guidance=5.0)
    Image.fromarray(b).save(OUT / "B_our_generate.png")
    results["B our generate (noise)"] = stripe_score(b)

    # --- C & D: inversion ---------------------------------------------
    lat = ed.encode_image(img)
    results["  (VAE only)"] = stripe_score(ed.decode_latent(lat))

    inv = ed.invert(lat, PROMPT, steps=50)
    d = ed.decode_latent(inv.noise)
    Image.fromarray(d).save(OUT / "D_inverted_latent.png")
    results["D decode(inverted noise)"] = stripe_score(d)

    c = ed.generate(inv, PROMPT, steps=50, guidance=1.0)
    Image.fromarray(c).save(OUT / "C_invert_then_generate.png")
    results["C invert + generate"] = stripe_score(c)

    # --- latent statistics --------------------------------------------
    print("\n--- latent statistics ---")
    print(f"  {'':<26} {'mean':>8} {'std':>8} {'absmax':>8}")
    for name, t in [("source latent", lat), ("inverted noise", inv.noise),
                    ("true N(0,1) ref", torch.randn_like(lat))]:
        f = t.float()
        print(f"  {name:<26} {f.mean():8.3f} {f.std():8.3f} {f.abs().max():8.2f}")
    print("  inverted noise should be close to N(0,1). A std far from 1.0 means")
    print("  the inversion is not landing on the noise distribution, and the VAE")
    print("  decoder is then being asked to decode something it never saw.")

    # --- verdict-free summary -----------------------------------------
    print("\n--- stripe measurements (16px = patch period) ---")
    print(f"  {'':<26} {'amp':>8} {'patch-E':>10} {'dom.period':>11}")
    for name, s in results.items():
        print(f"  {name:<26} {s['amp']:8.2f} {s['patch_freq_energy']:10.1f} "
              f"{s['dominant_period']:11.2f}")

    print(f"\n  images in {OUT}")
    print(
        "\n  Read across: the first row with a large patch-frequency energy is\n"
        "  where the artifact enters. No automatic verdict here on purpose --\n"
        "  compare the numbers and look at the images."
    )


if __name__ == "__main__":
    main()
