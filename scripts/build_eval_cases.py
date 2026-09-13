"""
Build an expanded Gate A eval set.

WHY. The re-scored Gate A rests on two usable cases. Ball and phone_booth came
back at oracle parity (1.10, 1.03); sheep was undefined because the
ground-truth edit itself scores 0.024 on grad_orient -- the object is ~30px and
darkening white wool to black leaves no gradient orientation to measure. The
verdict flips between REFUTED (1.065, excluding sheep) and INCONCLUSIVE (0.71,
including it), and the exclusion rule was written after seeing which case
failed. That is not a defensible position, and the fix is more cases rather
than a better argument.

SELECTION, fixed before looking at any model output:

  mask fraction        >= 3%    small objects are where gradient orientation
                                degenerates; the sheep at ~30px is the case
                                that broke the measurement
  mask fraction        <= 40%   above this "background" is barely defined
  oracle grad_orient   >= 0.30  the ground-truth edit must show structure the
                                metric can see, or it cannot serve as a
                                denominator. Same floor as
                                bench/decoupling.ORACLE_VALID_FLOOR, justified
                                there by a pure blur scoring 0.297.
  oracle dE            >= 15    the real edit must actually change colour, or
                                there is nothing to trade against structure
  turn_index == 1               later turns have a model output as source

The oracle checks need the VAE, so this runs on the GPU -- but only encode and
decode, no sampling. Budget a few minutes for the dev split.

    python scripts/build_eval_cases.py --data-root /data/datasets/MagicBrush \\
        --n 20 --out captions/eval_cases_v2.json

Output is in the captions.json shape ({id: [instruction, instruction]}) so
gate_a_transport.py eval consumes it unchanged:

    python scripts/gate_a_transport.py eval --ckpt <ckpt> \\
        --captions captions/eval_cases_v2.json --data-root ... --cache ...

KEEP THE THREE STAGE 0 CASES SEPARATE. They are the comparison with Stage 0 and
were chosen for that; folding them into a filtered set would make the new
numbers neither a replication nor a clean sample. Report both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIN_MASK, MAX_MASK = 0.03, 0.40
MIN_ORACLE_GRAD, MIN_ORACLE_DE = 0.30, 15.0
STAGE0_IDS = ("17320_t1", "158548_t1", "429011_t1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/datasets/MagicBrush")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--scan", type=int, default=400,
                    help="how many candidates to encode before stopping")
    ap.add_argument("--out", default="captions/eval_cases_v2.json")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusion3Pipeline

    from bench.decoupling import evaluate_point
    from bench.inversion import SD3EditPipeline
    from precompute_cache import iter_magicbrush
    from transport.conditioning import mask_to_latent_grid

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)

    kept, rejected, seen = {}, {}, 0
    for trip in iter_magicbrush(args.data_root, args.split, args.size,
                                limit=None):
        if len(kept) >= args.n or seen >= args.scan:
            break
        if trip.case_id in STAGE0_IDS:
            continue                      # kept as a separate comparison set
        seen += 1

        frac = float(np.asarray(trip.mask).mean())
        if not (MIN_MASK <= frac <= MAX_MASK):
            rejected["mask fraction"] = rejected.get("mask fraction", 0) + 1
            continue

        # The oracle must be a measurable edit: real colour change, and
        # structure the metric can still see.
        z_src = helper.encode_image(trip.image).float()
        z_tgt = helper.encode_image(trip.target_image).float()
        h, w = z_src.shape[-2:]
        m = mask_to_latent_grid(trip.mask, h, w).to(z_src.device)
        recon = helper.decode_latent(z_src)
        oracle_img = helper.decode_latent(z_src + (z_tgt - z_src) * m)
        p = evaluate_point(trip.case_id, 1.0, recon, oracle_img, trip.mask)

        if p.dE < MIN_ORACLE_DE:
            rejected["oracle dE too small"] = \
                rejected.get("oracle dE too small", 0) + 1
            continue
        if p.grad_orient is None or p.grad_orient < MIN_ORACLE_GRAD:
            rejected["oracle below metric floor"] = \
                rejected.get("oracle below metric floor", 0) + 1
            continue

        kept[trip.case_id] = [trip.instruction, trip.instruction]
        print(f"  + {trip.case_id:<14} mask {100*frac:>5.1f}%  "
              f"oracle dE {p.dE:>5.1f}  grad_or {p.grad_orient:.3f}  "
              f"{trip.instruction!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, indent=1))

    print(f"\nkept {len(kept)} of {seen} scanned -> {out}")
    for k, v in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  rejected, {k}: {v}")
    if len(kept) < args.n:
        print(f"\nOnly {len(kept)} cases met the criteria. Raise --scan, or "
              "widen the split to train (the eval cases must then be checked "
              "against the training cache -- the leakage guard in "
              "gate_a_transport.py eval will catch it).")
    print("\nNext:")
    print(f"  python scripts/gate_a_transport.py eval --ckpt <ckpt> \\")
    print(f"      --captions {out} --data-root {args.data_root} \\")
    print(f"      --cache <cache> --out runs/gate_a_full/eval_v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
