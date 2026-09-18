"""Per-item cos(u, v_clean) under the three treatments.

diagnose only reports quantiles; this writes one row per cached item with
its raw/pooled/centered cosine joined to the cache's own meta fields, so
the low tail of the centered column can be inspected directly.

Usage:
    python scripts/diagnose_per_item.py \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --bias-npy runs/bias.npy \
        --mask-encoder-kind global_local \
        --out runs/gate_a_sigma_0.0/per_item.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache, torch_dataset
from transport.pairflow import direction_position_diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bias-npy", required=True,
                    help="from ambiguity_probe.py --save-bias")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="global_local",
                    choices=("single", "global_local"))
    args = ap.parse_args()

    from diffusers import SD3Transformer2DModel
    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(args.device)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="pair", rank=args.rank,
                          mask_encoder_kind=args.mask_encoder_kind)
    ).to(args.device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")
    adapter.eval()
    print(f"loaded step {sd.get('step', '?')}")

    bias = torch.from_numpy(np.load(args.bias_npy)).float()
    print(f"bias shape {tuple(bias.shape)}")

    cache = TripletCache(args.cache)
    n_total = len(cache)
    ds = torch_dataset(cache)
    # shuffle=False so index i in the loader == cache[i]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    i0 = 0
    with out.open("w") as fh, torch.no_grad():
        for batch in dl:
            z_src = batch["z_src"].to(args.device).float()
            z_tgt = batch["z_tgt"].to(args.device).float()
            mask = batch["mask"].to(args.device).float()
            pe = batch["prompt_embeds"].to(args.device)
            pooled = batch["pooled"].to(args.device)
            b = z_src.shape[0]

            # s=0 query: the one-step readout's actual point
            s0 = torch.zeros(b, device=args.device)
            u = adapter(z_src, s0, s0, pe, pooled, mask=mask)
            v_clean = z_tgt - z_src

            diag = direction_position_diagnostics(u.float(), v_clean, mask,
                                                  bias=bias)

            for j in range(b):
                rec = cache[i0 + j]
                meta = rec["meta"]
                fh.write(json.dumps({
                    "idx": i0 + j,
                    "id": meta.get("id"),
                    "instruction": meta.get("instruction"),
                    "dataset": meta.get("dataset"),
                    "mask_frac": float(meta.get("mask_frac", -1.0)),
                    "moved": float(meta.get("moved", -1.0)),
                    "raw": float(diag["raw"][j]),
                    "pooled": float(diag["pooled"][j]),
                    "centered": (float(diag["centered"][j])
                                 if "centered" in diag else None),
                }) + "\n")
            i0 += b
            if i0 % 500 == 0:
                print(f"  {i0}/{n_total}")

    print(f"wrote {i0} rows to {out}")


if __name__ == "__main__":
    main()