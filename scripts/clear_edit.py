"""Synthetic unambiguous edit: white bg, red circle + green circle,
instruction 'change the red circle to blue', mask on red circle only.

If the model produces a clean blue circle, ambiguity was the bottleneck
and MSE is fine on unambiguous tasks. If it produces a smeared blue
blob, the problem is spatial precision / architecture, not ambiguity.

Usage:
    python scripts/clear_edit.py --ckpt runs/gate_a_dino/ckpt_final.pt \
        --r 1.0 --out runs/gate_a_dino/clear_edit

--r is the interval endpoint passed to the adapter. Checkpoints trained
before 9c67baa saw r = s, so use --r 0.0 for those; checkpoints trained
with interval conditioning (the current default) read out with --r 1.0,
which is what apply_transport and the Gate C wrapper pass for one step.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from bench.inversion import SD3EditPipeline
from diffusers import StableDiffusion3Pipeline

SRC_FILL = (220, 30, 30)
TGT_FILL = (30, 80, 220)


# --- 合成源图和目标图 ---
def make_pair(size=512):
    src = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(src)
    # 红圆
    d.ellipse([100, 100, 300, 300], fill=SRC_FILL)
    # 绿圆
    d.ellipse([320, 320, 480, 480], fill=(30, 180, 30))

    tgt = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(tgt)
    # 蓝圆（替换红圆）
    d.ellipse([100, 100, 300, 300], fill=TGT_FILL)
    # 绿圆不变
    d.ellipse([320, 320, 480, 480], fill=(30, 180, 30))

    # mask: 只有红圆区域
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([100, 100, 300, 300], fill=255)

    return src, tgt, mask


def in_mask_colour(img: np.ndarray, mask: np.ndarray) -> dict:
    """Mean RGB inside the mask and its distance to the source/target fills.

    blue_ok is the pre-registered synthetic check in DINO_EXPERIMENT.md:
    the disc must be blue-dominant and nearer the target fill than the
    source fill.
    """
    rgb = img[mask > 127].reshape(-1, 3).astype(np.float64).mean(axis=0)
    d_tgt = float(np.linalg.norm(rgb - np.array(TGT_FILL)))
    d_src = float(np.linalg.norm(rgb - np.array(SRC_FILL)))
    r, g, b = (float(x) for x in rgb)
    return {"mean_rgb": [round(r, 1), round(g, 1), round(b, 1)],
            "dist_to_target": round(d_tgt, 1),
            "dist_to_source": round(d_src, 1),
            "blue_ok": bool(b > r and b > g and d_tgt < d_src)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/gate_a_sigma_0.0/ckpt_latest.pt")
    ap.add_argument("--r", type=float, default=0.0,
                    help="interval endpoint passed to the adapter; see docstring")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
    ap.add_argument("--out", default=".",
                    help="directory for the PNGs and clear_edit.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    args = ap.parse_args()
    device, model = args.device, args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16).to(device)
    helper = SD3EditPipeline(pipe, device=device)

    adapter = TransportAdapter(
        pipe.transformer,
        AdapterConfig(mode="pair", rank=args.rank,
                      mask_encoder_kind=args.mask_encoder_kind),
    ).to(device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()

    sd = torch.load(args.ckpt,
                    map_location="cpu", weights_only=False)
    adapter.load_state_dict(sd["adapter"], strict=False)
    adapter.eval()

    src, tgt, mask = make_pair(512)
    src.save(out / "synthetic_src.png")
    tgt.save(out / "synthetic_tgt.png")
    mask.save(out / "synthetic_mask.png")

    z_src = helper.encode_image(src).float()
    z_tgt = helper.encode_image(tgt).float()
    h, w = z_src.shape[-2:]

    # 把 mask 降到 latent 分辨率
    from transport.conditioning import mask_to_latent_grid
    m = mask_to_latent_grid(mask, h, w).to(device).float()

    instr = "change the red circle to blue"
    pe, _, pooled, _ = helper.encode_prompt(instr)

    with torch.no_grad():
        s0 = torch.zeros(1, device=device)
        r0 = torch.full((1,), args.r, device=device)
        u = adapter(z_src, s0, r0, pe, pooled, mask=m)

        # 三个预测
        z_model = z_src + u * m
        z_oracle = z_src + (z_tgt - z_src) * m

        # cos
        v = z_tgt - z_src
        def _cos(a, b, mm):
            num = float((a * b * mm).sum())
            den = (float((a.pow(2) * mm).sum()) ** 0.5
                   * float((b.pow(2) * mm).sum()) ** 0.5)
            return num / den if den > 1e-8 else float("nan")
        cos_in = _cos(u, v, m)
        cos_full = _cos(u, v, torch.ones_like(m))
        print(f"cos(u, v) inside mask: {cos_in:+.3f}")
        print(f"cos(u, v) full frame:  {cos_full:+.3f}")

    img_model = helper.decode_latent(z_model)
    img_oracle = helper.decode_latent(z_oracle)
    img_src = np.asarray(src)
    img_tgt = np.asarray(tgt)

    # 拼四张
    panel = np.concatenate([
        img_src,
        img_tgt,
        np.asarray(Image.fromarray(img_oracle).resize((512, 512))),
        np.asarray(Image.fromarray(img_model).resize((512, 512))),
    ], axis=1)
    Image.fromarray(panel).save(out / "synthetic_result.png")
    print(f"saved {out / 'synthetic_result.png'}")

    mask_np = np.asarray(mask)
    report = {
        "ckpt": args.ckpt, "r": args.r,
        "cos_in_mask": cos_in, "cos_full": cos_full,
        "oracle": in_mask_colour(
            np.asarray(Image.fromarray(img_oracle).resize((512, 512))), mask_np),
        "model": in_mask_colour(
            np.asarray(Image.fromarray(img_model).resize((512, 512))), mask_np),
    }
    (out / "clear_edit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()