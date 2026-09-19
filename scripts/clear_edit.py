"""Synthetic unambiguous edit: white bg, red circle + green circle,
instruction 'change the red circle to blue', mask on red circle only.

If the model produces a clean blue circle, ambiguity was the bottleneck
and MSE is fine on unambiguous tasks. If it produces a smeared blue
blob, the problem is spatial precision / architecture, not ambiguity.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from bench.inversion import SD3EditPipeline
from diffusers import StableDiffusion3Pipeline

# --- 合成源图和目标图 ---
def make_pair(size=512):
    src = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(src)
    # 红圆
    d.ellipse([100, 100, 300, 300], fill=(220, 30, 30))
    # 绿圆
    d.ellipse([320, 320, 480, 480], fill=(30, 180, 30))

    tgt = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(tgt)
    # 蓝圆（替换红圆）
    d.ellipse([100, 100, 300, 300], fill=(30, 80, 220))
    # 绿圆不变
    d.ellipse([320, 320, 480, 480], fill=(30, 180, 30))

    # mask: 只有红圆区域
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([100, 100, 300, 300], fill=255)

    return src, tgt, mask


def main():
    device = "cuda"
    model = "stabilityai/stable-diffusion-3-medium-diffusers"

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16).to(device)
    helper = SD3EditPipeline(pipe, device=device)

    adapter = TransportAdapter(
        pipe.transformer,
        AdapterConfig(mode="pair", rank=16, mask_encoder_kind="single"),
    ).to(device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()

    sd = torch.load("runs/gate_a_sigma_0.0/ckpt_latest.pt",
                    map_location="cpu", weights_only=False)
    adapter.load_state_dict(sd["adapter"], strict=False)
    adapter.eval()

    src, tgt, mask = make_pair(512)
    src.save("synthetic_src.png")
    tgt.save("synthetic_tgt.png")
    mask.save("synthetic_mask.png")

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
        u = adapter(z_src, s0, s0, pe, pooled, mask=m)

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
        print(f"cos(u, v) inside mask: {_cos(u, v, m):+.3f}")
        print(f"cos(u, v) full frame:  {_cos(u, v, torch.ones_like(m)):+.3f}")

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
    Image.fromarray(panel).save("synthetic_result.png")
    print("saved synthetic_result.png")


if __name__ == "__main__":
    main()