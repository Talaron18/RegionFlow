"""Qwen-Image-Edit (2511 / 2509) behind the same call contract as Flux2KleinModel, so
`fresh_cases.py --edit --editor qwen` runs the identical plain / paste-back / crop-box
arms on the frozen arm-T cases.

Settings are the model card's (Qwen/Qwen-Image-Edit-2511 README, QwenImageEditPlusPipeline):
true_cfg_scale 4.0, negative_prompt " ", num_inference_steps 40, guidance_scale 1.0.

MEMORY. bf16 is about 40 GB of transformer plus a 7B text encoder: it does not fit a
24 GB card. Two ways, chosen with --qwen-mode:
  nf4        transformer and text encoder loaded in 4-bit (bitsandbytes), model CPU
             offload. Fits 24 GB; fast; 4-bit changes outputs slightly (report it).
  sequential bf16 with sequential CPU offload: exact weights, very slow, needs a lot of
             CPU RAM (~60 GB). Use only if nf4 is not acceptable.
Quantising both components in one process on the fly is heavy on RAM and a slow disk.
Quantise each once, in its own process, then load the saved 4-bit weights:
  python bench/qwen_edit_model.py --quantize transformer  --repo <snapshot> --out <dir>
  python bench/qwen_edit_model.py --quantize text_encoder --repo <snapshot> --out <dir>
  python bench/fresh_cases.py ... --editor qwen --qwen-mode nf4 --qwen-nf4-dir <dir>
SKIP presets (--skip): transformer modules kept in bf16 via llm_int8_skip_modules, because
NF4 on the modulation / embedding / output layers causes grainy images (text encoder unchanged).
NOT RUN ON A GPU when written; the adapter is tested with a fake pipe (self-test).
"""

from __future__ import annotations

SKIP_PRESETS = {
    # as in diffusers' Qwen-Image-Layered config
    "block0": ["transformer_blocks.0.img_mod", "img_in", "txt_in", "norm_out", "proj_out",
               "time_text_embed"],
    # as in diffusers/qwen-image-nf4 (all blocks' modulation in bf16; much larger)
    "allmod": ["img_mod", "txt_mod", "img_in", "txt_in", "norm_out", "proj_out",
               "time_text_embed"],
    "none": [],
}

QWEN_DEFAULTS = {"true_cfg_scale": 4.0, "negative_prompt": " ",
                 "num_inference_steps": 40, "guidance_scale": 1.0}


class QwenKleinAdapter:
    """Callable with Flux2KleinPipeline's keyword signature, forwarding to a
    QwenImageEditPlusPipeline. Flux2KleinModel (plain and crop modes) calls
    pipe(prompt=, image=, height=, width=, num_inference_steps=, guidance_scale=,
    generator=); the klein step count and guidance are replaced by Qwen's own."""

    def __init__(self, pipe, settings=None):
        self.pipe = pipe
        self.settings = dict(QWEN_DEFAULTS, **(settings or {}))

    def __call__(self, prompt, image, height, width, num_inference_steps=None,
                 guidance_scale=None, generator=None, **_):
        s = self.settings
        return self.pipe(image=[image], prompt=prompt, height=height, width=width,
                         true_cfg_scale=s["true_cfg_scale"],
                         negative_prompt=s["negative_prompt"],
                         num_inference_steps=s["num_inference_steps"],
                         guidance_scale=s["guidance_scale"], generator=generator)


def quantize_component(component: str, repo: str, out: str, skip: str = "block0"):
    """Load ONE component in NF4 from the snapshot, save it to <out>/<component>.
    For the transformer, SKIP_PRESETS[skip] modules stay in bf16."""
    import os  # noqa: PLC0415
    import torch  # noqa: PLC0415
    q = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
             bnb_4bit_compute_dtype=torch.bfloat16)
    if component == "transformer":
        from diffusers import BitsAndBytesConfig as DBnb  # noqa: PLC0415
        from diffusers import QwenImageTransformer2DModel  # noqa: PLC0415
        model = QwenImageTransformer2DModel.from_pretrained(
            repo, subfolder="transformer",
            quantization_config=DBnb(llm_int8_skip_modules=SKIP_PRESETS[skip], **q),
            torch_dtype=torch.bfloat16)
        n4, nbf, pbf, names = 0, 0, 0, []
        for name, m in model.named_modules():
            if "Linear4bit" in type(m).__name__:
                n4 += 1
            elif isinstance(m, torch.nn.Linear):
                nbf += 1
                pbf += sum(p.numel() for p in m.parameters())
                if len(names) < 10:
                    names.append(name)
        print(f"skip={skip}: Linear4bit {n4}; bf16 nn.Linear {nbf} "
              f"({pbf / 1e9:.3f}B params); first bf16: {names}")
    elif component == "text_encoder":
        from transformers import BitsAndBytesConfig as TBnb  # noqa: PLC0415
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo, subfolder="text_encoder", quantization_config=TBnb(**q),
            torch_dtype=torch.bfloat16, device_map={"": 0})
    else:
        raise SystemExit(f"unknown component {component!r}")
    dest = os.path.join(out, component)
    model.save_pretrained(dest)
    peak = torch.cuda.max_memory_allocated() / 2**30
    size = sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(dest) for f in fs) / 2**30
    print(f"{component}: peak GPU GB {peak:.2f}; saved {dest} ({size:.2f} GB)")


def load_qwen(repo: str, mode: str = "nf4", device: str = "cuda", nf4_dir=None):
    import torch  # noqa: PLC0415
    from diffusers import QwenImageEditPlusPipeline  # noqa: PLC0415
    if mode == "nf4":
        from diffusers import BitsAndBytesConfig as DBnb  # noqa: PLC0415
        from diffusers import QwenImageTransformer2DModel  # noqa: PLC0415
        from transformers import BitsAndBytesConfig as TBnb  # noqa: PLC0415
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415
        if nf4_dir:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                f"{nf4_dir}/transformer", torch_dtype=torch.bfloat16)
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                f"{nf4_dir}/text_encoder", torch_dtype=torch.bfloat16)
        else:
            q = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                     bnb_4bit_compute_dtype=torch.bfloat16)
            transformer = QwenImageTransformer2DModel.from_pretrained(
                repo, subfolder="transformer",
                quantization_config=DBnb(
                    llm_int8_skip_modules=SKIP_PRESETS["block0"], **q),
                torch_dtype=torch.bfloat16)
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repo, subfolder="text_encoder", quantization_config=TBnb(**q),
                torch_dtype=torch.bfloat16)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            repo, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        print(f"resident GB after load: {torch.cuda.memory_allocated() / 2**30:.2f}")
    elif mode == "sequential":
        pipe = QwenImageEditPlusPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
    elif mode == "full":
        pipe = QwenImageEditPlusPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        pipe.to(device)
    else:
        raise SystemExit(f"unknown --qwen-mode {mode!r}")
    pipe.set_progress_bar_config(disable=True)
    return QwenKleinAdapter(pipe)


def self_test():
    from types import SimpleNamespace
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return SimpleNamespace(images=["out"])
    a = QwenKleinAdapter(fake)
    r = a(prompt="p", image="IMG", height=768, width=1024, num_inference_steps=4,
          guidance_scale=1.0, generator="g")
    assert r.images == ["out"]
    assert seen["image"] == ["IMG"] and seen["num_inference_steps"] == 40
    assert seen["true_cfg_scale"] == 4.0 and seen["negative_prompt"] == " "
    assert (seen["height"], seen["width"], seen["generator"]) == (768, 1024, "g")
    assert set(SKIP_PRESETS) == {"block0", "allmod", "none"} and SKIP_PRESETS["none"] == []
    assert "img_mod" in SKIP_PRESETS["allmod"] and "img_mod" not in SKIP_PRESETS["block0"]
    print("SELF-TEST OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quantize", choices=("transformer", "text_encoder"))
    ap.add_argument("--repo")
    ap.add_argument("--out")
    ap.add_argument("--skip", choices=tuple(SKIP_PRESETS), default="block0",
                    help="transformer modules kept in bf16 with --quantize transformer")
    a = ap.parse_args()
    if a.quantize:
        if not (a.repo and a.out):
            ap.error("--quantize needs --repo and --out")
        quantize_component(a.quantize, a.repo, a.out, a.skip)
    else:
        self_test()
