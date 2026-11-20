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
NOT RUN ON A GPU when written; the adapter is tested with a fake pipe (self-test).
"""

from __future__ import annotations

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


def load_qwen(repo: str, mode: str = "nf4", device: str = "cuda"):
    import torch  # noqa: PLC0415
    from diffusers import QwenImageEditPlusPipeline  # noqa: PLC0415
    if mode == "nf4":
        from diffusers import BitsAndBytesConfig as DBnb  # noqa: PLC0415
        from diffusers import QwenImageTransformer2DModel  # noqa: PLC0415
        from transformers import BitsAndBytesConfig as TBnb  # noqa: PLC0415
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415
        q = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.bfloat16)
        transformer = QwenImageTransformer2DModel.from_pretrained(
            repo, subfolder="transformer", quantization_config=DBnb(**q),
            torch_dtype=torch.bfloat16)
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo, subfolder="text_encoder", quantization_config=TBnb(**q),
            torch_dtype=torch.bfloat16)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            repo, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
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
    print("SELF-TEST OK")


if __name__ == "__main__":
    self_test()
