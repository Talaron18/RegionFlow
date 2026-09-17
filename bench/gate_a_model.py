"""
Gate A (Direction A) as a run_gate_c.py-compatible model.

Wraps TransportAdapter + apply_transport (transport/pairflow.py) behind the
same `model(img, prompt, seed, mask=None) -> PIL.Image` interface every
other run_gate_c.py model already implements (IdentityModel,
SD3InstructPix2Pix) -- so a Direction A checkpoint can be scored on the
same 151-case Gate C scoreboard (mis_targeted classifier, VLM judge,
off_instance) as every other arm. This is GATE_A_REOPENING_PLAN.md SS8's
own "next concrete piece of work once a checkpoint exists to score" --
several now do.

Reuses bench/inversion.py::SD3EditPipeline for VAE encode/decode and text
encoding, and transport/pairflow.py::apply_transport for the actual edit --
the SAME pieces scripts/gate_a_transport.py's own `eval` subcommand already
combines for its 3-case Stage-0-comparable sweep. This is that combination,
wired to the Gate C harness's interface instead of the decoupling sweep's.

USAGE (via bench/run_gate_c.py):
    python bench/run_gate_c.py --model gate-a-transport \
        --gate-a-ckpt runs/gate_a_sigma_0.3/ckpt_final.pt \
        --gate-a-sigma0 0.3 --mask-mode oracle \
        --out runs/gate_c/gate_a_sigma0.3_oracle.jsonl

sigma0 HERE IS INFERENCE-TIME SAMPLING (apply_transport's own mode-selecting
noise), not read from the checkpoint -- pass whatever value the run was
ACTUALLY TRAINED WITH. A mismatch queries the model at a point its own
training never saw as a fixed value, off its own training distribution
(apply_transport's own docstring covers this). Use sigma0=0.0 to score a
sigma0=0 checkpoint, or to read ANY checkpoint deterministically as a
comparison point.

`--mask-mode none` is refused, matching sd3-ultraedit-mask's own existing
guard in load_model(): the mask is a required tensor input to this
architecture (AdapterConfig(mode="pair", use_mask=True)), not an optional
argument diffusers' pipeline can be called without. `--mask-mode white`
IS the no-spatial-information control for this model, same reasoning as
already established for sd3-ultraedit-mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class GateATransportModel:
    """run_gate_c.py model interface: __call__(img, prompt, seed, mask) -> PIL.Image."""

    def __init__(self, ckpt_path: str, sigma0: float = 0.0, rank: int = 16,
                model: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                device: str = "cuda", steps_readout: int = 1):
        import torch  # noqa: PLC0415
        from diffusers import StableDiffusion3Pipeline  # noqa: PLC0415

        from bench.inversion import SD3EditPipeline  # noqa: PLC0415
        from transport.adapter import AdapterConfig, TransportAdapter  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.sigma0 = sigma0
        self.steps_readout = steps_readout
        self.name = f"gate-a-transport(sigma0={sigma0},ckpt={Path(ckpt_path).name})"

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model, torch_dtype=torch.bfloat16).to(device)
        self.helper = SD3EditPipeline(pipe, device=device)

        self.adapter = TransportAdapter(
            pipe.transformer, AdapterConfig(mode="pair", rank=rank)
        ).to(device)
        for p in self.adapter.trainable_parameters():
            p.data = p.data.float()

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _missing, unexpected = self.adapter.load_state_dict(sd["adapter"], strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint has unexpected keys: {unexpected[:5]}")
        # Same live-weight check cmd_eval already does -- a checkpoint whose
        # lora_b blocks are all still zero is the pretrained model, and
        # scoring it would produce a confident null (identical to Gate A's
        # own "adapter is inert" guard in cmd_train).
        trained = [k for k in sd["adapter"] if "lora_b" in k]
        live = sum(float(sd["adapter"][k].abs().sum()) > 0 for k in trained)
        print(f"[gate-a-transport] checkpoint step {sd.get('step', '?')}, "
             f"{live}/{len(trained)} lora_b blocks non-zero")
        if live == 0:
            raise RuntimeError(
                "every lora_b is still zero -- this checkpoint is the "
                "pretrained model, scoring it would produce a confident null")
        self.adapter.eval()

    def __call__(self, img, prompt, seed, mask=None):
        from transport.conditioning import mask_to_latent_grid  # noqa: PLC0415
        from transport.pairflow import apply_transport  # noqa: PLC0415

        if mask is None:
            raise SystemExit(
                "gate-a-transport requires a mask (the architecture takes it "
                "as a required tensor input) -- use --mask-mode white for "
                "the no-spatial-information control, not none")

        z = self.helper.encode_image(np.asarray(img))
        h, w = z.shape[-2:]
        m_np = np.asarray(mask.convert("L")) > 127  # polarity: white=editable
        m = mask_to_latent_grid(m_np, h, w).to(self.device)
        pe, _npe, pooled, _npool = self.helper.encode_prompt(prompt)

        gen = (self.torch.Generator(device=self.device).manual_seed(seed)
              if self.sigma0 > 0 else None)
        z_e = apply_transport(self.adapter, z.float(), m, pe, pooled,
                              strength=1.0, steps=self.steps_readout,
                              pin_background=True, sigma0=self.sigma0,
                              generator=gen)
        out = self.helper.decode_latent(z_e)
        return Image.fromarray(out)
