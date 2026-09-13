"""
End-to-end assembly: request -> plan -> (inversion, biased generation).

The plan is deliberately a plain dataclass that can be built and inspected
without torch, SAM or a checkpoint. Every silent-failure mode in this project
is a shape or ordering mistake, so being able to dry-run the whole decision
chain on ground-truth masks is worth the small amount of ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lambda_field import (
    DEFAULT_PIXELS_PER_TOKEN,
    RegionFieldConfig,
    build_region_field,
    schedule_scale,
)
from .masking import TargetRegion, resolve_target
from .prompts import BAND_BY_TYPE, build_token_masks, infer_edit_type


@dataclass
class EditRequest:
    source_prompt: str
    target_prompt: str
    image: np.ndarray | None = None
    box: tuple | None = None
    ground_truth_mask: np.ndarray | None = None
    # Override the inferred band radius. Set explicitly during the r-sweep.
    band_tokens: float | None = None
    lam_in: float = 4.0
    lam_out: float = -2.0
    n_steps: int = 4


@dataclass
class EditPlan:
    region: TargetRegion
    field: np.ndarray               # (h_tok, w_tok)
    field_flat: np.ndarray          # (N_img,) raster order -- VERIFY vs patchifier
    edit_tokens: np.ndarray | None  # (N_txt,) bool, None in tokenizer-free dry run
    preserve_tokens: np.ndarray | None
    edit_type: str
    cfg: RegionFieldConfig
    n_steps: int

    def describe(self) -> str:
        h, w = self.field.shape
        lines = [
            f"edit type      : {self.edit_type}",
            f"band radius    : {self.cfg.band_tokens} tokens",
            f"region         : {self.region.report()}",
            f"field grid     : {h}x{w} = {h*w} image tokens",
            f"field range    : [{self.field.min():.2f}, {self.field.max():.2f}]",
            f"in-target toks : {(self.field > self.cfg.lam_in*0.95).sum()}",
            f"band tokens    : {((self.field > self.cfg.lam_out + 0.05) & (self.field <= self.cfg.lam_in*0.95)).sum()}",
            f"steps          : {self.n_steps}",
        ]
        if self.edit_tokens is not None:
            lines.append(
                f"text tokens    : {int(self.edit_tokens.sum())} edit / "
                f"{int(self.preserve_tokens.sum())} preserve"
            )
        if self.region.needs_crop_path:
            lines.append("ROUTE          : crop-and-zoom (target too small for graded field)")
        return "\n".join(lines)


def plan_edit(
    req: EditRequest,
    tokenizers: dict | None = None,
    segmenter: Any = None,
    pixels_per_token: int = DEFAULT_PIXELS_PER_TOKEN,
) -> EditPlan:
    """Build everything needed for a biased generation pass.

    tokenizers may be None for a dry run; token masks are then left unset.
    """
    region = resolve_target(
        image=req.image,
        box=req.box,
        ground_truth_mask=req.ground_truth_mask,
        segmenter=segmenter,
        pixels_per_token=pixels_per_token,
    )

    edit_type = infer_edit_type(req.source_prompt, req.target_prompt)
    band = req.band_tokens if req.band_tokens is not None else BAND_BY_TYPE[edit_type]

    cfg = RegionFieldConfig(
        band_tokens=band, lam_in=req.lam_in, lam_out=req.lam_out
    )
    field = build_region_field(region.mask, cfg, pixels_per_token)

    edit_tok = preserve_tok = None
    if tokenizers is not None:
        tm = build_token_masks(req.source_prompt, req.target_prompt, tokenizers)
        edit_tok, preserve_tok = tm.edit, tm.preserve

    return EditPlan(
        region=region,
        field=field,
        # Row-major flatten. The patch embedder's order is the assumption to
        # check first with a single-corner-token degenerate field.
        field_flat=field.reshape(-1),
        edit_tokens=edit_tok,
        preserve_tokens=preserve_tok,
        edit_type=edit_type,
        cfg=cfg,
        n_steps=req.n_steps,
    )


def step_scales(plan: EditPlan, mode: str = "front") -> list[float]:
    """Per-step multiplier on the field across the generation trajectory."""
    n = max(plan.n_steps, 1)
    return [schedule_scale((i + 0.5) / n, mode) for i in range(n)]


def run_edit(plan: EditPlan, model, invert_fn, generate_fn, source_latent=None):
    """Reference control flow. `model` is a region-wrapped MMDiT.

    Kept as a skeleton rather than a working call because the two function
    arguments are checkpoint-specific. The ordering below is the part that
    matters and is easy to get wrong.
    """
    from .attention import RegionAttentionSpec, set_region_all
    import torch

    # 1. Inversion runs UNBIASED. Biasing here corrupts the latent we are
    #    trying to recover and destroys background preservation.
    set_region_all(model, None, direction="invert")
    z = source_latent if source_latent is not None else invert_fn(model)

    # 2. Generation runs biased, with a per-step scale.
    field = torch.as_tensor(plan.field_flat)[None]
    edit = torch.as_tensor(plan.edit_tokens)[None]
    keep = torch.as_tensor(plan.preserve_tokens)[None]

    out = z
    for scale in step_scales(plan):
        spec = RegionAttentionSpec(
            field=field, edit_token_mask=edit,
            preserve_token_mask=keep, scale=scale,
        )
        set_region_all(model, spec, direction="generate")
        out = generate_fn(model, out)
    return out
