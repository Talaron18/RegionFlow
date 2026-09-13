"""
Injecting the region field into MMDiT joint attention.

SD3-style MMDiT runs full self-attention over the concatenated sequence
[text_tokens ; image_tokens]. The relationship we want to steer -- "this image
region should be driven by these prompt tokens" -- lives in the two
off-diagonal blocks of that matrix.

We add a bias before the softmax:

    attn = softmax(QK^T / sqrt(d) + B)

with B non-zero only in the text<->image blocks, and only for the edit tokens
(the tokens present in the target prompt but not the source prompt).

Two design points that are load-bearing:

1. ASYMMETRY ACROSS TIME DIRECTIONS. In a BiFM-style model the same weights run
   inversion (image -> noise) and generation (noise -> image). The bias must be
   applied on the GENERATION pass only. Inversion should be faithful
   everywhere; biasing it corrupts the latent you are trying to recover and
   destroys the background-preservation numbers that are the whole point of a
   learned inverse. Gate on `direction`.

2. BIAS, NOT MASK. B is additive on logits, not multiplicative on attention. A
   hard multiplicative mask makes background tokens unable to see the edit
   tokens at all, which is what produces the boundary discordance that
   mask-based inpainting baselines are criticised for. An additive bias only
   changes the competition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RegionAttentionSpec:
    """Everything a block needs to build its bias.

    field:
        (B, N_img) float tensor -- the pooled lambda field, flattened in the
        same raster order the patch embedder uses. Check this: a transposed
        field silently produces plausible-looking but wrong edits.
    edit_token_mask:
        (B, N_txt) bool -- True for tokens new in the target prompt.
    preserve_token_mask:
        (B, N_txt) bool or None -- tokens common to source and target. If
        given, they receive the inverted field, so background tokens hold the
        unedited content more firmly. Optional; ablate it.
    scale:
        Scalar multiplier from the timestep schedule.
    """

    field: torch.Tensor
    edit_token_mask: torch.Tensor
    preserve_token_mask: torch.Tensor | None = None
    scale: float = 1.0


def build_joint_bias(spec: RegionAttentionSpec, n_txt: int, n_img: int) -> torch.Tensor:
    """Assemble the (B, 1, N, N) additive bias over [text ; image].

    Returned in a head-broadcastable shape so it can be added to any
    scaled_dot_product_attention call site without per-head materialisation.
    """
    field = spec.field * spec.scale                       # (B, N_img)
    bsz = field.shape[0]
    n = n_txt + n_img
    device, dtype = field.device, field.dtype

    bias = torch.zeros(bsz, n, n, device=device, dtype=dtype)

    edit = spec.edit_token_mask.to(dtype)                 # (B, N_txt)

    # ORDER: diffusers concatenates [image, text], so image queries occupy the
    # first n_img rows and text keys the last n_txt columns. Getting this
    # backwards puts the bias in the text-text and image-image blocks, where
    # it does nothing useful and quietly corrupts the pass.
    block = field.unsqueeze(2) * edit.unsqueeze(1)        # (B, N_img, N_txt)
    bias[:, :n_img, n_img:] += block
    # edit-token queries attending back to image keys
    bias[:, n_img:, :n_img] += block.transpose(1, 2)

    if spec.preserve_token_mask is not None:
        keep = spec.preserve_token_mask.to(dtype)
        keep_block = (-field).unsqueeze(2) * keep.unsqueeze(1)
        bias[:, :n_img, n_img:] += keep_block
        bias[:, n_img:, :n_img] += keep_block.transpose(1, 2)

    return bias.unsqueeze(1)                              # (B, 1, N, N)


class RegionBiasedJointAttention(nn.Module):
    """Wraps an existing MMDiT joint-attention module.

    Keeps the wrapped module's projections untouched so LoRA adapters attach
    normally. Only the attention call is intercepted.

        for block in transformer.transformer_blocks:
            block.attn = RegionBiasedJointAttention(block.attn)
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._spec: RegionAttentionSpec | None = None
        self._direction: str = "generate"

    def set_region(self, spec: RegionAttentionSpec | None, direction: str = "generate"):
        self._spec = spec
        self._direction = direction

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        bias = None
        if self._spec is not None and self._direction == "generate":
            n_txt = encoder_hidden_states.shape[1]
            n_img = hidden_states.shape[1]
            bias = build_joint_bias(self._spec, n_txt, n_img)
        return self.inner(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=bias,
            **kwargs,
        )


def set_region_all(transformer: nn.Module, spec, direction: str = "generate"):
    """Broadcast a spec to every wrapped block."""
    for m in transformer.modules():
        if isinstance(m, RegionBiasedJointAttention):
            m.set_region(spec, direction)
