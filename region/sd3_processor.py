"""
A joint-attention processor for SD3 that honours the region bias.

WHY THIS EXISTS. The obvious approach -- wrap the block's `Attention` module and
pass `attention_mask=bias` -- does not work. diffusers' `JointAttnProcessor2_0`
accepts an `attention_mask` argument and then calls
`scaled_dot_product_attention` without forwarding it. The bias is dropped
silently: no exception, no warning, and every sweep number would be measured
against a model that never saw the field.

So the processor is replaced rather than the module wrapped. This mirrors
diffusers' joint processor, with two additions: the additive bias, and a
direction gate so inversion runs unbiased.

Verify against your installed diffusers version before trusting it -- the
reference implementation changes between releases. `assert_bias_applied` below
is the guard.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import RegionAttentionSpec, build_joint_bias


class RegionJointAttnProcessor:
    """Drop-in replacement for JointAttnProcessor2_0 with an additive bias.

    Install with:
        proc = RegionJointAttnProcessor()
        for block in transformer.transformer_blocks:
            block.attn.set_processor(proc)      # one shared instance is fine
    """

    def __init__(self):
        self.spec: RegionAttentionSpec | None = None
        self.direction: str = "generate"
        self.last_bias_norm: float | None = None   # for assert_bias_applied
        # Diagnostics. When capture_layer matches the block index, the
        # image-query -> text-key attention block is stored. Off by default:
        # materialising attention probabilities defeats the fused kernel and
        # costs real memory.
        self.capture: bool = False
        self.captured: torch.Tensor | None = None  # (n_img, n_txt), head-mean
        self._block_idx = 0
        self.capture_layer: int = 0

    def set_region(self, spec: RegionAttentionSpec | None, direction="generate"):
        self.spec = spec
        self.direction = direction
        self._block_idx = 0        # reset each forward pass

    def _maybe_capture(self, query, key, bias, n_img):
        """Store head-averaged attention from image queries to text keys.

        Computed explicitly rather than via SDPA because the fused kernel does
        not return probabilities. Run at reduced resolution -- at 1024px the
        full attention rows are several GB.
        """
        import math

        scale = 1.0 / math.sqrt(query.shape[-1])
        # image queries occupy the FIRST n_img positions; text keys the rest
        q_img = query[:, :, :n_img, :].float()
        logits = torch.matmul(q_img, key.float().transpose(-2, -1)) * scale
        if bias is not None:
            logits = logits + bias[:, :, :n_img, :].float()
        probs = logits.softmax(-1)
        self.captured = probs[0, :, :, n_img:].mean(0).detach().cpu()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        batch = hidden_states.shape[0]
        n_img = hidden_states.shape[1]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        heads = attn.heads
        head_dim = query.shape[-1] // heads

        def split(t):
            return t.view(batch, -1, heads, head_dim).transpose(1, 2)

        query, key, value = split(query), split(key), split(value)

        # QK-norm is PER STREAM. diffusers applies norm_q/norm_k to the image
        # tokens and separate norm_added_q/norm_added_k to the text tokens,
        # each BEFORE concatenation. Normalising the concatenated sequence with
        # the image norms (and never calling the added norms) silently corrupts
        # the model -- it produced a periodic striping artefact in the VAE
        # output that looked like inversion drift.
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        n_txt = 0
        if encoder_hidden_states is not None:
            n_txt = encoder_hidden_states.shape[1]
            enc_q = split(attn.add_q_proj(encoder_hidden_states))
            enc_k = split(attn.add_k_proj(encoder_hidden_states))
            enc_v = split(attn.add_v_proj(encoder_hidden_states))

            if getattr(attn, "norm_added_q", None) is not None:
                enc_q = attn.norm_added_q(enc_q)
            if getattr(attn, "norm_added_k", None) is not None:
                enc_k = attn.norm_added_k(enc_k)

            # IMAGE FIRST, then text -- matching diffusers' concatenation order.
            # build_joint_bias must use the same convention.
            query = torch.cat([query, enc_q], dim=2)
            key = torch.cat([key, enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        # --- the whole point of this class ---
        bias = attention_mask
        if self.spec is not None and self.direction == "generate" and n_txt:
            bias = build_joint_bias(self.spec, n_txt, n_img).to(query.dtype)
            self.last_bias_norm = float(bias.abs().sum().item())

        if self.capture and self._block_idx == self.capture_layer and n_txt:
            self._maybe_capture(query, key, bias, n_img)
        self._block_idx += 1

        hidden = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        hidden = hidden.transpose(1, 2).reshape(batch, -1, heads * head_dim)
        hidden = hidden.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden, enc_out = hidden[:, :n_img], hidden[:, n_img:]
            hidden = attn.to_out[0](hidden)
            hidden = attn.to_out[1](hidden)
            if not attn.context_pre_only:
                enc_out = attn.to_add_out(enc_out)
            return hidden, enc_out

        hidden = attn.to_out[0](hidden)
        hidden = attn.to_out[1](hidden)
        return hidden + 0.0 * residual.sum() * 0  # keep residual in graph


def install(transformer) -> RegionJointAttnProcessor:
    """Replace every joint-attention processor. Returns the shared instance."""
    proc = RegionJointAttnProcessor()
    n = 0
    for block in transformer.transformer_blocks:
        if hasattr(block, "attn"):
            block.attn.set_processor(proc)
            n += 1
    print(f"[region] installed processor on {n} blocks")
    if n == 0:
        raise RuntimeError(
            "no transformer_blocks with .attn found -- check the model class"
        )
    return proc


def assert_bias_applied(proc: RegionJointAttnProcessor) -> None:
    """Confirm the bias actually reached scaled_dot_product_attention.

    The failure this catches is silent by construction: a processor that
    accepts the mask and ignores it produces perfectly normal images. Call
    after a forward pass with a non-zero field.
    """
    if proc.last_bias_norm is None:
        raise RuntimeError(
            "no bias was applied. Either set_region was never called, the "
            "direction was 'invert', or the processor is not installed."
        )
    if proc.last_bias_norm == 0.0:
        raise RuntimeError(
            "bias was applied but is identically zero -- check that "
            "edit_token_mask has any True entries."
        )
