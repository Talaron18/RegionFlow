"""
Turning a (source prompt, target prompt) pair into token-level masks.

The bias in `region.attention` needs to know which positions in the MMDiT
context sequence are *edit* tokens and which are *preserve* tokens. Getting
this wrong is silent: the field lands on the wrong words and the edit either
does nothing or smears.

Two things make this less trivial than a set difference.

1. SUBWORD AND REPETITION. "a green truck" -> "a green bus" must mark only the
   final content token. Set difference over token ids breaks on repeated words
   ("a red car and a red bus" -> "...a blue bus") and on subword splits. We use
   a sequence alignment instead, so insertions and replacements are located
   positionally.

2. SD3 HAS THREE TEXT ENCODERS. The context sequence MMDiT attends over is the
   concatenation of CLIP-L and CLIP-G token embeddings (projected up) with T5
   embeddings. Each encoder tokenises differently and has its own padded
   length, so the mask must be built per-encoder and concatenated in exactly
   the order the pipeline builds the context. Verify that order against the
   model you actually load -- it is the second most likely silent bug in this
   codebase after raster order.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from difflib import SequenceMatcher

import numpy as np


@dataclass
class EncoderSpec:
    """One text encoder's contribution to the context sequence."""

    name: str
    max_length: int
    # Set False for encoders whose token embeddings are pooled into the global
    # conditioning vector rather than placed in the sequence.
    in_sequence: bool = True


# CORRECTED against the SD3/SD3.5 diffusers config.
#
# The two CLIP token embeddings are concatenated along the CHANNEL dimension
# and zero-padded to T5's width (4096), and only THEN is that combined CLIP
# sequence concatenated with the T5 sequence along the SEQUENCE dimension.
# The pooled CLIP embeddings go to global conditioning with the timestep, not
# into the sequence at all.
#
# So the context is 77 + 256 = 333 tokens, NOT 77 + 77 + 256 = 410.
#
# CLIP-L and OpenCLIP-G share the same BPE tokenizer, so position i in the
# combined CLIP block corresponds to position i in both. One tokenizer is
# enough to locate edit tokens there.
SD3_ENCODERS = [
    EncoderSpec("clip", 77),    # CLIP-L + CLIP-G, merged on channels
    EncoderSpec("t5", 256),
]

# Kept for reference: this is the layout to use ONLY if you verify that a
# particular checkpoint really does place both CLIP sequences separately.
# Check `transformer.config.joint_attention_dim` and the encoder output shape
# before trusting either; `assert_context_length` below is the real guard.
SD3_ENCODERS_SEPARATE_CLIP = [
    EncoderSpec("clip_l", 77),
    EncoderSpec("clip_g", 77),
    EncoderSpec("t5", 256),
]


def assert_context_length(masks, encoder_hidden_states) -> None:
    """Fail loudly if the token mask does not match the real context sequence.

    A silent length mismatch here misaligns the bias against the prompt, which
    produces edits that look wrong but never raise. Call this once at startup
    with a real forward pass's encoder_hidden_states.
    """
    got = encoder_hidden_states.shape[1]
    want = masks.edit.size
    if got != want:
        raise ValueError(
            f"context length mismatch: model produced {got} text tokens, "
            f"token masks are {want}. Fix SD3_ENCODERS to match the checkpoint "
            f"(333 = merged CLIP + T5; 410 = separate CLIP sequences)."
        )


@dataclass
class TokenMasks:
    edit: np.ndarray      # (N_txt,) bool
    preserve: np.ndarray  # (N_txt,) bool
    per_encoder: dict = _field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{int(self.edit.sum())} edit / {int(self.preserve.sum())} preserve "
            f"/ {self.edit.size} total"
        )


def _align_masks(src_ids, tgt_ids):
    """Positional edit/preserve masks over `tgt_ids`.

    Returns (edit, preserve), both length len(tgt_ids).

    'replace' and 'insert' spans in the target are edit tokens. 'equal' spans
    are preserve tokens. Deletions contribute nothing to the target.
    """
    n = len(tgt_ids)
    edit = np.zeros(n, dtype=bool)
    preserve = np.zeros(n, dtype=bool)

    sm = SequenceMatcher(a=list(src_ids), b=list(tgt_ids), autojunk=False)
    for op, _i1, _i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            preserve[j1:j2] = True
        elif op in ("replace", "insert"):
            edit[j1:j2] = True
    return edit, preserve


def _strip_special(ids, special_ids):
    keep = [k for k, t in enumerate(ids) if t not in special_ids]
    return keep


def build_token_masks(
    source_prompt: str,
    target_prompt: str,
    tokenizers: dict,
    encoders=SD3_ENCODERS,
    mark_padding_as_neither: bool = True,
) -> TokenMasks:
    """Build the concatenated edit/preserve masks for the context sequence.

    Args:
        source_prompt: prompt describing the image as it is.
        target_prompt: prompt describing the desired result.
        tokenizers: {encoder_name: HF tokenizer}. Each must expose
            `__call__(text, padding='max_length', max_length=..., truncation=True)`
            and `all_special_ids`.
        encoders: sequence of EncoderSpec in context-construction order.
        mark_padding_as_neither: pad/special positions get neither flag, so
            they receive zero bias.

    Returns:
        TokenMasks with concatenated boolean arrays.
    """
    edits, preserves, per_enc = [], [], {}

    for spec in encoders:
        if not spec.in_sequence:
            continue
        tok = tokenizers[spec.name]
        kw = dict(padding="max_length", max_length=spec.max_length, truncation=True)
        src = tok(source_prompt, **kw)["input_ids"]
        tgt = tok(target_prompt, **kw)["input_ids"]
        special = set(getattr(tok, "all_special_ids", []))

        # Align on content tokens only, then scatter back to padded positions.
        src_keep = _strip_special(src, special)
        tgt_keep = _strip_special(tgt, special)
        e_c, p_c = _align_masks([src[k] for k in src_keep], [tgt[k] for k in tgt_keep])

        e = np.zeros(spec.max_length, dtype=bool)
        p = np.zeros(spec.max_length, dtype=bool)
        for slot, pos in enumerate(tgt_keep):
            e[pos] = e_c[slot]
            p[pos] = p_c[slot]

        if not mark_padding_as_neither:
            p[[k for k in range(spec.max_length) if k not in tgt_keep]] = True

        edits.append(e)
        preserves.append(p)
        per_enc[spec.name] = (e, p)

    return TokenMasks(
        edit=np.concatenate(edits),
        preserve=np.concatenate(preserves),
        per_encoder=per_enc,
    )


def infer_edit_type(source_prompt: str, target_prompt: str) -> str:
    """Coarse edit-type guess -> feeds the band radius.

    Deliberately a stub with a keyword fallback. In v1 this is one call to the
    MLLM returning a single label; the whole planner reduces to this. Keeping
    the interface here now means v1 is a one-function swap, not a refactor.
    """
    COLOUR = {"red", "blue", "green", "yellow", "black", "white", "orange",
              "purple", "pink", "brown", "grey", "gray", "golden", "silver"}
    TEXTURE = {"wooden", "metal", "marble", "plaid", "bohemian", "desert",
               "grass", "leather", "glass", "brick", "furry", "striped"}
    LIGHT = {"sunset", "night", "shadow", "lit", "bright", "dark", "sunny",
             "cloudy", "dawn", "dusk", "neon"}

    new_words = set(target_prompt.lower().split()) - set(source_prompt.lower().split())
    if new_words & COLOUR:
        return "colour"
    if new_words & LIGHT:
        return "relight"
    if new_words & TEXTURE:
        return "texture"
    return "replace"


# Band radius in latent tokens, per edit type. See README band-semantics table.
BAND_BY_TYPE = {
    "colour": 1.0,
    "texture": 1.0,
    "replace": 3.0,
    "relight": 4.0,
}
