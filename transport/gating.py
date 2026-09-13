"""
Gated low-rank adaptation.

Every trainable path added to SD3 in Stage 1 is multiplied by a per-sample
scalar gate and zero-initialised on its output side. Two consequences, both
load-bearing:

1. AT INITIALISATION THE WRAPPED MODEL IS BIT-IDENTICAL TO PRETRAINED SD3.
   Not "close to". Identical. Every Stage 0 number therefore remains a valid
   control for Stage 1, and any change measured after training is attributable
   to training rather than to wrapping. Stage 0 lost three separate days to
   measurements that turned out to be artifacts of the harness; this is the
   cheap structural defence against a repeat.

2. THE BOUNDARY CONDITION IS ARCHITECTURAL, NOT LEARNED.
   MeanFlow requires u(z, t, t, c) = v(z, t, c): average velocity over a
   zero-width interval is instantaneous velocity. Gating every new path on
   Delta = t - r makes that hold exactly, for all parameter values, at every
   point in training. Learning it instead would spend capacity on a constraint
   we already know, and would leave it approximate exactly where the curriculum
   starts.

The gate is carried in a mutable holder rather than threaded through every
call signature, because the layers being wrapped live several frames down
inside diffusers and their signatures are not ours to change. `gate_scope`
sets it for the duration of one forward pass. Anything that reads the gate
outside a scope is a bug and raises.

CFG DOUBLES THE BATCH. The holder stores a (B,) tensor; if the caller builds a
[uncond, cond] batch it must set a gate of length 2B. `TransportAdapter` does
this for you. Direct users of the holder must not forget it -- a silently
broadcast scalar would apply the uncond gate to the cond half.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn


class GateHolder:
    """Per-forward gate, shared by every adapted layer in one model.

    GRADIENT CHECKPOINTING RE-RUNS THE FORWARD DURING BACKWARD, and that
    recomputation happens outside any context manager the original forward was
    wrapped in. A strictly scoped gate therefore raises on the recompute pass
    -- which is what killed Gate A's first launch, from inside
    torch.utils.checkpoint's unpack_hook.

    So the gate VALUE persists after its scope exits, and only the "never set
    at all" case raises. That is the case worth protecting: an adapted layer
    running before any gate was established is a wiring bug and would silently
    apply the wrong scaling. A value left over from the immediately preceding
    forward is exactly what the recompute needs.

    The residual risk is a stale gate applied to a forward that forgot to open
    a scope. `stale_reads` counts those so they are visible rather than
    silent, and `assert_fresh()` makes the check available where it matters
    (check_identity uses it). In practice every entry point into the
    transformer goes through TransportAdapter.forward, which always opens a
    scope, so a stale read means either a recompute -- expected -- or a new
    call path that needs one.
    """

    def __init__(self) -> None:
        self._gate: torch.Tensor | None = None
        self._ever_set: bool = False
        self._depth: int = 0
        self.stale_reads: int = 0

    @property
    def gate(self) -> torch.Tensor | None:
        if not self._ever_set:
            raise RuntimeError(
                "gate read before any gate_scope() was entered; an adapted "
                "layer ran with no gate established. Every path into the "
                "transformer must go through TransportAdapter.forward."
            )
        if self._depth == 0:
            # Outside a live scope: a checkpoint recompute, or a missing
            # scope. Same value either way; count it so it is not invisible.
            self.stale_reads += 1
        return self._gate

    def assert_fresh(self, where: str = "") -> None:
        if self._depth == 0:
            raise RuntimeError(
                f"expected to be inside gate_scope() {where}".strip())

    @contextlib.contextmanager
    def scope(self, gate: torch.Tensor | float | None):
        """Set the gate for one forward pass.

        gate=None disables every adapted path entirely (exact base model).
        A float is broadcast; a tensor must be shape (B,) matching the batch
        actually passed to the transformer, CFG doubling included.

        The value is deliberately NOT cleared on exit -- see the class
        docstring. The previous value is restored so nested scopes behave.
        """
        prev, prev_depth = self._gate, self._depth
        if isinstance(gate, (int, float)):
            gate = torch.tensor(float(gate))
        self._gate, self._ever_set = gate, True
        self._depth = prev_depth + 1
        try:
            yield
        finally:
            self._depth = prev_depth
            if prev_depth > 0:
                self._gate = prev      # restore only for genuine nesting


def _shape_gate(g: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a (B,) gate against a (B, ..., C) activation."""
    if g.dim() == 0:
        return g.to(ref.dtype).to(ref.device)
    if g.shape[0] != ref.shape[0]:
        raise ValueError(
            f"gate batch {g.shape[0]} != activation batch {ref.shape[0]}. "
            "With CFG the gate must be repeated to cover [uncond, cond]."
        )
    return g.reshape((-1,) + (1,) * (ref.dim() - 1)).to(ref.dtype).to(ref.device)


class GatedLoRALinear(nn.Module):
    """nn.Linear + gate * (B @ A) x, with B zero-initialised.

    Deliberately not peft. Three reasons: the per-sample gate is not something
    peft exposes cleanly, the identity guarantee needs to be auditable in one
    screen of code, and Stage 0 already showed how much a silently dropped
    argument costs (JointAttnProcessor2_0 accepting attention_mask and
    ignoring it invalidated every region number taken before it was caught).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float,
                 holder: GateHolder) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.holder = holder

        dtype = torch.float32   # adapter params stay fp32; see train.py
        self.lora_a = nn.Parameter(
            torch.randn(self.rank, base.in_features, dtype=dtype) / self.rank
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        g = self.holder.gate
        if g is None:
            return out
        h = torch.nn.functional.linear(x.to(self.lora_a.dtype), self.lora_a)
        d = torch.nn.functional.linear(h, self.lora_b) * self.scale
        d = d.to(out.dtype) * _shape_gate(g, out)
        return out + d


@dataclass
class InjectionReport:
    """What was actually wrapped. Print it; do not assume it."""

    wrapped: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"wrapped {len(self.wrapped)} Linear layers "
            f"({len(self.skipped)} candidates skipped)\n"
            + "\n".join(f"  + {n}" for n in self.wrapped[:8])
            + ("\n  ..." if len(self.wrapped) > 8 else "")
        )


#: SD3 MMDiT attention projections. Both streams: `to_*` is the image branch,
#: `add_*_proj` the text branch. Feed-forward layers are excluded by default --
#: they are the bulk of the parameters and Stage 1 is capacity-limited by data,
#: not by rank.
DEFAULT_TARGETS = (
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
)


def inject_lora(
    module: nn.Module,
    holder: GateHolder,
    rank: int = 16,
    alpha: float = 16.0,
    targets: tuple = DEFAULT_TARGETS,
    max_blocks: int | None = None,
) -> InjectionReport:
    """Replace matching nn.Linear layers in-place with GatedLoRALinear.

    max_blocks limits adaptation to the LAST n transformer blocks. On 24GB the
    full 24-block adaptation is affordable at rank 16, but restricting to the
    late blocks is a useful ablation: if a late-block-only adapter reproduces
    the effect, the claim that the transport is a shallow readout rather than a
    representational change gets much stronger.
    """
    report = InjectionReport()

    # Resolve block indices so max_blocks can be applied by name.
    names = [n for n, m in module.named_modules() if isinstance(m, nn.Linear)]
    block_ids = set()
    for n in names:
        for part in n.split("."):
            if part.isdigit():
                block_ids.add(int(part))
                break
    keep_from = -1
    if max_blocks is not None and block_ids:
        keep_from = max(block_ids) - max_blocks + 1

    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not any(name.endswith(t) for t in targets):
            continue
        if keep_from >= 0:
            idx = next((int(p) for p in name.split(".") if p.isdigit()), None)
            if idx is not None and idx < keep_from:
                report.skipped.append(name)
                continue

        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, GatedLoRALinear(child, rank, alpha, holder))
        report.wrapped.append(name)

    if not report.wrapped:
        raise RuntimeError(
            "inject_lora matched nothing. The diffusers attention naming has "
            "changed; print [n for n,m in transformer.named_modules() "
            "if isinstance(m, nn.Linear)] and update DEFAULT_TARGETS."
        )
    return report


def adapter_parameters(module: nn.Module):
    """Every parameter Stage 1 trains. Base weights are excluded by construction."""
    for p in module.parameters():
        if p.requires_grad:
            yield p
