"""
TransportAdapter: SD3-Medium wrapped as a conditional interval-transport model.

    u_theta(z, r, t, c, M)

with, by construction and for all parameter values:

    u_theta(z, t, t, c, M) == v_pretrained(z, t, c)          (boundary)
    u_theta at initialisation == v_pretrained                (identity init)

Both follow from gating every new path on Delta = t - r (see gating.py) and
zero-initialising every new output projection (see conditioning.py). Neither is
learned, neither drifts, and both are asserted by `check_identity()` -- run it
before every training run and after every checkpoint load.

TWO GATE MODES, because Stage 1 has two different experiments on one backbone:

  "interval"  gate = Delta.  The boundary condition is architectural: at
              Delta=0 the model IS SD3 and the adapter is inert.

              DO NOT USE THIS FOR MEANFLOW. The reference implementation
              trains 75% of every batch at r == t, where those samples reduce
              to plain flow matching and anchor the interval field to the
              instantaneous one. Under this gate their gradient is exactly
              zero, so the anchor population contributes nothing: the exact
              boundary guarantee costs three quarters of the training signal.
              Kept for ablation against "meanflow", not as the default.

  "meanflow"  gate = 1, u = u_base + Delta_u_theta, ungated. Reference-faithful.
              The boundary u(z,t,t,c) = v_base is no longer guaranteed by
              construction; it is TAUGHT by the r == t anchors and must be
              CHECKED empirically -- see check_identity, whose boundary_ok is
              informational rather than enforced in this mode.

  "pair"      gate = 1.  Gate A, source->target transport. There is no noise
              schedule and no boundary condition to respect -- t indexes
              progress along a straight path from source latent to target
              latent, and the adapter must be active at every t including t=0,
              since t=0 is exactly where the one-step edit is read out.

Get this wrong and Gate A trains a model that is inert at the only point it
will ever be evaluated at. It is the single easiest way to produce a confident
null result, so `mode` has no default.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from .conditioning import MaskEncoder, make_interval_embedding
from .gating import GateHolder, adapter_parameters, inject_lora


@contextlib.contextmanager
def no_checkpoint_without_grad(transformer):
    """Disable gradient checkpointing for the duration of a no-grad forward.

    THE RULE: checkpointing trades recompute for activation memory during a
    BACKWARD. With grad disabled there is no backward, nothing is stored, and
    the recompute buys nothing -- so it should be off. diffusers gates it on
    `self.training and self.gradient_checkpointing` only, and the transformer
    stays in .train() for the whole run (Trainer calls .eval() for validation
    alone), so every no-grad forward taken during a training step goes through
    checkpointing regardless.

    This module makes a LOT of those: central_difference's two probes, the
    checkpoint_safe jvp, tangent='boundary', guided_velocity's CFG pair, and
    base_velocity in every mode. The grad-enabled trainable forward is one
    call among them and the only one that wants checkpointing at all.

    It is also where one installed SD3Transformer2DModel breaks
    (diffusers==0.30.0.dev0, vendored for
    StableDiffusion3InstructPix2PixPipeline at
    /data/repos/UltraEdit/diffusers): its checkpointed branch assigns a
    JointTransformerBlock's tuple return to `hidden_states` without
    unpacking, so the next block's LayerNorm gets a tuple and raises
    `layer_norm(): argument 'input' ... must be Tensor, not tuple`. Note what
    that means: the bug is in the checkpointed branch as such, NOT in the
    no-grad part. This context manager keeps every no-grad forward out of it,
    which is correct on its own terms -- but the trainable forward still
    enters that branch and still breaks there, and no change on this side of
    the import fixes that. Use --no-grad-checkpointing (fd estimator only),
    or an install whose checkpointed branch unpacks.
    """
    if torch.is_grad_enabled():
        yield
        return
    was = getattr(transformer, "gradient_checkpointing", False)
    transformer.gradient_checkpointing = False
    try:
        yield
    finally:
        transformer.gradient_checkpointing = was


@dataclass
class AdapterConfig:
    mode: str                      # "interval" | "pair"
    rank: int = 16
    alpha: float = 16.0
    max_blocks: int | None = None
    use_mask: bool = True
    use_interval: bool = True
    #: "concat"   IntervalEmbedding: one MLP on [sin(t), sin(Delta)]. Stage 1.
    #: "centered" CenteredIntervalEmbedding: g(t) + [f(t,Delta) - f(t,0)].
    #: Default stays "concat" so every existing checkpoint loads unchanged.
    interval_kind: str = "concat"
    #: SD3 scales sigma by this before the transformer sees it.
    num_train_timesteps: int = 1000


class TransportAdapter(nn.Module):
    """Wraps a diffusers SD3Transformer2DModel. Base weights stay frozen."""

    def __init__(self, transformer: nn.Module, cfg: AdapterConfig) -> None:
        super().__init__()
        if cfg.mode not in ("interval", "pair", "meanflow"):
            raise ValueError(
                "mode must be 'interval', 'pair' or 'meanflow', got "
                f"{cfg.mode!r}")
        self.cfg = cfg
        self.transformer = transformer
        self.transformer.requires_grad_(False)

        self.holder = GateHolder()
        self.report = inject_lora(
            self.transformer, self.holder,
            rank=cfg.rank, alpha=cfg.alpha, max_blocks=cfg.max_blocks,
        )

        pooled_dim = int(transformer.config.pooled_projection_dim)
        latent_ch = int(transformer.config.in_channels)
        self.interval = (make_interval_embedding(cfg.interval_kind, pooled_dim)
                         if cfg.use_interval else None)
        self.mask_enc = MaskEncoder(latent_ch) if cfg.use_mask else None

    # ------------------------------------------------------------ plumbing --
    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    def _gate(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.cfg.mode in ("pair", "meanflow"):
            return torch.ones_like(t)
        return (t - r).clamp(min=0.0)

    def trainable_parameters(self):
        return adapter_parameters(self)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------- forward --
    def forward(
        self,
        z: torch.Tensor,                 # (B, C, h, w) fp32 ODE state
        t: torch.Tensor,                 # (B,) in [0, 1]
        r: torch.Tensor,                 # (B,) in [0, 1], r <= t
        prompt_embeds: torch.Tensor,     # (B, L, D)
        pooled: torch.Tensor,            # (B, P)
        mask: torch.Tensor | None = None,   # (B, 1, h, w) on the LATENT grid
        guidance: float = 1.0,
        negative_embeds: torch.Tensor | None = None,
        negative_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return u(z, r, t, c, M) in fp32.

        Mirrors bench/inversion.py::_velocity deliberately: fp32 state in,
        cast down for the transformer, fp32 out. The narrow dtype stays
        confined to the forward pass. Stage 0 bug 3 is what this costs to get
        wrong -- euler-200 moved 12.36 -> 21.16 dB on this one rule.
        """
        dtype = self.model_dtype
        gate = self._gate(t, r)

        # --- mask into the latent (zero-init conv; nothing at step 0) -------
        z_in = z
        if self.mask_enc is not None and mask is not None:
            delta_z = self.mask_enc(mask.float()).to(z.dtype)
            z_in = z + gate.reshape(-1, 1, 1, 1).to(z.dtype) * delta_z

        # --- interval into pooled projections ------------------------------
        pooled_in = pooled
        if self.interval is not None:
            d_pool = self.interval(t, r).to(pooled.dtype)
            pooled_in = pooled + gate.reshape(-1, 1).to(pooled.dtype) * d_pool

        timestep = (t.float() * self.cfg.num_train_timesteps)

        # Defensive casts. Every tensor entering the frozen transformer is
        # forced to the model dtype here, so no caller can reintroduce a
        # promotion. Gate A hit exactly this: a float32 keep-mask in
        # cond_dropout promoted bf16 embeddings back to float32 after the
        # caller had already cast them.
        prompt_embeds = prompt_embeds.to(dtype)
        if negative_embeds is not None:
            negative_embeds = negative_embeds.to(dtype)

        if guidance <= 1.0 or negative_embeds is None:
            with self.holder.scope(gate), \
                    no_checkpoint_without_grad(self.transformer):
                out = self.transformer(
                    hidden_states=z_in.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_in.to(dtype),
                    timestep=timestep,
                    return_dict=False,
                )[0]
            return out.float()

        # CFG: the gate must cover [uncond, cond] or the uncond half runs with
        # the wrong scaling. See gating.GateHolder docstring.
        batch_z = torch.cat([z_in, z_in]).to(dtype)
        neg_pooled = (negative_pooled if negative_pooled is not None
                      else pooled_in).to(dtype)
        with self.holder.scope(torch.cat([gate, gate])), \
                no_checkpoint_without_grad(self.transformer):
            out = self.transformer(
                hidden_states=batch_z,
                encoder_hidden_states=torch.cat([negative_embeds, prompt_embeds]),
                pooled_projections=torch.cat([neg_pooled, pooled_in]).to(dtype),
                timestep=timestep.repeat(2),
                return_dict=False,
            )[0]
        u_un, u_c = out.float().chunk(2)
        return u_un + guidance * (u_c - u_un)

    def base_velocity(self, z, t, prompt_embeds, pooled, guidance=1.0,
                      negative_embeds=None, negative_pooled=None) -> torch.Tensor:
        """Frozen pretrained v(z, t, c), adapter fully disabled.

        This is the teacher for Gate B and the control for everything else.
        """
        dtype = self.model_dtype
        timestep = t.float() * self.cfg.num_train_timesteps
        # no_grad first, so no_checkpoint_without_grad reads grad as already
        # disabled and takes its disabling branch. This call is never part of
        # a backward -- it is the frozen teacher -- so checkpointing it only
        # ever bought recompute nobody uses.
        with torch.no_grad(), no_checkpoint_without_grad(self.transformer), \
                self.holder.scope(None):
            if guidance <= 1.0 or negative_embeds is None:
                return self.transformer(
                    hidden_states=z.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled.to(dtype),
                    timestep=timestep, return_dict=False,
                )[0].float()
            out = self.transformer(
                hidden_states=torch.cat([z, z]).to(dtype),
                encoder_hidden_states=torch.cat([negative_embeds, prompt_embeds]),
                pooled_projections=torch.cat(
                    [negative_pooled if negative_pooled is not None else pooled,
                     pooled]).to(dtype),
                timestep=timestep.repeat(2), return_dict=False,
            )[0]
        v_un, v_c = out.float().chunk(2)
        return v_un + guidance * (v_c - v_un)

    # ------------------------------------------------------------- asserts --
    @torch.no_grad()
    def check_identity(self, z, t, prompt_embeds, pooled, mask=None,
                       tol: float = 1e-4) -> dict:
        """Assert the two structural guarantees. Run before every training run.

        - boundary: at r == t, u must equal the frozen base velocity. In
          "interval" mode this must hold at ALL times, including after
          training. In "pair" mode it holds only at initialisation, because the
          gate is 1 by design -- the check reports it rather than enforcing it.
        - the residual must be nonzero somewhere, or the adapter is not wired
          in at all and a null result would mean nothing. Stage 0 read
          byte-identical output as a finding three separate times.
        """
        stale_before = self.holder.stale_reads
        base = self.base_velocity(z, t, prompt_embeds, pooled)
        same = self(z, t, t.clone(), prompt_embeds, pooled, mask=mask)
        far_r = torch.zeros_like(t)
        far = self(z, t, far_r, prompt_embeds, pooled, mask=mask)

        boundary = (same - base).abs().max().item()
        residual = (far - base).abs().max().item()
        # In "meanflow" mode the boundary is learned, not architectural, so a
        # non-zero error early in training is expected and is the quantity to
        # WATCH rather than a failure. Enforced only for "interval".
        ok = boundary <= tol if self.cfg.mode == "interval" else True

        return {
            "mode": self.cfg.mode,
            "boundary_max_abs_err": boundary,
            "boundary_ok": bool(ok),
            "boundary_rel": float(
                boundary / max(float(base.abs().max()), 1e-8)),
            "boundary_is_enforced": self.cfg.mode == "interval",
            "residual_at_r0": residual,
            "adapter_is_live": residual > 0.0 or self.num_trainable() > 0,
            "n_trainable": self.num_trainable(),
            # Non-zero here means an adapted layer read the gate outside a
            # scope during a plain forward, which is a missing scope rather
            # than a checkpoint recompute (no backward runs in this check).
            "stale_gate_reads": self.holder.stale_reads - stale_before,
        }


def freeze_and_report(adapter: TransportAdapter) -> str:
    total = sum(p.numel() for p in adapter.parameters())
    train = adapter.num_trainable()
    return (
        f"{adapter.report}\n"
        f"trainable {train/1e6:.2f}M / {total/1e6:.1f}M "
        f"({100*train/total:.3f}%)  mode={adapter.cfg.mode}"
    )
