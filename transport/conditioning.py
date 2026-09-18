"""
The two new conditioning signals: the interval (r, t) and the instance mask M.

Both are injected at points that already exist in SD3, so nothing about the
pretrained computation graph changes shape:

    interval -> added to `pooled_projections`, which feeds the adaLN
                modulation. Global scalar conditioning belongs where the
                existing global scalar conditioning already is.

    mask     -> added to the latent, through a zero-initialised conv, BEFORE
                patch embedding. This avoids touching pos_embed.proj, whose
                input channel count is baked into the pretrained weights.
                Concatenating a mask channel instead would require surgery on
                that weight and would break the identity guarantee.

Both are zero-initialised on the output side, so at step 0 they contribute
exactly nothing.

WHY THE MASK ENTERS THE LATENT AND NOT ATTENTION. Stage 0's lambda field
biased attention and produced the negative result: attention bias controls how
strongly the instruction is heard in a region, which turns out to lie on the
same axis as source destruction. The Stage 1 hypothesis is that the mask should
instead select WHERE a learned displacement applies. That is a property of the
state being transported, not of the text-image attention, so it goes into the
state. Keeping the Stage 0 attention path available (region/sd3_processor.py,
untouched) means the two can be ablated against each other on one backbone.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """(B,) in [0,1] -> (B, dim). Standard transformer positional encoding.

    Computed in fp32 regardless of ambient dtype. Stage 0 lost 11% of a step to
    bf16 quantisation of a scaled timestep; the same arithmetic appears here and
    the same rule applies.
    """
    x = x.float().reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=x.device)
        / half
    )
    args = x[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class IntervalEmbedding(nn.Module):
    """(r, t) -> additive delta on pooled_projections.

    Parameterised as (t, Delta) with Delta = t - r rather than as (r, t).
    Equivalent information, but it puts the quantity the boundary condition
    depends on directly into the input, so the network does not have to learn
    to compute a difference in order to respect a constraint that is already
    enforced multiplicatively by the gate.
    """

    def __init__(self, pooled_dim: int = 2048, hidden: int = 512,
                 freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.net = nn.Sequential(
            nn.Linear(2 * freq_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, pooled_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net.float()

    def forward(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        delta = (t - r).clamp(min=0.0)
        e = torch.cat([sinusoidal(t, self.freq_dim),
                       sinusoidal(delta, self.freq_dim)], dim=-1)
        return self.net(e)


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, d_out),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net.float()


class CenteredIntervalEmbedding(nn.Module):
    """(r, t) -> additive delta on pooled_projections, with the Delta part centred.

        d_pool = g(t) + [ f(t, Delta) - f(t, 0) ]

    WHY. The concatenated IntervalEmbedding was probed on the P75 checkpoints
    (step 200 and 1000). Its readout was indistinguishable from noise pushed
    through its own hidden features (varying/const 0.100 trained vs 0.101
    feature-shaped noise), identical at 76% and 10% anchors (0.098), and the
    Delta-sensitivity of the hidden features fell from 0.48 to 0.26 over
    training. Delta enters the input strongly (ratio 0.62) and is eroded inside
    a single MLP whose updates concentrate on the shared mean direction, i.e.
    any correction collapses into a t-only offset.

    Here the Delta bracket is zero at r == t up to float rounding, so the
    Delta branch cannot hold an r-independent offset. That offset has
    somewhere else to go -- g(t) and the LoRA -- so the capacity the old
    embedding had is kept; only its ability to hide inside the Delta path is
    removed.

    NOT THE REMOVED DELTA-GATE. That gate multiplied EVERY trainable path by
    Delta, so the 75% anchor population produced zero gradient anywhere.
    Here only f's contribution vanishes at anchors; g and the LoRA keep their
    full anchor gradient, and f has nothing to learn at Delta = 0 by
    definition.

    Same sinusoidal basis and widths as IntervalEmbedding, deliberately: this
    run changes the structure only. Low-passing the basis (P5) is a separate
    run.
    """

    def __init__(self, pooled_dim: int = 2048, hidden: int = 512,
                 freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.t_branch = _mlp(freq_dim, hidden, pooled_dim)            # g
        self.delta_branch = _mlp(2 * freq_dim, hidden, pooled_dim)    # f

    def delta_part(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """f(t, Delta) - f(t, 0). Exposed so probes can measure it directly."""
        delta = (t - r).clamp(min=0.0)
        et = sinusoidal(t, self.freq_dim)
        e = torch.cat([et, sinusoidal(delta, self.freq_dim)], dim=-1)
        e0 = torch.cat([et, sinusoidal(torch.zeros_like(delta), self.freq_dim)],
                       dim=-1)
        # One batched call, so both halves go through identical kernels and
        # the anchor rows cancel as exactly as the hardware allows.
        f, f0 = self.delta_branch(torch.cat([e, e0], dim=0)).chunk(2, dim=0)
        return f - f0

    def forward(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        g = self.t_branch(sinusoidal(t, self.freq_dim))
        return g + self.delta_part(t, r)


#: interval_kind -> embedding class. Validated here rather than in
#: TransportAdapter.__init__, because tests/check_modes.py locates the mode
#: guard as the first `not in` comparison inside __init__, and a second guard
#: there would silently become the one it parses.
INTERVAL_KINDS = {
    "concat": IntervalEmbedding,
    "centered": CenteredIntervalEmbedding,
}


def make_interval_embedding(kind: str, pooled_dim: int) -> nn.Module:
    if kind not in INTERVAL_KINDS:
        raise ValueError(
            f"interval_kind must be one of {sorted(INTERVAL_KINDS)}, got {kind!r}")
    return INTERVAL_KINDS[kind](pooled_dim)


def interval_kind_from_keys(keys) -> str | None:
    """Which embedding a checkpoint was trained with, from its parameter names.

    Returns None if the checkpoint has no interval parameters. Raises on a
    mixture, which would mean two runs' weights were merged.
    """
    keys = list(keys)
    has_concat = any("interval.net." in k for k in keys)
    has_centered = any("interval.delta_branch" in k or "interval.t_branch" in k
                       for k in keys)
    if has_concat and has_centered:
        raise ValueError("checkpoint holds BOTH concat and centered interval "
                         "weights; it is not from a single run")
    if has_centered:
        return "centered"
    if has_concat:
        return "concat"
    return None


class MaskEncoder(nn.Module):
    """(B, 1, H, W) binary mask on the LATENT grid -> additive delta on the latent.

    Three input planes are built from the mask, not one:

        inside   the mask itself
        signed   normalised signed distance to the boundary, clipped
        band     a soft boundary indicator

    The distance planes exist because Stage 0's F4 found that a hard binary
    mask and a radius-r softened mask behaved identically for attention bias
    (selectivity flat across r in {0,1,2,4}). That was a finding about the
    attention path. Whether a transport field can use boundary geometry is a
    different question and it deserves the information to be present rather
    than assumed away. If the signed/band planes turn out to carry no gradient,
    that is itself the r-ablation repeated in the new setting.
    """

    def __init__(self, latent_channels: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(hidden, latent_channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.float()

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        planes = build_mask_planes(mask)
        return self.out(self.body(planes))


class GlobalLocalMaskEncoder(nn.Module):
    """MaskEncoder split into an explicit global operating-point shift and
    a mask-gated local edit.

    WHY THIS EXISTS. Direction A's DC-offset ablation (GATE_A_REOPENING_PLAN.md)
    found that MaskEncoder's spatial-mean component is LOAD-BEARING: removing
    it at inference dropped pooled cosine from 0.769 to 0.211 -- worse than
    removing the mask path entirely (0.681). The offset sets the operating
    point the transformer's query lands at; the structured, mask-conditioned
    part of the output only means anything relative to that baseline. The
    plain MaskEncoder entangles both roles in one head, with nothing forcing
    a particular split between them under ordinary MSE regression -- two
    heads summed and trained end to end can collapse back to exactly the
    current allocation, which would make a naive two-head version a
    reparameterisation with no new inductive bias, not a real change.

    THE SPLIT IS ENFORCED ARCHITECTURALLY, not by a training-time trick (a
    separate learning rate, a stop-gradient schedule, a variance penalty in
    the loss) that could fail to hold under enough optimisation pressure:

        global   a single learned (C,) vector with NO INPUT AT ALL -- it
                 cannot become a disguised, mask-conditioned copy of the
                 local signal, because it has nothing to condition on.
        local    the existing MaskEncoder CNN (mask-conditioned,
                 structured), with its own per-example spatial mean
                 SUBTRACTED before use, on every forward pass, not as a
                 post-hoc eval-time ablation -- it is architecturally
                 IMPOSSIBLE for the local head to contribute anything to
                 the operating point, whatever it learns.

        delta_z = global + (local - local.mean(over H, W))

    This guarantee is exact and checked directly in self_test(): the
    output's own spatial mean equals `global_bias` exactly, for ANY mask
    and ANY (including randomly re-initialised) local-head weights.

    NOT a drop-in replacement for MaskEncoder -- this is new model
    surface (the global parameter didn't exist before), selected via
    AdapterConfig.mask_encoder_kind="global_local"; the default stays
    "single" (plain MaskEncoder), byte-identical for every existing
    checkpoint. Same forward(mask) -> (B,C,h,w) interface, so nothing
    downstream (TransportAdapter.forward, apply_mask_conditioning, the
    inspect/diagnose checkpoint tooling) needs to know which one it has.
    """

    def __init__(self, latent_channels: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.local = MaskEncoder(latent_channels, hidden)
        # Zero-init: delta_z == 0 at initialisation regardless of mode,
        # matching the adapter's identity-at-init guarantee
        # (transport/adapter.py's own module docstring).
        self.global_bias = nn.Parameter(torch.zeros(latent_channels))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        local = self.local(mask)
        local_centered = local - local.mean(dim=(2, 3), keepdim=True)
        global_term = self.global_bias.reshape(1, -1, 1, 1).to(local.dtype)
        return global_term + local_centered


def build_mask_planes(mask: torch.Tensor, clip: float = 8.0) -> torch.Tensor:
    """(B,1,h,w) in {0,1} -> (B,3,h,w) float32 [inside, signed_dist, band].

    Distance transform runs on CPU via scipy because torch has no EDT and the
    grid is 64x64. It is called once per sample per step; if it ever shows up
    in a profile, precompute it in the dataset instead -- the mask does not
    change during training.
    """
    from scipy.ndimage import distance_transform_edt

    m = mask.detach().float()
    b = m.shape[0]
    out = torch.zeros(b, 3, m.shape[-2], m.shape[-1], dtype=torch.float32)
    arr = (m[:, 0].cpu().numpy() > 0.5)

    for i in range(b):
        inside = arr[i]
        if inside.any() and not inside.all():
            d_out = distance_transform_edt(~inside)
            d_in = distance_transform_edt(inside)
            signed = (d_in - d_out) / clip
        else:
            signed = np.zeros_like(inside, dtype=np.float64)
        signed = np.clip(signed, -1.0, 1.0)
        out[i, 0] = torch.from_numpy(inside.astype(np.float32))
        out[i, 1] = torch.from_numpy(signed.astype(np.float32))
        out[i, 2] = torch.from_numpy(
            np.exp(-(signed * clip) ** 2 / (2 * 2.0 ** 2)).astype(np.float32)
        )
    return out.to(mask.device)


def mask_to_latent_grid(mask: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Pixel-space HxW binary mask -> (1,1,h,w) on the latent grid.

    Area-averaged then thresholded at 0.5, matching how region/lambda_field.py
    pools to the token grid. Nearest-neighbour would drop thin structures and
    silently change the mask area, which is one of the confounds Stage 0's
    sweep had to log its way around.
    """
    m = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(m, (h, w))
    return (pooled > 0.5).float()


# --------------------------------------------------------------- self-test --

def self_test() -> None:
    """MaskEncoder / GlobalLocalMaskEncoder, CPU-only (needs scipy for
    build_mask_planes' distance transform, no GPU or transformer). Run
    before trusting GlobalLocalMaskEncoder in a real training run --
    this module had no self-test before the split was added.
    """
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 16, 16
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 4:12, 4:12] = 1.0

    # MaskEncoder: zero-init means delta_z == 0 exactly at construction,
    # the identity-at-init guarantee transport/adapter.py depends on.
    enc = MaskEncoder(C)
    enc.eval()
    with torch.no_grad():
        out0 = enc(mask)
    assert torch.equal(out0, torch.zeros_like(out0))

    # GlobalLocalMaskEncoder: same zero-at-init guarantee.
    gl = GlobalLocalMaskEncoder(C)
    gl.eval()
    with torch.no_grad():
        gl_out0 = gl(mask)
    assert torch.allclose(gl_out0, torch.zeros_like(gl_out0), atol=1e-6)

    # THE CORE GUARANTEE: the output's own per-channel spatial mean must
    # equal global_bias EXACTLY -- not approximately, not "usually" --
    # for ANY mask and ANY local-head weights, since local_centered is
    # constructed to have exactly zero spatial mean by subtraction. Give
    # the local head random (non-zero-init) weights first, since at true
    # zero-init this would hold trivially (both sides zero).
    with torch.no_grad():
        for p in gl.local.parameters():
            p.data = torch.randn_like(p)
        gl.global_bias.data = torch.tensor([1.0, -2.0, 0.5, 3.0])
    gl.eval()

    mask2 = torch.zeros(B, 1, H, W)
    mask2[:, :, 1:6, 9:14] = 1.0  # a DIFFERENT mask, different location/shape

    with torch.no_grad():
        out_a = gl(mask)
        out_b = gl(mask2)
    spatial_mean_a = out_a.mean(dim=(2, 3))  # (B, C)
    spatial_mean_b = out_b.mean(dim=(2, 3))
    expected = gl.global_bias.reshape(1, C).expand(B, C)
    # float32 accumulation over a 256-pixel mean subtraction leaves ~1e-4
    # residual noise for random-scale local weights -- not a real
    # violation (an actual architectural break would be O(1), not O(1e-4)).
    assert torch.allclose(spatial_mean_a, expected, atol=1e-3), spatial_mean_a
    assert torch.allclose(spatial_mean_b, expected, atol=1e-3), spatial_mean_b
    # Different masks, but the SAME global operating point -- exactly the
    # property the plain MaskEncoder does NOT guarantee (its DC offset is
    # entangled with, not independent of, the mask-conditioned content).
    assert torch.allclose(spatial_mean_a, spatial_mean_b, atol=1e-3)

    # The LOCAL component genuinely varies with the mask (this split isn't
    # a way to accidentally make the whole output mask-insensitive) --
    # subtract out the shared global term and confirm what's left differs.
    local_only_a = out_a - gl.global_bias.reshape(1, C, 1, 1)
    local_only_b = out_b - gl.global_bias.reshape(1, C, 1, 1)
    assert not torch.allclose(local_only_a, local_only_b, atol=1e-3)

    # A GlobalLocalMaskEncoder with global_bias forced to zero must have
    # EXACTLY zero spatial mean everywhere -- confirms the local head
    # alone never leaks into the operating point, whatever it computes.
    with torch.no_grad():
        gl.global_bias.data.zero_()
        out_c = gl(mask)
    assert torch.allclose(out_c.mean(dim=(2, 3)), torch.zeros(B, C), atol=1e-3)

    print("conditioning self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
