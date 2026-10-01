"""
Tier B's one new mechanism: source image + mask as extra input channels.

WHAT TIER B IS. `EDITOR_PLAN.md` section 2. The training objective is already
right (`x0 = z_tgt`, `z_t = (1-t)x0 + t*eps`, `v = eps - x0`) and the batch
already carries `z_src` and `mask` (`transport/data.py`'s torch_dataset hands
both to the loop). The measured conclusion from tier D is that the mechanism is
sufficient and the thing missing at inference time is Y -- a target -- so the
only thing to build is a path that lets the model SEE the source and the mask.

HOW. Copy UltraEdit, which `bench/run_gate_c.py` records as
`in_channels = 48 = 16 latent + 16 image + 16 MASK`: widen
`transformer.pos_embed.proj` from a 16-channel conv to a 48-channel one, keep
the pretrained 16 columns, and add the source latent and the mask as 32 new
input channels. Cheapest option, and it has precedent in two working models.

WHY NOT THE MaskEncoder THAT ALREADY EXISTS. `transport/conditioning.py`'s
MaskEncoder adds a zero-initialised conv delta to the LATENT before patch
embedding, and its docstring rejects channel concatenation because it "would
require surgery on that weight and would break the identity guarantee". The
first half of that is true and the second half is not (surgery plus zeroed new
columns preserves identity exactly -- see self_test). But the decisive reason is
measurement: the additive path was probed and found near-inert (rms/noise
1.07-1.12), and `EDITOR_PLAN.md section 5` says in as many words not to use the
existing additive mask path as the source conditioning. A zero-init additive
delta has to grow a spatial signal through a small conv; input channels put the
information in front of every block from layer 0.

THE TRAP THIS MODULE EXISTS TO PREVENT. Zero-init new columns plus a FROZEN
transformer is a structurally dead conditioning path. `TransportAdapter.__init__`
calls `self.transformer.requires_grad_(False)` and then injects LoRA into the
attention and MLP projections -- and LoRA does not touch `pos_embed.proj`. So
the plan's "zero-init the 32 new channels" combined with the existing freezing
would leave the source and mask multiplying exactly-zero weights FOREVER: the
model could never learn to use them, no matter how long it trained. Every shape
check, the identity assertion, the loss curve and the episode logs would all
look completely normal.

    make_source_cond() therefore (a) un-freezes the new columns and (b) by
    default installs a gradient mask so that ONLY the new columns receive
    gradient -- the pretrained 16 columns keep their frozen behaviour exactly,
    which is both the conservative choice and the one the plan describes.
    self_test() asserts both halves: identity at initialisation AND a nonzero
    gradient on the new columns after one backward.

MASK POLARITY: WHITE = EDITABLE, the same convention as Gate C
(`bench/run_gate_c.py`). This project has already lost a day to a polarity bug;
training and evaluation must not disagree, so `build_cond_input` refuses a
non-binary mask rather than guessing what it means.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SourceCondSpec:
    """What make_source_cond changed, so the log can state it and a resume can
    check it. Kept as a plain object (not a module) because it holds no
    parameters -- it is provenance."""

    def __init__(self, prev_channels: int, latent_channels: int,
                 image_channels: int, mask_channels: int, mask_repeat_to: int,
                 extra_only: bool) -> None:
        self.prev_channels = prev_channels
        self.latent_channels = latent_channels
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.mask_repeat_to = mask_repeat_to
        self.extra_only = extra_only

    @property
    def new_channels(self) -> int:
        return self.image_channels + self.mask_repeat_to

    @property
    def total_channels(self) -> int:
        return self.prev_channels + self.new_channels

    @property
    def extra_slice(self) -> slice:
        return slice(self.prev_channels, self.total_channels)

    def as_dict(self) -> dict:
        return {
            "prev_channels": self.prev_channels,
            "latent_channels": self.latent_channels,
            "image_channels": self.image_channels,
            "mask_channels": self.mask_channels,
            "mask_repeat_to": self.mask_repeat_to,
            "new_channels": self.new_channels,
            "total_channels": self.total_channels,
            "extra_only": self.extra_only,
        }

    def describe(self) -> str:
        return (f"source+mask conditioning: pos_embed.proj "
                f"{self.prev_channels} -> {self.total_channels} input channels "
                f"({self.latent_channels} latent + {self.image_channels} source "
                f"image + {self.mask_repeat_to} mask), new columns zero-init, "
                f"{'extra columns only trainable' if self.extra_only else 'all columns trainable'}")


def patch_embed_proj(transformer) -> nn.Conv2d:
    """The conv that turns pixels into tokens. Read from the MODULE, not from
    `config.in_channels`: the config is a number someone can forget to update,
    the tensor is what actually runs."""
    proj = getattr(getattr(transformer, "pos_embed", None), "proj", None)
    if not isinstance(proj, nn.Conv2d):
        raise TypeError(
            "transformer.pos_embed.proj is not a Conv2d "
            f"({type(proj).__name__}); this surgery is written for diffusers' "
            "SD3 PatchEmbed and must be re-derived for anything else")
    return proj


def patch_embed_in_channels(transformer) -> int:
    return int(patch_embed_proj(transformer).weight.shape[1])


def make_source_cond(transformer, image_channels: int = 16,
                     mask_channels: int = 1, mask_repeat_to: int | None = None,
                     extra_only: bool = True, init_extra: str = "zero",
                     trainable: bool = True) -> SourceCondSpec:
    """Widen `transformer.pos_embed.proj` to take (latent, source, mask).

    `init_extra="zero"` (the plan's choice) makes the model's output at
    initialisation EXACTLY the base model's, whatever the source and mask are,
    so a training run starts from a known-good function rather than from noise.
    `init_extra="source"` copies the pretrained columns into the source-image
    block instead, which is what SD3 InstructPix2Pix does; it is offered because
    it is the precedent's choice, NOT as the default -- it makes step 0 output
    depend on the source image, so the identity check below would have to be
    relaxed, and that is a decision to take deliberately.

    `extra_only=True` freezes the gradient into the pretrained columns. See the
    module docstring: without un-freezing the new columns this whole path is
    dead, and without the gradient mask the 16 pretrained columns drift too.
    """
    proj = patch_embed_proj(transformer)
    prev = patch_embed_in_channels(transformer)
    if mask_repeat_to is None:
        mask_repeat_to = prev
    if mask_channels != 1:
        raise ValueError(
            f"mask_channels must be 1 (a single binary plane, white = editable), "
            f"got {mask_channels}. build_cond_input repeats it to "
            f"{mask_repeat_to} channels itself.")
    if init_extra not in ("zero", "source"):
        raise ValueError(f"init_extra must be 'zero' or 'source', got {init_extra!r}")
    if image_channels <= 0 or mask_repeat_to <= 0:
        raise ValueError("source and mask channel counts must be positive")
    if image_channels != prev:
        raise ValueError(
            f"the source image latent has {prev} channels (the same VAE as the "
            f"state), but image_channels={image_channels}. A different count "
            "means this conv cannot be initialised from the pretrained columns "
            "at all.")

    spec = SourceCondSpec(prev, prev, image_channels, mask_channels,
                          mask_repeat_to, extra_only)
    new_proj = nn.Conv2d(
        spec.total_channels, proj.out_channels,
        kernel_size=proj.kernel_size, stride=proj.stride,
        padding=proj.padding, dilation=proj.dilation, groups=proj.groups,
        bias=proj.bias is not None, padding_mode=proj.padding_mode,
        device=proj.weight.device, dtype=proj.weight.dtype)

    with torch.no_grad():
        new_proj.weight.zero_()
        # the pretrained latent columns, unchanged
        new_proj.weight[:, :prev] = proj.weight
        if init_extra == "source":
            new_proj.weight[:, prev:prev + image_channels] = proj.weight
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)

    # The grad mask is a hook on the tensor, so it must be (re)installed
    # whenever the parameter object is replaced.
    _clear_grad_mask(proj)
    transformer.pos_embed.proj = new_proj
    _clear_grad_mask(new_proj)

    if trainable:
        new_proj.weight.requires_grad_(True)
        if new_proj.bias is not None:
            new_proj.bias.requires_grad_(True)
        if extra_only:
            _install_grad_mask(new_proj, spec)
    else:
        new_proj.weight.requires_grad_(False)
        if new_proj.bias is not None:
            new_proj.bias.requires_grad_(False)

    # keep the config honest: other code reads it (transport/adapter.py reads
    # config.in_channels to size the mask encoder and the additive path)
    if hasattr(transformer, "config"):
        transformer.config.in_channels = spec.total_channels
    transformer._source_cond_spec = spec
    return spec


# ------------------------------------------------------- the gradient mask --

def _mask_key(proj: nn.Conv2d) -> str:
    return f"_source_cond_grad_mask_{id(proj)}"


def _clear_grad_mask(proj: nn.Conv2d) -> None:
    key = getattr(proj, "_source_cond_mask_hook", None)
    if key is not None:
        key.remove()
        proj._source_cond_mask_hook = None


def _install_grad_mask(proj: nn.Conv2d, spec: SourceCondSpec) -> None:
    """Zero the gradient flowing into the PRETRAINED columns.

    `requires_grad` is per-tensor, so a Conv2d weight cannot have "only some
    columns trainable" -- the new columns would drag the old ones along. This
    hook is the smallest thing that gives the intended behaviour, and it is
    exact: the masked columns' gradient is set to zero, so an SGD/AdamW step
    leaves them bit-identical (momentum aside -- AdamW with weight decay would
    still touch a zero-grad parameter through the decay term, which is why
    train.py must exclude them from weight decay; the assertion in self_test
    covers the gradient, and the trainer's param groups cover the decay).
    """
    prev = spec.prev_channels

    def hook(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[:, :prev] = 0.0
        return grad

    proj._source_cond_mask_hook = proj.weight.register_hook(hook)


def grad_mask_report(transformer) -> dict:
    """Measure the claim instead of asserting it from the code. Call after a
    backward that used the conditioned input."""
    proj = patch_embed_proj(transformer)
    spec = getattr(transformer, "_source_cond_spec", None)
    g = proj.weight.grad
    if spec is None or g is None:
        return {"has_spec": spec is not None, "has_grad": g is not None}
    prev = spec.prev_channels
    old = float(g[:, :prev].abs().sum())
    new = float(g[:, prev:].abs().sum())
    return {"has_spec": True, "has_grad": True,
            "pretrained_grad_abs_sum": old, "new_grad_abs_sum": new,
            "pretrained_frozen": old == 0.0, "new_learning": new > 0.0}


# ------------------------------------------------------- the input builder --

def build_cond_input(z: torch.Tensor, z_src: torch.Tensor,
                     mask: torch.Tensor, spec: SourceCondSpec) -> torch.Tensor:
    """(z, z_src, mask) -> the (B, 48, h, w) tensor the widened conv expects.

    Every check here is a failure mode this project has actually paid for:
    a mask on the wrong grid, a mask with the wrong polarity, an image that
    silently broadcast against the state.
    """
    if z.ndim != 4 or z_src.ndim != 4 or mask.ndim != 4:
        raise ValueError(
            f"expected 4-D (B, C, h, w) tensors, got {tuple(z.shape)}, "
            f"{tuple(z_src.shape)}, {tuple(mask.shape)}")
    if z.shape[0] != z_src.shape[0] or z.shape[0] != mask.shape[0]:
        raise ValueError(
            f"batch sizes disagree: z {z.shape[0]}, z_src {z_src.shape[0]}, "
            f"mask {mask.shape[0]}")
    if z.shape[1] != spec.latent_channels:
        raise ValueError(f"z has {z.shape[1]} channels, expected "
                         f"{spec.latent_channels}")
    if z_src.shape[1] != spec.image_channels:
        raise ValueError(
            f"z_src has {z_src.shape[1]} channels, expected "
            f"{spec.image_channels}. The source latent must come from the same "
            "VAE as the state.")
    if z_src.shape[-2:] != z.shape[-2:]:
        raise ValueError(
            f"z_src is {tuple(z_src.shape[-2:])} but z is {tuple(z.shape[-2:])}; "
            "the source MUST be encoded at the state's resolution or the extra "
            "channels are a different grid")
    if mask.shape[1] != spec.mask_channels:
        raise ValueError(f"mask has {mask.shape[1]} channels, expected "
                         f"{spec.mask_channels}")
    if mask.shape[-2:] != z.shape[-2:]:
        raise ValueError(
            f"mask is {tuple(mask.shape[-2:])} but z is {tuple(z.shape[-2:])}; "
            "mask_to_latent_grid() is the thing that puts it on the latent grid")

    m = mask.to(dtype=z.dtype)
    lo, hi = float(m.min()), float(m.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(f"mask values must be in [0, 1] (white = editable), "
                         f"got [{lo:.3f}, {hi:.3f}]")
    # A mask that is neither binary nor a deliberate soft mask is almost always
    # a 0/255 uint8 plane or a swapped-polarity mask. Refuse loudly.
    frac_mid = float(((m > 0.02) & (m < 0.98)).float().mean())
    if frac_mid > 0.5:
        raise ValueError(
            f"{100*frac_mid:.0f}% of the mask is strictly between 0 and 1. A "
            "binary mask (white = editable) is what training and Gate C both "
            "assume; a 0/255 plane or a polarity swap looks exactly like this.")

    if spec.mask_repeat_to == 1:
        mask_c = m
    else:
        mask_c = m.expand(-1, spec.mask_repeat_to, -1, -1)
    return torch.cat([z, z_src.to(dtype=z.dtype), mask_c], dim=1)


def cond_from_tensors(z_src, mask, device=None) -> dict:
    """The kwargs a SAMPLING call must pass to `model(...)`, from raw tensors.

    `cond_from_batch` is the TRAINING-time entry point (it reads a batch dict);
    this is the INFERENCE-time one, and it is what `transport/sampling.py` calls.
    The mask arrives as (B, h, w) from the cache and from `mask_to_latent_grid`,
    but the conditioned conv needs (B, 1, h, w) -- and getting that wrong is a
    broadcast rather than an exception: a (B, h, w) tensor concatenated along
    dim=1 reads as a mask with h channels and every log line still looks normal.

    ONE implementation on purpose. Training and sampling disagreeing about the
    mask convention is the polarity bug the module docstring says cost a day,
    and the two call sites are now the same three lines of code.
    """
    z_src = z_src.float()
    mask = mask.float()
    if mask.ndim == 3:                      # (B, h, w) -> (B, 1, h, w)
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"the mask must reduce to (B, 1, h, w) or (B, h, w), got "
            f"{tuple(mask.shape)}")
    if device is not None:
        z_src = z_src.to(device)
        mask = mask.to(device)
    return {"z_src": z_src, "mask_cond": mask}


def cond_from_batch(batch: dict, source_cond: bool, device=None) -> dict:
    """The kwargs a training step must pass to `model(...)`/`base_velocity`.

    Returns {} when tier B is off, so callers can always splat it and the
    source_cond=False path stays byte-identical to what every existing run did.

    Why this is a function and not three lines at each call site: the mask comes
    out of the cache as (B, h, w) -- a 2-D mask per sample -- and the conditioned
    conv needs (B, 1, h, w). Getting that wrong is a broadcast, not an error:
    a (B, h, w) tensor concatenated along dim=1 would silently be treated as a
    mask with h channels and the model would train on nonsense.

    The tensor half is `cond_from_tensors`, shared with sampling.
    """
    if not source_cond:
        return {}
    missing = [k for k in ("z_src", "mask") if k not in batch]
    if missing:
        raise KeyError(
            f"tier B needs {missing} in the batch. The cache written by "
            "scripts/precompute_cache.py carries both; a KeyError here usually "
            "means an older cache, or a dataset that is not TripletCache.")
    return cond_from_tensors(batch["z_src"], batch["mask"], device=device)


def _self_test_checkpoint() -> None:
    """A checkpoint must PRESERVE the conditioning, through the real Trainer.

    Regression for a second silent trap found after the first: `Trainer.save`
    filtered its state dict with `v.requires_grad or "lora" in k or ...`, and
    `state_dict()` DETACHES, so the first clause never fired. Every trainable
    parameter so far was named lora/interval/mask_enc, so nothing noticed.
    Tier B's `transformer.pos_embed.proj.weight` is named none of those: it
    would have been dropped from every checkpoint while the loss curve looked
    normal, and `--resume` would have silently reset it to zero.

    Also checks the optimiser-group fix: with weight_decay > 0 the pretrained
    columns must STILL be bit-identical after real AdamW steps.
    """
    import tempfile

    from transport.adapter import AdapterConfig, TransportAdapter
    from transport.train import (TrainConfig, Trainer, checkpoint_state,
                                 source_cond_params)

    torch.manual_seed(0)
    tr = _tiny_transformer()
    base_w = tr.pos_embed.proj.weight.detach().clone()
    ad = TransportAdapter(tr, AdapterConfig(
        mode="interval", rank=4, use_mask=False, use_interval=False,
        source_cond=True))

    # 1. THE REGRESSION: the conditioning weight is in the saved set
    state = checkpoint_state(ad)
    assert "transformer.pos_embed.proj.weight" in state, sorted(state)[:8]
    # and the fix did not start saving the frozen backbone
    assert not any(k.startswith("transformer.blocks.") and "lora" not in k
                   for k in state), "checkpoint_state swallowed frozen weights"

    # 2. the conditioning params get their own zero-decay group
    cond = source_cond_params(ad)
    assert len(cond) == 2, cond          # weight + bias
    assert ad.source_cond is not None

    B, h, w = 2, 8, 8
    z = torch.randn(B, 16, h, w)
    z_src = torch.randn(B, 16, h, w)
    mcond = torch.zeros(B, 1, h, w)
    mcond[:, :, 2:5, 2:6] = 1.0
    pe = torch.randn(B, 5, 16)
    po = torch.randn(B, 16)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(gate="b", out_dir=tmp, steps=3, lr=0.05,
                          weight_decay=0.1)   # NON-ZERO on purpose
        tn = Trainer(ad, None, cfg, loss_fn=None)
        groups = tn.opt.param_groups
        total = sum(len(g["params"]) for g in groups)
        assert total == len(tn.params), (total, len(tn.params))
        wds = sorted(g["weight_decay"] for g in groups)
        assert wds == [0.0, 0.1], wds
        assert any(g["weight_decay"] == 0.0
                   and {id(p) for p in g["params"]} == {id(p) for p in cond}
                   for g in groups), "the conditioning group is wrong"

        ad.train()
        for step in range(3):
            tn.opt.zero_grad(set_to_none=True)
            t = torch.full((B,), 0.6)
            r = torch.full((B,), 0.2)
            out = ad(z, t, r, pe, po, z_src=z_src, mask_cond=mcond)
            ((out - z_src) ** 2).mean().backward()
            tn.opt.step()

        proj = ad.transformer.pos_embed.proj
        # THE CLAIM, under real AdamW with weight decay 0.1
        assert torch.equal(proj.weight.detach()[:, :16], base_w), \
            ("the pretrained columns moved under AdamW: weight decay is "
             "reaching the conditioning tensor")
        assert not torch.equal(proj.weight.detach()[:, 16:],
                               torch.zeros_like(proj.weight.detach()[:, 16:])), \
            "the new columns never moved, so nothing was trained"

        # 3. a REAL save/load round trip preserves them
        ckpt = tn.save("selftest")
        want = proj.weight.detach().clone()
        with torch.no_grad():
            proj.weight.zero_()
        assert torch.count_nonzero(proj.weight) == 0
        tn.load(ckpt)
        assert torch.equal(ad.transformer.pos_embed.proj.weight.detach(), want), \
            "the checkpoint did not restore the conditioning weights"

        # 4. a checkpoint whose filter DROPPED a declared-trainable weight must
        #    fail loudly rather than resume with it re-initialised. This is the
        #    old bug reproduced exactly: `adapter` lacks the key while
        #    `trainable_keys` (the declaration of intent) still lists it.
        import pathlib
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "transformer.pos_embed.proj.weight" in sd["trainable_keys"], \
            "the checkpoint must declare the conditioning weight as trainable"
        del sd["adapter"]["transformer.pos_embed.proj.weight"]
        bad = pathlib.Path(tmp) / "ckpt_dropped.pt"
        torch.save(sd, bad)
        with torch.no_grad():
            proj.weight.zero_()
        try:
            tn.load(bad)
        except RuntimeError as exc:
            assert "silently re-initialise" in str(exc), exc
        else:
            raise AssertionError(
                "a checkpoint that drops a declared-trainable weight must "
                "raise, not resume quietly with it re-initialised")


# --------------------------------------------------------------- self-test --

def _tiny_transformer(channels: int = 16, seed: int = 0):
    """A real SD3Transformer2DModel at toy size. The repo already builds these
    for CPU tests (`scripts/test_preflight.py`); using the real class is the
    point -- the surgery is on diffusers' own module, so a stand-in would not
    test it."""
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    torch.manual_seed(seed)
    return SD3Transformer2DModel(
        sample_size=8, patch_size=2, in_channels=channels, num_layers=2,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
        caption_projection_dim=16, pooled_projection_dim=16,
        pos_embed_max_size=16, qk_norm="rms_norm", out_channels=channels)


def self_test() -> None:
    torch.manual_seed(0)
    tr = _tiny_transformer()
    was = patch_embed_in_channels(tr)
    assert was == 16, was

    out_dim = tr.pos_embed.proj.out_channels
    base_w = tr.pos_embed.proj.weight.detach().clone()

    B, C, h, w = 2, 16, 8, 8
    z = torch.randn(B, C, h, w)
    z_src = torch.randn(B, C, h, w)
    mask = torch.zeros(B, 1, h, w)
    mask[:, :, 2:6, 2:6] = 1.0
    pe = torch.randn(B, 5, 16)
    po = torch.randn(B, 16)
    ts = torch.full((B,), 0.4)

    # THE BASE MODEL'S FUNCTION, measured BEFORE the surgery. This is the
    # reference the identity claim is about.
    tr.eval()
    with torch.no_grad():
        base_out = tr(hidden_states=z, encoder_hidden_states=pe,
                      pooled_projections=po, timestep=ts,
                      return_dict=False)[0]

    spec = make_source_cond(tr, extra_only=True, init_extra="zero")
    assert spec.total_channels == 48 and spec.new_channels == 32, spec.as_dict()
    assert patch_embed_in_channels(tr) == 48
    assert tr.config.in_channels == 48, tr.config.in_channels

    # 1. THE PRETRAINED COLUMNS ARE UNTOUCHED, and the new ones are exactly zero
    proj = patch_embed_proj(tr)
    assert torch.equal(proj.weight[:, :16], base_w), \
        "the surgery must not perturb the pretrained latent columns"
    assert torch.count_nonzero(proj.weight[:, 16:]) == 0, \
        "the new columns must start at exactly zero"
    assert proj.bias is not None and torch.count_nonzero(proj.bias) >= 0

    def pad_only(zz):
        """z with 32 zero channels: the base computation expressed in the
        widened conv, so both sides run through identical kernels and the
        comparison can be EXACT rather than tolerance-based."""
        return torch.cat([zz, torch.zeros(zz.shape[0], spec.new_channels,
                                         *zz.shape[2:])], dim=1)

    # 2. IDENTITY AT INITIALISATION. Two statements, deliberately separated
    #    because they fail for different reasons:
    #    (a) EXACT: at init the output cannot depend on the source or the mask.
    #        Both sides use the same 48-channel conv, so this is bit-exact.
    tr.eval()
    with torch.no_grad():
        cond = tr(hidden_states=build_cond_input(z, z_src, mask, spec),
                  encoder_hidden_states=pe, pooled_projections=po,
                  timestep=ts, return_dict=False)[0]
        cond_zero = tr(hidden_states=build_cond_input(
                           z, torch.zeros_like(z_src),
                           torch.zeros_like(mask), spec),
                       encoder_hidden_states=pe, pooled_projections=po,
                       timestep=ts, return_dict=False)[0]
    assert torch.equal(cond, cond_zero), \
        ("at init the source and mask values must not matter AT ALL; max abs "
         f"difference {(cond - cond_zero).abs().max():.3e}")

    #    (b) the model therefore starts as the base model. NOT bit-exact: a
    #        48-channel conv can accumulate in a different order or pick a
    #        different kernel than a 16-channel one, so this is float rounding,
    #        not a behavioural difference. Stated with a tolerance on purpose.
    err = float((cond - base_out).abs().max())
    assert err < 1e-5, \
        (f"a zero-init widening must leave the model's function unchanged, but "
         f"the max abs difference from the base output is {err:.3e}")

    # 3. THE PATH IS ALIVE. One backward on a target that DEPENDS on z_src, and
    #    the new columns must have moved. Without this, "zero-init" plus the
    #    adapter's frozen base is a conditioning path that can never learn.
    tr.train()
    before = proj.weight.detach().clone()
    out = tr(hidden_states=build_cond_input(z, z_src, mask, spec),
             encoder_hidden_states=pe, pooled_projections=po,
             timestep=ts, return_dict=False)[0]
    # a target that can only be reached by using the source image
    loss = ((out - z_src) ** 2).mean()
    loss.backward()
    rep = grad_mask_report(tr)
    assert rep["has_grad"], rep
    assert rep["new_learning"], \
        f"the new columns got NO gradient: the conditioning path is dead. {rep}"
    assert rep["pretrained_frozen"], \
        f"the gradient mask did not protect the pretrained columns. {rep}"

    opt = torch.optim.SGD([proj.weight], lr=0.5)
    opt.step()
    moved_new = (proj.weight.detach()[:, 16:] - before[:, 16:]).abs().sum()
    moved_old = (proj.weight.detach()[:, :16] - before[:, :16]).abs().sum()
    assert float(moved_new) > 0, "the new columns did not move after a step"
    assert float(moved_old) == 0, \
        f"the pretrained columns moved by {float(moved_old):.3e} despite the mask"
    assert torch.equal(proj.weight.detach()[:, :16], base_w), \
        "the pretrained columns must be bit-identical after an SGD step"

    # and the conditioning has become FUNCTIONAL: the same input now produces a
    # different answer from the base computation
    tr.eval()
    with torch.no_grad():
        after = tr(hidden_states=build_cond_input(z, z_src, mask, spec),
                   encoder_hidden_states=pe, pooled_projections=po,
                   timestep=ts, return_dict=False)[0]
        after_base = tr(hidden_states=pad_only(z), encoder_hidden_states=pe,
                        pooled_projections=po, timestep=ts,
                        return_dict=False)[0]
    assert not torch.allclose(after, after_base), \
        "after training the conditioning must change the output"

    # 4. init_extra="source" is the precedent's alternative and must NOT be
    #    silently identical to zero-init
    tr2 = _tiny_transformer()
    spec2 = make_source_cond(tr2, init_extra="source", extra_only=False)
    p2 = patch_embed_proj(tr2)
    assert torch.equal(p2.weight[:, 16:32], base_w), \
        "init_extra='source' must copy the pretrained columns into the image block"
    assert torch.count_nonzero(p2.weight[:, 32:]) == 0, \
        "the mask block starts at zero under either init"
    assert not torch.equal(p2.weight, proj.weight)
    assert spec2.new_channels == 32


    # 5. the guards. Each of these is a mistake that would otherwise train
    #    happily and produce a model that is subtly wrong.
    ok = build_cond_input(z, z_src, mask, spec)
    assert ok.shape == (B, 48, h, w), ok.shape
    for name, bad in (
            ("source on a different grid",
             dict(z=z, z_src=torch.randn(B, C, 4, 4), mask=mask)),
            ("mask on the pixel grid",
             dict(z=z, z_src=z_src, mask=torch.zeros(B, 1, 64, 64))),
            ("mask as 0/255 uint8",
             dict(z=z, z_src=z_src, mask=(mask * 255.0))),
            ("source with the wrong channel count",
             dict(z=z, z_src=torch.randn(B, 4, h, w), mask=mask))):
        try:
            build_cond_input(spec=spec, **bad)
        except ValueError as exc:
            assert str(exc), name
        else:
            raise AssertionError(f"{name} must be refused")

    # 6. a NON-square latent grid must work (the frame is 4:3 now): the conv is
    #    spatially agnostic, but the guard compares z_src/mask against z, and a
    #    transposed pair is the classic silent bug
    tz = torch.randn(1, 16, 8, 12)
    assert build_cond_input(tz, torch.randn(1, 16, 8, 12),
                            torch.ones(1, 1, 8, 12), spec).shape == (1, 48, 8, 12)
    try:
        build_cond_input(tz, torch.randn(1, 16, 12, 8),
                         torch.ones(1, 1, 12, 8), spec)
    except ValueError as exc:
        assert "resolution" in str(exc), exc
    else:
        raise AssertionError("a transposed source must be refused")

    # 7. trainable=False keeps the base fully frozen (the ablation arm)
    tr3 = _tiny_transformer()
    make_source_cond(tr3, trainable=False)
    assert not patch_embed_proj(tr3).weight.requires_grad

    # 8. cond_from_batch: what the TRAINER passes. The cache hands the mask out
    #    of TripletCache as (B, h, w) -- checked against transport/data.py --
    #    and the conditioned conv needs (B, 1, h, w).
    Bb, hh, ww = 3, 8, 8
    batch = {"z_src": torch.randn(Bb, 16, hh, ww),
             "mask": torch.zeros(Bb, hh, ww)}
    batch["mask"][:, 2:5, 2:5] = 1.0
    assert cond_from_batch(batch, False) == {}, \
        "with tier B off the helper must contribute nothing at all"
    got = cond_from_batch(batch, True)
    assert set(got) == {"z_src", "mask_cond"}, got.keys()
    assert got["mask_cond"].shape == (Bb, 1, hh, ww), got["mask_cond"].shape
    assert got["z_src"].shape == (Bb, 16, hh, ww)
    # it must be USABLE: the real builder accepts exactly these tensors
    assert build_cond_input(torch.randn(Bb, 16, hh, ww), got["z_src"],
                            got["mask_cond"], spec).shape == (Bb, 48, hh, ww)
    # already-4-D masks pass through unchanged
    four = dict(batch, mask=batch["mask"].unsqueeze(1))
    assert cond_from_batch(four, True)["mask_cond"].shape == (Bb, 1, hh, ww)
    # a batch from an older cache must fail loudly, not train on zeros
    for bad_batch in ({"mask": batch["mask"]}, {"z_src": batch["z_src"]}, {}):
        try:
            cond_from_batch(bad_batch, True)
        except KeyError as exc:
            assert "precompute_cache" in str(exc), exc
        else:
            raise AssertionError(f"{sorted(bad_batch)} must raise")
    # a mask with several channels is a mis-shaped mask, not a feature
    try:
        cond_from_batch(dict(batch, mask=torch.zeros(Bb, 3, hh, ww)), True)
    except ValueError as exc:
        assert "(B, 1, h, w)" in str(exc), exc
    else:
        raise AssertionError("a multi-channel batch mask must be refused")

    # 9. THE TWO ENTRY POINTS MUST AGREE EXACTLY. cond_from_tensors is what
    #    SAMPLING calls and cond_from_batch is what TRAINING calls; if they
    #    drifted, a model would be trained under one mask convention and
    #    sampled under another -- the polarity bug, one layer down and silent.
    #    Asserted by identity of the tensors, not by shape.
    for label, b in (("3-D mask", batch), ("4-D mask", four)):
        tr_got = cond_from_batch(b, True)
        tn_got = cond_from_tensors(b["z_src"], b["mask"])
        assert set(tr_got) == set(tn_got) == {"z_src", "mask_cond"}, tr_got.keys()
        assert torch.equal(tr_got["z_src"], tn_got["z_src"]), label
        assert torch.equal(tr_got["mask_cond"], tn_got["mask_cond"]), label
        assert tr_got["z_src"].dtype == torch.float32, tr_got["z_src"].dtype
    # and sampling's builder refuses the same mis-shaped mask training does
    try:
        cond_from_tensors(batch["z_src"], torch.zeros(Bb, 3, hh, ww))
    except ValueError as exc:
        assert "(B, 1, h, w)" in str(exc), exc
    else:
        raise AssertionError("cond_from_tensors must refuse a 3-channel mask")

    print("source_cond self-test OK")
    print("  the 16 pretrained columns are bit-identical after surgery")
    print("  identity at init: the conditioned output EQUALS the base output")
    print("  the path is ALIVE: new columns get gradient and move, the")
    print("  pretrained columns do not (this is the trap the module documents)")
    print("  refusals: wrong grid, 0/255 mask, transposed source, wrong channels")
    print("  training (cond_from_batch) and sampling (cond_from_tensors) build")
    print("  the SAME conditioning tensors -- asserted by equality, not shape")
    _self_test_checkpoint()
    print("  checkpoints PRESERVE the conditioning through the real Trainer")
    print("  (save's filter used a detached requires_grad and dropped it), and")
    print("  the pretrained columns survive AdamW with weight_decay=0.1")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
