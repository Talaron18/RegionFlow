"""
Backends for short_term: image IO, a CPU toy field, and the real editor.

Everything heavy (diffusers, the UltraEdit fork, the SD3 checkpoints) is
imported lazily inside the class that needs it, so this module -- and
`guidance.py` and `demo_toy.py` -- import and run on a CPU-only box with no
weights. That is deliberate: it is what lets the guidance core be tested here
rather than on the GPU box.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

try:                                   # imported as part of the package
    from .guidance import (GuidanceTerm, guided_velocity, one_step_clean,
                           renoise)
except ImportError:                    # run as a script: --self-test
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from short_term.guidance import (GuidanceTerm, guided_velocity,
                                    one_step_clean, renoise)

#: White = editable. This is the convention bench/run_gate_c.py uses for every
#: mask arm (its README polarity note: "White = editable, per UltraEdit's
#: README"), and the project has already lost a run to a polarity bug. Training
#: and evaluation must agree, so `short_term` inherits it rather than picking
#: its own.
WHITE_IS_EDITABLE = True


# ------------------------------------------------------------------- IO -----

def parse_size(size) -> tuple[int, int] | None:
    """-> (w, h) from 512, "512", "512x384", "512,384", (512, 384) or (w, h).

    ONE convention, stated once: "WxH", width first, matching `patch_rect`'s
    (x, y, w, h) and the way a person says "4:3, wide". numpy's (H, W) order is
    the opposite, which is exactly why this is a named function with a test
    instead of a cast at each call site.
    """
    if size is None:
        return None
    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError(f"size needs two numbers, got {size!r}")
        w, h = int(size[0]), int(size[1])
    elif isinstance(size, str):
        text = size.strip().lower().replace(",", "x").replace("*", "x")
        text = text.replace(" ", "")
        if ":" in text:
            raise ValueError(
                f"{size!r} is a ratio, not a pixel size. Give pixels, e.g. "
                f"'512x384' for 4:3 (or use app.frame_size('4:3')).")
        if "x" not in text:
            w = h = int(text)
        else:
            a, _, b = text.partition("x")
            w, h = int(a), int(b)
    else:
        w = h = int(size)
    if w <= 0 or h <= 0:
        raise ValueError(f"size must be positive, got {w}x{h}")
    return w, h


def load_image(path: str | Path, size=None) -> np.ndarray:
    """-> (H, W, 3) float32 in [0, 1]. `size` is WxH (see parse_size)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    want = parse_size(size)
    if want is not None and img.size != want:
        img = img.resize(want, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(arr: np.ndarray, path: str | Path) -> None:
    """(H, W, 3) in [0, 1] or (H, W) mask -> file. Creates parent dirs."""
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[..., None].repeat(3, axis=2)
    a = np.clip(a, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


def load_mask(path: str | Path, size=None,
              shape: tuple[int, int] | None = None) -> np.ndarray:
    """-> (H, W) float32 in {0, 1}, white = editable.

    `size` is WxH (see parse_size); `shape` is (H, W), numpy order. Both exist
    because callers legitimately have one or the other in hand.

    Accepts a path to an image, or the literal string "white" for a mask that
    is editable everywhere (which is the no-spatial-information control arm in
    Gate C, not the same thing as "no mask"). A Magenta/alpha PNG from
    MagicBrush-style datasets is NOT handled here on purpose -- the project
    measured that trap (`STAGE0_FINDINGS` SS6: the mask is `alpha < 128`, and
    converting to L composites the mask away and measures image brightness), so
    pass an explicit binary mask instead of guessing.
    """
    if isinstance(path, str) and path == "white":
        want = parse_size(size)
        if shape is None and want is None:
            raise ValueError('load_mask("white") needs size= or shape=')
        h, w = shape if shape is not None else (want[1], want[0])
        return np.ones((h, w), dtype=np.float32)
    from PIL import Image

    m = Image.open(path)
    if "A" in m.getbands():
        m = m.getchannel("A")
    else:
        m = m.convert("L")
    want = parse_size(size)
    if want is not None and m.size != want:
        m = m.resize(want, Image.NEAREST)
    arr = np.asarray(m, dtype=np.float32) / 255.0
    if not WHITE_IS_EDITABLE:
        arr = 1.0 - arr
    return (arr > 0.5).astype(np.float32)


def mask_to_latent(mask_px: np.ndarray, h: int, w: int,
                   hard: bool = False) -> torch.Tensor:
    """(H, W) pixel mask -> (1, 1, h, w) guidance weight.

    Takes the pixel-space (H, W) mask (from `load_mask`), NOT the 3-channel
    array `to_pipeline_mask` produces for the pipeline -- the guidance weight
    has one channel on purpose.

    AREA-AVERAGED, and soft by default. Two reasons the default is soft rather
    than the thresholded binary that `transport.conditioning.mask_to_latent_grid`
    uses: this is a *guidance weight*, not a conditioning input, and a hard
    boundary in a guidance weight is the same hard blend that
    `region/alpha_field.py` was written to avoid ("hard blending at a mask
    boundary is exactly what produces the seam artefacts"). Pass `hard=True`
    for the conditioning convention when you want the two to be comparable.
    """
    arr = np.asarray(mask_px)
    if arr.ndim == 3:             # handed the pipeline's 3-channel mask
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.size and arr.max() > 1.5:   # uint8 {0,255} rather than {0,1}
        arr = arr / 255.0
    m = torch.from_numpy(arr)[None, None]
    pooled = torch.nn.functional.adaptive_avg_pool2d(m, (h, w))
    return (pooled > 0.5).float() if hard else pooled


# ------------------------------------------------------------- toy field ----

def to_pipeline_image(arr) -> np.ndarray:
    """-> (H, W, 3) uint8, what the pipeline's `image=` wants.

    bench/run_gate_c.py passes numpy uint8 at `image=`, so match it rather than
    relying on the fork's image processor to also accept float [0,1].
    """
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    return (np.clip(a.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_pipeline_mask(mask_px) -> np.ndarray | None:
    """(H, W) float {0,1} -> (H, W, 3) uint8, white = editable.

    WHY THE MASK NEEDS THREE CHANNELS. It is not a mask argument that gets
    thresholded -- it is a conditioning IMAGE that goes through the same VAE as
    the source. `pipeline_stable_diffusion_3_instructpix2pix.py` line ~859:

        mask_image_latents = self.prepare_image_latents(...)
        ...  retrieve_latents(self.vae.encode(image), sample_mode="argmax")

    and the VAE encoder's first conv is 3-channel, so a (H, W) mask dies with

        RuntimeError: Given groups=1, weight of size [128, 3, 3, 3],
        expected input[1, 1, 512, 512] to have 3 channels, but got 1

    bench/run_gate_c.py's `cond_mask` has always done this
    (`Image.fromarray(...).convert("RGB")`, and `Image.new("RGB", ...)` for the
    white arm); this function is the same thing with numpy instead of PIL.
    """
    if mask_px is None:
        return None
    m = (np.asarray(mask_px) > 0.5).astype(np.uint8) * 255
    return np.repeat(m[..., None], 3, axis=2)


def check_pipeline_inputs(image: np.ndarray, mask: np.ndarray | None) -> None:
    """Fail here, with a readable message, rather than 20 frames deep in a conv.

    `region/pipeline.py`'s own note: "Every silent-failure mode in this project
    is a shape or ordering mistake". This is that check, at the boundary where
    the convention actually lives.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"source image must be (H, W, 3) uint8, got {image.shape} "
            f"{image.dtype}. Use backends.to_pipeline_image().")
    if mask is None:
        return
    if mask.ndim != 3 or mask.shape[2] != 3 or mask.dtype != np.uint8:
        raise ValueError(
            f"mask_img must be (H, W, 3) uint8 -- the pipeline VAE-encodes it, "
            f"so a single-channel mask raises in the VAE's first conv. Got "
            f"{mask.shape} {mask.dtype}. Use backends.to_pipeline_mask().")
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"mask {mask.shape[:2]} and image {image.shape[:2]} disagree; the "
            "pipeline will resize them independently and the edit region will "
            "not be where you think it is.")


def _step_index(sched):
    """The scheduler's current index, whichever private name this version uses."""
    for attr in ("step_index", "_step_index"):
        v = getattr(sched, attr, None)
        if v is not None:
            return int(v)
    return None


@contextmanager
def guided_scheduler(pipe, store: dict, terms=(), kappa_fn=None):
    """Inject guidance at `scheduler.step` -- the layer that actually steps.

    WHY THIS LAYER, AND WHAT THE TRANSFORMER HOOK GOT WRONG.

    The first guided run wrapped `pipe.transformer.forward` and hooked the
    denoiser's output. That was two layers too early, and both mistakes are
    visible in its log:

    1. "CFG batch detected: transformer saw 3x ... using chunk 2/2". The
       InstructPix2Pix loop evaluates three branches (uncond / text / image) in
       one batched call and then COMBINES them

           v = v_uncond + g (v_text - v_uncond) + ig (v_image - v_text)

       `scheduler.step` is handed that combination -- that is the velocity of
       the ODE. A single raw chunk is not, so the run was guiding a different
       vector from the one being integrated, and the choice of chunk was a
       guess that could not be checked.
    2. "48 channels ... latent block found at [0:16] (deviation 1.58e-02)".
       `hidden_states` carries the 16 image and 16 mask blocks; the scheduler
       never sees those. Reading sigma from `_step_index` *inside* the
       transformer call also failed on step 0 ("guidance was skipped on 1
       step"), because the index is not set up yet at that point.

    `scheduler.step(model_output, timestep, sample)` has none of these
    ambiguities: `sample` is exactly the 16-channel latent, `model_output` is
    exactly the velocity the solver integrates, and `sigmas[step_index]` is by
    the scheduler's own contract the sigma for THIS step. So guidance goes in
    as a correction to `model_output` -- literally equation (2),
    `v_guided = v + dv` -- with no chunk search, no channel search, and no
    sigma search.

    It also removes the dependency on `callback_on_step_end`, which an
    unfamiliar fork may or may not support.

    `store` receives: calls, idx, sigma, sigma_next, lat_ch, z, v, lat,
    max_dev, last_dev. `max_dev` is
    `|prev_sample - (sample + (sigma_next - sigma) * model_output)|`: ZERO for
    any solver whose step is `sample + ds * v` (which is what the re-noising
    algebra and the direct-kappa conversion assume). Non-zero means the solver
    is something else, and says so rather than silently mis-guiding.
    """
    sched = getattr(pipe, "scheduler", None)
    if sched is None or not hasattr(sched, "step"):
        raise RuntimeError(
            "pipe has no `.scheduler.step`; short_term's guided path enters "
            "there. Check that the UltraEdit diffusers fork is the installed "
            "one (stock diffusers has no "
            "StableDiffusion3InstructPix2PixPipeline at all).")
    orig = sched.step
    store.setdefault("calls", 0)
    store.setdefault("max_dev", 0.0)

    def wrapped(model_output, timestep, sample, *args, **kwargs):
        sigmas = getattr(sched, "sigmas", None)
        idx = _step_index(sched)
        sig = sig_n = None
        if sigmas is not None and idx is not None and idx < len(sigmas):
            sig = float(sigmas[idx])
            if idx + 1 < len(sigmas):
                sig_n = float(sigmas[idx + 1])

        store["calls"] += 1
        store["idx"] = idx
        store["sigma"] = sig
        store["sigma_next"] = sig_n
        store["lat_ch"] = (int(model_output.shape[1])
                           if torch.is_tensor(model_output) else None)
        store["z"] = sample.detach() if torch.is_tensor(sample) else None
        store["v"] = (model_output.detach()
                      if torch.is_tensor(model_output) else None)

        v_use = model_output
        if terms and sig is not None and sig_n is not None and sig_n != sig:
            if kappa_fn is None:
                v_use = guided_velocity(sample, sig, model_output, terms)
            else:
                # The direct form corrects the clean estimate and re-noises.
                # Express the result as an EQUIVALENT velocity so the solver
                # still does the stepping -- exact for a `sample + ds*v`
                # solver, which `max_dev` verifies.
                z_new = renoise(sample, sig, model_output, sig_n, terms,
                                direct_kappa=kappa_fn)
                ref = sample + (sig_n - sig) * model_output
                v_use = model_output + (z_new - ref) / (sig_n - sig)

        out = orig(v_use, timestep, sample, *args, **kwargs)

        prev = getattr(out, "prev_sample", None)
        if (torch.is_tensor(prev) and sig is not None and sig_n is not None
                and torch.is_tensor(model_output)):
            ref = sample + (sig_n - sig) * model_output
            dev = float((prev - ref).abs().mean())
            store["max_dev"] = max(store["max_dev"], dev)
            store["last_dev"] = dev
            store["lat"] = prev.detach()
        return out

    sched.step = wrapped
    try:
        yield store
    finally:
        sched.step = orig


@dataclass
class ToyMixtureField:
    """Exact marginal velocity field of a piecewise-constant two-mode target.

    The data is x0(p) = sum_r m_r * 1_r(p), with one independent sign m_r in
    {-1, +1} per region r, and eps ~ N(0, I) iid. z_t = (1-t) x0 + t eps, so

        v*(z, t) = E[eps - x0 | z_t = z] = (z - E[x0 | z]) / t

    and the posterior factorises over regions, each region's posterior being

        p(m_r = s | z) proportional to exp(-||(z - (1-t) s) * 1_r||^2 / (2 t^2))

    WHY THIS TOY IS THE RIGHT ONE. It is the smallest object that exhibits both
    of the things this project has been arguing about. Its data distribution is
    genuinely multimodal (2^len(regions) modes). Its *marginal* field is a
    conditional expectation -- the same mathematical object that collapses to a
    smear under MSE regression in Gate A -- and yet sampling from it recovers
    the distribution exactly, because the mode is selected by `z_init` (the
    noise), not by the conditioning. That is the whole claim of the noise-side
    route, in closed form, on CPU, in milliseconds.

    It is NOT a model of a real editor: regions are independent and the target
    is piecewise constant. Treat it as an instrument for the mechanism, the way
    `tests/test_composition_math.py`'s known-field control is an instrument for
    the composition metric.
    """

    regions: list[torch.Tensor]
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.regions)
        if n == 0:
            raise ValueError("ToyMixtureField needs at least one region")
        self.weights = list(self.weights) if self.weights else [1.0] * n
        if len(self.weights) != n:
            raise ValueError("weights must match regions")

    @classmethod
    def halves(cls, h: int, w: int, b: int = 1, dtype=torch.float64
               ) -> "ToyMixtureField":
        """Two regions: left half, right half. The natural setup for a local
        edit -- two independent attributes in one latent."""
        left = torch.zeros(b, 1, h, w, dtype=dtype)
        left[:, :, :, : w // 2] = 1.0
        right = 1.0 - left
        return cls([left, right])

    def posterior_mean(self, z: torch.Tensor, t: float) -> torch.Tensor:
        t_c = max(float(t), 1e-6)
        out = torch.zeros_like(z)
        for reg, wgt in zip(self.regions, self.weights):
            logits = []
            for s in (1.0, -1.0):
                d = (z - (1.0 - t_c) * s) * reg
                logits.append(float(np.log(wgt)) - (d ** 2).sum() / (2 * t_c ** 2))
            post = torch.softmax(torch.stack(logits), dim=0)
            out = out + (post[0] * 1.0 + post[1] * (-1.0)) * reg
        return out

    def __call__(self, z: torch.Tensor, t: float) -> torch.Tensor:
        return (z - self.posterior_mean(z, t)) / max(float(t), 1e-6)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """The "image" a latent corresponds to, in this toy: the sign it sits
        on, per region. Not a VAE -- just enough to read a mode off."""
        return torch.sign(self.posterior_mean(z, 0.01))


# ------------------------------------------------------------ real editor ---

# `captured_velocity` (the transformer.forward hook) lived here and is
# deliberately GONE, not merely unused. It hooked the denoiser two layers above
# the solver, which forced three guesses that all showed up as warnings in the
# first guided run: which of the 3 CFG chunks is "the" velocity, which of the 3
# channel blocks is the latent, and which sigma index `_step_index` means from
# inside the transformer call. `guided_scheduler` above answers all three by
# construction. See `edit_guided`'s own note and short_term/README.md SS3.


class SD3IP2PEditor:
    """An existing instruction editor, plus a guidance hook.

    `edit()` is tier A: call the checkpoint's own pipeline and write an image.
    `edit_guided()` is tier D: the same call, with a `callback_on_step_end` that
    corrects the clean estimate at every step -- so the checkpoint is never
    touched and the guidance can be switched off by passing no terms.

    UNVERIFIED ON THIS BOX. The CPU-only environment this package was written
    in has stock diffusers 0.36.0, which does not contain
    `StableDiffusion3InstructPix2PixPipeline` at all. Everything below is
    therefore written defensively and logs what it saw; the pieces that ARE
    verified here are `guidance.py` and `sampler.py` (see `--self-test`). Run
    `edit.py --dry-run` on the GPU box first: it constructs the callback and
    prints the scheduler arrays without loading a model.
    """

    def __init__(self, repo: str, device: str = "cuda", dtype=None,
                 steps: int = 50, image_guidance: float = 1.5,
                 guidance: float = 7.5):
        import torch as _torch
        try:
            from diffusers import StableDiffusion3InstructPix2PixPipeline
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit(
                "StableDiffusion3InstructPix2PixPipeline not found. It takes a\n"
                "`mask_img` argument and comes from UltraEdit's diffusers fork,\n"
                "not stock diffusers -- the same dependency bench/run_gate_c.py\n"
                "already documents. Install that fork before using this backend."
            ) from exc
        self.torch = _torch
        self.repo = repo
        self.device = device
        self.steps = steps
        self.ig = image_guidance
        self.g = guidance
        self.pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            repo, torch_dtype=dtype or _torch.float16).to(device)
        self.name = f"ip2p:{repo}"

    # -- tier A ------------------------------------------------------------
    def edit(self, image: np.ndarray, prompt: str, mask: np.ndarray | None,
             seed: int = 0, **over):
        """`image` may be (H,W,3) float [0,1] or uint8; `mask` is the (H,W)
        float {0,1} pixel mask from `load_mask`. Both are converted HERE, at
        the single choke point, so no caller can hand the pipeline a
        single-channel mask and get a VAE conv error 20 frames deep."""
        img_u8 = to_pipeline_image(image)
        mask_u8 = to_pipeline_mask(mask)
        check_pipeline_inputs(img_u8, mask_u8)
        gen = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            prompt, image=img_u8, mask_img=mask_u8,
            num_inference_steps=over.get("steps", self.steps),
            image_guidance_scale=over.get("image_guidance", self.ig),
            guidance_scale=over.get("guidance", self.g),
            generator=gen).images[0]

    # -- tier D ------------------------------------------------------------
    def edit_guided(self, image: np.ndarray, prompt: str, mask: np.ndarray | None,
                    terms, kappa_fn=None, seed: int = 0, verbose: bool = True,
                    cfg_chunk="auto", **over):
        """Same call, with guidance injected at `scheduler.step`.

        WHAT CHANGED AND WHY. The first version hooked `transformer.forward` and
        applied the correction as a latent delta from a
        `callback_on_step_end`. That hooked the denoiser two layers above the
        solver, and its log said so three times over:

            [guided] CFG batch detected: transformer saw 3x ... using chunk 2/2
            [guided] transformer input has 48 channels; latent block found at
                     [0:16] (deviation from the scheduler step 1.58e-02)
            [guided] WARNING: no channel block reproduces the scheduler's step
                     to 1e-2 ...
            [guided] WARNING: guidance was skipped on 1 step(s)

        Every one of those is a guess that hook forced: which of 3 CFG chunks is
        the velocity the ODE actually integrates (none of them -- the pipeline
        COMBINES them first), which of 3 concatenated channel blocks is the
        latent, and which sigma `_step_index` means when read from inside the
        transformer call (on step 0 it is not set up, hence the skipped step).

        `scheduler.step(model_output, timestep, sample)` has none of those
        ambiguities, so guidance now goes in there as a correction to
        `model_output` -- equation (2), `v_guided = v + dv`. Consequences:
        `cfg_factor`, `latent_block` and the channel search are all gone from
        the record because there is nothing left to guess, and the guided path
        no longer needs `callback_on_step_end` support in the fork at all.

        `scheduler_dev` is the remaining check and the one that matters: it is
        zero for any solver that steps `sample + ds * v`. A non-zero value means
        the re-noising algebra does not describe this solver, and the run says
        so rather than producing confidently mis-guided images.
        """
        img_u8 = to_pipeline_image(image)
        mask_u8 = to_pipeline_mask(mask)
        check_pipeline_inputs(img_u8, mask_u8)

        store: dict = {}

        gen = self.torch.Generator(device=self.device).manual_seed(seed)
        with guided_scheduler(self.pipe, store, terms=terms, kappa_fn=kappa_fn):
            out = self.pipe(
                prompt, image=img_u8, mask_img=mask_u8,
                num_inference_steps=over.get("steps", self.steps),
                image_guidance_scale=over.get("image_guidance", self.ig),
                guidance_scale=over.get("guidance", self.g),
                generator=gen).images[0]

        state = {
            "steps": store.get("calls", 0),
            "max_dev": store.get("max_dev", 0.0),
            "last_dev": store.get("last_dev"),
            "lat_ch": store.get("lat_ch"),
        }
        if verbose:
            print(f"[guided] {state['steps']} scheduler step(s); "
                  f"latent {state['lat_ch']}ch; "
                  f"max |solver_step - (sample + ds*v)| = "
                  f"{state['max_dev']:.3e}")
            if state["steps"] == 0:
                print("[guided] WARNING: scheduler.step was never called. "
                      "Either this pipeline does not step through "
                      "`self.scheduler.step` or the loop never ran. Guidance "
                      "was a no-op.")
            elif state["max_dev"] > 1e-2:
                print("[guided] WARNING: the solver's step is not "
                      "`sample + ds*v` to 1e-2, so it is not a plain Euler "
                      "flow-match loop. Injecting into model_output is the "
                      "right place for ANY solver, but the direct-kappa form's "
                      "velocity conversion assumes this form -- prefer "
                      "--kappa 0 (the velocity form) here.")
        return out, state

    # -- latents -----------------------------------------------------------
    @property
    def latent_shape(self) -> tuple[int, int]:
        vae = self.pipe.vae
        cfg = vae.config
        s = getattr(cfg, "sample_size", 64)
        f = 2 ** (len(getattr(cfg, "block_out_channels", [1, 2, 3, 4])) - 1)
        return (int(s) // f, int(s) // f)

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """(H, W, 3) in [0, 1] -> (1, C, h, w) in the pipeline's latent space.

        Same convention as transport/pairflow.py's perceptual loss, read
        backwards: there, decode is `z / scaling + shift`, so encode is
        `(z_raw - shift) * scaling`.
        """
        vae = self.pipe.vae
        dtype = next(vae.parameters()).dtype
        x = torch.from_numpy(np.asarray(image, dtype=np.float32))
        x = x.permute(2, 0, 1)[None].to(self.device, dtype) * 2.0 - 1.0
        with self.torch.no_grad():
            z = vae.encode(x).latent_dist.sample()
        scale = float(vae.config.scaling_factor)
        shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
        return (z - shift) * scale

    def decode_latent(self, z: torch.Tensor) -> np.ndarray:
        vae = self.pipe.vae
        dtype = next(vae.parameters()).dtype
        scale = float(vae.config.scaling_factor)
        shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
        with self.torch.no_grad():
            img = vae.decode((z / scale + shift).to(dtype)).sample
        img = img.float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
        return (img + 1.0) / 2.0


# ------------------------------------------------------------- self-test ----

def self_test() -> None:
    """CPU, no weights. The pipeline-input convention is the piece that has
    already cost a run on the GPU box, so it gets a test here like every other
    shape convention in this project.

        python short_term/backends.py --self-test
    """
    h, w = 8, 8
    px = np.zeros((256, 256), dtype=np.float32)
    px[64:192, 64:192] = 1.0

    # the mask handed to the pipeline is a 3-channel uint8 IMAGE, white inside
    m3 = to_pipeline_mask(px)
    assert m3.shape == (256, 256, 3), m3.shape
    assert m3.dtype == np.uint8, m3.dtype
    assert set(np.unique(m3)).issubset({0, 255}), np.unique(m3)
    assert abs(float((m3[..., 0] > 0).mean()) - float(px.mean())) < 1e-6
    assert to_pipeline_mask(None) is None
    # and it is the SAME thing run_gate_c.py's cond_mask builds
    from PIL import Image
    ref = np.asarray(Image.fromarray(((px > 0) * 255).astype(np.uint8))
                     .convert("RGB"))
    assert np.array_equal(m3, ref), "differs from run_gate_c.cond_mask's RGB"
    # a (H,W) mask is exactly what used to reach the VAE and blow up there
    try:
        check_pipeline_inputs(np.zeros((256, 256, 3), np.uint8), px)
    except ValueError as exc:
        assert "3, 3, 3" in str(exc) or "uint8" in str(exc), str(exc)
    else:
        raise AssertionError("a (H,W) mask must be rejected at the boundary")

    # source image: float [0,1] and uint8 both land on uint8 RGB
    img_f = np.random.default_rng(0).random((256, 256, 3)).astype(np.float32)
    u8 = to_pipeline_image(img_f)
    assert u8.shape == (256, 256, 3) and u8.dtype == np.uint8
    assert np.array_equal(to_pipeline_image(u8), u8)
    assert abs(float(u8[..., 0].mean()) / 255.0 - float(img_f[..., 0].mean())) \
        < 2e-3
    check_pipeline_inputs(u8, m3)

    # the guidance weight is ONE channel and survives both mask spellings
    lat = mask_to_latent(px, h, w)
    assert lat.shape == (1, 1, h, w), lat.shape
    assert abs(float(lat.mean()) - float(px.mean())) < 1e-6, \
        "area pooling must preserve the mask mean"
    assert torch.allclose(mask_to_latent(m3, h, w), lat)
    assert torch.allclose(mask_to_latent(px * 255.0, h, w), lat)
    assert torch.allclose(mask_to_latent(px, h, w, hard=True),
                          (lat > 0.5).float())
    # white mask = editable everywhere, which is NOT the same as "no mask"
    assert float(mask_to_latent(np.ones((256, 256), np.float32), h, w).mean()) \
        == 1.0

    # ---- the guidance seam, on a fake scheduler (the real one needs a GPU) --
    # This is the test that would have caught the layer mistake: it asserts
    # guidance reaches the SOLVER's velocity, and that the step-form check
    # fires on a solver that is not `sample + ds*v`.
    from types import SimpleNamespace

    class _EulerSched:
        """A minimal `sigmas`/`step_index`/`step` triple, same contract as
        FlowMatchEulerDiscreteScheduler."""

        def __init__(self, n=6):
            self.sigmas = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
            self.step_index = 0

        def step(self, model_output, timestep, sample, **kw):
            s = float(self.sigmas[self.step_index])
            sn = float(self.sigmas[self.step_index + 1])
            prev = sample + (sn - s) * model_output
            self.step_index += 1
            return SimpleNamespace(prev_sample=prev)

    class _NotEulerSched(_EulerSched):
        def step(self, model_output, timestep, sample, **kw):
            out = super().step(model_output, timestep, sample, **kw)
            return SimpleNamespace(prev_sample=out.prev_sample * 1.05)

    bb, cc, hh, ww = 1, 16, 4, 4
    g2 = torch.Generator().manual_seed(1)
    z0 = torch.randn(bb, cc, hh, ww, generator=g2, dtype=torch.float64)
    vel = torch.randn(bb, cc, hh, ww, generator=g2, dtype=torch.float64)
    y = torch.ones(bb, cc, hh, ww, dtype=torch.float64)
    wfull = torch.ones(bb, 1, hh, ww, dtype=torch.float64)
    term = GuidanceTerm(beta=2.0, weight=wfull, target=y, name="pull")

    # reference: the scheduler's own step, no hook at all. (The first version of
    # this test computed it INSIDE the guided context and then asserted the
    # guided result differed from it -- so of course it did not.)
    unguided = _EulerSched().step(vel, 0.5, z0).prev_sample

    pipe = SimpleNamespace(scheduler=_EulerSched())
    store: dict = {}
    with guided_scheduler(pipe, store, terms=[term]):
        guided = pipe.scheduler.step(vel, 0.5, z0).prev_sample
    assert store["calls"] == 1 and store["idx"] == 0, store
    assert abs(store["sigma"] - 1.0) < 1e-12, store["sigma"]
    assert abs(store["sigma_next"] - (1.0 - 1.0 / 6)) < 1e-12, store["sigma_next"]
    assert store["lat_ch"] == cc, store["lat_ch"]
    # the hook is restored afterwards
    assert pipe.scheduler.step.__func__ is _EulerSched.step

    # GUIDANCE REACHED THE SOLVER -- this is the assertion that would have
    # caught hooking the transformer instead of the scheduler.
    assert not torch.allclose(guided, unguided), \
        "guidance never reached the solver -- this is the layer bug"
    # The per-step blend is kappa = t*gain, so beta must be < 1/t at t=1 for the
    # error to SHRINK rather than flip. (Written the first time with beta=2.0
    # and the assertion below failed -- correctly: beta=2 at t=1 doubles-then-
    # mirrors the error instead of reducing it. That is a property of the
    # paper's gain, now stated in guidance.clean_blend.)
    small = GuidanceTerm(beta=0.5, weight=wfull, target=y, name="pull-soft")
    pipe = SimpleNamespace(scheduler=_EulerSched())
    store_s: dict = {}
    with guided_scheduler(pipe, store_s, terms=[small]):
        soft = pipe.scheduler.step(vel, 0.5, z0).prev_sample
    ds = store_s["sigma_next"] - store_s["sigma"]
    x0_s = one_step_clean(z0, store_s["sigma"], vel + (soft - unguided) / ds)
    x0_u = one_step_clean(z0, store_s["sigma"], vel)
    assert float((x0_s - y).abs().mean()) < float((x0_u - y).abs().mean()), \
        "a beta < 1/t correction must reduce the distance to the target"

    # no terms -> bit-identical to the unguided step
    pipe = SimpleNamespace(scheduler=_EulerSched())
    store = {}
    with guided_scheduler(pipe, store, terms=[]):
        same = pipe.scheduler.step(vel, 0.5, z0).prev_sample
    assert torch.equal(same, unguided)
    assert store["max_dev"] < 1e-12, store["max_dev"]

    # the step-form check FIRES on a solver that is not `sample + ds*v`
    pipe = SimpleNamespace(scheduler=_NotEulerSched())
    store = {}
    with guided_scheduler(pipe, store, terms=[term]):
        pipe.scheduler.step(vel, 0.5, z0)
    assert store["max_dev"] > 1e-2, \
        "a non-Euler solver must be detected, or the warning is decoration"

    print("backends self-test OK")
    print("  mask_img is (H,W,3) uint8 and byte-equal to run_gate_c.cond_mask")
    print("  a (H,W) mask is rejected at the boundary, not inside the VAE")
    print("  guidance weight is (1,1,h,w) and pooling preserves the mean")
    print("  guidance reaches scheduler.step (the layer the solver uses), and")
    print("  a non-Euler solver is detected rather than silently mis-guided")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
