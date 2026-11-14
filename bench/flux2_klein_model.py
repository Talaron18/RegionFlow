"""FLUX.2 [klein] as a Gate C model: instruction + source image -> edited image.

    python bench/run_gate_c.py --model flux2-klein-4b --mask-mode none --limit 4 \
        --out runs/gate_c/klein4b_smoke.jsonl \
        --save-images runs/gate_c/img_klein4b_smoke

WHY A SEPARATE FILE. It needs stock diffusers recent enough to ship
`Flux2KleinPipeline` (the repo's configs say 0.37.0.dev0; the model card says
install from git). The UltraEdit arms need UltraEdit's diffusers FORK at
0.30.0.dev0. The two cannot share an environment, so run this arm from its
own venv and leave the project image alone.

NO MASK. FLUX.2 klein takes a prompt and reference image(s); no mask or
region input is documented. Only --mask-mode none is meaningful, and the
wrapper refuses anything else rather than silently dropping the mask --
a dropped mask would be scored as an "oracle" arm that never saw one.

SETTINGS are the model card's for the distilled 4B checkpoint: 4 steps,
guidance_scale 1.0. run_gate_c.py's own --steps / --guidance defaults
(50 / 7.5) are UltraEdit's and are deliberately NOT forwarded.

COMPOSITE BASELINE (--klein-composite). The model still never sees the mask.
It edits the whole image exactly as in the no-mask run (same seed, same
call), and the result is then pasted into the SOURCE through the
conditioning mask run_gate_c builds (white = editable, already dilated by
--mask-dilate), with a Gaussian feather. Everything outside the mask is the
source pixel for pixel, so sibling leakage is removed by construction. This
is the trivial thing any mask-based method on this base has to beat: it
shows what a mask alone buys, and what it costs at the boundary.

    python bench/run_gate_c.py --model flux2-klein-4b --klein-composite \
        --mask-mode oracle --out runs/gate_c/klein4b_comp_oracle.jsonl \
        --save-images runs/gate_c/img_klein4b_comp_oracle

NOT RUN ON A GPU when written. `self_test` checks the call the wrapper
makes against a stub pipeline; whether the real pipeline accepts it is
what the 4-case smoke run above is for. Look at those four images first.
"""

from __future__ import annotations

REPOS = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
}

#: FLUX.2's VAE is 1/8 with 2x2 patches, so sides must be multiples of 16.
SIZE_MULTIPLE = 16


def composite(src, edited, mask, feather: float = 4.0):
    """Paste `edited` into `src` where `mask` is white. PIL images in, PIL out.

    mask is the runner's conditioning image (RGB or L, white = editable).
    feather is a Gaussian radius in pixels on the alpha; 0 gives a hard
    edge and leaves every pixel outside the mask exactly equal to `src`.
    With feather > 0 the blend reaches a few pixels past the mask edge.
    """
    from PIL import Image, ImageFilter  # noqa: PLC0415

    src = src.convert("RGB")
    if edited.size != src.size:
        edited = edited.resize(src.size, Image.BILINEAR)
    alpha = mask.convert("L")
    if alpha.size != src.size:
        alpha = alpha.resize(src.size, Image.NEAREST)
    alpha = alpha.point(lambda v: 255 if v >= 128 else 0)
    if feather and feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return Image.composite(edited.convert("RGB"), src, alpha)


def snap_size(w: int, h: int, multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """Nearest multiple-of-`multiple` size, never below one multiple."""
    def _snap(x):
        return max(multiple, int(round(x / multiple)) * multiple)
    return _snap(w), _snap(h)


class Flux2KleinModel:
    """Same call contract as run_gate_c.SD3InstructPix2Pix."""

    def __init__(self, name: str = "flux2-klein-4b", steps: int = 4,
                 guidance: float = 1.0, device: str = "cuda",
                 offload: bool = False, pipe=None,
                 composite_mask: bool = False, feather: float = 4.0):
        if name not in REPOS:
            raise SystemExit(f"unknown klein model {name!r}; "
                             f"known: {sorted(REPOS)}")
        self.repo = REPOS[name]
        self.composite_mask, self.feather = bool(composite_mask), float(feather)
        # distinct name so composite results never pool with the plain arm
        self.name = name + ("+composite" if self.composite_mask else "")
        self.steps, self.g = int(steps), float(guidance)
        self.device, self.offload = device, bool(offload)
        if pipe is not None:            # injected (self_test)
            self.pipe = pipe
            return
        import torch  # noqa: PLC0415
        try:
            from diffusers import Flux2KleinPipeline  # noqa: PLC0415
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit(
                "Flux2KleinPipeline not found in this diffusers install. It "
                "needs a recent stock diffusers (`pip install "
                "git+https://github.com/huggingface/diffusers.git`), in a "
                "SEPARATE environment from the UltraEdit fork the other Gate "
                "C arms use.") from exc
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.repo, torch_dtype=torch.bfloat16)
        if self.offload:
            # the model card's own low-VRAM path; slower per image
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    def _generator(self, seed: int):
        import torch  # noqa: PLC0415
        # offload moves modules between devices per call; a CPU generator is
        # valid in both modes there, a CUDA one is not.
        dev = "cpu" if self.offload else self.device
        return torch.Generator(device=dev).manual_seed(int(seed))

    def __call__(self, img, prompt, seed, mask=None):
        if mask is not None and not self.composite_mask:
            raise SystemExit(
                f"{self.name} has no mask input; run it with --mask-mode none, "
                "or add --klein-composite to paste the edit back through the "
                "mask. Refusing to drop the mask silently.")
        if mask is None and self.composite_mask:
            raise SystemExit(
                "--klein-composite needs a mask to paste through; use "
                "--mask-mode oracle (or white / sibling as controls).")
        src = img
        w, h = snap_size(*img.size)
        if (w, h) != img.size:
            img = img.resize((w, h))
        edited = self.pipe(prompt=prompt, image=img, height=h, width=w,
                           num_inference_steps=self.steps,
                           guidance_scale=self.g,
                           generator=self._generator(seed)).images[0]
        if self.composite_mask:
            return composite(src, edited, mask, self.feather)
        return edited


def self_test() -> None:
    from PIL import Image  # noqa: PLC0415

    assert snap_size(512, 512) == (512, 512)
    assert snap_size(500, 333) == (496, 336)
    assert snap_size(3, 3) == (16, 16)

    calls = []

    class _Out:
        def __init__(self, im):
            self.images = [im]

    class _Pipe:
        def __call__(self, **kw):
            calls.append(kw)
            return _Out(Image.new("RGB", (kw["width"], kw["height"])))

    m = Flux2KleinModel(pipe=_Pipe(), device="cpu")
    out = m(Image.new("RGB", (500, 333)), "make the left cup blue", seed=7)
    kw = calls[-1]
    assert set(kw) == {"prompt", "image", "height", "width",
                       "num_inference_steps", "guidance_scale", "generator"}, kw
    assert kw["prompt"] == "make the left cup blue"
    assert (kw["width"], kw["height"]) == (496, 336) == kw["image"].size
    assert out.size == (496, 336)
    # the card's distilled settings, not run_gate_c's UltraEdit defaults
    assert kw["num_inference_steps"] == 4 and kw["guidance_scale"] == 1.0

    # same seed -> same generator stream; different seed -> different
    import torch  # noqa: PLC0415
    a = torch.rand(3, generator=m._generator(7))
    b = torch.rand(3, generator=m._generator(7))
    c = torch.rand(3, generator=m._generator(8))
    assert torch.equal(a, b) and not torch.equal(a, c)

    # a mask must be refused, never dropped
    try:
        m(Image.new("RGB", (512, 512)), "x", seed=0,
          mask=Image.new("RGB", (512, 512)))
    except SystemExit:
        pass
    else:
        raise AssertionError("mask was accepted")
    # composite: the pipeline call is IDENTICAL to the no-mask arm (same
    # kwargs, no mask anywhere), and only the paste differs.
    import numpy as np  # noqa: PLC0415

    class _Red:
        def __call__(self, **kw):
            calls.append(kw)
            return _Out(Image.new("RGB", (kw["width"], kw["height"]), (255, 0, 0)))

    src = Image.new("RGB", (512, 512), (0, 0, 255))
    msk = Image.new("RGB", (512, 512), (0, 0, 0))
    msk.paste((255, 255, 255), (100, 100, 200, 200))
    hard = Flux2KleinModel(pipe=_Red(), device="cpu", composite_mask=True, feather=0)
    assert hard.name == "flux2-klein-4b+composite"
    o = np.asarray(hard(src, "x", seed=7, mask=msk))
    assert set(calls[-1]) == set(kw), "composite changed the pipeline call"
    inside = np.zeros((512, 512), bool); inside[100:200, 100:200] = True
    assert (o[inside] == (255, 0, 0)).all()
    assert (o[~inside] == (0, 0, 255)).all(), "pixels outside the mask changed"
    soft = Flux2KleinModel(pipe=_Red(), device="cpu", composite_mask=True, feather=4)
    o = np.asarray(soft(src, "x", seed=7, mask=msk))
    assert tuple(o[150, 150]) == (255, 0, 0)            # centre: the edit
    far = np.ones((512, 512), bool); far[70:230, 70:230] = False
    assert (o[far] == (0, 0, 255)).all(), "feather reached far from the mask"
    assert 0 < o[100, 150, 0] < 255                     # edge: a blend
    # mask size differing from the source is resized, not an error
    o = np.asarray(hard(src, "x", seed=7, mask=msk.resize((256, 256))))
    assert tuple(o[150, 150]) == (255, 0, 0) and tuple(o[20, 20]) == (0, 0, 255)
    # white mask == the plain model output (the "edit anywhere" control)
    o = np.asarray(hard(src, "x", seed=7, mask=Image.new("RGB", (512, 512), "white")))
    assert (o == (255, 0, 0)).all()
    try:
        hard(src, "x", seed=0)
    except SystemExit:
        pass
    else:
        raise AssertionError("composite without a mask was accepted")

    try:
        Flux2KleinModel(name="flux2-klein-99b", pipe=_Pipe())
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown name was accepted")
    print("flux2_klein_model self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
