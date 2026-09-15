"""
Encode editing triplets once; train many times.

    python scripts/precompute_cache.py --dataset magicbrush --root /data/magicbrush \
        --out /data/cache --limit 4000

Runs the VAE and all three text encoders, writes latents + embeddings to disk,
then exits. Training never loads T5 again. On 24GB that is the difference
between fitting and not: T5-XXL is ~9.5GB in bf16 and contributes nothing after
this step.

DETERMINISM. The VAE posterior is taken at its MODE, not sampled, exactly as
bench/inversion.py::encode_image does. Sampling would inject a different noise
into z_src and z_tgt and put a floor under the pair-flow loss that looks like
model error.

MASKS. MagicBrush's mask is the ALPHA channel with transparent = edit region
(`alpha < 128`), not an RGB channel. Stage 0 measured the wrong channel once
and produced a confident polarity claim that had to be withdrawn. bench/data.py
already encodes the right convention; this script calls it rather than
reimplementing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.conditioning import mask_to_latent_grid   # noqa: E402
from transport.data import CacheConfig, TripletCacheWriter  # noqa: E402


def load_pipeline(model: str, device: str):
    import torch
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_image(pipe, img: np.ndarray, device: str):
    import torch

    x = np.asarray(img, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1)[None] * 2.0 - 1.0
    t = t.to(device, pipe.vae.dtype)
    with torch.no_grad():
        lat = pipe.vae.encode(t).latent_dist.mode()
    cfg = pipe.vae.config
    return ((lat - cfg.shift_factor) * cfg.scaling_factor).float()[0].cpu().numpy()


def encode_text(pipe, text: str, device: str):
    """max_sequence_length is REQUESTED at 256, not left at the pipeline
    default -- see scripts/gate_b_meanflow.py::encode_null for why: a
    different `diffusers` install on sys.path can change that default and
    silently write a cache at a context length training's own encode_null
    does not reproduce, which then fails 40 steps into a run instead of here.

    Only passed if the installed encode_prompt's signature actually accepts
    it (checked, not assumed) -- a fork can drop the parameter entirely, and
    calling with an unknown kwarg raises TypeError before any of that context
    length reasoning even applies. TripletCacheWriter.add still asserts the
    written shape against CacheConfig.context_length regardless, so a
    mismatch this doesn't catch is caught there, at write time, per file.
    """
    import inspect

    import torch

    kwargs = dict(prompt=text, prompt_2=None, prompt_3=None, device=device,
                 do_classifier_free_guidance=False)
    if "max_sequence_length" in inspect.signature(
            pipe.encode_prompt).parameters:
        kwargs["max_sequence_length"] = 256
    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(**kwargs)
    return pe[0].float().cpu().numpy(), pooled[0].float().cpu().numpy()


def _missing_root_message(root: Path) -> str:
    """Explain a missing root, and look for the dataset where it probably is.

    The common failure is a HOST path passed to a CONTAINER-side script. `~`
    is expanded by the shell against the container user (/home/dev), while the
    data sits on the WSL host (/home/syarol) and is visible only through a
    mount. The paths look plausible and the error looks like a missing
    dataset, so search the usual mount points and name what is actually there.
    """
    lines = [f"{root} does not exist"]

    if str(root).startswith("/home/") and not Path("/home").joinpath(
            str(root).split("/")[2]).exists():
        lines.append(
            "  This looks like a HOST path used inside a container: `~` "
            "expanded to a container user that has no such directory. Pass "
            "the MOUNTED path instead (this project mounts data at /data)."
        )

    found = []
    for mount in (Path("/data"), Path("/mnt"), Path("/workspace")):
        if not mount.is_dir():
            continue
        try:
            for cand in mount.rglob("*"):
                if cand.is_dir() and cand.name.lower() == root.name.lower():
                    found.append(cand)
                if len(found) >= 5:
                    break
        except (PermissionError, OSError):
            continue
    if found:
        lines.append(f"  Found directories named {root.name!r}:")
        lines += [f"    --root {p}" for p in found[:5]]
    else:
        for mount in (Path("/data"), Path("/mnt")):
            if mount.is_dir():
                names = [p.name for p in list(mount.iterdir())[:10]]
                lines.append(f"  {mount} contains: {names}")
    return "\n".join(lines)


def resolve_root(root: str, split: str) -> Path:
    """Find the directory that actually holds the parquet shards.

    HF datasets land in several layouts depending on how they were fetched:

        <root>/data/train-00000-of-000NN.parquet     plain download
        <root>/snapshots/<sha>/data/*.parquet        hub cache layout
        <root>/*.parquet                             flattened by hand

    load_magicbrush only checks the first and third. Rather than make the user
    guess, search recursively for a directory containing shards whose filename
    carries the split, and say what was found if none does. A path error should
    cost one line of output, not a round trip.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(_missing_root_message(root))

    all_parquet = sorted(root.rglob("*.parquet"))
    if not all_parquet:
        raise FileNotFoundError(
            f"no .parquet anywhere under {root}\n"
            f"  contents: {[p.name for p in list(root.iterdir())[:10]]}\n"
            "  The dataset download may not have completed."
        )

    matching = [p for p in all_parquet if split in p.name]
    if not matching:
        raise FileNotFoundError(
            f"found {len(all_parquet)} parquet shards under {root}, but none "
            f"with '{split}' in the filename.\n"
            f"  shards: {[p.name for p in all_parquet[:6]]}\n"
            f"  Pass --split matching one of those."
        )

    # load_magicbrush globs <root>/data/*.parquet or <root>/*.parquet, so hand
    # it the shards' own parent, or its parent if that parent is named "data".
    holder = matching[0].parent
    resolved = holder.parent if holder.name == "data" else holder
    if resolved != root:
        print(f"[resolve] shards under {holder}\n"
              f"[resolve] using --root {resolved}")
    return resolved


#: MagicBrush parquet columns. The target column is the one Stage 0 never
#: needed: bench/data.py::load_magicbrush deliberately does not read it,
#: because an evaluation harness only requires source + mask + instruction.
#: Gate A trains a source -> target transport and needs the edited frame, so
#: this module reads the parquet directly rather than widening the Stage 0
#: loader, which is a validated artifact and should stay as it is.
TARGET_COLS = ("target_img", "edited_img", "output_img")


class Triplet:
    """source, target, mask, instruction -- what Gate A actually needs."""

    __slots__ = ("case_id", "image", "target_image", "mask", "instruction")

    def __init__(self, case_id, image, target_image, mask, instruction):
        self.case_id = case_id
        self.image = image
        self.target_image = target_image
        self.mask = mask
        self.instruction = instruction


def iter_magicbrush(root: str, split: str, size: int, limit: int | None,
                    turns=(1,)):
    """Yield Triplets straight from the parquet shards.

    Conventions copied verbatim from bench/data.py::load_magicbrush, each of
    which was established the hard way:

      - the mask is the ALPHA channel of mask_img, transparent = edit region
        (alpha < 128). The RGB channels carry picture content; converting to L
        composites the mask away and yields image brightness instead.
      - turn_index == 1 only. Later turns have a prior MODEL OUTPUT as their
        source, which would train the transport on generated images. On the
        dev split this drops 1186 of 2421 rows, which is expected, not a bug.
      - masks under 0.1% or over 99% carry no localisation.
    """
    import io

    import pandas as pd
    from PIL import Image

    root = Path(resolve_root(root, split))
    data_dir = root / "data" if (root / "data").is_dir() else root
    shards = sorted(p for p in data_dir.glob("*.parquet") if split in p.name)

    def _img(cell, mode):
        if cell is None:
            return None
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(raw))
        if mode == "mask":
            if im.mode != "RGBA":
                raise ValueError(
                    f"expected RGBA mask_img, got {im.mode} -- the alpha "
                    "channel carries the mask")
            return np.array(im.resize((size, size), Image.NEAREST))[..., 3] < 128
        return np.array(im.convert("RGB").resize((size, size), Image.BICUBIC))

    n = 0
    target_col = None
    kept = dropped = 0
    for shard in shards:
        df = pd.read_parquet(shard)
        if target_col is None:
            target_col = next((c for c in TARGET_COLS if c in df.columns), None)
            if target_col is None:
                raise KeyError(
                    f"no target-image column in {shard.name}. Looked for "
                    f"{TARGET_COLS}; the shard has {list(df.columns)}. Gate A "
                    "needs the edited frame -- without it there is no "
                    "transport to learn."
                )
            print(f"[triplets] target column: {target_col!r}")

        for _, row in df.iterrows():
            if turns is not None and int(row["turn_index"]) not in turns:
                dropped += 1
                continue
            src = _img(row["source_img"], "RGB")
            tgt = _img(row[target_col], "RGB")
            m = _img(row["mask_img"], "mask")
            if src is None or tgt is None or m is None:
                dropped += 1
                continue
            frac = float(m.mean())
            if frac < 0.001 or frac > 0.99:
                dropped += 1
                continue
            kept += 1
            yield Triplet(f"{row['img_id']}_t{row['turn_index']}", src, tgt, m,
                          str(row["instruction"]).strip())
            n += 1
            if limit and n >= limit:
                print(f"[triplets] {kept} kept, {dropped} dropped (limit hit)")
                return
    print(f"[triplets] {kept} kept, {dropped} dropped from {len(shards)} shard(s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="magicbrush",
                    choices=["magicbrush", "maskedit", "folder"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="/data/cache")
    ap.add_argument("--split", default="train")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = CacheConfig(size=args.size, latent_size=args.size // 8,
                      max_items=args.limit)
    writer = TripletCacheWriter(args.out, cfg,
                               source_tag=f"{args.dataset}:{args.split}")
    pipe = load_pipeline(args.model, args.device)

    n_ok = n_skip = 0
    consecutive = 0
    for case in iter_magicbrush(args.root, args.split, args.size, args.limit):
        try:
            z_src = encode_image(pipe, case.image, args.device)
            z_tgt = encode_image(pipe, case.target_image, args.device)
            h, w = z_src.shape[-2:]
            m = mask_to_latent_grid(case.mask, h, w)[0].numpy()
            pe, pooled = encode_text(pipe, case.instruction, args.device)
            writer.add(case.case_id, z_src, z_tgt, m, case.instruction,
                       prompt_embeds=pe, pooled=pooled,
                       thumb_src=case.image[::4, ::4],
                       thumb_tgt=case.target_image[::4, ::4],
                       meta={"dataset": args.dataset})
            n_ok += 1
            consecutive = 0
        except Exception as exc:                      # noqa: BLE001
            n_skip += 1
            consecutive += 1
            print(f"  skip {getattr(case, 'case_id', '?')}: "
                  f"{type(exc).__name__}: {exc}")
            # Fail fast. An earlier run ground through 1235 rows failing
            # identically and then wrote an EMPTY cache, which the trainer
            # only discovered later. If nothing has succeeded yet and the
            # same thing keeps failing, it is a bug, not bad data.
            if consecutive >= 20 and n_ok == 0:
                raise RuntimeError(
                    f"first {consecutive} items all failed with "
                    f"{type(exc).__name__}: {exc}\n"
                    "  This is a code or schema problem, not bad rows. "
                    "Nothing was written."
                ) from exc
        if (n_ok + n_skip) % 200 == 0:
            print(f"  {n_ok} written, {n_skip} skipped")

    if n_ok == 0:
        raise RuntimeError(
            f"0 items encoded, {n_skip} skipped -- refusing to write an empty "
            "cache. Read the skip reasons above."
        )
    root = writer.close()
    print(f"\ncache written: {root}\n  {n_ok} items, {n_skip} skipped")
    print("Sanity-check a few before training:")
    print(f"  python -c \"from transport.data import TripletCache as T; "
          f"c=T('{root}'); print(len(c), c.stratify().keys()); "
          f"print(c.items[0])\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
