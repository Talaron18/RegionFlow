"""
Benchmark loaders.

PIE-Bench ships two things this project can use directly and that are better
than the heuristics in `region.prompts`:

  * BRACKETS. The target prompt marks changed spans, e.g.
        "a slanted [rusty] mountain bicycle on the road in front of a building"
    That is a human annotation of exactly which tokens are edit tokens. Prefer
    it over the SequenceMatcher diff whenever it is present -- the diff is a
    fallback for datasets without it.

  * editing_type_id. Ten annotated edit categories, which replaces the keyword
    guess in `infer_edit_type` and feeds the band radius directly.

Schema varies between PIE-Bench v1 (cure-lab/PnPInversion) and PIE-Bench++
(UB-CVML-Group), which renamed the prompt keys and made blended_word a list.
Both are handled by alias. MAG-Bench's on-disk format is not documented in the
paper, so its loader is deliberately tolerant and paired with `inspect_schema`
-- run that first and adapt the aliases rather than trusting these defaults.

VERIFY ON FIRST LOAD. Every assumption here about key names, RLE convention and
type ids is from documentation, not from the files. `validate()` exists to
catch a mismatch loudly instead of producing a silently wrong mask.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from bench.sweep import BenchCase


# Folder names in PIE-Bench encode the type; counts sum to 700.
PIE_TYPE_NAMES = {
    "0": "random",
    "1": "change_object",
    "2": "add_object",
    "3": "delete_object",
    "4": "change_attribute_content",
    "5": "change_attribute_pose",
    "6": "change_attribute_color",
    "7": "change_attribute_material",
    "8": "change_background",
    "9": "change_style",
}

# Band radius in latent tokens per PIE edit type. See the band-semantics table
# in the README: seam only for colour, headroom for replacement, wide for
# anything with physical spillover.
PIE_BAND_TOKENS = {
    "change_attribute_color": 1.0,
    "change_attribute_material": 1.0,
    "change_attribute_content": 2.0,
    "change_object": 3.0,
    "add_object": 3.0,
    "delete_object": 3.0,
    "change_attribute_pose": 4.0,
    "change_background": 4.0,
    "change_style": 4.0,     # global edit; the hierarchy degenerates, see below
    "random": 2.0,
}

# Edits whose target is the whole image. The graded field has nothing to grade
# against, so these are excluded from the tiering sweep by default and reported
# separately.
GLOBAL_EDIT_TYPES = {"change_style", "change_background"}

_ALIASES = {
    "source_prompt": ("original_prompt", "source_prompt", "src_prompt", "prompt"),
    "target_prompt": ("editing_prompt", "target_prompt", "tgt_prompt", "edited_prompt"),
    "instruction": ("editing_instruction", "instruction", "edit_instruction"),
    "type_id": ("editing_type_id", "edit_type_id", "type_id"),
    "blended": ("blended_word", "blended_words", "blend_word"),
    "image_path": ("image_path", "img_path", "image", "file_name", "path"),
    "mask": ("mask", "mask_rle", "segmentation"),
}


def _pick(entry: dict, field: str):
    for key in _ALIASES[field]:
        if key in entry:
            return entry[key]
    return None


class PlaceholderMask(ValueError):
    """Raised when a mask string carries no spatial information.

    PIE-Bench++ stores '0 N' for entries with no annotation, where N is the
    pixel count. Decoding that yields a full-image mask, which silently turns
    a localised-editing case into a global one. It must be rejected, not
    decoded.
    """


def decode_rle(rle, shape) -> np.ndarray:
    """Decode a flat mask string, auto-detecting the encoding.

    Two formats appear in the PIE-Bench family:

    RUN-LENGTHS
        Alternating run lengths starting with a 0-value run, summing to H*W.
            "0 5 3 2"  ->  5 ones, 3 zeros, 2 ones

    (START, LENGTH) PAIRS
        Alternating absolute start index and run length. Identified by the
        total being far larger than H*W, because the starts are positions
        rather than counts. A 512x512 mask with ~900 segments sums to ~1.2e8
        against an image of 2.6e5 pixels.

    Raises PlaceholderMask on '0 N' entries, which encode "no annotation".
    """
    if isinstance(rle, str):
        vals = [int(x) for x in rle.split()]
    else:
        vals = [int(x) for x in rle]

    h, w = shape
    n = h * w

    # '0 N' covering the whole image is a placeholder, not a mask.
    if len(vals) == 2 and vals[0] == 0 and vals[1] >= n:
        raise PlaceholderMask(
            f"'0 {vals[1]}' is a no-annotation placeholder, not a mask"
        )

    total = int(np.sum(vals))
    flat = np.zeros(n, dtype=bool)

    if total == n:
        # Run-length form: the sum matching the pixel count exactly is
        # unambiguous.
        pos, val = 0, False
        for run in vals:
            if val and run:
                flat[pos : pos + run] = True
            pos += run
            val = not val
        return flat.reshape(h, w)

    # Otherwise try (start, length) pairs. Discriminate on STRUCTURE, not on
    # the size of the sum: a sparse pair-encoded mask with few segments can
    # total less than the pixel count, so a magnitude threshold misclassifies
    # it. Starts must be strictly increasing and in range; run lengths have no
    # such property.
    if len(vals) % 2 == 0:
        starts, lengths = vals[0::2], vals[1::2]
        monotonic = all(b > a for a, b in zip(starts, starts[1:]))
        in_range = starts[0] >= 0 and starts[-1] + lengths[-1] <= n
        positive = all(x > 0 for x in lengths)
        if monotonic and in_range and positive:
            for s, ln in zip(starts, lengths):
                flat[s : s + ln] = True
            return flat.reshape(h, w)

    raise ValueError(
        f"unrecognised mask encoding: {len(vals)} values summing to {total} "
        f"for {n} pixels (shape {shape}). Not run-lengths (sum != {n}) and not "
        f"start/length pairs (starts not strictly increasing or out of range)."
    )


def parse_brackets(target_prompt: str) -> tuple[str, list[str]]:
    """Split "a [rusty] bike" -> ("a rusty bike", ["rusty"]).

    The returned spans are the annotated edit tokens. Use them to build the
    edit-token mask instead of diffing, when available.
    """
    spans = re.findall(r"\[([^\]]*)\]", target_prompt)
    clean = re.sub(r"[\[\]]", "", target_prompt)
    return clean, [s.strip() for s in spans if s.strip()]


def parse_blended(blended) -> list[tuple[str, str]]:
    """Normalise blended_word forms to [(source_word, target_word), ...].

    v1:  "bicycle bicycle"        (space separated, one pair)
    ++:  ["cat,dog", "chair,chair"]  (list of comma-separated pairs)
    """
    if blended is None:
        return []
    items = [blended] if isinstance(blended, str) else list(blended)
    out = []
    for item in items:
        parts = [p.strip() for p in (item.split(",") if "," in item else item.split())]
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
        elif parts:
            out.append((parts[0], parts[0]))
    return out


def mask_bucket(mask: np.ndarray, small: float = 0.02, large: float = 0.15) -> str:
    """Area-based stratification.

    MAG-Bench ships its own three categories; PIE-Bench does not, so derive
    them. Thresholds are fractions of image area and are a starting guess --
    check the resulting distribution with `summarise` and adjust so the buckets
    are not wildly unbalanced.
    """
    frac = float((np.asarray(mask) > 0).mean())
    if frac < small:
        return "small"
    if frac < large:
        return "medium"
    return "large"


def inspect_schema(json_path, n: int = 3) -> str:
    """Dump the actual structure of an annotation file.

    Run this FIRST on any benchmark. Every loader below encodes assumptions
    that were read from documentation rather than from the files.
    """
    data = json.loads(Path(json_path).read_text())
    if isinstance(data, list):
        entries = data[:n]
        keys = Counter(k for e in data if isinstance(e, dict) for k in e)
    else:
        entries = [data[k] for k in list(data)[:n]]
        keys = Counter(k for e in data.values() if isinstance(e, dict) for k in e)

    lines = [f"{len(data)} entries", "", "key frequency:"]
    lines += [f"  {k:<28} {c}" for k, c in keys.most_common()]
    lines += ["", "first entries:"]
    for e in entries:
        for k, v in e.items():
            s = repr(v)
            lines.append(f"  {k:<28} {s[:110]}{'...' if len(s) > 110 else ''}")
        lines.append("")
    return "\n".join(lines)


def _type_from_entry(entry: dict) -> str:
    """Resolve the edit type, by explicit id or from the image path.

    PIE-Bench v1 has a top-level `editing_type_id`. PIE-Bench++ does not — it
    carries per-word `edit_action` entries instead — but both encode the type
    in the image path's folder prefix:

        "0_random_140/000000000000.jpg"      -> random
        "6_change_attribute_color_40/..."    -> change_attribute_color

    So the path is the reliable source across both variants.
    """
    tid = _pick(entry, "type_id")
    if tid is not None and str(tid) in PIE_TYPE_NAMES:
        return PIE_TYPE_NAMES[str(tid)]

    path = str(_pick(entry, "image_path") or "")
    prefix = path.split("/")[0].split("_")[0]
    if prefix in PIE_TYPE_NAMES:
        return PIE_TYPE_NAMES[prefix]
    return "unknown"


def load_pie_bench(
    root,
    json_name: str = "mapping_file.json",
    image_size: tuple = (512, 512),
    load_image=None,
    exclude_global: bool = True,
    limit: int | None = None,
) -> list[BenchCase]:
    """Load PIE-Bench into BenchCase objects.

    load_image(path) -> HxWx3 array. Injected so PIL is not a hard dependency
    and so a test can pass a stub.

    exclude_global drops style and background edits, whose target is the whole
    image. Keeping them in the r-sweep dilutes the tiering result with cases
    where tiering is meaningless by construction.
    """
    root = Path(root)
    data = json.loads((root / json_name).read_text())
    items = data.items() if isinstance(data, dict) else enumerate(data)

    cases, skipped = [], Counter()
    for key, entry in items:
        if limit and len(cases) >= limit:
            break

        type_name = _type_from_entry(entry)
        if exclude_global and type_name in GLOBAL_EDIT_TYPES:
            skipped[type_name] += 1
            continue

        raw_target = _pick(entry, "target_prompt") or ""
        target_prompt, spans = parse_brackets(raw_target)
        source_prompt = re.sub(r"[\[\]]", "", _pick(entry, "source_prompt") or "")

        rel = _pick(entry, "image_path")
        image = load_image(root / "annotation_images" / rel) if load_image else None
        shape = image.shape[:2] if image is not None else image_size

        try:
            mask = decode_rle(_pick(entry, "mask"), shape)
        except (ValueError, TypeError) as exc:
            skipped[f"mask_decode: {exc.__class__.__name__}"] += 1
            continue

        pairs = parse_blended(_pick(entry, "blended"))
        cases.append(
            BenchCase(
                case_id=str(key),
                image=image,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                target_mask=mask,
                mask_bucket=mask_bucket(mask),
                source_attribute=pairs[0][0] if pairs else None,
                target_attribute=pairs[0][1] if pairs else None,
            )
        )
        cases[-1].edit_type = type_name          # type: ignore[attr-defined]
        cases[-1].edit_spans = spans             # type: ignore[attr-defined]
        cases[-1].band_tokens = PIE_BAND_TOKENS.get(type_name, 2.0)  # type: ignore

    if skipped:
        print(f"[load_pie_bench] skipped: {dict(skipped)}")
    return cases


def load_mag_bench(
    root,
    json_name: str = "annotations.json",
    mask_dir: str = "masks",
    load_image=None,
    load_mask=None,
    limit: int | None = None,
) -> list[BenchCase]:
    """Load MAG-Bench. Tolerant by design.

    The paper describes 200 images with source prompt, target prompt, mask and
    three mask-size categories, but does not document the file layout. Run
    `inspect_schema` on the annotation file and adjust `_ALIASES` before
    trusting this. Masks are assumed to be PNGs named by case id; pass
    `load_mask` to override.
    """
    root = Path(root)
    data = json.loads((root / json_name).read_text())
    items = data.items() if isinstance(data, dict) else enumerate(data)

    cases = []
    for key, entry in items:
        if limit and len(cases) >= limit:
            break
        rel = _pick(entry, "image_path") or f"{key}.jpg"
        image = load_image(root / rel) if load_image else None

        if load_mask:
            mask = load_mask(root / mask_dir / f"{key}.png")
        elif _pick(entry, "mask") is not None:
            shape = image.shape[:2] if image is not None else (512, 512)
            mask = decode_rle(_pick(entry, "mask"), shape)
        else:
            raise ValueError(f"no mask for case {key}; pass load_mask=")

        bucket = entry.get("mask_size") or entry.get("category") or mask_bucket(mask)
        cases.append(
            BenchCase(
                case_id=str(key),
                image=image,
                source_prompt=_pick(entry, "source_prompt") or "",
                target_prompt=_pick(entry, "target_prompt") or "",
                target_mask=mask,
                mask_bucket=str(bucket),
            )
        )
    return cases


def attach_instance_annotations(cases, sidecar_path, load_mask=None) -> list:
    """Merge multi-instance annotations from a sidecar file.

    Neither benchmark ships masks for the non-selected same-category objects,
    so the instance-leakage metric cannot run without this. Sidecar format:

        {"000000000012": {"n_instances": 3,
                          "other_masks": ["ann/12_a.png", "ann/12_b.png"]}}

    or with inline RLE under "other_masks_rle". Cases absent from the sidecar
    keep other_instance_masks=None and are excluded from leakage reporting
    rather than counted as zero-leak, which would bias the rate downward.
    """
    side = json.loads(Path(sidecar_path).read_text())
    base = Path(sidecar_path).parent
    hit = 0
    for case in cases:
        rec = side.get(case.case_id)
        if not rec:
            continue
        if "other_masks_rle" in rec:
            shape = case.target_mask.shape
            masks = [decode_rle(r, shape) for r in rec["other_masks_rle"]]
        elif load_mask:
            masks = [load_mask(base / p) for p in rec.get("other_masks", [])]
        else:
            continue
        case.other_instance_masks = masks
        hit += 1
    print(f"[instances] annotated {hit}/{len(cases)} cases")
    return cases


def validate(cases) -> str:
    """Loud sanity report. Read it before running any sweep."""
    from region.lambda_field import target_token_area

    n = len(cases)
    if n == 0:
        return "no cases loaded"

    buckets = Counter(c.mask_bucket for c in cases)
    types = Counter(getattr(c, "edit_type", "unknown") for c in cases)
    areas = [float((c.target_mask > 0).mean()) for c in cases]
    toks = [target_token_area(c.target_mask) for c in cases]
    tiny = sum(t < 4.0 for t in toks)
    empty = sum(not c.target_mask.any() for c in cases)
    full = sum(c.target_mask.all() for c in cases)
    with_inst = sum(bool(c.other_instance_masks) for c in cases)
    with_spans = sum(bool(getattr(c, "edit_spans", None)) for c in cases)

    lines = [
        f"cases                 {n}",
        f"mask area  min/med/max {min(areas):.4f} / {np.median(areas):.4f} / {max(areas):.4f}",
        f"token area min/med/max {min(toks):.1f} / {np.median(toks):.1f} / {max(toks):.1f}",
        f"buckets               {dict(buckets)}",
        f"edit types            {dict(types)}",
        f"bracket annotations   {with_spans}/{n}",
        f"instance annotations  {with_inst}/{n}   <- leakage metric needs these",
        f"targets under 4 tok   {tiny}   <- no room for a graded field, crop path",
    ]
    if empty:
        lines.append(f"!! EMPTY MASKS         {empty}  -- RLE convention is probably wrong")
    if full:
        lines.append(f"!! FULL-IMAGE MASKS    {full}  -- global edits, exclude from sweep")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PIE-Bench++ (HuggingFace parquet mirror)
# ---------------------------------------------------------------------------
# The original PIE-Bench is behind a Google Form, so the HF mirror
# UB-CVML-Group/PIE_Bench_pp is often the practical route. Same 700 images,
# revised prompts with bracket annotations, per-word `edit_action` metadata,
# CC-BY-SA-4.0. Layout is parquet with embedded image bytes rather than JSON
# plus image folders, so it needs its own reader.


def load_pie_bench_pp(
    root,
    exclude_global: bool = True,
    limit: int | None = None,
) -> list[BenchCase]:
    """Load the parquet mirror. Requires `datasets` or pandas+pyarrow.

    Column names are not assumed: `describe_parquet` below prints what is
    actually there, and this reader resolves through the same alias table as
    the JSON loader.
    """
    import io

    import pandas as pd
    from PIL import Image

    root = Path(root)
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cases, skipped = [], Counter()

    for i, row in df.iterrows():
        if limit and len(cases) >= limit:
            break
        entry = row.to_dict()

        type_name = _type_from_entry(entry)
        if exclude_global and type_name in GLOBAL_EDIT_TYPES:
            skipped[type_name] += 1
            continue

        # Image may be a dict with 'bytes', raw bytes, or a PIL object.
        img = entry.get("image")
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        if img is None:
            skipped["no_image"] += 1
            continue
        image = np.array(img.convert("RGB"))

        raw_target = _pick(entry, "target_prompt") or ""
        target_prompt, spans = parse_brackets(str(raw_target))
        source_prompt = re.sub(r"[\[\]]", "", str(_pick(entry, "source_prompt") or ""))

        try:
            mask = decode_rle(_pick(entry, "mask"), image.shape[:2])
        except (ValueError, TypeError) as exc:
            skipped[f"mask_decode: {exc.__class__.__name__}"] += 1
            continue

        pairs = parse_blended(_pick(entry, "blended"))
        case_id = str(entry.get("image_path") or entry.get("id") or i)
        cases.append(
            BenchCase(
                case_id=case_id,
                image=image,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                target_mask=mask,
                mask_bucket=mask_bucket(mask),
                source_attribute=pairs[0][0] if pairs else None,
                target_attribute=pairs[0][1] if pairs else None,
            )
        )
        cases[-1].edit_type = type_name                              # type: ignore
        cases[-1].edit_spans = spans                                 # type: ignore
        cases[-1].band_tokens = PIE_BAND_TOKENS.get(type_name, 2.0)  # type: ignore

    if skipped:
        print(f"[load_pie_bench_pp] skipped: {dict(skipped)}")
    return cases


def describe_parquet(root, n: int = 2) -> str:
    """Print the real columns and a sample row. Run before trusting the reader."""
    import pandas as pd

    files = sorted(Path(root).rglob("*.parquet"))
    if not files:
        return f"no parquet under {root}"
    df = pd.read_parquet(files[0])
    lines = [f"{len(files)} parquet file(s), first has {len(df)} rows", "", "columns:"]
    lines += [f"  {c:<24} {str(df[c].dtype):<10}" for c in df.columns]
    lines += ["", "sample:"]
    for _, row in df.head(n).iterrows():
        for c in df.columns:
            v = repr(row[c])
            lines.append(f"  {c:<24} {v[:100]}{'...' if len(v) > 100 else ''}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- MagicBrush --
#
# MagicBrush (osunlp/MagicBrush) is the only dataset surveyed that ships a real
# per-edit mask on EVERY row, which is what makes the leakage metric runnable.
# It differs from the PIE-Bench family in three ways that matter:
#
#   1. INSTRUCTION, NOT CAPTION PAIR. Rows carry `instruction` ("make the sky
#      purple"), not source/target prompts. The lambda field derives edit tokens
#      by diffing the two captions, and there is nothing here to diff. See
#      `MAGICBRUSH_PROMPT_GAP` below -- this is a design decision, not a bug to
#      paper over in the loader.
#
#   2. MIXED RESOLUTIONS WITHIN A ROW. Observed on the dev split: source_img is
#      500x500 RGB while mask_img and target_img are 1024x1024, and mask_img is
#      RGBA. Anything that assumes one shape per row produces a silently
#      misaligned field. Everything is resampled to `size` here.
#
#   3. MULTI-TURN. `turn_index` counts edits in a chain; turn 2 is applied to
#      the OUTPUT of turn 1, not to source_img of turn 1. Mixing turns into one
#      eval set compares edits made from different starting images. Default is
#      turn_index == 1 only.
#
# MASK POLARITY IS UNVERIFIED. The convention (white = edited region, or the
# inverse) is not documented in the dataset card. `verify_magicbrush` below
# checks it empirically against the source/target difference. Run it before
# trusting a single case.

MAGICBRUSH_PROMPT_GAP = """\
MagicBrush has no source caption. Three options, in order of effort:

  (a) source_prompt="" -- generate() then builds no token mask and the lambda
      field falls back to spatial-only. Loses the token-level edit targeting
      that is half of the method. Fine for alpha-pin-only experiments.
  (b) Caption source_img with BLIP-2 and pair it with the instruction. Adds a
      caption-quality confound on top of the one already measured (prompt
      mismatch was worth 3.65 dB on can.jpg).
  (c) Use MagicBrush for the mask/leakage metrics only, and keep PIE-Bench for
      anything needing a caption pair.

(c) is the honest split and is what the loader assumes by default.
"""


def load_magicbrush(
    root,
    split: str = "dev",
    turns: tuple = (1,),
    size: int = 512,
    limit: int | None = None,
    source_prompt_fn=None,
) -> list:
    """Load MagicBrush parquet shards into BenchCase objects.

    Args:
      root: directory containing data/*.parquet
      split: "dev" or "train"; matched against the shard filename
      turns: which turn_index values to keep. (1,) keeps only first edits,
             whose source_img is a real photograph rather than a prior model
             output. Pass None to keep all turns and accept the confound.
      size: everything is resampled to size x size, matching the model frame
      source_prompt_fn: optional callable(image_ndarray) -> str used to fill
             source_prompt. Left None, source_prompt is "" -- see
             MAGICBRUSH_PROMPT_GAP.

    The instruction goes into target_prompt so the case is usable as-is; it is
    NOT a caption, and build_token_masks against an empty source will produce
    an empty edit mask. That is intended and visible rather than silent.
    """
    import io

    import pandas as pd
    from PIL import Image

    root = Path(root)
    data_dir = root / "data" if (root / "data").is_dir() else root
    shards = sorted(p for p in data_dir.glob("*.parquet") if split in p.name)
    if not shards:
        raise FileNotFoundError(
            f"no '{split}' parquet shards under {data_dir} -- "
            f"found: {[p.name for p in data_dir.glob('*.parquet')][:5]}"
        )

    def _img(cell, mode, resample):
        """Parquet image cell -> size x size ndarray.

        mask_img is the SOURCE PHOTO with the edit region punched out as
        transparency (DALL-E 2 inpainting convention). The RGB channels carry
        picture content, not the mask -- converting to L composites the mask
        away and yields image brightness. The mask is alpha, and transparent
        means edit, so it is inverted relative to the usual convention.
        """
        if cell is None:
            return None
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(raw))
        if mode == "mask":
            if im.mode != "RGBA":
                raise ValueError(
                    f"expected RGBA mask_img, got {im.mode} -- the alpha "
                    "channel carries the mask; verify before proceeding"
                )
            alpha = np.array(im.resize((size, size), Image.NEAREST))[..., 3]
            return alpha < 128          # True = transparent = region to edit
        im = im.convert("RGB")
        return np.array(im.resize((size, size), resample))

    cases, skipped = [], Counter()
    for shard in shards:
        df = pd.read_parquet(shard)
        for _, row in df.iterrows():
            if turns is not None and int(row["turn_index"]) not in turns:
                skipped["wrong_turn"] += 1
                continue

            src = _img(row["source_img"], "RGB", Image.BICUBIC)
            m = _img(row["mask_img"], "mask", Image.NEAREST)
            if src is None or m is None:
                skipped["no_image"] += 1
                continue

            frac = float(m.mean())
            # A mask covering everything or nothing carries no localisation.
            if frac < 0.001 or frac > 0.99:
                skipped[f"degenerate_mask({frac:.3f})"] += 1
                continue

            instr = str(row["instruction"]).strip()
            cases.append(
                BenchCase(
                    case_id=f"{row['img_id']}_t{row['turn_index']}",
                    image=src,
                    source_prompt=(
                        source_prompt_fn(src) if source_prompt_fn else ""
                    ),
                    target_prompt=instr,
                    target_mask=m,
                    box=_mask_to_box(m),
                    mask_bucket=mask_bucket(m),
                )
            )
            if limit and len(cases) >= limit:
                print(f"[load_magicbrush] {len(cases)} cases, skipped {dict(skipped)}")
                return cases

    print(f"[load_magicbrush] {len(cases)} cases from {len(shards)} shard(s), "
          f"skipped {dict(skipped)}")
    return cases


def _mask_to_box(m: np.ndarray) -> tuple | None:
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def verify_magicbrush(root, split: str = "dev", n: int = 40, size: int = 512) -> str:
    """Check mask polarity and alignment empirically. Run before using cases.

    The test: where source and target images differ, the mask should be True.
    If agreement is near 0 the polarity is inverted; if near chance the mask is
    misaligned with the images and nothing downstream is trustworthy.
    """
    import io

    import pandas as pd
    from PIL import Image

    root = Path(root)
    data_dir = root / "data" if (root / "data").is_dir() else root
    shards = sorted(p for p in data_dir.glob("*.parquet") if split in p.name)
    if not shards:
        return f"no '{split}' shards under {data_dir}"

    def _arr(cell, mode):
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(raw))
        if mode == "mask":
            a = np.array(im.convert("RGBA").resize((size, size), Image.NEAREST))
            return a[..., 3] < 128       # transparent = region to edit
        return np.array(im.convert("RGB").resize((size, size), Image.NEAREST)
                        ).astype(np.float32)

    recall, precision, shapes, cover, lift = [], [], Counter(), [], []
    df = pd.read_parquet(shards[0])
    for _, row in df.head(n).iterrows():
        if int(row["turn_index"]) != 1:
            continue
        src, tgt = _arr(row["source_img"], "RGB"), _arr(row["target_img"], "RGB")
        m = _arr(row["mask_img"], "mask")
        shapes[f"src{Image.open(io.BytesIO(row['source_img']['bytes'])).size}"
               f"/msk{Image.open(io.BytesIO(row['mask_img']['bytes'])).size}"] += 1
        changed = np.abs(src - tgt).mean(2) > 20
        if changed.sum() < 50 or m.sum() < 50:
            continue
        c = float(m.mean())
        r = float((changed & m).sum() / changed.sum())
        cover.append(c)
        recall.append(r)
        precision.append(float((changed & m).sum() / m.sum()))
        # A mask covering fraction c catches c of the changed pixels by chance.
        # Recall alone cannot tell a good mask from a big one; lift can.
        lift.append(r / max(c, 1e-6))

    if not recall:
        return "no usable rows -- all masks or diffs were empty"

    r, p = float(np.mean(recall)), float(np.mean(precision))
    cv, lf = float(np.mean(cover)), float(np.mean(lift))
    out = [
        f"MagicBrush {split}: {len(recall)} rows checked (mask = alpha < 128)",
        f"  native sizes seen: {dict(shapes)}",
        f"  mean mask coverage: {cv:.3f} of image  <- chance recall",
        f"  recall {r:.3f}   precision {p:.3f}",
        f"  LIFT OVER CHANCE: {lf:.2f}x",
    ]
    if lf > 2.5 and r > 0.6:
        out.append("  -> mask is real and correctly oriented. Usable.")
    elif lf < 1.5:
        out.append("  -> NO BETTER THAN CHANCE. The mask carries little or no")
        out.append("     information about where the edit happened. Do not use.")
    else:
        out.append("  -> WEAK. Better than chance but not by much. Inspect the")
        out.append("     overlay strips before relying on these masks.")
    out.append("  (low precision alone is expected -- masks are drawn loosely "
               "around the region, so they exceed the changed pixels.)")
    return "\n".join(out)


def dump_magicbrush_overlays(root, out_dir, split: str = "dev", n: int = 6,
                             size: int = 512) -> str:
    """Write source | target | mask | changed-pixels strips for eyeballing.

    Numbers cannot distinguish an inverted mask from a shifted one once both
    score badly. Looking can.
    """
    import io

    import pandas as pd
    from PIL import Image

    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data" if (root / "data").is_dir() else root
    shards = sorted(p for p in data_dir.glob("*.parquet") if split in p.name)
    df = pd.read_parquet(shards[0])

    def _im(cell, mode):
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(raw))
        if mode == "mask":
            a = np.array(im.convert("RGBA").resize((size, size), Image.NEAREST))
            return Image.fromarray(((a[..., 3] < 128).astype(np.uint8) * 255))
        return im.convert("RGB").resize((size, size), Image.NEAREST)

    written = 0
    for _, row in df.head(n * 4).iterrows():
        if int(row["turn_index"]) != 1:
            continue
        src, tgt = _im(row["source_img"], "RGB"), _im(row["target_img"], "RGB")
        msk = _im(row["mask_img"], "mask")
        diff = np.abs(np.asarray(src, np.float32)
                      - np.asarray(tgt, np.float32)).mean(2)
        dimg = Image.fromarray((diff > 20).astype(np.uint8) * 255).convert("RGB")
        strip = Image.new("RGB", (size * 4, size))
        for i, im in enumerate([src, tgt, msk.convert("RGB"), dimg]):
            strip.paste(im, (i * size, 0))
        strip.save(out_dir / f"mb_{row['img_id']}.png")
        written += 1
        if written >= n:
            break
    return (f"wrote {written} strips to {out_dir}\n"
            f"  panels: source | target | mask | changed pixels\n"
            f"  the mask panel should light up where the changed panel does")
