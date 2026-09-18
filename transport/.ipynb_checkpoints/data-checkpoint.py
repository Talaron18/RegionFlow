"""
Editing triplets (source, mask, target, instruction) with a precomputed cache.

WHY THE CACHE IS NOT OPTIONAL. On the 3090Ti, SD3's transformer, the VAE, CLIP
and T5 do not comfortably co-reside -- Stage 0 already had to segment first,
free SAM, then load the diffusion model. Training additionally needs optimiser
state and activations. The fix is the same shape as the Stage 0 fix: run the
encoders once, write latents and text embeddings to disk, then train with only
the transformer resident. T5-XXL alone is ~9.5GB in bf16; not loading it during
training is most of the headroom.

It also removes a whole class of confound. Encoders run once, deterministically,
in their own process; a training run cannot silently differ from an eval run
because of a dtype or a resize. The cache is content-addressed by a config hash
so a changed preprocessing setting produces a new cache rather than a
half-stale one -- Stage 0 lost three days to stale files read as findings.

LAYOUT

    cache/<hash>/
        meta.json           config + per-item index
        0000.npz            z_src, z_tgt, mask (latent grid), uint8 thumbs
        0000.emb.npz        prompt_embeds, pooled           (shared by ref)

Text embeddings are deduplicated by instruction string: editing datasets repeat
instructions heavily, and T5 embeddings at context 333 are the largest thing
per item.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class CacheConfig:
    size: int = 512
    latent_size: int = 64          # size // 8
    dtype: str = "float32"
    max_items: int | None = None
    #: SD3's real context length. Stage 0 measured 333, not the 410 assumed
    #: from the config. Asserted at write time so a mismatch fails here rather
    #: than as a shape error 40 minutes into training.
    context_length: int = 333


def config_hash(cfg: CacheConfig, source_tag: str) -> str:
    payload = json.dumps({**asdict(cfg), "source": source_tag}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class TripletCacheWriter:
    """Run once, on a machine where the encoders fit. GPU only for this phase."""

    def __init__(self, root: str | Path, cfg: CacheConfig, source_tag: str):
        self.cfg = cfg
        self.root = Path(root) / config_hash(cfg, source_tag)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index: list[dict] = []
        self.emb_ids: dict[str, str] = {}

    def add(self, item_id, z_src, z_tgt, mask_latent, instruction,
            prompt_embeds=None, pooled=None, thumb_src=None, thumb_tgt=None,
            meta: dict | None = None):
        z_src = np.asarray(z_src, dtype=np.float32)
        z_tgt = np.asarray(z_tgt, dtype=np.float32)
        mask_latent = np.asarray(mask_latent, dtype=np.float32)

        if z_src.shape != z_tgt.shape:
            raise ValueError(f"{item_id}: latent shape mismatch "
                             f"{z_src.shape} vs {z_tgt.shape}")
        if mask_latent.shape[-2:] != z_src.shape[-2:]:
            raise ValueError(
                f"{item_id}: mask grid {mask_latent.shape[-2:]} != latent grid "
                f"{z_src.shape[-2:]}. Pool the mask with "
                "conditioning.mask_to_latent_grid, do not resize with PIL."
            )

        # Reject degenerate triplets loudly. A pair whose latents are identical
        # contributes v = 0 everywhere and quietly teaches the model to do
        # nothing; enough of them and a null Gate A result means nothing.
        moved = float(np.abs(z_tgt - z_src).mean())
        if moved < 1e-6:
            raise ValueError(f"{item_id}: source and target latents identical")

        emb_key = None
        if prompt_embeds is not None:
            if prompt_embeds.shape[-2] != self.cfg.context_length:
                raise ValueError(
                    f"{item_id}: context {prompt_embeds.shape[-2]} != "
                    f"{self.cfg.context_length}"
                )
            emb_key = hashlib.sha1(instruction.encode()).hexdigest()[:16]
            if emb_key not in self.emb_ids:
                np.savez_compressed(
                    self.root / f"emb_{emb_key}.npz",
                    prompt_embeds=np.asarray(prompt_embeds, dtype=np.float32),
                    pooled=np.asarray(pooled, dtype=np.float32),
                )
                self.emb_ids[emb_key] = instruction

        stem = f"{len(self.index):05d}"
        payload = dict(z_src=z_src, z_tgt=z_tgt, mask=mask_latent)
        if thumb_src is not None:
            payload["thumb_src"] = np.asarray(thumb_src, dtype=np.uint8)
        if thumb_tgt is not None:
            payload["thumb_tgt"] = np.asarray(thumb_tgt, dtype=np.uint8)
        np.savez_compressed(self.root / f"{stem}.npz", **payload)

        self.index.append({
            "id": str(item_id), "stem": stem, "instruction": instruction,
            "emb": emb_key, "mask_frac": float(mask_latent.mean()),
            "moved": moved, **(meta or {}),
        })

    def close(self) -> Path:
        (self.root / "meta.json").write_text(json.dumps({
            "config": asdict(self.cfg),
            "n": len(self.index),
            "items": self.index,
        }, indent=1))
        return self.root


class TripletCache:
    """Read side. No torch import at module level so the CPU tools stay light."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text())
        self.cfg = CacheConfig(**meta["config"])
        self.items = meta["items"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        rec = self.items[i]
        d = np.load(self.root / f"{rec['stem']}.npz")
        out = {k: d[k] for k in d.files}
        if rec.get("emb"):
            e = np.load(self.root / f"emb_{rec['emb']}.npz")
            out["prompt_embeds"] = e["prompt_embeds"]
            out["pooled"] = e["pooled"]
        out["meta"] = rec
        return out

    def stratify(self, small: float = 0.02, large: float = 0.15) -> dict:
        """Mask-area buckets, matching bench/data.py::mask_bucket.

        Stage 0 swept mask scales from 0.93% to 15.1% and the coupling held
        across all of them. Keeping the same stratification means the Gate A
        comparison is against the Stage 0 numbers rather than against a
        differently-composed set.
        """
        buckets: dict = {"small": [], "medium": [], "large": []}
        for i, rec in enumerate(self.items):
            f = rec["mask_frac"]
            key = "small" if f < small else ("large" if f > large else "medium")
            buckets[key].append(i)
        return buckets


def torch_dataset(cache: TripletCache):
    """Wrap the cache as a torch Dataset. Imported lazily."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(cache)

        def __getitem__(self, i):
            d = cache[i]
            out = {
                "z_src": torch.from_numpy(d["z_src"]),
                "z_tgt": torch.from_numpy(d["z_tgt"]),
                "mask": torch.from_numpy(d["mask"]),
            }
            if "prompt_embeds" in d:
                out["prompt_embeds"] = torch.from_numpy(d["prompt_embeds"])
                out["pooled"] = torch.from_numpy(d["pooled"])
            return out

    return _DS()
