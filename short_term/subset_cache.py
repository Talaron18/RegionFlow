#!/usr/bin/env python3
"""
Extract a small SUBSET of a cache, for moving to the machine that has the GPU.

WHY THIS EXISTS. The obvious plan -- export PNGs on the cache machine, move the
PNGs -- does not work when that machine has no GPU: `cache_item.export` DECODES
the cached latents with an AutoencoderKL, and there is no GPU there. So the
thing to move is upstream of the decode: the latent triplets themselves.

THE PAYLOAD IS SMALLER THAN THE PNGs, AND MUCH SMALLER THAN THE CACHE.
Per item the cache stores:

    z_src   16 x 64 x 64 float32     262 KB
    z_tgt   16 x 64 x 64 float32     262 KB
    mask     1 x 64 x 64 float32      16 KB
    thumbs   2 x 128 x 128 x 3 uint8  96 KB
                                     ------
                                     639 KB uncompressed, ~300-450 KB compressed

and `emb_<key>.npz` holds the T5 embeddings -- context 333 x 4096 float32, about
5.5 MB PER DISTINCT INSTRUCTION, which is why the cache deduplicates them by
instruction string. Twenty items with twenty distinct instructions would be
~110 MB if the embeddings came along.

THEY DO NOT NEED TO. `cache_item.export` loads only `{stem}.npz` and reads the
instruction STRING out of the index; `edit.py` then re-encodes that string with
the pipeline's own T5/CLIP on the GPU machine. So this tool copies the item
npz files and NOT the embeddings, which makes 20 items about 8 MB.

SCOPE, STATED PLAINLY. A subset written here supports `cache_item.py` and
`suite.py`. It does NOT support `train.py`, which reads the embeddings -- the
`emb` key is kept in the records as provenance but the file is absent, and
`TripletCache.__getitem__` would raise on it. There is a note saying so inside
the subset's own meta.json.

    # on the cache machine (no GPU, and this only needs numpy)
    python short_term/subset_cache.py --cache /data/cache/<hash> --list 20
    python short_term/subset_cache.py --cache /data/cache/<hash> --sample 20 \\
        --out /tmp/sub20
    rsync -a /tmp/sub20/ gpu:/root/RegionFlow/runs/st/sub20/

    # on the GPU machine
    python short_term/subset_cache.py --out runs/st/sub20 --verify
    python short_term/cache_item.py --cache runs/st/sub20 --index 0 \\
        --out runs/st/one          # now the VAE decode can happen
    python short_term/suite.py --cache runs/st/sub20 --all \\
        --out runs/st/sheet20 --arms base,both,protect,selfy

    python short_term/subset_cache.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

#: Kept in step with cache_item.MASK_MIN / MASK_MAX by hand, because this file
#: must import NOTHING from short_term: backends.py pulls in torch, and the
#: machine this runs on may have only numpy. `self_test` asserts the two agree
#: whenever cache_item IS importable, so drift is caught on any dev box.
MASK_MIN, MASK_MAX = 0.02, 0.30


def load_index(cache: str | Path):
    root = Path(cache)
    mp = root / "meta.json"
    if not mp.is_file():
        raise SystemExit(
            f"{mp} not found. --cache takes the HASH directory itself "
            "(e.g. /data/cache/13061d51aee4), not its parent.")
    meta = json.loads(mp.read_text())
    return meta, meta["items"]


def in_band(items: list[dict]) -> list[dict]:
    """The usable population: big enough to edit, small enough that 'did the
    rest survive' is a real question."""
    return [r for r in items
            if MASK_MIN <= float(r.get("mask_frac", 0.0)) <= MASK_MAX]


def rank(items: list[dict]) -> list[dict]:
    return sorted(in_band(items), key=lambda r: -float(r.get("moved", 0.0)))


def pick(items: list[dict], indices, top, sample, seed):
    if indices:
        out = []
        for tok in indices.replace(",", " ").split():
            i = int(tok)
            if not 0 <= i < len(items):
                raise SystemExit(f"--indices {i} out of range "
                                 f"(cache has {len(items)} items)")
            out.append(i)
        return [items[i] for i in out], f"explicit:{indices}"
    cand = rank(items)
    if not cand:
        raise SystemExit(
            f"no item has mask_frac in [{MASK_MIN}, {MASK_MAX}]; widen the "
            "band here and in cache_item.py (they must stay equal) or pass "
            "--indices")
    if top:
        return cand[:top], f"top{top}_by_moved"
    if not sample:
        raise SystemExit("give --indices, --top or --sample (or --list)")
    n = min(int(sample), len(cand))
    rng = np.random.default_rng(seed)
    sel = sorted(rng.choice(len(cand), size=n, replace=False))
    return [cand[i] for i in sel], f"sample{n}_of{len(cand)}_inband_seed{seed}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_subset(cache: str | Path, recs: list[dict], rule: str,
                out: str | Path, copy: bool = True) -> dict:
    """Write a valid cache directory holding just `recs`.

    `stem` is PRESERVED, not renumbered: `cache_item.export` finds the npz by
    `rec["stem"]`, so renumbering would silently point every index at the wrong
    file. Indexing into the subset is by POSITION in the filtered item list,
    which is what `--index N` means; the filenames stay as they were.
    """
    root, out = Path(cache), Path(out)
    meta, _items = load_index(root)
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    total = 0
    for rec in recs:
        src = root / f"{rec['stem']}.npz"
        if not src.is_file():
            raise SystemExit(f"{src} is missing; the source cache is "
                             "incomplete and a subset of it would be too")
        dst = out / src.name
        if copy:
            shutil.copy2(src, dst)
        files[dst.name] = {"sha256": _sha256(dst), "bytes": dst.stat().st_size}
        total += dst.stat().st_size

    # Same `config`, so `config["size"]` -- which export uses to derive the
    # latent->pixel upsample factor -- still describes these items.
    sub_meta = {
        "config": meta["config"],
        "n": len(recs),
        "items": recs,
        "subset": {
            "of": str(root), "selection": rule, "created": time.strftime(
                "%Y-%m-%dT%H:%M:%S"),
            "n_source_items": len(meta["items"]),
            "embeddings_included": False,
            "note": ("emb_*.npz are intentionally NOT included: cache_item.py "
                     "and suite.py do not need them (the instruction STRING is "
                     "in the index and edit.py re-encodes it). TRAINING DOES "
                     "NEED THEM -- TripletCache.__getitem__ loads emb_<key> "
                     "and will raise on this subset. Do not train from it."),
        },
    }
    (out / "meta.json").write_text(json.dumps(sub_meta, indent=1))
    files["meta.json"] = {"sha256": _sha256(out / "meta.json"),
                          "bytes": (out / "meta.json").stat().st_size}
    total += files["meta.json"]["bytes"]

    man = {
        "created": sub_meta["subset"]["created"], "selection": rule,
        "source_cache": str(root), "n_items": len(recs),
        "n_files": len(files), "total_bytes": total, "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def verify(out: str | Path) -> int:
    out = Path(out)
    mp = out / "manifest.json"
    if not mp.is_file():
        print(f"no manifest at {mp}; nothing to verify")
        return 2
    man = json.loads(mp.read_text())
    missing, changed, ok = [], [], 0
    for rel, rec in man["files"].items():
        p = out / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size != rec["bytes"]:
            changed.append(f"{rel} (size {p.stat().st_size} != {rec['bytes']})")
        elif _sha256(p) != rec["sha256"]:
            changed.append(f"{rel} (sha256 differs)")
        else:
            ok += 1
    print(f"subset: {man['n_items']} item(s), {man['n_files']} file(s), "
          f"{man['total_bytes'] / 1e6:.1f} MB, selection {man['selection']}")
    print(f"  source cache: {man['source_cache']}")
    print(f"  verified {ok}/{man['n_files']}")
    for label, items in (("MISSING", missing), ("CHANGED", changed)):
        if items:
            print(f"  {label} ({len(items)}):")
            for x in items[:20]:
                print(f"    {x}")
    if not missing and not changed:
        print("  OK: every file matches. A truncated transfer shows up here.")
        return 0
    print("  FAILED: do not use this subset until it matches.")
    return 1


def _self_test() -> int:
    """No GPU, no torch needed for the core. The one part that does need
    cache_item (which imports torch) is attempted and skipped with a note when
    torch is absent -- this tool must be testable on the cache machine."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "cache" / "cafe"
        src.mkdir(parents=True)
        items = []
        for i, (mf, mv) in enumerate([(0.01, 0.9), (0.10, 0.5), (0.05, 0.8),
                                      (0.50, 0.9), (0.20, 0.1)]):
            np.savez_compressed(
                src / f"{i:05d}.npz",
                z_src=np.full((16, 64, 64), 0.1 * i, np.float32),
                z_tgt=np.full((16, 64, 64), 0.1 * i + 0.3, np.float32),
                mask=np.pad(np.ones((1, 8, 8), np.float32), ((0, 0), (28, 28),
                                                             (28, 28))))
            # an emb file that must NOT be copied (it is 200x bigger than an item)
            np.savez_compressed(src / f"emb_{i:016d}.npz",
                                prompt_embeds=np.zeros((1, 333, 4096), np.float32))
            items.append({"id": f"c{i}", "stem": f"{i:05d}", "instruction": "x",
                          "emb": f"{i:016d}", "mask_frac": mf, "moved": mv})
        (src / "meta.json").write_text(json.dumps({
            "config": {"size": 512, "latent_size": 64, "dtype": "float32",
                       "max_items": None, "context_length": 333},
            "n": len(items), "items": items}))

        # selection agrees with the band, and --top excludes both ends
        recs, rule = pick(items, None, 2, None, 0)
        assert [r["stem"] for r in recs] == ["00002", "00001"], recs
        assert rule == "top2_by_moved", rule
        recs_s, r2 = pick(items, None, None, 3, 7)
        assert len(recs_s) == 3 and 0 not in [r["stem"] for r in recs_s], recs_s
        assert r2 == "sample3_of3_inband_seed7", r2

        out = tmp / "sub"
        man = make_subset(src, recs, rule, out)
        assert man["n_items"] == 2 and man["n_files"] == 3, man  # 2 npz + meta
        # THE POINT: no emb file came along, so the subset is tiny
        assert not list(out.glob("emb_*")), list(out.glob("emb_*"))
        sub = json.loads((out / "meta.json").read_text())
        assert sub["n"] == 2 and len(sub["items"]) == 2
        assert sub["config"]["size"] == 512, "config must survive: export reads it"
        assert sub["subset"]["embeddings_included"] is False
        assert "TRAINING" in sub["subset"]["note"]
        # stems preserved, so the npz filenames still resolve
        assert [r["stem"] for r in sub["items"]] == ["00002", "00001"]
        # npz bytes identical to the source
        for r in sub["items"]:
            a = (src / f"{r['stem']}.npz").read_bytes()
            b = (out / f"{r['stem']}.npz").read_bytes()
            assert a == b, r["stem"]
        assert verify(out) == 0
        (out / "00002.npz").unlink()
        assert verify(out) == 1, "a missing npz must fail the check"
        (out / "manifest.json").unlink()
        assert verify(out) == 2

        # the one cross-check that needs cache_item (thus torch)
        #
        # sys.path: this file deliberately does NOT put the repo root on the
        # path, because the machine it runs on may have only numpy. Running it
        # as `python short_term/subset_cache.py` therefore has sys.path[0] =
        # short_term/, and `import short_term` fails -- which is correct at
        # RUNTIME and wrong for this check, so the root is added here only.
        root = str(Path(__file__).resolve().parent.parent)
        added = root not in sys.path
        if added:
            sys.path.insert(0, root)
        try:
            from short_term import cache_item as ci
            assert (ci.MASK_MIN, ci.MASK_MAX) == (MASK_MIN, MASK_MAX), (
                "the band here and in cache_item.py have drifted apart; the "
                "subset would be drawn from a different population than --list")
            _m, sub_items = ci.load_index(out)
            assert len(sub_items) == 2
            ranked = ci.rank(sub_items)
            assert all(MASK_MIN <= r["mask_frac"] <= MASK_MAX for r in ranked)
            print("  cross-checked against cache_item: band equal, subset loads")
        except ImportError as exc:
            print(f"  (skipped the cache_item cross-check: {exc}; this tool's "
                  "runtime path needs no short_term import, so that is fine on "
                  "a dataset-only machine)")
        finally:
            if added:
                sys.path.remove(root)

    print("subset_cache self-test OK")
    print("  emb_*.npz are NOT copied: the instruction string is in the index")
    print("  stems preserved (export finds the npz by stem, not by position)")
    print("  manifest verifies, and detects a missing file")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None,
                    help="source cache HASH directory (needs meta.json)")
    ap.add_argument("--indices", default=None)
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", type=int, default=0, metavar="N",
                    help="print the N best candidates and exit")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        if not args.out:
            raise SystemExit("--verify needs --out")
        return verify(args.out)

    if not args.cache:
        raise SystemExit("--cache is required (or --verify with --out)")
    meta, items = load_index(args.cache)

    if args.list:
        cand = rank(items)
        print(f"{len(items)} items; {len(cand)} in the mask_frac band "
              f"[{MASK_MIN}, {MASK_MAX}], most-changed first")
        print(f"\n{'index':>6}  {'id':<16}{'mask_frac':>10}{'moved':>8}  "
              "instruction")
        for r in cand[:args.list]:
            print(f"{items.index(r):>6}  {str(r['id'])[:16]:<16}"
                  f"{float(r.get('mask_frac', 0)):>10.4f}"
                  f"{float(r.get('moved', 0)):>8.3f}  "
                  f"{str(r['instruction'])[:56]}")
        return 0

    recs, rule = pick(items, args.indices, args.top, args.sample, args.seed)
    if not args.out:
        raise SystemExit("--out is required to write the subset")
    man = make_subset(args.cache, recs, rule, args.out)
    print(f"selection: {rule}")
    print(f"wrote {man['n_items']} item(s), {man['n_files']} file(s), "
          f"{man['total_bytes'] / 1e6:.1f} MB to {args.out}")
    print("  emb_*.npz deliberately NOT copied (see --help)")
    print("\nTRANSFER then, on the GPU machine:")
    print(f"  python short_term/subset_cache.py --out {args.out} --verify")
    print(f"  python short_term/cache_item.py --cache {args.out} --index 0 "
          "--out runs/st/one")
    print(f"  python short_term/suite.py --cache {args.out} --all "
          f"--out runs/st/sheet --arms base,both,protect,selfy")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
