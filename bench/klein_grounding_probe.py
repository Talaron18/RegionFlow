"""Does FLUX.2 klein know where the referent is?  (GROUNDING_DIAGNOSTIC.md)

    python bench/klein_grounding_probe.py --limit 4 \
        --out runs/gate_c/klein_grounding_smoke.jsonl \
        --save-maps runs/gate_c/klein_grounding_smoke_maps
    python bench/klein_grounding_probe.py \
        --out runs/gate_c/klein_grounding.jsonl \
        --save-maps runs/gate_c/klein_grounding_maps
    python bench/klein_grounding_probe.py --report runs/gate_c/klein_grounding.jsonl

For every case and every instruction (full referring expression, single-
instance instruction, and optionally a swapped instruction from another case)
it runs the ordinary 4-step no-mask edit and records, in every block and step,
the attention FROM the instruction's content tokens TO the reference-image
tokens. It then asks which annotated instance that attention favours: the
target, or a same-category sibling.

HOW IT RECORDS, AND WHY THAT IS SAFE. diffusers' FLUX.2 processors hand the
final (rotary-encoded) queries and keys to one function,
`dispatch_attention_fn`, for both block types. The probe wraps that function:
it calls the original unchanged and returns its result, and on the side
computes softmax(q k^T / sqrt(d)) for the few rows it needs. The model's
output is therefore the stock output; `self_test` checks that bit for bit on
a real (tiny, random) Flux2Transformer2DModel, and checks that the recorded
weights reproduce the stock attention output.

TOKEN LAYOUT in every attention call (read from diffusers 0.43.0):
    [ text (T, padded to 512) | noisy output (N) | reference image (R) ]
Reference tokens are packed row-major on a (H/16, W/16) grid.

PRIMARY STATISTIC (fixed in GROUNDING_DIAGNOSTIC.md before any run): pointing
accuracy of the map averaged over ALL blocks and ALL steps, with the full
instruction, compared with the single-instance instruction. Per-block curves
are exploratory.

Needs the same environment as bench/flux2_klein_model.py (recent stock
diffusers). NOT RUN ON A GPU when written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ----------------------------------------------------------- pure helpers ---

def pool_mask(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """(H, W) bool/float mask -> (gh, gw) float area fraction per token cell."""
    m = np.asarray(mask, dtype=np.float64)
    H, W = m.shape
    ys = np.linspace(0, H, gh + 1).round().astype(int)
    xs = np.linspace(0, W, gw + 1).round().astype(int)
    out = np.zeros((gh, gw))
    for i in range(gh):
        for j in range(gw):
            cell = m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[i, j] = cell.mean() if cell.size else 0.0
    return out


def densities(amap: np.ndarray, masks: list[np.ndarray]) -> list[float]:
    """Mean attention per unit area inside each (pooled) mask.

    amap sums to 1 over the grid. A density of 1/(gh*gw) is "uniform".
    Returns nan for a mask that covers no token cell.
    """
    out = []
    for m in masks:
        area = float(m.sum())
        out.append(float((amap * m).sum() / area) if area > 1e-9 else float("nan"))
    return out


def score(dens: list[float]) -> dict:
    """dens[0] is the target, the rest are siblings."""
    t, sib = dens[0], [d for d in dens[1:] if not math.isnan(d)]
    if math.isnan(t) or not sib:
        return {"pointing": None, "target_pref": None, "n_instances": 1 + len(sib)}
    return {"pointing": bool(t > max(sib)),
            "target_pref": float(t / (np.mean(sib) + 1e-12)),
            "n_instances": 1 + len(sib)}


def content_span(text: str, instruction: str, offsets) -> list[int]:
    """Token indices whose character span lies inside `instruction` within the
    chat-templated `text`. offsets: per-token (start, end); (0, 0) for padding."""
    a = text.rfind(instruction)
    if a < 0:
        raise ValueError("instruction not found in the templated prompt")
    b = a + len(instruction)
    return [i for i, (s, e) in enumerate(offsets) if e > s and s >= a and e <= b]


# -------------------------------------------------------------- recorder ---

class AttentionRecorder:
    """Wraps transformer_flux2.dispatch_attention_fn.

    After `configure(...)`, each attention call appends one row-normalised
    map: attention from `text_rows` to the reference columns, averaged over
    heads and rows, as a float64 numpy vector of length R. It also records
    how the text rows' attention mass splits over text / output / reference.
    """

    def __init__(self):
        self.orig = None
        self.module = None
        self.active = False
        self.maps: list[np.ndarray] = []
        self.mass: list[tuple[float, float, float]] = []
        self.text_rows = None
        self.T = self.N = self.R = 0

    def install(self):
        from diffusers.models.transformers import transformer_flux2 as mod  # noqa: PLC0415
        if self.orig is not None:
            return
        self.module, self.orig = mod, mod.dispatch_attention_fn
        rec = self

        def wrapped(query, key, value, *args, **kwargs):
            out = rec.orig(query, key, value, *args, **kwargs)
            if rec.active:
                rec._record(query, key)
            return out

        mod.dispatch_attention_fn = wrapped

    def remove(self):
        if self.orig is not None:
            self.module.dispatch_attention_fn = self.orig
            self.orig = None

    def configure(self, text_rows, T: int, N: int, R: int):
        self.text_rows, self.T, self.N, self.R = list(text_rows), T, N, R
        self.maps, self.mass = [], []

    @staticmethod
    def weights(query, key, rows):
        """softmax(q k^T / sqrt(d)) for `rows`. query/key: (B, S, H, D).
        Returns (H, len(rows), S) in float32, batch element 0."""
        import torch  # noqa: PLC0415
        q = query[0, rows].float().permute(1, 0, 2)          # (H, n, D)
        k = key[0].float().permute(1, 2, 0)                  # (H, D, S)
        logits = torch.matmul(q, k) / math.sqrt(q.shape[-1])
        return torch.softmax(logits, dim=-1)

    def _record(self, query, key):
        S = query.shape[1]
        if S != self.T + self.N + self.R:
            raise RuntimeError(
                f"attention call has {S} tokens, expected T+N+R = "
                f"{self.T}+{self.N}+{self.R}. The token layout assumption is "
                "wrong for this pipeline/version; do not trust any output.")
        w = self.weights(query, key, self.text_rows).mean(dim=(0, 1))   # (S,)
        t = float(w[:self.T].sum())
        o = float(w[self.T:self.T + self.N].sum())
        r = float(w[self.T + self.N:].sum())
        self.mass.append((t, o, r))
        ref = w[self.T + self.N:].double()
        self.maps.append((ref / ref.sum().clamp(min=1e-20)).cpu().numpy())


# ------------------------------------------------------------- the probe ---

def probe_case(pipe, rec, src_img, instruction, seed, steps, guidance, device):
    """Run one edit with recording on. Returns (maps (calls, gh, gw), mass, grid)."""
    import torch  # noqa: PLC0415

    W, H = src_img.size
    if W % 16 or H % 16:
        raise SystemExit(f"source size {W}x{H} is not a multiple of 16")
    gh, gw = H // 16, W // 16

    tok = pipe.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": instruction}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, padding="max_length", truncation=True, max_length=512,
              return_offsets_mapping=True)
    rows = content_span(text, instruction, enc["offset_mapping"])
    if not rows:
        raise SystemExit("no content tokens found for the instruction")

    rec.configure(rows, T=512, N=gh * gw, R=gh * gw)
    rec.active = True
    try:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        pipe(prompt=instruction, image=src_img, height=H, width=W,
             num_inference_steps=steps, guidance_scale=guidance, generator=gen)
    finally:
        rec.active = False
    maps = np.stack(rec.maps).reshape(len(rec.maps), gh, gw)
    return maps, np.asarray(rec.mass), (gh, gw), len(rows)


def run(args):
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from run_gate_c import PROMPT_COLS, build_masks, case_seed  # noqa: PLC0415
    from refedit_siblings import bench_image, load_bench, load_coco  # noqa: PLC0415
    from diffusers import Flux2KleinPipeline  # noqa: PLC0415

    blob = json.loads(Path(args.cases).read_text())
    cases = blob["cases"][:args.limit] if args.limit else blob["cases"]
    coco_index, _c, _u = load_coco(Path(args.coco_annotations),
                                   tuple(args.coco_splits.split(",")),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["id"], d["prompt_kind"]))
        print(f"resuming: {len(done)} results already in {out_path}")
    elif out_path.exists():
        out_path.unlink()
    map_dir = Path(args.save_maps) if args.save_maps else None
    if map_dir:
        map_dir.mkdir(parents=True, exist_ok=True)

    pipe = Flux2KleinPipeline.from_pretrained(args.repo, torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    n_blocks = (int(pipe.transformer.config.num_layers)
                + int(pipe.transformer.config.num_single_layers))
    rec = AttentionRecorder()
    rec.install()

    kinds = ["full", "single"] + (["swap"] if args.swap else [])
    ids = [c["id"] for c in cases]
    try:
        with out_path.open("a") as fh:
            for ci, case in enumerate(cases):
                idx = int(case["id"].rsplit("-", 1)[1])
                row = rows.get(idx)
                if row is None:
                    continue
                tmask, smasks = build_masks(case, coco_index)
                if tmask is None:
                    print(f"  {case['id']}: masks unavailable, skipped")
                    continue
                src_img = bench_image(row["input_image"])
                if src_img.size != (512, 512):
                    src_img = src_img.resize((512, 512), Image.BILINEAR)
                for kind in kinds:
                    if (case["id"], kind) in done:
                        continue
                    if kind == "swap":
                        # another case's full instruction on this image
                        other = cases[(ci + len(cases) // 2) % len(cases)]
                        orow = rows.get(int(other["id"].rsplit("-", 1)[1]))
                        if orow is None or other["id"] == case["id"]:
                            continue
                        instr = orow["editing_instruction"]
                    else:
                        instr = row.get(PROMPT_COLS[kind]) or row["editing_instruction"]
                    maps, mass, (gh, gw), n_tok = probe_case(
                        pipe, rec, src_img, instr, case_seed(case["id"], args.seed),
                        args.steps, args.guidance, args.device)
                    if maps.shape[0] != n_blocks * args.steps:
                        raise SystemExit(
                            f"recorded {maps.shape[0]} attention calls, expected "
                            f"{n_blocks} blocks x {args.steps} steps. The call "
                            "count assumption is wrong; stop.")
                    pm = [pool_mask(np.asarray(tmask) > 0, gh, gw)] + \
                         [pool_mask(np.asarray(s) > 0, gh, gw) for s in smasks]
                    per_call = [densities(m, pm) for m in maps]
                    agg = maps.mean(axis=0)
                    agg_d = densities(agg, pm)
                    rec_out = {
                        "id": case["id"], "prompt_kind": kind, "instruction": instr,
                        "n_content_tokens": n_tok, "grid": [gh, gw],
                        "n_blocks": n_blocks, "steps": args.steps,
                        "agg_densities": agg_d, **score(agg_d),
                        "uniform_density": 1.0 / (gh * gw),
                        # [step][block] -> densities per instance (target first)
                        "per_call_densities": per_call,
                        # mean share of the instruction tokens' attention going
                        # to text / output / reference tokens
                        "mass_text_out_ref": mass.mean(axis=0).tolist(),
                    }
                    fh.write(json.dumps(rec_out) + "\n")
                    fh.flush()
                    if map_dir:
                        np.save(map_dir / f"{case['id']}_{kind}.npy",
                                agg.astype(np.float32))
                        _save_overlay(src_img, agg, np.asarray(tmask) > 0,
                                      [np.asarray(s) > 0 for s in smasks],
                                      map_dir / f"{case['id']}_{kind}.png")
                print(f"  {ci + 1}/{len(cases)} {case['id']}")
    finally:
        rec.remove()
    report(out_path)


def _outline(mask: np.ndarray, px: int = 2) -> np.ndarray:
    m = np.asarray(mask, bool)
    er = m.copy()
    for _ in range(px):
        e = er.copy()
        e[1:] &= er[:-1]; e[:-1] &= er[1:]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]
        er = e
    return m & ~er


def _save_overlay(src_img, amap, tmask, smasks, path):
    """Source with the attention map as a heat tint; target outlined green,
    siblings red. For looking at before trusting any number."""
    from PIL import Image  # noqa: PLC0415
    src = np.asarray(src_img.convert("RGB")).astype(np.float64)
    a = amap / (amap.max() + 1e-12)
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
        src_img.size, Image.BILINEAR)).astype(np.float64) / 255.0
    out = src * (0.35 + 0.65 * a[..., None])
    out[_outline(tmask)] = (0, 255, 0)
    for s in smasks:
        out[_outline(s)] = (255, 0, 0)
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- report ---

def _acc(recs):
    v = [r["pointing"] for r in recs if r.get("pointing") is not None]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)


def report(path, outcomes=None):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by = {}
    for r in recs:
        by.setdefault(r["prompt_kind"], {})[r["id"]] = r
    print(f"\n{'=' * 72}\nKLEIN GROUNDING PROBE -- {path}\n{'=' * 72}")
    print("PRIMARY: pointing accuracy of the map averaged over all blocks and steps")
    print("  (does the instruction's attention favour the target over every sibling?)")
    for kind in ("full", "single", "swap"):
        if kind not in by:
            continue
        rs = list(by[kind].values())
        acc, n = _acc(rs)
        chance = np.mean([1.0 / r["n_instances"] for r in rs
                          if r.get("pointing") is not None]) if n else float("nan")
        pref = [r["target_pref"] for r in rs if r.get("target_pref") is not None]
        mass = np.mean([r["mass_text_out_ref"] for r in rs], axis=0)
        print(f"  {kind:<7} n={n:<4} pointing {100 * acc:5.1f}%   chance {100 * chance:5.1f}%   "
              f"target_pref median {np.median(pref):.2f}   "
              f"attention mass text/out/ref {mass[0]:.2f}/{mass[1]:.2f}/{mass[2]:.2f}")
    if "full" in by and "single" in by:
        common = [i for i in by["full"] if i in by["single"]
                  and by["full"][i].get("pointing") is not None
                  and by["single"][i].get("pointing") is not None]
        f = np.mean([by["full"][i]["pointing"] for i in common])
        s = np.mean([by["single"][i]["pointing"] for i in common])
        up = sum(by["full"][i]["pointing"] and not by["single"][i]["pointing"] for i in common)
        dn = sum(by["single"][i]["pointing"] and not by["full"][i]["pointing"] for i in common)
        print(f"\n  paired on {len(common)} cases: full {100 * f:.1f}%  single {100 * s:.1f}%  "
              f"difference {100 * (f - s):+.1f} points   (full-only correct {up}, single-only {dn})")
        print("  reading (GROUNDING_DIAGNOSTIC.md): >= +20 and full >= 70% in leaked cases "
              "-> grounds, edit spreads; < +10 -> grounding failure")
    if outcomes and "full" in by:
        print("\n  by the plain run's outcome class (full instruction):")
        groups = {}
        for i, r in by["full"].items():
            groups.setdefault(outcomes.get(i, "?"), []).append(r)
        for g, rs in sorted(groups.items()):
            acc, n = _acc(rs)
            print(f"    {g:<20} n={n:<4} pointing {100 * acc:5.1f}%")
    if "full" in by:
        rs = [r for r in by["full"].values() if r.get("pointing") is not None]
        if rs:
            nb, st = rs[0]["n_blocks"], rs[0]["steps"]
            print("\n  EXPLORATORY: pointing accuracy per block (rows) and step (cols), full instruction")
            for b in range(nb):
                cells = []
                for s_ in range(st):
                    v = []
                    for r in rs:
                        d = r["per_call_densities"][s_ * nb + b]
                        sc = score(d)
                        if sc["pointing"] is not None:
                            v.append(sc["pointing"])
                    cells.append(f"{100 * np.mean(v):5.1f}" if v else "   --")
                print(f"    block {b:>2}  " + "  ".join(cells))
            print("    (the best block is chosen after seeing the data; do not quote it as the result)")


# -------------------------------------------------------------- self-test ---

def self_test():
    # ---- pure helpers (numpy only) --------------------------------------
    m = np.zeros((512, 512), bool); m[0:256, 0:256] = True
    p = pool_mask(m, 32, 32)
    assert p.shape == (32, 32) and p[:16, :16].min() == 1.0 and p[16:, :].max() == 0.0
    half = np.zeros((32, 32), bool); half[:8] = True          # covers half of the first cell row
    assert abs(pool_mask(half, 2, 2)[0, 0] - 0.5) < 1e-9

    amap = np.zeros((4, 4)); amap[0, 0] = 0.75; amap[3, 3] = 0.25
    tgt = np.zeros((4, 4)); tgt[0, 0] = 1
    sib = np.zeros((4, 4)); sib[3, 3] = 1
    d = densities(amap, [tgt, sib])
    assert d == [0.75, 0.25]
    assert score(d) == {"pointing": True, "target_pref": 0.75 / (0.25 + 1e-12), "n_instances": 2}
    assert score([0.25, 0.75])["pointing"] is False
    # density is per unit AREA: a big sibling with the same total mass loses
    big = np.zeros((4, 4)); big[2:, 2:] = 1
    assert densities(amap, [tgt, big])[1] == 0.25 / 4
    assert score([0.5])["pointing"] is None                       # no sibling -> undefined
    empty = np.zeros((4, 4))
    assert math.isnan(densities(amap, [empty])[0])

    text = "<|im_start|>user\nMake the cow in back black<|im_end|>\n<|im_start|>assistant\n"
    instr = "Make the cow in back black"
    a = text.index(instr)
    offs = [(0, 12), (12, 17), (17, a)]                           # template tokens
    words, pos = [], a
    for w in ["Make", " the", " cow", " in", " back", " black"]:
        words.append((pos, pos + len(w))); pos += len(w)
    offs += words + [(pos, pos + 10), (0, 0), (0, 0)]            # template tail, padding
    assert content_span(text, instr, offs) == [3, 4, 5, 6, 7, 8]
    try:
        content_span(text, "not there", offs)
    except ValueError:
        pass
    else:
        raise AssertionError("missing instruction was accepted")
    print("klein_grounding_probe helpers self-test OK")

    # ---- the recorder, on the real diffusers attention path -------------
    try:
        import torch  # noqa: PLC0415
        from diffusers.models.transformers import transformer_flux2 as mod  # noqa: PLC0415
        from diffusers.models.transformers.transformer_flux2 import (  # noqa: PLC0415
            Flux2Transformer2DModel)
    except ImportError as e:
        print(f"  (recorder test SKIPPED, missing dependency: {e})")
        return

    torch.manual_seed(0)
    T, gh, gw = 6, 4, 4
    N = R = gh * gw
    model = Flux2Transformer2DModel(
        patch_size=1, in_channels=8, num_layers=2, num_single_layers=3,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
        timestep_guidance_channels=32, mlp_ratio=3.0,
        axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False).eval()

    def ids(n, t0, grid=None):
        x = torch.zeros(n, 4)
        x[:, 0] = t0
        if grid:
            yy, xx = torch.meshgrid(torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij")
            x[:, 1], x[:, 2] = yy.flatten(), xx.flatten()
        else:
            x[:, 3] = torch.arange(n)
        return x[None]

    hs = torch.randn(1, N + R, 8)
    ehs = torch.randn(1, T, 24)
    kw = dict(hidden_states=hs, encoder_hidden_states=ehs,
              timestep=torch.tensor([0.5]), guidance=None,
              img_ids=torch.cat([ids(N, 0, (gh, gw)), ids(R, 10, (gh, gw))], dim=1),
              txt_ids=ids(T, 0), return_dict=False)
    with torch.no_grad():
        ref_out = model(**kw)[0]

    rec = AttentionRecorder()
    rec.install()
    try:
        # the probe must not change the model's output at all
        rec.configure([1, 2, 3], T=T, N=N, R=R)
        rec.active = True
        with torch.no_grad():
            out = model(**kw)[0]
        rec.active = False
        assert torch.equal(out, ref_out), "recording changed the model output"
        assert len(rec.maps) == 2 + 3, f"expected one map per block, got {len(rec.maps)}"
        for mp, ms in zip(rec.maps, rec.mass):
            assert mp.shape == (R,) and abs(mp.sum() - 1) < 1e-9 and (mp >= 0).all()
            assert abs(sum(ms) - 1) < 1e-4, ms                    # text+out+ref mass = 1

        # the weights the probe computes are the model's own: they must
        # reproduce the stock attention output for every block
        seen = []
        orig = rec.orig

        def check(query, key, value, *a, **k):
            o = orig(query, key, value, *a, **k)
            w = AttentionRecorder.weights(query, key, list(range(query.shape[1])))  # (H,S,S)
            mine = torch.matmul(w, value[0].float().permute(1, 0, 2)).permute(1, 0, 2)
            seen.append(float((mine - o[0].float()).abs().max()))
            return o

        mod.dispatch_attention_fn = check
        with torch.no_grad():
            model(**kw)
        assert len(seen) == 5 and max(seen) < 1e-4, seen

        # a wrong layout must be refused, not silently recorded
        mod.dispatch_attention_fn = None
        rec.remove(); rec.install()
        rec.configure([1], T=T + 1, N=N, R=R)
        rec.active = True
        try:
            with torch.no_grad():
                model(**kw)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong token layout was accepted")
        finally:
            rec.active = False
    finally:
        rec.remove()
    assert mod.dispatch_attention_fn is not None
    with torch.no_grad():
        assert torch.equal(model(**kw)[0], ref_out), "removing the probe did not restore the model"
    print("klein_grounding_probe recorder self-test OK (real Flux2Transformer2DModel, CPU)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--repo", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--swap", action="store_true",
                   help="also probe each image with another case's instruction "
                        "(control: target preference should fall to about 1)")
    p.add_argument("--out", default="runs/gate_c/klein_grounding.jsonl")
    p.add_argument("--save-maps", default=None,
                   help="directory for per-case aggregate maps (.npy) and "
                        "overlays (.png): target outlined green, siblings red")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report", default=None, metavar="JSONL")
    p.add_argument("--outcomes", default=None, metavar="JSONL",
                   help="a run_gate_c results file (e.g. klein4b_none.jsonl) to "
                        "split the report by that run's outcome class")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.report:
        outcomes = None
        if args.outcomes:
            from run_gate_c import outcome_class  # noqa: PLC0415
            outcomes = {}
            for line in Path(args.outcomes).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("prompt_kind", "full") == "full":
                        outcomes[r["id"]] = r.get("outcome") or outcome_class(r)
        report(args.report, outcomes)
        return
    run(args)


if __name__ == "__main__":
    main()
