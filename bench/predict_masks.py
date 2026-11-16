"""Predicted-mask pipeline, step 1: does a referring-segmentation model find the
instance an editing instruction refers to?  (MODERN_BASE_EVAL.md, section
"Predicted-mask pipeline -- scope"; design and thresholds FROZEN there.)

    python bench/predict_masks.py --backend oracle                  # control: 100% target
    python bench/predict_masks.py --backend sibling                 # control: 0% target
    python bench/predict_masks.py --backend sa2va --limit 4 \
        --save-overlays runs/gate_c/pred_overlays_smoke \
        --out runs/gate_c/pred_masks_smoke.jsonl --out-dir runs/gate_c/pred_masks_smoke
    python bench/predict_masks.py --backend sa2va --prompt-kind full
    python bench/predict_masks.py --backend sa2va --prompt-kind single
    python bench/predict_masks.py --report runs/gate_c/pred_masks_sa2va_full.jsonl \
        --report-single runs/gate_c/pred_masks_sa2va_single.jsonl \
        --outcomes runs/gate_c/klein4b_none.jsonl

Every backend returns a bool HxW mask; it is saved as an 8-bit L PNG (0/255,
512x512) at <out-dir>/<id>.png, RE-LOADED from that file, and only then scored.
So the oracle control tests save / load / score, not just score.

SCORE (pure function, `score_masks`). P predicted, T target, S_k = sibling k with
target pixels removed:
    in_T = |P & T| / |P|     in_S = max_k |P & S_k| / |P|
    area_frac = |P| / pixels   iou_target = IoU(P, T) (secondary)
    empty    area_frac < 0.0005
    target   in_T >= t_in  and in_S <  t_leak          (t_in = 0.5, t_leak = 0.2)
    sibling  in_S >= t_in  and in_T <  t_leak
    both     in_T >= t_leak and in_S >= t_leak
    other    anything else
The thresholds are parameters only so the report can print a sensitivity table.

NOT RUN ON A GPU when written. The sa2va backend follows the Sa2VA model card
and the official demo script (see the comment above `make_sa2va`).
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from grounding_saliency import fisher

PROMPT = ("Please segment the existing object in the image that this editing "
          "instruction refers to: {instruction}")
PROMPT_COLS = {"full": "editing_instruction", "single": "edit_instruction_single"}
T_IN, T_LEAK, MIN_AREA = 0.5, 0.2, 0.0005
BARS = ((113, "<= 113 of 151: same-kind referents are hard for a pipeline too"),
        (135, "114-135: mixed"),
        (151, ">= 136 of 151: finding the instance is solved on this benchmark"))


# ----------------------------------------------------------- pure helpers ---

def classify(in_t: float, in_s: float, area_frac: float,
             t_in: float = T_IN, t_leak: float = T_LEAK) -> str:
    if area_frac < MIN_AREA:
        return "empty"
    if in_t >= t_in and in_s < t_leak:
        return "target"
    if in_s >= t_in and in_t < t_leak:
        return "sibling"
    if in_t >= t_leak and in_s >= t_leak:
        return "both"
    return "other"


def score_masks(pred, tmask, smasks) -> dict:
    """Pure scoring. Sibling pixels that are also target pixels do not count as
    sibling (S_k & ~T): a mask on an overlap is credited to the target only."""
    P, T = np.asarray(pred, bool), np.asarray(tmask, bool)
    sibs = [np.asarray(s, bool) & ~T for s in smasks]
    sibs = [s for s in sibs if s.any()]
    n = int(P.sum())
    if n == 0:
        in_t = in_s = iou = 0.0
    else:
        in_t = float((P & T).sum()) / n
        in_s = max((float((P & s).sum()) / n for s in sibs), default=0.0)
        iou = float((P & T).sum()) / float((P | T).sum())
    area = n / P.size
    return {"area_frac": area, "in_T": in_t, "in_S": in_s, "iou_target": iou,
            "n_instances": 1 + len(smasks), "cls": classify(in_t, in_s, area)}


def save_mask(mask, path) -> None:
    from PIL import Image  # noqa: PLC0415
    Image.fromarray((np.asarray(mask, bool) * 255).astype(np.uint8)).save(path)


def load_mask(path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415
    return np.asarray(Image.open(path).convert("L")) >= 128


def fit_mask(mask, size=(512, 512)) -> np.ndarray:
    """bool mask -> bool (H, W) = size[::-1], NEAREST."""
    from PIL import Image  # noqa: PLC0415
    m = np.asarray(mask, bool)
    if m.shape == (size[1], size[0]):
        return m
    im = Image.fromarray((m * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    return np.asarray(im) >= 128


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_done(out_path: Path, backend: str, model_path: str, overwrite: bool) -> set:
    """Resume on (id, prompt_kind). Refuses to append to rows of another
    backend / checkpoint: a mixed file would be scored as one run."""
    done = set()
    if not out_path.exists():
        return done
    if overwrite:
        out_path.unlink()
        return done
    for d in read_jsonl(out_path):
        if d.get("backend") != backend or d.get("model_path") != model_path:
            raise SystemExit(
                f"{out_path} holds rows for backend={d.get('backend')!r} "
                f"model_path={d.get('model_path')!r}, not backend={backend!r} "
                f"model_path={model_path!r}. Use a separate --out (or --overwrite).")
        done.add((d["id"], d["prompt_kind"]))
    return done


def process_case(case_id, kind, backend_name, model_path, backend, src_img, prompt,
                 tmask, smasks, out_dir: Path):
    """backend(src_img, prompt, tmask, smasks) -> (bool mask, raw_answer, n_masks).
    Save -> reload -> score. Returns (row, reloaded_mask)."""
    mask, raw, n_masks = backend(src_img, prompt, tmask, smasks)
    path = out_dir / f"{case_id}.png"
    save_mask(fit_mask(mask), path)
    P = load_mask(path)
    row = {"id": case_id, "prompt_kind": kind, "backend": backend_name,
           "model_path": model_path, "prompt": prompt, "raw_answer": raw,
           "decode": getattr(backend, "decode", ""),
           "model_revision": getattr(backend, "revision", ""),
           "n_masks_returned": n_masks, **score_masks(P, tmask, smasks),
           "mask_file": str(path)}
    return row, P


# --------------------------------------------------------------- backends ---

def oracle_backend(src_img, prompt, tmask, smasks):
    return np.asarray(tmask) > 0, "", 1


def sibling_backend(src_img, prompt, tmask, smasks):
    return max((np.asarray(s) > 0 for s in smasks), key=lambda x: int(x.sum())), "", 1


# Recipe, verified 2026-11-16 against these primary sources (not from memory):
#   model card  https://huggingface.co/ByteDance/Sa2VA-Qwen2_5-VL-7B  (license: apache-2.0)
#   demo        https://github.com/bytedance/Sa2VA/blob/main/projects/sa2va/demo/demo.py
#   remote code https://huggingface.co/ByteDance/Sa2VA-Qwen2_5-VL-7B/blob/main/modeling_sa2va_qwen.py
# Card, verbatim:
#   from transformers import AutoTokenizer, AutoModel
#   model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
#       low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
#   tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
#   text_prompts = "<image>Please segment ..."      # card: prompts start with <image>
#   image = Image.open(image_path).convert('RGB')
#   input_dict = {'image': image, 'text': text_prompts, 'past_text': '',
#                 'mask_prompts': None, 'tokenizer': tokenizer}
#   return_dict = model.predict_forward(**input_dict)
#   answer = return_dict["prediction"]
#   masks = return_dict['prediction_masks']  # segmentation masks, list(np.array(1, h, w), ...)
# DISCREPANCY the card does not mention: this checkpoint's modeling_sa2va_qwen.py
# `predict_forward(self, image, video, text, past_text, mask_prompts, tokenizer,
# processor)` does `assert processor is not None`, and the official demo.py builds
# `processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)`
# with tokenizer=None for any path containing "qwen". That is what is used here.
# Remote code also strips "<image>" from the text and takes an image-or-video
# argument; masks are bool numpy (1, h, w) at the input image's size, sigmoid > 0.5.
# Remote code imports qwen_vl_utils; config.json says transformers 4.56.1 (the
# card gives no version pin; the repo's "legacy" 4.42.3 pin is for InternVL2.5).
# Only the Qwen2.5-VL checkpoint's code was read; other checkpoints exit.
def make_sa2va(model_path: str, device: str):
    if "qwen" not in model_path.lower():
        raise SystemExit(
            f"recipe not verified for {model_path!r}: only the Sa2VA-Qwen2_5-VL "
            "predict_forward (processor=, tokenizer=None) was read. Do not guess "
            "the InternVL-based signature.")
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def backend(src_img, prompt, tmask, smasks):
        with torch.no_grad():
            out = model.predict_forward(
                image=src_img.convert("RGB"), text=prompt, past_text="",
                mask_prompts=None, tokenizer=None, processor=processor)
        masks = out.get("prediction_masks") or []
        raw = out.get("prediction", "")
        if not len(masks):
            return np.zeros((512, 512), bool), raw, 0
        m = np.asarray(masks[0], bool)
        if m.ndim == 3:                      # (frames=1, h, w)
            m = m[0]
        return m, raw, len(masks)

    backend.image_token = "<image>"
    # predict_forward calls self.model.generate(..., max_new_tokens=2048,
    # do_sample=False, ...) (modeling_sa2va_qwen.py): greedy, so no seeding.
    backend.decode = "greedy"
    backend.revision = getattr(model.config, "_commit_hash", None) or "unknown"
    return backend


def make_backend(name: str, model_path: str, device: str):
    if name == "oracle":
        return oracle_backend
    if name == "sibling":
        return sibling_backend
    if name == "sa2va":
        return make_sa2va(model_path, device)
    raise SystemExit(f"unknown backend {name!r}")


# --------------------------------------------------------------- overlays ---

def _outline(mask: np.ndarray, px: int = 2) -> np.ndarray:
    m = np.asarray(mask, bool)
    er = m.copy()
    for _ in range(px):
        e = er.copy()
        e[1:] &= er[:-1]; e[:-1] &= er[1:]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]
        er = e
    return m & ~er


def save_overlay(src_img, pred, tmask, smasks, path) -> None:
    """Source with the predicted mask as a translucent blue tint; target outlined
    green, siblings red. For looking at before trusting any number."""
    from PIL import Image  # noqa: PLC0415
    out = np.asarray(src_img.convert("RGB")).astype(np.float64)
    p = np.asarray(pred, bool)
    out[p] = 0.55 * out[p] + 0.45 * np.array([0, 80, 255.0])
    out[_outline(np.asarray(tmask) > 0)] = (0, 255, 0)
    for s in smasks:
        out[_outline(np.asarray(s) > 0)] = (255, 0, 0)
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(path)


# -------------------------------------------------------------------- run ---

def run(args):
    from PIL import Image  # noqa: PLC0415

    from run_gate_c import build_masks  # noqa: PLC0415
    from refedit_siblings import bench_image, load_bench, load_coco  # noqa: PLC0415

    kind = args.prompt_kind
    out_dir = Path(args.out_dir or f"runs/gate_c/pred_masks_{args.backend}_{kind}")
    out_path = Path(args.out or f"runs/gate_c/pred_masks_{args.backend}_{kind}.jsonl")
    model_path = args.model_path if args.backend == "sa2va" else ""

    blob = json.loads(Path(args.cases).read_text())
    cases = blob["cases"][:args.limit] if args.limit else blob["cases"]
    coco_index, _c, _u = load_coco(Path(args.coco_annotations),
                                   tuple(args.coco_splits.split(",")),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path, args.backend, model_path, args.overwrite)
    if done:
        print(f"resuming: {len(done)} results already in {out_path}")
    ov_dir = Path(args.save_overlays) if args.save_overlays else None
    if ov_dir:
        ov_dir.mkdir(parents=True, exist_ok=True)

    backend = make_backend(args.backend, model_path, args.device)
    token = getattr(backend, "image_token", "")
    with out_path.open("a") as fh:
        for ci, case in enumerate(cases):
            if (case["id"], kind) in done:
                continue
            row = rows.get(int(case["id"].rsplit("-", 1)[1]))
            if row is None:
                continue
            tmask, smasks = build_masks(case, coco_index)
            if tmask is None:
                print(f"  {case['id']}: masks unavailable, skipped")
                continue
            src_img = bench_image(row["input_image"])
            if src_img.size != (512, 512):
                src_img = src_img.resize((512, 512), Image.BILINEAR)
            instr = row.get(PROMPT_COLS[kind]) or row["editing_instruction"]
            prompt = token + PROMPT.format(instruction=instr)
            rec, P = process_case(case["id"], kind, args.backend, model_path, backend,
                                  src_img, prompt, np.asarray(tmask) > 0,
                                  [np.asarray(s) > 0 for s in smasks], out_dir)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if ov_dir:
                save_overlay(src_img, P, np.asarray(tmask) > 0,
                             [np.asarray(s) > 0 for s in smasks], ov_dir / f"{case['id']}.png")
            print(f"  {ci + 1}/{len(cases)} {case['id']}  {rec['cls']}  "
                  f"in_T {rec['in_T']:.2f} in_S {rec['in_S']:.2f}")
    report(out_path)


# ---------------------------------------------------------------- report ---

def _last_by_id(recs):
    return {r["id"]: r for r in recs}


def target_count(recs, t_in=T_IN, t_leak=T_LEAK) -> int:
    return sum(classify(r["in_T"], r["in_S"], r["area_frac"], t_in, t_leak) == "target"
               for r in recs)


def outcome_split(recs, outcomes: dict) -> dict:
    """target rate where the plain klein run's outcome is `measurable` vs not."""
    a = b = c = d = 0
    for r in recs:
        if r["id"] not in outcomes:
            continue
        is_t = classify(r["in_T"], r["in_S"], r["area_frac"]) == "target"
        if outcomes[r["id"]] == "measurable":
            a, b = a + is_t, b + (not is_t)
        else:
            c, d = c + is_t, d + (not is_t)
    return {"a": a, "b": b, "c": c, "d": d, "p": fisher(a, b, c, d)}


def paired_full_single(full: dict, single: dict) -> dict:
    ids = sorted(set(full) & set(single))
    keep = [i for i in ids if full[i]["prompt"] != single[i]["prompt"]]
    t = lambda r: classify(r["in_T"], r["in_S"], r["area_frac"]) == "target"  # noqa: E731
    f_only = sum(t(full[i]) and not t(single[i]) for i in keep)
    s_only = sum(t(single[i]) and not t(full[i]) for i in keep)
    n = f_only + s_only
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(f_only, s_only) + 1))
            / 2 ** n) if n else 1.0
    return {"n_paired": len(keep), "dropped_identical": len(ids) - len(keep),
            "full_only": f_only, "single_only": s_only, "p": p}


def _summary(name, recs) -> list[str]:
    M = len(recs)
    lines = [f"{name}: {M} cases"]
    if not M:
        return lines
    cls = {k: 0 for k in ("target", "sibling", "both", "other", "empty")}
    for r in recs:
        cls[classify(r["in_T"], r["in_S"], r["area_frac"])] += 1
    lines.append("  classes (t_in 0.5, t_leak 0.2): " + "  ".join(
        f"{k} {v}" for k, v in cls.items()))
    n = cls["target"]
    lines.append(f"  target: {n} of {M} ({100 * n / M:.1f}%)")
    if M == 151:
        band = BARS[0][1] if n <= 113 else BARS[1][1] if n <= 135 else BARS[2][1]
        lines.append(f"  pre-set bars: <= 113 of 151 | 114-135 | >= 136 of 151   "
                     f"-> this run is in: {band}")
    else:
        lines.append(f"  pre-set bars (<= 113 / 114-135 / >= 136 of 151) apply only to "
                     f"M = 151; M = {M} here, do not read them")
    lines.append(f"  empty: {cls['empty']} of {M}   median iou_target "
                 f"{np.median([r['iou_target'] for r in recs]):.3f}   "
                 f"chance (mean 1/n_instances) "
                 f"{100 * np.mean([1.0 / r['n_instances'] for r in recs]):.1f}%")
    lines.append("  sensitivity, `target` count   t_leak:   0.1     0.2     0.3")
    for ti in (0.5, 0.7, 0.9):
        lines.append(f"    t_in {ti}               " + "".join(
            f"{target_count(recs, ti, tl):>8}" for tl in (0.1, 0.2, 0.3)))
    return lines


def report(path, path_single=None, outcomes_path=None) -> None:
    full = _last_by_id(read_jsonl(path))
    print(f"\n{'=' * 72}\nPREDICTED-MASK PIPELINE -- {path}\n{'=' * 72}")
    meta = {(r.get("backend"), r.get("model_path")) for r in full.values()}
    print(f"backend / model_path: {sorted(meta)}")
    print("\n".join(_summary("prompt as stored in the file", list(full.values()))))
    single = None
    if path_single:
        single = _last_by_id(read_jsonl(path_single))
        print("\n" + "\n".join(_summary(f"second file {path_single}", list(single.values()))))
        pr = paired_full_single(full, single)
        print(f"\n  paired full vs single on {pr['n_paired']} cases whose prompts differ "
              f"({pr['dropped_identical']} dropped as identical):")
        print(f"    target with full only {pr['full_only']}   with single only "
              f"{pr['single_only']}   exact two-sided sign test p = {pr['p']:.4f}")
    if outcomes_path:
        outcomes = {}
        for r in read_jsonl(outcomes_path):
            if r.get("prompt_kind", "full") == "full":
                if r.get("outcome") is None:
                    from run_gate_c import outcome_class  # noqa: PLC0415
                outcomes[r["id"]] = r.get("outcome") or outcome_class(r)
        s = outcome_split(list(full.values()), outcomes)
        na, nc = s["a"] + s["b"], s["c"] + s["d"]
        print("\n  by the plain klein run's outcome (edit on target = `measurable`):")
        print(f"    measurable      target {s['a']} of {na}"
              f" ({100 * s['a'] / max(na, 1):.1f}%)")
        print(f"    not measurable  target {s['c']} of {nc}"
              f" ({100 * s['c'] / max(nc, 1):.1f}%)   Fisher exact two-sided p = {s['p']:.4f}")


# -------------------------------------------------------------- self-test ---

def self_test():
    N = 128
    z = lambda: np.zeros((N, N), bool)  # noqa: E731
    T = z(); T[20:60, 20:60] = True               # 1600
    S1 = z(); S1[20:100, 70:120] = True           # 4000, the largest sibling
    S2 = z(); S2[100:120, 10:30] = True           # 400
    sibs = [S2, S1]

    # ---- classification ---------------------------------------------------
    def cls(P, sb=sibs):
        return score_masks(P, T, sb)["cls"]
    sub = z(); sub[20:24, 20:60] = True           # 160 px = 10% of T
    blob = z(); blob[0:10, 0:10] = True
    assert cls(T) == "target"
    assert cls(S1) == "sibling"
    assert sub.sum() * 10 == T.sum() and cls(sub) == "target"
    assert cls(T | S1) == "both"
    assert cls(z()) == "empty"
    assert cls(blob) == "other"
    assert cls(np.ones((N, N), bool)) != "target"
    tiny = z(); tiny[0, 0:3] = True               # 3/16384 < 0.0005
    assert cls(tiny) == "empty"
    sc = score_masks(T, T, sibs)
    assert sc["in_T"] == 1.0 and sc["in_S"] == 0.0 and sc["iou_target"] == 1.0
    assert sc["n_instances"] == 3
    print("classification self-test OK")

    # ---- sibling shares use S_k & ~T --------------------------------------
    ov = z(); ov[40:80, 40:80] = True             # overlaps T in 400 px; 1200 px outside T
    sc = score_masks(ov, T, [ov])
    assert abs(sc["in_T"] - 0.25) < 1e-12 and abs(sc["in_S"] - 0.75) < 1e-12, sc
    assert sc["cls"] == "both"
    assert score_masks(T, T, [ov])["in_S"] == 0.0  # P on T: the overlap is T's, not the sibling's
    print("overlap self-test OK")

    # ---- save -> load round trip, and a real pass through process_case ----
    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(0)
        m = rng.random((512, 512)) > 0.5
        save_mask(m, td / "m.png")
        assert np.array_equal(load_mask(td / "m.png"), m)
        assert Image.open(td / "m.png").mode == "L" and Image.open(td / "m.png").size == (512, 512)
        small = z(); small[:64, :64] = True
        up = fit_mask(small)
        assert up.shape == (512, 512) and up[:256, :256].all() and not up[256:].any()

        T5 = np.zeros((512, 512), bool); T5[80:240, 80:240] = True
        A5 = np.zeros((512, 512), bool); A5[80:400, 300:500] = True
        B5 = np.zeros((512, 512), bool); B5[300:340, 20:60] = True
        img = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
        r_o, P_o = process_case("c-1", "full", "oracle", "", oracle_backend, img, "p",
                                T5, [A5, B5], td)
        r_s, _ = process_case("c-1", "full", "sibling", "", sibling_backend, img, "p",
                              T5, [B5, A5], td)
        assert r_o["cls"] == "target" and r_o["in_T"] == 1.0 and np.array_equal(P_o, T5)
        assert r_s["cls"] == "sibling" and r_s["in_S"] == 1.0
        assert (td / "c-1.png").exists() and r_o["n_instances"] == 3
        assert r_o["decode"] == "" and r_o["model_revision"] == ""
        fake = lambda *a: (T5, "x", 1)  # noqa: E731
        fake.decode, fake.revision = "greedy", "abc123"
        r_f, _ = process_case("c-2", "full", "sa2va", "m", fake, img, "p", T5, [A5], td)
        assert r_f["decode"] == "greedy" and r_f["model_revision"] == "abc123"
        save_overlay(img, P_o, T5, [A5, B5], td / "ov.png")
        assert Image.open(td / "ov.png").size == (512, 512)

        # ---- resume and refusal -------------------------------------------
        out = td / "r.jsonl"
        out.write_text(json.dumps(r_o) + "\n")
        assert load_done(out, "oracle", "", False) == {("c-1", "full")}
        try:
            load_done(out, "sibling", "", False)
        except SystemExit:
            pass
        else:
            raise AssertionError("append with a different backend was accepted")
        try:
            load_done(out, "oracle", "other/model", False)
        except SystemExit:
            pass
        else:
            raise AssertionError("append with a different model_path was accepted")
        assert load_done(out, "sibling", "", True) == set() and not out.exists()
        assert load_done(td / "none.jsonl", "oracle", "", False) == set()
    print("save/load, backends, resume self-test OK")

    # ---- report on a hand-built record list -------------------------------
    def rec(i, it, isb, area=0.1, n=2, prompt="p", iou=0.5):
        return {"id": i, "prompt_kind": "full", "backend": "x", "model_path": "m",
                "prompt": prompt, "in_T": it, "in_S": isb, "area_frac": area,
                "iou_target": iou, "n_instances": n}
    recs = [rec("r1", .95, 0.0), rec("r2", .6, 0.0), rec("r3", .8, .15),
            rec("r4", 0.0, .9), rec("r5", .3, .4), rec("r6", 0.0, 0.0, area=1e-4),
            rec("r7", .5, .25), rec("r8", .1, .1)]
    want = {"target": 3, "sibling": 1, "both": 2, "empty": 1, "other": 1}
    got = {k: 0 for k in want}
    for r in recs:
        got[classify(r["in_T"], r["in_S"], r["area_frac"])] += 1
    assert got == want, got
    grid = {(ti, tl): target_count(recs, ti, tl)
            for ti in (0.5, 0.7, 0.9) for tl in (0.1, 0.2, 0.3)}
    assert grid == {(.5, .1): 2, (.5, .2): 3, (.5, .3): 4, (.7, .1): 1, (.7, .2): 2,
                    (.7, .3): 2, (.9, .1): 1, (.9, .2): 1, (.9, .3): 1}, grid
    # r3 flips between t_leak 0.1 and 0.3 at t_in 0.5
    assert classify(.8, .15, .1, .5, .1) != "target" and classify(.8, .15, .1, .5, .3) == "target"
    split = outcome_split(recs, {"r1": "measurable", "r2": "measurable", "r3": "measurable",
                                 "r4": "measurable", "r5": "mis_targeted", "r6": "no_tracked_change",
                                 "r7": "off_instance_edit", "r8": "mis_targeted"})
    assert (split["a"], split["b"], split["c"], split["d"]) == (3, 1, 0, 4)
    assert abs(split["p"] - 10 / 70) < 1e-9, split
    full = {i: rec(i, it, 0.0, prompt="F") for i, it in
            zip("abcdef", (.9, .9, .9, 0.0, 0.0, .9))}
    sing = {i: rec(i, it, 0.0, prompt="F" if i == "f" else "S") for i, it in
            zip("abcdef", (.9, 0.0, 0.0, 0.0, 0.0, 0.0))}
    pr = paired_full_single(full, sing)
    assert pr == {"n_paired": 5, "dropped_identical": 1, "full_only": 2,
                  "single_only": 0, "p": 0.5}, pr
    with tempfile.TemporaryDirectory() as td:
        a, b, o = Path(td) / "a.jsonl", Path(td) / "b.jsonl", Path(td) / "o.jsonl"
        a.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        b.write_text("\n".join(json.dumps(r) for r in sing.values()) + "\n")
        o.write_text("\n".join(json.dumps({"id": r["id"], "prompt_kind": "full",
                                           "outcome": "measurable" if r["id"] < "r5" else "x"})
                               for r in recs) + "\n")
        report(a, b, o)
    lines = _summary("t", [rec(f"x{i}", .9, 0.0) for i in range(151)])
    assert any("target: 151 of 151 (100.0%)" in l for l in lines)
    assert any(">= 136 of 151: finding" in l and "this run is in" in l for l in lines)
    assert any("apply only to M = 151" in l for l in _summary("t", recs))
    print("report self-test OK")
    print("SELF-TEST OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("oracle", "sibling", "sa2va"), default="sa2va")
    p.add_argument("--model-path", default="ByteDance/Sa2VA-Qwen2_5-VL-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt-kind", choices=tuple(PROMPT_COLS), default="full")
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="default runs/gate_c/pred_masks_<backend>_<kind>.jsonl")
    p.add_argument("--out-dir", default=None,
                   help="mask PNGs; default runs/gate_c/pred_masks_<backend>_<kind>")
    p.add_argument("--save-overlays", default=None, metavar="DIR")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report", default=None, metavar="JSONL")
    p.add_argument("--report-single", default=None, metavar="JSONL")
    p.add_argument("--outcomes", default=None, metavar="JSONL",
                   help="a run_gate_c results file (e.g. klein4b_none.jsonl)")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.report:
        report(args.report, args.report_single, args.outcomes)
    else:
        run(args)


if __name__ == "__main__":
    main()
