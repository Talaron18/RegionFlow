"""
VLM-judged TARGET-REGION edit quality, VIEScore-adapted.

WHY. `instance_selectivity` measures WHERE change happened, never whether it
was any good -- the scope limit already on record: thorough destruction of
the target reads as a confidently "measurable" edit, so `destroyed` and
`wrong_edit` (quality_audit.VERDICTS) are scored as successes by every
leakage figure. `quality_audit.py` / `blind_audit.py` already measure this by
hand, on a 40-case stratified sample, at real human labor cost. This module
asks the same question with a VLM judge instead of a person, at the scale of
every edit already on disk -- 302 = 151 cases x 2 prompt kinds (full,
single) -- so it can run unattended while the curriculum arms train, and it
is RE-SCORING: the images already exist, nothing here does inference on the
editing model.

PROTOCOL, adapted from VIEScore (TIGER-AI-Lab, ACL 2024, arXiv:2312.14867),
the metric RefEdit-Bench's own paper reports against. Closing this gap makes
RegionFlow's quality number directly comparable to theirs, not merely
adjacent to it -- which is the whole point of doing this instead of adding a
fourth hand-rolled verdict scheme.

  SC (semantic consistency)  ONE score, 0-10. Read from VIEScore's own
      Text-guided Image Editing (TIE) prompt template
      (viescore/vie_prompts.py in the TIGER-AI-Lab/VIEScore repo):
        "A score from 0 to 10 will be given based on the success of the
        editing. (0 indicates that the scene in the edited image does not
        follow the editing instruction at all. 10 indicates that the scene
        in the edited image follow the editing instruction text perfectly.)"

  PQ (perceptual quality)    TWO sub-scores, naturalness and artifacts, each
      0-10, combined as PQ = sqrt(naturalness x artifacts) -- confirmed
      independently across two descriptions of the paper as a geometric
      mean, not a minimum or an average. VIEScore's own artifact rubric:
        "0 indicates that the image contains a large portion of distortion,
        or watermark, or scratches, or blurred faces, or unusual body parts"

  Overall = sqrt(SC x PQ)   the paper's headline aggregate.

CONFIRMED against the actual prompt-template file in TIGER-AI-Lab/VIEScore,
fetched directly, not against a paraphrase of the paper. NOT CONFIRMED: which
underlying VLM RefEdit-Bench's own paper used as judge, or the EXACT output
format VIEScore's own parser expects -- arxiv.org is blocked from this
sandbox's egress proxy, so the paper itself could not be read here, only the
repo's prompt file. `parse_scores` below is a lenient number-extractor, not a
byte-exact reimplementation of VIEScore's own parsing code. If exact
number-for-number comparability to a published table matters, verify the
judge model and output format against the paper directly before quoting this
against it; what's confirmed here is the SCORING PROTOCOL (which questions
are asked, on what scale, combined how), not a guarantee every implementation
detail matches.

TARGET-REGION ADAPTATION -- NOT part of the original VIEScore paper, and
state it as such wherever this number is quoted. The generic protocol scores
the WHOLE image, which conflates target quality with everything else in the
frame -- exactly the gap quality_audit.py's rubric exists to close for a
human judge (a destroyed target next to an untouched background can still
look globally "natural"). This module reuses quality_audit.side_by_side's
own convention -- yellow outline on the target, red on siblings, same source
| edited panel a human judge sees -- and the judge prompt explicitly
restricts SC and PQ to the region outlined in yellow.

JUDGE IS PLUGGABLE. The interface is one function:
    judge_fn(panel_image: PIL.Image, instruction: str) -> dict(sc: float,
        naturalness: float, artifacts: float)
`claude_judge` is one implementation, calling this session's own model
family via the Anthropic API -- explicitly NOT a claim that it reproduces
whatever model RefEdit-Bench's paper used. Point --judge-module at your own
callable (module:function) for GPT-4V or anything else; self_test exercises
the whole pipeline with a scripted fake judge and never touches the network.

USAGE
    python bench/edit_quality_vlm.py --self-test
    ANTHROPIC_API_KEY=... python bench/edit_quality_vlm.py --build \\
        --results runs/gate_c/refedit_sd3.jsonl --edits runs/gate_c/edits \\
        --out runs/gate_c/vlm_quality.jsonl
    python bench/edit_quality_vlm.py --score runs/gate_c/vlm_quality.jsonl
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from quality_audit import masks_for_case, side_by_side
from refedit_siblings import bench_image, load_bench, load_coco
from run_gate_c import outcome_class

PROMPT_COLS = {"full": "editing_instruction", "single": "edit_instruction_single"}

# VIEScore's own TIE-task SC prompt, quoted verbatim from
# viescore/vie_prompts.py (TIGER-AI-Lab/VIEScore). The instruction and the
# target-region restriction are this project's additions, appended rather
# than interleaved, so the VIEScore text itself stays byte-identical to the
# source and any future diff against it is a diff, not a rewrite.
SC_PROMPT = """You are given a source image and an edited image, side by \
side (source on the left, edited on the right). The instance the editing \
instruction refers to is outlined in YELLOW on both panels; any other \
instances of the same category are outlined in RED. Judge the edit ONLY \
within the yellow-outlined region -- ignore everything outside it, \
including the red-outlined siblings and the rest of the frame.

The editing instruction was: "{instruction}"

A score from 0 to 10 will be given based on the success of the editing. \
(0 indicates that the scene in the edited image does not follow the \
editing instruction at all. 10 indicates that the scene in the edited \
image follow the editing instruction text perfectly.) Give a float score \
strictly following this format: "[[score]]", e.g., "[[5.0]]"."""

# VIEScore's own PQ prompt, same sourcing. The target-region restriction is
# the same addition as SC_PROMPT's.
PQ_PROMPT = """You are given an edited image. The instance the editing \
instruction targeted is outlined in YELLOW. Judge quality ONLY within the \
yellow-outlined region.

A score from 0 to 10 will be given based on the naturalness of the image \
within that region. (0 indicates the region looks totally unnatural or is \
completely a work of art. 10 indicates the region looks completely \
natural and photorealistic.)
A second score from 0 to 10 will rate the image artifacts within that \
region. (0 indicates that the image contains a large portion of \
distortion, or watermark, or scratches, or blurred faces, or unusual body \
parts. 10 indicates the image has no artifacts.)
Put the score in this format: "[[naturalness, artifacts]]", e.g., \
"[[9.0, 8.0]]"."""

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_scores(text: str, n: int) -> list[float]:
    """Extract the first `n` numbers from a judge's free-text response.

    LENIENT ON PURPOSE, and that is a deviation stated up front: VIEScore's
    own repo has a specific "||V^=^V||"-delimited output format this module
    does not reproduce (unconfirmed from here -- see module docstring). A
    judge asked for "[[7.0]]" or "[[9.0, 8.0]]" reliably puts the numbers in
    that order somewhere in its response; grabbing the first `n` numbers by
    regex is robust to minor formatting drift (extra whitespace, a stray
    sentence before the brackets) at the cost of being fooled by a judge that
    states an unrelated number FIRST, e.g. "on a scale of 0 to 10, ...". The
    prompts above are written to minimise that by asking for the bracketed
    format explicitly; self_test checks the common cases, not every one.
    """
    nums = [float(x) for x in _SCORE_RE.findall(text)]
    if len(nums) < n:
        raise ValueError(f"expected >= {n} numbers in judge output, got "
                         f"{len(nums)}: {text!r}")
    return nums[:n]


def judge_case(judge_fn, src: Image.Image, edited: Image.Image,
               tmask: np.ndarray, smasks: list, instruction: str) -> dict:
    """Score one edit. Returns sc, naturalness, artifacts, pq, overall.

    ONE panel, ONE outline convention, for both calls -- side_by_side is
    quality_audit's own function, imported not restated, so a human re-check
    of a VLM score sees exactly the image the judge saw.
    """
    panel = side_by_side(src, edited, tmask, smasks)
    sc_text = judge_fn(panel, SC_PROMPT.format(instruction=instruction))
    pq_text = judge_fn(panel, PQ_PROMPT)
    sc = float(np.clip(parse_scores(sc_text, 1)[0], 0.0, 10.0))
    nat, art = (float(np.clip(x, 0.0, 10.0))
               for x in parse_scores(pq_text, 2))
    pq = math.sqrt(max(nat, 0.0) * max(art, 0.0))
    overall = math.sqrt(max(sc, 0.0) * pq)
    return {"sc": sc, "pq_naturalness": nat, "pq_artifacts": art,
           "pq": pq, "overall": overall}


# ------------------------------------------------------------------ judges --

def claude_judge(model: str = "claude-opus-5"):
    """One reference implementation, using this session's own model family.

    NOT a claim of parity with whatever judge RefEdit-Bench's paper used --
    see the module docstring. Returns a judge_fn(panel, prompt) -> str;
    imports anthropic lazily, matching this project's own convention for
    heavy/optional dependencies (torch is imported inside functions
    throughout transport/ and scripts/, not at module scope).
    """
    import base64
    import io

    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic()

    def fn(panel: Image.Image, prompt: str) -> str:
        buf = io.BytesIO()
        panel.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = client.messages.create(
            model=model, max_tokens=100,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png",
                                             "data": b64}},
                {"type": "text", "text": prompt},
            ]}])
        return "".join(b.text for b in msg.content if b.type == "text")

    return fn


def load_judge(spec: str):
    """--judge-module MODULE:FUNC, called with no args, returning judge_fn.

    Lets --judge-module bench.edit_quality_vlm:claude_judge select the
    built-in, or point anywhere else (module:function) for a GPT-4V or
    Gemini implementation this file does not need to know about.
    """
    mod_name, _, fn_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)()


# ------------------------------------------------------------------- build --

def build(args) -> None:
    recs = [json.loads(l) for l in Path(args.results).read_text().splitlines()
            if l.strip()]
    for r in recs:
        r.setdefault("outcome", outcome_class(r))
    recs = [r for r in recs if r["prompt_kind"] in args.prompt_kinds]

    cases = {c["id"]: c for c in
             json.loads(Path(args.cases).read_text())["cases"]}
    coco_index, _c, _u = load_coco(Path(args.coco_annotations),
                                   tuple(args.coco_splits.split(",")),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # RESUMABLE, same pattern as run_gate_c.run: an interrupted 302-case VLM
    # run should not silently re-pay for cases already scored.
    done = set()
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["id"], d["prompt_kind"]))
        print(f"resuming: {len(done)} already scored in {out_path}")

    judge_fn = load_judge(args.judge_module)
    todo = [r for r in recs if r["id"] in cases
           and (r["id"], r["prompt_kind"]) not in done]
    print(f"{len(todo)} to score ({len(recs)} total, {len(done)} done)")

    t0, n = time.time(), 0
    with out_path.open("a") as fh:
        for r in todo:
            case = cases[r["id"]]
            idx = int(r["id"].rsplit("-", 1)[1])
            row, rec = rows.get(idx), coco_index.get(case["coco_image_id"])
            edit_path = Path(args.edits) / f"{r['id']}_{r['prompt_kind']}.png"
            if row is None or rec is None or not edit_path.exists():
                continue
            tmask, smasks = masks_for_case(case, rec)
            if tmask is None:
                continue

            src = bench_image(row["input_image"])
            edited = Image.open(edit_path)
            try:
                q = judge_case(judge_fn, src, edited, tmask, smasks,
                              row["editing_instruction"])
            except Exception as exc:  # noqa: BLE001
                print(f"  {r['id']} ({r['prompt_kind']}): judge failed, "
                      f"skipped -- {type(exc).__name__}: {exc}")
                continue

            out = {"id": r["id"], "prompt_kind": r["prompt_kind"],
                  "outcome": r["outcome"], "valid": r["valid"],
                  "target_change": r["target_change"],
                  "leakage_ratio": r["leakage_ratio"],
                  "difficulty": case["difficulty"], "category": case["category"],
                  **{k: round(v, 3) for k, v in q.items()}}
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / n:.1f}s/case  "
                      f"eta {(len(todo) - n) * el / n / 60:.1f}m")

    print(f"\nwrote {n} new scores -> {out_path}")


# ------------------------------------------------------------------- score --

def score(path: Path) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        print("no scores")
        return
    print(f"{len(recs)} VLM-judged edits\n")

    for kind in sorted({r["prompt_kind"] for r in recs}):
        rs = [r for r in recs if r["prompt_kind"] == kind]
        print(f"prompt = {kind}  (n={len(rs)})")
        for key in ("sc", "pq", "overall"):
            vals = np.array([r[key] for r in rs])
            print(f"  {key:<8} mean {vals.mean():.2f}  median "
                  f"{np.median(vals):.2f}  p10 {np.percentile(vals, 10):.2f}")

        print("\n  quality BY the metric's outcome class (median overall)")
        for oc in ("measurable", "mis_targeted", "no_tracked_change",
                  "off_instance_edit"):
            sub = [r for r in rs if r["outcome"] == oc]
            if not sub:
                continue
            vals = np.array([r["overall"] for r in sub])
            print(f"    {oc:<18} n={len(sub):<4} median overall "
                  f"{np.median(vals):.2f}")

        # THE CONDITIONAL every leakage figure inherits, now at n=up-to-151
        # rather than n=40: what does the VLM judge think of the cases that
        # actually produce the reported leakage numbers?
        contrib = [r for r in rs if r["valid"] and r["outcome"] == "measurable"]
        if contrib:
            vals = np.array([r["overall"] for r in contrib])
            low = int((vals < 5.0).sum())
            print(f"\n  CASES THAT PRODUCE THE REPORTED LEAKAGE NUMBERS "
                  f"(valid+measurable): {len(contrib)}")
            print(f"    median overall {np.median(vals):.2f}   "
                  f"overall < 5.0 (VIEScore's own rough failure line): "
                  f"{low}/{len(contrib)} = {low / len(contrib):.0%}")
            print("    -> every leakage figure is conditional on this "
                  "distribution. Quote it alongside them.")
        print()


# --------------------------------------------------------------- self-test --

def _fake_judge(responses: dict):
    """A scripted judge_fn for self_test: keyed on whether the prompt asks
    for SC (one number) or PQ (two), never touches the network."""
    def fn(panel, prompt):
        return responses["pq"] if "naturalness" in prompt else responses["sc"]
    return fn


def self_test() -> None:
    assert parse_scores("[[7.0]]", 1) == [7.0]
    assert parse_scores("Looking at the region, I'd say [[9.0, 8.0]]", 2) \
        == [9.0, 8.0]
    assert parse_scores("score: [[ 3.5 , 4 ]]", 2) == [3.5, 4.0]
    try:
        parse_scores("no numbers here", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("parse_scores must reject text with too few numbers")

    src = Image.fromarray(np.full((256, 256, 3), 90, np.uint8))
    edited = Image.fromarray(np.full((256, 256, 3), 90, np.uint8))
    t = np.zeros((256, 256), bool); t[50:150, 50:150] = True
    s = np.zeros((256, 256), bool); s[180:220, 180:220] = True

    judge = _fake_judge({"sc": "[[8.0]]", "pq": "[[9.0, 6.0]]"})
    q = judge_case(judge, src, edited, t, [s], "make it blue")
    assert q["sc"] == 8.0
    assert abs(q["pq"] - math.sqrt(9.0 * 6.0)) < 1e-9, q["pq"]
    assert abs(q["overall"] - math.sqrt(8.0 * q["pq"])) < 1e-9, q["overall"]

    # Clamping: a judge that ignores the 0-10 scale must not blow up Overall.
    judge_ood = _fake_judge({"sc": "[[15.0]]", "pq": "[[-2.0, 20.0]]"})
    q2 = judge_case(judge_ood, src, edited, t, [s], "x")
    assert q2["sc"] == 10.0 and q2["pq_naturalness"] == 0.0 \
        and q2["pq_artifacts"] == 10.0
    assert q2["pq"] == 0.0 and q2["overall"] == 0.0

    # SC_PROMPT / PQ_PROMPT must actually carry the instruction and stay
    # distinguishable by _fake_judge's own routing key ("naturalness").
    assert "make it blue" in SC_PROMPT.format(instruction="make it blue")
    assert "naturalness" in PQ_PROMPT and "naturalness" not in SC_PROMPT

    print("edit_quality_vlm self-test OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="runs/gate_c/refedit_sd3.jsonl")
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--edits", default="runs/gate_c/edits")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--prompt-kinds", nargs="+", default=["full", "single"],
                   choices=("full", "single"))
    p.add_argument("--judge-module", default="edit_quality_vlm:claude_judge",
                   help="MODULE:FUNC, called with no args, returning "
                        "judge_fn(panel, prompt) -> str")
    p.add_argument("--out", default="runs/gate_c/vlm_quality.jsonl")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--score", default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
    elif args.score:
        score(Path(args.score))
    elif args.build:
        build(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
