"""
Diagnose the 2 residual GLM-5.3-Flash high scorers from the full-40 `none`
calibration run (refedit-134: overall 9.22, refedit-83: overall 9.08).

Same method used for refedit-117/160/170/29 and refedit-119 earlier in this
project: rebuild the EXACT panel the judge saw (crop_to_target + side_by_side,
not the raw uncropped images), call the judge_fn directly instead of going
through judge_case so the RAW SC/PQ text survives (judge_case throws it away
after parsing), and put that next to the human blind verdict + notes for the
same panel. Reading the raw text alongside the cropped image is what found
refedit-119's outline-override and refedit-172's cross-object-context
mechanism -- the parsed score alone never explains why it's wrong.

Run from bench/:
    python diagnose_high_scorers.py
Needs bench/.env with SILICONFLOW_API_KEY, and the usual local paths
(runs/gate_c/, /data/datasets/coco/, blind_key.json, blind_ledger.json).
Saves the two cropped panels next to this script so you can look at exactly
what the judge looked at.
"""
import json
from pathlib import Path

from PIL import Image

from edit_quality_vlm import SC_PROMPT, PQ_PROMPT, crop_to_target, siliconflow_judge
from quality_audit import load_bench, load_coco, masks_for_case, side_by_side
from refedit_siblings import bench_image

IDS = [ "refedit-83"]

CASES = "bench/cases_refedit_final.json"
BENCH = "/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet"
COCO_ANN = "/data/datasets/coco/annotations"
COCO_CACHE = "/data/datasets/coco/desc"
NONE_RESULTS = "runs/gate_c/refedit_sd3.jsonl"
NONE_EDITS = Path("runs/gate_c/edits")
PROMPT_KIND = "full"

BLIND_KEY = "runs/gate_c/blind_key.json"
BLIND_LEDGER = "runs/gate_c/blind_ledger.json"


def main() -> None:
    cases = {c["id"]: c for c in json.loads(Path(CASES).read_text())["cases"]}
    coco_index, _cats, _urls = load_coco(Path(COCO_ANN), ("train2014",),
                                         cache=Path(COCO_CACHE))
    rows = {r["index"]: r for r in load_bench(BENCH)}
    recs = {json.loads(l)["id"]: json.loads(l)
            for l in Path(NONE_RESULTS).read_text().splitlines() if l.strip()}

    key = {e["id"]: e for e in json.loads(Path(BLIND_KEY).read_text())
           if e["arm"] == "none"}
    ledger = {e["panel"]: e for e in json.loads(Path(BLIND_LEDGER).read_text())}

    judge_fn = siliconflow_judge()

    for cid in IDS:
        print(f"\n{'=' * 70}\n{cid}\n{'=' * 70}")
        case = cases[cid]
        idx = int(cid.rsplit("-", 1)[1])
        row, rec = rows[idx], coco_index[case["coco_image_id"]]
        tmask, smasks = masks_for_case(case, rec)
        src = bench_image(row["input_image"])
        edited = Image.open(NONE_EDITS / f"{cid}_{PROMPT_KIND}.png")
        instruction = row["editing_instruction"]

        k = key.get(cid)
        if k is not None:
            l = ledger.get(k["panel"], {})
            print(f"human verdict: {l.get('verdict')!r}   notes: {l.get('notes')!r}")
            print(f"metric outcome: {k.get('outcome')!r}  valid: {k.get('valid')}  "
                  f"leakage_ratio: {k.get('leakage_ratio')}")
        else:
            print("not in blind_key.json (not part of the 40-case audit) -- "
                  "no human verdict to compare against.")
        print(f"instruction: {instruction!r}")

        c_src, c_edited, c_tmask, c_smasks = crop_to_target(src, edited, tmask, smasks)
        panel = side_by_side(c_src, c_edited, c_tmask, c_smasks)
        out = Path(__file__).parent / f"{cid}_panel.png"
        panel.save(out)
        print(f"panel saved -> {out}")

        sc_text = judge_fn(panel, SC_PROMPT.format(instruction=instruction))
        pq_text = judge_fn(panel, PQ_PROMPT)
        print(f"\n--- raw SC response ---\n{sc_text}")
        print(f"\n--- raw PQ response ---\n{pq_text}")


if __name__ == "__main__":
    main()
