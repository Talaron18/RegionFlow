"""
Recompute outcome_class counts for every arm under the new
OFF_INSTANCE_EDIT_OVER_TARGET threshold (GATE_C_REPORT.md SS7), CPU-only --
no images, no GPU, no API. Just reapplies the fixed outcome_class to
results files that already exist on disk.

WHY THIS MATTERS. run_gate_c.report()'s own outcome breakdown counts
no_tracked_change and off_instance_edit as SEPARATE Counter keys. SS3's
headline table's "no *tracked* change" row is the no_tracked_change count
alone -- so it's the one number in that table that can move now that the
classifier changed from ratio>10 to over_target>1. Everything else
(mis_targeted, measurable, valid, leakage figures, and SS7's own "louder
than the target" diagnostic, which already used over_target>1 unconditionally
and doesn't go through outcome_class at all) is untouched by this and does
NOT need re-running.

Run from the repo root:
    python recompute_outcomes.py
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "bench")
from run_gate_c import outcome_class  # noqa: E402

ARMS = {
    "none": "runs/gate_c/refedit_sd3.jsonl",
    "white": "runs/gate_c/ue_white.jsonl",
    "oracle": "runs/gate_c/ue_oracle.jsonl",
    "sibling": "runs/gate_c/ue_sibling.jsonl",
    "freeform": "runs/gate_c/ultraedit_freeform.jsonl",
}

print(f"{'arm':<10} {'n':>4}  {'mis_targeted':>13}  {'no_tracked_change':>18}  "
      f"{'off_instance_edit':>18}  {'measurable':>11}")
for arm, path in ARMS.items():
    p = Path(path)
    if not p.exists():
        print(f"{arm:<10} SKIP -- {path} not found")
        continue
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("prompt_kind", "full") == "full"]
    n = len(recs)
    old = collections.Counter(r.get("outcome") for r in recs)
    new = collections.Counter(outcome_class(r) for r in recs)
    print(f"{arm:<10} {n:>4}  "
          f"{new['mis_targeted']:>13}  {new['no_tracked_change']:>18}  "
          f"{new['off_instance_edit']:>18}  {new['measurable']:>11}")
    moved = sum(1 for r in recs if r.get("outcome") != outcome_class(r))
    if moved:
        print(f"           {moved} cases changed bucket vs the outcome "
              f"already stored in the file (old breakdown: {dict(old)})")
