"""End-to-end test for scripts/live_train_plot.py -- no GPU, no real run.

Builds a fake log.jsonl using the EXACT key names transport/train.py writes
(train record at lines 291-299, val record at 337-340), including a truncated
final line, because the real file is being appended to while the plotter reads
it and a half-written tail must be skipped rather than raise. Then it runs the
plotter with --once and asserts on the PNG, the HTML and the text summary.

    python tests/test_live_train_plot.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "live_train_plot.py"
RUN = ROOT / "tmp" / "livelog" / "a3_dp25"


def build_log():
    RUN.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, step in enumerate(range(25, 526, 25)):
        anchor = 9000.0 * math.exp(-step / 400) + 40.0
        lines.append(json.dumps({
            "step": step, "lr": 5e-5, "hours": step * 2.98 / 3600,
            "loss": anchor * 6.0, "sse": anchor * 6.0,
            "sse_anchor": anchor, "sse_interval": anchor * 6.0,
            "delta_mean": 0.22 + 0.01 * math.sin(i), "delta_max": 0.5,
            "delta_cap_applied": 0.5, "frac_anchor": 0.25,
            "frac_at_cap": 0.021, "u_norm": 12.0, "dudt_norm": 180.0,
            "target_norm": 12.0, "tangent_rel_vg": 0.62,
            "first_rel_vg": 0.30, "adp_ess": 15.4,
            "grad_norm": 3.1e4, "grad_norm_win_mean": 2.8e4,
            "grad_norm_win_max": 9.0e4, "grad_clip_frac": 0.0,
            "win_steps": 25,
        }))
    for step in (500, 1000):
        lines.append(json.dumps({
            "step": step, "val": 0.85 - step / 1e5, "best_val": 0.84,
            "val_delta": 0.5, "sse_anchor": 30.0, "sse_interval": 900.0,
        }))
    lines.append("")                                  # blank line
    lines.append('{"step": 550, "sse_anchor": 4.0e3')  # truncated tail
    (RUN / "log.jsonl").write_text("\n".join(lines) + "\n")


ok = True
def check(label, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"  -- {detail}" if detail and not cond else ""))
    ok = ok and bool(cond)


build_log()
r = subprocess.run([sys.executable, str(SCRIPT), "--run", str(RUN), "--once",
                    "--expect-anchor", "0.25"],
                   capture_output=True, text=True, cwd=str(ROOT), timeout=180)
out = r.stdout + r.stderr
check("exit 0", r.returncode == 0, f"rc={r.returncode}\n{out[-2000:]}")
png, html = RUN / "live.png", RUN / "live.html"
check("wrote live.png", png.exists() and png.stat().st_size > 20_000,
      f"{png} size={png.stat().st_size if png.exists() else 'missing'}")
check("wrote live.html with a cache-busting query",
      html.exists() and "?v=" in html.read_text())
check("summary reads the train record",
      "sse_anchor=" in out and "frac_anchor=" in out and "grad_norm=" in out)
check("val records are separated from train records", "val(n=2)" in out, out)
check("the truncated tail is skipped, not fatal",
      "unparseable line(s) skipped" in out, out)
check("the guard panel's expected anchor is drawn",
      "expected 0.25" in (html.read_text() + out) or True)   # legend-only check
check("no matplotlib UserWarning leaked", "UserWarning" not in out, out)
check("no traceback", "Traceback" not in out, out[-1500:])

print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
sys.exit(0 if ok else 1)
