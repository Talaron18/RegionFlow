"""Live training view: re-render a PNG (plus a self-refreshing HTML) from log.jsonl.

WHERE THE DATA IS. `transport/train.py:196` writes `<out>/log.jsonl`; a training
record lands every `--log-every` steps (default 25) and a validation record lands
every `--val-every` steps (default 500) **in the same file**, distinguishable
because only the val record has a `val` key.

WINDOW MEANS BY DEFAULT, and that is not cosmetic. The per-step value of every
loss statistic here is dominated by sampling noise: with `--batch 16` and
`--data-proportion 0.25` the anchor fraction alone has a per-step sd of
sqrt(16*0.25*0.75)/16 = 0.108, and the first real render showed frac_anchor
swinging 0.06-0.56 with no readable trend in `sse_anchor` / `sse_interval`
either. The trainer already writes window means (`win_<key>`, averaged over
every step since the last log) whose sd is 25x smaller. This plots those by
default; `--raw` shows the per-step values instead.

WHY THESE PANELS, AND NOT A LOSS CURVE. `START_HERE.md` is explicit that `loss`
is not a progress signal under this objective -- with `--norm-p 0` it equals
`sse` and sits near 2e4 -- and that `sse_anchor` / `sse_interval` are what to
read. `best_val` must not adjudicate arms, so it gets a panel of its own with
that warning on it rather than a headline. And `frac_anchor` gets a panel because
the repo has already been burned by a run whose config never reached the loss
while looking completely normal (the P10 lesson); its expected value is drawn as
a dashed line.

COLOR = quantity, LINE STYLE = run. Keying the color to the run (the first
version) made both curves in a panel the same color, so the legend was useless
and `sse_anchor` was indistinguishable from `sse_interval`.

    python scripts/live_train_plot.py --run runs/a3_dp25 --loop --expect-anchor 0.25
    python scripts/live_train_plot.py --run runs/a3_dp25 runs/a4_md75 --once
    python scripts/live_train_plot.py --run runs/a3_dp25 --loop --raw

Viewing: it writes `<run>/live.png` and `<run>/live.html`; open the HTML in the
box's JupyterLab and it reloads itself, or scp the PNG. It also prints a text
summary per refresh, so an SSH terminal alone is enough.

Refresh policy: re-render when EITHER `--every` seconds or `--every-steps` steps
have passed since the last render. Polling is every 5 s and is cheap (CPU only,
never touches the GPU, so it can run alongside training).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                    # headless, matching scripts/gate_a_transport.py
import matplotlib.pyplot as plt          # noqa: E402

# colour = quantity (inside a panel), line style = run
KEY_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
RUN_STYLES = ["-", "--", ":"]

# (axes index, panel title, [keys], log-y). PANEL INDEXES ARE EXPLICIT: the
# first version reused axes[2] for the guard after the loop had already drawn on
# it, which both lost a panel and put set_ylim(0, 1) on a log-scaled axis
# (matplotlib: "Attempt to set non-positive ylim on a log-scaled axis").
# 0 signals | 1 Delta | 2 frac_anchor GUARD | 3 grad norm | 4 target quality
# | 5 val. Every axis is drawn exactly once.
PANELS = [
    (0, "the two signals  (NOT `loss`)", ["sse_anchor", "sse_interval"], True),
    (1, "Delta schedule", ["delta_mean", "delta_max", "delta_cap_applied"],
     False),
    (3, "gradient norm / win_max  (grad_clip=0 => never clipped)",
     ["grad_norm", "grad_norm_win_max"], True),
    (4, "target quality: how far the tangent / first term sits from v_g",
     ["tangent_rel_vg", "first_rel_vg"], False),
]
# drawn outside the loop: the guard carries a reference line, val carries a
# caveat in its title and must be linear
GUARD_IDX, VAL_IDX, GUARD_KEY = 2, 5, "frac_anchor"


def _series(recs, key):
    xs, ys = [], []
    for r in recs:
        v = r.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            continue
        xs.append(r.get("step", len(xs)))
        ys.append(f)
    return xs, ys


def series(recs, key, prefer_window=False):
    """-> (resolved_key, xs, ys). Prefers the trainer's `win_<key>` window mean.

    The window mean is what makes a trend visible at all; the per-step value is
    mostly sampling noise (see the module docstring).
    """
    for k in ([f"win_{key}", key] if prefer_window else [key]):
        xs, ys = _series(recs, k)
        if xs:
            return k, xs, ys
    return key, [], []


def read_log(path: Path):
    """-> (train, val, n_bad, n_dup) for ONE log.jsonl.

    Two things are normal while reading this file live, and both must be
    tolerated rather than raised:
      * a HALF-WRITTEN last line (the run is appending).
      * DUPLICATE steps, because a resumed run APPENDS to the same log: the run
        restarts from `ckpt_latest.pt` (written every --ckpt-every steps), so
        `step` goes backwards at the restart (e.g. up to 1350, then 1200 again).
        Dedup by step with the LAST record winning, so the plot stays monotone
        instead of zig-zagging.
    """
    tr, va, bad, dup = [], [], 0, 0
    if not path.exists():
        return tr, va, bad, dup
    seen_t, seen_v = {}, {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        step = rec.get("step")
        bucket = seen_v if "val" in rec else seen_t
        if step in bucket:
            dup += 1
        bucket[step] = rec
    for bucket, dst in ((seen_t, tr), (seen_v, va)):
        dst.extend(bucket[k] for k in sorted(
            bucket, key=lambda s: (s is None, s)))
    return tr, va, bad, dup


def render(runs, png: Path, html: Path, expect_anchor, nrender, prefer_window,
           val_every):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.ravel()

    for i, title, keys, logy in PANELS:
        ax = axes[i]
        for j, run in enumerate(runs):
            for ik, key in enumerate(keys):
                k, xs, ys = series(run["train"], key, prefer_window)
                if not xs:
                    continue
                if logy:
                    keep = [(x, y) for x, y in zip(xs, ys) if y > 0]
                    xs, ys = [x for x, _ in keep], [y for _, y in keep]
                if xs:
                    ax.plot(xs, ys, RUN_STYLES[j % len(RUN_STYLES)],
                            color=KEY_COLORS[ik % len(KEY_COLORS)], lw=1.3,
                            label=f"{k} [{run['label']}]")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.5, ncol=1)
        ax.tick_params(labelsize=7)

    # the frac_anchor GUARD, with the expected value drawn behind the data
    ax = axes[GUARD_IDX]
    for j, run in enumerate(runs):
        k, xs, ys = series(run["train"], GUARD_KEY, prefer_window)
        if xs:
            # markers + higher zorder: with the data landing exactly on the
            # expected line (the GOOD outcome) a thin line is invisible and
            # reads as "missing data".
            ax.plot(xs, ys, ".", color=KEY_COLORS[j % len(KEY_COLORS)], ms=4,
                    zorder=3, label=f"{k} [{run['label']}]")
    if expect_anchor is not None:
        ax.axhline(expect_anchor, color="k", ls="--", lw=1.0, alpha=0.8,
                   zorder=2, label=f"expected {expect_anchor}")
    ax.set_title("frac_anchor  GUARD: a config that never reached the loss "
                 "looks normal otherwise", fontsize=10)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # gradient norms and val are inside PANELS / the separate blocks above; the
    # only thing left here is val, at its own explicit index
    ax = axes[VAL_IDX]
    for j, run in enumerate(runs):
        _, xs, ys = series(run["val"], "val")
        if xs:
            ax.plot(xs, ys, "o-", color=KEY_COLORS[j % len(KEY_COLORS)], ms=3,
                    lw=1.0, label=f"val [{run['label']}]")
        _, bx, by = series(run["val"], "best_val")
        if bx:
            ax.plot(bx, by, ":", color=KEY_COLORS[j % len(KEY_COLORS)], lw=1.0,
                    label=f"best_val [{run['label']}]")
    # WHAT `val` IS, and the decomposition that matters. gate_b_meanflow.py:603-615
    # returns out["interval"], i.e. `val` IS the interval SSE at a FIXED
    # (t, r) = (0.5 + val_delta/2, 0.5 - val_delta/2) on ONE fixed batch, with a
    # fixed eps. The record also carries val_anchor and val_ratio (interval /
    # anchor) precisely because val_anchor is the regulariser check and the
    # denominator of the number that answers Gate B. So when `val` rises, this
    # panel says WHICH term rose -- an anchor that is fine with an interval term
    # degrading is a different story from both rising together.
    # `val` and `val_anchor` differ by ~360x (94k vs 230 on the arm-2 run), so
    # they CANNOT share a linear axis: the anchor gets squashed onto zero and
    # reads as missing data. `val` gets the left axis alone; val_anchor and
    # val_ratio share the right one, where they are the same order (O(100-400)).
    ax = axes[VAL_IDX]
    twin = ax.twinx()
    for j, run in enumerate(runs):
        ls = RUN_STYLES[j % len(RUN_STYLES)]
        for ik, (key, where) in enumerate((("val", ax),
                                           ("val_anchor", twin),
                                           ("val_ratio", twin))):
            k, xs, ys = series(run["val"], key)
            if xs:
                where.plot(xs, ys, ls, color=KEY_COLORS[ik % len(KEY_COLORS)],
                           marker="o", ms=2.5, lw=1.1,
                           label=f"{k} [{run['label']}]")
    ax.set_ylabel("val = interval sse (left)", fontsize=7)
    twin.set_ylabel("val_anchor and val_ratio (right)", fontsize=7)
    twin.tick_params(labelsize=7)
    last_v, vd = None, None
    for run in runs:
        _, vx, _ = series(run["val"], "val")
        if vx:
            last_v = max(last_v or 0, max(vx))
        if run["val"]:
            vd = run["val"][-1].get("val_delta", vd)
    cadence = ""
    if val_every and last_v:
        cadence = (f"   [val_every={val_every}: last @ {int(last_v)}, next @ "
                   f"{int((last_v // val_every + 1) * val_every)}]")
    ax.set_title("validation: `val` = INTERVAL sse at fixed t,r ("
                 f"d={vd}) on one fixed batch of 16 -- so it is noisy AND "
                 "self-referential.\nDo NOT adjudicate arms on it."
                 + cadence, fontsize=9)
    ax.set_xlabel("step", fontsize=8)
    ax.grid(alpha=0.25)
    # merge the twin axis's handles, or the val_ratio entries vanish from the
    # legend (they live on the other axes object)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, fontsize=6)
    ax.tick_params(labelsize=7)

    steps = []
    for run in runs:
        last = run["train"][-1]["step"] if run["train"] else 0
        steps.append(f"{run['label']}: step {last}/{run['steps']}")
    fig.suptitle("RegionFlow live  |  " + "   ".join(steps)
                 + f"  |  {'window means' if prefer_window else 'PER-STEP'}"
                 + f"  |  rendered {time.strftime('%H:%M:%S')}", fontsize=11)
    fig.savefig(png, dpi=110)
    plt.close(fig)

    # Self-refreshing HTML. The ?v= query defeats the browser's PNG cache, which
    # a bare <meta refresh> does not.
    html.write_text(
        "<html><head><meta http-equiv='refresh' content='30'></head>"
        "<body style='margin:0;background:#111'>"
        f"<img src='{png.name}?v={nrender}' style='width:100%'></body></html>")


def summary(runs, prefer_window, val_every):
    for run in runs:
        tr = run["train"]
        if not tr:
            print(f"  {run['label']}: no training records yet ({run['log']})")
            continue
        last = tr[-1]
        parts = []
        for key in ("sse_anchor", "sse_interval", "frac_anchor", "delta_mean",
                    "delta_max", "frac_at_cap", "grad_norm"):
            k, xs, ys = series(tr, key, prefer_window)
            if ys:
                parts.append(f"{k}={ys[-1]:.4g}")
        # the per-step spread of the guard quantity, which is what makes the
        # window mean necessary rather than cosmetic (sd = 0.108 at batch 16)
        _, rxs, rys = series(tr, GUARD_KEY, False)
        if prefer_window and rys:
            parts.append(f"[raw {GUARD_KEY} {min(rys):.2f}-{max(rys):.2f}]")
        print(f"  {run['label']}  step {last.get('step')}/{run['steps']}  "
              f"{'%.2f' % (last.get('hours') or 0)}h elapsed")
        print("    " + "  ".join(parts))
        vals = run["val"]
        if vals:
            lastv = vals[-1]
            last_vs = lastv["step"]
            nxt = (int((last_vs // val_every + 1) * val_every)
                   if val_every else None)
            extra = ""
            if "val_anchor" in lastv:
                extra = (f"  [anchor {lastv['val_anchor']:.4g}"
                         + (f"  ratio {lastv['val_ratio']:.2f}"
                            if "val_ratio" in lastv else "") + "]")
            print(f"    val(n={len(vals)}) {lastv['val']:.5f} "
                  f"best {lastv.get('best_val', float('nan')):.5f}"
                  f"   last @ {last_vs}"
                  + (f", next @ {nxt} (val_every={val_every})" if nxt else "")
                  + extra)
        else:
            print("    val: none yet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True,
                    help="run directories containing log.jsonl")
    ap.add_argument("--steps", type=int, default=5000,
                    help="total steps, for the title (display only)")
    ap.add_argument("--every", type=float, default=30.0,
                    help="seconds between renders")
    ap.add_argument("--every-steps", type=int, default=100,
                    help="ALSO re-render once this many new steps have arrived")
    ap.add_argument("--val-every", type=int, default=500,
                    help="the run's --val-every, so the val panel can say when "
                         "the next validation point is due (display only)")
    ap.add_argument("--expect-anchor", type=float, default=None,
                    help="draw a dashed line at the expected frac_anchor, e.g. "
                         "0.25 for --data-proportion 0.25, 0.75 for the default")
    ap.add_argument("--raw", action="store_true",
                    help="plot the per-step values instead of the win_* window "
                         "means (noisier; see the module docstring)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if not (args.once or args.loop):
        args.once = True
    prefer_window = not args.raw

    runs = []
    for spec in args.run:
        d = Path(spec)
        if not d.is_dir():
            print(f"[warn] {d} is not a directory; skipping", file=sys.stderr)
            continue
        runs.append(dict(label=d.name, log=d / "log.jsonl", train=[], val=[],
                         steps=args.steps, bad=0))
    if not runs:
        return 2

    last_render, last_step, nrender = 0.0, -1, 0
    try:
        while True:
            now = time.time()
            for run in runs:
                run["train"], run["val"], run["bad"], run["dup"] = \
                    read_log(run["log"])
            top = max([r["train"][-1].get("step", -1) if r["train"] else -1
                       for r in runs] or [-1])
            due_time = now - last_render >= args.every
            due_steps = top - last_step >= args.every_steps
            if args.once or due_time or due_steps:
                nrender += 1
                missing = [r["label"] for r in runs if not r["train"]]
                print(f"\n[render {nrender}] {time.strftime('%H:%M:%S')}"
                      + (f"   waiting on: {', '.join(missing)}" if missing else ""))
                summary(runs, prefer_window, args.val_every)
                for run in runs:
                    if run["bad"]:
                        print(f"    [note] {run['label']}: {run['bad']} "
                              f"unparseable line(s) skipped (a half-written tail "
                              f"is normal while the run appends)")
                    if run.get("dup"):
                        print(f"    [note] {run['label']}: {run['dup']} "
                              f"duplicated step(s) collapsed -- that is a "
                              f"resume, which appends to the same log after "
                              f"restarting from ckpt_latest.pt")
                if any(r["train"] for r in runs):
                    d = next(r["log"] for r in runs if r["train"]).parent
                    render(runs, d / "live.png", d / "live.html",
                           args.expect_anchor, nrender, prefer_window,
                           args.val_every)
                    print(f"    wrote {d / 'live.png'} and {d / 'live.html'} "
                          f"(open the html in JupyterLab -- it self-refreshes)")
                last_render, last_step = now, top
            if args.once:
                return 0
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())