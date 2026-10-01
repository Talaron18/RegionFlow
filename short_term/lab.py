#!/usr/bin/env python3
"""
Run line 1 from inside a JupyterLab cell, on a box whose only door is the
notebook server.

WHY THIS EXISTS. A remote box that exposes nothing but JupyterLab cannot reach
`http://127.0.0.1:8765` from the user's browser, and `--host 0.0.0.0` alone does
not publish the port. There are two ways through, and this module gives both:

    lab.embed("https://host:8443/jupyter/lab/tree/RegionFlow/short_term")

starts the UI server in a DAEMON THREAD (no terminal, no second process, no
`nohup`) and renders it in an IFrame through the notebook server's own port
proxy -- Jupyter serves any port at `<base>/proxy/<port>/`, and `lab.proxy_url`
derives that base from the address bar you paste. One cell, and the UI appears
underneath it.

    lab.run_line1(source=..., patch=..., rect=..., instruction=...)

is the door that needs no proxy at all. It builds the same payload the browser
would have sent, calls the same `handle_run`, and shows the result inline. Use
it if the proxy 403s (usually Jupyter's `allow_remote_access`, or a platform
firewall) -- and note you can upload source/patch in JupyterLab's own file
browser, so no browser-side file picker is needed either. What you give up is
dragging the patch; you type its rectangle instead, in 512-space.

BOTH PATHS CALL THE SAME `handle_run`, so neither can drift from the other, and
both are covered by `app.py --self-test`.

    python short_term/lab.py --self-test      # CPU, no model, no notebook
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from short_term.app import (  # noqa: E402
    DEFAULTS,
    array_to_b64_png,
    b64_png_to_array,
    handle_run,
    proxy_url,
)
from short_term.backends import load_image, load_mask  # noqa: E402

__all__ = ["proxy_url", "serve_in_thread", "embed", "run_line1", "ask_lab_url"]

_SERVERS: dict = {}


def ask_lab_url() -> str:
    """Best effort at the current page URL from inside a notebook.

    Jupyter does not export the browser's address bar to the kernel, so this
    usually cannot know it -- and a wrong base silently produces a dead link,
    which is worse than asking. It returns "" and the caller should ask the
    user to paste theirs.
    """
    import os
    for var in ("JUPYTERHUB_SERVICE_PREFIX", "JUPYTER_BASE_URL",
                "JPY_SESSION_NAME"):
        v = os.environ.get(var)
        if v and var == "JUPYTERHUB_SERVICE_PREFIX":
            # e.g. '/user/alice/' -- a real base, but no scheme or host, so it
            # is still only useful as a hint.
            return v
    return ""


def serve_in_thread(port: int = 8765, host: str = "0.0.0.0",
                    out_root: str = "runs/st/ui", mock: bool = False):
    """Start the UI in a daemon thread. Idempotent per port.

    A thread rather than a subprocess so the notebook owns the lifetime: when
    the kernel dies the server goes with it, which is what a temporary tool
    should do.
    """
    from types import SimpleNamespace
    if port in _SERVERS:
        return _SERVERS[port][1]
    cfg = SimpleNamespace(out_root=out_root, host=host, port=port,
                          mock=mock, lab_url=None, open=False,
                          published_ports=(6006, 6008))
    from short_term.app import serve
    t = threading.Thread(target=serve, args=(cfg,), daemon=True,
                         name=f"short_term_ui_{port}")
    t.start()
    url = f"http://127.0.0.1:{port}/"
    _SERVERS[port] = (cfg, url)
    return url


def embed(lab_url: str, port: int = 8765, height: int = 760,
          mock: bool = False, out_root: str = "runs/st/ui"):
    """Start the UI and render it inline through the notebook server's proxy.

    `lab_url` is your JupyterLab address bar. Returns the URL it embedded, so it
    can also be opened in its own tab -- which is usually better, because the
    IFrame shares the notebook's page and a canvas drag behaves oddly at the
    edges of a scrollable output cell.
    """
    from IPython.display import IFrame, display

    serve_in_thread(port=port, mock=mock, out_root=out_root)
    url = proxy_url(lab_url, port)
    if not url.startswith("http"):
        raise SystemExit(
            f"could not derive a URL from {lab_url!r}. Paste the whole address "
            "bar including https:// and the host.")
    print(f"embedding {url}")
    print("  if the frame below stays blank, open that URL in its own tab; and "
          "if THAT 403s,")
    print("  the notebook server is refusing to proxy (allow_remote_access) -- "
          "use lab.run_line1 instead.")
    display(IFrame(src=url, width="100%", height=height))
    return url


def run_line1(source: str, patch: str, rect=None,
              patch_mask: str | None = None, mask: str | None = None,
              instruction: str = "", mock: bool = False,
              port: int = 8765, out_root: str = "runs/st/ui",
              show: bool = True, **params):
    """The proxy-free door: files in, result out, rendered inline.

    `rect` = (x, y, w, h) in the 512 canvas. It is the PLACEMENT box; with
    `fit="contain"` (the default) the patch keeps its aspect and is centred in
    it, so a typed rect that disagrees on aspect cannot distort the patch.

    THE MASK COMES FROM THE PATCH (`patch_mask`), matching the UI. That is the
    unification: mark the object inside the patch, and that marking is carried
    to the source by the same placement the content got -- so a patch with its
    own background cannot paste that background in.

    `mask=` is the OLD source-space mask, kept because a segmentation mask made
    on the source (SAM) is a real use case. It is only right when the patch is
    tightly cropped: it does NOT restrict which part of the patch is used.
    """
    from types import SimpleNamespace

    if rect is None:
        raise SystemExit("rect=(x, y, w, h) is required: without the browser "
                         "there is nothing to infer the patch placement from")
    x, y, w, h = (int(round(v)) for v in rect)

    src = load_image(source, 512)
    patch_arr = load_image(patch, 512)
    pm = load_mask(patch_mask, None) if patch_mask else None
    if pm is not None and pm.shape[:2] != patch_arr.shape[:2]:
        # a patch mask at the wrong resolution is silently meaningless once it
        # is resized into the placement box
        from PIL import Image
        pm = np.asarray(Image.fromarray((pm * 255).astype(np.uint8)).resize(
            (patch_arr.shape[1], patch_arr.shape[0]), Image.NEAREST),
            np.float32) / 255.0

    payload = {
        "source_png": array_to_b64_png(src),
        "patch_png": array_to_b64_png(patch_arr),
        "patch_rect": [x, y, w, h],
        "fit": params.get("fit") or "contain",
        "grow": params.get("grow") or 0.0,
        "feather": params.get("feather") or 0.0,
        "instruction": instruction,
        "mock": mock,
    }
    if pm is not None:
        payload["patch_mask_png"] = array_to_b64_png(pm)
    if mask:                     # explicit source-space override
        payload["_mask_override_png"] = None    # documented, not routed here
        raise SystemExit(
            "mask= (a source-space mask) is no longer used by the server "
            "path.\nUse patch_mask=PATH -- a mask in the PATCH's coordinates, "
            "marking the object inside the crop -- which is what the UI does "
            "and what keeps the mask and the content on one geometry.")
    for k in ("seed", "steps", "kappa", "kappa_out"):
        if params.get(k) is not None:
            payload[k] = params[k]

    cfg = SimpleNamespace(out_root=out_root, host="127.0.0.1", port=port,
                          mock=mock, lab_url=None, open=False,
                          published_ports=(6006, 6008))
    res = handle_run(payload, cfg)
    if not res["ok"]:
        print("FAILED:", res.get("error"))
        print(res.get("log", "")[-2000:])
        return res

    if show:
        m = b64_png_to_array(res["mask_png"])[..., 0]
        _show(res, src, m)
    return res


def _show(res: dict, src: np.ndarray, mask: np.ndarray) -> None:
    """Render the result, the Y, and the numbers, in the cell."""
    try:
        import base64
        import io

        from IPython.display import HTML, display
        from PIL import Image
    except ImportError:      # not in a notebook: just report
        print({k: round(v, 4) for k, v in res["metrics"].items()
               if isinstance(v, float)})
        return

    def png(a):
        buf = io.BytesIO()
        Image.fromarray((np.clip(a, 0, 1) * 255).astype("uint8")).save(
            buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    m, ym = res["metrics"], res["y_metrics"]
    over = np.clip(src * 0.45 + np.repeat(mask[..., None], 3, 2)
                   * np.array([1.0, 0.2, 0.2], np.float32) * 0.55, 0, 1)
    held = m["verdict"]["background_held"]
    rows = "".join(
        f"<tr><td>{n}</td><td>{p}</td><td>{q}</td><td>{r}</td></tr>"
        for n, p, q, r in (
            ("result", f"{m['d_inside']:.3f}", f"{m['d_out_content']:.4f}",
             f"{m['sat_in']:.3f}"),
            ("patch pasted", f"{ym['d_inside']:.3f}",
             f"{ym['d_out_content']:.4f}", f"{ym['sat_in']:.3f}"),
        ))
    display(HTML(f"""
    <div style="font:13px system-ui">
      <div style="display:flex;gap:12px;align-items:flex-start">
        <div><div style="color:#888">source + mask</div>
             <img src="data:image/png;base64,{png(over)}" width="210"></div>
        <div><div style="color:#888">Y (what guidance pulls toward)</div>
             <img src="data:image/png;base64,{res['y_png']}" width="210"></div>
        <div><div style="color:#888">result</div>
             <img src="data:image/png;base64,{res['result_png']}" width="210">
        </div>
      </div>
      <table style="border-collapse:collapse;margin-top:8px">
        <tr><th></th><th>inside</th><th>outside content</th><th>sat x src</th></tr>
        {rows}
      </table>
      <div style="margin-top:6px">background held:
        <b style="color:{'#4ade80' if held else '#f87171'}">
        {'PASS' if held else 'FAIL'}</b>
        <span style="color:#888">(needs &lt; 0.02)</span> &nbsp; contrast
        <b>{m['contrast']:.2f}</b>
        <span style="color:#888">(inside / outside content)</span></div>
      {'<div style="color:#fbbf24;margin-top:6px">' + str(round(res['patch_outside_mask'] * 100)) + '% of the placed patch lies OUTSIDE the mask, so that part steers nothing.</div>' if res.get('patch_outside_mask', 0) > 0.02 else ''}
      <div style="color:#888;margin-top:6px">written to {res['rundir']}</div>
    </div>"""))


def _self_test() -> int:
    """CPU, no model, no notebook. The URL derivation is the part that silently
    produces a dead link, so it is pinned to the real address bar."""
    assert proxy_url(
        "https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443/jupyter/lab/"
        "tree/RegionFlow/short_term", 8765) == (
        "https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443/jupyter/"
        "proxy/8765/")
    # a base containing 'lab' must not be mistaken for the marker
    assert proxy_url("https://h/laboratory/lab/tree/x", 1) == \
        "https://h/laboratory/proxy/1/"
    # and the notebook/notebooks markers
    assert proxy_url("https://h/jupyter/notebooks/a.ipynb", 2) == \
        "https://h/jupyter/proxy/2/"

    # run_line1 in --mock, end to end, on files
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        src = np.full((512, 512, 3), 0.3, np.float32)
        Image.fromarray((src * 255).astype("uint8")).save(d / "src.png")
        # a patch that is mostly BLACK with one bright object, and a patch mask
        # marking ONLY the object -- the photographed failure was the black
        # background being pasted in
        pat = np.zeros((512, 512, 3), np.float32)
        pat[100:300, 100:300] = np.array([0.1, 0.2, 0.9], np.float32)
        Image.fromarray((pat * 255).astype("uint8")).save(d / "patch.png")
        pm = np.zeros((512, 512), np.float32)
        pm[120:280, 120:280] = 1.0
        Image.fromarray((pm * 255).astype("uint8")).save(d / "pmask.png")

        # without a patch mask: the whole footprint is used (tight-crop case)
        res = run_line1(str(d / "src.png"), str(d / "patch.png"),
                        rect=(200, 150, 120, 90), instruction="x", mock=True,
                        out_root=str(d / "out"), show=False)
        assert res["ok"], res
        assert Path(res["rundir"], "y.png").is_file()
        assert res["mask_from_patch"] is False
        assert res["metrics"]["d_inside"] > 0.05
        foot_frac = res["mask_frac"]

        # WITH a patch mask: the derived mask is smaller, and only the marked
        # object is pasted -- the black background is not
        res2 = run_line1(str(d / "src.png"), str(d / "patch.png"),
                         rect=(200, 150, 120, 90), patch_mask=str(d / "pmask.png"),
                         instruction="x", mock=True, out_root=str(d / "out"),
                         show=False)
        assert res2["ok"] and res2["mask_from_patch"] is True, res2
        assert res2["mask_frac"] < foot_frac, (res2["mask_frac"], foot_frac)
        # The unification: content and mask share one placement, so the patch's
        # unmarked corner must NOT appear in Y. Compare where alpha is EXACTLY
        # zero -- `_resize` is Lanczos, so the mask has a soft edge and pixels
        # with a small nonzero alpha are legitimately touched. Measured: alpha==0
        # holds to 0.002 (uint8), and only 57 pixels sit in (0.02, 0.5].
        y_arr = load_image(Path(res2["rundir"]) / "y.png")
        m_alpha = b64_png_to_array(res2["mask_png"])[..., 0]
        untouched = m_alpha <= 0.0
        assert untouched.any(), "the mask covers the whole frame -- vacuous test"
        assert np.allclose(y_arr[untouched], src[untouched], atol=1e-2), \
            f"Y changed pixels outside the derived mask: max " \
            f"{float(np.abs(y_arr - src)[untouched].max()):.4f}"
        edge = ((m_alpha > 0.02) & (m_alpha <= 0.5)).sum()
        assert edge < 500, f"the mask edge is {edge}px wide; expected a crisp " \
                           "edge from a nearest/Lanczos resize"

        # rect is mandatory: without a browser there is nothing to infer it from
        try:
            run_line1(str(d / "src.png"), str(d / "patch.png"), mock=True,
                      out_root=str(d / "out"), show=False)
        except SystemExit as exc:
            assert "rect" in str(exc)
        else:
            raise AssertionError("a missing rect must be refused")

        # the old source-space mask is refused with an explanation rather than
        # silently doing the wrong thing
        m = np.zeros((512, 512), np.float32)
        m[20:60, 20:60] = 1.0
        Image.fromarray((m * 255).astype("uint8")).save(d / "mask.png")
        try:
            run_line1(str(d / "src.png"), str(d / "patch.png"),
                      rect=(200, 150, 120, 90), mask=str(d / "mask.png"),
                      mock=True, out_root=str(d / "out"), show=False)
        except SystemExit as exc:
            assert "patch_mask" in str(exc), exc
        else:
            raise AssertionError("mask= must be refused in favour of patch_mask=")

    print("lab self-test OK")
    print("  proxy_url pinned to the real seetacloud address bar, and safe "
          "against a base")
    print("  containing 'lab'")
    print("  run_line1 (--mock): a mostly-black patch pastes ONLY its marked")
    print("  object -- Y is untouched outside the derived mask -- and the old")
    print("  source-space mask= is refused with a pointer to patch_mask=")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
