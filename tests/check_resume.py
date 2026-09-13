"""Save/resume round-trip, including device placement.

    python tests/check_resume.py --device cuda

Resume is load-bearing: an 8-hour run under a 6-hour block cap only works if
`--max-hours` -> save -> rerun -> load reproduces the training state. Nothing
in the short tests exercised it, and the first real chained run died on the
first EMA update because the shadow came back on CPU while the parameters
were on CUDA.
"""
import argparse
import tempfile
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transport.train import EMA  # noqa: E402

dev = args.device
params = [torch.nn.Parameter(torch.randn(8, device=dev)) for _ in range(3)]
ema = EMA(params, 0.99)
for _ in range(5):
    for p in params:
        p.data.add_(torch.randn_like(p) * 0.01)
    ema.update(params)

# Round-trip through disk exactly as Trainer.save/load does.
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "ckpt.pt"
    torch.save({"ema": ema.state_dict()}, path)
    sd = torch.load(path, map_location="cpu", weights_only=False)

    naive = EMA(params, 0.99)
    naive.load_state_dict(sd["ema"])          # no params -> stays on CPU
    on_cpu = [t.device.type == "cpu" for t in naive.shadow]
    print(f"without params=: shadow on cpu = {all(on_cpu)}"
          f"  (params are on {dev})")

    fixed = EMA(params, 0.99)
    fixed.load_state_dict(sd["ema"], params=params)
    print(f"with params=:    shadow devices = "
          f"{sorted({str(t.device) for t in fixed.shadow})}")

    try:
        fixed.update(params)
        print("update() after resume: OK")
    except RuntimeError as exc:
        print(f"update() after resume: FAIL {exc}")
        raise SystemExit(1)

    fixed.copy_to(params)
    print("copy_to() after resume: OK")

    same = all(torch.allclose(a, b.to(a.device))
               for a, b in zip(fixed.shadow, ema.shadow))
    print(f"shadow values preserved through save/load: {same}")
    raise SystemExit(0 if same else 1)
