import json, numpy as np
from refedit_siblings import raster, bench_image, load_bench, load_coco

for r in map(json.loads, open("runs/gate_c/smoke_oracle.jsonl")):
    y0, x0, y1, x1 = r["off_instance_bbox"]
    t = np.asarray(load_target_mask(r["id"]))          # 与 run_gate_c 中的调用一致
    ys, xs = np.nonzero(t)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).min()  # 距离目标边界的像素距离
    print(f"{r['id']:<14} peak {d:6.1f}px from target   "
          f"over_target {r['off_instance_over_target']:.2f}")