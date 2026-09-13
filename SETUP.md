# Setup manifest

Two different targets, do not confuse them:

* **Local 3090 Ti** — Stages 0, 1 and 3. This is where nearly all the work
  happens. Nothing needs renting.
* **Rented 8×A100** — Stage 2 (BiFM reimplementation) only, and only after
  Stage 0 produces a result.

---

## 1. What you move

Almost nothing. The repo is a few hundred kilobytes of Python.

```
regionflow/
  region/     lambda_field, alpha_field, attention, masking, prompts, pipeline
  bench/      data, metrics, sweep
  docker/     Dockerfile, compose.yml, requirements.txt, smoke_test.py
  scripts/    fetch_assets.sh
  README.md
```

Everything else downloads on the target machine. **Never** put weights or
datasets in the git repo or the Docker image — a rebuild would re-pull 25 GB.

## 2. What downloads there

| asset | size | needed for | source |
|---|---|---|---|
| SD3-Medium (+ T5-XXL, 2×CLIP) | ~25 GB | everything | HF, **gated** |
| PIE-Bench | ~1 GB | Stage 0 | `cure-lab/PnPInversion` |
| MAG-Bench | ~0.5 GB | Stage 0 (optional) | `HelenMao/MAG-Edit` — release unconfirmed |
| SAM ViT-H | 2.4 GB | instance annotation | fbaipublicfiles |
| LPIPS + CLIP ViT-L | ~2 GB | metrics | auto |
| MagicBrush | ~30 GB | Stages 2–3 only | `osunlp/MagicBrush`, gated |

Under 60 GB total. Your 3 TB disk is not a constraint.

**SD3-Medium is gated.** Accept the licence on Hugging Face *before* anything
else — a missing token produces a 401 that reads like a network failure and
will waste an hour.

## 3. Repos to clone

```bash
# the benchmark you will actually use
git clone --depth 1 https://github.com/cure-lab/PnPInversion.git

# MAG-Edit — the baseline, and the teacher if you do distillation later
git clone --depth 1 https://github.com/HelenMao/MAG-Edit.git
```

Two notes on these.

`PnPInversion` ships `mapping_file.json` in-repo, but the 700 images may be a
separate download linked from its README. Check for `data/annotation_images/`
after cloning; the mapping file alone is useless.

MAG-Edit's paper describes MAG-Bench, but I could not confirm a public release
of the benchmark itself. If it is not there, **proceed with PIE-Bench alone** —
it has 700 cases against 200, plus bracket annotations and edit-type ids that
MAG-Bench lacks. MAG-Bench's only advantage is its mask-size stratification,
which `bench.data.mask_bucket` reproduces from area.

You do not need a BiFM repo. There isn't one.

## 4. Commands, in order

```bash
# --- host ---
git clone <your-fork>/regionflow && cd regionflow

# .env goes in docker/, NOT the repo root. Compose v2 takes the project
# directory from the compose file's location, so a root-level .env is
# silently ignored when you pass -f docker/compose.yml and DATA_DIR falls
# back to the literal default.
cp docker/.env.example docker/.env      # set HF_TOKEN and DATA_DIR

# The bind mount shadows the image's ownership of /data, so the HOST
# directory must be owned by your uid or every write inside fails.
sudo mkdir -p /mnt/data/regionflow
sudo chown -R $(id -u):$(id -g) /mnt/data/regionflow

export UID=$(id -u) GID=$(id -g)

docker compose -f docker/compose.yml build

# verify DATA_DIR resolved before starting anything
docker compose -f docker/compose.yml config | grep -A3 volumes

docker compose -f docker/compose.yml run --rm dev bash

# --- inside the container, from here on ---
python docker/smoke_test.py       # STOP if this fails
bash scripts/fetch_assets.sh stage0
```

Then, before running any sweep, three verification steps. Skipping these is how
you get a clean-looking run that means nothing:

```bash
# 4a. what is actually in the annotation file
python -c "
from bench.data import inspect_schema
print(inspect_schema('/data/repos/PnPInversion/data/mapping_file.json'))"

# 4b. do the masks decode correctly
python -c "
from bench.data import load_pie_bench, validate
from PIL import Image; import numpy as np
cases = load_pie_bench('/data/repos/PnPInversion/data',
                       load_image=lambda p: np.array(Image.open(p).convert('RGB')))
print(validate(cases))"
```

In 4b, `!! EMPTY MASKS` means the RLE convention is wrong — fix `decode_rle`
before going further, or every number you produce afterwards is noise.

```bash
# 4c. raster-order check — the highest-value single test in this project
python scripts/check_raster_order.py
```

That one puts all field mass on a single corner token and confirms the edit
lands in that corner. If the patch embedder's flatten order disagrees with
`field.reshape(-1)`, every edit silently lands in the wrong place and the
outputs look like plausible bad results rather than a bug.

Only then:

```bash
# 4d. time ten cases, rescale, decide whether the estimates hold
python -m bench.run --cases 10 --steps 50 --time-only

# 4e. lambda pre-sweep (100 cases) — must precede the r-sweep
python -m bench.run --sweep lambda --cases 100 --out /data/outputs/lambda

# 4f. the r-sweep. this is the result.
python -m bench.run --sweep r --lam-in <chosen> --lam-out <chosen> \
                    --out /data/outputs/rsweep
```

## 5. If you do rent the 8×A100

Different needs entirely:

* MagicBrush instead of PIE-Bench (`fetch_assets.sh stage2`)
* **Checkpoint and resume from step one.** If your free 6-hour grant repeats,
  eight chained sessions equal the entire 50-hour rental at no cost. Build this
  in before you ever pay.
* Measure seconds/step locally first, scale by GPU count and the A100/3090Ti
  ratio, and rent exactly the hours that buy the epochs you need. An hour of
  measuring protects a budget you cannot refill.
* Do not rent until Stage 0 has produced a result. If the graded field does not
  beat the binary control at 50 steps, no amount of few-step machinery rescues
  it.

## 6. Still missing

`bench/inversion.py` and `bench/run.py` do not exist yet — steps 4c through 4f
reference them. They are the next thing to write.
