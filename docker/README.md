# Container setup

Target machine: RTX 3090 Ti (24 GB, sm_86), Ryzen 9950X (16c), 64 GB RAM, 3 TB disk.

That card is enough for **Stages 0, 1 and 3 locally**. Only a BiFM
reimplementation (Stage 2) needs rented GPUs.

## First run

```bash
# host prerequisites: docker, nvidia-container-toolkit
cp docker/.env.example .env      # then set HF_TOKEN and DATA_DIR
export UID=$(id -u) GID=$(id -g)

docker compose -f docker/compose.yml build
docker compose -f docker/compose.yml run --rm dev python docker/smoke_test.py
```

SD3-Medium is gated. Accept the licence on Hugging Face and put the token in
`.env` before the first model download, or diffusers fails with a 401 that
looks like a network error.

## What the smoke test is really checking

Checks 1–3 are routine. Checks 4 and 5 are the ones that decide whether Stage 2
is possible at all:

* **JVP works, but only with `PYTORCH_JIT=0`.** On this stack (torch
  2.6.0+cu124, py3.11, Zen 5) TorchScript's compiler segfaults. Forward-mode AD
  reaches it lazily on the first `make_dual`:

  ```
  make_dual -> _maybe_load_decompositions -> decompositions_for_jvp
            -> torch.jit.script -> SIGSEGV
  ```

  The crash is in the JIT compiler, not in autograd, and it is uncatchable from
  Python — hence the subprocess probes. `PYTORCH_JIT=0` makes `jit.script` a
  passthrough and everything passes. It is set in both compose.yml and the
  Dockerfile, and the build fails if the check does not pass.

  Diagnostic history worth keeping: the first hypothesis was that the fused
  attention kernels do not support forward-mode AD. That was wrong. The tell
  was a CPU-only `nn.Linear` probe crashing identically — no attention, no
  CUDA. Whenever every probe fails the same way, suspect the shared setup step,
  not the thing each probe was meant to isolate.

* **JVP vs gradient checkpointing.** These do not compose. Checkpointing is
  exactly what you would otherwise use to fit a batch into 24 GB, so losing it
  changes the VRAM arithmetic for training. The test expects this to fail; the
  point is to see it in minute one rather than week three.

Neither affects Stages 0 or 1, which are inference only.

## Memory

| component | bf16 |
|---|---|
| T5-XXL | ~9.4 GB |
| SD3-Medium DiT | ~4.3 GB |
| CLIP-L + CLIP-G | ~1.6 GB |
| VAE | ~0.3 GB |
| **subtotal** | **~15.6 GB** |

Roughly 8 GB left for activations at 512², which is comfortable for inference
and workable for LoRA at batch 1–2. If it gets tight, cache text embeddings
once offline and unload T5 — the prompts are fixed across the whole sweep, so
this costs one pass and buys back 9 GB permanently.

`expandable_segments:True` is set in compose.yml; fragmentation, not raw
capacity, is what usually kills a run on a single card.

## Layout

Weights and datasets live on the host 3 TB disk at `$DATA_DIR`, mounted to
`/data`, with `HF_HOME` pointed there. The image stays small and a rebuild
never re-downloads 20 GB.

```
/data
  hf/            # HF cache: SD3, T5, CLIP
  torch/         # SAM checkpoints
  datasets/      # PIE-Bench, MAG-Bench
  outputs/       # sweep results
```

## Caveats

This image has not been built or run — no Docker or network in the environment
where it was written. Expect to adjust:

* the `torch==2.6.0 / torchvision==0.21.0` pin, if that pair has moved;
* `nvidia/cuda:12.4.1` if your host driver is older than 550;
* `segment-anything` from git, which has no release cadence.

Nothing else in the image is version-sensitive.

## Runtime estimates on this machine

| task | scope | estimate |
|---|---|---|
| Stage 0 r-sweep | 4 radii × ~540 PIE cases, 50+50 steps @512² | ~8–12 h |
| λ pre-sweep | 9 configs × 100 cases | ~4 h |
| Stage 1 α ablation | subset | ~3 h |

Overnight jobs, not rentals. Measure a 10-case run first and rescale — these
are extrapolations, not measurements.
