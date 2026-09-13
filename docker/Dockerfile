# RegionFlow dev image — targets a single RTX 3090 Ti (Ampere, sm_86, 24 GB).
#
# Build:
#   docker build -t regionflow:dev -f docker/Dockerfile .
#
# Weights live on the host, NOT in the image. SD3-Medium plus T5-XXL is well
# over 20 GB and rebuilding the image would re-download it. See compose.yml.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # 8.6 is the 3090 Ti. Add 8.9 for a 4090, 9.0 for H100 if you rent later;
    # extra arches only cost build time for compiled extensions.
    TORCH_CUDA_ARCH_LIST="8.6" \
    # Ampere tensor cores for fp32 matmuls — free speedup, no accuracy cost
    # that matters here.
    NVIDIA_TF32_OVERRIDE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git git-lfs curl wget ca-certificates \
        build-essential ninja-build \
        libgl1 libglib2.0-0 \
        tmux htop \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# --- torch first, from the CUDA 12.4 index ---
# Not tightly pinned: if this exact version has been pulled, bump it rather
# than fighting the resolver. Any torch >= 2.4 with cu124 works.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0

# CRITICAL: lpips, torchmetrics and open-clip-torch all declare torch as a
# dependency. Without a constraint, pip can resolve that to a different torch
# build and install it over the cu124 one. The constraints file pins torch so
# no transitive dependency can move it.
RUN printf 'torch==2.6.0\ntorchvision==0.21.0\n' > /tmp/constraints.txt

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -c /tmp/constraints.txt -r /tmp/requirements.txt

# TorchScript's compiler segfaults on this stack (torch 2.6.0+cu124, py3.11,
# Zen 5). It is reached lazily on the first forward-mode AD call: make_dual ->
# _maybe_load_decompositions -> decompositions_for_jvp -> torch.jit.script ->
# crash. Confirmed by faulthandler; the crash is in the JIT compiler, not in
# autograd.
#
# PYTORCH_JIT=0 turns torch.jit.script into a passthrough that returns the
# original Python function. Nothing in this project needs TorchScript --
# diffusers, transformers and peft all run eager -- so the only cost is that
# any library relying on jit.script for speed falls back to eager.
#
# Without this, MeanFlow/BiFM (stage 2) is impossible: every jvp call dies.
ENV PYTORCH_JIT=0

# Fail the build rather than ship an image where stage 2 is silently blocked.
RUN python -c "\
import torch; from torch.func import jvp; \
net = torch.nn.Linear(8, 8); x = torch.randn(2, 8); \
jvp(net, (x,), (torch.randn_like(x),)); \
assert torch.version.cuda == '12.4', torch.version.cuda; \
print('torch ok:', torch.__version__, torch.version.cuda, '| jvp ok')"

# segment-anything is git-only. Checkpoints are downloaded at runtime to the
# mounted volume, not baked in.
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Non-root, uid 1000 so bind-mounted files stay writable from the host.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} dev 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash dev

# Every cache on the mounted volume. The image stays small and a rebuild
# never re-downloads 20 GB of weights.
ENV HF_HOME=/data/hf \
    TORCH_HOME=/data/torch \
    XDG_CACHE_HOME=/data/cache \
    PYTHONPATH=/workspace

RUN mkdir -p /data /workspace && chown -R ${UID}:${GID} /data /workspace

USER dev
WORKDIR /workspace

CMD ["bash"]
