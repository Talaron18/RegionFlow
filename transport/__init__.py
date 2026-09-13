"""Stage 1: conditional interval transport on a pretrained flow model.

    adapter.py       TransportAdapter -- SD3 wrapped as u(z, r, t, c, M)
    gating.py        gated LoRA; the identity and boundary guarantees
    conditioning.py  interval and mask conditioning, both zero-initialised
    pairflow.py      Gate A objective: source -> target transport
    meanflow.py      Gate B objective: average velocity, JVP or finite difference
    sampling.py      few-step sampler
    data.py          triplet cache (encoders run once, offline)
    train.py         trainer with resumable checkpoint chaining
"""

from .adapter import AdapterConfig, TransportAdapter
from .gating import GateHolder, inject_lora

__all__ = ["AdapterConfig", "TransportAdapter", "GateHolder", "inject_lora"]
