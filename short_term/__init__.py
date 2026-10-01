"""short_term: get a working local editor, then get it to edit the right thing.

Two tiers, deliberately separated:

  TIER A -- run an editor that already exists. `edit.py --guide none`.
            The checkpoints in this repo's Gate C harness (RefEdit-SD3,
            SD3_UltraEdit_*) ARE instruction editors with mask support. Tier A
            is a CLI over them, so a usable editor and a fixed evaluation
            convention exist before any training is spent.

  TIER D -- inference-time guidance (PiGDM-style) on top of that editor, with
            the edit mask as the per-coordinate weight. Zero training. See
            `guidance.py` for the equations and what was verified about them.

Order: A, then D, then the training work (tier B) in `EDITOR_PLAN.md`. D is
cheaper than B and independent of it, so it comes first.

    python short_term/guidance.py --self-test    # CPU, seconds, no weights
    python short_term/demo_toy.py                # the mechanism, end to end
    python short_term/edit.py --help
"""

__all__ = ["guidance", "sampler", "backends"]
