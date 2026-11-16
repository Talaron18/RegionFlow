# Prior-art check for the referent-grounded editing direction (2026-11-16)

From a search-based sweep by a research subagent (about 25 queries). Verification
level per item: **[FULL]** arXiv page read, **[ABS]** abstract read, **[SNIP]** search
snippet only. Absence from search is not proof of novelty; a full sweep (Semantic
Scholar / Google Scholar) is still required before any novelty claim in a paper.

## Corrections to ids cited earlier in this repo (GATE_C_REPORT §9)

| id | actual title | what it does | masks at inference |
|---|---|---|---|
| 2602.08749 [FULL] | Shifting the Breaking Point of Flow Matching for Multi-Instance Editing | Instance-Disentangled Attention in MMDiT joint attention; optional LoRA with the plain flow-matching loss | yes, every step |
| 2606.31278 [ABS] | Editing Everything Everywhere All at Once (MICE; page says ECCV 2026) | training-free bias on MMDiT joint attention from segmentation masks | yes |
| 2604.05180 [ABS] | MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing | training-free: a VLM splits the instruction into regions, parallel denoising; introduces MIRA-Bench, also evaluates RefEdit-Bench | not stated |

2606.31278 was labelled "biased joint attention" in this repo; that describes the
method, it is not the title. Per a snippet, MIRAGE reports that FLUX.2 and
Qwen-Image-Edit over-edit when three or more similar instances are present, which
overlaps our "a 2026 editor still leaks" finding.

## Verdict per element of the proposal

| | element | verdict | closest work |
|---|---|---|---|
| A | mask-supervised training, mask-free inference | **known in instruction editing** | AdaptEdit / "Edit Where You Mean" 2604.23763 [ABS]: adapters + region-aware loss, GT masks in training only, learned mask predictor. FlowMimic 2607.18227 [ABS, partial]: video-first, referring segmentation as a training task. |
| B | counterfactual pairs differing only in the referent | **no prior art found** | neighbouring: VIBE 2601.02242 (same source, different instruction, object swaps), SuperEdit 2505.02370 (triplet loss on instructions) |
| C | attention loss tying referring tokens to the instance mask | **known in neighbouring tasks, not found for referring edits** | FastComposer 2305.10431, Break-A-Scene 2305.16311, TokenCompose 2312.03626 (UNet generation/personalisation); RegionRoute 2602.19254 (regional style); FlowMimic has an "attention-level loss" whose target was not verified |
| D | penalty on change to same-category siblings | **no prior art found for siblings** | generic non-edited-region preservation is common (CoCoEdit 2602.14068, GIE-Bench 2505.11493) |
| E | crop-edit-paste teacher for data | **known in a neighbouring form** | RefEdit 2506.03448 (masked synthetic data), SmartEraser 2501.08279 |
| F | reading the editor's attention as a no-training diagnostic | **known** | FoI 2312.10113, ZONE 2312.16794, ConceptAttention 2502.04320 (FLUX). The target-vs-sibling test on klein was not found. |

No paper was found combining A+B+C+D. Evaluation lineage to cite: GIE-Bench
(masked preservation), "text embedding leakage" TELS/TILS 2412.04715, EditInspector
2506.09988 (VLM edit judges perform poorly). No paper was found showing a VLM judge
misses same-category sibling leakage specifically.

## Consequences for the plan

- "Train with masks, run without" cannot be the headline; AdaptEdit must be a compared
  baseline, and MIRAGE (code released) should be run on our 151 cases.
- The defensible core is **same-category referents**: referent-only counterfactual
  pairs (B), sibling-negative supervision (C applied with D), and sibling-level
  evaluation including the judge-blindness result.
- 2602.08749 and 2606.31278 share authors and work on multi-instance MMDiT editing with
  masks at inference; a trained or mask-free follow-up from that group is plausible.

## Not verified

RefEdit's training recipe and whether its results cover same-category cases;
ReferPix2Pix (openreview unreachable); RegionRoute's backbone; whether BindEdit
2606.18906 regularises attention at training or per sample; VIBE's negatives and loss;
FlowMimic's attention target; MIRAGE's mask use and numbers; every numeric result (none
are quoted here). Not checked: MGIE, InstructEdit, InstDiffEdit, GLIGEN, MIGC, LoMOE,
InfoEdit, OmniEdit, AnyEdit, ImgEdit, GEdit, the Kontext and Qwen-Image-Edit reports.
