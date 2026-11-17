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

## Referent-counterfactual change attribution — prior-art check (2026-11-18)

Search-level check by a research subagent (abstracts and HTML pages, not full reads;
Semantic Scholar was unreachable from the session, so DiffEdit's citing papers were NOT
enumerated). Result: **not found**, which is not the same as novel.

- **Closest prior work.** DiffEdit (arXiv 2210.11427) builds its mask by contrasting
  noise predictions under a query text vs a reference text. Watch Your Steps (ECCV 2024,
  arXiv 2308.08947) contrasts predictions with vs without the instruction. Both contrast
  *edit vs no edit*, so neither separates same-category instances: "red jacket" applies
  to every girl.
- **Mask-free localisation** (Prompt-to-Prompt local blend, InstructEdit, ZONE, FoI,
  LEDITS++) uses attention or grounding; none found contrasting two referents.
- **Multi-instance leakage** (MIRAGE 2604.05180, Instance-Disentangled Attention
  2602.08749, MICE 2606.31278, BindEdit 2606.18906, AdaptEdit 2604.23763) separates
  instances through attention or per-instance masks in one pass; none described runs
  two edits that differ only in the referent.
- **Attachment / interaction editing:** Add-it (ICLR 2025, placement affordance),
  InteractEdit/IEBench (2503.09130), HOI-Edit (2606.19073, instance-level HOI evaluator,
  closest), MPIE-Bench (2607.27616, multi-person interaction, not opened). No benchmark
  for extra limbs in instruction editing found.

Reviewer objections to expect: "DiffEdit with the reference text set to a sibling
expression" (answer: the contrast is between two full edited outputs, used to subtract
spillover shared across referents, not to localise the edit); and a comparison against
single-pass leakage fixes (MICE, BindEdit, IDA, MIRAGE). Owed before any novelty claim:
DiffEdit's citation list and a read of RefEdit's method section.

### DiffEdit, read from the paper (user-supplied PDF, 2026-11-18)

- **Mask (Sec. 3.2, Step 1):** noise estimates of the input image at 50% noise, under
  the query text vs a reference text (or empty text); extreme values removed, spatial
  differences averaged over n = 10 noises, rescaled to [0, 1], binarised at 0.5. Masks
  "generally somewhat overshoot". Editing then runs DDIM decoding with the latents
  outside the mask replaced by the DDIM-encoded source (ỹ_t = M y_t + (1 − M) x_t).
- **Selecting an object** (App. A.5, examples 7–12) is done by contrasting a query
  with a similar reference caption ("a bowl of fruits" vs "a bowl of pears"). That
  selects an object *type*, not one of several instances.
- **Its own failure modes (Fig. 19):** columns 4–5, "mask does not isolate object to be
  edited, both objects": multiple similar objects enter the mask when the query needs
  only one, causing over-editing. Columns 1–3: inserting an object fails or lands in the
  wrong place, because the mask "seeks an anchor visual element".

**What this means for us.** DiffEdit names both of our problems as unsolved failure
modes of its contrast: same-kind instances (Level 1) and additions outside an existing
object (Level 2). Our contrast differs in three ways: (1) between two *referents* under
the same edit, not edit vs no-edit; (2) between two finished outputs in pixel space,
after editing, not between noise estimates before; (3) the referent's own region comes
from a segmenter, and the contrast only decides what to keep *outside* it. The direct
ablation is the same rule with no counterfactual (keep every change vs the source that
connects to the mask): `<id>_connected.png`, written by `--edit` at no extra klein cost.
A DiffEdit-mask baseline on klein (contrasting velocity predictions under the
instruction vs empty text) is still owed if reviewers want the original method.
