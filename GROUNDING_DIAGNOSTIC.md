# Does klein know where the referent is? — diagnostic scope

Status: **probe run on all 151 cases (2026-11-16); result and the join with the audit
are at the end of this file.** Readings below were fixed before any run.

Checked on CPU (2026-11-16, diffusers 0.43.0): on a small random
`Flux2Transformer2DModel`, installing the probe leaves the model output bit-identical,
the recorded weights reproduce the stock attention output in all five blocks (max
difference < 1e-4), a wrong token-layout assumption is refused, and removing the probe
restores the model. Not checked: the real pipeline's token counts and call count per
step; the script stops with an error if either differs from what it expects.

```bash
python bench/klein_grounding_probe.py --self-test
python bench/klein_grounding_probe.py --limit 4 --swap \
  --out runs/gate_c/klein_grounding_smoke.jsonl \
  --save-maps runs/gate_c/klein_grounding_smoke_maps      # look at the PNGs
python bench/klein_grounding_probe.py --swap \
  --out runs/gate_c/klein_grounding.jsonl --save-maps runs/gate_c/klein_grounding_maps
python bench/klein_grounding_probe.py --report runs/gate_c/klein_grounding.jsonl \
  --outcomes runs/gate_c/klein4b_none.jsonl
```

## Question

FLUX.2 klein 4B edits a sibling in roughly half of the multi-instance cases
(`MODERN_BASE_EVAL.md`). Two different failures would look the same from outside:

- **Grounding failure:** the model never resolves "the cow in back" to one instance.
- **Generation failure:** it resolves the referent, and the edit spreads anyway.

They call for different training losses, so measure which one it is before designing
the method. No training is involved.

## What the model looks like inside (read from diffusers 0.43.0, `transformer_flux2.py`
and `pipeline_flux2_klein.py`)

- Text: Qwen3 (text-only) hidden states from layers 9, 18, 27, up to 512 tokens, built
  with a chat template. The text encoder never sees the image, so any grounding has to
  arise inside the transformer.
- Image stream: `[noisy output tokens, reference-image tokens]`, concatenated in that
  order; reference tokens carry their own position index. For a 512² source, 1024 of
  each.
- Blocks: 5 dual-stream then 20 single-stream. In both kinds, attention runs over the
  concatenation `[text, output, reference]`, with rotary positions applied to queries
  and keys. Attention goes through swappable processor classes
  (`Flux2AttnProcessor`, `Flux2ParallelSelfAttnProcessor`), which is the hook point.
- Distilled checkpoint: 4 steps, so 25 blocks × 4 steps = 100 attention maps per run.

## Measurement

For each case and each of the two instructions the benchmark provides (full referring
expression; single-instance instruction that names only the category):

1. Run the normal 4-step edit with recording processors installed.
2. In every block and step, take attention **from the instruction's content tokens to
   the reference-image tokens** (rows renormalised over reference tokens only), average
   over heads and over the content tokens. Content tokens = the user's instruction,
   excluding chat-template and padding tokens, found with tokenizer offsets.
3. Reshape to the reference token grid and compare with the target and sibling masks
   pooled to that grid (area pooling, as elsewhere in this repo).

Two scores per map:
- **Pointing:** among the annotated instances (target + siblings), does the instance
  with the highest mean attention density equal the target? Chance = 1 / (number of
  instances).
- **Target preference:** mean density on the target ÷ mean density on the siblings.

Also record the reverse direction (which reference tokens attend to the instruction)
as a secondary view; it is the DAAM-style map and may behave differently.

**Primary statistic, fixed in advance:** pointing accuracy of the map averaged over
ALL blocks and ALL steps, full instruction. Per-layer and per-step curves are reported
as exploratory; the best layer is never quoted as the result (it is chosen after
seeing the data).

## Built-in controls

- **Single-instance instruction.** It names the category only, so it cannot pick out
  one instance. Its pointing accuracy is the null for "attention prefers the target
  for reasons other than the referring expression" (size, centrality, saliency).
- **Swapped instruction.** Another case's instruction on this image: target preference
  should fall to about 1.
- **Sanity on the probe itself (before any real run):** a synthetic check that the
  recording processor reproduces the model's output exactly (same image with and
  without the probe installed), so the probe is measuring the path that executes.

## How the result will be read

Let `P_full` and `P_single` be primary pointing accuracy with the two instructions,
and split cases by the plain run's outcome (measurable / mis-targeted / off-instance;
and clean / leaked for the 40 audited ones).

| observation | reading | what the method should centre on |
|---|---|---|
| `P_full − P_single` ≥ 20 points, and `P_full` ≥ 70% **also in the leaked and mis-targeted cases** | The model grounds the referent; the edit spreads in generation. | Sibling-preservation loss; grounding loss is secondary. |
| `P_full − P_single` < 10 points | The referring expression does not move attention: a grounding failure. | Grounding loss + counterfactual referent pairs. |
| `P_full` high in clean cases, near `P_single` in leaked/mis-targeted ones | Grounding succeeds when the edit succeeds and fails when it fails. | Grounding loss, with attention on the target as the training signal; the probe itself becomes a predictor of failure. |
| anything else | Mixed. Report the table; do not force a story. | Both losses, ablated. |

Whatever the outcome, the per-case target preference is a candidate automatic leakage
predictor to compare against the human audit, which the current judge cannot provide.

## To build

1. `scripts/klein_grounding_probe.py`: two recording processors mirroring the stock
   ones (they must recompute attention weights explicitly, since the stock path calls a
   fused kernel that does not return them), content-token span finder, mask pooling,
   scores, one JSON line per case/prompt/block/step. Medium. Memory: text rows × all
   keys is small; the reverse direction is ~2k × 2.5k per head and must be reduced over
   heads immediately.
2. A small report script: primary table, per-layer curves, split by outcome class.
3. Reuse `bench/run_gate_c.py`'s case, mask and seed plumbing so cases pair with every
   existing arm.

## Limits

Attention is evidence about where information flows, not proof of what the model
"knows"; 151 cases; one base model; the audit split has only 40 cases. A result on
klein 4B need not hold for the 9B model or for Qwen-Image-2.1.

---

## Result (2026-11-16, user's GPU run, 151 cases, `--swap`)

Primary statistic, map averaged over all 25 blocks and 4 steps:

| instruction | pointing accuracy | chance | target preference (median) |
|---|---|---|---|
| full referring expression | **78.1%** | 42.4% | 1.44 |
| single-instance (category only) | 59.6% | 42.4% | 1.19 |
| swapped (another case's instruction) | 51.0% | 42.4% | 1.07 |

Paired, full vs single: +18.5 points; 35 cases correct only with the full
instruction, 7 only with the single one (exact two-sided sign test p < 0.0001).
The instruction tokens put about 12% of their attention on reference-image tokens,
69% on text, 19% on the output tokens.

Full instruction, split by what the plain no-mask run did:

| plain run's class | n | pointing accuracy |
|---|---|---|
| measurable (target changed most) | 108 | 87.0% |
| mis-targeted (a sibling changed more) | 34 | 61.8% |
| off-instance | 9 | 33.3% |

**Reading against the table fixed above.** Row 1 is not met (+18.5 is under 20, and
pointing in mis-targeted cases is 61.8%, under 70). Row 2 is not met (well over +10).
Row 3 fits: pointing is high where the edit landed on the target and falls to about the
single-instruction level (59.6%) where it did not. So **mis-aiming goes with a
grounding failure**, and attention on the target is a usable training signal.

Derived from the counts: when the probe points at the target (118 cases), the edit was
mis-aimed in 24 (20%); when it points elsewhere (33 cases), in 19 (58%).

**What this does not yet answer.** The audit found sibling leakage in 9 of 16 audited
`measurable` cases, a class where pointing is 87%. If those leaked cases point
correctly, leakage there is a generation failure (the model finds the referent and the
edit spreads) and needs the preservation loss; if they do not, it is grounding again.
That needs the per-case file joined with the audit ledgers.

Controls behaved as intended, with one thing to note: the swapped and single
instructions score 51–60%, above the 42.4% chance. Attention density favours the target
somewhat regardless of the text (size, position or saliency), so the referring
expression's own contribution is the +18.5, not the gap to chance.

### Join with the audit ledgers (2026-11-16, the 40 audited ids)

Per-case file joined with the plain audit (`blind_klein_*`) and the paste-back audit
(`blind_comp_*`). The join is valid: the probe's instruction equals the plain run's
prompt in 151 of 151 cases and both take their seed from `case_seed(id, 0)`.
Pointing = primary statistic, full instruction.

**The question fixed above** (audited `measurable` cases judged `wrong_edit`):

| group | n | points at target | single instruction | chance |
|---|---|---|---|---|
| measurable, judged clean | 6 | 6 | 3 | 36% |
| measurable, judged wrong_edit | 9 | **7** | 4 | 46% |
| — of those, leaks (paste-back verdict clean or degraded) | 7 | **5** | 3 | 45% |
| — of those, a different edit on the target (refedit-72, -150) | 2 | 2 | 1 | 50% |

The leaked cases mostly point correctly (5 of 7; misses are refedit-154 and -196), so
by the rule above leakage in this class **leans towards a generation failure**: the
referent is found and the edit spreads anyway. It is 7 cases and one more miss would
put it at 4 of 7, so this is a direction, not a result.

**Exploratory (grouping chosen after seeing the data, stratified sample).** The
paste-back verdict says what happened to the target itself, so it splits the 40 by
whether the target was edited at all:

| what happened to the target (paste-back verdict) | n | points at target | chance |
|---|---|---|---|
| edited as asked (clean or degraded) | 23 | 20 (87%) | 43% |
| not edited (no_edit) | 13 | 5 (38%) | 43% |
| a different edit (wrong_edit) | 4 | 2 | 46% |

Fisher exact, edited vs not edited: p = 0.006. This mostly restates the 151-case
outcome split, since all 13 `no_edit` cases come from the mis-targeted and off-instance
classes. Across all classes, leaks (plain `wrong_edit`, paste-back clean or degraded)
point correctly in 12 of 15, against 7 of 7 for cases clean in the plain arm whose
target was also edited as asked.

**Reading.** Two failures with different signatures: where the target is not edited,
attention is at chance (grounding); where the target is edited and a sibling changes
too, attention is mostly on the target (generation). That is the table's "both losses,
ablated" outcome rather than grounding alone. Not shown: that either is causal, or that
the leak figure holds beyond 15 cases and one rater.

### Looking at the maps

`python bench/grounding_heatmaps.py` writes one panel per case (source, plain edit,
attention for the full / single / swapped instruction) plus `index.html` to
`runs/gate_c/klein_grounding_heat/`. It needs only numpy and Pillow and the files the
probe saved; the source image is recovered from the probe's own overlay. Colour is a
multiple of uniform attention, clipped at the pooled 99th percentile per case. The
probe's own overlays are scaled to the map's maximum and come out nearly black.

Caveat the pictures make visible: `pointing` compares MEAN density per instance, so a
"yes" can be a near-tie. Of the 118 full-instruction "yes" cases, 8 have target
preference under 1.1 and 24 under 1.25 (refedit-113 is one: 1.08, with the visible hot
spots on the sibling). Look at the low end of each "points" section before leaning on
78.1%.

### Attention that does not depend on the referring expression (2026-11-16, exploratory)

Prompted by looking at the heat maps. Seen in the panels by the rater: under the
swapped instruction, and often the single one, attention goes to large, near, bright
or strongly coloured objects and to people. **Which feature drives it is not tested**:
this machine has no masks or source images. Everything below was defined after seeing
the data, and **none of it revises the primary statistic or its reading above.**

What the per-case file allows is using the swapped-instruction map as a stand-in for
"attention not caused by this case's referring expression". It is not text-free: the
swap is another case's full instruction, and in 52 of 151 cases the partner has the
same category (82 of 151 cases are `person`), so it can pick out an instance by
attribute. Numbers are given for all 151 and for the 99 different-category cases.

**A referent-independent component exists.** The swapped instruction points at the
target in 51.0% of all cases and 53.5% of the different-category ones, against a
chance of 42–43%.

**It decides where pointing is lost, not where the edit lands.**

| swapped map favours | n | full instruction points at target | plain edit off target |
|---|---|---|---|
| the target | 77 | 69 (90%) | 19 (25%) |
| a sibling | 74 | 49 (66%) | 24 (32%) |
| the target, different-category only | 53 | 46 (87%) | 13 (25%) |
| a sibling, different-category only | 46 | 32 (70%) | 14 (30%) |

The referring expression overturns the swapped map's preference in about two thirds of
the cases where it has to. The swapped map's preference does not predict an off-target
edit (25% against 32%, p = 0.37). The full instruction's pointing does, in both halves:
off target in 14 of 69 and 10 of 49 when it points at the target, 5 of 8 and 14 of 25
when it does not.

**Dividing the swapped or single map out** (per instance, density under the full
instruction ÷ density under the other map; is the target raised most, ties not
counted). "Off target" = mis-targeted + off-instance, 43 cases:

| statistic | all | measurable | mis-targeted (34) | off target (43) | p, measurable vs off target |
|---|---|---|---|---|---|
| raw full (primary) | 78.1% | 87% | 62% | 56% | 0.0001 |
| full ÷ swap | 70.2% | 76% | 65% | 56% | 0.02 |
| full ÷ swap, different-category (99) | 68.7% | 75% | 65% | 52% | 0.05 |
| full ÷ single, 141 cases with distinct prompts | 61.0% | 63% | 62% | 56% | 0.56 |

The gap between well-aimed and off-target cases shrinks when the other map is divided
out (31 points raw, 20 with swap, 7 with single) and whether it survives depends on
the denominator. Two things confound this: a ratio of two noisy maps is noisier, which
shrinks any real gap; and the single instruction is not a null (10 cases have
identical full and single prompts, and single ÷ swap raises the target in 55% of the
rest). So: part of the raw gap is attention that would sit on the target whatever the
referring expression said, and how large that part is cannot be read from this data.

**For the method (suggested, not shown).** The loss should supervise where attention
ends up, since that is what tracks the edit. The cases where the swapped map favours a
sibling are a candidate hard subset to report separately; that needs the feature check
below first. Referent-only pairs fit: each instance is the target once, so every pair
has a member that runs against whatever the image-driven preference is.

### Feature check — scope (written before any run)

`bench/grounding_saliency.py`, CPU only, needs the container's masks and source images.

```bash
python bench/grounding_saliency.py --self-test
python bench/grounding_saliency.py --extract
python bench/grounding_saliency.py --report runs/gate_c/klein_grounding_features.jsonl
```

`--extract` rebuilds each case's masks the way the probe did and measures, per
instance: `area` (share of the image), `low` (centroid height; lower in the frame is
the stand-in for nearer), `lum` (mean luminance) and `sat` (mean saturation). It first
recomputes the attention densities from the saved maps with those masks and stops if
they differ from what the probe recorded, so a feature cannot be attached to the wrong
instance.

**Question:** when the instruction cannot pick out the referent, is the instance with
the highest attention density also the larger / lower / brighter / more saturated one?
Only the target and its same-category siblings are compared; attention on other
objects or people in the scene, which the rater also saw, is not tested.

**Size is partly built into density.** `pointing` weights each token cell by how much
of it the mask covers. For a map that is flat on objects and flat on background, an
instance's density is `a_bg + (a_obj − a_bg) · Σc²/Σc`, with `c` the per-cell coverage.
That factor is near 1 for a large compact mask and small for one a couple of cells
wide, so equal attention per object pixel still ranks the larger instance higher (the
self-test reproduces a 1.20 ratio from this alone). The script therefore also stores
**interior densities**, the mean over cells a mask covers at least 90%, which do not
have the bias, and prints the primary line a second time with them. This also applies,
unexamined until now, to `pointing` itself wherever target and siblings differ in size.

**Primary, fixed here:** swapped instruction whose source case has a different
category, two-instance cases. That is 60 cases, chance 50%. Four features are tested,
so the bar is p < 0.0125 (exact, two-sided), which is **41 or more of 60** (68%). If
extraction skips a case or a feature ties, n changes and the bar is the p-value.

| result for a feature | reading |
|---|---|
| ≥ 41 of 60 | Attention's referent-independent part goes with that feature. Report referent results split by it (target is the smaller / higher / darker one = hard), and balance it when building referent-only pairs. |
| ≤ 19 of 60 | The opposite preference. Report it as found. |
| 20–40 of 60 | Not shown at this sample size. Do not describe the prior by that feature. |

**Extra condition for `area` and `low`** (they travel together, and both travel with
the coverage factor): a result over the bar is called size or position only if the
interior-only line points the same way, at 60% or more of its cases. Otherwise it is
reported as confounded with token resolution. If the interior line keeps fewer than
30 cases, say so and call neither.

If no feature reaches the bar: stop here. Keep the swapped map itself as the definition of
the hard subset, and do not add features after the fact. The visual impression may
then be total attention (size) rather than density.

Everything else the report prints (single and full instructions, all-instance cases,
which feature attention follows when two disagree, the target-side and edit-side
splits) is context and is not tested against the bar. Large objects also tend to sit
lower in the frame, so two features clearing together need the disagreement table
before either is named. Checked on CPU: the self-test (planted area effect recovered,
planted-null brightness not; a planted effect that is all coverage bias disappears on
interior densities; a swapped mask order is caught) and a dry run of `--report` on the
real attention file with random features, which gave chance-level rows. The extraction
itself has not run.

Exploratory only: pointing is at or below chance in blocks 0 and 3, rises through the
single-stream blocks to about 80% around blocks 12–19, and falls in the last few; it
declines from the first step to the last. Association, not causation: nothing here
shows that fixing attention would fix the edit.

### Joined with the two blind audits (40 cases; per-case file pushed 2026-11-17)

The two audits together give each failed case a type: a plain-arm wrong edit that
paste-back **fixed** was leakage onto siblings; one that paste-back turned into
**no edit** was aimed at the wrong place; one that stayed wrong is a different edit
on the target.

| case type (from the two audits) | n | probe points at target (full) | same, single instruction | target preference (median) |
|---|---|---|---|---|
| clean in the plain arm | 9 | 7 (78%) | 6 (67%) | 1.92 |
| **leak**: wrong, fixed by paste-back | 15 | 12 (80%) | 8 (53%) | 1.32 |
| **mis-aimed**: wrong, no edit after paste-back | 10 | 4 (40%) | 1 (10%) | 0.86 |
| true wrong edit (wrong in both arms) | 3 | 2 | 1 | 2.86 |

- **Leak cases are grounded.** Attention points at the target as often as in clean
  cases (80% vs 78%), and the edit spreads anyway. For leakage, the model finds the
  referent and fails in generation.
- **Mis-aimed cases are not grounded** (40%; target preference below 1, i.e. attention
  favours a sibling).
- **Target preference orders the three outcomes:** 1.92 clean, 1.32 leak, 0.86
  mis-aimed. Leak cases point at the right instance but by a smaller margin. This
  suggests one continuous quantity, how strongly attention separates the target from
  its siblings, behind both failures. Suggestive only: n = 9 / 15 / 10, one rater,
  medians.

So the two failures have different immediate causes, as the split by metric class
already hinted, and both losses are motivated: a sibling-preservation loss for leakage
in grounded cases, a grounding loss (with referent-only pairs) for mis-aimed ones. The
ordering above adds a testable prediction: a grounding loss that widens the
target-vs-sibling margin should reduce leakage as well as mis-aiming.
