# Does klein know where the referent is? — diagnostic scope

Status: **scope only; nothing built or run.** Readings are fixed here first.

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
