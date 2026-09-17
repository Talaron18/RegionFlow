# Gate C — mask-conditioning arms

Companion to `gate_c_note.md` (benchmark construction, metric calibration,
RefEdit-SD3 baseline). Every threshold frozen in that document's §9 is used
here unchanged. Nothing was retuned after a score was seen.

**Verdict: branch A.** Spatial conditioning eliminates wrong-instance editing
and buys no measurable edit quality. Localisation is solved by conditioning;
transformation is the residual, and it is what RegionFlow has to attack.

---

## 1. What was run

Four arms over the same 151 `hard` RefEdit-Bench × COCO cases, same
`case_seed(id, 0)`, so every arm pairs case-by-case with every other.

| arm | checkpoint | `mask_img` |
|---|---|---|
| `none` | `bpathir1/RefEdit-SD3` | none (the completed Gate C run) |
| `white` | `BleachNick/SD3_UltraEdit_w_mask` | blank white — "edit anywhere" |
| `oracle` | same | target instance, dilated 8px |
| `sibling` | same | largest sibling — positive control |

`--mask-dilate 8` frozen from a 4-case smoke run before any arm was scored.

### Why there is no no-mask arm for the mask checkpoint

`SD3_UltraEdit_w_mask`'s transformer takes `in_channels=48` = 16 latent + 16
image + **16 mask**. The mask is a channel block, not an optional argument, so
`--mask-mode none` raises a `ValueError` from the pipeline and no plumbing
fixes it. `white` therefore *is* the no-spatial-information condition for this
checkpoint — it is what UltraEdit's own demo passes when the user draws no
region — and the within-checkpoint comparison is **white → oracle**.

`none → oracle` crosses checkpoints and is reported only as context.

A guard in `run_gate_c.load_model` now refuses `sd3-ultraedit-mask` +
`--mask-mode none` with this explanation, so the hour is not spent twice.

---

## 2. Premise checks — both had to pass before any branch could be read

**P1, does the mask arm composite or condition?** Background change medians:
`none` 0.0238, `white` 0.0316, `oracle` 0.0167, `sibling` 0.0166. All
comfortably nonzero, so the pipeline conditions rather than pasting the source
back outside the mask. Had it composited, `sibling_change` would be zero by
construction and `mis_targeted` identically 0 for arithmetic reasons.

**P2, does the model use the mask?** Pointing it at a sibling drives
mis-targeting 28.5% → 94.0%, discordant 101:2, p < 0.0001. The mask moves the
edit. Without this control, a mis-targeting collapse in the oracle arm and a
model quietly ignoring masks are the same observation.

---

## 3. Headline results (151 cases, prompt = full)

| | `none` | `white` | `oracle` | `sibling` |
|---|---|---|---|---|
| mis-targeted | 43 (28%) | 46 (30%) | **0 (0%)** | 142 (94%) |
| no *tracked* change | 10 (7%) | 16 (11%) | 2 (1%) | 5 (3%) |
| measurable | 98 (65%) | 89 (59%) | 149 (99%) | 4 (3%) |
| valid (K=10) | 38 (25%) | 2 (1%) | 16 (11%) | 0 |
| leakage ratio median (K=10) | 0.082 | 0.857 | 0.154 | — |
| sibling excess median (K=10) | +0.0083 | +0.0884 | **+0.0097** | — |

Reference: ground truth 0.011 / −0.005; category-level 1.001 / +0.157.

McNemar vs `none`: `oracle` 0:43, p < 0.0001. `white` 28:25, p = 0.784.

### Mis-targeting is saturated, not won

0/151 is the measure hitting its floor. `mis_targeted` asks whether sibling
change exceeds target change; a model handed the target region cannot fail
that. **Any** mask-respecting model scores 0 here, so this number motivates the
problem against text-only models and cannot discriminate among
region-conditioned ones. RegionFlow will not be able to post a headline figure
on it, because the row above already reads 0%.

### The number that did not move

Sibling excess over background went **+0.0083 → +0.0097**. Unchanged, if
anything marginally worse. Conditioning eliminated the gross failure — editing
the wrong object entirely — and did nothing for fine contamination of the
instances that were supposed to stay untouched.

Quote the excess, not the ratio. `oracle`'s ratio is higher than `none`'s
(0.154 vs 0.082) largely because oracle edits are lower-amplitude relative to
ambient, which inflates a ratio without any more sibling contamination. The
validity counts cross over across K (oracle 16 vs none 38 at K=10; 128 vs 92 at
K=3), which is the same fact seen from the other side.

`white`'s leakage ratio of 0.857 sits next to the category-level reference of
1.001. With 2 valid cases at K=10 it is not a measurement, but it is consistent
with what the audit found independently: `white` regenerates globally rather
than editing.

---

## 4. Edit quality — the blind audit

120 panels: 40 ids × 3 arms, pooled and shuffled, arm identity and metric
outcome hidden, one rubric, one sitting. Ids carried over from the original
40-case audit so the figures stay comparable. `sibling` excluded — at 94%
mis-targeted the control had already done its job.

| arm | clean | degraded | destroyed | wrong_edit | no_edit | clean % | 95% CI |
|---|---|---|---|---|---|---|---|
| `none` | 6 | 8 | 3 | 0 | 23 | 15% | [7%, 29%] |
| `white` | 3 | 8 | 3 | 3 | 23 | 8% | [3%, 20%] |
| `oracle` | 9 | 12 | 3 | 6 | 10 | **22%** | [12%, 38%] |

Paired on the same 40 ids:

- `none` → `oracle`: not→clean 6, clean→not 3, **p = 0.508**
- `none` → `white`: not→clean 2, clean→not 5, p = 0.453
- `white` → `oracle`: not→clean 7, clean→not 1, **p = 0.070**

### Why this reads as A and not C

The pre-registered key said "at or below ~18% → A" and "well above ~18% → C."
22% fell in the gap between them. **That gap was a defect in the
pre-registration** — a criterion with no number in it, which is the failure
mode this project has otherwise guarded against. It is resolved by the paired
test registered alongside it, which is null at p = 0.508 with intervals
overlapping almost entirely. The within-checkpoint comparison at p = 0.070 also
fails the pre-registered α = 0.01, and is treated the way the 28/35 ablation at
p = 0.064 was treated: suggestive, not stated flatly.

Independent of any p-value: at 22%, **four in five oracle edits are not clean**,
and `destroyed` is unmoved at 3 in every arm.

### `wrong_edit` 0 → 6

Within the blind audit, the no-mask arm never once produced a different edit
than asked inside the target. The oracle arm did so six times. Conditioning
converted inaction into action, and that action landed in the right place with
the wrong transformation. This is the localisation/transformation split
observed directly rather than inferred from rates.

### The white → oracle transition matrix

|  white \ oracle | clean | degraded | destroyed | wrong_edit | no_edit |
|---|---|---|---|---|---|
| clean | 2 | 0 | 0 | 1 | 0 |
| degraded | 3 | 3 | 1 | 1 | 0 |
| destroyed | 0 | 1 | 1 | 0 | 1 |
| wrong_edit | 0 | 1 | 0 | 1 | 1 |
| no_edit | **4** | **7** | **1** | **3** | 8 |

15 cases left `no_edit`; 4 reached `clean` and 11 did not. 8 stayed inert even
with the mask.

**The mask is a trigger, not a steer.** Told "edit anywhere," the model usually
does nothing. Told "edit here," it acts — and roughly three quarters of the
time what it does is not the requested edit.

This observation is **exploratory**. The `no_edit` test was constructed after
seeing the marginals it explains, and §7 records two failed attempts to confirm
it on held-out data. It carries the weight of the 28/35 ablation, no more.

---

## 5. VLM-judged edit quality (RefEdit-SD3 baseline, `none` arm)

`bench/edit_quality_vlm.py`, VIEScore-adapted (TIGER-AI-Lab, ACL 2024,
arXiv:2312.14867): SC (semantic consistency, 0–10), PQ = sqrt(naturalness ×
artifacts), Overall = sqrt(SC × PQ) — all three restricted to the
yellow-outlined target region only, same outline convention as §4's human
panels. This is a re-scorer: the 302 edits (151 cases × {full, single} prompt)
already existed on disk from the completed `none`-arm run; nothing here does
new inference on the editing model. **Only the baseline `none` arm has been
scored this way — `white`/`oracle`/`sibling` have not, so this section adds a
quality axis to the baseline, not a fourth arm to §3's or §4's comparisons.**

Judge: `Qwen/Qwen3.6-35B-A3B` via SiliconFlow, picked as a budget substitute
after confirming with a real image probe (not just its presence in
SiliconFlow's model listing) that it actually reads pixels rather than
ignoring the image. **Not a claim of parity with RefEdit-Bench's own GPT-4o
judge** — see the module docstring for what is and is not confirmed against
the VIEScore paper itself (arxiv.org is blocked from the training sandbox, so
the prompt protocol was confirmed against the VIEScore repo's own prompt file,
not the paper).

### Sanity check, and it passes

`mis_targeted` cases score `Overall = 0.00` at the median, in both prompt
kinds (43/151 full, 53/151 single — consistent with §3's 28%/30%-scale
mis-targeting figures for `none`/`white`). This is the target-region
restriction working as intended: when the edit lands on the wrong instance,
nothing happens inside the yellow-outlined *target* region, so SC correctly
reads zero there. Not assumed — checked.

### When the metric says `measurable`, quality is decent

| prompt | n | median Overall |
|---|---|---|
| full | 98 | 6.43 |
| single | 90 | 7.20 |

Edits that land on the right instance tend to look reasonably good by VLM
judgment, not merely "technically located there."

### The caveat every §3 leakage figure now carries

| prompt | valid + measurable (n) | median Overall | Overall < 5.0 |
|---|---|---|---|
| full | 34 | 5.46 | 16/34 = 47% |
| single | 29 | 6.17 | 11/29 = 38% |

`valid + measurable` is exactly the subset that feeds §3's leakage-ratio and
sibling-excess figures — already flagged there as resting on very few cases
(§7). Nearly half of those cases (47% full, 38% single) score below VIEScore's
own rough failure line (Overall < 5.0). The leakage numbers are not only thin;
a large share of the edits producing them are independently judged poor
quality. **Quote every leakage figure in §3 alongside this distribution, not
instead of it.**

### One discrepancy, not resolved

`no_tracked_change` median Overall is 6.75 (full, n=10) vs 3.08 (single, n=8)
— the same outcome class reading very differently depending on prompt kind, at
a sample size too small to adjudicate (n=8–10). Possibly noise, possibly a
real difference in what "no tracked change" means between full and
single-object instructions. Flagged, not interpreted.

### Scope limits specific to this section

- **Judge substitution is unvalidated against GPT-4o.** Everything above is
  Qwen3.6-35B-A3B's reading, not a reproduction of RefEdit-Bench's own
  published numbers.
- **`parse_scores` is a lenient number-extractor**, not VIEScore's own exact
  output parser (unconfirmed from this sandbox — see module docstring).
- **Baseline only.** No VLM-judged comparison across `white`/`oracle`/`sibling`
  exists yet; extending it is the natural next step if the mask arms' edits
  are re-scored the same way.

---

## 6. The two audits disagree, and the disagreement is a finding

The original audit (§14 of the note) and this blind audit judged **the same 40
outputs of the same model**. They agree on almost nothing.

| verdict | unblinded | blind | Δ |
|---|---|---|---|
| clean | 7 | 6 | −1 |
| degraded | 9 | 8 | −1 |
| destroyed | 2 | 3 | +1 |
| wrong_edit | **22** | **0** | **−22** |
| no_edit | **0** | **23** | **+23** |

The clean counts nearly matching is coincidental agreement in one cell, not
evidence of a stable rubric. Reading it as a replication — as was done on first
inspection — is wrong.

Two differences could produce this. The rubric was restated between sittings,
specifically routing "right object, wrong result" to `wrong_edit`. And the
original audit was **unblinded**: `quality_audit` names each panel
`<id>_<outcome>.png` and captions it with the metric's outcome class,
`target_change`, `leakage_ratio` and `valid`. The blind deck shows the
instruction and nothing else.

An anchoring effect is the natural explanation for a wholesale
`wrong_edit` → `no_edit` migration: a panel labelled `measurable` with a large
`target_change` invites the reading that something happened. It cannot be
proven from this data, and it need not be — the blind audit is preferable by
construction and **supersedes §14's composition**. The 18% and its 55%
`wrong_edit` breakdown should not be quoted again.

Consequences:

- Every cross-audit comparison is void. All quality claims here are
  within-blind-audit, which is why branch A survives: `none` 15% and `oracle`
  22% were judged in the same sitting under the same rubric.
- The blind key is reconstructible from a fixed `shuffle_seed`. Blinding rested
  on the file not being opened, not on it being unrecoverable. State this.
- Future audits use `blind_audit.py`. `quality_audit --sample` prints the
  metric's verdict on the image being judged.

---

## 7. Dead ends — do not retry these

### The metric cannot detect "an edit happened" independently of "it happened here"

Two automated proxies for the human `no_edit` verdict were validated on the
audited 40 and tested on the 111 unaudited cases. Both were confounded, and the
confound is structural.

**Absolute `target_change < 0.03`** (81% precision on the 40). Held out: 17
inert→active, 0 reversals, p < 0.0001. Diagnostic: oracle's `target_change`
exceeds white's in **86%** of all 151 cases, median shift +0.018. The 17:0 is a
threshold slicing a distribution that shifted as a whole. Compounding it,
`white`'s target-to-background ratio is 1.83, so at 0.03 the test largely asks
whether a case's ambient regeneration sat above or below the line.

**Ratio `target_change / background_change < 1.5`** (81% precision, recall 46%;
threshold frozen for precision before the held-out run). Held out: 37:0,
p < 0.0001. Diagnostic: oracle's ratio exceeds white's in **100%** of 111,
median shift +3.14. Worse than the first: a target-over-background ratio *is* a
concentration measure, and concentrating change inside the target is exactly
what mask conditioning does by construction.

**The general statement.** Any pixel-difference proxy for "did an edit occur"
compares change in the target against change elsewhere, and that comparison is
the quantity spatial conditioning directly manipulates. For a mask-conditioned
model the instrument cannot separate *acted* from *acted here*. This is §17's
blindness generalised from edit quality to edit occurrence, and it constrains
how RegionFlow itself can be evaluated later.

At the absolute threshold the proxy never exceeded 55% recall at usable
precision. The human `no_edit` verdict sees something the pixel metric largely
cannot.

### Off-instance measure has no null floor yet

`off_instance.py` (new) reports peak windowed change outside the tracked
instance set, over the ambient median. Arms: `white` ratio 5.47 / over-target
1.40, louder than the target in 106/151; `oracle` 4.01 / 0.46, 16/151;
`sibling` 3.91 / 1.95, 126/151. The `none` arm predates the measure.

Under a noiseless uniform shift the null ratio is 1.0, but real regeneration is
not uniform and the max over ~700 windows will exceed the median under the null
too. **Report `over_target` only**; the ratio needs a ground-truth floor from
the MagicBrush pairs, exactly as `leakage_ratio` has one.

---

## 8. Scope limits — state with any result

Everything in §10 of the note still applies. Additionally:

- **Mis-targeting is saturated for region-conditioned models.** It cannot rank
  them.
- **`none` vs `oracle` crosses checkpoints.** The matched free-form partner is
  UltraEdit's `*_wo_mask` checkpoint, not RefEdit-SD3. Not yet run.
- **The audit is 40 ids.** The 20 additional ids were not drawn: `read_arm`
  collapsed both prompt kinds into one dict before filtering, emptying the
  pool. Fixed; the audit was not rerun because rebuilding the deck would
  invalidate the completed verdicts.
- **Nine discordant pairs.** The clean-rate comparison is underpowered to
  detect a moderate improvement. "No detectable improvement", not "no
  improvement."
- **Leakage figures rest on very few cases.** Among the audited, valid +
  measurable is n=8 / n=4 / n=1 for none / oracle / white. Those cells are
  unusable and were not interpreted. §5 quantifies the same thinness on the
  full 151-case `none` run, independently, via VLM judgment rather than
  the audited-40 subset.

---

## 9. The defensible claim

> Conditioning a flow-matching editor on the target instance mask eliminates
> wrong-instance editing entirely (28% → 0% mis-targeted, 151 paired cases),
> and buys no measurable improvement in edit quality (15% → 22% clean, 40
> blind-audited paired cases, p = 0.51; four in five outputs remain unclean,
> and `destroyed` is unchanged). Leakage onto same-category instances that were
> designated to stay untouched is likewise unchanged (+0.0083 → +0.0097). The
> mask determines *where* the model edits and not *what* it does there: within
> one checkpoint, conditioning moved 15 of 40 cases out of "no edit", of which
> 11 produced something other than the requested edit.

Lead with mis-targeting and the unchanged quality. Do not lead with a leakage
magnitude, and never quote one without its K.

**Positioning.** Every 2026 method in this space — instance-disentangled
attention (2602.08749), MIRAGE (2604.05180), biased joint attention
(2606.31278) — is an inference-time intervention that routes a frozen field to
the right region. This result measures the ceiling of that family on a
benchmark none of them can be scored on, because no existing benchmark
(LoMOE-Bench, MIRA-Bench, InfoEdit) leaves any instance unedited. Routing
works; routing is not enough.

---

## 10. Bugs worth not repeating

- **`{r["id"]: r for r in recs}` before filtering `prompt_kind`.** A results
  file that ran `--ablation` holds both kinds; the dict keeps whichever came
  last. Silently emptied the audit sample pool and put stripped-prompt metadata
  on full-prompt panels. Filter, then key.
- **`in_channels=48`.** Check a checkpoint's transformer config before writing
  an arm that assumes an argument is optional.
- **Default paths copied between tools.** `--bench` was set to a PNG directory
  rather than the parquet. Failed loudly; the dangerous version is the one that
  succeeds on the wrong input.
- **Judging one cell and calling it a replication.** See §6.

---

## 11. Next

1. UltraEdit `*_wo_mask` arm — the matched free-form partner, one hour, turns
   the cross-checkpoint context row into a clean ablation.
2. MagicBrush ground-truth floor for `off_instance`.
3. Δ-order curriculum (`delta_curriculum.py`) — three arms at ~2 h. Now
   motivated by this gate: direction is the residual it isolated, and the
   anchor-SSE climb is the one lead anyone has on it.
