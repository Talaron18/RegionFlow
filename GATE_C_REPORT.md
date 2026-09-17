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

**Added after the above, not part of the original pre-registration**: a
fifth arm, `freeform` (`BleachNick/SD3_UltraEdit_freeform`, `--mask-mode
none`) — UltraEdit's own base checkpoint, the free-form partner this section
originally said was not yet run (§11). It closes the `none`/`oracle`
cross-checkpoint gap; results are in §3.

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

| | `none` | `white` | `oracle` | `sibling` | `freeform` |
|---|---|---|---|---|---|
| mis-targeted | 43 (28%) | 46 (30%) | **0 (0%)** | 142 (94%) | 60 (40%) |
| no *tracked* change | 10 (7%) | 16 (11%) | 2 (1%) | 5 (3%) | 12 (8%) |
| measurable | 98 (65%) | 89 (59%) | 149 (99%) | 4 (3%) | 79 (52%) |
| valid (K=10) | 38 (25%) | 2 (1%) | 16 (11%) | 0 | 1 (0.7%) |
| leakage ratio median (K=10) | 0.082 | 0.857 | 0.154 | — | 1.121 |
| sibling excess median (K=10) | +0.0083 | +0.0884 | **+0.0097** | — | +0.1634 |

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

### `freeform` → `oracle`: the matched ablation §8 flagged as missing

`sd3-ultraedit-freeform` is UltraEdit's own base checkpoint — the one
`oracle`'s `SD3_UltraEdit_w_mask` and `none` (RefEdit-SD3) both descend from,
before RefEdit-SD3's referring-expression finetuning. Scoring it closes the
"`none` vs `oracle` crosses checkpoints" gap §8 flagged as not yet run.

It is not a gentler baseline than `none` — it is worse than `white`. 40%
mis-targeted (vs `white`'s 30%, `none`'s 28%), 0.7% valid at K=10 (vs
`white`'s 1%), and every one of its cases has an off-instance hot spot louder
than the target (§7's `off_instance` measure, `over_target` median
1.27 full / 1.34 single). With no finetuning and no mask, this checkpoint
barely respects instance boundaries at all — worse than even the
"edit anywhere" `white` control.

`freeform → oracle`, same checkpoint family, is now the cleanest version of
this report's headline result: **40% mis-targeted → 0%**, a bigger swing than
`none → oracle`'s 28% → 0%, with no cross-checkpoint caveat attached. Mask
conditioning did not just clean up an already-tuned model's residual leakage
— it took a checkpoint with no instance-selectivity behaviour at all and
eliminated wrong-instance editing completely.

**McNemar confirms it: 60:0 discordant, p < 0.0001.** Every one of
`freeform`'s 60 mis-targeted cases becomes correctly targeted under `oracle`;
zero go the other way — a more extreme, more one-sided result than
`none → oracle`'s own 0:43. (P2 — does the model actually use the mask,
rather than a real effect being confounded with a model that ignores masks
entirely — was already established in §2 via the `sibling` positive control
on this same `oracle` checkpoint; `compare_arms.py`'s own P2 warning on a
2-arm invocation is about that invocation, not a gap in this report.)

Leakage/excess figures for `freeform` rest on 1 valid case at K=10 — worse
than `white`'s 2, same "not a measurement" caveat applies.

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

Three instruments now tile the space: `instance_selectivity` measures WHERE
change landed, this VLM axis measures how good it was GIVEN that it landed
there, and `off_instance` (§7, still missing its null floor) measures whether
anything happened elsewhere. That structural independence is real and worth
having. The calibration check below is what determines whether the middle
instrument can be trusted yet — right now it cannot.

### Sanity check: real, but not as clean as a median suggests

`mis_targeted` cases score `Overall = 0.00` at the **median**, in both prompt
kinds (43/151 full, 53/151 single). But the median hides a bimodal split, not
a clean zero:

| prompt | n | Overall = 0.0 | Overall in 9–10 |
|---|---|---|---|
| full | 43 | 23 (53%) | 14 (33%) |
| single | 53 | 27 (51%) | 17 (32%) |

About half read as intended (nothing happened in the target region). The
other third score as good as any successful edit. This is not a parsing bug —
all 302 rows are present with no duplicates (0% parse-failure rate, checked
directly against the resumable output). It is `mis_targeted`'s own definition:
`outcome_class` calls a case `mis_targeted` when `sibling_change >
target_change`, not when target_change is near zero. A case where both the
target AND a sibling were genuinely edited, sibling slightly more, is
correctly flagged `mis_targeted` by the pixel metric while still showing a
real, well-executed edit inside the yellow region. **Restate accordingly: the
sanity check confirms the target-region restriction works, not that
`mis_targeted` implies an untouched target.**

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

### Calibration against the 40 human blind verdicts: it fails (first attempt)

The `none` arm already has 40 blind-audited human verdicts (§4). Joining VLM
`Overall` (full prompt) against them by id is a free check with no GPU and no
new API cost — and a stronger warrant than parity with GPT-4o would be,
since the blind-audit rubric, not GPT-4o's, is what this report's own §9
claim rests on.

| human verdict | n | median Overall |
|---|---|---|
| clean | 6 | 3.46 |
| degraded | 8 | **7.59** |
| destroyed | 3 | 3.57 |
| no_edit | 23 | 0.00 (mean 3.38) |

`clean` scores *below* `degraded` — inverted, not just noisy: Mann-Whitney
U = 15.5 against a maximum of 48 (n=6 vs n=8), meaning clean cases mostly
rank lower than degraded ones, not scattered around them. Spearman ρ between
Overall and an ordinal human-quality scale (`no_edit`=0 → `clean`=4) across
all 40 cases is **0.170** — negligible.

`no_edit` (n=23) has a separate problem: median is 0 as the sanity check
above would predict, but 7/23 (30%) score ≥9.4, including a perfect 10.0 on
**refedit-172** — the same case where both the pixel metric and the blind
human verdict independently agree nothing happened. Its instruction ("Add a
bird to the bench next to the bench with **the bird on it**") describes a
bird that already exists elsewhere in the source frame; the working
hypothesis is the judge's SC score is reading that existing bird rather than
staying inside the yellow-outlined target region, but this is unconfirmed —
`edit_quality_vlm.py` does not persist the judge's raw text, only the parsed
numbers, so checking it needs a targeted re-query of just these 7 cases.

**Consequence: do not extend this judge to `white`/`oracle` to strengthen §9's
low-power quality claim.** That would report a more "powered" number built on
an instrument shown not to track the ground truth it is meant to proxy for —
worse than the existing 40-case audit, not better. Before any re-scoring at
n=151 means anything, either this judge needs to pass the calibration above,
or a different judge does.

### Re-calibration after `crop_to_target`: better, not sufficient

Visual inspection of 7 of the misfiring `no_edit` cases (images pushed to
`tmp/` on this branch at the time, not retained in the repo) found the judge
crediting content ANYWHERE in the full frame to the target, by at least three
distinct mechanisms — a decoy object named in the instruction's own landmark
clause recoloured (`refedit-172`), the SIBLING's own edit credited to the
target (`refedit-169`), an unrelated background change credited to the
target (`refedit-165`, `refedit-126`), a frisbee near the sibling credited to
the target (`refedit-73`), and a target left completely untouched by an
"erase" instruction scored a perfect 10.0 (`refedit-81`). `crop_to_target`
crops the panel to the target's own padded bounding box before the judge
ever sees it, mechanically removing the option instead of relying on
instruction-following for it (`edit_quality_vlm.py`). Two more issues
surfaced alongside it and were also fixed: response variance from an unset
`temperature` (0.7 default; the same panel+prompt scored 4.3 then 0.0 on
`refedit-144`), fixed by pinning `temperature=0.0`; and a case
(`refedit-172` again) where the judge reasoned to the CORRECT answer in
text — *"within the yellow region, no change occurred... the edit did not
occur inside the yellow region"* — then kept re-litigating it for the full
600-token budget and never emitted `[[score]]`, fixed with a closing
"commit and stop" instruction appended to both prompts.

Re-running the full 40-case calibration with all three fixes:

| human verdict | n | median Overall (before → after) |
|---|---|---|
| clean | 6 | 3.46 → 4.44 |
| degraded | 8 | 7.59 → 4.93 |
| destroyed | 3 | 3.57 → 2.06 |
| no_edit | 23 | 0.00 → 0.00 (mean 3.38 → 2.72) |

**The clean/degraded inversion is resolved.** Mann-Whitney U went from
15.5/48 (clean scoring mostly *lower* than degraded — backwards) to 22/48,
essentially the null midpoint of 24 — no detectable difference in rank
either way, which is at least the right qualitative shape even if it isn't
separating them cleanly yet. Spearman ρ against the ordinal human-quality
scale went **0.170 → 0.333** — meaningfully better, still weak.

**A second failure mode survived the fix, and the original 7-case check
never saw it.** Of the 23 `no_edit` cases, 6 still score ≥9.0 — one is
`refedit-172` (already attributed to a real limitation of cropping itself:
its instruction is a *compound* reference, "the bench next to the bench
with the bird on it," which structurally requires seeing both benches to
verify, and a tight crop around only the target can starve exactly the
context a relational instruction needs). The other five are new:
`refedit-117` (9.0), `refedit-170` (9.0), `refedit-119` (9.5), `refedit-160`
(9.5), `refedit-29` (9.5). None of these were in the original 7 checked by
eye, so their failure mechanism is not yet known — this is the open item,
not a closed one. `refedit-126` (4.5, was 9.7) similarly stopped being a
clean hallucination and became a genuine judgment call: the raw SC text now
reasons correctly inside the yellow region and calls a plain sky/cloud
screen image a "scenic beach wallpaper," which a stricter rater might
reasonably not.

**Still the same consequence as the first attempt, for a different reason.**
ρ=0.333 and an unexplained ~22% high-scoring failure rate on `no_edit` are
not grounds to extend this judge to `white`/`oracle` at n=151 yet. The
crop fix is real progress — worth keeping — but the next step is
diagnosing `refedit-117`/`170`/`119`/`160`/`29` the same way as the
original 7 (cropped panel + raw SC text) before deciding whether this needs
another mechanical fix or a different judge model entirely.

### Diagnosing the five: one clear rule violation, not a single mechanism

Visual inspection (panels pushed the same way as before) of the 5 cases
above found a mix, not one bug:

- **`refedit-119`** (a red scarf added to the RED dog, not yellow) —
  genuinely fixable. The judge's own text: *"the instruction specifically
  targeted the passenger seat dog (red outline)... the edit was applied
  correctly to that region."* It independently re-derived which instance
  "the dog in the passenger seat" refers to from the instruction's wording,
  decided that meant the red one, and scored success against its own
  re-derived target instead of the yellow one it was told to trust —
  overriding region restriction outright when its own reading of the
  instruction disagreed with the case's target assignment.
- **`refedit-117`** (suit) — looks like real fabrication: the visible change
  is only a suit-colour shift, but the judge described a full gender change
  and a "purple top" not clearly present in the pixels.
- **`refedit-160`** (microwave) and **`refedit-29`** (parrot) — real edits,
  imprecisely credited, not hallucinated: the microwave was fully replaced
  with a different-looking one (not "light off," which is what the judge
  called it), and a parrot really was added, but sitting on the counter near
  the chef's hand, not on his shoulder as instructed.
- **`refedit-170`** (tattoo) — too subtle to tell from a compressed image
  whether a real mark is there.

Added one sentence to `SC_PROMPT` making the yellow outline authoritative
regardless of what the instruction's wording seems to suggest, targeting
`refedit-119`'s mechanism directly. Re-testing the same 5:
`refedit-119` 10.0→0.0 and `refedit-170` 8.0→0.0 (fixed, the second
unexpectedly), `refedit-29` 10.0→4.2 (partial), `refedit-160` 10.0→9.5 and
`refedit-117` 9.0→8.5 (unaffected, as expected — neither is an
outline-authority problem). Running tally across all 12 cases checked by
eye so far: 7/12 clean, 2/12 partial, 3/12 still bad
(`refedit-172`/`160`/`117`), each with a distinct, identified cause. None
of the three remaining look fixable by further prompt wording.

### Switching judges: `Qwen/Qwen3.6-35B-A3B` → `zai-org/GLM-5.3-Flash`

Re-ran the 40-case calibration (36/40 obtained; `refedit-188`/`153`/`28`/`85`
missing from an interrupted run, not yet backfilled) with GLM-5.3-Flash as
the judge — a reasoning-only model, so `enable_thinking=False` (400s on it)
was swapped for `thinking_budget="low"` in `siliconflow_judge`. This run
already carries all four fixes above (crop/temperature/commit/outline
-authority live in the shared prompt and `judge_case` code, not per-model
logic), so it is the most-fixed comparison point either judge has had.

| human verdict | n | Qwen (v2, 3 fixes) | GLM-5.3-Flash (n=36, all 4 fixes) |
|---|---|---|---|
| clean | 6 / 5 | median 4.44 | median **9.50** |
| degraded | 8 / 7 | median 4.93 | median 5.20 |
| destroyed | 3 / 2 | median 2.06 | median 5.45 |
| no_edit | 23 / 22 | median 0.00 (mean 2.72) | median 0.00 (mean 2.93) |

**Better, on the two numbers that matter most.** Spearman ρ = **0.527**
(vs Qwen's 0.333) — clean/degraded is now correctly *ordered*, not merely
un-inverted: clean's median (9.5) sits above degraded's (5.2), where Qwen's
v2 result only got as far as "no detectable difference either way."

**Not better on the specific failure this section has been chasing.**
`no_edit` cases scoring ≥9.0: 5/22 (23%) for GLM vs 6/23 (26%) for Qwen —
essentially the same rate, including `refedit-172` still at 9.5 (consistent
with that being a structural limit of cropping itself, not a model-specific
failure, since it persists across models). One new false positive appears
(`refedit-138`, 9.5) not seen in Qwen's run.

**Net effect on the consequence stated above, at the time: unchanged** —
see below, this has since changed.

### Full backfill (n=39): ρ jumps to 0.717, ordering clean across all four classes

The 4 missing cases (`refedit-188`/`153`/`28`/`85`) were backfilled for a
complete run: 39/40 scored (1 API failure), same judge, same four fixes,
no further prompt changes.

| human verdict | n | mean | median |
|---|---|---|---|
| clean | 6 | 6.41 | 6.74 |
| degraded | 7 | 5.59 | 5.48 |
| destroyed | 3 | 2.53 | 2.65 |
| no_edit | 23 | 1.23 | 0.00 |

**Spearman ρ = 0.717** (n=39), up from the n=36 partial run's 0.527 and far
past Qwen's v1/v2 (0.170 / 0.333). The ordering is now clean on both mean
and median across all four classes, not just clean-vs-degraded: every
median step (clean → degraded → destroyed → no_edit) moves the right
direction, and `no_edit`'s median of exactly 0.00 with a low mean (1.23,
pulled up by a small number of outliers rather than a systematic bias) is
the first time this judge has read `no_edit` as unambiguously low rather
than bimodal.

**The specific failure this section was chasing is mostly gone.** Only 2 of
39 cases score ≥9.0 in this run (`refedit-134` at 9.22, `refedit-83` at
9.08), down from 5 of 22 `no_edit` cases alone in the partial run.
`refedit-172` — the compound-reference case flagged earlier as a
*structural* limit of cropping, expected to persist across judges — now
scores **0.0**, not 9.5. Whether that is the outline-authority fix, the
model switch, or both is not separated here; it is no longer a live
concern either way.

### Diagnosing the 2 remaining high scorers: one is not a failure, one is a new mechanism

Rebuilt the exact cropped panel each judge call saw and captured the raw
SC/PQ text (not just the parsed score), same method as before, and joined
both against `blind_key.json`/`blind_ledger.json` for the human verdict.

**`refedit-134` is not a judge failure.** Human verdict: **clean**.
Instruction "Add a cap to the man with a black sweater" — the edit adds a
cap that reads as genuinely well-integrated (pose, lighting, shading all
consistent). SC=9.5, and the raw text describes exactly what happened in
the crop. This is the judge being right, not a residual bug — the earlier
framing of "2 unresolved high scorers" overcounted by one. (Separately:
the automated metric marked this case `valid: False`, leakage_ratio
0.373, while a human called it clean — a metric/human disagreement worth
noting but orthogonal to the VLM judge.)

**`refedit-83` is a real false positive, and it is a new mechanism.**
Human verdict: **degraded**. Instruction: "Change the woman in a
checkered shirt to a woman in a plain blue shirt." The edit did change the
shirt to blue — but it also replaced the woman's face, hair, and
expression with a different person's. SC scored this **10.0**
("perfectly aligns with the instruction... preserving the surrounding
context and the person's pose") and PQ scored it high too (naturalness
8.5) — both prompts check whether the *instructed* attribute changed and
whether the pixels look artifact-free, and neither asks whether the
edited instance is still recognizably *the same individual* apart from
the requested change. An attribute-edit instruction that swaps identity
along the way currently reads as a clean success by this rubric.

This is a different mechanism from every prior failure diagnosed in this
section (outline override, fabrication, imprecise credit, cropped-context
loss) — none of those were about identity preservation specifically. It
is one case. Per this project's own standard, one case does not license a
prompt change (a `refedit-119`-style "add an instruction" fix here would
be curve-fitting a rubric to n=1). Not acted on. **Watch for the same
pattern** — instructed attribute correct, but the subject's identity
drifted — when `white`/`oracle` are scored; if it recurs, that is the
threshold to add an identity-preservation check to `PQ_PROMPT`, mirroring
how the outline-authority sentence was added only after `refedit-119` and
`refedit-170` both showed the same override.

**Net effect on the consequence stated above: reversed, with one caveat
now named.** This is no longer just "better correlation" — it is a judge
that orders all four verdict classes correctly and has resolved its worst
failure mode down to one confirmed miss in 39 cases, with that miss's
mechanism identified rather than mysterious. Extending it to `white`/
`oracle` at n=151 is now defensible in a way it was not at ρ=0.527; that
extension has not been run yet (this section still covers `none` only),
and doing it is the natural next step (§11) — watching for identity-drift
false positives in that run is now a specific, not just general, thing
to check.

### One discrepancy, not resolved

`no_tracked_change` median Overall is 6.75 (full, n=10) vs 3.08 (single, n=8)
— the same outcome class reading very differently depending on prompt kind, at
a sample size too small to adjudicate (n=8–10). Possibly noise, possibly a
real difference in what "no tracked change" means between full and
single-object instructions. Flagged, not interpreted.

### Scope limits specific to this section

- **The current judge (GLM-5.3-Flash) is now calibrated on the full
  backfilled `none` set: ρ = 0.717 (n=39), all four verdict classes
  correctly ordered on both mean and median, and only 1 confirmed false
  positive (`refedit-83`, of the 2 candidates — `refedit-134` turned out
  to be a correctly-scored clean edit).** This is a real improvement over
  the n=36 partial run (ρ=0.527) and over Qwen (ρ=0.170 → 0.333), not
  just a larger sample of the same result. It is still calibrated on
  `none` only — `white`/`oracle`/`sibling`/`freeform` have not been
  VLM-judged at all. The one confirmed miss has a named mechanism
  (identity drift on an attribute-edit instruction, not checked by either
  SC_PROMPT or PQ_PROMPT) rather than being unexplained, but it rests on
  n=1 and was not acted on for that reason — see the diagnosis above.
  Treat every number in this section as this judge's own read on the
  `none` arm, now reasonably well-validated there, but not yet extended.
- **Judge substitution is unvalidated against GPT-4o**, separately from the
  calibration limit above. Everything in this section is GLM-5.3-Flash's
  reading, not a reproduction of RefEdit-Bench's own published numbers.
- **`parse_scores` is a lenient number-extractor**, not VIEScore's own exact
  output parser (unconfirmed from this sandbox — see module docstring). Ruled
  out as the explanation for the calibration failure: parse-failure rate is
  0/302, checked directly.
- **Baseline only.** No VLM-judged comparison across `white`/`oracle`/`sibling`
  exists — and per the calibration result above, extending this judge to them
  is not advisable until it is recalibrated or replaced.

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

### Off-instance measure has no null floor yet — now it does, and the ratio fails it

`off_instance.py` (new) reports peak windowed change outside the tracked
instance set, over the ambient median. Arms: `white` ratio 5.47 / over-target
1.40, louder than the target in 106/151; `oracle` 4.01 / 0.46, 16/151;
`sibling` 3.91 / 1.95, 126/151; `freeform` 4.21 / 1.27, 100/151 (full prompt).
The `none` arm predates the measure.

Under a noiseless uniform shift the null ratio is 1.0, but real regeneration is
not uniform and the max over ~700 windows will exceed the median under the null
too. `calibrate_metric.py` now computes `off_instance_change` alongside
`leakage_ratio` on the same MagicBrush×COCO controls, including the real
`ground_truth` pairs (not a synthetic control): **`ratio` median 83.27,
`over_target` median 0.619, n=8 of 9 valid.**

That ratio is not a typo. Real, human-approved MagicBrush edits — edits with
no mis-targeting, no instance-selectivity problem, nothing an auditor would
flag — read a peak/ambient ratio nearly **8× the frozen
`OFF_INSTANCE_EDIT_RATIO = 10.0`** that `run_gate_c.outcome_class` uses to
call a case `off_instance_edit`:

```python
if oi_peak > OFF_INSTANCE_EDIT_RATIO * (oi_ambient + 1e-9):
    return "off_instance_edit"
```

A real edit's own floor clears that threshold by roughly an order of
magnitude. This means the ratio-based classifier is very likely over-firing
`off_instance_edit` on genuinely good, well-localized edits throughout §3/§6 —
the false-positive rate for this specific failure mode has not been
quantified, but the ground-truth floor being this far above the cutoff is
itself the finding, independent of any single arm's count. This is the
concrete case the `off_instance.py` docstring warned about before there was
a floor to check it against: **use `over_target`, not `ratio`, as the
detection statistic.** `over_target`'s own ground-truth median (0.619) sits
well inside the range real arms report (0.46–1.95 above) and does not need
an order-of-magnitude correction to be legible.

The other four controls (`oracle_local`/`category_level`/`mis_targeted`/
`global_shift`) all still pass their `off_instance` self-test bounds
(sibling/category changes read `ratio < 0.1`; uniform shift reads
`0.5 < ratio < 2.0`) — only the real-data floor exposes the gap, which is
exactly why a synthetic control alone was never going to catch this.

### The decision: switch to `over_target`, threshold 1.0

`outcome_class`'s off-instance branch now reads:

```python
if r.get("off_instance_over_target", 0.0) > OFF_INSTANCE_EDIT_OVER_TARGET:
    return "off_instance_edit"
```

`OFF_INSTANCE_EDIT_OVER_TARGET = 1.0` (`off_instance.py`), replacing
`OFF_INSTANCE_EDIT_RATIO = 10.0`. Why `over_target` and not a recalibrated
ratio: `ratio` divides by `off_instance_ambient`, a property of how noisy
*this model's regeneration* is — which is exactly what made it read fine on
the synthetic self-test's clean frame and 83x on real MagicBrush noise. It
has no portable scale across datasets. `over_target` divides by the case's
own `target_change` instead — same units, no cross-case or cross-dataset
denominator to recalibrate. 1.0 reads literally: *more happened off-instance
than happened at the target itself*.

1.0 is bracketed the same way 10.0 originally was, now from two independent
sources instead of one: synthetically, `off_instance.py`'s own self-test
(genuine off-instance edit reads `over_target > 20`; clean target edit reads
`< 0.1`; boundary bleed, newly asserted, reads `< 0.1`) and, for the first
time, against real data (`ground_truth`'s median of 0.619 sits comfortably
under 1.0, with an order of magnitude of margin either side matching the
synthetic controls' own spread). `OFF_INSTANCE_EDIT_RATIO` stays defined —
`off_instance_ratio` is still a reported diagnostic — but is no longer read
by any classifier; `stratify_analysis.py`'s own fallback copy of
`outcome_class` (used only when its `run_gate_c` import fails) was updated
in lockstep, the same discipline its docstring already demanded of itself
after the 47-vs-43 disagreement.

**What this does and does not change in §3/§6.** `mis_targeted`,
`measurable`, `valid`, and every leakage figure never touch the off-instance
branch and are unaffected. §7's own "louder than the target" counts above
(106/151 `white`, etc.) already used `over_target > 1` unconditionally, not
gated through `outcome_class`, and are unaffected too. The one number that
*can* move is §3's "no *tracked* change" row, which is `outcome_class`'s
`no_tracked_change` count specifically (a case only reaches the off-instance
branch at all when target and sibling both read at background) — some of
those 10/16/2/5/12 cases may now reclassify as `off_instance_edit`. Not
recomputed in this pass — `recompute_outcomes.py` (CPU-only, reads the
existing per-arm results files, no new inference) gives the corrected
counts; §3 should be updated once it's run.

---

## 8. Scope limits — state with any result

Everything in §10 of the note still applies. Additionally:

- **Mis-targeting is saturated for region-conditioned models.** It cannot rank
  them.
- **`none` vs `oracle` still crosses checkpoints** (RefEdit-SD3 vs
  `SD3_UltraEdit_w_mask`) and that comparison is reported only as context, as
  it always was. §3 now also has `freeform` (UltraEdit's own base
  checkpoint) vs `oracle` as a same-checkpoint-family ablation, which is the
  one to cite for the headline claim — this bullet used to say that
  comparison had not been run; it has.
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

**The mis-targeting half of this claim now has a cleaner, same-checkpoint
version**: `freeform → oracle`, both `SD3_UltraEdit` variants, goes
40% → 0% mis-targeted (§3) — a bigger swing than the `none → oracle` figure
above, and without that comparison's cross-checkpoint caveat. Use `freeform →
oracle` when the claim needs to be checkpoint-controlled; the quality half of
the claim (15% → 22% clean, p = 0.51) still rests on `none` and `oracle` only,
since `freeform` has not been through the blind audit.

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

1. ~~UltraEdit `*_wo_mask` arm~~ — done (§3): `sd3-ultraedit-freeform`,
   40% mis-targeted, worse than `white`. ~~Formal McNemar for `freeform` vs
   `oracle`~~ — done: 60:0 discordant, p < 0.0001 (§3).
2. ~~MagicBrush ground-truth floor for `off_instance`~~ — obtained, and
   ~~the threshold decision~~ made (§7): `outcome_class`'s off-instance
   branch now reads `over_target > OFF_INSTANCE_EDIT_OVER_TARGET (1.0)`,
   replacing the ratio-based check that read `ratio > 10.0` while real
   MagicBrush ground truth reads a ratio median of 83.27. Bracketed by both
   the synthetic self-test and, now, the real ground-truth floor (median
   0.619). `mis_targeted`/`measurable`/`valid`/leakage figures never touch
   this branch and are unaffected; the one number that can move is §3's
   "no *tracked* change" row (which excludes `off_instance_edit` from its
   count) — `recompute_outcomes.py` (CPU-only) gives the corrected per-arm
   counts; run it and update §3.
3. ~~Backfill the 4 missing cases~~ — done: GLM-5.3-Flash, n=39/40, ρ=0.717,
   all four verdict classes correctly ordered. ~~Diagnose the 2 residual
   high scorers~~ — done: `refedit-134` is a correctly-scored clean edit,
   not a miss; `refedit-83` is a real false positive with a named
   mechanism (identity drift on an attribute-edit instruction — SC/PQ
   check the instructed attribute and pixel artifacts, not whether the
   subject is still recognizably the same individual). Not prompt-fixed
   on n=1; watch for recurrence when `white`/`oracle` are scored. Judge is
   now reasonably well-calibrated on `none`. What's left: run `white`/
   `oracle`/`sibling`/`freeform` through the judge. That's the one step
   not done; it's what would turn §9's quality claim (currently 40
   audited cases only) into a full-151 comparison.
4. ~~Δ-order curriculum~~ — done, two seeds each, see `DELTA_CURRICULUM_REPORT.md`.
   Headline: A1 (Δ-order curriculum) beats the uncurriculed baseline by
   ~3% on `vs_base_at_a_opt` at the Δ=0.5 operating point, a gap that
   clears seed noise in both seeds; A2 (anchor replacement) is a
   reproducible regression with a code-confirmed mechanism. This is
   separate MeanFlow-training work, not part of the mask-conditioning gate
   this report covers.
