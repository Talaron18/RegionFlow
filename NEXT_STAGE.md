# Next stage: two-stage graph editing with a reversible detail flow

Status: **plan, 2026-11-24. Nothing here is built or trained yet.** Evidence behind it is in
`MODERN_BASE_EVAL.md` (measured) and `RELATED_WORK.md` (prior art). Older plans are in
`docs/archive/`.

**Objective.** A demo model that works on a small set, used to make the case to the tutor
(and for compute). It must show that an explicit part graph fixes the failures current
editors make on multi-person attachment edits. Light, style and scale are later work.

---

## 1. Why: what the evaluation established

| finding | evidence |
|---|---|
| Picking the referred instance is solved by segment-then-paste | Sa2VA finds the target 137/151 (RefEdit-Bench), 64/70 (arm T); MIRAGE does the same in-sampler |
| What remains is attachment: an object that must go to one person's body | arm T, same people: clothing controls pass, attachment fails — klein crop-box 7 vs 0 discordant pairs, Qwen plain 6 vs 0 |
| Failure mode depends on the editor | klein crop-box (user-judged): A 29/45 = 64%, mostly **wrong owner** / invented holder (5/5 invented holders failed). Qwen plain (Claude-judged, user-checked): A 22/45 = 49%, mostly **identity**: right person, redrawn (15/45) |
| Region rules cannot fix it | paste, crop-box, attribution, erase-then-redraw only keep or cut what the editor drew |
| Structure given zero-shot is ignored | stick-figure reference: 0/8 with, 0/8 without |
| Not explained by hand visibility | occluded / low-contrast hand: 9/15 vs 9/15 |
| VLM judges miss it | our judge missed sibling leakage; MPIE-Bench finds VLM checklists > 0.95 on anatomically broken images |

The common cause, in the user's terms: editors have no notion of **parts** — that a sleeve
moves with the forearm, a glove with the hand, a handle belongs to the umbrella, and that each
hand ends one arm of one person. A pose solver gives joints; a segmenter gives instance masks;
neither gives attachment and ownership. So the condition must combine them.

## 2. The representation: a part graph

The graph is the single source of truth. Nodes:

| node | examples | attributes |
|---|---|---|
| instance | `girl_3`, `umbrella_1`, `chair_2` | class, name, appearance code (§3.3) |
| part | `girl_3.forearm_L`, `girl_3.glove_L`, `girl_3.jacket.sleeve_L`, `umbrella_1.handle` | 2D mask, depth order, visible / hidden |
| joint | `girl_3.wrist_L` | 2D position, visible / hidden |

Edges:

| edge | meaning |
|---|---|
| `part_of` | glove_L → hand_L → forearm_L → girl_3; handle → umbrella_1 |
| `attached_to` | sleeve_L skinned to forearm_L; glove_L to hand_L (moves with it) |
| `holds` / `rests_on` | girl_3.hand_L holds umbrella_1.handle; cat_1 rests_on girl_3.shoulder_R |
| `occludes` | girl_2.arm_R in front of girl_3.torso |
| side | every paired part carries L / R from the person's own view |

**Rules** checked by a program on any edited graph, before rendering: each person has two arms
and two hands; each hand ends one arm of one person; a held object touches exactly one hand;
a part not named by the edit keeps its mask.

**Three views of the same graph**

| view | consumer | why |
|---|---|---|
| JSON | the coarse model (LLM/VLM) | reads it and writes edits as code |
| **map image** | the detail model | part IDs colour-coded per instance + joints, given as extra reference tokens |
| masks | the compositor | parts the edit does not touch are pasted back from the source |

Map image, not text, for the detail model (research, 2026-11-24): map-as-tokens with a LoRA is
verified on FLUX (OminiControl: canny, depth, colourization); segmentation maps are learnable
from ~1k images (ControlNet); text-graph conditioning needed 540k pairs and a GNN for a modest
gain and does not edit (LAION-SG). No existing label set has attachment edges (LIP/CIHP have
L/R arms but sleeves inside "upper clothes"; PASCAL-Part has objects).

## 3. The model

```
                 ┌──────────────── coarse stage ────────────────┐
source image ──► │ parse:  image → part graph G  (+ appearance) │
instruction  ──► │ edit:   G, instruction → G'  (graph ops)     │
                 │ check:  rules(G') else reject / retry        │
                 └──────────────────┬───────────────────────────┘
                                    │ map(G'), masks(G, G')
                 ┌──────────────── detail stage ────────────────┐
source image ──► │ klein-base-4B + LoRA, references:            │
                 │   [source image] [map(G')]                   │ ──► edited image
                 │ compositor: paste back parts unchanged in G' │
                 └──────────────────────────────────────────────┘
```

### 3.1 Coarse stage
- **Edit** (`G, instruction → G'`): an LLM/VLM writes graph operations as code:
  `attach(umbrella_1.handle, girl_3.hand_L)`, `rotate(girl_3.shoulder_L, ...)`,
  `recolor(girl_3.jacket, red)`, `place(cat_1, girl_3.shoulder_R)`. A deterministic step
  redraws the map. Testable on text alone with the 70 arm-T instructions.
- **Parse** (`image → G`): the hard step on real images. For the demo it is replaced by
  hand-made graphs on a handful of real cases and by ground truth on renders.

### 3.2 Detail stage
LoRA on FLUX.2 klein-base-4B (undistilled; LoRA training on 24 GB verified by others for a
style task, not for a new condition), OminiControl-style: the map is an extra reference image
next to the source. Trained on rendered edit pairs (§4) plus real images with an unchanged
map (reconstruction), so appearance and identity come from the source image. Fallback if the
LoRA ignores the map: a ControlNet-style branch.

**Preservation is explicit**, not left to the condition (every graph-based editor found keeps
unedited content by masks or compositing, never by conditioning alone): the compositor pastes
back the source wherever G' leaves parts unchanged. This is what should stop Qwen-style
identity redraws and neighbour changes.

### 3.3 Reversibility
The user's idea: image → sketch → edit sketch → image, one model for parsing, editing and
generation. What it requires:
1. **An encoder, trained with the renderer.** Inverting a flow model gives noise, not
   structure; the sketch must come from a parser trained so that render(parse(x)) ≈ x
   (TRELLIS also has a separate encoder for its structured latent).
2. **Appearance codes per node.** A graph says "girl in a yellow jacket", not her face. Each
   instance/part carries a code from the source image; edits copy codes for untouched nodes.
3. **Cycle check as a training signal**: parse(render(G)) ≈ G on renders, where G is known.

For the demo, reversibility is shown on renders (where G is ground truth) and on a few real
images with hand-made graphs; a general real-image parser is later work.

## 4. Training data from 3D assets

Why renders: they give the full graph for free — per-part ID pass, depth, 2D bone projections
including hidden joints, the scene hierarchy as ownership. The same asset placed several times
gives same-kind siblings with distinct names; camera moves give overlaps; rigs re-pose easily.
**Edit pairs are exact**: render, apply one graph op, render again from the same camera;
everything not edited is identical.

| op | how in Blender |
|---|---|
| attach | parent prop to `hand_L` bone (+ pose the arm) |
| re-pose | rotate shoulder / elbow bones |
| in-body | swap a clothing material |
| place | parent an animal to a shoulder / lap |

Pipeline (research, 2026-11-24): Blender 4.2 (user has it), **Cycles (OptiX) headless** —
EEVEE needs an OpenGL display. Exact IDs from object / material index passes (Cryptomatte is
anti-aliased; alpha-blended hair drops out). Joints: bone head → `world_to_camera_view`,
visibility by `scene.ray_cast` against depth. No ready-made pipeline fits: BlenderProc and
Kubric have no humans; BEDLAM / AGORA / SURREAL are SMPL-based and licence-gated.

Assets (CC0 for anything released): **MPFB2** humans with clothes as separate skinned meshes
(confirm clothing licence), Quaternius animals, Kenney and Poly Haven props. Avoid for
release: SMPL-X / AMASS (non-commercial), Mixamo (reported to bar ML training), Ready Player
Me (CC BY-NC-SA).

Work needed: split clothing into per-part objects/materials, occlusion test for joints,
`occludes` edges from IDs + depth, L/R naming, a diff mask for shadow changes in "unchanged"
regions after re-posing.

## 5. Experiments, in order of risk (kill rules fixed before each runs)

| step | question | pass | stop |
|---|---|---|---|
| **E0 render smoke** | do IDs, joints (with visibility), graph JSON and one attach pair come out right? | all correct on 1 scene, viewed | IDs or joints wrong → fix before any scale-up |
| **E1 map learning** | does a klein-base LoRA follow a part map at all? few hundred rendered pairs, held-out scenes | with the map: right owner, no added/shared limb, siblings unchanged in most cases; **clearly better than the same LoRA without the map** | map ignored, or no-map baseline equal → try ControlNet branch once; if still equal, the condition route is closed |
| E2 copy check | does it edit, or copy the source? | edits happen at the rate the pairs demand | copying → change pair design (OminiControl §4.3) |
| E3 real images | 10–20 arm-T cases with hand-made G' | beats klein crop-box and Qwen plain on owner + anatomy + identity, blind-judged | synthetic gains vanish on real → renders premise fails, part-graph idea does not |
| E4 coarse editor | LLM writes graph ops for the 70 arm-T instructions | ops correct on most, rules catch the rest | — |
| E5 reversibility | parse(render(G)) ≈ G on renders; then the full loop | — | — |

Numbers for the "pass" columns are set when each test set exists, before the run.

## 6. Risks

1. **The LoRA ignores the map** (OminiControl: FLUX "tends to ignore the condition"; our
   stick-figure test). Mitigation: pairs where ignoring the map costs the loss; no-map
   baseline; ControlNet-style fallback.
2. **Sim-to-real.** No published number found for structure transferring from renders to
   photos for diffusion; LightLab (relighting) found synthetic-only generalizes poorly, real +
   synthetic best. Mitigation: edit a real source image; mix real reconstruction pairs.
3. **Parsing real images** is the hardest part; the demo avoids it.
4. **24 GB for klein-base LoRA with two references** is unverified.
5. **Re-posing** changes shadows in "unchanged" regions of renders (diff mask).
6. **Licences** of assets for any released data.

## 7. Closest prior work (to position against)

OminiControl (map as tokens + LoRA on FLUX), ControlNet, PartEdit (learned token per part,
edits one part), SGEdit (scene graph + masks, per-image fine-tune), LAION-SG / SDXL-SG
(text scene graphs), Spatial CoT (coordinate layouts in Qwen-Image-Edit), MIRAGE
(VLM-planned regions), HOI-Edit / MPIE-Bench (interaction and anatomy evaluation). Not found:
editing through a part graph with attachment edges, or rendered multi-person hand–object edit
pairs with per-part labels.

## 8. Immediate next step

E0: a Blender 4.2 + MPFB2 script (headless, Cycles) that builds one scene — two or three
copies of one clothed character, one prop — and writes RGB, part-ID pass, depth, 2D joints with
visibility and the graph JSON, then applies `attach(prop, char_2.hand_L)` and renders the pair.
