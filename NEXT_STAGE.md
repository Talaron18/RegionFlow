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

**Division of labour (user, 2026-11-26).** Editing — deciding *which* parts change, where an
object goes, whose hand holds it, how an arm moves — is reasoning and geometry. Language models
are already good at it; image models are hard to train for it and fail at it (§1). So the
**language model does all of the editing**, on the graph. The **image model only does what it
was trained for**: follow a prompt and generate pixels, inside the region the graph says has
changed. Image models are not taught to edit.

```
                 ┌──────────────── coarse stage (LLM/VLM) ──────┐
source image ──► │ parse:  image → part graph G  (+ appearance) │
instruction  ──► │ edit:   G, instruction → G'  (graph ops)     │
                 │ check:  rules(G') else reject / retry        │
                 │ plan:   changed region R = parts(G') Δ G,    │
                 │         a prompt per region, map(G') in R    │
                 └──────────────────┬───────────────────────────┘
                                    │ R, prompts, map(G')
                 ┌──────────────── detail stage (image model) ──┐
source image ──► │ generate inside R only, from the prompt      │
                 │   (+ map(G') as guidance, only if needed)    │ ──► edited image
                 │ outside R: source pixels, untouched          │
                 └──────────────────────────────────────────────┘
```

Consequences:
- Wrong owner and neighbour changes become impossible by construction: the image model never
  sees a choice of whom to edit, and it cannot write outside R.
- Identity of the referent outside the edited parts is kept the same way (R excludes the face
  unless the edit names it). This targets Qwen's main failure.
- What is left for the image model is drawing a hand holding an umbrella in a region whose
  shape already says where the arm and hand are. That is ordinary prompted inpainting — the
  task image models are trained on — not instruction editing.
- Training the image model becomes optional (§3.2), not the core of the method.

### 3.1 Coarse stage
- **Edit** (`G, instruction → G'`): an LLM/VLM writes graph operations as code:
  `attach(umbrella_1.handle, girl_3.hand_L)`, `rotate(girl_3.shoulder_L, ...)`,
  `recolor(girl_3.jacket, red)`, `place(cat_1, girl_3.shoulder_R)`. A deterministic step
  redraws the map. Testable on text alone with the 70 arm-T instructions.
- **Parse** (`image → G`): the hard step on real images. For the demo it is replaced by
  hand-made graphs on a handful of real cases and by ground truth on renders.

### 3.2 Detail stage
Order of preference, cheapest first:
1. **Zero-shot inpainting** with the LLM's region and prompt: `Flux2KleinInpaintPipeline`
   (in diffusers 0.43) or Qwen's inpaint pipeline. No training.
2. **Map as guidance, no training**: the map region drawn as an extra reference image, or a
   pose control where one exists (FLUX.2-dev Fun ControlNet Union has pose; none found for
   klein).
3. **LoRA, only if 1–2 draw the wrong structure inside R** (e.g. a third arm inside the
   region): klein-base-4B, map as an extra reference (OminiControl-style), trained on rendered
   pairs (§4). Fallback: a ControlNet-style branch.

**Preservation is explicit**: outside R the output is the source, pixel for pixel (every
graph-based editor found keeps unedited content by masks or compositing, never by
conditioning alone). A thin feathered border blends R into the source.

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

Pipeline (research, 2026-11-24): Blender **5.2.2 LTS** (user's version; the research looked at
4.2 docs, and MPFB2 was stated for ≥ 4.2 — compatibility with 5.x to be checked in E0),
**Cycles (OptiX) headless** —
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
| **E1 LLM plan + zero-shot inpaint** | on real arm-T attach cases with hand-made graphs: LLM writes G', region R and prompt; klein inpaint fills R | right owner by construction; question is the drawing inside R: no added/shared limb, object held, edit happens | structure wrong inside R in most cases → E1b |
| E1b map guidance / LoRA | does the image model draw the right structure inside R when given map(G')? rendered pairs, held-out scenes | clearly better than E1 without the map | map ignored → ControlNet branch once; if still equal, the image-side route is closed (the LLM side still stands) |
| E2 copy check (only if E1b trains a LoRA) | does it edit, or copy the source? | edits happen at the rate the pairs demand | copying → change pair design (OminiControl §4.3) |
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

E0: a Blender 5.2 + MPFB2 script (headless, Cycles) that builds one scene — two or three
copies of one clothed character, one prop — and writes RGB, part-ID pass, depth, 2D joints with
visibility and the graph JSON, then applies `attach(prop, char_2.hand_L)` and renders the pair.
