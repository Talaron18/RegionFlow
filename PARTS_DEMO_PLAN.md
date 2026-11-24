# Part-graph editor — demo plan (draft, 2026-11-24)

**Objective.** A demo model that works on a small set, to show the tutor that an explicit
*part graph* fixes the failures every current editor has (wrong owner, invented holder,
extra limbs, redrawn identity, changed neighbours). Not a production model; light, style and
scale are later work for whoever has the GPUs.

## Why parts, not skeletons or masks

A pose solver (MediaPipe, OpenPose) gives joints; a segmenter gives instance masks. Neither
says what is *attached* to what. The stick-figure test failed because klein does not know
that a sleeve moves with the arm and a glove with the hand. So the sketch is a **part
graph**, for people and objects alike:

| node | example | carries |
|---|---|---|
| instance | girl_3, umbrella_1, chair_2 | name, class, appearance code (later) |
| part | girl_3.forearm_L, girl_3.glove_L, girl_3.jacket.sleeve_L, umbrella_1.handle | 2D mask, depth order, visible / hidden |
| joint | girl_3.wrist_L | 2D position, visible / hidden |

| edge | meaning |
|---|---|
| `part_of` | glove_L → hand_L → forearm_L → girl_3 |
| `attached_to` (bone) | sleeve_L moves with forearm_L; glove_L with hand_L |
| `holds` / `rests_on` | hand_L → umbrella_1.handle; cat_1 → girl_3.shoulder_R |
| `occludes` | girl_2.arm_R in front of girl_3.torso |

The rules an editor breaks become checks on this graph: each person has two hands; each hand
ends one arm of one person; a held object touches exactly one hand; parts not touched by the
edit keep their masks.

## Training data: rendered 3D scenes, not photos + detectors

Rigged 3D characters give the whole graph for free: per-part ID pass (every mesh / material
gets an ID), depth, 2D projection of every bone including hidden ones, and the scene
hierarchy as ownership. Placing the same asset several times gives same-category siblings
with distinct names; moving the camera gives overlaps between people and between people and
objects; re-posing a rig is a few bone rotations.

**Edit pairs come out exactly.** Render a scene, apply one change, render again from the same
camera with the same lights:
- *attach*: parent a prop to `girl_3.hand_L` (and pose the arm if needed);
- *re-pose*: rotate girl_3's shoulder/elbow;
- *in-body*: change girl_3's jacket material;
- *place*: put a cat on girl_3's shoulder.

Everything not edited is pixel-identical by construction, so the pair teaches "change the
referent's parts, keep everyone else", which is the failure we measured.

Candidate assets (licences to check per asset before use):
- **MakeHuman / MPFB2 for Blender**: CC0 humans whose clothes are *separate meshes skinned to
  the skeleton* — part masks and attachment with no extra work. First choice.
- Animations: CMU motion capture (free); Mixamo (free to use, not to redistribute).
- Props: Poly Haven, Kenney, Quaternius (CC0); rigged animals from Quaternius.
- Renderer: Blender (`bpy`), headless, Cycles or EEVEE on the 3090 Ti.

## The demo, staged by risk

**D1 — renderer that obeys parts (the core claim).** Input: source image + *target* part map
(a colour-coded image of part IDs + joints) + instruction. Output: edited image. Train a LoRA
on FLUX.2 klein-base-4B with the part map as an extra reference image, on synthetic pairs, plus
real images with an unchanged part map (reconstruction), so appearance and identity are learnt
from the source. This is exactly what the stick-figure test showed klein cannot do zero-shot.
Test with the *oracle* target part map.

**D2 — parser (the "reverse" direction).** Image → part map. Trained on the same renders;
on real images it is the domain-gap risk. For the demo it can be replaced by hand-made part
maps on a handful of real cases.

**D3 — coarse editor.** An LLM/VLM reads the part graph (as JSON) and the instruction and
writes graph edits as code (`attach(umbrella, girl_3.hand_L)`, `rotate(girl_3.shoulder_L, …)`);
a deterministic step redraws the part map. The graph checks reject impossible edits before
rendering. For the demo, D3 can be checked on its own on text: does it produce the right edit
for the 70 arm-T instructions?

Order: D1 first, since if the renderer does not obey a correct part map, D2 and D3 do not
matter. The demo is D1 with oracle part maps, then D1 + D3, then the full loop.

## Kill rules (fixed before training; numbers to be set when the test set exists)

- D1 on held-out synthetic scenes (new assets, new poses, new cameras): with the oracle part
  map, the edit lands on the right owner with no added/shared limb and unchanged siblings in
  most cases; the same model *without* the part map does clearly worse (otherwise the map is
  not what helps).
- D1 on real images with hand-made part maps (≈ 10–20 arm-T cases): better than klein
  crop-box and Qwen plain on owner + anatomy + identity, blind-judged as before.
- Synthetic-only gains that vanish on real images end the "train on renders" premise, not the
  part-graph idea.

## Known risks

1. **Domain gap**: renders look fake. Mitigation for the demo: the renderer edits a *real
   source image*, so appearance comes from the source; synthetic data mainly teaches structure.
   Mix in real reconstruction pairs.
2. **Part maps on real images** (D2) are the hardest step; the demo avoids it with hand-made
   maps.
3. **LoRA on klein-base-4B in 24 GB**: reported by others, not verified here.
4. **Re-posing in 2D** (D3): rotating an arm needs the part map redrawn consistently; easy in
   the renderer for synthetic data, harder for real maps.

## First concrete step

A Blender data-generation smoke test: one scene, three copies of one clothed MPFB2 character,
one prop; render RGB, part-ID pass, depth, 2D joints (with visibility), and the graph JSON;
then apply one *attach* edit and render the pair. Look at the outputs before writing anything
larger.

## Research findings (2026-11-24; two research passes, sources tagged as verified or not in the reports)

**Conditioning format.** Map image as extra reference tokens, not a text graph.
- Verified: OminiControl appends the condition image as tokens + rank-4 LoRA on FLUX.1-dev
  (canny, depth, colourization, inpaint; 223k images, 2×H100). ControlNet learns segmentation
  maps with ~1k images (recognizable output). klein-4B LoRA trains on 24 GB (HF blog: 120 pairs,
  RTX 4090, ~1 h) — but for a style edit, not a new condition.
- Text graphs: LAION-SG needed 540k pairs + a GNN for SG-IoU 0.23 → 0.34; no editing.
- Not found: any LoRA learning a colour-coded part-ID + joint map; any map-vs-text comparison.
- Known failures: FLUX "tends to ignore the condition" (OminiControl README; matches our
  stick-figure test); augmentation-only pairs teach copying (OminiControl §4.3).
- Every graph/scene-based editor (SGEdit, SAGE) keeps unedited content by masks / attention /
  compositing, never by the condition alone → our design needs explicit preservation.

**Part labels.** Human parsing (LIP/CIHP) has left/right arm, leg, shoe, one glove class,
sleeves inside upper-clothes; PASCAL-Part covers objects. **No label set has attachment edges**
(sleeve→forearm, glove→hand). Renders from skinned assets provide them; this is a real gap.

**Rendering.** Blender + Cycles (OptiX) headless; EEVEE needs an OpenGL display. Exact IDs from
object/material index passes (Cryptomatte is anti-aliased); alpha-blended hair/lashes drop out
of passes. Joints: bone head in world space → `world_to_camera_view`; occlusion by
`scene.ray_cast` against depth. No ready-made script found. Existing pipelines don't fit:
BlenderProc and Kubric have no humans; BEDLAM/AGORA/SURREAL are SMPL-based (licence-gated,
clothing masks not stated); TDW (Unity) has humanoids + ID passes but more setup.

**Assets (licence for anything released).** CC0: MPFB2 assets (clothing coverage to confirm),
Quaternius (animals), Kenney, Poly Haven. Avoid for release: SMPL-X / AMASS (non-commercial),
Mixamo (reported to bar ML training), Ready Player Me (CC BY-NC-SA).

**Sim-to-real.** No 2024-26 number found for diffusion trained on renders transferring structure
to photos. LightLab (relighting): synthetic-only generalizes poorly; real + synthetic best. So
mix real images (reconstruction pairs) and edit a real source image.

**Existing edit-pair data.** CLEVR-Change, RS-Objects, SceneForge (Infinigen counterfactuals),
ObMan (hand–object renders). None: multi-person, per-part, hand–object edit pairs.

**Risks added.** Re-posing changes shadows/indirect light in "unchanged" regions (use a diff
mask); LoRA may ignore the map or copy the source (fallback: ControlNet-style branch; compare
against a no-map baseline).
