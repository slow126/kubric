# Plan: Geometry-Only Search + Object/Camera SE(3) Motion Parameterization

Status: PLANNING ONLY — no code edits yet.
Owner: spencer
Created: 2026-06-04

Goal of this experiment: extend the existing interventional Kubric search so it
(1) scores candidates with a **cheap geometry-only render** instead of full RGB,
and (2) searches over **object SE(3) motion** (not just camera motion), with
**number of moving objects** as a parameter. This is the next step toward
target-aware motion-matched procedural generation (Project 3 generator arm).

--------------------------------------------------------------------------------
## 0. Current state (grounded in the existing code)

Two repos are involved:

- `~/Projects/kubric` (fork `slow126/kubric`, branch `main`) — the generator,
  scene builders, and Blender renderer. **This is where the motion + render
  changes live** (hence this plan.md location).
- `~/Projects/interventional-study` — the search loop, search space, scoring,
  and dataset materialization that *drive* the generator.

What exists today:

1. **Search loop** — `interventional-study/search_loop.py`
   - Optuna TPE over θ. Per trial: render flow scenelets → extract 13-dim
     features vs benchmarks → score via the pickled full-fit predictor
     (`score_candidate.py`, fitness = mean calibration gain `g`). Resumable
     SQLite study. **No per-candidate model training** — predictor is the
     surrogate. This loop is the validated interventional pipeline
     (KITTI+Middlebury confirmed).

2. **Search space** — `interventional-study/search_spaces.py`
   - `_sample_kubric_geometry()` samples: `start_distance`, `end_distance`,
     `temporal_gap`, `camera_azimuth`, `camera_elevation`, `target_z`,
     `focal_length`, `num_assets` (3–8), `asset_scale`. Appearance axes
     (`asset_source`, `background_mode`, `keep_asset_materials`) added by
     `sample_kubric` / pinned by `sample_kubric_hq`.
   - All priors are Optuna **uniform** ranges (`suggest_float` / `suggest_int`).
   - **Motion = camera dolly only.** Objects are placed but do not move.

3. **Scene builder** — `kubric/interface/asset_camera_motion_scene.py`
   - `AssetDollyInScene(StaticDollyInScene)` → `_add_assets()` places assets and
     sets `obj.static = True`. Camera dolly-in is keyframed by the base class.
   - **This is the file to extend for object motion.**

4. **Renderer** — `kubric/kubric/renderer/blender.py`
   - `render(return_layers=("rgba","backward_flow","forward_flow","depth",
     "segmentation"))` — layers are selectable; `rgba` can be dropped.
   - `samples_per_pixel` (default 128) and `use_denoising` are settable.
   - `post_processing.compute_visibility()` already gives per-asset pixel
     visibility from the segmentation pass — useful for occlusion stats later.

5. **Materialize** — `interventional-study/render_kubric_dataset.py`
   - Full RGB render of the winning θ (`samples_per_pixel=16`, 5000 pairs) for
     downstream training. Unchanged by this experiment.

Keyframing reference: `kubric/examples/keyframing.py` shows the exact API —
`obj.position = ...; obj.quaternion = ...; obj.keyframe_insert("position", f);
obj.keyframe_insert("quaternion", f)`. Same API works for camera and objects.

--------------------------------------------------------------------------------
## 1. Workstream A — Geometry-only ("lightweight") render for search

Rationale: the motion descriptor (flow stats, segmentation, depth) is geometric.
RGB shading (high Cycles samples, denoise, materials, lighting) is the expensive
part and is **not needed to score a candidate**. Render only the data passes.

Steps:

A1. **Audit the search render path** in `interventional-study/generators/kubric.py`
    (`run_renderer` / the per-scenelet render script). Confirm whether the
    *search* render currently produces `rgba` and at what `samples_per_pixel`.
    (Materialize uses spp=16; search may share that path.) Record the baseline
    per-scenelet wall-clock.

A2. **Add a `geometry_only` / `search_mode` flag** to the render call so it:
    - calls `renderer.render(return_layers=("forward_flow","backward_flow",
      "segmentation","depth"))` — **no `rgba`**;
    - sets `samples_per_pixel = 1` and `use_denoising = False`;
    - optionally renders at reduced resolution (e.g. 256² instead of 512²) —
      flow stats are scale-normalized so low res is acceptable for *scoring*.
    - skips HDRI/material setup where it only affects RGB (keep geometry/assets).

A3. **Verify equivalence of the descriptor** under geometry-only vs full render:
    extract the 13-dim feature vector both ways for ~10 θ and confirm the flow
    features match within noise. (Appearance/DINO features will differ — those
    are not used for the motion search anyway; pin or skip them in search mode.)

A4. **Measure speedup** (target: ≥5–10× per scenelet). This is the unlock that
    makes a larger search (object motion adds dimensions) affordable.

Acceptance: same flow descriptor (±noise) as full render, ≥5× faster.

--------------------------------------------------------------------------------
## 2. Workstream B — Object SE(3) motion parameterization

Rationale: today only the camera moves; the generator cannot produce the
independent-object motion present in KITTI/FlyingThings targets. Add per-object
rigid motion via keyframing (bypassing PyBullet — explicit control, two-frame).

Steps:

B1. **New scene class** in `kubric/interface/asset_camera_motion_scene.py`
    (or a sibling `asset_object_motion_scene.py`):
    `MovingAssetDollyInScene(AssetDollyInScene)` that overrides `_add_assets()`
    to, per object:
    - keep frame-0 placement;
    - sample an SE(3) delta (translation + rotation) for frame-1;
    - set `obj.static = False`, keyframe `position`/`quaternion` at frame_start
      and frame_end (two-frame scenelet, same pattern as keyframing.py);
    - leave a configurable fraction of objects static (background clutter).

B2. **Per-object motion axes** (the new θ dimensions, all *distribution params*,
    not per-object values — see §3):
    - translation magnitude (world units / depth-normalized)
    - translation direction (in-image vs in-depth ratio; or full 3D direction)
    - rotation axis + rotation angle magnitude
    - (optional) object depth / screen-size coupling
    Keep camera dolly motion as-is and *combined* with object motion — the
    rendered flow is the joint field (we never decompose; see project notes).

B3. **Fraction / count of moving objects** — see Workstream C.

B4. **Sanity render**: a handful of geometry-only scenelets, visualize flow +
    segmentation, confirm moving objects show distinct flow vs background and
    occlusion boundaries appear.

Acceptance: flow field shows independent object motion + camera motion jointly;
segmentation IDs let us attribute (for the Kubric *oracle* check only).

--------------------------------------------------------------------------------
## 3. Workstream C — number of objects + distribution family

### C1. num_objects as a parameter
`num_assets` already exists (3–8). Split it into:
- `num_assets` (total placed objects), and
- `num_moving` — how many of them get SE(3) motion (the rest stay static).
  Parameterize as either an integer (`suggest_int`) or a fraction
  `moving_frac ∈ [0,1]` mapped to `round(moving_frac * num_assets)`.
Note `_asset_positions()` currently only has a handful of base slots — extend it
to support larger counts cleanly if we raise the upper bound.

### C2. Distribution family — "are we thinking just Gaussian?"
**Recommendation: not pure Gaussian.** A single Gaussian per axis is the wrong
prior for several of these quantities. Use a small *mixed* family, one
low-dimensional distribution per motion primitive, matching each axis's support:

| Axis | Support | Suggested family | θ params |
|------|---------|------------------|----------|
| translation magnitude | ≥ 0 | half-normal **or** uniform[lo,hi] | scale (or lo,hi) |
| translation direction (2D image) | angle | von Mises (or uniform) | mean, concentration |
| in-plane vs in-depth ratio | [0,1] | uniform / Beta | (a,b) or lo,hi |
| rotation angle magnitude | ≥ 0 | half-normal / uniform | scale |
| rotation axis | S² | uniform on sphere (or fixed up-axis) | — |
| num_moving | integer | uniform int / Binomial(num_assets, p) | p |
| camera dolly (existing) | — | keep current uniform ranges | — |

Design rules:
- **Optimize *distributions over* motion primitives, not per-object values.**
  θ is the handful of distribution parameters above; the generator *samples*
  each scene's concrete object transforms from them. (Same philosophy as the
  SDF sampler θ already in `search_spaces.py`.)
- **Keep θ low-dimensional** (target ≲ 12–15 dims total incl. camera) so the
  distribution-distance / predictor signal is estimable and TPE/CMA-ES stay
  efficient.
- Start with the **simplest** member that has the right support (uniform/half-
  normal); add concentration params (von Mises κ, Beta) only if the oracle shows
  the simple family can't reach the target motion structure.

### C3. New sampler function
Add `sample_kubric_motion(trial)` in `search_spaces.py` = `_sample_kubric_geometry`
+ the per-object motion distribution params above + `num_moving`. Register in
`SAMPLERS`. Keep `sample_kubric_hq` (appearance-pinned) as the variant used for
the clean motion-only run.

--------------------------------------------------------------------------------
## 4. Scoring objective — keep predictor first, distribution-matching later

Two options for what the search maximizes:

- **(Current) predictor fitness** — extract 13-dim features vs benchmark, score
  with the full-fit predictor. Proven, validated (KITTI+Middlebury). **Use this
  for the first object-motion run** — change only the generator + render, not the
  objective, so any improvement is attributable to the new motion axis.

- **(Phase B) direct distribution matching** — match the generated motion
  descriptor to the cached target descriptor (whole-dataset aggregate BFV +
  structural stats). This is the cleaner "match the target distribution" objective
  discussed in project notes, but it's a *separate* change. Decision gate: only
  adopt if it beats predictor-scoring on the Kubric oracle.

Descriptor granularity (the "whole-dataset BFV vs more processing" question):
**start with whole-dataset aggregate BFV both sides** (lowest friction, matches
existing 13-dim extractor). Enrich with structural stats (motion-boundary
density, occlusion/validity fraction from `compute_visibility`, per-segment
coherence) **only if the Kubric oracle shows the aggregate match admits
pathological camera/object splits.** Target descriptor is computed once and
cached (cheap to process heavily); generated side must stay geometry-only/cheap.

--------------------------------------------------------------------------------
## 5. Phasing (cheap-test discipline)

- **Phase 0 — geometry-only render (Workstream A).** No new motion. Confirm
  descriptor equivalence + speedup. Pure infra win, de-risks everything else.
- **Phase 1 — object SE(3) motion, predictor scoring (B + C, objective unchanged).**
  Run the camera+object search against a KITTI-like target. Compare to the
  camera-only baseline (existing trial19). Question: does adding object motion
  let the search reach better predicted transfer?
- **Phase 2 — Kubric oracle check.** Because Kubric gives GT segmentation +
  known transforms, verify whether matched candidates produce sensible vs
  pathological camera/object splits, and whether structural stats are needed.
- **Phase 3 (optional) — distribution-matching objective** and/or **per-pair
  descriptor**. Only if Phases 1–2 motivate it.

Scope guard: this is the Project-3 generator arm / future-work extension. It is
NOT a dependency for Projects 2 or the curation result. Keep camera+rigid-object
(KITTI-like) targets only; articulation/deformation (SPair) is out of scope here.

--------------------------------------------------------------------------------
## 6. Integration touchpoints (files to change, when we get there)

kubric (this repo):
- `interface/asset_camera_motion_scene.py` — new `MovingAssetDollyInScene`
  (object SE(3) keyframing; `_add_assets` override; `num_moving`).
- render driver / `interface/run_*` entry used by the search — add
  `geometry_only` mode (return_layers w/o rgba, spp=1, no denoise, optional 256²).
- possibly `kubric/post_processing.py` usage — wire `compute_visibility` for
  occlusion stats (Phase 2+ only).

interventional-study (driver repo):
- `search_spaces.py` — `sample_kubric_motion`, `num_moving`, motion dist params.
- `generators/kubric.py` (`run_renderer`) — plumb `geometry_only` + motion θ.
- `search_loop.py` — register the new sampler / objective (objective unchanged
  in Phase 1).
- `score_candidate.py` / extractor — only if/when switching to distribution
  matching (Phase 3).

--------------------------------------------------------------------------------
## 7. Open decisions (resolve before coding)

1. Two-frame scenelets only, or short multi-frame trajectories? (Two-frame is
   simplest and matches the current temporal_gap design — recommend two-frame.)
2. Object motion in world units or depth-normalized? (Depth-normalized makes the
   flow magnitude prior more transferable across scene scales.)
3. Reduced search resolution (256²) acceptable, or keep 512² for descriptor
   fidelity? (Measure in Phase 0.)
4. num_moving as integer vs Binomial(p) — affects θ dimensionality.
5. Geometry-only render still needs Docker? Confirm the lightweight path works
   outside the heavy `kubricdockerhub/kubruntu` RGB image if possible.

## 8. Risks
- Geometry-only flow may differ subtly from full-render flow (anti-aliasing at
  motion boundaries). Mitigated by Phase 0 equivalence check.
- Adding motion dims enlarges the search; TPE may need more trials or a switch to
  CMA-ES. The geometry-only speedup (Phase 0) is what buys this back.
- Pathological camera/object split under aggregate-BFV matching (Phase 2 catches).
- Generator expressivity ceiling: rigid SE(3) only — cannot match deformation/
  articulation targets. Stay on KITTI-like targets.
