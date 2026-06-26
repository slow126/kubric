"""Render compact static Kubric camera-intervention scenelets.

Each scenelet contains two visible 512x512 frames and the minimum files needed
by OnlineSyntheticCorrespondence's KubricInterventionDataset. Object poses are
static during rendered frames; motion comes only from the camera dolly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import kubric as kb
from kubric.renderer.blender import Blender
from kubric.safeimport.bpy import bpy

from interface.asset_camera_motion_scene import AssetDollyInConfig, AssetDollyInScene
from interface.asset_object_motion_scene import MovingAssetDollyInConfig, MovingAssetDollyInScene
from interface.run_asset_dolly_in_smoke import ASSET_MANIFESTS
from interface.run_dolly_in_smoke import to_uint8_rgb
from interface.run_hdri_dolly_in_smoke import HDRI_MANIFEST, add_hdri_environment


def _resolution(theta: dict[str, Any]) -> tuple[int, int]:
  raw = theta.get("resolution", [512, 512])
  if isinstance(raw, int):
    return int(raw), int(raw)
  if len(raw) != 2:
    raise ValueError(f"resolution must be int or [H, W], got {raw!r}")
  return int(raw[0]), int(raw[1])


def _camera_direction(theta: dict[str, Any]) -> tuple[float, float, float]:
  if "camera_direction" in theta:
    direction = np.asarray(theta["camera_direction"], dtype=np.float64)
  else:
    azimuth = float(theta.get("camera_azimuth", math.pi * 0.75))
    elevation = float(theta.get("camera_elevation", 0.35))
    direction = np.array([
        math.cos(azimuth) * math.cos(elevation),
        math.sin(azimuth) * math.cos(elevation),
        math.sin(elevation),
    ], dtype=np.float64)
  norm = float(np.linalg.norm(direction))
  if norm <= 0:
    raise ValueError("camera_direction must be nonzero")
  direction = direction / norm
  return tuple(float(v) for v in direction)


def _manifest_for_source(asset_source: str, theta: dict[str, Any]) -> str:
  if theta.get("asset_manifest"):
    return str(theta["asset_manifest"])
  if asset_source not in ASSET_MANIFESTS:
    raise ValueError(f"unknown asset_source={asset_source!r}; expected one of {sorted(ASSET_MANIFESTS)}")
  return ASSET_MANIFESTS[asset_source]


def _frame1_distance(theta: dict[str, Any]) -> float:
  start = float(theta.get("start_distance", 14.0))
  end = float(theta.get("end_distance", 3.5))
  gap = float(theta.get("temporal_gap", 1.0))
  gap = float(np.clip(gap, 0.0, 1.0))
  return start + (end - start) * gap


def _write_flow_frame(output_dir: Path, name: str, frame_index: int, flow: np.ndarray) -> None:
  min_value = float(np.min(flow))
  max_value = float(np.max(flow))
  if max_value == min_value:
    encoded = np.zeros(flow.shape, dtype=np.uint16)
  else:
    encoded = ((flow - min_value) * 65535.0 / (max_value - min_value)).astype(np.uint16)
  kb.write_png(encoded, output_dir / f"{name}_{frame_index:05d}.png")

  range_path = output_dir / "data_ranges.json"
  if range_path.exists():
    with range_path.open("r", encoding="utf-8") as fp:
      ranges = json.load(fp)
  else:
    ranges = {}
  ranges[name] = {"min": min_value, "max": max_value}
  with range_path.open("w", encoding="utf-8") as fp:
    json.dump(ranges, fp, indent=2, sort_keys=True)


def _write_metadata(scene: kb.Scene, output_dir: Path, theta: dict[str, Any], scene_seed: int) -> None:
  payload = {
      "metadata": kb.get_scene_metadata(scene),
      "camera": kb.get_camera_info(scene.camera),
      "intervention_theta": theta,
      "scene_seed": int(scene_seed),
      "format": "kubric_intervention_scenelet_v1",
  }
  with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
    json.dump(payload, fp, indent=2, default=lambda value: np.asarray(value).tolist())
  with (output_dir / "events.json").open("w", encoding="utf-8") as fp:
    json.dump({"collisions": []}, fp, indent=2)


def _chmod_tree(path: Path) -> None:
  """Keep Docker-root renders removable by the host user."""
  if os.getenv("KUBRIC_CHMOD_OUTPUT", "1").lower() not in ("1", "true", "t"):
    return
  for item in [path, *path.rglob("*")]:
    try:
      if item.is_dir():
        item.chmod(0o777)
      else:
        item.chmod(0o666)
    except OSError:
      pass


def _force_cycles_gpu(backend: str) -> None:
  """Enable non-CPU Cycles devices when Blender runs inside NVIDIA Docker."""
  if os.getenv("KUBRIC_USE_GPU", "1").lower() not in ("1", "true", "t", "yes"):
    # CPU-only render: skip Cycles device enumeration and its GPU-fallback
    # warnings entirely. Kubric scenelet rendering is CPU-bound (scene build
    # dominates; Cycles sampling is a few seconds), and this image's Blender
    # 2.93 has no CUDA kernels anyway, so GPU forcing is pure log noise.
    return
  prefs = bpy.context.preferences.addons["cycles"].preferences
  backends = [backend]
  for fallback in ("OPTIX", "CUDA"):
    if fallback not in backends:
      backends.append(fallback)

  selected_backend = None
  for candidate in backends:
    try:
      prefs.compute_device_type = candidate
      prefs.get_devices()
      gpu_devices = [device for device in prefs.devices if device.type != "CPU"]
      if gpu_devices:
        selected_backend = candidate
        break
    except Exception as exc:  # Blender backend availability varies by image.
      print(f"[kubric-render] GPU backend {candidate} unavailable: {exc}", flush=True)

  if selected_backend is None:
    print("[kubric-render] WARNING: no non-CPU Cycles devices detected", flush=True)
    return

  used = []
  for device in prefs.devices:
    device.use = device.type != "CPU"
    if device.use:
      used.append(f"{device.name}({device.type})")
  bpy.context.scene.cycles.device = "GPU"
  print(
      f"[kubric-render] Cycles GPU backend={selected_backend} devices={used}",
      flush=True,
  )


# --- Per-scene camera-motion randomness ("the ladder") ----------------------
# By default a dataset applies ONE fixed camera motion to every scene (the frozen
# rung): scene_seed only varies asset content/HDRI, so camera azimuth/elevation/
# distances are identical across all scenes. Object SE(3) motion already supports
# per-scene spread via motion_*_std; this adds the SAME capability to the CAMERA
# params. Set any *_std key (or camera_motion_choices) in theta to draw per-scene
# camera motion from N(mean, std), seeded by scene_seed (deterministic, so renders
# stay resumable). Backward compatible: with no *_std keys theta is returned
# unchanged.
#   frozen rung : no *_std keys                (legacy behaviour)
#   fitted rung : small *_std toward target    (the "robust" generalizing set)
#   broad  rung : large *_std / motion choices (movi_f-like spread)
_CAMERA_JITTER_KEYS = {
    "camera_azimuth": "camera_azimuth_std",
    "camera_elevation": "camera_elevation_std",
    "start_distance": "start_distance_std",
    "end_distance": "end_distance_std",
    "temporal_gap": "temporal_gap_std",
    "asset_scale": "asset_scale_std",
    "focal_length": "focal_length_std",
}


def _camera_jitter_active(theta: dict[str, Any]) -> bool:
  if theta.get("camera_motion_choices"):
    return True
  return any(float(theta.get(std_key, 0.0)) > 0.0 for std_key in _CAMERA_JITTER_KEYS.values())


def _perturb_theta_for_scene(theta: dict[str, Any], scene_seed: int) -> dict[str, Any]:
  """Resample camera params per scene from N(mean, *_std) for the randomness
  ladder. Returns theta unchanged when no jitter keys are set (frozen rung), so
  this is fully backward compatible. Deterministic in scene_seed."""
  if not _camera_jitter_active(theta):
    return theta
  rng = np.random.default_rng([int(scene_seed), 0xCA3E1A])
  scene_theta = dict(theta)
  for mean_key, std_key in _CAMERA_JITTER_KEYS.items():
    std = float(theta.get(std_key, 0.0))
    if std > 0.0:
      scene_theta[mean_key] = float(theta.get(mean_key, 0.0)) + float(rng.normal(0.0, std))
  choices = theta.get("camera_motion_choices")
  if choices:
    scene_theta["camera_motion"] = str(choices[int(rng.integers(len(choices)))])
  # Keep physical params positive after jitter.
  for key, lo in (("start_distance", 0.5), ("end_distance", 0.5),
                  ("asset_scale", 0.05), ("focal_length", 1.0),
                  ("temporal_gap", 1e-3)):
    if key in scene_theta:
      scene_theta[key] = max(lo, float(scene_theta[key]))
  # Preserve forward dolly-in structure (end < start) so jitter varies the
  # magnitude/heading of the motion without flipping its focus-of-expansion.
  if str(scene_theta.get("camera_motion", "dolly")) == "dolly" \
      and "start_distance" in scene_theta and "end_distance" in scene_theta \
      and scene_theta["end_distance"] >= scene_theta["start_distance"]:
    scene_theta["end_distance"] = max(0.5, scene_theta["start_distance"] - float(theta.get("dolly_span", 0.1)))
  return scene_theta


def render_scenelet(
    theta: dict[str, Any],
    output_dir: Path,
    scene_scratch_dir: Path,
    asset_scratch_dir: Path,
    scene_seed: int,
    samples_per_pixel: int,
    debug_artifacts: bool,
    gpu_backend: str,
    geometry_only: bool = False,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  scene_scratch_dir.mkdir(parents=True, exist_ok=True)
  asset_scratch_dir.mkdir(parents=True, exist_ok=True)

  # Randomness ladder: resample camera motion per scene if *_std keys are set
  # (no-op for frozen thetas). Downstream config + metadata use the per-scene theta.
  theta = _perturb_theta_for_scene(theta, scene_seed)

  asset_source = str(theta.get("asset_source", "kubasic"))
  manifest = _manifest_for_source(asset_source, theta)
  target_z = float(theta.get("target_z", 0.8))
  config_kwargs = dict(
      resolution=_resolution(theta),
      frame_start=1,
      frame_end=2,
      frame_rate=1,
      start_distance=float(theta.get("start_distance", 14.0)),
      end_distance=_frame1_distance(theta),
      target=(0.0, 0.0, target_z),
      camera_direction=_camera_direction(theta),
      focal_length=float(theta.get("focal_length", 35.0)),
      end_focal_length=float(theta.get("end_focal_length", 0.0)),
      sensor_width=float(theta.get("sensor_width", 32.0)),
      camera_motion=str(theta.get("camera_motion", "dolly")),
      camera_baseline=float(theta.get("camera_baseline", 0.0)),
      orbit_azimuth_span=float(theta.get("orbit_azimuth_span", 0.0)),
      orbit_elevation_span=float(theta.get("orbit_elevation_span", 0.0)),
      asset_manifest=manifest,
      seed=int(scene_seed),
      num_assets=int(theta.get("num_assets", 4)),
      asset_scale=float(theta.get("asset_scale", 1.1)),
      asset_ids=tuple(theta.get("asset_ids", ())),
      color_assets=not bool(theta.get("keep_asset_materials", False)),
  )

  # Object SE(3) motion is opt-in: with num_moving<=0 (the default) we build the
  # original static-asset scene and the path below is unchanged.
  num_moving = int(theta.get("num_moving", 0))
  if num_moving > 0:
    config = MovingAssetDollyInConfig(
        **config_kwargs,
        num_moving=num_moving,
        motion_translation=float(theta.get("motion_translation", 0.0)),
        motion_translation_std=float(theta.get("motion_translation_std", 0.0)),
        motion_rotation=float(theta.get("motion_rotation", 0.0)),
        motion_rotation_std=float(theta.get("motion_rotation_std", 0.0)),
        motion_vertical_frac=float(theta.get("motion_vertical_frac", 0.0)),
        motion_seed=int(theta.get("motion_seed", 0)),
        spawn_mode=str(theta.get("spawn_mode", "floor")),
        keep_floor=bool(theta.get("keep_floor", True)),
        spawn_xy_extent=float(theta.get("spawn_xy_extent", 1.8)),
        spawn_z_min=float(theta.get("spawn_z_min", 0.4)),
        spawn_z_max=float(theta.get("spawn_z_max", 2.6)),
    )
    builder = MovingAssetDollyInScene(config, asset_scratch_dir=asset_scratch_dir)
  else:
    config = AssetDollyInConfig(**config_kwargs)
    builder = AssetDollyInScene(config, asset_scratch_dir=asset_scratch_dir)
  scene = builder.build()
  # Geometry-only ("search") render: the motion descriptor (flow/seg/depth) is
  # geometric, so we skip the expensive RGB shading -- 1 sample/pixel and no
  # denoise -- and never request the rgba layer. This is the cheap scoring path;
  # use the full path (geometry_only=False) only to materialize a winning theta.
  use_denoising = not geometry_only
  renderer = Blender(
      scene,
      scratch_dir=scene_scratch_dir,
      samples_per_pixel=samples_per_pixel,
      use_denoising=use_denoising,
  )
  _force_cycles_gpu(gpu_backend)

  background_mode = str(theta.get("background_mode", "matte"))
  if geometry_only:
    # HDRI only affects RGB shading; skip its fetch/dome setup in search mode.
    if background_mode not in ("matte", "hdri"):
      raise ValueError(f"unknown background_mode={background_mode!r}")
  elif background_mode == "hdri":
    add_hdri_environment(
        scene=scene,
        renderer=renderer,
        scratch_dir=asset_scratch_dir,
        hdri_manifest=str(theta.get("hdri_manifest", HDRI_MANIFEST)),
        hdri_id=theta.get("hdri_id"),
        hdri_split=str(theta.get("hdri_split", "train")),
        seed=int(scene_seed),
        use_dome=not bool(theta.get("no_hdri_dome", False)),
        strength=float(theta.get("hdri_strength", 1.0)),
    )
  elif background_mode != "matte":
    raise ValueError(f"unknown background_mode={background_mode!r}")

  builder.keyframe_camera_path(scene)
  if isinstance(builder, MovingAssetDollyInScene):
    builder.keyframe_object_paths(scene)

  # Opt-in: also emit the segmentation pass in full renders so a foreground/background
  # mask can be derived downstream. Default (False) keeps renders byte-identical.
  emit_segmentation = bool(theta.get("emit_segmentation", False))
  if geometry_only:
    # Restrict to the geometric passes; no rgba/normal/object_coordinates.
    frames = renderer.render(
        return_layers=("forward_flow", "backward_flow", "segmentation", "depth"))
  elif emit_segmentation:
    frames = renderer.render(
        return_layers=("rgba", "forward_flow", "backward_flow", "segmentation"))
  else:
    # Unchanged full-render path: Blender's default return_layers.
    frames = renderer.render()

  if not geometry_only:
    rgb = to_uint8_rgb(frames["rgba"])
    imageio.imwrite(output_dir / "rgba_00000.png", rgb[0])
    imageio.imwrite(output_dir / "rgba_00001.png", rgb[1])
  _write_flow_frame(output_dir, "forward_flow", 0, frames["forward_flow"][0])
  _write_flow_frame(output_dir, "backward_flow", 1, frames["backward_flow"][1])
  if emit_segmentation and "segmentation" in frames:
    # Segmentation of the TARGET frame (frame 1) -- matches backward_flow_00001 that
    # KubricInterventionDataset reads. uint16 instance ids; background (no object) = 0.
    seg = np.asarray(frames["segmentation"])[1]
    if seg.ndim == 3:
      seg = seg[..., 0]
    imageio.imwrite(output_dir / "segmentation_00001.png", seg.astype(np.uint16))
  _write_metadata(scene, output_dir, theta, scene_seed)

  if debug_artifacts:
    kb.write_image_dict({k: v for k, v in frames.items() if k not in {"forward_flow", "backward_flow"}}, output_dir)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--theta-json", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--scratch-dir", type=Path, default=Path("/mnt/nvme_1tb_a/kubric_interventions/scratch"))
  parser.add_argument("--asset-scratch-dir", type=Path, default=None)
  parser.add_argument("--n-pairs", type=int, default=10_000)
  parser.add_argument("--start-index", type=int, default=0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--samples-per-pixel", type=int, default=None,
                      help="Cycles samples/pixel. Default: 1 with --geometry-only, else 32.")
  parser.add_argument("--geometry-only", action="store_true",
                      help="Cheap search render: flow/seg/depth only, spp=1, no denoise, no rgba/HDRI.")
  parser.add_argument("--debug-artifacts", action="store_true")
  parser.add_argument("--keep-scene-scratch", action="store_true")
  parser.add_argument("--gpu-backend", default="CUDA", choices=["CUDA", "OPTIX"])
  args = parser.parse_args()

  samples_per_pixel = args.samples_per_pixel
  if samples_per_pixel is None:
    samples_per_pixel = 1 if args.geometry_only else 32

  with args.theta_json.open() as fp:
    theta = json.load(fp)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.scratch_dir.mkdir(parents=True, exist_ok=True)
  asset_scratch_dir = args.asset_scratch_dir or (args.scratch_dir / "assets")
  asset_scratch_dir.mkdir(parents=True, exist_ok=True)
  for offset in range(args.n_pairs):
    scene_index = args.start_index + offset
    scene_seed = args.seed + scene_index
    scene_dir = args.output_dir / f"scene_{scene_index:06d}"
    print(f"[kubric-render] scene={scene_index} seed={scene_seed} -> {scene_dir}", flush=True)
    render_scenelet(
        theta=theta,
        output_dir=scene_dir,
        scene_scratch_dir=args.scratch_dir / "scene_scratch" / f"scene_{scene_index:06d}",
        asset_scratch_dir=asset_scratch_dir,
        scene_seed=scene_seed,
        samples_per_pixel=samples_per_pixel,
        debug_artifacts=args.debug_artifacts,
        gpu_backend=args.gpu_backend,
        geometry_only=args.geometry_only,
    )
    _chmod_tree(scene_dir)
    if not args.keep_scene_scratch:
      shutil.rmtree(args.scratch_dir / "scene_scratch" / f"scene_{scene_index:06d}", ignore_errors=True)


if __name__ == "__main__":
  main()
