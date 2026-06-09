"""Dolly-in scene where a subset of the assets also undergo rigid SE(3) motion.

Extends :class:`AssetDollyInScene` (camera dolly only, all objects static) by
keyframing per-object translation + rotation for ``num_moving`` of the placed
assets. Motion is set explicitly via keyframes (no PyBullet) so the two-frame
scenelet has deterministic, known object transforms -- the rendered flow is the
*joint* camera + object field (we never decompose it).

The per-object transforms are sampled from a few low-dimensional motion
parameters on the config (max translation / rotation). The distribution-family
refinement (von Mises directions, half-normal magnitudes, ``num_moving`` as a
Binomial, etc. -- see kubric/plan.md Workstream C) lives on the search sampler
side; this builder just consumes concrete bounds and samples within them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyquaternion as pyquat

import kubric as kb

from interface.asset_camera_motion_scene import AssetDollyInConfig, AssetDollyInScene


@dataclass(frozen=True)
class MovingAssetDollyInConfig(AssetDollyInConfig):
  """Adds rigid object-motion parameters to the static asset-dolly config."""

  num_moving: int = 0
  motion_translation: float = 0.0   # mean world-unit displacement magnitude per mover
  motion_translation_std: float = 0.0  # std of per-mover translation magnitude (0 => fixed = legacy)
  motion_rotation: float = 0.0      # mean rotation angle magnitude (radians) per mover
  motion_rotation_std: float = 0.0  # std of per-mover rotation angle (0 => fixed = legacy)
  motion_vertical_frac: float = 0.0  # 0 => grounded slide; >0 allows out-of-plane (in-depth) motion
  motion_seed: int = 0              # decorrelates motion sampling from asset placement


class MovingAssetDollyInScene(AssetDollyInScene):
  """Static dolly-in scene with per-object rigid SE(3) keyframed motion."""

  def __init__(
      self,
      config: MovingAssetDollyInConfig | None = None,
      asset_scratch_dir: str | Path | None = None,
  ):
    super().__init__(config or MovingAssetDollyInConfig(), asset_scratch_dir)
    self.config: MovingAssetDollyInConfig
    # Each entry: dict(obj, start_pos, end_pos, q_start, q_end).
    self._movers: list[dict] = []

  def build(self) -> kb.Scene:
    scene = super().build()
    scene.metadata["motion_intervention"] = {
        "type": "camera_dolly_in_moving_asset_scene",
        **asdict(self.config),
    }
    return scene

  def _add_static_geometry(self, scene: kb.Scene) -> None:
    """Place assets like the parent, then mark/keyframe-prep the movers.

    Mirrors ``AssetDollyInScene._add_static_geometry`` so placement stays
    identical, but the first ``num_moving`` assets get ``static = False`` and a
    sampled end-of-scenelet pose recorded for ``keyframe_object_paths``.
    """
    scene += kb.Cube(
        name="matte_floor",
        scale=(8.0, 8.0, 0.05),
        position=(0.0, 0.0, -0.05),
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(0.55, 0.56, 0.58), roughness=0.9, specular=0.1
        ),
        static=True,
        background=True,
    )

    self.asset_source = kb.AssetSource.from_manifest(
        self.config.asset_manifest,
        scratch_dir=self.asset_scratch_dir,
    )
    rng = np.random.RandomState(self.config.seed)
    asset_ids = self._select_asset_ids(self.asset_source, rng)
    positions = self._asset_positions(len(asset_ids))

    num_moving = int(np.clip(self.config.num_moving, 0, len(asset_ids)))
    motion_rng = np.random.RandomState(self.config.seed + 1 + self.config.motion_seed)

    self._movers = []
    for idx, (asset_id, position_xy) in enumerate(zip(asset_ids, positions)):
      obj = self.asset_source.create(asset_id=asset_id, name=f"asset_{idx:02d}_{asset_id}")
      assert isinstance(obj, kb.FileBasedObject)
      self._normalize_asset_size(obj)
      self._place_asset_on_floor(obj, position_xy)
      if self.config.color_assets:
        self._assign_asset_material(obj, rng)
      obj.metadata["asset_smoke_index"] = idx

      is_moving = idx < num_moving
      obj.static = not is_moving
      if is_moving:
        self._prepare_object_motion(obj, motion_rng)
      scene += obj

    scene.metadata["asset_ids"] = list(asset_ids)
    scene.metadata["num_moving"] = num_moving

  @staticmethod
  def _sample_magnitude(rng: np.random.RandomState, mean: float, std: float) -> float:
    """Per-mover motion magnitude. std>0 => Gaussian(mean, std) clamped to >=0;
    std==0 => the fixed mean with no RNG draw (legacy-identical)."""
    if std > 0.0:
      return max(0.0, float(rng.normal(mean, std)))
    return float(mean)

  def _prepare_object_motion(self, obj: kb.FileBasedObject, rng: np.random.RandomState) -> None:
    """Sample a rigid end pose for one mover and record it for keyframing."""
    start_pos = np.asarray(obj.position, dtype=np.float64)
    q_start = pyquat.Quaternion(*np.asarray(obj.quaternion, dtype=np.float64))

    # Translation: random horizontal direction + optional out-of-plane component.
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([np.cos(phi), np.sin(phi),
                          self.config.motion_vertical_frac * rng.uniform(-1.0, 1.0)])
    direction /= np.linalg.norm(direction)
    # Per-mover magnitude ~ Gaussian(mean, std) so movers vary in speed (FlyingThings
    # has a distribution of per-object speeds, not one shared magnitude). std == 0
    # reproduces the legacy fixed-magnitude behavior *and* draws no RNG, so existing
    # thetas render bit-identically.
    trans_mag = self._sample_magnitude(
        rng, self.config.motion_translation, self.config.motion_translation_std)
    end_pos = start_pos + direction * trans_mag
    # Keep grounded objects from sinking below the floor.
    end_pos[2] = max(end_pos[2], float(start_pos[2]))

    # Rotation: per-mover angle magnitude about a random axis.
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rot_mag = self._sample_magnitude(
        rng, self.config.motion_rotation, self.config.motion_rotation_std)
    delta = pyquat.Quaternion(axis=axis, angle=rot_mag)
    q_end = (delta * q_start).normalised

    obj.metadata["object_motion"] = {
        "start_position": start_pos.tolist(),
        "end_position": end_pos.tolist(),
        "start_quaternion": list(q_start.elements),
        "end_quaternion": list(q_end.elements),
        "translation_magnitude": trans_mag,
        "rotation_angle": rot_mag,
    }
    self._movers.append({
        "obj": obj,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "q_start": q_start,
        "q_end": q_end,
    })

  def keyframe_object_paths(self, scene: kb.Scene) -> None:
    """Insert per-object motion keyframes.

    Call this after creating the Blender renderer (same contract as
    ``keyframe_camera_path``) so the keyframes mirror onto Blender objects.
    Poses are clamped outside ``[frame_start, frame_end]`` so the boundary
    frames produce valid forward/backward flow, matching the camera path.
    """
    cfg = self.config
    visible_span = max(1, cfg.frame_end - cfg.frame_start)
    for frame in range(cfg.frame_start - 1, cfg.frame_end + 2):
      interp = float(np.clip((frame - cfg.frame_start) / visible_span, 0.0, 1.0))
      for mover in self._movers:
        obj = mover["obj"]
        pos = (1.0 - interp) * mover["start_pos"] + interp * mover["end_pos"]
        quat = pyquat.Quaternion.slerp(mover["q_start"], mover["q_end"], interp)
        obj.position = tuple(float(v) for v in pos)
        obj.quaternion = tuple(float(v) for v in quat.elements)
        obj.keyframe_insert("position", frame)
        obj.keyframe_insert("quaternion", frame)
