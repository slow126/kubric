"""Static dolly-in scene populated with Kubric manifest assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import kubric as kb

from interface.camera_motion_scene import DollyInConfig, StaticDollyInScene


KUBASIC_SMOKE_ASSETS = (
    "cone",
    "torus",
    "gear",
    "torus_knot",
    "sponge",
    "spot",
    "teapot",
    "suzanne",
)


@dataclass(frozen=True)
class AssetDollyInConfig(DollyInConfig):
  """Configuration for a static asset scene with dolly-in camera motion."""

  asset_manifest: str = "gs://kubric-public/assets/KuBasic/KuBasic.json"
  seed: int = 7
  num_assets: int = 4
  asset_scale: float = 1.1
  asset_ids: tuple[str, ...] = ()
  color_assets: bool = True


class AssetDollyInScene(StaticDollyInScene):
  """Builds a static scene using random assets from a Kubric manifest."""

  def __init__(
      self,
      config: AssetDollyInConfig | None = None,
      asset_scratch_dir: str | Path | None = None,
  ):
    super().__init__(config or AssetDollyInConfig())
    self.config: AssetDollyInConfig
    self.asset_scratch_dir = asset_scratch_dir
    self.asset_source: kb.AssetSource | None = None

  def build(self) -> kb.Scene:
    scene = super().build()
    scene.metadata["motion_intervention"] = {
        "type": "camera_dolly_in_static_asset_scene",
        **asdict(self.config),
    }
    return scene

  def _add_static_geometry(self, scene: kb.Scene) -> None:
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

    for idx, (asset_id, position_xy) in enumerate(zip(asset_ids, positions)):
      obj = self.asset_source.create(asset_id=asset_id, name=f"asset_{idx:02d}_{asset_id}")
      assert isinstance(obj, kb.FileBasedObject)
      self._normalize_asset_size(obj)
      self._place_asset_on_floor(obj, position_xy)
      if self.config.color_assets:
        self._assign_asset_material(obj, rng)
      obj.static = True
      obj.metadata["asset_smoke_index"] = idx
      scene += obj

    scene.metadata["asset_ids"] = list(asset_ids)

  def _select_asset_ids(
      self,
      asset_source: kb.AssetSource,
      rng: np.random.RandomState,
  ) -> list[str]:
    if self.config.asset_ids:
      return list(self.config.asset_ids)

    available = asset_source.all_asset_ids
    if self.config.asset_manifest.endswith("KuBasic.json"):
      candidates = [asset_id for asset_id in KUBASIC_SMOKE_ASSETS if asset_id in available]
    else:
      candidates = [asset_id for asset_id in available if asset_id != "dome"]

    if not candidates:
      raise ValueError(f"No usable assets found in manifest {self.config.asset_manifest}")

    count = min(self.config.num_assets, len(candidates))
    return list(rng.choice(candidates, size=count, replace=False))

  def _normalize_asset_size(self, obj: kb.FileBasedObject) -> None:
    bounds = np.asarray(obj.bounds, dtype=np.float32)
    largest_extent = float(np.max(bounds[1] - bounds[0]))
    if largest_extent <= 0:
      largest_extent = 1.0
    obj.scale = self.config.asset_scale / largest_extent

  def _place_asset_on_floor(
      self,
      obj: kb.FileBasedObject,
      position_xy: tuple[float, float],
  ) -> None:
    bounds = np.asarray(obj.bounds, dtype=np.float32)
    scaled_bounds = bounds * np.asarray(obj.scale, dtype=np.float32)
    z = float(-scaled_bounds[0, 2])
    obj.position = (position_xy[0], position_xy[1], z)

  @staticmethod
  def _assign_asset_material(
      obj: kb.FileBasedObject,
      rng: np.random.RandomState,
  ) -> None:
    color = kb.Color.from_hsv(
        h=float(rng.uniform(0.0, 1.0)),
        s=float(rng.uniform(0.45, 0.85)),
        v=float(rng.uniform(0.65, 0.95)),
    )
    obj.material = kb.PrincipledBSDFMaterial(
        color=color,
        roughness=float(rng.uniform(0.45, 0.8)),
        specular=float(rng.uniform(0.15, 0.35)),
    )
    obj.metadata["smoke_material_color"] = color.rgb

  @staticmethod
  def _asset_positions(count: int) -> Sequence[tuple[float, float]]:
    base_positions = [
        (-1.35, -0.55),
        (-0.25, 0.65),
        (0.95, -0.45),
        (1.65, 0.55),
        (-1.75, 0.85),
        (1.95, -0.95),
    ]
    if count <= len(base_positions):
      return base_positions[:count]

    positions = list(base_positions)
    for idx in range(count - len(base_positions)):
      row = idx // 3
      col = idx % 3
      positions.append((-1.5 + col * 1.5, 1.45 + row * 1.1))
    return positions
