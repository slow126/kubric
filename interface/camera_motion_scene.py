"""Static Kubric scene with an interventive camera motion path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np

import kubric as kb


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class DollyInConfig:
  """Configuration for a static scene rendered with dolly-in camera motion."""

  resolution: Tuple[int, int] = (256, 256)
  frame_start: int = 1
  frame_end: int = 4
  frame_rate: int = 1
  start_distance: float = 14.0
  end_distance: float = 3.5
  target: Vector3 = (0.0, 0.0, 0.8)
  camera_direction: Vector3 = (1.0, -1.0, 0.45)
  focal_length: float = 35.0
  sensor_width: float = 32.0

  @property
  def num_frames(self) -> int:
    return self.frame_end - self.frame_start + 1


class StaticDollyInScene:
  """Builds a deterministic static scene and keyframes only the camera.

  The object poses are constant for all frames. Camera keyframes are inserted
  one frame before and after the rendered range so Blender can produce valid
  forward/backward optical flow on the boundary frames.
  """

  def __init__(self, config: DollyInConfig | None = None):
    self.config = config or DollyInConfig()

  def build(self) -> kb.Scene:
    cfg = self.config
    scene = kb.Scene(
        resolution=cfg.resolution,
        frame_start=cfg.frame_start,
        frame_end=cfg.frame_end,
        frame_rate=cfg.frame_rate,
    )
    scene.metadata["motion_intervention"] = {
        "type": "camera_dolly_in_static_scene",
        **asdict(cfg),
    }

    self._add_static_geometry(scene)
    self._add_lighting(scene)
    self._add_camera(scene)
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
    scene += kb.Cube(
        name="red_cube",
        scale=(0.55, 0.55, 0.55),
        position=(-0.9, -0.25, 0.55),
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(0.85, 0.18, 0.14), roughness=0.55, specular=0.2
        ),
        static=True,
    )
    scene += kb.Sphere(
        name="blue_sphere",
        scale=0.5,
        position=(0.45, 0.35, 0.5),
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(0.12, 0.36, 0.95), roughness=0.35, specular=0.35
        ),
        static=True,
    )
    scene += kb.Cube(
        name="green_tall_box",
        scale=(0.35, 0.35, 0.9),
        position=(1.15, -0.7, 0.9),
        material=kb.PrincipledBSDFMaterial(
            color=kb.Color(0.18, 0.62, 0.24), roughness=0.65, specular=0.15
        ),
        static=True,
    )

  def _add_lighting(self, scene: kb.Scene) -> None:
    scene += kb.DirectionalLight(
        name="key_light",
        position=(-3.0, -4.0, 7.0),
        look_at=(0.0, 0.0, 0.5),
        intensity=2.2,
    )
    scene += kb.PointLight(
        name="fill_light",
        position=(3.5, 2.0, 4.0),
        intensity=70.0,
    )
    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

  def _add_camera(self, scene: kb.Scene) -> None:
    cfg = self.config
    scene.camera = kb.PerspectiveCamera(
        name="dolly_camera",
        focal_length=cfg.focal_length,
        sensor_width=cfg.sensor_width,
    )
    self._set_camera_pose(scene, cfg.frame_start)

  def keyframe_camera_path(self, scene: kb.Scene) -> None:
    """Insert dolly-in camera keyframes.

    Call this after creating the Blender renderer so the keyframes are mirrored
    onto Blender's camera object.
    """
    cfg = self.config
    visible_span = max(1, cfg.frame_end - cfg.frame_start)
    for frame in range(cfg.frame_start - 1, cfg.frame_end + 2):
      interp = (frame - cfg.frame_start) / visible_span
      interp = float(np.clip(interp, 0.0, 1.0))
      self._set_camera_pose(scene, frame=frame, interp=interp)
      scene.camera.keyframe_insert("position", frame)
      scene.camera.keyframe_insert("quaternion", frame)

  def _set_camera_pose(
      self,
      scene: kb.Scene,
      frame: int,
      interp: float | None = None,
  ) -> None:
    cfg = self.config
    if interp is None:
      visible_span = max(1, cfg.frame_end - cfg.frame_start)
      interp = (frame - cfg.frame_start) / visible_span
      interp = float(np.clip(interp, 0.0, 1.0))
    direction = np.asarray(cfg.camera_direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    target = np.asarray(cfg.target, dtype=np.float64)
    distance = (1.0 - interp) * cfg.start_distance + interp * cfg.end_distance
    scene.camera.position = tuple(target + direction * distance)
    scene.camera.look_at(tuple(target))
