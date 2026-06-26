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
  end_focal_length: float = 0.0   # if >0: keyframe focal_length -> end_focal_length over the visible frames (pure magnification zoom, NO camera translation -> no parallax/disocclusion)
  sensor_width: float = 32.0
  camera_motion: str = "dolly"   # "dolly" (forward zoom) | "lateral" (stereo baseline) | "orbit" (arc around target)
  camera_baseline: float = 0.0   # lateral translation (world units) when camera_motion="lateral"
  orbit_azimuth_span: float = 0.0    # radians swept in azimuth for camera_motion="orbit" (pi ~= 180-deg multi-view sweep)
  orbit_elevation_span: float = 0.0  # radians swept in elevation for camera_motion="orbit"

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
    """Insert camera keyframes (dolly-in or lateral stereo baseline).

    Call this after creating the Blender renderer so the keyframes are mirrored
    onto Blender's camera object.
    """
    self._keyframe_focal(scene)  # focal zoom (if end_focal_length>0); independent of camera path
    if self.config.camera_motion == "lateral":
      self._keyframe_lateral_path(scene)
      return
    if self.config.camera_motion == "orbit":
      self._keyframe_orbit_path(scene)
      return
    cfg = self.config
    visible_span = max(1, cfg.frame_end - cfg.frame_start)
    for frame in range(cfg.frame_start - 1, cfg.frame_end + 2):
      interp = (frame - cfg.frame_start) / visible_span
      interp = float(np.clip(interp, 0.0, 1.0))
      self._set_camera_pose(scene, frame=frame, interp=interp)
      scene.camera.keyframe_insert("position", frame)
      scene.camera.keyframe_insert("quaternion", frame)

  def _keyframe_focal(self, scene: kb.Scene) -> None:
    """Keyframe the camera focal_length from focal_length -> end_focal_length over
    the visible frames. Pure magnification (no camera translation), so it produces
    a radial zoom flow WITHOUT parallax or disocclusion -- the 3D analog of an
    occlusion-free warp. No-op unless end_focal_length>0 and differs from focal_length."""
    cfg = self.config
    end_fl = float(getattr(cfg, "end_focal_length", 0.0) or 0.0)
    if end_fl <= 0.0 or end_fl == cfg.focal_length:
      return
    visible_span = max(1, cfg.frame_end - cfg.frame_start)
    for frame in range(cfg.frame_start - 1, cfg.frame_end + 2):
      interp = float(np.clip((frame - cfg.frame_start) / visible_span, 0.0, 1.0))
      scene.camera.focal_length = (1.0 - interp) * cfg.focal_length + interp * end_fl
      scene.camera.keyframe_insert("focal_length", frame)

  def _keyframe_lateral_path(self, scene: kb.Scene) -> None:
    """Rectified stereo: camera slides along its right axis at fixed distance and
    fixed (parallel) orientation, producing horizontal-disparity flow."""
    cfg = self.config
    direction = np.asarray(cfg.camera_direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    target = np.asarray(cfg.target, dtype=np.float64)
    base_pos = target + direction * cfg.start_distance
    # Aim once at the target, then lock orientation for all frames (parallel).
    scene.camera.position = tuple(float(v) for v in base_pos)
    scene.camera.look_at(tuple(float(v) for v in target))
    fixed_quat = tuple(float(q) for q in scene.camera.quaternion)
    view_dir = target - base_pos
    view_dir = view_dir / np.linalg.norm(view_dir)
    right = np.cross(view_dir, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    visible_span = max(1, cfg.frame_end - cfg.frame_start)
    for frame in range(cfg.frame_start - 1, cfg.frame_end + 2):
      interp = float(np.clip((frame - cfg.frame_start) / visible_span, 0.0, 1.0))
      pos = base_pos + (interp - 0.5) * cfg.camera_baseline * right
      scene.camera.position = tuple(float(v) for v in pos)
      scene.camera.quaternion = fixed_quat
      scene.camera.keyframe_insert("position", frame)
      scene.camera.keyframe_insert("quaternion", frame)

  def _keyframe_orbit_path(self, scene: kb.Scene) -> None:
    """Camera arcs along the sphere around the target, re-aiming at it each frame.

    Sweeps azimuth/elevation by ``orbit_azimuth_span``/``orbit_elevation_span``
    over the visible frames at fixed distance (or a spiral if start/end differ),
    producing multi-view / rotational flow rather than the radial field of a
    forward dolly. Uses the same azimuth/elevation -> direction convention as
    ``_camera_direction``: [cos(az)cos(el), sin(az)cos(el), sin(el)]."""
    cfg = self.config
    target = np.asarray(cfg.target, dtype=np.float64)
    d = np.asarray(cfg.camera_direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    el0 = float(np.arcsin(np.clip(d[2], -1.0, 1.0)))   # recover base elevation
    az0 = float(np.arctan2(d[1], d[0]))                # recover base azimuth
    visible_span = max(1, cfg.frame_end - cfg.frame_start)
    for frame in range(cfg.frame_start - 1, cfg.frame_end + 2):
      interp = float(np.clip((frame - cfg.frame_start) / visible_span, 0.0, 1.0))
      az = az0 + interp * cfg.orbit_azimuth_span
      el = el0 + interp * cfg.orbit_elevation_span
      direction = np.array([np.cos(az) * np.cos(el),
                            np.sin(az) * np.cos(el),
                            np.sin(el)], dtype=np.float64)
      distance = (1.0 - interp) * cfg.start_distance + interp * cfg.end_distance
      scene.camera.position = tuple(float(v) for v in (target + direction * distance))
      scene.camera.look_at(tuple(float(v) for v in target))
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
