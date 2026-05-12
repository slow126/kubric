"""Render a four-frame static-scene dolly-in camera smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import kubric as kb
from kubric.renderer.blender import Blender

from interface.camera_motion_scene import DollyInConfig, StaticDollyInScene


def parse_resolution(value: str) -> tuple[int, int]:
  if "x" in value:
    height, width = value.lower().split("x", maxsplit=1)
    return int(height), int(width)
  size = int(value)
  return size, size


def to_uint8_rgb(rgba: np.ndarray) -> np.ndarray:
  rgb = rgba[..., :3]
  if np.issubdtype(rgb.dtype, np.floating):
    rgb = np.clip(rgb, 0.0, 1.0) * 255.0
  return rgb.astype(np.uint8)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
  h = hsv[..., 0] % 1.0
  s = np.clip(hsv[..., 1], 0.0, 1.0)
  v = np.clip(hsv[..., 2], 0.0, 1.0)

  i = np.floor(h * 6.0).astype(np.int32)
  f = h * 6.0 - i
  p = v * (1.0 - s)
  q = v * (1.0 - f * s)
  t = v * (1.0 - (1.0 - f) * s)
  i = i % 6

  rgb = np.zeros(hsv.shape, dtype=np.float32)
  choices = [
      (v, t, p),
      (q, v, p),
      (p, v, t),
      (p, q, v),
      (t, p, v),
      (v, p, q),
  ]
  for idx, channels in enumerate(choices):
    mask = i == idx
    for channel, values in enumerate(channels):
      rgb[..., channel] = np.where(mask, values, rgb[..., channel])
  return rgb


def make_flow_colorwheel() -> np.ndarray:
  """Middlebury-style optical flow color wheel."""
  transitions = [
      (15, [255, 0, 0], [255, 255, 0]),      # red -> yellow
      (6, [255, 255, 0], [0, 255, 0]),       # yellow -> green
      (4, [0, 255, 0], [0, 255, 255]),       # green -> cyan
      (11, [0, 255, 255], [0, 0, 255]),      # cyan -> blue
      (13, [0, 0, 255], [255, 0, 255]),      # blue -> magenta
      (6, [255, 0, 255], [255, 0, 0]),       # magenta -> red
  ]
  wheel = []
  for n_steps, start, end in transitions:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    for idx in range(n_steps):
      alpha = idx / n_steps
      wheel.append((1.0 - alpha) * start + alpha * end)
  return np.asarray(wheel, dtype=np.float32) / 255.0


FLOW_COLORWHEEL = make_flow_colorwheel()


def flow_to_rgb(
    flow: np.ndarray,
    max_magnitude: float | None = None,
    zero_color: str = "black",
) -> np.ndarray:
  """Convert (delta_row, delta_col) flow to a color-wheel image.

  Direction selects a color from the optical-flow wheel. Magnitude controls the
  interpolation from the zero-flow color to that direction color.
  """
  magnitude = np.linalg.norm(flow, axis=-1)
  if max_magnitude is None:
    max_magnitude = float(np.percentile(magnitude, 99.0))
  max_magnitude = max(max_magnitude, 1.0e-6)
  normalized_magnitude = np.clip(magnitude / max_magnitude, 0.0, 1.0)

  delta_row = flow[..., 0]
  delta_col = flow[..., 1]
  angle = np.arctan2(-delta_row, -delta_col) / np.pi
  wheel_position = (angle + 1.0) / 2.0 * (len(FLOW_COLORWHEEL) - 1)
  lower = np.floor(wheel_position).astype(np.int32)
  upper = (lower + 1) % len(FLOW_COLORWHEEL)
  weight = (wheel_position - lower)[..., None]
  direction_rgb = (
      (1.0 - weight) * FLOW_COLORWHEEL[lower] + weight * FLOW_COLORWHEEL[upper]
  )

  if zero_color == "white":
    zero_rgb = np.ones_like(direction_rgb)
  elif zero_color == "black":
    zero_rgb = np.zeros_like(direction_rgb)
  else:
    raise ValueError(f"Unsupported zero flow color: {zero_color}")

  rgb = (
      (1.0 - normalized_magnitude[..., None]) * zero_rgb
      + normalized_magnitude[..., None] * direction_rgb
  )
  return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def robust_flow_scale(flow: np.ndarray, percentile: float) -> float:
  magnitude = np.linalg.norm(flow, axis=-1)
  nonzero = magnitude[magnitude > 1.0e-6]
  if nonzero.size == 0:
    return 1.0
  return max(float(np.percentile(nonzero, percentile)), 1.0e-6)


def flow_magnitude_to_rgb(flow: np.ndarray, max_magnitude: float) -> np.ndarray:
  magnitude = np.linalg.norm(flow, axis=-1)
  normalized = np.clip(magnitude / max(max_magnitude, 1.0e-6), 0.0, 1.0)
  # Simple black -> blue -> cyan -> yellow -> white heatmap without matplotlib.
  stops = np.array([
      [0, 0, 0],
      [0, 64, 255],
      [0, 220, 255],
      [255, 230, 0],
      [255, 255, 255],
  ], dtype=np.float32)
  scaled = normalized * (len(stops) - 1)
  lower = np.floor(scaled).astype(np.int32)
  upper = np.clip(lower + 1, 0, len(stops) - 1)
  lower = np.clip(lower, 0, len(stops) - 1)
  weight = (scaled - lower)[..., None]
  rgb = (1.0 - weight) * stops[lower] + weight * stops[upper]
  return rgb.astype(np.uint8)


def make_contact_sheet(images: Iterable[np.ndarray], pad: int = 8) -> np.ndarray:
  frames = list(images)
  if not frames:
    raise ValueError("Cannot build a contact sheet from zero images.")
  height, width, channels = frames[0].shape
  sheet = np.full(
      (height, len(frames) * width + (len(frames) - 1) * pad, channels),
      255,
      dtype=np.uint8,
  )
  x = 0
  for frame in frames:
    sheet[:, x:x + width] = frame
    x += width + pad
  return sheet


def draw_disk(
    image: np.ndarray,
    row: float,
    col: float,
    radius: int,
    color: tuple[int, int, int],
) -> None:
  center_row = int(round(row))
  center_col = int(round(col))
  height, width = image.shape[:2]
  for y in range(max(0, center_row - radius), min(height, center_row + radius + 1)):
    for x in range(max(0, center_col - radius), min(width, center_col + radius + 1)):
      if (y - center_row) ** 2 + (x - center_col) ** 2 <= radius ** 2:
        image[y, x] = color


def draw_line(
    image: np.ndarray,
    row0: float,
    col0: float,
    row1: float,
    col1: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
  steps = int(max(abs(row1 - row0), abs(col1 - col0))) + 1
  if steps <= 1:
    draw_disk(image, row0, col0, thickness, color)
    return
  rows = np.linspace(row0, row1, steps)
  cols = np.linspace(col0, col1, steps)
  for row, col in zip(rows, cols):
    draw_disk(image, row, col, thickness, color)


def sample_correspondence_points(
    rgb_frame: np.ndarray,
    flow_frame: np.ndarray,
    stride: int,
    max_points: int,
    brightness_threshold: float,
    seed: int,
) -> list[tuple[int, int]]:
  height, width = rgb_frame.shape[:2]
  candidates = []
  for row in range(stride // 2, height, stride):
    for col in range(stride // 2, width, stride):
      delta_row, delta_col = flow_frame[row, col]
      target_row = row + float(delta_row)
      target_col = col + float(delta_col)
      if not np.isfinite(delta_row) or not np.isfinite(delta_col):
        continue
      if not (0 <= target_row < height and 0 <= target_col < width):
        continue
      if np.linalg.norm(flow_frame[row, col]) <= 1.0e-6:
        continue
      if float(np.mean(rgb_frame[row, col])) < brightness_threshold:
        continue
      candidates.append((row, col))

  if len(candidates) <= max_points:
    return candidates

  rng = np.random.RandomState(seed)
  selected = rng.choice(len(candidates), size=max_points, replace=False)
  return [candidates[idx] for idx in sorted(selected)]


def make_keypoint_flow_contact_sheet(
    rgb: np.ndarray,
    forward_flow: np.ndarray,
    stride: int = 32,
    max_points_per_pair: int = 45,
    brightness_threshold: float = 12.0,
    pad: int = 8,
    seed: int = 0,
) -> np.ndarray:
  sheet = make_contact_sheet(rgb, pad=pad)
  frame_count, height, width = rgb.shape[:3]
  palette = [
      (255, 64, 64),
      (255, 180, 0),
      (80, 220, 80),
      (0, 210, 255),
      (120, 120, 255),
      (255, 80, 230),
  ]

  for frame_idx in range(frame_count - 1):
    points = sample_correspondence_points(
        rgb[frame_idx],
        forward_flow[frame_idx],
        stride=stride,
        max_points=max_points_per_pair,
        brightness_threshold=brightness_threshold,
        seed=seed + frame_idx,
    )
    source_offset = frame_idx * (width + pad)
    target_offset = (frame_idx + 1) * (width + pad)
    for point_idx, (row, col) in enumerate(points):
      delta_row, delta_col = forward_flow[frame_idx, row, col]
      target_row = row + float(delta_row)
      target_col = col + float(delta_col)
      color = palette[point_idx % len(palette)]
      draw_line(
          sheet,
          row,
          source_offset + col,
          target_row,
          target_offset + target_col,
          color=color,
          thickness=1,
      )
      draw_disk(sheet, row, source_offset + col, radius=2, color=(255, 255, 255))
      draw_disk(sheet, row, source_offset + col, radius=1, color=color)
      draw_disk(sheet, target_row, target_offset + target_col, radius=2, color=(0, 0, 0))
      draw_disk(sheet, target_row, target_offset + target_col, radius=1, color=color)

  return sheet


def write_flow_batch_safely(flow: np.ndarray, output_dir: Path, name: str) -> None:
  min_value = float(np.min(flow))
  max_value = float(np.max(flow))
  if max_value == min_value:
    encoded = np.zeros(flow.shape, dtype=np.uint16)
  else:
    encoded = ((flow - min_value) * 65535.0 / (max_value - min_value)).astype(np.uint16)

  for idx, frame in enumerate(encoded):
    kb.write_png(frame, output_dir / f"{name}_{idx:05d}.png")

  range_path = output_dir / "data_ranges.json"
  if range_path.exists():
    with range_path.open("r", encoding="utf-8") as fp:
      ranges = json.load(fp)
  else:
    ranges = {}
  ranges[name] = {"min": min_value, "max": max_value}
  with range_path.open("w", encoding="utf-8") as fp:
    json.dump(ranges, fp, indent=2, sort_keys=True)


def write_camera_metadata(scene: kb.Scene, output_dir: Path) -> None:
  payload = {
      "metadata": kb.get_scene_metadata(scene),
      "camera": kb.get_camera_info(scene.camera),
  }
  with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
    json.dump(payload, fp, indent=2, default=lambda value: np.asarray(value).tolist())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", default="interface/output/dolly_in_smoke")
  parser.add_argument("--scratch-dir", default="interface/output/dolly_in_smoke_scratch")
  parser.add_argument("--resolution", default="256x256")
  parser.add_argument("--frame-count", type=int, default=4)
  parser.add_argument("--start-distance", type=float, default=14.0)
  parser.add_argument("--end-distance", type=float, default=3.5)
  parser.add_argument("--samples-per-pixel", type=int, default=64)
  parser.add_argument("--flow-viz-percentile", type=float, default=95.0)
  parser.add_argument("--flow-zero-color", choices=["black", "white"], default="black")
  parser.add_argument("--keypoint-stride", type=int, default=32)
  parser.add_argument("--keypoint-max-points", type=int, default=45)
  parser.add_argument("--save-blend", action="store_true")
  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  scratch_dir = Path(args.scratch_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  scratch_dir.mkdir(parents=True, exist_ok=True)

  config = DollyInConfig(
      resolution=parse_resolution(args.resolution),
      frame_end=args.frame_count,
      start_distance=args.start_distance,
      end_distance=args.end_distance,
  )
  builder = StaticDollyInScene(config)
  scene = builder.build()
  renderer = Blender(scene, scratch_dir=scratch_dir, samples_per_pixel=args.samples_per_pixel)
  builder.keyframe_camera_path(scene)

  if args.save_blend:
    renderer.save_state(output_dir / "scene.blend")

  frames = renderer.render()
  rgb = to_uint8_rgb(frames["rgba"])
  forward_flow_scale = robust_flow_scale(frames["forward_flow"], args.flow_viz_percentile)
  backward_flow_scale = robust_flow_scale(frames["backward_flow"], args.flow_viz_percentile)
  forward_flow_rgb = np.stack([
      flow_to_rgb(
          flow,
          max_magnitude=forward_flow_scale,
          zero_color=args.flow_zero_color,
      )
      for flow in frames["forward_flow"]
  ])
  backward_flow_rgb = np.stack([
      flow_to_rgb(
          flow,
          max_magnitude=backward_flow_scale,
          zero_color=args.flow_zero_color,
      )
      for flow in frames["backward_flow"]
  ])
  forward_flow_mag_rgb = np.stack([
      flow_magnitude_to_rgb(flow, max_magnitude=forward_flow_scale)
      for flow in frames["forward_flow"]
  ])
  backward_flow_mag_rgb = np.stack([
      flow_magnitude_to_rgb(flow, max_magnitude=backward_flow_scale)
      for flow in frames["backward_flow"]
  ])

  non_flow_frames = {
      key: value for key, value in frames.items()
      if key not in {"forward_flow", "backward_flow"}
  }
  kb.write_image_dict(non_flow_frames, output_dir)
  write_flow_batch_safely(frames["forward_flow"], output_dir, "forward_flow")
  write_flow_batch_safely(frames["backward_flow"], output_dir, "backward_flow")
  imageio.mimsave(output_dir / "rgb.gif", list(rgb), duration=0.6)
  imageio.imwrite(output_dir / "rgb_contact_sheet.png", make_contact_sheet(rgb))
  imageio.imwrite(
      output_dir / "forward_flow_contact_sheet.png",
      make_contact_sheet(forward_flow_rgb),
  )
  imageio.imwrite(
      output_dir / "backward_flow_contact_sheet.png",
      make_contact_sheet(backward_flow_rgb),
  )
  imageio.imwrite(
      output_dir / "forward_flow_magnitude_contact_sheet.png",
      make_contact_sheet(forward_flow_mag_rgb),
  )
  imageio.imwrite(
      output_dir / "backward_flow_magnitude_contact_sheet.png",
      make_contact_sheet(backward_flow_mag_rgb),
  )
  imageio.imwrite(
      output_dir / "keypoint_correspondence_contact_sheet.png",
      make_keypoint_flow_contact_sheet(
          rgb,
          frames["forward_flow"],
          stride=args.keypoint_stride,
          max_points_per_pair=args.keypoint_max_points,
      ),
  )
  write_camera_metadata(scene, output_dir)

  print(f"Wrote render outputs to {output_dir}")
  print(f"Wrote RGB GIF to {output_dir / 'rgb.gif'}")
  print(f"Wrote RGB contact sheet to {output_dir / 'rgb_contact_sheet.png'}")
  print(f"Wrote flow contact sheet to {output_dir / 'forward_flow_contact_sheet.png'}")
  print(
      f"Forward flow viz scale: {forward_flow_scale:.3f} px "
      f"at p{args.flow_viz_percentile:g}"
  )
  print(
      f"Backward flow viz scale: {backward_flow_scale:.3f} px "
      f"at p{args.flow_viz_percentile:g}"
  )


if __name__ == "__main__":
  main()
