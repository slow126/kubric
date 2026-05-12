"""Preview one pre-rendered MOVi TFDS example.

This reads an already-built Kubric MOVi dataset from TFDS/GCS, writes an RGB
GIF/contact sheet, and writes optical-flow preview sheets. It does not render
new Kubric scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import imageio
import numpy as np
import tensorflow_datasets as tfds


def to_numpy_example(dataset_name: str, data_dir: str, split: str, index: int) -> dict:
  ds = tfds.load(
      dataset_name,
      data_dir=data_dir,
      split=split,
      shuffle_files=False,
  )
  if index:
    ds = ds.skip(index)
  for example in tfds.as_numpy(ds.take(1)):
    return example
  raise ValueError(f"No example found for split={split!r}, index={index}")


def decode_flow(flow_uint16: np.ndarray, flow_range: np.ndarray) -> np.ndarray:
  flow = flow_uint16.astype(np.float32)
  min_value, max_value = [float(value) for value in flow_range]
  return flow / 65535.0 * (max_value - min_value) + min_value


def make_contact_sheet(images: Iterable[np.ndarray], pad: int = 8) -> np.ndarray:
  frames = [np.asarray(image, dtype=np.uint8) for image in images]
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


def make_flow_colorwheel() -> np.ndarray:
  transitions = [
      (15, [255, 0, 0], [255, 255, 0]),
      (6, [255, 255, 0], [0, 255, 0]),
      (4, [0, 255, 0], [0, 255, 255]),
      (11, [0, 255, 255], [0, 0, 255]),
      (13, [0, 0, 255], [255, 0, 255]),
      (6, [255, 0, 255], [255, 0, 0]),
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


def robust_flow_scale(flow: np.ndarray, percentile: float) -> float:
  magnitude = np.linalg.norm(flow, axis=-1)
  nonzero = magnitude[magnitude > 1.0e-6]
  if nonzero.size == 0:
    return 1.0
  return max(float(np.percentile(nonzero, percentile)), 1.0e-6)


def flow_to_rgb(
    flow: np.ndarray,
    max_magnitude: float,
    zero_color: str = "black",
) -> np.ndarray:
  magnitude = np.linalg.norm(flow, axis=-1)
  normalized_magnitude = np.clip(magnitude / max(max_magnitude, 1.0e-6), 0.0, 1.0)

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


def flow_magnitude_to_rgb(flow: np.ndarray, max_magnitude: float) -> np.ndarray:
  magnitude = np.linalg.norm(flow, axis=-1)
  normalized = np.clip(magnitude / max(max_magnitude, 1.0e-6), 0.0, 1.0)
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


def bytes_to_text(value) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, np.ndarray) and value.shape == ():
    return bytes_to_text(value.item())
  return str(value)


def write_preview(example: dict, output_dir: Path, args: argparse.Namespace) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  video = np.asarray(example["video"], dtype=np.uint8)
  metadata = example["metadata"]
  forward_flow = decode_flow(
      np.asarray(example["forward_flow"]),
      np.asarray(metadata["forward_flow_range"]),
  )
  backward_flow = decode_flow(
      np.asarray(example["backward_flow"]),
      np.asarray(metadata["backward_flow_range"]),
  )

  frame_count = min(args.frames, video.shape[0])
  frame_indices = np.linspace(0, video.shape[0] - 1, frame_count).round().astype(int)

  forward_scale = robust_flow_scale(forward_flow[frame_indices], args.flow_viz_percentile)
  backward_scale = robust_flow_scale(backward_flow[frame_indices], args.flow_viz_percentile)

  imageio.mimsave(
      output_dir / "video.gif",
      list(video),
      duration=args.gif_duration_ms / 1000.0,
  )
  imageio.imwrite(output_dir / "rgb_contact_sheet.png", make_contact_sheet(video[frame_indices]))
  imageio.imwrite(
      output_dir / "forward_flow_contact_sheet.png",
      make_contact_sheet([
          flow_to_rgb(forward_flow[idx], forward_scale, zero_color=args.flow_zero_color)
          for idx in frame_indices
      ]),
  )
  imageio.imwrite(
      output_dir / "backward_flow_contact_sheet.png",
      make_contact_sheet([
          flow_to_rgb(backward_flow[idx], backward_scale, zero_color=args.flow_zero_color)
          for idx in frame_indices
      ]),
  )
  imageio.imwrite(
      output_dir / "forward_flow_magnitude_contact_sheet.png",
      make_contact_sheet([
          flow_magnitude_to_rgb(forward_flow[idx], forward_scale)
          for idx in frame_indices
      ]),
  )

  summary = {
      "dataset": args.dataset,
      "split": args.split,
      "index": args.index,
      "video_name": bytes_to_text(metadata["video_name"]),
      "video_shape": list(video.shape),
      "frame_indices": frame_indices.tolist(),
      "forward_flow_range": np.asarray(metadata["forward_flow_range"]).tolist(),
      "backward_flow_range": np.asarray(metadata["backward_flow_range"]).tolist(),
      "forward_flow_viz_scale": forward_scale,
      "backward_flow_viz_scale": backward_scale,
  }
  if "background" in example:
    summary["background"] = bytes_to_text(example["background"])
  with (output_dir / "preview_summary.json").open("w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2, sort_keys=True)

  print(f"Wrote MOVi preview to {output_dir}")
  print(f"Video shape: {tuple(video.shape)}")
  print(f"Video name: {summary['video_name']}")
  if "background" in summary:
    print(f"Background: {summary['background']}")
  print(f"Forward flow viz scale: {forward_scale:.3f} px at p{args.flow_viz_percentile:g}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="movi_f/256x256")
  parser.add_argument("--data-dir", default="gs://kubric-public/tfds")
  parser.add_argument("--split", default="train")
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument("--output-dir", default="interface/output/movi_f_preview")
  parser.add_argument("--frames", type=int, default=6)
  parser.add_argument("--gif-duration-ms", type=int, default=80)
  parser.add_argument("--flow-viz-percentile", type=float, default=95.0)
  parser.add_argument("--flow-zero-color", choices=["black", "white"], default="black")
  args = parser.parse_args()

  example = to_numpy_example(
      dataset_name=args.dataset,
      data_dir=args.data_dir,
      split=args.split,
      index=args.index,
  )
  write_preview(example, Path(args.output_dir), args)


if __name__ == "__main__":
  main()
