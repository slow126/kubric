"""Render a static-scene dolly-in smoke test with random Kubric assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import kubric as kb
from kubric.renderer.blender import Blender

from interface.asset_camera_motion_scene import AssetDollyInConfig, AssetDollyInScene
from interface.run_dolly_in_smoke import (
    flow_magnitude_to_rgb,
    flow_to_rgb,
    make_keypoint_flow_contact_sheet,
    make_contact_sheet,
    parse_resolution,
    robust_flow_scale,
    to_uint8_rgb,
    write_camera_metadata,
    write_flow_batch_safely,
)


ASSET_MANIFESTS = {
    "kubasic": "gs://kubric-public/assets/KuBasic/KuBasic.json",
    "gso": "gs://kubric-public/assets/GSO/GSO.json",
}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", default="interface/output/asset_dolly_in_smoke")
  parser.add_argument("--scratch-dir", default="interface/output/asset_dolly_in_smoke_scratch")
  parser.add_argument("--resolution", default="192x192")
  parser.add_argument("--frame-count", type=int, default=4)
  parser.add_argument("--start-distance", type=float, default=14.0)
  parser.add_argument("--end-distance", type=float, default=3.5)
  parser.add_argument("--samples-per-pixel", type=int, default=32)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--num-assets", type=int, default=4)
  parser.add_argument("--asset-scale", type=float, default=1.1)
  parser.add_argument("--asset-source", choices=sorted(ASSET_MANIFESTS), default="kubasic")
  parser.add_argument("--asset-manifest", default=None)
  parser.add_argument("--asset-ids", nargs="*", default=())
  parser.add_argument("--keep-asset-materials", action="store_true")
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

  manifest = args.asset_manifest or ASSET_MANIFESTS[args.asset_source]
  config = AssetDollyInConfig(
      resolution=parse_resolution(args.resolution),
      frame_end=args.frame_count,
      start_distance=args.start_distance,
      end_distance=args.end_distance,
      asset_manifest=manifest,
      seed=args.seed,
      num_assets=args.num_assets,
      asset_scale=args.asset_scale,
      asset_ids=tuple(args.asset_ids),
      color_assets=not args.keep_asset_materials,
  )
  builder = AssetDollyInScene(config, asset_scratch_dir=scratch_dir)
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
      flow_to_rgb(flow, max_magnitude=forward_flow_scale, zero_color=args.flow_zero_color)
      for flow in frames["forward_flow"]
  ])
  backward_flow_rgb = np.stack([
      flow_to_rgb(flow, max_magnitude=backward_flow_scale, zero_color=args.flow_zero_color)
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
  imageio.imwrite(output_dir / "forward_flow_contact_sheet.png", make_contact_sheet(forward_flow_rgb))
  imageio.imwrite(output_dir / "backward_flow_contact_sheet.png", make_contact_sheet(backward_flow_rgb))
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

  print(f"Wrote asset render outputs to {output_dir}")
  print(f"Asset IDs: {', '.join(scene.metadata['asset_ids'])}")
  print(f"Wrote RGB GIF to {output_dir / 'rgb.gif'}")
  print(f"Wrote RGB contact sheet to {output_dir / 'rgb_contact_sheet.png'}")
  print(f"Wrote flow contact sheet to {output_dir / 'forward_flow_contact_sheet.png'}")
  print(f"Forward flow viz scale: {forward_flow_scale:.3f} px at p{args.flow_viz_percentile:g}")
  print(f"Backward flow viz scale: {backward_flow_scale:.3f} px at p{args.flow_viz_percentile:g}")


if __name__ == "__main__":
  main()
