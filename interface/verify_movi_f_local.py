"""Verify a local MOVi-F TFDS mirror by writing one preview."""

from __future__ import annotations

import argparse
from pathlib import Path

from interface.preview_movi_tfds import to_numpy_example, write_preview


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="movi_f/512x512")
  parser.add_argument("--data-dir", default="data/kubric_tfds")
  parser.add_argument("--split", default="train")
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument("--output-dir", default="interface/output/movi_f_512_local_preview")
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

