"""PyTorch/Lightning dataset wrapper for pre-rendered MOVi-F TFDS.

This is intended as a bridge/reference implementation for correspondence
training code. It does not render Kubric scenes. It reads the public TFDS MOVi-F
records and returns adjacent frame pairs with either dense optical flow,
sampled keypoint correspondences, or both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader, IterableDataset

try:
  import lightning.pytorch as pl
except ImportError:  # pragma: no cover - depends on the consuming project
  try:
    import pytorch_lightning as pl
  except ImportError:  # pragma: no cover
    pl = None


FlowConvention = Literal["target_to_source", "source_to_target"]
Representation = Literal["flow", "keypoints", "both"]


@dataclass(frozen=True)
class MoviFPairConfig:
  """Configuration for MOVi-F adjacent-frame pair sampling."""

  dataset: str = "movi_f/256x256"
  data_dir: str = "gs://kubric-public/tfds"
  split: str = "train"
  flow_convention: FlowConvention = "target_to_source"
  representation: Representation = "both"
  image_range: Literal["0_1", "minus1_1"] = "0_1"
  shuffle_files: bool = True
  shuffle_buffer: int = 128
  repeat: bool = True
  pairs_per_video: int = 1
  random_pair: bool = True
  keypoints_per_pair: int = 512
  keypoint_seed: int = 0
  min_flow_magnitude: float = 0.0


def _decode_flow(flow_uint16: np.ndarray, flow_range: np.ndarray) -> np.ndarray:
  flow = flow_uint16.astype(np.float32)
  min_value, max_value = [float(value) for value in flow_range]
  return flow / 65535.0 * (max_value - min_value) + min_value


def _image_to_tensor(image: np.ndarray, image_range: str) -> torch.Tensor:
  tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
  if image_range == "minus1_1":
    tensor = tensor * 2.0 - 1.0
  return tensor


def _flow_to_tensor(flow: np.ndarray) -> torch.Tensor:
  return torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1)


def _valid_mask_from_flow(flow: np.ndarray) -> np.ndarray:
  height, width = flow.shape[:2]
  rows, cols = np.meshgrid(
      np.arange(height, dtype=np.float32),
      np.arange(width, dtype=np.float32),
      indexing="ij",
  )
  target_rows = rows + flow[..., 0]
  target_cols = cols + flow[..., 1]
  return (
      np.isfinite(flow).all(axis=-1)
      & (target_rows >= 0)
      & (target_rows <= height - 1)
      & (target_cols >= 0)
      & (target_cols <= width - 1)
  )


def _sample_keypoints(
    flow: np.ndarray,
    valid_mask: np.ndarray,
    count: int,
    rng: np.random.RandomState,
    min_flow_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  height, width = flow.shape[:2]
  magnitude = np.linalg.norm(flow, axis=-1)
  valid = valid_mask & (magnitude >= min_flow_magnitude)
  rows, cols = np.nonzero(valid)

  points0 = np.zeros((count, 2), dtype=np.float32)
  points1 = np.zeros((count, 2), dtype=np.float32)
  keypoint_valid = np.zeros((count,), dtype=np.bool_)

  if len(rows) == 0:
    return points0, points1, keypoint_valid

  replace = len(rows) < count
  selected = rng.choice(len(rows), size=count, replace=replace)
  sampled_rows = rows[selected].astype(np.float32)
  sampled_cols = cols[selected].astype(np.float32)
  sampled_flow = flow[rows[selected], cols[selected]]

  points0[:, 0] = sampled_cols
  points0[:, 1] = sampled_rows
  points1[:, 0] = sampled_cols + sampled_flow[:, 1]
  points1[:, 1] = sampled_rows + sampled_flow[:, 0]
  keypoint_valid[:] = True

  points1[:, 0] = np.clip(points1[:, 0], 0, width - 1)
  points1[:, 1] = np.clip(points1[:, 1], 0, height - 1)
  return points0, points1, keypoint_valid


class MoviFPairIterableDataset(IterableDataset):
  """Iterable PyTorch dataset for adjacent MOVi-F correspondence pairs.

  Returned dense flow convention:
    - source image = frame t
    - target image = frame t + 1
    - default `flow` is target -> source, using Kubric `backward_flow[t + 1]`

  For keypoints with the default convention:
    - `points_target` are pixel coordinates in the target frame, shape [N, 2]
    - `points_source` are corresponding pixel coordinates in the source frame
  Coordinates are `(x, y)` in pixel units.
  """

  def __init__(self, config: MoviFPairConfig | None = None):
    super().__init__()
    self.config = config or MoviFPairConfig()

  def _tf_dataset(self, worker_info) -> tf.data.Dataset:
    ds = tfds.load(
        self.config.dataset,
        data_dir=self.config.data_dir,
        split=self.config.split,
        shuffle_files=self.config.shuffle_files,
    )
    if worker_info is not None:
      ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
    if self.config.shuffle_buffer > 0:
      ds = ds.shuffle(self.config.shuffle_buffer)
    if self.config.repeat:
      ds = ds.repeat()
    return ds

  def __iter__(self):
    worker = torch.utils.data.get_worker_info()
    worker_id = 0 if worker is None else worker.id
    rng = np.random.RandomState(self.config.keypoint_seed + worker_id)

    for example in tfds.as_numpy(self._tf_dataset(worker)):
      video = np.asarray(example["video"], dtype=np.uint8)
      num_frames = video.shape[0]
      if num_frames < 2:
        continue

      pair_indices = self._pair_indices(num_frames, rng)
      for frame_idx in pair_indices:
        yield self._format_pair(example, video, int(frame_idx), rng)

  def _pair_indices(self, num_frames: int, rng: np.random.RandomState) -> np.ndarray:
    max_start = num_frames - 2
    count = max(1, self.config.pairs_per_video)
    if self.config.random_pair:
      return rng.randint(0, max_start + 1, size=count)
    return np.arange(min(count, max_start + 1))

  def _format_pair(
      self,
      example: dict,
      video: np.ndarray,
      frame_idx: int,
      rng: np.random.RandomState,
  ) -> dict:
    source = video[frame_idx]
    target = video[frame_idx + 1]
    metadata = example["metadata"]

    forward_flow = _decode_flow(
        np.asarray(example["forward_flow"][frame_idx]),
        np.asarray(metadata["forward_flow_range"]),
    )
    backward_flow = _decode_flow(
        np.asarray(example["backward_flow"][frame_idx + 1]),
        np.asarray(metadata["backward_flow_range"]),
    )

    if self.config.flow_convention == "source_to_target":
      flow = forward_flow
      valid_mask = _valid_mask_from_flow(flow)
    elif self.config.flow_convention == "target_to_source":
      flow = backward_flow
      valid_mask = _valid_mask_from_flow(flow)
    else:
      raise ValueError(f"Unknown flow convention: {self.config.flow_convention}")

    sample = {
        "image_source": _image_to_tensor(source, self.config.image_range),
        "image_target": _image_to_tensor(target, self.config.image_range),
        "frame_source": torch.tensor(frame_idx, dtype=torch.long),
        "frame_target": torch.tensor(frame_idx + 1, dtype=torch.long),
        "flow_convention": self.config.flow_convention,
        "video_name": _bytes_to_text(metadata["video_name"]),
    }

    if self.config.representation in {"flow", "both"}:
      sample["flow"] = _flow_to_tensor(flow)
      sample["valid_mask"] = torch.from_numpy(valid_mask.astype(np.bool_)).unsqueeze(0)

    if self.config.representation in {"keypoints", "both"}:
      points0, points1, keypoint_valid = _sample_keypoints(
          flow=flow,
          valid_mask=valid_mask,
          count=self.config.keypoints_per_pair,
          rng=rng,
          min_flow_magnitude=self.config.min_flow_magnitude,
      )
      if self.config.flow_convention == "target_to_source":
        sample["points_target"] = torch.from_numpy(points0)
        sample["points_source"] = torch.from_numpy(points1)
      else:
        sample["points_source"] = torch.from_numpy(points0)
        sample["points_target"] = torch.from_numpy(points1)
      sample["keypoint_valid"] = torch.from_numpy(keypoint_valid)

    return sample


def _bytes_to_text(value) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, np.ndarray) and value.shape == ():
    return _bytes_to_text(value.item())
  return str(value)


class MoviFDataModule(pl.LightningDataModule if pl is not None else object):
  """LightningDataModule for MOVi-F adjacent correspondence pairs."""

  def __init__(
      self,
      train_config: MoviFPairConfig | None = None,
      val_config: MoviFPairConfig | None = None,
      batch_size: int = 8,
      num_workers: int = 4,
      pin_memory: bool = True,
  ):
    if pl is not None:
      super().__init__()
    self.train_config = train_config or MoviFPairConfig(split="train")
    self.val_config = val_config or MoviFPairConfig(
        split="validation",
        shuffle_files=False,
        shuffle_buffer=0,
        repeat=False,
        random_pair=False,
        pairs_per_video=4,
    )
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pin_memory = pin_memory

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
        MoviFPairIterableDataset(self.train_config),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
    )

  def val_dataloader(self) -> DataLoader:
    return DataLoader(
        MoviFPairIterableDataset(self.val_config),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
    )
