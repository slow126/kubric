# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import difflib
import functools
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile

import numpy as np
import tensorflow as tf

from typing import Optional, Dict, Any, Type
import weakref

try:
  import fcntl  # POSIX-only; used for cross-process asset-cache locking.
except ImportError:  # pragma: no cover - non-POSIX fallback
  fcntl = None

from kubric import core
from kubric import file_io
from kubric.kubric_typing import PathLike


@contextlib.contextmanager
def _asset_lock(lock_path: pathlib.Path):
  """Exclusive cross-process lock so concurrent workers fetch an asset once.

  Coordinates parallel render workers (separate processes/containers sharing
  the asset cache over a bind mount) so they do not download/extract the same
  asset simultaneously. A no-op where fcntl is unavailable.
  """
  if fcntl is None:
    yield
    return
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  handle = open(lock_path, "w")
  try:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    yield
  finally:
    try:
      fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
      handle.close()


class ClosableResource:
  """TODO(klausg): documentation."""
  _set_of_open_resources = weakref.WeakSet()

  def __init__(self):
    super().__init__()
    self.is_closed = False
    self._set_of_open_resources.add(self)

  def close(self):
    try:
      self._set_of_open_resources.remove(self)
    except (ValueError, KeyError):
      pass  # not listed anymore. Ignore.

  @classmethod
  def close_all(cls):
    while True:
      try:
        r = cls._set_of_open_resources.pop()
      except KeyError:
        break
      r.close()


class AssetSource(ClosableResource):
  """TODO(klausg): documentation."""

  @classmethod
  def from_manifest(
      cls,
      manifest_path: PathLike,
      scratch_dir: Optional[PathLike] = None
  ) -> "AssetSource":
    if manifest_path == "gs://kubric-public/assets/ShapeNetCore.v2.json":
      raise ValueError(f"The path `{manifest_path}` is a placeholder for the real path. "
                       "Please visit https://shapenet.org, agree to terms and conditions."
                       "After logging in, you will find the manifest URL here:"
                       "https://shapenet.org/download/kubric")

    manifest_path = file_io.as_path(manifest_path)
    manifest = file_io.read_json(manifest_path)
    name = manifest.get("name", manifest_path.stem)  # default to filename
    data_dir = manifest.get("data_dir", manifest_path.parent)  # default to manifest dir
    assets = manifest["assets"]
    return cls(name=name, data_dir=data_dir, assets=assets, scratch_dir=scratch_dir)

  def __init__(
      self,
      name: str,
      data_dir: PathLike,
      assets: Dict[str, Any],
      scratch_dir: Optional[PathLike] = None
  ):
    super().__init__()
    self.name = name
    self.data_dir = file_io.as_path(data_dir)
    logging.info("Created AssetSource '%s' with '%d' assets at URI='%s'",
                 name, len(assets), self.data_dir)
    if scratch_dir is not None:
      # Stable, shared cache keyed by manifest name so every scene/worker/run
      # reuses one copy of each asset instead of re-downloading per scene.
      # Concurrency is handled per-asset in fetch() via _asset_lock.
      self.local_dir = pathlib.Path(scratch_dir) / name
      self.local_dir.mkdir(parents=True, exist_ok=True)
    else:
      self.local_dir = pathlib.Path(tempfile.mkdtemp(prefix=name, dir=scratch_dir))
    self._assets = assets

  def close(self):
    if self.is_closed:
      return
    try:
      shutil.rmtree(self.local_dir)
    finally:
      super().close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  @functools.cached_property
  def db(self):
    import pandas as pd
    db = pd.DataFrame([{"id": k} | v["kwargs"] | v["metadata"]
                       for k, v in self._assets.items()])

    def get_category_id(x):
      if x['category'] in self.categories:
        return self.categories.index(x['category'])
      else:
        return np.nan

    if "category_id" not in db:
      db["category_id"] = db.apply(get_category_id, axis=1)
    return db

  @functools.cached_property
  def categories(self):
    return sorted(filter(None, {v["metadata"].get("category", "")
                                for v in self._assets.values()}))

  @functools.cached_property
  def all_asset_ids(self):
    return sorted(self._assets.keys())

  @staticmethod
  def _resolve_asset_type(asset_type: str) -> Type:
    types = {
        "FileBasedObject": core.FileBasedObject,
        "Texture": core.Texture,
    }
    if asset_type not in types:
      raise KeyError(f"Unknown asset_type {asset_type!r}. "
                     f"Available types: {types!r}")
    return types[asset_type]

  def _resolve_asset_path(self, path: Optional[str], asset_id: str) -> Optional[PathLike]:
    if path is None:
      return None
    elif path == "":
      path = f"{asset_id}.tar.gz"

    return self.data_dir / path

  @staticmethod
  def _adjust_paths(asset_kwargs: Dict[str, Any], asset_dir: PathLike) -> Dict[str, Any]:
    """If present, replace '{asset_dir}' prefix with actual asset_dir in each kwarg value."""
    def _adjust_path(p):
      if isinstance(p, str) and p.startswith("{asset_dir}/"):
        return str(asset_dir / p[12:])
      elif isinstance(p, dict):
        return {key: _adjust_path(value) for key, value in p.items()}
      else:
        return p

    return {k: _adjust_path(v) for k, v in asset_kwargs.items()}

  def create(self, asset_id: str, add_metadata: bool = True, **kwargs) -> Type[core.Asset]:
    """
    Create an instance of an asset by a given id.

    Performs the following steps
    1. check if asset_id is found in manifest and retrieve entry
    2. determine Asset class and full path (can be remote or local cache or missing)
    3. if path is not none, then fetch and unpack the zipped asset to scratch_dir
    4. construct kwargs from asset_entry->kwargs, override with **kwargs and then
    adjust paths (ones that start with “{{asset_dir}}”
    5. create asset by calling constructor with kwargs
    6. set metadata (if add_metadata is True)
    7. return asset

    Args:
        asset_id (str): the id of the asset to be created
                        (corresponds to its key in the manifest file and
                        typically also to the filename)
        add_metadata (bool): whether to add the metadata from the asset to the instance
        **kwargs: additional kwargs to be passed to the asset constructor

    Returns:
      An instance of the specified asset (subtype of kubric.core.Asset)
    """
    # find corresponding asset entry
    asset_entry = self._assets.get(asset_id)
    if not asset_entry:
      close_matches = difflib.get_close_matches(asset_id, possibilities=self.all_asset_ids, n=1)
      if close_matches:
        raise KeyError(f"Unknown asset with id='{asset_id}'. Did you mean '{close_matches[0]}'?")

    # determine type and path
    asset_type = self._resolve_asset_type(asset_entry["asset_type"])
    asset_path = self._resolve_asset_path(asset_entry.get("path", ""), asset_id)

    # fetch and unpack tar.gz file if necessary
    asset_dir = None if asset_path is None else self.fetch(asset_path, asset_id)

    # construct kwargs
    asset_kwargs = asset_entry.get("kwargs", {})
    asset_kwargs.update(kwargs)
    asset_kwargs = self._adjust_paths(asset_kwargs, asset_dir)
    if asset_type == core.FileBasedObject:
      asset_kwargs["asset_id"] = asset_id
    # create the asset
    asset = asset_type(**asset_kwargs)
    # set the metadata
    if add_metadata:
      asset.metadata.update(asset_entry.get("metadata", {}))

    return asset

  def fetch(self, asset_path, asset_id):
    asset_dir = self.local_dir / asset_id
    done_marker = self.local_dir / (asset_id + ".done")
    # Fast path: a previous scene/worker already cached and extracted this asset.
    if done_marker.exists():
      return asset_dir

    self.local_dir.mkdir(parents=True, exist_ok=True)
    lock_path = self.local_dir / (asset_id + ".lock")
    with _asset_lock(lock_path):
      # Re-check under the lock: another worker may have just finished.
      if done_marker.exists():
        return asset_dir

      # Download to a unique temp file then atomically rename, so a partial
      # download can never be mistaken for a complete archive.
      local_path = self.local_dir / (asset_id + ".tar.gz")
      tmp_path = self.local_dir / f"{asset_id}.tar.gz.{os.getpid()}.tmp"
      logging.debug("Copying %s to %s", str(asset_path), str(local_path))
      tf.io.gfile.copy(asset_path, str(tmp_path), overwrite=True)
      os.replace(tmp_path, local_path)

      # Drop any partial extraction left by an earlier crash before extracting.
      if asset_dir.exists():
        shutil.rmtree(asset_dir, ignore_errors=True)
      with tarfile.open(local_path, "r:gz") as tar:
        # We support two kinds of archives:
        #  1. flat archives that do not contain any directories
        #  2. archives where the content is in a directory with the name of the asset
        list_of_files = tar.getnames()
        if asset_id in list_of_files and tar.getmember(asset_id).isdir():
          # tarfile contains directory with name object_id, so we can just extract
          assert f"{asset_id}/data.json" in list_of_files, list_of_files
          tar.extractall(self.local_dir)
        else:
          # tarfile contains files only, so extract into a new directory
          assert "data.json" in list_of_files, list_of_files
          tar.extractall(self.local_dir / asset_id)
        logging.debug("Extracted %s", repr([m.name for m in tar.getmembers()]))

      # Extraction succeeded: drop the archive (halves cache size) and mark done.
      local_path.unlink(missing_ok=True)
      done_marker.touch()

    return asset_dir

  def get_test_split(self, fraction=0.1):
    """
    Generates a train/test split for the asset source.

    Args:
      fraction: the fraction of the asset source to use for the held-out set.

    Returns:
      train_ids: list of asset ID strings
      test_ids: list of asset ID strings
    """
    rng = np.random.default_rng(42)
    test_size = int(round(len(self.all_asset_ids) * fraction))
    test_ids = rng.choice(self.all_asset_ids, size=test_size, replace=False)
    train_ids = [i for i in self.all_asset_ids if i not in test_ids]
    return train_ids, test_ids
