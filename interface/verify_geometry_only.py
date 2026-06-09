"""Phase 0 acceptance check: geometry-only vs full render.

Renders the same theta (N seeds) twice -- full RGB and --geometry-only -- via the
Docker wrapper, then compares the decoded forward/backward optical flow and the
wall-clock. The acceptance gate (kubric/plan.md A3/A4) is:

  * flow descriptor matches within noise  (low EPE, high per-channel correlation)
  * geometry-only is >=5x faster per scenelet

This only *reads* renders and times the two batches; it changes nothing in the
render path. Run it from the kubric repo root so the Docker wrapper mounts the
repo (and this work dir) at /kubric.

Example:
  KUBRIC_USE_GPU=1 KUBRIC_DOCKER_USE_GPUS_FLAG=1 KUBRIC_CUDA_DEVICE=1 \
    python3 interface/verify_geometry_only.py --theta-json theta.json --n 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "interface" / "run_intervention_scenelets_docker.sh"

# (layer name, frame index) pairs written by the driver.
FLOW_LAYERS = (("forward_flow", 0), ("backward_flow", 1))


def _container_path(host_path: Path) -> str:
  """Map a host path under the repo to its /kubric mount path."""
  rel = host_path.resolve().relative_to(REPO_ROOT)
  return f"/kubric/{rel.as_posix()}"


def _decode_flow(scene_dir: Path, name: str, frame: int) -> np.ndarray:
  """Invert _write_flow_frame: uint16 png + data_ranges.json -> float flow."""
  png = scene_dir / f"{name}_{frame:05d}.png"
  ranges = json.loads((scene_dir / "data_ranges.json").read_text())
  encoded = np.asarray(imageio.imread(png), dtype=np.float64)
  lo = float(ranges[name]["min"])
  hi = float(ranges[name]["max"])
  if hi == lo:
    return np.zeros_like(encoded)
  return encoded * (hi - lo) / 65535.0 + lo


def _run_batch(mode: str, theta_c: str, out_host: Path, scratch_host: Path,
               assets_host: Path, n: int, seed: int, full_spp: int) -> float:
  """Render a batch via the Docker wrapper; return wall-clock seconds."""
  out_host.mkdir(parents=True, exist_ok=True)
  scratch_host.mkdir(parents=True, exist_ok=True)
  cmd = [
      "bash", str(WRAPPER),
      "--theta-json", theta_c,
      "--output-dir", _container_path(out_host),
      "--scratch-dir", _container_path(scratch_host),
      "--asset-scratch-dir", _container_path(assets_host),
      "--n-pairs", str(n),
      "--seed", str(seed),
  ]
  if mode == "geometry":
    cmd.append("--geometry-only")
  else:
    cmd += ["--samples-per-pixel", str(full_spp)]
  print(f"\n[verify] === {mode} batch (n={n}) ===\n  {' '.join(cmd)}", flush=True)
  start = time.perf_counter()
  subprocess.run(cmd, cwd=REPO_ROOT, check=True)
  elapsed = time.perf_counter() - start
  print(f"[verify] {mode} batch wall-clock: {elapsed:.1f}s ({elapsed / n:.1f}s/scene)")
  return elapsed


def _compare_scene(full_dir: Path, geom_dir: Path) -> dict:
  per_layer = {}
  for name, frame in FLOW_LAYERS:
    a = _decode_flow(full_dir, name, frame)
    b = _decode_flow(geom_dir, name, frame)
    if a.shape != b.shape:
      raise ValueError(f"shape mismatch {name}: {a.shape} vs {b.shape}")
    epe = float(np.mean(np.sqrt(np.sum((a - b) ** 2, axis=-1))))
    mag = float(np.mean(np.sqrt(np.sum(a ** 2, axis=-1))))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    per_layer[name] = {
        "epe_px": epe,
        "mean_flow_mag_px": mag,
        "rel_epe": epe / (mag + 1e-6),
        "pearson_r": corr,
    }
  return per_layer


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--theta-json", type=Path, required=True,
                      help="theta to render both ways (host path).")
  parser.add_argument("--n", type=int, default=5, help="number of seeds/scenes.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--full-spp", type=int, default=32,
                      help="samples/pixel for the full reference render.")
  parser.add_argument("--workdir", type=Path,
                      default=REPO_ROOT / "_geom_verify",
                      help="scratch+output root (must be under the repo).")
  parser.add_argument("--report-json", type=Path, default=None)
  args = parser.parse_args()

  workdir = args.workdir.resolve()
  try:
    workdir.relative_to(REPO_ROOT)
  except ValueError:
    sys.exit(f"--workdir must be under the repo root {REPO_ROOT} (it is mounted at /kubric)")

  # The theta also has to be visible inside the container.
  theta_host = args.theta_json.resolve()
  try:
    theta_c = _container_path(theta_host)
  except ValueError:
    theta_c = "/kubric/_geom_verify/theta.json"
    dst = workdir / "theta.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(theta_host.read_text())
    print(f"[verify] copied theta into workdir -> {dst}")

  assets_host = workdir / "assets"
  full_out = workdir / "full"
  geom_out = workdir / "geom"

  t_full = _run_batch("full", theta_c, full_out, workdir / "scratch_full",
                      assets_host, args.n, args.seed, args.full_spp)
  t_geom = _run_batch("geometry", theta_c, geom_out, workdir / "scratch_geom",
                      assets_host, args.n, args.seed, args.full_spp)

  scenes = []
  for i in range(args.n):
    # Both batches use start_index=0 and the same --seed, so scene_{i} shares a
    # scene_seed (seed+i) across modes -> identical geometry, comparable flow.
    name = f"scene_{i:06d}"
    full_dir = full_out / name
    geom_dir = geom_out / name
    if not full_dir.exists() or not geom_dir.exists():
      print(f"[verify] WARNING: missing {name} in one mode; skipping")
      continue
    scenes.append({"scene": name, "layers": _compare_scene(full_dir, geom_dir)})

  if not scenes:
    sys.exit("[verify] no comparable scenes were produced")

  # Aggregate.
  agg = {}
  for name, _ in FLOW_LAYERS:
    epes = [s["layers"][name]["epe_px"] for s in scenes]
    rels = [s["layers"][name]["rel_epe"] for s in scenes]
    corrs = [s["layers"][name]["pearson_r"] for s in scenes]
    agg[name] = {
        "mean_epe_px": float(np.mean(epes)),
        "max_epe_px": float(np.max(epes)),
        "mean_rel_epe": float(np.mean(rels)),
        "min_pearson_r": float(np.min(corrs)),
    }

  speedup = t_full / t_geom if t_geom > 0 else float("inf")
  report = {
      "n_scenes": len(scenes),
      "full_seconds": t_full,
      "geometry_seconds": t_geom,
      "speedup": speedup,
      "flow_agg": agg,
      "per_scene": scenes,
  }

  print("\n" + "=" * 64)
  print("PHASE 0 EQUIVALENCE REPORT")
  print("=" * 64)
  print(f"scenes compared : {len(scenes)}")
  print(f"full   wall-clock: {t_full:7.1f}s ({t_full / args.n:.1f}s/scene)")
  print(f"geom   wall-clock: {t_geom:7.1f}s ({t_geom / args.n:.1f}s/scene)")
  print(f"SPEEDUP         : {speedup:5.2f}x   (gate: >= 5x)")
  for name, _ in FLOW_LAYERS:
    a = agg[name]
    print(f"{name:14s}: EPE mean={a['mean_epe_px']:.4f}px max={a['max_epe_px']:.4f}px "
          f"rel={a['mean_rel_epe']:.4f} minR={a['min_pearson_r']:.5f}")
  gate_speed = speedup >= 5.0
  gate_flow = all(agg[n]["min_pearson_r"] >= 0.999 and agg[n]["mean_rel_epe"] <= 0.02
                  for n, _ in FLOW_LAYERS)
  print(f"\nGATE speedup>=5x : {'PASS' if gate_speed else 'FAIL'}")
  print(f"GATE flow~equal  : {'PASS' if gate_flow else 'REVIEW'} "
        f"(rel_epe<=0.02 & r>=0.999)")
  print("=" * 64)

  report_path = args.report_json or (workdir / "phase0_report.json")
  report_path.write_text(json.dumps(report, indent=2))
  print(f"[verify] wrote {report_path}")


if __name__ == "__main__":
  main()
