"""Fit Kubric (object + camera motion) to a REAL benchmark flow distribution.

Searches Kubric theta to minimize mean_nn_sym(candidate, real_benchmark) in your
normalized joint space, using the same pipeline as search_loop:
  render (geometry-only, 512²) -> extract_flow_vectors_to_file -> load_candidate
  -> compute_pair_metrics  vs  load_flow_vectors(benchmark).

Two regimes:
  flyingthings (test): stationary camera + dense object motion -> camera_motion
    "dolly" (search end_distance; expect ~static) + object motion.
  middlebury  (val) : stereo rigid scene -> camera_motion "lateral" (search
    camera_baseline) + object motion (expect num_moving -> 0).

No theta* (real target). Floor = the target's own bootstrap self-distance at the
candidate sample size (lower bound; Kubric likely can't reach it -> the gap is the
generator-expressivity cost). Reports best theta_hat + near-best cluster spread
(which axes the descriptor pins down). 512² is REQUIRED: benchmark vectors are
normalized at IMG_W=512. Run in the `cuda` env from the kubric repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna

REPO = Path("/home/spencer/Projects/kubric")
INTERV = Path("/home/spencer/Projects/interventional-study")
OSC = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")
IMAGE = "kubricdockerhub/kubruntu"
VEC_DIR = Path("/mnt/nvme_1tb_b/coverage_vectors")

sys.path.insert(0, str(INTERV))
sys.path.insert(0, str(OSC / "scripts" / "transfer_analysis_v3"))
sys.path.insert(0, str(OSC))
from extract_candidate_features import load_candidate, compute_candidate_knnself  # noqa: E402
from compute_pairwise_self_distances import compute_pair_metrics, load_flow_vectors  # noqa: E402
from generators.builders import build_dataset  # noqa: E402
from generators.extract import extract_flow_vectors_to_file  # noqa: E402

BASE = dict(
    asset_source="kubasic", num_assets=4, asset_scale=1.1,
    start_distance=9.0, temporal_gap=0.30,
    camera_azimuth=2.356, camera_elevation=0.35, target_z=0.8,
    focal_length=35.0, background_mode="matte", resolution=[512, 512],
)
BENCH = {
    "flyingthings": dict(split="test", camera_motion="dolly"),
    "middlebury":   dict(split="val",  camera_motion="lateral"),
    "kitti2015":    dict(split="val",  camera_motion="dolly"),  # forward ego-motion
}
OBJECTIVE_METRIC = "mean_nn_sym"
VECTORS_PER_PAIR = 2000
MAX_VECTORS = 60_000
USE_GPU = False


def render(work, theta, tag, n, seed):
    out = work / tag
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{tag}_theta.json").write_text(json.dumps(theta))
    cmd = ["docker", "run", "--rm", "-i", "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{REPO}:/kubric", "-v", f"{work}:/work", "--env", "KUBRIC_USE_GPU=0",
           IMAGE, "/usr/bin/python3", "interface/render_intervention_scenelets.py",
           "--theta-json", f"/work/{tag}_theta.json", "--output-dir", f"/work/{tag}",
           "--scratch-dir", f"/work/scratch/{tag}", "--asset-scratch-dir", "/work/assets",
           "--n-pairs", str(n), "--seed", str(seed), "--geometry-only"]
    subprocess.run(cmd, cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def candidate_vectors(work, theta, n, seed):
    rd = render(work, theta, "cand", n, seed)
    built = build_dataset("kubric", theta, split="train", kubric_intervention_datapath=rd)
    vp = extract_flow_vectors_to_file(
        built.dataset, work / "cand_vec", n_vectors=n * VECTORS_PER_PAIR,
        batch_size=1, num_workers=0, collate_fn=getattr(built, "collate_fn", None),
        seed=0, vectors_per_pair=VECTORS_PER_PAIR, max_flow_magnitude=None)
    return load_candidate(vp, MAX_VECTORS, seed=0)


def dist_to_target(cand, target, target_knn):
    cknn = compute_candidate_knnself(cand, USE_GPU)
    return float(compute_pair_metrics(cand, target, use_gpu=USE_GPU,
                                      knn_self_a=cknn, knn_self_b=target_knn)[OBJECTIVE_METRIC])


def bootstrap_floor(target, n_vec, reps=3):
    rng = np.random.default_rng(0)
    n = len(target); m = min(n_vec, n // 2)
    out = []
    for _ in range(reps):
        idx = rng.permutation(n)
        a = np.ascontiguousarray(target[idx[:m]]); b = np.ascontiguousarray(target[idx[m:2 * m]])
        out.append(float(compute_pair_metrics(
            a, b, use_gpu=USE_GPU,
            knn_self_a=compute_candidate_knnself(a, USE_GPU),
            knn_self_b=compute_candidate_knnself(b, USE_GPU))[OBJECTIVE_METRIC]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCH), required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--n-cand", type=int, default=24)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--bench-cap", type=int, default=2_000_000)
    args = ap.parse_args()
    bcfg = BENCH[args.benchmark]
    cam = bcfg["camera_motion"]
    work = args.work
    t0 = time.time()

    # 1) real target
    target, _ = load_flow_vectors(args.benchmark, bcfg["split"], VEC_DIR, args.bench_cap)
    if target is None:
        sys.exit(f"no flow vectors for {args.benchmark}/{bcfg['split']} in {VEC_DIR}")
    target = np.ascontiguousarray(target, dtype=np.float32)
    target_knn = compute_candidate_knnself(target, USE_GPU)
    print(f"[fit] target={args.benchmark}/{bcfg['split']} vecs={len(target):,} camera_motion={cam}", flush=True)

    floor = bootstrap_floor(target, args.n_cand * VECTORS_PER_PAIR)
    floor_mean, floor_std = float(np.mean(floor)), float(np.std(floor))
    print(f"[fit] target self-floor (sampling noise): {floor_mean:.5f} +/- {floor_std:.5f}", flush=True)

    base = {**BASE, "camera_motion": cam}

    def objective(trial):
        theta = dict(base)
        if cam == "dolly":
            theta["end_distance"] = trial.suggest_float("end_distance", 5.0, 9.0)
        else:
            theta["camera_baseline"] = trial.suggest_float("camera_baseline", 0.0, 3.0)
        theta["num_moving"] = trial.suggest_int("num_moving", 0, 4)
        theta["motion_translation"] = trial.suggest_float("motion_translation", 0.0, 1.5)
        theta["motion_translation_std"] = trial.suggest_float("motion_translation_std", 0.0, 1.0)
        theta["motion_rotation"] = trial.suggest_float("motion_rotation", 0.0, 1.5)
        theta["motion_rotation_std"] = trial.suggest_float("motion_rotation_std", 0.0, 1.0)
        theta["motion_vertical_frac"] = trial.suggest_float("motion_vertical_frac", 0.0, 1.0)
        cand = candidate_vectors(work, theta, args.n_cand, seed=0)
        d = dist_to_target(cand, target, target_knn)
        camv = theta.get("end_distance", theta.get("camera_baseline"))
        print(f"[fit] trial {trial.number:02d} {OBJECTIVE_METRIC}={d:.5f}  cam={camv:.2f} "
              f"nm={theta['num_moving']} tr={theta['motion_translation']:.2f}±{theta['motion_translation_std']:.2f} "
              f"rot={theta['motion_rotation']:.2f}±{theta['motion_rotation_std']:.2f} "
              f"vfrac={theta['motion_vertical_frac']:.2f}", flush=True)
        return d

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # cluster analysis: spread of params among near-best trials
    tr = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)
    thr = study.best_value * 1.5
    near = [t.params for t in tr if t.value <= thr]
    cam_key = "end_distance" if cam == "dolly" else "camera_baseline"
    axes = [cam_key, "num_moving", "motion_translation", "motion_translation_std",
            "motion_rotation", "motion_rotation_std", "motion_vertical_frac"]
    spread = {k: [float(min(p[k] for p in near)), float(max(p[k] for p in near))] for k in axes} if len(near) >= 2 else {}

    print("\n" + "=" * 60)
    print(f"BENCHMARK FIT: {args.benchmark}  (camera_motion={cam}, objective={OBJECTIVE_METRIC})")
    print("=" * 60)
    print(f"target self-floor : {floor_mean:.5f} +/- {floor_std:.5f}  (sampling noise lower bound)")
    print(f"best distance     : {study.best_value:.5f}  ({study.best_value/floor_mean:.1f}x floor)")
    print(f"best theta_hat    : {study.best_params}")
    print(f"\nnear-best cluster (<=1.5x best, n={len(near)}) -- which axes the descriptor pins:")
    for k in axes:
        if k in spread:
            lo, hi = spread[k]
            print(f"  {k:<20} [{lo:.3f}, {hi:.3f}]  spread={hi-lo:.3f}")
    print("=" * 60)

    report = {
        "benchmark": args.benchmark, "camera_motion": cam,
        "objective_metric": OBJECTIVE_METRIC,
        "best_distance": study.best_value, "best_theta": study.best_params,
        "target_self_floor_mean": floor_mean, "target_self_floor_std": floor_std,
        "near_best_spread": spread, "n_near": len(near),
        "n_cand": args.n_cand, "n_trials": args.n_trials,
        "trials": [{"params": t.params, "dist": t.value} for t in study.trials],
        "elapsed_sec": time.time() - t0,
    }
    (work / f"fit_{args.benchmark}_report.json").write_text(json.dumps(report, indent=2))
    print(f"[fit] wrote {work}/fit_{args.benchmark}_report.json  (elapsed {report['elapsed_sec']/60:.1f} min)")


if __name__ == "__main__":
    main()
