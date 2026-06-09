"""Self-recovery oracle for object-motion parameters (uses YOUR feature pipeline).

Question: given a target Kubric set generated with KNOWN object-motion params
theta*, can a cheap geometry-only render + TPE search recover theta* by matching
the target's flow distribution under your validated metric (mean_nn_sym)?

Each candidate (and the target) goes through the exact search path:
  render (geometry-only) -> extract_flow_vectors_to_file -> load_candidate
  (normalize_flow_vectors + to_joint_space) -> compute_pair_metrics(cand, target*)
Objective = mean_nn_sym(candidate, target*), minimized. Recovery is reported in
descriptor-space (distance vs the finite-sample noise floor) AND parameter-space
(theta_hat vs theta*). Win-or-learn: descriptor at floor but params off ==
mixture-collapse / non-identifiability.

Run in the `cuda` conda env from the kubric repo root.
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

# --- your feature pipeline (mirror search_loop.py's imports) ---
sys.path.insert(0, str(INTERV))
sys.path.insert(0, str(OSC / "scripts" / "transfer_analysis_v3"))
sys.path.insert(0, str(OSC))
from extract_candidate_features import load_candidate, compute_candidate_knnself  # noqa: E402
from compute_pairwise_self_distances import compute_pair_metrics  # noqa: E402
from generators.builders import build_dataset  # noqa: E402
from generators.extract import extract_flow_vectors_to_file  # noqa: E402

# Scene params shared by target and all candidates (everything except the motion
# axes we are recovering). Isolates the object-motion recovery question.
BASE = dict(
    asset_source="kubasic", num_assets=4, asset_scale=1.1,
    start_distance=9.0, temporal_gap=0.30,
    camera_azimuth=2.356, camera_elevation=0.35, target_z=0.8,
    focal_length=35.0, background_mode="matte", resolution=[256, 256],
)
# Confounded target: a specific JOINT of camera dolly (end_distance, smaller =>
# more dolly => more camera flow) and object motion. The search varies BOTH, so
# it can reach the target's flow via different camera/object splits. theta_hat
# matching the descriptor (floor) but NOT theta* => mixture-collapse: camera and
# object motion are non-identifiable from the aggregate flow distribution.
THETA_STAR = dict(end_distance=7.0, num_moving=2, motion_translation=0.5, motion_rotation=0.7)

OBJECTIVE_METRIC = "mean_nn_sym"
VECTORS_PER_PAIR = 2000
MAX_VECTORS = 60_000
USE_GPU = False  # faiss CPU; vector counts here are small


def render(work: Path, theta: dict, tag: str, n: int, seed: int) -> Path:
    """Render n geometry-only scenelets for theta into work/tag (on nvme)."""
    out = work / tag
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{tag}_theta.json").write_text(json.dumps(theta))
    cmd = [
        "docker", "run", "--rm", "-i", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{REPO}:/kubric", "-v", f"{work}:/work", "--env", "KUBRIC_USE_GPU=0",
        IMAGE, "/usr/bin/python3", "interface/render_intervention_scenelets.py",
        "--theta-json", f"/work/{tag}_theta.json", "--output-dir", f"/work/{tag}",
        "--scratch-dir", f"/work/scratch/{tag}", "--asset-scratch-dir", "/work/assets",
        "--n-pairs", str(n), "--seed", str(seed), "--geometry-only",
    ]
    subprocess.run(cmd, cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def vectors_for(work: Path, theta: dict, tag: str, n: int, seed: int) -> np.ndarray:
    """render -> extract_flow_vectors_to_file -> load_candidate (normalized joint)."""
    render_dir = render(work, theta, tag, n, seed)
    built = build_dataset("kubric", theta, split="train",
                          kubric_intervention_datapath=render_dir)
    vec_path = extract_flow_vectors_to_file(
        built.dataset, work / f"{tag}_vec",
        n_vectors=n * VECTORS_PER_PAIR, batch_size=1, num_workers=0,
        collate_fn=getattr(built, "collate_fn", None),
        seed=0, vectors_per_pair=VECTORS_PER_PAIR, max_flow_magnitude=None,
    )
    return load_candidate(vec_path, MAX_VECTORS, seed=0)


def match_distance(cand: np.ndarray, target: np.ndarray, target_knn) -> float:
    cand_knn = compute_candidate_knnself(cand, USE_GPU)
    m = compute_pair_metrics(cand, target, use_gpu=USE_GPU,
                             knn_self_a=cand_knn, knn_self_b=target_knn)
    return float(m[OBJECTIVE_METRIC])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/mnt/nvme_1tb_a/kubric_oracle"))
    ap.add_argument("--n-target", type=int, default=120)
    ap.add_argument("--n-cand", type=int, default=40)
    ap.add_argument("--n-trials", type=int, default=28)
    ap.add_argument("--floor-reps", type=int, default=3)
    args = ap.parse_args()
    work = args.work
    t0 = time.time()

    # 1) target descriptor (theta* at a held-out seed)
    target_theta = {**BASE, **THETA_STAR}
    print(f"[oracle] target: {args.n_target} pairs, theta*={THETA_STAR}  metric={OBJECTIVE_METRIC}", flush=True)
    target_vecs = vectors_for(work, target_theta, "target", args.n_target, seed=10_000)
    target_knn = compute_candidate_knnself(target_vecs, USE_GPU)

    # 2) noise floor: independent theta* re-draws at candidate sample size
    floor = []
    for k in range(args.floor_reps):
        v = vectors_for(work, target_theta, f"floor{k}", args.n_cand, seed=11_000 + 1000 * k)
        floor.append(match_distance(v, target_vecs, target_knn))
    floor_mean, floor_std = float(np.mean(floor)), float(np.std(floor))
    print(f"[oracle] noise floor (n={args.n_cand}): {floor_mean:.5f} +/- {floor_std:.5f}  "
          f"reps={[round(x,5) for x in floor]}", flush=True)

    # 3) TPE search over the motion axes
    def objective(trial: optuna.Trial) -> float:
        theta = {
            **BASE,
            "end_distance": trial.suggest_float("end_distance", 5.0, 9.0),
            "num_moving": trial.suggest_int("num_moving", 0, 4),
            "motion_translation": trial.suggest_float("motion_translation", 0.0, 1.0),
            "motion_rotation": trial.suggest_float("motion_rotation", 0.0, 1.5),
        }
        v = vectors_for(work, theta, "cand", args.n_cand, seed=0)
        d = match_distance(v, target_vecs, target_knn)
        trial.set_user_attr("dist", d)
        print(f"[oracle] trial {trial.number:02d} {OBJECTIVE_METRIC}={d:.5f}  "
              f"end={theta['end_distance']:.2f}(cam) nm={theta['num_moving']} "
              f"tr={theta['motion_translation']:.3f} rot={theta['motion_rotation']:.3f}", flush=True)
        return d

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # 4) report
    best = study.best_params
    reached = study.best_value <= floor_mean + 2 * floor_std
    print("\n" + "=" * 62)
    print(f"SELF-RECOVERY ORACLE REPORT  (objective = {OBJECTIVE_METRIC})")
    print("=" * 62)
    print(f"noise floor                : {floor_mean:.5f} +/- {floor_std:.5f}")
    print(f"best candidate distance    : {study.best_value:.5f}")
    print(f"reached floor (<=mean+2std)? {'YES' if reached else 'NO'}")
    print("\nparameter recovery (theta_hat vs theta*):")
    print(f"{'param':<20}{'theta*':>10}{'theta_hat':>12}{'abs_err':>10}")
    for kk in ("end_distance", "num_moving", "motion_translation", "motion_rotation"):
        gt, hat = THETA_STAR[kk], best[kk]
        tag = " <-camera" if kk == "end_distance" else ""
        print(f"{kk:<20}{gt:>10.3f}{hat:>12.3f}{abs(gt - hat):>10.3f}{tag}")
    print("=" * 62)

    report = {
        "objective_metric": OBJECTIVE_METRIC,
        "theta_star": THETA_STAR, "theta_hat": best,
        "best_distance": study.best_value,
        "noise_floor_mean": floor_mean, "noise_floor_std": floor_std,
        "reached_floor": bool(reached),
        "n_target": args.n_target, "n_cand": args.n_cand, "n_trials": args.n_trials,
        "trials": [{"params": t.params, "dist": t.value} for t in study.trials],
        "elapsed_sec": time.time() - t0,
    }
    (work / "oracle_report.json").write_text(json.dumps(report, indent=2))
    print(f"[oracle] wrote {work/'oracle_report.json'}  (elapsed {report['elapsed_sec']/60:.1f} min)")


if __name__ == "__main__":
    main()
