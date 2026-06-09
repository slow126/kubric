"""Score a Kubric theta with the full-fit predictor (predicted transfer).

Renders the config (512²), extracts flow vectors, computes the motion (mean_nn)
features vs every eval benchmark, and runs the pickled predictor to report
predicted transfer (pred = L + g) per benchmark -- so we can see what the
predictor thinks a given Kubric set should do, e.g. on FlyingThings.

Reusable for the appearance ablations (pass a different theta json / overrides).
Run in the `cuda` conda env from the kubric repo root.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/spencer/Projects/kubric")
INTERV = Path("/home/spencer/Projects/interventional-study")
OSC = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")
VEC_DIR = Path("/mnt/nvme_1tb_b/coverage_vectors")
sys.path.insert(0, str(INTERV))
sys.path.insert(0, str(OSC / "scripts" / "transfer_analysis_v3"))
sys.path.insert(0, str(OSC))
from extract_candidate_features import load_candidate, compute_candidate_knnself  # noqa: E402
from compute_pairwise_self_distances import (  # noqa: E402
    compute_pair_metrics, load_flow_vectors, EVAL_DATASETS)
from transfer_predictor_prototype import SELFDIST_METRICS  # noqa: E402
from score_candidate import load_predictor, score  # noqa: E402
from generators.builders import build_dataset  # noqa: E402
from generators.extract import extract_flow_vectors_to_file  # noqa: E402

import importlib.util  # reuse render() helper from the fit driver
_spec = importlib.util.spec_from_file_location("ofb", REPO / "interface" / "oracle_fit_benchmark.py")
ofb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ofb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, help="fit report json; uses its best_theta")
    ap.add_argument("--theta-json", type=Path, help="explicit theta json (overrides --report)")
    ap.add_argument("--overrides", type=str, default="{}", help="json dict merged into theta (appearance ablations)")
    ap.add_argument("--predictor", type=Path,
                    default=INTERV / "full_fit/peak_pck/motion_mean_nn.pkl")
    ap.add_argument("--work", type=Path, default=Path("/mnt/nvme_1tb_a/kubric_score"))
    ap.add_argument("--n-pairs", type=int, default=80)
    ap.add_argument("--bench-cap", type=int, default=150_000)
    ap.add_argument("--label", type=str, default="config")
    args = ap.parse_args()

    if args.theta_json:
        theta = json.loads(args.theta_json.read_text())
    else:
        best = json.loads(args.report.read_text())["best_theta"]
        cam = json.loads(args.report.read_text()).get("camera_motion", "dolly")
        theta = {**ofb.BASE, "camera_motion": cam, **best}
    theta.update(json.loads(args.overrides))
    print(f"[score] {args.label} theta={ {k: theta[k] for k in ('camera_motion','end_distance','num_moving','motion_translation','motion_rotation','motion_vertical_frac') if k in theta} }", flush=True)

    # 1) render + extract candidate flow vectors (512²)
    rd = ofb.render(args.work, theta, f"{args.label}_src", args.n_pairs, seed=0)
    built = build_dataset("kubric", theta, split="train", kubric_intervention_datapath=rd)
    vp = extract_flow_vectors_to_file(
        built.dataset, args.work / f"{args.label}_vec", n_vectors=args.n_pairs * 2000,
        batch_size=1, num_workers=0, collate_fn=getattr(built, "collate_fn", None),
        seed=0, vectors_per_pair=2000, max_flow_magnitude=None)
    cand = load_candidate(vp, 60_000, seed=0)
    cknn = compute_candidate_knnself(cand, False)

    # 2) motion features vs every benchmark
    rows = []
    for name, split in EVAL_DATASETS:
        bv, _ = load_flow_vectors(name, split, VEC_DIR, args.bench_cap)
        if bv is None:
            continue
        bv = np.ascontiguousarray(bv, np.float32)
        m = compute_pair_metrics(cand, bv, use_gpu=False, knn_self_a=cknn,
                                 knn_self_b=compute_candidate_knnself(bv, False))
        row = {"benchmark": name, "split": split}
        for mm in SELFDIST_METRICS:
            row[f"se_flow_{mm}"] = float(m.get(mm, float("nan")))
        rows.append(row)
    features = pd.DataFrame(rows)

    # 3) predictor
    pred = load_predictor(args.predictor)
    res = score(pred, features)
    by_bench = res["per_benchmark_pred_cal"]
    by_bench_g = res["per_benchmark_g"]

    print("\n" + "=" * 60)
    print(f"PREDICTED TRANSFER  ({args.label})  target={res['target']} family={res['family']}")
    print(f"predictor={args.predictor.name}  fitness(mean g)={res['fitness']:.4f}")
    print("=" * 60)
    print(f"{'benchmark':<16}{'pred (L+g)':>12}{'within-ctx g':>14}")
    order = sorted(by_bench, key=lambda b: -by_bench[b])
    for b in order:
        star = "  <- FlyingThings" if b == "flyingthings" else ""
        print(f"{b:<16}{by_bench[b]:>12.3f}{by_bench_g.get(b, float('nan')):>14.4f}{star}")
    print("=" * 60)
    out = args.work / f"score_{args.label}.json"
    out.write_text(json.dumps({"theta": theta, "by_bench_pred": by_bench,
                               "by_bench_g": by_bench_g, "fitness": res["fitness"]}, indent=2))
    print(f"[score] wrote {out}")


if __name__ == "__main__":
    main()
