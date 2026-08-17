"""
Sweeps IsolationForest's threshold_fpr to find a well-calibrated value, independent of LOF/DBSCAN's shared THRESHOLD_FPR.

Usage:
    python -m ml.tune_if --baseline-dir data/processed/clean
    python -m ml.tune_if --baseline-dir data/processed/clean --attack-dir data/processed/attack --ground-truth data/ground_truth/attack_windows.json
"""
from __future__ import annotations
import argparse
from pathlib import Path
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
from . import config
from .features import extract_features
from .train import BASELINE_DIR, HOLDOUT_FRACTION, _load_session_features, _split_sessions

CANDIDATE_THRESHOLD_FPR = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

def _load_malicious_pids(ground_truth_path: Path) -> set:
    if ground_truth_path is None or not Path(ground_truth_path).exists():
        return set()
    import json
    data = json.loads(Path(ground_truth_path).read_text())
    return set(data.get("malicious_pids", []))

def _score_attack_dir(attack_dir: Path, clf, threshold: float, malicious_pids: set):
    if attack_dir is None or not Path(attack_dir).exists():
        return None

    all_y_true, all_y_pred = [], []
    total_nodes = 0
    for gexf_path in sorted(Path(attack_dir).glob("*.gexf")):
        G = nx.read_gexf(gexf_path)
        X, node_list, _ = extract_features(G)
        if X.shape[0] == 0:
            continue
        scores = clf.decision_function(X)
        flagged = scores < threshold
        pids = []
        for n in node_list:
            try:
                pids.append(int(G.nodes[n].get("pid", -1)))
            except (TypeError, ValueError):
                pids.append(-1)
        y_true = [1 if p in malicious_pids else 0 for p in pids]
        all_y_true.extend(y_true)
        all_y_pred.extend(flagged.astype(int).tolist())
        total_nodes += X.shape[0]

    if not all_y_true or not malicious_pids:
        return None

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, int(y_pred.sum()), total_nodes

def main():
    parser = argparse.ArgumentParser(description="Sweep IsolationForest's threshold_fpr to find a well-calibrated value.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--attack-dir", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--candidates", type=float, nargs="+", default=CANDIDATE_THRESHOLD_FPR)
    args = parser.parse_args()

    gexf_paths = sorted(args.baseline_dir.glob("*.gexf"))
    if not gexf_paths:
        raise FileNotFoundError(f"No .gexf files found in {args.baseline_dir}")

    train_paths, holdout_paths = _split_sessions(gexf_paths, HOLDOUT_FRACTION, args.seed)
    print(f"Training sessions: {[p.name for p in train_paths]}")
    print(f"Held-out sessions: {[p.name for p in holdout_paths]}\n")

    X_train = np.vstack([x for x in (_load_session_features(p) for p in train_paths) if x.shape[0] > 0])
    if holdout_paths:
        X_holdout = np.vstack([x for x in (_load_session_features(p) for p in holdout_paths) if x.shape[0] > 0])
    else:
        X_holdout = X_train
        print("WARNING: no held-out sessions -- calibrating against training data itself (optimistic).\n")

    malicious_pids = _load_malicious_pids(args.ground_truth)

    # IF itself only needs to be fit ONCE (unlike LOF/DBSCAN, threshold_fpr
    # doesn't change how the model is fit, only where the threshold lands).
    clf = IsolationForest(contamination=config.CONTAMINATION, random_state=args.seed, n_estimators=200)
    clf.fit(X_train)
    holdout_scores = clf.decision_function(X_holdout)

    print(f"{'threshold_fpr':>13} | {'threshold':>10} | {'actual_FPR':>10} | {'attack_precision':>16} | {'attack_recall':>13} | {'attack_f1':>9} | {'n_flagged/n_nodes':>17}")
    print("-" * 105)

    results = []
    for target_fpr in args.candidates:
        threshold = float(np.percentile(holdout_scores, target_fpr * 100))
        actual_fpr = float(np.mean(holdout_scores < threshold))

        attack_result = _score_attack_dir(args.attack_dir, clf, threshold, malicious_pids)
        if attack_result:
            precision, recall, f1, n_flagged, n_nodes = attack_result
            attack_str = f"{precision:>16.3f} | {recall:>13.3f} | {f1:>9.3f} | {n_flagged}/{n_nodes:>13}"
        else:
            attack_str = f"{'n/a':>16} | {'n/a':>13} | {'n/a':>9} | {'n/a':>17}"

        print(f"{target_fpr:>13.3f} | {threshold:>10.4f} | {actual_fpr:>9.3%} | {attack_str}")
        results.append({
            "threshold_fpr": target_fpr, "threshold": threshold, "actual_fpr": actual_fpr,
            "attack_result": attack_result,
        })

    non_degenerate = [
        r for r in results
        if r["attack_result"] and r["attack_result"][3] < r["attack_result"][4]
    ]
    candidates_with_recall = [r for r in non_degenerate if r["attack_result"][1] >= 1.0]

    if candidates_with_recall:
        best = min(candidates_with_recall, key=lambda r: r["threshold_fpr"])
        selection_note = "strictest (lowest) threshold_fpr among non-degenerate candidates achieving recall=1.0"
    elif non_degenerate:
        best = max(non_degenerate, key=lambda r: r["attack_result"][1])  # highest recall achieved
        selection_note = "highest recall among non-degenerate candidates (none achieved recall=1.0 -- inspect the full table before trusting this)"
    elif any(r["attack_result"] for r in results):
        print(
            "\nERROR: every candidate threshold_fpr flagged EVERY node in the "
            "attack graph. There is no usable value here -- NOT writing "
            "if_best_params.json. Review the full table above."
        )
        return
    else:
        best = results[len(results) // 2]
        selection_note = "no attack ground truth available -- picked a middle-of-range default"

    print(f"\nSelection method: {selection_note}")
    print(f"Best threshold_fpr: {best['threshold_fpr']}")
    print(f"  -> actual FPR: {best['actual_fpr']:.3%}")
    if best["attack_result"]:
        p, r, f1, nf, nn = best["attack_result"]
        print(f"  -> attack precision/recall/f1: {p:.3f} / {r:.3f} / {f1:.3f}")

    import json
    from datetime import datetime, timezone
    best_params_path = Path(__file__).resolve().parent / "if_best_params.json"
    best_params_path.write_text(json.dumps({
        "threshold_fpr": best["threshold_fpr"],
        "actual_fpr": best["actual_fpr"],
        "tuned_at": datetime.now(timezone.utc).isoformat(),
        "candidates_tried": args.candidates,
    }, indent=2))
    print(f"\nSaved recommendation to {best_params_path}")
    print("ml/train.py will automatically pick this up on the next run.")

if __name__ == "__main__":
    main()