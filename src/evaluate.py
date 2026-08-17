"""
src/evaluate.py
Loads a trained K-Guard baseline model, scores an attack-window .gexf graph, and reports detection performance.

Usage:
    python -m src.evaluate --attack-dir data/processed/attack --model ml/models/baseline_model.joblib --ground-truth data/ground_truth/attack_windows.json --out-json results.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import networkx as nx
import numpy as np
from ml.features import extract_features
from .ground_truth import load_malicious_pids


def evaluate_graph(gexf_path: Path, model_path: Path, malicious_pids: set):
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    lof = bundle.get("lof_model")
    dbscan = bundle.get("dbscan_model")
    threshold = bundle["meta"]["threshold"]
    lof_threshold = bundle["meta"].get("lof_threshold")
    dbscan_threshold = bundle["meta"].get("dbscan_threshold")

    # Ensemble calibration stats (z-score normalization + combined threshold).
    # Only present in bundles trained after ensemble support was added.
    ensemble_threshold = bundle["meta"].get("ensemble_threshold")
    if_mean = bundle["meta"].get("ensemble_if_mean")
    if_std = bundle["meta"].get("ensemble_if_std")
    lof_mean = bundle["meta"].get("ensemble_lof_mean")
    lof_std = bundle["meta"].get("ensemble_lof_std")
    dbscan_mean = bundle["meta"].get("ensemble_dbscan_mean")
    dbscan_std = bundle["meta"].get("ensemble_dbscan_std")
    has_ensemble = ensemble_threshold is not None and None not in (if_mean, if_std, lof_mean, lof_std, dbscan_mean, dbscan_std)

    G = nx.read_gexf(gexf_path)
    X, node_list, feature_names = extract_features(G)

    if X.shape[0] == 0:
        print(f"No process nodes found in {gexf_path}, skipping.")
        return None

    scores = clf.decision_function(X)
    if_flagged = scores < threshold

    # LOF: second independent detector. 
    if lof is not None and lof_threshold is not None:
        lof_scores = lof.decision_function(X)
        lof_flagged = lof_scores < lof_threshold
    else:
        lof_scores = np.full(X.shape[0], np.nan)
        lof_flagged = np.zeros(X.shape[0], dtype=bool)

    # DBSCAN-style scorer: third independent detector. Same fallback pattern
    # for bundles saved before this model was added.
    if dbscan is not None and dbscan_threshold is not None:
        dbscan_scores = dbscan.decision_function(X)
        dbscan_flagged = dbscan_scores < dbscan_threshold
    else:
        dbscan_scores = np.full(X.shape[0], np.nan)
        dbscan_flagged = np.zeros(X.shape[0], dtype=bool)

    flagged = if_flagged | lof_flagged | dbscan_flagged

    # Ensemble score: combine all three raw scores via the same z-score normalization used at training time
    # can flag nodes that no single modelindividually caught
    if has_ensemble:
        z_if = (scores - if_mean) / if_std
        z_lof = (lof_scores - lof_mean) / lof_std
        z_dbscan = (dbscan_scores - dbscan_mean) / dbscan_std
        ensemble_scores = (z_if + z_lof + z_dbscan) / 3.0
        ensemble_flagged = ensemble_scores < ensemble_threshold
    else:
        ensemble_scores = np.full(X.shape[0], np.nan)
        ensemble_flagged = np.zeros(X.shape[0], dtype=bool)

    flagged = flagged | ensemble_flagged

    # Look up each scored node's real pid from the graph to compare against
    # ground truth. node_list entries are graph node IDs (hex UUIDs).
    node_pids = []
    for n in node_list:
        pid = G.nodes[n].get("pid", -1)
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = -1
        node_pids.append(pid)

    results = {
        "gexf_path": str(gexf_path),
        "n_nodes": int(X.shape[0]),
        "n_flagged": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "n_flagged_by_if": int(if_flagged.sum()),
        "n_flagged_by_lof": int(lof_flagged.sum()),
        "n_flagged_by_dbscan": int(dbscan_flagged.sum()),
        "n_flagged_by_ensemble_only": int((ensemble_flagged & ~(if_flagged | lof_flagged | dbscan_flagged)).sum()),
        "n_flagged_by_all_three": int((if_flagged & lof_flagged & dbscan_flagged).sum()),
    }

    if malicious_pids:
        y_true = np.array([1 if pid in malicious_pids else 0 for pid in node_pids])
        y_pred = flagged.astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results.update({
            "mode": "pid_level",
            "n_ground_truth_malicious_nodes_present": int(y_true.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        })
        if y_true.sum() == 0:
            results["note"] = (
                "None of the ground-truth malicious PIDs were found as process "
                "nodes in this graph -- likely wrong session/host, or the "
                "attacker process didn't trigger any OPENS/EXECUTES/CONNECTED_TO "
                "edges captured in this window. Precision/recall above are not "
                "meaningful in that case."
            )
    else:
        results["mode"] = "window_level_approx"
        results["note"] = (
            "No malicious PIDs provided -- flagged_fraction is a rough "
            "proxy only. Provide --ground-truth for real precision/recall."
        )

    # Top-N flagged anomalies, with pid + whether it matches ground truth.Ranked by IsolationForest score (kept as the primary ranking signal) with LOF and DBSCAN scores/flags reported alongside.
    order = np.argsort(scores)  # most anomalous first (lowest score)
    top_n = min(15, len(order))
    top_rows = []
    for idx in order[:top_n]:
        row = {
            "node": node_list[idx],
            "pid": node_pids[idx],
            "score": float(scores[idx]),
            "if_flagged": bool(if_flagged[idx]),
            "lof_score": float(lof_scores[idx]) if lof is not None else None,
            "lof_flagged": bool(lof_flagged[idx]),
            "dbscan_score": float(dbscan_scores[idx]) if dbscan is not None else None,
            "dbscan_flagged": bool(dbscan_flagged[idx]),
            "ensemble_score": float(ensemble_scores[idx]) if has_ensemble else None,
            "ensemble_flagged": bool(ensemble_flagged[idx]),
            "ground_truth_malicious": bool(node_pids[idx] in malicious_pids) if malicious_pids else "unknown",
        }
        row.update({fname: float(X[idx, i]) for i, fname in enumerate(feature_names)})
        top_rows.append(row)
    results["top_flagged"] = top_rows

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate K-Guard model against DARPA attack graph(s).")
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=None,
                         help="Path to data/ground_truth/attack_windows.json (provides malicious_pids)")
    parser.add_argument("--out-json", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    malicious_pids = load_malicious_pids(args.ground_truth) if args.ground_truth else set()
    if malicious_pids:
        print(f"Loaded {len(malicious_pids)} ground-truth malicious PID(s): {malicious_pids}")
    else:
        print("No ground-truth PIDs loaded -- running in window-level approximate mode.")

    if args.model.exists():
        bundle_check = joblib.load(args.model)
        active = ["IsolationForest"]
        if bundle_check.get("lof_model") is not None:
            active.append("LOF")
        if bundle_check.get("dbscan_model") is not None:
            active.append("DBSCAN")
        has_ensemble_check = bundle_check.get("meta", {}).get("ensemble_threshold") is not None
        suffix = " + ensemble (z-score combination)" if has_ensemble_check else ""
        print(f"Scoring with: {' + '.join(active)}{suffix} (flagged if any fires).")

    all_results = []
    for gexf_path in sorted(args.attack_dir.glob("*.gexf")):
        print(f"\nEvaluating {gexf_path}...")
        res = evaluate_graph(gexf_path, args.model, malicious_pids)
        if res:
            all_results.append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "top_flagged"}, indent=2))

    args.out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved full results (incl. top-flagged tables) to {args.out_json}")

if __name__ == "__main__":
    main()