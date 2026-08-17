"""
inspect_score_gap.py

Diagnoses why IF/LOF are missing the attack node on the synthetic attack graph, by showing exactly where the malicious node ranks relative
to all other nodes for each of the three models plus the ensemble.

Usage (run from project root):
    python inspect_score_gap.py
"""
import joblib
import networkx as nx
import numpy as np
from pathlib import Path

from .features import extract_features

MODEL_PATH = Path("ml/models/baseline_model.joblib")
ATTACK_GEXF = Path("data/processed/attack/synthetic_theia_target1_attack.gexf")
MALICIOUS_PID = 534051

bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
lof = bundle.get("lof_model")
dbscan = bundle.get("dbscan_model")
meta = bundle["meta"]

threshold = meta["threshold"]
lof_threshold = meta.get("lof_threshold")
dbscan_threshold = meta.get("dbscan_threshold")

G = nx.read_gexf(ATTACK_GEXF)
X, node_list, feature_names = extract_features(G)

if_scores = clf.decision_function(X)
lof_scores = lof.decision_function(X) if lof is not None else None
dbscan_scores = dbscan.decision_function(X) if dbscan is not None else None

# Find the malicious node's index and rank (1 = most anomalous) for each model
mal_idx = None
for i, n in enumerate(node_list):
    pid = int(G.nodes[n].get("pid", -1)) if str(G.nodes[n].get("pid", "-1")).lstrip("-").isdigit() else -1
    if pid == MALICIOUS_PID:
        mal_idx = i
        break

if mal_idx is None:
    print(f"Malicious PID {MALICIOUS_PID} not found among {len(node_list)} scored nodes -- check the attack graph / PID.")
else:
    n_nodes = len(node_list)

    def rank_and_gap(scores, threshold, name):
        if scores is None:
            print(f"{name}: model not present in this bundle.")
            return
        # Lower score = more anomalous, matching your decision_function convention.
        order = np.argsort(scores)  # ascending: index 0 = most anomalous
        rank = int(np.where(order == mal_idx)[0][0]) + 1  # 1-based
        mal_score = scores[mal_idx]
        gap = mal_score - threshold if threshold is not None else float("nan")
        flagged = (mal_score < threshold) if threshold is not None else None
        print(
            f"{name}: score={mal_score:.4f}  rank={rank}/{n_nodes} (1=most anomalous)  "
            f"threshold={threshold if threshold is None else round(threshold, 4)}  "
            f"gap={gap:+.4f}  flagged={flagged}"
        )

    print(f"=== Malicious node (pid={MALICIOUS_PID}), index {mal_idx} of {n_nodes} ===\n")
    rank_and_gap(if_scores, threshold, "IsolationForest")
    rank_and_gap(lof_scores, lof_threshold, "LOF            ")
    rank_and_gap(dbscan_scores, dbscan_threshold, "DBSCAN         ")

    print("\nFeature values for the malicious node:")
    for name, val in zip(feature_names, X[mal_idx]):
        print(f"  {name}: {val}")

    print(f"\n=== Score distributions across all {n_nodes} nodes ===")
    for scores, name in [(if_scores, "IsolationForest"), (lof_scores, "LOF"), (dbscan_scores, "DBSCAN")]:
        if scores is not None:
            print(f"{name}: min={scores.min():.4f} p25={np.percentile(scores,25):.4f} "
                  f"median={np.median(scores):.4f} p75={np.percentile(scores,75):.4f} max={scores.max():.4f}")

    # Nearest clean neighbor comparison: which normal node looks MOST similar to the malicious one on raw features
    print(f"\n=== Nodes with the most similar raw feature vector to the malicious node ===")
    diffs = np.linalg.norm(X - X[mal_idx], axis=1)
    diffs[mal_idx] = np.inf
    closest = np.argsort(diffs)[:5]
    for idx in closest:
        pid = G.nodes[node_list[idx]].get("pid", "?")
        print(f"  pid={pid} euclidean_dist={diffs[idx]:.4f}")