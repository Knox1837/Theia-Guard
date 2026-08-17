"""
find_lof_outlier.py
Identifies which clean held-out node(s) have extreme LOF scores that are dragging the percentile-based LOF threshold out to an unreachable value.

Usage (run from project root):
    python find_lof_outlier.py
"""
import joblib
import networkx as nx
import numpy as np
from pathlib import Path

from .features import extract_features
from .train import _load_session_features

MODEL_PATH = Path("ml/models/baseline_model.joblib")
HOLDOUT_GEXF = Path("data/processed/clean/theia_official1_clean.gexf")
N_WORST = 10

bundle = joblib.load(MODEL_PATH)
lof = bundle.get("lof_model")
if lof is None:
    raise SystemExit("No lof_model in this bundle.")

G = nx.read_gexf(HOLDOUT_GEXF)
X, node_list, feature_names = extract_features(G)

scores = lof.decision_function(X)
order = np.argsort(scores)  # ascending: most extreme (most "anomalous") first

print(f"Held-out session: {HOLDOUT_GEXF.name}")
print(f"Total nodes scored: {len(scores)}")
print(f"Score stats: min={scores.min():.4f} p1={np.percentile(scores,1):.4f} "
      f"p5={np.percentile(scores,5):.4f} median={np.median(scores):.4f} max={scores.max():.4f}\n")

print(f"=== {N_WORST} most extreme (most 'anomalous') clean nodes by LOF score ===\n")
for rank, idx in enumerate(order[:N_WORST], start=1):
    n = node_list[idx]
    pid = G.nodes[n].get("pid", "?")
    node_type = G.nodes[n].get("type", G.nodes[n].get("label", "?"))
    print(f"#{rank}  score={scores[idx]:.4f}  pid={pid}  type={node_type}")
    for name, val in zip(feature_names, X[idx]):
        print(f"      {name}: {val}")
    print()

# How much is the single worst point actually distorting the threshold?
# Compare the threshold with vs without the single most extreme point.
target_fpr = bundle["meta"].get("threshold_fpr_target", 0.03)
threshold_with = float(np.percentile(scores, target_fpr * 100))
scores_without_worst = np.delete(scores, order[0])
threshold_without_worst = float(np.percentile(scores_without_worst, target_fpr * 100))

print("=== Threshold sensitivity to the single most extreme point ===")
print(f"Threshold WITH worst point:    {threshold_with:.4f}")
print(f"Threshold WITHOUT worst point: {threshold_without_worst:.4f}")
print(f"Difference: {threshold_without_worst - threshold_with:.4f}")
if abs(threshold_without_worst - threshold_with) > abs(threshold_with) * 0.5:
    print(
        "\n-> The threshold shifts by more than 50% when a single point is "
        "removed. This confirms one extreme outlier is dominating the "
        "threshold calibration- worth deciding whether that node is a "
        "legitimate rare-but-normal event (keep it, but consider a more "
        "robust threshold statistic) or a data/feature-extraction artifact "
        "(fix or exclude it)."
    )