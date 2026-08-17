import joblib
import networkx as nx
import numpy as np
from pathlib import Path

from .features import extract_features
from .train import _load_session_features

MODEL_PATH = Path("ml/models/baseline_model_20260815T103523Z.joblib")
HOLDOUT_GEXF = Path("data/processed/clean/theia_official1_clean.gexf")  # from the training run's printed "Held-out sessions"
MALICIOUS_SCORE = -0.2530  # from inspect_score_gap.py output

bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]

X_holdout = _load_session_features(HOLDOUT_GEXF)
holdout_scores = clf.decision_function(X_holdout)

# What fraction of held-out clean scores fall BELOW the malicious node's score
percentile_rank = float(np.mean(holdout_scores < MALICIOUS_SCORE))

print(f"Held-out clean score stats: min={holdout_scores.min():.4f} max={holdout_scores.max():.4f} mean={holdout_scores.mean():.4f}")
print(f"Malicious node's score: {MALICIOUS_SCORE}")
print(f"Fraction of held-out CLEAN scores below this value: {percentile_rank:.4%}")
print(f"\n=> A THRESHOLD_FPR of {percentile_rank:.4f} (or slightly above) would have flagged this node.")
print(f"   Current THRESHOLD_FPR = 0.01 (1%)")
print(f"   Needed  THRESHOLD_FPR ~ {percentile_rank:.4f} ({percentile_rank:.2%})")

# Sanity table across a few candidate target FPRs
print("\nWhat different target FPRs would produce as an actual threshold:")
for target_fpr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
    thresh = float(np.percentile(holdout_scores, target_fpr * 100))
    would_flag = MALICIOUS_SCORE < thresh
    print(f"  target_fpr={target_fpr:.2f} -> threshold={thresh:.4f} -> would flag malicious node: {would_flag}")