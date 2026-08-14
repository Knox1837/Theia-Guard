# K-Guard DARPA Trainer

Converts DARPA Transparent Computing (TC) Engagement 5 Theia data into
K-Guard-schema provenance graphs, trains K-Guard's anomaly detector
(IsolationForest + LOF) on them, and evaluates detection performance
against DARPA's labeled ground truth.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 1. Get the data

Place raw DARPA `.bin` (Avro) files in `data/raw/`, and ground-truth
files in `data/ground_truth/`.

## 2. Convert to K-Guard graphs

```bash
python -m src.darpa_to_gexf --input data/raw/<file>.bin --schema data/schema/TCCDMDatum.avsc --ground-truth data/ground_truth/operational_event_log.md --out-clean data/processed/clean --out-attack data/processed/attack
```

Splits the capture into benign ("clean") and attack-window `.gexf` graphs.

## 3. Train the baseline models

```bash
python -m ml.train --baseline-dir data/processed/clean
```

Fits both an IsolationForest and a LocalOutlierFactor on the clean
sessions. Each is promoted into `ml/models/baseline_model.joblib`
independently -- only if it's not worse than the currently active
version of that same model. Use `--force` to promote both regardless.

## 4. Evaluate against labeled attack data

```bash
python -m src.evaluate --attack-dir data/processed/attack --model ml/models/baseline_model.joblib --ground-truth data/ground_truth/attack_windows.json --out-json results.json
```

Scores with both models (a node is flagged if either fires) and reports
precision/recall/F1, plus a per-model flag breakdown.

## 5. Notebooks

Run in order under `notebooks/`:
1. `01_data_exploration.ipynb`
2. `02_feature_analysis.ipynb`
3. `03_train_and_calibrate.ipynb`
4. `04_results_and_case_study.ipynb`

## Notes

- Recommended minimum for a reliable baseline: 4+ diverse clean sessions,
  1000+ training rows. Below that, treat models (especially LOF) as
  provisional.
- No eBPF / kernel hooks are involved -- everything here runs on plain
  historical files, so Windows is fine.