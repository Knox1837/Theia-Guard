# Theia-Guard

Converts DARPA Transparent Computing (TC) Engagement 5 (Theia) data into K-Guard-schema provenance graphs, trains an ensemble anomaly detector (IsolationForest + LOF + DBSCAN-style novelty scorer) on them, and evaluates detection performance against DARPA's labeled ground truth.

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Project structure

```
Theia-Guard/
├── data/
│   ├── ground_truth/
│   │   ├── attack_windows.json
│   │   └── Engagement-5-Event-Log.md
│   ├── raw/
│   │   └── ta1-theia-3-e5-official-1.bin
│   ├── processed/
│   │   ├── clean/
│   │   └── attack/
│   └── schema/
│       └── TCCDMDatum.avsc
├── ml/
│   ├── train.py
│   ├── features.py
│   ├── lof_novelty.py
│   ├── dbscan_novelty.py
│   ├── config.py
│   └── models/
├── src/
│   ├── darpa_parser.py
│   ├── darpa_json_parser.py
│   ├── darpa_to_gexf.py
│   ├── ground_truth.py
│   └── evaluate.py
├── check_timestamp_range.py
├── requirements.txt
└── README.md
```

## Pipeline

### 1. Get the data

Place the raw DARPA CDM20 `.bin` (Avro) file in `data/raw/`, and the Avro schema in `data/schema/TCCDMDatum.avsc`.

### 2. DARPA parsing

`src/darpa_parser.py` streams the `.bin` file block-by-block using `fastavro` rather than loading the whole capture into memory. It parses `RECORD_SUBJECT` and `RECORD_EVENT` records, extracting subject UUIDs, PIDs (`cid`), command lines, parent-process links, and — for events — event type, timestamp, subject/object UUIDs, object path, and size.

Malformed or corrupted Avro blocks are skipped individually (both `block_reader` failures and per-record parse errors are caught) so a handful of bad blocks don't abort the whole run. Progress is printed periodically, and the parser reports whether it stopped due to true EOF (`StopIteration`) or an error partway through.

`src/darpa_to_gexf.py` also has a JSON-input path (`src/darpa_json_parser.py`) and auto-dispatches based on the input file's extension (`.bin` → Avro parser, `.json`/`.json.gz` → JSON parser).

### 3. Convert to K-Guard graphs

```
python -m src.darpa_to_gexf --input data/raw/ta1-theia-3-e5-official-1.bin --schema data/schema/TCCDMDatum.avsc --ground-truth data/ground_truth/attack_windows.json --out-clean data/processed/clean --out-attack data/processed/attack --session-name ta1-theia-3-e5-official-1
```

This splits the capture into a benign ("clean") graph and an attack-window graph, written as `.gexf` files named `<session-name>_clean.gexf` / `<session-name>_attack.gexf`.

Currently supported event → edge mappings:

| DARPA event type | Graph relation                |
| ---------------- | ----------------------------- |
| `EVENT_EXECUTE`  | `EXECUTES`                    |
| `EVENT_OPEN`     | `OPENS`                       |
| `EVENT_CONNECT`  | `CONNECTED_TO` (network node) |

`EVENT_SENDMSG` / `EVENT_RECVMSG` don't create new edges — they update `bytes_sent` / `bytes_recv` on an existing edge between the same subject/object pair, if one already exists.

Node/edge attributes written to the graph:

- Process nodes: `type="process"`, `label` (command line, else path, else UUID), `pid` (sentinel `-1` if unknown — GEXF doesn't allow `None` attributes)
- File nodes: `type="file"`, `label` (object path or UUID)
- Network nodes: `type="network"`, `label`
- Edges: `relation`, `timestamp`; `CONNECTED_TO` edges also carry `dest_port` (currently always `-1` — no `NetFlowObject` records are parsed for real port data yet), `bytes_sent`, `bytes_recv`

If `--ground-truth` is omitted or the file doesn't exist, every event is written to the clean graph (no attack split), with a warning printed.

While streaming, `darpa_to_gexf.py` also tracks the minimum and maximum event timestamp seen in the capture, and counts how many events fell inside the configured attack window(s). At the end of the run it prints:

```
Timestamp range: <min_ns> - <max_ns>
Events inside attack windows: <count>
```

This doesn't change the routing logic itself (events are still split the same way) — it just surfaces, at a glance, whether the capture's timestamps actually overlap the configured attack window(s).

### 4. Ground truth

Ground truth is a hand-built JSON file, `data/ground_truth/attack_windows.json`, not auto-extracted from the DARPA PDF/event-log text (those formats aren't structured enough to regex reliably). It contains:

- `attack_windows`: a list of `{start_ns, end_ns, description}` objects. `darpa_to_gexf.py` routes any event whose timestamp falls inside one of these windows into the attack graph instead of the clean graph.
- `malicious_pids`: a list of ground-truth attacker-controlled PIDs (DARPA's `cid` field). `src/evaluate.py` uses these — not UUIDs — to check whether a flagged process node is a true positive, since DARPA's internal UUIDs aren't available from the after-action report.

`src/ground_truth.py` also exposes `load_malicious_uuids()` for compatibility with `evaluate.py`'s original UUID-based design, but this project's ground truth is PID-based, so it always returns an empty set — use `load_malicious_pids()`.

Note: PIDs are only unique within a single boot/session, which is fine for this project's single-session evaluation but wouldn't be safe to reuse across different capture sessions without also checking start timestamps.

### 5. Train the models

```
python -m ml.train --baseline-dir data/processed/clean
```

Fits three independent models on the clean session(s): `IsolationForest`, a scaled/clipped LOF novelty scorer, and a DBSCAN-style novelty scorer. Each model:

- gets its own decision threshold, calibrated on a held-out split of the clean data (default 20% holdout) to hit a target false-positive rate (`THRESHOLD_FPR`, default 3%, overridable per-model via `ml/if_best_params.json`, `ml/lof_best_params.json`, `ml/dbscan_best_params.json` — generated by the corresponding `python -m ml.tune_if` / `ml.tune_lof` / `ml.tune_dbscan` scripts, if present)
- is promoted into the active `ml/models/baseline_model.joblib` independently of the other two — only if its held-out false-positive rate isn't worse than the currently active model of that same type, and it wasn't trained on much less data (less than half the row count) than the current one
- also gets a versioned, timestamped copy saved alongside the active model

A combined "ensemble" score (average of each model's z-scored output) and its own threshold are also calibrated and stored, but the ensemble stats are only recalculated from a run if all three models were promoted together in that same run — if models are promoted individually (a mix of old and new), the previous ensemble calibration is kept instead, since a fresh ensemble would describe a combination of models not actually in the active bundle.

Use `--force` to promote all three models from a run regardless of whether they look worse than the current ones.

### 6. Evaluate against labeled attack data

```
python -m src.evaluate --attack-dir data/processed/attack --model ml/models/baseline_model.joblib --ground-truth data/ground_truth/attack_windows.json --out-json results.json
```

Scores each attack-window graph with all three models. A node is flagged if IsolationForest, LOF, DBSCAN, or the combined ensemble score fires. Ground-truth malicious PIDs (loaded from `attack_windows.json`) are matched against the flagged process nodes' `pid` attribute to compute precision/recall/F1.

`results.json` includes, per graph: total flagged count, a breakdown of how many nodes each individual model flagged, how many were flagged only by the ensemble (missed by all three individual models), how many were flagged by all three, precision/recall/F1 (when malicious PIDs are found in the graph), and a top-15 most-anomalous-nodes table with each model's score/flag for manual inspection.

If none of the ground-truth malicious PIDs appear as nodes in a given graph, `evaluate.py` still runs but flags this in the output (`note` field) — precision/recall aren't meaningful in that case. If no `--ground-truth` is passed at all, evaluation falls back to a rough window-level flagged-fraction approximation instead of PID-level precision/recall.

## Diagnostic utilities

`check_timestamp_range.py` (repo root) is a standalone debugging script, **not part of the core pipeline**. It reads the **raw `.bin` capture file(s) directly** (via `src.darpa_parser.stream_records`, the same parser used by the main pipeline) — it does not read a `.gexf` file. For each `.bin` file in its (currently hardcoded) file list, it streams every event, and reports the total event count plus the minimum and maximum event timestamp (both raw nanoseconds and converted to UTC). It exists to check whether a raw capture actually contains events that fall inside the configured ground-truth attack window, independent of anything downstream in the graph-building step.

`decompress_theia.py` (repo root) is a one-off utility that gzip-decompresses a downloaded DARPA capture into `data/raw/`. It has hardcoded, machine-specific source/destination paths (currently a Windows `Downloads` folder path), so it needs editing before it can be re-run on another machine or for another file — it's not a general-purpose or parameterized tool.

## Current status / known limitations

- On the current `ta1-theia-3-e5-official-1.bin` capture, the pipeline reaches true EOF and processes roughly 1,223 subjects and 9,973,273 events, producing a clean graph of about 5,514 nodes / 23,521 edges.
- **No attack graph is currently produced for this capture** — the run reports "No attack-window events found -- attack graph not written." This is because none of the event timestamps in this particular `.bin` file fall inside the configured ground-truth attack window (2019-05-15, 14:47–14:52 ET); it is not a bug in the graph-splitting logic itself. Use `check_timestamp_range.py` against the raw `.bin` file(s) to confirm the capture's actual time range, and check whether additional `.bin` parts covering the attack date are needed.
- `CONNECTED_TO` edges do not yet have real destination-port data (`dest_port` is always `-1`) since `NetFlowObject` records aren't parsed.
- No eBPF / kernel hooks are involved — everything here runs on plain historical DARPA files, so Windows is fine.

## Notes

- Recommended minimum for a reliable baseline: 4+ diverse clean sessions, 1,000+ training rows. Below that, all three models (and especially LOF/DBSCAN) should be treated as provisional — `ml/train.py` prints a warning in this case.
