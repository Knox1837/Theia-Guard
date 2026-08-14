"""
Loads hand-built ground-truth JSON (data/ground_truth/attack_windows.json), deliberately not auto-parsed from the PDF/log since the source formats aren't structured enough to regex reliably.
Used by darpa_to_gexf.py to split the DARPA CDM20 .bin into "clean" and "attack" graphs, and by evaluate.py to check whether flagged process nodes match real attacker-controlled PIDs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set


@dataclass
class AttackWindow:
    start_ns: int
    end_ns: int
    description: str = ""


def load_attack_windows(gt_json_path: Path) -> List[AttackWindow]:
    """
    Loads attack windows from a structured ground truth JSON
    (data/ground_truth/attack_windows.json format). Returns an empty list
    if the file doesn't exist or has no windows -- caller should treat
    that as "write everything as clean."
    """
    path = Path(gt_json_path)
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    windows = []
    for w in data.get("attack_windows", []):
        windows.append(AttackWindow(
            start_ns=int(w["start_ns"]),
            end_ns=int(w["end_ns"]),
            description=w.get("description", ""),
        ))
    return windows


def load_malicious_pids(gt_json_path: Path) -> Set[int]:
    """
    Loads the ground-truth malicious PIDs (DARPA CDM 'cid' field) from the
    same structured JSON. Use this in evaluate.py to check whether flagged
    process nodes match real attacker-controlled PIDs -- more reliable
    than UUID matching since we know the real PIDs from the report, but
    don't have DARPA's internal UUIDs.

    NOTE: PIDs are only unique within a single boot/session. Fine for a
    single-session evaluation like this project's, but don't reuse across
    different capture sessions without checking startTimestampNanos too.
    """
    path = Path(gt_json_path)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(int(p) for p in data.get("malicious_pids", []))


def load_malicious_uuids(gt_json_path) -> set:
    """
    Kept for compatibility with evaluate.py's original UUID-based design.
    This project's ground truth is PID-based (see load_malicious_pids)
    since we don't have DARPA's internal UUIDs from the report -- this
    always returns an empty set. Use load_malicious_pids instead.
    """
    return set()