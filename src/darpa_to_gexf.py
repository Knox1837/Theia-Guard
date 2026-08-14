"""
Converts a DARPA TC CDM20 .bin file into K-Guard-schema .gexf graphs, split into "clean" (benign) and "attack" (ground-truth window) sessions.
Used by the graph-building pipeline to generate the graphs that are fed into the ML training and evaluation scripts.

Usage:
    python -m src.darpa_to_gexf \
        --input data/raw/ta1-theia-3-e5-official-1-combined.bin \
        --schema data/schema/TCCDMDatum.avsc \
        --ground-truth data/ground_truth/operational_event_log.md \
        --out-clean data/processed/clean \
        --out-attack data/processed/attack

"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import networkx as nx

from .darpa_parser import stream_records as stream_records_avro, Subject, Event
from .darpa_json_parser import stream_records as stream_records_json
from .ground_truth import load_attack_windows


def _stream_records_auto(input_path: Path, schema_path: Path):
    """Dispatches to the Avro or JSON parser based on input file extension."""
    suffixes = "".join(input_path.suffixes).lower()
    if suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        print(f"Detected JSON input ({input_path.name}) -- using darpa_json_parser.")
        yield from stream_records_json(input_path)
    else:
        print(f"Detected Avro .bin input ({input_path.name}) -- using darpa_parser.")
        yield from stream_records_avro(input_path, schema_path)

EVENT_TYPE_TO_RELATION = {
    "EVENT_EXECUTE": "EXECUTES",
    "EVENT_OPEN": "OPENS",
    "EVENT_CONNECT": "CONNECTED_TO",
}
SEND_EVENT_TYPES = {"EVENT_SENDMSG"}
RECV_EVENT_TYPES = {"EVENT_RECVMSG"}


def build_graphs(bin_path: Path, schema_path: Path, attack_windows):
    clean_g = nx.DiGraph()
    attack_g = nx.DiGraph()

    subjects: Dict[str, Subject] = {}

    def in_attack_window(ts_ns: int) -> bool:
        return any(w.start_ns <= ts_ns <= w.end_ns for w in attack_windows)

    def ensure_process_node(g: nx.DiGraph, uuid: str):
        if uuid not in g:
            subj = subjects.get(uuid)
            if subj and subj.cmdline:
                label = subj.cmdline
            elif subj and subj.path:
                label = subj.path
            else:
                label = uuid
            # pid stored as sentinel -1 when unknown -- GEXF disallows None attrs,
            # and this lets evaluate.py match ground-truth PIDs from the report.
            pid = subj.pid if (subj and subj.pid is not None) else -1
            g.add_node(uuid, type="process", label=label, pid=pid)

    print("Streaming records (subjects cached, events written directly)...")
    event_count = 0
    subject_count = 0
    total_yielded = 0
    start_time = time.time()
    PROGRESS_EVERY = 20000

    for kind, obj in _stream_records_auto(bin_path, schema_path):
        total_yielded += 1
        if total_yielded % PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            rate = total_yielded / elapsed if elapsed > 0 else 0
            print(f"...processed {total_yielded} records "
                  f"({subject_count} subjects, {event_count} events, "
                  f"{elapsed:.1f}s elapsed, {rate:.0f} records/sec, "
                  f"clean_g: {clean_g.number_of_nodes()}n/{clean_g.number_of_edges()}e)")

        if kind == "subject":
            subjects[obj.uuid] = obj
            subject_count += 1
            continue

        # kind == "event"
        ev: Event = obj
        event_count += 1
        relation = EVENT_TYPE_TO_RELATION.get(ev.event_type)
        target_g = attack_g if in_attack_window(ev.timestamp_ns) else clean_g

        if relation and ev.subject_uuid and ev.predicate_object_uuid:
            ensure_process_node(target_g, ev.subject_uuid)
            target_uuid = ev.predicate_object_uuid
            target_label = ev.predicate_object_path or target_uuid

            if relation == "CONNECTED_TO":
                if target_uuid not in target_g:
                    target_g.add_node(target_uuid, type="network", label=target_label)
                if not target_g.has_edge(ev.subject_uuid, target_uuid):
                    target_g.add_edge(
                        ev.subject_uuid, target_uuid,
                        relation="CONNECTED_TO",
                        timestamp=ev.timestamp_ns,
                        dest_port=-1,  # sentinel: no NetFlowObject records available for real port data; GEXF disallows None attrs
                        bytes_sent=0,
                        bytes_recv=0,
                    )
            else:  # OPENS / EXECUTES
                if target_uuid not in target_g:
                    target_g.add_node(target_uuid, type="file", label=target_label)
                target_g.add_edge(
                    ev.subject_uuid, target_uuid,
                    relation=relation,
                    timestamp=ev.timestamp_ns,
                )

        elif ev.event_type in SEND_EVENT_TYPES or ev.event_type in RECV_EVENT_TYPES:
            if ev.subject_uuid and ev.predicate_object_uuid:
                g = target_g
                if g.has_edge(ev.subject_uuid, ev.predicate_object_uuid):
                    key = "bytes_sent" if ev.event_type in SEND_EVENT_TYPES else "bytes_recv"
                    g[ev.subject_uuid][ev.predicate_object_uuid][key] += (ev.size or 0)

    print(f"Processed {subject_count} subjects, {event_count} events.")
    return clean_g, attack_g


def main():
    parser = argparse.ArgumentParser(description="Convert DARPA TC CDM .bin to K-Guard .gexf graphs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=False, default=None,
                         help="Path to TCCDMDatum.avsc (only needed for .bin input, not .json/.json.gz)")
    parser.add_argument("--ground-truth", type=Path, required=False, default=None,
                         help="Path to operational_event_log.md. If omitted or "
                              "missing, everything is written as 'clean' (no attack split).")
    parser.add_argument("--out-clean", type=Path, required=True)
    parser.add_argument("--out-attack", type=Path, required=True)
    parser.add_argument("--session-name", type=str, default=None)
    args = parser.parse_args()

    args.out_clean.mkdir(parents=True, exist_ok=True)
    args.out_attack.mkdir(parents=True, exist_ok=True)

    if args.ground_truth is None or not args.ground_truth.exists():
        windows = []
        print(f"WARNING: no ground truth file found at {args.ground_truth}. "
              f"Everything will be written as 'clean' (no attack split). "
              f"Pass --ground-truth once you have operational_event_log.md.")
    else:
        windows = load_attack_windows(args.ground_truth)
        if not windows:
            print("WARNING: ground truth file found but no attack windows parsed from it. "
                  "Everything written as 'clean'. Check src/ground_truth.py's regex against the real file.")
        else:
            for w in windows:
                print(f"Attack window: {w.start_ns} - {w.end_ns} ({w.description})")

    clean_g, attack_g = build_graphs(args.input, args.schema, windows)

    name = args.session_name or args.input.stem
    clean_path = args.out_clean / f"{name}_clean.gexf"
    attack_path = args.out_attack / f"{name}_attack.gexf"

    nx.write_gexf(clean_g, clean_path)
    print(f"Wrote {clean_g.number_of_nodes()} nodes / {clean_g.number_of_edges()} edges -> {clean_path}")

    if attack_g.number_of_nodes() > 0:
        nx.write_gexf(attack_g, attack_path)
        print(f"Wrote {attack_g.number_of_nodes()} nodes / {attack_g.number_of_edges()} edges -> {attack_path}")
    else:
        print("No attack-window events found -- attack graph not written.")


if __name__ == "__main__":
    main()