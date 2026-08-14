"""
Streams a DARPA TC CDM20 newline-delimited JSON file (.json.gz) directly from its gzip, without full decompression to disk.
Used as a fallback when the Avro .bin file is not available, or for debugging / inspection of the raw JSON.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Any


_NULL_UUID = "00000000-0000-0000-0000-000000000000"


def _unwrap(v: Any) -> Any:
    """Unwrap Avro-JSON union type tags: {"long": 5} -> 5, null -> None."""
    if v is None:
        return None
    if isinstance(v, dict) and len(v) == 1:
        return next(iter(v.values()))
    return v


@dataclass
class Subject:
    uuid: str
    cmdline: Optional[str] = None
    pid: Optional[int] = None
    parent_uuid: Optional[str] = None
    path: Optional[str] = None


@dataclass
class Event:
    event_type: str
    timestamp_ns: int
    subject_uuid: Optional[str] = None
    predicate_object_uuid: Optional[str] = None
    predicate_object_path: Optional[str] = None
    size: Optional[int] = None


def parse_subject(inner: dict) -> Subject:
    props = _unwrap(inner.get("properties")) or {}
    return Subject(
        uuid=inner.get("uuid"),
        cmdline=_unwrap(inner.get("cmdLine")),
        pid=_unwrap(inner.get("cid")),
        parent_uuid=_unwrap(inner.get("parentSubject")),
        path=props.get("path") if isinstance(props, dict) else None,
    )


def parse_event(inner: dict) -> Event:
    return Event(
        event_type=inner.get("type"),
        timestamp_ns=int(_unwrap(inner.get("timestampNanos")) or 0),
        subject_uuid=_unwrap(inner.get("subject")),
        predicate_object_uuid=_unwrap(inner.get("predicateObject")),
        predicate_object_path=_unwrap(inner.get("predicateObjectPath")),
        size=_unwrap(inner.get("size")),
    )


_RECORD_DISPATCH = {
    "RECORD_SUBJECT": ("subject", parse_subject),
    "RECORD_EVENT": ("event", parse_event),
}


def _open_maybe_gz(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def stream_records(json_path: Path) -> Iterator[tuple]:
    """
    Yields (record_kind, parsed_object) for "subject" and "event" records,
    streaming line-by-line (works directly on .gz, no full decompression
    to disk needed). A malformed line is skipped, not fatal -- logged so
    we can see if it's a handful (fine) or systematic (needs investigation).
    """
    path = Path(json_path)
    bad_lines = 0
    line_num = 0

    with _open_maybe_gz(path) as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines += 1
                if bad_lines <= 10:
                    print(f"[darpa_json_parser] Skipped malformed line #{line_num}: {e}")
                continue

            top_type = raw.get("type")
            dispatch = _RECORD_DISPATCH.get(top_type)
            if dispatch is None:
                continue
            kind, parser = dispatch

            datum_wrapper = raw.get("datum", {})
            if not isinstance(datum_wrapper, dict) or not datum_wrapper:
                continue
            # datum is {"com.bbn.tc.schema.avro.cdm20.X": {...actual fields...}}
            inner = next(iter(datum_wrapper.values()))

            try:
                obj = parser(inner)
            except Exception:
                continue

            if kind == "subject":
                if not obj.uuid or obj.uuid == _NULL_UUID:
                    continue
            elif kind == "event":
                if not obj.subject_uuid or obj.subject_uuid == _NULL_UUID:
                    continue

            yield kind, obj

    if bad_lines > 10:
        print(f"[darpa_json_parser] Total malformed lines skipped: {bad_lines}")