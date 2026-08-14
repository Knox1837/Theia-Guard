"""
Streams a DARPA TC CDM20 .bin file (Avro Object Container format) directly, with per-block error handling for corrupted records.
Used by the graph-building pipeline to avoid having to pre-convert the entire .bin file to JSON first.
"""
from __future__ import annotations

import fastavro
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


def _hex(raw: Optional[bytes]) -> Optional[str]:
    if raw is None:
        return None
    return raw.hex()


_NULL_UUID_HEX = "00" * 16


@dataclass
class Subject:
    uuid: str
    cmdline: Optional[str] = None
    pid: Optional[int] = None
    parent_uuid: Optional[str] = None
    path: Optional[str] = None  # from properties.path, e.g. /usr/bin/sudo


@dataclass
class Event:
    event_type: str
    timestamp_ns: int
    subject_uuid: Optional[str] = None
    predicate_object_uuid: Optional[str] = None
    predicate_object_path: Optional[str] = None
    size: Optional[int] = None


def parse_subject(datum: dict) -> Subject:
    props = datum.get("properties") or {}
    return Subject(
        uuid=_hex(datum.get("uuid")),
        cmdline=datum.get("cmdLine"),
        pid=datum.get("cid"),
        parent_uuid=_hex(datum.get("parentSubject")),
        path=props.get("path") if isinstance(props, dict) else None,
    )


def parse_event(datum: dict) -> Event:
    return Event(
        event_type=datum.get("type"),
        timestamp_ns=int(datum.get("timestampNanos") or 0),
        subject_uuid=_hex(datum.get("subject")),
        predicate_object_uuid=_hex(datum.get("predicateObject")),
        predicate_object_path=datum.get("predicateObjectPath"),
        size=datum.get("size"),
    )


_RECORD_DISPATCH = {
    "RECORD_SUBJECT": ("subject", parse_subject),
    "RECORD_EVENT": ("event", parse_event),
}


def stream_records(bin_path: Path, schema_path: Path) -> Iterator[tuple]:
    """
    Yields (record_kind, parsed_object) for "subject" and "event" records.
    Reads block-by-block so a handful of corrupted blocks (observed: ~9 per
    10M records) don't kill the whole stream -- each bad block is silently
    skipped and the stream continues.
    """
    schema = fastavro.schema.load_schema(str(schema_path))
    with open(bin_path, "rb") as f:
        block_reader = fastavro.block_reader(f, reader_schema=schema)
        block_num = 0
        while True:
            try:
                block = next(block_reader)
            except StopIteration:
                print(f"[darpa_parser] Reached StopIteration (true EOF) at block #{block_num}")
                break
            except Exception as e:
                print(f"[darpa_parser] Stopped at block #{block_num}: "
                      f"{type(e).__name__}: {e}")
                break

            block_num += 1

            try:
                for raw in block:
                    top_type = raw.get("type")
                    dispatch = _RECORD_DISPATCH.get(top_type)
                    if dispatch is None:
                        continue
                    kind, parser = dispatch
                    datum = raw.get("datum", {})
                    try:
                        obj = parser(datum)
                    except Exception:
                        continue

                    if kind == "subject":
                        if obj.uuid is None or obj.uuid == _NULL_UUID_HEX:
                            continue
                    elif kind == "event":
                        # Event has no top-level .uuid -- validate on the
                        # fields we actually need downstream instead.
                        if not obj.subject_uuid or obj.subject_uuid == _NULL_UUID_HEX:
                            continue

                    yield kind, obj
            except Exception as e:
                # corrupted block content -- safe to skip, next(block_reader)
                # above has already advanced past it
                print(f"[darpa_parser] Skipped corrupt block #{block_num}: "
                      f"{type(e).__name__}: {e}")
                continue