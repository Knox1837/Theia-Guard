import fastavro
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SCHEMA_PATH = SCRIPT_DIR / "schema" / "TCCDMDatum.avsc"
BIN_PATH = SCRIPT_DIR / "raw" / "ta1-theia-3-e5-official-1-combined.bin"

WANTED_EVENTS = {"EVENT_CONNECT", "EVENT_OPEN", "EVENT_EXECUTE", "EVENT_SENDMSG", "EVENT_RECVMSG"}
found = {}

schema = fastavro.schema.load_schema(str(SCHEMA_PATH))
with open(BIN_PATH, "rb") as f:
    block_reader = fastavro.block_reader(f, reader_schema=schema)
    total = 0
    while len(found) < len(WANTED_EVENTS):
        try:
            block = next(block_reader)
        except StopIteration:
            break
        try:
            for raw in block:
                total += 1
                if raw.get("type") != "RECORD_EVENT":
                    continue
                datum = raw.get("datum", {})
                et = datum.get("type")
                if et in WANTED_EVENTS and et not in found:
                    found[et] = datum
                    print(f"=== {et} (record #{total}) ===")
                    print(datum)
                    print()
                if len(found) == len(WANTED_EVENTS):
                    break
        except Exception:
            continue

print("Found:", list(found.keys()))
print("Missing:", WANTED_EVENTS - found.keys())