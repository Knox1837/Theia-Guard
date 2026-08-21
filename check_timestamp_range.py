from pathlib import Path
from src.darpa_parser import stream_records
from datetime import datetime, timezone

files = [
    Path("data/raw/ta1-theia-3-e5-official-1.bin"),
    Path("data/raw/ta1-theia-3-e5-official-2.bin"),
]

schema = Path("data/schema/TCCDMDatum.avsc")

for input_path in files:
    print("\n" + "=" * 60)
    print("FILE:", input_path)
    print("=" * 60)

    min_ts = None
    max_ts = None
    count = 0

    for kind, obj in stream_records(input_path, schema):
        if kind != "event":
            continue

        count += 1
        ts = obj.timestamp_ns

        if min_ts is None or ts < min_ts:
            min_ts = ts

        if max_ts is None or ts > max_ts:
            max_ts = ts

        if count % 500000 == 0:
            print(f"Processed {count:,} events")

    print("\nEvents:", f"{count:,}")
    print("Min timestamp:", min_ts)
    print("Max timestamp:", max_ts)

    if min_ts:
        print(
            "Min date:",
            datetime.fromtimestamp(min_ts / 1e9, timezone.utc)
        )

    if max_ts:
        print(
            "Max date:",
            datetime.fromtimestamp(max_ts / 1e9, timezone.utc)
        )