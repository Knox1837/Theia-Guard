import fastavro
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SCHEMA_PATH = SCRIPT_DIR / "schema" / "TCCDMDatum.avsc"
BIN_PATH = SCRIPT_DIR / "raw" / "ta1-theia-3-e5-official-2.bin"

# Step 1: check the raw header bytes to know which reader to use
with open(BIN_PATH, "rb") as f:
    header = f.read(16)
    print("First 16 bytes:", header)
    print("Magic check:", header[:4])
    is_container_format = header[:4] == b"Obj\x01"
    print("Is Avro container format:", is_container_format)

# Step 2: load schema
schema = fastavro.schema.load_schema(SCHEMA_PATH)
print("Schema loaded OK. Top-level type:", schema.get("type") if isinstance(schema, dict) else type(schema))

# Step 3: try reading records with the appropriate method
count = 0
with open(BIN_PATH, "rb") as f:
    if is_container_format:
        records = fastavro.reader(f, reader_schema=schema)
    else:
        records = fastavro.schemaless_reader(f, schema)
        records = [records]  # schemaless_reader returns a single record, not a generator

    for record in records:
        print(record)
        count += 1
        if count >= 3:   # just print first 3 records to sanity-check structure
            break

print(f"\nSuccessfully read {count} record(s) (stopped early for inspection).")