import networkx as nx
from datetime import datetime, timezone
from pathlib import Path

GEXF_PATH = Path(__file__).parent / "processed" / "clean" / "theia_official2_target1_clean.gexf"

G = nx.read_gexf(GEXF_PATH)

timestamps = []
for _, _, data in G.edges(data=True):
    ts = data.get("timestamp")
    if ts is not None:
        try:
            timestamps.append(int(float(ts)))
        except (TypeError, ValueError):
            pass

if not timestamps:
    print("No timestamps found on any edge -- something else is wrong.")
else:
    min_ts, max_ts = min(timestamps), max(timestamps)
    min_dt = datetime.fromtimestamp(min_ts / 1e9, tz=timezone.utc)
    max_dt = datetime.fromtimestamp(max_ts / 1e9, tz=timezone.utc)
    print(f"Total edges with timestamps: {len(timestamps)}")
    print(f"Earliest event: {min_dt} UTC  (ns={min_ts})")
    print(f"Latest event:   {max_dt} UTC  (ns={max_ts})")

    attack_start_ns = 1557946020000000000
    attack_end_ns = 1557946320000000000
    attack_start_dt = datetime.fromtimestamp(attack_start_ns / 1e9, tz=timezone.utc)
    attack_end_dt = datetime.fromtimestamp(attack_end_ns / 1e9, tz=timezone.utc)
    print(f"\nAttack window we're looking for: {attack_start_dt} to {attack_end_dt} UTC")

    if max_ts < attack_start_ns:
        gap_hours = (attack_start_ns - max_ts) / 1e9 / 3600
        print(f"\n>>> Data ENDS {gap_hours:.1f} hours BEFORE the attack window starts.")
        print(">>> This combined .bin file doesn't reach far enough chronologically.")
        print(">>> You likely need additional .bin.N parts to reach 2019-05-15.")
    elif min_ts > attack_end_ns:
        print("\n>>> Data STARTS after the attack window -- unexpected, check date assumptions.")
    else:
        print("\n>>> Attack window IS within this file's time range -- something else is wrong.")