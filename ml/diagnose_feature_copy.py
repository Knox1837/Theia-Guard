from pathlib import Path

import networkx as nx
import numpy as np
from .features import extract_features

ATTACK_GEXF = Path("data/processed/attack/synthetic_theia_target1_attack.gexf")
CLEAN_DIR = Path("data/processed/clean")

# Load the attack graph and extract features from it
G_attack = nx.read_gexf(ATTACK_GEXF)
X_attack, nodes_attack, names = extract_features(G_attack)

# Find a few "benign0-...", "benign1-..." sampled process nodes
benign_nodes_in_attack = [n for n in nodes_attack if n.startswith("benign")][:5]
print(f"Checking {len(benign_nodes_in_attack)} sampled benign nodes...\n")

# Load all clean sessions fresh, build a lookup: original_node_id -> (graph, feature_row)
clean_graphs = {}
for p in sorted(CLEAN_DIR.glob("*.gexf")):
    clean_graphs[p.name] = nx.read_gexf(p)

for node_id in benign_nodes_in_attack:
    # node_id looks like "benign{i}-{original_node_id}" strip the "benign{i}-" prefix
    original_id = node_id.split("-", 1)[1]

    idx_in_attack = nodes_attack.index(node_id)
    attack_features = X_attack[idx_in_attack]

    # Find which original clean session graph actually contains this node id
    found_in = None
    for session_name, G_clean in clean_graphs.items():
        if original_id in G_clean.nodes:
            found_in = session_name
            X_clean, nodes_clean, _ = extract_features(G_clean)
            idx_in_clean = nodes_clean.index(original_id)
            original_features = X_clean[idx_in_clean]
            break

    print(f"=== {node_id} (original: {original_id}) ===")
    if found_in is None:
        print("  COULD NOT FIND original node in any clean session -- possible id mismatch bug")
    else:
        print(f"  Found in: {found_in}")
        diffs = attack_features - original_features
        max_diff = np.abs(diffs).max()
        print(f"  Max abs difference across all features: {max_diff}")
        if max_diff > 1e-6:
            print("  !!! MISMATCH DETECTED !!!")
            for name, a, o in zip(names, attack_features, original_features):
                if abs(a - o) > 1e-6:
                    print(f"    {name}: in_attack_graph={a}  vs  original={o}")
        else:
            print("  MATCH- features identical, no copy bug for this node.")
    print()