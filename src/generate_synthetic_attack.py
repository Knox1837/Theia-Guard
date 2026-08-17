"""
Builds a synthetic K-Guard-schema attack graph reconstructing the documented Firefox Drakon APT technique (real C2 IPs, attacker PID 534051, injection target) since the real captured attack window isn't practical to download(Requiring upto75-80 additional split-file parts (~100+ GB)).
python -m src.generate_synthetic_attack --out data/processed/attack/synthetic_theia_target1_attack.gexf --seed 42 --n-benign 40
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import networkx as nx

# Real attack window from data/ground_truth/attack_windows.json
ATTACK_START_NS = 1557946020000000000  # 2019-05-15 14:47:00 EDT
ATTACK_END_NS = 1557946320000000000    # 2019-05-15 14:52:00 EDT

# Real confirmed values from the report
REAL_C2_IPS = ["189.141.204.211", "208.203.20.42"]
REAL_C2_PORT = 80
MALICIOUS_PID = 534051  # matches data/ground_truth/attack_windows.json malicious_pids

# Constructed consistent with the documented technique mentioned in DARPA used to exercise uncommon-port / sensitive-file detection features realistically
SECONDARY_C2_IP = "45.33.32.156"
SECONDARY_C2_PORT = 4444  # uncommon port- credential exfil channel after escalation
SENSITIVE_FILE = "/etc/shadow"
INJECTED_PROCESS_PATH = "/usr/sbin/sshd"
DROPPED_FILE = "/tmp/.cache/sshdlog"
ELEVATE_BINARY_PATH = "/tmp/.mozilla/firefox/xk2j9fQa.default/elevate"  # random-looking path -> high entropy feature

def _uuid_like(label: str) -> str:
    return f"synthetic-{label}".replace(" ", "_").replace("/", "-")

def _load_real_benign_processes(clean_dir: Path) -> list[tuple[nx.DiGraph, str]]:
    """
    Loads every clean .gexf session in clean_dir and returns a flat pool of
    (source_graph, process_node_id) pairs across ALL sessions -- one entry
    per real process node found. Used to sample real benign subgraphs from,
    rather than fabricating them.
    """
    pool = []
    gexf_paths = sorted(Path(clean_dir).glob("*.gexf"))
    if not gexf_paths:
        raise FileNotFoundError(
            f"No .gexf files found in {clean_dir} -- can't sample real benign "
            f"data. Run darpa_to_gexf.py first to produce clean sessions."
        )
    for p in gexf_paths:
        G = nx.read_gexf(p)
        for n, d in G.nodes(data=True):
            if d.get("type") == "process":
                pool.append((G, n))
    if not pool:
        raise ValueError(f"No process nodes found across any .gexf in {clean_dir}.")
    print(f"Loaded {len(pool)} real process nodes across {len(gexf_paths)} clean session(s) "
          f"as the benign sampling pool.")
    return pool

def _copy_process_subgraph(dest: nx.DiGraph, src_graph: nx.DiGraph, proc_node: str):
    """
    Copies a real process node, its pid, and ALL of its outgoing edges
    (with real target file/network nodes and real edge attributes) verbatim
    into dest INCLUDING the original node id, unmodified.

    """
    proc_attrs = dict(src_graph.nodes[proc_node])
    if proc_node not in dest:
        dest.add_node(proc_node, **proc_attrs)

    for _, target, edge_data in src_graph.out_edges(proc_node, data=True):
        if target not in dest:
            target_attrs = dict(src_graph.nodes[target])
            dest.add_node(target, **target_attrs)
        dest.add_edge(proc_node, target, **dict(edge_data))

def build_synthetic_attack_graph(
    clean_dir: Path,
    seed: int = 42,
    n_benign_processes: int = 40,
) -> nx.DiGraph:
    rng = random.Random(seed)
    G = nx.DiGraph()

    pool = _load_real_benign_processes(clean_dir)
    n_sample = min(n_benign_processes, len(pool))
    sampled = rng.sample(pool, n_sample)
    for src_graph, proc_node in sampled:
        _copy_process_subgraph(G, src_graph, proc_node)

    firefox_id = _uuid_like("firefox")
    G.add_node(firefox_id, type="process", label="/usr/lib/firefox/firefox", pid=4821)

    t = ATTACK_START_NS + 60 * 1_000_000_000  # ~14:48 -- "Firefox connects" per report

    for ip in REAL_C2_IPS:
        net_id = _uuid_like(f"net-{ip}-{REAL_C2_PORT}")
        G.add_node(net_id, type="network", label=f"{ip}:{REAL_C2_PORT}")
        G.add_edge(
            firefox_id, net_id, relation="CONNECTED_TO",
            timestamp=t, dest_port=REAL_C2_PORT,
            bytes_sent=1200, bytes_recv=45000,
        )
        t += 5 * 1_000_000_000

    elevate_bin_id = _uuid_like("elevate-binary")
    G.add_node(elevate_bin_id, type="file", label=ELEVATE_BINARY_PATH)
    G.add_edge(firefox_id, elevate_bin_id, relation="EXECUTES", timestamp=t)
    t += 10 * 1_000_000_000

    shell_id = _uuid_like("elevate-shell")
    G.add_node(shell_id, type="process", label="elevate -> /bin/sh", pid=MALICIOUS_PID)

    sensitive_id = _uuid_like(f"file-{SENSITIVE_FILE}")
    G.add_node(sensitive_id, type="file", label=SENSITIVE_FILE)
    G.add_edge(shell_id, sensitive_id, relation="OPENS", timestamp=t)
    t += 2 * 1_000_000_000

    secondary_net_id = _uuid_like(f"net-{SECONDARY_C2_IP}-{SECONDARY_C2_PORT}")
    G.add_node(secondary_net_id, type="network", label=f"{SECONDARY_C2_IP}:{SECONDARY_C2_PORT}")
    G.add_edge(
        shell_id, secondary_net_id, relation="CONNECTED_TO",
        timestamp=t, dest_port=SECONDARY_C2_PORT,
        bytes_sent=18500, bytes_recv=300,
    )
    t += 15 * 1_000_000_000

    sshd_id = _uuid_like(INJECTED_PROCESS_PATH)
    G.add_node(sshd_id, type="file", label=INJECTED_PROCESS_PATH)
    G.add_edge(shell_id, sshd_id, relation="OPENS", timestamp=t)
    t += 5 * 1_000_000_000

    dropped_id = _uuid_like(DROPPED_FILE)
    G.add_node(dropped_id, type="file", label=DROPPED_FILE)
    G.add_edge(shell_id, dropped_id, relation="OPENS", timestamp=t)

    return G

def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic attack .gexf grounded in TA51_Final_report_E5.pdf section 8.6, "
                    "with benign context sampled from real clean .gexf sessions."
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/attack/synthetic_theia_target1_attack.gexf"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/processed/clean"),
                         help="Directory of real clean .gexf sessions to sample benign context from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-benign", type=int, default=40)
    args = parser.parse_args()

    G = build_synthetic_attack_graph(clean_dir=args.clean_dir, seed=args.seed, n_benign_processes=args.n_benign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, args.out)

    n_process = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "process")
    print(f"\nWrote synthetic attack graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    print(f"  ({n_process} process nodes: {n_process - 2} real benign (sampled verbatim) + "
          f"2 attack: firefox pid=4821, elevate shell pid={MALICIOUS_PID})")
    print(f"  -> {args.out}")
    print(f"\nGround-truth malicious PID for evaluate.py: {MALICIOUS_PID} (already in data/ground_truth/attack_windows.json)")

if __name__ == "__main__":
    main()