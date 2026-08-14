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

# Constructed (not directly stated in the report, but consistent with the
# documented technique) -- used to exercise uncommon-port / sensitive-file
# detection features realistically
SECONDARY_C2_IP = "45.33.32.156"
SECONDARY_C2_PORT = 4444  # uncommon port -- credential exfil channel after escalation
SENSITIVE_FILE = "/etc/shadow"
INJECTED_PROCESS_PATH = "/usr/sbin/sshd"
DROPPED_FILE = "/tmp/.cache/sshdlog"
ELEVATE_BINARY_PATH = "/tmp/.mozilla/firefox/xk2j9fQa.default/elevate"  # random-looking path -> high entropy feature

BENIGN_CMDLINES = [
    "/usr/bin/bash", "/usr/sbin/cron", "/usr/bin/systemd-journald",
    "/usr/bin/rsyslogd", "/usr/sbin/sshd", "/usr/bin/python3 /opt/monitor/check.py",
    "/usr/bin/gnome-terminal", "/usr/bin/vim /etc/hosts", "/usr/bin/curl -s https://updates.example.com",
    "/usr/bin/apt-get update", "/usr/bin/systemd-resolved", "/usr/bin/dbus-daemon",
]
BENIGN_FILES = [
    "/etc/hosts", "/var/log/syslog", "/etc/resolv.conf", "/home/darpa/.bashrc",
    "/tmp/session.lock", "/var/log/auth.log", "/etc/nsswitch.conf",
]
BENIGN_PORTS = [80, 443, 53]
BENIGN_IPS = ["93.184.216.34", "142.250.72.14", "151.101.1.69"]


def _uuid_like(label: str) -> str:
    """Generate a stable-ish synthetic node id for readability in case-study viz."""
    return f"synthetic-{label}".replace(" ", "_").replace("/", "-")


def build_synthetic_attack_graph(seed: int = 42, n_benign_processes: int = 40) -> nx.DiGraph:
    rng = random.Random(seed)
    G = nx.DiGraph()

    # --- Benign population -------------------------------------------------
    for i in range(n_benign_processes):
        pid = 1000 + i
        cmdline = rng.choice(BENIGN_CMDLINES)
        proc_id = _uuid_like(f"benign-proc-{i}")
        G.add_node(proc_id, type="process", label=cmdline, pid=pid)

        # 1-2 benign file opens, non-sensitive
        for _ in range(rng.randint(1, 2)):
            f = rng.choice(BENIGN_FILES)
            file_id = _uuid_like(f"file-{f}-{i}-{rng.randint(0,9999)}")
            G.add_node(file_id, type="file", label=f)
            ts = ATTACK_START_NS - rng.randint(60, 3600) * 1_000_000_000
            G.add_edge(proc_id, file_id, relation="OPENS", timestamp=ts)

        # some benign processes make a normal outbound connection
        if rng.random() < 0.4:
            ip = rng.choice(BENIGN_IPS)
            port = rng.choice(BENIGN_PORTS)
            net_id = _uuid_like(f"net-{ip}-{port}")
            if net_id not in G:
                G.add_node(net_id, type="network", label=f"{ip}:{port}")
            ts = ATTACK_START_NS - rng.randint(60, 3600) * 1_000_000_000
            G.add_edge(
                proc_id, net_id, relation="CONNECTED_TO",
                timestamp=ts, dest_port=port,
                bytes_sent=rng.randint(100, 2000),
                bytes_recv=rng.randint(500, 5000),
            )

    # --- Attack chain (grounded in the real report) -------------------------
    firefox_id = _uuid_like("firefox")
    G.add_node(firefox_id, type="process", label="/usr/lib/firefox/firefox", pid=4821)

    t = ATTACK_START_NS + 60 * 1_000_000_000  # ~14:48 -- "Firefox connects" per report

    # Firefox -> real C2 IPs on port 80 (matches report exactly)
    for ip in REAL_C2_IPS:
        net_id = _uuid_like(f"net-{ip}-{REAL_C2_PORT}")
        G.add_node(net_id, type="network", label=f"{ip}:{REAL_C2_PORT}")
        G.add_edge(
            firefox_id, net_id, relation="CONNECTED_TO",
            timestamp=t, dest_port=REAL_C2_PORT,
            bytes_sent=1200, bytes_recv=45000,  # payload download -- recv-heavy
        )
        t += 5 * 1_000_000_000

    # Firefox executes the dropped elevate binary (high-entropy random path)
    elevate_bin_id = _uuid_like("elevate-binary")
    G.add_node(elevate_bin_id, type="file", label=ELEVATE_BINARY_PATH)
    G.add_edge(firefox_id, elevate_bin_id, relation="EXECUTES", timestamp=t)
    t += 10 * 1_000_000_000  # ~14:49 -- "getpid/whoami" per report

    # The elevated shell itself -- THIS is the ground-truth malicious PID
    shell_id = _uuid_like("elevate-shell")
    G.add_node(shell_id, type="process", label="elevate -> /bin/sh", pid=MALICIOUS_PID)

    # Reads a sensitive file (post-escalation recon)
    sensitive_id = _uuid_like(f"file-{SENSITIVE_FILE}")
    G.add_node(sensitive_id, type="file", label=SENSITIVE_FILE)
    read_ts = t
    G.add_edge(shell_id, sensitive_id, relation="OPENS", timestamp=read_ts)
    t += 2 * 1_000_000_000  # ~14:50 -- "elevate success", within the 5s sensitive-read-then-connect window

    # Exfil-style connect on an uncommon port shortly after reading the sensitive file
    secondary_net_id = _uuid_like(f"net-{SECONDARY_C2_IP}-{SECONDARY_C2_PORT}")
    G.add_node(secondary_net_id, type="network", label=f"{SECONDARY_C2_IP}:{SECONDARY_C2_PORT}")
    G.add_edge(
        shell_id, secondary_net_id, relation="CONNECTED_TO",
        timestamp=t, dest_port=SECONDARY_C2_PORT,
        bytes_sent=18500, bytes_recv=300,  # send-heavy -- exfil pattern
    )
    t += 15 * 1_000_000_000

    # Injects into sshd (matches report)
    sshd_id = _uuid_like(INJECTED_PROCESS_PATH)
    G.add_node(sshd_id, type="file", label=INJECTED_PROCESS_PATH)
    G.add_edge(shell_id, sshd_id, relation="OPENS", timestamp=t)
    t += 5 * 1_000_000_000

    # Drops sshdlog (matches report)
    dropped_id = _uuid_like(DROPPED_FILE)
    G.add_node(dropped_id, type="file", label=DROPPED_FILE)
    G.add_edge(shell_id, dropped_id, relation="OPENS", timestamp=t)

    return G


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic attack .gexf grounded in TA51_Final_report_E5.pdf section 8.6."
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/attack/synthetic_theia_target1_attack.gexf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-benign", type=int, default=40)
    args = parser.parse_args()

    G = build_synthetic_attack_graph(seed=args.seed, n_benign_processes=args.n_benign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, args.out)

    n_process = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "process")
    print(f"Wrote synthetic attack graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    print(f"  ({n_process} process nodes, {args.n_benign} benign + 2 attack: firefox pid=4821, elevate shell pid={MALICIOUS_PID})")
    print(f"  -> {args.out}")
    print(f"\nGround-truth malicious PID for evaluate.py: {MALICIOUS_PID} (already in data/ground_truth/attack_windows.json)")


if __name__ == "__main__":
    main()