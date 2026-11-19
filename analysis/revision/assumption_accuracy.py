# analysis/revision/assumption_accuracy.py
"""
How good are a drone's assumptions about what its peers will do?

Each SPPDEC decision record carries `proposals.assumed`: for each reasoning model, the task the drone
assumed every peer would take. The peer's actual next delivered parcel after that decision is the
outcome. Accuracy = share of assumptions that equal it. Peers carrying a parcel at the time are not
assigned a task by the shared-assumption arms, so every assumption concerns a free peer.

Usage:
    python -m analysis.revision.assumption_accuracy --root experiments/revision_fixed_dt/assumption_diag
"""
import argparse
import collections
import glob
import json


def source_of(arm: str) -> str:
    return "hungarian" if "hungarian" in arm else "cbba"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    stats = collections.defaultdict(collections.Counter)
    for f in glob.glob(f"{a.root}/*/experiment_*/sppdec_*_instance_*/agents.json"):
        arm = f.split("/")[-2].rsplit("_instance_", 1)[0]
        src = source_of(arm)
        agents = json.load(open(f))
        deliveries = {ag["agent_id"]: sorted((t["time"]["step"], t["parcel_id"]) for t in ag["tasks"]) for ag in agents}
        for ag in agents:
            for e in ag.get("slm_decisions", []):
                prop = e.get("proposals") or {}
                assumed = (prop.get("assumed") or {}).get(src) or {}
                step = e["time"]["step"]
                for peer, task in assumed.items():
                    nxt = next((pid for s, pid in deliveries.get(peer, []) if s >= step), None)
                    c = stats[arm]
                    c["assumptions"] += 1
                    if nxt is None:
                        c["peer finished nothing after"] += 1
                    else:
                        c["scored"] += 1
                        c["correct"] += (nxt == task)
    print(f"{'arm':40s} {'assumptions':>11s} {'scored':>8s} {'accuracy':>9s}")
    for arm, c in sorted(stats.items()):
        acc = c["correct"] / c["scored"] if c["scored"] else float("nan")
        print(f"{arm:40s} {c['assumptions']:11d} {c['scored']:8d} {acc:9.3f}")


if __name__ == "__main__":
    main()
