# analysis/revision/belief_drift.py
"""
Does the peer-energy belief drift? (R2.3) Each SPPDEC drone keeps, for every peer, the energy it believes the peer has used (`hybrid_log.predicted_energy`: the trips the
drone booked for the peer plus an accrual for unscheduled time; it is NOT summed from sensed motion, so position noise does not enter it directly). The truth is
the peer's own energy used at that step (capacity minus battery, from the peer's energy log). Error = belief - truth, in energy units, by time since the start.
Usage: python -m analysis.revision.belief_drift <batch dir> [out.txt]   (a batch of SPPDEC arms with different sensing conditions)
"""
import collections, glob, json, os, re, sys
import numpy as np
BINS = [(0, 200), (200, 400), (400, 700), (700, 1000), (1000, 10 ** 9)]

def load(path):
    ags = json.load(open(os.path.join(path, "agents.json")))
    used = {}
    for g in ags:
        e = g["energy"]
        cap = e[0]["battery"] / (e[0]["battery_pct"] / 100.0) if e and e[0]["battery_pct"] else 100.0
        used[g["agent_id"]] = {x["time"]["step"]: cap - x["battery"] for x in e}
    return ags, used

def main():
    root = sys.argv[1]
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in sorted(glob.glob(f"{root}/A*_P*/experiment_*/sppdec_*_instance_*")):
        cond = re.sub(r".*nogamble_limited_?", "", os.path.basename(d).rsplit("_instance_", 1)[0]) or "exact"
        ags, used = load(d)
        for g in ags:
            for h in g.get("hybrid_log", []):
                u = used.get(h["peer_id"], {}).get(h["step"])
                if u is None:
                    continue
                for b in BINS:
                    if b[0] <= h["step"] < b[1]:
                        res[cond][b].append(h["predicted_energy"] - u)
    lines = [f"{'condition':8s} " + " ".join(f"{str(b[0])+'-'+(str(b[1]) if b[1] < 10**9 else 'end'):>22s}" for b in BINS), "  (mean error / mean absolute error in energy units; n)"]
    for cond in ["exact", "n05", "n1", "n2", "l50", "n2l50", "d30", "d120", "bo"]:
        if cond not in res:
            continue
        cells = []
        for b in BINS:
            v = np.array(res[cond][b])
            cells.append(f"{v.mean():+6.1f}/{np.abs(v).mean():5.1f} n={len(v):5d}" if len(v) else f"{'-':>22s}")
        lines.append(f"{cond:8s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\n".join(lines))
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
