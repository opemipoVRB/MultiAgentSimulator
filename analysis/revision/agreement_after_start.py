# analysis/revision/agreement_after_start.py
"""
Agreement between SPPDEC drones AFTER the start (R1.1 follow-up). Drones rarely decide at the same step, so tables are compared across drones within a time window.
For two decisions (drone i at step s, drone j at step t, i != j, both after step 1, |s - t| <= W) take the parcels that appear in both tables and ask whether the two
tables give each of them to the same drone. Agreement = share of such (parcel, pair-of-decisions) cases with the same assignee, averaged per mission, then over missions.
The exact-readings arm is the control: tables made a few steps apart differ a little even when every reading is exact, because the drones re-solve as parcels are taken.
Usage: python -m analysis.revision.agreement_after_start [batch dir]   (default experiments/peer_model/agreement_check)
"""
import glob, json, os, re, sys, collections
import numpy as np
ROOT = sys.argv[1] if len(sys.argv) > 1 else "experiments/peer_model/agreement_check"
WINDOWS = (0, 10, 30, 90)

def decisions(path):
    out = []
    for a in json.load(open(os.path.join(path, "agents.json"))):
        for s in a.get("slm_decisions", []):
            t = (s.get("proposals") or {}).get("tables")
            if t and s["time"]["step"] > 1:
                tab = next(iter(t.values()))
                owner = {p: d for d, b in tab.items() for p in b}
                out.append((s["time"]["step"], a["agent_id"], owner))
    return out

def mission_agreement(dec, w):
    same = tot = 0
    for x in range(len(dec)):
        for y in range(x + 1, len(dec)):
            (s, i, oi), (t, j, oj) = dec[x], dec[y]
            if i == j or abs(s - t) > w:
                continue
            for p in oi.keys() & oj.keys():
                tot += 1; same += oi[p] == oj[p]
    return (same, tot)

def main():
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in sorted(glob.glob(f"{ROOT}/A*_P*/experiment_*/sppdec_*_instance_*")):
        cond = re.sub(r".*nogamble_limited_?", "", os.path.basename(d).rsplit("_instance_", 1)[0]) or "exact"
        dec = decisions(d)
        for w in WINDOWS:
            s, t = mission_agreement(dec, w)
            if t:
                res[cond][w].append((s, t))
    print(f"{'condition':8s} " + " ".join(f"{'W='+str(w):>16s}" for w in WINDOWS))
    for cond in ["exact", "n05", "n1", "n2", "l50", "n2l50", "d30", "d120", "bo"]:
        if cond not in res:
            continue
        cells = []
        for w in WINDOWS:
            v = res[cond][w]
            if not v:
                cells.append(f"{'-':>16s}"); continue
            per = [s / t for s, t in v]
            cells.append(f"{100*np.mean(per):6.1f}% (n={len(v):3d})")
        print(f"{cond:8s} " + " ".join(cells))

if __name__ == "__main__":
    main()
