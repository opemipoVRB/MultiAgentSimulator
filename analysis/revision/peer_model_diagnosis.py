# analysis/revision/peer_model_diagnosis.py
"""
Where does the peer model go wrong? Against the truth rebuilt from the mission logs, for peers not sensed at the decision (shadow arm `pmm_<sensing>`):
A. speed: the drone's own steps-per-cell estimate (in Manhattan cells) against the true steps per cell of straight-line motion (a different unit: the two differ by the diagonal factor);
B. position error (cells) of the model, of the last sensed position and of SPPDEC's old extrapolation, by the phase of the peer at its last sighting (carrying or empty)
   and the steps since the sighting;
C. carrying at the sighting: did the model say 'delivered' when the peer had delivered, and where was the true drop compared with the model's;
D. empty at the sighting: the true next parcel the peer picked against the model's choice (the open parcel nearest to the peer): how often equal, and the rank of the true one.
Usage: python -m analysis.revision.peer_model_diagnosis <batch dir> <arm>
"""
import collections
import glob
import json
import statistics as st
import sys

import numpy as np

CELL_PX = 100.0


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def main(root, arm):
    speed_true, speed_est = [], []
    err = collections.defaultdict(lambda: [[], [], []])      # (phase, since bucket) -> [model, last, old]
    carry = collections.Counter()
    drop_err = []
    nxt = collections.Counter()
    ranks = []
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        exp = json.load(open(ep))
        for d in glob.glob(ep.replace("experiment.json", f"{arm}_instance_*")):
            fleet = json.load(open(f"{d}/fleet.json"))
            parcels = {p["parcel_id"]: tuple(p["spawn_cell"]) for p in fleet["parcels"]}
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
            pix = {a["agent_id"]: np.array([t["pos"] for t in a["trajectory"]]) / CELL_PX for a in ags}
            picks = {a["agent_id"]: sorted(((x["time"]["step"], x["parcel_id"], tuple(x["cell"])) for x in a["actions"] if x["type"] == "pick")) for a in ags}
            drops = {a["agent_id"]: sorted(((x["time"]["step"], tuple(x["cell"])) for x in a["actions"] if x["type"] == "drop")) for a in ags}
            picked_at = {}
            for aid, pl in picks.items():
                for t, pid, c in pl:
                    picked_at[pid] = t
            for aid, pp in pix.items():
                moving = np.linalg.norm(np.diff(pp, axis=0), axis=1)
                moving = moving[moving > 1e-3]
                if len(moving):
                    speed_true.append(1.0 / float(np.median(moving)))
            for g in ags:
                for dec in g.get("slm_decisions", []):
                    m = (dec.get("proposals") or {}).get("measured")
                    pm = m.get("pm") if m else None
                    if not pm or "peers" not in pm:
                        continue
                    t = dec["time"]["step"]
                    speed_est.append(pm["spc"])
                    pmpos = {p[0]: p for p in pm["peers"]}
                    for pid, last, old, since, vis in m["peers"]:
                        if vis or since < 1 or pid not in pmpos or pid not in cells:
                            continue
                        t0 = t - since
                        traj = cells[pid]
                        true = traj[min(t, len(traj)) - 1]
                        # phase at the last sighting
                        carrying0 = any(pt <= t0 < next((dt for dt, _ in drops[pid] if dt >= pt), 10 ** 9) for pt, _, _ in picks[pid])
                        phase = "carrying" if carrying0 else "empty"
                        b = 0 if since <= 30 else 1 if since <= 100 else 2 if since <= 300 else 3
                        mod = man(pmpos[pid][1], true)
                        err[(phase, b)][0].append(mod)
                        err[(phase, b)][1].append(man(last, true))
                        err[(phase, b)][2].append(man(old, true))
                        if carrying0:
                            drop_t = next((dt for dt, _ in drops[pid] if dt >= t0), None)
                            did = drop_t is not None and drop_t <= t
                            said = pmpos[pid][3] > 0
                            carry[(did, said)] += 1
                            if did and said:
                                drop_err.append(0)
                        else:
                            nxt_pick = next(((pt, pidd, pc) for pt, pidd, pc in picks[pid] if pt > t0), None)
                            open0 = [(i, c) for i, c in parcels.items() if picked_at.get(i, 10 ** 9) > t0]
                            if nxt_pick and open0:
                                order = sorted(open0, key=lambda x: (man(tuple(last), x[1]), x[0]))
                                ids = [i for i, _ in order]
                                if nxt_pick[1] in ids:
                                    r = ids.index(nxt_pick[1]) + 1
                                    ranks.append(r)
                                    nxt["top1" if r == 1 else "top3" if r <= 3 else "other"] += 1
    print(f"A. speed: drone's own steps-per-cell estimate: median {st.median(speed_est):.1f}; true steps per cell while moving: median {st.median(speed_true):.1f} (mean {st.mean(speed_true):.1f})")
    print("B. position error (cells) by phase at the last sighting and steps since; n, model / last sensed / old extrapolation")
    names = ["1-30", "31-100", "101-300", "301+"]
    for phase in ("carrying", "empty"):
        for b in range(4):
            e = err.get((phase, b))
            if e and len(e[0]) >= 20:
                print(f"   {phase:9s} {names[b]:8s} n={len(e[0]):5d}  model {np.mean(e[0]):5.1f} | last {np.mean(e[1]):5.1f} | old {np.mean(e[2]):5.1f}")
    tot = sum(carry.values())
    print(f"C. peers carrying at the sighting (n={tot}): delivered by now and model says delivered {carry[(True, True)]}, delivered but model says not {carry[(True, False)]}, "
          f"not delivered but model says delivered {carry[(False, True)]}, neither {carry[(False, False)]}")
    tn = sum(nxt.values())
    if tn:
        print(f"D. peers empty at the sighting (n={tn}): the true next parcel was the nearest open one {nxt['top1'] / tn:.0%}, within the 3 nearest {(nxt['top1'] + nxt['top3']) / tn:.0%}; median rank {st.median(ranks)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
