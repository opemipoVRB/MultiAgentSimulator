# analysis/revision/personality_belief_fit.py
"""
Does a drone's belief about its peers shape its choice differently by personality? Each drone's decisions, among the tasks it offered itself, with only what the drone
believed (logged with the decision):
  mine         its own distance to the parcel
  peer_gap     how much farther the nearest peer is believed to be than the drone (believed position; positive: the drone is ahead of every peer)
  peer_gap_last  the same with the last sensed positions
  age          how long ago the parcel was last confirmed in place (as a share of 2000 steps)
  leg          the delivery leg (parcel to the station centre)
Conditional logit per personality (mixed fleet, personalities by drone index in turn); a personality that yields to peers has a larger weight on peer_gap.
Usage: python -m analysis.revision.personality_belief_fit <batch dir> <arm>
"""
import glob
import json
import sys

import numpy as np

from analysis.revision.peer_choice_fit import fit, score, man

MIX = ["rational", "selfish", "altruistic", "reckless"]
NAMES = ["mine", "peer_gap", "peer_gap_last", "age", "leg"]


def collect(root, arm):
    data = []
    mission = 0
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        exp = json.load(open(ep))
        st = exp["station"]
        centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
        for d in sorted(glob.glob(ep.replace("experiment.json", f"{arm}_instance_*"))):
            mission += 1
            fleet = json.load(open(f"{d}/fleet.json"))
            ppos = {p["parcel_id"]: tuple(p["spawn_cell"]) for p in fleet["parcels"]}
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            for g in ags:
                for dec in g.get("slm_decisions", []):
                    pr = dec.get("proposals") or {}
                    m = pr.get("measured")
                    chosen = pr.get(dec.get("policy"))
                    if not m or not isinstance(chosen, str):
                        continue
                    rows, ids = [], []
                    for tid, age, mine, near_pred, near_last in m["tasks"]:
                        if tid not in ppos:
                            continue
                        gap = (near_pred - mine) if near_pred >= 0 else 30.0
                        gap_l = (near_last - mine) if near_last >= 0 else 30.0
                        rows.append([mine, gap, gap_l, age / 2000.0, man(ppos[tid], centre)])
                        ids.append(tid)
                    if len(ids) < 2 or chosen not in ids:
                        continue
                    data.append((mission, np.array(rows, float), ids.index(chosen), int(g["agent_id"].split("_")[-1])))
    return data


def main(root, arm):
    data = collect(root, arm)
    train = [d for d in data if d[0] % 2 == 1]
    test = [d for d in data if d[0] % 2 == 0]
    cols = [0, 1, 3, 4]
    w, mu, sd = fit([d[1][:, cols] for d in train], [d[2] for d in train])
    ll, t1, t3 = score(w, mu, sd, [d[1][:, cols] for d in test], [d[2] for d in test])
    print(f"{len(train)} training and {len(test)} test decisions. Pooled, features {[NAMES[c] for c in cols]}: log-lik {ll:.3f}, top-1 {t1:.1%}, top-3 {t3:.1%}; standardised weights {np.round(w, 2)}")
    tot_ll = tot_n = 0.0
    for k, name in enumerate(MIX):
        tr = [d for d in train if d[3] % 4 == k]
        te = [d for d in test if d[3] % 4 == k]
        wk, muk, sdk = fit([d[1][:, cols] for d in tr], [d[2] for d in tr])
        llk, t1k, t3k = score(wk, muk, sdk, [d[1][:, cols] for d in te], [d[2] for d in te])
        llp, _, _ = score(w, mu, sd, [d[1][:, cols] for d in te], [d[2] for d in te])
        tot_ll += llk * len(te)
        tot_n += len(te)
        print(f"  {name:11s} n={len(te):5d} own fit log-lik {llk:6.3f} top-1 {t1k:5.1%} (pooled {llp:6.3f}) | weights mine/peer_gap/age/leg {np.round(wk, 2)}")
    print(f"  own fits together {tot_ll / tot_n:.3f} against pooled {ll:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
