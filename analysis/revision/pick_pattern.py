# analysis/revision/pick_pattern.py
"""
What does a drone pick, and does what is learned from 2-drone missions carry to 4-drone missions? (batch experiments/peer_model/pattern.sh: full sensing, so a drone's offered tasks are
the open parcels; declared orders; weights stretched to PARCEL_WEIGHT_RANGE.) Every decision of every drone is one choice among its offered tasks; features of a candidate parcel:
  assumed   the energy cost SPPDEC assumes for a peer today: 0.2 x pickup distance x (1 + 0.5 x parcel weight)
  pick      pickup distance                       leg   distance from the parcel to the station centre
  weight    declared weight                       trip  energy of the whole trip: pickup leg empty + delivery leg loaded
  gap       how much nearer the nearest other drone is to the parcel than this drone (>= 0)      rank   rank by pickup distance among the offered
Conditional logit fitted on the 2-drone missions; scored on the held-out 4-drone missions (top-1, top-3, log-likelihood), against the assumed cost alone and against a fit on the 4-drone
missions themselves (the ceiling). Usage: python -m analysis.revision.pick_pattern <batch dir> [lo,hi weight range]
"""
import glob
import json
import os
import sys

import numpy as np

from analysis.revision.peer_choice_fit import fit, score, man

NAMES = ["assumed", "pick", "leg", "weight", "trip", "gap", "rank"]


def collect(root, config, lo, hi):
    data = []
    mission = 0
    for ep in sorted(glob.glob(f"{root}/{config}/experiment_*/experiment.json")):
        exp = json.load(open(ep))
        st = exp["station"]
        centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
        wbycell = {(p["col"], p["row"]): lo + (float(p["weight"]) - 0.5) / 1.5 * (hi - lo) for p in exp["parcels"]}
        for d in sorted(glob.glob(ep.replace("experiment.json", "sppdec_aw_full_a0_instance_*"))):
            mission += 1
            fleet = json.load(open(f"{d}/fleet.json"))
            ppos = {p["parcel_id"]: tuple(p["spawn_cell"]) for p in fleet["parcels"]}
            wt = {i: wbycell.get(c, 1.0) for i, c in ppos.items()}
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
            for g in ags:
                me = g["agent_id"]
                for dec in g.get("slm_decisions", []):
                    pr = dec.get("proposals") or {}
                    m = pr.get("measured")
                    chosen = pr.get(dec.get("policy"))
                    if not m or not isinstance(chosen, str):
                        continue
                    offered = [x[0] for x in m["tasks"] if x[0] in ppos]
                    if len(offered) < 2 or chosen not in offered:
                        continue
                    t = dec["time"]["step"]
                    pos = cells[me][min(t, len(cells[me])) - 1]
                    others = [cells[o][min(t, len(cells[o])) - 1] for o in cells if o != me]
                    dp = np.array([man(pos, ppos[i]) for i in offered], float)
                    dl = np.array([man(ppos[i], centre) for i in offered], float)
                    w = np.array([wt[i] for i in offered], float)
                    assumed = 0.2 * dp * (1 + 0.5 * w)
                    trip = 0.2 * dp + 0.2 * dl * (1 + 0.5 * w)
                    gap = np.array([max(0.0, d0 - min(man(o, ppos[i]) for o in others)) for d0, i in zip(dp, offered)], float)
                    rank = np.argsort(np.argsort(dp)).astype(float)
                    data.append((mission, np.stack([assumed, dp, dl, w, trip, gap, rank], axis=1), offered.index(chosen), me))
    return data


def evaluate(train, test, cols, label):
    w, mu, sd = fit([d[1][:, cols] for d in train], [d[2] for d in train])
    ll, t1, t3 = score(w, mu, sd, [d[1][:, cols] for d in test], [d[2] for d in test])
    return ll, t1, t3, w


def main(root, wr="0.2,8.0"):
    lo, hi = (float(v) for v in wr.split(","))
    two = sum((collect(root, c, lo, hi) for c in ("A2_P12", "A2_P24")), [])
    four = sum((collect(root, c, lo, hi) for c in ("A4_P12", "A4_P24")), [])
    four_train = [d for d in four if d[0] % 2 == 1]
    four_test = [d for d in four if d[0] % 2 == 0]
    chance = np.mean([1.0 / len(d[1]) for d in four_test])
    print(f"{len(two)} choices from 2-drone missions, {len(four_train)} + {len(four_test)} from 4-drone missions (chance top-1 {chance:.1%})\n")
    sets = [("assumed cost alone (today)", [0]), ("pickup distance alone", [1]), ("pickup + delivery leg", [1, 2]), ("+ weight", [1, 2, 3]), ("whole-trip cost alone", [4]),
            ("+ competition gap", [1, 2, 3, 5]), ("all features", [1, 2, 3, 4, 5, 6])]
    print(f"{'feature set':32s} | learned on 2 drones, tested on 4: log-lik top-1 top-3 | fitted on 4, tested on 4 (ceiling): log-lik top-1 top-3 | learned weights (standardised)")
    for label, cols in sets:
        ll, t1, t3, w = evaluate(two, four_test, cols, label)
        llc, t1c, t3c, _ = evaluate(four_train, four_test, cols, label)
        print(f"{label:32s} | {ll:7.2f} {t1:6.1%} {t3:6.1%}                          | {llc:7.2f} {t1c:6.1%} {t3c:6.1%}                       | {np.round(w, 2)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0.2,8.0")
