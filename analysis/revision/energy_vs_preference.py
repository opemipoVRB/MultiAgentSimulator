# analysis/revision/energy_vs_preference.py
"""
Does knowing a peer's energy make its choice predictable? Each decision of each drone (full sensing, so beliefs are the truth) is one choice. The same conditional logit (pickup distance, delivery leg)
is scored on three candidate sets: (1) every open parcel; (2) the open parcels the drone can afford with its true battery (its capacity minus the energy of the trips it has done, reserve floor kept);
(3) the tasks the drone itself offered (its own belief and feasibility). If (2) is far better than (1) and close to (3), energy explains the gap; if it is no better than (1), it does not.
Usage: python -m analysis.revision.energy_vs_preference <batch dir> <arm> <config> [<config> ...]
"""
import glob
import json
import sys

import numpy as np

from analysis.revision.peer_choice_fit import fit, score
from src.physics.physics_2d import Grid2DPhysics

PHYS = Grid2DPhysics()
FLOOR = 3.0
CAPACITY = 100.0


def d(a, b):
    return PHYS.distance(a, b)


def collect(root, arm, cfgs, lo=0.2, hi=8.0):
    sets = {"open": [], "affordable": [], "offered": []}
    mission = 0
    for cfg in cfgs:
        for ep in sorted(glob.glob(f"{root}/{cfg}/experiment_*/experiment.json")):
            exp = json.load(open(ep))
            st = exp["station"]
            centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
            wbycell = {(p["col"], p["row"]): lo + (float(p["weight"]) - 0.5) / 1.5 * (hi - lo) for p in exp["parcels"]}
            for dd in sorted(glob.glob(ep.replace("experiment.json", f"{arm}_instance_*"))):
                mission += 1
                fleet = json.load(open(f"{dd}/fleet.json"))
                ppos = {q["parcel_id"]: tuple(q["spawn_cell"]) for q in fleet["parcels"]}
                wt = {i: wbycell.get(c, 1.0) for i, c in ppos.items()}
                ags = json.load(open(f"{dd}/agents.json"))
                ags = list(ags.values()) if isinstance(ags, dict) else ags
                cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
                picks_all = {i: t for a in ags for t, i in ((x["time"]["step"], x["parcel_id"]) for x in a["actions"] if x["type"] == "pick")}
                for g in ags:
                    me = g["agent_id"]
                    acts = sorted((x["time"]["step"], x["type"], tuple(x["cell"]), x.get("parcel_id")) for x in g["actions"] if x["type"] in ("pick", "drop"))
                    for dec in g.get("slm_decisions", []):
                        pr = dec.get("proposals") or {}
                        m = pr.get("measured")
                        chosen = pr.get(dec.get("policy"))
                        if not m or not isinstance(chosen, str):
                            continue
                        t = dec["time"]["step"]
                        pos = cells[me][min(t, len(cells[me])) - 1]
                        # true energy used so far: the drone's finished trips (deduced from the log)
                        cur, used, pick = tuple(g["start_cell"]), 0.0, None
                        for at, kind, cell, pid in acts:
                            if at >= t:
                                break
                            if kind == "pick":
                                used += PHYS.energy_cost(distance=d(cur, cell), weight=0.0)
                                pick, cur = (cell, wt.get(pid, 1.0)), cell
                            elif kind == "drop" and pick:
                                used += PHYS.energy_cost(distance=d(cur, cell), weight=pick[1]) + 2.0 * PHYS.pick_drop_cost(pick[1])
                                cur, pick = cell, None
                        usable = CAPACITY - used - FLOOR
                        openset = [i for i in ppos if picks_all.get(i, 10 ** 9) >= t]
                        offered = [x[0] for x in m["tasks"] if x[0] in ppos]
                        if chosen not in offered or chosen not in openset:
                            continue

                        def feats(ids):
                            return np.array([[d(pos, ppos[i]), d(ppos[i], centre)] for i in ids], float)

                        def trip(i):
                            return PHYS.energy_cost(distance=d(pos, ppos[i]), weight=0.0) + PHYS.energy_cost(distance=d(ppos[i], centre), weight=wt[i]) + 2.0 * PHYS.pick_drop_cost(wt[i])
                        affordable = [i for i in openset if trip(i) <= usable]
                        for name, ids in (("open", openset), ("affordable", affordable), ("offered", offered)):
                            if len(ids) >= 2 and chosen in ids:
                                sets[name].append((mission, feats(ids), ids.index(chosen)))
    return sets


def main(root, arm, cfgs):
    sets = collect(root, arm, cfgs)
    n = min(len(v) for v in sets.values())
    print(f"{n}+ decisions per candidate set ({arm}, {', '.join(cfgs)}; fit on odd missions, scored on even)")
    for name in ("open", "affordable", "offered"):
        data = sets[name]
        tr = [x for x in data if x[0] % 2 == 1]
        te = [x for x in data if x[0] % 2 == 0]
        w, mu, sd = fit([x[1] for x in tr], [x[2] for x in tr])
        ll, t1, t3 = score(w, mu, sd, [x[1] for x in te], [x[2] for x in te])
        chance = np.mean([1.0 / len(x[1]) for x in te])
        print(f"   {name:11s} n={len(data):5d} mean candidates {np.mean([len(x[1]) for x in data]):5.1f} (chance top-1 {chance:.0%}) | log-lik {ll:6.2f}  top-1 {t1:5.1%}  top-3 {t3:5.1%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
