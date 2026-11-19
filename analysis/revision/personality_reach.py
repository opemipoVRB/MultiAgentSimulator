# analysis/revision/personality_reach.py
"""
Which personality reaches closer and which farther? From the fleets of one personality (batch fleets_by_personality, full sensing): for every decision, the chosen parcel's pickup distance and
delivery leg against the mean of the offered parcels, and the conditional-logit weights on (pickup distance, delivery leg) per personality. A larger negative weight on pickup distance means
reaching closer; a larger positive weight on the delivery leg means reaching farther out. Usage: python -m analysis.revision.personality_reach <batch dir> [lo,hi]
"""
import glob
import json
import sys

import numpy as np

from analysis.revision.peer_choice_fit import fit, man


def collect(root, p, s, lo, hi):
    rows, rel = [], []
    mission = 0
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        exp = json.load(open(ep))
        st = exp["station"]
        centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
        for d in sorted(glob.glob(ep.replace("experiment.json", f"sppdec_pers_{p}_{s}_instance_*"))):
            mission += 1
            fleet = json.load(open(f"{d}/fleet.json"))
            ppos = {q["parcel_id"]: tuple(q["spawn_cell"]) for q in fleet["parcels"]}
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
            for g in ags:
                for dec in g.get("slm_decisions", []):
                    pr = dec.get("proposals") or {}
                    m = pr.get("measured")
                    chosen = pr.get(dec.get("policy"))
                    if not m or not isinstance(chosen, str):
                        continue
                    offered = [x[0] for x in m["tasks"] if x[0] in ppos]
                    if len(offered) < 3 or chosen not in offered:
                        continue
                    t = dec["time"]["step"]
                    pos = cells[g["agent_id"]][min(t, len(cells[g["agent_id"]])) - 1]
                    dp = np.array([man(pos, ppos[i]) for i in offered], float)
                    dl = np.array([man(ppos[i], centre) for i in offered], float)
                    k = offered.index(chosen)
                    rows.append((mission, np.stack([dp, dl], axis=1), k))
                    rel.append((dp[k] - dp.mean(), dl[k] - dl.mean()))
    return rows, np.array(rel)


def main(root, wr="0.2,8.0"):
    lo, hi = (float(v) for v in wr.split(","))
    for s in ("full",):
        print(f"{s} sensing (beliefs = truth); per personality over all configs")
        for p in ("rational", "selfish", "altruistic", "reckless"):
            rows, rel = collect(root, p, s, lo, hi)
            w, mu, sd = fit([r[1] for r in rows], [r[2] for r in rows])
            raw = w / sd
            print(f"  {p:10s} n={len(rows):5d} | chosen pickup distance vs offered mean {rel[:, 0].mean():+.2f} cells, chosen delivery leg vs offered mean {rel[:, 1].mean():+.2f} cells | "
                  f"logit weight per cell: pickup {raw[0]:+.3f}, delivery leg {raw[1]:+.3f}  (leg / pickup = {raw[1] / -raw[0]:.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0.2,8.0")
