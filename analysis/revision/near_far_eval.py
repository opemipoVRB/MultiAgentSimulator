# analysis/revision/near_far_eval.py
"""
A near-reaching personality (selfish) with a far-reaching one (rational) in the same fleet (arm mixsr) against the all-selfish and all-rational fleets (batch near_far.sh). Per config and
sensing, matched complete-case: makespan (ratio to the mixed fleet, 95% paired interval), contested plans, plans whose target was gone, distance flown per drone, and runs without
every parcel delivered. Usage: python -m analysis.revision.near_far_eval <batch dir>
"""
import collections
import glob
import json
import math
import statistics as st
import sys

from analysis.revision.marl_evaluation import load

CFGS = ("A2_P12", "A2_P24", "A4_P12", "A4_P24")


def per_run(root, arm):
    out = {}
    for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
        cfg = ep.split("/")[-3]
        seed = json.load(open(ep)).get("seed")
        for d in glob.glob(ep.replace("experiment.json", f"sppdec_{arm}_instance_*")):
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            n = len(ags)
            oc = collections.Counter(str(o.get("outcome")) for a in ags for o in a.get("plan_outcomes", []))
            out[(cfg, seed)] = {"contested": oc["interrupt_contested"] / n, "gone": (oc["target_gone"] + oc["interrupt_target_gone"]) / n,
                                "distance": st.mean((a.get("distance_cells") or 0.0) for a in ags)}
    return out


def main(root):
    rec = load(root)
    for s in ("full", "noisy"):
        arms = [f"pers_selfish_{s}", f"pers_rational_{s}", f"mixsr_{s}"]
        runs = {a: per_run(root, a) for a in arms}
        for cfg in CFGS:
            keys = [k for k in rec if k[0] == cfg and all(a in rec[k] for a in arms)]
            done = [k for k in keys if all(rec[k][a]["delivered"] == rec[k][a]["parcels"] for a in arms)]
            if len(done) < 3:
                print(f"{s:5s} {cfg}: only {len(done)} complete of {len(keys)}")
                continue
            line = []
            for a in arms:
                ms = st.mean(rec[k][a]["makespan"] for k in done)
                f = lambda key: st.mean(runs[a][(cfg, k[1])][key] for k in done if (cfg, k[1]) in runs[a])
                d = [rec[k][a]["makespan"] / rec[k]["mixsr_" + s if False else f"mixsr_{s}"]["makespan"] for k in done] if False else None
                line.append((a.replace(f"_{s}", ""), ms, f("contested"), f("gone"), f("distance")))
            mix = rec
            print(f"{s:5s} {cfg} (n={len(done)}/{len(keys)}):")
            for name, ms, c, g, dist in line:
                if name == "mixsr":
                    r = ""
                else:
                    ratio = [rec[k][f"{name}_{s}"]["makespan"] / rec[k][f"mixsr_{s}"]["makespan"] for k in done]
                    r = f" (x{st.mean(ratio):.3f}+-{2.04 * st.stdev(ratio) / math.sqrt(len(ratio)):.3f} of the mix)"
                print(f"      {name:14s} makespan {ms:6.0f}{r} | contested/drone {c:.2f} | gone/drone {g:.2f} | distance {dist:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
