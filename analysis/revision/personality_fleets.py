# analysis/revision/personality_fleets.py
"""
Does a fleet of one personality behave differently from the others? Batch experiments/peer_model/fleets_by_personality.sh (declared orders, weights 0.2-8.0). Per config and
sensing level, over the replicates in which every fleet delivered everything (matched complete-case): mean makespan, tail, wasted plans (plans whose target was already gone
or was contested) and energy per delivery; the ratio to the all-rational fleet with a 95% paired interval; and the number of runs without every parcel delivered.
Usage: python -m analysis.revision.personality_fleets <batch dir>
"""
import collections
import glob
import json
import math
import statistics as st
import sys

from analysis.revision.marl_evaluation import load

CFGS = ("A4_P36", "A8_P24", "A12_P24")
FLEETS = ["aw_{s}_a0", "pers_rational_{s}", "pers_selfish_{s}", "pers_altruistic_{s}", "pers_reckless_{s}"]


def wasted(root, arm):
    """Plans per run whose target was gone or contested, by seed."""
    out = {}
    for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
        cfg = ep.split("/")[-3]
        seed = json.load(open(ep)).get("seed")
        for d in glob.glob(ep.replace("experiment.json", f"sppdec_{arm}_instance_*")):
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            out[(cfg, seed)] = sum(1 for a in ags for o in a.get("plan_outcomes", []) if "target_gone" in str(o.get("outcome")) or "contested" in str(o.get("outcome")))
    return out


def main(root):
    rec = load(root)
    for s in ("full", "noisy"):
        arms = [f.format(s=s) for f in FLEETS]
        w = {a: wasted(root, a) for a in arms}
        for cfg in CFGS:
            keys = [k for k in rec if k[0] == cfg and all(a in rec[k] for a in arms)]
            done = [k for k in keys if all(rec[k][a]["delivered"] == rec[k][a]["parcels"] for a in arms)]
            inc = {a: sum(1 for k in keys if rec[k][a]["delivered"] != rec[k][a]["parcels"]) for a in arms}
            print(f"\n== {cfg}, {s} sensing: {len(done)}/{len(keys)} replicates complete in every fleet (when the batteries cannot carry every parcel, 'delivered' below is over all runs)")
            base = arms[1]
            for a in arms:
                if done:
                    ms = st.mean(rec[k][a]["makespan"] for k in done)
                    tail = ms - st.mean(rec[k][a]["last"] for k in done)
                    en = st.mean(rec[k][a]["energy"] for k in done)
                    ws = st.mean(w[a].get(k, float("nan")) for k in done)
                dl = st.mean(rec[k][a]["delivered"] for k in keys) if keys else float("nan")
                if not done:
                    print(f"   {a.replace('_' + s, '').replace('pers_', 'all ').replace('aw_a0', 'default'):18s} no complete replicate | delivered per run {dl:5.1f} | incomplete runs {inc[a]}")
                    continue
                if a == base:
                    rr = "(reference)"
                else:
                    d = [rec[k][a]["makespan"] / rec[k][base]["makespan"] for k in done]
                    rr = f"x{st.mean(d):.3f}+-{2.04 * st.stdev(d) / math.sqrt(len(d)):.3f}"
                print(f"   {a.replace('_' + s, '').replace('pers_', 'all ').replace('aw_a0', 'default'):18s} makespan {ms:6.0f} {rr:16s} tail {tail:4.0f} | wasted plans {ws:5.1f} | energy {en:6.1f} | delivered per run {dl:5.1f} | incomplete runs {inc[a]}")


if __name__ == "__main__":
    main(sys.argv[1])
