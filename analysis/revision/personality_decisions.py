# analysis/revision/personality_decisions.py
"""
Do personalities change the DECISIONS, not only the outcome? Same seeds, same world, fleets of one personality (batch fleets_by_personality). Compared with the all-rational fleet:
1. the first decision of every drone (taken at the start, before any run has diverged): share of drones that choose a different parcel;
2. over all decisions: share where the chosen parcel is not the greedy (nearest-cost) one, i.e. the drone deferred to what it believes about its peers;
3. how the plans end: picked / target gone / contested by a peer, per drone;
4. distance flown per drone.
Usage: python -m analysis.revision.personality_decisions <batch dir> <full|noisy>
"""
import collections
import glob
import json
import statistics as st
import sys

PERS = ["rational", "selfish", "altruistic", "reckless"]


def main(root, s):
    first = {p: {} for p in PERS}
    stats = {p: collections.defaultdict(list) for p in PERS}
    for p in PERS:
        for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
            cfg = ep.split("/")[-3]
            seed = json.load(open(ep))["seed"]
            for d in glob.glob(ep.replace("experiment.json", f"sppdec_pers_{p}_{s}_instance_*")):
                cell_of = {q["parcel_id"]: tuple(q["spawn_cell"]) for q in json.load(open(f"{d}/fleet.json"))["parcels"]}     # parcel ids are random per run: compare positions
                ags = json.load(open(f"{d}/agents.json"))
                ags = list(ags.values()) if isinstance(ags, dict) else ags
                for a in ags:
                    decs = [x for x in a.get("slm_decisions", []) if isinstance(((x.get("proposals") or {}).get(x.get("policy"))), str)]
                    if decs:
                        pr = decs[0]["proposals"]
                        first[p][(cfg, seed, a["agent_id"])] = cell_of.get(pr.get(decs[0]["policy"]))
                    nd = deferred = 0
                    for x in decs:
                        pr = x["proposals"]
                        nd += 1
                        deferred += pr.get(x["policy"]) != pr.get("greedy")
                    stats[p]["deferred"].append(deferred / max(1, nd))
                    oc = collections.Counter(str(o.get("outcome")) for o in a.get("plan_outcomes", []))
                    stats[p]["picked"].append(oc["picked"])
                    stats[p]["gone"].append(oc["target_gone"] + oc["interrupt_target_gone"])
                    stats[p]["contested"].append(oc["interrupt_contested"])
                    stats[p]["distance"].append(a.get("distance_cells") or 0.0)
    ref = first["rational"]
    print(f"{s} sensing, all configs")
    for p in PERS:
        common = [k for k in ref if k in first[p]]
        diff = sum(first[p][k] != ref[k] for k in common)
        print(f"  all {p:10s} first decision differs from all-rational: {diff / max(1, len(common)):5.1%} (n={len(common)}) | "
              f"decisions that defer from greedy {st.mean(stats[p]['deferred']):.3f} | per drone: picked {st.mean(stats[p]['picked']):.2f}, target gone {st.mean(stats[p]['gone']):.2f}, "
              f"contested {st.mean(stats[p]['contested']):.2f} | distance {st.mean(stats[p]['distance']):.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
