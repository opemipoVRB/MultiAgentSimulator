# analysis/revision/fresh_grid_ranking.py
"""
Extra mission metrics and a ranking of the six strategies on the fresh-seed grid, on the instances in which every strategy delivered every parcel (the common set).
Per instance each strategy gets a rank per metric (1 = best, ties share the mean rank); the table gives the mean rank, and an overall rank by the mean of the mean ranks
(equal weights over the metrics listed; a different weighting gives a different order, so the per-metric ranks are shown too).
Usage: python -m analysis.revision.fresh_grid_ranking [batch dir]
"""
import glob
import json
import statistics as st
import sys

from analysis.revision.fresh_grid_metrics import ARMS, HZ, load_runs


def jain(x):
    s = sum(x)
    return (s * s) / (len(x) * sum(v * v for v in x)) if s > 0 else 1.0


def ranks(values, higher_is_better=False):
    order = sorted(range(len(values)), key=lambda i: -values[i] if higher_is_better else values[i])
    out = [0.0] * len(values); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            out[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return out


def main(root="experiments/peer_model/fresh_grid_proc"):
    runs = load_runs(root)
    inst = {}
    for r in runs:
        inst.setdefault((r["cfg"], r["rep"]), {})[r["arm"]] = r
    names = [a for _, a in ARMS]
    for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
        cfg = ep.split("/")[-3]; e = json.load(open(ep)); rep = e["provenance"]["replicate"]
        for res in e["results"]:
            fam = res["family"][len("sppdec_"):] if res["family"].startswith("sppdec_") else res["family"]
            r = inst[(cfg, rep)][fam]
            ags = json.load(open(ep.rsplit("/", 1)[0] + f"/{res['strategy_name']}/agents.json")); ags = list(ags.values()) if isinstance(ags, dict) else ags
            r["jain_energy"] = jain([g["energy_consumed"] for g in ags])
            r["jain_deliveries"] = jain([sum(1 for x in g["actions"] if x["type"] == "drop") for g in ags])
            r["energy_per_parcel"] = r["energy"] / int(cfg.split("_P")[1])
            r["tail_s"] = (r["makespan"] - r["last"]) / HZ if r["last"] is not None else None
            r["last_s"] = r["last"] / HZ if r["last"] is not None else None
            r["makespan_s"] = r["makespan"] / HZ
            r["idle_agents"] = sum(1 for g in ags if not any(x["type"] == "drop" for x in g["actions"]))
            r["task_conflicts"] = sum(1 for g in ags for o in g.get("plan_outcomes", []) if "target_gone" in o["outcome"])   # a plan that ended because another drone had already taken the parcel
    common = [k for k, v in inst.items() if len(v) == 6 and all(x["complete"] for x in v.values())]
    allk = list(inst)
    print(f"common set: {len(common)} instances of {len(allk)} (every strategy delivered every parcel)\n")
    metrics = [("time to last delivery (s)", "last_s", False), ("makespan (s)", "makespan_s", False), ("energy consumed", "energy", False), ("energy per parcel", "energy_per_parcel", False),
               ("messages sent", "messages", False), ("task conflicts (drone flew to a parcel already taken)", "task_conflicts", False), ("co-occupancy events (informational; not physical contact)", "collisions", False), ("parking tail after last delivery (s)", "tail_s", False),
               ("energy fairness across drones (Jain, 1 = equal)", "jain_energy", True), ("delivery fairness across drones (Jain)", "jain_deliveries", True),
               ("drones that delivered nothing", "idle_agents", False)]
    print("MEAN VALUE on the common set")
    print(f"{'':44s}" + "".join(f"{nm:>13s}" for nm, _ in ARMS))
    for label, key, hib in metrics:
        print(f"{label:44s}" + "".join(f"{st.mean(inst[k][a][key] for k in common):13.2f}" for _, a in ARMS))
    print("\nMEAN RANK per metric (1 = best of six; per instance, then averaged)")
    print(f"{'':44s}" + "".join(f"{nm:>13s}" for nm, _ in ARMS))
    mr = {}
    for label, key, hib in metrics:
        rk = [ranks([inst[k][a][key] for a in names], hib) for k in common]
        mr[key] = [st.mean(r[i] for r in rk) for i in range(6)]
        print(f"{label:44s}" + "".join(f"{v:13.2f}" for v in mr[key]))
    core = ["last_s", "makespan_s", "energy", "messages", "task_conflicts", "jain_energy"]      # co-occupancy is shown but not scored: the paper defines it as spatial interference pressure, not contact
    overall = [st.mean(mr[k][i] for k in core) for i in range(6)]
    order = sorted(range(6), key=lambda i: overall[i])
    print(f"\nOVERALL RANK (mean of the mean ranks of: {', '.join(core)}; equal weights; co-occupancy is listed above for information and is not scored)")
    for pos, i in enumerate(order, 1):
        print(f"  {pos}. {ARMS[i][0]:12s} mean rank {overall[i]:.2f}   per metric: " + ", ".join(f"{k} {mr[k][i]:.2f}" for k in core))
    first = [ARMS[i][0] for i in range(6)]
    print("\nHOW OFTEN IS EACH STRATEGY THE FASTEST (time to last delivery), of the common instances; ties counted for each")
    for i, (nm, a) in enumerate(ARMS):
        win = sum(1 for k in common if inst[k][a]["last"] == min(inst[k][b]["last"] for b in names))
        top2 = sum(1 for k in common if sorted(inst[k][b]["last"] for b in names).index(inst[k][a]["last"]) <= 1)
        print(f"  {nm:12s} fastest in {100 * win / len(common):5.1f}%   in the top two in {100 * top2 / len(common):5.1f}%")
    print("\nCOMPLETION over all instances (delivered every parcel)")
    for nm, a in ARMS:
        v = [inst[k][a] for k in allk if a in inst[k]]
        print(f"  {nm:12s} {sum(x['complete'] for x in v)}/{len(v)}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
