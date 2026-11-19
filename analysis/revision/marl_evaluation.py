# analysis/revision/marl_evaluation.py
"""
Matched evaluation of the MARL baseline against the rule arms (design: .ignore/revision/marl_baseline_design.md section 3).
Per config and sensing: for each arm the mean makespan, the step of the last delivery and the tail until the fleet is parked, over the held-out replicates;
for the MARL policies the ratio to the plain rule list and to coded territory computed per training seed (paired over replicates), then the mean over the
training seeds with a t-interval over seeds (n = 5), never the best seed. Completion counts alongside.
Usage: python -m analysis.revision.marl_evaluation <batch dir>
"""
import collections
import glob
import json
import math
import re
import statistics as st
import sys

T975 = {4: 2.776, 5: 2.571, 7: 2.365, 15: 2.131}      # t quantiles (97.5%) by degrees of freedom


def load(root):
    rec = collections.defaultdict(dict)          # (cfg, seed) -> arm -> metrics
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        cfg = ep.split("/")[-3]
        exp = json.load(open(ep))
        for r in exp["results"]:
            name = r.get("strategy_name") or r.get("name")
            arm = re.sub(r"_instance_\d+$", "", name).replace("sppdec_", "")
            ags = json.load(open(ep.replace("experiment.json", f"{name}/agents.json")))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            steps = [t["time"]["step"] for a in ags for t in a.get("tasks", [])]
            m = r["metrics"]
            rec[(cfg, exp["seed"])][arm] = {"makespan": m["mission_makespan_steps"], "delivered": m["tasks_completed"],
                                            "last": max(steps) if steps else 0, "status": r.get("status"),
                                            "parcels": len(exp["parcels"]), "energy": m["total_energy_consumed"]}
    return rec


def interval(xs):
    m = st.mean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, T975.get(len(xs) - 1, 2.0) * st.stdev(xs) / math.sqrt(len(xs))


def main(root):
    rec = load(root)
    cfgs = sorted({k[0] for k in rec})
    for cfg in cfgs:
        for tag in ("full", "noisy"):
            base, terr = f"channel_{tag}", f"territory_{tag}"
            keys = [k for k in rec if k[0] == cfg]
            seeds = sorted({int(a.split("_s")[1].split("_")[0]) for k in keys for a in rec[k] if a.startswith("marl_ippo_s")})
            marl = {s: f"marl_ippo_s{s}_{tag}" for s in seeds}
            ok = [k for k in keys if base in rec[k] and terr in rec[k] and all(m in rec[k] for m in marl.values())]
            if not ok:
                continue
            print(f"\n== {cfg} {tag}: {len(ok)} replicates, {len(seeds)} MARL training seeds")
            def mean(arm, f):
                return st.mean(rec[k][arm][f] for k in ok)
            for arm in (base, terr):
                print(f"   {arm:22s} makespan {mean(arm, 'makespan'):7.0f}  last delivery {mean(arm, 'last'):6.0f}  tail {mean(arm, 'makespan') - mean(arm, 'last'):6.0f}  delivered all: {sum(rec[k][arm]['delivered'] == rec[k][arm]['parcels'] for k in ok)}/{len(ok)}")
            ms, ls = [mean(m, "makespan") for m in marl.values()], [mean(m, "last") for m in marl.values()]
            print(f"   {'MARL (mean of seeds)':22s} makespan {st.mean(ms):7.0f}  last delivery {st.mean(ls):6.0f}  tail {st.mean(ms) - st.mean(ls):6.0f}  delivered all: "
                  f"{sum(rec[k][m]['delivered'] == rec[k][m]['parcels'] for k in ok for m in marl.values())}/{len(ok) * len(marl)}")
            print(f"      per seed makespan: {[round(x) for x in ms]}")
            for ref, name in ((base, "plain rule list"), (terr, "coded territory")):
                per_seed = [st.mean(rec[k][m]["makespan"] / rec[k][ref]["makespan"] for k in ok) for m in marl.values()]
                mu, hw = interval(per_seed)
                print(f"      MARL / {name:16s}: mean over seeds {mu:.3f}  95% interval over seeds [{mu - hw:.3f}, {mu + hw:.3f}]  (seeds: {[round(x, 2) for x in per_seed]})")
            per_seed_last = [st.mean(rec[k][m]["last"] / rec[k][base]["last"] for k in ok) for m in marl.values()]
            mu, hw = interval(per_seed_last)
            print(f"      MARL / plain rule list, time to last delivery: {mu:.3f} [{mu - hw:.3f}, {mu + hw:.3f}]")


if __name__ == "__main__":
    main(sys.argv[1])
