# analysis/revision/reward_validation.py
"""
SPPDEC's own reward, computed offline from existing runs, and its validation (design: .ignore/revision/marl_baseline_design.md section 4).

Realized reward of one decision (scenario-free, from what the drone's own log holds): the k-th decision of a drone pairs with the k-th plan outcome.
  delivered trip      e = ideal time / time taken, ideal = cells (drone position at the decision -> pickup -> drop cell) x TAU steps per cell,
                      TAU = the 5th percentile of (time taken / cells) over all delivered trips in the data (an unobstructed trip); e is capped at 1
  wasted trip         -1 (the target was gone on arrival, or the plan was interrupted because a peer was ahead)
Validation 1 (mission level): across runs, the mean reward of a run against its makespan, overall and inside each stratum (config and sensing).
Validation 2 (source level): the mean reward of each arm against its mission-level result, within each stratum, paired over replicates.
Usage: python reward_validation.py <batch dir> [<batch dir> ...]
"""
import collections
import glob
import json
import math
import re
import sys

import numpy as np

WASTED = -1.0


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_runs(batch_dirs):
    runs = []
    for root in batch_dirs:
        for exp_path in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
            cfg = exp_path.split("/")[-3]
            exp = json.load(open(exp_path))
            for r in exp["results"]:
                name = r.get("strategy_name") or r.get("name")
                if name.startswith("centralized"):
                    continue
                path = exp_path.replace("experiment.json", f"{name}/agents.json")
                ags = json.load(open(path))
                ags = list(ags.values()) if isinstance(ags, dict) else ags
                trips = []
                for ag in ags:
                    cells = [tuple(t["cell"]) for t in ag["trajectory"]]
                    drops = {a["parcel_id"]: a for a in ag["actions"] if a["type"] == "drop"}
                    picks = {a["parcel_id"]: a for a in ag["actions"] if a["type"] == "pick"}
                    decisions = ag.get("decision_actions", [])
                    outcomes = ag.get("plan_outcomes", [])
                    for dec, out in zip(decisions, outcomes):
                        t = dec["time"]["step"]
                        pid = dec["decided_parcel"]
                        pos = cells[min(t, len(cells)) - 1]
                        if out["outcome"] == "picked" and pid in drops and pid in picks:
                            pk, dr = picks[pid], drops[pid]
                            cells_total = (abs(pos[0] - pk["cell"][0]) + abs(pos[1] - pk["cell"][1])
                                           + abs(pk["cell"][0] - dr["cell"][0]) + abs(pk["cell"][1] - dr["cell"][1]))
                            trips.append({"policy": dec["policy"], "ok": True, "cells": max(1, cells_total), "elapsed": max(1, dr["time"]["step"] - t if "time" in dr else dr["step"] - t)})
                        else:
                            trips.append({"policy": dec["policy"], "ok": False, "cells": 1, "elapsed": 0})
                m = r["metrics"]
                arm = re.sub(r"_instance_\d+$", "", name).replace("sppdec_", "")
                runs.append({"cfg": cfg, "seed": exp.get("seed"), "arm": arm, "makespan": m["mission_makespan_steps"],
                             "delivered": m["tasks_completed"], "trips": trips, "batch": root})
    return runs


def main(batch_dirs):
    runs = load_runs(batch_dirs)
    per_cell = [t["elapsed"] / t["cells"] for run in runs for t in run["trips"] if t["ok"]]
    tau = float(np.percentile(per_cell, 5))
    print(f"{len(runs)} runs, {sum(len(r['trips']) for r in runs)} decisions; TAU (5th percentile of steps per cell) = {tau:.2f}")
    for run in runs:
        rewards = [min(1.0, tau * t["cells"] / t["elapsed"]) if t["ok"] else WASTED for t in run["trips"]]
        run["reward"] = float(np.mean(rewards)) if rewards else float("nan")
        run["wasted_share"] = float(np.mean([not t["ok"] for t in run["trips"]])) if run["trips"] else float("nan")
    runs = [r for r in runs if not math.isnan(r["reward"]) and r["delivered"] >= 20]     # missions that delivered (rate scenario stalls differently)
    # stratum: config plus sensing class from the arm name
    def sensing(arm):
        return "noisy" if "noisy" in arm else "full" if "full" in arm else "other"
    strata = collections.defaultdict(list)
    for r in runs:
        strata[(r["cfg"], sensing(r["arm"]))].append(r)
    print("\nValidation 1: mean decision reward of a run against its makespan (Spearman; negative = a better reward goes with a faster mission)")
    allr = spearman([r["reward"] for r in runs], [r["makespan"] for r in runs])
    print(f"  all runs pooled: n={len(runs)} rho={allr:.3f}")
    for k, rs in sorted(strata.items()):
        if len(rs) >= 8:
            print(f"  {k[0]:7s} {k[1]:6s}: n={len(rs):3d} rho={spearman([r['reward'] for r in rs], [r['makespan'] for r in rs]):.3f}   "
                  f"(wasted-trip share vs makespan rho={spearman([r['wasted_share'] for r in rs], [r['makespan'] for r in rs]):.3f})")
    print("\nValidation 2: arms ranked by mean reward against mean makespan, per stratum (the ranks should agree)")
    for k, rs in sorted(strata.items()):
        by_arm = collections.defaultdict(list)
        for r in rs:
            by_arm[r["arm"]].append(r)
        arms = [(a, np.mean([x["reward"] for x in v]), np.mean([x["makespan"] for x in v]), len(v)) for a, v in by_arm.items() if len(v) >= 4]
        if len(arms) >= 3:
            rho = spearman([a[1] for a in arms], [a[2] for a in arms])
            print(f"  {k[0]:7s} {k[1]:6s}: {len(arms)} arms, rank correlation of mean reward with mean makespan rho={rho:.3f}")
            for a in sorted(arms, key=lambda a: -a[1])[:8]:
                print(f"       {a[0][:44]:44s} reward {a[1]:.3f}  makespan {a[2]:7.0f}  (n={a[3]})")
    print("\nValidation 3 (paired, the environment held fixed): within one seed and sensing, do two arms order the same way by reward and by makespan?")
    groups = collections.defaultdict(list)
    for r in runs:
        if sensing(r["arm"]) in ("full", "noisy"):
            groups[(r["cfg"], r["seed"], sensing(r["arm"]))].append(r)
    agree = collections.defaultdict(lambda: [0, 0])
    big = collections.defaultdict(lambda: [0, 0])
    for key, rs in groups.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["arm"] == b["arm"] or a["makespan"] == b["makespan"]:
                    continue
                same = (a["reward"] - b["reward"]) * (b["makespan"] - a["makespan"]) > 0       # higher reward <-> shorter makespan
                stratum = (key[0], key[2])
                agree[stratum][0] += int(same); agree[stratum][1] += 1
                if abs(a["makespan"] - b["makespan"]) / min(a["makespan"], b["makespan"]) >= 0.10:   # pairs that differ by at least 10% in makespan
                    big[stratum][0] += int(same); big[stratum][1] += 1
    for k in sorted(agree):
        print(f"  {k[0]:7s} {k[1]:6s}: agree on {agree[k][0]}/{agree[k][1]} pairs ({agree[k][0] / agree[k][1]:.2f}); on pairs differing by >=10% in makespan: {big[k][0]}/{big[k][1]} ({big[k][0] / max(1, big[k][1]):.2f})")
    tot = [sum(v[0] for v in agree.values()), sum(v[1] for v in agree.values())]
    totb = [sum(v[0] for v in big.values()), sum(v[1] for v in big.values())]
    print(f"  all strata: {tot[0]}/{tot[1]} ({tot[0] / tot[1]:.2f}); pairs differing by >=10%: {totb[0]}/{totb[1]} ({totb[0] / max(1, totb[1]):.2f})")
    return runs, tau


if __name__ == "__main__":
    main(sys.argv[1:])
