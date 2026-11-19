# analysis/revision/consensus_beliefs.py
"""
How good are the beliefs the consensus drone reasons from? Three measurements on one batch (an outside check against the run logs; the drones never see the truth):
  1. Agreement: two drones that decide at the same step, do their Hungarian and CBBA bundles for each agent match?
  2. Predicted gone: of the parcels a drone predicted gone, how many were truly picked by then, how early or late; and how many plans still ended at an empty pickup cell
     (target_gone), the ghost chases.
  3. Peers' energy: the energy a drone credits to each peer at a decision against the energy that peer had really used.
Usage: python -m analysis.revision.consensus_beliefs <batch dir> <arm> <config> [<config> ...]
"""
import collections
import glob
import json
import statistics as st
import sys


def runs(root, arm, cfgs):
    for cfg in cfgs:
        for ep in sorted(glob.glob(f"{root}/{cfg}/experiment_*/experiment.json")):
            for d in glob.glob(ep.replace("experiment.json", f"{arm}_instance_*")):
                agents = json.load(open(f"{d}/agents.json"))
                agents = list(agents.values()) if isinstance(agents, dict) else agents
                yield cfg, d, json.load(open(f"{d}/fleet.json")), agents


def main(root, arm, cfgs):
    agree = collections.defaultdict(collections.Counter)
    gone_stats = collections.defaultdict(list)
    right_taker = collections.defaultdict(list)
    clock_error = collections.defaultdict(list)
    ghosts = collections.defaultdict(list)
    marked_per_run = collections.defaultdict(list)
    ratios = collections.defaultdict(list)
    abs_err = collections.defaultdict(list)
    for cfg, d, fleet, agents in runs(root, arm, cfgs):
        cell_to_id = {tuple(p["spawn_cell"]): p["parcel_id"] for p in fleet["parcels"]}
        pick_at, pick_by = {}, {}
        for a in agents:
            for x in a["actions"]:
                if x["type"] == "pick":
                    pick_at[x["parcel_id"]] = x["time"]["step"]
                    pick_by[x["parcel_id"]] = a["agent_id"]
        used_at = {}
        for a in agents:
            bat = [e["battery"] for e in a["energy"]]
            used_at[a["agent_id"]] = [bat[0] - b for b in bat]
        # 1. agreement between drones deciding at the same step
        by_step = collections.defaultdict(dict)
        for a in agents:
            for x in a.get("slm_decisions", []):
                bl = ((x.get("proposals") or {}).get("beliefs") or {})
                if bl:
                    by_step[x["time"]["step"]][a["agent_id"]] = bl
        for t, dec in by_step.items():
            ids = list(dec)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    A, B = dec[ids[i]]["bundles"], dec[ids[j]]["bundles"]
                    for q in set(A["hungarian"]) | set(B["hungarian"]):
                        for model in ("hungarian", "cbba_rounds"):
                            a1, b1 = A[model].get(q, []), B[model].get(q, [])
                            agree[(cfg, model + " first task")][(a1[:1] == b1[:1])] += 1
                        ag = lambda X: [t_ for t_ in X["cbba_rounds"].get(q, []) if t_ in set(X["hungarian"].get(q, []))]
                        agree[(cfg, "agreed tasks (both models)")][ag(A) == ag(B)] += 1
        # 2. predicted gone and ghost chases
        n_marked = 0
        for a in agents:
            marks, taker = {}, {}
            for x in a.get("slm_decisions", []):
                bl = (x.get("proposals") or {}).get("beliefs") or {}
                for tid, m in (bl.get("gone") or {}).items():
                    marks.setdefault(tid, m)
                taker.update(bl.get("gone_by") or {})
            n_marked += len(marks)
            for tid, m in marks.items():
                p = pick_at.get(tid)
                if p is not None:
                    gone_stats[cfg].append(p - m)          # <= 0: truly gone when predicted; > 0: predicted too early by this many steps
                    if tid in taker:
                        right = pick_by.get(tid) == taker[tid]
                        right_taker[cfg].append(right)
                        if right:
                            clock_error[cfg].append(p - m)   # the assumed peer did take it: what is left is the clock
            ghosts[cfg].append(sum(1 for o in a.get("plan_outcomes", []) if o["outcome"] == "target_gone"))
        marked_per_run[cfg].append(n_marked / max(1, len(agents)))
        # 3. peers' energy
        for a in agents:
            for x in a.get("slm_decisions", []):
                t = x["time"]["step"]
                for q, belief in ((((x.get("proposals") or {}).get("beliefs") or {}).get("peer_energy")) or {}).items():
                    series = used_at.get(q)
                    if not series or t >= len(series):
                        continue
                    truth = series[t]
                    if truth > 5.0:
                        ratios[cfg].append(belief / truth)
                    abs_err[cfg].append(abs(belief - truth))
    for cfg in cfgs:
        print(f"== {cfg}")
        for k in sorted(k for k in agree if k[0] == cfg):
            c = agree[k]
            print(f"   agreement, {k[1]:28s}: {c[True] / max(1, c[True] + c[False]):4.0%} of {c[True] + c[False]} drone-pair comparisons")
        g = gone_stats[cfg]
        if g:
            early = sum(1 for v in g if v > 0)
            print(f"   predicted gone: {len(g)} parcels; truly gone when predicted {1 - early / len(g):4.0%}; predicted early by median {st.median([v for v in g if v > 0] or [0]):.0f} steps; "
                  f"truly gone before the prediction by median {st.median([-v for v in g if v <= 0] or [0]):.0f} steps; {st.mean(marked_per_run[cfg]):.1f} parcels per drone")
        rt = right_taker[cfg]
        if rt:
            ce = clock_error[cfg]
            print(f"   the assumed peer really was the taker in {sum(rt) / len(rt):4.0%} of {len(rt)} predictions; when it was, the clock was off by median {st.median(ce):+.0f} steps "
                  f"(positive: predicted early), quartiles {st.quantiles(ce, n=4)[0]:+.0f} to {st.quantiles(ce, n=4)[2]:+.0f}" if len(ce) >= 4 else
                  f"   the assumed peer really was the taker in {sum(rt) / len(rt):4.0%} of {len(rt)} predictions")
        print(f"   ghost chases (plans that ended at an empty pickup): {st.mean(ghosts[cfg]):.2f} per drone per run")
        r = ratios[cfg]
        if r:
            qs = st.quantiles(r, n=4)
            print(f"   peers' energy, belief / truth (truth > 5): median {st.median(r):.2f}, quartiles {qs[0]:.2f} to {qs[2]:.2f}; mean absolute error {st.mean(abs_err[cfg]):.1f} energy units ({len(r)} readings)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
