# analysis/revision/lower_bound.py
"""Lower bound and exact optimum on the time of the last delivery (R1.6, R3.2).

Both bounds RELAX the problem: no battery limit, no pick or drop time, no hover, no station queue, no collisions, straight flight at nominal speed,
a free choice of drop cell inside the station. Whatever a strategy does, its last delivery cannot come sooner, so the ratio of a strategy to the
bound shows how far it is from the best any plan could do in the relaxed problem.

  far   : the parcel whose cheapest trip (best drone) is longest.
  load  : every parcel needs a loaded leg from its cell to the station and an approach leg of at least the same length from the station (every drone
          starts in the station and returns to it after each drop), so the longest of A routes is at least (sum of these) / A.
  exact : (small fleets, P <= 12) the best static assignment of parcels to drones and order of each route in the relaxed problem, by dynamic
          programming over subsets: this is the true optimum of the relaxed problem.

Usage: python -m analysis.revision.lower_bound [raw_root] [summary_csv] [out_csv]
Defaults: experiments/peer_model/fresh_grid_realistic, data/batch_summaries/fresh_grid_realistic.csv, data/batch_summaries/lower_bound.csv
"""
import csv
import glob
import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

STEPS_PER_CELL = 14.2857      # sim_clock.STEPS_PER_CELL: steps to fly one cell at nominal speed


def _cells(st):
    return [(c, r) for c in range(st["col"], st["col"] + st["w"]) for r in range(st["row"], st["row"] + st["h"])]


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bounds(station, starts, pickups, exact_limit=12):
    cells = _cells(station)
    P, A = len(pickups), len(starts)
    to_station = [min(_d(p, c) for c in cells) for p in pickups]            # loaded leg, and the least approach leg
    trip = [[_d(s, p) + to_station[j] for j, p in enumerate(pickups)] for s in starts]
    far = max(min(trip[a][j] for a in range(A)) for j in range(P))
    load = sum(2.0 * t for t in to_station) / A
    out = {"far": far * STEPS_PER_CELL, "load": load * STEPS_PER_CELL}
    out["lb"] = max(out["far"], out["load"])
    if P <= exact_limit and A <= 4:
        out["exact"] = _exact(station, starts, pickups) * STEPS_PER_CELL
    return out


def _exact(station, starts, pickups):
    cells = _cells(station)
    P, A = len(pickups), len(starts)
    # leg[i][j]: from picking i, drop in the best cell, then fly to pick j
    leg = np.array([[min(_d(pickups[i], c) + _d(c, pickups[j]) for c in cells) for j in range(P)] for i in range(P)])
    end = np.array([min(_d(pickups[i], c) for c in cells) for i in range(P)])
    full = 1 << P
    route = []
    for s in starts:
        start = np.array([_d(s, p) for p in pickups])
        dp = np.full((full, P), np.inf)
        for j in range(P):
            dp[1 << j, j] = start[j]
        for mask in range(1, full):
            row = dp[mask]
            if not np.isfinite(row).any():
                continue
            for nxt in range(P):
                if mask & (1 << nxt):
                    continue
                cand = np.min(row + leg[:, nxt])
                nm = mask | (1 << nxt)
                if cand < dp[nm, nxt]:
                    dp[nm, nxt] = cand
        t = np.full(full, np.inf)
        t[0] = 0.0
        for mask in range(1, full):
            t[mask] = np.min(dp[mask] + end)
        route.append(t)
    # g[k][S]: least over assignments of the parcels S to the first k drones of the longest route
    g = route[0].copy()
    for k in range(1, A):
        new = np.full(full, np.inf)
        for S in range(full):
            T = S
            best = np.inf
            while True:
                v = max(g[S ^ T], route[k][T])
                if v < best:
                    best = v
                T = (T - 1) & S
                if T == S:
                    break
            new[S] = best
        g = new
    return float(g[full - 1])


def missions(raw_root):
    for ep in sorted(glob.glob(f"{raw_root}/*/experiment_*/experiment.json")):
        e = json.load(open(ep))
        cfg = ep.split("/")[-3]
        starts = [(a["start_col"], a["start_row"]) for a in e["agents"]]
        pick = [(p["col"], p["row"]) for p in e["parcels"]]
        yield cfg, e["seed"], e["station"], starts, pick


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else "experiments/peer_model/fresh_grid_realistic"
    summ = sys.argv[2] if len(sys.argv) > 2 else "data/batch_summaries/fresh_grid_realistic.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "data/batch_summaries/lower_bound.csv"
    rows = []
    for cfg, seed, station, starts, pick in missions(raw):
        b = bounds(station, starts, pick)
        rows.append(dict(config=cfg, seed=seed, agents=len(starts), parcels=len(pick), **{k: round(v, 1) for k, v in b.items()}))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "agents", "parcels", "far", "load", "lb", "exact"])
        w.writeheader()
        for r in rows:
            r.setdefault("exact", "")
            w.writerow(r)
    print(len(rows), "missions ->", out)
    # ratio of each strategy's last delivery to the bound, on the missions every listed strategy completed
    lbm = {(r["config"], r["seed"]): r for r in rows}
    per = defaultdict(dict)
    for r in csv.DictReader(open(summ)):
        try:
            if int(r["delivered"]) == int(r["parcels"]) and r["last_delivery"] not in ("", "None"):
                per[(r["config"], int(r["seed"]))][r["arm"]] = float(r["last_delivery"])
        except (ValueError, KeyError):
            pass
    arms = sorted({a for d in per.values() for a in d})
    keys = [k for k, d in per.items() if k in lbm and all(a in d for a in arms)]
    print(len(keys), "missions completed by all", len(arms), "strategies")
    for name, col in (("lower bound", "lb"), ("exact relaxed optimum (A<=4, P<=12)", "exact")):
        ks = [k for k in keys if lbm[k].get(col) not in ("", None)]
        if not ks:
            continue
        print(f"\nlast delivery / {name}: {len(ks)} missions")
        for a in arms:
            ratio = [per[k][a] / float(lbm[k][col]) for k in ks]
            print(f"  {a:55s} mean {np.mean(ratio):.3f}  (+{(np.mean(ratio) - 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
