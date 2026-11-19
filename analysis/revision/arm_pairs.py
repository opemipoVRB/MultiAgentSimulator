# analysis/revision/arm_pairs.py
"""Paired comparison of arms of one batch summary CSV: x minus y on matched complete missions, mean +/- 95%, lower in k of n; all configs, more than 4 agents, 4 or fewer.
Usage: python -m analysis.revision.arm_pairs summary.csv x=armX y=armY [x=armX y=armY ...]   (arm names as full family names; pairs given as X:Y)"""
import csv, collections, math, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1]))); by = collections.defaultdict(dict)
for r in rows: by[(r["config"], r["replicate"])][r["arm"]] = r
ok = lambda r: r["delivered"] == r["parcels"] and r["last_delivery"] not in ("", "None") and int(float(r["dead"] or 0)) == 0
nag = lambda c: int(c[1:].split("_")[0])
def pair(x, y, flt):
    m = [v for k, v in by.items() if flt(k[0]) and x in v and y in v and ok(v[x]) and ok(v[y])]
    out = []
    for met in ("last_delivery", "makespan", "energy", "task_conflicts"):
        d = [float(v[x][met]) - float(v[y][met]) for v in m]
        out.append(f"{met.split('_')[0][:8]} {st.mean(d):+7.1f}+/-{1.96 * st.stdev(d) / math.sqrt(len(d)):5.1f} ({sum(z < 0 for z in d)}/{len(d)})" if len(d) > 1 else f"{met} n/a")
    return len(m), "  ".join(out)
for arg in sys.argv[2:]:
    x, y = arg.split(":")
    for lab, f in (("all", lambda c: True), (">4", lambda c: nag(c) > 4), ("<=4", lambda c: nag(c) <= 4)):
        n, s = pair(x, y, f); print(f"{x[-26:]:26s} - {y[-22:]:22s} {lab:4s} n={n:3d}  {s}")
