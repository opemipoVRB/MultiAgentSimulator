# analysis/revision/speed_analysis.py
"""Heterogeneous-speed development test: per-arm means on matched complete missions, and each arm minus the arm of record (speed blind, energy only), by fleet size.
Usage: python -m analysis.revision.speed_analysis data/batch_summaries/speed_dev.csv"""
import csv, math, sys, collections, statistics as st
P = "sppdec_consensus_rule_nogamble"
ARMS = [("R", P + "_hs_limited"), ("S", P + "_speed_hs_limited"), ("T", P + "_timeh_hs_limited"), ("TS", P + "_timeh_speed_hs_limited"),
        ("KT", P + "_keep_timeh_hs_limited"), ("KTS", P + "_keep_timeh_speed_hs_limited"),
        ("central", "centralized_cc_hungarian_hs_hungarian"), ("CBBA1", "cbba_single_task_hs_limited"), ("auction", "decentralized_auction_hs_limited")]
rows = list(csv.DictReader(open(sys.argv[1]))); by = collections.defaultdict(dict)
for r in rows: by[(r["config"], r["replicate"])][r["arm"]] = r
done = lambda r: r["delivered"] == r["parcels"] and r["last_delivery"] not in ("", "None") and int(float(r["dead"] or 0)) == 0
nag = lambda c: int(c[1:].split("_")[0])
print("completion (of %d missions):" % len(by), {k: sum(1 for v in by.values() if a in v and done(v[a])) for k, a in ARMS})
def block(title, flt):
    m = [(k, v) for k, v in by.items() if flt(k[0]) and all(a in v and done(v[a]) for _, a in ARMS)]
    print(f"\n== {title}: {len(m)} matched missions (mean last / makespan / energy / task conflicts; last delivery minus R with 95%)")
    base = {k: float(v[ARMS[0][1]]["last_delivery"]) for k, v in m}
    for lab, a in ARMS:
        f = lambda key: st.mean(float(v[a][key]) for _, v in m)
        d = [float(v[a]["last_delivery"]) - base[k] for k, v in m]
        ci = 1.96 * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 and lab != "R" else 0.0
        print(f"  {lab:8s} {f('last_delivery'):7.1f} {f('makespan'):7.1f} {f('energy'):6.1f} {f('task_conflicts'):5.2f}   vs R {st.mean(d):+7.1f} +/- {ci:5.1f}  faster in {sum(x < 0 for x in d):3d}/{len(d)}")
block("all configs", lambda c: True)
block("more than 4 agents (Hungarian regime)", lambda c: nag(c) > 4)
block("4 agents or fewer (CBBA regime)", lambda c: nag(c) <= 4)
for c in sorted({k[0] for k in by}, key=lambda c: (nag(c), int(c.split('_P')[1]))):
    m = [v for k, v in by.items() if k[0] == c and all(a in v and done(v[a]) for _, a in ARMS)]
    print(f"  {c:8s} n={len(m):2d} last: " + "  ".join(f"{l} {st.mean(float(v[a]['last_delivery']) for v in m):6.0f}" for l, a in ARMS[:6]))
