# analysis/revision/schedule_time_analysis.py
"""Pre-registered analysis of the fresh-seed validation (.ignore/revision/schedule_time_protocol.md): variants K, T, KT against the arm of record R.
Matched complete-case, variant minus R per instance, averaged within config, paired t-test across configs, Wilcoxon as check, Holm over the three variants.
Usage: python -m analysis.revision.schedule_time_analysis data/batch_summaries/schedule_time_fresh.csv"""
import csv, math, sys, collections, statistics as st
from scipy import stats
P = "sppdec_consensus_rule_nogamble_"
ARMS = {"R": P + "limited", "K": P + "keep_limited", "T": P + "timeh_limited", "KT": P + "keep_timeh_limited"}
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(dict)
for r in rows: by[(r["config"], r["replicate"])][r["arm"]] = r
def done(r): return r["delivered"] == r["parcels"] and r["last_delivery"] not in ("", "None") and int(float(r["dead"] or 0)) == 0
print("completion (all instances):", {k: sum(1 for v in by.values() if a in v and done(v[a])) for k, a in ARMS.items()}, "of", len(by))
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); m = 0
    for rank, i in enumerate(o): m = max(m, min(1.0, (len(ps) - rank) * ps[i])); out[i] = m
    return out
def compare(var, metric, cfg_filter=lambda c: True):
    per = collections.defaultdict(list)
    for (cfg, rep), v in by.items():
        if not cfg_filter(cfg): continue
        if ARMS["R"] in v and ARMS[var] in v and done(v[ARMS["R"]]) and done(v[ARMS[var]]):
            per[cfg].append(float(v[ARMS[var]][metric]) - float(v[ARMS["R"]][metric]))
    d = [st.mean(x) for x in per.values()]
    if len(d) < 3: return None
    t, p = stats.ttest_1samp(d, 0.0); w = stats.wilcoxon(d).pvalue if any(x != 0 for x in d) else 1.0
    ci = 1.96 * st.stdev(d) / math.sqrt(len(d))
    return dict(mean=st.mean(d), ci=ci, p=p, w=w, n=len(d), better=sum(x < 0 for x in d))
nag = lambda c: int(c[1:].split("_")[0])
for name, flt in (("ALL CONFIGS", lambda c: True), ("MORE THAN 4 AGENTS", lambda c: nag(c) > 4), ("4 AGENTS OR FEWER", lambda c: nag(c) <= 4)):
    print(f"\n== {name}: variant minus R, mean over configs +/- 95% (negative = faster / fewer)")
    res = {v: compare(v, "last_delivery", flt) for v in ("K", "T", "KT")}
    hp = holm([res[v]["p"] if res[v] else 1.0 for v in ("K", "T", "KT")]); 
    for (v, h) in zip(("K", "T", "KT"), hp):
        x = res[v]
        if x: print(f"  {v:3s} last delivery {x['mean']:+7.1f} +/- {x['ci']:5.1f}  (configs {x['n']}, faster in {x['better']}, t p={x['p']:.3g}, Holm {h:.3g}, Wilcoxon {x['w']:.3g})")
    for m, lab in (("makespan", "makespan"), ("energy", "energy"), ("task_conflicts", "task conflicts")):
        print("     " + "  ".join(f"{v} {lab} {compare(v, m, flt)['mean']:+.1f}" for v in ("K", "T", "KT") if compare(v, m, flt)))
print("\n== by load regime (agents per parcel), last delivery, variant minus R")
for lab, lo, hi in (("low <=0.25", 0, 0.25), ("middle", 0.25, 0.6), ("high >0.6", 0.6, 99)):
    f = lambda c, lo=lo, hi=hi: lo < nag(c) / int(c.split("_P")[1]) <= hi if lo else nag(c) / int(c.split("_P")[1]) <= hi
    print(f"  {lab:12s} " + "  ".join(f"{v} {compare(v, 'last_delivery', f)['mean']:+.1f}" for v in ("K", "T", "KT") if compare(v, 'last_delivery', f)))
print("\n== means over instances (matched across the four arms)")
m4 = [v for v in by.values() if all(a in v and done(v[a]) for a in ARMS.values())]
print(f"  matched instances {len(m4)}")
for k, a in ARMS.items():
    print(f"  {k:3s} last {st.mean(float(v[a]['last_delivery']) for v in m4):7.1f} makespan {st.mean(float(v[a]['makespan']) for v in m4):7.1f} energy {st.mean(float(v[a]['energy']) for v in m4):6.1f} conflicts {st.mean(float(v[a]['task_conflicts']) for v in m4):5.2f}")
