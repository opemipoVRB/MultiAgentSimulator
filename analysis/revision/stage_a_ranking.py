# analysis/revision/stage_a_ranking.py
"""Ranking of the six strategies on every metric kept in the stage A summaries (fresh seeds 410-412, 21 missions per setting). Lower is better for every metric.
For each baseline the variant (as published, or with time x3) with the lower mean last delivery in that setting is used (the choice that favours the baseline). SPPDEC is the completion-bid arm (C).
Ranks are taken per mission (1 = best, ties share the average rank) and averaged; the overall score is the mean of the metric ranks (equal weights). Co-occupancy is informational and not ranked.
Usage: python -m analysis.revision.stage_a_ranking"""
import csv, collections, statistics as st
P = "sppdec_consensus_rule_nogamble"
SETUPS = {
    "EQUAL SPEED (stage A piece 1)": ("data/batch_summaries/stage_a1.csv", {
        "SPPDEC": [P + "_cbid_limited"], "centralised": ["centralized_cc_hungarian_hungarian", "centralized_cc_hungarian_time3_hungarian"],
        "CBBA single": ["cbba_single_task_cbba_single", "cbba_single_task_time3_limited"], "CBBA bundle": ["cbba_full_cbba_bundle", "cbba_full_time3_limited"],
        "auction": ["decentralized_auction_auction_baseline", "decentralized_auction_time3_limited"], "naive": ["naive_decentralized_reactive", "naive_decentralized_time3_limited"]}),
    "MIXED SPEED (stage A piece 2)": ("data/batch_summaries/stage_a2.csv", {
        "SPPDEC": [P + "_cbid_share_hs_limited"], "centralised": ["centralized_cc_hungarian_hs_hungarian", "centralized_cc_hungarian_time3_hs_hungarian"],
        "CBBA single": ["cbba_single_task_hs_limited", "cbba_single_task_time3_hs_limited"], "CBBA bundle": ["cbba_full_hs_limited", "cbba_full_time3_hs_limited"],
        "auction": ["decentralized_auction_hs_limited", "decentralized_auction_time3_hs_limited"], "naive": ["naive_decentralized_hs_limited", "naive_decentralized_time3_hs_limited"]}),
}
METRICS = [("last delivery", "last_delivery"), ("makespan", "makespan"), ("energy (fleet total)", "energy"), ("energy per parcel", None), ("messages sent", "messages"), ("wasted trips (task conflicts)", "task_conflicts"), ("drones lost", "dead")]
def val(r, key):
    if key is None: return float(r["energy"]) / float(r["parcels"])
    return float(r[key] or 0)
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); out = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]: j += 1
        for k in range(i, j + 1): out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out
for title, (path, fam) in SETUPS.items():
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)): by[(r["config"], r["replicate"])][r["arm"]] = r
    ok = lambda r: r["delivered"] == r["parcels"] and r["last_delivery"] not in ("", "None")
    allarms = [a for v in fam.values() for a in v]
    m = [v for v in by.values() if all(a in v and ok(v[a]) for a in allarms)]
    pick = {}
    for s, arms in fam.items():
        pick[s] = min(arms, key=lambda a: st.mean(float(v[a]["last_delivery"]) for v in m))
    names = list(fam)
    print(f"\n=== {title}: {len(m)} missions, every strategy completed all of them")
    print("variant used: " + "; ".join(f"{s}: {'time x3' if 'time3' in pick[s] else 'as published' if s != 'SPPDEC' else 'completion bid'}" for s in names))
    print(f"\n  {'metric':32s}" + "".join(f"{s:>13s}" for s in names) + "      (mean value, then average rank per mission in brackets)")
    allr = {s: [] for s in names}
    for lab, key in METRICS:
        per = {s: [val(v[pick[s]], key) for v in m] for s in names}
        rk = {s: [] for s in names}
        for i in range(len(m)):
            rr = ranks([per[s][i] for s in names])
            for s, x in zip(names, rr): rk[s].append(x)
        line = f"  {lab:32s}"
        for s in names:
            line += f"{st.mean(per[s]):8.1f}({st.mean(rk[s]):.1f})"
            allr[s].append(st.mean(rk[s]))
        print(line)
    print(f"\n  {'OVERALL (mean of the metric ranks, equal weights)':52s}" + "".join(f"{st.mean(allr[s]):9.2f}" for s in names))
    order = sorted(names, key=lambda s: st.mean(allr[s]))
    print("  overall order: " + " > ".join(f"{s} ({st.mean(allr[s]):.2f})" for s in order))
    # without the metrics where SPPDEC has no stake being a tie at zero for others (drones lost), and with only the three headline metrics
    head = [0, 2, 5]
    hs = {s: st.mean(allr[s][i] for i in head) for s in names}
    print("  headline (last delivery, energy, wasted trips): " + " > ".join(f"{s} ({hs[s]:.2f})" for s in sorted(names, key=lambda s: hs[s])))
