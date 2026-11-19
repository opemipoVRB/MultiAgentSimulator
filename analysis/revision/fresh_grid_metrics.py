# analysis/revision/fresh_grid_metrics.py
"""
The mission metrics per strategy on the fresh-seed grid: energy consumed, messages sent, lost drones, time to complete the deliveries, makespan (until every drone is parked),
co-occupancy events (the paper's metric: same grid cell per step in the 2D projection of shared airspace, not physical contact). Read from the experiment.json of each instance (results[i].metrics and .termination), the numbers the simulator wrote.
Usage: python -m analysis.revision.fresh_grid_metrics [batch dir]
"""
import csv
import glob
import json
import statistics as st
import sys

ARMS = [("SPPDEC", "consensus_rule_nogamble_limited"), ("Centralised", "centralized_cc_hungarian_hungarian"), ("CBBA single", "cbba_single_task_cbba_single"),
        ("CBBA bundle", "cbba_full_cbba_bundle"), ("Auction", "decentralized_auction_auction_baseline"), ("Naive", "naive_decentralized_reactive")]
HZ = 60.0


def load_runs(root):
    runs = []   # one dict per (instance, strategy)
    for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
        cfg = ep.split("/")[-3]
        e = json.load(open(ep))
        parcels = int(cfg.split("_P")[1])
        for r in e["results"]:
            fam = r["family"][len("sppdec_"):] if r["family"].startswith("sppdec_") else r["family"]
            m, t = r["metrics"], r.get("termination") or {}
            d = glob.glob(ep.rsplit("/", 1)[0] + f"/{r['strategy_name']}")
            drops = []
            if d:
                ags = json.load(open(d[0] + "/agents.json"))
                ags = list(ags.values()) if isinstance(ags, dict) else ags
                drops = [x["time"]["step"] for g in ags for x in g["actions"] if x["type"] == "drop"]
            runs.append(dict(cfg=cfg, rep=e["provenance"]["replicate"], arm=fam, complete=(len(drops) == parcels), last=max(drops) if drops else None,
                             makespan=m["mission_makespan_steps"], energy=m["total_energy_consumed"], messages=m["messages_sent"], collisions=m["unsafe_failures"],
                             dead=t.get("dead_total", 0), lost_field=t.get("lost_in_field", 0), dead_station=t.get("dead_at_station", 0)))
    return runs


def main(root="experiments/peer_model/fresh_grid_proc"):
    runs = load_runs(root)
    inst = {}
    for r in runs:
        inst.setdefault((r["cfg"], r["rep"]), {})[r["arm"]] = r
    common = [k for k, v in inst.items() if len(v) == 6 and all(x["complete"] for x in v.values())]
    print(f"{len(inst)} instances; {len(common)} in which all six strategies delivered every parcel (the common set for the means below)\n")
    ms = lambda v: f"{st.mean(v):8.1f} ({st.stdev(v):6.1f})"
    print("on the common set: mean (sd) per mission")
    print(f"{'':12s} {'energy':>17s} {'messages sent':>17s} {'time to last delivery s':>25s} {'makespan s':>17s} {'co-occupancy':>17s}")
    rows = []
    for nm, a in ARMS:
        v = [inst[k][a] for k in common]
        print(f"{nm:12s} {ms([x['energy'] for x in v])} {ms([x['messages'] for x in v])} {ms([x['last'] / HZ for x in v]):>25s} {ms([x['makespan'] / HZ for x in v])} {ms([x['collisions'] for x in v])}")
        rows.append([nm, len(v), st.mean(x["energy"] for x in v), st.mean(x["messages"] for x in v), st.mean(x["last"] / HZ for x in v), st.mean(x["makespan"] / HZ for x in v), st.mean(x["collisions"] for x in v)])
    print("\nover all 320 instances (every instance counts, completed or not): lost drones")
    print(f"{'':12s} {'missions with a lost drone':>28s} {'drones lost in the field':>26s} {'drones dead at the station':>28s} {'total messages':>16s}")
    for nm, a in ARMS:
        v = [inst[k][a] for k in inst if a in inst[k]]
        print(f"{nm:12s} {sum(1 for x in v if x['dead'] > 0):>28d} {sum(x['lost_field'] for x in v):>26d} {sum(x['dead_station'] for x in v):>28d} {sum(x['messages'] for x in v):>16d}")
    sizes = [(2, 4), (5, 8), (9, 12), (13, 20)]
    metrics = [("energy consumed", "energy", 1.0), ("messages sent", "messages", 1.0), ("time to last delivery (s)", "last", 1 / HZ), ("makespan (s)", "makespan", 1 / HZ), ("co-occupancy events", "collisions", 1.0)]
    for label, key, scale in metrics:
        print(f"\n{label} by fleet size: mean per mission over the instances in which the strategy delivered every parcel (n in brackets)")
        print(f"{'':12s}" + "".join(f"{f'A{lo}-{hi}':>16s}" for lo, hi in sizes))
        for nm, a in ARMS:
            cells = []
            for lo, hi in sizes:
                v = [inst[k][a][key] * scale for k in inst if a in inst[k] and inst[k][a]["complete"] and lo <= int(k[0].split("_")[0][1:]) <= hi]
                cells.append(f"{st.mean(v):9.1f} ({len(v):3d})" if v else f"{'-':>16s}")
            print(f"{nm:12s}" + "".join(f"{c:>16s}" for c in cells))
    with open(".ignore/revision/fresh_grid_metrics_per_config.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "strategy", "n_complete", "energy", "messages", "time_to_last_delivery_s", "makespan_s", "collisions", "lost_drones"])
        for cfg in sorted({k[0] for k in inst}, key=lambda c: (int(c.split("_")[0][1:]), int(c.split("_P")[1]))):
            for nm, a in ARMS:
                v = [inst[k][a] for k in inst if k[0] == cfg and a in inst[k]]; c = [x for x in v if x["complete"]]
                if c:
                    w.writerow([cfg, nm, len(c), round(st.mean(x["energy"] for x in c), 1), round(st.mean(x["messages"] for x in c), 1), round(st.mean(x["last"] / HZ for x in c), 2),
                                round(st.mean(x["makespan"] / HZ for x in c), 2), round(st.mean(x["collisions"] for x in c), 1), sum(x["dead"] for x in v)])
    with open(".ignore/revision/fresh_grid_metrics_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["strategy", "n_common", "energy", "messages", "time_to_last_delivery_s", "makespan_s", "collisions"]); w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:2])
