# analysis/revision/batch_summary.py
"""Compact per-mission summary of a batch (one row per arm-mission), so the raw trajectories can be archived once a result is logged.
Columns: config, replicate, arm, completed, last_delivery, makespan, energy, messages, task_conflicts, close_pair... (kept small on purpose).
Usage: python -m analysis.revision.batch_summary experiments/peer_model/<batch> out.csv"""
import csv, glob, json, sys

def rows(root):
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        e = json.load(open(ep)); base = ep.rsplit("/", 1)[0]; cfg = ep.split("/")[-3]
        for r in e["results"]:
            m = r["metrics"]
            try:
                ags = json.load(open(f"{base}/{r['artifacts']['agents']}")); ags = list(ags.values()) if isinstance(ags, dict) else ags
                drops = [x["time"]["step"] for g in ags for x in g["actions"] if x["type"] == "drop"]
                conf = sum(1 for g in ags for o in g.get("plan_outcomes", []) if o["outcome"] in ("target_gone", "interrupt_target_gone"))
                last = max(drops) if drops else None
            except Exception:
                last = conf = None
            yield dict(batch=root.rstrip("/").split("/")[-1], config=cfg, replicate=e.get("provenance", {}).get("replicate"), seed=e["seed"], arm=r["family"],
                       agents=m["n_drones"], parcels=m["n_parcels"], delivered=m["tasks_completed"], last_delivery=last, makespan=m["mission_makespan_steps"],
                       energy=m["total_energy_consumed"], messages=m["messages_sent"], task_conflicts=conf, co_occupancy=m["unsafe_failures"], dead=r.get("termination", {}).get("dead_total"))

if __name__ == "__main__":
    out = list(rows(sys.argv[1]))
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"{len(out)} rows -> {sys.argv[2]}")
