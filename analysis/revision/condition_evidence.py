# analysis/revision/condition_evidence.py
"""
Evidence for the context arbiters about what the drone's own conditions do to each reasoning model.

Reads a batch of fixed-model arms (`sppdec_evidence_<scenario>_<model>`, run by the same source on the same seeds)
and writes, per scenario and model: parcels delivered, share of runs that delivered everything, drones lost
(ran out of battery), mean energy, and mission end relative to the best model on runs where every model finished.

Usage:
    python -m analysis.revision.condition_evidence --root .ignore/sandbox/evidence \
        --out src/strategies/sppdec/condition_table.json
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SCENARIOS = {"": "rate", "weight_": "weight", "mixed_": "mixed", "shift_": "shift"}
MODELS = ["greedy", "hungarian", "cbba_direct", "cbba_rounds"]


def load(root: Path):
    last = {}
    for f in glob.glob(str(root / "*" / "experiment_*" / "experiment.json")):
        cfg = Path(f).parts[-3]
        exp = Path(f).parts[-2]
        for r in json.load(open(f))["results"]:
            last[(cfg, exp, r.get("strategy_name") or r.get("name"))] = r["metrics"]
    return last


def lost_drones(root: Path):
    """Drones that ended with an empty battery, per (config, experiment, strategy instance)."""
    out = {}
    for d in glob.glob(str(root / "*" / "experiment_*" / "sppdec_evidence_*")):
        p = Path(d)
        n = sum(1 for ag in json.load(open(p / "agents.json")) if ag.get("final_battery", 1) <= 0)
        out[(p.parts[-3], p.parts[-2], p.name)] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    res, lost = load(root), lost_drones(root)
    runs = defaultdict(dict)  # (scenario, config, replicate key) -> model -> metrics
    for (cfg, exp, name), m in res.items():
        mm = re.match(r"sppdec_evidence_(weight_|mixed_|shift_)?(greedy|hungarian|cbba_direct|cbba_rounds)_instance_(\d+)$", name)
        if not mm:
            continue
        sc, model, inst = mm.group(1) or "", mm.group(2), mm.group(3)
        m = dict(m, lost=lost.get((cfg, exp, name), 0))
        runs[(sc, cfg, inst, exp)][model] = m
    table = {}
    for sc, label in SCENARIOS.items():
        keys = [k for k in runs if k[0] == sc]
        cell = {}
        for model in MODELS:
            v = [runs[k][model] for k in keys if model in runs[k]]
            cell[model] = {
                "runs": len(v),
                "delivered_share": round(sum(x["tasks_completed"] for x in v) / max(1, sum(x["n_parcels"] for x in v)), 3),
                "all_delivered_runs": round(float(np.mean([x["tasks_completed"] == x["n_parcels"] for x in v])), 3),
                "drones_lost_per_run": round(float(np.mean([x["lost"] for x in v])), 2),
                "mean_energy": round(float(np.mean([x["total_energy_consumed"] for x in v])), 1),
            }
        full = [k for k in keys if all(m in runs[k] and runs[k][m]["tasks_completed"] == runs[k][m]["n_parcels"] for m in MODELS)]
        if full:
            gm = {m: float(np.exp(np.mean([np.log(runs[k][m]["mission_makespan_steps"]) for k in full]))) for m in MODELS}
            best = min(gm.values())
            for m in MODELS:
                cell[m]["mission_end_vs_best"] = round(gm[m] / best, 3)
        table[label] = {"models": cell, "runs_where_all_models_finished": len(full), "runs": len(keys)}
    Path(a.out).write_text(json.dumps(table, indent=1))
    for label, t in table.items():
        print(f"\n{label}: {t['runs']} runs, all models finished in {t['runs_where_all_models_finished']}")
        for m, c in t["models"].items():
            print(f"  {m:12s} delivered {c['delivered_share']:.3f}  all-delivered {c['all_delivered_runs']:.2f}  "
                  f"lost/run {c['drones_lost_per_run']:.2f}  end vs best {c.get('mission_end_vs_best', 'n/a')}")


if __name__ == "__main__":
    main()
