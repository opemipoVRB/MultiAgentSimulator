# analysis/revision/delivery_times.py
"""
Step of the last delivery in every run of a batch, next to the recorded mission end.

The mission end (makespan in the exports) needs every parcel delivered, nothing carried and every
drone parked at the station. Under limited observation a drone can keep checking parcels it still
believes exist, so the mission end can lag far behind the last delivery. The last delivery step does
not depend on that.

With --standdown the telemetry of every drone's parking is read too (slower): the step at which each
drone stands down for good (parked from then until the end), reported as the first, median and last
stand-down, and the number of drones that parked before the last delivery. That count is not a measure
of a false belief that the mission is over: with a global view a drone parks as soon as no task is left
for it while others still deliver; under local observation drones keep waiting while they believe a
parcel may remain.

Usage:
    python -m analysis.revision.delivery_times --root experiments/revision_fixed_dt/local_tracking --out .ignore/sandbox/delivery_times_local.csv [--standdown]
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from src.strategies.sppdec.legacy_ids import modernise


def standdown_steps(fleet_path: Path):
    """Per agent: first step from which its mode stays 'idle' (parked) to the end, or None."""
    last_idle = {}
    state = {}
    steps = 0
    for e in json.load(open(fleet_path)).get("fleet_events", []):
        if e.get("type") != "runner_state":
            continue
        steps = e["step"]
        for aid, mode in e["agent_modes"].items():
            if mode == "idle":
                state.setdefault(aid, e["step"])
            else:
                state.pop(aid, None)
    return state, steps


def one(path: str, with_standdown: bool = False):
    parts = Path(path).parts
    config, run_dir, arm_dir = parts[-4], parts[-3], parts[-2]
    arm, inst = arm_dir.rsplit("_instance_", 1)
    last, n = 0, 0
    for ag in json.load(open(path)):
        for t in ag.get("tasks", []):
            last = max(last, t["time"]["step"])
            n += 1
    row = {"config": config, "run": run_dir, "family": modernise(arm), "instance": int(inst), "last_delivery": last, "deliveries": n}
    if with_standdown:
        done, _ = standdown_steps(Path(path).with_name("fleet.json"))
        vals = sorted(done.values())
        n_agents = len(json.load(open(path)))
        row.update({
            "agents": n_agents, "stood_down": len(vals),
            "standdown_first": vals[0] if vals else None,
            "standdown_median": vals[len(vals) // 2] if vals else None,
            "standdown_last": vals[-1] if len(vals) == n_agents else None,
            "early_standdowns": sum(1 for v in vals if v < last),
        })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--standdown", action="store_true")
    a = ap.parse_args()
    files = glob.glob(f"{a.root}/*/experiment_*/*_instance_*/agents.json")
    with ProcessPoolExecutor(max_workers=24) as ex:
        rows = list(ex.map(one, files, [a.standdown] * len(files), chunksize=8))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(len(rows), "runs ->", a.out)


if __name__ == "__main__":
    main()
