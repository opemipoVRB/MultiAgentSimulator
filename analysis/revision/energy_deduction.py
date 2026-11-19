# analysis/revision/energy_deduction.py
"""
Can a peer's energy used be deduced from the work it did? For each drone of a mission: its trips from the log (the cell it picked from, the parcel's declared weight, the cell it dropped at),
priced with the shared consumption formula (empty pickup leg, loaded delivery leg, pick and drop), against the energy the drone actually consumed. Reports the ratio deduced / actual and the
correlation over drones, for the missions of one arm. Usage: python -m analysis.revision.energy_deduction <batch dir> <config> <arm> [lo,hi weight range]
"""
import glob
import json
import statistics as st
import sys

import numpy as np

from src.physics.physics_2d import Grid2DPhysics


def man(a, b):
    return Grid2DPhysics().distance(a, b)       # the travel (straight-line) distance, as the drone flies


def main(root, cfg, arm, wr="0.2,8.0"):
    lo, hi = (float(v) for v in wr.split(","))
    phys = Grid2DPhysics()
    dedu, act, ratio, rows = [], [], [], []
    for ep in sorted(glob.glob(f"{root}/{cfg}/experiment_*/experiment.json")):
        exp = json.load(open(ep))
        wbycell = {(p["col"], p["row"]): lo + (float(p["weight"]) - 0.5) / 1.5 * (hi - lo) for p in exp["parcels"]}
        for d in glob.glob(ep.replace("experiment.json", f"{arm}_instance_*")):
            fleet = json.load(open(f"{d}/fleet.json"))
            wt = {q["parcel_id"]: wbycell.get(tuple(q["spawn_cell"]), 1.0) for q in fleet["parcels"]}
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            for a in ags:
                cur = tuple(a["start_cell"])
                total = 0.0
                acts = sorted((x["time"]["step"], x["type"], tuple(x["cell"]), x.get("parcel_id")) for x in a["actions"] if x["type"] in ("pick", "drop"))
                pick = None
                for t, kind, cell, pid in acts:
                    if kind == "pick":
                        total += phys.energy_cost(distance=man(cur, cell), weight=0.0)
                        pick, cur = (cell, wt.get(pid, 1.0)), cell
                    elif kind == "drop" and pick:
                        w = pick[1]
                        total += phys.energy_cost(distance=man(cur, cell), weight=w) + 2.0 * phys.pick_drop_cost(w)
                        cur, pick = cell, None
                actual = float(a.get("energy_consumed") or 0.0)
                if actual > 0:
                    dedu.append(total)
                    act.append(actual)
                    ratio.append(total / actual)
    print(f"{cfg} {arm}: {len(ratio)} drones; deduced / actual energy: mean {st.mean(ratio):.3f}, sd {st.pstdev(ratio):.3f}, min {min(ratio):.2f}, max {max(ratio):.2f}; "
          f"correlation {np.corrcoef(dedu, act)[0, 1]:.3f}; mean deduced {st.mean(dedu):.1f}, mean actual {st.mean(act):.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
