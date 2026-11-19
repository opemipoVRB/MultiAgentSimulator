# analysis/revision/station_prediction_eval.py
"""
How well does the drone know the state of the station? At each logged decision of the shadow arm (`pmm_<sensing>`): the number of occupied station cells the drone
remembers (`occ`), the count the peer model predicts (`occ_pred`: memory plus the drops of simulated peers and of parcels known to be flying to the station) and the
count from the old rate extrapolation (`occ_rate`), against the true count (drops in the mission log up to that step). Mean absolute error and signed error (positive:
over-counts) by how long ago the drone last looked at the station, and the peer-model criteria of stage A (design section 8) on the same data.
Usage: python -m analysis.revision.station_prediction_eval <batch dir> <noisy|full> [replicate indices seed filter: fresh|eval|all]
"""
import glob
import json
import sys

import numpy as np

from analysis.revision.peer_model_eval import collect
from analysis.revision.belief_error import auc, bucket, AGE_BINS


def rows(root, arm, seeds):
    out = []
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        cfg = ep.split("/")[-3]
        exp = json.load(open(ep))
        rep_seed = exp.get("seed")
        for d in glob.glob(ep.replace("experiment.json", f"{arm}_instance_*")):
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            drops = sorted((a.get("time") or {}).get("step", a.get("step")) for g in ags for a in g["actions"] if a["type"] == "drop")
            for g in ags:
                for dec in g.get("slm_decisions", []):
                    m = (dec.get("proposals") or {}).get("measured")
                    pm = m.get("pm") if m else None
                    if not pm or pm.get("occ_pred") is None:
                        continue
                    t = dec["time"]["step"]
                    true = sum(x <= t for x in drops)
                    out.append((cfg, rep_seed, t, true, pm["occ"], pm["occ_pred"], pm.get("occ_rate")))
    return out


def main(root, tag, which="all"):
    r = rows(root, f"sppdec_pmm_{tag}", which)
    print(f"{len(r)} decisions with a prediction")
    for cfg in ["ALL"] + sorted({x[0] for x in r}):
        sel = [x for x in r if cfg == "ALL" or x[0] == cfg]
        e_mem = np.array([x[4] - x[3] for x in sel], float)
        e_pred = np.array([x[5] - x[3] for x in sel], float)
        e_rate = np.array([(x[6] if x[6] is not None else x[4]) - x[3] for x in sel], float)
        print(f"{cfg:8s} n={len(sel):5d} | mean abs error, occupied cells: memory only {np.abs(e_mem).mean():.2f} | rate extrapolation {np.abs(e_rate).mean():.2f} | "
              f"predicted from flights {np.abs(e_pred).mean():.2f}  (signed: memory {e_mem.mean():+.2f}, rate {e_rate.mean():+.2f}, flights {e_pred.mean():+.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "all")
