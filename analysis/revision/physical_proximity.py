# analysis/revision/physical_proximity.py
"""
Physical proximity of drones, from the continuous positions of the trajectories: the number of (pair, step) instances closer than a radius, and the number of distinct
close-pair episodes (a pair that comes inside the radius and leaves it), per mission and strategy. A radius is in cells (100 px). The recorder's own "collision" counts
drones sharing a committed grid cell, which is not a distance (see change_log 2026-10-14).
Usage: python -m analysis.revision.physical_proximity [batch dir]
"""
import glob
import json
import statistics as st
import sys

import numpy as np

FAM = [("SPPDEC", "sppdec_consensus_rule_nogamble_limited"), ("Centralised", "centralized_cc_hungarian_hungarian"), ("CBBA single", "cbba_single_task_cbba_single"),
       ("CBBA bundle", "cbba_full_cbba_bundle"), ("Auction", "decentralized_auction_auction_baseline"), ("Naive", "naive_decentralized_reactive")]
RADII = (0.25, 0.5, 1.0)


def mission_proximity(agents_path):
    ags = json.load(open(agents_path)); ags = list(ags.values()) if isinstance(ags, dict) else ags
    T = min(len(g["trajectory"]) for g in ags)
    P = np.array([[g["trajectory"][t]["pos"] for g in ags] for t in range(T)], dtype=float) / 100.0      # T x n x 2, in cells
    n = P.shape[1]
    iu = np.triu_indices(n, 1)
    D = np.linalg.norm(P[:, iu[0], :] - P[:, iu[1], :], axis=2)                                        # T x pairs
    out = {}
    for r in RADII:
        close = D < r
        out[f"pairsteps_{r}"] = int(close.sum())
        out[f"episodes_{r}"] = int(((close[1:] & ~close[:-1]).sum()) + close[0].sum())
        out[f"steps_any_{r}"] = int(close.any(axis=1).sum())
    out["min_dist"] = float(D.min()) if D.size else 9.9
    return out


def main(root="experiments/peer_model/fresh_grid_proc"):
    rows = {}
    for ep in glob.glob(f"{root}/*/experiment_*/experiment.json"):
        cfg = ep.split("/")[-3]; base = ep.rsplit("/", 1)[0]; e = json.load(open(ep)); P = int(cfg.split("_P")[1])
        for nm, f in FAM:
            d = glob.glob(base + f"/{f}_instance_*")
            if not d:
                continue
            ags = json.load(open(d[0] + "/agents.json")); ags = list(ags.values()) if isinstance(ags, dict) else ags
            complete = sum(1 for g in ags for x in g["actions"] if x["type"] == "drop") == P
            rows.setdefault((cfg, e["provenance"]["replicate"]), {})[nm] = dict(complete=complete, **mission_proximity(d[0] + "/agents.json"))
    common = [k for k, v in rows.items() if len(v) == 6 and all(x["complete"] for x in v.values())]
    print(f"common set {len(common)} instances; mean per mission")
    print(f"{'':12s}" + "".join(f"{'close pair-steps <'+str(r):>22s}" for r in RADII) + "".join(f"{'episodes <'+str(r):>16s}" for r in RADII))
    res = {}
    for nm, _ in FAM:
        v = [rows[k][nm] for k in common]
        res[nm] = [st.mean(x[f"pairsteps_{r}"] for x in v) for r in RADII] + [st.mean(x[f"episodes_{r}"] for x in v) for r in RADII]
        print(f"{nm:12s}" + "".join(f"{res[nm][i]:22.1f}" for i in range(3)) + "".join(f"{res[nm][3 + i]:16.1f}" for i in range(3)))
    print("\nrank by close pair-steps within 0.5 cell (1 = fewest) and by episodes within 0.5 cell")
    for key, idx in (("pair-steps", 1), ("episodes", 4)):
        order = sorted(res, key=lambda n: res[n][idx]); print(f"  {key}: " + " < ".join(f"{n} ({res[n][idx]:.0f})" for n in order))
    print("\nmissions in which two drones came closer than 0.25 cell at any time:")
    for nm, _ in FAM:
        print(f"  {nm:12s} {sum(1 for k in common if rows[k][nm]['pairsteps_0.25'] > 0)}/{len(common)}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
