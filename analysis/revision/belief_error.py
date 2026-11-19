# analysis/revision/belief_error.py
"""
What SPPDEC believes about its peers and about the parcels it could take, against the truth, in every configuration (logging: `_measured_context`).
The truth is rebuilt from the mission logs: where every drone was at the decision step, and which parcels had been picked by then.

1. Peers: the error (grid steps, Manhattan) of the believed (extrapolated) position and of the last sensed position, by the steps since the peer was last sensed.
2. Parcels: how often a parcel the drone believes open is already gone, by the age of its information; whether a peer believed nearer than the drone predicts it, and how well
   each belief separates gone parcels from present ones (area under the ROC curve; 0.5 = no information).
Usage: python -m analysis.revision.belief_error <batch dir> [<batch dir> ...]
"""
import collections
import glob
import json
import re
import sys

import numpy as np

AGE_BINS = [(0, 0), (1, 30), (31, 100), (101, 300), (301, 10 ** 9)]


def auc(score, label):
    """Area under the ROC curve of `score` for the binary `label` (rank based)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bucket(age):
    for k, (lo, hi) in enumerate(AGE_BINS):
        if lo <= age <= hi:
            return k
    return len(AGE_BINS) - 1


def collect(roots):
    peer_rows, parcel_rows = [], []
    for root in roots:
        for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
            cfg = ep.split("/")[-3]
            exp = json.load(open(ep))
            for r in exp["results"]:
                name = r.get("strategy_name") or r.get("name")
                arm = re.sub(r"_instance_\d+$", "", name).replace("sppdec_", "")
                ags = json.load(open(ep.replace("experiment.json", f"{name}/agents.json")))
                ags = list(ags.values()) if isinstance(ags, dict) else ags
                cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
                picked = {}
                for a in ags:
                    for act in a["actions"]:
                        if act["type"] == "pick":
                            picked[act["parcel_id"]] = min(picked.get(act["parcel_id"], 10 ** 9), act["time"]["step"] if "time" in act else act["step"])
                for a in ags:
                    for d in a.get("slm_decisions", []):
                        m = (d.get("proposals") or {}).get("measured")
                        if not m or "peers" not in m:
                            continue
                        t = d["time"]["step"]
                        for pid, last, pred, since, vis in m["peers"]:
                            traj = cells.get(pid)
                            if not traj:
                                continue
                            true = traj[min(t, len(traj)) - 1]
                            peer_rows.append((cfg, arm, since, vis, abs(pred[0] - true[0]) + abs(pred[1] - true[1]),
                                              abs(last[0] - true[0]) + abs(last[1] - true[1])))
                        for tid, age, mine, near_pred, near_last in m["tasks"]:
                            gone = picked.get(tid, 10 ** 9) < t
                            parcel_rows.append((cfg, arm, age, mine, near_pred, near_last, gone))
    return peer_rows, parcel_rows


def main(roots):
    peers, parcels = collect(roots)
    print(f"{len(peers)} peer beliefs and {len(parcels)} parcel beliefs from {len({r[0] for r in parcels})} configs\n")
    print("1. Peer position error (grid steps) by steps since the peer was last sensed -- believed (extrapolated) vs last sensed position; all configs / per config")
    for cfg in ["ALL"] + sorted({r[0] for r in peers}):
        rows = [r for r in peers if cfg == "ALL" or r[0] == cfg]
        line = []
        for k, (lo, hi) in enumerate(AGE_BINS):
            sel = [r for r in rows if bucket(r[2]) == k and not (r[3] and k > 0)]
            if len(sel) >= 30:
                line.append(f"since {lo}-{hi if hi < 10 ** 8 else 'inf'}: n={len(sel):5d} believed {np.mean([r[4] for r in sel]):5.1f}  last-sensed {np.mean([r[5] for r in sel]):5.1f}")
        print(f"   {cfg:8s}", " | ".join(line))
    print("\n2. Parcels believed open: share already gone, by age of the information (all configs and per config)")
    for cfg in ["ALL"] + sorted({r[0] for r in parcels}):
        rows = [r for r in parcels if cfg == "ALL" or r[0] == cfg]
        line = []
        for k, (lo, hi) in enumerate(AGE_BINS):
            sel = [r for r in rows if bucket(r[2]) == k]
            if len(sel) >= 30:
                line.append(f"age {lo}-{hi if hi < 10 ** 8 else 'inf'}: gone {np.mean([r[6] for r in sel]):.2f} (n={len(sel)})")
        print(f"   {cfg:8s}", " | ".join(line))
    print("\n3. How well each belief separates gone from present parcels (AUC; 0.5 = no information), offered parcels with a known age > 0")
    for cfg in ["ALL"] + sorted({r[0] for r in parcels}):
        rows = [r for r in parcels if (cfg == "ALL" or r[0] == cfg) and r[2] > 0 and r[4] >= 0]
        if len(rows) < 100:
            continue
        gone = [r[6] for r in rows]
        age = [r[2] for r in rows]
        rel_pred = [r[3] - r[4] for r in rows]       # own distance minus the nearest peer's (believed position): large = a peer is nearer
        rel_last = [r[3] - r[5] for r in rows]
        both = [a / 2000.0 + 0.5 * (rp > 0) for a, rp in zip(age, rel_last)]
        print(f"   {cfg:8s} n={len(rows):6d} gone {np.mean(gone):.2f} | age {auc(age, gone):.3f} | peer nearer (believed pos.) {auc(rel_pred, gone):.3f} | "
              f"peer nearer (last sensed) {auc(rel_last, gone):.3f} | age + peer nearer {auc(both, gone):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
