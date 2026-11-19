# analysis/revision/peer_model_eval.py
"""
Stage A of the peer model (shadow mode; design: .ignore/revision/marl_baseline_design.md section 8). The model's predictions are logged with each decision
(`proposals.measured.pm`); here they are compared with the truth rebuilt from the mission logs. Criteria fixed beforehand, on every config:
  (a) beyond 30 steps since a peer was sensed, the model's position error is at least 20% below the last sensed position's;
  (b) the area under the ROC curve for 'parcel already gone' with the model's gone flag added to the age is at least 0.03 above the age alone;
  (c) in every age bucket the predicted gone share is within 0.10 of the observed one.
Only decisions where the model had a prediction are used (not before the drone has seen three moves of its own). Also reports the control check: decisions
equal to the plain arm's (same makespan and energy).
Usage: python -m analysis.revision.peer_model_eval <batch dir> <noisy|full> [arm prefix: pm (default) | pmm] [control arm prefix: channel (default) | aw]
"""
import glob
import json
import re
import sys

import numpy as np

from analysis.revision.belief_error import AGE_BINS, auc, bucket
from analysis.revision.marl_evaluation import load


def collect(root, arm_name):
    peer_rows, parcel_rows = [], []
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        cfg = ep.split("/")[-3]
        for d in glob.glob(ep.replace("experiment.json", f"{arm_name}_instance_*")):
            ags = json.load(open(f"{d}/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            cells = {a["agent_id"]: [tuple(t["cell"]) for t in a["trajectory"]] for a in ags}
            picked = {}
            for a in ags:
                for act in a["actions"]:
                    if act["type"] == "pick":
                        picked[act["parcel_id"]] = min(picked.get(act["parcel_id"], 10 ** 9), (act.get("time") or {}).get("step", act.get("step")))
            for a in ags:
                for dec in a.get("slm_decisions", []):
                    m = (dec.get("proposals") or {}).get("measured")
                    if not m or not m.get("pm") or "peers" not in m["pm"]:
                        continue
                    t = dec["time"]["step"]
                    pm_pos = {p[0]: p[1] for p in m["pm"]["peers"]}
                    for pid, last, _pred, since, vis in m["peers"]:
                        traj = cells.get(pid)
                        if not traj or pid not in pm_pos:
                            continue
                        true = traj[min(t, len(traj)) - 1]
                        peer_rows.append((cfg, since, vis, abs(pm_pos[pid][0] - true[0]) + abs(pm_pos[pid][1] - true[1]),
                                          abs(last[0] - true[0]) + abs(last[1] - true[1])))
                    gone_pred = set(m["pm"]["gone"])
                    for tid, age, *_ in m["tasks"]:
                        parcel_rows.append((cfg, age, tid in gone_pred, picked.get(tid, 10 ** 9) < t))
    return peer_rows, parcel_rows


def main(root, tag, prefix="pm", control="channel"):
    arm = f"sppdec_{prefix}_{tag}"
    rec = load(root)
    same = tot = 0
    for k, v in rec.items():
        if f"{prefix}_{tag}" in v and f"{control}_{tag}" in v:
            tot += 1
            same += v[f"{prefix}_{tag}"]["makespan"] == v[f"{control}_{tag}"]["makespan"] and abs(v[f"{prefix}_{tag}"]["energy"] - v[f"{control}_{tag}"]["energy"]) < 1e-6
    print(f"control: shadow arm equals the plain arm in {same} of {tot} runs (makespan and energy)\n")
    peers, parcels = collect(root, arm)
    ok_all = True
    for cfg in sorted({r[0] for r in parcels}):
        pr = [r for r in peers if r[0] == cfg and r[1] > 30 and not r[2]]
        a_model = np.mean([r[3] for r in pr]) if pr else float("nan")
        a_last = np.mean([r[4] for r in pr]) if pr else float("nan")
        ca = len(pr) > 0 and a_model <= 0.8 * a_last
        rows = [r for r in parcels if r[0] == cfg and r[1] > 0]
        age = [r[1] for r in rows]
        gone = [r[3] for r in rows]
        base = auc(age, gone)
        both = auc([x / 2000.0 + 1.0 * r[2] for x, r in zip(age, rows)], gone)
        cb = both - base >= 0.03
        cal = []
        for k, (lo, hi) in enumerate(AGE_BINS):
            sel = [r for r in rows if bucket(r[1]) == k]
            if len(sel) >= 30:
                cal.append((lo, hi, np.mean([r[2] for r in sel]), np.mean([r[3] for r in sel]), len(sel)))
        cc = all(abs(p - o) <= 0.10 for _, _, p, o, _ in cal)
        ok_all &= ca and cb and cc
        print(f"== {cfg}: {len(pr)} peer beliefs beyond 30 steps, {len(rows)} parcel beliefs with an age")
        print(f"   (a) position error: model {a_model:.1f}, last sensed {a_last:.1f} cells (ratio {a_model / a_last if a_last else float('nan'):.2f}; need <= 0.80): {'PASS' if ca else 'FAIL'}")
        print(f"   (b) gone AUC: age alone {base:.3f}, age + model {both:.3f} (gain {both - base:+.3f}; need >= 0.03): {'PASS' if cb else 'FAIL'}")
        print("   (c) predicted vs observed gone share by age: " + " | ".join(f"{lo}-{hi if hi < 10 ** 8 else 'inf'}: {p:.2f} vs {o:.2f} (n={n})" for lo, hi, p, o, n in cal)
              + f"  -> {'PASS' if cc else 'FAIL'}")
    print(f"\nStage A on {tag} sensing: {'PASS on every config' if ok_all else 'does not pass on every config (stop and report)'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
