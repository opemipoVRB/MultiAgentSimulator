# analysis/revision/everything_eval.py
"""
The combined batch (experiments/peer_model/everything.sh): station memory, predicted station and personalities together. Seeds: 'fresh' = replicates 100-115, 'eval' = 0-15.
1. Makespan by awareness level, homogeneous fleet and mixed-personality fleet, noisy and full sensing: matched complete-case, ratio to the level without awareness (a0),
   with the baselines' absolute makespan for reference.
2. Completion counts (runs where not every parcel was delivered).
3. Peer model against the truth in the mixed fleet, with each peer's declared personality (pmm) and with peers assumed like the drone (pmmnp):
   position error against the last sensed position (beyond 30 steps), area under the ROC curve for 'parcel already gone', calibration of the gone share.
Usage: python -m analysis.revision.everything_eval <batch dir>
"""
import collections
import math
import statistics as st
import sys

import numpy as np

from analysis.revision.belief_error import AGE_BINS, auc, bucket
from analysis.revision.marl_evaluation import load
from analysis.revision.peer_model_eval import collect

CFGS = ("A4_P36", "A8_P24", "A12_P24")
BASELINES = ("naive_decentralized", "decentralized_auction", "cbba_full", "cbba_single_task")


def split(rec):
    out = {}
    for cfg in CFGS:
        seeds = sorted(k[1] for k in rec if k[0] == cfg)
        out[cfg] = {"eval": set(seeds[:16]), "fresh": set(seeds[16:])}
    return out


def ratio_table(rec, sp, title, arms, subset):
    print(f"\n=== {title} | seeds: {subset}")
    for cfg in CFGS:
        keys = [k for k in rec if k[0] == cfg and k[1] in sp[cfg][subset] and all(a in rec[k] for a in arms)]
        done = [k for k in keys if all(rec[k][a]["delivered"] == rec[k][a]["parcels"] for a in arms)]
        if not done:
            print(f"  {cfg:8s} no complete replicates ({len(keys)} run)")
            continue
        cells = []
        for a in arms:
            m = st.mean(rec[k][a]["makespan"] for k in done)
            t = m - st.mean(rec[k][a]["last"] for k in done)
            name = a.replace("sppdec_", "").replace("aw_", "").replace("noisy_", "").replace("full_", "")
            if a == arms[0]:
                cells.append(f"{name} {m:5.0f} (tail {t:4.0f})")
                continue
            d = [rec[k][a]["makespan"] / rec[k][arms[0]]["makespan"] for k in done]
            hw = 2.13 * st.stdev(d) / math.sqrt(len(d)) if len(d) > 2 else float("nan")
            cells.append(f"{name} {m:5.0f} (tail {t:4.0f}) x{st.mean(d):.3f}+-{hw:.3f}")
        print(f"  {cfg:8s} n={len(done)}/{len(keys)} | " + " | ".join(cells))


def peer_metrics(root, arm, sp_cfg_filter=None):
    peers, parcels = collect(root, arm)
    out = {}
    for cfg in CFGS:
        pr = [r for r in peers if r[0] == cfg and r[1] > 30 and not r[2]]
        rows = [r for r in parcels if r[0] == cfg and r[1] > 0]
        age = [r[1] for r in rows]
        gone = [r[3] for r in rows]
        base = auc(age, gone)
        both = auc([x / 2000.0 + 1.0 * r[2] for x, r in zip(age, rows)], gone)
        cal = [(np.mean([r[2] for r in rows if bucket(r[1]) == k]), np.mean([r[3] for r in rows if bucket(r[1]) == k]))
               for k in range(len(AGE_BINS)) if sum(bucket(r[1]) == k for r in rows) >= 30]
        out[cfg] = (len(pr), np.mean([r[3] for r in pr]) / np.mean([r[4] for r in pr]) if pr else float("nan"), both - base,
                    max(abs(p - o) for p, o in cal) if cal else float("nan"))
    return out


def main(root):
    rec = load(root)
    sp = split(rec)
    for subset in ("fresh", "eval"):
        for tag in ("noisy", "full"):
            ratio_table(rec, sp, f"homogeneous fleet, {tag} sensing", [f"aw_{tag}_a0", f"aw_{tag}_a1", f"aw_{tag}_a2", f"aw_{tag}_a5", f"sm_{tag}"], subset)
            ratio_table(rec, sp, f"mixed personalities, {tag} sensing", [f"mix_{tag}_a0", f"mix_{tag}_a2", f"mix_{tag}_a5", f"mix_{tag}_sm", f"mix_{tag}_smnp"], subset)
    print("\n=== baselines (original form, absolute makespan, complete-case among the four), noisy | full")
    for subset in ("fresh", "eval"):
        for cfg in CFGS:
            line = []
            for tag in ("noisy", "full"):
                arms = [f"{n}_aw_{tag}_a0" if n != "sppdec" else f"aw_{tag}_a0" for n in BASELINES] + [f"aw_{tag}_a0", f"mix_{tag}_a0"]
                keys = [k for k in rec if k[0] == cfg and k[1] in sp[cfg][subset] and all(a in rec[k] for a in arms)]
                done = [k for k in keys if all(rec[k][a]["delivered"] == rec[k][a]["parcels"] for a in arms)]
                if done:
                    line.append(f"{tag}: n={len(done)}/{len(keys)} " + " ".join(f"{a.replace('_aw_'+tag+'_a0','').replace('aw_'+tag+'_a0','SPPDEC plain').replace('mix_'+tag+'_a0','SPPDEC mixed')}={st.mean(rec[k][a]['makespan'] for k in done):.0f}" for a in arms))
            print(f"  {subset:5s} {cfg:8s} " + " | ".join(line))
    inc = collections.Counter(arm for v in rec.values() for arm, r in v.items() if r["delivered"] != r["parcels"])
    print("\nruns without every parcel delivered, by arm:", dict(inc) or "none")
    print("\n=== peer model in the mixed fleet, noisy sensing: (peer beliefs beyond 30 steps, position error model/last sensed, gone-AUC gain over age, worst calibration gap)")
    for arm, label in (("sppdec_mix_noisy_pmm", "each peer's declared personality"), ("sppdec_mix_noisy_pmmnp", "peers assumed like the drone")):
        m = peer_metrics(root, arm)
        print(f"  {label}: " + " | ".join(f"{c}: n={v[0]} pos {v[1]:.2f} auc {v[2]:+.3f} cal {v[3]:.2f}" for c, v in m.items()))
    for arm, label in (("sppdec_pmm_noisy", "homogeneous fleet")):
        pass
    m = peer_metrics(root, "sppdec_pmm_noisy")
    print("  homogeneous fleet: " + " | ".join(f"{c}: n={v[0]} pos {v[1]:.2f} auc {v[2]:+.3f} cal {v[3]:.2f}" for c, v in m.items()))


if __name__ == "__main__":
    main(sys.argv[1])
