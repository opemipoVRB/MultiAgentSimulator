# analysis/revision/marl_ablation.py
"""
How much the trained MAPPO policies (seeds 0-2) rely on each feature group: each group replaced by an uninformative value at run time (experiments/marl/ablate.sh),
makespan, time to last delivery and tail against the intact policy on the same replicates (paired, noisy sensing). Mean over the three seeds of the per-seed mean ratio.
Usage: python -m analysis.revision.marl_ablation
"""
import collections
import glob
import json
import re
import statistics as st

BASE = "experiments/marl/evaluation_mappo"
GROUPS = ["age", "peer_proximity", "peers", "peer_seen", "peer_start", "parcel_position", "distance", "energy", "weight", "step", "sensing_flags"]


def load(root, arm_pattern):
    out = {}
    for ep in sorted(glob.glob(f"{root}/*/experiment_*/experiment.json")):
        cfg = ep.split("/")[-3]
        exp = json.load(open(ep))
        for r in exp["results"]:
            name = r.get("strategy_name") or r.get("name")
            arm = re.sub(r"_instance_\d+$", "", name)
            if not re.fullmatch(arm_pattern, arm):
                continue
            ags = json.load(open(ep.replace("experiment.json", f"{name}/agents.json")))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            steps = [t["time"]["step"] for a in ags for t in a.get("tasks", [])]
            m = r["metrics"]
            out[(cfg, exp["seed"], arm)] = {"makespan": m["mission_makespan_steps"], "last": max(steps) if steps else 0,
                                           "done": m["tasks_completed"] == len(exp["parcels"])}
    return out


def main():
    base = load(BASE, r"marl_ippo_s[012]_noisy")
    print(f"{'ablated feature group':22s} {'makespan x':>10s} {'last delivery x':>16s} {'tail (steps)':>13s} {'all delivered':>14s}   per config (A8_P24 / A12_P24 makespan x)")
    ref_tail = {c: st.mean(v["makespan"] - v["last"] for k, v in base.items() if k[0] == c) for c in ("A8_P24", "A12_P24")}
    print(f"{'(intact policy)':22s} {1.0:10.3f} {1.0:16.3f} {st.mean(ref_tail.values()):13.0f} {'':>14s}   tails: {{ {', '.join(f'{c} {t:.0f}' for c, t in ref_tail.items())} }}")
    rows = []
    for g in GROUPS:
        ab = load(f"experiments/marl/ablation/{g}", r"marl_ippo_s[012]_noisy")
        keys = [k for k in ab if k in base]
        if not keys:
            continue
        ratio = st.mean(ab[k]["makespan"] / base[k]["makespan"] for k in keys)
        lratio = st.mean(ab[k]["last"] / max(1, base[k]["last"]) for k in keys)
        tail = st.mean(ab[k]["makespan"] - ab[k]["last"] for k in keys)
        per = {c: st.mean(ab[k]["makespan"] / base[k]["makespan"] for k in keys if k[0] == c) for c in ("A8_P24", "A12_P24")}
        done = sum(ab[k]["done"] for k in keys)
        rows.append((ratio, g, lratio, tail, done, len(keys), per))
    for ratio, g, lratio, tail, done, n, per in sorted(rows, reverse=True):
        print(f"{g:22s} {ratio:10.3f} {lratio:16.3f} {tail:13.0f} {done:>8d}/{n:<5d}   {per['A8_P24']:.3f} / {per['A12_P24']:.3f}")


if __name__ == "__main__":
    main()
