# analysis/revision/ledger_timing.py
"""
When does a drone learn that the job is done? For each run of an awareness arm: the step of the last delivery (the job is done), and for each drone the step of its
first `interrupt_ledger_complete` plan outcome (the step it believed the station full). Reports the delay after the last delivery (mean over drones that stopped),
the share of drones that stopped before the last delivery (early stops), and the share that never stopped by a belief (they found out by running out of tasks).
Usage: python -m analysis.revision.ledger_timing <batch dir> <config> <arm> [<arm> ...]
"""
import glob
import json
import re
import statistics as st
import sys


def main(root, cfg, arms):
    for arm in arms:
        delay, early, never, drones, runs = [], 0, 0, 0, 0
        for ep in sorted(glob.glob(f"{root}/{cfg}/experiment_*")):
            for d in glob.glob(f"{ep}/{arm}_instance_*"):
                ags = json.load(open(f"{d}/agents.json"))
                ags = list(ags.values()) if isinstance(ags, dict) else ags
                drops = [(a.get("time") or {}).get("step", a.get("step")) for g in ags for a in g["actions"] if a["type"] == "drop"]
                if not drops:
                    continue
                last = max(drops)
                runs += 1
                for g in ags:
                    drones += 1
                    steps = [(o.get("time") or {}).get("step") for o in g.get("plan_outcomes", []) if "ledger_complete" in str(o.get("outcome"))]
                    if not steps:
                        never += 1
                        continue
                    t = min(steps)
                    if t < last:
                        early += 1
                    else:
                        delay.append(t - last)
        print(f"{arm}: {runs} runs, {drones} drones; stopped by belief {drones - never} ({(drones - never) / drones:.0%}); of those before the last delivery {early}; "
              f"mean delay after the last delivery {st.mean(delay) if delay else float('nan'):.0f} steps (median {st.median(delay) if delay else float('nan'):.0f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
