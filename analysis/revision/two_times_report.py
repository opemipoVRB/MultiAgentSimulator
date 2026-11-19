# analysis/revision/two_times_report.py
"""
Completion time (step of the last delivery) and fleet-registered time (every drone parked: the recorded
mission end), per arm, from a batch plus its `delivery_times.py --standdown` CSV.

Usage:
    python -m analysis.revision.two_times_report --root experiments/revision_fixed_dt/local_tracking_final \
        --csv .ignore/sandbox/main_results/delivery_times_final.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.revision.revision_stats import load_sweep, LABELS, regime_of
import builtins as _b
from analysis.revision.names import pretty as _pretty


def print(*args, **kwargs):  # printed names use the plain scheme of arm_names.py
    _b.print(*[_pretty(str(a)) for a in args], **kwargs)


MODELS = [("Greedy + tracking", "Always-G + obs", "S: Always-G + observation"),
          ("CBBA one task + tracking", "CBBA shortcut + obs", "S: CBBA shortcut + observation"),
          ("Hungarian + tracking + memory", "Hungarian + obs + mem", "S: Hungarian shared + observation + memory"),
          ("CBBA 3 tasks + tracking + memory", "CBBA protocol + obs + mem", "S: CBBA protocol (bundle 3) + observation + memory")]
SETTINGS = ["R=12", "R=8", "R=4", "R=8, loss 30%", "R=8, noise 1"]


def ratio(m, ref, arm, col):
    a = m[m.strategy == ref].set_index(["config", "instance"])[col]
    b = m[m.strategy == arm].set_index(["config", "instance"])[col]
    j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    lr = np.log(j.b / j.a)
    rng = np.random.default_rng(1)
    bs = [rng.choice(lr.values, len(lr)).mean() for _ in range(1500)]
    return float(np.exp(lr.mean())), float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5))), len(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    df = load_sweep(Path(a.root))
    d = pd.read_csv(a.csv)
    d["strategy"] = d.family.map(LABELS)
    m = df.merge(d, on=["config", "instance", "strategy"])
    m = m[m.config != "A2_P36"].copy()
    m["regime"] = m.c.map(regime_of)
    m["mission_end"] = m.makespan
    m["registered_gap"] = m.standdown_last - m.last_delivery
    print("runs", len(m), "| statuses", m.status.value_counts().to_dict(), "| runs with every drone stood down", int(m.standdown_last.notna().sum()))
    print("\n== Completion time (last delivery), local arm / same model seeing everything ==")
    rows = []
    for name, short, unl in MODELS:
        rows.append([name] + [f"{ratio(m, unl, f'L: {short}, {t}', 'last_delivery')[0]:.3f}" for t in SETTINGS])
    print(pd.DataFrame(rows, columns=["model"] + SETTINGS).to_string(index=False))
    print("\n== Fleet-registered time (all parked), local arm / same model seeing everything ==")
    rows = []
    for name, short, unl in MODELS:
        rows.append([name] + [f"{ratio(m, unl, f'L: {short}, {t}', 'mission_end')[0]:.3f}" for t in SETTINGS])
    print(pd.DataFrame(rows, columns=["model"] + SETTINGS).to_string(index=False))
    print("\n== Time the last drone stood down (decided it was done), local arm / same model seeing everything ==")
    rows = []
    for name, short, unl in MODELS:
        rows.append([name] + [f"{ratio(m, unl, f'L: {short}, {t}', 'standdown_last')[0]:.3f}" for t in SETTINGS])
    print(pd.DataFrame(rows, columns=["model"] + SETTINGS).to_string(index=False))
    print("\n== Completion time, local arm / plain greedy seeing everything ==")
    rows = []
    for name, short, unl in MODELS:
        rows.append([name] + [f"{ratio(m, 'S: Always-G', f'L: {short}, {t}', 'last_delivery')[0]:.3f}" for t in SETTINGS])
    print(pd.DataFrame(rows, columns=["model"] + SETTINGS).to_string(index=False))
    print("\n== Same local setting: model / greedy with the same local view (completion time) ==")
    for t in SETTINGS:
        g = f"L: Always-G + obs, {t}"
        print(f"  {t:14s}", " | ".join(f"{n}: {ratio(m, g, f'L: {s}, {t}', 'last_delivery')[0]:.3f}" for n, s, _ in MODELS[1:]))
    print("\n== References (completion time, / plain greedy seeing everything) ==")
    for ref in ["CBBA-ST", "CBBA-B", "C-Hung.", "Naive (RG)", "S: Always-G + observation", "S: CBBA shortcut + observation", "S: CBBA protocol (bundle 3) + observation + memory"]:
        r = ratio(m, "S: Always-G", ref, "last_delivery")
        print(f"  {ref:52s} {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]")
    print("\n== Stand-down, CBBA shortcut arms: median steps (first / median / last drone to stand down, and last minus last delivery) ==")
    for arm in ["S: CBBA shortcut + observation"] + [f"L: CBBA shortcut + obs, {t}" for t in SETTINGS]:
        g = m[m.strategy == arm]
        print(f"  {arm:44s} first {g.standdown_first.median():6.0f} | median {g.standdown_median.median():6.0f} | last {g.standdown_last.median():6.0f} | lag {g.registered_gap.median():6.0f} | parked before last delivery (mean drones) {g.early_standdowns.mean():4.1f}")
    print("\n== CBBA protocol + obs + mem (R=8) against CBBA-ST by regime, completion time (not matched information) ==")
    for reg in ["task-rich (c<0.3)", "balanced (0.3<=c<1)", "agent-surplus (c>=1)"]:
        mm = m[m.regime == reg]
        r = ratio(mm, "CBBA-ST", "L: CBBA protocol + obs + mem, R=8", "last_delivery")
        print(f"  {reg:22s} {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] n={r[3]}")


if __name__ == "__main__":
    main()
