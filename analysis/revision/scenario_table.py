# analysis/revision/scenario_table.py
"""
Scenario-to-performance table: which reasoning model finishes fastest in which situation, from the
final local-observation batch. Situations are described only by what a drone can observe for itself:

  sensing class   full     it sees every peer and positions are exact
                  partial  it sees only some peers (radius or random loss), positions exact
                  noisy    observed positions jitter (about one cell)
  regime          task-rich (agents per parcel < 0.3), balanced (0.3 to 1), agent-surplus (>= 1)

The table is context for any arbiter (rules, classifier, SLM). A leave-one-configuration-out version
is built too, so an arm run on a configuration can use a table that never saw it.

Usage:
    python -m analysis.revision.scenario_table --root experiments/revision_fixed_dt/local_tracking_final \
        --csv .ignore/sandbox/main_results/delivery_times_final.csv --out src/strategies/sppdec/scenario_table.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.revision.revision_stats import load_sweep, LABELS, regime_of
import builtins as _b
from analysis.revision.names import pretty as _pretty


def print(*args, **kwargs):  # printed names use the plain scheme of arm_names.py
    _b.print(*[_pretty(str(a)) for a in args], **kwargs)


MODELS = {
    "greedy": ("S: Always-G + observation", "L: Always-G + obs, {}"),
    "cbba": ("S: CBBA shortcut + observation", "L: CBBA shortcut + obs, {}"),
    "hungarian": ("S: Hungarian shared + observation + memory", "L: Hungarian + obs + mem, {}"),
    "cbba_bid_rounds": ("S: CBBA protocol (bundle 3) + observation + memory", "L: CBBA protocol + obs + mem, {}"),
}
SENSING = {"full": [None], "partial": ["R=12", "R=8", "R=4", "R=8, loss 30%"], "noisy": ["R=8, noise 1"]}
REGIMES = {"task-rich (c<0.3)": "task_rich", "balanced (0.3<=c<1)": "balanced", "agent-surplus (c>=1)": "agent_surplus"}


def build(m, exclude_config=None):
    cells = {}
    for sclass, tags in SENSING.items():
        for reg_label, reg in REGIMES.items():
            per_model = {}
            for k, (g, l) in MODELS.items():
                names = [g if t is None else l.format(t) for t in tags]
                sel = m[m.strategy.isin(names) & (m.regime == reg_label)]
                if exclude_config:
                    sel = sel[sel.config != exclude_config]
                per_model[k] = sel
            n = min(len(v) for v in per_model.values()) if per_model else 0
            if n == 0:
                continue
            gm = {k: float(np.exp(np.log(v.last_delivery).mean())) for k, v in per_model.items()}
            best = min(gm, key=gm.get)
            cells[f"{sclass}|{reg}"] = {
                "mean_completion_steps": {k: round(v, 1) for k, v in gm.items()},
                "relative_to_best": {k: round(v / gm[best], 3) for k, v in gm.items()},
                "best": best, "runs_per_model": int(n),
            }
    return cells


def describe(cells) -> str:
    lines = ["sensing class | regime | best | relative completion time (1.00 = best): Greedy / CBBA one task / Hungarian / CBBA 3 tasks"]
    for key, c in cells.items():
        r = c["relative_to_best"]
        lines.append(f"{key.replace('|', ' | ')} | { {'greedy':'Greedy (nearest task first)','cbba':'CBBA, one task (bids guessed)','hungarian':'Hungarian (optimal assignment)','cbba_bid_rounds':'CBBA, 3 tasks (bids guessed)'}[c['best']] } | {r['greedy']:.2f} / {r['cbba']:.2f} / {r['hungarian']:.2f} / {r['cbba_bid_rounds']:.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = load_sweep(Path(a.root))
    d = pd.read_csv(a.csv)
    d["strategy"] = d.family.map(LABELS)
    m = df.merge(d[["config", "instance", "strategy", "last_delivery"]], on=["config", "instance", "strategy"])
    m = m[m.config != "A2_P36"].copy()
    m["regime"] = m.c.map(regime_of)
    out = {"all": build(m), "leave_one_config_out": {c: build(m, c) for c in sorted(m.config.unique())}}
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(describe(out["all"]))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
