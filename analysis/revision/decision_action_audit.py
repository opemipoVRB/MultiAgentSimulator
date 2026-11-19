# analysis/revision/decision_action_audit.py
"""
Does the drone act on what was decided? (revision: decision must match action)

Two audits over batch outputs, computed for every strategy with the same code:

1. Pickups vs plans (all strategies). The controller records, at each pickup stop, the
   parcel cell the current plan names, and flags stops on a parcel the plan does not name
   (`unplanned_parcel_present`, or `unplanned_parcel_ignored` when CONTROLLER_PICK_POLICY=enforce).
   Also counted per strategy: picked, pick_failed, target_gone (someone else took it first).

2. Decisions vs directives vs pickups (SPPDEC). Each arbitration-based decision is linked to the
   directive it produced (`decision_actions`), and each directive to the next pickup by the
   same drone: acted / target taken by another drone / different parcel picked (a mismatch).

Usage:
  python -m analysis.revision.decision_action_audit --root <batch dir> [--root <batch dir> ...]
"""
import argparse
import glob
import json
import re
from src.strategies.sppdec.legacy_ids import modernise
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from analysis.revision.revision_stats import LABELS


def family_of(strategy_dir: str) -> str:
    fam = modernise(re.sub(r"_instance_\d+$", "", strategy_dir))
    return LABELS.get(fam, fam)


def audit_run(agents: list):
    """Counters for one run's agents.json."""
    pick_outcomes = Counter()
    dec = Counter()
    for a in agents:
        for o in a.get("plan_outcomes", []):
            pick_outcomes[o["outcome"]] += 1

        decisions = [d for d in a.get("decision_actions", [])]
        picks = [x for x in a.get("actions", []) if x.get("type") == "pick"]
        outcomes = a.get("plan_outcomes", [])
        for i, d in enumerate(decisions):
            dec["decisions"] += 1
            if d["reason_not_acted"]:
                dec[f"not_acted: {d['reason_not_acted']}"] += 1
                continue
            dec["directives"] += 1
            t0 = d["time"]["step"]
            t1 = decisions[i + 1]["time"]["step"] if i + 1 < len(decisions) else float("inf")
            # Decisions are stamped one step behind the controller's own events, so the previous
            # plan's outcome can carry the same step as this decision. Match this decision's plan
            # by the parcel cell it named, not by time alone.
            nxt = next((o for o in outcomes
                        if t0 <= o["time"]["step"] <= t1
                        and o["outcome"] in ("picked", "pick_failed", "target_gone")
                        and o["planned_cell"] == d["decided_cell"]), None)
            if nxt is None:
                dec["unresolved (run ended or drone parked)"] += 1
            elif nxt["outcome"] == "picked":
                dec["acted: picked the decided parcel" if nxt["cell"] == d["decided_cell"]
                    else "MISMATCH: picked a different parcel"] += 1
            elif nxt["outcome"] == "target_gone":
                dec["target taken by another drone before arrival"] += 1
            else:
                dec["pick failed"] += 1
    return pick_outcomes, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="batch directory containing A*_P*/experiment_*/")
    ap.add_argument("--out", default=None, help="write CSVs here")
    args = ap.parse_args()

    pick_rows, dec_rows = [], []
    for root in args.root:
        for f in glob.glob(str(Path(root) / "A*_P*" / "experiment_*" / "*_instance_*" / "agents.json")):
            strat = family_of(Path(f).parent.name)
            cfg = Path(f).parts[-4]
            agents = json.load(open(f))
            p, d = audit_run(agents)
            pick_rows.append({"strategy": strat, "config": cfg, **p})
            if d:
                dec_rows.append({"strategy": strat, "config": cfg, **d})

    picks = pd.DataFrame(pick_rows).fillna(0)
    cols = [c for c in ("picked", "pick_failed", "target_gone", "unplanned_parcel_present",
                        "unplanned_parcel_ignored") if c in picks]
    by_strategy = picks.groupby("strategy")[cols].sum()
    by_strategy["runs"] = picks.groupby("strategy").size()
    if "target_gone" in by_strategy:
        # share of planned pickups that found the parcel already taken (contested targeting)
        by_strategy["planned pickups that found the target gone"] = (
            by_strategy["target_gone"] / (by_strategy["target_gone"] + by_strategy["picked"])).map("{:.0%}".format)
    if "unplanned_parcel_present" in by_strategy:
        by_strategy["unplanned per 100 pickups"] = (100 * by_strategy["unplanned_parcel_present"]
                                                   / by_strategy["picked"].clip(lower=1)).round(2)
    print("== 1. Pickups vs plans, all strategies ==")
    print(by_strategy.to_string())

    if dec_rows:
        dec = pd.DataFrame(dec_rows).fillna(0)
        dsum = dec.drop(columns=["config"]).groupby("strategy").sum()
        print("\n== 2. Decisions -> directives -> pickups (decisions with a directive) ==")
        print(dsum.T.to_string())
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        by_strategy.to_csv(out / "pickups_vs_plans.csv")
        if dec_rows:
            dsum.T.to_csv(out / "decisions_to_actions.csv")


if __name__ == "__main__":
    main()
