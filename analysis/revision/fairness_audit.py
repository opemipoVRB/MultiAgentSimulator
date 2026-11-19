# analysis/revision/fairness_audit.py
"""
Are all strategies compared under the same conditions? Checked from the exported data.

  1. Same environment: for each (config, replicate), every strategy saw the same parcels
     (cells and weights) and the same drone start cells, across all batches.
  2. Same rules: code version, clock, grid, controller pick policy, prompt style and model
     per batch (provenance), so version differences between batches are visible.
  3. Complete design: every (config, replicate) has every strategy that is being compared.
  4. Same treatment of stalls: how each strategy's runs ended, and the longest stretch
     without any pick or drop (the stall rule ends a run after 3,600 steps of none).

Usage:
  python -m analysis.revision.fairness_audit --root experiments/revision_fixed_dt/fast --root ...
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from analysis.revision.revision_stats import LABELS

STALL_STEPS = 60 * 60


def label(strategy_dir: str) -> str:
    return LABELS.get(re.sub(r"_instance_\d+$", "", strategy_dir), strategy_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--compare", nargs="*", default=None, help="strategy labels that must all be present")
    args = ap.parse_args()

    env = defaultdict(set)          # (config, replicate) -> {environment fingerprint}
    starts = defaultdict(set)       # (config, replicate, agent) -> {start cell}
    prov_rows, present, ends, gaps = [], defaultdict(set), [], []

    for root in args.root:
        for exp_file in glob.glob(str(Path(root) / "A*_P*" / "experiment_*" / "experiment.json")):
            exp = json.load(open(exp_file))
            cfg = Path(exp_file).parts[-3]
            prov = exp.get("provenance") or {}
            rep = prov.get("replicate")
            if rep is None:
                continue
            key = (cfg, rep)
            fp = tuple((p["col"], p["row"], round(p.get("weight", -1), 9)) for p in exp["parcels"])
            env[key].add(fp)
            prov_rows.append({"batch": Path(root).name, "src": prov.get("src_sha256"),
                              "fixed_dt": prov.get("fixed_timestep"), "grid": str(prov.get("grid")),
                              "pick_policy": prov.get("controller_pick_policy", "(not recorded)"),
                              "prompt_style": prov.get("sppdec_prompt_style_default",
                                                       prov.get("sppdec_prompt_version_default", "(not recorded)")),
                              "model": prov.get("ollama_model")})
            for r in exp["results"]:
                strat = label(r["strategy_name"])
                present[key].add(strat)
                ends.append({"strategy": strat, "status": r.get("status")})
                agents_file = Path(exp_file).parent / r["strategy_name"] / "agents.json"
                if not agents_file.exists():
                    continue
                agents = json.load(open(agents_file))
                for a in agents:
                    starts[(cfg, rep, a["agent_id"])].add(tuple(a["start_cell"]))
                steps = sorted(x["time"]["step"] for a in agents for x in a.get("actions", [])
                               if x.get("type") in ("pick", "drop"))
                if len(steps) > 1:
                    gaps.append({"strategy": strat, "max_gap": max(b - a for a, b in zip(steps, steps[1:]))})

    print("== 1. Same environment across strategies and batches ==")
    bad_env = [k for k, v in env.items() if len(v) > 1]
    bad_start = [k for k, v in starts.items() if len(v) > 1]
    print(f"(config, replicate) pairs checked: {len(env)}; with differing parcels/weights: {len(bad_env)}"
          f"; drone start cells checked: {len(starts)}, differing: {len(bad_start)}")
    for k in bad_env[:5]:
        print("  parcel/weight mismatch:", k)

    print("\n== 2. Rules and code per batch ==")
    print(pd.DataFrame(prov_rows).drop_duplicates().to_string(index=False))

    print("\n== 3. Complete design ==")
    everything = sorted({s for v in present.values() for s in v})
    want = set(args.compare) if args.compare else set(everything)
    missing = [(k, sorted(want - v)) for k, v in sorted(present.items()) if want - v]
    print(f"strategies seen: {everything}")
    print(f"(config, replicate) pairs missing at least one compared strategy: {len(missing)} of {len(present)}")
    for k, m in missing[:8]:
        print("  ", k, "missing", m)

    print("\n== 4. How runs ended, and longest stretch without a pick/drop ==")
    e = pd.DataFrame(ends)
    print(pd.crosstab(e.strategy, e.status).to_string())
    g = pd.DataFrame(gaps).groupby("strategy").max_gap.agg(["median", "max"])
    g[f"runs with a gap >= {STALL_STEPS}"] = pd.DataFrame(gaps).groupby("strategy").max_gap.apply(
        lambda s: int((s >= STALL_STEPS).sum()))
    print(g.to_string())


if __name__ == "__main__":
    main()
