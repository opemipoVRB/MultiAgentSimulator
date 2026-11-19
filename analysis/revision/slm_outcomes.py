# analysis/revision/slm_outcomes.py
"""
How SLM-arbitrated decisions were reached in a batch: which gate or rule decided, how often the model
was consulted, its status (success / failed), attempts used, and how often the decision guard replaced
its answer. Reads the per-agent `slm_decisions` records.

Usage:
    python -m analysis.revision.slm_outcomes --root experiments/revision_fixed_dt/main_batch --arm sppdec_judge_model
"""
import argparse
import collections
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", nargs="+", required=True, help="strategy directory prefix(es), e.g. sppdec_judge_model")
    a = ap.parse_args()
    for arm in a.arm:
        c = collections.Counter()
        files = glob.glob(f"{a.root}/*/experiment_*/{arm}_instance_*/agents.json")
        for f in files:
            for ag in json.load(open(f)):
                for e in ag.get("slm_decisions", []):
                    gate = e.get("gate") or ""
                    consulted = gate.startswith("slm")
                    c["decisions"] += 1
                    c["consulted"] += consulted
                    c["rule/gate"] += (not consulted)
                    st = e.get("status", "success")
                    c[f"status={st}"] += 1
                    if consulted:
                        ch = e.get("check") or {}
                        c[f"policy={e.get('policy')}"] += 1
                        c["guard_overrides"] += bool(ch.get("overridden_to"))
                        c["guard_violations"] += bool(ch.get("violations"))
                        if e.get("attempts"):
                            c[f"attempts={e['attempts']}"] += 1
        n = c["decisions"] or 1
        print(f"\n{arm}: {len(files)} runs, {c['decisions']} decisions")
        print(f"  consulted the model: {c['consulted']} ({100 * c['consulted'] / n:.1f}%), rules or gates: {c['rule/gate']}")
        for k in sorted(c):
            if k.startswith(("status=", "attempts=", "policy=")) or k.startswith("guard"):
                print(f"  {k}: {c[k]}")


if __name__ == "__main__":
    main()
