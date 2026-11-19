# analysis/revision/assignment_agreement.py
"""
Do the drones of a mission reach the same assignment? Read from the logged decisions of the SPPDEC arm.
At each decision a drone logs the first task it expects each peer to take (`assumed`) and the task it takes itself.
Measure A (start): at step 1 every drone has identical inputs. Compare drone i's expected first task for drone j with the task j took.
Measure B (later): at a later decision of drone i, compare its expectation for j with j's latest own decision at or before that step
  (same step = "fresh"; earlier = "stale"). Disagreement is counted per ordered pair of drones.
Output: data/batch_summaries/assignment_agreement.csv
"""
import glob, json, os, re, sys, csv, collections
ROOT = "experiments/peer_model/fresh_grid_realistic"
ARM = "sppdec_consensus_rule_nogamble_limited_instance_*"
OUT = "data/batch_summaries/assignment_agreement.csv"

def mission(path):
    agents = json.load(open(os.path.join(path, "agents.json")))
    own = {}     # agent -> [(step, parcel)]
    belief = {}  # agent -> [(step, {peer: parcel})]
    for a in agents:
        aid = a["agent_id"]
        own[aid] = [(d["time"]["step"], d.get("decided_parcel")) for d in a.get("decision_actions", []) if d.get("decided_parcel")]
        belief[aid] = []
        for s in a.get("slm_decisions", []):
            asm = (s.get("proposals") or {}).get("assumed") or {}
            tab = next(iter(asm.values()), None) if asm else None
            if tab:
                belief[aid].append((s["time"]["step"], tab))
    return own, belief

def main():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/A*_P*/experiment_*/{ARM}")):
        cfg = d.split("/")[-3]
        own, belief = mission(d)
        n = len(own)
        st = dict(pairs0=0, agree0=0, fresh=0, fresh_ok=0, stale=0, stale_ok=0)
        for i, bl in belief.items():
            for step, tab in bl:
                for j, p in tab.items():
                    if j not in own or not own[j]:
                        continue
                    if step <= 1:
                        j0 = [x for x in own[j] if x[0] <= 1]
                        if j0:
                            st["pairs0"] += 1; st["agree0"] += (j0[-1][1] == p)
                    else:
                        prior = [x for x in own[j] if x[0] <= step]
                        if not prior:
                            continue
                        k = "fresh" if prior[-1][0] == step else "stale"
                        st[k] += 1; st[k + "_ok"] += (prior[-1][1] == p)
        rows.append(dict(cfg=cfg, agents=n, mission=d.split("/")[-1], **st))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def tot(rs, a, b):
        x = sum(r[a] for r in rs); y = sum(r[b] for r in rs)
        return f"{y}/{x} = {100*y/x:.1f}%" if x else "n/a"
    print("missions", len(rows))
    for name, rs in [("all", rows), ("N<=4", [r for r in rows if r["agents"] <= 4]), ("N>4", [r for r in rows if r["agents"] > 4])]:
        print(name, "| start:", tot(rs, "pairs0", "agree0"), "| fresh:", tot(rs, "fresh", "fresh_ok"), "| stale:", tot(rs, "stale", "stale_ok"))
    full0 = sum(1 for r in rows if r["pairs0"] and r["pairs0"] == r["agree0"])
    print("missions with full agreement at start:", full0, "of", sum(1 for r in rows if r["pairs0"]))
main()
