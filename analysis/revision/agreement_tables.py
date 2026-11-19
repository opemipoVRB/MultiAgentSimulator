# analysis/revision/agreement_tables.py
"""
Direct agreement between SPPDEC drones, from the full bundle table each drone logs at every decision (`proposals["tables"]`).
For two drones i and j that decide at the same step s, compare the table of i with the table of j:
  whole table equal (every drone's bundle, in order)   |   first task of every drone equal   |   i's bundle for j equals j's own bundle for itself.
Decisions at step 1 (identical inputs) and later steps are reported separately. Grouped by sensing condition (arm suffix).
Usage: python -m analysis.revision.agreement_tables
"""
import glob, json, os, collections, re
ROOT = "experiments/peer_model/agreement_check"

def tables(path):
    out = collections.defaultdict(dict)       # step -> agent -> table
    for a in json.load(open(os.path.join(path, "agents.json"))):
        for s in a.get("slm_decisions", []):
            t = (s.get("proposals") or {}).get("tables")
            if t:
                out[s["time"]["step"]][a["agent_id"]] = next(iter(t.values()))
    return out

def main():
    tot = collections.defaultdict(lambda: collections.Counter())
    for d in sorted(glob.glob(f"{ROOT}/A*_P*/experiment_*/sppdec_*_instance_*")):
        cond = re.sub(r".*nogamble_limited_?", "", os.path.basename(d).rsplit("_instance_", 1)[0]) or "exact"
        for step, by in tables(d).items():
            ids = sorted(by)
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    i, j = ids[x], ids[y]
                    ti, tj = by[i], by[j]
                    k = (cond, "start" if step <= 1 else "later")
                    c = tot[k]; c["pairs"] += 1
                    c["whole"] += ti == tj
                    c["first"] += {a: (b[:1]) for a, b in ti.items()} == {a: (b[:1]) for a, b in tj.items()}
                    c["own"] += ti.get(j) == tj.get(j) and tj.get(i) == ti.get(i)
    print(f"{'condition':8s} {'when':6s} {'pairs':>6s} {'whole table':>12s} {'first tasks':>12s} {'own bundles':>12s}")
    for (cond, when), c in sorted(tot.items(), key=lambda kv: (["exact","l50","n1","n2","n2l50"].index(kv[0][0]) if kv[0][0] in ["exact","l50","n1","n2","n2l50"] else 9, kv[0][1])):
        p = c["pairs"]
        print(f"{cond:8s} {when:6s} {p:6d} {100*c['whole']/p:11.1f}% {100*c['first']/p:11.1f}% {100*c['own']/p:11.1f}%")

if __name__ == "__main__":
    main()
