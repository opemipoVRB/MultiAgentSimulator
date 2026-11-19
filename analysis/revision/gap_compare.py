# analysis/revision/gap_compare.py
"""
Compare gap batches that differ in the environment, not in the arm (radio profiles, map sizes, dynamic arrival, obstacles).
For each batch, per strategy (all arms are the unstressed arm of that environment): missions completed, agents lost, mean messages, last delivery and energy on the missions the strategy completed in
both the batch and the reference batch (% later / % more than the reference), and SPPDEC's lead over the strategy inside the batch (over the missions both SPPDEC and the strategy completed).
A strategy that sends no message (SPPDEC, Naive, MARL) is read from the reference batch when the batch has no run of it (the radio profile cannot change it).
Usage: python -m analysis.revision.gap_compare <reference batch> <batch> [<batch> ...]
"""
import csv, sys
import numpy as np
from analysis.revision.gap_tables import BASES, NAMES, split_arm

def load(batch):
    d = {}
    for r in csv.DictReader(open(f"data/batch_summaries/gap_{batch}.csv")):
        s, t = split_arm(r["arm"])
        if s and t == "ref":
            d[(r["config"], r["replicate"], s)] = r
    return d

def ok(r):
    return r is not None and r["last_delivery"] not in ("", "None") and int(r["delivered"]) == int(r["parcels"])

def main():
    ref = load(sys.argv[1])
    for b in sys.argv[2:]:
        cur = load(b)
        missions = sorted({(c, r) for c, r, _ in ref})
        def get(s, m):
            return cur.get((m[0], m[1], s)) or (ref.get((m[0], m[1], s)) if s in ("S", "N", "M") and b.startswith("radio") else None)
        shorts = [s for s, _ in BASES if any(get(s, m) for m in missions)]
        keys = [m for m in missions if all(ok(get(s, m)) for s in shorts)]
        print(f"\n== {b} against {sys.argv[1]}: {len(keys)} missions completed by all of {len(shorts)} strategies")
        print(f"{'strategy':24s} {'done':>5s} {'/run':>5s} {'lost':>5s} {'msgs':>7s} {'last delivery':>14s} {'vs ref':>8s} {'energy vs ref':>14s} {'SPPDEC lead (pairwise)':>24s}")
        for s in shorts:
            rows = [get(s, m) for m in missions if get(s, m)]
            nd = sum(ok(r) for r in rows); lost = sum(int(r["dead"] or 0) for r in rows)
            msgs = np.mean([float(r["messages"]) for r in rows])
            both = [m for m in missions if ok(get(s, m)) and ok(ref.get((m[0], m[1], s)))]
            vs = en = ""
            if both:
                a = np.mean([float(get(s, m)["last_delivery"]) for m in both]); c = np.mean([float(ref[(m[0], m[1], s)]["last_delivery"]) for m in both])
                vs = f"{100 * (a / c - 1):+.1f}%"
                ea = np.mean([float(get(s, m)["energy"]) for m in both]); ec = np.mean([float(ref[(m[0], m[1], s)]["energy"]) for m in both])
                en = f"{100 * (ea / ec - 1):+.1f}%"
            mean = np.mean([float(get(s, m)["last_delivery"]) for m in keys]) if keys else float("nan")
            lead = ""
            pair = [m for m in missions if ok(get("S", m)) and ok(get(s, m))]      # pairwise: SPPDEC and this strategy both completed
            if s != "S" and pair and "S" in shorts:
                lead = f"{np.mean([100 * (1 - float(get('S', m)['last_delivery']) / float(get(s, m)['last_delivery'])) for m in pair]):+.1f}% (n={len(pair)})"
            print(f"{NAMES[s]:24s} {nd:5d} {len(rows):5d} {lost:5d} {msgs:7.1f} {mean:14.1f} {vs:>8s} {en:>14s} {lead:>24s}")

if __name__ == "__main__":
    main()
