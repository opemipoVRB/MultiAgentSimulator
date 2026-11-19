# analysis/revision/model_context.py
"""
When does CBBA reason better than Hungarian, and when the reverse? For each scenario of the context study (fleet size, parcel count, sensing radius, battery capacity, parcel weights),
the work phase (step of the last delivery) and the makespan of the consensus drone reasoning with Hungarian alone, CBBA alone, or both, as paired ratios over the seeds.
Usage: python -m analysis.revision.model_context <study dir>      (the dir holds scenarios.txt and one sub-dir per scenario)
"""
import glob
import json
import math
import statistics as st
import sys

from analysis.revision.marl_evaluation import load


def last_delivery(root, cfg, arm):
    out = {}
    for ep in glob.glob(f"{root}/{cfg}/experiment_*/experiment.json"):
        seed = json.load(open(ep))["seed"]
        for d in glob.glob(ep.replace("experiment.json", f"sppdec_{arm}_instance_*")):
            ags = json.load(open(d + "/agents.json"))
            ags = list(ags.values()) if isinstance(ags, dict) else ags
            drops = [x["time"]["step"] for g in ags for x in g["actions"] if x["type"] == "drop"]
            out[seed] = max(drops) if drops else None
    return out


def ratio(a, b):
    r = [a[k] / b[k] for k in a if k in b and a[k] and b[k]]
    if not r:
        return None, None, 0, 0
    m = st.mean(r)
    ci = (2.0 * st.stdev(r) / math.sqrt(len(r))) if len(r) > 2 else float("nan")
    return m, ci, sum(1 for x in r if x < 1), len(r)


def main(study):
    rows = [ln.split() for ln in open(f"{study}/scenarios.txt") if ln.strip()]
    print(f"{'scn':4s} {'N':>3s} {'P':>3s} {'N/P':>5s} {'rad':>3s} {'cap':>4s} {'weights':>8s} | complete H/C/B | last delivery  H     C     B | CBBA/Hung work (faster) | both/Hung work | makespan H     C     B | CBBA/Hung makespan")
    for sid, n, p, r, c, w in rows:
        cfg = f"A{n}_P{p}"
        ctag = f"r{r}" + (f"_c{c}" if c != "0" else "")
        arms = {m: f"ctx_{m}_{ctag}" for m in ("hungarian", "cbba", "both")}
        rec = load(f"{study}/{sid}")
        ld = {m: last_delivery(f"{study}/{sid}", cfg, a) for m, a in arms.items()}
        mk, comp = {}, {}
        for m, a in arms.items():
            mk[m] = {k[1]: v[a]["makespan"] for k, v in rec.items() if a in v and v[a]["delivered"] == v[a]["parcels"]}
            comp[m] = f"{sum(1 for v in rec.values() if a in v and v[a]['delivered'] == v[a]['parcels'])}/{sum(1 for v in rec.values() if a in v)}"
        mean = lambda d: (st.mean(v for v in d.values() if v) if any(d.values()) else float("nan"))
        wc = ratio(ld["cbba"], ld["hungarian"]); wb = ratio(ld["both"], ld["hungarian"]); mc = ratio(mk["cbba"], mk["hungarian"])
        f = lambda x: "  n/a " if x[0] is None else f"x{x[0]:.2f}+-{x[1]:.2f} ({x[2]}/{x[3]})"
        print(f"{sid:4s} {n:>3s} {p:>3s} {int(n) / int(p):5.2f} {r:>3s} {c:>4s} {w:>8s} | {comp['hungarian']} {comp['cbba']} {comp['both']} | {mean(ld['hungarian']):6.0f} {mean(ld['cbba']):6.0f} {mean(ld['both']):6.0f} | {f(wc)} | {f(wb)} | {mean(mk['hungarian']):6.0f} {mean(mk['cbba']):6.0f} {mean(mk['both']):6.0f} | {f(mc)}")


if __name__ == "__main__":
    main(sys.argv[1])
