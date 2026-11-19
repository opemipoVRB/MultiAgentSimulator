# analysis/revision/gap_paper_tables.py
"""
LaTeX tables for the gap-closing batches (revision). Percentages give each configuration equal weight (per-mission percent, averaged within a configuration), with a 95% bootstrap interval over the
seven configurations (10,000 resamples, seed 0), as in the main comparison. Usage: python -m analysis.revision.gap_paper_tables <out dir>
"""
import csv, sys
from collections import defaultdict
import numpy as np
from analysis.revision.gap_tables import split_arm, load, done

def lead_by_config(data, s_arm, o_arm, tag_s, tag_o, keys):
    """per-config mean of 100*(1 - S/other) over missions in keys (both completed)."""
    per = defaultdict(list)
    for k in keys:
        v = data[k]
        a, b = v.get((s_arm, tag_s)) or v.get((s_arm, "ref")), v.get((o_arm, tag_o)) or v.get((o_arm, "ref"))
        if done(a) and done(b):
            per[k[0]].append(100 * (1 - float(a["last_delivery"]) / float(b["last_delivery"])))
    return [np.mean(v) for v in per.values()]

def boot(vals, n=10000, seed=0):
    vals = np.array(vals)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    m = rng.choice(vals, (n, len(vals))).mean(1)
    return vals.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def fmt(b):
    return "--" if b is None else f"{b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}]\\%"

def own_change(data, s, tag):
    """pooled: mean last delivery under the condition over mean last delivery of the same missions without it, minus one (percent)."""
    a, b = [], []
    for k, v in data.items():
        x, y = v.get((s, tag)), v.get((s, "ref"))
        if done(x) and done(y):
            a.append(float(x["last_delivery"])); b.append(float(y["last_delivery"]))
    return (100 * (np.mean(a) / np.mean(b) - 1),)

def main(out):
    d = load("sensing_all")
    keys = list(d)
    rows = [("Exact", "ref"), ("0.5 cell", "n05"), ("1 cell", "n1"), ("2 cells", "n2"), ("50\\% lost", "l50"), ("2 cells, 50\\% lost", "n2l50"),
            ("Delay 0.5 s", "d30"), ("Delay 2 s", "d120"), ("Blackout 1/3", "bo")]
    L = ["\\begin{tabular}{lrrrr}", "\\toprule", "\\textbf{Condition} & \\textbf{SPPDEC slower} & \\textbf{vs CBBA single} & \\textbf{vs Centralised} & \\textbf{vs MARL} \\\\", "\\midrule"]
    for name, tag in rows:
        own = "--" if tag == "ref" else f"{own_change(d, 'S', tag)[0]:+.1f}\\%"
        cells = [fmt(boot(lead_by_config(d, "S", o, tag, tag, keys))) for o in ("C", "H", "M")]
        L.append(f"{name} & {own} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{out}/sensing_gap.tex", "w").write("\n".join(L) + "\n")
    # environments: lead of SPPDEC in other settings, each batch against its own runs of the other strategies
    envs = [("Unequal batteries (100/60)", "unequal_batteries", "ub"), ("Map 16$\\times$10", "map_16x10", "ref"), ("Map 40$\\times$25", "map_40x25", "ref"),
            ("Obstacles, 18\\% blocked", "obstacles_8_3_3", "ref"), ("Obstacles, 32\\% blocked", "obstacles_14_3_3", "ref"), ("Late orders (half after the start)", "dynamic", "ref")]
    L = ["\\begin{tabular}{lrrr}", "\\toprule", "\\textbf{Setting} & \\textbf{vs CBBA single} & \\textbf{vs Centralised} & \\textbf{vs MARL} \\\\", "\\midrule"]
    for name, batch, tag in envs:
        dd = load(batch)
        cells = [fmt(boot(lead_by_config(dd, "S", o, tag, tag, list(dd)))) for o in ("C", "H", "M")]
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{out}/environments.tex", "w").write("\n".join(L) + "\n")
    # radio
    ref = load("radio_ideal")
    prof = [("Ideal", "radio_ideal"), ("50\\% lost", "radio_lossy50"), ("80\\% lost", "radio_lossy80"), ("Delay 10--30 steps", "radio_slow"), ("Outage 1/3", "radio_outage"),
            ("50\\% lost, 3 tries", "radio_lossy50_retx3"), ("80\\% lost, 5 tries", "radio_lossy80_retx5")]
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule", "\\textbf{Radio} & \\textbf{CBBA single} & \\textbf{Cap. bundle} & \\textbf{CBBA bundle} & \\textbf{Auction} & \\textbf{Centr. (done)} \\\\", "\\midrule"]
    for name, batch in prof:
        try:
            dd = load(batch)
        except FileNotFoundError:
            continue
        cells = []
        for s in ("C", "CK", "CB", "A"):
            ch = own_change_between(dd, ref, s)
            cells.append("--" if ch is None else f"{ch:+.0f}\\%")
        nh = sum(done(v.get(("H", "ref"))) for v in dd.values())
        L.append(f"{name} & " + " & ".join(cells) + f" & {nh}/{len(dd)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{out}/radio_stress.tex", "w").write("\n".join(L) + "\n")

def own_change_between(dd, ref, s):
    per = defaultdict(list)
    for k, v in dd.items():
        a, b = v.get((s, "ref")), ref.get(k, {}).get((s, "ref"))
        if done(a) and done(b):
            per[k[0]].append(100 * (float(a["last_delivery"]) / float(b["last_delivery"]) - 1))
    return np.mean([np.mean(x) for x in per.values()]) if per else None

if __name__ == "__main__":
    main(sys.argv[1])
