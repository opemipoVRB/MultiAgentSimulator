# analysis/revision/marl_learning_curve.py
"""Learning curves of the MARL training runs, for the paper: training return per round and validation return, against the number of training missions (the cost).
Writes <out>/data/marl_learning_curve_data.csv (every round of every run: missions, hours, return, entropy, validation) and <out>/marl_learning_curve_training_and_validation.png.
Usage: python -m analysis.revision.marl_learning_curve <out dir> name=<run dir> [name=<run dir> ...]   (a run dir holds training_log.jsonl)"""
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]       # categorical slots 1-4 of the reference palette, in order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"

def load(path):
    return [json.loads(l) for l in open(os.path.join(path, "training_log.jsonl"))]

def smooth(v, k=5):
    return [sum(v[max(0, i - k + 1):i + 1]) / len(v[max(0, i - k + 1):i + 1]) for i in range(len(v))]

def main(out, specs):
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    runs = {n: load(p) for n, p in (s.split("=", 1) for s in specs)}
    with open(os.path.join(out, "data", "marl_learning_curve_data.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "round", "missions_total", "hours", "mean_return", "entropy", "validation_return", "validation_makespan"])
        for n, rows in runs.items():
            for r in rows:
                w.writerow([n, r["round"], r["episodes_total"], round(r["hours"], 4), round(r["mean_return"], 4), round(r["entropy"], 4), r.get("validation_return"), r.get("validation_makespan")])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for a in ax:
        a.set_facecolor(SURFACE); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
        for s in ("top", "right"): a.spines[s].set_visible(False)
        for s in ("left", "bottom"): a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9); a.set_xlabel("training missions", color=MUTED, fontsize=9)
    for i, (n, rows) in enumerate(runs.items()):
        c = SERIES[i % len(SERIES)]; x = [r["episodes_total"] for r in rows]; y = [r["mean_return"] for r in rows]
        ax[0].plot(x, y, color=c, lw=0.8, alpha=0.35); ax[0].plot(x, smooth(y), color=c, lw=2, label=n)
        v = [(r["episodes_total"], r["validation_return"]) for r in rows if r.get("validation_return") is not None]
        if v: ax[1].plot(*zip(*v), color=c, lw=2, marker="o", ms=5, mfc=SURFACE, mec=c, label=n)
    ax[0].set_title("Training return per round (line: 5-round mean)", color=INK, fontsize=10, loc="left")
    ax[1].set_title("Validation return (A4_P12, A8_P24, A12_P24; unseen seeds)", color=INK, fontsize=10, loc="left")
    ax[0].set_ylabel("return (higher is better)", color=MUTED, fontsize=9)
    ax[1].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, "marl_learning_curve_training_and_validation.png"), dpi=160, facecolor=SURFACE)
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
