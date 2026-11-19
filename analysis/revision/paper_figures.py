# analysis/revision/paper_figures.py
"""Paper figures and their data tables from the saved per-mission summaries (batch_summaries/*.csv). Writes <out>/<name>.png and <out>/data/<name>.csv.
Figures: baseline_gap_last_delivery (full grid, bootstrap over configurations), baseline_gap_by_load_regime, sensing_stress_last_delivery, energy_by_source.
Usage: python -m analysis.revision.paper_figures <batch_summaries dir> <out dir>"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]       # categorical slots 1-4 of the reference palette, in order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"
BLUES = ["#9ec5f0", "#4f94dd", "#17508f"]                    # one hue, light to dark: low, middle, high load
S = "sppdec_consensus_rule_nogamble_limited"
NAMES = {"centralized_cc_hungarian_hungarian": "Centralised reference", "cbba_single_task_cbba_single": "CBBA single", "cbba_full_capability_bundle": "CBBA capability bundle",
         "marl_ippo_limited": "MARL (IPPO)", "cbba_full_cbba_bundle": "CBBA bundle", "naive_decentralized_reactive": "Naive", "decentralized_auction_auction_baseline": "Auction"}


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def save(fig, out, name, table):
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    table.to_csv(os.path.join(out, "data", name + ".csv"), index=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, name + ".png"), dpi=160, facecolor=SURFACE); plt.close(fig); print("wrote", name)


def boot(per, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = per.values
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def regime(cfg):
    a, p = int(cfg.split("_")[0][1:]), int(cfg.split("_P")[1]); r = a / p
    return "low" if r <= 0.25 else "middle" if r <= 0.6 else "high"


def gaps(d, metric="last_delivery"):
    d = d.copy(); d["ok"] = d.delivered == d.parcels
    w = d.pivot_table(index=["config", "replicate"], columns="arm", values=metric); ok = d.pivot_table(index=["config", "replicate"], columns="arm", values="ok", aggfunc="min").astype(bool)
    out = {}
    for a in NAMES:
        keep = ok[S] & ok[a]; df = (w[a] - w[S])[keep].dropna().reset_index(name="d"); out[a] = df.groupby("config").d.mean()
    return out


def pgaps(d, metric="last_delivery"):
    """Per configuration: 100 x (baseline - SPPDEC) / baseline of the mean over the matched complete-case instances (positive: SPPDEC sooner)."""
    d = d.copy(); d["ok"] = d.delivered == d.parcels
    w = d.pivot_table(index=["config", "replicate"], columns="arm", values=metric); ok = d.pivot_table(index=["config", "replicate"], columns="arm", values="ok", aggfunc="min").astype(bool)
    out = {}
    for a in NAMES:
        keep = ok[S] & ok[a]
        g = pd.DataFrame({"s": w[S][keep], "b": w[a][keep]}).dropna().reset_index().groupby("config")[["s", "b"]].mean()
        out[a] = 100 * (g.b - g.s) / g.b
    return out


def main(src, out):
    os.makedirs(out, exist_ok=True)
    grid = pd.read_csv(os.path.join(src, "fresh_grid_realistic.csv")); g = pgaps(grid)
    # 1. gap to SPPDEC, all baselines
    rows = []
    for a, per in g.items():
        m, lo, hi = boot(per); rows.append(dict(strategy=NAMES[a], configs=len(per), mean_gap=m, ci_low=lo, ci_high=hi, sppdec_faster_in=int((per > 0).sum()), unit="percent sooner"))
    t = pd.DataFrame(rows).sort_values("mean_gap")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE); style(ax)
    y = np.arange(len(t))
    ax.hlines(y, t.ci_low, t.ci_high, color=SERIES[0], lw=2.2); ax.plot(t.mean_gap, y, "o", color=SERIES[0], ms=7, mfc=SURFACE, mec=SERIES[0], mew=2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{n}  ({k}/{c})" for n, k, c in zip(t.strategy, t.sppdec_faster_in, t.configs)], color=INK, fontsize=9)
    ax.set_xlabel("SPPDEC sooner to the last delivery, % of the strategy's time\nbars: 95% bootstrap over configurations", color=MUTED, fontsize=8.5)
    ax.set_title("Full grid, 64 configurations x 5 seeds", color=INK, fontsize=10, loc="left")
    ax.text(0.99, 0.02, "in brackets: configurations where SPPDEC is earlier", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    save(fig, out, "baseline_gap_last_delivery", t)
    # 2. by load regime
    rows = []
    for a, per in g.items():
        for rg in ("low", "middle", "high"):
            sub = per[[regime(c) == rg for c in per.index]]; m, lo, hi = boot(sub); rows.append(dict(strategy=NAMES[a], regime=rg, configs=len(sub), mean_gap=m, ci_low=lo, ci_high=hi, unit="percent sooner"))
    t2 = pd.DataFrame(rows); order = list(t.strategy)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE); style(ax)
    for k, rg in enumerate(("low", "middle", "high")):
        s = t2[t2.regime == rg].set_index("strategy").loc[order]; yy = np.arange(len(order)) + (k - 1) * 0.24
        ax.hlines(yy, s.ci_low, s.ci_high, color=BLUES[k], lw=2); ax.plot(s.mean_gap, yy, "o", color=BLUES[k], ms=6, mfc=SURFACE, mec=BLUES[k], mew=2, label={"low": "low load (agents/parcels <= 0.25)", "middle": "middle (0.25 to 0.6)", "high": "high (> 0.6)"}[rg])
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, color=INK, fontsize=9)
    ax.set_xlabel("SPPDEC sooner to the last delivery, % of the strategy's time", color=MUTED, fontsize=9)
    ax.set_title("The lead over the strongest baselines shrinks with load; over the others it grows", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    save(fig, out, "baseline_gap_by_load_regime", t2)
    # 3. sensing stress
    st = pd.read_csv(os.path.join(src, "sensing_stress.csv")); n1 = pd.read_csv(os.path.join(src, "sensing_stress_n1.csv")); n1["arm"] = S + "_n1"
    st = pd.concat([st, n1], ignore_index=True)
    fam = {"sppdec": "SPPDEC", "cbba_single": "CBBA single", "centralized": "Centralised reference", "marl": "MARL (IPPO)"}
    st["strategy"] = st.arm.map(lambda a: next(v for k, v in fam.items() if a.startswith(k)))
    def cond(a):
        for t_ in ("n2l50", "n2", "n1", "l50"):
            if a.endswith("_" + t_) or ("_" + t_ + "_") in a: return t_
        return "ref"
    st["condition"] = st.arm.map(cond)
    lab = {"ref": "exact", "l50": "half of readings lost", "n1": "1-cell error", "n2": "2-cell error", "n2l50": "2-cell error and half lost"}
    t3 = st.groupby(["strategy", "condition"]).agg(missions=("last_delivery", "size"), last_delivery=("last_delivery", "mean"), makespan=("makespan", "mean"), energy=("energy", "mean")).reset_index()
    t3["condition"] = pd.Categorical(t3.condition, ["ref", "l50", "n1", "n2", "n2l50"], ordered=True); t3 = t3.sort_values(["strategy", "condition"])
    # percent: SPPDEC sooner than each baseline under each condition, on the same missions (CBBA single and the centralised reference read no peer from the sensor, so their exact values stand in for the 1-cell level; MARL was not run at 1 cell)
    base = {"CBBA single": "cbba_single", "Centralised reference": "centralized", "MARL (IPPO)": "marl"}
    def cfg_means(strategy, c):
        sub = st[(st.strategy == strategy) & (st.condition == c)]
        return sub.groupby("config").last_delivery.mean()
    rows3 = []
    fig, ax = plt.subplots(figsize=(7.6, 4.0), facecolor=SURFACE); style(ax)
    for i, (nm, _k) in enumerate(base.items()):
        xs, ys = [], []
        for x, c in enumerate(lab):
            b = cfg_means(nm, c if len(st[(st.strategy == nm) & (st.condition == c)]) else "ref") if not (nm == "MARL (IPPO)" and c == "n1") else None
            if b is None:
                xs.append(x); ys.append(float("nan")); continue     # break the line: this level was not run
            sp = cfg_means("SPPDEC", c)
            pct = (100 * (b - sp) / b).dropna()
            m_, lo_, hi_ = boot(pct); xs.append(x); ys.append(m_)
            rows3.append(dict(baseline=nm, condition=c, configs=len(pct), mean_pct_sooner=m_, ci_low=lo_, ci_high=hi_))
        ax.plot(xs, ys, "-o", color=SERIES[i + 1], lw=2, ms=6, mfc=SURFACE, mec=SERIES[i + 1], mew=2, label=f"vs {nm}")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(5)); ax.set_xticklabels([lab[c] for c in lab], color=INK, fontsize=8.5, rotation=12)
    ax.set_ylabel("SPPDEC sooner than the baseline by (% of its time)", color=MUTED, fontsize=9)
    ax.set_title("Peer-position error removes SPPDEC's lead; lost readings do not (21 missions per point)", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.text(0.01, 0.01, "1-cell error was run for SPPDEC only; below zero, the baseline is sooner", transform=ax.transAxes, color=MUTED, fontsize=8)
    t3 = pd.DataFrame(rows3)
    save(fig, out, "sensing_stress_last_delivery", t3)
    # 4. energy by source
    eb = pd.read_csv(os.path.join(src, "energy_breakdown_grid.csv"))
    t4 = eb.groupby("arm")[["motion_kJ", "hover_kJ", "radio_kJ", "computation_kJ_est"]].mean().reset_index()
    t4["strategy"] = t4.arm.map(lambda a: NAMES.get(a, "SPPDEC" if a == S else a)); t4["total_kJ"] = t4[["motion_kJ", "hover_kJ", "radio_kJ", "computation_kJ_est"]].sum(axis=1)
    t4 = t4.sort_values("total_kJ", ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), facecolor=SURFACE); style(ax); y = np.arange(len(t4)); left = np.zeros(len(t4))
    for col, lbl, c in (("motion_kJ", "motion", SERIES[0]), ("hover_kJ", "hover", SERIES[1]), ("radio_kJ", "radio", SERIES[2]), ("computation_kJ_est", "computation (estimated)", SERIES[3])):
        ax.barh(y, t4[col], left=left, color=c, height=0.62, label=lbl, edgecolor=SURFACE, linewidth=1.5); left += t4[col].values
    for yy, tot in zip(y, t4.total_kJ): ax.text(tot + 12, yy, f"{tot:.0f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(t4.strategy, color=INK, fontsize=9); ax.set_xlabel("energy per mission (kJ); 12 missions per strategy", color=MUTED, fontsize=9)
    ax.set_title("Motion dominates; computation is under 1 kJ for every strategy", color=INK, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    save(fig, out, "energy_by_source", t4.drop(columns=["arm"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
