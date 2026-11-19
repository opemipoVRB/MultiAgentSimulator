# analysis/revision/paper_tables.py
"""LaTeX tables for the revised manuscript, generated from the saved per-mission summaries (batch_summaries/*.csv), so no number is typed by hand.
Writes <out>/<name>.tex (tabular bodies wrapped in a table environment is left to the manuscript: each file is one tabular).
Usage: python -m analysis.revision.paper_tables <batch_summaries dir> <out dir>"""
import os, sys
import numpy as np
import pandas as pd
from analysis.revision.paper_figures import S, NAMES, boot, gaps, pgaps, regime

ORDER = ["sppdec", "centralized_cc_hungarian_hungarian", "cbba_single_task_cbba_single", "cbba_full_capability_bundle", "cbba_full_cbba_bundle", "marl_ippo_limited", "naive_decentralized_reactive", "decentralized_auction_auction_baseline"]
SHORT = dict(NAMES, sppdec="SPPDEC")


def tex(path, header, rows, colspec):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def ci(m, lo, hi, d=1):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def arm(a):
    return S if a == "sppdec" else a


def main(src, out):
    os.makedirs(out, exist_ok=True)
    grid = pd.read_csv(os.path.join(src, "fresh_grid_realistic.csv")); grid["ok"] = grid.delivered == grid.parcels
    # 1. grid summary on the instances that every strategy completed
    allok = grid.pivot_table(index=["config", "replicate"], columns="arm", values="ok", aggfunc="min").all(axis=1)
    m = grid.set_index(["config", "replicate"]).loc[allok[allok].index].reset_index()
    rows = []
    for k in ORDER:
        a = arm(k); g = grid[grid.arm == a]; mm = m[m.arm == a]
        rows.append([SHORT[k], int(g.ok.sum()), int(g.dead.sum()), f"{mm.last_delivery.mean():.1f}", f"{mm.makespan.mean():.1f}", f"{mm.energy.mean():.1f}", f"{mm.messages.mean():.0f}"])
    tex(os.path.join(out, "grid_summary.tex"), "\\textbf{Strategy} & \\textbf{Compl.} & \\textbf{Lost} & \\textbf{Last del.} & \\textbf{Makesp.} & \\textbf{Energy} & \\textbf{Msg.}", rows, "lrrrrrr")
    open(os.path.join(out, "grid_summary_n.txt"), "w").write(str(int(allok.sum())))
    # 2. gaps with bootstrap intervals
    gl, gm, ge = gaps(grid, "last_delivery"), gaps(grid, "makespan"), gaps(grid, "energy")
    def pct_gap(metric):
        w_ = grid.pivot_table(index=["config", "replicate"], columns="arm", values=metric)
        ok_ = grid.pivot_table(index=["config", "replicate"], columns="arm", values="ok", aggfunc="min").astype(bool)
        res = {}
        for a in NAMES:
            keep = ok_[S] & ok_[a]
            df_ = pd.DataFrame({"s": w_[S][keep], "b": w_[a][keep]}).dropna().reset_index().groupby("config")[["s", "b"]].mean()
            res[a] = 100 * (df_.b - df_.s) / df_.b
        return res
    pl_, pm_ = pct_gap("last_delivery"), pct_gap("makespan")
    rows = []
    for a in [x for x in NAMES]:
        r = [NAMES[a], len(gl[a])]
        mu, lo, hi = boot(pl_[a]); r.append(ci(mu, lo, hi) + "\\%")
        mu, lo, hi = boot(gl[a]); r.append(ci(mu, lo, hi))
        mu, lo, hi = boot(pm_[a]); r.append(ci(mu, lo, hi) + "\\%")
        mu, lo, hi = boot(gm[a]); r.append(ci(mu, lo, hi))
        mu, lo, hi = boot(ge[a]); r.append(ci(mu, lo, hi))
        r.append(f"{int((gl[a] > 0).sum())}/{len(gl[a])}")
        rows.append(r)
    rows.sort(key=lambda r: float(r[2].split("\\%")[0].split()[0]))
    tex(os.path.join(out, "grid_gaps.tex"), "\\textbf{Strategy} & \\textbf{Cfg.} & \\textbf{Last del. (\\%)} & \\textbf{Last del. (steps)} & \\textbf{Makesp. (\\%)} & \\textbf{Makesp. (steps)} & \\textbf{Energy (units)} & \\textbf{Earlier in}", rows, "lrrrrrrr")
    # 3. by load regime (percent sooner, 95% bootstrap over the configurations of the regime, configurations in which SPPDEC is sooner)
    pg = pgaps(grid)
    rows = []
    for a in NAMES:
        r = [NAMES[a]]
        for rg in ("low", "middle", "high"):
            sub = pg[a][[regime(c) == rg for c in pg[a].index]]; mu, lo, hi = boot(sub)
            r.append(f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}]\\% ({int((sub > 0).sum())}/{len(sub)})")
        rows.append(r)
    rows.sort(key=lambda r: float(r[1].split()[0]))
    tex(os.path.join(out, "grid_regime.tex"), "\\textbf{Strategy} & \\textbf{Low load} & \\textbf{Middle load} & \\textbf{High load}", rows, "lrrr")
    # 4. sensing stress
    st = pd.read_csv(os.path.join(src, "sensing_stress.csv")); n1 = pd.read_csv(os.path.join(src, "sensing_stress_n1.csv")); n1["arm"] = S + "_n1"
    st = pd.concat([st, n1], ignore_index=True)
    fam = [("sppdec", "SPPDEC"), ("cbba_single", "CBBA single"), ("centralized", "Centralised reference"), ("marl", "MARL (IPPO)")]
    st["s"] = st.arm.map(lambda a: next(v for k, v in fam if a.startswith(k)))
    def cond(a):
        for t in ("n2l50", "n2", "n1", "l50"):
            if a.endswith("_" + t) or ("_" + t + "_") in a: return t
        return "ref"
    st["c"] = st.arm.map(cond)
    cols = [("ref", "Exact"), ("l50", "50\\% lost"), ("n1", "1 cell"), ("n2", "2 cells"), ("n2l50", "2 cells, 50\\% lost")]
    rows = []
    for _, name in fam:
        r = [name]
        for c, _l in cols:
            sub = st[(st.s == name) & (st.c == c)]
            r.append(f"{sub.last_delivery.mean():.0f}" if len(sub) else "--")
        rows.append(r)
    tex(os.path.join(out, "sensing_stress.tex"), "\\textbf{Strategy} & " + " & ".join(f"\\textbf{{{l}}}" for _, l in cols), rows, "lrrrrr")
    # sensing: paired change for SPPDEC
    w = st[st.s == "SPPDEC"].pivot_table(index=["config", "replicate"], columns="c", values="last_delivery")
    rows = []
    for c, l in cols[1:]:
        x = (w[c] - w["ref"]).dropna(); pc = 100 * x.mean() / w["ref"].mean()
        rows.append([l, len(x), f"{pc:+.1f}\\%", f"{x.mean():+.1f} $\\pm$ {2.086*x.std()/len(x)**.5:.1f}", f"{int((x > 0).sum())}/{len(x)}"])
    tex(os.path.join(out, "sensing_sppdec_change.tex"), "\\textbf{Condition} & \\textbf{Missions} & \\textbf{Change (\\%)} & \\textbf{Change (steps)} & \\textbf{Worse in}", rows, "lrrrr")
    # SPPDEC sooner than each baseline under each condition, in percent of the baseline's time (per configuration, bootstrap over the 7 configurations)
    rows = []
    def cm(strategy, c):
        sub = st[(st.s == strategy) & (st.c == c)]
        return sub.groupby("config").last_delivery.mean()
    for c, l in cols:
        r = [l]
        for nm in ("CBBA single", "Centralised reference", "MARL (IPPO)"):
            if nm == "MARL (IPPO)" and c == "n1":
                r.append("--"); continue
            bc = c if len(st[(st.s == nm) & (st.c == c)]) else "ref"      # CBBA single and the centralised reference read no peer from the sensor: the exact values stand in
            b, sp = cm(nm, bc), cm("SPPDEC", c)
            pct = (100 * (b - sp) / b).dropna(); mu, lo, hi = boot(pct)
            r.append(f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}]\\%")
        rows.append(r)
    tex(os.path.join(out, "sensing_gap.tex"), "\\textbf{Condition} & \\textbf{vs CBBA single} & \\textbf{vs Centralised} & \\textbf{vs MARL}", rows, "lrrr")
    # 5. energy by source
    eb = pd.read_csv(os.path.join(src, "energy_breakdown_grid.csv"))
    t = eb.groupby("arm")[["motion_kJ", "hover_kJ", "radio_kJ", "computation_kJ_est", "cpu_s"]].mean().reset_index()
    t["tot"] = t[["motion_kJ", "hover_kJ", "radio_kJ", "computation_kJ_est"]].sum(axis=1)
    rows = []
    for k in ORDER:
        r = t[t.arm == arm(k)].iloc[0]
        rows.append([SHORT[k], f"{r.motion_kJ:.0f}", f"{r.hover_kJ:.1f}", f"{r.radio_kJ:.3f}", f"{r.computation_kJ_est:.2f}", f"{r.tot:.0f}", f"{r.cpu_s:.1f}"])
    tex(os.path.join(out, "energy_by_source.tex"), "\\textbf{Strategy} & \\textbf{Motion} & \\textbf{Hover} & \\textbf{Radio} & \\textbf{Comp.} & \\textbf{Total} & \\textbf{CPU s}", rows, "lrrrrrr")
    # 6. mixed speed
    sp = pd.read_csv(os.path.join(src, "speed_transfer.csv"))
    sp["s"] = sp.arm.map(lambda a: "SPPDEC" if a.startswith("sppdec") else "Centralised reference" if a.startswith("centralized") else "CBBA single" if a.startswith("cbba_single") else "CBBA bundle" if a.startswith("cbba_full") else "Auction" if a.startswith("decentralized") else "Naive" if a.startswith("naive") else "MARL (IPPO), 3 speeds" if a.endswith("hs3") else "MARL (IPPO)")
    base = sp[sp.s == "SPPDEC"].set_index(["config", "replicate"])
    rows = []
    for s in ["SPPDEC", "Centralised reference", "CBBA single", "MARL (IPPO)", "Naive", "CBBA bundle", "Auction", "MARL (IPPO), 3 speeds"]:
        g = sp[sp.s == s]; r = [s, f"{g.last_delivery.mean():.0f}", f"{g.makespan.mean():.0f}", f"{g.energy.mean():.1f}"]
        if s in ("SPPDEC", "MARL (IPPO), 3 speeds"): r.extend(["--", "--"])
        else:
            gi = g.set_index(["config", "replicate"]); x = (gi.last_delivery - base.last_delivery).dropna()
            pc = (100 * (gi.last_delivery.groupby("config").mean() - base.last_delivery.groupby("config").mean()) / gi.last_delivery.groupby("config").mean()).dropna()
            r.append(f"{pc.mean():+.1f}\\%"); r.append(f"{x.mean():+.0f} $\\pm$ {2.086*x.std()/len(x)**.5:.0f} ({int((x > 0).sum())}/{len(x)})")
        rows.append(r)
    tex(os.path.join(out, "mixed_speed.tex"), "\\textbf{Strategy} & \\textbf{Last del.} & \\textbf{Makesp.} & \\textbf{Energy} & \\textbf{SPPDEC sooner (\\%)} & \\textbf{Gap (steps)}", rows, "lrrrrr")
    # 7. sensitivity
    se = pd.read_csv(os.path.join(src, "sensitivity.csv")); se["ok"] = se.delivered == se.parcels
    lab = {"": "Arm of record", "_b0": "$\\beta=0$", "_b10": "$\\beta=0.1$", "_w0": "$\\bar w=0$", "_w50": "$\\bar w=0.5$", "_h10": "Belief window 10", "_h60": "Belief window 60", "_boff": "Belief off"}
    ref = se[se.arm == S].set_index(["config", "replicate"])
    rows = []
    for suf, l in lab.items():
        g = se[se.arm == S + suf]; x = (g.set_index(["config", "replicate"]).last_delivery - ref.last_delivery)
        same = int(((g.set_index(["config", "replicate"]).last_delivery == ref.last_delivery) & (g.set_index(["config", "replicate"]).makespan == ref.makespan) & ((g.set_index(["config", "replicate"]).energy - ref.energy).abs() < 1e-6)).sum())
        rows.append([l, int(g.ok.sum()), int(g.dead.sum()), f"{g.last_delivery.mean():.1f}", f"{g.energy.mean():.1f}", f"{same}/{len(g)}"])
    tex(os.path.join(out, "sensitivity.tex"), "\\textbf{Setting} & \\textbf{Compl.} & \\textbf{Lost} & \\textbf{Last del.} & \\textbf{Energy} & \\textbf{Identical}", rows, "lrrrrr")

    # 8. configurations in which SPPDEC's mean last delivery is above the best baseline's (instances every strategy completed)
    rows = []
    for cfg, g in m.groupby("config"):
        means = {a: g[g.arm == a].last_delivery.mean() for a in g.arm.unique()}
        base = {a: v for a, v in means.items() if a != S}
        if not base: continue
        best = min(base, key=base.get)
        if means[S] > base[best]:
            rows.append((cfg, g.replicate.nunique(), means[S], NAMES[best], base[best]))
    rows.sort(key=lambda r: (int(r[0].split("_")[0][1:]), int(r[0].split("_P")[1])))
    tex(os.path.join(out, "config_exceptions.tex"), "\\textbf{Config.} & \\textbf{Inst.} & \\textbf{SPPDEC} & \\textbf{Best baseline} & \\textbf{Its mean}",
        [[c.replace("_", "\\_"), n, f"{a:.0f}", b, f"{v:.0f}"] for c, n, a, b, v in rows], "lrrlr")
    open(os.path.join(out, "config_exceptions_n.txt"), "w").write(f"{len(rows)} of {m.config.nunique()}")
    # 9. completion and agent loss per strategy
    rows = []
    for k in ORDER:
        g = grid[grid.arm == arm(k)]
        rows.append([SHORT[k], int(g.ok.sum()), int((~g.ok).sum()), int((g.dead.fillna(0) > 0).sum()), int(g.dead.sum())])
    tex(os.path.join(out, "completion_loss.tex"), "\\textbf{Strategy} & \\textbf{Complete} & \\textbf{Incomplete} & \\textbf{Runs with loss} & \\textbf{Agents lost}", rows, "lrrrr")

    # 10. Holm-corrected p-values (paired t-test on per-configuration mean differences, Holm over the seven baselines)
    from scipy import stats as _st
    def holm(ps):
        order = sorted(range(len(ps)), key=lambda i: ps[i]); out_ = [None] * len(ps); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out_[i] = run
        return out_
    def fmt(p_):
        if p_ < 1e-6: return "$<10^{-6}$"
        if p_ < 1e-3:
            m_, e_ = f"{p_:.1e}".split("e"); return f"${m_}{{\\times}}10^{{{int(e_)}}}$"
        return f"{p_:.3f}"
    keys = list(NAMES)
    cols = {}
    for metric, g in (("last", gl), ("mk", gm), ("en", ge)):
        ps = [_st.ttest_1samp(g[a].values, 0.0).pvalue for a in keys]; cols[metric] = holm(ps)
    rows = [[NAMES[a], fmt(cols["last"][i]), fmt(cols["mk"][i]), fmt(cols["en"][i])] for i, a in enumerate(keys)]
    tex(os.path.join(out, "grid_pvalues.tex"), "\\textbf{Strategy} & \\textbf{Last delivery} & \\textbf{Makespan} & \\textbf{Energy}", rows, "lrrr")

    # 11. as recorded: all 320 missions per strategy, including incomplete ones (sensitivity to the matched complete-case rule)
    rows = []
    for k in ORDER:
        g = grid[grid.arm == arm(k)]
        rows.append([SHORT[k], f"{g.makespan.mean():.0f}", f"{g.energy.mean():.1f}", f"{g.delivered.sum()/g.parcels.sum()*100:.1f}\\%"])
    tex(os.path.join(out, "as_recorded.tex"), "\\textbf{Strategy} & \\textbf{Makespan} & \\textbf{Energy} & \\textbf{Parcels delivered}", rows, "lrrr")
    print("wrote tables to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
