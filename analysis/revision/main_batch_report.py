# analysis/revision/main_batch_report.py
"""
Comparison tables for the main revision batch: makespan of each arm relative to a reference arm
(matched complete-case pairs, geometric mean, bootstrap interval), overall and by load regime.

Usage:
    python -m analysis.revision.main_batch_report --root experiments/revision_fixed_dt/main_batch
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.revision.revision_stats import load_sweep, regime_of, holm
import builtins as _b
from analysis.revision.names import pretty as _pretty


def print(*args, **kwargs):  # printed names use the plain scheme of arm_names.py
    _b.print(*[_pretty(str(a)) for a in args], **kwargs)


N_BOOT = 2000


def relative(df: pd.DataFrame, ref: str, arm: str, scope_col=None, scope=None):
    """Makespan of `arm` over `ref` on matched complete replicates. Below 1 = arm faster."""
    a = df[(df.strategy == ref) & df.complete]
    b = df[(df.strategy == arm) & df.complete]
    m = a.merge(b, on=["config", "instance"], suffixes=("_ref", "_arm"))
    if scope_col is not None:
        m = m[m[scope_col + "_ref"] == scope]
    if len(m) == 0:
        return None
    lr = np.log(m["makespan_arm"].to_numpy() / m["makespan_ref"].to_numpy())
    rng = np.random.default_rng(1)
    boots = [rng.choice(lr, size=len(lr), replace=True).mean() for _ in range(N_BOOT)]
    lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
    # sign-flip test on the log ratios
    obs = abs(lr.mean())
    flips = [abs((lr * rng.choice([-1, 1], size=len(lr))).mean()) for _ in range(N_BOOT)]
    p = (1 + sum(f >= obs for f in flips)) / (1 + N_BOOT)
    return {"n": len(m), "ratio": float(np.exp(lr.mean())), "lo": float(lo), "hi": float(hi), "p": p,
            "share_arm_faster": float((lr < 0).mean())}


def table(df, ref, arms, title, by_regime=True):
    scopes = [("all", None)] + ([(r, r) for r in ("task-rich (c<0.3)", "balanced (0.3<=c<1)", "agent-surplus (c>=1)")]
                                if by_regime else [])
    rows = []
    for arm in arms:
        if arm not in set(df.strategy):
            continue
        row = {"arm": arm}
        for name, sc in scopes:
            r = relative(df, ref, arm, "regime" if sc else None, sc)
            row[name] = None if r is None else r
        rows.append(row)
    # Holm across arms for the overall column
    ps = [r["all"]["p"] for r in rows if r["all"]]
    adj = holm(ps) if ps else []
    out = [f"### {title}", "", f"Reference: **{ref}**. Values are makespan of the arm divided by the reference "
           "(below 1 = faster), geometric mean over matched complete replicates, 95% bootstrap interval; "
           "Holm-adjusted p (sign-flip test) over the arms listed.", ""]
    header = "| arm | n | all | Holm p | " + " | ".join(n for n, _ in scopes[1:]) + " |"
    out += [header, "|" + "---|" * (4 + len(scopes) - 1)]
    k = 0
    for r in rows:
        a = r["all"]
        if a is None:
            continue
        cells = [f"{x['ratio']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]" if x else "n/a" for n, x in
                 ((n, r[n]) for n, _ in scopes[1:])]
        out.append(f"| {r['arm']} | {a['n']} | {a['ratio']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] | {adj[k]:.3f} | "
                   + " | ".join(cells) + " |")
        k += 1
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/revision_fixed_dt/main_batch")
    ap.add_argument("--out", default=".ignore/sandbox/main_results/tables.md")
    a = ap.parse_args()
    df = load_sweep(Path(a.root))
    df["regime"] = df["c"].map(regime_of)
    md = []
    md.append(table(df, "S: Always-G", [
        "S: Always-H", "S: Rule", "S: CBBA-in-head (asym)",
        "S: Hungarian symmetric-cost only", "S: Hungarian no-carrying-peers only", "S: Hungarian shared",
        "S: CBBA-in-head symmetric-cost only", "S: CBBA-in-head no-carrying-peers only", "S: CBBA-in-head shared",
        "Naive (RG)", "CBBA-B", "CBBA-ST", "C-Hung."], "Peer reasoning against greedy (with the yield rule)"))
    md.append(table(df, "S: shared, conflict -> trend", [
        "S: shared, conflict -> competition", "S: shared, conflict -> model judgement",
        "S: CBBA-in-head shared", "S: Rule", "S: Comb. Qwen3-VL (paired_steps)"],
        "Who decides when the trend and the competition conflict"))
    md.append(table(df, "S: shared, conflict -> model judgement", [
        "S: judgement, +33 steps", "S: judgement, +60 steps", "S: judgement, +120 steps",
        "S: judgement, +33 steps, non-blocking", "S: judgement, +120 steps, non-blocking"],
        "SLM inference latency on the judgement arm (reference = no latency)"))
    md.append(table(df, "S: Comb. Qwen3-VL (paired_steps)", [
        "S: Comb. +1s", "S: Comb. +2s", "S: Comb. +2s, non-blocking"],
        "SLM inference latency on the original guarded arm (reference = no latency)"))
    md.append(table(df, "S: CBBA-in-head shared", [
        "S: CBBA shared, R=12", "S: CBBA shared, R=8", "S: CBBA shared, R=4",
        "S: CBBA shared, R=8, noise 1", "S: CBBA shared, R=8, loss 30%"],
        "Limited observation, shared CBBA in head (reference = unlimited observation)"))
    md.append(table(df, "S: Always-G", [
        "S: Always-G, R=12", "S: Always-G, R=8", "S: Always-G, R=4",
        "S: Always-G, R=8, noise 1", "S: Always-G, R=8, loss 30%"],
        "Limited observation, greedy (reference = unlimited observation)"))
    md.append(table(df, "S: Always-G, hetero fleet", [
        "S: CBBA shared, hetero fleet, peers known", "S: CBBA shared, hetero fleet, peers assumed alike",
        "Naive (RG), hetero fleet", "CBBA-ST, hetero fleet", "C-Hung., hetero fleet"],
        "Heterogeneous fleet (rates 0.75, 1.0, 1.5), reference = greedy on the same fleet"))
    md.append(table(df, "S: CBBA shared, hetero fleet, peers known", [
        "S: CBBA shared, hetero fleet, peers assumed alike"],
        "Heterogeneous fleet: correct against wrong assumptions about peers' consumption", by_regime=True))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
