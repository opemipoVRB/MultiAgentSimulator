# analysis/revision/revision_stats.py
"""
Revision statistics for the IEEE IoT-J resubmission (tickets T0.8, T0.9, T4.1-T4.4).

Regenerates, from the raw experiment exports:
  1. Table III reproduction check against the submitted manuscript
  2. Completion and agent-loss table
  3. Per-regime paired comparisons of the primary method vs every baseline (Holm-corrected)
  4. Per-config makespan ratios vs reference baselines (geometric mean, median, IQR, bootstrap CI)
  5. Fairness on/off contrasts, pooled and per regime
  6. Mixed-effects model of log makespan with a strategy x competition interaction
  7. Variant-selection check (post-hoc best variant 3B+W vs pre-specified Combined)
  8. Energy overhead vs the centralised Hungarian reference, per regime

Usage:
  python -m analysis.revision.revision_stats
  python -m analysis.revision.revision_stats --sweep experiments/station_scaling --out analysis/revision/output

Data rule
---------
experiment.json `results` can contain appended reruns (the Table IV prediction run and the
belief-tracking ablation were appended to A3/A6/A12/A18 P36). The FIRST occurrence of each
strategy run is the submitted sweep: it reproduces the paper's A12 P36 numbers exactly and
Table III energy exactly. Per-run folders (fleet.json etc.) were overwritten by the reruns,
so everything here is read from `results[].metrics` only.

Parcels delivered are in `metrics.tasks_completed`; one batch of A12 P36 runs
uses the older name `tasks_pending` for the same count.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from src.strategies.sppdec.legacy_ids import modernise
import pandas as pd
from scipy import stats

RNG_SEED = 20260928
MAX_STEPS = 18000  # simulation step limit (60 steps per simulated second)
N_BOOT = 10_000

LABELS = {
    "naive_decentralized_reactive": "Naive (RG)",
    "decentralized_auction_auction_baseline": "Dec-Auct.",
    "cbba_single_task_cbba_single": "CBBA-ST",
    "cbba_full_cbba_bundle": "CBBA-B",
    "centralized_cc_hungarian_hungarian": "C-Hung.",
    "centralized_cc_auction": "C-Auct.",
    "centralized_cc_greedy_greedy": "C-Grdy.",
    "sppdec_combined": "S: Comb.",
    "sppdec_control": "S: Ctrl",
    "sppdec_baseline_only": "S: Base",
    "sppdec_fairness_only": "S: FO",
    "sppdec_personality_mix_3_brains_1_wildcard": "S: 3B+W",
    "sppdec_personality_mix_efficiency_overdrive": "S: EO",
    "sppdec_personality_mix_no_reckless": "S: NR",
    "sppdec_personality_mix_controlled_chaos": "S: CC",
    "sppdec_personality_mix_balanced_quartet": "S: BQ",
    # Revision arbitration ablation (Combined configuration)
    "sppdec_arb_always_greedy": "S: Always-G",
    "sppdec_arb_always_hungarian": "S: Always-H",
    "sppdec_arb_rule_based": "S: Rule",
    # Revision (beta, w-bar) ablation with the rule-based arbiter
    "sppdec_control_rule": "S: Ctrl (rule)",
    "sppdec_fairness_only_rule": "S: FO (rule)",
    "sppdec_baseline_only_rule": "S: Base (rule)",
    # Revision SLM latency in the loop (Combined configuration)
    "sppdec_lat60": "S: Comb. +1s",
    "sppdec_lat120": "S: Comb. +2s",
    "sppdec_combined_qwen3vl_v2": "S: Comb. Qwen3-VL (cost_gap_closed)",
    "sppdec_combined_qwen3vl_paired": "S: Comb. Qwen3-VL (paired_steps)",
    "sppdec_cbba_assumed_bids_asym": "S: CBBA-in-head (asym)",
    "sppdec_hungarian_symmetric": "S: Hungarian shared",
    "sppdec_cbba_assumed_bids_symmetric": "S: CBBA-in-head shared",
    "sppdec_hungarian_sym": "S: Hungarian symmetric-cost only",
    "sppdec_hungarian_nocarry": "S: Hungarian no-carrying-peers only",
    "sppdec_cbba_assumed_bids_sym": "S: CBBA-in-head symmetric-cost only",
    "sppdec_sense_greedy_r4": "S: Always-G, R=4",
    "sppdec_sense_cbba_symmetric_r4": "S: CBBA shared, R=4",
    "sppdec_sense_greedy_r8": "S: Always-G, R=8",
    "sppdec_sense_cbba_symmetric_r8": "S: CBBA shared, R=8",
    "sppdec_sense_greedy_r12": "S: Always-G, R=12",
    "sppdec_sense_cbba_symmetric_r12": "S: CBBA shared, R=12",
    "sppdec_sense_greedy_r8_noise1": "S: Always-G, R=8, noise 1",
    "sppdec_sense_cbba_symmetric_r8_noise1": "S: CBBA shared, R=8, noise 1",
    "sppdec_sense_greedy_r8_loss30": "S: Always-G, R=8, loss 30%",
    "sppdec_sense_cbba_symmetric_r8_loss30": "S: CBBA shared, R=8, loss 30%",
    "naive_decentralized_unequal_batteries_reactive": "Naive (RG), hetero fleet",
    "cbba_single_task_unequal_batteries_cbba_single": "CBBA-ST, hetero fleet",
    "centralized_cc_hungarian_unequal_batteries_hungarian": "C-Hung., hetero fleet",
    "sppdec_greedy_unequal_batteries": "S: Always-G, hetero fleet",
    "sppdec_cbba_assumed_bids_symmetric_unequal_batteries_known": "S: CBBA shared, hetero fleet, peers known",
    "sppdec_cbba_assumed_bids_symmetric_unequal_batteries_assumed_equal": "S: CBBA shared, hetero fleet, peers assumed alike",
    "sppdec_cbba_assumed_bids_nocarry": "S: CBBA-in-head no-carrying-peers only",
    "sppdec_cbba_assumed_bids_symmetric_wait": "S: CBBA-in-head shared+wait",
    "sppdec_cbba_assumed_bids_symmetric_fallback_chain": "S: CBBA-in-head shared+fallback chain",
    "sppdec_hungarian_symmetric_wait": "S: Hungarian shared+wait",
    "sppdec_judge_trend_wins": "S: shared, conflict -> trend",
    "sppdec_judge_competition_wins": "S: shared, conflict -> competition",
    "sppdec_judge_model": "S: shared, conflict -> model judgement",
    "sppdec_cbba_bid_rounds_bundle1_symmetric": "S: CBBA protocol in head (bundle 1)",
    "sppdec_cbba_assumed_bids_symmetric_memory": "S: CBBA shortcut + memory",
    "sppdec_greedy_tracking": "S: Always-G + observation",
    "sppdec_greedy_tracking_gone": "S: Always-G + observation (vanished targets only)",
    "sppdec_greedy_tracking_contested": "S: Always-G + observation (contested targets only)",
    "sppdec_cbba_assumed_bids_symmetric_tracking": "S: CBBA shortcut + observation",
    "sppdec_rule_tracking": "A: Rule table + obs",
    "sppdec_local_rule_tracking_r8": "A: Rule table + obs, R=8",
    "sppdec_classifier_regime_tracking": "A: Regime classifier + obs",
    "sppdec_local_classifier_regime_tracking_r8": "A: Regime classifier + obs, R=8",
    "sppdec_classifier_distilled_tracking": "A: Tree distilled from rules + obs",
    "sppdec_local_classifier_distilled_tracking_r8": "A: Tree distilled from rules + obs, R=8",
    "sppdec_classifier_slm_imitation_tracking": "A: Tree imitating SLM answers + obs",
    "sppdec_local_classifier_slm_imitation_tracking_r8": "A: Tree imitating SLM answers + obs, R=8",
    "sppdec_local_greedy_tracking_r4": "L: Always-G + obs, R=4",
    "sppdec_local_cbba_tracking_r4": "L: CBBA shortcut + obs, R=4",
    "sppdec_local_hungarian_tracking_memory_r4": "L: Hungarian + obs + mem, R=4",
    "sppdec_local_bid_rounds_bundle3_tracking_memory_r4": "L: CBBA protocol + obs + mem, R=4",
    "sppdec_local_greedy_tracking_r8": "L: Always-G + obs, R=8",
    "sppdec_local_cbba_tracking_r8": "L: CBBA shortcut + obs, R=8",
    "sppdec_local_hungarian_tracking_memory_r8": "L: Hungarian + obs + mem, R=8",
    "sppdec_local_bid_rounds_bundle3_tracking_memory_r8": "L: CBBA protocol + obs + mem, R=8",
    "sppdec_local_greedy_tracking_r12": "L: Always-G + obs, R=12",
    "sppdec_local_cbba_tracking_r12": "L: CBBA shortcut + obs, R=12",
    "sppdec_local_hungarian_tracking_memory_r12": "L: Hungarian + obs + mem, R=12",
    "sppdec_local_bid_rounds_bundle3_tracking_memory_r12": "L: CBBA protocol + obs + mem, R=12",
    "sppdec_local_greedy_tracking_r8_noise1": "L: Always-G + obs, R=8, noise 1",
    "sppdec_local_cbba_tracking_r8_noise1": "L: CBBA shortcut + obs, R=8, noise 1",
    "sppdec_local_hungarian_tracking_memory_r8_noise1": "L: Hungarian + obs + mem, R=8, noise 1",
    "sppdec_local_bid_rounds_bundle3_tracking_memory_r8_noise1": "L: CBBA protocol + obs + mem, R=8, noise 1",
    "sppdec_local_greedy_tracking_r8_loss30": "L: Always-G + obs, R=8, loss 30%",
    "sppdec_local_cbba_tracking_r8_loss30": "L: CBBA shortcut + obs, R=8, loss 30%",
    "sppdec_local_hungarian_tracking_memory_r8_loss30": "L: Hungarian + obs + mem, R=8, loss 30%",
    "sppdec_local_bid_rounds_bundle3_tracking_memory_r8_loss30": "L: CBBA protocol + obs + mem, R=8, loss 30%",
    "sppdec_cbba_assumed_bids_symmetric_tracking_gone": "S: CBBA shortcut + observation (vanished targets only)",
    "sppdec_cbba_assumed_bids_symmetric_tracking_contested": "S: CBBA shortcut + observation (contested targets only)",
    "sppdec_cbba_assumed_bids_symmetric_tracking_memory": "S: CBBA shortcut + observation + memory",
    "sppdec_hungarian_symmetric_tracking_memory": "S: Hungarian shared + observation + memory",
    "sppdec_cbba_bid_rounds_bundle3_symmetric_tracking_memory": "S: CBBA protocol (bundle 3) + observation + memory",
    "sppdec_hungarian_symmetric_memory": "S: Hungarian shared + memory",
    "sppdec_cbba_bid_rounds_bundle3_symmetric_memory": "S: CBBA protocol (bundle 3) + memory",
    "sppdec_cbba_bid_rounds_bundle3_symmetric": "S: CBBA protocol in head (bundle 3)",
    "sppdec_judge_model_lat33": "S: judgement, +33 steps",
    "sppdec_judge_model_lat60": "S: judgement, +60 steps",
    "sppdec_judge_model_lat120": "S: judgement, +120 steps",
    "sppdec_judge_model_lat33_prefetch": "S: judgement, +33 steps, non-blocking",
    "sppdec_judge_model_lat120_prefetch": "S: judgement, +120 steps, non-blocking",
    "sppdec_lat120_prefetch": "S: Comb. +2s, non-blocking",
    "sppdec_conflict_competition": "S: conflict -> competition",
    "sppdec_conflict_judgement": "S: conflict -> model judgement",
}

PRIMARY = "S: Comb."
BASELINES = ["CBBA-ST", "C-Hung.", "C-Auct.", "C-Grdy.", "CBBA-B", "Dec-Auct.", "Naive (RG)"]
SPPDEC_VARIANTS = [v for v in LABELS.values() if v.startswith("S: ")]
REVISION_ARMS = ("S: Always-G", "S: Always-H", "S: Rule", "S: Comb. +1s", "S: Comb. +2s",
                 "S: Ctrl (rule)", "S: FO (rule)", "S: Base (rule)")
SUBMITTED_VARIANTS = [v for v in SPPDEC_VARIANTS if v not in REVISION_ARMS]

# Submitted manuscript, Table III: (MS mean, MS sd, energy, co-occupancy, messages)
PAPER_TABLE_III = {
    "CBBA-ST": (930.5, 688.4, 144.4, 553.8), "C-Hung.": (937.6, 698.6, 113.5, 459.6),
    "S: 3B+W": (938.9, 510.8, 174.6, 0.0), "C-Auct.": (940.1, 698.3, 113.0, 461.7),
    "S: EO": (945.7, 514.4, 174.1, 0.0), "S: NR": (945.9, 495.2, 175.9, 0.0),
    "S: CC": (949.8, 500.9, 176.4, 0.0), "S: BQ": (956.6, 486.9, 175.7, 0.0),
    "S: Comb.": (958.1, 499.0, 178.0, 0.0), "S: Ctrl": (960.9, 494.8, 178.3, 0.0),
    "S: Base": (962.6, 500.4, 176.6, 0.0), "S: FO": (963.0, 488.4, 176.6, 0.0),
    "C-Grdy.": (994.0, 767.5, 119.7, 437.6), "CBBA-B": (1149.7, 719.2, 188.6, 1292.7),
    "Dec-Auct.": (1156.5, 518.2, 615.7, 6692.4), "Naive (RG)": (1268.8, 606.4, 180.6, 0.0),
}

# Proposed load regimes on c = agents / parcels (T4.1; needs author approval)
REGIMES = [("task-rich (c<0.3)", 0.0, 0.3), ("balanced (0.3<=c<1)", 0.3, 1.0), ("agent-surplus (c>=1)", 1.0, np.inf)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def regime_of(c: float) -> str:
    for name, lo, hi in REGIMES:
        if lo <= c < hi:
            return name
    raise ValueError(c)


def load_sweep(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for cfg_dir in sorted(sweep_dir.glob("A*_P*")):
        m = re.fullmatch(r"A(\d+)_P(\d+)", cfg_dir.name)
        if not m:
            continue
        n_agents, n_parcels = int(m.group(1)), int(m.group(2))
        # Revision batches may split one config across several experiment files
        # (e.g. baselines and SPPDEC arms run separately); a run name must not repeat across files.
        exp_files = sorted(cfg_dir.glob("experiment_*/experiment.json"))
        loaded = set()
        for exp_file in exp_files:
            exp = json.loads(exp_file.read_text())
            seen = Counter()
            for res in exp["results"]:
                name = res["strategy_name"]
                seen[name] += 1
                if seen[name] > 1:
                    continue  # appended rerun, not part of the sweep being analysed
                if name in loaded:
                    raise RuntimeError(f"{name} appears in more than one experiment file under {cfg_dir}")
                loaded.add(name)
                rows.append(_row(cfg_dir.name, n_agents, n_parcels, name, res, exp_file))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["complete"] = df["delivered"] >= df["P"]
    # Runs that do not deliver every parcel end differently by strategy (baselines abort as
    # "infeasible" early, SPPDEC keeps trying until it stalls or times out), so their recorded
    # makespans are not comparable. The censored column gives every such run the step limit.
    df["makespan_censored"] = df["makespan"].where(df["complete"], MAX_STEPS)
    df["regime"] = df["c"].map(regime_of)
    return df


def _row(config: str, n_agents: int, n_parcels: int, name: str, res: dict, exp_file: Path) -> dict:
    family = modernise(re.sub(r"_instance_\d+$", "", name))
    if family not in LABELS:
        raise KeyError(f"Unknown strategy family {family!r} in {exp_file}")
    met = res["metrics"]
    return {
        "config": config,
        "A": n_agents,
        "P": n_parcels,
        "c": n_agents / n_parcels,
        "strategy": LABELS[family],
        "instance": int(re.search(r"_instance_(\d+)$", name).group(1)),
        "makespan": float(met["mission_makespan_steps"]),
        "energy": float(met["total_energy_consumed"]),
        "distance": float(met["total_distance_traveled"]),
        "messages": float(met["messages_sent"]),
        "delivered": int(met["tasks_completed"] if "tasks_completed" in met else met["tasks_pending"]),
        "deaths": int(res.get("termination", {}).get("dead_total", 0)),
        "status": res.get("status"),
    }


def config_means(df: pd.DataFrame, metric: str, complete_only: bool = False) -> pd.DataFrame:
    """Config x strategy table of instance means."""
    sub = df[df["complete"]] if complete_only else df
    return sub.pivot_table(index="config", columns="strategy", values=metric, aggfunc="mean")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = stat(values[idx], axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_comparison(x: pd.Series, y: pd.Series) -> Optional[Dict]:
    """x = primary, y = comparator; both indexed by config. Negative diff = primary faster."""
    pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    n = len(pair)
    if n < 3:
        return None
    d = (pair["x"] - pair["y"]).to_numpy()
    sd = d.std(ddof=1)
    t = stats.ttest_rel(pair["x"], pair["y"])
    try:
        w_p = stats.wilcoxon(d).pvalue
    except ValueError:
        w_p = np.nan
    lo, hi = bootstrap_ci(d)
    log_r = np.log(pair["x"].to_numpy() / pair["y"].to_numpy())
    r_lo, r_hi = bootstrap_ci(log_r)
    return {
        "n_configs": n,
        "mean_primary": pair["x"].mean(),
        "mean_comparator": pair["y"].mean(),
        "mean_diff": d.mean(),
        "pct_diff": 100 * d.mean() / pair["y"].mean(),
        "diff_ci_lo": lo,
        "diff_ci_hi": hi,
        "t": t.statistic,
        "p_t": t.pvalue,
        "p_wilcoxon": w_p,
        "cohen_dz": d.mean() / sd if sd > 0 else np.nan,
        "geo_ratio": float(np.exp(log_r.mean())),
        "geo_ratio_ci_lo": float(np.exp(r_lo)),
        "geo_ratio_ci_hi": float(np.exp(r_hi)),
        "median_ratio": float(np.median(np.exp(log_r))),
        "ratio_q1": float(np.percentile(np.exp(log_r), 25)),
        "ratio_q3": float(np.percentile(np.exp(log_r), 75)),
        "share_primary_better": float(np.mean(d < 0)),
    }


def comparison_family(cm: pd.DataFrame, primary: str, comparators: List[str], configs) -> pd.DataFrame:
    rows = []
    for comp in comparators:
        if comp not in cm.columns:
            continue
        res = paired_comparison(cm.loc[configs, primary], cm.loc[configs, comp])
        if res:
            rows.append({"primary": primary, "comparator": comp, **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_t_holm"] = holm(out["p_t"].tolist())
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].fillna(1.0).tolist())
    return out


def by_scope(df: pd.DataFrame, cm: pd.DataFrame, primary: str, comparators: List[str]) -> pd.DataFrame:
    cfg_regime = df.drop_duplicates("config").set_index("config")["regime"]
    parts = []
    scopes = [("all", cfg_regime.index)] + [
        (name, cfg_regime[cfg_regime == name].index) for name, _, _ in REGIMES
    ]
    for scope, configs in scopes:
        configs = [c for c in configs if c in cm.index]
        fam = comparison_family(cm, primary, comparators, configs)
        fam.insert(0, "scope", scope)
        parts.append(fam)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def by_scope_pairwise(df: pd.DataFrame, primary: str, comparators: List[str], metric: str = "makespan") -> pd.DataFrame:
    """
    Matched complete-case comparison, matched per pair: for each comparator, only replicates in
    which BOTH the primary strategy and that comparator delivered every parcel. Each strategy is
    measured on the same task environments, and a comparator that often fails does not shrink the
    data available for the others.
    """
    cfg_regime = df.drop_duplicates("config").set_index("config")["regime"]
    scopes = [("all", list(cfg_regime.index))] + [
        (name, list(cfg_regime[cfg_regime == name].index)) for name, _, _ in REGIMES]
    have = set(df["strategy"])
    parts = []
    for scope, configs in scopes:
        rows = []
        for comp in comparators:
            if comp not in have or comp == primary:
                continue
            sub = df[df["strategy"].isin([primary, comp]) & df["config"].isin(configs)]
            ok = sub.groupby(["config", "instance"])["complete"].agg(["all", "size"])
            keys = ok[ok["all"] & (ok["size"] == 2)].index
            sub = sub[sub.set_index(["config", "instance"]).index.isin(keys)]
            cm = sub.pivot_table(index="config", columns="strategy", values=metric, aggfunc="mean")
            if primary not in cm or comp not in cm:
                continue
            res = paired_comparison(cm[primary], cm[comp])
            if res:
                rows.append({"primary": primary, "comparator": comp, "matched_replicates": len(keys), **res})
        fam = pd.DataFrame(rows)
        if not fam.empty:
            fam["p_t_holm"] = holm(fam["p_t"].tolist())
            fam["p_wilcoxon_holm"] = holm(fam["p_wilcoxon"].fillna(1.0).tolist())
            fam.insert(0, "scope", scope)
            parts.append(fam)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def table_iii_check(df: pd.DataFrame) -> pd.DataFrame:
    ms = config_means(df, "makespan")
    en = config_means(df, "energy")
    msg = config_means(df, "messages")
    rows = []
    for strat, (p_ms, p_sd, p_en, p_msg) in PAPER_TABLE_III.items():
        if strat not in ms.columns:
            continue
        rows.append({
            "strategy": strat,
            "MS_mean": ms[strat].mean(), "paper_MS_mean": p_ms,
            "MS_sd": ms[strat].std(ddof=1), "paper_MS_sd": p_sd,
            "energy": en[strat].mean(), "paper_energy": p_en,
            "messages": msg[strat].mean(), "paper_messages": p_msg,
        })
    out = pd.DataFrame(rows)
    out["MS_delta"] = out["MS_mean"] - out["paper_MS_mean"]
    out["energy_delta"] = out["energy"] - out["paper_energy"]
    return out


def completion_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("strategy")
    out = pd.DataFrame({
        "runs": g.size(),
        "complete_runs": g["complete"].sum(),
        "completion_rate": g["complete"].mean(),
        "delivered_share": g.apply(lambda s: (s["delivered"] / s["P"]).mean()),
        "runs_with_agent_loss": g.apply(lambda s: int((s["deaths"] > 0).sum())),
        "agents_lost_total": g["deaths"].sum(),
        "timeouts": g["status"].apply(lambda st: int((st == "timeout").sum())),
        "stalled": g["status"].apply(lambda st: int((st == "stalled").sum())),
        "configs_not_fully_complete": g.apply(
            lambda s: int((~s.groupby("config")["complete"].all()).sum())),
    })
    return out.sort_values("completion_rate")


def regime_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    cm = config_means(df, "makespan")
    cfg_regime = df.drop_duplicates("config").set_index("config")["regime"]
    rows = []
    for name, _, _ in REGIMES:
        configs = cfg_regime[cfg_regime == name].index
        sub = cm.loc[configs]
        for strat in sub.columns:
            rows.append({"regime": name, "n_configs": len(configs), "strategy": strat,
                         "mean_makespan": sub[strat].mean(), "median_makespan": sub[strat].median()})
    return pd.DataFrame(rows)


def fairness_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """beta on vs off, holding w-bar fixed: FO vs Ctrl (w=0), Comb vs Base (w=0.15)."""
    cm = config_means(df, "makespan")
    cfg_regime = df.drop_duplicates("config").set_index("config")["regime"]
    scopes = [("all", cfg_regime.index)] + [
        (n, cfg_regime[cfg_regime == n].index) for n, _, _ in REGIMES]
    rows = []
    for scope, configs in scopes:
        fam = []
        for on, off, label in [("S: FO", "S: Ctrl", "beta on vs off @ w=0"),
                               ("S: Comb.", "S: Base", "beta on vs off @ w=0.15"),
                               ("S: FO (rule)", "S: Ctrl (rule)", "rule arbiter: beta on vs off @ w=0"),
                               ("S: Rule", "S: Base (rule)", "rule arbiter: beta on vs off @ w=0.15"),
                               ("S: Base (rule)", "S: Ctrl (rule)", "rule arbiter: w=0.15 vs 0 @ beta=0"),
                               ("S: Rule", "S: FO (rule)", "rule arbiter: w=0.15 vs 0 @ beta=0.02")]:
            if on not in cm.columns or off not in cm.columns:
                continue
            res = paired_comparison(cm.loc[configs, on], cm.loc[configs, off])
            if res:
                fam.append({"scope": scope, "contrast": label, "beta_on": on, "beta_off": off, **res})
        if fam:
            p_adj = holm([r["p_t"] for r in fam])
            for r, p in zip(fam, p_adj):
                r["p_t_holm"] = p
            rows.extend(fam)
    return pd.DataFrame(rows)


def mixed_model(df: pd.DataFrame, primary: str = PRIMARY):
    import statsmodels.formula.api as smf

    candidates = [primary, "S: Ctrl", "S: FO", "S: Base", "S: Always-G", "S: Always-H", "S: Rule",
                  "C-Hung.", "CBBA-ST"]
    keep = list(dict.fromkeys(c for c in candidates if c in set(df["strategy"])))
    sub = df[df["strategy"].isin(keep)].copy()
    sub["log_ms"] = np.log(sub["makespan"])
    sub["logA_c"] = np.log(sub["A"]) - np.log(sub["A"]).mean()
    sub["logc_c"] = np.log(sub["c"]) - np.log(sub["c"]).mean()
    sub["strategy"] = pd.Categorical(sub["strategy"], categories=keep)
    # log c = log A - log P, so (logA, logc) spans the same space as (logA, logP)
    # while keeping the strategy x logc interaction identifiable. Both centred, so
    # strategy coefficients are gaps at the geometric-mean fleet size and load.
    model = smf.mixedlm(
        f"log_ms ~ C(strategy, Treatment('{primary}')) * logc_c + logA_c",
        sub, groups=sub["config"])
    fit = model.fit(reml=True)
    return fit


def selection_check(df: pd.DataFrame) -> pd.DataFrame:
    cm = config_means(df, "makespan")
    rows = []
    for v in [v for v in SPPDEC_VARIANTS if v in cm.columns]:
        res = paired_comparison(cm[v], cm["C-Hung."])
        rows.append({"variant": v, "mean_makespan": cm[v].mean(),
                     "diff_vs_C-Hung.": res["mean_diff"], "p_t_vs_C-Hung.": res["p_t"]})
    out = pd.DataFrame(rows).sort_values("mean_makespan")
    out["p_t_holm_across_variants"] = holm(out["p_t_vs_C-Hung."].tolist())
    return out


def energy_overhead(df: pd.DataFrame, primary: str = PRIMARY) -> pd.DataFrame:
    cm = config_means(df, "energy")
    cfg_regime = df.drop_duplicates("config").set_index("config")["regime"]
    rows = []
    for scope, configs in [("all", cfg_regime.index)] + [
            (n, cfg_regime[cfg_regime == n].index) for n, _, _ in REGIMES]:
        for comp in ["C-Hung.", "CBBA-ST"]:
            r = np.log(cm.loc[configs, primary] / cm.loc[configs, comp]).to_numpy()
            lo, hi = bootstrap_ci(r)
            rows.append({"scope": scope, "comparator": comp, "n_configs": len(configs),
                         "ratio_of_means": cm.loc[configs, primary].mean() / cm.loc[configs, comp].mean(),
                         "geo_mean_ratio": float(np.exp(r.mean())),
                         "ci_lo": float(np.exp(lo)), "ci_hi": float(np.exp(hi))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="experiments/station_scaling",
                    help="one or more sweep directories (A*_P*/experiment_*/experiment.json)", nargs="+")
    ap.add_argument("--out", default="analysis/revision/output")
    ap.add_argument("--primary", default=PRIMARY)
    ap.add_argument("--title", default="submitted sweep")
    args = ap.parse_args()
    sweeps = args.sweep if isinstance(args.sweep, list) else [args.sweep]
    primary = args.primary

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([load_sweep(Path(sw)) for sw in sweeps], ignore_index=True)
    dup = df.duplicated(["config", "strategy", "instance"])
    if dup.any():
        raise RuntimeError(f"Duplicate runs across sweeps: {df[dup].head()}")
    df.to_csv(out / "runs.csv", index=False)
    present = set(df["strategy"])

    ms = config_means(df, "makespan")
    ms_censored = config_means(df, "makespan_censored")
    # Matched complete-case: only (config, replicate) pairs in which EVERY compared strategy
    # delivered all parcels, so each strategy is measured on the same task environments.
    keyed = df.set_index(["config", "instance"])
    all_done = keyed.groupby(level=[0, 1])["complete"].agg(["all", "size"])
    matched = all_done[(all_done["all"]) & (all_done["size"] == df["strategy"].nunique())].index
    dfm = df[keyed.index.isin(matched)]
    ms_matched = config_means(dfm, "makespan")
    matched_note = (f"{len(matched)} of {all_done.shape[0]} (config, replicate) pairs are matched "
                    f"(every strategy delivered every parcel).")
    regime_counts = df.drop_duplicates("config")["regime"].value_counts()
    cmp_cols = ["scope", "comparator", "n_configs", "mean_primary", "mean_comparator", "mean_diff",
                "pct_diff", "diff_ci_lo", "diff_ci_hi", "cohen_dz", "p_t_holm", "p_wilcoxon_holm",
                "geo_ratio", "geo_ratio_ci_lo", "geo_ratio_ci_hi", "share_primary_better"]
    fair_cols = ["scope", "contrast", "n_configs", "mean_primary", "mean_comparator", "mean_diff",
                 "pct_diff", "diff_ci_lo", "diff_ci_hi", "p_t", "p_t_holm", "share_primary_better"]

    report = [
        f"# Revision statistics — {args.title}",
        "",
        f"Generated by `python -m analysis.revision.revision_stats` from {', '.join(f'`{s}`' for s in sweeps)}.",
        f"Runs: {len(df)} ({df['config'].nunique()} configs × {df['strategy'].nunique()} strategies).",
        f"Unit of analysis for inference: configuration (instance means). Negative differences mean the primary method is faster.",
        f"Primary method: **{primary}**. Bootstrap: {N_BOOT} resamples over configs, seed {RNG_SEED}.",
        "",
        "Regimes (c = agents / parcels): " + ", ".join(f"{k}: {v} configs" for k, v in regime_counts.items()),
        "",
    ]
    sections = []
    counter = [0]

    def heading(title):
        counter[0] += 1
        return f"## {counter[0]}. {title}"

    def section(title, frame, name, floatfmt="{:.3g}", note=None):
        frame.to_csv(out / f"{name}.csv", index=False)
        sections.extend([heading(title), "",
                         (note + "\n") if note else "", md_table(frame, floatfmt), ""])

    t3 = table_iii_check(df)
    if not t3.empty:
        section("Table III reproduction check", t3, "table_iii_check", "{:.1f}")
    section("Completion and agent loss", completion_table(df).reset_index(), "completion", "{:.3f}")

    if primary in present:
        section(f"{primary} vs baselines, makespan, matched complete-case (primary)",
                by_scope_pairwise(df, primary, BASELINES).reindex(columns=["matched_replicates"] + cmp_cols),
                "primary_vs_baselines",
                note="Matched per pair: only replicates in which both strategies delivered every parcel "
                     "(column matched_replicates). Holm correction within each scope across the baselines present.")
        section(f"{primary} vs baselines, makespan, censored (sensitivity)",
                by_scope(df, ms_censored, primary, BASELINES)[cmp_cols], "primary_vs_baselines_censored",
                note="Runs that did not deliver every parcel are set to the step limit for every strategy.")
        section(f"{primary} vs baselines, makespan, as recorded (sensitivity; not comparable for failed runs)",
                by_scope(df, ms, primary, BASELINES)[cmp_cols], "primary_vs_baselines_as_recorded")
        arbiters = [a for a in ["S: Always-G", "S: Always-H", "S: Rule", "S: Comb.", "S: Comb. +1s", "S: Comb. +2s"]
                    if a in present and a != primary]
        if arbiters:
            section(f"Arbitration ablation: {primary} vs alternative arbiters, makespan",
                    by_scope(df, ms, primary, arbiters)[cmp_cols], "arbiter_ablation")
            en = config_means(df, "energy")
            section(f"Arbitration ablation: {primary} vs alternative arbiters, energy",
                    by_scope(df, en, primary, arbiters)[cmp_cols], "arbiter_ablation_energy")
        if {"C-Hung.", "CBBA-ST"} <= present:
            section(f"Energy: {primary} relative to references", energy_overhead(df, primary), "energy_overhead", "{:.3f}")
        fit = mixed_model(df, primary)
        (out / "mixed_model.txt").write_text(str(fit.summary()))
        sections.extend([heading("Mixed-effects model (log makespan)"), "",
                         f"Formula: `log_ms ~ strategy * logc_c + logA_c` (centred logs; c = agents/parcels), "
                         f"random intercept per config, reference = {primary}.\n",
                         "```\n" + str(fit.summary()) + "\n```", ""])

    fair = fairness_contrasts(df)
    if not fair.empty:
        section("Fairness on/off (β = 0.02 vs 0), makespan", fair[fair_cols], "fairness_contrasts",
                note="Holm correction within each scope across the two contrasts.")
    if "C-Hung." in present:
        sel = selection_check(df)
        if not sel.empty:
            section("Every SPPDEC variant vs C-Hung.", sel, "variant_selection")

    desc = regime_descriptives(df)
    section("Mean makespan by regime",
            desc.pivot(index="strategy", columns="regime", values="mean_makespan").reset_index(),
            "regime_descriptives", "{:.0f}")

    (out / "REPORT.md").write_text("\n".join(report + sections))
    print(f"Wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
