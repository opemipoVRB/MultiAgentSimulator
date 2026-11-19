# analysis/revision/grid_bootstrap_mixed.py
"""Bootstrap intervals and a mixed-effects model for the full-grid comparison (R1.7, R3.6).
Input: per-mission summary CSV from batch_summary.py. Matched complete-case, baseline minus SPPDEC in last delivery.
(1) Cluster bootstrap over configurations (10,000 resamples of configs; each config contributes its mean paired difference).
(2) Mixed model: difference ~ agents + parcels + agents/parcels, random intercept per config.
Usage: python -m analysis.revision.grid_bootstrap_mixed <summary csv> <out txt>"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

S = "sppdec_consensus_rule_nogamble_limited"
B = {"Centralised": "centralized_cc_hungarian_hungarian", "CBBA single": "cbba_single_task_cbba_single", "CBBA capability bundle": "cbba_full_capability_bundle",
     "CBBA bundle": "cbba_full_cbba_bundle", "MARL": "marl_ippo_limited", "Naive": "naive_decentralized_reactive", "Auction": "decentralized_auction_auction_baseline"}


def main(csv, out):
    d = pd.read_csv(csv)
    d["ok"] = d.delivered == d.parcels
    rng = np.random.default_rng(0)
    lines = []
    for metric in ("last_delivery", "makespan", "energy"):
        lines.append(f"\n=== {metric}: baseline minus SPPDEC, matched complete-case ===")
        w = d.pivot_table(index=["config", "replicate"], columns="arm", values=metric)
        ok = d.pivot_table(index=["config", "replicate"], columns="arm", values="ok", aggfunc="min")
        for nm, a in B.items():
            keep = ok[S].astype(bool) & ok[a].astype(bool)
            diff = (w[a] - w[S])[keep].dropna().reset_index(name="d")
            per = diff.groupby("config").d.mean()
            boots = [per.sample(len(per), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            line = f"  {nm:24s} configs {len(per):2d}  mean {per.mean():+8.1f}  bootstrap 95% CI [{lo:+8.1f}, {hi:+8.1f}]"
            if metric == "last_delivery":
                diff["A"] = diff.config.str.extract(r"A(\d+)_").astype(float)
                diff["P"] = diff.config.str.extract(r"_P(\d+)").astype(float)
                diff["c"] = diff.A / diff.P
                try:
                    m = smf.mixedlm("d ~ A + P + c", diff, groups=diff["config"]).fit(reml=True)
                    line += f" | mixed: A {m.params['A']:+.2f} (p {m.pvalues['A']:.2g}), P {m.params['P']:+.2f} (p {m.pvalues['P']:.2g}), A/P {m.params['c']:+.1f} (p {m.pvalues['c']:.2g})"
                except Exception as e:
                    line += f" | mixed model failed: {type(e).__name__}"
            lines.append(line)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
