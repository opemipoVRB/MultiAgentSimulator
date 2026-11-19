# analysis/revision/fresh_grid_analysis.py
"""
Fresh-seed full-grid comparison, analysed as the protocol states (.ignore/revision/fresh_grid_protocol.md).
Primary: last delivery, matched complete-case, paired baseline-minus-SPPDEC differences averaged within a config (each config counts once),
paired t-test across configs with a Wilcoxon signed-rank check, Holm correction over the five baselines.
Usage: python -m analysis.revision.fresh_grid_analysis [baseline batch dir] [SPPDEC batch dir: its SPPDEC arm replaces the one in the first]
"""
import collections
import statistics as st
import sys

from scipy import stats

from analysis.revision.marl_evaluation import load

S = "consensus_rule_nogamble_limited"
BASE = [("Centralised", "centralized_cc_hungarian_hungarian"), ("CBBA single", "cbba_single_task_cbba_single"), ("CBBA bundle", "cbba_full_cbba_bundle"),
        ("Auction", "decentralized_auction_auction_baseline"), ("Naive", "naive_decentralized_reactive"),
        ("CBBA capability bundle", "cbba_full_capability_bundle"), ("MARL", "marl_ippo_limited")]


def complete(r):
    return r["delivered"] == r["parcels"]


def regime(cfg):
    a, p = int(cfg.split("_")[0][1:]), int(cfg.split("_P")[1])
    return "low (<=0.25)" if a / p <= 0.25 else "middle (0.25-0.6)" if a / p <= 0.6 else "high (>0.6)"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def paired(R, a, metric, cfgs):
    """Per-config means of (baseline - SPPDEC) over matched complete instances; returns {cfg: (mean difference, n instances)} and the instance-level list."""
    per, inst = {}, []
    for cfg in cfgs:
        ks = [k for k in R if k[0] == cfg and S in R[k] and a in R[k] and complete(R[k][S]) and complete(R[k][a])]
        if ks:
            d = [R[k][a][metric] - R[k][S][metric] for k in ks]
            per[cfg] = (st.mean(d), len(d)); inst += d
    return per, inst


def main(root="experiments/peer_model/fresh_grid", sppdec_root=None):
    R = load(root)
    if sppdec_root:
        for k, v in load(sppdec_root).items():
            if k in R and S in v:
                R[k][S] = v[S]
    cfgs = sorted({k[0] for k in R}, key=lambda c: (int(c.split("_")[0][1:]), int(c.split("_P")[1])))
    print(f"{len(cfgs)} configs, {len(R)} instances, strategies: {sorted({a for v in R.values() for a in v})}")
    print("\ncompletion (instances in which every parcel was delivered, of", len(R), ")")
    for nm, a in [("SPPDEC", S)] + BASE:
        print(f"  {nm:12s} {sum(complete(R[k][a]) for k in R if a in R[k])}")
    for metric, label in (("last", "LAST DELIVERY (primary)"), ("makespan", "MAKESPAN"), ("energy", "FLEET ENERGY")):
        print(f"\n=== {label}: baseline minus SPPDEC (positive = SPPDEC lower), matched complete-case ===")
        rows = []
        for nm, a in BASE:
            per, inst = paired(R, a, metric, cfgs)
            d = [v[0] for v in per.values()]
            t, p = stats.ttest_1samp(d, 0.0); w = stats.wilcoxon(d).pvalue if any(x != 0 for x in d) else 1.0
            ti, pi = stats.ttest_1samp(inst, 0.0)
            rows.append((nm, st.mean(d), 1.96 * st.stdev(d) / len(d) ** 0.5, len(d), len(inst), sum(x > 0 for x in d), p, w, pi))
        ph = holm([r[6] for r in rows])
        for r, h in zip(rows, ph):
            print(f"  {r[0]:12s} mean over {r[3]} configs {r[1]:+7.1f} +/- {r[2]:5.1f} | SPPDEC lower in {r[5]}/{r[3]} configs | n instances {r[4]} | p(t) {r[6]:.2g}  Holm {h:.2g} | Wilcoxon {r[7]:.2g} | instance-level p {r[8]:.2g}")
    print("\n=== by load regime (agents per parcel), last delivery: baseline minus SPPDEC, mean over configs in the regime ===")
    for reg in ("low (<=0.25)", "middle (0.25-0.6)", "high (>0.6)"):
        sub = [c for c in cfgs if regime(c) == reg]
        line = []
        for nm, a in BASE:
            per, _ = paired(R, a, "last", sub)
            d = [v[0] for v in per.values()]
            line.append(f"{nm} {st.mean(d):+6.1f} ({sum(x > 0 for x in d)}/{len(d)})")
        print(f"  {reg:18s} {len(sub):2d} configs | " + " | ".join(line))
    print("\n=== per config: mean last delivery (SPPDEC | best baseline | its name), instances in which SPPDEC was not faster than the best baseline ===")
    worse = []
    for cfg in cfgs:
        ks = [k for k in R if k[0] == cfg and all(a in R[k] and complete(R[k][a]) for a in [S] + [b for _, b in BASE])]
        if not ks:
            continue
        means = {nm: st.mean(R[k][a]["last"] for k in ks) for nm, a in [("SPPDEC", S)] + BASE}
        bn = min((n for n in means if n != "SPPDEC"), key=lambda n: means[n])
        slower = sum(R[k][S]["last"] > min(R[k][a]["last"] for _, a in BASE) for k in ks)
        print(f"  {cfg:8s} n={len(ks)} {means['SPPDEC']:6.0f} | {means[bn]:6.0f} | {bn:12s} | SPPDEC not fastest in {slower}/{len(ks)}")
        if means["SPPDEC"] > means[bn]:
            worse.append(cfg)
    print("\nconfigs in which SPPDEC's mean last delivery is above the best baseline's:", worse or "none")


if __name__ == "__main__":
    main(*sys.argv[1:3])
