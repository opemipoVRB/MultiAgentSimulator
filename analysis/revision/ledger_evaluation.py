# analysis/revision/ledger_evaluation.py
"""
Matched evaluation of the ledger probe (design: .ignore/revision/marl_baseline_design.md section 9). Per config and sensing: makespan, step of the last delivery and tail
(mean over the held-out replicates) for the plain rule list, coded territory, each with the ledger, and the MAPPO baseline (mean over its five training seeds); paired ratios;
the pre-registered criteria. Usage: python -m analysis.revision.ledger_evaluation <batch dir>
"""
import math
import statistics as st
import sys

from analysis.revision.marl_evaluation import interval, load


def main(root):
    rec = load(root)
    verdict = []
    for cfg in ("A4_P36", "A8_P24", "A12_P24"):
        for tag in ("full", "noisy"):
            arms = {"plain": f"channel_{tag}", "territory": f"territory_{tag}", "ledger": f"ledger_{tag}", "ledger+territory": f"ledger_territory_{tag}"}
            marl = [f"marl_ippo_s{k}_{tag}" for k in range(5)]
            keys = [k for k in rec if k[0] == cfg and all(a in rec[k] for a in arms.values()) and all(m in rec[k] for m in marl)]
            if not keys:
                continue
            print(f"\n== {cfg} {tag}: {len(keys)} replicates")
            def mean(arm, f):
                return st.mean(rec[k][arm][f] for k in keys)
            for name, arm in arms.items():
                print(f"   {name:18s} makespan {mean(arm, 'makespan'):7.0f}  last delivery {mean(arm, 'last'):6.0f}  tail {mean(arm, 'makespan') - mean(arm, 'last'):5.0f}  all delivered {sum(rec[k][arm]['delivered'] == rec[k][arm]['parcels'] for k in keys)}/{len(keys)}")
            ml = st.mean(mean(m, "makespan") for m in marl)
            ll = st.mean(mean(m, "last") for m in marl)
            print(f"   {'MARL (5 seeds)':18s} makespan {ml:7.0f}  last delivery {ll:6.0f}  tail {ml - ll:5.0f}")
            def ratio(a, b):
                d = [rec[k][a]["makespan"] / rec[k][b]["makespan"] for k in keys]
                m = st.mean(d)
                se = st.stdev(d) / math.sqrt(len(d))
                return m, 2.131 * se, sum(x < 1 for x in d)
            for a, b, label in ((arms["ledger"], arms["plain"], "ledger / plain rules"), (arms["ledger+territory"], arms["territory"], "ledger+territory / territory"),
                                (arms["ledger+territory"], arms["plain"], "ledger+territory / plain rules")):
                m, hw, f = ratio(a, b)
                print(f"      {label:32s}: {m:.3f} +- {hw:.3f}  (faster in {f} of {len(keys)})")
            for a, label in ((arms["ledger"], "ledger"), (arms["ledger+territory"], "ledger+territory")):
                per_seed = [st.mean(rec[k][a]["makespan"] / rec[k][m]["makespan"] for k in keys) for m in marl]
                mu, hw = interval(per_seed)
                print(f"      {label:32s} / MARL baseline: {mu:.3f} [{mu - hw:.3f}, {mu + hw:.3f}] over the 5 seeds")
            verdict.append((cfg, tag, {n: (mean(a, 'makespan') - mean(a, 'last')) for n, a in arms.items()},
                            ratio(arms["ledger"], arms["plain"])[0], ratio(arms["ledger+territory"], arms["territory"])[0]))
    print("\n== Pre-registered criteria")
    for cfg, tag, tails, r_plain, r_terr in verdict:
        if tag == "noisy":
            ok_tail = tails["ledger"] <= 250 and tails["ledger+territory"] <= 250
            need = 0.85 if cfg != "A4_P36" else 1.0
            ok_ratio = r_plain <= need and r_terr <= need
            print(f"   {cfg} noisy: tails {tails['ledger']:.0f} / {tails['ledger+territory']:.0f} (need <= 250): {'PASS' if ok_tail else 'FAIL'}; ratios {r_plain:.3f} / {r_terr:.3f} (need <= {need}): {'PASS' if ok_ratio else 'FAIL'}")
        else:
            ok = r_plain <= 1.02 and r_terr <= 1.02
            print(f"   {cfg} full:  ratios {r_plain:.3f} / {r_terr:.3f} (need <= 1.02): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
