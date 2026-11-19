# analysis/revision/awareness_evaluation.py
"""
What station awareness is worth to each strategy (batch: experiments/marl/evaluate_awareness.sh). Levels: a0 none (the original arm), a1 station memory
refreshed by sensing, a2 stop when the live station count says every parcel is delivered, a3 memory and stop, a4 predictive memory (rate over the last four full views) and stop, a5 the same with the average rate since the start (the tuned version). Per strategy, config and sensing: mean makespan, last
delivery and tail over the replicates in which every arm delivered everything (matched complete-case), the ratio to a0 (paired, 95% interval), and completion counts.
Usage: python -m analysis.revision.awareness_evaluation <batch dir>
"""
import math
import statistics as st
import sys

from analysis.revision.marl_evaluation import load

STRATEGIES = ["sppdec", "naive_decentralized", "decentralized_auction", "cbba_full", "cbba_single_task"]
LEVELS = ["a0", "a1", "a2", "a3", "a4", "a5"]
T95 = {15: 2.131, 14: 2.145, 13: 2.160, 12: 2.179, 11: 2.201, 10: 2.228, 9: 2.262, 8: 2.306}


def main(root):
    rec = load(root)
    for tag in ("full", "noisy"):
        for cfg in ("A4_P36", "A8_P24", "A12_P24"):
            print(f"\n== {cfg}, sensing {tag}")
            for strat in STRATEGIES:
                arms = {l: f"{strat}_aw_{tag}_{l}" if strat != "sppdec" else f"aw_{tag}_{l}" for l in LEVELS}
                arms = {l: a for l, a in arms.items() if any(a in rec[k] for k in rec if k[0] == cfg)}     # the levels this batch ran
                keys = [k for k in rec if k[0] == cfg and all(a in rec[k] for a in arms.values())]
                done = [k for k in keys if all(rec[k][a]["delivered"] == rec[k][a]["parcels"] for a in arms.values())]
                if not done:
                    print(f"   {strat:24s} no complete replicates ({len(keys)} run)")
                    continue
                cells = []
                for l in LEVELS:
                    ms = st.mean(rec[k][arms[l]]["makespan"] for k in done)
                    tl = ms - st.mean(rec[k][arms[l]]["last"] for k in done)
                    if l == "a0":
                        cells.append(f"a0 {ms:6.0f} (tail {tl:4.0f})")
                        continue
                    d = [rec[k][arms[l]]["makespan"] / rec[k][arms["a0"]]["makespan"] for k in done]
                    mu = st.mean(d)
                    hw = T95.get(len(d) - 1, 2.0) * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
                    cells.append(f"{l} {ms:6.0f} (tail {tl:4.0f}) x{mu:.3f}+-{hw:.3f}")
                print(f"   {strat:24s} n={len(done)}/{len(keys)} | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1])
