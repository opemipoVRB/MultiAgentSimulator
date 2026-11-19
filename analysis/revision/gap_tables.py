# analysis/revision/gap_tables.py
"""
Tables for the gap-closing batches (experiments/peer_model/gaps/*; per-mission summaries in data/batch_summaries/gap_<batch>.csv).

For one batch and one tag (the condition) the table gives, per strategy: missions completed, agents lost, mean last delivery of the strategy under the condition against its own
reference run (the same missions without the condition; % later), and the lead of SPPDEC over it under the condition (% sooner; steps; standard error), matched complete-case over the
missions all listed strategies completed. A strategy that has no arm for the tag (it does not read what the tag degrades) is read at its reference.

Usage: python -m analysis.revision.gap_tables <batch name> [tag ...]       e.g.  sensing_all n1 d120      radio_lossy50      unequal_batteries ub
       Tag "ref" is the unstressed arm. For batches where every strategy runs under one environment (radio_*, dynamic, obstacles_*, map_*), the arms have no tag: use "ref" and compare batches.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

SUMMARY = "data/batch_summaries/gap_{}.csv"
BASES = [("S", "sppdec_consensus_rule_nogamble_limited"), ("H", "centralized_cc_hungarian"), ("C", "cbba_single_task_cbba_single"), ("CB", "cbba_full_cbba_bundle"),
         ("CK", "cbba_full_capability_bundle"), ("A", "decentralized_auction_auction_baseline"), ("N", "naive_decentralized_reactive"), ("M", "marl_ippo_limited")]
NAMES = {"S": "SPPDEC", "H": "Centralised reference", "C": "CBBA single", "CB": "CBBA bundle", "CK": "CBBA capability bundle", "A": "Auction", "N": "Naive", "M": "MARL"}


def split_arm(arm):
    if arm.startswith("centralized_cc_hungarian"):
        rest = arm[len("centralized_cc_hungarian"):]
        if rest.endswith("_hungarian"):
            rest = rest[: -len("_hungarian")]
        return "H", rest.lstrip("_") or "ref"
    for short, base in BASES:
        if arm == base:
            return short, "ref"
        if arm.startswith(base + "_"):
            return short, arm[len(base) + 1:]
    return None, arm


def load(batch):
    out = defaultdict(dict)       # (config, replicate) -> {(short, tag): row}
    for r in csv.DictReader(open(SUMMARY.format(batch))):
        s, t = split_arm(r["arm"])
        if s is None:
            continue
        out[(r["config"], r["replicate"])][(s, t)] = r
    return out


def done(r):
    return r is not None and r["last_delivery"] not in ("", "None") and int(r["delivered"]) == int(r["parcels"])


def table(batch, tag, strategies=None):
    data = load(batch)
    shorts = strategies or [s for s, _ in BASES]
    def get(v, s, t):
        return v.get((s, t)) or v.get((s, "ref"))
    keys = [k for k, v in data.items() if all(done(get(v, s, tag)) for s in shorts)]
    print(f"\n== {batch}, condition '{tag}': {len(keys)} missions completed by all of {len(shorts)} strategies ({len(data)} missions run)")
    print(f"{'strategy':24s} {'done':>5s} {'lost':>5s} {'last delivery':>14s} {'vs own ref':>11s} {'SPPDEC lead':>12s} {'steps (se)':>14s}")
    for s in shorts:
        rows = [get(v, s, tag) for v in data.values()]
        nd = sum(done(r) for r in rows)
        lost = sum(int(r["dead"] or 0) for r in rows if r)
        mean_last = np.mean([float(get(data[k], s, tag)["last_delivery"]) for k in keys]) if keys else float("nan")
        own = ""
        ref_keys = [k for k in keys if done(data[k].get((s, "ref")))]
        if tag != "ref" and ref_keys and (s, tag) in next(iter(data.values())):
            a = np.mean([float(data[k][(s, tag)]["last_delivery"]) for k in ref_keys])
            b = np.mean([float(data[k][(s, "ref")]["last_delivery"]) for k in ref_keys])
            own = f"{100 * (a / b - 1):+.1f}%"
        lead = steps = ""
        if s != "S" and keys and "S" in shorts:
            pct = [100 * (1 - float(get(data[k], "S", tag)["last_delivery"]) / float(get(data[k], s, tag)["last_delivery"])) for k in keys]
            d = [float(get(data[k], s, tag)["last_delivery"]) - float(get(data[k], "S", tag)["last_delivery"]) for k in keys]
            lead = f"{np.mean(pct):+.1f}%"
            steps = f"{np.mean(d):+.1f} ({np.std(d, ddof=1) / np.sqrt(len(d)):.1f})" if len(d) > 1 else f"{np.mean(d):+.1f}"
        print(f"{NAMES[s]:24s} {nd:5d} {lost:5d} {mean_last:14.1f} {own:>11s} {lead:>12s} {steps:>14s}")


if __name__ == "__main__":
    batch = sys.argv[1]
    tags = sys.argv[2:] or ["ref"]
    for t in tags:
        table(batch, t)
