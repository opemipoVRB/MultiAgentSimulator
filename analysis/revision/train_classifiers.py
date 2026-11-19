# analysis/revision/train_classifiers.py
"""
Train the two lightweight classifiers that stand in for the SLM arbiter, and report how well they do.

1. Regime classifier (labels from outcomes). Per (configuration, replicate): which of two fixed policies
   finished the mission faster on the same parcels and start cells, greedy or coordination (both with
   the same observation layer). Inputs known at the start of a mission: agents, parcels, agents per
   parcel. Evaluated leave-one-configuration-out: the tree used for a configuration was trained
   without that configuration. The leave-one-out trees are stored under the configuration's name.
2. Distilled tree (labels from the rule table). Decision-time inputs (uncertainty, its trend, the
   prediction-error trend, competition) labelled by the rule table the rule-based arbiter applies.
   It shows whether a tiny model can stand in for the rules.

The SLM's recorded decisions are not used as the main labels: its raw answers contradicted its rules and the
guarded answers equal the rule table. A third tree (--slm-root) imitates the SLM's raw answers anyway, as a
description of what the model actually did: it shows what the model's own behaviour is worth when run cheaply.

Usage:
    PYTHONPATH=.:src python3 -m analysis.revision.train_classifiers \
        --root experiments/revision_fixed_dt/components_final
"""
import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from analysis.revision.revision_stats import load_sweep
from src.strategies.sppdec.classifier import DecisionTree, MODELS_PATH
from src.strategies.sppdec.graph import SPPDECLangGraph

GREEDY = "S: Always-G + observation"
COORD = "S: CBBA shortcut + observation"
TREND = {"RISING": 1, "STABLE": 0, "FALLING": -1}


def regime_data(root: Path):
    df = load_sweep(root)
    df = df[df.complete]
    g = df[df.strategy == GREEDY].set_index(["config", "instance"])
    c = df[df.strategy == COORD].set_index(["config", "instance"])
    j = g[["A", "P", "makespan"]].join(c[["makespan"]], lsuffix="_g", rsuffix="_c", how="inner").dropna()
    rows = []
    for (cfg, inst), r in j.iterrows():
        rows.append({"config": cfg, "A": float(r.A), "P": float(r.P), "c": float(r.A) / float(r.P),
                     "y": int(r.makespan_c < r.makespan_g), "w": abs(float(np.log(r.makespan_g / r.makespan_c))),
                     "mk_g": float(r.makespan_g), "mk_c": float(r.makespan_c)})
    return rows


def fit_regime(rows, depth=2):
    X = [[r["c"], r["A"], r["P"]] for r in rows]
    y = [r["y"] for r in rows]
    w = [r["w"] + 1e-6 for r in rows]
    return DecisionTree(max_depth=depth, min_leaf_weight=0.05 * sum(w)).fit(X, y, ["c", "A", "P"], w)


def rule_label(b, bt, et, comp):
    policy, _ = SPPDECLangGraph._rule_based_policy(b, bt, et, comp)
    return 1 if policy == "hungarian" else 0


def distilled_data(n_random=20000, seed=3):
    rng = random.Random(seed)
    rows = []
    trends = list(TREND)
    for b in [i / 20 for i in range(21)]:
        for bt, et in itertools.product(trends, trends):
            for comp in [0.01, 0.05, 0.1, 0.15, 0.16, 0.25, 0.5, 1, 2, 5, 10, 20]:
                rows.append((b, bt, et, comp))
    for _ in range(n_random):
        rows.append((rng.random(), rng.choice(trends), rng.choice(trends), 10 ** rng.uniform(-2, 1.4)))
    X = [[b, TREND[bt], TREND[et], comp] for b, bt, et, comp in rows]
    y = [rule_label(*r) for r in rows]
    return X, y


def slm_raw_data(root: Path, arm: str = "sppdec_combined_qwen3vl_paired"):
    """Decision contexts and the model's raw answers (before the decision guard) from a batch."""
    X, y, rule_y = [], [], []
    for f in root.glob(f"*/experiment_*/{arm}_instance_*/agents.json"):
        for ag in json.loads(f.read_text()):
            for e in ag.get("slm_decisions", []):
                if not (e.get("gate") or "").startswith("slm"):
                    continue
                ans = (e.get("check") or {}).get("answer_policy")
                c = e["context"]
                if ans not in ("greedy", "hungarian") or c.get("belief_trend") not in TREND:
                    continue
                row = [c["belief_infeasibility"], TREND[c["belief_trend"]], TREND[c["error_trend"]], c["competition_ratio"]]
                X.append(row)
                y.append(1 if ans == "hungarian" else 0)
                rule_y.append(rule_label(c["belief_infeasibility"], c["belief_trend"], c["error_trend"], c["competition_ratio"]))
    return X, y, rule_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=str(MODELS_PATH))
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--slm-root", default=None, help="batch with an SLM arm, to imitate its raw answers")
    a = ap.parse_args()

    rows = regime_data(Path(a.root))
    configs = sorted({r["config"] for r in rows})
    print(f"regime data: {len(rows)} (configuration, replicate) pairs over {len(configs)} configurations")
    print("coordination faster in", f"{100 * np.mean([r['y'] for r in rows]):.0f}% of pairs")

    models = {"regime": {}, "distilled": None}
    chosen_mk, best_mk, g_mk, c_mk, correct, total = [], [], [], [], 0.0, 0.0
    for cfg in configs:
        train = [r for r in rows if r["config"] != cfg]
        test = [r for r in rows if r["config"] == cfg]
        tree = fit_regime(train, a.depth)
        models["regime"][cfg] = tree.to_json()
        for r in test:
            p = tree.predict({"c": r["c"], "A": r["A"], "P": r["P"]})
            chosen_mk.append(r["mk_c"] if p else r["mk_g"])
            best_mk.append(min(r["mk_g"], r["mk_c"]))
            g_mk.append(r["mk_g"])
            c_mk.append(r["mk_c"])
            correct += (p == r["y"]) * r["w"]
            total += r["w"]
    models["regime"]["all"] = fit_regime(rows, a.depth).to_json()
    gm = lambda x, y: float(np.exp(np.mean(np.log(np.array(x) / np.array(y)))))
    print("\nLeave-one-configuration-out, makespan relative to the best of the two fixed policies per replicate:")
    print(f"  regime classifier  {gm(chosen_mk, best_mk):.3f}")
    print(f"  always coordinate  {gm(c_mk, best_mk):.3f}")
    print(f"  always greedy      {gm(g_mk, best_mk):.3f}")
    print(f"  weighted accuracy  {100 * correct / total:.1f}%")
    print("\nTree trained on all configurations (for inspection):")
    print(DecisionTree.from_json(models["regime"]["all"]).describe())

    X, y = distilled_data()
    cut = int(0.8 * len(X))
    idx = list(range(len(X)))
    random.Random(1).shuffle(idx)
    tr, te = idx[:cut], idx[cut:]
    tree = DecisionTree(max_depth=6, min_leaf_weight=1.0).fit([X[i] for i in tr], [y[i] for i in tr],
                                                              ["belief", "belief_trend", "error_trend", "competition"])
    acc = np.mean([tree.predict(dict(zip(tree.features, X[i]))) == y[i] for i in te])
    full = DecisionTree(max_depth=6, min_leaf_weight=1.0).fit(X, y, ["belief", "belief_trend", "error_trend", "competition"])
    acc_all = np.mean([full.predict(dict(zip(full.features, X[i]))) == y[i] for i in range(len(X))])
    print(f"\nDistilled tree (labels from the rule table): held-out agreement with the rules {100 * acc:.2f}% "
          f"({len(te)} contexts); agreement of the stored tree over all {len(X)} contexts {100 * acc_all:.2f}%")
    models["distilled"] = full.to_json()
    if a.slm_root:
        Xs, ys, rs = slm_raw_data(Path(a.slm_root))
        feats = ["belief", "belief_trend", "error_trend", "competition"]
        order = list(range(len(Xs)))
        random.Random(2).shuffle(order)
        k = int(0.8 * len(order))
        t = DecisionTree(max_depth=6, min_leaf_weight=1.0).fit([Xs[i] for i in order[:k]], [ys[i] for i in order[:k]], feats)
        held = np.mean([t.predict(dict(zip(feats, Xs[i]))) == ys[i] for i in order[k:]])
        compliance = np.mean([ys[i] == rs[i] for i in range(len(ys))])
        imit = DecisionTree(max_depth=6, min_leaf_weight=1.0).fit(Xs, ys, feats)
        agree_rules = np.mean([imit.predict(dict(zip(feats, X[i]))) == y[i] for i in range(len(X))])
        print(f"\nSLM imitation: {len(Xs)} consulted decisions; the model's raw answers agree with the rule table "
              f"{100 * compliance:.1f}% of the time; a tree imitates its raw answers with held-out agreement {100 * held:.1f}%; "
              f"that tree agrees with the rule table on {100 * agree_rules:.1f}% of all contexts")
        models["slm_imitation"] = imit.to_json()
    Path(a.out).write_text(json.dumps(models, separators=(",", ":")))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
