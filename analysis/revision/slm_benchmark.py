# analysis/revision/slm_benchmark.py
"""
Replay benchmark for the SPPDEC arbitration SLM (tickets T2.3, T2.6, T2.9).

Replays logged SLM-decided arbitration cases through any Ollama model with the
simulator's exact prompt builder and decoding settings, and measures what a
shared arbiter needs:

  - rule agreement : does the model follow the prompt's own DECISION RULES?
  - determinism    : does an identical prompt give the same decision on repeat?
                     (text identity is reported separately; only the decision matters
                     for a shared arbiter)
  - valid output   : is a policy parseable from the reply?
  - latency        : seconds per call (and tokens generated)

Cases come from `agents.json` slm_decisions with gate == "slm". Runs recorded
before prompt inputs were logged are reconstructed: peers = agents - 1 (every
peer is visible at infinite radius), tasks = round(peers / competition_ratio),
five-cycle histories rebuilt from the logged trend labels. Newer runs carry
`prompt_inputs` and are replayed exactly.

Usage:
  python -m analysis.revision.slm_benchmark --models llama3:8b qwen2.5:7b --cases 120 --repeats 20
"""
import argparse
import glob
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

from src.strategies.sppdec.graph import SPPDECLangGraph, arbitration_output_schema, build_arbitration_prompt

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
rule_policy = SPPDECLangGraph._rule_based_policy


def cost_diff_label(greedy_cost: float, hungarian_cost: float) -> str:
    """Same wording as SPPDECLangGraph.reason_with_slm."""
    cost_diff = hungarian_cost - greedy_cost
    cost_diff_pct = abs(cost_diff) / max(greedy_cost, 1e-6) * 100
    if abs(cost_diff) < 1e-6:
        return "identical — no coordination benefit available"
    if cost_diff > 0:
        return (f"Hungarian costs {cost_diff:+.2f} more ({cost_diff_pct:.0f}% premium) "
                f"— you sacrifice personal efficiency for team benefit")
    return (f"Hungarian costs {cost_diff:+.2f} less ({cost_diff_pct:.0f}% saving) "
            f"— coordination is strictly better")


def history(value: float, trend: str):
    step = {"RISING": -0.15, "FALLING": 0.15}.get(trend, 0.0)
    return [f"{value + step * (4 - i) / 4:.2f}" for i in range(5)]


def load_cases(sweep: Path, max_file_mb: float = 400):
    cases = []
    for f in glob.glob(str(sweep / "*/experiment_*/*/agents.json")):
        if os.path.getsize(f) > max_file_mb * 1e6:
            continue
        n_agents = int(re.search(r"/A(\d+)_P\d+/", f).group(1))
        for agent in json.load(open(f)):
            for d in agent.get("slm_decisions", []):
                if d.get("gate") != "slm":
                    continue
                c = d["context"]
                if d.get("prompt_inputs"):
                    inputs, exact = d["prompt_inputs"], True
                else:
                    peers = n_agents - 1
                    ratio = c["competition_ratio"]
                    g, h = d["costs"]["greedy"], d["costs"]["hungarian"]
                    inputs = dict(
                        greedy_choice={"parcel_id": "p_greedy", "cost": g},
                        hungarian_choice={"parcel_id": "p_hungarian", "cost": h},
                        cost_diff_label=cost_diff_label(g, h),
                        belief_infeasibility=c["belief_infeasibility"],
                        energy_error=c["energy_error"],
                        num_peers=peers,
                        num_tasks=max(1, round(peers / ratio)) if ratio > 0 else peers,
                        competition_ratio=ratio,
                        belief_trend=c["belief_trend"],
                        belief_history=history(c["belief_infeasibility"], c["belief_trend"]),
                        error_trend=c["error_trend"],
                        error_history=history(c["energy_error"], c["error_trend"]),
                        competition_trend=c["competition_trend"],
                        competition_history=history(ratio, c["competition_trend"]),
                    )
                    exact = False
                expected, rule = rule_policy(c["belief_infeasibility"], c["belief_trend"],
                                             c["error_trend"], c["competition_ratio"])
                cases.append({"inputs": inputs, "expected": expected, "rule": rule,
                              "logged_policy": d["policy"], "exact": exact})
    return cases


def stratified_sample(cases, n, seed=7):
    rng = random.Random(seed)
    by_rule = defaultdict(list)
    for c in cases:
        by_rule[c["rule"]].append(c)
    per = max(1, n // len(by_rule))
    sample = []
    for rule, group in sorted(by_rule.items()):
        rng.shuffle(group)
        sample.extend(group[:per])
    return sample


def parse_policy(text: str):
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        pol = str(obj.get("policy", "")).strip().lower()
        if pol in ("greedy", "hungarian"):
            return pol
        if obj.get("p") in ("G", "H"):  # compact styles
            return "greedy" if obj["p"] == "G" else "hungarian"
    return None


def expected_fields(inputs: dict) -> dict:
    """Correct value of every field any prompt style may ask for."""
    from src.strategies.sppdec.graph import _derived_facts
    band, trend, comp = _derived_facts(inputs["belief_infeasibility"], inputs["belief_trend"],
                                       inputs["error_trend"], inputs["competition_ratio"])
    s1 = {"above_0.7": "G", "below_0.3": "H"}.get(band, "next")
    s2 = "skip" if s1 != "next" else {"either_rising": "G", "both_falling": "H"}.get(trend, "next")
    s3 = "skip" if s2 != "next" else ("H" if comp == "yes" else "G")
    s12 = s1 if s1 != "next" else s2
    long = {"G": "GREEDY", "H": "HUNGARIAN", "next": None, "skip": "not_reached"}
    return {"uncertainty_band": band, "trend_case": trend, "competition_above_0.15": comp,
            "step1_outcome": long[s1] or "go_to_step_2", "step2_outcome": long[s2] or "go_to_step_3",
            "step3_outcome": long[s3], "s1": s1, "s2": s2, "s3": s3, "s12": s12}


def step_check(text: str, inputs: dict) -> dict:
    """Share of the structured fields in the answer that are correct for these inputs."""
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return {}
    exp = expected_fields(inputs)
    fields = [k for k in obj if k in exp]
    if not fields:
        return {}
    return {"fields_correct": sum(obj[k] == exp[k] for k in fields) / len(fields)}


def call(model: str, prompt: str, timeout: int = 120, schema=None):
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "top_p": 0.1, "num_predict": 256}}
    if schema:
        payload["format"] = schema  # same constrained decoding as the simulator
    if model.startswith("qwen3"):
        payload["think"] = False  # arbitration answer only, no hidden reasoning trace
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    call.last_timing = {"prompt_eval_s": data.get("prompt_eval_duration", 0) / 1e9,
                        "generate_s": data.get("eval_duration", 0) / 1e9,
                        "prompt_tokens": data.get("prompt_eval_count")}
    return data.get("response", ""), time.time() - t0, data.get("eval_count")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="experiments/revision_fixed_dt/sppdec_combined")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--repeats", type=int, default=20, help="cases re-asked to test determinism")
    ap.add_argument("--sample-seed", type=int, default=7, help="change to draw a fresh validation sample")
    ap.add_argument("--prompt-version", default=None,
                    help="prompt style: submitted, cost_gap_closed, stepwise, stepwise_compact, paired_steps, single_answer")
    ap.add_argument("--out", default="analysis/revision/output_slm_benchmark")
    args = ap.parse_args()

    all_cases = load_cases(Path(args.sweep))
    freq = Counter(c["rule"] for c in all_cases)
    print(f"logged SLM cases by rule: {dict(freq)}")
    cases = stratified_sample(all_cases, args.cases, seed=args.sample_seed)
    print(f"{len(cases)} cases; rules: {dict(Counter(c['rule'] for c in cases))}; "
          f"exact prompts: {sum(c['exact'] for c in cases)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in args.models:
        call(model, "Reply with OK.")  # load weights before timing
        for i, c in enumerate(cases):
            inputs = {k: v for k, v in c["inputs"].items() if k != "prompt_version"}
            prompt = build_arbitration_prompt(**inputs, prompt_version=args.prompt_version)
            schema = arbitration_output_schema(args.prompt_version)
            text, secs, toks = call(model, prompt, schema=schema)
            pol = parse_policy(text)
            steps = step_check(text, c["inputs"]) if schema else {}
            repeat_same = repeat_text = None
            if i < args.repeats:
                text2, _, _ = call(model, prompt, schema=schema)
                repeat_same = parse_policy(text2) == pol
                repeat_text = text2 == text
            rows.append({"model": model, "case": i, "rule": c["rule"], "expected": c["expected"],
                         "policy": pol, "agrees": pol == c["expected"], "valid": pol is not None,
                         "repeat_same_decision": repeat_same, "repeat_same_text": repeat_text,
                         "latency_s": secs, "tokens": toks, **steps, **call.last_timing})
        m = pd.DataFrame([r for r in rows if r["model"] == model])
        print(f"{model:32s} agree {m.agrees.mean():.0%}  valid {m.valid.mean():.0%}  "
              f"same decision on repeat {m.repeat_same_decision.dropna().mean():.0%}  "
              f"latency p50 {m.latency_s.median():.2f}s  tokens p50 {m.tokens.median():.0f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "calls.csv", index=False)
    summary = df.groupby("model").agg(
        rule_agreement=("agrees", "mean"), valid_output=("valid", "mean"),
        same_decision_on_repeat=("repeat_same_decision", lambda s: s.dropna().mean()),
        same_text_on_repeat=("repeat_same_text", lambda s: s.dropna().mean()),
        latency_p50_s=("latency_s", "median"), latency_p90_s=("latency_s", lambda s: s.quantile(0.9)),
        tokens_p50=("tokens", "median"))
    by_rule = df.pivot_table(index="model", columns="rule", values="agrees", aggfunc="mean")
    # Agreement weighted by how often each rule case occurs in the simulation
    weights = pd.Series(freq) / sum(freq.values())
    summary["rule_agreement_sim_weighted"] = by_rule[weights.index.intersection(by_rule.columns)].mul(
        weights, axis=1).sum(axis=1) / weights[weights.index.intersection(by_rule.columns)].sum()
    for col in ("fields_correct", "prompt_eval_s", "generate_s", "prompt_tokens"):
        if col in df:
            summary[col] = df.groupby("model")[col].median() if col != "fields_correct" else df.groupby("model")[col].mean()
    summary.join(by_rule).to_csv(out / "summary.csv")
    print(summary.join(by_rule).round(2).to_string())


if __name__ == "__main__":
    main()
