# analysis/revision/slm_exhaustive.py
"""
Exhaustive rule-following test for the compact arbitration prompt styles.

In these styles the model sees only three facts computed in code (uncertainty band, trend case,
competition), so the model's whole input space is 10 distinct prompts. Testing all 10 replaces
sampling: with temperature 0, a style that is right on all 10 is right on every input.

Usage: python -m analysis.revision.slm_exhaustive --models qwen3-vl:8b-instruct --styles paired_steps ...
"""
import argparse
import itertools
import json

import requests

from src.strategies.sppdec.graph import (arbitration_output_schema, build_arbitration_prompt,
                                         check_arbitration_answer, expected_arbitration)

OLLAMA = "http://localhost:11434"

# one representative raw input for every reachable (band, trend case, competition) combination
COMBOS = []
for band_u in (0.9, 0.1):                      # above_0.7, below_0.3: trend and step 3 not reached
    for comp_r in (0.5, 0.05):
        COMBOS.append((band_u, "STABLE", "STABLE", comp_r))
for bt, et in (("RISING", "STABLE"), ("STABLE", "RISING"), ("FALLING", "FALLING"), ("STABLE", "STABLE")):
    for comp_r in (0.5, 0.05):
        COMBOS.append((0.5, bt, et, comp_r))
# ("RISING","STABLE") and ("STABLE","RISING") give the same trend case; both kept: 12 prompts,
# 10 distinct fact combinations for the compact styles.


def make_inputs(u, bt, et, cr):
    return dict(greedy_choice={"parcel_id": "p1", "cost": 1.0}, hungarian_choice={"parcel_id": "p2", "cost": 1.2},
                cost_diff_label="Hungarian costs +0.20 more (20% premium)", belief_infeasibility=u,
                energy_error=0.1, num_peers=8, num_tasks=int(round(8 / cr)), competition_ratio=cr,
                belief_trend=bt, belief_history=[f"{u:.2f}"] * 5, error_trend=et, error_history=["0.10"] * 5,
                competition_trend="STABLE", competition_history=[f"{cr:.2f}"] * 5)


def ask(model, prompt, schema):
    payload = {"model": model, "prompt": prompt, "stream": False, "format": schema,
               "options": {"temperature": 0.0, "top_p": 0.1, "num_predict": 256}}
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=120).json()
    return json.loads(r["response"]), r.get("eval_count"), r.get("total_duration", 0) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-vl:8b-instruct"])
    ap.add_argument("--styles", nargs="+", default=["stepwise_compact", "paired_steps", "single_answer"])
    args = ap.parse_args()
    for model in args.models:
        for style in args.styles:
            schema = arbitration_output_schema(style)
            wrong, secs, toks = [], [], []
            for u, bt, et, cr in COMBOS:
                inp = make_inputs(u, bt, et, cr)
                ans, t, dur = ask(model, build_arbitration_prompt(**inp, prompt_version=style), schema)
                chk = check_arbitration_answer(ans, u, bt, et, cr)
                secs.append(dur); toks.append(t)
                if chk["violations"]:
                    wrong.append(((u, bt, et, cr), ans, chk["violations"][0]))
            print(f"{model} | {style:17s} | wrong decisions on {len(wrong)} of {len(COMBOS)} inputs | "
                  f"median {sorted(secs)[len(secs)//2]:.2f} s, {sorted(toks)[len(toks)//2]} tokens")
            for w in wrong:
                print("     ", w)


if __name__ == "__main__":
    main()
