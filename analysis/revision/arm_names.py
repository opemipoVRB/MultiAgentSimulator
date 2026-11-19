# analysis/revision/arm_names.py
"""
Plain names for every arm, generated from the arm's own settings so they cannot drift.

Scheme:  <how a drone picks its task> <layers> | <what the drones can see> [| fleet or latency]
  how it picks  Nearest task first
                Optimal team assignment            (peer costs guessed)
                One-task auction, bids guessed     (CBBA in the drone's head, one task each)
                Three-task auction, bids guessed   (CBBA in the drone's head, up to 3 tasks, played out in rounds)
                One-task / CBBA-Bundle (three-task auction, bids sent by message)   (the communicating CBBA baselines)
                Optimal assignment by a central planner
                Rules | SLM | Classifier choosing between nearest-first and a coordination method
  layers        + tracks peers every step | + remembers who took what
  what they see global view | local view (8 cells) | local view, noisy | local view, 30% dropouts

Usage:  PYTHONPATH=.:src python3 -m analysis.revision.arm_names     (rewrites arm_names.json)
"""
import json
from pathlib import Path

NAMES_PATH = Path(__file__).with_name("arm_names.json")


def _view(sensing):
    if not sensing:
        return "global view"
    r = sensing.get("radius")
    base = f"local view ({r:g} cells)" if (r is not None and r != float("inf")) else "local view"
    extra = []
    if sensing.get("noise_std"):
        extra.append("noisy")
    if sensing.get("loss"):
        extra.append(f"{int(sensing['loss'] * 100)}% dropouts")
    return base + (", " + ", ".join(extra) if extra else "")


def spec_name(s: dict) -> str:
    st, name, variant = s.get("strategy_type"), s["name"], s.get("variant", "")
    if st == "naive":
        return "Naive (random nearest-first rollouts)" + (" | unequal batteries" if s.get("fleet_profile") else "")
    if name.startswith("decentralized_auction"):
        return "Auction (older auction, bids sent by message)"
    if name.startswith("cbba_single"):
        return "CBBA-ST (one-task auction, bids sent by message)" + (" | unequal batteries" if s.get("fleet_profile") else "")
    if name.startswith("cbba_full"):
        return "CBBA-Bundle (three-task auction, bids sent by message)"
    if name.startswith("centralized_cc"):
        kind = {"hungarian": "Hungarian (optimal assignment)", "greedy": "Greedy (nearest task first)",
                "auction": "Auction"}.get(variant, variant)
        return f"Centralised {kind}, by a central planner" + (" | unequal batteries" if s.get("fleet_profile") else "")
    if st != "sppdec":
        return f"{name}_{variant}"
    arbiter = s.get("arbiter", "slm")
    k = s.get("cbba_bundle_size", 1)
    k = k if k == "auto" else int(k)
    coord = ("CBBA, one task (auction, bids guessed)" if s.get("cbba_impl", "best_pair_first") == "best_pair_first"
             else (("CBBA, tasks per drone = fair share of open tasks (auction, bids guessed, played out in rounds)" if k == "auto"
                   else f"CBBA, {k} task{'s' if k != 1 else ''} (auction, bids guessed, played out in rounds)")))
    if arbiter == "always_greedy":
        who = "Greedy (nearest task first)"
    elif arbiter == "always_hungarian":
        who = "Hungarian (optimal team assignment)"
    elif arbiter == "always_cbba":
        who = coord
    else:
        over = coord if s.get("coordination_model") == "cbba" else "Hungarian (optimal team assignment)"
        if arbiter == "rule_based":
            who = "Rules" + (" (competition wins a conflict)" if s.get("conflict_policy") == "competition" else "")
        elif arbiter == "classifier_regime":
            who = "Classifier (regime)"
        elif arbiter == "classifier_tree":
            who = "Classifier (" + {"distilled": "distilled from rules", "slm_imitation": "imitating the SLM"}.get(
                s.get("classifier_model", "distilled"), "tree") + ")"
        else:
            who = "SLM (" + ("judgement" if s.get("prompt_version") == "judgement" else "guarded") + ")"
        who += f" choosing Greedy or {over}"
    layers = ""
    if s.get("observe_every_step"):
        layers += " + tracks peers every step"
        if not s.get("observe_contested", True):
            layers += " (reacts to vanished targets only)"
        if not s.get("observe_gone", True):
            layers += " (reacts to contested targets only)"
    if s.get("assumption_memory"):
        layers += " + remembers who took what"
    tail = []
    if not s.get("symmetric_costs") and arbiter not in ("always_greedy",) and "personality" not in variant:
        tail.append("drone and peers scored differently (original paper)")
    tail.append(_view(s.get("sensing")))
    if s.get("fleet_profile"):
        tail.append("unequal batteries" + (", peers assumed alike" if s.get("peer_rate_assumption") == "same_as_self" else ""))
    if s.get("slm_latency_steps"):
        tail.append(f"latency {s['slm_latency_steps']} steps ({s.get('slm_latency_mode', 'blocking')})")
    return f"{who}{layers} | " + ", ".join(tail)


def build():
    import src.run_experiment as rx
    names = {}
    for s in rx.build_strategy_vault("local"):
        names[f"{s['name']}_{s.get('variant', 'base')}"] = spec_name(s)
    return names


def display(arm_id: str) -> str:
    try:
        return json.loads(NAMES_PATH.read_text())[arm_id]
    except Exception:
        return arm_id


if __name__ == "__main__":
    n = build()
    NAMES_PATH.write_text(json.dumps(n, indent=1, sort_keys=True))
    print(len(n), "arms ->", NAMES_PATH)
