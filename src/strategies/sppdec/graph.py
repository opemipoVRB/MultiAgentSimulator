# src/strategies/sppdec/graph.py
"""
SPPDEC LangGraph-based Agent Strategy (Phase 1)

This module implements a local, predictive-decentralized agent controller
based on the SPPDEC (Structured Predictive Purpose-Driven Emergent Coordination)
architecture.

Phase 1 Scope
-------------
This implementation represents the minimal SPPDEC agent:

- Single-agent, single-step reasoning
- No coordination, arbitration, or reservation
- No long-horizon planning
- No execution control (movement handled elsewhere)
- Physics-grounded peer energy belief tracking (minimal latent state)

High-level responsibilities
---------------------------
- Construct a LangGraph execution graph per agent
- Interpret a compact world snapshot into local percepts
- Track peer energy expenditure through observable motion
- Generate feasible pickup tasks with energy-adjusted peer cost predictions
- Invoke a Small Language Model (SLM) for strategic policy reasoning
- Select a concrete pickup to dropoff action
- Emit controller-compatible directives every cycle

Design constraints
------------------
- One pickup and one dropoff per decision cycle
- No centralized authority
- No implicit coordination
- No communication or shared state
- Peer beliefs derived purely from observable physics (position changes)
- Compatible with existing controller and planner contracts

Explicit non-goals
------------------
- Execute movement
- Maintain global reservations
- Perform inter-agent arbitration
- Perform long-horizon planning
- Infer peer intentions or policies
- Reconstruct hidden peer state beyond observable motion
"""
import json
import os
import logging
import math
import random
import re
from .cbba_assumed_bids import bid_cost, capability_caps, cbba_with_assumed_bids, hungarian_bundles
from .self_model import SelfEnergyModel
from typing import TypedDict, List, Dict, Optional, Literal, Tuple, cast
from typing_extensions import TypeAlias, Required

from langgraph.graph import StateGraph, END

from src.strategies.sppdec.personalities import SPPDEC_PERSONALITIES
from src.strategies.sppdec.sppdec_experiment_config import get_sppdec_config

# ===========================================================================
# Logging
# ===========================================================================

logger = logging.getLogger("SPPDEC")
logger.setLevel(logging.INFO)


# ===========================================================================
# Data contracts
# ===========================================================================


class BeliefSample(TypedDict):
    x: List[float]
    belief: float


class LocalPrediction(TypedDict):
    """
    Local predictive belief about peers.

    Invariants (Phase 1):
    --------------------
    - Predictions are mobility-based only.
    - Parcel payload weight is NEVER used.
    - All energy estimates assume self.baseline_payload.
    - No peer intent or commitment is assumed.
    """
    peer_positions: Dict[str, Tuple[int, int]]
    peer_energy_costs: Dict[str, float]
    predicted_conflicts: List[str]
    confidence_penalty: float


DecisionMode: TypeAlias = Literal["candidate", "solo", "coordinate", "wait", "idle"]


class SPPDECAgentState(TypedDict, total=False):
    """
    Mutable state propagated across LangGraph nodes.

    Invariant
    ---------
    Every execution path MUST terminate with `directive` populated.
    The controller must never exit without an explicit directive.
    """

    # Required fields - always present
    snapshot: Required[Dict]

    # Perception
    visible_parcels: List[Dict]
    visible_agents: List[Dict]

    # Predictive context
    peer_distance_map: Dict[str, Dict[str, float]]
    predicted_conflicts: List[str]

    # Planning
    feasible_tasks: List[Dict]
    slm_preferred_parcel_id: Optional[str]
    slm_confidence: float
    slm_status: Literal["success", "abstain", "failed"]

    # Decision
    selected_task: Optional[Dict]
    decision_mode: DecisionMode

    # Epistemic beliefs
    belief_infeasibility: float  # 0.0 to 1.0, informational only

    # Learning signals
    last_battery: float
    predicted_cost: float
    energy_prediction_error: float

    # Output
    directive: Optional[Dict]

    # Strategy meta-choice
    slm_policy: Optional[Literal["greedy", "hungarian", "greedy_only"]]
    greedy_choice: Optional[Dict]
    hungarian_choice: Optional[Dict]
    cbba_choice: Optional[Dict]

    # Values that cross from one graph step to the next. LangGraph silently drops any key that
    # is not declared here, so every key written by one step and read by another belongs here.
    slm_attempted: bool           # the model was consulted (charges its latency)
    slm_bundle: List[str]         # plan ahead: parcel ids the model named to take after the first task
    measured_context: Dict        # quantities the drone measures at the decision, logged for the reward analysis (not shown to any model)
    slm_gate: str                 # which gate or arbiter decided
    wait_reason: str              # why a decision did not become a plan
    slm_prompt_inputs: Dict       # exact prompt inputs, for replay
    slm_attempts: int             # generation attempts used by the decision that was kept
    cbba_info: Dict               # protocol CBBA: rounds, convergence, own bundle length
    assumed_assignment: Dict      # what this drone assumes each peer takes: {model: {peer: parcel id}}
    model_choices: Dict           # what each reasoning model proposes for this drone: {model: task or None}
    arbiter_context: Dict         # the shared context the context arbiters read (see _arbiter_context)
    consensus: Dict               # what the two models agreed on this round (consensus option)
    consensus_bundle: List[Dict]  # the tasks after the first in the drone's agreed plan (consensus option)
    scenario: Dict                # facts the drone can observe about its situation (see _scenario_descriptors)

    # Tracing and debugging - initialized dynamically
    cycle_id: int
    trace: List[str]


# ===========================================================================
# Helper functions
# ===========================================================================


def build_execution_directive(task: Dict, confidence: float) -> Dict:
    """
    Wrap an executable pickup to dropoff task into a controller directive.

    This is the only structure the controller executes.
    """
    return {
        "mode": "plan",
        "plan": {
            "plan": [
                {
                    "pickup": task["pickup"],
                    "dropoff": task["dropoff"],
                    "weight": task.get("weight", 1.0),
                }
            ],
            "confidence": confidence,
            "strategy": "sppdec-local",
            "narration": "SPPDEC local single-step decision",
        },
    }


# ===========================================================================
# LangGraph Controller
# ===========================================================================

# ---------------------------------------------------------------------------
# Arbitration prompt styles
#
#   submitted         the prompt in the submitted paper (reproduces its results)
#   cost_gap_closed   submitted text with Step 1 fixed: the cost premium never decides
#                     on its own (the submitted rules had no rule above a 25% premium)
#   stepwise          full context + facts computed in code + JSON schema with one
#                     outcome field per rule step before the policy (100% rule agreement)
#   stepwise_compact  rules first (cacheable prefix), facts only, short fields: one
#                     outcome per step, then the policy
#   paired_steps      as stepwise_compact, but steps 1 and 2 share one outcome field
#                     (default; a code guard overrides answers that contradict the written steps)
#   single_answer     as stepwise_compact, policy field only
#
# ---------------------------------------------------------------------------
PROMPT_STYLE = os.getenv("SPPDEC_PROMPT_STYLE", "paired_steps")  # default for arms that do not set one


def prompt_style(name=None) -> str:
    style = name or PROMPT_STYLE
    return style


STEP1_RULES = {
    "submitted": (
        "    Step 1 — Cost differential (primary signal):\n"
        "      If Hungarian costs 0-10% more → coordination is cheap; proceed to Step 2\n"
        "      If Hungarian costs 10-25% more → coordination requires justification; proceed to Step 2\n"
    ),
    "cost_gap_closed": (
        "    Step 1 — Cost differential:\n"
        "      A higher personal cost never decides on its own, whatever the premium (including above 25%).\n"
        "      Leave the team assignment only for the reasons in Steps 2 and 3. Proceed to Step 2.\n"
    ),
}

STEPWISE_RULES = """    === DECISION RULES ===

    Apply the steps in order. Each step either decides or passes to the next step.
    The cost differential never decides on its own, whatever the premium.

    Step 1 — Uncertainty band:
      above_0.7   → step1_outcome = GREEDY
      below_0.3   → step1_outcome = HUNGARIAN
      0.3_to_0.7  → step1_outcome = go_to_step_2

    Step 2 — Trend case (only if step1_outcome is go_to_step_2, otherwise not_reached):
      either_rising → step2_outcome = GREEDY
      both_falling  → step2_outcome = HUNGARIAN
      other         → step2_outcome = go_to_step_3

    Step 3 — Competition tiebreak (only if step2_outcome is go_to_step_3, otherwise not_reached):
      Peers-per-task ratio above 0.15 is yes → step3_outcome = HUNGARIAN
      Peers-per-task ratio above 0.15 is no  → step3_outcome = GREEDY

    policy = the GREEDY or HUNGARIAN outcome of the step that decided.

    === YOUR ANSWER ===

    Fill in every field in order, copying the DERIVED FACTS, then each step outcome, then the
    policy. Output only the JSON object.
    """

STEPWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "uncertainty_band": {"type": "string", "enum": ["above_0.7", "below_0.3", "0.3_to_0.7"]},
        "trend_case": {"type": "string", "enum": ["either_rising", "both_falling", "other", "not_needed"]},
        "competition_above_0.15": {"type": "string", "enum": ["yes", "no"]},
        "step1_outcome": {"type": "string", "enum": ["GREEDY", "HUNGARIAN", "go_to_step_2"]},
        "step2_outcome": {"type": "string", "enum": ["GREEDY", "HUNGARIAN", "go_to_step_3", "not_reached"]},
        "step3_outcome": {"type": "string", "enum": ["GREEDY", "HUNGARIAN", "not_reached"]},
        "policy": {"type": "string", "enum": ["greedy", "hungarian"]},
        "confidence": {"type": "number"},
    },
    "required": ["uncertainty_band", "trend_case", "competition_above_0.15", "step1_outcome",
                 "step2_outcome", "step3_outcome", "policy", "confidence"],
}

# Compact styles: fixed text first so Ollama can reuse its cached computation of the
# prefix across calls; only the three facts at the end change. Short field names and
# values keep the output to a few tokens (output tokens dominate latency).
COMPACT_RULES = """You are the arbitration step of a delivery drone. Choose G (GREEDY: my cheapest task) or H (HUNGARIAN: my slot in the team assignment) from the FACTS at the end.
Steps run in order and stop at the first G or H.
Step 1 (always): band above_0.7 -> G; below_0.3 -> H; 0.3_to_0.7 -> next.
Step 2 (only if step 1 is next, otherwise skip): trend either_rising -> G; both_falling -> H; other -> next.
Step 3 (only if step 2 is next, otherwise skip): competition yes -> H; no -> G.
The policy p is the first G or H.
"""

COMPACT_ANSWER = {
    "stepwise_compact": "Answer with JSON only: s1 = step 1 outcome, s2 = step 2 outcome, s3 = step 3 outcome, p = policy.\n",
    "paired_steps": "Answer with JSON only: s12 = step 1 outcome if it is G or H, otherwise the step 2 outcome (G, H or next); s3 = step 3 outcome; p = policy.\n",
    "single_answer": "Answer with JSON only: p = policy.\n",
}

_OUT = {"type": "string", "enum": ["G", "H", "next"]}
_LAST = {"type": "string", "enum": ["G", "H", "skip"]}
_MID = {"type": "string", "enum": ["G", "H", "next", "skip"]}
_POL = {"type": "string", "enum": ["G", "H"]}
COMPACT_SCHEMAS = {
    "stepwise_compact": {"type": "object", "properties": {"s1": _OUT, "s2": _MID, "s3": _LAST, "p": _POL},
                         "required": ["s1", "s2", "s3", "p"]},
    "paired_steps": {"type": "object", "properties": {"s12": _OUT, "s3": _LAST, "p": _POL},
                     "required": ["s12", "s3", "p"]},
    "single_answer": {"type": "object", "properties": {"p": _POL}, "required": ["p"]},
}



# ---------------------------------------------------------------------------
# Judgement style: the model weighs competition against the trend
# ---------------------------------------------------------------------------
def judgement_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio):
    """
    Everything the rules can decide, computed in code.
      forced   G / H when the decision is fixed (extreme uncertainty band, or the trend and the
               competition lean the same way, or the trend has no lean), else None
      conflict True when the trend and the competition lean opposite ways (medium band only)
    """
    band = ("above_0.7" if belief_infeasibility > 0.7
            else "below_0.3" if belief_infeasibility < 0.3 else "0.3_to_0.7")
    if "RISING" in (belief_trend, error_trend):
        trend_lean = "G"          # estimates of peers are degrading
    elif belief_trend == error_trend == "FALLING":
        trend_lean = "H"          # estimates of peers are improving
    else:
        trend_lean = "none"
    competition_lean = "H" if competition_ratio > 0.15 else "G"
    if band == "above_0.7":
        forced, conflict = "G", False
    elif band == "below_0.3":
        forced, conflict = "H", False
    elif trend_lean == "none" or trend_lean == competition_lean:
        forced, conflict = competition_lean, False
    else:
        forced, conflict = None, True
    return {"band": band, "trend_lean": trend_lean, "competition_lean": competition_lean,
            "forced": forced, "conflict": conflict}


JUDGEMENT_RULES = (
    "You are the arbitration step of a delivery drone that coordinates with other drones without messages. "
    "Choose G (GREEDY: take my cheapest task) or H (HUNGARIAN: take my slot in the team assignment computed from what I believe about my peers).\n"
    "TREND: a rising uncertainty or prediction error means my beliefs about peers are getting less reliable, so the team assignment "
    "may not match what peers really do; that favours G. Falling means they are improving; that favours H.\n"
    "COMPETITION: peers per task. If several peers target each task, choosing greedily means we collide on the same parcels and waste trips; "
    "coordination pays. That favours H. If there are few peers per task, collisions are rare; that favours G.\n"
)

JUDGEMENT_ANSWER = (
    "The trend and the competition pull in opposite directions. First rate how strong the trend is (weak, moderate, strong) from its history "
    "and how high the competition is (low, moderate, high, extreme) from its value; then weigh the stronger against the weaker and decide.\n"
    "Answer with JSON only: trend_strength, competition_level, weigh (trend or competition), reason (at most 12 words), verdict (G or H).\n"
)

JUDGEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "trend_strength": {"type": "string", "enum": ["weak", "moderate", "strong"]},
        "competition_level": {"type": "string", "enum": ["low", "moderate", "high", "extreme"]},
        "weigh": {"type": "string", "enum": ["trend", "competition"]},
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["G", "H"]},
    },
    "required": ["trend_strength", "competition_level", "weigh", "reason", "verdict"],
}


def _judgement_facts_text(facts, belief_infeasibility, belief_trend, belief_history, error_trend,
                          error_history, competition_ratio, num_peers, num_tasks, cost_diff_label):
    lines = [
        "FACTS",
        f"uncertainty {belief_infeasibility:.2f}, trend {belief_trend}, history {belief_history}",
        f"prediction error trend {error_trend}, history {error_history}",
        f"competition {competition_ratio:.2f} peers per task ({num_peers} peers, {num_tasks} tasks)",
    ]
    return "\n".join(lines) + "\n"


def check_judgement_answer(answer: dict, belief_infeasibility, belief_trend, error_trend, competition_ratio) -> dict:
    """
    Verify a judgement answer. The decision must follow the reasoning it states, not a fixed table:
      - where the code fixes the decision, the verdict must equal it;
      - where the leans conflict, either verdict is allowed, but it must equal the lean of the
        consideration the model says it followed (weigh = trend or competition).
    Returns the same structure as check_arbitration_answer, plus the conflict flag.
    """
    f = judgement_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio)
    verdict = answer.get("verdict")
    policy = "greedy" if verdict == "G" else "hungarian" if verdict == "H" else None
    weigh = answer.get("weigh")
    reason = (answer.get("reason") or "").strip()
    violations, warnings = [], []
    if f["conflict"]:
        followed = {"trend": f["trend_lean"], "competition": f["competition_lean"]}.get(weigh)
        # in a conflict the model chooses; the safe default if its answer is incoherent is the
        # submitted rule: degrading estimates favour self-reliance
        expected = "greedy" if f["trend_lean"] == "G" else "hungarian"
        if followed is None:
            violations.append(f"weigh '{weigh}' is not a consideration in a conflict")
        elif verdict != followed:
            violations.append(f"verdict '{verdict}' does not follow the consideration it named "
                              f"('{weigh}' leans {followed})")
    else:
        expected = "greedy" if f["forced"] == "G" else "hungarian"
        if verdict != f["forced"]:
            violations.append(f"verdict '{verdict}' differs from the fixed decision '{f['forced']}'")
        if weigh not in ("agree", "trend", "competition"):
            warnings.append(f"weigh '{weigh}' invalid")
    if len(reason) < 3:
        warnings.append("no reason given")
    return {"violations": violations, "warnings": warnings, "expected_policy": expected,
            "written_policy": policy, "conflict": f["conflict"], "weigh": weigh, "reason": reason}


def arbitration_output_schema(name=None):
    """JSON schema for constrained decoding, or None for the free-text styles."""
    style = prompt_style(name)
    if style == "stepwise":
        return STEPWISE_SCHEMA
    if style == "judgement":
        return JUDGEMENT_SCHEMA
    return COMPACT_SCHEMAS.get(style)


def _derived_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio):
    """Threshold comparisons computed in code: small models misjudge them."""
    band = ("above_0.7" if belief_infeasibility > 0.7
            else "below_0.3" if belief_infeasibility < 0.3 else "0.3_to_0.7")
    if band != "0.3_to_0.7":
        trend = "not_needed"
    elif "RISING" in (belief_trend, error_trend):
        trend = "either_rising"
    elif belief_trend == error_trend == "FALLING":
        trend = "both_falling"
    else:
        trend = "other"
    return band, trend, "yes" if competition_ratio > 0.15 else "no"


_STEP_WORDS = {"GREEDY": "G", "HUNGARIAN": "H", "go_to_step_2": "next", "go_to_step_3": "next",
               "not_reached": "skip"}


def expected_arbitration(band: str, trend: str, comp: str):
    """The rule table applied to the facts: (step1, step2, step3, policy) as G/H/next/skip."""
    s1 = {"above_0.7": "G", "below_0.3": "H"}.get(band, "next")
    s2 = "skip" if s1 != "next" else {"either_rising": "G", "both_falling": "H"}.get(trend, "next")
    s3 = "skip" if s2 != "next" else ("H" if comp == "yes" else "G")
    decided = next(x for x in (s1, s2, s3) if x in ("G", "H"))
    return s1, s2, s3, "greedy" if decided == "G" else "hungarian"


def check_arbitration_answer(answer: dict, belief_infeasibility, belief_trend, error_trend, competition_ratio) -> dict:
    """
    Verify a model's arbitration answer. The decision must (1) equal the decision implied by
    the steps it wrote, and (2) equal the rule table applied to the facts.

    Returns {"violations": [...], "warnings": [...], "expected_policy": str, "written_policy": str|None}.
    A violation means the answer must not be acted on; a warning means only a written
    intermediate field is off while the decision is still right.
    """
    band, trend, comp = _derived_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio)
    e1, e2, e3, expected = expected_arbitration(band, trend, comp)
    a = dict(answer)
    policy = a.get("policy")
    if policy is None and a.get("p") in ("G", "H"):
        policy = "greedy" if a["p"] == "G" else "hungarian"
    violations, warnings = [], []

    # steps as written by the model, in the G/H/next/skip vocabulary
    if "step1_outcome" in a:
        written = [_STEP_WORDS.get(a.get(k), a.get(k)) for k in ("step1_outcome", "step2_outcome", "step3_outcome")]
        expect = [e1, e2, e3]
    elif "s1" in a:
        written, expect = [a.get("s1"), a.get("s2"), a.get("s3")], [e1, e2, e3]
    elif "s12" in a:
        written = [a.get("s12"), a.get("s3")]
        expect = [e1 if e1 != "next" else e2, e3]
    else:
        written, expect = None, None

    written_decision = None
    if written is not None:
        open_ = True
        for i, v in enumerate(written):
            if open_:
                if v in ("G", "H"):
                    written_decision, open_ = v, False
                elif v != "next":
                    warnings.append(f"step {i + 1} written '{v}' before any decision")
            elif v != "skip":
                warnings.append(f"step {i + 1} written '{v}' after a step already decided")
        if open_ and "s12" not in a:
            warnings.append("no step decided")
        for i, (v, e) in enumerate(zip(written, expect)):
            if v != e:
                warnings.append(f"step {i + 1}: wrote '{v}', rules give '{e}'")
        if written_decision is None:
            violations.append("steps wrote no decision")
        elif ("greedy" if written_decision == "G" else "hungarian") != policy:
            violations.append(f"policy '{policy}' differs from the decision its steps wrote ('{written_decision}')")
    for key, want in (("uncertainty_band", band), ("trend_case", trend), ("competition_above_0.15", comp)):
        if key in a and a[key] != want:
            warnings.append(f"fact {key}: wrote '{a[key]}', computed '{want}'")
    if policy != expected:
        violations.append(f"policy '{policy}' differs from the rule result '{expected}'")
    return {"violations": violations, "warnings": warnings, "expected_policy": expected,
            "written_policy": None if written_decision is None else ("greedy" if written_decision == "G" else "hungarian")}


MODEL_WORDS = {
    "greedy": "take the nearest task; ignore what peers will do (the yield rule still applies)",
    "hungarian": "optimal team assignment from believed positions and energy of all drones",
    "cbba_rounds": "auction run in the drone's head in rounds, each drone holding several tasks, peers' bids guessed",
}

CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "situation": {"type": "string"},
        "table_row": {"type": "string"},
        "table_best": {"type": "string", "enum": ["greedy", "hungarian", "cbba_rounds"]},
        "own_condition": {"type": "string", "enum": ["none", "rate", "weight", "mixed", "shift"]},
        "deviation_from": {"type": "string"},
        "reason": {"type": "string"},
        "model": {"type": "string", "enum": ["greedy", "hungarian", "cbba_rounds"]},
    },
    "required": ["situation", "table_row", "table_best", "own_condition", "deviation_from", "reason", "model"],
}


def context_schema(ctx: dict, table_cells: dict, available) -> dict:
    """
    The answer schema for this decision: the model and the named table row and fact can only be ones that exist
    (a loosely worded name would otherwise get a sound answer thrown out by the guard on form alone).
    """
    names = [m for m in ("greedy", "hungarian", "cbba_rounds") if m in available]
    facts = ["none"] + [k for k, v in _flat_context(ctx).items()
                        if not k.startswith("proposals") and isinstance(v, (int, float))]
    props = dict(CONTEXT_SCHEMA["properties"])
    props["table_row"] = {"type": "string", "enum": list(table_cells)}
    props["table_best"] = {"type": "string", "enum": ["greedy", "hungarian", "cbba_rounds"]}
    props["model"] = {"type": "string", "enum": names}
    props["deviation_from"] = {"type": "string", "enum": facts}
    return dict(CONTEXT_SCHEMA, properties=props)


def _flat_context(ctx: dict) -> dict:
    """Every numeric or categorical fact in the shared context, by name (nested names joined with '.')."""
    out = {}

    def walk(prefix, v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[prefix] = v
    walk("", ctx)
    return out


CONDITION_WORDS = {
    "rate": "every drone burns more energy per cell than the nominal model says",
    "weight": "carried weight costs every drone much more than the nominal model says",
    "mixed": "each drone burns differently (some more, some less than nominal)",
    "shift": "consumption changes part-way through the mission, without warning",
}


def _condition_text(cond_table: dict) -> str:
    lines = []
    for label, t in cond_table.items():
        lines.append(f"  {label} ({CONDITION_WORDS[label]}); {t['runs']} runs:")
        for m, c in t["models"].items():
            lines.append(f"    {m}: delivered {c['delivered_share']:.2f} of parcels, all parcels in {c['all_delivered_runs']:.2f} "
                         f"of runs, drones lost per run {c['drones_lost_per_run']:.2f}, mission end vs best "
                         f"{c.get('mission_end_vs_best', 'n/a')}")
    return "\n".join(lines)


def build_free_context_prompt(ctx: dict, table_cells: dict, available, cond_table: dict = None,
                              suggest: bool = True, world: str = None, plan_ahead: bool = False) -> str:
    """
    Prompt of the free SLM arbiter. The drone's SLM decides which reasoning model acts; everything it is shown
    (tables, facts, what each model proposes, a suggestion computed from the tables) is advice it may use, weigh or
    ignore. No required form beyond the choice and a reason. With `suggest` False the prompt does not compute the
    tables' answer for the model (the row and its best model): it has to find them in the tables itself. (Measured:
    with the computed suggestion, models from 0.6B to 8B copy it and ignore availability and the own-energy condition.)
    """
    base = build_context_prompt(ctx, table_cells, available, cond_table, free=True)
    head, _, _ = base.partition("Steps. 1)")
    ahead = ("""
Also give in "plan" the ordered parcel ids this drone will deliver next, starting with the task of the model you chose.
Plan every task you expect this drone to take, at least three when at least three are open and affordable. Order them by
how this drone should take them, using distance, energy margin and which peers are nearer. The drone follows the list
without asking you again and skips any task a peer takes first.
""" if plan_ahead else "")
    names = {"cbba_bid_rounds": "cbba_rounds"}
    row_now = f"{ctx['sensing_class']}|{ctx['regime']}"
    best_now = names.get((table_cells.get(row_now) or {}).get("best"), (table_cells.get(row_now) or {}).get("best", "unknown"))
    if world:
        head = head.replace("EVERYTHING THE DRONE KNOWS (name: value)", world + "\n\nEVERYTHING THE DRONE KNOWS (name: value)")
    head = head.replace("EVIDENCE: measured relative", "SUGGESTION (measured, not binding): relative")
    head = head.replace("EVIDENCE: what the drone's own", "SUGGESTION (measured, not binding): what the drone's own")
    head = head.replace("Computed for you: this drone's table row", "A suggestion from the tables: this drone's table row")
    if not suggest:
        head = head.split("A suggestion from the tables:")[0]
        return head + """The tables and facts above are evidence. You decide which reasoning model acts for this drone now, and
your choice is carried out as given; choose among the models listed as available. Say in "weighing" which of the
evidence you relied on and which you set aside.""" + ahead + "Answer in JSON with exactly: situation, weighing, model, reason" + (", plan." if plan_ahead else ".")
    return head + f"""The tables and facts above are suggestions. You decide which reasoning model acts for this drone now,
and your choice is carried out as given. The first table suggests {best_now} for "{row_now}"; take it, or not, as
you judge. Say in "weighing" which of the suggestions and facts you relied on and which you set aside.""" + ahead + "Answer in JSON with exactly: situation, weighing, model, reason" + (", plan." if plan_ahead else ".")


FREE_SCHEMA = {
    "type": "object",
    "properties": {"situation": {"type": "string", "maxLength": 160},
                   "weighing": {"type": "string", "maxLength": 400},
                   "model": {"type": "string", "enum": ["greedy", "hungarian", "cbba_rounds"]},
                   "reason": {"type": "string", "maxLength": 200}},
    "required": ["situation", "weighing", "model", "reason"],
}


TASK_SCHEMA = {
    "type": "object",
    "properties": {"situation": {"type": "string", "maxLength": 160},
                   "weighing": {"type": "string", "maxLength": 400},
                   "parcel": {"type": "string"},
                   "reason": {"type": "string", "maxLength": 200}},
    "required": ["situation", "weighing", "parcel", "reason"],
}


TERRITORY_CONVENTION = {
    "labels": """TERRITORY CONVENTION (every drone follows it and nobody sends messages): each parcel belongs to the drone whose start cell
is nearest to it (ties go to the lowest drone id). Each open task below says whose territory it lies in. Take a parcel of your
own territory while one is left; only when none of your own is left, take a parcel from another territory.""",
    "raw": """TERRITORY CONVENTION (every drone follows it and nobody sends messages): each parcel belongs to the drone whose start cell
is nearest to it (nearest in grid steps, |column difference| + |row difference|; ties go to the lowest drone id). The start cell of
every drone is listed below, so every drone can work out the same owner for every parcel. Take a parcel of your own territory while
one is left; only when none of your own is left, take a parcel from another territory.""",
    "both": """TERRITORY CONVENTION (every drone follows it and nobody sends messages): each parcel belongs to the drone whose start cell
is nearest to it (nearest in grid steps, |column difference| + |row difference|; ties go to the lowest drone id). The start cell of
every drone is listed below, and each open task says whose territory it lies in. Take a parcel of your own territory while one is
left; only when none of your own is left, take a parcel from another territory.""",
}


def build_task_prompt(ctx: dict, world: str, available, convention: Optional[str] = None) -> str:
    """
    Prompt of the task-level SLM arbiter: the SLM chooses the parcel this drone takes next, not a reasoning model. It is
    shown the world as the drone believes it (its body, the terrain, each task with its geometry and energy margin, its peers,
    which reasoning models propose which task) and every fact the drone knows. The models' proposals are advice.
    """
    models = "\n".join(f"  {m}: {MODEL_WORDS[m]}" for m in available)
    facts = "\n".join(f"  {k}: {round(v, 3) if isinstance(v, float) else v}"
                      for k, v in _flat_context(ctx).items() if not k.startswith("proposals"))
    return f"""You choose the next task (parcel to deliver) for one drone in a multi-drone parcel delivery mission.
No messages are exchanged; the drone works from what it believes about its peers. The mission is finished when every
parcel is delivered, so the aim is that the whole fleet finishes soon and no drone runs out of energy.

REASONING MODELS whose proposals are marked on the tasks below (advice, not instructions)
{models}

{world}

EVERYTHING THE DRONE KNOWS (name: value)
{facts}

{(convention + chr(10) + chr(10)) if convention else ""}You decide which open task this drone takes now; your choice is carried out as given. Weigh distance, the energy margin, which
peers are nearer to a task or believed to be heading for it, and what the reasoning models propose. Answer in JSON with
exactly: situation, weighing, parcel (the id of the chosen parcel, taken from the open tasks above), reason."""


def build_context_prompt(ctx: dict, table_cells: dict, available, cond_table: dict = None, free: bool = False) -> str:
    """The prompt of the context SLM arbiter: the models, the evidence table, every fact, the proposals."""
    names = {"cbba_bid_rounds": "cbba_rounds"}
    rows = []
    for key, c in table_cells.items():
        rel = {names.get(k, k): v for k, v in c["relative_to_best"].items()}
        rows.append(f"  {key}: best {names.get(c['best'], c['best'])}; relative completion time "
                    + ", ".join(f"{m} {rel[m]:.2f}" for m in ("greedy", "hungarian", "cbba_rounds")))
    models = "\n".join(f"  {m}: {MODEL_WORDS[m]}" for m in available)
    props = "\n".join(
        f"  {m}: " + ("no task for this drone" if ctx["proposals"][m] is None else
                      f"parcel {ctx['proposals'][m]['parcel']}, cost {ctx['proposals'][m]['cost']}, "
                      f"{'same parcel as greedy' if ctx['proposals'][m]['same_as_greedy'] else 'different parcel from greedy'}")
        for m in ctx["proposals"])
    facts = "\n".join(f"  {k}: {round(v, 3) if isinstance(v, float) else v}"
                      for k, v in _flat_context(ctx).items() if not k.startswith("proposals"))
    row_now = f"{ctx['sensing_class']}|{ctx['regime']}"
    best_now = names.get((table_cells.get(row_now) or {}).get("best"), (table_cells.get(row_now) or {}).get("best", "unknown"))
    cond_block = ("EVIDENCE: what the drone's own energy conditions do to each model (measured, small-battery missions, "
                  "self-check on). Compare with self_check.cost_vs_nominal, self_check.weight_cost_vs_nominal, "
                  "self_check.shift_alarms and self_check.affordable_vs_open below.\n" + _condition_text(cond_table) + "\n\n"
                  if cond_table else "")
    choice_rule = ("The table's best model is evidence, not an instruction: weigh it against the other facts and "
                   "your own reading of the situation, and choose the model you judge best for this drone now. If you "
                   "choose a different model from table_best, put the name of the fact that decided it in "
                   "deviation_from, otherwise write \"none\"."
                   if free else
                   "Choose table_best unless a fact above gives a reason to prefer another; in that case put the name "
                   "of that fact in deviation_from, otherwise write \"none\".")
    return f"""You choose which reasoning model decides this drone's next task in a multi-drone parcel delivery mission.
No messages are exchanged; every model works from what the drone believes about its peers.

REASONING MODELS (choose one of these)
{models}

EVIDENCE: measured relative completion time per situation (1.00 = fastest model there)
{chr(10).join(rows)}
The situation is "<sensing class>|<regime>". Sensing class: full (sees everything), partial (limited radius),
noisy (limited radius and position noise). Regime: task_rich, balanced, agent_surplus (agents per parcel).

{cond_block}WHAT EACH MODEL PROPOSES NOW
{props}

EVERYTHING THE DRONE KNOWS (name: value)
{facts}

Computed for you: this drone's table row is "{row_now}" and the table's best model in it is {best_now}.
Steps. 1) Say the situation in one sentence. 2) Give table_row (it is "{row_now}") and table_best (it is {best_now}). 3) Say which own condition (rate, weight, mixed, shift) the self_check
facts point to, or none. 4) Choose the model. {choice_rule}
5) Give the reason in one sentence. Answer in JSON with exactly: situation, table_row, table_best,
own_condition, deviation_from, reason, model."""


def check_context_answer(answer: dict, ctx: dict, table_cells: dict, available, strict: bool = True) -> dict:
    """
    The decision must follow the reasoning it states:
      - the model must be one that has a proposal;
      - the named table row must exist and table_best must be what that row says (it read the table);
      - choosing a model other than table_best requires naming a fact of the context in deviation_from.
    Violations make the drone use the rule list instead of the answer.
    """
    names = {"cbba_bid_rounds": "cbba_rounds"}
    violations, warnings = [], []
    model = answer.get("model")
    row = table_cells.get((answer.get("table_row") or "").strip())
    if not strict:
        # Free mode: the SLM judges the evidence itself. Only validity is checked (the model has a proposal);
        # whether its departures from the tables help is measured on mission outcomes, not enforced.
        if model not in available:
            violations.append(f"model '{model}' has no proposal")
        suggested = table_cells.get(f"{ctx['sensing_class']}|{ctx['regime']}") or {}
        best = names.get(suggested.get("best"), suggested.get("best"))
        return {"violations": violations, "warnings": warnings,
                "answer_model": model, "deviated": best is not None and model != best,
                "deviation_from": answer.get("weighing"), "reason": (answer.get("reason") or "").strip()}
    actual_row = f"{ctx['sensing_class']}|{ctx['regime']}"
    if model not in available:
        violations.append(f"model '{model}' has no proposal")
    if row is None:
        violations.append(f"table row '{answer.get('table_row')}' does not exist")
    else:
        if (answer.get("table_row") or "").strip() != actual_row:
            violations.append(f"named row '{answer.get('table_row')}' is not this situation ('{actual_row}')")
        best = names.get(row["best"], row["best"])
        if answer.get("table_best") != best:
            violations.append(f"table_best '{answer.get('table_best')}' is not what row says ('{best}')")
        if (answer.get("own_condition") or "none") != (ctx.get("condition") or "none"):
            violations.append(f"own_condition '{answer.get('own_condition')}' is not what the self-check facts give "
                              f"('{ctx.get('condition')}')")
        if model in available and model != answer.get("table_best"):
            dev = (answer.get("deviation_from") or "").strip()
            if dev == "none" or dev not in _flat_context(ctx):
                violations.append(f"chose '{model}' over table best without naming a fact (deviation_from '{dev}')")
    if len((answer.get("reason") or "").strip()) < 3:
        warnings.append("no reason given")
    return {"violations": violations, "warnings": warnings, "answer_model": model,
            "deviated": bool(row) and model != names.get(row["best"], row["best"]),
            "deviation_from": answer.get("deviation_from"), "reason": (answer.get("reason") or "").strip()}


def build_arbitration_prompt(*, greedy_choice, hungarian_choice, cost_diff_label, belief_infeasibility, energy_error, num_peers, num_tasks, competition_ratio, belief_trend, belief_history, error_trend, error_history, competition_trend, competition_history, prompt_version=None) -> str:
    """SLM arbitration prompt (greedy vs Hungarian). Shared by the strategy and the replay benchmark."""
    style = prompt_style(prompt_version)
    if style == "judgement":
        facts = judgement_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio)
        return (JUDGEMENT_RULES + JUDGEMENT_ANSWER
                + _judgement_facts_text(facts, belief_infeasibility, belief_trend, belief_history, error_trend,
                                        error_history, competition_ratio, num_peers, num_tasks, cost_diff_label))
    if style in COMPACT_SCHEMAS:
        band, trend, comp = _derived_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio)
        return (COMPACT_RULES + COMPACT_ANSWER[style]
                + f"FACTS\nband: {band}\ntrend: {trend}\ncompetition: {comp}\n")
    step1_rules = STEP1_RULES["cost_gap_closed" if style == "stepwise" else style]
    prompt = f"""You are the coordination meta-controller for a delivery drone agent.
    Your job is to choose between two pre-computed assignment strategies for this decision cycle.

    === THE TWO OPTIONS ===

    GREEDY
      Task: {greedy_choice['parcel_id']}
      My cost: {greedy_choice['cost']:.3f}
      Logic: take the cheapest task for me personally, ignore team

    HUNGARIAN
      Task: {hungarian_choice['parcel_id']}
      My cost: {hungarian_choice['cost']:.3f}
      Logic: take my slot in the globally optimal team assignment

    === CURRENT SITUATION ===

    Cost differential: {cost_diff_label}

    Epistemic state:
      Belief infeasibility: {belief_infeasibility:.3f}  (0 = confident in peer cost estimates, 1 = very uncertain)
      Energy prediction error: {energy_error:.3f}  (running average of how much actual costs exceeded predictions)

    Competition:
      Peers visible: {num_peers}
      Tasks available: {num_tasks}
      Peers-per-task ratio: {competition_ratio:.3f}

    Trends over last {len(belief_history)} decision cycles:
      Uncertainty:        {belief_trend:8s}  history={belief_history}
      Prediction error:   {error_trend:8s}  history={error_history}
      Competition:        {competition_trend:8s}  history={competition_history}

    === DECISION RULES ===

{step1_rules}
    Step 2 — Uncertainty gating:
      Belief infeasibility > 0.7 → peer cost estimates are unreliable → choose GREEDY
      Belief infeasibility < 0.3 → estimates are trustworthy → coordination safe, choose HUNGARIAN
      Belief infeasibility 0.3-0.7 → proceed to Step 3

    Step 3 — Trend resolution (only reached under medium uncertainty):
      Uncertainty FALLING AND error FALLING → estimates improving → lean HUNGARIAN
      Uncertainty RISING OR error RISING → estimates degrading → lean GREEDY
      Both STABLE → use competition as tiebreaker: ratio > 0.15 → HUNGARIAN, else GREEDY

    === YOUR ANSWER ===

    Work through the steps above using the numbers provided.
    Then output a single JSON object with no other text:

    {{
      "policy": "<greedy or hungarian>",
      "confidence": <float 0.0-1.0 reflecting how clear-cut this decision was>
    }}
    """
    if style == "stepwise":
        # Threshold comparisons are computed here, not by the model: small models misjudge
        # them (e.g. 0.794 placed in "0.3 to 0.7"), while applying the step logic to given
        # facts is reliable (RuleArena: oracle arithmetic/logic tools give large gains).
        band, trend_case, comp = _derived_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio)
        derived = (
            "    === DERIVED FACTS (computed exactly; use these, do not recompute) ===\n\n"
            f"      Uncertainty band: {band}\n"
            f"      Trend case: {trend_case}\n"
            f"      Peers-per-task ratio above 0.15: {comp}\n\n"
        )
        situation = prompt.split("    === DECISION RULES ===")[0]
        return situation + derived + STEPWISE_RULES
    return prompt


class SPPDECLangGraph:
    """
    Phase 1 SPPDEC agent controller implemented using LangGraph.

    One instance is created per agent.
    The graph is authoritative over directive emission.

    Key Features (Phase 1):
    ----------------------
    - Physics-grounded peer energy belief tracking
    - Energy-adjusted peer cost predictions for task allocation
    - SLM-based strategic policy selection between:
      * Greedy: Local optimization (lowest cost for self)
      * Hungarian: Global optimization (optimal team assignment using peer cost beliefs)
    - True Hungarian assignment algorithm using believed cost matrix
    - No communication, coordination, or shared state
    - All beliefs derived from observable motion only
    """

    free_slm = False  # context_slm_free: the SLM may depart from the tables; only validity is enforced
    ARBITERS = ("slm", "always_greedy", "always_hungarian", "always_cbba", "rule_based",
                "classifier_regime", "classifier_tree", "context_rules", "context_slm", "context_slm_free", "context_slm_task")
    # Arbiters that choose among all four reasoning models from one shared context (see _arbiter_context).
    CONTEXT_ARBITERS = ("context_rules", "context_slm", "context_slm_free", "context_slm_task")
    CONTEXT_MODELS = ("greedy", "hungarian", "cbba_rounds")
    ASSUMPTION_AWAY_STEPS = 4     # consecutive observations of a peer moving away from its assumed task
    ASSUMPTION_TTL_STEPS = 600    # an assumption about an unobserved peer is dropped after this long
    INTENT_WINDOW_STEPS = 12      # observed motion over this many steps is read as intent
    CONTEST_MARGIN_CELLS = 1      # a peer must be this much closer to a target before the drone yields it
    INTENT_FRESH_STEPS = 24       # an observed intent counts only if the peer was seen closing in this recently
    YIELD_PATIENCE_STEPS = 120  # limited observation only: stop yielding after this long without a task

    def __init__(self, physics, variant: str = "combined", arbiter: str = "slm", prompt_version: Optional[str] = None,
                 symmetric_costs: bool = False, exclude_carrying_peers: bool = False, no_slot_policy: str = "greedy",
                 sensing: Optional[Dict] = None, peer_rate_assumption: str = "known",
                 conflict_policy: str = "trend", coordination_model: str = "hungarian",
                 cbba_bundle_size=1, peer_prior: str = "like_self",
                 assumption_memory: bool = False, observe_every_step: bool = False,
                 classifier_model: str = "distilled",
                 use_self_check: bool = False, own_rate_known: bool = True,
                 observe_gone: bool = True, observe_contested: bool = True,
                 world_view: bool = False, slm_suggest: bool = True, plan_ahead: bool = False,
                 slm_think: bool = False, rule_perturb: float = 0.0, freshness: bool = False,
                 territory: bool = False, territory_hint: Optional[str] = None,
                 ledger_complete: bool = False):
        """
        Parameters
        ----------
        physics
            Shared physics and energy cost model.
        energy_penalty_alpha : float, optional
            Linear penalty coefficient applied to peer costs based on
            their cumulative observed energy expenditure.

            Higher values increase penalty for peers who have moved more,
            making the local agent more competitive for tasks those peers
            might otherwise take.

            Recommended range: 0.01 - 0.05
            Default: 0.02 (empirically balanced)
        """
        self.physics = physics
        self.drone = None
        if arbiter not in self.ARBITERS:
            raise ValueError(f"Unknown arbiter {arbiter!r}; expected one of {self.ARBITERS}")
        # Who resolves greedy vs Hungarian (revision ablation, R1.3/R3.5):
        #   slm              deterministic pre-filter, then the SLM (the submitted method)
        #   always_greedy    greedy every cycle (no pre-filter, no SLM)
        #   always_hungarian Hungarian every cycle (no pre-filter, no SLM)
        #   rule_based       pre-filter, then the prompt's decision rules applied in code
        self.arbiter = arbiter
        # Shared-reasoning switches (all off = the submitted behaviour)
        self.symmetric_costs = symmetric_costs
        self.exclude_carrying_peers = exclude_carrying_peers
        self.no_slot_policy = no_slot_policy
        # Rule-based arbiter only: who wins when the trend and the competition lean opposite ways
        self.conflict_policy = conflict_policy
        # Which reasoning model the "hungarian" policy stands for: the Hungarian assignment or
        # CBBA simulated in the drone's head. Both are computed every decision.
        self.coordination_model = coordination_model
        # CBBA in the head runs both phases for the believed fleet, every message replaced by the bid the drone assumes the peer would send (bundle size below).
        # an integer, or "auto": each drone's fair share of the open tasks (see _bundle_size)
        self.cbba_bundle_size = cbba_bundle_size if cbba_bundle_size == "auto" else int(cbba_bundle_size)
        # Assumption memory: the drone keeps its assumption about which task each peer took in
        # the assignment it computed, and holds it until its own observations contradict it.
        # A peer with a live assumption is treated as busy and its task as taken.
        self.assumption_memory = assumption_memory
        # Per-step observation: the controller calls observe() every step, so peer tracking,
        # intent inference and assumption memory run continuously, and observation can end the
        # drone's own plan (target gone, or a closer peer is closing on it).
        self.observe_every_step = observe_every_step
        # Lightweight-classifier arbiters (src/strategies/sppdec/classifier.py): `classifier_tree` applies a
        # decision tree (`classifier_model`: "distilled" from the rule table, "slm_imitation" of the model's
        # raw answers) to the decision-time facts; `classifier_regime` picks greedy or coordination once per
        # mission from the fleet size and the number of parcels, with a tree that was trained without this
        # configuration.
        self.classifier_model = classifier_model
        # Track record of the drone's own assumptions about peers, per reasoning model: what each model
        # said a peer would take, against the parcel the peer was then seen to take (decayed hit counts).
        # Self-check: the drone compares the energy it predicted for its own task with what it actually used
        # and keeps a running correction factor; plan success is tracked as well. The signals are always
        # computed and logged. `use_self_check` is the ablation switch: when on, the correction scales every
        # energy estimate the drone makes (its own and its peers'), on the assumption that peers resemble it.
        # `own_rate_known` False hides the true consumption rate from the drone (it assumes nominal).
        self.use_self_check = use_self_check
        self.own_rate_known = own_rate_known
        # world_view: the free SLM is also shown the drone's own body, the terrain, every open task with its geometry and
        # energy margin, and its peers (see _world_view). slm_suggest: whether the prompt carries the tables' computed answer.
        self.world_view = world_view
        self.slm_suggest = slm_suggest
        # plan_ahead: the free SLM also gives an ordered plan of tasks; the tasks after the first one are the drone's bundle. The drone keeps
        # them in `planned_queue` and takes the next one that is still open without asking the model again (a task a peer
        # took meanwhile is skipped); the strategy plans the next bundle in the background (see SPPDECStrategy).
        self.plan_ahead = plan_ahead
        # slm_think: the free SLM reasons before it answers (qwen3 thinking mode); slower, the reply is free text that holds a JSON answer.
        self.slm_think = slm_think
        # rule_perturb: a measuring instrument for the headroom check, not a strategy. With this probability the rule list's
        # choice is replaced, at a decision where the reasoning models disagree, by a random one of the available models; the
        # spread of outcomes it produces says how much outcomes vary from chance alone.
        self.rule_perturb = rule_perturb
        self._perturb_rng = None
        # freshness: a parcel the drone has not seen for a while may already be gone, so a trip to it is worth less. Cost of a
        # trip to a parcel = cost / q, q the chance the parcel is still there (see _parcel_presence). Default off.
        self.freshness = freshness
        self._pick_steps: List[int] = []   # steps at which this drone picked up a parcel
        # territory: a convention every drone computes alike from the shared start information (no messages): a parcel belongs
        # to the drone whose start cell is nearest to it (ties: the lowest id). A drone with parcels of its own left considers
        # only those, and no peer takes part in its assignment; with none left it takes any parcel, coordinating as usual.
        self.territory = territory
        self._territory_owner: Optional[Dict[str, str]] = None
        # territory_hint (task-level SLM only): the SLM is told the territory convention in words and chooses the parcel itself.
        #   "labels"  every task is marked with its owner, computed by the code
        #   "raw"     the drones' start cells are listed and the SLM works out who owns what
        #   "both"    every task is marked with its owner and the drones' start cells are listed too
        self.territory_hint = territory_hint
        # ledger_complete: the station is a shared ledger. A delivered parcel stays on its cell, so the number of occupied station cells is the
        # number of parcels delivered; the total is known from the start information. When the station holds every parcel the job is finished and
        # whatever the drone still believes open is gone, so it stops looking and goes to park. (Probe: uses the station occupancy as the observation
        # gives it today, live and from anywhere; a faithful version needs the drone's memory of the station and visits to read it.)
        self.ledger_complete = ledger_complete
        # consensus (the Author's rules): the drone holds Hungarian and CBBA rounds as its two models, for itself and every peer, on one shared set of expected bids,
        # each with a bundle. It takes the tasks both give it, leaves the tasks both give a peer, and for the rest goes for what was not agreed. A drone with a
        # task available is never left waiting; a parcel assumed taken by a peer is predicted gone once the peer should have collected it.
        self.consensus = False
        self.time_weight = 0.0           # the energy-equivalent of one step of flight added to every bid (0: bids are energy alone)
        self.consensus_models = ("hungarian", "cbba_rounds")     # the models the drone reasons with: both (act on what they agree on), or one alone
        self.count_first = False      # when every parcel the drone believes open is predicted gone, it counts before it flies: the exact station count (all parcels minus delivered minus carriers in view) says whether any can still be on the ground
        self.time_weight_hungarian = 0.0  # the time weight applied only while the fleet-size rule has the drones reason with Hungarian (energy alone with CBBA)
        self.share_bundles = False    # bundle length per agent from its share of the team's capability (flight speed, limited by remaining energy at its own consumption rate), not from the head count
        self.rolling = False          # a drone commits to one trip at a time: it flies the first task of its bundle and decides again after the delivery; a peer is assumed to hold only the trip it is flying (the rest of its bundle stays open)
        self.equal_bid = False        # with the completion bid: time and energy count equally, each as a share of its ideal for the whole mission (ideal finish time = total flight work / total speed; ideal energy = every open task by its cheapest agent)
        self.completion_bid = False   # bids are the step at which an agent would complete its whole path (own declared speed, from when it is free), energy only a tie-break; a peer's live schedule keeps its parcels and sets when and where the peer is free
        self.speed_aware = False      # the drones' declared flight speeds enter the bids' time term and the timing of a peer's schedule (off: every drone is taken to fly at the nominal speed)
        self.keep_schedule = False    # a peer's agreed schedule is only ended by observation that contradicts it; a guess from the peer's motion does not replace it
        self.commitments = False      # an assignment stands until observation contradicts it: the parcels in a peer's live schedule are the peer's, and a peer with a live schedule is not a free agent in the solve
        self.no_gamble = False       # a drone acts only on the tasks the shared reasoning gives it; with none it goes to the station and decides again (no free, disputed or peer-owned task is taken)
        self.recheck = False         # a drone flying to its pickup asks the same reasoning again each time its cell changes, and gives the target up if it now belongs to a peer
        self.model_rule = None       # {"max_agents": n}: reason with CBBA alone in a fleet of at most n agents, else Hungarian alone (fleet size from the declared start information)
        self._bundles = {}           # {model: {agent id: [task ids]}} of the latest decision
        self._committed_tasks = set()  # parcels held by a peer's live schedule (only with `commitments`)
        self._predicted_gone = {}    # parcel id -> step it was predicted gone
        self._own_cells = []         # (step, cell) at each change of the drone's own cell, for its speed
        self._leg_start = None       # (step, cell) where the drone's current leg to a pickup began: its decision, or its last drop
        self._loaded_start = None    # (step, cell) of its last pickup
        self._pick_ratios = []       # steps per cell the drone took on its own legs to a pickup (waiting and handling included)
        self._loaded_ratios = []     # steps per cell on its own loaded legs
        self._compete_cell = None    # the pickup the drone chose knowing a peer is expected to take it (the reaction to a nearer peer leaves it alone)
        self._booked_trips = 0       # peers' trips booked so far (a delivery each, when the schedule runs out)
        self._booked_at_view = 0     # the count when the station was last in full view (what was booked since is not yet in the drone's memory of the cells)
        self._looking = False        # the drone has asked to see the station and has not yet got there
        self._gone_by = {}           # parcel id -> the peer the drone assumed would take it (logged, to tell a wrong taker from a wrong clock)
        self._energy_clock = None    # the step the peers' unscheduled energy was last accrued
        self._drop_plan = {}         # parcel id -> the drop cell expected for it this round (a cell assumed taken by one task is not assumed free for another)
        self.planned_queue: List[str] = []
        self.excluded_parcels: set = set()    # parcels left out of the tasks considered (already planned for this drone)
        self.last_slm_usage = None  # tokens read and written by the latest model call (for the hardware latency)
        self._start_info = None  # what the fleet knew before the first step (declared specs, map size)
        self.self_model = None  # SelfEnergyModel, created on the first step the drone is known (see watch_own_energy)
        self._watch_prev = None
        self._plan_hits = [0.0, 0.0]
        self._brier = [0.0, 0]  # squared error of the drone's own plan-success forecast, and how many forecasts
        self._model_preds = {}
        self._track = {m: [0.0, 0.0] for m in ("cbba", "cbba_rounds", "hungarian", "greedy")}
        self._prev_open = None
        self._classifier_models = None
        self._classifier_trees = {}
        self._regime_policy_cache = None
        # What per-step observation may do: end a plan whose target has vanished (observe_gone),
        # and give up a target a closer peer is closing on, also at the next decision (observe_contested).
        self.observe_gone = observe_gone
        self.observe_contested = observe_contested
        self._assignments = {}
        self._busy_peers = set()
        self._carrying_peers = set()
        self._self_pos = None
        self.prompt_version = prompt_style(prompt_version)
        # Decision guard: with the step-structured prompt styles the model's decision must
        # match the steps it wrote and the rule table; otherwise the rule result is used.
        # The free-text styles (submitted, cost_gap_closed) cannot be checked step by step:
        # they are only recorded, keeping the submitted setup reproducible.
        self.enforce_arbitration = self.prompt_version not in ("submitted", "cost_gap_closed")
        config = get_sppdec_config(variant)
        self.visibility_radius = float("inf")  # inf = full observability (Phase 1 default)
        # Observation model (see .ignore/revision/information_model.md). Default: exact, unlimited.
        #   radius     peers, and the state and weight of parcels, are observed only within this distance
        #   noise_std  Gaussian error (cells) on an observed peer position
        #   loss       probability that an observation of a peer is lost in a given step
        # Known at the start of the mission: parcel locations, the start state of every peer.
        sensing = sensing or {}
        self.sensing_radius = float(sensing.get("radius", float("inf")))
        self.sensing_noise_std = float(sensing.get("noise_std", 0.0))
        self.sensing_loss = float(sensing.get("loss", 0.0))
        # (parcel_radius: the range of parcel readings when it differs from the range of peer readings)
        self.sensing_active = (self.sensing_radius != float("inf") or self.sensing_noise_std > 0.0
                               or float(sensing.get("parcel_radius", float("inf"))) != float("inf")
                               or self.sensing_loss > 0.0)
        self.visibility_radius = self.sensing_radius
        # Energy profile of a peer: "known" = its true consumption rate (part of the start state),
        # "same_as_self" = the drone wrongly assumes every peer burns energy like itself.
        self.peer_rate_assumption = peer_rate_assumption
        # How the self-check carries over to peers: "like_self" or "nominal" (ablation of the self-model as a prior).
        self.peer_prior = peer_prior
        self._yield_wait = 0          # consecutive cycles in which yielding left no task
        self._start_state_done = False

        # Hybrid belief state per peer
        # Structure: {peer_id: {
        #   "last_pos":             (x, y),   # last *observed* position
        #   "last_seen_time":       int,       # sim step when last seen
        #   "energy_used":          float,     # cumulative energy from observed motion
        #   "predicted_position":   (x, y),   # current best position estimate
        #   "predicted_energy":     float,     # current best energy estimate
        #   "assigned_task":        None,      # hypothesis: parcel_id or None
        #   "confidence":           float,     # 1.0 = just seen, decays when invisible
        #   "is_visible":           bool,      # was peer visible this cycle?
        #   "time_since_last_seen": int,       # steps since last observation
        #   "last_velocity"
        # }}
        self.peer_beliefs = {}
        self.disable_belief_tracking = False  # ablation flag
        self.current_step = 0  # sim clock for confidence decay

        # Tunable parameter: controls strength of energy penalty on peer costs
        # Applied as: adjusted_cost = base_cost + (alpha * cumulative_energy_used)
        self.baseline_payload = config.baseline_payload
        self.energy_penalty_alpha = config.energy_penalty_alpha
        self.flight_recorder = None  # injected after construction

        # Agent tracking for Hungarian assignment
        self.agent_id = None
        self.last_visible_agents = []

        self.app = self._build()

    def set_drone(self, drone):
        """
            Bind a drone instance to this SPPDEC controller and apply agent-specific personality parameters.

            This method associates the physical drone with the cognitive controller and configures
            per-agent behavioral bias using the drone's assigned personality. Personality influences
            how the agent trades off self-cost (payload sensitivity) against inferred team cost
            (peer energy penalty).

            Behavior:
            - Reads `drone.personality` if present, otherwise defaults to "rational".
            - Maps personality to parameters via `SPPDEC_PERSONALITIES`.
            - Sets:
                - self.baseline_payload: controls sensitivity to own effort and payload cost.
                - self.energy_penalty_alpha: controls fairness pressure from inferred peer workload.
            - Emits a structured log for reproducibility and analysis.

            Must be called before executing the decision graph.
            """
        self.drone = drone

        if getattr(drone, "personality", None) is None:
            self._apply_sweep()
            return  # keep variant-based config intact

        personality = drone.personality
        profile = SPPDEC_PERSONALITIES.get(personality)
        if profile:
            self.baseline_payload = profile["baseline_payload"]
            self.energy_penalty_alpha = profile["energy_penalty_alpha"]
            self.avoid_width = profile.get("avoid_width", self.avoid_width)
        self._apply_sweep()

        print(
            f"[SPPDEC][{drone.agent_id}] Personality={personality} "
            f"payload={self.baseline_payload} "
            f"alpha={self.energy_penalty_alpha}"
        )

    def _apply_sweep(self):
        """Sensitivity study (R1.9): an arm may fix the fairness weight and the assumed payload of every drone, whatever its personality. Empty by default, so no existing arm changes."""
        sw = getattr(self, "sweep", None) or {}
        if "energy_penalty_alpha" in sw:
            self.energy_penalty_alpha = float(sw["energy_penalty_alpha"])
        if "baseline_payload" in sw:
            self.baseline_payload = float(sw["baseline_payload"])

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self):
        """
        Construct the LangGraph execution pipeline.

        Execution order:
            perceive
                -> evaluate_tasks
                -> reason_with_slm
                -> decide_mode
                -> select_drop_cell
                -> emit_directive or finalize
        """
        g = StateGraph(SPPDECAgentState)

        g.add_node("perceive", self.perceive)
        g.add_node("evaluate_tasks", self.evaluate_tasks)
        g.add_node("reason_with_slm", self.reason_with_slm)
        g.add_node("decide_mode", self.decide_mode)
        g.add_node("select_drop_cell", self.select_drop_cell)
        g.add_node("emit_directive", self.emit_directive)
        g.add_node("finalize", self.finalize)

        g.set_entry_point("perceive")

        g.add_edge("perceive", "evaluate_tasks")
        g.add_edge("evaluate_tasks", "reason_with_slm")
        g.add_edge("reason_with_slm", "decide_mode")
        g.add_edge("decide_mode", "select_drop_cell")

        g.add_conditional_edges(
            "select_drop_cell",
            self._route_after_decision,
            {
                "solo": "emit_directive",
                "wait": "finalize",
                "idle": "finalize",
                "look": "emit_directive",
                "coordinate": "finalize",
                "candidate": "finalize",
            },
        )

        g.add_edge("emit_directive", "finalize")
        g.add_edge("finalize", END)

        return g.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def perceive(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Perception and early termination logic for SPPDEC (Hybrid Phase).

        This node is responsible for state interpretation, not planning.

        Responsibilities
        ----------------
        1. Execution-phase guard:
           - Prevents replanning while the agent is already executing a plan
             (for example, carrying a parcel).

        2. Mission completion detection:
           - Determines whether the task environment is exhausted
             (no remaining parcels anywhere in the world).
           - This is treated as an objective world fact, not a planning decision.

        3. Local perception construction:
           - Identifies visible parcels.
           - Identifies visible peer agents within visibility_radius.
           - Derives station geometry for downstream feasibility and dropoff selection.

        4. Hybrid peer belief updates:
           - Case 1 (first seen): initialises full belief struct for new peer.
           - Case 1/3 (visible): snaps predicted position to actual observation,
             resets confidence to 1.0, re-estimates cumulative energy from observed
             position delta via local physics model (no communication).
           - Case 3 (reappearance): logs correction error between predicted and
             actual position before snapping — ground truth for prediction quality.
           - Case 2 (not visible): holds last known position, decays confidence
             on a half-life of 10 steps (floor 0.05), advances time_since_last_seen.
           - No communication, intent inference, or hidden state reconstruction.
           - All observations derived from local sensing within visibility_radius only.

        5. Energy prediction error learning:
           - Compares actual vs predicted energy consumption from completed tasks.
           - Updates running average of prediction errors for belief formation.
        """

        self._init_trace(state, "perceive")

        snap = state["snapshot"]
        agent = snap["agent"]
        agent_id = agent["id"]
        self._note_start_info(snap)

        # Limited observation: the snapshot is already what this drone senses; set up its start-state beliefs.
        if self.sensing_active:
            self._init_start_peers(snap, agent)

        # Track agent info for Hungarian assignment
        self.agent_id = agent_id

        # ==================================================
        # 0. Update prediction error (learning signal)
        # ==================================================
        current_battery = self.drone.power.level

        # Get persistent state from agent (not cycle state)
        if not hasattr(self, '_last_battery'):
            self._last_battery = None
            self._last_predicted_cost = 0.0
            self._prediction_error_history = []

        if self._last_battery is not None and self._last_predicted_cost > 0:
            actual_consumption = self._last_battery - current_battery
            predicted_consumption = self._last_predicted_cost

            if actual_consumption > 0:  # Only learn from actual task execution
                error = abs(actual_consumption - predicted_consumption)
                self._prediction_error_history.append(error)

                # Keep last 10 prediction errors for running average
                if len(self._prediction_error_history) > 10:
                    self._prediction_error_history.pop(0)

                # Update running average
                avg_error = sum(self._prediction_error_history) / len(self._prediction_error_history)
                state["energy_prediction_error"] = avg_error

                print(
                    f"[ENERGY LEARNING] Agent {agent_id}: predicted={predicted_consumption:.2f}, actual={actual_consumption:.2f}, error={error:.2f}, avg_error={avg_error:.3f}")
            else:
                # No energy consumed (idle/waiting), use previous error
                state["energy_prediction_error"] = state.get("energy_prediction_error", 0.0)
        else:
            # No previous prediction to compare against
            state["energy_prediction_error"] = state.get("energy_prediction_error", 0.0)

        self._last_battery = current_battery

        # ==================================================
        # 1. Execution-phase guard
        # ==================================================
        # NOTE:
        # This check uses self-proprioception only.
        # The agent may observe whether *itself* is carrying to avoid replanning mid-execution.
        # Peer carrying state is intentionally ignored to preserve the design constraint:
        # all peer reasoning must be derived from observable motion, not symbolic state.
        if snap.get("carrying", {}).get("has", False):
            state["decision_mode"] = "wait"
            state["directive"] = {"mode": "wait"}

            print(
                "[SPPDEC][%s] carrying parcel -> planning suspended",
                agent_id,
            )
            return state

        # ==================================================
        # 2. Mission completion check (world-level fact)
        # ==================================================
        remaining_parcels = [
            p for p in snap.get("all_parcels", [])
            if not p.get("picked") and not p.get("delivered")
        ]

        if self.ledger_complete and self._ledger_says_done(snap):
            remaining_parcels = []

        if not remaining_parcels:
            state["decision_mode"] = "idle"
            state["directive"] = {
                "mode": "idle",
                "reason": "no_remaining_parcels",
            }

            print(
                "[SPPDEC][%s] mission complete -> idle (no remaining parcels)",
                agent_id,
            )
            return state

        visible_agents = self._update_peers(state, snap, agent, agent_id)

        # Captures both visible-snapped and invisible-predicted peers in one pass
        actual_peer_energies = {
            a["id"]: a.get("battery", None)
            for a in visible_agents
            if a.get("battery") is not None
        }

        if self.flight_recorder is not None:
            self.flight_recorder.tick_hybrid(
                observer_id=agent_id,
                step=self.current_step,
                sim_time=self.current_step * (1.0 / 60.0),
                peer_beliefs=self.peer_beliefs,
                actual_peer_energies=actual_peer_energies,
            )
        # ==================================================
        # 5. Parcel visibility
        # ==================================================
        state["visible_parcels"] = remaining_parcels

        print(
            "[SPPDEC][%s] perceive: %d visible parcels",
            agent_id,
            len(state["visible_parcels"]),
        )

        # ==================================================
        # 6. Station geometry derivation
        # ==================================================
        station = snap.get("nearest_station")
        if station:
            snap["station_cells"] = [
                (c, r)
                for r in range(station["row"], station["row"] + station["h"])
                for c in range(station["col"], station["col"] + station["w"])
            ]

            print(
                "[SPPDEC][%s] station geometry resolved (%d cells)",
                agent_id,
                len(snap["station_cells"]),
            )
        else:
            snap["station_cells"] = None
            logger.warning(
                "[SPPDEC][%s] no station visible in snapshot",
                agent_id,
            )

        return state

    def evaluate_tasks(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Convert visible parcels into locally feasible pickup tasks.

        Responsibilities
        ----------------
        - Filters parcels based on local energy feasibility
        - Estimates self cost using physics model
        - Predicts peer costs using physics + cumulative energy penalty
        - Produces task list and peer-adjusted cost map for SLM reasoning
        - Detects local infeasibility and emits idle directive when appropriate

        Important semantic distinction
        ------------------------------
        - This function does NOT declare mission completion.
        - It only expresses local incapacity: "I cannot act given my constraints".
        - The runner may still decide that other agents can complete the mission.

        Energy model
        ------------
        - Self cost uses true task weight
        - Peer cost uses baseline payload only
        - Peer cost is adjusted by:
            adjusted_cost = base_cost + alpha * cumulative_energy_used
        """
        self._init_trace(state, "evaluate_tasks")

        if state.get("decision_mode") == "idle":
            state["feasible_tasks"] = []
            state["peer_distance_map"] = {}
            return state

        snap = state["snapshot"]
        agent = snap["agent"]
        agent_id = agent["id"]
        peers = state.get("visible_agents", [])
        self._carrying_peers = {a["id"] for a in peers if a.get("carrying")}
        # Peers out of sensing range still compete: they enter by their predicted position.
        seen_ids = {a["id"] for a in peers}
        peers = list(peers) + [
            {"id": pid, "col": int(round(b["predicted_position"][0])),
             "row": int(round(b["predicted_position"][1])), "carrying": False}
            for pid, b in self.peer_beliefs.items() if pid not in seen_ids
        ]
        self._self_pos = (agent["col"], agent["row"])

        # --------------------------------------------------
        # NEW: Competition / scarcity awareness
        # --------------------------------------------------
        num_agents = len(self.peer_beliefs) + 1
        num_tasks = len(state.get("visible_parcels", [])) or 1
        competition_ratio = num_agents / num_tasks
        allow_yield = competition_ratio < 0.8

        print(f"[YIELD-GATE] competition={competition_ratio:.2f}, allow_yield={allow_yield}")

        state["belief_infeasibility"] = self._compute_infeasibility_belief(state)
        station = snap.get("nearest_station")
        station_center = ((station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
                          if station else (agent["col"], agent["row"]))
        _drops: Dict[tuple, tuple] = {}

        def drop_of(pk):
            """The drop cell a task at `pk` is priced to: the nearest station cell the drone remembers free (every cost in SPPDEC is a whole trip)."""
            if pk not in _drops:
                _drops[pk] = self._drop_for(pk, snap) if station else station_center
            return _drops[pk]

        battery = self.drone.power.level
        feasible: List[Dict] = []
        yielded_any = False
        peer_distance_map: Dict[str, Dict[str, float]] = {}

        visible_parcels = [p for p in state.get("visible_parcels", []) if p["id"] not in self.excluded_parcels]
        if self.consensus:
            self._note_own_cell(snap)
            visible_parcels = self._without_predicted_gone(visible_parcels, snap)
            state["visible_parcels"] = visible_parcels
        presence = self._parcel_presence(snap, visible_parcels)    # parcel id -> chance it is still there
        # Territory: with parcels of its own left, the drone considers only those and assigns among no peers (they have their own).
        cost_peers = peers
        if self.territory:
            owner = self._territory_owners()
            mine = [p for p in visible_parcels if owner.get(p["id"]) == agent_id]
            if mine:
                visible_parcels, cost_peers = mine, []

        # --------------------------------------------------
        # Peer intent modeling
        # --------------------------------------------------
        peer_best_task: Dict[str, Tuple[str, float]] = {}
        peer_second_best_cost: Dict[str, float] = {}
        peer_cost_rows: Dict[str, Dict[str, float]] = {}      # peer -> parcel id -> its adjusted cost (to rank each peer's best parcels)
        peer_effective_conf: Dict[str, float] = {}

        # ---- Confidence calibration ----
        for peer in peers:
            peer_id = peer["id"]
            belief = self.peer_beliefs.get(peer_id)
            if not belief:
                continue

            effective_conf = belief["confidence"]

            if "predicted_energy" in belief and "last_actual_energy" in belief:
                error = abs(belief["predicted_energy"] - belief["last_actual_energy"])
                norm_error = error / (belief["last_actual_energy"] + 1e-6)
                effective_conf = math.exp(-2.0 * norm_error)

            belief["effective_confidence"] = effective_conf
            peer_effective_conf[peer_id] = effective_conf

        # ---- Compute best + second-best ----
        for p in visible_parcels:
            parcel_id = p["id"]
            pickup = (p["col"], p["row"])

            for peer in cost_peers:
                peer_id = peer["id"]
                belief = self.peer_beliefs.get(peer_id)

                effective_conf = peer_effective_conf.get(peer_id, 0.0)
                adjusted_cost = (self._model_peer_cost(peer, pickup, belief, effective_conf, snap, p.get("weight", 1.0))
                                 if self.peer_model_cost else None)
                if adjusted_cost is None:
                    if belief and not belief["is_visible"]:
                        peer_pos = belief["predicted_position"]
                    else:
                        peer_pos = (peer["col"], peer["row"])

                    base_cost = self._trip_energy(peer_id, peer_pos, pickup, drop_of(pickup), p.get("weight", 1.0))

                    if belief:
                        penalty = self.energy_penalty_alpha * belief["predicted_energy"] * effective_conf
                        adjusted_cost = base_cost + penalty
                    else:
                        adjusted_cost = base_cost
                adjusted_cost /= presence.get(parcel_id, 1.0)
                peer_cost_rows.setdefault(peer_id, {})[parcel_id] = adjusted_cost

                best = peer_best_task.get(peer_id)

                if best is None:
                    peer_best_task[peer_id] = (parcel_id, adjusted_cost)
                    peer_second_best_cost[peer_id] = float("inf")
                else:
                    _, best_cost = best

                    if adjusted_cost < best_cost:
                        peer_second_best_cost[peer_id] = best_cost
                        peer_best_task[peer_id] = (parcel_id, adjusted_cost)
                    else:
                        if adjusted_cost < peer_second_best_cost.get(peer_id, float("inf")):
                            peer_second_best_cost[peer_id] = adjusted_cost

        # ---- Persist intent ----
        for peer_id, (best_task_id, best_cost) in peer_best_task.items():
            belief = self.peer_beliefs.get(peer_id)
            if not belief:
                continue

            belief["predicted_task"] = best_task_id

            second_cost = peer_second_best_cost.get(peer_id, float("inf"))

            if second_cost < float("inf"):
                commitment = 1.0 - (best_cost / (second_cost + 1e-6))
            else:
                commitment = 1.0

            belief["task_commitment_score"] = commitment

            print(
                f"[BELIEF] peer={peer_id} "
                f"task={best_task_id[:8]} "
                f"conf={belief.get('effective_confidence', 0.0):.2f} "
                f"commit={commitment:.2f}"
            )

        # --------------------------------------------------
        # Avoid the peers' best parcels (option avoid_top_k): no waiting, the drone goes for its best parcel outside every free peer's k best, where k shrinks with
        # the open parcels per peer so that something is always left; if energy leaves nothing outside, it competes as usual.
        # --------------------------------------------------
        avoid_set: set = set()
        avoided: List[Dict] = []
        self._peer_top3 = {pid: sorted(row, key=row.get)[:3] for pid, row in peer_cost_rows.items()}
        if self.avoid_top_k and peer_cost_rows:
            free_peers = [pid for pid in peer_cost_rows if pid not in self._carrying_peers]
            k = min(int(self.avoid_width if self.avoid_width_fixed is None else self.avoid_width_fixed), (len(visible_parcels) - 1) // max(1, len(free_peers)))
            ghost_rows = self._ghost_rows(snap, cost_peers, peer_effective_conf) if (self.avoid_beliefs and self._peer_model is not None) else {}
            self._peer_top3_open = {pid: sorted(r, key=r.get)[:3] for pid, r in peer_cost_rows.items()}
            self._peer_top3 = {pid: sorted({**peer_cost_rows[pid], **ghost_rows.get(pid, {})}, key=lambda i: {**peer_cost_rows[pid], **ghost_rows.get(pid, {})}[i])[:3]
                               for pid in peer_cost_rows}
            if k > 0:
                for pid in free_peers:
                    rows = {**peer_cost_rows[pid], **ghost_rows.get(pid, {})}      # parcels the peer is believed to think open (ghosts take up its slots)
                    avoid_set.update(sorted(rows, key=rows.get)[:k])

        # --------------------------------------------------
        # Build feasible task set
        # --------------------------------------------------
        for p in visible_parcels:
            parcel_id = p["id"]
            pickup = (p["col"], p["row"])

            task = {
                "parcel_id": parcel_id,
                "pickup": pickup,
                "weight": p.get("weight", 1.0),
            }

            self_cost = self._trip_energy(self.agent_id, (snap["agent"]["col"], snap["agent"]["row"]), pickup, drop_of(pickup), task["weight"])
            self_cost /= presence.get(parcel_id, 1.0)

            # Feasible trip: the same definition for every strategy. Usable energy (battery above the drone's hardware
            # floor) must cover the pickup leg, the delivery leg to the station and the pick and the drop.
            if battery - float(self.drone.power.floor) < self._trip_energy(
                    self.agent_id, (snap["agent"]["col"], snap["agent"]["row"]), pickup,
                    self._bid_drop(pickup, station) if station else station_center, task["weight"]):
                continue

            peer_costs: Dict[str, float] = {}

            for peer in cost_peers:
                peer_id = peer["id"]
                belief = self.peer_beliefs.get(peer_id)

                effective_conf = peer_effective_conf.get(peer_id, 0.0)
                adjusted_cost = (self._model_peer_cost(peer, pickup, belief, effective_conf, snap, task["weight"])
                                 if self.peer_model_cost else None)
                if adjusted_cost is None:
                    if belief and not belief["is_visible"]:
                        peer_pos = belief["predicted_position"]
                    else:
                        peer_pos = (peer["col"], peer["row"])

                    base_cost = self._trip_energy(peer_id, peer_pos, pickup, drop_of(pickup), task["weight"])

                    if belief:
                        penalty = self.energy_penalty_alpha * belief["predicted_energy"] * effective_conf
                        adjusted_cost = base_cost + penalty
                    else:
                        adjusted_cost = base_cost

                peer_costs[peer_id] = adjusted_cost / presence.get(parcel_id, 1.0)

            should_yield = False

            # With limited observation, a drone that has been held back by yielding for a long
            # time stops yielding and goes to look (a parcel it believes is wanted may have been
            # delivered long ago, and it only finds out by going there).
            patient = self.sensing_active and self._yield_wait >= self.YIELD_PATIENCE_STEPS

            for peer_id, peer_cost in ({} if patient else peer_costs).items():
                belief = self.peer_beliefs.get(peer_id)
                if not belief:
                    continue

                effective_conf = peer_effective_conf.get(peer_id, 0.0)
                predicted_task = belief.get("predicted_task")
                commitment = belief.get("task_commitment_score", 0.0)

                if effective_conf < 0.6:
                    continue

                if predicted_task == parcel_id and commitment > 0.8:
                    print(
                        f"[YIELD-COMMIT] parcel={parcel_id[:8]} skipped — peer {peer_id} committed ({commitment:.2f})")
                    should_yield = True
                    break

                if abs(peer_cost - self_cost) < 0.05 * self_cost:
                    continue

                margin = (0.1 + (1 - effective_conf)) * self_cost

                if allow_yield and peer_cost + margin < self_cost and predicted_task == parcel_id and commitment > 0.6:
                    print(
                        f"[YIELD] parcel={parcel_id[:8]} skipped — peer {peer_id} better (conf={effective_conf:.2f}, commit={commitment:.2f})")
                    should_yield = True
                    break

            # Anti-deadlock: override yield when all peers are at commitment=1.0
            # This fires specifically when only one task remains, because the
            # commitment formula collapses to 1.0 for all peers when there is
            # no second-best task. Every agent yields to every other agent,
            # producing infinite wait. The cheapest agent breaks the tie.
            if should_yield:
                all_peers_max_committed = all(
                    b.get("task_commitment_score", 0.0) >= 1.0
                    for b in self.peer_beliefs.values()
                )

                if all_peers_max_committed:
                    all_known_agents = sorted(
                        list(self.peer_beliefs.keys()) + [self.agent_id]
                    )
                    if all_known_agents[0] == self.agent_id:
                        print(
                            f"[DEADLOCK-BREAK] parcel={parcel_id[:8]} all peers at max commitment "
                            f"→ lowest agent ID {self.agent_id} takes the task"
                        )
                        should_yield = False

            if not should_yield and not patient and self.observe_every_step and self.observe_contested:
                # A peer seen closing on this task from closer than the drone is takes it.
                d_me = self.physics.distance(self._self_pos, pickup)
                for pid, b in self.peer_beliefs.items():
                    a = b.get("assumed")
                    if (a and a["task"] == parcel_id and a.get("source") == "observed" and b.get("is_visible")
                            and snap.get("step", 0) - a.get("seen_moving", -10 ** 9) <= self.INTENT_FRESH_STEPS
                            and self.physics.distance(b.get("smooth_pos") or b["predicted_position"], pickup)
                            + self.CONTEST_MARGIN_CELLS <= d_me):
                        should_yield = True
                        break

            if should_yield and not self.consensus:
                yielded_any = True
                continue

            if parcel_id in avoid_set:
                avoided.append({"task_type": "feasible", "parcel_id": parcel_id, "pickup": pickup, "weight": task["weight"], "cost": self_cost,
                                "_peers": peer_costs})
                continue

            feasible.append({
                "task_type": "feasible",
                "parcel_id": parcel_id,
                "pickup": pickup,
                "weight": task["weight"],
                "cost": self_cost,
            })

            peer_distance_map[parcel_id] = peer_costs

        if not feasible and avoided:                 # nothing outside the peers' best parcels is affordable: compete for them rather than idle
            for t in avoided:
                peer_distance_map[t["parcel_id"]] = t.pop("_peers")
                feasible.append(t)
        state["feasible_tasks"] = feasible
        state["peer_distance_map"] = peer_distance_map
        if self.sensing_active:
            self._yield_wait = self._yield_wait + 1 if (yielded_any and not feasible) else 0

        self._assignments = {}
        self._bundles = {}
        self._busy_peers = set()
        coord_tasks = feasible
        if self.assumption_memory:
            claimed = self._refresh_assumptions(state)
            if not self.consensus:                      # consensus keeps every peer in the matrix and every parcel a candidate
                self._busy_peers = set(claimed.values())
                coord_tasks = [t for t in feasible if t["parcel_id"] not in claimed]
            elif self.commitments or self.completion_bid:   # unless the assignments stand: a peer's live schedule keeps its parcels and the peer
                self._committed_tasks, self._busy_peers = self._peer_commitments()
                if self.completion_bid:
                    self._busy_peers = set()                # the completion bid prices a busy peer by when it is free, so it stays in the matrix

        if feasible:
            state["greedy_choice"] = min(feasible, key=lambda t: t["cost"])
            # Every reasoning model runs on every decision; the arbiter decides from all of them.
            # With assumption memory the coordination models see only the tasks no peer is
            # assumed to have taken, and only the peers assumed to be free.
            def run_models(tasks):
                # In consensus mode only the model(s) the drone reasons with are solved: the other one would not be read.
                used = self._models() if self.consensus else ("hungarian", "cbba_rounds")
                state["hungarian_choice"] = self._solve_hungarian(tasks, peer_distance_map, state) if tasks and "hungarian" in used else None
                if not tasks or self.consensus:
                    state["cbba_choice"] = None
                else:
                    state["cbba_choice"] = self._solve_cbba_bid_rounds(state, tasks)
                if self.arbiter in self.CONTEXT_ARBITERS:
                    # The context arbiters choose among the reasoning models, so they are computed here.
                    rounds = (self._solve_cbba_bid_rounds(state, tasks, key="cbba_rounds")
                              if tasks and "cbba_rounds" in used else None)
                    state["model_choices"] = {"greedy": state["greedy_choice"], "hungarian": state["hungarian_choice"],
                                              "cbba_rounds": rounds}

            run_models(coord_tasks)
            state["assumed_assignment"] = {
                m: {p: t for p, t in a.items() if p != self.agent_id}
                for m, a in self._assignments.items()
            }
            if self.assumption_memory:
                self._record_assumptions(state)
            self._remember_predictions(state)
            state["scenario"] = self._scenario_descriptors(state)
            state["measured_context"] = self._measured_context(state)

        return state

    def _refresh_assumptions(self, state) -> Dict[str, str]:
        """Assumption refresh at a decision (see _refresh_core)."""
        snap = state["snapshot"]
        step = snap.get("step", self.current_step)
        open_tasks = {p["id"]: p for p in state.get("visible_parcels", [])}
        return self._refresh_core(step, open_tasks)

    def _refresh_core(self, step, open_tasks) -> Dict[str, str]:
        """
        Drop assumptions the drone's own observations contradict; return the live ones as
        {task id: peer id}. An assumption ends when the task is no longer open as far as the drone
        knows, when the peer is seen carrying a parcel, when the peer is seen moving away from
        the task several observations in a row, or when the peer has not been seen for a long time.
        """
        claimed: Dict[str, str] = {}
        if self.consensus:
            self._accrue_unscheduled_energy(step, open_tasks)
        for pid, b in self.peer_beliefs.items():
            if self.consensus and pid not in self._carrying_peers:
                b["carry_booked"] = False
            a = b.get("assumed")
            if not a:
                continue
            tid = a["task"]
            if self.consensus and a.get("gone_at") is not None:
                if pid in self._carrying_peers:
                    if not b.get("carry_booked"):
                        # seen carrying: the peer has begun a trip. The scheduled task's trip stands for it and is booked now; the schedule moves on.
                        self._book_trip(pid, b, a)
                        b["carry_booked"] = True
                        queue = a.get("queue") or []
                        b["assumed"] = ({"since": step, "seen": step, "away": 0, **queue[0], "queue": queue[1:]} if queue else None)
                    continue
                if not a.get("marked") and tid not in open_tasks:
                    b["assumed"] = None
                    continue
                if not a.get("marked") and step >= a["gone_at"]:
                    self._predicted_gone[tid] = step     # predicted gone: the peer should have collected it by now
                    self._gone_by[tid] = pid
                    a["marked"] = True
                if a.get("marked"):
                    if step >= a["trip_end"]:            # the trip is over: its energy is booked and the peer's next agreed task begins
                        self._book_trip(pid, b, a)
                        queue = a.get("queue") or []
                        b["assumed"] = ({"since": step, "seen": step, "away": 0, **queue[0], "queue": queue[1:]} if queue else None)
                    continue
            if tid not in open_tasks or pid in self._carrying_peers:
                b["assumed"] = None
                continue
            if b.get("is_visible"):
                pickup = (open_tasks[tid]["col"], open_tasks[tid]["row"])
                d = self.physics.distance(b["predicted_position"], pickup)
                if d > a["last_dist"] + 1e-9:
                    a["away"] += 1
                elif d < a["last_dist"] - 1e-9:
                    a["away"] = 0
                a["last_dist"] = d
                a["seen"] = step
                if a["away"] >= self.ASSUMPTION_AWAY_STEPS:
                    b["assumed"] = None
                    continue
            elif step - a["seen"] > self.ASSUMPTION_TTL_STEPS:
                b["assumed"] = None
                continue
            claimed[tid] = pid
        return claimed

    def _record_assumptions(self, state) -> None:
        """Remember which task the assignment gave each peer that had no live assumption."""
        source = "cbba" if (self.arbiter == "always_cbba" or self.coordination_model == "cbba") else "hungarian"
        if self.arbiter in self.CONTEXT_ARBITERS:
            # Both reasoning models are held. A peer is assumed to take a parcel only where Hungarian and CBBA rounds converge on it; where they differ the drone
            # assumes nothing about that peer (it abstains). No observation of what peers did is used to choose between the models.
            h = self._assignments.get("hungarian") or {}
            c = self._assignments.get("cbba_rounds") or {}
            assignment = {pid: tid for pid, tid in h.items() if c.get(pid) == tid}
        else:
            assignment = self._assignments.get(source) or {}
        step = state["snapshot"].get("step", self.current_step)
        open_tasks = {p["id"]: p for p in state.get("visible_parcels", [])}
        if self.consensus:
            self._record_agreed_bundles(state, step, open_tasks)
            return
        for pid, tid in assignment.items():
            if pid == self.agent_id or pid not in self.peer_beliefs or tid not in open_tasks:
                continue
            b = self.peer_beliefs[pid]
            if b.get("assumed") or pid in self._carrying_peers:
                continue
            pickup = (open_tasks[tid]["col"], open_tasks[tid]["row"])
            b["assumed"] = {"task": tid, "since": step, "seen": step, "away": 0,
                            "last_dist": self.physics.distance(b["predicted_position"], pickup)}
            if self.consensus:
                station = state["snapshot"].get("nearest_station")
                spc = self._steps_per_cell()
                start = tuple(int(round(v)) for v in b["predicted_position"])
                d_pick = self.physics.distance(start, pickup)
                spc *= self._peer_pace(pid)
                b["assumed"].update({
                    "start_pos": start, "pickup": pickup, "weight": open_tasks[tid].get("weight", 1.0),
                    "drop": ((station["col"] + station["w"] // 2, station["row"] + station["h"] // 2) if station else pickup),
                    # by this step the peer should have collected the parcel (its distance at the fleet's speed, with slack)
                    "gone_at": step + 1.25 * d_pick * spc + 10,
                })

    def _models(self):
        """The models the drone reasons with. Under `model_rule` the choice reads only the declared fleet size, which every drone shares, so all drones choose alike."""
        rule, info = self.model_rule, self._start_info
        if not rule or info is None:
            return self.consensus_models
        return ("cbba_rounds",) if len(info.fleet) <= rule["max_agents"] else ("hungarian",)

    def _look_cell(self, snap):
        """
        The station cell this drone goes to when it looks (a cell from which the whole station is in view). Every drone holds the same fleet list and the same station, so each
        takes a different cell by the same rule (its rank among the fleet's ids, cells ordered from the centre outward): no two idle drones share a cell and nothing is sent.
        """
        st = snap.get("nearest_station")
        info = self._start_info
        if not st:
            return None
        cells = [(c, r) for r in range(st["row"], st["row"] + st["h"]) for c in range(st["col"], st["col"] + st["w"])]
        rad = getattr(self, "visibility_radius", float("inf"))
        view = [c for c in cells if all(abs(c[0] - o[0]) + abs(c[1] - o[1]) <= rad for o in cells)]
        if not view:
            return (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
        mid = (st["col"] + (st["w"] - 1) / 2.0, st["row"] + (st["h"] - 1) / 2.0)
        view.sort(key=lambda c: (abs(c[0] - mid[0]) + abs(c[1] - mid[1]), c[0], c[1]))
        ids = sorted(info.fleet) if info is not None else [self.agent_id]
        k = ids.index(self.agent_id) if self.agent_id in ids else 0
        return view[k % len(view)]

    def _equal_refs(self, agents, tasks, drop, snap):
        """The two ideals of the equal-importance bid, from information every drone holds: the ideal finish time (all open flight work shared by the whole fleet at its total speed) and the ideal
        energy (every open task flown by its cheapest agent from the station). The agents' fairness penalty is brought to the same scale."""
        spc = self.physics.steps_per_cell
        dist = self.physics.distance
        work = 0.0
        e_ideal = 0.0
        for t in tasks:
            d = self._drop_plan.get(t["id"], drop) if hasattr(self, "_drop_plan") else drop
            work += (dist(drop, t["pickup"]) + dist(t["pickup"], d)) * spc
            e_ideal += min(a.get("rate", 1.0) for a in agents) * (self.physics.energy_cost(distance=dist(drop, t["pickup"]), weight=0.0)
                                                                 + self.physics.energy_cost(distance=dist(t["pickup"], d), weight=t["weight"]) + 2.0 * self.physics.pick_drop_cost(t["weight"]))
        total_speed = sum(float(a.get("speed", 1.0)) or 1.0 for a in agents)
        t_ref, e_ref = max(1.0, work / total_speed), max(1e-6, e_ideal)
        for a in agents:
            a["t_ref"], a["e_ref"] = t_ref, e_ref
            a["penalty"] = a.get("penalty", 0.0) / e_ref

    def _free_when(self, pid, b, pos, snap):
        """Where a peer will be and in how many steps it is free, as far as the drone knows: at the end of its live schedule (the last trip's drop and end step), or at the station when it is
        seen carrying (the rest of its flight at its declared speed), else here and now."""
        step = snap.get("step", self.current_step)
        a = b.get("assumed")
        if a and a.get("trip_end") is not None:
            last = (a.get("queue") or [a])[-1]
            return tuple(last["drop"]), max(0.0, float(last["trip_end"]) - step)
        if pid in self._carrying_peers:
            st = snap.get("nearest_station")
            if st:
                centre = (st["col"] + st["w"] // 2, st["row"] + st["h"] // 2)
                return centre, self.physics.distance(pos, centre) * self.physics.steps_per_cell / self._speed_of(pid)
        return pos, 0.0

    def _peer_commitments(self):
        """The parcels held by the live schedules of the peers (the one it is flying to and the rest of its agreed bundle) and the peers that hold them. A schedule is live until
        observation or its own clock ends it (`_refresh_core`); a parcel already predicted gone is not listed (it is out of the open set)."""
        tasks, busy = set(), set()
        for pid, b in self.peer_beliefs.items():
            a = b.get("assumed")
            if not a:
                continue
            busy.add(pid)
            if not a.get("marked"):
                tasks.add(a["task"])
            tasks.update(q["task"] for q in (a.get("queue") or []) if q.get("task"))
        return tasks, busy

    def _agreed_for(self, aid) -> List[str]:
        """The tasks every chosen model holds in its bundle for `aid`, in the order of the last chosen model (CBBA's when it is chosen)."""
        lists = [self._bundles.get(m, {}).get(aid, []) for m in self._models()]
        if not lists:
            return []
        return [t for t in lists[-1] if all(t in set(l) for l in lists)]

    def _record_agreed_bundles(self, state, step, open_tasks) -> None:
        """
        For each peer with no live schedule: the tasks both of its bundles hold, in CBBA's order, flown one after another from its predicted position
        (to the pickup, to the task's expected drop cell, on to the next pickup).
        Each task carries the step by which the peer should have collected the parcel (`gone_at`) and the step its trip ends (`trip_end`); the legs take what the
        drone's own legs have taken per cell (`_leg_ratios`), plus 5 steps of handling.
        """
        station = state["snapshot"].get("nearest_station")
        centre = (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2) if station else None
        r_pick, r_load = self._leg_ratios()
        for pid in self.peer_beliefs:
            if pid == self.agent_id or pid in self._carrying_peers:
                continue
            b = self.peer_beliefs[pid]
            if b.get("assumed"):
                continue
            agreed = [t for t in self._agreed_for(pid) if t in open_tasks]
            if self.rolling:
                agreed = agreed[:1]
            if not agreed:
                continue
            pos = tuple(int(round(v)) for v in b["predicted_position"])
            t0 = step
            queue = []
            pace = self._peer_pace(pid)
            for tid in agreed:
                pick = (open_tasks[tid]["col"], open_tasks[tid]["row"])
                drop = self._drop_plan.get(tid) or centre or pick
                pick_at = t0 + self.physics.distance(pos, pick) * r_pick * pace + 5
                end_at = pick_at + self.physics.distance(pick, drop) * r_load * pace + 5
                queue.append({"task": tid, "pickup": pick, "weight": open_tasks[tid].get("weight", 1.0), "drop": drop, "start_pos": pos,
                              "gone_at": pick_at, "trip_end": end_at, "last_dist": self.physics.distance(pos, pick)})
                pos, t0 = drop, end_at
            b["assumed"] = {"since": step, "seen": step, "away": 0, **queue[0], "queue": queue[1:]}

    def observe(self, snap: Dict, own_target=None) -> Dict:
        """
        One observation step, called by the controller every step while the drone executes or
        waits. No model call and no planning. Updates beliefs about the peers within sensing range,
        reads their observed motion as intent, refreshes the assumption memory, and tells the
        controller when the drone's own plan should end.
        Returns {} or {"replan": "target_gone" | "contested"}.
        """
        agent = snap["agent"]
        agent_id = agent["id"]
        self.agent_id = agent_id
        if self.consensus:
            self._note_own_cell(snap)
            if snap.get("station_in_full_view"):
                self._booked_at_view = self._booked_trips        # the memory of the cells is exact again: nothing booked before now is outstanding
        if self.sensing_active:
            self._init_start_peers(snap, agent)
        step = snap.get("step", self.current_step)
        self._last_station = snap.get("nearest_station")
        visible = self._update_peers({}, snap, agent, agent_id)
        self._carrying_peers = {a["id"] for a in visible if a.get("carrying")}
        open_tasks = {p["id"]: p for p in snap.get("all_parcels", [])
                      if not p.get("picked") and not p.get("delivered")}
        if self.peer_model_shadow or self.station_model or self.peer_model_cost or self.avoid_beliefs:
            if self._peer_model is None:
                from .predictors import PeerModel
                self._peer_model = PeerModel(self.physics.distance)
            self._peer_model.observe(int(step), (agent["col"], agent["row"]), visible, snap.get("start_info"), agent_id, snap.get("nearest_station"))
        self._observe_intents(visible, open_tasks, step)
        self._score_predictions(visible, open_tasks)
        if self.assumption_memory:
            self._refresh_core(step, open_tasks)
        return self._own_plan_signal(agent, open_tasks, own_target, step)

    # ------------------------------------------------------------------
    # Scenario descriptors and the track record of assumptions
    # ------------------------------------------------------------------
    TRACK_DECAY = 0.97

    def _remember_predictions(self, state) -> None:
        """What each model says every peer will take, to be checked against what the peer is seen to take."""
        step = state["snapshot"].get("step", self.current_step)
        open_tasks = state.get("visible_parcels", [])
        for pid, b in self.peer_beliefs.items():
            if pid in self._carrying_peers:
                continue
            preds = {"step": step}
            for m in ("cbba", "cbba_rounds", "hungarian"):
                preds[m] = (self._assignments.get(m) or {}).get(pid)
            pos = b.get("predicted_position")
            if pos is not None and open_tasks:
                preds["greedy"] = min(
                    open_tasks, key=lambda t: (self.physics.distance(pos, (t["col"], t["row"])), t["col"], t["row"]))["id"]
            self._model_preds[pid] = preds

    def _score_predictions(self, visible, open_tasks) -> None:
        """When a parcel disappears next to exactly one peer we predicted for, credit the models that named it."""
        if self._prev_open is not None:
            for tid in set(self._prev_open) - set(open_tasks):
                pk = self._prev_open[tid]
                near = [a for a in visible if a["id"] in self._model_preds
                        and self.physics.distance((a["col"], a["row"]), pk) <= 1.5]
                if len(near) != 1:
                    continue
                preds = self._model_preds.pop(near[0]["id"])
                for m, rec in self._track.items():
                    if preds.get(m) is None:
                        continue
                    rec[0] = rec[0] * self.TRACK_DECAY + (1.0 if preds[m] == tid else 0.0)
                    rec[1] = rec[1] * self.TRACK_DECAY + 1.0
        self._prev_open = {tid: (p["col"], p["row"]) for tid, p in open_tasks.items()}

    def _jitter_estimate(self) -> Optional[float]:
        """How jittery observed peer positions look: median residual of their recent tracks around a straight line."""
        vals = []
        for b in self.peer_beliefs.values():
            hist = b.get("_hist") or []
            if len(hist) < 10:
                continue
            ts = [h[0] for h in hist]
            tm = sum(ts) / len(ts)
            den = sum((t - tm) ** 2 for t in ts) or 1.0
            sq = 0.0
            for axis in (0, 1):
                ys = [h[1][axis] for h in hist]
                ym = sum(ys) / len(ys)
                slope = sum((t - tm) * (y - ym) for t, y in zip(ts, ys)) / den
                sq += sum((y - (ym + slope * (t - tm))) ** 2 for t, y in zip(ts, ys)) / len(ys)
            vals.append((sq / 2.0) ** 0.5)
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    def _measured_context(self, state) -> Dict:
        """
        What the drone measures about its situation at this decision, in scenario-free terms, for the reward analysis
        (.ignore/revision/marl_baseline_design.md section 4). Logged with the decision; no model or rule reads it.
        staleness_mean   how long ago the offered parcels were last confirmed in place (mean, as a share of 2,000 steps)
        unseen_share     share of the offered parcels not confirmed in place now
        own_territory_share  share of the offered parcels whose nearest start cell is this drone's
        peers_in_view_share  share of the peers sensed now
        battery_margin   usable energy as a share of the capacity
        n_tasks, n_peers     the offered tasks and the peers
        """
        snap = state["snapshot"]
        station = snap.get("nearest_station")
        step = float(snap.get("step", 0))
        by_id = {p["id"]: p for p in state.get("visible_parcels", [])}
        tasks = state.get("feasible_tasks", [])
        ages = [max(0.0, step - float(by_id.get(t["parcel_id"], {}).get("seen_step", 0))) for t in tasks]
        owners = self._territory_owners()
        power = getattr(self.drone, "power", None)
        peers = list(self.peer_beliefs)
        in_view = [p for p in peers if self.peer_beliefs[p].get("is_visible")]
        # Beliefs about each peer and each offered parcel, for the belief-error analysis (compared offline with the truth rebuilt from the logs):
        # peers: [id, last sensed position, believed (extrapolated) position, steps since sensed, sensed now]
        # tasks: [parcel id, steps since confirmed in place, my distance, distance of the nearest peer by believed position, by last sensed position]
        me = (snap["agent"]["col"], snap["agent"]["row"])
        peers_log = []
        for pid in sorted(peers):
            b = self.peer_beliefs[pid]
            last = b.get("last_pos") or b.get("predicted_position")
            pred = b.get("predicted_position") or last
            peers_log.append([pid, [int(round(last[0])), int(round(last[1]))], [int(round(pred[0])), int(round(pred[1]))],
                              int(b.get("time_since_last_seen", 0)), bool(b.get("is_visible"))])
        def near(positions, cell):
            return min((self.physics.distance(p_, cell) for p_ in positions), default=-1)
        pred_pos = [tuple(x[2]) for x in peers_log]
        last_pos = [tuple(x[1]) for x in peers_log]
        tasks_log = [[t["parcel_id"], int(a_), self.physics.distance(me, t["pickup"]), near(pred_pos, t["pickup"]), near(last_pos, t["pickup"])]
                     for t, a_ in zip(tasks, ages)]
        pm = None
        if self.peer_model_shadow and self._peer_model is not None and station:
            ps = self._predicted_station(snap)
            if ps is None or ps["pred"] is None:
                pm = {"missing": "speed", "n_intervals": len(self._peer_model._ratios), "occ": ps["mem"] if ps else None,
                      "occ_rate": snap.get("station_count_hat")}
            else:
                pred = ps["model"]
                pm = {"spc": self._peer_model.steps_per_cell(),
                      "peers": [[pid, list(v["pos"]), len(v["taken"]), v["delivered"]] for pid, v in pred["peers"].items()],
                      "gone": sorted({i for v in pred["peers"].values() for i in v["taken"]}),
                      "occ": ps["mem"], "occ_pred": ps["pred"], "occ_rate": snap.get("station_count_hat")}
        assign = None
        if self.peer_model_shadow:
            # what the drone's own reasoning models say each peer takes (parcel id), to be checked against what the peers pick next
            assign = {m: {k: v for k, v in (getattr(self, "_assignments", {}).get(m) or {}).items() if k != self.agent_id} for m in ("hungarian", "cbba", "cbba_rounds")}
        return {
            "pm": pm, "assign": assign, "top3": dict(getattr(self, "_peer_top3", {})), "top3_open": dict(getattr(self, "_peer_top3_open", {})),
            "peers": peers_log, "tasks": tasks_log,
            "staleness_mean": round(min(1.0, (sum(ages) / len(ages)) / 2000.0), 4) if ages else 0.0,
            "unseen_share": round(sum(a > 0 for a in ages) / len(ages), 4) if ages else 0.0,
            "own_territory_share": round(sum(owners.get(t["parcel_id"]) == self.agent_id for t in tasks) / len(tasks), 4) if tasks else 0.0,
            "peers_in_view_share": round(len(in_view) / len(peers), 4) if peers else 1.0,
            "battery_margin": round((float(power.level) - float(power.floor)) / max(1e-9, float(power.capacity)), 4) if power else 0.0,
            "n_tasks": len(tasks), "n_peers": len(peers),
        }

    def _scenario_descriptors(self, state) -> Dict:
        """
        Facts about the situation that this drone can observe or compute for itself: how much of the fleet
        it sees, how jittery what it sees is, the competition for tasks, its own energy, how unequal the
        fleet's consumption rates are, and how well each reasoning model has predicted its peers so far.
        """
        snap = state["snapshot"]
        peers = list(self.peer_beliefs)
        visible = [p for p in peers if self.peer_beliefs[p].get("is_visible")]
        open_n = len(state.get("visible_parcels", []))
        rates = [self._rate(p) for p in peers] + [self._rate(self.agent_id)]
        power = getattr(self.drone, "power", None)
        d = {
            "fleet_size": len(peers) + 1,
            "parcels_total": len(snap.get("all_parcels", [])),
            "open_tasks": open_n,
            "competition": round((len(peers) + 1) / max(1, open_n), 3),
            "peers_visible_fraction": round(len(visible) / max(1, len(peers)), 3),
            "sensing_radius": None if self.sensing_radius == float("inf") else self.sensing_radius,
            "position_jitter": None,
            "sensor_noise": self.sensing_noise_std,
            "battery_fraction": round(power.level / power.capacity, 3) if power is not None else None,
            "rate_spread": round(max(rates) / max(1e-9, min(rates)), 3),
            "model_hit_rate": {m: (round((rec[0] + 1.0) / (rec[1] + 2.0), 3), round(rec[1], 1))
                               for m, rec in self._track.items()},
            "self_check": {"energy_bias": self._self_scale(), "energy_samples": 0 if self.self_model is None else self.self_model.samples,
                           "own_model": None if self.self_model is None else self.self_model.snapshot(),
                           "plan_success": round((self._plan_hits[0] + 1.0) / (self._plan_hits[1] + 2.0), 3),
                           "plans": round(self._plan_hits[1], 1),
                           "plan_brier": None if not self._brier[1] else round(self._brier[0] / self._brier[1], 4)},
        }
        j = self._jitter_estimate()
        d["position_jitter"] = None if j is None else round(j, 3)
        d["self_check"].update(self._own_energy_facts(snap, power))
        return d

    def _own_energy_facts(self, snap, power) -> Dict:
        """
        What the learned own-energy model says, as facts an arbiter can use: how costly the drone is per cell and
        per unit of weight relative to nominal, how sure it is, whether its consumption is changing, and how many
        tasks of its mean open-parcel distance its battery still affords (affordable_tasks) against the tasks open (affordable_vs_open).
        All None until the model has data.
        """
        m = self.self_model
        if m is None or m.samples < 2:
            return {"cost_vs_nominal": None, "weight_cost_vs_nominal": None, "cost_uncertainty": None,
                    "trend": None, "shift_alarms": 0, "affordable_tasks": None, "affordable_vs_open": None}
        nominal_cell = m.nominal[0]
        nominal_weight = m.nominal[1]
        w = self.baseline_payload
        # A typical leg is the drone's mean distance to the parcels that are still open (nothing assumed about the map).
        me = (snap["agent"]["col"], snap["agent"]["row"])
        legs = [self.physics.distance(me, (p["col"], p["row"])) for p in snap.get("all_parcels", [])
                if not p.get("picked") and not p.get("delivered")]
        typical_leg = max(1.0, sum(legs) / len(legs)) if legs else 1.0
        per_task = max(1e-9, m.cost(typical_leg, w) + m.cost(typical_leg, 0.0))
        left = max(0.0, float(power.level)) if power is not None else 0.0
        affordable = left / per_task
        open_n = max(1, len([p for p in snap.get("all_parcels", []) if not p.get("picked") and not p.get("delivered")]))
        tr = m.trend()
        return {
            "cost_vs_nominal": round(m.theta[0] / max(1e-9, nominal_cell), 3),
            "weight_cost_vs_nominal": round(m.theta[1] / max(1e-9, nominal_weight), 3),
            "cost_uncertainty": round(m.std(typical_leg, w) / max(1e-9, m.cost(typical_leg, w)), 4),
            "trend": round(max(abs(tr["per_cell"]), abs(tr["per_weight"])), 4),
            "shift_alarms": m.alarms,
            "affordable_tasks": round(affordable, 2),
            "affordable_vs_open": round(affordable / open_n, 3),
        }

    @staticmethod
    def _smooth_track(hist):
        """
        Least-squares line through the observed positions of a peer: returns the smoothed start and
        end of the track and the number of samples. Used when observed positions are noisy.
        """
        n = len(hist)
        ts = [h[0] for h in hist]
        t0 = ts[0]
        out = []
        for axis in (0, 1):
            ys = [h[1][axis] for h in hist]
            tm, ym = sum(ts) / n, sum(ys) / n
            den = sum((t - tm) ** 2 for t in ts)
            slope = sum((t - tm) * (y - ym) for t, y in zip(ts, ys)) / den if den > 0 else 0.0
            out.append((ym + slope * (t0 - tm), ym + slope * (ts[-1] - tm)))
        return (out[0][0], out[1][0]), (out[0][1], out[1][1])

    def _observe_intents(self, visible, open_tasks, step) -> None:
        """A peer that is moving is assumed to be heading for the open task it is closing in on."""
        noisy = self.sensing_noise_std > 0.0
        for a in visible:
            b = self.peer_beliefs.get(a["id"])
            if not b or a.get("carrying"):
                continue
            if self.keep_schedule and (b.get("assumed") or {}).get("gone_at") is not None:
                continue
            hist = b.get("_hist") or []
            if noisy:
                # Observed positions jitter: read intent from the smoothed track over the whole
                # window, and count a peer as moving only if the track covers a clear distance.
                if len(hist) < 6:
                    continue
                old_hat, now_hat = self._smooth_track(hist)
                b["smooth_pos"] = now_hat
                moved = abs(now_hat[0] - old_hat[0]) + abs(now_hat[1] - old_hat[1])
                recent = ([(hist[0][0], old_hat), (hist[-1][0], now_hat)]
                          if moved >= max(2.0, 3.0 * self.sensing_noise_std) else [(0, old_hat)])
            else:
                recent = [h for h in hist if h[0] >= step - self.INTENT_WINDOW_STEPS] or hist[-2:]
            if len(recent) < 2 or recent[0][1] == recent[-1][1]:
                # Not seen moving. A computed assumption stays; an observed intent goes stale.
                cur = b.get("assumed")
                if cur and cur.get("source") == "observed" and step - cur["seen_moving"] > self.INTENT_FRESH_STEPS:
                    b["assumed"] = None
                continue
            old, now = recent[0][1], recent[-1][1]
            best = None
            # Equally close targets are ranked by pickup cell: parcel ids are random per run.
            for tid in sorted(open_tasks, key=lambda i: (open_tasks[i]["col"], open_tasks[i]["row"])):
                t = open_tasks[tid]
                pk = (t["col"], t["row"])
                d_now, d_old = self.physics.distance(now, pk), self.physics.distance(old, pk)
                # With noisy positions a task counts only if the peer has clearly closed on it.
                if d_old - d_now > (1.5 if noisy else 0.0) and (best is None or d_now < best[0]):
                    best = (d_now, tid)
            if best:
                b["assumed"] = {"task": best[1], "since": step, "seen": step, "away": 0,
                                "last_dist": best[0], "source": "observed", "seen_moving": step}
            else:
                cur = b.get("assumed")
                if cur and cur.get("source") == "observed":
                    b["assumed"] = None  # moving, but toward none of the open tasks

    def _contested_targets(self, snap, open_tasks) -> set:
        """The tasks a visible peer is seen closing on from nearer than this drone is: the same test that makes the drone give up a target on its way."""
        me = (snap["agent"]["col"], snap["agent"]["row"])
        step_now = snap.get("step", self.current_step)
        out = set()
        for pid, b in self.peer_beliefs.items():
            a = b.get("assumed")
            if (not a or a.get("source") != "observed" or not b.get("is_visible") or a["task"] not in open_tasks
                    or step_now - a.get("seen_moving", -10 ** 9) > self.INTENT_FRESH_STEPS):
                continue
            t = open_tasks[a["task"]]
            target = (t["col"], t["row"])
            if self.physics.distance(b.get("smooth_pos") or b["predicted_position"], target) + self.CONTEST_MARGIN_CELLS <= self.physics.distance(me, target):
                out.add(a["task"])
        return out

    def _own_plan_signal(self, agent, open_tasks, own_target, step_now=0) -> Dict:
        """Should the drone give up the pickup it is flying to?"""
        if own_target is None:
            return {}
        target = tuple(own_target)
        tid = next((i for i, t in open_tasks.items() if (t["col"], t["row"]) == target), None)
        me = (agent["col"], agent["row"])
        d_me = self.physics.distance(me, target)
        if d_me == 0:
            return {}
        if tid is None:
            return {"replan": "target_gone"} if self.observe_gone else {}
        if not self.observe_contested:
            return {}
        if self.consensus and self._compete_cell is not None and tuple(self._compete_cell) == target:
            return {}                                    # the drone chose this knowing a peer is expected to take it: no peer can surprise it
        for pid, b in self.peer_beliefs.items():
            a = b.get("assumed")
            if (not a or a["task"] != tid or a.get("source") != "observed" or not b.get("is_visible")
                    or step_now - a.get("seen_moving", -10 ** 9) > self.INTENT_FRESH_STEPS):
                continue
            d_p = self.physics.distance(b.get("smooth_pos") or b["predicted_position"], target)
            if d_p + self.CONTEST_MARGIN_CELLS <= d_me:
                return {"replan": "contested"}
        return {}

    @staticmethod
    def _proposal_summary(state) -> Dict:
        """What each reasoning model proposed for this drone, and whether they agree."""
        def pid(key):
            t = state.get(key)
            return t["parcel_id"] if t else None

        def pid_of(choices, name):
            t = (choices or {}).get(name)
            return t["parcel_id"] if t else None
        proposals = {"greedy": pid("greedy_choice"), "hungarian": pid("hungarian_choice"),
                     "cbba": pid("cbba_choice")}
        named = [v for v in proposals.values() if v is not None]
        proposals["distinct"] = len(set(named))
        proposals["unanimous"] = len(named) == 3 and len(set(named)) == 1
        proposals["cbba_no_slot"] = proposals["cbba"] is None and proposals["hungarian"] is not None
        proposals["cbba_info"] = state.get("cbba_info")
        proposals["assumed"] = state.get("assumed_assignment")
        proposals["scenario"] = state.get("scenario")
        proposals["measured"] = state.get("measured_context")
        proposals["arbiter_context"] = state.get("arbiter_context")
        proposals["cbba_rounds"] = pid_of(state.get("model_choices"), "cbba_rounds")
        return proposals

    def _no_slot(self, state, greedy_choice, agent_id, convention):
        """The convention assigned this drone no task."""
        if self.no_slot_policy == "wait":
            # Act on the decision: no slot means no task. Wait; do not chase a parcel another
            # drone was assigned.
            state["selected_task"] = None
            state["decision_mode"] = "wait"
            state["wait_reason"] = f"{convention} assigned this drone no task"
            state["slm_status"] = "abstain"  # decide_mode turns an abstain without a task into a wait
            state["slm_policy"] = "no_slot_wait"
            state["slm_gate"] = f"arbiter: {convention} no slot"
            return state
        backup = state.get("hungarian_choice") if convention == "cbba" else None
        if self.no_slot_policy == "hungarian_greedy" and backup:
            # Ordered fallback: the primary model gave no task, so use the next model's task.
            print(f"[SPPDEC FALLBACK] Agent: {agent_id}: {convention} gave no assignment; hungarian fallback")
            state["selected_task"] = backup
            state["decision_mode"] = "candidate"
            state["slm_status"] = "abstain"
            state["slm_policy"] = "hungarian_fallback"
            return state
        print(f"[SPPDEC FALLBACK] Agent: {agent_id}: {convention} gave no assignment; greedy fallback")
        state["selected_task"] = greedy_choice
        state["decision_mode"] = "candidate"
        state["slm_status"] = "abstain"
        state["slm_policy"] = "greedy_fallback"
        return state

    def _expected_drops(self, tasks, snap: Dict) -> Dict[str, tuple]:
        """
        The drop cell the drone expects for each task: the station cells it remembers free (minus the cells it predicts filled), matched one-to-one to the
        tasks by the shortest total pickup-to-cell distance, so a cell assumed taken by one task is not assumed free for another. Deterministic for tasks
        sorted by pickup cell. With more tasks than free cells the surplus tasks take their nearest cell.
        """
        station = snap.get("nearest_station")
        if not station or not tasks:
            return {}
        free = sorted(set(self._predicted_free_cells(snap) or [(c, r) for r in range(station["row"], station["row"] + station["h"])
                                                               for c in range(station["col"], station["col"] + station["w"])]))
        if not free:
            return {}
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = np.array([[self.physics.distance(t["pickup"], c) for c in free] for t in tasks], dtype=float)
        rows, cols = linear_sum_assignment(cost)
        plan = {tasks[i]["id"]: free[j] for i, j in zip(rows, cols)}
        for t in tasks:
            plan.setdefault(t["id"], min(free, key=lambda c: (self.physics.distance(t["pickup"], c), c)))
        return plan

    def _believes_all_picked(self, snap: Dict) -> bool:
        """
        The drone's soft belief that every parcel has been picked up: the cells it remembers occupied, the parcel it carries, and the trips it has booked for its peers
        since it last saw the whole station, against the number of parcels in the start information. A belief, not a fact: the cells it has not looked at since may
        not be occupied, and a booked trip may not have happened. Only the sight of the station settles it.
        """
        station, info = snap.get("nearest_station"), self._start_info
        if not station or info is None:
            return False
        occupied = station["w"] * station["h"] - len(station.get("free_cells") or [])
        carrying = 1 if snap.get("carrying", {}).get("has") else 0
        return occupied + carrying + (self._booked_trips - self._booked_at_view) >= info.n_orders

    def _travel_cost(self):
        """The energy-equivalent of the time a flight of `cells` takes (the simulator's speed times the arm's time weight), or None when bids are energy alone."""
        w = self.time_weight
        if w <= 0.0 and self.time_weight_hungarian > 0.0 and self._models() == ("hungarian",):
            w = self.time_weight_hungarian
        return (lambda cells: w * self.physics.travel_steps(cells)) if w > 0.0 else None

    def _note_own_cell(self, snap: Dict) -> None:
        """The steps at which the drone's own cell changed: its speed, and with it the fleet's, is read from these."""
        a = snap["agent"]
        cell = (a["col"], a["row"])
        if not self._own_cells or self._own_cells[-1][1] != cell:
            self._own_cells.append((snap.get("step", self.current_step), cell))
            del self._own_cells[:-60]

    def _steps_per_cell(self) -> float:
        """Steps the drone takes per cell when it moves (lower quartile of the gaps between its cell changes; waiting only lengthens a gap)."""
        gaps = sorted(b[0] - a[0] for a, b in zip(self._own_cells, self._own_cells[1:]) if 0 < b[0] - a[0] < 60)
        return gaps[max(0, (len(gaps) - 1) // 4)] if len(gaps) >= 5 else 8.0

    def _without_predicted_gone(self, parcels, snap=None):
        """The believed-open parcels minus those predicted gone. A parcel sensed in place after the prediction is back; if only predicted-gone
        parcels are left the drone goes to look instead of idling."""
        keep = []
        for p in parcels:
            marked = self._predicted_gone.get(p["id"])
            if marked is not None and p.get("seen_step", 0) > marked:
                del self._predicted_gone[p["id"]]
                marked = None
            if marked is None:
                keep.append(p)
        if not keep and parcels:
            if self.count_first and snap is not None and self._nothing_left_by_count(snap) is not False:
                # everything the drone believes open is predicted gone: it does not fly out to check. At the station the count is exact; elsewhere it goes there to count (the look directive)
                return []
            self._predicted_gone.clear()
            return list(parcels)
        return keep

    def _nothing_left_by_count(self, snap):
        """True if the exact station count says no parcel can be on the ground (all parcels - delivered - carriers in view <= 0); False if it says some can; None when the drone cannot count (the whole station is not in view)."""
        station, info = snap.get("nearest_station"), self._start_info
        if not station or info is None or not snap.get("station_in_full_view"):
            return None
        occupied = station["w"] * station["h"] - len(station.get("free_cells") or [])
        return info.n_orders - occupied - len(self._carrying_peers) <= 0

    def note_pickup(self, snap: Dict) -> None:
        """The drone has just picked a parcel up: how long its leg to the pickup took, per cell (its own timing, used for what it expects of its peers)."""
        if not self.consensus:
            return
        a = snap["agent"]
        step, cell = snap.get("step", self.current_step), (a["col"], a["row"])
        if self._leg_start is not None:
            cells = self.physics.distance(self._leg_start[1], cell)
            if cells >= 3:
                self._pick_ratios.append((step - self._leg_start[0]) / cells)
                del self._pick_ratios[:-20]
        self._loaded_start = (step, cell)

    def note_drop(self, snap: Dict) -> None:
        """The drone has just delivered: how long its loaded leg took, per cell; the next leg to a pickup begins here."""
        if not self.consensus:
            return
        a = snap["agent"]
        step, cell = snap.get("step", self.current_step), (a["col"], a["row"])
        if self._loaded_start is not None:
            cells = self.physics.distance(self._loaded_start[1], cell)
            if cells >= 3:
                self._loaded_ratios.append((step - self._loaded_start[0]) / cells)
                del self._loaded_ratios[:-20]
        self._leg_start = (step, cell)

    def _leg_ratios(self):
        """
        Steps per cell the drone expects a peer to need to reach a pickup and to fly loaded. The prior is 1.8 times the drone's steady speed (a leg is
        longer than the straight-line distance, and waiting and handling add to it; 1.8 is calibrated on the timed legs of seeds 0-7, which came out near 14.5
        steps per cell against a steady 8). Each leg the drone has timed itself (the last 20) counts as much as the prior, so the estimate moves to the
        drone's own timing from its first trip.
        """
        prior = 1.8 * self._steps_per_cell()
        est = lambda xs: (sum(xs[-20:]) + prior) / (len(xs[-20:]) + 1)
        return est(self._pick_ratios), est(self._loaded_ratios)

    def _accrue_unscheduled_energy(self, step, open_tasks) -> None:
        """
        For the time a peer has no schedule (and is not seen carrying), its energy used grows as the drone's own does per step, scaled by the peer's declared
        consumption relative to the drone's own: unless the drone's schedule says otherwise, the peer has been working as hard as the drone. A live schedule
        accounts for its own time (its trip is booked at the end), so the accrual waits. Only while parcels are believed open; never beyond the capacity.
        """
        last, self._energy_clock = self._energy_clock, step
        if last is None or step <= last or not open_tasks:
            return
        rate = self._own_energy_used() / max(1, step)
        if rate <= 0.0:
            return
        mine = max(self._declared_rate(self.agent_id), 1e-6)
        for pid, b in self.peer_beliefs.items():
            a = b.get("assumed")
            if (a and a.get("gone_at") is not None) or pid in self._carrying_peers:
                continue                                  # a live schedule accounts for the time; an observed heading does not book anything
            spec = self._spec(pid)
            cap = spec.capacity if spec is not None else float("inf")
            used = min(cap, b.get("booked_energy", 0.0) + rate * (step - last) * self._declared_rate(pid) / mine)
            b["booked_energy"] = b["energy_used"] = b["predicted_energy"] = used

    def _book_trip(self, pid, belief, a) -> None:
        """A peer's trip the drone assumed is booked as the peer's energy used: empty leg, loaded leg at the declared weight, pick and drop; the peer is
        then expected at the drop."""
        w = a.get("weight", 1.0)
        e = (self._energy(pid, self.physics.distance(a["start_pos"], a["pickup"]), 0.0)
             + self._energy(pid, self.physics.distance(a["pickup"], a["drop"]), w) + 2.0 * self.physics.pick_drop_cost(w))
        self._booked_trips += 1
        belief["booked_energy"] = belief.get("booked_energy", 0.0) + e
        belief["energy_used"] = belief["predicted_energy"] = belief["booked_energy"]
        if not belief.get("is_visible"):
            belief["predicted_position"] = a["drop"]

    def _consensus_decide(self, state):
        """
        The drone acts on what its two reasoning models agree on. Each model gives every believed agent a bundle (same expected bids, same order).
          the tasks in both bundles for this drone   it takes them, in CBBA's order;
          the tasks in both bundles for a peer       they are the peer's: the drone leaves them;
          otherwise                                   it goes for what was not agreed, parcels the models disagree about last, the lowest bid first;
          nothing but the peers' agreed tasks left    it competes for the lowest bid rather than idle.
        """
        agent_id = self.agent_id
        feasible = state.get("feasible_tasks", [])
        by_id = {t["parcel_id"]: t for t in feasible}
        H = self._bundles.get("hungarian") or {}
        C = self._bundles.get("cbba_rounds") or {}
        owners = [{t: a for a, b in (self._bundles.get(m) or {}).items() for t in b} for m in self._models()]
        contested = (self._contested_targets(state["snapshot"], {p["id"]: p for p in state.get("visible_parcels", [])}) if self.observe_contested else set())
        mine = [t for t in self._agreed_for(agent_id) if t in by_id and t not in contested]
        every_task = set().union(*[set(o) for o in owners]) if owners else set()
        peers_agreed = {t for t in every_task if owners[0].get(t) not in (None, agent_id) and all(o.get(t) == owners[0].get(t) for o in owners)}
        peers_agreed |= {t for t in contested if t in by_id}          # a nearer peer seen closing on it: it is the peer's
        disputed = {t for t in every_task if len({o.get(t) for o in owners}) > 1}
        no_win = False
        if mine:
            chosen, label = mine, "consensus: both models give this drone %d task(s)" % len(mine)
        elif self.no_gamble and state["snapshot"].get("nearest_station"):
            candidates, no_win = list(by_id), True
            label = "consensus: only the peers' agreed tasks left; none is this drone's, goes to the station"
            bids = self._own_bids(state, feasible)
            chosen = [min(candidates, key=lambda t: (bids.get(t, float("inf")), by_id[t]["pickup"]))]
        else:
            pool = [t for t in by_id if t not in peers_agreed]
            free = [t for t in pool if t not in disputed]
            contested = [t for t in pool if t in disputed]
            if free:
                candidates, label = free, "consensus: nothing agreed for this drone; takes a task not agreed for anyone"
            elif contested:
                candidates, label = contested, "consensus: only disputed tasks left; takes one"
            else:
                candidates, label = list(by_id), "consensus: only the peers' agreed tasks left; competes rather than idle"
            bids = self._own_bids(state, feasible)
            chosen = [min(candidates, key=lambda t: (bids.get(t, float("inf")), by_id[t]["pickup"]))]
        first = by_id[chosen[0]]
        snap_l = state["snapshot"]
        if (label.startswith("consensus: only the peers' agreed tasks left") and snap_l.get("nearest_station")
                and (no_win or (not snap_l.get("station_in_full_view") and self._believes_all_picked(snap_l)))):
            # Everything left is a peer's and either the drone believes every parcel is already picked up (a flight to a parcel would almost surely find it gone) or it
            # has no task of its own (no gambling). It goes to the station; from the centre every cell is in view, so the memory becomes exact and the drone
            # decides again on what it saw.
            state["selected_task"] = None
            state["consensus_bundle"] = []
            state["decision_mode"] = "look"
            state["slm_status"] = "success"
            state["slm_policy"] = "look"
            state["slm_gate"] = label if no_win else "consensus: believes every parcel is picked up; looks at the station before chasing a peer's parcel"
            if self.flight_recorder is not None and not self._looking:
                self.flight_recorder.record_slm_decision(
                    agent_id=agent_id, step=snap_l.get("step", 0), sim_time=self.current_step * (1.0 / 60.0), policy="look", confidence=1.0,
                    belief_infeasibility=state.get("belief_infeasibility", 0.0), energy_error=0.0, competition_ratio=0.0,
                    belief_trend="STABLE", error_trend="STABLE", competition_trend="STABLE", greedy_cost=first["cost"], hungarian_cost=None,
                    gate=state["slm_gate"], arbiter="consensus", proposals=self._proposal_summary(state), status="success")
            self._looking = True
            self._compete_cell = None
            return state
        self._looking = False
        self._compete_cell = first["pickup"] if label.startswith("consensus: only the peers' agreed tasks left") else None
        snap0 = state["snapshot"]
        self._leg_start = (snap0.get("step", self.current_step), (snap0["agent"]["col"], snap0["agent"]["row"]))
        state["selected_task"] = first
        state["consensus_bundle"] = [] if self.rolling else [by_id[t] for t in chosen[1:]]
        state["decision_mode"] = "candidate"
        state["slm_status"] = "success"
        state["slm_policy"] = "consensus"
        state["slm_confidence"] = 1.0
        state["slm_gate"] = label
        state["consensus"] = {"mine": mine, "peers_agreed": sorted(peers_agreed), "disputed": sorted(disputed), "chosen": chosen}
        if self.flight_recorder is not None:
            snap = state["snapshot"]
            proposals = self._proposal_summary(state)
            # log only: the whole table of bundles this drone computed for every drone (read by analysis/revision/assignment_agreement.py)
            proposals["tables"] = {m: {aid: list(b) for aid, b in self._bundles.get(m, {}).items()} for m in self._models()}
            # log only: what this drone believes each peer has used so far (read by analysis/revision/belief_drift.py)
            proposals["peer_energy_belief"] = {pid: round(float(b.get("energy_used", 0.0)), 3) for pid, b in self.peer_beliefs.items()}
            proposals["consensus"] = state["consensus"]
            proposals["beliefs"] = {
                "bundles": {"hungarian": H, "cbba_rounds": C},
                "gone": dict(self._predicted_gone),
                "gone_by": dict(self._gone_by),
                "leg_ratios": [round(float(r), 2) for r in self._leg_ratios()],
                "peer_energy": {pid: round(float(b.get("predicted_energy", 0.0)), 2) for pid, b in self.peer_beliefs.items()},
                "peer_pos": {pid: [round(float(b["predicted_position"][0]), 1), round(float(b["predicted_position"][1]), 1)] for pid, b in self.peer_beliefs.items()},
            }
            self.flight_recorder.record_slm_decision(
                agent_id=agent_id, step=snap.get("step", 0), sim_time=self.current_step * (1.0 / 60.0), policy="consensus", confidence=1.0,
                belief_infeasibility=state.get("belief_infeasibility", 0.0), energy_error=0.0, competition_ratio=0.0,
                belief_trend="STABLE", error_trend="STABLE", competition_trend="STABLE",
                greedy_cost=first["cost"], hungarian_cost=None, gate=label, arbiter="consensus", proposals=proposals, status="success")
        return state

    def _own_bids(self, state, feasible) -> Dict[str, float]:
        """This drone's own expected bid for each task on the shared matrix."""
        agents, tasks, drop = self._bid_inputs(state, feasible)
        if agents is None:
            return {}
        me = next((a for a in agents if a["id"] == self.agent_id), None)
        if me is None:
            return {}
        margin = float(self.drone.power.floor)
        return {t["id"]: bid_cost(me, t, drop, lambda a, b: self.physics.distance(a, b), lambda d, w: self.physics.energy_cost(distance=d, weight=w),
                                  lambda w: self.physics.pick_drop_cost(w), margin, travel_cost=self._travel_cost()) for t in tasks}

    def _matrix_agents(self):
        """Agents that take part in the assignment: every believed peer plus self."""
        agents = list(self.peer_beliefs.keys()) + [self.agent_id]
        agents = [a for a in agents if a == self.agent_id or a not in getattr(self, "_busy_peers", ())]
        if self.exclude_carrying_peers:
            agents = [a for a in agents if a == self.agent_id or a not in self._carrying_peers]
        return agents

    def _own_energy_used(self) -> float:
        power = self.drone.power
        return max(0.0, power.capacity - power.level)

    def _caps(self, agents, tasks, drop):
        """Per-agent bundle caps from the team's capability shares (only with `share_bundles`), else None: every agent has the common bundle size."""
        if not self.share_bundles or not tasks:
            return None
        return capability_caps(agents, tasks, drop, lambda a, b: self.physics.distance(a, b), lambda d, w: self.physics.energy_cost(distance=d, weight=w),
                               lambda w: self.physics.pick_drop_cost(w), float(self.drone.power.floor))

    def _bundle_size(self, n_tasks: int, n_agents: int) -> int:
        """Bundle length for the CBBA rounds. Fixed when configured as a number; with "auto" it is the
        fair share of the open tasks, ceil(tasks / agents), so that tasks run out evenly across the
        fleet: 1 when drones outnumber tasks, more as tasks pile up per drone. No other limit: the
        share is already bounded by the problem (it cannot exceed the number of open tasks)."""
        if self.cbba_bundle_size != "auto":
            return int(self.cbba_bundle_size)
        return max(1, -(-int(n_tasks) // max(1, int(n_agents))))

    def _bid_inputs(self, state, feasible):
        """
        The one set of expected bids both reasoning models work from: every believed agent (itself included) with its assumed state, and every parcel the
        drone believes open. Rows are sorted by agent id and columns by pickup cell, the same for every drone (parcel ids are random per run), so two drones holding
        the same beliefs build the same matrix. The drone itself may bid only on `feasible`; the peers on every believed-open parcel.
        """
        snap = state["snapshot"]
        station = snap.get("nearest_station")
        if not station or not feasible:
            return None, None, None
        drop = (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
        tasks = [{"id": p["id"], "pickup": (p["col"], p["row"]), "weight": p.get("weight", 1.0)}
                 for p in state.get("visible_parcels", []) if p["id"] not in self._committed_tasks]
        if self.consensus and self.freshness:
            q = self._parcel_presence(snap, state.get("visible_parcels", []))
            for t in tasks:
                t["presence"] = q.get(t["id"], 1.0)
        if not self.consensus:
            for t in tasks:
                t["drop"] = self._drop_for(t["pickup"], snap)           # priced to the drop cell the drone expects, from its station memory and its predictions
        ps = self._predicted_station(snap) if self.peer_model_cost else None
        pm_peers = ps["model"]["peers"] if ps and ps.get("model") else {}
        agents = []
        for aid in self._matrix_agents():
            if aid == self.agent_id:
                agents.append({
                    "id": aid, "pos": self._self_pos, "rate": self._rate(aid), "speed": self._speed_of(aid),
                    "completion": self.completion_bid, "spc": self.physics.steps_per_cell, "free_in": 0.0,
                    "penalty": self.energy_penalty_alpha * self._own_energy_used(),
                    "remaining": float(self.drone.power.level),
                    "eligible": {t["parcel_id"] for t in feasible},
                })
                continue
            b = self.peer_beliefs[aid]
            px, py = b["predicted_position"]
            pos, busy, beta = (int(round(px)), int(round(py))), 0.0, self.energy_penalty_alpha
            pm = pm_peers.get(aid)
            if pm:                                  # the peer model: predicted position, the rest of a carrying peer's flight, the peer's own beta and w-bar
                prof = self._peer_profile(aid)
                pos, beta = pm["pos"], prof["energy_penalty_alpha"]
                if pm.get("carrying") and pm.get("dest") is not None:
                    busy = self.physics.energy_cost(distance=self.physics.distance(pos, pm["dest"]), weight=prof["baseline_payload"])
                    pos = pm["dest"]
            spec = self._spec(aid)
            cap = spec.capacity if spec is not None else None
            free_in = 0.0
            if self.completion_bid:
                pos, free_in = self._free_when(aid, b, pos, snap)
            agents.append({
                "id": aid, "pos": pos, "rate": self._rate(aid), "busy": busy, "speed": self._speed_of(aid),
                "completion": self.completion_bid, "spc": self.physics.steps_per_cell, "free_in": free_in,
                "penalty": beta * b.get("predicted_energy", 0.0),
                "remaining": (cap - b.get("predicted_energy", 0.0)) if cap is not None else None,
                "eligible": None,
            })
        agents.sort(key=lambda a: str(a["id"]))
        tasks.sort(key=lambda t: (t["pickup"][0], t["pickup"][1]))
        if self.consensus and self.completion_bid and self.equal_bid and tasks:
            self._equal_refs(agents, tasks, drop, snap)
        if self.consensus:
            # every task is priced to its own expected drop cell, a cell remembered free that no other task of the round is expected to fill
            self._drop_plan = self._expected_drops(tasks, snap)
            for t in tasks:
                t["drop"] = self._drop_plan.get(t["id"], drop)
        return agents, tasks, drop

    def _solve_cbba_bid_rounds(self, state, feasible, key: str = "cbba"):
        """
        CBBA with its messages replaced by assumptions (see cbba_assumed_bids.py). Returns this drone's
        first task in its resulting bundle, or None if the protocol gives it none.
        """
        agents, tasks, drop = self._bid_inputs(state, feasible)
        if agents is None:
            return None
        result = cbba_with_assumed_bids(
            agents, tasks, drop,
            dist=lambda a, b: self.physics.distance(a, b),
            energy=lambda d, w: self.physics.energy_cost(distance=d, weight=w),
            bundle_size=self._bundle_size(len(tasks), len(agents)), feasibility_margin=float(self.drone.power.floor),
            pick_drop=lambda w: self.physics.pick_drop_cost(w), travel_cost=self._travel_cost(), caps=self._caps(agents, tasks, drop),
        )
        state["cbba_info"] = {"rounds": result["rounds"], "converged": result["converged"],
                              "settled": result["settled"],
                              "bundle_len": len(result["bundles"].get(self.agent_id, []))}
        if not hasattr(self, "_assignments"):
            self._assignments = {}
        self._assignments[key] = {aid: b[0] for aid, b in result["bundles"].items() if b}
        self._bundles[key] = {aid: list(b) for aid, b in result["bundles"].items()}
        mine = result["bundles"].get(self.agent_id, [])
        if not mine:
            return None
        return next((t for t in feasible if t["parcel_id"] == mine[0]), None)

    def _solve_hungarian(self, tasks, peer_map, state=None):
        """
        Hungarian assignment of the believed fleet to the believed-open parcels, on the same expected bids CBBA works from (`_bid_inputs`, `bid_cost`):
        the whole trip of each agent for each parcel, rows sorted by agent id and columns by pickup cell, so two drones holding the same beliefs
        solve the same matrix and reach the same assignment. Every agent's row is solved together (min-sum); the result for every agent is kept as what
        the drone expects each of them to take. Returns this drone's task among `tasks`, or None if the assignment gives it none.
        """
        if not tasks:
            return None
        agents, cols, drop = self._bid_inputs(state, tasks)
        if agents is None:
            return None
        margin = float(self.drone.power.floor)
        dist = lambda a, b: self.physics.distance(a, b)
        energy = lambda d, w: self.physics.energy_cost(distance=d, weight=w)
        pick_drop = lambda w: self.physics.pick_drop_cost(w)
        big = 1e9
        cost_matrix = []
        for a in agents:
            row = []
            for t in cols:
                c = bid_cost(a, t, drop, dist, energy, pick_drop, margin, travel_cost=self._travel_cost())
                row.append(big if c == float("inf") else c)
            cost_matrix.append(row)
        try:
            assignment = self._hungarian_algorithm(cost_matrix)
        except Exception as e:
            logger.warning(f"Hungarian algorithm computation failed: {e}")
            return None
        if not hasattr(self, "_assignments"):
            self._assignments = {}
        self._assignments["hungarian"] = {
            agents[i]["id"]: cols[j]["id"]
            for i, j in enumerate(assignment) if j is not None and j < len(cols) and cost_matrix[i][j] < big
        }
        if self.consensus:
            self._bundles["hungarian"] = hungarian_bundles(
                agents, cols, drop, dist, energy, bundle_size=self._bundle_size(len(cols), len(agents)),
                feasibility_margin=margin, pick_drop=pick_drop, travel_cost=self._travel_cost(), caps=self._caps(agents, cols, drop))["bundles"]
            if "greedy" in self.consensus_models:
                self._bundles["greedy"] = hungarian_bundles(
                    agents, cols, drop, dist, energy, bundle_size=self._bundle_size(len(cols), len(agents)),
                    feasibility_margin=margin, pick_drop=pick_drop, travel_cost=self._travel_cost(), matcher="greedy")["bundles"]
        mine = self._assignments["hungarian"].get(self.agent_id)
        if mine is None:
            return None
        return next((t for t in tasks if t["parcel_id"] == mine), None)

    def _hungarian_algorithm(self, cost_matrix):
        """
        Implementation of the Hungarian algorithm for assignment problem.
        Returns list of assigned task indices for each agent (None if unassigned).
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        costs = np.array(cost_matrix, dtype=float)
        n_agents, n_tasks = costs.shape

        # When agents exceed tasks, pad with dummy columns so surplus
        # agents get assigned there and are correctly returned as None
        if n_agents > n_tasks:
            dummy_cost = 0.0  # surplus agents assigned to dummies cheaply
            padding = np.full((n_agents, n_agents - n_tasks), dummy_cost)
            costs = np.hstack([costs, padding])

        max_finite = np.max(costs[np.isfinite(costs)]) if np.any(np.isfinite(costs)) else 1000
        costs[np.isinf(costs)] = max_finite * 10

        row_indices, col_indices = linear_sum_assignment(costs)

        # None for agents assigned to dummy columns
        assignment = [None] * n_agents
        for i, j in zip(row_indices, col_indices):
            if j < n_tasks:
                assignment[i] = j

        return assignment

    def _update_peers(self, state, snap: Dict, agent: Dict, agent_id: str) -> List[Dict]:
        """
        Peer observation and belief update (steps 3 and 4 of perceive): which peers are within the
        sensing radius, hybrid belief update for each, trajectory extrapolation for the rest.
        Shared by perceive (at a decision) and observe (every step).
        """
        # ==================================================
        # 3. Peer visibility (radius-gated)
        # ==================================================
        # Only peers within self.visibility_radius are directly observed.
        # Peers beyond the radius are handled via belief prediction in Step 4.
        agent_pos = (agent["col"], agent["row"])
        visible_agents = []
        for a in snap.get("other_agents", []):
            if a.get("id") == agent_id:
                continue
            peer_pos = (a["col"], a["row"])
            dist = self.physics.distance(agent_pos, peer_pos)
            if dist <= self.visibility_radius:
                visible_agents.append(a)

        state["visible_agents"] = visible_agents

        # Track for Hungarian assignment
        self.last_visible_agents = [a["id"] for a in visible_agents]

        print(
            "[SPPDEC][%s] perceive: %d peer agents visible",
            agent_id,
            len(state["visible_agents"]),
        )

        # ==================================================
        # 4. Hybrid peer belief updates
        # ==================================================
        # Advance sim clock
        self.current_step = snap.get("step", self.current_step)

        # Build set of visible peer ids this cycle
        visible_peer_ids = {a["id"] for a in visible_agents}

        # --- Case 1 & 3: visible peers (first seen or reappearing) ---
        for peer in visible_agents:
            peer_id = peer["id"]
            peer_pos = (peer["col"], peer["row"])

            if peer_id not in self.peer_beliefs:
                # Case 1: first time seeing this peer — initialise full belief struct
                self.peer_beliefs[peer_id] = {
                    "last_pos": peer_pos,
                    "last_seen_time": self.current_step,
                    "energy_used": 0.0,
                    "predicted_position": peer_pos,
                    "predicted_energy": 0.0,
                    "assigned_task": None,
                    "confidence": 1.0,
                    "is_visible": True,
                    "time_since_last_seen": 0,
                    "last_velocity": (0.0, 0.0),  # (dx, dy) per step — zero until second observation
                }
            else:
                belief = self.peer_beliefs[peer_id]

                # Case 3: reappearance — log correction error before snapping
                if not belief["is_visible"]:
                    pred_pos = belief["predicted_position"]
                    correction_error = self.physics.distance(pred_pos, peer_pos)
                    print(
                        f"[BELIEF CORRECTION] peer={peer_id} "
                        f"predicted={pred_pos} actual={peer_pos} "
                        f"error={correction_error:.2f} "
                        f"after {belief['time_since_last_seen']} steps"
                    )
                    if self.flight_recorder is not None:
                        self.flight_recorder.record_correction(
                            observer_id=agent_id,
                            peer_id=peer_id,
                            predicted_pos=pred_pos,
                            actual_pos=peer_pos,
                            error_dist=correction_error,
                            steps_since_seen=belief["time_since_last_seen"],
                            step=self.current_step,
                            sim_time=self.current_step * (1.0 / 60.0),
                        )

                # Case 1 & 3: update from observation — snap to actual position
                old_pos = belief["last_pos"]

                # Compute velocity BEFORE updating last_pos
                belief["last_velocity"] = (
                    peer_pos[0] - old_pos[0],
                    peer_pos[1] - old_pos[1],
                )

                dist = self.physics.distance(old_pos, peer_pos)

                if not self.disable_belief_tracking and not self.consensus:
                    belief["energy_used"] += self._energy(peer_id, dist, self.baseline_payload)
                    belief["predicted_energy"] = belief["energy_used"]

                # Velocity over a window of recent observations (cells per step); used to
                # extrapolate a peer that goes out of view.
                hist = belief.setdefault("_hist", [])
                hist.append((self.current_step, peer_pos))
                if len(hist) > 30:
                    hist.pop(0)
                span = hist[-1][0] - hist[0][0]
                if span > 0:
                    belief["v_est"] = ((hist[-1][1][0] - hist[0][1][0]) / span,
                                       (hist[-1][1][1] - hist[0][1][1]) / span)

                belief["last_pos"] = peer_pos
                belief["predicted_position"] = peer_pos
                belief["last_seen_time"] = self.current_step
                belief["confidence"] = 1.0
                belief["is_visible"] = True
                belief["time_since_last_seen"] = 0

        # Event-based corrections from observable parcel state changes
        self.update_beliefs_from_events(snap)

        # --- Case 2: non-visible peers — forward predict via trajectory extrapolation ---
        for peer_id, belief in self.peer_beliefs.items():
            if peer_id in visible_peer_ids:
                continue

            belief["is_visible"] = False
            belief["time_since_last_seen"] = self.current_step - belief["last_seen_time"]

            t = belief["time_since_last_seen"]
            vx, vy = belief.get("last_velocity", (0.0, 0.0))
            last_x, last_y = belief["last_pos"]

            # Extrapolate position
            if self.sensing_active:
                # Limited observation: windowed velocity, bounded horizon, kept inside the grid.
                vx, vy = belief.get("v_est", (0.0, 0.0))
                t_eff = min(t, 150)
                predicted_x = last_x + vx * t_eff
                predicted_y = last_y + vy * t_eff
                bounds = self._grid_bounds()
                if bounds:
                    predicted_x = min(max(predicted_x, 0.0), bounds[0] - 1)
                    predicted_y = min(max(predicted_y, 0.0), bounds[1] - 1)
            else:
                predicted_x = last_x + vx * t
                predicted_y = last_y + vy * t
            belief["predicted_position"] = (predicted_x, predicted_y)

            # Energy extrapolation
            extrapolated_dist = self.physics.distance(
                belief["last_pos"],
                belief["predicted_position"],
            )

            belief["predicted_energy"] = (
                    belief["energy_used"]
                    + self._energy(peer_id, extrapolated_dist, self.baseline_payload)
            )

            # Confidence decay
            decay = 0.5 ** (t / 10.0)
            belief["confidence"] = max(0.05, decay)

            print(
                f"[BELIEF PREDICT] peer={peer_id} "
                f"predicted_pos={belief['predicted_position']} "
                f"velocity=({vx:.2f},{vy:.2f}) "
                f"predicted_energy={belief['predicted_energy']:.2f} "
                f"confidence={belief['confidence']:.3f} "
                f"t_since_seen={t}"
            )

        return visible_agents

    def _tree(self, name: str):
        """A trained tree by name, loaded once."""
        from .classifier import DecisionTree, load_models
        if self._classifier_models is None:
            self._classifier_models = load_models()
        if name not in self._classifier_trees:
            self._classifier_trees[name] = DecisionTree.from_json(self._classifier_models[name])
        return self._classifier_trees[name]

    def _regime_policy(self, snap) -> str:
        """Greedy or coordination for the whole mission, decided at the first decision."""
        if self._regime_policy_cache is None:
            from .classifier import DecisionTree, load_models
            if self._classifier_models is None:
                self._classifier_models = load_models()
            agents = len(self.peer_beliefs) + 1
            parcels = max(1, len(snap.get("all_parcels", [])))
            regime = self._classifier_models["regime"]
            # Leave-one-configuration-out: use the tree trained without this configuration if there is one.
            tree = DecisionTree.from_json(regime.get(f"A{agents}_P{parcels}") or regime["all"])
            go = tree.predict({"c": agents / parcels, "A": float(agents), "P": float(parcels)})
            self._regime_policy_cache = "hungarian" if go else "greedy"
        return self._regime_policy_cache

    def _grid_bounds(self):
        """(columns, rows) of the grid, or None if unknown."""
        info = self._start_info
        return tuple(info.grid) if info is not None and info.grid is not None else None

    def _base_rate(self, aid) -> float:
        """Consumption multiplier the drone assumes for `aid` before any self-check correction."""
        if aid == self.agent_id:
            if not self.own_rate_known:
                return 1.0
            return float(self.drone.energy_rate) if self.drone is not None else 1.0
        # A drone that does not know its own consumption knows nobody's: peers are assumed to be like it.
        # (Reading a peer's true rate here while also applying the self-check correction would count the
        # difference twice.)
        if self.peer_rate_assumption == "same_as_self" or not self.own_rate_known:
            return self._base_rate(self.agent_id)
        return self._declared_rate(aid)

    def _nominal(self, distance: float, weight: float) -> float:
        return self.physics.energy_cost(distance=distance, weight=weight)

    def _self_ratio(self, distance: float, weight: float) -> float:
        """Learned own cost over the cost the drone would otherwise assume for itself, at this leg."""
        nominal = self._nominal(distance, weight) * self._base_rate(self.agent_id)
        if self.self_model is None or nominal <= 1e-9:
            return 1.0
        return self.self_model.cost(distance, weight) / nominal

    def _self_scale(self) -> float:
        """Single-number summary of the self-check (learned over assumed cost at one cell with the baseline payload)."""
        if not self.use_self_check or self.self_model is None or self.self_model.samples < 2:
            return 1.0
        return round(self._self_ratio(1.0, self.baseline_payload), 3)

    def _rate(self, aid) -> float:
        """Consumption multiplier of `aid` as a single number, for callers that cannot take a cost function
        (the CBBA rounds). Exact costs go through `_energy`, which uses the learned distance and weight terms."""
        scale = self._self_scale()
        return self._base_rate(aid) * scale

    def _note_start_info(self, snap: Dict) -> None:
        if self._start_info is None and snap.get("start_info") is not None:
            self._start_info = snap["start_info"]

    def _spec(self, aid):
        """A drone's declared spec from the start information (None if not known)."""
        info = self._start_info
        return info.fleet.get(aid) if info is not None else None

    def _speed_of(self, aid) -> float:
        """A drone's declared flight speed relative to the nominal (1.0 when the arm is not speed aware or nothing is declared)."""
        spec = self._spec(aid) if self.speed_aware else None
        return float(getattr(spec, "speed", 1.0) or 1.0) if spec is not None else 1.0

    def _peer_pace(self, pid) -> float:
        """Factor from the drone's own pace to a peer's: how many times longer a flight takes the peer (2.0 for a peer half as fast)."""
        return self._speed_of(self.agent_id) / self._speed_of(pid)

    def _declared_rate(self, aid) -> float:
        """A drone's declared unloaded consumption relative to the nominal model (1.0 if not declared)."""
        spec = self._spec(aid)
        if spec is None:
            return 1.0
        return round(spec.unloaded_per_cell / self.physics.energy_cost(distance=1.0, weight=0.0), 9)

    def watch_own_energy(self, snap: Dict) -> None:
        """Self-check, every step: what did the last step of motion cost, carrying what? Feeds the own-energy model.
        Reads the drone's own odometry and battery from its observation."""
        self._note_start_info(snap)
        own = snap["own"]
        pos = (float(own["pos_px"][0]), float(own["pos_px"][1]))
        level = float(own["battery"])
        if self.self_model is None:
            nominal_cell = self.physics.energy_cost(distance=1.0, weight=0.0)
            nominal_wf = round(self.physics.energy_cost(distance=1.0, weight=1.0) / nominal_cell - 1.0, 9)
            rate = self._base_rate(self.agent_id)
            spec = self._spec(self.agent_id)
            wf = float(spec.weight_sensitivity) if (self.own_rate_known and spec is not None) else nominal_wf
            base = nominal_cell * rate
            self.self_model = SelfEnergyModel(base, base * wf)
        if self._watch_prev is not None:
            px, py, plevel = self._watch_prev
            cells = ((pos[0] - px) ** 2 + (pos[1] - py) ** 2) ** 0.5 / float(own["cell_px"])
            spent = plevel - level
            if cells > 1e-9 and spent >= 0:
                self.self_model.update(cells, float(own["carrying_weight"]), spent)
        self._watch_prev = (pos[0], pos[1], level)

    avoid_top_k = False         # set by the factory: the drone leaves out of its candidates the parcels in each free peer's k best (by the peer's own cost), k = min(width, (open - 1) // free peers)
    _last_station = None
    avoid_beliefs = False       # set by the factory: each peer's best parcels are ranked over the parcels the peer is believed to think open, including parcels already taken that it has not yet found gone
    avoid_width_fixed = None    # set by the factory: one width for every drone (overrides the personality's)
    avoid_width = 1             # the personality's width (SPPDEC_PERSONALITIES["avoid_width"]; 1 without a personality)
    declared_weights = False    # set by the factory: the weight of each order is known from the start, so a peer's bid for a parcel is priced at the parcel's weight (as the drone's own is), not at a fixed payload
    peer_model_cost = False     # set by the factory: peers' costs in the assignment (Hungarian and CBBA alike) come from the peer model: its predicted position, the rest of a
                                # carrying peer's flight to the predicted drop cell, the peer's own declared w-bar and beta
    peer_personalities = True   # set by the factory: the peer model uses each peer's declared personality (False: peers are assumed to share the drone's own beta and w-bar)
    station_model = False       # set by the factory: the drone predicts the station (remembered cells plus the drops of simulated peers) and uses it to decide the job is done
    peer_model_shadow = False   # set by the factory: log the peer model's predictions with each decision (nothing reads them)
    _peer_model = None

    station_bids = False        # set by the factory from the arm's spec (see BaseStrategy.station_bids)

    def _bid_drop(self, pick, station):
        """The delivery point a trip is priced to: the cell the drone would drop at when station_bids is on, otherwise the station centre."""
        if self.station_bids:
            free = station.get("free_cells") or [(c, r) for r in range(station["row"], station["row"] + station["h"])
                                                 for c in range(station["col"], station["col"] + station["w"])]
            return min(free, key=lambda c: abs(c[0] - pick[0]) + abs(c[1] - pick[1]))
        return (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)

    def _predicted_free_cells(self, snap: Dict):
        """The station cells the drone believes free now: remembered free cells minus the cells the predicted drops have filled."""
        ps = self._predicted_station(snap) if self._peer_model is not None else None
        station = snap.get("nearest_station")
        if not station:
            return None
        free = [tuple(c) for c in station.get("free_cells") or []]
        if ps is None or ps.get("filled") is None:
            return free
        return [c for c in free if c not in ps["filled"]] or free

    def _drop_for(self, pickup, snap: Dict):
        """Where a parcel from `pickup` is dropped by the drone's belief: the nearest predicted-free station cell (the nearest remembered-free one without the model)."""
        station = snap.get("nearest_station")
        if not station:
            return pickup
        free = self._predicted_free_cells(snap) or [(c, r) for r in range(station["row"], station["row"] + station["h"])
                                                     for c in range(station["col"], station["col"] + station["w"])]
        return min(free, key=lambda c: (abs(c[0] - pickup[0]) + abs(c[1] - pickup[1]), c))

    def _model_peer_cost(self, peer: Dict, pickup, belief, effective_conf: float, snap: Dict, weight: float):
        """
        A peer's cost of taking the parcel at `pickup`, from the peer model (None when it has no prediction yet, and the caller keeps its old estimate): the
        whole trip from where the peer will be (a carrying peer first finishes its flight to its predicted drop cell, at the peer's own w-bar): empty leg to the
        pickup, loaded leg to the drop cell, the pick and the drop, and the fairness penalty at the peer's own beta.
        """
        ps = self._predicted_station(snap)
        pred = ps["model"]["peers"].get(peer["id"]) if ps and ps.get("model") else None
        if pred is None:
            return None
        prof = self._peer_profile(peer["id"])
        pos = pred["pos"]
        cost = 0.0
        if pred.get("carrying") and pred.get("dest") is not None:
            cost += self._energy(peer["id"], self.physics.distance(pos, pred["dest"]), prof["baseline_payload"])
            pos = pred["dest"]
        cost += self._trip_energy(peer["id"], pos, pickup, self._drop_for(pickup, snap), weight)
        if belief:
            cost += prof["energy_penalty_alpha"] * belief["predicted_energy"] * effective_conf
        return cost

    def _ghost_rows(self, snap: Dict, cost_peers, peer_effective_conf) -> Dict[str, Dict[str, float]]:
        """
        Parcels the drone knows are taken that a peer has not yet found out about: the peer was not sensed within sensor range of the parcel's original place since the drone
        learned of the take, so the peer is believed to still think it open. Returns {peer id: {parcel id: the peer's cost of flying to it}}, priced like the real candidates.
        """
        info = self._start_info
        if info is None:
            return {}
        radius = self.sensing_radius
        origin = {q["id"]: (q["col"], q["row"], q.get("declared_weight", 1.0)) for q in info.parcels}
        taken = [(p["id"], p["taken_step"]) for p in snap.get("all_parcels", []) if (p.get("picked") or p.get("delivered")) and p.get("taken_step") is not None and p["id"] in origin]
        out: Dict[str, Dict[str, float]] = {}
        for peer in cost_peers:
            pid = peer["id"]
            belief = self.peer_beliefs.get(pid)
            rows: Dict[str, float] = {}
            for tid, since in taken:
                col, row, w = origin[tid]
                if self._peer_model.saw_after(pid, (col, row), int(since), radius):
                    continue
                c = (self._model_peer_cost(peer, (col, row), belief, peer_effective_conf.get(pid, 0.0), snap, w) if self.peer_model_cost else None)
                if c is None:
                    pos = (belief["predicted_position"] if belief and not belief["is_visible"] else (peer["col"], peer["row"]))
                    c = self._energy(pid, self.physics.distance(pos, (col, row)), w if self.declared_weights else self.baseline_payload)
                rows[tid] = c
            out[pid] = rows
        return out

    def _peer_profile(self, pid: str) -> Dict:
        """beta (energy_penalty_alpha) and w-bar (baseline_payload) of a peer: its declared personality from the shared start information, else the drone's own values."""
        if self.peer_personalities and self._start_info is not None:
            spec = self._start_info.fleet.get(pid)
            profile = SPPDEC_PERSONALITIES.get(getattr(spec, "personality", None))
            if profile:
                return profile
        return {"baseline_payload": self.baseline_payload, "energy_penalty_alpha": self.energy_penalty_alpha}

    def _peer_adjusted_cost(self, pid: str, a, b) -> float:
        """What a peer would pay to fly from a to b by its own reckoning: the energy at its w-bar plus beta times the energy it is believed to have used."""
        prof = self._peer_profile(pid)
        used = (self.peer_beliefs.get(pid) or {}).get("energy_used", 0.0)
        cost = self._energy(pid, self.physics.distance(a, b), prof["baseline_payload"]) + prof["energy_penalty_alpha"] * used
        free = getattr(self, "_trip_free", None)
        if free:
            # the peer's choice is predicted by the same cost as the drone's own: the delivery leg from the parcel to the drop cell counts
            # (the remembered free cells; the prediction itself is still being made, so it cannot be asked for here)
            drop = min(free, key=lambda c: (abs(c[0] - b[0]) + abs(c[1] - b[1]), c))
            cost += self._energy(pid, self.physics.distance(b, drop), prof["baseline_payload"])
        return cost

    def _predicted_station(self, snap: Dict) -> Optional[Dict]:
        """
        The station as the drone believes it is now: the cells it remembers occupied (refreshed whenever it sensed them) plus the cells the simulated
        peers have filled since. A simulated drop at step t into a cell the drone has looked at since t, and found free, did not happen there and is
        not counted (the look wins). Returns {"mem": remembered occupied cells, "pred": believed occupied cells or None before the peer model has
        the drone's own speed, "model": the peer model's output}.
        """
        station = snap.get("nearest_station")
        if not station or self._peer_model is None:
            return None
        cells = [(c, r) for r in range(station["row"], station["row"] + station["h"]) for c in range(station["col"], station["col"] + station["w"])]
        free = [tuple(c) for c in station.get("free_cells") or []]
        occupied = {c for c in cells if c not in set(free)}
        step = int(snap.get("step", 0))
        key = (step // 10, len(free), self._peer_model.version)       # the prediction moves slowly: recomputed every ten steps or when the memory or a sighting changes
        cached = getattr(self, "_station_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        open_now = [p for p in snap.get("all_parcels", []) if not p.get("picked") and not p.get("delivered")]
        own_carried = bool(snap.get("carrying", {}).get("has"))
        taken = [(p["id"], (p["col"], p["row"]), p["taken_step"]) for p in snap.get("all_parcels", [])
                 if p.get("picked") and not p.get("delivered") and p.get("taken_step") is not None] if not own_carried else []
        self._trip_free = free
        model = self._peer_model.predict(step, open_now, free, cells, taken, self._peer_adjusted_cost)
        if model is None:
            result = {"mem": len(occupied), "pred": None, "model": None, "filled": None}
            self._station_cache = (key, result)
            return result
        seen = snap.get("station_seen_step", {})
        # Inferred drops are kept: when a peer is sighted again its earlier drops would otherwise be forgotten, though they may have filled a cell the
        # drone cannot see. A drop is identified by the peer and the time (to 20 steps); the first inference stands.
        committed = getattr(self, "_committed_drops", None)
        if committed is None:
            committed = self._committed_drops = {}
        for t, c, who in model["drops"]:
            committed.setdefault((who, t // 20), (t, c))
        # A drop into a cell the memory already holds occupied is taken as that cell (nothing to add). A drop into a cell the drone has looked at since,
        # and found free, did not happen there; it went to another cell the drone has not looked at since, the nearest one.
        filled = set()
        for t, c in sorted(committed.values()):
            if c in occupied:
                continue
            if c not in filled and seen.get(c, -1) < t:
                filled.add(c)
                continue
            alt = [x for x in cells if x not in occupied and x not in filled and seen.get(x, -1) < t]
            if alt:
                filled.add(min(alt, key=lambda x: (self.physics.distance(x, c), x)))
        result = {"mem": len(occupied), "pred": len(occupied) + len(filled), "model": model, "filled": filled}
        self._station_cache = (key, result)
        return result

    def predicts_job_done(self, snap: Dict) -> bool:
        """The predicted station holds every parcel."""
        if not self.station_model or self._start_info is None:
            return False
        ps = self._predicted_station(snap)
        return bool(ps and ps["pred"] is not None and ps["pred"] >= self._start_info.n_orders)

    def _ledger_says_done(self, snap: Dict) -> bool:
        """The station holds as many parcels as exist: every parcel has been delivered."""
        station, info = snap.get("nearest_station"), self._start_info
        if not station or info is None:
            return False
        occupied = station["w"] * station["h"] - len(station.get("free_cells") or [])
        return occupied >= info.n_orders

    def _territory_owners(self) -> Dict[str, str]:
        """parcel id -> the drone whose territory it lies in (nearest start cell, ties to the lowest drone id); computed once
        from the start information every drone shares."""
        if self._territory_owner is None:
            info = self._start_info
            if info is None:
                return {}
            starts = sorted((pid, tuple(spec.start_cell)) for pid, spec in info.fleet.items())
            self._territory_owner = {
                p["id"]: min(starts, key=lambda s: (self.physics.distance(s[1], (p["col"], p["row"])), s[0]))[0]
                for p in info.parcels}
        return self._territory_owner

    FRESH_Q_MIN = 0.2   # a parcel's chance of being still there is not taken below this (the cost at most quintuples)

    def _parcel_presence(self, snap: Dict, parcels: List[Dict]) -> Dict[str, float]:
        """
        Chance q that each parcel is still where the drone last saw it. The fleet takes parcels at a rate R, estimated from
        this drone's own pickups (its peers behave like it): R = (peers + 1) x own pickups / steps so far; the parcels the
        drone believes open share these takes equally, so each disappears at rate h = R / open; after `age` steps without
        being seen, q = exp(-h x age). Parcels in sensor range have age 0. Before the drone's first pickup nothing is
        known about the rate and q = 1. Returns {} when freshness is off.
        """
        if not self.freshness or not self._pick_steps:
            return {}
        step = max(1, snap.get("step", self.current_step))
        open_n = max(1, sum(1 for p in snap.get("all_parcels", []) if not p.get("picked") and not p.get("delivered")))
        h = (len(self.peer_beliefs) + 1) * len(self._pick_steps) / step / open_n
        return {p["id"]: max(self.FRESH_Q_MIN, math.exp(-h * max(0, step - p.get("seen_step", 0)))) for p in parcels}

    def note_plan_result(self, success: bool) -> None:
        """The plan the drone was executing ended: it picked its parcel, or it had to give the plan up."""
        if success:
            self._pick_steps.append(self.current_step)
        # Calibration of the drone's own forecast: it forecast this success rate before the plan ended.
        forecast = (self._plan_hits[0] + 1.0) / (self._plan_hits[1] + 2.0)
        self._brier[0] += (forecast - (1.0 if success else 0.0)) ** 2
        self._brier[1] += 1
        self._plan_hits[0] = self._plan_hits[0] * self.TRACK_DECAY + (1.0 if success else 0.0)
        self._plan_hits[1] = self._plan_hits[1] * self.TRACK_DECAY + 1.0

    def _pick_drop_energy(self, aid, weight) -> float:
        """Energy of one pick or one drop for drone `aid`: the action's nominal cost scaled by its declared consumption
        and weight sensitivity."""
        spec = self._spec(aid)
        s = spec.weight_sensitivity if spec is not None else \
            round(self.physics.pick_drop_cost(1.0) / self.physics.pick_drop_cost(0.0) - 1.0, 9)
        return self._base_rate(aid) * self.physics.pick_drop_cost(0.0) * (1.0 + weight * s)

    def _trip_energy(self, aid, frm, pickup, drop, weight) -> float:
        """The energy of one delivery trip, the definition every strategy shares: the pickup leg flown empty, the
        delivery leg flown loaded, and the energy of the pick and of the drop."""
        return (self._energy(aid, self.physics.distance(frm, pickup), 0.0)
                + self._energy(aid, self.physics.distance(pickup, drop), weight)
                + 2.0 * self._pick_drop_energy(aid, weight))

    def _energy(self, aid, distance, weight) -> float:
        """Energy agent `aid` would spend moving `distance` cells carrying `weight`."""
        nominal = self._nominal(distance, weight)
        if not self.use_self_check or self.self_model is None or self.self_model.samples < 2:
            return nominal * self._base_rate(aid)
        if aid == self.agent_id:
            return self.self_model.cost(distance, weight)
        # peers: what the drone assumes about them. "like_self": corrected by how its own cost differs from its
        # assumption (peers resemble it); "nominal": no correction, peers are assumed to follow the nominal model.
        if self.peer_prior == "nominal":
            return nominal * self._base_rate(aid)
        return nominal * self._base_rate(aid) * self._self_ratio(distance, weight)

    def _init_start_peers(self, snap: Dict, agent: Dict) -> None:
        """
        Under limited observation: the start state of every peer (position, full charge, carrying nothing) is known
        before the first step, from the start information. After that a peer is known only while sensed.
        """
        if self._start_state_done:
            return
        self._start_state_done = True
        agent_id = agent["id"]
        info = snap.get("start_info")
        starts = [(pid, tuple(spec.start_cell)) for pid, spec in info.fleet.items()] if info else []
        for pid, ppos in starts:
            if pid == agent_id or pid in self.peer_beliefs:
                continue
            self.peer_beliefs[pid] = {
                "last_pos": ppos,
                "last_seen_time": snap.get("step", 0),
                "energy_used": 0.0,
                "predicted_position": ppos,
                "predicted_energy": 0.0,
                "assigned_task": None,
                "confidence": 1.0,
                "is_visible": True,
                "time_since_last_seen": 0,
                "last_velocity": (0.0, 0.0),
            }

    def update_beliefs_from_events(self, snap: Dict):
        """
        Event-based belief correction triggered by observable parcel state changes.

        When a parcel transitions from unpicked → picked or picked → delivered,
        we infer which agent most likely caused the event and update their
        belief state accordingly.

        This is purely inferential — no communication or shared state.
        Assignment is made to the closest believed agent at the time of detection.

        Called once per cycle from perceive(), after belief updates.
        """

        current_step = snap.get("step", self.current_step)

        for parcel in snap.get("all_parcels", []):
            parcel_id = parcel["id"]
            parcel_pos = (parcel["col"], parcel["row"])

            # -----------------------------------------------
            # Event: parcel was just picked up
            # -----------------------------------------------
            if parcel.get("picked") and not parcel.get("delivered"):
                # Find the closest believed agent to the parcel pickup location
                closest_peer = self._closest_believed_agent(parcel_pos)
                if closest_peer and closest_peer in self.peer_beliefs:
                    belief = self.peer_beliefs[closest_peer]

                    # Only assign if not already tracking a task
                    if belief["assigned_task"] is None:
                        belief["assigned_task"] = parcel_id

                        # Snap predicted position to parcel location
                        # — agent must be at or near pickup to have triggered this
                        belief["predicted_position"] = parcel_pos

                        print(
                            f"[EVENT] parcel={parcel_id[:8]} picked — "
                            f"assigned to believed agent={closest_peer} "
                            f"confidence={belief['confidence']:.2f}"
                        )

            # -----------------------------------------------
            # Event: parcel was just delivered
            # -----------------------------------------------
            if parcel.get("delivered"):
                # Find whichever peer was believed to be carrying this parcel
                for peer_id, belief in self.peer_beliefs.items():
                    if belief["assigned_task"] == parcel_id:
                        belief["assigned_task"] = None

                        print(
                            f"[EVENT] parcel={parcel_id[:8]} delivered — "
                            f"clearing task hypothesis for agent={peer_id}"
                        )
                        break

    def _closest_believed_agent(self, pos: Tuple[int, int]) -> Optional[str]:
        """
        Return the peer_id of the believed agent closest to pos.
        Uses predicted_position for all peers regardless of visibility.
        Returns None if no beliefs exist.
        """
        if not self.peer_beliefs:
            return None

        return min(
            self.peer_beliefs.keys(),
            key=lambda pid: self.physics.distance(
                self.peer_beliefs[pid]["predicted_position"], pos
            )
        )

    def _compute_infeasibility_belief(self, state: SPPDECAgentState) -> float:
        """
        Compute an epistemic belief that emerges from experience rather than heuristics.

        This belief is an online learned probability model trained from the agent's
        own outcomes. It estimates:

            P(local failure | current grounded signals)

        The model updates itself continuously using observed consequences such as:
        - Running out of battery
        - Being forced into fallback
        - Task infeasibility
        - Unexpected aborts

        No fixed semantic constants are used. All parameters are learned online.

        Signals used (all grounded):
        - normalized energy prediction error
        - environmental pressure (competition and scarcity)
        - energy margin pressure

        Returns
        -------
        float
            Learned probability of local infeasibility in [0, 1]
        """

        # --------------------------------------------------
        # 1. Extract grounded signals
        # --------------------------------------------------
        error = state.get("energy_prediction_error", 0.0)

        tasks = len(state.get("feasible_tasks", []))
        peers = len(state.get("visible_agents", []))

        # Competition grounded in observation
        competition_pressure = peers / max(tasks, 1)
        competition_pressure = max(0.0, min(1.0, competition_pressure))

        # Scarcity grounded in observed historical maximum
        prev_max_tasks = getattr(self, "_max_tasks_seen", tasks)
        self._max_tasks_seen = max(prev_max_tasks, tasks)

        if self._max_tasks_seen > 0:
            scarcity_pressure = 1.0 - (tasks / self._max_tasks_seen)
        else:
            scarcity_pressure = 0.0

        environmental_pressure = max(competition_pressure, scarcity_pressure)

        # Energy pressure grounded in observed battery range
        battery = self.drone.power.level if self.drone else 0.0
        prev_max_battery = getattr(self, "_max_battery_seen", battery)
        self._max_battery_seen = max(prev_max_battery, battery)

        if self._max_battery_seen > 0:
            energy_pressure = 1.0 - (battery / self._max_battery_seen)
        else:
            energy_pressure = 0.0

        # Error normalization grounded in observed maximum error
        prev_max_error = getattr(self, "_max_error_seen", error)
        self._max_error_seen = max(prev_max_error, error)

        if self._max_error_seen > 0:
            normalized_error = error / self._max_error_seen
        else:
            normalized_error = 0.0

        # Final bounded signals
        x = [
            max(0.0, min(1.0, normalized_error)),
            max(0.0, min(1.0, environmental_pressure)),
            max(0.0, min(1.0, energy_pressure)),
        ]

        # --------------------------------------------------
        # 2. Initialize belief memory if needed
        # --------------------------------------------------
        if not hasattr(self, "_belief_history"):
            self._belief_history: List[List[float]] = []

        self._belief_history.append(x)

        # Keep last 30 observations only (an arm of the sensitivity study may change the window or switch the belief off)
        sw = getattr(self, "sweep", None) or {}
        window = int(sw.get("belief_window", 30))
        if len(self._belief_history) > window:
            self._belief_history.pop(0)
        if sw.get("belief_off"):
            return 0.5

        # --------------------------------------------------
        # 3. Emergent belief computation (experience driven)
        # --------------------------------------------------
        # Belief is proportional to how extreme current signals are
        # compared to what the agent has previously experienced.

        if len(self._belief_history) < 5:
            # Not enough experience yet, remain epistemically cautious but neutral
            belief = 0.5
        else:
            # Compute empirical rarity of current signal vector
            distances = []
            for past in self._belief_history:
                dist = sum(abs(a - b) for a, b in zip(x, past)) / len(x)
                distances.append(dist)

            avg_distance = sum(distances) / len(distances)

            # Track max deviation ever observed
            prev_max_dev = getattr(self, "_max_belief_deviation", avg_distance)
            self._max_belief_deviation = max(prev_max_dev, avg_distance)

            if self._max_belief_deviation > 0:
                belief = avg_distance / self._max_belief_deviation
            else:
                belief = 0.0

            belief = max(0.0, min(1.0, belief))

        print(
            f"[BELIEF] agent={self.agent_id} "
            f"err={x[0]:.3f} "
            f"env={x[1]:.3f} "
            f"energy={x[2]:.3f} "
            f"belief={belief:.3f}"
        )

        return belief

    def extract_first_json(self, text: str) -> dict:
        """
        Extract the first JSON-like object from text, repair common LLM issues,
        and return a validated Python dict.

        Raises ValueError only if extraction and repair both fail.
        """
        import json
        import re

        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")

        raw = text.strip()

        # -------------------------------------------------
        # 1. Try direct JSON parse
        # -------------------------------------------------
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return self._validate_payload(obj)
        except Exception:
            pass

        # -------------------------------------------------
        # 2. Handle double-encoded JSON
        # Example: "\"{...}\""
        # -------------------------------------------------
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, str):
                obj = json.loads(decoded)
                if isinstance(obj, dict):
                    return self._validate_payload(obj)
        except Exception:
            pass

        # -------------------------------------------------
        # 3. Extract first balanced {...} block
        # -------------------------------------------------
        start = raw.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in:\n{raw[:300]}")

        depth = 0
        end = None

        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            raise ValueError("Unclosed JSON object")

        candidate = raw[start:end].strip()

        # -------------------------------------------------
        # 4. Repair common issues
        # -------------------------------------------------

        # Replace single quotes only if no double quotes exist
        if "'" in candidate and '"' not in candidate:
            candidate = candidate.replace("'", '"')

        # Remove trailing commas
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)

        # Reject obvious template placeholders
        if "{your answer}" in candidate or "{correct parcel id}" in candidate:
            raise ValueError(f"Template echoed instead of answer:\n{candidate}")

        # -------------------------------------------------
        # 5. Parse repaired JSON
        # -------------------------------------------------
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON extraction failed.\nCandidate:\n{candidate}\n\nOriginal:\n{raw[:300]}"
            ) from e

        if not isinstance(obj, dict):
            raise ValueError(f"JSON must be an object, got {type(obj)}")

        return self._validate_payload(obj)

    @staticmethod
    def _validate_payload(data: dict) -> dict:
        """
        Enforce schema and apply safe defaults for policy selection.

        Policy values are normalised to lowercase before validation so that
        model outputs like "GREEDY" or "Hungarian" are accepted. The old
        prompt provided lowercase templates to copy; the new chain-of-thought
        prompt elicits natural sentence-case reasoning before the JSON, so
        the model naturally writes uppercase policy names in its conclusion.
        """
        if "policy" not in data and data.get("p") in ("G", "H"):
            # Compact prompt styles answer p = G | H
            data["policy"] = "greedy" if data["p"] == "G" else "hungarian"
        if "policy" not in data and data.get("verdict") in ("G", "H"):
            data["policy"] = "greedy" if data["verdict"] == "G" else "hungarian"
        if "policy" in data:
            # Normalise case before validation — model outputs GREEDY/HUNGARIAN
            # when reasoning in natural language before the JSON block
            policy = str(data["policy"]).lower().strip()
            data["policy"] = policy
            if policy not in ("greedy", "hungarian"):
                raise ValueError(f"Invalid policy value: {policy}")
        elif "parcel_id" in data:
            # Legacy support - treat as task selection
            if data["parcel_id"] in (None, "", "{your answer}", "{correct parcel id}"):
                raise ValueError(f"Invalid parcel_id value: {data['parcel_id']}")
        else:
            raise ValueError(f"Missing 'policy' or 'parcel_id' in {data}")

        try:
            conf = float(data.get("confidence", 0.5))
            data["confidence"] = max(0.0, min(1.0, conf))
        except Exception:
            data["confidence"] = 0.5

        return data

    def reason_with_slm(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Strategic meta-controller for SPPDEC coordination policy selection.

        This function represents the core intelligence layer of SPPDEC, where a Small Language Model
        (SLM) acts as a learned meta-controller that arbitrates between two grounded coordination
        strategies based on environmental context, uncertainty beliefs, and temporal trends.

        Architecture Overview
        ---------------------
        The SLM operates as a bounded arbitrator over two deterministic planners:
        1. **Greedy Planner**: Local optimization (lowest personal cost)
        2. **Hungarian Planner**: Global coordination (optimal team assignment using peer beliefs)

        Rather than inventing plans or choosing specific tasks, the SLM selects which planning
        philosophy to apply, then the system executes the corresponding pre-computed assignment.
        This ensures all plans remain physics-grounded while enabling strategic adaptability.

        Strategic Context Inputs
        ------------------------
        The SLM receives rich temporal and environmental context:

        **Uncertainty Signals:**
        - Current belief infeasibility (0.0-1.0): Agent's epistemic uncertainty about environment
        - Energy prediction error: How much actual costs exceeded predicted costs
        - Temporal trends: RISING/FALLING/STABLE patterns over last 5 decision cycles

        **Environmental Pressure:**
        - Task competition ratio: How many peers compete for available tasks
        - Task scarcity indicators: Absolute task availability
        - Competition trends: Whether competitive pressure is increasing

        **Coordination Trade-offs:**
        - Greedy task cost vs Hungarian task cost differential
        - Risk assessment: Personal efficiency vs team optimality
        - Strategic timing: When coordination attempts are worth the risk

        Decision Framework
        ------------------
        A deterministic pre-filter handles mathematically unambiguous cases before
        the SLM is invoked, ensuring correctness on clear-cut decisions and reserving
        SLM calls for genuine ambiguity.

        **Pre-filter (deterministic, no SLM call):**
        - Identical costs AND low competition  → GREEDY (no benefit, no pressure)
        - Identical costs AND high competition → HUNGARIAN (break symmetry under pressure)
        - Hungarian strictly cheaper           → HUNGARIAN (dominates unconditionally)
        - Small cost premium AND high competition → HUNGARIAN (cheap coordination opportunity)
        - Cost premium >25% AND belief > 0.7  → GREEDY (too uncertain to justify sacrifice)
        - Belief >= 0.9 (any premium)         → GREEDY (peer estimates too unreliable)

        **SLM reasoning (genuine ambiguity only):**

        Cost Differential (primary signal):
        - 0-10% premium → cheap coordination, proceed to uncertainty gating
        - 10-25% premium → coordination requires justification

        Uncertainty Gating (secondary):
        - Belief infeasibility > 0.7 → peer cost estimates unreliable → GREEDY
        - Belief infeasibility < 0.3 → estimates trustworthy → coordination safe
        - Medium uncertainty (0.3-0.7) → resolve via trend analysis

        Trend Resolution (tiebreaker):
        - Uncertainty FALLING AND error FALLING → estimates improving → HUNGARIAN
        - Uncertainty RISING OR error RISING → estimates degrading → GREEDY
        - Both STABLE → competition ratio > 0.15 → HUNGARIAN, else GREEDY

        **Adaptive Behavior:**
        - Early mission: High errors, unknown trends → Conservative greedy choices
        - Mid mission: Established trends, learning → Strategic coordination attempts
        - Late mission: Stable patterns → Confident optimal decisions

        Output Contract
        ---------------
        **Direct Task Selection:**
        - Sets `state["selected_task"]` directly (no parcel-level indirection)
        - Sets `state["decision_mode"] = "candidate"` to proceed with execution
        - Preserves selected strategy in `state["slm_policy"]` for analysis

        **Honest Epistemics:**
        - Success: SLM made strategic choice → `"slm_status": "success"`
        - Fallback: Hungarian failed → `"slm_status": "abstain", "slm_policy": "greedy_fallback"`
        - SLM failure: Complete failure → `"slm_status": "failed", "slm_policy": "greedy_fallback"`

        **Experimental Validity:**
        - All decisions traceable and categorizable
        - No hidden fallbacks disguised as intelligent choices
        - Confidence levels reflect actual decision quality

        Research Significance
        ---------------------
        This approach represents a principled integration of learned strategic reasoning with
        deterministic coordination algorithms. The SLM handles the "when to coordinate" meta-
        decision while grounded planners handle the "how to coordinate" mechanics.

        Key advantages:
        - **Bounded Intelligence**: SLM cannot hallucinate invalid plans
        - **Temporal Adaptability**: Strategic choices evolve based on experience
        - **Experimentally Valid**: All decisions are traceable and honest
        - **Performance**: Maintains SPPDEC's 30-40% speed advantages over centralized approaches

        Failure Modes & Robustness
        ---------------------------
        **SLM Failures:**
        - JSON parsing errors → Retry with backoff → Greedy fallback
        - Invalid policy choices → Immediate greedy fallback
        - Network/model failures → Graceful degradation to deterministic behavior

        **Coordination Failures:**
        - Hungarian computation failure → Deterministic greedy selection (no fake SLM choice)
        - No feasible tasks → Early termination with honest "abstain" status

        **Learning Signal Quality:**
        - Energy prediction errors tracked over time for self-trust estimation
        - Belief formation grounded in physics (peer motion → energy expenditure)
        - Uncertainty quantification through multiple environmental pressure signals

        Parameters
        ----------
        state : SPPDECAgentState
            Mutable state containing:
            - feasible_tasks: Pre-computed grounded task options
            - greedy_choice: Lowest-cost task for this agent
            - hungarian_choice: Globally optimal assignment for this agent
            - belief_infeasibility: Current uncertainty estimate
            - energy_prediction_error: Running average of prediction errors
            - visible_agents: Peer agents for competition assessment

        Returns
        -------
        SPPDECAgentState
            Updated state with:
            - selected_task: Chosen task (greedy or Hungarian)
            - decision_mode: "candidate" (proceed) or original (early termination)
            - slm_policy: "greedy", "hungarian", or "greedy_fallback"
            - slm_status: "success", "abstain", or "failed"
            - slm_confidence: [0.0, 1.0] confidence in the strategic choice

        Raises
        ------
        RuntimeError
            - If drone not bound before SLM invocation
            - If SLM not enabled on bound drone
            - If invalid policy returned from SLM (after retries)
        """
        self._init_trace(state, f"reason_with_slm({state.get('slm_status', 'pre')})")

        if state.get("decision_mode") == "idle":
            return state

        tasks = state.get("feasible_tasks", [])
        agent_id = state["snapshot"]["agent"]["id"]

        if not tasks:
            self._init_trace(state, "reason_with_slm(abstain_empty)")
            state["slm_status"] = "abstain"
            if self.consensus:
                state["decision_mode"] = "idle"          # nothing affordable is believed open: the drone parks instead of waiting
            return state

        if self.consensus:
            return self._consensus_decide(state)

        # Plan ahead: the next planned task that is still open (affordable, not taken) is the decision; no model call.
        # A planned task that is gone or no longer affordable is skipped.
        while self.plan_ahead and self.planned_queue:
            planned_id = self.planned_queue.pop(0)
            planned = next((t for t in tasks if t["parcel_id"] == planned_id), None)
            if planned is not None:
                state["selected_task"] = planned
                state["decision_mode"] = "candidate"
                state["slm_status"] = "success"
                state["slm_policy"] = "plan_ahead_queue"
                state["slm_confidence"] = 0.5
                state["slm_gate"] = "plan ahead: the next planned task is still open"
                print(f"[SPPDEC][{agent_id}] Plan ahead: taking the planned task {planned['parcel_id']}")
                return state

        if self.drone is None:
            raise RuntimeError(
                f"[SPPDEC][{agent_id}] Drone not bound to LangGraph before SLM call"
            )

        if not getattr(self.drone, "slm_enabled", False):
            raise RuntimeError(
                f"[SPPDEC][{agent_id}] Drone SLM not active"
            )

        # Get the precomputed policy choices
        greedy_choice = state.get("greedy_choice")
        hungarian_choice = state.get("hungarian_choice")

        if not greedy_choice:
            state["slm_status"] = "abstain"
            return state

        if self.arbiter == "always_cbba" or self.coordination_model == "cbba":
            cbba_choice = state.get("cbba_choice")
            if not cbba_choice:
                return self._no_slot(state, greedy_choice, agent_id, "cbba")
            hungarian_choice = cbba_choice

        if not hungarian_choice:
            if self.arbiter in ("always_hungarian", "always_cbba"):
                return self._no_slot(state, greedy_choice, agent_id, "hungarian")
            print(f"\n[SPPDEC FALLBACK] Agent: {agent_id}")
            print("  Hungarian assignment computation failed")
            print(f"  Using greedy fallback: {greedy_choice['parcel_id']} (cost: {greedy_choice['cost']:.2f})")

            state["selected_task"] = greedy_choice
            state["decision_mode"] = "candidate"
            state["slm_status"] = "abstain"
            state["slm_policy"] = "greedy_fallback"

            return state

        # Extract contextual signals for policy decision
        belief_infeasibility = state.get("belief_infeasibility", 0.0)
        energy_error = state.get("energy_prediction_error", 0.0)
        num_peers = len(self.peer_beliefs)
        num_tasks = len(tasks)

        # Track time series for trend analysis
        if not hasattr(self, '_decision_history'):
            self._decision_history = {
                'belief_series': [],
                'error_series': [],
                'competition_series': []
            }

        # Update history (keep last 5 cycles)
        self._decision_history['belief_series'].append(belief_infeasibility)
        self._decision_history['error_series'].append(energy_error)
        self._decision_history['competition_series'].append(num_peers / max(num_tasks, 1))

        for key in self._decision_history:
            if len(self._decision_history[key]) > 5:
                self._decision_history[key].pop(0)

        # Compute trend summaries
        def get_trend(series, threshold=0.1):
            if len(series) < 2:
                return "STABLE"
            change = series[-1] - series[0]
            if abs(change) < threshold:
                return "STABLE"
            return "RISING" if change > 0 else "FALLING"

        belief_trend = get_trend(self._decision_history['belief_series'], 0.1)
        error_trend = get_trend(self._decision_history['error_series'], 0.1)
        competition_trend = get_trend(self._decision_history['competition_series'], 0.05)

        # Compute task competition ratio (visible peers only — used in SLM prompt)
        competition_ratio = num_peers / max(num_tasks, 1)

        # ------------------------------------------------------------------
        # Confidence adjustment: incorporate competition and coordination gain
        #
        # Uses all known peers (visible + predicted) for competition pressure,
        # and the greedy/hungarian cost differential for coordination gain.
        #
        # FIX 1: coordination_signal is now signed, not one-sided.
        # Positive signal (greedy cheaper than hungarian) → coordination_penalty > 0
        #   → reduces confidence in the chosen path (coordination is costly).
        # Negative signal (hungarian cheaper than greedy) → coordination_penalty < 0
        #   → slightly boosts confidence (coordination is free, go for it).
        # Clamped to [-0.5, 1.0] so the boost can't overwhelm competition_penalty.
        # ------------------------------------------------------------------
        num_agents = len(self.peer_beliefs) + 1  # all known peers + self
        competition_ratio_agents = num_agents / max(num_tasks, 1)

        coordination_gain = greedy_choice["cost"] - hungarian_choice["cost"]
        coordination_signal = coordination_gain / max(greedy_choice["cost"], 1e-6)

        competition_penalty = min(1.0, competition_ratio_agents / 2.0)
        # Signed: positive penalises (greedy cheaper), negative rewards (hungarian cheaper)
        coordination_penalty = max(-0.5, min(1.0, coordination_signal))

        def adjust_confidence(base: float) -> float:
            return max(0.0, min(1.0, base * (1.0 - competition_penalty) * (1.0 - coordination_penalty)))

        print(f"\n[SPPDEC SLM CONTEXT] Agent: {agent_id}")
        print(f"  Belief infeasibility: {belief_infeasibility:.3f}")
        print(f"  Energy prediction error: {energy_error:.3f}")
        print(f"  Competition ratio (visible peers/tasks): {competition_ratio:.2f}")
        print(f"  Competition ratio (all known peers/tasks): {competition_ratio_agents:.2f}")
        print(f"  Greedy choice: {greedy_choice['parcel_id']} (cost: {greedy_choice['cost']:.2f})")
        print(f"  Hungarian choice: {hungarian_choice['parcel_id']} (cost: {hungarian_choice['cost']:.2f})")
        print(f"  Coordination gain (greedy-hungarian): {coordination_gain:.3f} ({coordination_signal * 100:.0f}%)")
        print(f"  Competition penalty: {competition_penalty:.3f}")
        print(f"  Coordination penalty (signed): {coordination_penalty:.3f}")
        print(f"  Trends: belief={belief_trend}, error={error_trend}, competition={competition_trend}")

        # Format recent history for prompt
        belief_history = [f"{x:.2f}" for x in self._decision_history['belief_series']]
        error_history = [f"{x:.2f}" for x in self._decision_history['error_series']]
        competition_history = [f"{x:.2f}" for x in self._decision_history['competition_series']]

        # ------------------------------------------------------------------
        # Build cost differential signals
        # ------------------------------------------------------------------
        cost_diff = hungarian_choice["cost"] - greedy_choice["cost"]
        cost_diff_pct = abs(cost_diff) / max(greedy_choice["cost"], 1e-6) * 100
        costs_identical = abs(cost_diff) < 1e-6

        if costs_identical:
            cost_diff_label = "identical — no coordination benefit available"
        elif cost_diff > 0:
            cost_diff_label = (
                f"Hungarian costs {cost_diff:+.2f} more ({cost_diff_pct:.0f}% premium) "
                f"— you sacrifice personal efficiency for team benefit"
            )
        else:
            cost_diff_label = (
                f"Hungarian costs {cost_diff:+.2f} less ({cost_diff_pct:.0f}% saving) "
                f"— coordination is strictly better"
            )

        # ------------------------------------------------------------------
        # Shared commit path
        # ------------------------------------------------------------------
        def _commit(policy: str, task: dict, confidence: float, label: str, check=None):
            state["selected_task"] = task
            state["decision_mode"] = "candidate"
            state["slm_status"] = "success"
            state["slm_policy"] = policy
            state["slm_confidence"] = confidence
            state["slm_gate"] = label

            if self.flight_recorder is not None:
                snap = state["snapshot"]
                self.flight_recorder.record_slm_decision(
                    agent_id=agent_id,
                    step=snap.get("step", 0),
                    sim_time=self.current_step * (1.0 / 60.0),
                    policy=policy,
                    confidence=confidence,
                    belief_infeasibility=belief_infeasibility,
                    energy_error=energy_error,
                    competition_ratio=competition_ratio,
                    belief_trend=belief_trend,
                    error_trend=error_trend,
                    competition_trend=competition_trend,
                    greedy_cost=greedy_choice["cost"],
                    hungarian_cost=hungarian_choice["cost"] if hungarian_choice else None,
                    gate=label,
                    arbiter=self.arbiter,
                    prompt_inputs=state.get("slm_prompt_inputs") if label.startswith("slm") else None,
                    check=check,
                    proposals=self._proposal_summary(state),
                    status="success",
                    attempts=state.get("slm_attempts") if label.startswith("slm") else None,
                )

            print(f"\n[SPPDEC SLM DECISION] Agent: {agent_id}")
            print(f"  Policy: {policy.upper()}  [{label}]")
            print(f"  Selected task: {task['parcel_id']}")
            print(f"  Task cost: {task['cost']:.2f}")
            print(f"  Confidence: {confidence:.2f}")

        def _record_failed(reason: str, attempts: int):
            """The model gave no usable answer: the drone falls back to greedy; keep the outcome."""
            if self.flight_recorder is None:
                return
            snap = state["snapshot"]
            self.flight_recorder.record_slm_decision(
                agent_id=agent_id,
                step=snap.get("step", 0),
                sim_time=self.current_step * (1.0 / 60.0),
                policy="greedy_fallback",
                confidence=state.get("slm_confidence", 0.0),
                belief_infeasibility=belief_infeasibility,
                energy_error=energy_error,
                competition_ratio=competition_ratio,
                belief_trend=belief_trend,
                error_trend=error_trend,
                competition_trend=competition_trend,
                greedy_cost=greedy_choice["cost"],
                hungarian_cost=hungarian_choice["cost"] if hungarian_choice else None,
                gate="slm: failed",
                arbiter=self.arbiter,
                proposals=self._proposal_summary(state),
                status="failed",
                failure_reason=reason,
                attempts=attempts,
            )

        # ------------------------------------------------------------------
        # Context arbiters: one shared context, all four reasoning models to choose from
        # ------------------------------------------------------------------
        if self.arbiter in self.CONTEXT_ARBITERS:
            facts = {"belief_infeasibility": belief_infeasibility, "energy_error": energy_error,
                     "belief_trend": belief_trend, "error_trend": error_trend,
                     "competition_ratio": competition_ratio, "competition_trend": competition_trend,
                     "num_peers": num_peers, "num_tasks": num_tasks}
            if self.arbiter == "context_slm_task":
                chosen, label, check = self._decide_task(state, facts)
                _commit("slm_task", chosen, adjust_confidence(0.8), label, check=check)
                return state
            model, label, check = self._decide_with_context(state, facts)
            chosen = state["model_choices"][model]
            _commit(model, chosen, adjust_confidence(0.8), label, check=check)
            return state

        # ------------------------------------------------------------------
        # Fixed-policy arbiters (ablation): bypass pre-filter and SLM
        # ------------------------------------------------------------------
        if self.arbiter == "always_greedy":
            _commit("greedy", greedy_choice, adjust_confidence(1.0), "arbiter: always_greedy")
            return state
        if self.arbiter == "always_hungarian":
            _commit("hungarian", hungarian_choice, adjust_confidence(1.0), "arbiter: always_hungarian")
            return state
        if self.arbiter == "always_cbba":
            _commit("cbba", hungarian_choice, adjust_confidence(1.0), "arbiter: always_cbba")
            return state
        if self.arbiter == "classifier_regime":
            policy = self._regime_policy(state["snapshot"])
            chosen = greedy_choice if policy == "greedy" else hungarian_choice
            _commit(policy, chosen, adjust_confidence(1.0), f"arbiter: regime classifier ({policy})")
            return state

        # ------------------------------------------------------------------
        # Deterministic pre-filter
        # ------------------------------------------------------------------

        # Case 1: costs are identical.
        # FIX 2: under meaningful competition, break symmetry toward hungarian
        # instead of always defaulting to greedy. When multiple agents face
        # identical costs they will all pick the same task; routing the
        # higher-competition case to hungarian distributes them.
        if costs_identical:
            if competition_ratio_agents > 0.5:
                _commit(
                    "hungarian", hungarian_choice, adjust_confidence(1.0),
                    "pre-filter: identical costs, symmetry-break under competition"
                )
            else:
                _commit(
                    "greedy", greedy_choice, adjust_confidence(1.0),
                    "pre-filter: identical costs, low competition"
                )
            return state

        # Case 2: Hungarian is strictly cheaper — coordination dominates unconditionally
        if cost_diff < 0:
            _commit("hungarian", hungarian_choice, adjust_confidence(1.0), "pre-filter: hungarian strictly cheaper")
            return state

        # Case 3: FIX 3 — cheap coordination under meaningful competition.
        # When hungarian costs less than 10% more AND agents outnumber tasks,
        # the coordination benefit outweighs the small personal sacrifice.
        # This gate runs BEFORE the large-premium and high-uncertainty cases
        # so it only fires when the cost is genuinely small.
        if cost_diff_pct < 10.0 and competition_ratio_agents > 0.5:
            _commit(
                "hungarian", hungarian_choice, adjust_confidence(0.95),
                f"pre-filter: {cost_diff_pct:.0f}% premium, competition_ratio={competition_ratio_agents:.2f} > 0.5"
            )
            return state

        # Case 4: large premium AND high uncertainty
        if cost_diff_pct > 25.0 and belief_infeasibility > 0.7:
            _commit(
                "greedy", greedy_choice, adjust_confidence(0.9),
                f"pre-filter: {cost_diff_pct:.0f}% premium + belief={belief_infeasibility:.2f} > 0.7"
            )
            return state

        # Case 5: near-maximum uncertainty regardless of cost premium
        if belief_infeasibility >= 0.9:
            _commit(
                "greedy", greedy_choice, adjust_confidence(0.85),
                f"pre-filter: belief={belief_infeasibility:.2f} >= 0.9, estimates unreliable"
            )
            return state

        # ------------------------------------------------------------------
        # Rule-based arbiter (ablation): the prompt's DECISION RULES in code
        # ------------------------------------------------------------------
        if self.arbiter == "classifier_tree":
            tree = self._tree(self.classifier_model)
            trend = {"RISING": 1, "STABLE": 0, "FALLING": -1}
            go = tree.predict({"belief": belief_infeasibility, "belief_trend": trend.get(belief_trend, 0),
                               "error_trend": trend.get(error_trend, 0), "competition": competition_ratio})
            policy = "hungarian" if go else "greedy"
            chosen = greedy_choice if policy == "greedy" else hungarian_choice
            _commit(policy, chosen, adjust_confidence(0.8), f"classifier ({self.classifier_model}): {policy}")
            return state

        if self.arbiter == "rule_based":
            policy, label = self._rule_based_policy(
                belief_infeasibility, belief_trend, error_trend, competition_ratio,
                conflict_policy=self.conflict_policy)
            chosen = greedy_choice if policy == "greedy" else hungarian_choice
            _commit(policy, chosen, adjust_confidence(0.8), label)
            return state

        # ------------------------------------------------------------------
        # SLM reasoning — genuine ambiguity only
        # ------------------------------------------------------------------
        slm_prompt_inputs = {
            "greedy_choice": greedy_choice,
            "hungarian_choice": hungarian_choice,
            "cost_diff_label": cost_diff_label,
            "belief_infeasibility": belief_infeasibility,
            "energy_error": energy_error,
            "num_peers": num_peers,
            "num_tasks": num_tasks,
            "competition_ratio": competition_ratio,
            "belief_trend": belief_trend,
            "belief_history": belief_history,
            "error_trend": error_trend,
            "error_history": error_history,
            "competition_trend": competition_trend,
            "competition_history": competition_history,
        }
        slm_prompt_inputs["prompt_version"] = self.prompt_version
        state["slm_prompt_inputs"] = slm_prompt_inputs
        prompt = build_arbitration_prompt(**slm_prompt_inputs)

        # Reached only when the SLM is actually consulted; used to charge inference latency.
        state["slm_attempted"] = True

        MAX_RETRIES = 2
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                state["slm_attempts"] = attempt + 1
                raw = self.drone.slm_generate(prompt, output_schema=arbitration_output_schema(self.prompt_version))
                self.last_slm_usage = self._usage_of(prompt, raw, getattr(self.drone, "slm_last_usage", None))

                if not isinstance(raw, str):
                    raise RuntimeError(
                        f"[SPPDEC][{agent_id}] SLM returned non-text response: {type(raw)}"
                    )

                result = self.extract_first_json(raw)

                # --------------------------------------------------
                # FIX 1: Invalid / missing JSON → immediate fallback
                # --------------------------------------------------
                if not result or "policy" not in result:
                    print(f"[SLM-FALLBACK] Agent {agent_id} invalid SLM output → greedy fallback")

                    adjusted = adjust_confidence(0.3)

                    state["selected_task"] = greedy_choice
                    state["decision_mode"] = "candidate"
                    state["slm_status"] = "failed"
                    state["slm_policy"] = "greedy_fallback"
                    state["slm_confidence"] = adjusted
                    _record_failed("invalid output: no policy in reply", attempt + 1)

                    return state

                policy = result.get("policy")
                confidence = result.get("confidence", 0.5)

                # --------------------------------------------------
                # FIX 2: Invalid policy → fallback (NOT retry)
                # --------------------------------------------------
                if policy not in ("greedy", "hungarian"):
                    print(f"[SLM-FALLBACK] Agent {agent_id} invalid policy '{policy}' → greedy fallback")

                    adjusted = adjust_confidence(0.3)

                    state["selected_task"] = greedy_choice
                    state["decision_mode"] = "candidate"
                    state["slm_status"] = "failed"
                    state["slm_policy"] = "greedy_fallback"
                    state["slm_confidence"] = adjusted
                    _record_failed(f"invalid policy: {policy}", attempt + 1)

                    return state

                # --------------------------------------------------
                # Normal SLM success path (RESTORED)
                # --------------------------------------------------
                try:
                    confidence = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError):
                    confidence = 0.5

                if self.prompt_version == "judgement":
                    result = dict(result)
                    result["policy"] = "greedy" if result.get("verdict") == "G" else "hungarian"
                    result.setdefault("confidence", 0.8)  # same confidence as the rule-based arm
                    check = check_judgement_answer(result, belief_infeasibility, belief_trend,
                                                   error_trend, competition_ratio)
                else:
                    check = check_arbitration_answer(result, belief_infeasibility, belief_trend,
                                                     error_trend, competition_ratio)
                check["answer_policy"] = policy
                gate_label = "slm"
                if check["violations"] and self.enforce_arbitration:
                    policy = check["expected_policy"]  # act on the rule result, not on a contradicting answer
                    gate_label = "slm_overridden"
                    check["overridden_to"] = policy
                    print(f"[SLM-GUARD] Agent {agent_id}: {'; '.join(check['violations'])} -> using rules: {policy}")

                chosen_task = greedy_choice if policy == "greedy" else hungarian_choice
                adjusted = adjust_confidence(confidence)

                _commit(policy, chosen_task, adjusted, gate_label, check=check)

                logger.info(
                    "[SPPDEC][%s] SLM policy=%s, task=%s, cost=%.2f, confidence=%.2f",
                    agent_id, policy, chosen_task['parcel_id'], chosen_task['cost'], adjusted,
                )
                print(f"  Raw SLM output: {raw[:200]}")

                return state

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[SPPDEC][%s] SLM attempt %d failed: %s",
                    agent_id,
                    attempt + 1,
                    str(exc),
                )
        # ------------------------------------------------------------------
        # Fallback on complete SLM failure
        # ------------------------------------------------------------------
        logger.error(
            "[SPPDEC][%s] SLM failed after retries. Using greedy fallback. Last error: %s",
            agent_id,
            last_error,
        )

        adjusted_fallback = adjust_confidence(0.3)
        state["selected_task"] = greedy_choice
        state["decision_mode"] = "candidate"
        state["slm_confidence"] = adjusted_fallback
        state["slm_status"] = "failed"
        state["slm_policy"] = "greedy_fallback"
        _record_failed(f"no usable reply after retries: {last_error}", MAX_RETRIES + 1)

        return state

    # ------------------------------------------------------------------
    # Context arbiters
    # ------------------------------------------------------------------
    def _arbiter_context(self, state, facts: Dict) -> Dict:
        """
        The one context every context arbiter reads, rules and SLM alike: the decision-time facts, the
        scenario descriptors (sensing, noise, battery, rate spread, how well each model has predicted the
        peers, the self-check facts), the regime and sensing class derived from them, and what each model
        proposes. A rule uses what it can; the SLM is shown all of it.
        """
        sc = state.get("scenario") or {}
        c = sc.get("fleet_size", 1) / max(1, sc.get("parcels_total", 1))
        regime = "task_rich" if c < 0.3 else "balanced" if c < 1.0 else "agent_surplus"
        if sc.get("sensing_radius") is None and not sc.get("sensor_noise"):
            sensing = "full"
        else:
            sensing = "noisy" if sc.get("sensor_noise") else "partial"
        greedy = state["model_choices"]["greedy"]
        proposals = {}
        for m, t in state["model_choices"].items():
            proposals[m] = None if t is None else {
                "parcel": t["parcel_id"], "cost": round(t["cost"], 3), "same_as_greedy": t["parcel_id"] == greedy["parcel_id"]}
        # The drone's own energy condition, from its learned own-energy model (a drone cannot tell "mixed" from
        # inside: it only knows its own costs, so that label is for the SLM's table and is never detected).
        own = sc.get("self_check") or {}
        cost, wcost = own.get("cost_vs_nominal"), own.get("weight_cost_vs_nominal")
        if cost is None:
            condition = "none"
        elif own.get("shift_alarms"):
            condition = "shift"
        elif cost >= 1.2:
            condition = "rate"
        elif wcost is not None and wcost >= 2.0:
            condition = "weight"
        else:
            condition = "none"
        ctx = {"decision": facts, "scenario": sc, "regime": regime, "sensing_class": sensing, "condition": condition,
               "agents_per_parcel": round(c, 3), "proposals": proposals}
        state["arbiter_context"] = ctx
        return ctx

    def _world_view(self, state) -> str:
        """
        The world as this drone believes it, for the free SLM: its own body (position, energy, what a cell and a unit of
        weight cost it), the terrain (grid, station), every open task with its geometry (distance, energy margin, which
        peers are nearer, which models propose it) and each peer (believed position, distance, energy used, the task it
        is believed to hold). Everything comes from the drone's own perception, memory and declared specs; distances
        are in cells (grid steps) and energy is in battery units.
        """
        snap = state["snapshot"]
        agent = snap["agent"]
        me = (agent["col"], agent["row"])
        drone = self.drone
        station = snap.get("nearest_station")
        grid = self._grid_bounds()
        sc = ((station["col"] + station["w"] // 2, station["row"] + station["h"] // 2) if station else me)
        usable = drone.power.level - float(drone.power.floor)
        carrying = snap.get("carrying", {})
        own_cell = self._energy(self.agent_id, 1.0, 0.0)
        own_weight = self._energy(self.agent_id, 1.0, 1.0) - own_cell
        lines = ["WORLD AS THIS DRONE BELIEVES IT (cells are grid steps; energy is in battery units)"]
        lines.append(f"THIS DRONE ({self.agent_id}) at column {me[0]}, row {me[1]}; battery {drone.power.level:.1f} of "
                     f"{drone.power.capacity:.0f}, {usable:.1f} usable above the reserve floor {float(drone.power.floor):.1f}; "
                     + (f"carrying a parcel of weight {carrying.get('weight', 0):.2f}; " if carrying.get("has") else "carrying nothing; ")
                     + f"it believes a cell costs it {own_cell:.3f} and each unit of carried weight adds {own_weight:.3f} per cell.")
        lines.append(f"TERRAIN: open grid {grid[0]} columns by {grid[1]} rows, no obstacles; "
                     + (f"delivery station {station['w']}x{station['h']} cells with its top-left at column {station['col']}, row {station['row']} "
                        f"(centre {sc[0]}, {sc[1]}), {self.physics.distance(me, sc):.0f} cells from this drone." if station else "station unknown."))
        owners = self._territory_owners() if self.territory_hint else {}
        if self.territory_hint in ("raw", "both") and self._start_info is not None:
            lines.append("START CELLS of every drone (known to all): " + "; ".join(
                f"{pid} at column {spec.start_cell[0]}, row {spec.start_cell[1]}" for pid, spec in sorted(self._start_info.fleet.items())))
        peers = []
        for pid, b in sorted(self.peer_beliefs.items()):
            pos = b.get("smooth_pos") or b["predicted_position"]
            pos = (int(round(pos[0])), int(round(pos[1])))
            spec = drone.declared_spec(pid) if hasattr(drone, "declared_spec") else None
            a = b.get("assumed") or {}
            peers.append((pid, pos, b, a, spec))
        lines.append("OPEN TASKS this drone can afford (distance and energy are this drone's; 'nearer peers' lists peers believed closer to the parcel):")
        proposed = {}
        for m, t in state["model_choices"].items():
            if t is not None:
                proposed.setdefault(t["parcel_id"], []).append(m)
        for t in sorted(state["feasible_tasks"], key=lambda x: self.physics.distance(me, x["pickup"])):
            d = self.physics.distance(me, t["pickup"])
            trip = self._trip_energy(self.agent_id, me, t["pickup"], sc, t["weight"])
            nearer = sorted(((self.physics.distance(pos, t["pickup"]), pid) for pid, pos, _b, _a, _s in peers
                             if self.physics.distance(pos, t["pickup"]) < d))
            claimed = [pid for pid, _pos, _b, a, _s in peers if a.get("task") == t["parcel_id"]]
            lines.append(f"  parcel {t['parcel_id']}: at column {t['pickup'][0]}, row {t['pickup'][1]}, weight {t['weight']:.2f}, "
                         f"{d:.0f} cells away, whole trip {trip:.1f} energy (margin {usable - trip:.1f}), "
                         f"to the station {self.physics.distance(t['pickup'], sc):.0f} cells; "
                         f"nearer peers: {', '.join(f'{p} ({x:.0f})' for x, p in nearer[:4]) or 'none'}"
                         + (f"; believed taken by {', '.join(claimed)}" if claimed else "")
                         + (f"; proposed by {', '.join(proposed[t['parcel_id']])}" if t["parcel_id"] in proposed else "")
                         + (f"; territory: {'yours' if owners.get(t['parcel_id']) == self.agent_id else owners.get(t['parcel_id'], 'unknown')}"
                            if self.territory_hint in ("labels", "both") else ""))
        lines.append("PEERS (believed positions; energy used so far is what this drone has inferred):")
        for pid, pos, b, a, spec in peers:
            lines.append(f"  {pid}: at column {pos[0]}, row {pos[1]}, {self.physics.distance(me, pos):.0f} cells from this drone, "
                         f"{'in sight' if b.get('is_visible') else 'out of sight for ' + str(b.get('time_since_last_seen', '?')) + ' steps'}, "
                         f"energy used about {b.get('energy_used', 0.0):.1f}"
                         + (f" of {spec.capacity:.0f}" if spec is not None else "")
                         + (f", believed heading for parcel {a['task']}" if a.get("task") else ", no believed task"))
        return "\n".join(lines)

    def _condition_table(self) -> Dict:
        """What the drone's own energy conditions do to each model (condition_table.json); empty if absent."""
        if not hasattr(self, "_cond_table"):
            from pathlib import Path
            path = Path(__file__).with_name("condition_table.json")
            self._cond_table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return self._cond_table

    def _scenario_table_cells(self) -> Dict:
        if not hasattr(self, "_table_cells"):
            from pathlib import Path
            path = Path(__file__).with_name("scenario_table.json")
            self._table_cells = json.loads(path.read_text(encoding="utf-8"))["all"]
        return self._table_cells

    @staticmethod
    def _table_model_name(key: str) -> str:
        return {"cbba_bid_rounds": "cbba_rounds"}.get(key, key)

    def _context_rules(self, ctx: Dict, available) -> Tuple[str, str]:
        """
        Rule list from the scenario table: in this kind of situation (sensing class, load regime) the table
        says which reasoning model finishes the mission soonest. Take the best model that has a proposal; if
        it has none (no slot), the next best. Facts the table has no evidence for are not used by the rules.
        """
        cell = self._scenario_table_cells().get(f"{ctx['sensing_class']}|{ctx['regime']}")
        if not cell:
            return "greedy", "rule: no table entry, nearest task"
        order = sorted(cell["relative_to_best"].items(), key=lambda kv: kv[1])
        pick, label = None, None
        for key, rel in order:
            name = self._table_model_name(key)
            if name in available:
                pick, label = name, f"rule: {ctx['sensing_class']}|{ctx['regime']} -> {name} (table {rel:.3f} of best)"
                break
        if pick is None:
            return "greedy", "rule: no coordination model had a proposal"
        # Second table: the drone's own energy condition. Depart from the situation table's choice only when the
        # condition table shows a clear gain: at least 3 points more parcels delivered, or 0.3 fewer drones lost
        # per run (both are noisy at this sample size, so smaller differences are ignored).
        ev = self._condition_table().get(ctx.get("condition"))
        if ev:
            ms = {m: c for m, c in ev["models"].items() if m in available}
            if pick in ms and len(ms) > 1:
                best = min(ms, key=lambda m: (-ms[m]["delivered_share"], ms[m]["drones_lost_per_run"]))
                gain_delivered = ms[best]["delivered_share"] - ms[pick]["delivered_share"]
                gain_lost = ms[pick]["drones_lost_per_run"] - ms[best]["drones_lost_per_run"]
                if best != pick and (gain_delivered >= 0.03 or gain_lost >= 0.3):
                    return best, (f"rule: condition {ctx['condition']} -> {best} (condition table: delivered "
                                  f"{ms[best]['delivered_share']:.2f} vs {ms[pick]['delivered_share']:.2f}, drones lost per run "
                                  f"{ms[best]['drones_lost_per_run']:.2f} vs {ms[pick]['drones_lost_per_run']:.2f})")
        return pick, label

    def _decide_with_context(self, state, facts: Dict):
        """Choose a reasoning model from the shared context. Returns (model, label, check)."""
        ctx = self._arbiter_context(state, facts)
        available = [m for m, t in state["model_choices"].items() if t is not None]
        if self.arbiter == "context_rules":
            model, label = self._context_rules(ctx, available)
            if self.rule_perturb > 0.0 and len({state["model_choices"][m]["parcel_id"] for m in available}) > 1:
                if self._perturb_rng is None:
                    import random
                    self._perturb_rng = random.Random(f"perturb|{self.agent_id}")
                if self._perturb_rng.random() < self.rule_perturb:
                    model = self._perturb_rng.choice(available)
                    label = f"rule list perturbed: random model {model} ({label})"
            return model, label, None
        # context_slm / context_slm_free
        self.free_slm = self.arbiter == "context_slm_free"
        rule_model, rule_label = self._context_rules(ctx, available)
        proposed = {state["model_choices"][m]["parcel_id"] for m in available}
        if len(proposed) == 1:  # every model names the same parcel: nothing to arbitrate
            return rule_model, "all models agree", None
        table_cells = self._scenario_table_cells()
        world = self._world_view(state) if (self.free_slm and self.world_view) else None
        prompt = (build_free_context_prompt(ctx, table_cells, available, self._condition_table(),
                                            suggest=self.slm_suggest, world=world, plan_ahead=self.plan_ahead) if self.free_slm
                  else build_context_prompt(ctx, table_cells, available, self._condition_table()))
        state["slm_prompt_inputs"] = {"prompt": prompt, "context": ctx, "prompt_version": "context"}
        state["slm_attempted"] = True
        last_error = None
        for attempt in range(3):
            state["slm_attempts"] = attempt + 1
            try:
                open_ids = [t["parcel_id"] for t in state["feasible_tasks"]]
                if self.free_slm:
                    props = dict(FREE_SCHEMA["properties"], model={"type": "string", "enum": list(available)})
                    schema = dict(FREE_SCHEMA, properties=props)
                    if self.plan_ahead:
                        props["plan"] = {"type": "array", "items": {"type": "string", "enum": open_ids}}
                        schema["required"] = list(FREE_SCHEMA["required"]) + ["plan"]
                else:
                    schema = context_schema(ctx, table_cells, available)
                raw = (self._cached_slm(prompt, None, num_predict=12000, think=True) if (self.free_slm and self.slm_think)
                       else self._cached_slm(prompt, schema, num_predict=512 if self.free_slm else 256))
                answer = self._parse_answer_object(raw)
                check = check_context_answer(answer, ctx, table_cells, available, strict=not self.free_slm)
                check["raw_model"] = answer.get("model")
                if self.free_slm and self.plan_ahead:
                    head_id = state["model_choices"][answer["model"]]["parcel_id"] if answer.get("model") in available else None
                    listed = answer.get("plan") or []
                    state["slm_bundle"] = [p for i, p in enumerate(listed)
                                           if p in open_ids and p != head_id and p not in listed[:i]]
                    print(f"[SPPDEC][{self.agent_id}] Plan ahead: the model listed {len(listed)} tasks, "
                          f"{len(state['slm_bundle'])} kept after the first")
                if check["violations"]:
                    check["overridden_to"] = rule_model
                    print(f"[SLM-GUARD] {self.agent_id}: {'; '.join(check['violations'])} -> rule list: {rule_model}")
                    return rule_model, "slm_overridden: " + rule_label, check
                return answer["model"], f"slm: {answer['model']} ({'departs from the tables' if check['deviated'] else 'table best'})", check
            except Exception as exc:  # no usable answer: keep the rule list's choice and say so
                last_error = exc
        return rule_model, f"slm_failed ({last_error}): " + rule_label, {"violations": ["no usable answer"], "error": str(last_error)}

    def _decide_task(self, state, facts: Dict):
        """
        Task-level SLM arbiter: the SLM names the parcel this drone takes. Returns (task, label, check). When all four
        reasoning models propose the same parcel there is nothing to arbitrate (no model call). An answer that is not an open,
        affordable task is replaced by the rule list's choice.
        """
        ctx = self._arbiter_context(state, facts)
        available = [m for m, t in state["model_choices"].items() if t is not None]
        rule_model, rule_label = self._context_rules(ctx, available)
        rule_task = state["model_choices"][rule_model]
        proposers: Dict[str, List[str]] = {}
        for m, t in state["model_choices"].items():
            if t is not None:
                proposers.setdefault(t["parcel_id"], []).append(m)
        if len(proposers) == 1 and not self.territory_hint:
            return rule_task, "all models agree", None
        by_id = {t["parcel_id"]: t for t in state["feasible_tasks"]}
        owners = self._territory_owners()
        own_left = sum(1 for t in state["feasible_tasks"] if owners.get(t["parcel_id"]) == self.agent_id)
        prompt = build_task_prompt(ctx, self._world_view(state), available,
                                   TERRITORY_CONVENTION.get(self.territory_hint) if self.territory_hint else None)
        state["slm_prompt_inputs"] = {"prompt": prompt, "context": ctx, "prompt_version": "task"}
        state["slm_attempted"] = True
        last_error = None
        for attempt in range(3):
            state["slm_attempts"] = attempt + 1
            try:
                schema = dict(TASK_SCHEMA, properties=dict(TASK_SCHEMA["properties"], parcel={"type": "string", "enum": list(by_id)}))
                raw = (self._cached_slm(prompt, None, num_predict=12000, think=True) if self.slm_think
                       else self._cached_slm(prompt, schema, num_predict=512))
                answer = self._parse_answer_object(raw)
                pid = str(answer.get("parcel") or "").strip()
                if pid not in by_id:
                    last_error = f"parcel '{pid}' is not an open, affordable task"
                    continue
                by = proposers.get(pid, [])
                check = {"violations": [], "warnings": [], "answer_parcel": pid, "proposed_by": by,
                         "rule_model": rule_model, "agrees_with_rule": pid == rule_task["parcel_id"],
                         # adherence to the territory convention (recorded for every task-level arm)
                         "owner": owners.get(pid), "own_territory_left": own_left, "in_own_territory": owners.get(pid) == self.agent_id,
                         "deviation_from": answer.get("weighing"), "reason": (answer.get("reason") or "").strip()}
                label = f"slm task: {pid} (proposed by {', '.join(by) if by else 'no model'}" + \
                        ("" if check["agrees_with_rule"] else "; departs from the rule list") + ")"
                return by_id[pid], label, check
            except Exception as exc:
                last_error = exc
        return rule_task, f"slm_failed ({last_error}): " + rule_label, {"violations": ["no usable answer"], "error": str(last_error)}

    @staticmethod
    def _parse_answer_object(raw: str) -> dict:
        """The answer object in the reply: the last JSON object that parses (a free-text reply, as from thinking mode, may
        hold braces before its answer)."""
        dec = json.JSONDecoder()
        found = None
        for i, ch in enumerate(raw):
            if ch == "{":
                try:
                    obj, _ = dec.raw_decode(raw[i:])
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    found = obj
        if found is None:
            raise ValueError("no JSON object in the reply")
        return found

    _SLM_CACHE: Dict[str, str] = {}

    def _cached_slm(self, prompt: str, schema: dict, num_predict: int = 256, think: bool = False) -> str:
        """The model decodes deterministically, so an answer is reusable: the same prompt gets the same answer
        (cache shared by all drones in the process, optionally persisted in the file named by SPPDEC_SLM_CACHE)."""
        import hashlib
        key = hashlib.sha256((str(getattr(self.drone, "slm_model", "")) + ("|think|" if think else "") + prompt).encode("utf-8")).hexdigest()
        path = os.environ.get("SPPDEC_SLM_CACHE")
        if path and not SPPDECLangGraph._SLM_CACHE and os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    SPPDECLangGraph._SLM_CACHE[rec["k"]] = (rec["v"], rec.get("u"))
                except ValueError:
                    pass
        if key in SPPDECLangGraph._SLM_CACHE:
            raw, usage = SPPDECLangGraph._SLM_CACHE[key]
            self.last_slm_usage = self._usage_of(prompt, raw, usage)
            return raw
        raw = (self.drone.slm_generate(prompt, output_schema=None, num_predict=num_predict, think=True, timeout=600)
               if think else self.drone.slm_generate(prompt, output_schema=schema, num_predict=num_predict))
        if not isinstance(raw, str):
            raise RuntimeError(f"SLM returned non-text response: {type(raw)}")
        usage = getattr(self.drone, "slm_last_usage", None)
        SPPDECLangGraph._SLM_CACHE[key] = (raw, usage)
        self.last_slm_usage = self._usage_of(prompt, raw, usage)
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": key, "v": raw, "u": usage}) + "\n")
        return raw

    @staticmethod
    def _usage_of(prompt: str, raw: str, usage) -> Dict[str, int]:
        """Tokens read and written by one model call (as the server counted them; estimated for old cache entries)."""
        from src.hardware import estimate_tokens
        usage = usage or {}
        return {"prompt_tokens": usage.get("prompt_tokens") or estimate_tokens(prompt),
                "gen_tokens": usage.get("gen_tokens") or estimate_tokens(raw)}

    @staticmethod
    def _rule_based_policy(belief_infeasibility: float, belief_trend: str, error_trend: str,
                           competition_ratio: float, conflict_policy: str = "trend") -> Tuple[str, str]:
        """
        Deterministic version of the SLM prompt's DECISION RULES (Steps 2-3).

        Reached only for cases the pre-filter leaves ambiguous (Hungarian costs more,
        not identical, belief < 0.9). Matches prompt rules v2: the premium never
        decides on its own, so every case goes to Step 2.
        """
        if conflict_policy == "competition":
            f = judgement_facts(belief_infeasibility, belief_trend, error_trend, competition_ratio)
            if f["conflict"]:
                return ("greedy" if f["competition_lean"] == "G" else "hungarian",
                        "rule: conflict, competition wins")
        if belief_infeasibility > 0.7:
            return "greedy", "rule: belief > 0.7"
        if belief_infeasibility < 0.3:
            return "hungarian", "rule: belief < 0.3"
        if belief_trend == "FALLING" and error_trend == "FALLING":
            return "hungarian", "rule: uncertainty and error falling"
        if belief_trend == "RISING" or error_trend == "RISING":
            return "greedy", "rule: uncertainty or error rising"
        if competition_ratio > 0.15:
            return "hungarian", "rule: stable, competition > 0.15"
        return "greedy", "rule: stable, competition <= 0.15"

    def decide_mode(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Simplified decision mode resolution that respects the new architecture.

        SLM or fallback already selected a task directly, so just proceed.
        """
        if state.get("decision_mode") in ("idle", "look"):
            return state

        agent_id = state["snapshot"]["agent"]["id"]

        # If SLM or fallback already selected a task, proceed
        if state.get("selected_task"):
            state["decision_mode"] = "candidate"
            print(
                f"[SPPDEC][{agent_id}] Task selected -> CANDIDATE"
            )
            return state

        # If SLM abstained, wait
        if state.get("slm_status") == "abstain":
            state["decision_mode"] = "wait"
            print(f"[SPPDEC][{agent_id}] No decision -> WAIT")
            return state

        raise RuntimeError(f"[SPPDEC][{agent_id}] Invalid decision state")

    def select_drop_cell(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Select a concrete station cell for dropoff.

        Selection uses real-world perception by querying terrain occupancy.
        """
        self._init_trace(state, f"select_drop_cell({state.get('decision_mode')})")

        if state.get("decision_mode") != "candidate":
            return state

        snap = state["snapshot"]
        task = state.get("selected_task")

        if not task:
            state["decision_mode"] = "wait"
            state["wait_reason"] = "no selected task"
            agent_id = snap["agent"]["id"]
            print(f"[SPPDEC][{agent_id}] No task -> WAIT")
            return state

        station = snap.get("nearest_station")
        if not station:
            state["decision_mode"] = "wait"
            state["wait_reason"] = "no station in snapshot"
            return state

        # The station's free cells as the drone senses them on site; if all are blocked, any station cell
        free_cells = station.get("free_cells") or [
            (c, r)
            for r in range(station["row"], station["row"] + station["h"])
            for c in range(station["col"], station["col"] + station["w"])
        ]

        # The same rule as every strategy: the free station cell nearest to the pickup
        best_cell = min(
            free_cells,
            key=lambda c: self.physics.distance(task["pickup"], c),
        )

        if self.consensus:
            planned = self._drop_plan.get(task["parcel_id"])
            if planned in free_cells:
                best_cell = planned
        # Assign dropoff
        task["dropoff"] = best_cell
        task["task_type"] = "executable"

        # Compute predicted cost: the whole trip to the chosen cell
        total_predicted_cost = self._trip_energy(
            self.agent_id, (snap["agent"]["col"], snap["agent"]["row"]), task["pickup"], best_cell, task["weight"])

        state["predicted_cost"] = total_predicted_cost
        self._last_predicted_cost = total_predicted_cost

        state["decision_mode"] = "solo"

        agent_id = snap["agent"]["id"]
        print(
            f"[SPPDEC][{agent_id}] Drop cell selected from the station's free cells: {best_cell}, "
            f"predicted_cost={total_predicted_cost:.2f}"
        )

        return state

    def emit_directive(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Emit a controller-compatible execution directive.
        """
        self._init_trace(state, "emit_directive")

        task = state.get("selected_task")
        if state.get("decision_mode") == "look":
            state["directive"] = {"mode": "look", "reason": "see the station before deciding", "cell": self._look_cell(state["snapshot"])}
            return state
        if task and task.get("task_type") == "executable":
            state["directive"] = build_execution_directive(
                task,
                confidence=state.get("slm_confidence", 0.5),
            )
            for extra in state.get("consensus_bundle") or []:
                state["directive"]["plan"]["plan"].append({
                    "pickup": extra["pickup"], "dropoff": self._drop_plan.get(extra["parcel_id"]) or self._drop_for(extra["pickup"], state["snapshot"]),
                    "weight": extra.get("weight", 1.0)})
        self._record_decision_action(state)
        return state

    def _record_decision_action(self, state: SPPDECAgentState):
        """
        Link the decision to the directive handed to the controller: the chosen parcel, the
        pickup cell the directive sends the drone to, and whether they agree. If a decision
        did not become a plan, say why. The drone's pick actions complete the chain
        (see analysis/revision/decision_action_audit.py).
        """
        if self.flight_recorder is None:
            return
        snap = state["snapshot"]
        task = state.get("selected_task")
        directive = state.get("directive") or {}
        plan = (directive.get("plan") or {}).get("plan") or []
        directive_cell = list(plan[0]["pickup"]) if plan else None
        decided_cell = list(task["pickup"]) if task else None
        if task is None:
            reason = "no task was selected"
        elif not plan:
            reason = state.get("wait_reason") or "task was not executable"
        elif directive_cell != decided_cell:
            reason = "directive pickup differs from the decided task"
        else:
            reason = None
        self.flight_recorder.record_decision_action(
            agent_id=snap["agent"]["id"],
            step=snap.get("step", 0),
            sim_time=self.current_step * (1.0 / 60.0),
            policy=state.get("slm_policy"),
            gate=state.get("slm_gate"),
            decided_parcel=task["parcel_id"] if task else None,
            decided_cell=decided_cell,
            directive_mode="plan" if plan else "wait",
            directive_cell=directive_cell,
            reason_not_acted=reason,
        )

    def finalize(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Finalize the decision cycle.

        Guarantees that a directive is always returned.
        """
        self._init_trace(state, "finalize")

        if state.get("directive") is not None:
            return state

        if state.get("decision_mode") == "idle":
            state["directive"] = {
                "mode": "idle",
                "reason": "agent_declares_infeasible",
            }

        else:
            state["directive"] = {"mode": "wait"}

        self._final_trace(state)
        return state

    def _route_after_decision(self, state: SPPDECAgentState) -> str:
        """
        Determine routing after dropoff selection.
        """
        return state.get("decision_mode", "wait")

    def _init_trace(self, state: SPPDECAgentState, node: str):
        agent_id = state["snapshot"]["agent"]["id"]

        if "cycle_id" not in state:
            state["cycle_id"] = state["snapshot"].get("step", 0)

        if "trace" not in state:
            state["trace"] = []

        state["trace"].append(node)

        print(f"[TRACE][{agent_id}][cycle={state['cycle_id']}] enter {node}")

    def _final_trace(self, state: SPPDECAgentState):
        agent_id = state["snapshot"]["agent"]["id"]
        directive = state.get("directive", {})
        mode = directive.get("mode", "none")
        reason = directive.get("reason")

        path = " -> ".join(state.get("trace", []))

        if reason:
            print(f"[TRACE][{agent_id}][cycle={state['cycle_id']}] PATH: {path} | final: {mode} ({reason})")
        else:
            print(f"[TRACE][{agent_id}][cycle={state['cycle_id']}] PATH: {path} | final: {mode}")
