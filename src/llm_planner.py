# src/llm_planner.py
"""
Planner client that returns a full delivery plan (sequence of pickups and delivery targets)
and a short human-readable narration explaining the plan.

This file contains two main parts:

1) A simple Ollama client:
   - `generate(prompt, model="phi3:mini", ...)` that talks to a local Ollama server.
   - Useful for quick, local LLM queries (TinyLLMs etc).

2) A delivery PlannerClient:
   - If `use_llm` is True, it uses an LLM backend (OpenAI via LangChain or Ollama)
     to propose a plan, optionally running several trials and picking the best.
   - Otherwise it falls back to a deterministic local planner that respects
     battery constraints.

Local planner supports two search modes:
   - "greedy"  : single nearest-first greedy pass.
   - "multi"   : runs several greedy simulations with different orderings and picks
                 the best plan based on:
                    1) maximum parcels picked
                    2) then maximum remaining battery

LLM planner:
   - Runs `num_trials` calls to the chosen LLM backend.
   - Each trial returns a candidate plan JSON.
   - Each candidate is scored using:
        1) picked_count = len(plan)
        2) remaining_battery = estimated using the same cost model
   - The best plan by this metric is returned with strategy "llm_openai" or "llm_ollama".

Plan format (for callers):
{
  "plan": [
     {"pickup": [col,row], "dropoff": [col,row], "weight": 1.0 },
     ...
  ],
  "confidence": 0.8,
  "narration": "Short summary sentence(s)...",
  "strategy": "greedy" or "multi" or "llm_openai" or "llm_ollama"
}
"""

from __future__ import annotations

import json
import random
import time
from typing import Dict, Any, List, Optional, Tuple

import requests

from artifacts import Drone  # for BASE_COST_PER_CELL and WEIGHT_FACTOR


# ---------------------------------------------------------------------------
# 1. Simple Ollama client (local TinyLLM, for example phi3:mini)
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "phi3:mini"


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call a local Ollama model and return the raw text response.

    This uses Ollama's /api/generate endpoint:
      - `model` is something like "phi3:mini", "tinyllama", etc.
      - `prompt` is plain text.
      - If `stream` is False, returns the complete response string.
      - If `stream` is True, streams the response and concatenates chunks.

    Note:
      This does not use LangChain. It is a bare HTTP client you can use
      anywhere in your project for local LLM calls.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }

    if options:
        payload["options"] = options

    url = f"{OLLAMA_URL}/api/generate"

    if not stream:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")

    response = requests.post(url, json=payload, stream=True, timeout=60)
    response.raise_for_status()

    chunks: List[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        data = requests.utils.json.loads(line)
        chunks.append(data.get("response", ""))
    return "".join(chunks)


# ---------------------------------------------------------------------------
# 2. PlannerClient with optional LLM backend and local fallback
# ---------------------------------------------------------------------------

# Try to import LangChain / OpenAI. If missing, we will fall back.
try:
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    from langchain.llms import OpenAI  # Classic LangChain OpenAI LLM

    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK = False
    LLMChain = None  # type: ignore
    PromptTemplate = None  # type: ignore
    OpenAI = None  # type: ignore


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _est_cost_cells(dist_cells: float, weight: float) -> float:
    """
    Simple battery cost estimate that mirrors the Drone energy model.
    """
    return (
        dist_cells
        * Drone.BASE_COST_PER_CELL
        * (1.0 + weight * Drone.WEIGHT_FACTOR)
    )


def _simulate_greedy_plan(
    snapshot: Dict[str, Any],
    max_items: int,
    parcel_order: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run a single greedy simulation from the current snapshot.

    - If `parcel_order` is provided, it is used as the initial list of parcels.
      The greedy step then sorts by distance and weight at each step.
    - Returns a dict with:
        {
          "plan": [...],
          "remaining_battery": float,
          "picked_count": int,
        }

    This version also chooses a concrete station cell (not just station center)
    for each drop, using a simple round-robin over the station footprint.
    """
    agent = snapshot.get("agent", {})
    battery = float(agent.get("battery_pct", 0.0))
    start = (agent.get("col", 0), agent.get("row", 0))
    station = snapshot.get("nearest_station", None)

    # Prepare list of station cells (if a station exists)
    station_cells: List[Tuple[int, int]] = []
    if station:
        s_col = int(station["col"])
        s_row = int(station["row"])
        s_w = int(station["w"])
        s_h = int(station["h"])
        for r in range(s_row, s_row + s_h):
            for c in range(s_col, s_col + s_w):
                station_cells.append((c, r))

    # Prepare parcels
    if parcel_order is not None:
        parcels = [
            dict(p)
            for p in parcel_order
            if not p.get("picked", False) and not p.get("delivered", False)
        ]
    else:
        parcels = [
            dict(p)
            for p in snapshot.get("all_parcels", [])
            if not p.get("picked", False) and not p.get("delivered", False)
        ]

    plan: List[Dict[str, Any]] = []
    cur_pos = start
    remaining_batt = battery

    # index into station_cells so we do not always use the same cell
    drop_slot_idx = 0

    attempts = 0
    while parcels and len(plan) < max_items and attempts < 200:
        # sort by distance then weight from current position
        parcels.sort(
            key=lambda p: (
                _manhattan(cur_pos, (p["col"], p["row"])),
                p.get("weight", 1.0),
            )
        )

        chosen = None
        chosen_drop_cell: Tuple[int, int]

        for p in parcels:
            pick_cell = (int(p["col"]), int(p["row"]))

            # Decide which station cell this step will drop to
            if station_cells:
                chosen_drop_cell = station_cells[drop_slot_idx % len(station_cells)]
            elif station:
                # Fallback to approximate station center if something is odd
                chosen_drop_cell = (
                    station["col"] + station["w"] // 2,
                    station["row"] + station["h"] // 2,
                )
            else:
                # No station, drop where we picked
                chosen_drop_cell = pick_cell

            dist_to_pick = _manhattan(cur_pos, pick_cell)
            dist_pick_to_drop = _manhattan(pick_cell, chosen_drop_cell)

            need = _est_cost_cells(dist_to_pick, 0.0) + _est_cost_cells(
                dist_pick_to_drop, p.get("weight", 1.0)
            )

            # keep a small safety margin
            if remaining_batt >= need + 3:
                chosen = p
                break

        if not chosen:
            break

        # We commit the selected drop cell for this step
        if station_cells:
            drop_cell = station_cells[drop_slot_idx % len(station_cells)]
            drop_slot_idx += 1
        elif station:
            drop_cell = (
                station["col"] + station["w"] // 2,
                station["row"] + station["h"] // 2,
            )
        else:
            drop_cell = (chosen["col"], chosen["row"])

        plan.append(
            {
                "pickup": [int(chosen["col"]), int(chosen["row"])],
                "dropoff": [int(drop_cell[0]), int(drop_cell[1])],
                "weight": float(chosen.get("weight", 1.0)),
            }
        )

        # Update battery based on this step
        remaining_batt -= _est_cost_cells(
            _manhattan(cur_pos, (chosen["col"], chosen["row"])), 0.0
        )
        remaining_batt -= _est_cost_cells(
            _manhattan((chosen["col"], chosen["row"]), drop_cell),
            chosen.get("weight", 1.0),
        )

        cur_pos = drop_cell
        parcels.remove(chosen)
        attempts += 1

    result = {
        "plan": plan,
        "remaining_battery": remaining_batt,
        "picked_count": len(plan),
    }
    return result


def _choose_best_simulation(
    sims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Pick the best simulation based on:
      1) maximum picked_count
      2) then maximum remaining_battery
    """
    best = None
    for sim in sims:
        if best is None:
            best = sim
            continue
        if sim["picked_count"] > best["picked_count"]:
            best = sim
        elif sim["picked_count"] == best["picked_count"] and sim[
            "remaining_battery"
        ] > best["remaining_battery"]:
            best = sim
    return best or {"plan": [], "remaining_battery": 0.0, "picked_count": 0}


def _local_plan_snapshot(
    snapshot: Dict[str, Any],
    max_items: int = 20,
    search_mode: str = "greedy",
    num_trials: int = 8,
) -> Dict[str, Any]:
    """
    Local planner. Does not call any LLM.

    search_mode:
      - "greedy": single greedy simulation.
      - "multi" : multiple greedy simulations with random parcel orderings.
                  Best plan is selected by:
                      1) max picked_count
                      2) then max remaining_battery
    """
    agent = snapshot.get("agent", {})
    battery = float(agent.get("battery_pct", 0.0))
    station = snapshot.get("nearest_station", None)

    # Collect all candidate parcels once
    base_parcels = [
        dict(p)
        for p in snapshot.get("all_parcels", [])
        if not p.get("picked", False) and not p.get("delivered", False)
    ]

    num_trials_run = 0
    best_trial_index = 0

    if search_mode == "multi" and base_parcels:
        simulations: List[Dict[str, Any]] = []

        # Trial 0: deterministic ordering as is
        sim0 = _simulate_greedy_plan(
            snapshot, max_items=max_items, parcel_order=base_parcels
        )
        simulations.append(sim0)

        # Additional random trials
        for _ in range(1, num_trials):
            shuffled = base_parcels[:]
            random.shuffle(shuffled)
            sim = _simulate_greedy_plan(
                snapshot, max_items=max_items, parcel_order=shuffled
            )
            simulations.append(sim)

        # Choose best simulation
        best_sim = _choose_best_simulation(simulations)
        plan = best_sim["plan"]
        remaining_batt = best_sim["remaining_battery"]
        picked_count = best_sim["picked_count"]
        strategy_name = "multi"

        # Instrumentation: how many trials and which index won
        num_trials_run = len(simulations)
        try:
            best_trial_index = simulations.index(best_sim)
        except ValueError:
            best_trial_index = 0

        # Debug logging for all trials and the chosen plan
        try:
            print(
                f"[PLANNER][local-multi] ran {num_trials_run} simulation(s); "
                f"best_trial_index={best_trial_index}"
            )
            for idx, sim in enumerate(simulations):
                plan_steps = [
                    f"{step['pickup'][0]},{step['pickup'][1]}->"
                    f"{step['dropoff'][0]},{step['dropoff'][1]}"
                    for step in sim["plan"]
                ]
                print(
                    f"[PLANNER][local-multi] trial {idx}: "
                    f"picked={sim['picked_count']}, "
                    f"remaining_batt={sim['remaining_battery']:.1f}, "
                    f"plan={plan_steps}"
                )
        except Exception:
            pass

    else:
        # Simple single greedy pass
        sim = _simulate_greedy_plan(snapshot, max_items=max_items)
        plan = sim["plan"]
        remaining_batt = sim["remaining_battery"]
        picked_count = sim["picked_count"]
        strategy_name = "greedy"
        num_trials_run = 1
        best_trial_index = 0

        # Optional debug for greedy mode too
        try:
            plan_steps = [
                f"{step['pickup'][0]},{step['pickup'][1]}->"
                f"{step['dropoff'][0]},{step['dropoff'][1]}"
                for step in plan
            ]
            print(
                "[PLANNER][local-greedy] single simulation: "
                f"picked={picked_count}, remaining_batt={remaining_batt:.1f}, "
                f"plan={plan_steps}"
            )
        except Exception:
            pass

    # Build a concise narration from the plan
    if not plan:
        if battery < 10:
            narration = (
                "No feasible pickups planned. Battery low, "
                "returning to station is recommended."
            )
        else:
            narration = (
                "No suitable pickups found that meet battery constraints. "
                "Idle or reposition to find parcels."
            )
    else:
        # Short human readable view of the plan
        step_summaries = [
            f"({s['pickup'][0]},{s['pickup'][1]})→({s['dropoff'][0]},{s['dropoff'][1]})"
            for s in plan
        ]
        step_preview = ", ".join(step_summaries[:4])

        if station:
            narration = (
                f"Planned {len(plan)} pickup(s) using {strategy_name} search "
                f"over {num_trials_run} simulation(s). "
                f"Best plan steps: {step_preview}. "
                "Will deliver to nearest station to maximize successful returns."
            )
        else:
            narration = (
                f"Planned {len(plan)} pickup(s) using {strategy_name} search "
                f"over {num_trials_run} simulation(s). "
                f"Best plan steps: {step_preview}. "
                "No station available so drops will be local."
            )

    return {
        "plan": plan,
        "confidence": 0.6,
        "narration": narration,
        "created_at": time.time(),
        "strategy": strategy_name,
        "picked_count": picked_count,
        "remaining_battery": remaining_batt,
        "num_trials_run": num_trials_run,
        "best_trial_index": best_trial_index,
    }


# ---------------------------------------------------------------------------
# 3. LLM planning helpers (OpenAI via LangChain or Ollama)
# ---------------------------------------------------------------------------

_LLM_PROMPT_TEXT = (
    "You are a delivery planner. Given this compact snapshot of the world, "
    "return exactly valid JSON with keys:\n"
    " - plan: a list of steps where each step has pickup [col,row], "
    "dropoff [col,row], weight (float)\n"
    " - confidence: float 0..1\n"
    " - narration: a short (1-3 sentence) explanation of why this plan "
    "is efficient for maximizing deliveries given battery constraints.\n"
    "Return JSON only.\n\n"
    "Search mode hint: {search_mode}\n"
    "Snapshot:\n{snapshot}\n\n"
    "Objective: maximize successful deliveries within remaining time "
    "while avoiding lost drones."
)

if LANGCHAIN_OK:
    _PROMPT = PromptTemplate(
        input_variables=["snapshot", "search_mode"],
        template=_LLM_PROMPT_TEXT,
    )
    _CHAIN = None
    # _CHAIN = LLMChain(
    #     llm=OpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=700),
    #     prompt=_PROMPT,
    # )
else:
    _CHAIN = None  # type: ignore


def _parse_llm_json(resp: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a JSON object from an LLM response.
    Tries full string first, then first-to-last brace slice.
    """
    resp = resp.strip()
    if not resp:
        return None
    try:
        return json.loads(resp)
    except Exception:
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(resp[start : end + 1])
            except Exception:
                return None
    return None


def _score_plan_from_snapshot(
    snapshot: Dict[str, Any],
    plan: List[Dict[str, Any]],
) -> Tuple[int, float]:
    """
    Given a snapshot and an LLM plan, estimate:
      - picked_count: number of steps
      - remaining_battery: using the same cost model

    This does not mutate the snapshot. It only simulates movement
    following the plan steps.
    """
    agent = snapshot.get("agent", {})
    battery = float(agent.get("battery_pct", 0.0))
    cur_pos = (agent.get("col", 0), agent.get("row", 0))
    remaining_batt = battery

    for step in plan:
        try:
            pickup = step["pickup"]
            dropoff = step["dropoff"]
            weight = float(step.get("weight", 1.0))
            pick_pos = (int(pickup[0]), int(pickup[1]))
            drop_pos = (int(dropoff[0]), int(dropoff[1]))
        except Exception:
            # If step is malformed, stop counting further
            break

        dist_to_pick = _manhattan(cur_pos, pick_pos)
        dist_pick_to_drop = _manhattan(pick_pos, drop_pos)
        need = _est_cost_cells(dist_to_pick, 0.0) + _est_cost_cells(
            dist_pick_to_drop, weight
        )

        remaining_batt -= need
        if remaining_batt < 0:
            # Out of battery, clamp to 0 and stop
            remaining_batt = 0.0
            break

        cur_pos = drop_pos

    picked_count = len(plan)
    return picked_count, remaining_batt


def _call_openai_llm_for_plan(
    snapshot: Dict[str, Any],
    num_trials: int,
    search_mode: str,
    max_items: int,
) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI (via LangChain) num_trials times and pick the best plan.
    Returns None if LANGCHAIN_OK is False or all attempts fail.
    """
    if not LANGCHAIN_OK or _CHAIN is None:
        return None

    s = json.dumps(snapshot, sort_keys=True)
    sims: List[Dict[str, Any]] = []

    for _ in range(max(1, num_trials)):
        try:
            resp = _CHAIN.run(snapshot=s, search_mode=search_mode)
        except Exception:
            continue
        obj = _parse_llm_json(resp)
        if not obj:
            continue
        plan = obj.get("plan", [])
        if not isinstance(plan, list):
            continue

        # Truncate plan to max_items to keep it sane
        plan = plan[:max_items]
        picked_count, remaining_batt = _score_plan_from_snapshot(snapshot, plan)
        sims.append(
            {
                "plan": plan,
                "remaining_battery": remaining_batt,
                "picked_count": picked_count,
                "confidence": float(obj.get("confidence", 0.8)),
                "narration": obj.get("narration", ""),
            }
        )

    if not sims:
        return None

    best = _choose_best_simulation(sims)
    # Pick a narration and confidence from first matching sim
    for sim in sims:
        if (
            sim["picked_count"] == best["picked_count"]
            and sim["remaining_battery"] == best["remaining_battery"]
        ):
            narration = sim.get("narration") or ""
            confidence = float(sim.get("confidence", 0.8))
            break
    else:
        narration = ""
        confidence = 0.8

    return {
        "plan": best["plan"],
        "remaining_battery": best["remaining_battery"],
        "picked_count": best["picked_count"],
        "confidence": confidence,
        "narration": narration,
        "strategy": "llm_openai",
        "created_at": time.time(),
    }


def _call_ollama_llm_for_plan(
    snapshot: Dict[str, Any],
    num_trials: int,
    search_mode: str,
    max_items: int,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Call Ollama num_trials times and pick the best plan.
    Uses the same prompt as the OpenAI variant, but via the local `generate(...)` client.
    """
    s = json.dumps(snapshot, sort_keys=True)
    sims: List[Dict[str, Any]] = []

    for _ in range(max(1, num_trials)):
        prompt = _LLM_PROMPT_TEXT.format(snapshot=s, search_mode=search_mode)
        try:
            resp = generate(prompt, model=model, stream=False)
        except Exception:
            continue
        obj = _parse_llm_json(resp)
        if not obj:
            continue
        plan = obj.get("plan", [])
        if not isinstance(plan, list):
            continue

        plan = plan[:max_items]
        picked_count, remaining_batt = _score_plan_from_snapshot(snapshot, plan)
        sims.append(
            {
                "plan": plan,
                "remaining_battery": remaining_batt,
                "picked_count": picked_count,
                "confidence": float(obj.get("confidence", 0.8)),
                "narration": obj.get("narration", ""),
            }
        )

    if not sims:
        return None

    best = _choose_best_simulation(sims)
    for sim in sims:
        if (
            sim["picked_count"] == best["picked_count"]
            and sim["remaining_battery"] == best["remaining_battery"]
        ):
            narration = sim.get("narration") or ""
            confidence = float(sim.get("confidence", 0.8))
            break
    else:
        narration = ""
        confidence = 0.8

    return {
        "plan": best["plan"],
        "remaining_battery": best["remaining_battery"],
        "picked_count": best["picked_count"],
        "confidence": confidence,
        "narration": narration,
        "strategy": "llm_ollama",
        "created_at": time.time(),
    }


def _call_llm_for_plan(
    snapshot: Dict[str, Any],
    backend: str,
    num_trials: int,
    search_mode: str,
    max_items: int,
    ollama_model: str,
) -> Dict[str, Any]:
    """
    Dispatcher: choose OpenAI or Ollama backend, run trials, pick best.
    If backend fails, fall back to local planner.
    """
    backend = backend.lower()
    if backend == "openai":
        res = _call_openai_llm_for_plan(snapshot, num_trials, search_mode, max_items)
    elif backend == "ollama":
        res = _call_ollama_llm_for_plan(
            snapshot, num_trials, search_mode, max_items, model=ollama_model
        )
    else:
        res = None

    if res is None:
        # Fall back to local planner if LLM fails
        return _local_plan_snapshot(
            snapshot,
            max_items=max_items,
            search_mode=search_mode,
            num_trials=num_trials,
        )
    return res


# ---------------------------------------------------------------------------
# 4. Public PlannerClient
# ---------------------------------------------------------------------------

class PlannerClient:
    """
    Planner that returns a dict with keys plan, confidence, narration.

    Config:

      use_llm: bool
        - If True, use LLM-based planner (OpenAI or Ollama) with `num_trials`.
        - If False, use local planner (greedy or multi).

      llm_backend: str
        - "openai" to use LangChain + OpenAI (requires LANGCHAIN_OK).
        - "ollama" to use local Ollama via `generate(...)`.

      search_mode: str
        - For local planner:
            "greedy"  - single greedy pass.
            "multi"   - multiple simulations and select best by
                        (picked_count, remaining_battery).
        - For LLM planner:
            passed as a hint in the prompt, but trials control variation.

      num_trials: int
        - For local "multi": number of greedy simulations.
        - For LLM: number of LLM calls to sample candidate plans.

      max_items: int
        - Hard cap on how many pickups a plan should contain.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_backend: str = "ollama",
        ollama_model: str = DEFAULT_MODEL,
        search_mode: str = "multi",
        num_trials: int = 8,
        max_items: int = 20,
    ):
        self.use_llm = use_llm and (llm_backend == "ollama" or LANGCHAIN_OK)
        self.llm_backend = llm_backend
        self.ollama_model = ollama_model
        self.search_mode = search_mode  # "greedy" or "multi"
        self.num_trials = num_trials
        self.max_items = max_items

        self._last_plan: Optional[Dict[str, Any]] = None
        self._last_snapshot_ts: float = 0.0

    def request_plan(
        self,
        snapshot: Dict[str, Any],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        now = time.time()
        # simple rate limiting: do not replan too often if not forced
        if (
            not force_refresh
            and self._last_plan
            and (now - self._last_snapshot_ts) < 3.0
        ):
            return self._last_plan

        if self.use_llm:
            plan = _call_llm_for_plan(
                snapshot,
                backend=self.llm_backend,
                num_trials=self.num_trials,
                search_mode=self.search_mode,
                max_items=self.max_items,
                ollama_model=self.ollama_model,
            )
        else:
            plan = _local_plan_snapshot(
                snapshot,
                max_items=self.max_items,
                search_mode=self.search_mode,
                num_trials=self.num_trials,
            )

        # guarantee presence of narration key
        if "narration" not in plan:
            p = plan.get("plan", [])
            if not p:
                plan["narration"] = "No feasible steps found by planner."
            else:
                picks = [f"{s['pickup'][0]},{s['pickup'][1]}" for s in p[:4]]
                plan["narration"] = (
                    f"Planned {len(p)} pickups ({', '.join(picks)})."
                )

        self._last_plan = plan
        self._last_snapshot_ts = now
        return plan
