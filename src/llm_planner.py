# src/llm_planner.py
"""
Planner client that returns a full delivery plan (sequence of pickups and delivery targets)
and a short human-readable narration explaining the plan.
- If LangChain + an LLM is available, it will ask the model for a compact JSON plan.
- Otherwise falls back to a deterministic nearest-first planner that respects battery constraints.

Plan format:
{
  "plan": [
     {"pickup": [col,row], "dropoff": [col,row], "weight": 1.0 },
     ...
  ],
  "confidence": 0.8,
  "narration": "Short summary sentence(s)..."
}
"""

from __future__ import annotations
import json
import random
import time
from typing import Dict, Any, List

# Try to import LangChain/OpenAI -> optional. If missing, use fallback planner.
try:
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    from langchain.llms import OpenAI
    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK = False


def _local_plan_snapshot(snapshot: Dict[str, Any], max_items: int = 20) -> Dict[str, Any]:
    """
    Simple greedy nearest-first planner + synthesized narration.
    """
    agent = snapshot.get("agent", {})
    battery = float(agent.get("battery_pct", 0.0))
    start = (agent.get("col", 0), agent.get("row", 0))
    parcels = [p for p in snapshot.get("all_parcels", []) if not p.get("picked", False) and not p.get("delivered", False)]
    parcels = [dict(p) for p in parcels]
    plan = []
    cur_pos = start
    remaining_batt = battery
    station = snapshot.get("nearest_station", None)

    def manh(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def est_cost_cells(dist_cells, weight):
        return dist_cells * (1.0 + 0.4 * weight)

    attempts = 0
    while parcels and len(plan) < max_items and attempts < 200:
        parcels.sort(key=lambda p: (manh(cur_pos, (p["col"], p["row"])), p.get("weight", 1.0)))
        chosen = None
        for p in parcels:
            pick_cell = (p["col"], p["row"])
            dist_to_pick = manh(cur_pos, pick_cell)
            if station:
                station_center = (station["col"] + station["w"]//2, station["row"] + station["h"]//2)
                dist_pick_to_drop = manh(pick_cell, station_center)
            else:
                station_center = pick_cell
                dist_pick_to_drop = 0
            need = est_cost_cells(dist_to_pick, 0.0) + est_cost_cells(dist_pick_to_drop, p.get("weight", 1.0))
            if remaining_batt >= need + 3:
                chosen = p
                break
        if not chosen:
            break
        drop = station_center
        plan.append({"pickup": [chosen["col"], chosen["row"]], "dropoff": [int(drop[0]), int(drop[1])], "weight": float(chosen.get("weight", 1.0))})
        remaining_batt -= est_cost_cells(manh(cur_pos, (chosen["col"], chosen["row"])), 0.0)
        remaining_batt -= est_cost_cells(manh((chosen["col"], chosen["row"]), drop), chosen.get("weight", 1.0))
        cur_pos = drop
        parcels.remove(chosen)
        attempts += 1

    # Build a concise narration from the plan
    if not plan:
        if battery < 10:
            narration = "No feasible pickups planned. Battery low; returning to station is recommended."
        else:
            narration = "No suitable pickups found that meet battery constraints. Idle or reposition to find parcels."
    else:
        # summarize up to 4 pickups
        picks = [f"({p['pickup'][0]},{p['pickup'][1]})" for p in plan[:4]]
        if station:
            narration = f"Planned {len(plan)} pickup(s), prioritizing nearby parcels ({', '.join(picks)}). Will deliver to nearest station to maximize successful returns."
        else:
            narration = f"Planned {len(plan)} pickup(s), prioritizing nearby parcels ({', '.join(picks)}). No station available; drops will be local."

    return {"plan": plan, "confidence": 0.6, "narration": narration, "created_at": time.time()}


# If LangChain is available we prepare a prompt that asks for both plan and narration.
if LANGCHAIN_OK:
    _LLM = OpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=700)

    _PROMPT = PromptTemplate(
        input_variables=["snapshot"],
        template=(
            "You are a delivery planner operating with the following deterministic energy model:\n"
            " - To travel N cell-units costs: energy = N * base_cost_per_cell * (1 + weight * weight_factor)\n"
            " - Picking or dropping a parcel costs pick_drop_cost*(1 + weight*weight_factor)\n"
            " - Distances: use euclidean cell distances provided in snapshot (or compute euclidean if needed).\n\n"

            "You will be given exactly one variable: {snapshot}\n\n"

            "Return valid JSON only, with keys:\n"
            "  - plan: a list of step objects in execution order. Each step must be {\"pickup\":[col,row], \"dropoff\":[col,row], \"weight\":float}\n"
            "  - confidence: float between 0.0 and 1.0 indicating how confident you are the plan respects energy constraints\n"
            "  - narration: 1-4 short sentences explaining why the plan is efficient and any risky assumptions (if any)\n\n"

            "Rules:\n"
            "  - Compute energy required for each leg using the energy_model constants in the snapshot.\n"
            "  - If a planned leg requires more energy than the agent's battery_level, mark it as risky and include that information in narration.\n"
            "  - Prefer pickups with lower total estimated cost first, and prefer delivering to free station cells (snapshot includes station center and per-parcel estimates).\n"
            "  - Do NOT output any other keys or extra text. Only output JSON.\n\n"
            "Objective: maximize number of completed deliveries while minimizing the chance of the drone becoming lost."
        )
    )

    _CHAIN = LLMChain(llm=_LLM, prompt=_PROMPT)


def _call_llm_for_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not LANGCHAIN_OK:
        return _local_plan_snapshot(snapshot)
    s = json.dumps(snapshot, sort_keys=True)
    try:
        resp = _CHAIN.run(snapshot=s)
        # parse JSON response robustly
        try:
            obj = json.loads(resp)
        except Exception:
            # try to extract first JSON object in text
            start = resp.find("{")
            end = resp.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(resp[start:end+1])
            else:
                return _local_plan_snapshot(snapshot)

        # ensure plan list exists
        if "plan" in obj and isinstance(obj["plan"], list):
            # ensure narration exists; if not synthesize
            if "narration" not in obj or not obj["narration"]:
                # synthesize short narration from plan
                plan = obj.get("plan", [])
                if not plan:
                    obj["narration"] = "No plan provided by LLM; fallback recommended."
                else:
                    picks = [f"{p['pickup'][0]},{p['pickup'][1]}" for p in plan[:4]]
                    if snapshot.get("nearest_station"):
                        obj["narration"] = f"LLM: plan {len(plan)} pickups, prioritized by proximity ({', '.join(picks)}); delivering to station to minimize lost drones."
                    else:
                        obj["narration"] = f"LLM: plan {len(plan)} pickups, prioritized by proximity ({', '.join(picks)}); no station available so drops are local."
            return obj
        else:
            return _local_plan_snapshot(snapshot)
    except Exception:
        return _local_plan_snapshot(snapshot)


class PlannerClient:
    """

    Planner that returns a dict with keys plan, confidence, narration.

    """
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm and LANGCHAIN_OK
        self._last_plan = None
        self._last_snapshot_ts = 0.0

    def request_plan(self, snapshot: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force_refresh and self._last_plan and (now - self._last_snapshot_ts) < 3.0:
            return self._last_plan

        if self.use_llm:
            plan = _call_llm_for_plan(snapshot)
        else:
            plan = _local_plan_snapshot(snapshot)

        # guarantee presence of narration key
        if "narration" not in plan:
            # synthesize fallback narration
            p = plan.get("plan", [])
            if not p:
                plan["narration"] = "No feasible steps found by planner."
            else:
                picks = [f"{s['pickup'][0]},{s['pickup'][1]}" for s in p[:4]]
                plan["narration"] = f"Planned {len(p)} pickups ({', '.join(picks)})."

        self._last_plan = plan
        self._last_snapshot_ts = now
        return plan
