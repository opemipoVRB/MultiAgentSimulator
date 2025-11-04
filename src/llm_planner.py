# src/llm_planner.py
"""
Planner client that returns a full delivery plan (sequence of pickups and delivery targets)
and a short human-readable narration explaining the plan.

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
from typing import Dict, Any, List, Optional

# Try to import user's llm_client first (common in your codebase).
# Try both 'llm_client' and 'src.llm_client' so IDE/run-mode differences are handled.
_llm_client = None
_HAS_USER_LLM_CLIENT = False
try:
    from llm_client import llm_client  # type: ignore
    _llm_client = llm_client  # type: ignore
    _HAS_USER_LLM_CLIENT = True
except Exception:
    try:
        from src.llm_client import llm_client  # type: ignore
        _llm_client = llm_client  # type: ignore
        _HAS_USER_LLM_CLIENT = True
    except Exception:
        _llm_client = None  # type: ignore
        _HAS_USER_LLM_CLIENT = False

# Try to import LangChain pieces if user's client isn't available
try:
    from langchain.chains import LLMChain  # type: ignore
    from langchain_core.prompts import PromptTemplate  # type: ignore
    from langchain.llms import OpenAI  # type: ignore
    _LANGCHAIN_AVAILABLE = True
except Exception:
    LLMChain = None
    PromptTemplate = None
    OpenAI = None
    _LANGCHAIN_AVAILABLE = False

# If neither user's client nor LangChain available, we'll use the local planner fallback.
USE_LLM = _HAS_USER_LLM_CLIENT or _LANGCHAIN_AVAILABLE


# -----------------------
# Local greedy planner fallback
# -----------------------
def _local_plan_snapshot(snapshot: Dict[str, Any], max_items: int = 20) -> Dict[str, Any]:
    """
    Greedy nearest-first planner + concise narration.
    Deterministic and safe when no LLM is present.
    """
    agent = snapshot.get("agent", {})
    battery = float(agent.get("battery_pct", 0.0))
    start = (agent.get("col", 0), agent.get("row", 0))
    parcels = [p for p in snapshot.get("all_parcels", []) if not p.get("picked", False) and not p.get("delivered", False)]
    parcels = [dict(p) for p in parcels]
    plan: List[Dict[str, Any]] = []
    cur_pos = start
    remaining_batt = battery
    station = snapshot.get("nearest_station", None)

    def manh(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # use a small, conservative weight factor similar to earlier code
    def est_cost_cells(dist_cells, weight):
        return dist_cells * (snapshot.get("energy_model", {}).get("base_cost_per_cell", 0.2) *
                             (1.0 + (weight * snapshot.get("energy_model", {}).get("weight_factor", 0.4))))

    attempts = 0
    while parcels and len(plan) < max_items and attempts < 200:
        parcels.sort(key=lambda p: (manh(cur_pos, (p["col"], p["row"])), p.get("weight", 1.0)))
        chosen = None
        for p in parcels:
            pick_cell = (p["col"], p["row"])
            dist_to_pick = manh(cur_pos, pick_cell)
            if station:
                station_center = (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
                dist_pick_to_drop = manh(pick_cell, station_center)
            else:
                station_center = pick_cell
                dist_pick_to_drop = 0
            need = est_cost_cells(dist_to_pick, 0.0) + est_cost_cells(dist_pick_to_drop, p.get("weight", 1.0))
            # leave a small buffer to avoid overly optimistic plans
            if remaining_batt >= need + 2:
                chosen = p
                break
        if not chosen:
            break
        drop = station_center
        plan.append({"pickup": [int(chosen["col"]), int(chosen["row"])],
                     "dropoff": [int(drop[0]), int(drop[1])],
                     "weight": float(chosen.get("weight", 1.0))})
        remaining_batt -= est_cost_cells(manh(cur_pos, (chosen["col"], chosen["row"])), 0.0)
        remaining_batt -= est_cost_cells(manh((chosen["col"], chosen["row"]), drop), chosen.get("weight", 1.0))
        cur_pos = drop
        parcels.remove(chosen)
        attempts += 1

    # narration
    if not plan:
        if battery < 10:
            narration = "No feasible pickups planned. Battery low; returning to station is recommended."
        else:
            narration = "No suitable pickups found that meet battery constraints. Idle or reposition to find parcels."
    else:
        picks = [f"({p['pickup'][0]},{p['pickup'][1]})" for p in plan[:4]]
        if station:
            narration = f"Planned {len(plan)} pickup(s), prioritizing nearby parcels ({', '.join(picks)}). Delivering to nearest station."
        else:
            narration = f"Planned {len(plan)} pickups ({', '.join(picks)}). No station available; drops will be local."

    return {"plan": plan, "confidence": 0.6, "narration": narration, "created_at": time.time()}


# -----------------------
# LLM-based planner (if available)
# -----------------------
_PROMPT_TEXT = (
    "You are a deterministic delivery planner. Use the provided `snapshot` JSON to\n"
    "compute a plan that maximizes completed deliveries while minimizing the chance the drone runs out\n"
    "of energy. The snapshot contains an `energy_model` object explaining how to compute costs.\n\n"
    "INPUT: a single JSON object `snapshot` (provided as a variable). Output MUST be valid JSON with only the keys:\n"
    "  - plan: list of steps, each step is {\"pickup\": [col,row], \"dropoff\": [col,row], \"weight\": float}\n"
    "  - confidence: float between 0.0 and 1.0 (how confident you are about energy constraints)\n"
    "  - narration: 1-4 short sentences explaining the plan and any risky assumptions\n\n"
    "Rules:\n"
    " - Use the energy_model values from snapshot (base_cost_per_cell, weight_factor, pick_drop_cost) to compute costs.\n"
    " - If a planned leg requires more energy than the agent's battery_level, label it as risky in narration.\n"
    " - Prefer pickups with lower estimated total cost first and prefer delivering to free station center when available.\n"
    " - Output only JSON (no additional commentary)."
)


def _call_llm_for_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the LLM to request a plan. If anything fails (parsing, LLM errors) fall back to the local planner.
    """
    if not USE_LLM:
        return _local_plan_snapshot(snapshot)

    # Prepare the snapshot string (compact)
    snapshot_json = json.dumps(snapshot, sort_keys=True)

    # 1) If user provided a llm_client singleton, prefer using it.
    if _HAS_USER_LLM_CLIENT and _llm_client is not None:
        try:
            # Build a prompt by injecting snapshot as variable in the template-like message
            prompt = _PROMPT_TEXT + "\n\nSNAPSHOT:\n" + snapshot_json
            lm = getattr(_llm_client, "get_llm_model", None)
            if lm:
                llm = _llm_client.get_llm_model()
            else:
                llm = None

            # If we have an LLM instance compatible with LangChain LLMChain
            if llm is not None and LLMChain is not None and PromptTemplate is not None:
                chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["text"], template=prompt))
                try:
                    raw = chain.run(text=snapshot_json)
                except Exception:
                    # Some LLM instances expect a single string call; try direct call if available
                    raw = llm(snapshot_json) if callable(llm) else None
            else:
                # Fallback: if chat model is provided, try a simple call
                chat = getattr(_llm_client, "get_chat_model", None)
                if chat:
                    chat_llm = _llm_client.get_chat_model()
                    # Try to call .generate/.call depending on object interface
                    try:
                        raw = chat_llm.generate([{"role": "user", "content": prompt}])  # langchain_chat style
                        # `generate` may return an object; try to extract text
                        if isinstance(raw, str):
                            pass
                        else:
                            # best-effort extraction
                            try:
                                raw = raw.generations[0][0].text  # type: ignore
                            except Exception:
                                raw = str(raw)
                    except Exception:
                        # fallback: call llm() if callable
                        try:
                            raw = chat_llm(prompt)
                        except Exception:
                            raw = None
                else:
                    raw = None

            if not raw:
                return _local_plan_snapshot(snapshot)

            # raw might sometimes be a structured object; ensure string
            if not isinstance(raw, str):
                raw = str(raw)

            # Try parsing JSON robustly
            try:
                obj = json.loads(raw)
            except Exception:
                # fallback: extract first {...} JSON substring
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        obj = json.loads(raw[start:end+1])
                    except Exception:
                        return _local_plan_snapshot(snapshot)
                else:
                    return _local_plan_snapshot(snapshot)

            # Ensure plan exists
            if "plan" in obj and isinstance(obj["plan"], list):
                if "narration" not in obj or not obj["narration"]:
                    # synthesize short narration
                    plan = obj.get("plan", [])
                    if not plan:
                        obj["narration"] = "No plan provided by LLM; fallback recommended."
                    else:
                        picks = [f"{p['pickup'][0]},{p['pickup'][1]}" for p in plan[:4]]
                        if snapshot.get("nearest_station"):
                            obj["narration"] = f"LLM: plan {len(plan)} pickups, prioritized by proximity ({', '.join(picks)}); delivering to station."
                        else:
                            obj["narration"] = f"LLM: plan {len(plan)} pickups ({', '.join(picks)}); no station available."
                return obj
            else:
                return _local_plan_snapshot(snapshot)

        except Exception:
            return _local_plan_snapshot(snapshot)

    # 2) If user client not available, but LangChain installed, use LLMChain + OpenAI
    if _LANGCHAIN_AVAILABLE:
        try:
            prompt = _PROMPT_TEXT + "\n\nSNAPSHOT:\n{{snapshot}}"
            # Create a minimal prompt template that takes "snapshot" as input
            prompt_tpl = PromptTemplate(input_variables=["snapshot"], template=prompt)
            # instantiate an LLM if not using user's client
            llm = OpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=700)
            chain = LLMChain(llm=llm, prompt=prompt_tpl)
            raw = chain.run(snapshot=snapshot_json)
            try:
                obj = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    obj = json.loads(raw[start:end+1])
                else:
                    return _local_plan_snapshot(snapshot)

            if "plan" in obj and isinstance(obj["plan"], list):
                if "narration" not in obj or not obj["narration"]:
                    plan = obj.get("plan", [])
                    if not plan:
                        obj["narration"] = "No plan provided by LLM; fallback recommended."
                    else:
                        # FIXED: corrected f-string indices here (was syntax error before)
                        picks = [f"{p['pickup'][0]},{p['pickup'][1]}" for p in plan[:4]]
                        if snapshot.get("nearest_station"):
                            obj["narration"] = f"LLM: plan {len(plan)} pickups, prioritized by proximity ({', '.join(picks)}); delivering to station."
                        else:
                            obj["narration"] = f"LLM: plan {len(plan)} pickups ({', '.join(picks)}); no station available."
                return obj
            else:
                return _local_plan_snapshot(snapshot)
        except Exception:
            return _local_plan_snapshot(snapshot)

    # If we get here, no LLM path was usable

    return _local_plan_snapshot(snapshot)


# -----------------------
# PlannerClient - exposed to the rest of the code
# -----------------------
class PlannerClient:
    """
    Planner that returns a dict with keys plan, confidence, narration.
    """
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm and USE_LLM
        self._last_plan = None
        self._last_snapshot_ts = 0.0
        self._cache_ttl = 3.0  # seconds

    def request_plan(self, snapshot: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force_refresh and self._last_plan and (now - self._last_snapshot_ts) < self._cache_ttl:
            return self._last_plan

        if self.use_llm:
            plan = _call_llm_for_plan(snapshot)
        else:
            plan = _local_plan_snapshot(snapshot)

        # ensure narration key
        if "narration" not in plan:
            p = plan.get("plan", [])
            if not p:
                plan["narration"] = "No feasible steps found by planner."
            else:
                # ✅ Correctly formatted f-string (no syntax error)
                picks = [f"{s['pickup'][0]},{s['pickup'][1]}" for s in p[:4]]
                plan["narration"] = f"Planned {len(p)} pickups ({', '.join(picks)})."

        # ensure valid structure
        if "plan" not in plan or not isinstance(plan["plan"], list):
            plan = _local_plan_snapshot(snapshot)

        self._last_plan = plan
        self._last_snapshot_ts = now
        return plan
