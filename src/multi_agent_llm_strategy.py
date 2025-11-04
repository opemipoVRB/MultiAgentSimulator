# src/strategies/multi_agent_llm_strategy.py
from __future__ import annotations
import time
import json
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Deque, Optional

# Try to use the user's llm_client if available
_HAS_USER_LLM = False
try:
    from llm_client import llm_client  # type: ignore
    _HAS_USER_LLM = True
except Exception:
    llm_client = None  # type: ignore
    _HAS_USER_LLM = False

# fallback auction if LLM not available
try:
    from coordinator import AuctionStrategy  # type: ignore
except Exception:
    AuctionStrategy = None  # type: ignore


class MultiAgentLLMStrategy:
    """
    Multi-agent LLM strategy.

    Behavior summary:
      1) Build compact snapshot of agents and available parcels.
      2) For R rounds (default 2), call the LLM once per agent asking for a short,
         deterministic set of proposals. Each agent sees other agents' last proposals when revising.
      3) After rounds complete, deterministically arbitrate conflicts using a scorer.
      4) Return mapping controller -> deque(parcel objects).

    Notes:
      - This Coordinator-level strategy simulates peer-to-peer negotiation by making
        independent LLM calls for each agent. In a true P2P deployment, each drone
        would run its own LLM and exchange messages over the network.
      - Keep rounds small to avoid oscillation and cost.
    """

    strategy_name = "MultiAgentLLMStrategy"

    def __init__(self,
                 tick_interval: float = 1.0,
                 max_assign_per_agent: int = 2,
                 rounds: int = 2,
                 cache_ttl: float = 1.5,
                 per_agent_min_interval: float = 0.4):
        self.tick_interval = float(tick_interval)
        self.max_assign_per_agent = int(max_assign_per_agent)
        self.rounds = max(1, int(rounds))
        self._last_tick = 0.0
        self._cache_snapshot = None
        self._cache_assignments = None
        self._cache_ts = 0.0
        self._cache_ttl = float(cache_ttl)
        self._agent_last_call: Dict[int, float] = {}
        self._per_agent_min_interval = float(per_agent_min_interval)
        # scorer weights, tunable
        self._w_distance = 1.0
        self._w_battery = -0.6
        self._w_weight = 0.8
        self._w_conflicts = 1.6

    def should_run(self) -> bool:
        return (time.time() - self._last_tick) >= self.tick_interval

    def mark_ran(self):
        self._last_tick = time.time()

    def _build_snapshot(self, controllers: List[Any], terrain: Any) -> Dict[str, Any]:
        agents = []
        for idx, c in enumerate(controllers):
            try:
                col = int(c.drone.col)
                row = int(c.drone.row)
                battery_pct = int(getattr(c.drone, "power").percent()) if getattr(c.drone, "power", None) else 100
                agents.append({"id": idx, "col": col, "row": row, "battery_pct": battery_pct})
            except Exception:
                continue

        parcels = []
        for p in terrain.parcels:
            if getattr(p, "picked", False) or getattr(p, "delivered", False):
                continue
            parcels.append({"col": int(p.col), "row": int(p.row), "weight": float(getattr(p, "weight", 1.0))})

        grid = {"cols": int(terrain.screen_size[0] // terrain.grid_size),
                "rows": int(terrain.screen_size[1] // terrain.grid_size)}

        return {"agents": agents, "parcels": parcels, "grid": grid, "ts": time.time()}

    def _llm_call_for_agent(self, agent_idx: int, snapshot: Dict, other_proposals: List[Dict]) -> Optional[Dict]:
        """
        Call LLM for a single agent. other_proposals is used to simulate what the agent
        saw from peers in previous rounds.
        Returns parsed JSON on success, else None.
        """
        if not _HAS_USER_LLM or llm_client is None:
            return None

        now = time.time()
        last = self._agent_last_call.get(agent_idx, 0.0)
        if (now - last) < self._per_agent_min_interval:
            return None
        self._agent_last_call[agent_idx] = now

        prompt_obj = {
            "role": "agent",
            "agent_id": agent_idx,
            "instruction": (
                "You are an agent proposing parcel pickup coordinates. "
                "Return JSON only with key 'proposals' which is a list of [col,row]. "
                f"Return at most {self.max_assign_per_agent} coords. Be conservative about battery."
            ),
            "snapshot": snapshot,
            "other_proposals": other_proposals
        }

        prompt_text = ("AGENT PROPOSAL REQUEST\n"
                       "Input is a JSON object. Output must be JSON only.\n\n"
                       f"{json.dumps(prompt_obj, sort_keys=True)}\n\nOUTPUT:")

        try:
            chat_getter = getattr(llm_client, "get_chat_model", None)
            llm = None
            if chat_getter:
                llm = llm_client.get_chat_model()
            else:
                llm_getter = getattr(llm_client, "get_llm_model", None)
                if llm_getter:
                    llm = llm_client.get_llm_model()

            if llm is None:
                return None

            raw = None
            try:
                if hasattr(llm, "generate"):
                    out = llm.generate([{"role": "user", "content": prompt_text}])
                    try:
                        raw = out.generations[0][0].text  # type: ignore
                    except Exception:
                        raw = str(out)
                elif callable(llm):
                    raw = llm(prompt_text)
                else:
                    raw = str(llm)
            except Exception:
                try:
                    raw = llm(prompt_text)
                except Exception:
                    raw = None

            if not raw:
                return None
            if not isinstance(raw, str):
                raw = str(raw)

            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            raw_json = raw[start:end+1]
            parsed = json.loads(raw_json)
            return parsed
        except Exception:
            return None

    def _score_assignment_candidate(self, ctrl_idx: int, parcel_coord: Tuple[int, int], controllers: List[Any], terrain: Any) -> float:
        try:
            ctrl = controllers[ctrl_idx]
            a_col, a_row = int(ctrl.drone.col), int(ctrl.drone.row)
            d = abs(a_col - parcel_coord[0]) + abs(a_row - parcel_coord[1])
            battery_pct = int(getattr(ctrl.drone, "power").percent()) if getattr(ctrl.drone, "power", None) else 100
            batt_pen = (100.0 - battery_pct) / 100.0
            parcel_obj = terrain.parcel_at_cell(parcel_coord[0], parcel_coord[1])
            weight = float(getattr(parcel_obj, "weight", 1.0)) if parcel_obj is not None else 1.0
            score = (self._w_distance * float(d)
                     + self._w_battery * float(batt_pen)
                     + self._w_weight * float(weight))
            return float(score)
        except Exception:
            return float("inf")

    def _arbitrate(self, controllers: List[Any], terrain: Any, candidate_map: Dict[int, List[Tuple[int,int]]]) -> Dict[Any, Deque]:
        assignments: Dict[Any, Deque] = defaultdict(deque)
        entries: List[Tuple[float, int, Tuple[int,int]]] = []
        for aid, coords in candidate_map.items():
            for coord in coords:
                score = self._score_assignment_candidate(aid, coord, controllers, terrain)
                entries.append((score, aid, coord))

        entries.sort(key=lambda t: (t[0], t[1], t[2][0], t[2][1]))

        taken = set()
        for score, aid, coord in entries:
            if coord in taken:
                continue
            if not (0 <= aid < len(controllers)):
                continue
            ctrl = controllers[aid]
            if len(assignments[ctrl]) >= self.max_assign_per_agent:
                continue
            parcel = terrain.parcel_at_cell(coord[0], coord[1])
            if parcel is None:
                continue
            if getattr(parcel, "picked", False) or getattr(parcel, "delivered", False):
                continue
            assignments[ctrl].append(parcel)
            taken.add(coord)

        return assignments

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        now = time.time()
        snapshot = self._build_snapshot(controllers, terrain)
        if self._cache_snapshot and (now - self._cache_ts) < self._cache_ttl:
            prev_agents = self._cache_snapshot.get("agents", [])
            prev_parcels = self._cache_snapshot.get("parcels", [])
            if len(prev_agents) == len(snapshot["agents"]) and len(prev_parcels) == len(snapshot["parcels"]):
                self.mark_ran()
                return self._cache_assignments or defaultdict(deque)

        if not _HAS_USER_LLM or llm_client is None:
            if AuctionStrategy is not None:
                print("LLM Falling back to AuctionStrategy")
                fallback = AuctionStrategy(tick_interval=self.tick_interval, max_assign_per_agent=self.max_assign_per_agent)
                return fallback.compute_assignments(controllers, terrain)
            return defaultdict(deque)

        candidate_map: Dict[int, List[Tuple[int,int]]] = {a["id"]: [] for a in snapshot["agents"]}
        other_proposals: List[Dict] = []

        for r in range(self.rounds):
            for agent in snapshot["agents"]:
                aid = int(agent["id"])
                per_agent_snapshot = {
                    "agents": snapshot["agents"],
                    "parcels": snapshot["parcels"],
                    "grid": snapshot["grid"],
                    "me": agent,
                    "round": r + 1
                }
                resp = self._llm_call_for_agent(aid, per_agent_snapshot, other_proposals)
                if resp and isinstance(resp, dict):
                    props = resp.get("proposals")
                    if isinstance(props, list):
                        coords = []
                        for item in props[: self.max_assign_per_agent]:
                            try:
                                if isinstance(item, (list, tuple)) and len(item) >= 2:
                                    coords.append((int(item[0]), int(item[1])))
                                elif isinstance(item, dict) and "col" in item and "row" in item:
                                    coords.append((int(item["col"]), int(item["row"])))
                            except Exception:
                                continue
                        candidate_map[aid] = coords
                    elif "scores" in resp and isinstance(resp["scores"], list):
                        scs = []
                        for s in resp["scores"]:
                            try:
                                coord = s.get("parcel")
                                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                                    scs.append(((int(coord[0]), int(coord[1])), float(s.get("score", 1.0))))
                            except Exception:
                                continue
                        scs.sort(key=lambda t: t[1])
                        candidate_map[aid] = [c for c, _ in scs[: self.max_assign_per_agent]]
                other_proposals = [{"agent_id": k, "proposals": v} for k, v in candidate_map.items() if v]

        final_assignments = self._arbitrate(controllers, terrain, candidate_map)

        self._cache_snapshot = snapshot
        cache_assign_repr = {}
        for ctrl, dq in final_assignments.items():
            try:
                idx = controllers.index(ctrl)
            except Exception:
                continue
            cache_assign_repr[idx] = [(int(p.col), int(p.row)) for p in dq]
        self._cache_assignments = final_assignments
        self._cache_ts = now

        self.mark_ran()
        return final_assignments
