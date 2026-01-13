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
- Invoke a Small Language Model (SLM) for preference reasoning
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
import logging
import re
from typing import TypedDict, List, Dict, Optional, Literal, Tuple

from langgraph.graph import StateGraph, END

from src.strategies.sppdec.sppdec_experiment_config import get_sppdec_config

# ===========================================================================
# Logging
# ===========================================================================

logger = logging.getLogger("SPPDEC")
logger.setLevel(logging.INFO)


# ===========================================================================
# Data contracts
# ===========================================================================

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


class AgentIntent(TypedDict):
    """
    Declarative intent inferred from SLM reasoning.

    Important:
    This intent is informational only.
    It does NOT directly trigger execution.
    """
    agent_id: str
    target_parcel_id: Optional[str]
    intent_type: Literal["pickup", "deliver", "idle", "wait"]
    target_station_cell: Optional[Tuple[int, int]]
    confidence: float
    prediction: LocalPrediction
    timestamp: float


class SPPDECAgentState(TypedDict, total=False):
    """
    Mutable state propagated across LangGraph nodes.

    Invariant
    ---------
    Every execution path MUST terminate with `directive` populated.
    The controller must never exit without an explicit directive.

    """

    snapshot: Dict

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
    decision_mode: Literal["candidate", "solo", "coordinate", "wait", "idle"]

    # Output
    directive: Optional[Dict]


# ===========================================================================
# Helper functions
# ===========================================================================

def estimate_task_cost(snapshot: Dict, task: Dict, physics) -> float:
    """
    Estimate the energy cost to reach a pickup location.

    This estimate is:
    - Local
    - Conservative
    - Independent of dropoff selection

    Delivery cost is resolved after station cell selection.
    """
    agent = snapshot["agent"]
    distance = physics.distance(
        (agent["col"], agent["row"]),
        task["pickup"],
    )
    return physics.energy_cost(distance, task.get("weight", 1.0))


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

class SPPDECLangGraph:
    """
    Phase 1 SPPDEC agent controller implemented using LangGraph.

    One instance is created per agent.
    The graph is authoritative over directive emission.

    Key Features (Phase 1):
    ----------------------
    - Physics-grounded peer energy belief tracking
    - Energy-adjusted peer cost predictions for task allocation
    - SLM-based collaborative task selection
    - No communication, coordination, or shared state
    - All beliefs derived from observable motion only
    """

    def __init__(self, physics, variant: str = "combined"):
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
        config = get_sppdec_config(variant)
        # Physics-grounded peer energy belief tracking
        # Structure: {peer_id: {"last_pos": (x, y), "energy_used": float}}
        # - last_pos: Last observed position of peer (for computing movement delta)
        # - energy_used: Cumulative energy estimated from observed motion
        #   using baseline payload assumption
        self.peer_beliefs = {}

        # Tunable parameter: controls strength of energy penalty on peer costs
        # Applied as: adjusted_cost = base_cost + (alpha * cumulative_energy_used)
        self.baseline_payload = config.baseline_payload
        self.energy_penalty_alpha = config.energy_penalty_alpha

        self.app = self._build()

    def set_drone(self, drone):
        """
        Bind a drone instance to this LangGraph controller.

        The drone provides access to the SLM for reasoning.
        Must be called before executing the graph.
        """
        self.drone = drone

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
        Perception and early termination logic for SPPDEC (Phase 1).

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
           - Identifies visible peer agents (for predictive reasoning).
           - Derives station geometry for downstream feasibility and dropoff selection.

        4. Peer energy belief updates (Phase 1 with beliefs):
           - Observes peer position changes since last perception cycle
           - Computes energy expenditure from observed motion using physics model
           - Updates cumulative energy belief for each peer
           - No communication, intent inference, or hidden state reconstruction
           - Beliefs derived purely from observable position deltas
        """

        snap = state["snapshot"]
        agent = snap["agent"]
        agent_id = agent["id"]

        # ==================================================
        # 1. Execution-phase guard
        # ==================================================
        if agent.get("carrying") is not None:
            state["decision_mode"] = "wait"
            state["directive"] = {"mode": "wait"}

            logger.info(
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

        if not remaining_parcels:
            state["decision_mode"] = "idle"
            state["directive"] = {
                "mode": "idle",
                "reason": "no_remaining_parcels",
            }

            logger.info(
                "[SPPDEC][%s] mission complete -> idle (no remaining parcels)",
                agent_id,
            )
            return state

        # ==================================================
        # 3. Peer visibility
        # ==================================================
        state["visible_agents"] = [
            a for a in snap.get("other_agents", [])
            if a.get("id") != agent_id
        ]

        logger.info(
            "[SPPDEC][%s] perceive: %d peer agents visible",
            agent_id,
            len(state["visible_agents"]),
        )

        # ==================================================
        # 4. Peer energy belief updates (physics-grounded)
        # ==================================================
        for peer in snap.get("other_agents", []):
            peer_id = peer["id"]
            peer_pos = (peer["col"], peer["row"])

            if peer_id not in self.peer_beliefs:
                self.peer_beliefs[peer_id] = {
                    "last_pos": peer_pos,
                    "energy_used": 0.0,
                }
            else:
                belief = self.peer_beliefs[peer_id]
                dist = self.physics.distance(belief["last_pos"], peer_pos)
                energy_cost = self.physics.energy_cost(dist, self.baseline_payload)
                belief["energy_used"] += energy_cost
                belief["last_pos"] = peer_pos

        # ==================================================
        # 5. Parcel visibility
        # ==================================================
        state["visible_parcels"] = remaining_parcels

        logger.info(
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

            logger.info(
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

        Additionally:
        - Computes peer mobility energy estimates for prediction
        - Uses self.baseline_payload for peers (payload-agnostic prediction)
        - Applies energy penalty based on peer cumulative energy expenditure
        - Does NOT infer intent or commitment

        Energy penalty mechanism:
        - base_cost = physics model using self.baseline_payload
        - adjusted_cost = base_cost + (alpha * cumulative_energy_used)
        """
        if state.get("decision_mode") == "idle":
            state["feasible_tasks"] = []
            state["peer_distance_map"] = {}
            return state

        snap = state["snapshot"]
        agent = snap["agent"]
        peers = state.get("visible_agents", [])

        feasible: List[Dict] = []
        peer_distance_map: Dict[str, Dict[str, float]] = {}

        for p in state.get("visible_parcels", []):
            parcel_id = p["id"]
            pickup = (p["col"], p["row"])

            task = {
                "parcel_id": parcel_id,
                "pickup": pickup,
                "weight": p.get("weight", 1.0),
            }

            self_cost = estimate_task_cost(snap, task, self.physics)

            feasible.append({
                "task_type": "feasible",
                "parcel_id": parcel_id,
                "pickup": pickup,
                "weight": task["weight"],
                "cost": self_cost,
            })

            peer_distance_map[parcel_id] = {}

            for peer in peers:
                peer_id = peer["id"]
                peer_pos = (peer["col"], peer["row"])
                dist = self.physics.distance(peer_pos, pickup)

                base_cost = self.physics.energy_cost(
                    distance=dist,
                    weight=self.baseline_payload,
                )

                if peer_id in self.peer_beliefs:
                    penalty = (
                        self.energy_penalty_alpha
                        * self.peer_beliefs[peer_id]["energy_used"]
                    )
                    adjusted_cost = base_cost + penalty
                else:
                    adjusted_cost = base_cost

                peer_distance_map[parcel_id][peer_id] = adjusted_cost

        state["feasible_tasks"] = feasible
        state["peer_distance_map"] = peer_distance_map

        logger.info(
            "[SPPDEC][%s] evaluate_tasks: %d tasks, energy-adjusted peer costs computed",
            agent["id"],
            len(feasible),
        )

        return state

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
        Enforce schema and apply safe defaults.
        """
        if "parcel_id" not in data:
            raise ValueError(f"Missing 'parcel_id' in {data}")

        if data["parcel_id"] in (None, "", "{your answer}", "{correct parcel id}"):
            raise ValueError(f"Invalid parcel_id value: {data['parcel_id']}")

        try:
            conf = float(data.get("confidence", 0.5))
            data["confidence"] = max(0.0, min(1.0, conf))
        except Exception:
            data["confidence"] = 0.5

        return data

    def reason_with_slm(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Invoke the SLM to select a locally preferable task
        using predictive peer awareness for collaborative task allocation.

        Peer energy beliefs are already integrated into peer costs upstream.
        The SLM sees only adjusted team_efficiency scores and remains unchanged.

        Robustness addition:
        - Retries SLM call if JSON is invalid
        - Does NOT change prompt semantics
        - Falls back to abstain instead of crashing
        """

        if state.get("decision_mode") == "idle":
            return state

        tasks = state.get("feasible_tasks", [])
        peer_map = state.get("peer_distance_map", {})
        agent_id = state["snapshot"]["agent"]["id"]

        if not tasks:
            state["slm_status"] = "abstain"
            return state

        if self.drone is None:
            raise RuntimeError(
                f"[SPPDEC][{agent_id}] Drone not bound to LangGraph before SLM call"
            )

        if not getattr(self.drone, "slm_enabled", False):
            raise RuntimeError(
                f"[SPPDEC][{agent_id}] Drone SLM not active"
            )

        task_scores = []
        for task in tasks:
            pid = task["parcel_id"]
            my_cost = task["cost"]

            peer_costs = peer_map.get(pid, {})
            min_peer = min(peer_costs.values()) if peer_costs else float("inf")

            team_efficiency = min_peer - my_cost

            task_scores.append({
                "parcel_id": pid,
                "my_cost": round(my_cost, 1),
                "best_teammate_cost": round(min_peer, 1),
                "team_efficiency": round(team_efficiency, 1),
            })

        task_scores.sort(key=lambda x: x["team_efficiency"], reverse=True)
        valid_ids = [t["parcel_id"] for t in task_scores]

        print("\n" + "=" * 80)
        print(f"[SPPDEC SLM INPUT] Agent: {agent_id}")
        print("=" * 80)

        if self.peer_beliefs:
            print("\nPeer Energy Beliefs (Physics-Grounded):")
            for peer_id, belief in self.peer_beliefs.items():
                energy_used = belief["energy_used"]
                penalty = self.energy_penalty_alpha * energy_used
                print(
                    f"  {peer_id}: cumulative_energy_used={energy_used:.1f}, cost_penalty={penalty:.2f}"
                )
            print()

        print("Task Scores Fed to SLM (with energy-adjusted peer costs):")
        for score in task_scores:
            print(
                f"  Parcel {score['parcel_id']}: my_cost={score['my_cost']}, "
                f"best_teammate={score['best_teammate_cost']}, team_efficiency={score['team_efficiency']}"
            )
        print("\nOptimal choice (greedy): " + task_scores[0]["parcel_id"])
        print("=" * 80 + "\n")

        prompt = f"""You are a delivery drone selecting a parcel to pick up.

        ONLY OUTPUT JSON. NO OTHER TEXT.
        CRITICAL RULE: Always pick the parcel with the HIGHEST team_efficiency value.

        Examples to learn from:

        Example 1:
        Parcels: [{{\"id\": \"A\", \"team_eff\": 5.2}}, {{\"id\": \"B\", \"team_eff\": -1.0}}]
        Correct choice: \"A\" (because 5.2 > -1.0, higher is better)

        Example 2:
        Parcels: [{{\"id\": \"X\", \"team_eff\": -0.5}}, {{\"id\": \"Y\", \"team_eff\": -3.0}}]
        Correct choice: \"X\" (because -0.5 > -3.0, less negative is better)

        Example 3:
        Parcels: [{{\"id\": \"P1\", \"team_eff\": 2.0}}, {{\"id\": \"P2\", \"team_eff\": 3.5}}, {{\"id\": \"P3\", \"team_eff\": 1.0}}]
        Correct choice: \"P2\" (because 3.5 is the highest value)

        Your available parcels (already sorted highest to lowest):
        {json.dumps([{"parcel_id": t["parcel_id"], "team_efficiency": t["team_efficiency"]} for t in task_scores], indent=2)}

        The parcel with HIGHEST team_efficiency is: {task_scores[0]["parcel_id"]} ({task_scores[0]["team_efficiency"]})

        Select this parcel unless you have a strong strategic reason to pick a different one.

        Valid choices: {valid_ids}

        Respond with JSON only (no explanation):
        {{\"parcel_id\": \"<id>\", \"confidence\": <0.0-1.0>}}
        """.strip()

        MAX_RETRIES = 2
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = self.drone.slm_generate(prompt)

                if not isinstance(raw, str):
                    raise RuntimeError(
                        f"[SPPDEC][{agent_id}] SLM returned non-text response: {type(raw)}"
                    )

                result = self.extract_first_json(raw)

                chosen = result.get("parcel_id")
                conf = result.get("confidence", 0.5)

                if chosen not in valid_ids:
                    raise RuntimeError(
                        f"[SPPDEC][{agent_id}] Invalid parcel_id from SLM\n"
                        f"Chosen: {chosen}\n"
                        f"Allowed: {valid_ids}\n"
                        f"Raw output: {raw[:200]}"
                    )

                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 0.5

                conf = max(0.0, min(1.0, conf))

                state["slm_preferred_parcel_id"] = chosen
                state["slm_confidence"] = conf
                state["slm_status"] = "success"

                selected_efficiency = next(
                    t["team_efficiency"] for t in task_scores if t["parcel_id"] == chosen
                )

                selected_rank = next(
                    i for i, t in enumerate(task_scores) if t["parcel_id"] == chosen
                ) + 1

                is_optimal = chosen == task_scores[0]["parcel_id"]

                print("\n" + "=" * 80)
                print(f"[SPPDEC SLM DECISION] Agent: {agent_id}")
                print("=" * 80)
                print(f"Selected Parcel: {chosen}")
                print(f"Team Efficiency: {selected_efficiency}")
                print(f"Confidence: {conf:.2f}")
                print(f"Rank: {selected_rank}/{len(task_scores)}")
                print(
                    f"Optimal Choice: {'YES' if is_optimal else 'NO (greedy would pick ' + task_scores[0]['parcel_id'] + ')'}"
                )
                print("Raw SLM Output:")
                print(raw[:1000])  # show up to 1000 chars instead of 150
                if len(raw) > 1000:
                    print("... (truncated)")
                print("=" * 80 + "\n")

                logger.info(
                    "[SPPDEC][%s] SLM selected parcel=%s (team_efficiency=%.1f, confidence=%.2f, rank=%d/%d, optimal=%s)",
                    agent_id,
                    chosen,
                    selected_efficiency,
                    conf,
                    selected_rank,
                    len(task_scores),
                    is_optimal,
                )

                return state

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[SPPDEC][%s] SLM attempt %d failed: %s",
                    agent_id,
                    attempt + 1,
                    str(exc),
                )

        logger.error(
            "[SPPDEC][%s] SLM failed after retries. Falling back to abstain. Last error: %s",
            agent_id,
            last_error,
        )

        state["slm_status"] = "abstain"
        state["slm_preferred_parcel_id"] = None
        state["slm_confidence"] = 0.0
        return state

    def decide_mode(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Resolve high-level decision mode after SLM reasoning.

        This node NEVER invents a task.
        It only interprets the SLM outcome.

        Semantics:
        - success -> proceed with selected task
        - abstain -> wait
        - failed -> abort experiment
        """
        if state.get("decision_mode") == "idle":
            return state

        agent_id = state["snapshot"]["agent"]["id"]

        slm_status = state.get("slm_status")

        if slm_status == "failed":
            raise RuntimeError(
                f"[SPPDEC][{agent_id}] SLM failed. SPPDEC requires prediction to proceed."
            )

        if slm_status == "abstain":
            state["decision_mode"] = "wait"
            logger.info(
                "[SPPDEC][%s] SLM abstained -> WAIT",
                agent_id,
            )
            return state

        if slm_status == "success":
            preferred = state.get("slm_preferred_parcel_id")
            tasks = state.get("feasible_tasks", [])

            selected = next(
                (t for t in tasks if t["parcel_id"] == preferred),
                None,
            )

            if selected is None:
                raise RuntimeError(
                    f"[SPPDEC][{agent_id}] SLM selected parcel not present in feasible_tasks"
                )

            state["selected_task"] = selected
            state["decision_mode"] = "candidate"

            logger.info(
                "[SPPDEC][%s] Decision accepted: parcel=%s",
                agent_id,
                preferred,
            )
            return state

        raise RuntimeError(
            f"[SPPDEC][{agent_id}] Invalid SLM status: {slm_status}"
        )

    def select_drop_cell(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Select a concrete station cell for dropoff.

        Selection is greedy and distance-minimizing.
        """
        if state.get("decision_mode") != "candidate":
            return state

        snap = state["snapshot"]
        task = state.get("selected_task")
        station_cells = snap.get("station_cells")

        if not task or not station_cells:
            state["decision_mode"] = "wait"
            agent_id = snap["agent"]["id"]
            logger.info(
                f"[SPPDEC][{agent_id}] No task or station cells -> WAIT"
            )
            return state

        agent = snap["agent"]
        agent_pos = (agent["col"], agent["row"])

        best_cell = min(
            station_cells,
            key=lambda c: self.physics.distance(agent_pos, c),
        )

        task["dropoff"] = best_cell
        task["task_type"] = "executable"
        state["decision_mode"] = "solo"

        agent_id = snap["agent"]["id"]
        logger.info(
            f"[SPPDEC][{agent_id}] Complete plan: pickup={task['pickup']} -> dropoff={best_cell}, cost={task['cost']:.1f}"
        )

        return state

    def emit_directive(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Emit a controller-compatible execution directive.
        """
        task = state.get("selected_task")
        if task and task.get("task_type") == "executable":
            state["directive"] = build_execution_directive(
                task,
                confidence=state.get("slm_confidence", 0.5),
            )
        return state

    def finalize(self, state: SPPDECAgentState) -> SPPDECAgentState:
        """
        Finalize the decision cycle.

        Guarantees that a directive is always returned.
        """
        if state.get("directive") is not None:
            return state

        if state.get("decision_mode") == "idle":
            state["directive"] = {
                "mode": "idle",
                "reason": "no_remaining_parcels",
            }
        else:
            state["directive"] = {"mode": "wait"}

        return state

    def _route_after_decision(self, state: SPPDECAgentState) -> str:
        """
        Determine routing after dropoff selection.
        """
        return state.get("decision_mode", "wait")
