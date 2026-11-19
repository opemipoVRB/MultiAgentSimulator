# src/strategies/naive.py
"""
Naive decentralised single-agent planning strategy with operational robustness.

This strategy implements Monte Carlo greedy rollouts with configurable
energy efficiency trade-offs, producing plans with quantified operational
confidence scores based on both mission accomplishment and robustness to
uncertainties.

Key Philosophical Shift:
- Confidence is not merely "task completion probability"
- Confidence reflects "probability of mission success despite operational uncertainties"
- Higher battery provides margin for: navigation errors, weight misestimation,
  obstacle avoidance, coordination conflicts, and other unforeseen challenges
- Plans with more battery buffer are operationally more robust
"""

import math
import random
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
from .base import BaseStrategy


class NaiveStrategy(BaseStrategy):
    """
        Naive snapshot-based single-agent planning strategy
        with operational robustness and local autonomy.

        This strategy assumes a fully decentralized execution model
        in which each agent is responsible for completing tasks
        without external coordination.

        Key characteristics:
        --------------------
        - Single-agent planning based on local snapshots
        - Monte Carlo greedy rollouts with energy-awareness
        - Explicit operational confidence estimation
        - Robustness-focused plan selection

        Autonomy model:
        ---------------
        The agent is allowed to adapt locally during execution.

        If execution constraints invalidate a planned step:
          - the controller MAY substitute a feasible alternative
          - confidence is revised using posterior evidence
          - execution continues without external intervention

        This makes the strategy resilient under uncertainty,
        but unsuitable for coordinated multi-agent task allocation.

        Reactive fallback is therefore ENABLED by design.
        """
    enable_reactive_fallback = True
    requires_plan_completion_before_requery = False

    def __init__(self, physics, num_trials: int = 10):
        """
        Initialize the naive strategy with operational robustness.

        Parameters:
        -----------
        num_trials : int, default=10
            Number of greedy rollouts to generate per planning cycle.
            Higher values increase exploration of task-battery trade-offs.
        """
        super().__init__()
        self.num_trials = num_trials
        self.physics = physics
        self.drone = None

    def set_drone(self, drone):
        super().set_drone(drone)
        # Per-drone generator, seeded from the run and the drone, so rollouts do not share a generator and the
        # same run gives the same result whatever the thread timing or process layout.
        self._rng = random.Random(f"naive-{self.drone.agent_id}-{self.drone.run_seed}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, snapshot: Dict, agent_id: str = None) -> Dict:
        plan_obj = self.request_plan(snapshot)

        if not plan_obj.get("plan"):
            return {
                "mode": "idle",
                "reason": "no_feasible_plan",
            }

        return {
            "mode": "plan",
            "plan": plan_obj,
        }

    def request_plan(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a single open-loop plan with quantified operational confidence.

        This method represents the epistemic core of the naive decentralized agent.
        It answers the question: given the agent's current knowledge and battery,
        is there a safe and meaningful course of action?

        The method must return one of three outcomes:
        1) A valid plan with mode implicitly assumed as "plan"
        2) An idle response when no work exists
        3) An abort response when the agent determines the mission is infeasible

        Abort is not an error. Abort is a rational conclusion that the agent
        cannot proceed safely or meaningfully under its operational constraints.

        Parameters
        ----------
        snapshot : Dict[str, Any]
            World snapshot containing:
            - agent state
            - available parcels
            - station geometry

        Returns
        -------
        Dict[str, Any]
            Either:
            - A plan response
            - An idle response
            - An abort response with mode="abort"
        """

        agent_id = snapshot["agent"].get("id", "unknown")

        if self.drone is None:
            raise RuntimeError("[NAIVE] Drone not injected into strategy")

        current_battery = float(self.drone.power.percent())
        N = self.num_trials

        print(f"[NAIVE-PLANNER][agent={agent_id}] Planning with {current_battery:.1f}% battery")

        # -------------------------------
        # Operational robustness parameters
        # -------------------------------
        BASE_RISK_DENSITY = 0.02
        SIGMOID_SHARPNESS = 3.0
        SIGMOID_OFFSET = 0.5
        MIN_SAFE_BATTERY = self.FEASIBILITY_MARGIN
        EPSILON = 1e-3

        # -------------------------------
        # STEP 1: Check for available parcels
        # -------------------------------
        parcels_available = [
            dict(p)
            for p in snapshot.get("all_parcels", [])
            if not p.get("picked") and not p.get("delivered")
        ]

        if not parcels_available:
            print(f"[NAIVE-PLANNER][agent={agent_id}] NO PARCELS: environment is idle")
            return self._idle_response(snapshot, "No parcels available")

        print(f"[NAIVE-PLANNER][agent={agent_id}] Found {len(parcels_available)} parcels")

        # -------------------------------
        # STEP 2: Feasibility test
        # -------------------------------
        # If the agent cannot safely deliver even one parcel, this is not idle.
        # This is epistemic failure and should terminate the strategy.
        if not self.is_globally_feasible(snapshot):
            print(f"[NAIVE-PLANNER][agent={agent_id}] ABORT: no feasible parcel given current battery")

            return {
                "mode": "abort",
                "agent_id": agent_id,
                "reason": "agent_energy_infeasible",
            }

        # -------------------------------
        # STEP 3: Generate rollouts
        # -------------------------------
        rollouts: List[Dict[str, Any]] = []

        print(f"[NAIVE-PLANNER][agent={agent_id}] Generating {N} rollouts")

        for i in range(N):
            efficiency_weight = i / max(1, N - 1)

            plan, projected_battery = self._simulate(
                snapshot,
                shuffle=True,
                efficiency_weight=efficiency_weight
            )

            if plan:
                rollouts.append({
                    "trial": i + 1,
                    "task_completion": len(plan),
                    "projected_battery": projected_battery,
                    "plan": plan,
                    "efficiency_weight": round(efficiency_weight, 2),
                })

                print(
                    f"[NAIVE-PLANNER][agent={agent_id}][trial {i + 1}] "
                    f"tasks={len(plan)}, battery={projected_battery:.1f}%"
                )
            else:
                print(f"[NAIVE-PLANNER][agent={agent_id}][trial {i + 1}] no feasible plan")

        # -------------------------------
        # STEP 4: Abort if exploration fails
        # -------------------------------
        if not rollouts:
            print(f"[NAIVE-PLANNER][agent={agent_id}] ABORT: planner explored but found no plans")

            return {
                "mode": "abort",
                "agent_id": agent_id,
                "reason": "no_feasible_rollout",
            }

        # -------------------------------
        # STEP 5: Compute confidence for each rollout
        # -------------------------------
        max_tasks = max(r["task_completion"] for r in rollouts)

        for r in rollouts:
            confidence, components = self._compute_operational_confidence_with_components(
                task_completion=r["task_completion"],
                battery_percent=r["projected_battery"],
                max_tasks=max_tasks,
                base_risk_density=BASE_RISK_DENSITY,
                sigmoid_sharpness=SIGMOID_SHARPNESS,
                sigmoid_offset=SIGMOID_OFFSET,
                epsilon=EPSILON,
            )

            r["confidence"] = confidence
            r["confidence_components"] = components

        # -------------------------------
        # STEP 6: Safety filter using prefixes
        # -------------------------------
        acceptable_plans = []

        for r in rollouts:
            repaired = self._best_safe_prefix(r, snapshot, MIN_SAFE_BATTERY)

            if repaired is None:
                print(f"[NAIVE-PLANNER][agent={agent_id}] rollout rejected: no safe prefix")
                continue

            acceptable_plans.append(repaired)

        # If nothing survives safety filtering, abort
        if not acceptable_plans:
            print(f"[NAIVE-PLANNER][agent={agent_id}] ABORT: no safe prefix for any plan")

            return {
                "mode": "abort",
                "agent_id": agent_id,
                "reason": "no_safe_prefix",
            }

        # -------------------------------
        # STEP 7: Rank plans
        # -------------------------------
        ranked = sorted(
            acceptable_plans,
            key=lambda r: (
                r["confidence"],
                r["task_completion"],
                r["projected_battery"],
            ),
            reverse=True,
        )

        for idx, r in enumerate(ranked, start=1):
            r["rank"] = idx

        # -------------------------------
        # STEP 8: Select best plan
        # -------------------------------
        best = ranked[0]

        # -------------------------------
        # STEP 9: Commitment horizon
        # -------------------------------
        if best["confidence"] > 0.8:
            horizon = 3
        elif best["confidence"] > 0.5:
            horizon = 2
        else:
            horizon = 1

        committed_plan = best["plan"][:horizon]

        narration = (
                self._generate_detailed_narration(
                    best,
                    max_tasks,
                    best["confidence_components"]
                )
                + f" Agent commits to next {len(committed_plan)} task(s) before replanning."
        )

        print(f"[NAIVE-PLANNER][agent={agent_id}][NARRATION] {narration}")

        # -------------------------------
        # STEP 10: Return valid plan
        # -------------------------------
        return {
            "agent_id": agent_id,
            "plan": committed_plan,
            "confidence": best["confidence"],
            "projected_battery_remaining": best["projected_battery"],
            "ranking": [
                {
                    "rank": r["rank"],
                    "task_completion": r["task_completion"],
                    "projected_battery": round(r["projected_battery"], 2),
                    "confidence": round(r["confidence"], 4),
                    "efficiency_weight": r.get("efficiency_weight", 0.0),
                }
                for r in ranked
            ],
            "selected_rank": best["rank"],
            "narration": narration,
            "strategy": "naive-multi-greedy-receding",
        }

    def _best_safe_prefix(self, rollout, snapshot, min_battery, agent_id=None):
        """
        Try longest safe prefix of a rollout plan.
        Returns modified rollout or None if nothing feasible.
        """
        full_plan = rollout["plan"]

        for k in range(len(full_plan), 0, -1):
            prefix = full_plan[:k]

            battery = float(self.drone.power.level)   # energy units
            cur_pos = (snapshot["agent"]["col"], snapshot["agent"]["row"])
            station = snapshot.get("nearest_station")

            feasible = True

            for step in prefix:
                pick = tuple(step["pickup"])
                drop = tuple(step["dropoff"])
                weight = step["weight"]

                battery -= self.trip_energy(cur_pos, pick, drop, weight)
                cur_pos = drop

                if battery < min_battery:
                    feasible = False
                    break

            if feasible:
                rollout_copy = dict(rollout)
                rollout_copy["plan"] = prefix
                rollout_copy["task_completion"] = len(prefix)
                rollout_copy["projected_battery"] = battery / float(self.drone.power.capacity) * 100.0
                return rollout_copy
            else:
                # 👇 THIS is where rejection belongs
                if agent_id:
                    print(
                        f"[NAIVE-PLANNER][agent={agent_id}] PREFIX REJECTED: "
                        f"tasks={k:2d}, "
                        f"battery={battery:5.1f} units "
                        f"(min_battery={min_battery})"
                    )

        return None

    # ------------------------------------------------------------------
    # Improved Operational Confidence Calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_operational_confidence_with_components(
            task_completion: int,
            battery_percent: float,
            max_tasks: int,
            base_risk_density: float = 0.02,
            sigmoid_sharpness: float = 3.0,
            sigmoid_offset: float = 0.5,
            epsilon: float = 1e-3
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute operational confidence using IMPROVED robustness model.

        Key Improvement: Risk grows sublinearly (sqrt) to avoid over-penalizing
        multi-task plans. This reflects that operational risks don't scale
        linearly with task count (some efficiencies of scale).

        Mathematical Formulation:
            C = V × P_robust

            where:
            V = T/T_max                         (mission value)
            P_robust = σ(β·(Γ - offset))        (robustness probability)
            Γ = B_buffer / (ρ·√T + ε)           (risk coverage ratio - SUBLINEAR!)
            B_buffer = B/100                    (normalized battery buffer)
            ρ = base risk density               (risk per sqrt(task))

        Parameters:
        -----------
        task_completion : int
            Number of tasks in the plan
        battery_percent : float
            Projected battery remaining after plan execution (0-100%)
        max_tasks : int
            Maximum tasks achievable among all plans
        base_risk_density : float, default=0.02
            Base risk density (scales with sqrt(tasks))
        sigmoid_sharpness : float, default=3.0
            Sigmoid sharpness (β) for robustness probability transition
        sigmoid_offset : float, default=0.5
            Offset for sigmoid (lower = more forgiving)
        epsilon : float, default=1e-3
            Numerical stability term

        Returns:
        --------
        Tuple[float, Dict[str, float]]
            - Confidence score [0, 1]
            - Dictionary with all intermediate calculations
        """
        # Guard clauses
        if max_tasks == 0 or task_completion == 0:
            return 0.0, {
                "mission_value": 0.0,
                "battery_buffer": 0.0,
                "inherent_risk": 0.0,
                "risk_coverage": 0.0,
                "robustness_prob": 0.0,
                "confidence": 0.0
            }

        # 1. Mission Value: Normalized task completion
        mission_value = task_completion / max_tasks

        # 2. Normalized Battery Buffer
        battery_buffer = battery_percent / 100.0

        # 3. Inherent Operational Risk - SUBLINEAR GROWTH (sqrt)
        # This is the key improvement: risk grows with sqrt(tasks), not linearly
        # This reflects that operational risks don't scale linearly
        # (e.g., navigation errors might be similar for 10 vs 16 tasks)
        inherent_risk = base_risk_density * math.sqrt(task_completion)

        # 4. Risk Coverage Ratio
        risk_coverage = battery_buffer / (inherent_risk + epsilon)

        # 5. Robustness Probability (sigmoid with offset)
        # Using offset = 0.5 means we need less buffer for decent robustness
        # This produces more reasonable confidence values
        robustness_prob = 1.0 / (1.0 + math.exp(-sigmoid_sharpness * (risk_coverage - sigmoid_offset)))

        # 6. Operational Confidence
        confidence = mission_value * robustness_prob

        components = {
            "mission_value": round(mission_value, 4),
            "battery_buffer": round(battery_buffer, 4),
            "inherent_risk": round(inherent_risk, 4),
            "risk_coverage": round(risk_coverage, 4),
            "robustness_prob": round(robustness_prob, 4),
            "confidence": round(confidence, 4)
        }

        return confidence, components

    # Backward compatibility wrapper

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _idle_response(self, snapshot: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Construct a standardized IDLE response.

        This represents a benign state where:
        - No work exists in the environment, OR
        - The mission is effectively complete, OR
        - The agent should simply remain parked

        This MUST NOT be used for failure conditions.
        Failures must return: {"mode": "abort", ...}

        Parameters:
        -----------
        snapshot : Dict[str, Any]
            Current world snapshot
        reason : str
            Human-readable explanation for idling

        Returns:
        --------
        Dict[str, Any]
            Idle response payload
        """
        return {
            "mode": "idle",
            "agent_id": snapshot["agent"].get("id", "unknown"),
            "plan": [],
            "confidence": 0.0,
            "projected_battery_remaining": self.drone.power.percent(),
            "ranking": [],
            "selected_rank": None,
            "narration": reason,
            "strategy": "naive-multi-greedy",
        }

    @staticmethod
    def _generate_detailed_narration(plan_info: Dict[str, Any],
                                     max_tasks: int,
                                     components: Dict[str, float]) -> str:
        """
        Generate comprehensive human-readable explanation of plan selection.

        Parameters:
        -----------
        plan_info : Dict[str, Any]
            Selected plan information
        max_tasks : int
            Maximum tasks possible
        components : Dict[str, float]
            Confidence calculation components

        Returns:
        --------
        str
            Detailed narration explaining the selection
        """
        task_completion = plan_info["task_completion"]
        battery = plan_info["projected_battery"]
        confidence = plan_info["confidence"]

        # Confidence level with precise ranges
        if confidence >= 0.8:
            conf_level = "very high"
            robustness_desc = "excellent robustness to operational uncertainties"
        elif confidence >= 0.6:
            conf_level = "high"
            robustness_desc = "good robustness to operational uncertainties"
        elif confidence >= 0.4:
            conf_level = "moderate"
            robustness_desc = "reasonable robustness to operational uncertainties"
        elif confidence >= 0.2:
            conf_level = "low"
            robustness_desc = "limited robustness to operational uncertainties"
        else:
            conf_level = "very low"
            robustness_desc = "minimal robustness to operational uncertainties"

        # Battery analysis
        if battery >= 40:
            battery_desc = "high"
            return_desc = "excellent probability of returning to station"
        elif battery >= 25:
            battery_desc = "moderate"
            return_desc = "good probability of returning to station"
        elif battery >= 15:
            battery_desc = "low"
            return_desc = "adequate probability of returning to station"
        elif battery >= 10:
            battery_desc = "very low"
            return_desc = "marginal probability of returning to station"
        else:
            battery_desc = "critical"
            return_desc = "high risk of failing to return to station"

        # Mission accomplishment
        if max_tasks > 0:
            task_ratio = task_completion / max_tasks
            if task_ratio >= 0.9:
                accomplishment = "excellent"
                task_desc = "nearly maximum possible task completion"
            elif task_ratio >= 0.7:
                accomplishment = "good"
                task_desc = "substantial task completion"
            elif task_ratio >= 0.5:
                accomplishment = "moderate"
                task_desc = "moderate task completion"
            elif task_ratio >= 0.3:
                accomplishment = "limited"
                task_desc = "limited task completion"
            else:
                accomplishment = "minimal"
                task_desc = "minimal task completion"
        else:
            accomplishment = "none"
            task_desc = "no task completion"

        # Risk coverage analysis
        risk_coverage = components.get("risk_coverage", 0.0)
        if risk_coverage >= 2.0:
            risk_desc = "substantial operational risk buffer"
        elif risk_coverage >= 1.5:
            risk_desc = "adequate operational risk buffer"
        elif risk_coverage >= 1.0:
            risk_desc = "minimal operational risk buffer"
        elif risk_coverage >= 0.5:
            risk_desc = "insufficient operational risk buffer"
        else:
            risk_desc = "critically insufficient operational risk buffer"

        narration = (
            f"Selected plan achieves {accomplishment} mission accomplishment "
            f"({task_completion} of {max_tasks} possible tasks, {task_desc}) "
            f"with {conf_level} operational confidence ({confidence:.3f}). "
            f"Projected battery after execution: {battery:.1f}% ({battery_desc} level), "
            f"providing {robustness_desc}. "
            f"Risk coverage ratio: {risk_coverage:.2f} ({risk_desc}). "
            f"This plan has {return_desc}."
        )

        return narration

    # ------------------------------------------------------------------
    # Core simulation logic (unchanged from original)
    # ------------------------------------------------------------------

    def _simulate(
            self,
            snapshot: Dict[str, Any],
            shuffle: bool,
            known_weight: bool = True,
            efficiency_weight: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Same as original - generates plan candidates."""
        agent = snapshot["agent"]

        # Filter available parcels (not picked or delivered)
        parcels = [
            dict(p)
            for p in snapshot.get("all_parcels", [])
            if not p.get("picked") and not p.get("delivered")
        ]

        if shuffle:
            getattr(self, "_rng", random).shuffle(parcels)

        # Initial state
        start_battery = float(self.drone.power.level)   # energy units
        battery = start_battery
        cur_pos = (agent["col"], agent["row"])
        station = snapshot.get("nearest_station")

        plan: List[Dict[str, Any]] = []

        # Greedy parcel selection loop
        while parcels and battery > self.FEASIBILITY_MARGIN:
            scored_parcels = []

            # Score each feasible parcel
            for p in parcels:
                pick_pos = (p["col"], p["row"])
                drop_pos = self._choose_drop_cell(pick_pos, station)

                true_weight = float(p.get("weight", 1.0))
                est_weight = true_weight if known_weight else self.NOMINAL_WEIGHT

                # The energy of the whole trip (pickup leg, delivery leg, pick and drop)
                total_cost = self.trip_energy(cur_pos, pick_pos, drop_pos, est_weight)

                # Feasibility check
                if battery < total_cost + self.FEASIBILITY_MARGIN:
                    continue  # Insufficient battery

                # Efficiency-weighted scoring
                # Higher score = more desirable parcel
                task_value = 1.0  # Each parcel contributes 1 task
                efficiency = 1.0 / max(0.1, total_cost + self.time_weight * self.trip_time(cur_pos, pick_pos, drop_pos))  # Avoid division by zero

                # Combined score: weighted average of task value and efficiency
                score = (efficiency_weight * task_value +
                         (1 - efficiency_weight) * efficiency)

                scored_parcels.append({
                    "parcel": p,
                    "pick_pos": pick_pos,
                    "drop_pos": drop_pos,
                    "true_weight": true_weight,
                    "total_cost": total_cost,
                    "score": score,
                })

            # If no feasible parcels, stop planning
            if not scored_parcels:
                break

            # Select highest-scoring parcel
            best = max(scored_parcels, key=lambda x: x["score"])

            # Add to plan and update state
            plan.append({
                "pickup": [best["pick_pos"][0], best["pick_pos"][1]],
                "dropoff": [best["drop_pos"][0], best["drop_pos"][1]],
                "weight": best["true_weight"],
            })

            battery -= best["total_cost"]
            cur_pos = best["drop_pos"]

            # Remove selected parcel from consideration
            parcels.remove(best["parcel"])

        # reported as a percentage of capacity (the confidence and narration code works in percent)
        projected_battery_remaining = max(0.0, battery) / float(self.drone.power.capacity) * 100.0
        return plan, projected_battery_remaining

    # ------------------------------------------------------------------
    # Reactive fallback logic (unchanged from original)
    # ------------------------------------------------------------------

    def choose_best_feasible_next_parcel(
            self,
            *,
            progress: dict,
            parcels: list[dict],
            station: dict,
            known_weight: bool = True,
            safety_margin: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Reactive fallback: select the next feasible parcel and emit
        execution evidence for posterior confidence analysis.
        """
        current_cell = progress["current_cell"]
        battery = float(progress.get("battery_energy", progress["battery_pct"]))   # energy units
        if safety_margin is None:
            safety_margin = self.FEASIBILITY_MARGIN

        attempted = progress.get("attempted_parcels", set())
        failed = progress.get("failed_parcels", set())

        NOMINAL_WEIGHT = 1.0
        candidates = []

        for p in parcels:
            pid = p.get("id")

            # Skip unavailable or previously attempted parcels
            if p.get("picked") or p.get("delivered"):
                continue
            if pid in attempted or pid in failed:
                continue

            pick = (int(p["col"]), int(p["row"]))
            drop = self._choose_drop_cell(pick, station)

            true_weight = float(p.get("weight", NOMINAL_WEIGHT))
            est_weight = true_weight if known_weight else NOMINAL_WEIGHT

            # Estimated energy cost
            est_cost = self.trip_energy(current_cell, pick, drop, est_weight)

            # Feasibility check
            if battery < est_cost + safety_margin:
                continue

            evidence = self._compute_reactive_evidence(
                battery_before=battery,
                estimated_cost=est_cost,
                safety_margin=safety_margin,
                current_cell=current_cell,
                pickup_cell=pick,
            )

            candidates.append({
                "parcel": p,
                "pickup": pick,
                "dropoff": drop,
                "true_weight": true_weight,
                "estimated_cost": est_cost,
                "evidence": evidence,
            })

        if not candidates:
            return None

        # Select nearest feasible parcel, then lowest cost
        candidates.sort(
            key=lambda c: (
                self.physics.distance(a=current_cell, b=c["pickup"]),
                c["estimated_cost"],
            )
        )

        chosen = candidates[0]

        return {
            "id": chosen["parcel"].get("id"),
            "pickup": [chosen["pickup"][0], chosen["pickup"][1]],
            "dropoff": [chosen["dropoff"][0], chosen["dropoff"][1]],
            "weight": chosen["true_weight"],
            "estimated_cost": chosen["estimated_cost"],
            "posterior_evidence": chosen["evidence"],
        }

    def _compute_reactive_evidence(
            self,
            *,
            battery_before: float,
            estimated_cost: float,
            safety_margin: float,
            current_cell: tuple[int, int],
            pickup_cell: tuple[int, int],
    ) -> dict:
        """
        Extract execution evidence for posterior confidence analysis.
        This method makes no judgement. It only reports facts.
        """
        expected_remaining = battery_before - estimated_cost
        travel_distance = self.physics.distance(a=current_cell, b=pickup_cell)

        return {
            "battery_before": round(battery_before, 3),
            "estimated_cost": round(estimated_cost, 3),
            "expected_remaining": round(expected_remaining, 3),
            "safety_margin": safety_margin,
            "buffer_ok": expected_remaining >= safety_margin,
            "distance_to_pickup": travel_distance,
        }

    @staticmethod
    def compute_posterior_confidence_modifier(
            *,
            prior_confidence: float,
            evidence: dict,
    ) -> float:
        """
        Compute posterior confidence modifier after reactive fallback.

        Rules:
        - No decay if safety margin is preserved
        - Mild decay if buffer is reduced
        - Never increases confidence
        """
        if evidence.get("buffer_ok", False):
            modifier = 1.0
            reason = "energy buffer preserved"
        else:
            modifier = 0.95
            reason = "energy buffer reduced"

        print(
            "[NAIVE-PLANNER][POSTERIOR] "
            f"prior={prior_confidence:.3f} "
            f"modifier={modifier:.3f} "
            f"remaining={evidence['expected_remaining']:.2f}% "
            f"→ {reason}"
        )

        return modifier
