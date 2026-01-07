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
from .utils import manhattan, est_cost_cells


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

    def __init__(self, num_trials: int = 10):
        """
        Initialize the naive strategy with operational robustness.

        Parameters:
        -----------
        num_trials : int, default=10
            Number of greedy rollouts to generate per planning cycle.
            Higher values increase exploration of task-battery trade-offs.
        """
        self.num_trials = num_trials

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

        Process:
        1. Check if any parcels exist at all
        2. If no parcels, return empty plan (controller will park)
        3. Generate multiple greedy rollouts with varying efficiency trade-offs
        4. Compute operational confidence for each rollout using improved robustness model
        5. Select plan with highest confidence
        6. Return detailed ranking with comprehensive confidence rationale

        Key Improvements from Original:
        - Sublinear risk growth (sqrt) to avoid over-penalizing multi-task plans
        - Reasonable confidence thresholds that work across battery levels
        - Clear, complete logging showing all calculations
        - No dynamic thresholds that break abstraction

        Parameters:
        -----------
        snapshot : Dict[str, Any]
            World snapshot containing agent state and available parcels.

        Returns:
        --------
        Dict[str, Any]
            Plan response containing:
            - plan: selected sequence of pickup/drop actions (empty if no feasible plans)
            - confidence: operational robustness score [0, 1]
            - projected_battery_remaining: estimated battery after execution
            - ranking: full ranking of all evaluated rollouts
            - narration: detailed human-readable explanation of selection
        """
        agent_id = snapshot["agent"].get("id", "unknown")
        current_battery = float(snapshot["agent"]["battery_pct"])
        N = self.num_trials

        print(f"[NAIVE-PLANNER][agent={agent_id}] Planning with {current_battery:.1f}% battery")

        # -------------------------------------------------
        # Operational Robustness Parameters
        # -------------------------------------------------
        # These are calibrated to produce reasonable confidence values
        # Risk grows sublinearly (sqrt) to avoid over-penalizing multi-task plans
        BASE_RISK_DENSITY = 0.02  # Base risk per task (2%)
        SIGMOID_SHARPNESS = 3.0  # Reasonable transition sharpness
        SIGMOID_OFFSET = 0.5  # Offset for sigmoid (0.5 means we need less buffer)
        MIN_CONFIDENCE = 0.2  # Minimum acceptable confidence
        MIN_SAFE_BATTERY = 10.0  # Absolute minimum battery after execution
        EPSILON = 1e-3

        # -------------------------------------------------
        # STEP 1: Check for available parcels
        # -------------------------------------------------
        parcels_available = [
            dict(p)
            for p in snapshot.get("all_parcels", [])
            if not p.get("picked") and not p.get("delivered")
        ]

        if not parcels_available:
            print(f"[NAIVE-PLANNER][agent={agent_id}] NO PARCELS: No parcels available for delivery.")
            return self._empty_plan_response(
                snapshot,
                "No parcels available for delivery."
            )

        print(f"[NAIVE-PLANNER][agent={agent_id}] Found {len(parcels_available)} available parcels")

        # -------------------------------------------------
        # STEP 2: Quick feasibility check
        # -------------------------------------------------
        if not self._any_parcel_feasible(snapshot, parcels_available):
            print(f"[NAIVE-PLANNER][agent={agent_id}] NO FEASIBLE PARCELS: Insufficient battery for any delivery.")
            return self._empty_plan_response(
                snapshot,
                "No feasible parcels (insufficient battery for any delivery)."
            )

        # -------------------------------------------------
        # STEP 3: Generate greedy rollouts
        # -------------------------------------------------
        print(f"[NAIVE-PLANNER][agent={agent_id}] Generating {N} greedy rollouts...")
        rollouts: List[Dict[str, Any]] = []

        for i in range(N):
            efficiency_weight = i / max(1, N - 1)

            plan, projected_battery = self._simulate(
                snapshot,
                shuffle=True,
                efficiency_weight=efficiency_weight
            )

            # Only consider non-empty plans
            if plan:
                rollouts.append({
                    "trial": i + 1,
                    "task_completion": len(plan),
                    "projected_battery": projected_battery,
                    "plan": plan,
                    "efficiency_weight": round(efficiency_weight, 2),
                })

                print(
                    f"[NAIVE-PLANNER][agent={agent_id}][trial {i + 1}/{N}] "
                    f"efficiency_weight={efficiency_weight:.2f} → "
                    f"tasks={len(plan):2d}, "
                    f"projected_battery={projected_battery:5.1f}%"
                )
            else:
                print(
                    f"[NAIVE-PLANNER][agent={agent_id}][trial {i + 1}/{N}] "
                    f"efficiency_weight={efficiency_weight:.2f} → "
                    f"NO FEASIBLE PLAN"
                )

        # -------------------------------------------------
        # STEP 4: Handle case where all rollouts are empty
        # -------------------------------------------------
        if not rollouts:
            print(f"[NAIVE-PLANNER][agent={agent_id}] ALL ROLLOUTS EMPTY: No plans could be generated.")
            return self._empty_plan_response(
                snapshot,
                "No delivery plans could be generated (all rollouts failed)."
            )

        # -------------------------------------------------
        # STEP 5: Compute confidence for each rollout
        # -------------------------------------------------
        max_tasks = max(r["task_completion"] for r in rollouts)
        print(f"[NAIVE-PLANNER][agent={agent_id}] Maximum tasks among rollouts: {max_tasks}")

        for r in rollouts:
            confidence, components = self._compute_operational_confidence_with_components(
                task_completion=r["task_completion"],
                battery_percent=r["projected_battery"],
                max_tasks=max_tasks,
                base_risk_density=BASE_RISK_DENSITY,
                sigmoid_sharpness=SIGMOID_SHARPNESS,
                sigmoid_offset=SIGMOID_OFFSET,
                epsilon=EPSILON
            )

            r["confidence"] = confidence
            r["confidence_components"] = components

        # -------------------------------------------------
        # STEP 6: Filter and rank plans
        # -------------------------------------------------
        # Filter plans that are too risky or have insufficient battery
        acceptable_plans = []
        for r in rollouts:
            if r["confidence"] >= MIN_CONFIDENCE and r["projected_battery"] >= MIN_SAFE_BATTERY:
                acceptable_plans.append(r)
            else:
                print(
                    f"[NAIVE-PLANNER][agent={agent_id}] REJECTED PLAN: "
                    f"tasks={r['task_completion']:2d}, "
                    f"battery={r['projected_battery']:5.1f}%, "
                    f"confidence={r['confidence']:.3f} "
                    f"(min_conf={MIN_CONFIDENCE}, min_battery={MIN_SAFE_BATTERY})"
                )

        if not acceptable_plans:
            print(
                f"[NAIVE-PLANNER][agent={agent_id}] NO ACCEPTABLE PLANS: All plans rejected by confidence or battery thresholds.")
            return self._empty_plan_response(
                snapshot,
                f"No acceptable plans found (all plans below confidence {MIN_CONFIDENCE} or battery {MIN_SAFE_BATTERY}% threshold)."
            )

        # Rank by confidence (primary), then tasks, then battery
        ranked = sorted(
            acceptable_plans,
            key=lambda r: (
                r["confidence"],  # Highest confidence first
                r["task_completion"],  # More tasks at same confidence
                r["projected_battery"],  # More battery at same confidence/tasks
            ),
            reverse=True
        )

        # Add rank numbers
        for idx, r in enumerate(ranked, start=1):
            r["rank"] = idx

        # -------------------------------------------------
        # STEP 7: Log detailed ranking
        # -------------------------------------------------
        print(f"[NAIVE-PLANNER][agent={agent_id}] FINAL RANKING OF ACCEPTABLE PLANS:")
        print(
            f"[NAIVE-PLANNER][agent={agent_id}] {'Rank':<4} {'Tasks':<6} {'Battery':<8} {'Confidence':<10} {'RiskCoverage':<12} {'RobustProb':<10}")
        print(f"[NAIVE-PLANNER][agent={agent_id}] {'-' * 4} {'-' * 6} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 10}")

        for r in ranked[:5]:  # Show top 5
            components = r["confidence_components"]
            marker = " ← SELECTED" if r["rank"] == 1 else ""

            print(
                f"[NAIVE-PLANNER][agent={agent_id}] "
                f"{r['rank']:<4} "
                f"{r['task_completion']:<6} "
                f"{r['projected_battery']:<8.1f}% "
                f"{r['confidence']:<10.3f} "
                f"{components['risk_coverage']:<12.3f} "
                f"{components['robustness_prob']:<10.3f}"
                f"{marker}"
            )

        # -------------------------------------------------
        # STEP 8: Select best plan
        # -------------------------------------------------
        best = ranked[0]
        components = best["confidence_components"]

        print(f"[NAIVE-PLANNER][agent={agent_id}] SELECTED BEST PLAN:")
        print(
            f"[NAIVE-PLANNER][agent={agent_id}]   Tasks: {best['task_completion']}/{max_tasks} (Mission Value: {components['mission_value']:.3f})")
        print(
            f"[NAIVE-PLANNER][agent={agent_id}]   Projected Battery: {best['projected_battery']:.1f}% (Normalized Buffer: {components['battery_buffer']:.3f})")
        print(
            f"[NAIVE-PLANNER][agent={agent_id}]   Inherent Risk: {components['inherent_risk']:.3f} (Risk Density: {BASE_RISK_DENSITY:.3f} × sqrt({best['task_completion']}))")
        print(
            f"[NAIVE-PLANNER][agent={agent_id}]   Risk Coverage Ratio: {components['risk_coverage']:.3f} (Buffer/Risk)")
        print(f"[NAIVE-PLANNER][agent={agent_id}]   Robustness Probability: {components['robustness_prob']:.3f}")
        print(f"[NAIVE-PLANNER][agent={agent_id}]   FINAL CONFIDENCE: {best['confidence']:.3f}")

        # -------------------------------------------------
        # STEP 9: Generate comprehensive narration
        # -------------------------------------------------
        narration = self._generate_detailed_narration(best, max_tasks, components)

        # -------------------------------------------------
        # STEP 10: Prepare response
        # -------------------------------------------------
        ranking_summary = [
            {
                "rank": r["rank"],
                "task_completion": r["task_completion"],
                "projected_battery": round(r["projected_battery"], 2),
                "confidence": round(r["confidence"], 4),
                "efficiency_weight": r.get("efficiency_weight", 0.0),
            }
            for r in ranked
        ]

        return {
            "agent_id": agent_id,
            "plan": best["plan"],
            "confidence": best["confidence"],
            "projected_battery_remaining": best["projected_battery"],
            "ranking": ranking_summary,
            "selected_rank": best["rank"],
            "narration": narration,
            "strategy": "naive-multi-greedy",
        }

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
    def _compute_operational_confidence(self,
                                        task_completion: int,
                                        battery_percent: float,
                                        max_tasks: int,
                                        risk_density: float = 0.02,
                                        sigmoid_sharpness: float = 3.0,
                                        epsilon: float = 1e-3) -> float:
        """Legacy wrapper for backward compatibility."""
        confidence, _ = self._compute_operational_confidence_with_components(
            task_completion=task_completion,
            battery_percent=battery_percent,
            max_tasks=max_tasks,
            base_risk_density=risk_density,
            sigmoid_sharpness=sigmoid_sharpness,
            sigmoid_offset=0.5,
            epsilon=epsilon
        )
        return confidence

    def _get_confidence_components(self,
                                   task_completion: int,
                                   battery_percent: float,
                                   max_tasks: int,
                                   risk_density: float = 0.02,
                                   sigmoid_sharpness: float = 3.0,
                                   epsilon: float = 1e-3) -> Dict[str, float]:
        """Legacy wrapper for backward compatibility."""
        _, components = self._compute_operational_confidence_with_components(
            task_completion=task_completion,
            battery_percent=battery_percent,
            max_tasks=max_tasks,
            base_risk_density=risk_density,
            sigmoid_sharpness=sigmoid_sharpness,
            sigmoid_offset=0.5,
            epsilon=epsilon
        )
        return components

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_plan_response(snapshot: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Create a standardized empty plan response.

        Parameters:
        -----------
        snapshot : Dict[str, Any]
            Current world snapshot
        reason : str
            Detailed human-readable reason for empty plan

        Returns:
        --------
        Dict[str, Any]
            Empty plan response
        """
        return {
            "agent_id": snapshot["agent"].get("id", "unknown"),
            "plan": [],
            "confidence": 0.0,
            "projected_battery_remaining": snapshot["agent"]["battery_pct"],
            "ranking": [],
            "selected_rank": None,
            "narration": reason,
            "strategy": "naive-multi-greedy",
        }

    @staticmethod
    def _any_parcel_feasible(snapshot: Dict[str, Any], parcels: List[Dict[str, Any]]) -> bool:
        """
        Quick check if ANY parcel can be delivered with current battery.

        Parameters:
        -----------
        snapshot : Dict[str, Any]
            World snapshot
        parcels : List[Dict[str, Any]]
            Available parcels to check

        Returns:
        --------
        bool
            True if at least one parcel is feasible, False otherwise
        """
        if not parcels:
            return False

        agent = snapshot["agent"]
        current_battery = float(agent["battery_pct"])
        SAFETY_MARGIN = 3.0
        NOMINAL_WEIGHT = 1.0

        cur_pos = (agent["col"], agent["row"])
        station = snapshot.get("nearest_station")

        # Quick drop estimate (station center)
        def quick_drop(pick_pos):
            if not station:
                return pick_pos
            center_col = station["col"] + station["w"] // 2
            center_row = station["row"] + station["h"] // 2
            return (center_col, center_row)

        # Check first few parcels (avoid O(n^2) in planning phase)
        for p in parcels[:min(10, len(parcels))]:
            pick_pos = (p["col"], p["row"])
            drop_pos = quick_drop(pick_pos)

            # Conservative energy estimate
            pickup_dist = manhattan(cur_pos, pick_pos)
            delivery_dist = manhattan(pick_pos, drop_pos)
            weight = float(p.get("weight", NOMINAL_WEIGHT))

            pickup_cost = est_cost_cells(pickup_dist, 0.0)
            delivery_cost = est_cost_cells(delivery_dist, weight)
            total_cost = pickup_cost + delivery_cost

            if current_battery >= total_cost + SAFETY_MARGIN:
                return True

        return False

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
            random.shuffle(parcels)

        # Initial state
        start_battery = float(agent["battery_pct"])
        battery = start_battery
        cur_pos = (agent["col"], agent["row"])
        station = snapshot.get("nearest_station")

        plan: List[Dict[str, Any]] = []
        NOMINAL_WEIGHT = 1.0
        SAFETY_MARGIN = 3.0  # Minimum battery buffer

        # Greedy parcel selection loop
        while parcels and battery > SAFETY_MARGIN:
            scored_parcels = []

            # Score each feasible parcel
            for p in parcels:
                pick_pos = (p["col"], p["row"])
                drop_pos = self._choose_drop_cell(pick_pos, station)

                true_weight = float(p.get("weight", 1.0))
                est_weight = true_weight if known_weight else NOMINAL_WEIGHT

                # Calculate energy costs
                pickup_dist = manhattan(cur_pos, pick_pos)
                delivery_dist = manhattan(pick_pos, drop_pos)

                pickup_cost = est_cost_cells(pickup_dist, 0.0)
                delivery_cost = est_cost_cells(delivery_dist, est_weight)
                total_cost = pickup_cost + delivery_cost

                # Feasibility check
                if battery < total_cost + SAFETY_MARGIN:
                    continue  # Insufficient battery

                # Efficiency-weighted scoring
                # Higher score = more desirable parcel
                task_value = 1.0  # Each parcel contributes 1 task
                efficiency = 1.0 / max(0.1, total_cost)  # Avoid division by zero

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

        projected_battery_remaining = max(0.0, battery)
        return plan, projected_battery_remaining

    def _choose_drop_cell(
            self,
            pick: Tuple[int, int],
            station: Optional[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Same as original - selects drop cell."""
        if not station:
            return pick

        # Generate all cells within station bounds
        cells: List[Tuple[int, int]] = [
            (c, r)
            for r in range(station["row"], station["row"] + station["h"])
            for c in range(station["col"], station["col"] + station["w"])
        ]

        # Compute distance-based weights
        distances = [manhattan(pick, cell) for cell in cells]
        weights = [1.0 / (d + 1.0) for d in distances]

        return random.choices(cells, weights=weights, k=1)[0]

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
            safety_margin: float = 3.0,
    ) -> Optional[Dict]:
        """
        Reactive fallback: select the next feasible parcel and emit
        execution evidence for posterior confidence analysis.
        """
        current_cell = progress["current_cell"]
        battery = float(progress["battery_pct"])

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
            est_cost = (
                    est_cost_cells(manhattan(current_cell, pick), 0.0)
                    + est_cost_cells(manhattan(pick, drop), est_weight)
            )

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
                manhattan(current_cell, c["pickup"]),
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

    @staticmethod
    def _compute_reactive_evidence(
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
        travel_distance = manhattan(current_cell, pickup_cell)

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


