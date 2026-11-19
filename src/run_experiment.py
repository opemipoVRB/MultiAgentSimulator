# src/run_experiment.py
"""
GUI-only experiment runner for the Drone Agent simulator.

Usage:
  python src/run_experiment.py

Behavior:
 - Everything driven via GUI screens.
 - Mirrors visuals from games.py to keep look and feel identical.
 - Experiment JSONs: created as temp_<run_id>.json while running, renamed to experiment_<run_id>.json on success.
 - Resumes the most recent unfinished temp_*.json or experiment_*.json when possible.
 - Supports per-strategy snapshots so an aborted run can resume from the exact world state.
 - Stores agents and comm_config; snapshot contains agents_state list.
"""
import json
import os
import random
import time
import uuid
from collections import Counter

import pygame
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from src.artifacts import Terrain, Drone, Parcel
from src.config import OLLAMA_MODE
from src.controllers import HumanAgentController, AIAgentController, ControllerSwitcher
from src.network.profiles import DEFAULT_PROFILE, apply_profile
from src.process_world import RemoteController, allocate_cores
from src.sensor import DEFAULT_SENSING, SensorModel
from src.world_port import WorldPort
from src.observability import FlightRecorder, MetricsAccumulator
from src.physics.physics_2d import Grid2DPhysics
from src.replay import ReplayViewer, has_telemetry
from src.strategies.sppdec.personalities import SPPDEC_PERSONALITIES
from src.system.abort import AbortReason
from src.utils import load_image, scale_to_cell
from src.system.capacity import SystemCapacityChecker, AgentCapacityError
from src import sim_clock
from enum import Enum

# ---------------------------------------------------------------------------
#  Physics Coonfig
# ---------------------------------------------------------------------------

physics = Grid2DPhysics()
NAIVE_NUM_TRIALS = 10  # Monte Carlo greedy rollouts per agent

GRID_SIZE = sim_clock.CELL_PX
SPEED = sim_clock.DRONE_SPEED_PX_S
# Hover power as a fraction of the power of flying empty. 1.0: a hovering multirotor draws about as much as it does in flight
# (hover power is the upper bound of flight power, Dorling et al., arXiv 1608.02305). Sources: .ignore/revision/simulated_hardware.md.
HOVER_POWER_RATIO = 1.0
DRONE_SCALE = 1.5
PARCEL_SCALE = 0.70
ANIM_FPS = 12
FLASH_DURATION = 0.15

BASE_DIR = Path(__file__).parent
GRAPHICS_DIR = BASE_DIR / "graphics"
EXPERIMENTS_DIR = BASE_DIR.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_PREFIX = "temp_"
EXPERIMENT_PREFIX = "experiment_"

USE_SWEEP = True

# With the fixed timestep, a run with parcels remaining in which no drone picks or
# drops a parcel for this many steps (60 simulated seconds) is recorded as "stalled".
STALL_STEPS = 60 * 60

try:
    from src.observability.configs import SCALING_CONFIG, generate_station_scaling
except ImportError:
    SCALING_CONFIG = None
    generate_station_scaling = None


def deterministic_parcel_positions(seed: Optional[int], n_parcels: int, cols: int, rows: int,
                                   forbidden_cells: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
    rng = random.Random(seed if seed is not None else int(time.time()))
    all_cells = [(c, r) for r in range(rows) for c in range(cols)]
    if forbidden_cells:
        forbidden = set(forbidden_cells)
        all_cells = [c for c in all_cells if c not in forbidden]
    rng.shuffle(all_cells)
    return all_cells[:max(0, min(n_parcels, len(all_cells)))]


def _base_strategy_vault(selected_planner: str) -> List[Dict]:
    """Strategy specs run for every scaling configuration (one entry per arm)."""
    return [
        # -------------------------------
        # Decentralized Experimental Variants
        # -------------------------------
        {
            "name": "naive_decentralized",
            "strategy_type": "naive",
            "variant": "reactive",
            "topology": "decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Reactive decentralized baseline. Agents greedily select tasks using only local perception without coordination, prediction, or communication."
        },
        {
            "name": "decentralized_auction",
            "variant": "auction_baseline",
            "strategy_type": "auction",
            "topology": "decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Decentralized auction baseline with explicit message passing. Agents broadcast bids for tasks and resolve conflicts through communication."
        },
        {
            "name": "cbba_single_task",
            "variant": "cbba_single",
            "strategy_type": "cbba",
            "topology": "decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "CBBA variant limited to single task allocation. Tests consensus-based coordination under restricted bundle formation."
        },
        {
            "name": "cbba_full",
            "variant": "cbba_bundle",
            "strategy_type": "cbba",
            "topology": "decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Full CBBA implementation with bundle construction and consensus convergence. Represents structured decentralized coordination with communication."
        },

        # -------------------------------
        # Centralized Experimental Variants
        # -------------------------------
        {
            "name": "centralized_cc_greedy",
            "variant": "greedy",
            "strategy_type": "centralized",
            "topology": "centralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Centralized controller using greedy assignment. Global information available but allocation performed using local-minimum heuristic."
        },
        {
            "name": "centralized_cc_hungarian",
            "variant": "hungarian",
            "strategy_type": "centralized",
            "topology": "centralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Centralized optimal task allocation using the Hungarian algorithm. Provides upper-bound performance baseline under full observability."
        },
        {
            "name": "centralized_cc",
            "variant": "auction",
            "strategy_type": "centralized",
            "topology": "centralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Centralized auction mechanism. Tasks are globally coordinated via a single auctioneer that assigns parcels based on agent bids."
        },

        # -------------------------------
        # SPPDEC Experimental Variants
        # -------------------------------
        {
            "name": "sppdec",
            "variant": "control",  # baseline_payload=0.0, alpha=0.0
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC with SLM policy selection but without payload-aware self modeling and without peer energy penalties. Uses perception, feasibility, and learned uncertainty only."
        },
        {
            "name": "sppdec",
            "variant": "baseline_only",  # baseline_payload=0.15, alpha=0.0
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC with SLM policy selection and payload-aware self-cost modeling, but without fairness pressure. Tests the effect of improved self modeling alone."
        },
        {
            "name": "sppdec",
            "variant": "fairness_only",  # baseline_payload=0.0, alpha=0.02
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC with SLM policy selection and belief-based peer energy penalties, but without payload-aware self modeling. Tests fairness pressure from inferred peer workload."
        },
        {
            "name": "sppdec",
            "variant": "combined",  # baseline_payload=0.15, alpha=0.02
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Full SPPDEC Phase 1. Uses SLM policy selection with both payload-aware self modeling and belief-based peer energy penalties for adaptive decentralized coordination."
        },
        {
            "name": "sppdec",
            "variant": "personality_mix_balanced_quartet",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Is complete social coverage valuable without extremes?",
            "agent_personalities": [
                "rational",
                "selfish",
                "altruistic",
                "reckless"
            ],
            "personality_distribution": {
                "rational": 1,
                "selfish": 1,
                "altruistic": 1,
                "reckless": 1,
            }
        },
        {
            "name": "sppdec",
            "variant": "personality_mix_no_reckless",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Heterogeneous SPPDEC team with mixed personalities",
            "agent_personalities": [
                "rational",
                "selfish",
                "altruistic",
                "reckless"
            ],
            "personality_distribution": {
                "rational": 2,
                "selfish": 1,
                "altruistic": 1,
                "reckless": 0,
            }
        },
        {
            "name": "sppdec",
            "variant": "personality_mix_controlled_chaos",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Find the sweet spot between too much and too little disruption. Double Rationals to mediate, Selfish for efficiency, Reckless as catalytic disruptor",
            "agent_personalities": [
                "rational",
                "selfish",
                "altruistic",
                "reckless"
            ],
            "personality_distribution": {
                "rational": 2,
                "selfish": 1,
                "altruistic": 0,
                "reckless": 1,
            }
        },
        {
            "name": "sppdec",
            "variant": "personality_mix_3_brains_1_wildcard",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Test if Rational majority can fully harness Reckless's catalytic power",
            "agent_personalities": [
                "rational",
                "selfish",
                "altruistic",
                "reckless"
            ],
            "personality_distribution": {
                "rational": 3,
                "selfish": 0,
                "altruistic": 0,
                "reckless": 1,
            }
        },
        {
            "name": "sppdec",
            "variant": "personality_mix_efficiency_overdrive",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "What happens when efficiency pressure meets disruption?",
            "agent_personalities": [
                "rational",
                "selfish",
                "altruistic",
                "reckless"
            ],
            "personality_distribution": {
                "rational": 1,
                "selfish": 2,
                "altruistic": 0,
                "reckless": 1,
            }
        },
        # -------------------------------
        # Revision: arbitration ablation on the Combined configuration (R1.3, R3.5)
        # -------------------------------
        {
            "name": "sppdec",
            "variant": "arb_always_greedy",
            "config_variant": "combined",
            "arbiter": "always_greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC Combined with the greedy candidate every cycle: no pre-filter, no SLM."
        },
        {
            "name": "sppdec",
            "variant": "arb_always_hungarian",
            "config_variant": "combined",
            "arbiter": "always_hungarian",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC Combined with the Hungarian (belief-adjusted) candidate every cycle: no pre-filter, no SLM."
        },
        {
            "name": "sppdec",
            "variant": "arb_rule_based",
            "config_variant": "combined",
            "arbiter": "rule_based",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC Combined with the pre-filter followed by the SLM prompt's decision rules in code instead of the SLM."
        },
        # -------------------------------
        # Revision: arbitration SLM switched to Qwen3-VL 8B Instruct with complete rules (T9.1)
        # -------------------------------
        {
            "name": "sppdec",
            "variant": "combined_qwen3vl_v2",
            "config_variant": "combined",
            "slm_model": "qwen3-vl:8b-instruct",
            "prompt_version": "cost_gap_closed",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC Combined; SLM arbiter Qwen3-VL 8B Instruct with arbitration rules v2 (cost premium never decides on its own)."
        },
        {
            "name": "sppdec",
            "variant": "combined_qwen3vl_paired",
            "config_variant": "combined",
            "slm_model": "qwen3-vl:8b-instruct",
            "prompt_version": "paired_steps",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC Combined; SLM arbiter Qwen3-VL 8B Instruct with the paired_steps prompt (rules first, facts computed in code, two rule steps per answer field)."
        },
        # -------------------------------
        # Revision: reasoning-model prototypes (shared reasoning architecture)
        # -------------------------------
        {
            "name": "sppdec",
            "variant": "cbba_assumed_bids_asym",
            "config_variant": "combined",
            "arbiter": "always_cbba",
            "symmetric_costs": False,
            "exclude_carrying_peers": False,
            "no_slot_policy": "greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "CBBA simulated in each drone's head on the same (asymmetric) cost matrix as the Hungarian arm: isolates the algorithm."
        },
        {
            "name": "sppdec",
            "variant": "hungarian_symmetric",
            "config_variant": "combined",
            "arbiter": "always_hungarian",
            "symmetric_costs": True,
            "exclude_carrying_peers": True,
            "no_slot_policy": "greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Hungarian with symmetric self/peer costs and carrying peers excluded."
        },
        {
            "name": "sppdec",
            "variant": "cbba_assumed_bids_symmetric",
            "config_variant": "combined",
            "arbiter": "always_cbba",
            "symmetric_costs": True,
            "exclude_carrying_peers": True,
            "no_slot_policy": "greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "CBBA in head with symmetric self/peer costs and carrying peers excluded."
        },
        {
            "name": "sppdec",
            "variant": "cbba_assumed_bids_symmetric_wait",
            "config_variant": "combined",
            "arbiter": "always_cbba",
            "symmetric_costs": True,
            "exclude_carrying_peers": True,
            "no_slot_policy": "wait",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "As cbba_assumed_bids_symmetric, and a drone the assignment gives no task waits instead of chasing another drone's parcel."
        },
        {
            "name": "sppdec",
            "variant": "cbba_assumed_bids_symmetric_fallback_chain",
            "config_variant": "combined",
            "arbiter": "always_cbba",
            "symmetric_costs": True,
            "exclude_carrying_peers": True,
            "no_slot_policy": "hungarian_greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "As cbba_assumed_bids_symmetric with an ordered fallback: CBBA in head, then the Hungarian task, then greedy, so a drone the CBBA assignment leaves without a task still acts."
        },
        {
            "name": "sppdec",
            "variant": "hungarian_sym",
            "config_variant": "combined",
            "arbiter": "always_hungarian",
            "symmetric_costs": True,
            "exclude_carrying_peers": False,
            "no_slot_policy": "greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Peer-reasoning ablation: Hungarian with symmetric self/peer costs only."
        },
        {
            "name": "sppdec",
            "variant": "hungarian_nocarry",
            "config_variant": "combined",
            "arbiter": "always_hungarian",
            "symmetric_costs": False,
            "exclude_carrying_peers": True,
            "no_slot_policy": "greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Peer-reasoning ablation: Hungarian with carrying peers excluded only."
        },
        {
            "name": "sppdec",
            "variant": "cbba_assumed_bids_sym",
            "config_variant": "combined",
            "arbiter": "always_cbba",
            "symmetric_costs": True,
            "exclude_carrying_peers": False,
            "no_slot_policy": "greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Peer-reasoning ablation: CBBA in head with symmetric self/peer costs only."
        },
        {
            "name": "sppdec",
            "variant": "cbba_assumed_bids_nocarry",
            "config_variant": "combined",
            "arbiter": "always_cbba",
            "symmetric_costs": False,
            "exclude_carrying_peers": True,
            "no_slot_policy": "greedy",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Peer-reasoning ablation: CBBA in head with carrying peers excluded only."
        },
        {
            "name": "sppdec",
            "variant": "hungarian_symmetric_wait",
            "config_variant": "combined",
            "arbiter": "always_hungarian",
            "symmetric_costs": True,
            "exclude_carrying_peers": True,
            "no_slot_policy": "wait",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "As hungarian_symmetric, and a drone the assignment gives no task waits."
        },
        # -------------------------------
        # Revision: who decides when the trend and the competition conflict?
        # (trend wins = sppdec_arb_rule_based; competition wins; the model weighs the numbers)
        # -------------------------------
        {
            "name": "sppdec",
            "variant": "conflict_competition",
            "config_variant": "combined",
            "arbiter": "rule_based",
            "conflict_policy": "competition",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "Rule-based arbitration where the competition wins when it conflicts with the trend."
        },
        {
            "name": "sppdec",
            "variant": "conflict_judgement",
            "config_variant": "combined",
            "slm_model": "qwen3-vl:8b-instruct",
            "prompt_version": "judgement",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SLM arbitration: code fixes every decision except where the trend and the competition conflict; there the model weighs the numbers and decides, and its verdict must follow the consideration it names."
        },
        # -------------------------------
        # Revision: (beta, w-bar) ablation with the rule-based arbiter (R1.2, R1.9)
        # -------------------------------
        {
            "name": "sppdec",
            "variant": "control_rule",
            "config_variant": "control",
            "arbiter": "rule_based",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC control (no payload modelling, no fairness penalty (w=0, beta=0)) with the rule-based arbiter instead of the SLM."
        },
        {
            "name": "sppdec",
            "variant": "fairness_only_rule",
            "config_variant": "fairness_only",
            "arbiter": "rule_based",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC fairness_only (fairness penalty only (w=0, beta=0.02)) with the rule-based arbiter instead of the SLM."
        },
        {
            "name": "sppdec",
            "variant": "baseline_only_rule",
            "config_variant": "baseline_only",
            "arbiter": "rule_based",
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC baseline_only (payload modelling only (w=0.15, beta=0)) with the rule-based arbiter instead of the SLM."
        },
        # -------------------------------
        # Revision: SLM inference latency in the loop (R1.4, R2, R3.5)
        # 60 / 120 steps = 1 / 2 s at 60 Hz (Jetson AGX Orin range for Llama 3 8B)
        # -------------------------------
        {
            "name": "sppdec",
            "variant": "lat60",
            "config_variant": "combined",
            "slm_latency_steps": 60,
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC Combined; every SLM-consulted decision is held 60 steps (1 s) before execution."
        },
        {
            "name": "sppdec",
            "variant": "lat120",
            "config_variant": "combined",
            "slm_latency_steps": 120,
            "strategy_type": "sppdec",
            "topology": "predictive_decentralized",
            "planner": selected_planner,
            "snapshot": None,
            "status": "pending",
            "description": "SPPDEC Combined; every SLM-consulted decision is held 120 steps (2 s) before execution."
        },
    ]


# Fleet profile of the heterogeneous-fleet ablation: consumption-rate multipliers cycled over the
# fleet (nominal = 1.0). Every compared arm runs on the same fleet.
HETERO_FLEET = {"rates": [0.75, 1.0, 1.5]}
UNEQUAL_BATTERIES = {"capacities": [100, 60]}

# Observation-model sweep of the peer-reasoning arms (radius in cells, noise in cells, loss per step).
SENSITIVITY = {"b0": {"energy_penalty_alpha": 0.0}, "b10": {"energy_penalty_alpha": 0.1}, "w0": {"baseline_payload": 0.0}, "w50": {"baseline_payload": 0.5},
               "h10": {"belief_window": 10}, "h60": {"belief_window": 60}, "boff": {"belief_off": True}}

SENSING_STRESS = {"n1": {"noise_std": 1.0}, "n2": {"noise_std": 2.0}, "l50": {"loss": 0.5}, "n2l50": {"noise_std": 2.0, "loss": 0.5},
                  # added in the revision: finer position error, delayed readings (steps; the simulation runs at 60 steps a second), intermittent readings
                  "n05": {"noise_std": 0.5}, "d30": {"delay": 30}, "d120": {"delay": 120}, "bo": {"blackout": (200, 100)}}

SENSING_SETTINGS = {
    "r4": {"radius": 4},
    "r8": {"radius": 8},
    "r12": {"radius": 12},
}


def build_strategy_vault(selected_planner: str) -> List[Dict]:
    """Base arms plus the arms derived from them (sensing sweep, heterogeneous fleet).

    Vocabulary of the arm ids (the old spellings are mapped in strategies/sppdec/legacy_ids.py):
      cbba_assumed_bids   CBBA run inside the drone; every message the protocol would receive is replaced
                          by the bid the drone assumes the peer would send. Without "bid_rounds" it is
                          the one-task form computed directly (best drone-and-task pair first).
      bid_rounds_bundleN  both CBBA phases played out in rounds, up to N tasks per drone;
      bid_rounds_fair_share  N is each drone's fair share of the open tasks, ceil(tasks / agents).
      symmetric           the drone scores itself and its peers with the same cost estimator and leaves
                          out peers that are carrying a parcel (the "shared assumptions" of the notes).
      tracking            peers are tracked every step; a vanished or contested target ends the plan.
      memory              the drone remembers which task it assumed each peer took until it sees otherwise.
      unequal_batteries   the fleet has mixed consumption rates and capacities (HETERO_FLEET).
      sense_ / local_     limited sensing radius (sense_: no tracking layer; local_: with the layers named).
    """
    base = _base_strategy_vault(selected_planner)
    by_id = {f"{s['name']}_{s.get('variant', 'base')}": s for s in base}
    derived: List[Dict] = []

    def clone(src_id, variant, description, rename=False, **extra):
        spec = dict(by_id[src_id])
        spec.update(extra)
        if rename:
            # Baselines whose variant names their mode (centralised) keep it; the suffix goes on the name.
            spec["name"] = spec["name"] + "_unequal_batteries"
        else:
            spec["variant"] = variant
        spec["status"] = "pending"
        spec["snapshot"] = None
        spec["description"] = description
        derived.append(spec)
        by_id[f"{spec['name']}_{spec.get('variant', 'base')}"] = spec

    # Sensing sweep: the floor (greedy with the yield rule) and the shared-assumption arm.
    for tag, sensing in SENSING_SETTINGS.items():
        clone("sppdec_arb_always_greedy", f"sense_greedy_{tag}",
              f"Always-greedy with limited observation ({tag}).", sensing=sensing)
        clone("sppdec_cbba_assumed_bids_symmetric", f"sense_cbba_symmetric_{tag}",
              f"CBBA in head, shared costs, limited observation ({tag}).", sensing=sensing)

    # Who decides when the trend and the competition conflict? Same shared-assumption setup
    # (symmetric costs, carrying peers dropped, CBBA in head as the coordination model) for all three.
    clone("sppdec_cbba_assumed_bids_symmetric", "judge_trend_wins",
          "Shared assumptions, CBBA coordination; arbiter rules, the trend wins a conflict.",
          arbiter="rule_based", conflict_policy="trend", coordination_model="cbba")
    clone("sppdec_cbba_assumed_bids_symmetric", "judge_competition_wins",
          "Shared assumptions, CBBA coordination; arbiter rules, the competition wins a conflict.",
          arbiter="rule_based", conflict_policy="competition", coordination_model="cbba")
    clone("sppdec_cbba_assumed_bids_symmetric", "judge_model",
          "Shared assumptions, CBBA coordination; the code fixes every decision except a trend-versus-competition conflict, which the SLM weighs.",
          arbiter="slm", prompt_version="judgement", slm_model="qwen3-vl:8b-instruct", coordination_model="cbba")

    # SLM inference latency in the loop on the judgement arm. 33 steps = 0.55 s, the measured
    # per-call time of the 8B model on the test GPU; 60 and 120 steps bracket slower hardware.
    # Blocking: the drone waits for the decision. Prefetch: it decides while delivering.
    for steps, mode in [(33, "blocking"), (60, "blocking"), (120, "blocking"),
                        (33, "prefetch"), (120, "prefetch")]:
        clone("sppdec_judge_model", f"judge_model_lat{steps}" + ("_prefetch" if mode == "prefetch" else ""),
              f"Judgement arm with {steps} steps of inference latency per consulted decision ({mode}).",
              slm_latency_steps=steps, slm_latency_mode=mode)
    clone("sppdec_lat120", "lat120_prefetch",
          "Original SLM arbitration with 120 steps of latency, non-blocking (decides while delivering).",
          slm_latency_mode="prefetch")

    # CBBA in the head: how much of the protocol the drone simulates (shortcut, single-task
    # protocol, standard bundle protocol). Same shared assumptions in every arm.
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_bid_rounds_bundle1_symmetric",
          "CBBA in head, shared assumptions: both CBBA phases run over the believed fleet with assumed peer bids, bundle size 1.",
          cbba_bundle_size=1)
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_bid_rounds_bundle3_symmetric",
          "CBBA in head, shared assumptions: both CBBA phases run over the believed fleet with assumed peer bids, bundle size 3 (as the CBBA-Bundle baseline).",
          cbba_bundle_size=3)

    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_bid_rounds_fair_share_symmetric",
          "CBBA in head, shared assumptions, rounds with an adaptive bundle: each drone's fair share of the open tasks, ceil(tasks/agents), no other limit.",
          cbba_bundle_size="auto")

    # Assumption memory: the drone keeps what it assumed each peer took until it observes otherwise.
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_assumed_bids_symmetric_memory",
          "CBBA in head, shared assumptions, with assumption memory (a peer assumed to have taken a task stays busy until observation contradicts it).",
          assumption_memory=True)
    clone("sppdec_hungarian_symmetric", "hungarian_symmetric_memory",
          "Hungarian, shared assumptions, with assumption memory.", assumption_memory=True)
    clone("sppdec_cbba_bid_rounds_bundle3_symmetric", "cbba_bid_rounds_bundle3_symmetric_memory",
          "Protocol CBBA in head (bundle 3), shared assumptions, with assumption memory.", assumption_memory=True)

    # Per-step observation: tracking, intent from observed motion, assumption memory, and plan
    # interruption run alongside execution and decision.
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_assumed_bids_symmetric_tracking",
          "CBBA in head, shared assumptions, per-step observation (no memory): peers tracked every step, a closer peer closing on a target or a vanished target ends the drone's plan.",
          observe_every_step=True)
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_assumed_bids_symmetric_tracking_memory",
          "CBBA in head, shared assumptions, per-step observation and assumption memory.",
          observe_every_step=True, assumption_memory=True)
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_assumed_bids_symmetric_tracking_gone",
          "As cbba_assumed_bids_symmetric_tracking, but observation only ends a plan whose target has vanished (no contested-target handling).",
          observe_every_step=True, observe_contested=False)
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_assumed_bids_symmetric_tracking_contested",
          "As cbba_assumed_bids_symmetric_tracking, but observation only handles contested targets (a vanished target is still found on arrival).",
          observe_every_step=True, observe_gone=False)
    # Control: greedy given the same observation layer (is the gain reasoning, or reacting?)
    clone("sppdec_arb_always_greedy", "greedy_tracking",
          "Always-greedy (with the yield rule) plus per-step observation: ends a plan whose target vanished or is contested, and leaves contested targets out.",
          observe_every_step=True)
    clone("sppdec_arb_always_greedy", "greedy_tracking_gone",
          "Always-greedy plus per-step observation limited to vanished targets.",
          observe_every_step=True, observe_contested=False)
    clone("sppdec_arb_always_greedy", "greedy_tracking_contested",
          "Always-greedy plus per-step observation limited to contested targets.",
          observe_every_step=True, observe_gone=False)
    clone("sppdec_hungarian_symmetric", "hungarian_symmetric_tracking_memory",
          "Hungarian, shared assumptions, per-step observation and assumption memory.",
          observe_every_step=True, assumption_memory=True)
    clone("sppdec_cbba_bid_rounds_bundle3_symmetric", "cbba_bid_rounds_bundle3_symmetric_tracking_memory",
          "Protocol CBBA (bundle 3), shared assumptions, per-step observation and assumption memory.",
          observe_every_step=True, assumption_memory=True)

    # Local observation: the components with the observation layer under the limited observation
    # model (radius in cells, noise, loss). Unlimited radius is the labelled upper bound.
    for tag, sensing in SENSING_SETTINGS.items():
        for src_id, short in [("sppdec_greedy_tracking", "greedy_tracking"),
                              ("sppdec_cbba_assumed_bids_symmetric_tracking", "cbba_tracking"),
                              ("sppdec_hungarian_symmetric_tracking_memory", "hungarian_tracking_memory"),
                              ("sppdec_cbba_bid_rounds_bundle3_symmetric_tracking_memory", "bid_rounds_bundle3_tracking_memory")]:
            clone(src_id, f"local_{short}_{tag}",
                  f"{short} under limited observation ({tag}).", sensing=sensing)

    # Lightweight classifiers in place of the SLM arbiter, on the same footing: shared assumptions,
    # CBBA coordination, per-step observation. See analysis/revision/train_classifiers.py.
    for name, kw, desc in [
        ("rule_tracking", dict(arbiter="rule_based", conflict_policy="trend"), "Rule-table arbiter"),
        ("classifier_regime_tracking", dict(arbiter="classifier_regime"),
         "Regime classifier (outcome-trained, leave-one-configuration-out): greedy or coordination for the whole mission"),
        ("classifier_distilled_tracking", dict(arbiter="classifier_tree", classifier_model="distilled"),
         "Tree distilled from the rule table, applied at every consulted decision"),
        ("classifier_slm_imitation_tracking", dict(arbiter="classifier_tree", classifier_model="slm_imitation"),
         "Tree imitating the SLM's raw answers, applied at every consulted decision"),
    ]:
        clone("sppdec_cbba_assumed_bids_symmetric", name, desc + " (shared assumptions, CBBA coordination, per-step observation).",
              coordination_model="cbba", observe_every_step=True, **kw)
        for tag, sensing in [("r8", SENSING_SETTINGS["r8"])]:
            clone(f"sppdec_{name}", f"local_{name}_{tag}", f"{desc}, sees nearby ({tag}).", sensing=sensing)

    # Context arbiters: one shared context (decision facts, scenario descriptors including the self-check facts,
    # what each model proposes); the arbiter picks one of greedy, Hungarian, CBBA one-task, CBBA in rounds
    # (fair-share bundle), and that model's task is the drone's action. Per-step tracking and memory are on.
    for name, kw, desc in [
        ("context_rules_selfcheck_off", dict(arbiter="context_rules", use_self_check=False),
         "Rule list from the scenario table over all four reasoning models"),
        ("context_rules_selfcheck_on", dict(arbiter="context_rules", use_self_check=True),
         "Rule list from the scenario table over all four reasoning models, with the self-check"),
        ("context_slm_selfcheck_off", dict(arbiter="context_slm", use_self_check=False),
         "SLM over all four reasoning models, reading the whole context and the scenario table"),
        ("context_slm_selfcheck_on", dict(arbiter="context_slm", use_self_check=True),
         "SLM over all four reasoning models, reading the whole context (self-check facts included) and the scenario table"),
    ]:
        clone("sppdec_cbba_assumed_bids_symmetric", name,
              desc + " (shared assumptions, per-step tracking, memory, fair-share CBBA bundle).",
              coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
              cbba_bundle_size="auto", **kw)
        for tag, sensing in SENSING_SETTINGS.items():
            clone(f"sppdec_{name}", f"local_{name}_{tag}", f"{desc}, sees nearby ({tag}).", sensing=sensing)

    # Self-check test scenarios. The drone's own energy use differs from the nominal model it plans with, and it does
    # not know how. The self-check learns what a cell costs it and what weight costs it, from its battery as it flies.
    #   rate    every drone burns 40% more per cell than nominal;
    #   weight  distance costs as nominal but carried weight costs three times the nominal extra;
    #   mixed   each drone differs in both (cycled over the fleet), so peers are not like the drone itself.
    # Ablation per scenario: self-check off, self-check on, and the true values known from the start (reference).
    SELFCHECK_SCENARIOS = [
        ("", {"rates": [1.4], "capacities": [30]}, "burns 40% more per cell than nominal"),
        ("weight_", {"rates": [1.0], "weight_factors": [1.5], "capacities": [30]},
         "pays three times the nominal weight surcharge per cell"),
        ("mixed_", {"rates": [0.8, 1.4, 1.0], "weight_factors": [0.3, 1.5, 0.5], "capacities": [30]},
         "has its own consumption (mixed across the fleet)"),
        # shift: consumption is nominal at first; at step 600 every drone burns 50% more per cell and pays four
        # times the weight surcharge. Nothing announces it. "known" is the truth at the start, stale after the shift.
        ("shift_", {"rates": [1.0], "capacities": [30],
                    "shift": {"step": 600, "rate_scale": 1.5, "weight_scale": 4.0}},
         "changes its consumption part-way through the mission"),
    ]
    for stag, profile, what in SELFCHECK_SCENARIOS:
        for tag, extra, desc in [
            ("off", dict(use_self_check=False, own_rate_known=False), "without the self-check; the true consumption is hidden"),
            ("on", dict(use_self_check=True, own_rate_known=False), "with the self-check; the true consumption is hidden"),
            ("known", dict(use_self_check=True, own_rate_known=True), "the true consumption is known from the start (reference)"),
        ]:
            if tag == "on" and stag in ("mixed_", "shift_"):
                # ablation of the self-model as a prior for peers (they are assumed to follow the nominal model)
                clone("sppdec_cbba_assumed_bids_symmetric_tracking", f"selfcheck_{stag}on_peers_nominal",
                      f"CBBA (direct) + tracking, each drone {what}; self-check on, peers assumed nominal.",
                      fleet_profile=profile, peer_prior="nominal", **extra)
            clone("sppdec_cbba_assumed_bids_symmetric_tracking", f"selfcheck_{stag}{tag}",
                  f"CBBA (direct) + tracking, small batteries, each drone {what}; {desc}.",
                  fleet_profile=profile, **extra)
            clone("sppdec_greedy_tracking", f"selfcheck_greedy_{stag}{tag}",
                  f"Greedy + tracking, small batteries, each drone {what}; {desc}.", fleet_profile=profile, **extra)

    # Evidence for the context arbiters: each reasoning model fixed (no arbiter), with the self-check on, in each of the
    # self-check scenarios. Completion time, parcels delivered and drones lost per model say what those conditions do to
    # the mission, which the arbiters then read.
    EVIDENCE_MODELS = [
        ("greedy", dict(arbiter="always_greedy")),
        ("hungarian", dict(arbiter="always_hungarian", coordination_model="hungarian")),
        ("cbba_rounds", dict(arbiter="always_cbba", coordination_model="cbba",
                             cbba_bundle_size="auto")),
    ]
    for stag, profile, what in SELFCHECK_SCENARIOS:
        for mname, mkw in EVIDENCE_MODELS:
            clone("sppdec_cbba_assumed_bids_symmetric", f"evidence_{stag}{mname}",
                  f"{mname} fixed, with tracking, memory and the self-check, each drone {what}.",
                  fleet_profile=profile, observe_every_step=True, assumption_memory=True,
                  use_self_check=True, own_rate_known=False, **mkw)

    # Context arbiters in the self-check scenarios, to be compared with the fixed models above in the same batch.
    # Self-check off: the arbiter does not see the learned own-energy facts and the drone's estimates are not corrected.
    for stag, profile, what in SELFCHECK_SCENARIOS:
        label = (stag or "rate_").rstrip("_")
        for kind in ("rules", "slm"):
            for tag, use in (("on", True), ("off", False)):
                clone("sppdec_cbba_assumed_bids_symmetric", f"context_{kind}_in_{label}_selfcheck_{tag}",
                      f"{kind} arbiter over all four models, self-check {tag}, each drone {what}.",
                      arbiter=f"context_{kind}", coordination_model="hungarian", observe_every_step=True,
                      assumption_memory=True, cbba_bundle_size="auto", fleet_profile=profile,
                      use_self_check=use, own_rate_known=False)

    # The arbiters in the self-check scenarios, to be compared with each fixed model (the evidence arms above) in the
    # same batch: rule list and SLM, each with the self-check off and on (drones do not know their true consumption).
    for stag, profile, what in SELFCHECK_SCENARIOS:
        for aname, akw in [("rules", dict(arbiter="context_rules")), ("slm", dict(arbiter="context_slm")),
                            ("slm_free", dict(arbiter="context_slm_free"))]:
            for ctag, on in [("off", False), ("on", True)]:
                clone("sppdec_cbba_assumed_bids_symmetric", f"arbiter_{stag}{aname}_{ctag}",
                      f"Context arbiter ({aname}) over all four models, self-check {ctag}, each drone {what}.",
                      fleet_profile=profile, observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto",
                      coordination_model="hungarian", use_self_check=on, own_rate_known=False, **akw)

    # SLM contribution ladder. Same decision facts and the same four reasoning models for every arm; only the arbiter
    # differs: the rule list, or the free SLM (authority over the choice, context as suggestions) at each model size.
    # Self-check on, drones do not know their true consumption. Scenarios: limited sensing;
    # unequal batteries. No decision time is charged to the model.
    LADDER_SCENARIOS = [
        ("limited", dict(sensing=SENSING_SETTINGS["r8"])),
        ("unequal", dict(fleet_profile=HETERO_FLEET)),
        # Small batteries and a consumption the drone's declared spec does not match (headroom for the choice):
        ("rate", dict(fleet_profile={"rates": [1.4], "capacities": [30]})),
        ("weight", dict(fleet_profile={"rates": [1.0], "weight_factors": [1.5], "capacities": [30]})),
    ]
    LADDER_MODELS = [
        ("vl8b", "qwen3-vl:8b-instruct"), ("q8b", "qwen3:8b"), ("q4b", "qwen3:4b"),
        ("q1_7b", "qwen3:1.7b"), ("q0_6b", "qwen3:0.6b"),
    ]
    for scn, skw in LADDER_SCENARIOS:
        common = dict(coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
                      cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, **skw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"ladder_{scn}_rules",
              f"Rule list over all four reasoning models, scenario {scn}.", arbiter="context_rules", **common)
        for mtag, model in LADDER_MODELS:
            clone("sppdec_cbba_assumed_bids_symmetric", f"ladder_{scn}_slm_{mtag}",
                  f"Free SLM ({model}) over all four reasoning models, scenario {scn}.",
                  arbiter="context_slm_free", slm_model=model, **common)
        # The same free SLM also shown the world (its body, the terrain, every task with its geometry and energy margin, its
        # peers, a small map): with the tables' computed suggestion, and without it.
        for mtag, model in LADDER_MODELS[:1]:
            clone("sppdec_cbba_assumed_bids_symmetric", f"ladder_{scn}_slm_{mtag}_world",
                  f"Free SLM ({model}) shown the world and the tables' suggestion, scenario {scn}.",
                  arbiter="context_slm_free", slm_model=model, world_view=True, **common)
            clone("sppdec_cbba_assumed_bids_symmetric", f"ladder_{scn}_slm_{mtag}_world_nosug",
                  f"Free SLM ({model}) shown the world and the tables as evidence, no computed suggestion, scenario {scn}.",
                  arbiter="context_slm_free", slm_model=model, world_view=True, slm_suggest=False, **common)
        # The hybrid qwen3:8b reasoning before it answers (thinking mode), shown the world and the tables as evidence.
        clone("sppdec_cbba_assumed_bids_symmetric", f"ladder_{scn}_slm_q8b_world_think",
              f"Free SLM (qwen3:8b, thinking mode) shown the world and the tables as evidence, no computed suggestion, scenario {scn}.",
              arbiter="context_slm_free", slm_model="qwen3:8b", world_view=True, slm_suggest=False, slm_think=True, **common)

    # Headroom check (no model calls): how much a better planner could recover. Per scenario next to the rule list
    # (ladder_<scenario>_rules): the greedy floor, the rule list perturbed at 30% of the decisions where the models disagree
    # (outcome spread from chance), and centralised Hungarian with full information (the ceiling).
    for scn, skw in [LADDER_SCENARIOS[0], LADDER_SCENARIOS[2]]:
        common = dict(coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
                      cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, **skw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"headroom_{scn}_greedy",
              f"Always-greedy floor, scenario {scn}.", arbiter="always_greedy", **common)
        clone("sppdec_cbba_assumed_bids_symmetric", f"headroom_{scn}_perturbed",
              f"Rule list perturbed at 30% of the decisions where the models disagree, scenario {scn}.",
              arbiter="context_rules", rule_perturb=0.3, **common)
        fleet = {"fleet_profile": skw["fleet_profile"]} if "fleet_profile" in skw else {}
        # The centralised strategy reads its assignment mode from the variant, so the scenario goes in the name.
        clone("centralized_cc_hungarian_hungarian", None,
              f"Centralised Hungarian with full information (the ceiling), scenario {scn}.", rename=True, **fleet)
        derived[-1]["name"] = f"centralized_cc_hungarian_headroom_{scn}"

    # Information channels under limited sensing (no model calls): the rule list with one channel limited at a time, to find which
    # missing information costs the time. `radius` limits the readings of peers; `parcel_radius` the readings of parcels
    # (default: the same as `radius`). Not included: exact peers with limited parcels, where the drones keep planning for parcels
    # that peers already delivered and the run never ends (see the change log).
    CHANNELS = [
        ("limited", {"radius": 8}),
    ]
    for tag, sens in CHANNELS:
        clone("sppdec_cbba_assumed_bids_symmetric", f"channel_{tag}",
              f"Rule list, sensing channels: {tag}.", arbiter="context_rules", coordination_model="hungarian",
              observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
              own_rate_known=False, sensing=sens)

    # Parcel freshness (no model calls): the rule list with a trip's cost divided by the chance the parcel is still there, from
    # the time since the drone last saw it. Under limited sensing, and in full sensing as a control (nothing should change).
    for tag, sens in CHANNELS[:1]:
        clone("sppdec_cbba_assumed_bids_symmetric", f"fresh_{tag}",
              f"Rule list with parcel freshness, sensing: {tag}.", arbiter="context_rules", coordination_model="hungarian",
              observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
              own_rate_known=False, sensing=sens, freshness=True)

    # Territory convention (no model calls): every drone computes the same partition of the parcels from the shared start
    # information and prefers its own territory; no messages. Under limited sensing, and in full sensing as a control.
    for tag, sens in CHANNELS[:1]:
        clone("sppdec_cbba_assumed_bids_symmetric", f"territory_{tag}",
              f"Rule list with the territory convention, sensing: {tag}.", arbiter="context_rules",
              coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto",
              use_self_check=True, own_rate_known=False, sensing=sens, territory=True)

    # Ledger probe (no model calls): the station's occupancy counts the delivered parcels; when it equals the total, the drone stops looking and parks.
    for tag, sens in CHANNELS[:1]:
        for terr in (False, True):
            clone("sppdec_cbba_assumed_bids_symmetric", f"ledger_{'territory_' if terr else ''}{tag}",
                  f"Rule list{' with territory' if terr else ''} that stops when the station holds every parcel, sensing: {tag}.",
                  arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
                  cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, sensing=sens, territory=terr,
                  ledger_complete=True)

    # Station awareness (information model shared by every strategy, src/perception.py): the drone's knowledge of the station is a memory refreshed
    # only by what it senses (station_memory="observed", against the simulator's live free-cell list), and it stops when that knowledge says the
    # station holds every parcel (ledger_stop). Levels: a0 neither (the original arm), a1 memory only, a2 stop on the live count, a3 memory and stop, a4 predictive memory (extrapolated count) and stop. Each strategy
    # uses the information through its own logic; the allocation code is unchanged. Centralised keeps its global view and is not included.
    AWARENESS = {"a0": {}, "a1": {"station_memory": "observed"}, "a2": {"ledger_stop": True},
                 "a3": {"station_memory": "observed", "ledger_stop": True},
                 "a4": {"station_memory": "predictive", "ledger_stop": True},
                 "a5": {"station_memory": "predictive_avg", "ledger_stop": True},
                 "a6": {"station_memory": "predictive_work", "ledger_stop": True},
                 # b: bids and feasibility priced to the cell the drone would drop at (b0: the live free cells, b1: the drone's own station memory)
                 "b0": {"station_bids": True},
                 "b1": {"station_memory": "observed", "station_bids": True}}
    for tag, sens in CHANNELS[:1]:
        for level, opts in AWARENESS.items():
            for src_id in ("naive_decentralized_reactive", "decentralized_auction_auction_baseline", "cbba_single_task_cbba_single",
                           "cbba_full_cbba_bundle"):
                clone(src_id, f"aw_{tag}_{level}", f"Baseline with station awareness {level}, sensing: {tag}.", sensing=sens, **opts)
            clone("sppdec_cbba_assumed_bids_symmetric", f"aw_{tag}_{level}", f"Rule list with station awareness {level}, sensing: {tag}.",
                  arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
                  cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, sensing=sens, **opts)

    # Peer model, shadow mode (stage A of marl_baseline_design.md section 8): the rule list as in channel_<sensing> with the peer model logging its
    # predictions (positions, parcels probably taken, drops) with each decision. Decisions must equal channel_<sensing> exactly.
    for tag, sens in CHANNELS[:1]:
        clone("sppdec_cbba_assumed_bids_symmetric", f"pm_{tag}", f"Rule list with the peer model in shadow mode, sensing: {tag}.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto",
              use_self_check=True, own_rate_known=False, sensing=sens, peer_model_shadow=True)

    # Predicted station (Author's design): the drone remembers the station, refreshes it whenever it senses cells, and adds the drops it predicts
    # for the peers (peer model). pmm_: shadow mode with the memory on (nothing reads it; compare with aw_<tag>_a1, which has the same memory).
    # sm_: the drone stops when its predicted station holds every parcel (reversible: a fresh look that says otherwise restarts it).
    for tag, sens in CHANNELS[:1]:
        base_kwargs = dict(arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
                           cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, sensing=sens)
        clone("sppdec_cbba_assumed_bids_symmetric", f"pmm_{tag}", f"Rule list, station memory, peer model in shadow mode, sensing: {tag}.",
              station_memory="observed", peer_model_shadow=True, **base_kwargs)
        clone("sppdec_cbba_assumed_bids_symmetric", f"sm_{tag}", f"Rule list that stops when its predicted station holds every parcel, sensing: {tag}.",
              station_memory="observed", ledger_stop=True, station_model=True, **base_kwargs)

    # Uneven batteries (capacities 30, 60, 100 in turn over the fleet), one fleet per personality, to see whether a personality spares the weak drones:
    # unq_<personality>_<tag>; unq_none: no personality (the variant's own beta and w-bar). Same rule list as the other arms.
    UNEQ = {"capacities": [30, 60, 100]}
    for tag, sens in CHANNELS[:1]:
        kwu = dict(arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto",
                   use_self_check=True, own_rate_known=False, sensing=sens, fleet_profile=UNEQ)
        clone("sppdec_cbba_assumed_bids_symmetric", f"unq_none_{tag}", f"Uneven batteries, no personality, sensing: {tag}.", **kwu)
        for pers in ("rational", "selfish", "altruistic", "reckless"):
            clone("sppdec_cbba_assumed_bids_symmetric", f"unq_{pers}_{tag}", f"Uneven batteries, all {pers}, sensing: {tag}.", agent_personalities=[pers], **kwu)
        # one mixed fleet: personalities (4) and batteries (3) in turn, so a fleet of 12 holds every combination once
        clone("sppdec_cbba_assumed_bids_symmetric", f"mixu_{tag}", f"Uneven batteries and mixed personalities, sensing: {tag}.",
              agent_personalities=["rational", "selfish", "altruistic", "reckless"], **kwu)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mixuh_{tag}", f"Uneven batteries, mixed personalities, parcels x3 heavier, sensing: {tag}.",
              agent_personalities=["rational", "selfish", "altruistic", "reckless"], parcel_weight_scale=3.0, **kwu)
        # the same with parcels three times heavier
        clone("sppdec_cbba_assumed_bids_symmetric", f"unqh_none_{tag}", f"Uneven batteries, parcels x3 heavier, no personality, sensing: {tag}.", parcel_weight_scale=3.0, **kwu)
        for pers in ("rational", "selfish", "altruistic"):
            clone("sppdec_cbba_assumed_bids_symmetric", f"unqh_{pers}_{tag}", f"Uneven batteries, parcels x3 heavier, all {pers}, sensing: {tag}.",
                  agent_personalities=[pers], parcel_weight_scale=3.0, **kwu)

    # Avoid the peers' best parcels (no waiting): avoid_<tag> rule list (width 1 without a personality); mixavoid_<tag> mixed fleet (widths: rational 1, selfish 0, altruistic 3,
    # reckless 0); avoid3_<tag> width 3 for every drone (personality-free, to see the effect of the width alone).
    for tag, sens in CHANNELS[:1]:
        kwa = dict(arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto",
                   use_self_check=True, own_rate_known=False, sensing=sens)
        clone("sppdec_cbba_assumed_bids_symmetric", f"avoid_{tag}", f"Rule list that avoids the peers' best parcels, sensing: {tag}.", avoid_top_k=True, **kwa)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mixavoid_{tag}", f"Mixed fleet that avoids the peers' best parcels (width by personality), sensing: {tag}.",
              avoid_top_k=True, agent_personalities=["rational", "selfish", "altruistic", "reckless"], **kwa)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_a0b", f"Mixed fleet without avoidance, sensing: {tag}.",
              agent_personalities=["rational", "selfish", "altruistic", "reckless"], **kwa)

    # Avoidance with the whole-trip ranking: avoidt_<tag> = rt_<tag> (peer-model costs, whole-trip cost, predicted-station stop) plus avoid_top_k; mixavoidt_<tag> in the mixed fleet.
    for tag, sens in CHANNELS[:1]:
        kwt = dict(arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
                   own_rate_known=False, sensing=sens, station_memory="observed", ledger_stop=True, station_model=True, peer_model_cost=True)
        clone("sppdec_cbba_assumed_bids_symmetric", f"avoidt_{tag}", f"Whole-trip costs, avoids the peers' best parcels, sensing: {tag}.", avoid_top_k=True, **kwt)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mixavoidt_{tag}", f"Mixed fleet, whole-trip costs, avoids the peers' best parcels, sensing: {tag}.",
              avoid_top_k=True, agent_personalities=["rational", "selfish", "altruistic", "reckless"], **kwt)

    # Avoidance with the fixed width the Author asked for (outside every free peer's top 3, shrinking with open parcels per peer): avoid3_<tag>.
    for tag, sens in CHANNELS[:1]:
        clone("sppdec_cbba_assumed_bids_symmetric", f"avoid3_{tag}", f"Rule list, goes outside every free peer's top 3 (width 3 for all), sensing: {tag}.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
              own_rate_known=False, sensing=sens, avoid_top_k=True, avoid_width=3)

    # Avoidance with the peer-belief tracker: avoid3b_<tag> = avoid3 plus each peer's best parcels ranked over the parcels it is believed to think open (ghosts included).
    for tag, sens in CHANNELS[:1]:
        clone("sppdec_cbba_assumed_bids_symmetric", f"avoid3b_{tag}", f"Rule list, outside every free peer's top 3 over the peer's believed-open set, sensing: {tag}.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
              own_rate_known=False, sensing=sens, avoid_top_k=True, avoid_width=3, avoid_beliefs=True)

    # Avoidance with width 2 (outside every free peer's top 2): avoid2_<tag> over the open set, avoid2b_<tag> over each peer's believed-open set.
    for tag, sens in CHANNELS[:1]:
        kw2 = dict(arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
                   own_rate_known=False, sensing=sens, avoid_top_k=True, avoid_width=2)
        clone("sppdec_cbba_assumed_bids_symmetric", f"avoid2_{tag}", f"Rule list, outside every free peer's top 2, sensing: {tag}.", **kw2)
        clone("sppdec_cbba_assumed_bids_symmetric", f"avoid2b_{tag}", f"Rule list, outside every free peer's top 2 over its believed set, sensing: {tag}.", avoid_beliefs=True, **kw2)

    # One near-reaching personality (selfish, beta 0) with one far-reaching (rational, beta 0.02) in the same fleet, drones alternating: mixsr_<tag>. Single-personality fleets: pers_<p>_<tag>.
    for tag, sens in CHANNELS[:1]:
        clone("sppdec_cbba_assumed_bids_symmetric", f"mixsr_{tag}", f"Fleet alternating selfish (near) and rational (far), sensing: {tag}.", arbiter="context_rules",
              coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing=sens, agent_personalities=["selfish", "rational"])

    # Consensus (the Author's rules): the drone acts on what Hungarian and CBBA agree on, on one shared set of expected bids; no yield filter, no wait.
    for tag, sens in CHANNELS[:1]:
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_{tag}", f"Consensus of Hungarian and CBBA bundles on shared bids, sensing: {tag}.", arbiter="context_rules",
              coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing=sens, station_memory="observed", freshness=True, ledger_complete=True, consensus=True)
        # The same, reasoning with one model alone (its bundle is the drone's plan and its assumption about every peer).
        for mname, models in (("hungarian", ["hungarian"]), ("cbba", ["cbba_rounds"])):
            clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_{mname}_{tag}", f"Reasoning with {mname} alone on shared bids, sensing: {tag}.", arbiter="context_rules",
                  coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
                  sensing=sens, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, consensus_models=models)

    # Context study: which model should the drone reason with, and when? The consensus drone with Hungarian alone, CBBA alone or both, across the mission
    # parameters an SLM was given (sensing radius, battery capacity; the fleet size, the parcel count and the weight range come from the config and the environment).
    for radius in (4, 8, 12):
        for cap in (None, 50, 30):
            ctag = f"r{radius}" + (f"_c{cap}" if cap else "")
            for mname, models in (("hungarian", ["hungarian"]), ("cbba", ["cbba_rounds"]), ("both", None)):
                extra = dict(consensus_models=models) if models else {}
                if cap:
                    extra["fleet_profile"] = {"capacities": [cap]}
                clone("sppdec_cbba_assumed_bids_symmetric", f"ctx_{mname}_{ctag}", f"Context study: {mname}, sensing radius {radius}" + (f", battery {cap}" if cap else "") + ".",
                      arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
                      own_rate_known=False, sensing={"radius": radius}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, **extra)

    # Model choice by fleet size: CBBA alone in a fleet of at most 4 agents, Hungarian alone in a larger one (cutoff fitted on the revision batches: CBBA ahead at 2-3 agents, no difference at 4-8, Hungarian ahead from 10).
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_rule_limited", "Reasoning with CBBA alone when the fleet is at most 4 agents, else Hungarian alone, radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4})

    E_STEP_1 = Drone.BASE_COST_PER_CELL / sim_clock.STEPS_PER_CELL
    # No gambling (Author): the drone acts only on the tasks the shared reasoning gives it; with none it goes to the station and decides again. With and without the re-check of its target.
    for suffix, extra in (("nogamble", dict(no_gamble=True)), ("nogamble_recheck", dict(no_gamble=True, recheck=True))):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_{suffix}_limited", f"Fleet-size rule, {suffix.replace('_', ', ')}, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, **extra)

    # Hungarian alone and CBBA alone in the no-gamble design (the model comparison after the feasibility fix).
    for mname, models in (("hungarian", ["hungarian"]), ("cbba", ["cbba_rounds"])):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_{mname}_nogamble_limited", f"No gambling, reasoning with {mname} alone, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, consensus_models=models, no_gamble=True, observe_contested=False)

    # Both models in the drone, no gambling: the drone acts only on the tasks both Hungarian and CBBA give it (what they disagree about is nobody's until they agree).
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_both_nogamble_limited", "No gambling, consensus of Hungarian and CBBA, radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, no_gamble=True, observe_contested=False)

    # Shared greedy matching in the drone (cheapest pair first on the same bids), no gambling.
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_greedy_nogamble_limited", "No gambling, reasoning with a shared greedy matching alone, radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, consensus_models=["greedy"], no_gamble=True, observe_contested=False)

    # Assignments stand until observation contradicts them (a peer's live schedule keeps its parcels), on top of the fleet-size rule and no gambling.
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_rule_nogamble_commit_limited", "Fleet-size rule, no gambling, assignments stand (commitments), radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True, commitments=True)

    # Time in the bid on the arm of record (fleet-size rule, no gambling): weights as tested on seeds 16-31, not refitted.
    for mult in (1, 3):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble_time{mult}_limited", f"Fleet-size rule, no gambling, time in the bid x{mult}, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
              time_weight=mult * (Drone.BASE_COST_PER_CELL / sim_clock.STEPS_PER_CELL))

    # Hungarian alone on the arm of record's other settings (no gambling), without and with time in the bid, for the small fleets where the fleet-size rule uses CBBA.
    for label, mult in (("", 0), ("_time3", 3)):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_hungarian_nogamble{label}_limited", f"Hungarian alone, no gambling{', time in the bid x3' if mult else ''}, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, consensus_models=["hungarian"], observe_contested=False, no_gamble=True,
              time_weight=mult * (Drone.BASE_COST_PER_CELL / sim_clock.STEPS_PER_CELL))

    # Time in the bid only where the fleet-size rule uses Hungarian (more than 4 agents); alone and with the schedules kept. Weight x3 as tested on seeds 16-31 and 130-145, not refitted.
    for label, extra in (("timeh", {}), ("keep_timeh", {"keep_schedule": True})):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble_{label}_limited", f"Fleet-size rule, no gambling, {'schedules kept, ' if extra else ''}time x3 in the bid with Hungarian only, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
              time_weight_hungarian=3 * (Drone.BASE_COST_PER_CELL / sim_clock.STEPS_PER_CELL), **extra)

    # Heterogeneous flight speed (Author: speed matters alongside time and energy): drones alternate between the nominal speed and half of it, each declaring its own.
    # Energy per cell is unchanged. Arms: the arm of record and the three changes, each with and without speed awareness (declared speeds in the bids' time term and in the timing of a peer's schedule),
    # and two baselines as published (energy only, speed blind) for reference. The profile was fixed before any run.
    SPEEDS = {"speeds": [1.0, 0.5]}
    W3 = 3 * (Drone.BASE_COST_PER_CELL / sim_clock.STEPS_PER_CELL)
    BLIND = {"speed_aware": False, "share_bundles": False}      # the speed-blind controls keep the head-count rule and nominal speed for every drone, whatever the default
    for label, extra in (("", dict(BLIND)), ("_speed", {"speed_aware": True, "share_bundles": False}), ("_timeh", {"time_weight_hungarian": W3, **BLIND}), ("_timeh_speed", {"time_weight_hungarian": W3, "speed_aware": True, "share_bundles": False}),
                         ("_keep_timeh", {"keep_schedule": True, "time_weight_hungarian": W3, **BLIND}), ("_keep_timeh_speed", {"keep_schedule": True, "time_weight_hungarian": W3, "speed_aware": True, "share_bundles": False})):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble{label}_hs_limited", f"Fleet-size rule, no gambling{label.replace('_', ' ')}, drones at nominal and half speed, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
              fleet_profile=SPEEDS, **extra)
    # Time x3 in the bid for every fleet size (CBBA included, where the arm of record bids on energy alone), on the mixed-speed fleet, speed blind and speed aware.
    for label, extra in (("", {}), ("_speed", {"speed_aware": True})):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble_time3{label}_hs_limited", f"Fleet-size rule, no gambling, time x3 in every bid{label.replace('_', ' ')}, drones at nominal and half speed, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
              fleet_profile=SPEEDS, time_weight=W3, **{"share_bundles": False, **({} if extra else {"speed_aware": False}), **extra})

    # Completion-time bid (Author: design the bid from the objective; Lagoudakis et al. 2005: the makespan bid is the whole path completion), for every fleet size, with the schedules kept and speed aware.
    # Equal speed (suffix none) and the mixed-speed fleet (suffix hs).
    for label, prof in (("", {}), ("_hs", {"fleet_profile": SPEEDS})):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble_cbid{label}_limited", f"Fleet-size rule, no gambling, completion-time bid, schedules kept{', drones at nominal and half speed' if prof else ''}, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
              keep_schedule=True, speed_aware=True, completion_bid=True, **prof)

    # The completion-time bid with one trip committed at a time (rolling): a drone flies the first task of its bundle and decides again after the delivery.
    for label, prof in (("", {}), ("_hs", {"fleet_profile": SPEEDS})):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble_cbid_roll{label}_limited", f"Fleet-size rule, no gambling, completion-time bid, schedules kept, one trip committed at a time{', drones at nominal and half speed' if prof else ''}, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
              keep_schedule=True, speed_aware=True, completion_bid=True, rolling=True, **prof)

    # Bundle length from each agent's share of the team's capability (speed, limited by energy), on the arm of record, the speed-aware time bid and the completion bid; equal speed and the mixed-speed fleet.
    BASES = {"": ({}, "arm of record"), "_keep_timeh_speed": ({"keep_schedule": True, "time_weight_hungarian": W3}, "schedules kept, time x3 with Hungarian"), "_cbid": ({"keep_schedule": True, "completion_bid": True}, "completion-time bid")}
    for bkey, (bx, bdesc) in BASES.items():
        for label, prof in (("", {}), ("_hs", {"fleet_profile": SPEEDS})):
            if bkey == "_keep_timeh_speed" and not prof:
                continue
            clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble{bkey}_share{label}_limited", f"Fleet-size rule, no gambling, {bdesc}, bundles by capability share{', drones at nominal and half speed' if prof else ''}, radius 8.",
                  arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
                  sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
                  speed_aware=True, share_bundles=True, **bx, **prof)

    # Ablation: the head-count bundle rule ceil(open tasks / agents) and nominal speed for every drone (what every consensus arm did before capability shares became the default).
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_rule_nogamble_headcount_limited", "Fleet-size rule, no gambling, bundles by head count and no declared speed (ablation of capability shares), radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
          share_bundles=False, speed_aware=False)

    # The baselines with time x3 in the bid on the mixed-speed fleet: each prices the trip at the flying drone's declared speed (the centralised one at each drone's speed).
    for src_id in ("naive_decentralized_reactive", "decentralized_auction_auction_baseline", "cbba_single_task_cbba_single", "cbba_full_cbba_bundle"):
        clone(src_id, "time3_hs_limited", "Baseline with time x3 in the bid, drones at nominal and half speed, radius 8.", time_weight=W3, fleet_profile=SPEEDS)
    cspec = dict(by_id["centralized_cc_hungarian_hungarian"])
    cspec.update(name="centralized_cc_hungarian_time3_hs", status="pending", snapshot=None, time_weight=W3, fleet_profile=SPEEDS, description="Centralised Hungarian with time x3 in the bid, drones at nominal and half speed, radius 8.")
    derived.append(cspec)
    by_id[f"{cspec['name']}_{cspec['variant']}"] = cspec

    # CBBA bundle with the bundle length set by each agent's capability share (as SPPDEC does), beside the published fixed length of 3.
    clone("cbba_full_cbba_bundle", "capability_bundle", "CBBA with bundle length by capability share of the open tasks (declared fleet information), published conflict resolution.", capability_bundle=True)
    for src_id in ("naive_decentralized_reactive", "cbba_full_cbba_bundle"):
        clone(src_id, "hs_limited", "Baseline as published, drones at nominal and half speed, radius 8.", fleet_profile=SPEEDS)
    # Equal-importance bid (Author: the mission is energy efficiency and speed, both important): the completion bid with time and energy each as a share of its own ideal for the whole mission, summed 1:1.
    for label, prof in (("", {}), ("_hs", {"fleet_profile": SPEEDS})):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_rule_nogamble_equalbid{label}_limited", f"Fleet-size rule, no gambling, equal-importance time and energy bid, schedules kept{', drones at nominal and half speed' if prof else ''}, radius 8.",
              arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True,
              keep_schedule=True, completion_bid=True, equal_bid=True, **prof)

    for src_id in ("cbba_single_task_cbba_single", "decentralized_auction_auction_baseline"):
        clone(src_id, "hs_limited", "Baseline as published, drones at nominal and half speed, radius 8.", fleet_profile=SPEEDS)
    cspec = dict(by_id["centralized_cc_hungarian_hungarian"])
    cspec.update(name="centralized_cc_hungarian_hs", status="pending", snapshot=None, fleet_profile=SPEEDS, description="Centralised Hungarian as published, drones at nominal and half speed, radius 8.")
    derived.append(cspec)
    by_id[f"{cspec['name']}_{cspec['variant']}"] = cspec

    # A peer's agreed schedule is kept until observation contradicts it (a guess from the peer's motion does not replace it), alone and with the assignments standing.
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_rule_nogamble_keep_limited", "Fleet-size rule, no gambling, a peer's schedule is not replaced by a motion guess, radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True, keep_schedule=True)
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_rule_nogamble_keep_commit_limited", "Fleet-size rule, no gambling, schedules kept and assignments stand, radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True, keep_schedule=True, commitments=True)

    # When every parcel the drone believes open is predicted gone, it counts (exact at the station) before it flies, on top of the fleet-size rule and no gambling.
    clone("sppdec_cbba_assumed_bids_symmetric", "consensus_rule_nogamble_count_limited", "Fleet-size rule, no gambling, count before chasing a parcel believed gone, radius 8.",
          arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
          sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, model_rule={"max_agents": 4}, observe_contested=False, no_gamble=True, count_first=True)

    # Does the fairness weight (beta = 0.02 x the energy used, added to a bid) steer the last parcel away from the best-placed agent? The single-model consensus arms with beta = 0.
    for mname, models in (("hungarian", ["hungarian"]), ("cbba", ["cbba_rounds"])):
        clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_{mname}_beta0_limited", f"Reasoning with {mname} alone, no fairness weight (beta 0), radius 8.", arbiter="context_rules",
              coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False,
              sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True,
              consensus_models=models, config_variant="baseline_only")

    # Time in the bid (Author): a task's bid is its energy plus time_weight times the steps its trip takes; time_weight is a multiple of the energy a drone spends flying one
    # step unloaded. The same option for every strategy: the baselines as published (energy alone, no suffix) and time-matched (suffix time1 / time3).
    E_STEP = Drone.BASE_COST_PER_CELL / sim_clock.STEPS_PER_CELL
    for mult in (1, 3):
        weight = mult * E_STEP
        for mname, models in (("hungarian", ["hungarian"]), ("cbba", ["cbba_rounds"])):
            clone("sppdec_cbba_assumed_bids_symmetric", f"consensus_{mname}_time{mult}_limited", f"Reasoning with {mname} alone, time in the bid x{mult}, radius 8.",
                  arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
                  own_rate_known=False, sensing={"radius": 8}, station_memory="observed", freshness=True, ledger_complete=True, consensus=True, consensus_models=models,
                  time_weight=weight)
        for src_id in ("naive_decentralized_reactive", "decentralized_auction_auction_baseline", "cbba_single_task_cbba_single", "cbba_full_cbba_bundle"):
            clone(src_id, f"time{mult}_limited", f"Baseline with time in the bid x{mult}, radius 8.", time_weight=weight)
        # the centralised strategy takes its assignment mode from the variant, so the label goes on the name
        cspec = dict(by_id["centralized_cc_hungarian_hungarian"])
        cspec.update(name=f"centralized_cc_hungarian_time{mult}", status="pending", snapshot=None, time_weight=weight,
                     description=f"Centralised Hungarian with time in the bid x{mult}, radius 8.")
        derived.append(cspec)
        by_id[f"{cspec['name']}_{cspec['variant']}"] = cspec

    # Whole fleets of one personality (standard batteries): does a personality change the fleet's results? pers_<personality>_<tag>.
    for tag, sens in CHANNELS[:1]:
        for pers in ("rational", "selfish", "altruistic", "reckless"):
            clone("sppdec_cbba_assumed_bids_symmetric", f"pers_{pers}_{tag}", f"Fleet of all {pers}, sensing: {tag}.", arbiter="context_rules",
                  coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
                  own_rate_known=False, sensing=sens, agent_personalities=[pers])

    # Declared weights (information model, shared by every strategy): the weight of an order is known from the start like its location; the true weight is learned on
    # pickup. Without it a parcel not yet sensed carries an unseen default (1.0; SPPDEC: its own w-bar). dw: the same arm with declared weights, limited sensing.
    for src_id in ("naive_decentralized_reactive", "decentralized_auction_auction_baseline", "cbba_single_task_cbba_single", "cbba_full_cbba_bundle"):
        clone(src_id, "aw_limited_dw", "Baseline with declared weights, limited sensing.", sensing=CHANNELS[0][1], declared_weights=True)
    clone("sppdec_cbba_assumed_bids_symmetric", "aw_limited_dw", "Rule list with declared weights, limited sensing.", arbiter="context_rules", coordination_model="hungarian",
          observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, sensing=CHANNELS[0][1], declared_weights=True)
    clone("sppdec_cbba_assumed_bids_symmetric", "sm_limited_dw", "Predicted-station stop with declared weights, limited sensing.", arbiter="context_rules", coordination_model="hungarian",
          observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, sensing=CHANNELS[0][1], station_memory="observed",
          ledger_stop=True, station_model=True, declared_weights=True)

    # Mixed personalities (rational, selfish, altruistic, reckless: different w-bar and beta; assigned in turn to the drones). Peers know each other's
    # personality from the shared start information. mix_: the rule list without station awareness (a0), with the stop on the live count (a2), on the rate
    # estimate (a5), and on the predicted station with the peer model using each peer's own personality (sm) or assuming peers are like the drone (smnp).
    MIX = ["rational", "selfish", "altruistic", "reckless"]
    for tag, sens in CHANNELS[:1]:
        kw = dict(arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto",
                  use_self_check=True, own_rate_known=False, sensing=sens, agent_personalities=MIX)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_a0", f"Mixed fleet, rule list, sensing: {tag}.", **kw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_a2", f"Mixed fleet, stop on the live station count, sensing: {tag}.", ledger_stop=True, **kw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_a5", f"Mixed fleet, stop on the rate estimate, sensing: {tag}.",
              station_memory="predictive_avg", ledger_stop=True, **kw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_sm", f"Mixed fleet, stop on the predicted station, personality-aware peer model, sensing: {tag}.",
              station_memory="observed", ledger_stop=True, station_model=True, **kw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_smnp", f"Mixed fleet, stop on the predicted station, peers assumed like the drone, sensing: {tag}.",
              station_memory="observed", ledger_stop=True, station_model=True, peer_personalities=False, **kw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_pmmnp", f"Mixed fleet, peer model in shadow mode, peers assumed like the drone, sensing: {tag}.",
              station_memory="observed", peer_model_shadow=True, peer_personalities=False, **kw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mix_{tag}_pmm", f"Mixed fleet, peer model in shadow mode with the station memory, sensing: {tag}.",
              station_memory="observed", peer_model_shadow=True, **kw)

    # The peer model in the reasoning (Hungarian and CBBA alike read the same peer cost function): rc_ peers' costs from the model (predicted position, the rest of a
    # carrying peer's flight to its predicted drop cell, the peer's own declared personality); rt_ the same plus every agent's delivery leg to the predicted drop cell.
    # Both with the predicted-station stop (as sm_). mixrc_ / mixrt_: the same in the mixed-personality fleet.
    for tag, sens in CHANNELS[:1]:
        kw0 = dict(arbiter="context_rules", coordination_model="hungarian", observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto",
                    use_self_check=True, own_rate_known=False, sensing=sens, station_memory="observed", ledger_stop=True, station_model=True)
        clone("sppdec_cbba_assumed_bids_symmetric", f"smv_{tag}", f"Rule list, predicted-station stop with a look at the station centre before parking, sensing: {tag}.",
              **kw0, ledger_verify=True)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mixsmv_{tag}", f"Mixed fleet, predicted-station stop with a look before parking, sensing: {tag}.",
              agent_personalities=["rational", "selfish", "altruistic", "reckless"], **kw0, ledger_verify=True)
        for pre, extra in (("rcv", dict(peer_model_cost=True, ledger_verify=True)), ("rc", dict(peer_model_cost=True))):
            clone("sppdec_cbba_assumed_bids_symmetric", f"{pre}_{tag}", f"Rule list, predicted station, {pre}: peer model in the costs, sensing: {tag}.", **kw0, **extra)
            clone("sppdec_cbba_assumed_bids_symmetric", f"mix{pre}_{tag}", f"Mixed fleet, predicted station, {pre}: peer model in the costs, sensing: {tag}.",
                  agent_personalities=["rational", "selfish", "altruistic", "reckless"], **kw0, **extra)
        # CBBA in the head, the bundle protocol on the whole path cost: cbba_rounds with the peers from the peer model; mixcbba_rounds the same in the mixed fleet.
        clone("sppdec_cbba_assumed_bids_symmetric", f"cbba_rounds_{tag}", f"Rule list, CBBA bundle protocol in the head on the whole path, sensing: {tag}.",
              **dict(kw0, coordination_model="cbba"), peer_model_cost=True)
        clone("sppdec_cbba_assumed_bids_symmetric", f"mixcbba_rounds_{tag}", f"Mixed fleet, CBBA bundle protocol in the head on the whole path, sensing: {tag}.",
              agent_personalities=["rational", "selfish", "altruistic", "reckless"],
              **dict(kw0, coordination_model="cbba"), peer_model_cost=True)
        clone("sppdec_cbba_assumed_bids_symmetric", f"cbba_rc_{tag}", f"Rule list with CBBA in the head, predicted station, peer model in the costs, sensing: {tag}.",
              **dict(kw0, coordination_model="cbba"), peer_model_cost=True, ledger_verify=True)

    # Communication-free MARL baseline (IPPO, learned task choice; .ignore/revision/marl_baseline_design.md). `marl_ippo_<sensing>` takes
    # its policy file from the environment variable MARL_WEIGHTS (training and ad hoc runs); `marl_ippo_s<k>_<sensing>` is the policy of
    # independent training run k (experiments/marl/seed<k>/best.npz), for the matched evaluation batches.
    def marl_arm(variant, sensing, weights=None):
        spec = {"name": "marl_ippo", "variant": variant, "strategy_type": "marl", "topology": "decentralized",
                "planner": selected_planner, "snapshot": None, "status": "pending", "sensing": sensing,
                "description": f"Communication-free MARL baseline (IPPO), sensing {variant}."}
        if weights:
            spec["marl_weights"] = weights
        derived.append(spec)
        by_id[f"{spec['name']}_{variant}"] = spec

    for tag, sens in CHANNELS[:1]:
        marl_arm(tag, sens)
        for k in range(5):
            marl_arm(f"s{k}_{tag}", sens, f"experiments/marl/seed{k}/best.npz")
        for k in range(3):                      # the ablation: local critic (IPPO), no global information even in training
            marl_arm(f"local_s{k}_{tag}", sens, f"experiments/marl/local_seed{k}/best.npz")
        # Version 2.1 (2026-10-16; marl_baseline_design.md section 10): policies trained on the current world with scale-free features and a wide mixture; tags give the speed profile of the fleet
        # (`limited` equal; `limited_hs` 1.0 / 0.5; `limited_hs75` 1.0 / 0.75; `limited_hs3` 1.0 / 0.75 / 0.5, the transfer test only). `marl_ippo_<tag>` takes the policy from MARL_WEIGHTS (training);
        # `marl_ippo_v2s<k>_<tag>` is experiments/marl2/seed<k>/best.npz, `marl_ippo_v2local_s<k>_<tag>` experiments/marl2/local_seed<k>/best.npz.
        for tg, speeds in (("limited", None), ("limited_hs", [1.0, 0.5]), ("limited_hs75", [1.0, 0.75]), ("limited_hs3", [1.0, 0.75, 0.5])):
            if speeds:
                marl_arm(tg, sens)
                derived[-1]["fleet_profile"] = {"speeds": speeds}
            for k in range(5):
                marl_arm(f"v2s{k}_{tg}", sens, f"experiments/marl2/seed{k}/best.npz")
                if speeds:
                    derived[-1]["fleet_profile"] = {"speeds": speeds}
                marl_arm(f"v2local_s{k}_{tg}", sens, f"experiments/marl2/local_seed{k}/best.npz")
                if speeds:
                    derived[-1]["fleet_profile"] = {"speeds": speeds}

    # Task-level SLM told the territory convention in words (qwen3-vl 8B instruct, no thinking), limited sensing: with every task
    # labelled with its owner, and with only the drones' start cells listed (the SLM works out the owners). The reference arms are
    # the plain rule list, the coded territory rule and the task-level SLM without the convention (task_limited_slm_vl8b).
    for hint in ("labels", "raw", "both"):
        clone("sppdec_cbba_assumed_bids_symmetric", f"tslm_limited_{hint}",
              f"Task-level SLM (qwen3-vl:8b-instruct) told the territory convention ({hint}), limited sensing.",
              arbiter="context_slm_task", slm_model="qwen3-vl:8b-instruct", coordination_model="hungarian",
              observe_every_step=True, assumption_memory=True, cbba_bundle_size="auto", use_self_check=True,
              own_rate_known=False, sensing=CHANNELS[0][1], territory_hint=hint)

    # Task-level SLM: it chooses the parcel the drone takes next (not a reasoning model), shown the world and the four
    # models' proposals as advice. Non-thinking (qwen3-vl 8B instruct) and thinking (qwen3:8b).
    for scn, skw in LADDER_SCENARIOS[:2]:
        common = dict(coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
                      cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, **skw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"task_{scn}_slm_vl8b",
              f"Task-level SLM (qwen3-vl:8b-instruct) chooses the parcel, shown the world, scenario {scn}.",
              arbiter="context_slm_task", slm_model="qwen3-vl:8b-instruct", **common)
        clone("sppdec_cbba_assumed_bids_symmetric", f"task_{scn}_slm_q8b_think",
              f"Task-level SLM (qwen3:8b, thinking) chooses the parcel, shown the world, scenario {scn}.",
              arbiter="context_slm_task", slm_model="qwen3:8b", slm_think=True, **common)

    # Decision time charged (Jetson-class onboard hardware, the model's fixed prompt cached): the free SLM shown the world,
    # blocking (the drone waits for each decision) against plan ahead (the model plans a bundle of tasks; the drone takes the
    # next one still open without asking, and the next bundle is planned while it delivers).
    for scn, skw in LADDER_SCENARIOS:
        common = dict(coordination_model="hungarian", observe_every_step=True, assumption_memory=True,
                      cbba_bundle_size="auto", use_self_check=True, own_rate_known=False, **skw)
        clone("sppdec_cbba_assumed_bids_symmetric", f"latency_{scn}_slm_vl8b_world_blocking",
              f"Free SLM (qwen3-vl:8b-instruct) shown the world, Jetson-class decision time, the drone waits for each decision, scenario {scn}.",
              arbiter="context_slm_free", slm_model="qwen3-vl:8b-instruct", world_view=True, hardware="jetson_orin_nano_8gb",
              prompt_cache=True, **common)
        clone("sppdec_cbba_assumed_bids_symmetric", f"latency_{scn}_slm_vl8b_world_plan_ahead",
              f"Free SLM (qwen3-vl:8b-instruct) shown the world, Jetson-class decision time, plan ahead with background planning, scenario {scn}.",
              arbiter="context_slm_free", slm_model="qwen3-vl:8b-instruct", world_view=True, plan_ahead=True,
              hardware="jetson_orin_nano_8gb", prompt_cache=True, **common)

    # Heterogeneous fleet: every compared arm runs on the same fleet.
    for src_id, tag in [
        ("naive_decentralized_reactive", "naive"),
        ("cbba_single_task_cbba_single", "cbba_st"),
        ("centralized_cc_hungarian_hungarian", "c_hung"),
        ("sppdec_arb_always_greedy", "greedy"),
        ("sppdec_cbba_assumed_bids_symmetric", "cbba_symmetric_known"),
    ]:
        if tag == "cbba_symmetric_known":
            clone(src_id, "cbba_assumed_bids_symmetric_unequal_batteries_known",
                  f"{tag} on a heterogeneous fleet (rates {HETERO_FLEET['rates']}).", fleet_profile=HETERO_FLEET)
        elif tag == "greedy":
            clone(src_id, "greedy_unequal_batteries",
                  f"{tag} on a heterogeneous fleet (rates {HETERO_FLEET['rates']}).", fleet_profile=HETERO_FLEET)
        else:
            clone(src_id, None, f"{tag} on a heterogeneous fleet (rates {HETERO_FLEET['rates']}).",
                  rename=True, fleet_profile=HETERO_FLEET)
    clone("sppdec_cbba_assumed_bids_symmetric", "cbba_assumed_bids_symmetric_unequal_batteries_assumed_equal",
          "CBBA in head, shared costs, heterogeneous fleet, peers wrongly assumed to burn energy like self.",
          fleet_profile=HETERO_FLEET, peer_rate_assumption="same_as_self")
    # Energy self-calibration check (Author question: is learning its own consumption online an advantage?): the arm of record with the declared consumption rate read from the specification and no online fitting.
    clone("sppdec_consensus_rule_nogamble_limited", f"{by_id['sppdec_consensus_rule_nogamble_limited']['variant']}_declared_energy", "Arm of record with the declared consumption rate and no online energy fitting.", use_self_check=False, own_rate_known=True)
    # The eight compared strategies (revision: every stress test now covers all of them).
    ALL_EIGHT = ("sppdec_consensus_rule_nogamble_limited", "cbba_single_task_cbba_single", "centralized_cc_hungarian_hungarian", "marl_ippo_limited",
                 "cbba_full_cbba_bundle", "cbba_full_capability_bundle", "decentralized_auction_auction_baseline", "naive_decentralized_reactive")
    # Unequal batteries for all eight (capacities alternate 100 and 60 over the fleet; speeds and rates equal), tag "ub".
    for src_id in ALL_EIGHT:
        if src_id.startswith("centralized_cc"):
            spec = dict(by_id[src_id]); spec.update(name=f"{spec['name']}_ub", status="pending", snapshot=None, fleet_profile=UNEQUAL_BATTERIES,
                                                   description=f"{src_id} with capacities alternating 100 and 60.")
            derived.append(spec); by_id[f"{spec['name']}_{spec['variant']}"] = spec
        else:
            clone(src_id, f"{by_id[src_id]['variant']}_ub", f"{src_id} with capacities alternating 100 and 60.", fleet_profile=UNEQUAL_BATTERIES)
    # Reserve-floor test (isolates what prevents agent loss): the hardware reserve every strategy checks against, set to 0 and to 6 instead of 3, tags "rf0" and "rf6".
    for rtag, floor in (("rf0", 0.0), ("rf6", 6.0)):
        for src_id in ALL_EIGHT:
            if src_id.startswith("centralized_cc"):
                spec = dict(by_id[src_id]); spec.update(name=f"{spec['name']}_{rtag}", status="pending", snapshot=None, fleet_profile={"reserve_floor": floor},
                                                       description=f"{src_id} with the reserve floor at {floor}.")
                derived.append(spec); by_id[f"{spec['name']}_{spec['variant']}"] = spec
            else:
                clone(src_id, f"{by_id[src_id]['variant']}_{rtag}", f"{src_id} with the reserve floor at {floor}.", fleet_profile={"reserve_floor": floor})
    # Both together: reserve floor 0 and the infeasibility belief switched off (what is left to prevent loss is the shared battery check).
    clone("sppdec_consensus_rule_nogamble_limited", f"{by_id['sppdec_consensus_rule_nogamble_limited']['variant']}_rf0_boff", "Reserve floor 0 and the infeasibility belief off.",
          fleet_profile={"reserve_floor": 0.0}, sweep={"belief_off": True})
    # Sensing stress test (R1.1, R3.3; .ignore/revision/sensing_test_design.md): the same sensor degradation for every compared strategy, on top of the default radius of 8 cells.
    # n2 = position error of 2 cells (standard deviation); l50 = half of the peer readings lost; n2l50 = both.
    for tag, stress in SENSING_STRESS.items():
        for src_id in ALL_EIGHT:
            if src_id.startswith("centralized_cc"):
                # The centralised variant names its assignment mode, so the tag goes on the name.
                spec = dict(by_id[src_id]); spec.update(name=f"{spec['name']}_{tag}", status="pending", snapshot=None, sensing=dict(DEFAULT_SENSING, **stress),
                                                       description=f"{src_id} with degraded sensing ({tag}: {stress}).")
                derived.append(spec); by_id[f"{spec['name']}_{spec['variant']}"] = spec
            else:
                clone(src_id, f"{by_id[src_id]['variant']}_{tag}", f"{src_id} with degraded sensing ({tag}: {stress}).", sensing=dict(DEFAULT_SENSING, **stress))
    # Sensitivity study of the arm of record (R1.9; .ignore/revision/sensitivity_design.md): one setting changed at a time, every drone, whatever its personality.
    for tag, sweep in SENSITIVITY.items():
        clone("sppdec_consensus_rule_nogamble_limited", f"{by_id['sppdec_consensus_rule_nogamble_limited']['variant']}_{tag}", f"Arm of record with {sweep}.", sweep=sweep)
    vault = base + derived
    for spec in vault:
        if not spec.get("sensing"):
            spec["sensing"] = dict(DEFAULT_SENSING)      # no arm runs on unlimited sensing
    return vault



class AgentSpawnError(Exception):
    pass


class ExperimentState(Enum):
    NEW = "new"
    RESUMABLE = "resumable"
    COMPLETED = "completed"


class ExperimentRunnerGUI:
    SCHEMA_VERSION = 1

    def __init__(self):
        # experiment metadata
        self.seed = int(time.time()) % (2 ** 31)
        # GUI-controlled defaults
        self.parcels = 15
        self.planners = ["local"]
        self.run_id = str(uuid.uuid4())[:8]
        self.experiment: Dict = {}
        self.experiment_dir: Optional[Path] = None
        self.experiment_json_path: Optional[Path] = None
        self._sweep_queue = None
        self._sweep_initialized = False
        self._strategy_initialized = False

        # communication config (placeholder)
        self.default_comm_config = {"reliability": 1.0, "delay_mean": 0.0, "delay_std": 0.0}

        # behavior controls
        self.max_steps = 60 * 60 * 5
        self.stall_seconds = 6.0  # if idle longer than this considered a stall when parcels remain

        # Drones as processes need a core each (and the world one); a drone process is about 0.1 GB, an embedded
        # model about 1.2 GB.
        self.checker = SystemCapacityChecker(
            ram_per_agent_gb=1.2 if os.getenv("OLLAMA_MODE") == "embedded" else 0.3,
            process_mode=True)
        self.system_limit = self.checker.max_supported_agents()
        self.fast_mode = True  # New: fast execution with status display

        # Normal pygame initialization for GUI mode
        pygame.init()
        pygame.key.set_repeat(250, 35)
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_size = pygame.display.get_surface().get_size()
        pygame.display.set_caption("Drone Agent - Experiment Runner")
        self.clock = pygame.time.Clock()

        # fonts to match games.py
        self.font = pygame.font.SysFont("Consolas", 20)
        self.narration_font = pygame.font.SysFont("Consolas", 22)
        self.title_font = pygame.font.SysFont("Consolas", 72)
        self.large_font = pygame.font.SysFont("Consolas", 48)
        self.images = self._load_graphics()

    def _show_modal(
            self,
            *,
            title: str,
            message: str,
            background=(30, 30, 30),
            title_color=(255, 80, 80),
            hint: str = "Press any key to continue",
            exit_keys=None,
    ):
        """
        Generic blocking modal renderer.

        Parameters
        ----------
        title : str
            Modal title text.
        message : str
            Message body. Newlines are respected.
        background : tuple
            RGB background color.
        title_color : tuple
            RGB title color.
        hint : str
            Hint text displayed at the bottom.
        exit_keys : set[int] | None
            Specific pygame keys that close the modal.
            If None, any key or mouse press closes it.
        """
        if exit_keys is not None:
            exit_keys = set(exit_keys)

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()

                if e.type == pygame.KEYDOWN:
                    if exit_keys is None or e.key in exit_keys:
                        return

                if exit_keys is None and e.type == pygame.MOUSEBUTTONDOWN:
                    return

            self.screen.fill(background)

            title_surf = self.large_font.render(title, True, title_color)
            self.screen.blit(
                title_surf,
                title_surf.get_rect(center=(self.screen_size[0] // 2, 180)),
            )

            y = 260
            for line in message.split("\n"):
                txt = self.font.render(line, True, (230, 230, 230))
                self.screen.blit(
                    txt,
                    txt.get_rect(center=(self.screen_size[0] // 2, y)),
                )
                y += 34

            hint_surf = self.font.render(hint, True, (180, 180, 180))
            self.screen.blit(
                hint_surf,
                hint_surf.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 120)),
            )

            pygame.display.flip()
            self.clock.tick(30)

    def show_error(self, title: str, message: str):
        """
        Blocking modal error dialog.

        Used for fatal or informational errors.
        Returns after user acknowledgement.
        """
        self._show_modal(
            title=title,
            message=message,
            background=(30, 30, 30),
            title_color=(255, 80, 80),
            hint="Press any key to continue",
            exit_keys=None,  # any key or mouse
        )

    def _show_error_and_restart_setup(self, message: Optional[str] = None):
        """
        Show a configuration error and restart setup flow.
        """
        self._show_modal(
            title="Invalid Configuration",
            message=message,
            background=(40, 44, 52),
            title_color=(240, 100, 100),
            hint="Press ENTER to return to setup",
            exit_keys={pygame.K_RETURN, pygame.K_ESCAPE},
        )

    def scan_sweep_config_state(self, sweep_name: str, agents: int, parcels: int):
        """
        Returns a dict describing the true state of a sweep configuration.
        """
        scale_dir = (
                EXPERIMENTS_DIR
                / sweep_name
                / f"A{agents}_P{parcels}"
        )

        if not scale_dir.exists():
            return {
                "exists": False,
                "completed": False,
                "pending_strategies": set(),
                "terminal_strategies": set(),
            }

        pending = set()
        terminal = set()
        experiment_completed = False  # Add this flag

        for exp_dir in scale_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
                continue

            exp_json = exp_dir / "experiment.json"
            if not exp_json.exists():
                continue

            try:
                data = json.loads(exp_json.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Check experiment-level completion first
            exp_status = data.get("status", "in_progress")
            current_index = data.get("current_index", 0)
            total_strategies = len(data.get("strategies", []))

            # Experiment is truly completed if:
            # 1. Status is "completed", OR
            # 2. current_index >= total_strategies (all strategies processed)
            if exp_status == "completed": # or current_index >= total_strategies:
                experiment_completed = True

            for s in data.get("strategies", []):
                name = s.get("family") or s.get("name")
                status = s.get("status")

                if status in {"completed", "aborted", "failed", "infeasible", "timeout", "stalled"}:
                    terminal.add(name)
                else:
                    pending.add(name)

        # A strategy is truly pending only if it never appears terminal anywhere
        truly_pending = pending - terminal

        return {
            "exists": True,
            "completed": experiment_completed,  # Use experiment-level completion
            "pending_strategies": truly_pending,
            "terminal_strategies": terminal,
        }

    def show_experiment_setup(
            self,
            initial_agents: int = 1,
            initial_parcels: int = 15,
            min_agents: int = 1,
            max_agents: int = 50,
            min_parcels: int = 0,
            max_parcels: int = 1000,
    ):
        """
        GUI screen to choose:
          - number of agents
          - number of parcels

        Controls:
          - TAB switches between fields
          - UP / DOWN increment selected field
          - Type digits to enter value
          - ENTER to accept
          - ESC to abort
        Returns:
          (n_agents, n_parcels)
        """
        max_agents = self.system_limit
        agents = initial_agents
        parcels = initial_parcels
        typing = ""
        active_field = "agents"  # or "parcels"

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()

                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        raise SystemExit()

                    elif e.key == pygame.K_TAB:
                        active_field = "parcels" if active_field == "agents" else "agents"
                        typing = ""

                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return agents, parcels

                    elif e.key == pygame.K_BACKSPACE:
                        typing = typing[:-1]

                    elif e.key == pygame.K_UP:
                        if active_field == "agents":
                            agents = min(max_agents, agents + 1)
                        else:
                            parcels = min(max_parcels, parcels + 1)
                        typing = ""

                    elif e.key == pygame.K_DOWN:
                        if active_field == "agents":
                            agents = max(min_agents, agents - 1)
                        else:
                            parcels = max(min_parcels, parcels - 1)
                        typing = ""
                    elif e.type == pygame.KEYDOWN and e.unicode.isdigit():
                        typing += e.unicode
                        v = int(typing)

                        if active_field == "agents":
                            agents = max(min_agents, min(max_agents, v))
                        else:
                            parcels = max(min_parcels, min(max_parcels, v))

            # -------- render --------
            self.screen.fill((40, 44, 52))

            header = self.large_font.render("Experiment Setup", True, (230, 230, 230))
            self.screen.blit(header, header.get_rect(center=(self.screen_size[0] // 2, 140)))
            limit_txt = self.font.render(
                f"System agent limit: {max_agents}",
                True,
                (160, 160, 160),
            )
            self.screen.blit(
                limit_txt,
                limit_txt.get_rect(center=(self.screen_size[0] // 2, 190)),
            )
            agents_color = (240, 240, 200) if active_field == "agents" else (180, 180, 180)
            parcels_color = (240, 240, 200) if active_field == "parcels" else (180, 180, 180)

            agents_txt = self.title_font.render(str(agents), True, agents_color)
            parcels_txt = self.title_font.render(str(parcels), True, parcels_color)

            self.screen.blit(
                self.font.render("Agents", True, agents_color),
                (self.screen_size[0] // 2 - 260, 300),
            )
            self.screen.blit(agents_txt, agents_txt.get_rect(center=(self.screen_size[0] // 2 - 200, 360)))

            self.screen.blit(
                self.font.render("Parcels", True, parcels_color),
                (self.screen_size[0] // 2 + 100, 300),
            )
            self.screen.blit(parcels_txt, parcels_txt.get_rect(center=(self.screen_size[0] // 2 + 200, 360)))

            hint = self.font.render(
                "TAB switch field | UP/DOWN adjust | Type digits | ENTER start | ESC cancel",
                True,
                (200, 200, 200),
            )
            self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 120)))

            pygame.display.flip()
            self.clock.tick(30)

    def _load_graphics(self):
        def _load_and_scale(name, scale_factor):
            p = GRAPHICS_DIR / name
            img = load_image(p)
            return scale_to_cell(img, GRID_SIZE, scale_factor) if img else None

        drone_static = _load_and_scale("drone_static.png", DRONE_SCALE)

        drone_rot_frames = []
        i = 0
        while True:
            p = GRAPHICS_DIR / f"drone_rotating_{i}.png"
            if p.exists():
                try:
                    img = pygame.image.load(str(p)).convert_alpha()
                    drone_rot_frames.append(scale_to_cell(img, GRID_SIZE, DRONE_SCALE))
                except Exception:
                    pass
                i += 1
            else:
                break
        if not drone_rot_frames:
            r = load_image(GRAPHICS_DIR / "drone_rotating.png")
            if r:
                drone_rot_frames = [scale_to_cell(r, GRID_SIZE, DRONE_SCALE)]

        drone_static_with_parcel = _load_and_scale("drone_static_with_parcel.png", DRONE_SCALE)

        drone_rot_with_parcel_frames = []
        i = 0
        while True:
            p = GRAPHICS_DIR / f"drone_rotating_with_parcel_{i}.png"
            if p.exists():
                try:
                    img = pygame.image.load(str(p)).convert_alpha()
                    drone_rot_with_parcel_frames.append(scale_to_cell(img, GRID_SIZE, DRONE_SCALE))
                except Exception:
                    pass
                i += 1
            else:
                break
        if not drone_rot_with_parcel_frames:
            r = load_image(GRAPHICS_DIR / "drone_rotating_with_parcel.png")
            if r:
                drone_rot_with_parcel_frames = [scale_to_cell(r, GRID_SIZE, DRONE_SCALE)]

        parcel_img = None
        try:
            parcel_img_raw = load_image(GRAPHICS_DIR / "parcel.png")
            parcel_img = scale_to_cell(parcel_img_raw, GRID_SIZE, PARCEL_SCALE) if parcel_img_raw else None
        except Exception:
            parcel_img = None

        return {
            "drone_static": drone_static,
            "drone_rot_frames": drone_rot_frames,
            "drone_static_with_parcel": drone_static_with_parcel,
            "drone_rot_with_parcel_frames": drone_rot_with_parcel_frames,
            "parcel_img": parcel_img,
        }

    def build_sweep_status(self, sweep_name: str, sweep_configs: list[dict]):
        rows = []
        for cfg in sweep_configs:
            agents = cfg["agents"]
            parcels = cfg["parcels"]

            scan = self.scan_sweep_config_state(
                sweep_name=sweep_name,
                agents=agents,
                parcels=parcels,
            )

            completed = scan["completed"]
            pending = len(scan["pending_strategies"])

            if completed:
                state = "completed"
            elif pending > 0:
                state = "partial"
            else:
                state = "new"

            rows.append({
                "agents": agents,
                "parcels": parcels,
                "state": state,
                "pending": pending,
                "terminal": len(scan["terminal_strategies"]),
            })

        return rows

    def _load_or_create_experiment(self, root: Path) -> ExperimentState:
        """
        Load or resume an experiment under a specific root directory.

        Rules:
        - If an experiment exists and has runnable strategies, let user choose which to run
        - Completed strategies are blocked at UI level
        - If all strategies are terminal, treat experiment as completed
        - Never create a second experiment under the same root
        """

        self._resumed = False
        self._experiment_completed = False

        TERMINAL = {"completed", "failed", "infeasible", "timeout", "stalled"}

        # -------------------------------------------------
        # 1) Find existing experiment under this root
        # -------------------------------------------------
        exp_jsons = sorted(
            root.glob("experiment_*/experiment.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not exp_jsons:
            return ExperimentState.NEW

        candidate = exp_jsons[0]

        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            strategies = data.get("strategies", [])

            if not strategies:
                print("[EXPERIMENT] No strategies found")
                self._experiment_completed = True
                return ExperimentState.COMPLETED

            # -------------------------------------------------
            # Normalize experiment status
            # -------------------------------------------------
            exp_status = data.get("status", "in_progress")

            if exp_status == "completed":
                if any(s.get("status") in {"pending", "aborted"} for s in strategies):
                    data["status"] = "in_progress"
                    exp_status = "in_progress"

            # Clear snapshots for terminal strategies only
            for s in strategies:
                if s.get("status") in TERMINAL:
                    s["snapshot"] = None

            # -------------------------------------------------
            # Check if anything is runnable at all
            # -------------------------------------------------
            runnable_exists = any(
                s.get("status") in {"pending", "aborted"}
                for s in strategies
            )

            if not runnable_exists:
                data["status"] = "completed"
                self._experiment_completed = True
                return ExperimentState.COMPLETED

            # -------------------------------------------------
            # Extract sweep context safely
            # -------------------------------------------------
            try:
                sweep_name = root.parents[1].name
            except IndexError:
                sweep_name = "manual"

            scale = root.name  # A{agents}_P{parcels}
            try:
                parts = scale.split("_")
                agents = int(parts[0][1:])
                parcels = int(parts[1][1:])
            except Exception:
                agents = parcels = -1

            # -------------------------------------------------
            # STRATEGY SELECTION MODAL (AUTHORITATIVE)
            # -------------------------------------------------
            if getattr(self, '_strategy_initialized', False):
                # Skip modal - use current_index from experiment (works for both sweep and manual)
                selected_index = data.get("current_index", 0)
                print(f"[AUTO] Auto-resuming at strategy index {selected_index}")
            elif USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False):
                # In sweep mode, automatically skip completed strategies and start from next pending
                pending_strategies = [i for i, s in enumerate(strategies) if s.get("status") in {"pending", "aborted"}]
                if pending_strategies:
                    selected_index = pending_strategies[0]  # Start from first pending strategy
                    print(f"[SWEEP] Auto-selecting first pending strategy at index {selected_index}")
                else:
                    # All strategies completed, mark experiment as completed
                    print(f"[SWEEP] All strategies completed, marking experiment as done")
                    data["status"] = "completed"
                    self._experiment_completed = True
                    return ExperimentState.COMPLETED
            else:
                # Show selection modal for manual mode only
                selected_index = self._show_experiment_status_modal(
                    sweep_name=sweep_name,
                    agents=agents,
                    parcels=parcels,
                    run_id=data.get("run_id", "unknown"),
                    current_index=data.get("current_index", 0),
                    strategies=strategies,
                    experiment_dir=candidate.parent,
                )

                if selected_index is None:
                    print("[EXPERIMENT] Experiment discarded by user")
                    self._experiment_completed = True
                    return ExperimentState.COMPLETED

                # Mark strategy selection as initialized after first choice
                self._strategy_initialized = True
            # -------------------------------------------------
            # RESUME PATH (STRATEGY-LEVEL)
            # -------------------------------------------------
            self.experiment = data

            self.experiment.setdefault("run_id", str(uuid.uuid4())[:8])
            self.experiment.setdefault("schema_version", self.SCHEMA_VERSION)
            self.experiment.setdefault("comm_config", self.default_comm_config.copy())
            self.experiment.setdefault(
                "created_at",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            )
            self.experiment.setdefault("results", [])

            # User-selected strategy index is authoritative
            self.experiment["current_index"] = int(selected_index)

            # Seed consistency
            if "seed" in self.experiment:
                self.seed = self.experiment["seed"]
            else:
                self.experiment["seed"] = self.seed

            self.experiment["status"] = "in_progress"
            self.experiment_json_path = candidate
            self.experiment_dir = candidate.parent
            self.parcels = len(self.experiment.get("parcels", []))
            self.run_id = self.experiment["run_id"]

            self._resumed = True

            print(f"[EXPERIMENT] Resuming experiment {candidate}")
            print(f"[EXPERIMENT] Run ID: {self.run_id}")
            print(f"[EXPERIMENT] Starting at strategy index {selected_index}")

            return ExperimentState.RESUMABLE

        except Exception as e:
            print(f"[EXPERIMENT] Failed to load experiment {candidate}: {e}")
            self._experiment_completed = True
            return ExperimentState.COMPLETED

    def _write_flight_artifacts(self, run_id, planner, strategy_name, flight_data):
        """
        Persist flight recorder data into the current experiment directory.
        """

        if self.experiment_dir is None:
            raise RuntimeError("Experiment directory not initialized")

        # Create a subfolder per strategy
        strategy_dir = self.experiment_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        agents_path = strategy_dir / "agents.json"
        agents_path.write_text(json.dumps(flight_data["agents"], indent=2), encoding="utf-8")
        paths["agents"] = f"{strategy_name}/agents.json"

        fleet_path = strategy_dir / "fleet.json"
        fleet_path.write_text(
            json.dumps(
                {
                    "parcels": flight_data["parcels"],
                    "fleet_events": flight_data["fleet_events"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["fleet"] = f"{strategy_name}/fleet.json"

        heatmap_path = strategy_dir / "heatmap.json"
        heatmap_path.write_text(
            json.dumps(flight_data["heatmap"], indent=2),
            encoding="utf-8",
        )
        paths["heatmap"] = f"{strategy_name}/heatmap.json"

        meta_path = strategy_dir / "meta.json"
        meta_path.write_text(
            json.dumps(flight_data["meta"], indent=2),
            encoding="utf-8",
        )
        paths["meta"] = f"{strategy_name}/meta.json"

        return paths

    def _write_experiment_json(self):
        if not self.experiment_json_path:
            return

        # Ensure directory exists
        self.experiment_json_path.parent.mkdir(parents=True, exist_ok=True)

        self.experiment_json_path.write_text(
            json.dumps(self.experiment, indent=2),
            encoding="utf-8",
        )

    # -----------------------
    # Snapshot and logging helpers
    # -----------------------
    def _ensure_snapshot_dict(self, strategy_idx: int) -> Dict:
        """
        Ensure that strategies[strategy_idx]['snapshot'] is a dict. Return the snapshot dict.
        This prevents NoneType errors when appending events/messages.
        """
        strategies = self.experiment.setdefault("strategies", [])
        if strategy_idx < 0 or strategy_idx >= len(strategies):
            raise IndexError("strategy_idx out of range")
        strat = strategies[strategy_idx]
        if strat.get("snapshot") is None:
            strat["snapshot"] = {}
        return strat["snapshot"]

    def _append_event(self, strategy_idx: int, ev: Dict):
        """
        Append an event dict to the current strategy's snapshot.events list.
        Useful for moves, pick/drop attempts, unsafe events, etc.
        """
        try:
            snap = self._ensure_snapshot_dict(strategy_idx)
        except IndexError:
            return
        evs = snap.setdefault("events", [])
        evs.append(ev)
        if len(evs) % 500 == 0:
            self._write_experiment_json()

    def _restore_snapshot(self, snapshot: Dict, strategy_name: str, flight_recorder: FlightRecorder) -> Tuple[
        Terrain, List[Drone], int, List[Parcel]]:
        """
        Restore world state from a snapshot and return (terrain, agents_list, steps, parcels).
        Snapshot is authoritative.
        """

        # ----------------------------
        # Validate snapshot
        # ----------------------------
        agents_state = snapshot.get("agents_state")
        if not agents_state:
            raise ValueError("Snapshot missing agents_state")

        # ----------------------------
        # Lock experiment agents to snapshot
        # ----------------------------
        self.experiment["agents"] = [
            {
                "id": a["agent_id"],
                "type": "drone",
            }
            for a in agents_state
        ]

        # ----------------------------
        # Recreate terrain
        # ----------------------------
        terrain = Terrain(
            GRID_SIZE,
            self.screen_size,
            parcel_img=self.images.get("parcel_img"),
            parcel_scale=PARCEL_SCALE,
        )

        cols = self.screen_size[0] // GRID_SIZE
        rows = self.screen_size[1] // GRID_SIZE
        center_col = max(2, cols // 2 - 2)
        center_row = max(2, rows // 2 - 2)
        terrain.add_station(center_col, center_row, w=5, h=5)

        # ----------------------------
        # Recreate parcels
        # ----------------------------
        recreated_parcels: List[Parcel] = []

        for ps in snapshot.get("parcels", []):
            snapshot_parcel_id = ps.get("id")

            p = Parcel(
                int(ps["col"]),
                int(ps["row"]),
                terrain.grid_size
            )

            if snapshot_parcel_id is not None:
                p.id = str(snapshot_parcel_id)

            p.picked = bool(ps.get("picked", False))
            p.delivered = bool(ps.get("delivered", False))

            if "weight" in ps:
                p.weight = float(ps["weight"])

            recreated_parcels.append(p)

            print(f"[RESTORE] Parcel at ({int(ps['col'])},{int(ps['row'])}) - "
                  f"ID: {p.id}, picked: {p.picked}, delivered: {p.delivered}")

        terrain.parcels = recreated_parcels

        # ----------------------------
        # Recreate agents (with onboard SLM)
        # ----------------------------
        agents: List[Drone] = []
        base_port = 11434

        for idx, astate in enumerate(agents_state):
            agent_id = astate["agent_id"]

            drone = Drone(
                start_cell=(int(astate["col"]), int(astate["row"])),
                grid_size=GRID_SIZE,
                screen_size=self.screen_size,
                terrain=terrain,
                agent_id=agent_id,
                flight_recorder=flight_recorder
            )

            # precise position
            drone.pos = pygame.Vector2(
                float(astate["pos_x"]),
                float(astate["pos_y"]),
            )
            drone.col = int(astate["col"])
            drone.row = int(astate["row"])
            drone.last_cell = (drone.col, drone.row)

            # battery restore
            cap = float(astate.get("battery_capacity", drone.power.capacity))
            lvl = float(astate.get("battery_level", drone.power.level))
            drone.power.capacity = cap
            drone.power.level = min(lvl, cap)
            drone.energy_rate = float(astate.get("energy_rate", 1.0))
            drone.weight_factor = float(astate.get("weight_factor", drone.WEIGHT_FACTOR))

            # carried parcel restore
            carrying_id = astate.get("carrying_id")
            if carrying_id is not None:
                carrying_id_str = str(carrying_id)

                for p in recreated_parcels:
                    if getattr(p, "id", None) == carrying_id_str:
                        drone.carrying = p
                        print(f"[RESTORE] Agent {agent_id} is carrying parcel {carrying_id_str}")
                        break
                else:
                    print(
                        f"[RESTORE] WARNING: Agent {agent_id} should be carrying parcel {carrying_id_str} but it wasn't found"
                    )

            # ----------------------------
            # Restore embodied cognition
            # ----------------------------
            if OLLAMA_MODE == "external":
                drone.slm_base_url = "http://127.0.0.1:11434"
                drone.slm_enabled = True
                print(f"[RESTORE][SLM] Using external Ollama for {agent_id}")
            else:
                slm_port = base_port + idx + 1
                print(f"[RESTORE][SLM] Restarting {agent_id} on port {slm_port}")
                drone.start_slm(port=slm_port)

            terrain.register_agent_cell(drone, drone.col, drone.row)
            terrain.agents[drone.agent_id] = drone
            agents.append(drone)

            print(f"[RESTORE] Agent {agent_id} at ({drone.col},{drone.row}) - "
                  f"battery: {drone.power.level:.1f}%")

        steps = int(snapshot.get("steps", 0))

        print(f"[RESTORE] Snapshot restored: {len(agents)} agents, {len(recreated_parcels)} parcels, step {steps}")

        return terrain, agents, steps, recreated_parcels

    # -----------------------
    # Existing UI helpers
    # -----------------------
    def show_splash(self, timeout=2.0):
        # same look and feel as games.py splash
        splash_img = None
        p = GRAPHICS_DIR / "drone_static.png"
        if p.exists():
            splash_img = load_image(p)
            if splash_img:
                max_w = int(self.screen_size[0] * 0.6)
                max_h = int(self.screen_size[1] * 0.4)
                w, h = splash_img.get_size()
                scale = min(1.0, max_w / w, max_h / h)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                splash_img = pygame.transform.smoothscale(splash_img, new_size)

        start = time.time()
        while True:
            now = time.time()
            elapsed = now - start
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN or e.type == pygame.MOUSEBUTTONDOWN:
                    return
            self.screen.fill((24, 32, 48))
            if splash_img:
                img_rect = splash_img.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2 - 80))
                self.screen.blit(splash_img, img_rect)
                text_y = img_rect.bottom + 28
            else:
                text_y = self.screen_size[1] // 2 - 60

            title = self.title_font.render("Agentic Swarm Lab", True, (220, 230, 240))
            subtitle = self.large_font.render("Episodic LLM Guidance", True, (180, 200, 220))
            hint = self.font.render("Press any key to continue or wait...", True, (180, 180, 200))

            self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, text_y)))
            self.screen.blit(subtitle, subtitle.get_rect(center=(self.screen_size[0] // 2, text_y + 64)))
            self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, text_y + 140)))

            pygame.display.flip()
            self.clock.tick(60)
            if elapsed >= timeout:
                return

    def show_setup_and_planner_select(self):
        """
        Show a GUI screen that displays:
         - current experiment info (seed, parcels, planners)
         - allows choosing planner index with left/right or TAB
         - Enter to start
         - ESC to quit (abort)
        """
        planners = list(self.planners)
        planner_index = 0
        typing = ""
        info_lines = [
            f"Seed: {self.seed}",
            f"Parcels: {self.parcels}",
        ]

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        print("[EXPERIMENT] Setup aborted by user")
                        raise SystemExit()
                    elif e.key in (pygame.K_RIGHT, pygame.K_TAB):
                        planner_index = (planner_index + 1) % max(1, len(planners))
                    elif e.key == pygame.K_LEFT:
                        planner_index = (planner_index - 1) % max(1, len(planners))
                    elif e.key == pygame.K_RETURN or e.key == pygame.K_KP_ENTER:
                        return planners[planner_index]
                    elif e.key == pygame.K_BACKSPACE:
                        typing = typing[:-1]
                    elif e.unicode and e.unicode.isdigit():
                        typing += e.unicode
                        try:
                            v = int(typing)
                            if v > 1000:
                                typing = str(1000)
                        except ValueError:
                            typing = ""
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    planner_index = (planner_index + 1) % max(1, len(planners))

            self.screen.fill((40, 44, 52))
            header = self.large_font.render("Experiment Setup", True, (230, 230, 230))
            self.screen.blit(header, header.get_rect(center=(self.screen_size[0] // 2, 120)))

            for i, ln in enumerate(info_lines):
                text_surf = self.font.render(ln, True, (210, 210, 210))
                self.screen.blit(text_surf, (self.screen_size[0] // 2 - 220, 220 + i * 30))

            left = self.screen_size[0] // 2 - 220
            top = 340
            for i, p in enumerate(planners):
                txt = self.font.render(f"[{i + 1}] {p}", True,
                                       (240, 240, 200) if i == planner_index else (180, 180, 180))
                self.screen.blit(txt, (left + (i % 3) * 220, top + (i // 3) * 36))

            bottom_hint = self.font.render(f"Selected planner: {planners[planner_index]}   Press Enter to start", True,
                                           (230, 230, 230))
            self.screen.blit(bottom_hint, (self.screen_size[0] // 2 - 320, self.screen_size[1] - 120))

            pygame.display.flip()
            self.clock.tick(30)

    def _configure_planner(self, name: str):
        """
        Planner configuration hook.

        For snapshot-based planners (e.g., NaiveStrategy),
        planner selection is structural rather than parametric.
        No runtime configuration is performed.
        """
        print(f"[PLANNER] using planner '{name}' (no runtime configuration)")

    def _setup_agents(self, terrain: Terrain, *, strategy_name: str, strategy_spec: dict):
        agents = {}
        self._current_strategy = strategy_spec

        station = terrain.nearest_station(
            self.screen_size[0] // GRID_SIZE // 2,
            self.screen_size[1] // GRID_SIZE // 2,
        )

        if not station:
            raise AgentSpawnError("No station found. Cannot spawn agents.")

        station_cells = [
            (c, r)
            for r in range(station.row, station.row + station.h)
            for c in range(station.col, station.col + station.w)
        ]

        rng = random.Random(self.seed)
        rng.shuffle(station_cells)

        if len(station_cells) < len(self.experiment["agents"]):
            raise AgentSpawnError(
                f"Too many agents ({len(self.experiment['agents'])}) "
                f"for station capacity ({len(station_cells)}).\n\n"
                "Reduce the number of agents."
            )

        # ----------------------------------------
        # Build personality assignment pool
        # ----------------------------------------
        expanded_personalities = None

        if "personality_distribution" in strategy_spec:
            expanded_personalities = []
            for p, count in strategy_spec["personality_distribution"].items():
                expanded_personalities.extend([p] * count)

        elif "agent_personalities" in strategy_spec:
            expanded_personalities = list(strategy_spec["agent_personalities"])

        # ----------------------------------------
        # SLM port allocation (infrastructure-level)
        # ----------------------------------------
        base_port = 11434
        used_ports = set()

        for idx, (agent_def, start_cell) in enumerate(
                zip(self.experiment["agents"], station_cells)
        ):
            drone = Drone(
                start_cell=start_cell,
                grid_size=GRID_SIZE,
                screen_size=self.screen_size,
                terrain=terrain,
                agent_id=agent_def["id"],
            )

            # ----------------------------------------
            # Personality assignment
            # ----------------------------------------
            # Personalities apply only to arms that specify a composition. Plain SPPDEC
            # variants (combined, control, fairness_only, baseline_only) must keep their
            # variant (beta, w-bar); before the revision every drone got a cycled personality
            # here, which silently overrode the variant for all of them.
            if expanded_personalities:
                drone.personality = expanded_personalities[idx % len(expanded_personalities)]
            else:
                drone.personality = None

            print(f"[PERSONALITY] {drone.agent_id} -> {drone.personality}")

            # ----------------------------------------
            # Assign SLM ports for SPPDEC
            # ----------------------------------------
            if strategy_name.startswith("sppdec"):
                if OLLAMA_MODE == "external":
                    drone.slm_base_url = "http://127.0.0.1:11434"
                    drone.slm_enabled = True
                    print(f"[SLM] Using external Ollama for {drone.agent_id}")
                else:
                    slm_port = base_port + idx + 1
                    used_ports.add(slm_port)
                    drone.start_slm(port=slm_port)
                    print(f"[SLM] Booting {drone.agent_id} on port {slm_port}")

            # Heterogeneous fleet: per-drone consumption rate and battery capacity, cycled over
            # the fleet (spec["fleet_profile"] = {"rates": [...], "capacities": [...]}).
            profile = strategy_spec.get("fleet_profile") or {}
            if profile.get("rates"):
                drone.energy_rate = float(profile["rates"][idx % len(profile["rates"])])
            if profile.get("speeds"):
                drone.speed_scale = float(profile["speeds"][idx % len(profile["speeds"])])
            if profile.get("weight_factors"):
                drone.weight_factor = float(profile["weight_factors"][idx % len(profile["weight_factors"])])
            if profile.get("capacities"):
                cap = float(profile["capacities"][idx % len(profile["capacities"])])
                drone.power.capacity = cap
                drone.power.level = cap
            # What the drone declares, when it differs from what it actually consumes (miscalibration: a spec sheet
            # from other conditions). Without these keys the declared spec is whatever calibration measures.
            if profile.get("declared_rates"):
                drone.declared_rate = float(profile["declared_rates"][idx % len(profile["declared_rates"])])
            if profile.get("declared_weight_factors"):
                drone.declared_weight_factor = float(
                    profile["declared_weight_factors"][idx % len(profile["declared_weight_factors"])])
            # Battery reading quality during calibration (default: exact).
            drone.calibration_noise = float(profile.get("calibration_noise", 0.0))
            drone.calibration_resolution = float(profile.get("calibration_resolution", 0.0))
            if profile.get("reserve_floor") is not None:
                drone.reserve_floor = float(profile["reserve_floor"])

            agents[agent_def["id"]] = drone
            terrain.register_agent_cell(drone, start_cell[0], start_cell[1])
            terrain.agents[drone.agent_id] = drone

        if used_ports:
            print(f"[SLM] Active ports: {sorted(used_ports)}")

        self._active_drones = list(agents.values())
        return agents, 0, None

    def _setup_scene(self, maybe_snapshot: Optional[Dict], strategy_name: str, strategy_spec: dict,
                     flight_recorder: FlightRecorder) -> \
            Tuple[
                Terrain, List[Drone], int, List[Parcel]]:
        """
        Create Terrain and agents.
        If maybe_snapshot provided, restore world state from it.
        Returns (terrain, agents_list, steps_already_taken, all_parcels)
        """

        # ---- Snapshot path (AUTHORITATIVE) ----
        if maybe_snapshot is not None:
            terrain, agents, steps, all_parcels = self._restore_snapshot(snapshot=maybe_snapshot,
                                                                         strategy_name=strategy_name,
                                                                         flight_recorder=flight_recorder
                                                                         )  # Changed to 4 values
            self._active_drones = agents
            return terrain, agents, steps, all_parcels

        # ---- Fresh scene path (from experiment schema) ----

        # 1. Terrain
        terrain = Terrain(
            GRID_SIZE,
            self.screen_size,
            parcel_img=self.images.get("parcel_img"),
            parcel_scale=PARCEL_SCALE,
        )

        cols = self.screen_size[0] // GRID_SIZE
        rows = self.screen_size[1] // GRID_SIZE
        center_col = max(2, cols // 2 - 2)
        center_row = max(2, rows // 2 - 2)
        s = self.experiment["station"]
        terrain.add_station(s["col"], s["row"], w=s["w"], h=s["h"])

        # 2. Parcels (schema-driven)
        for pdef in self.experiment.get("parcels", []):
            terrain.add_parcel(int(pdef["col"]), int(pdef["row"]))
            p = terrain.parcel_at_cell(
                int(pdef["col"]),
                int(pdef["row"]),
                include_delivered=True,
            )
            if p:
                p.picked = bool(pdef.get("picked", False))
                p.delivered = bool(pdef.get("delivered", False))
                # Experiments created before v1.0-iotj revision have no stored weight
                # and keep the per-run random draw.
                if "weight" in pdef:
                    p.weight = float(pdef["weight"])
                if os.environ.get("PARCEL_WEIGHT_RANGE"):
                    # high-variance orders for a whole batch: the stored weights (uniform 0.5 to 2.0) are stretched linearly to [lo, hi]; every strategy of the batch sees the same weights
                    lo, hi = (float(v) for v in os.environ["PARCEL_WEIGHT_RANGE"].split(","))
                    p.weight = lo + (float(p.weight) - 0.5) / 1.5 * (hi - lo)
                if os.environ.get("ORDER_ARRIVAL"):
                    # dynamic arrival (revision): "frac,window": the last `frac` of the orders are released at steps spread evenly over 1..window (the same for every strategy)
                    frac, window = os.environ["ORDER_ARRIVAL"].split(",")
                    n_all = len(self.experiment.get("parcels", []))
                    idx = [i for i, q in enumerate(self.experiment.get("parcels", [])) if q is pdef][0]
                    n_late = int(round(float(frac) * n_all))
                    if idx >= n_all - n_late:
                        k = idx - (n_all - n_late)
                        rng = random.Random(f"arrival-{self.seed}-{idx}")
                        p.release_step = max(1, int((k + rng.random()) / max(1, n_late) * int(window)))
                if strategy_spec.get("parcel_weight_scale"):
                    p.weight = float(p.weight) * float(strategy_spec["parcel_weight_scale"])      # heavier orders (arm option)

        # 2b. Map with no-fly blocks (revision; off unless OBSTACLES is set): one layout per run seed, the same for every strategy
        from src import obstacles as _obs
        if os.environ.get("OBSTACLES"):
            keep = [(int(q["col"]), int(q["row"])) for q in self.experiment.get("parcels", [])]
            keep += [(int(a["start_col"]), int(a["start_row"])) for a in self.experiment.get("agents", []) if "start_col" in a]
            _obs.set_obstacles(_obs.make_layout(self.seed, (cols, rows), os.environ["OBSTACLES"], keep, s), (cols, rows))
        else:
            _obs.set_obstacles([], (cols, rows))

        # 3. Agents

        agents_dict, steps, error = self._setup_agents(
            terrain,
            strategy_name=strategy_name,
            strategy_spec=strategy_spec
        )

        if error:
            self._show_error_and_restart_setup(error)
            raise RuntimeError("Restart setup")

        # Collect all parcels (both newly created and existing)
        all_parcels = list(terrain.parcels)

        return terrain, list(agents_dict.values()), steps, all_parcels

    @staticmethod
    def agent_at_base(drone, terrain) -> bool:
        return terrain.get_station_at(
            int(drone.col),
            int(drone.row),
        ) is not None

    @staticmethod
    def agent_dead(drone) -> bool:
        return (
                hasattr(drone, "power")
                and drone.power.level <= 0.0
        )

    def mission_exhausted(self, terrain, agents) -> bool:
        # 1. Check all parcels - if any are not delivered, mission is not exhausted
        for p in terrain.parcels:
            if not getattr(p, "delivered", False):
                return False

        # 2. No agent is carrying a parcel
        for a in agents:
            if getattr(a, "carrying", None) is not None:
                return False

        # 3. No agent has a pending task
        for agent in agents:
            controller = getattr(agent, "controller", None)
            if controller and getattr(controller, "has_pending_task", False):
                return False

        # 4. ALL ALIVE AGENTS MUST BE AT THE STATION (PARKED)
        for agent in agents:
            if not self.agent_dead(agent):
                # Check if agent is at the station
                if not self.agent_at_base(agent, terrain):
                    return False
                # Also check if the controller recognizes it's parked
                controller = getattr(agent, "controller", None)
                if controller and hasattr(controller, "parked"):
                    if not controller.parked:
                        return False

        return True

    def _finalize_strategy(
            self,
            *,
            idx: int,
            status: str,
            planner_name: str,
            strategy_name: str,
            strategy_type: str,
            steps: int,
            flight_recorder: FlightRecorder,
    ):
        assert status in {"completed", "aborted", "failed", "infeasible", "timeout", "stalled"}, f"Invalid status: {status}"

        # Resolve authoritative strategy spec from experiment
        strategy = self.experiment["strategies"][idx]

        # Pull metrics from composition instead of param
        metrics = flight_recorder.metrics
        if metrics is None:
            raise RuntimeError("FlightRecorder.metrics is None. MetricsAccumulator was never attached.")

        print(
            f"[FINALIZE] Strategy {idx} '{strategy_name}' | "
            f"mode={strategy_type} | planner={planner_name} | "
            f"status={status} | steps={steps}"
        )

        # -------------------------------------------------
        # Export flight artifacts first
        # -------------------------------------------------
        flight_data = flight_recorder.export()

        # -----------------------------------
        # Capture personality composition
        # -----------------------------------
        active_drones = getattr(self, "_active_drones", [])

        personality_map = {
            d.agent_id: getattr(d, "personality", None) or "variant_default"
            for d in active_drones
        }
        personality_distribution = dict(Counter(personality_map.values()))

        artifact_paths = self._write_flight_artifacts(
            run_id=self.experiment["run_id"],
            planner=planner_name,
            strategy_name=strategy_name,
            flight_data=flight_data,
        )

        print(
            f"[FINALIZE] Artifacts written | "
            f"keys={list(artifact_paths.keys())}"
        )

        # -------------------------------------------------
        # Derive authoritative metrics from FlightRecorder
        # -------------------------------------------------
        agents = flight_data.get("agents", [])
        fleet_events = flight_data.get("fleet_events", [])

        total_distance = sum(a.get("distance_cells", 0.0) for a in agents)

        total_energy_consumed = 0.0
        for a in agents:
            energy_series = a.get("energy", [])
            if len(energy_series) >= 2:
                start = energy_series[0]["battery_pct"]
                end = energy_series[-1]["battery_pct"]
                total_energy_consumed += max(0.0, start - end)

        tasks_completed = sum(len(a.get("tasks", [])) for a in agents)
        unsafe_failures = len([e for e in fleet_events if e.get("type") == "collision"])

        runtime_seconds = round(time.time() - metrics.start_time, 3)

        derived_metrics = {
            "run_id": self.experiment["run_id"],
            "planner": planner_name,
            "seed": self.experiment["seed"],
            "n_drones": len(agents),
            "n_parcels": len(self.experiment.get("parcels", [])),
            "comm_reliability": self.experiment.get("comm_config", {}).get("reliability", 1.0),
            "battery_capacity": metrics.fields.get("battery_capacity", 100.0),
            "episodic_interval": 1.0,
            "total_energy_consumed": round(total_energy_consumed, 3),
            "total_distance_traveled": round(total_distance, 3),
            "tasks_completed": tasks_completed,
            "mission_makespan_steps": steps,
            "unsafe_failures": unsafe_failures,
            "messages_sent": metrics.messages_sent,
            "runtime_seconds": runtime_seconds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(
            f"[FINALIZE] Derived metrics | "
            f"distance={derived_metrics['total_distance_traveled']:.2f} | "
            f"energy={derived_metrics['total_energy_consumed']:.3f} | "
            f"tasks={tasks_completed}"
        )
        # -------------------------------
        # Termination diagnostics
        # -------------------------------
        final_agents = flight_data.get("agents", [])

        dead_total = sum(1 for a in final_agents if a.get("energy", []) and a["energy"][-1]["battery_pct"] <= 0)
        dead_at_station = sum(
            1 for a in final_agents
            if a.get("energy", []) and a["energy"][-1]["battery_pct"] <= 0 and a.get("at_station", False)
        )
        lost_in_field = dead_total - dead_at_station

        termination_summary = {
            "dead_total": dead_total,
            "dead_at_station": dead_at_station,
            "lost_in_field": lost_in_field,
        }

        # -------------------------------------------------
        # Append result (rich identity preserved)
        # -------------------------------------------------
        self.experiment["results"].append({
            "strategy_index": idx,

            # Identity
            "strategy_name": strategy_name,  # e.g. sppdec_combined_instance_3
            "family": strategy.get("family"),  # e.g. sppdec_combined
            "base_name": strategy.get("name"),  # original base name
            "variant": strategy.get("variant"),  # combined, control, etc
            "instance_id": strategy.get("instance_id"),  # 0..N
            "description": strategy.get("description"),
            "personality_map": personality_map,
            "personality_distribution": personality_distribution,
            # Metadata
            "strategy_type": strategy_type,
            "planner": planner_name,
            "seed": self.experiment["seed"],
            "status": status,
            "steps_executed": steps,
            "termination": termination_summary,
            # Outputs
            "metrics": derived_metrics,
            "artifacts": artifact_paths,
        })

        print(
            f"[FINALIZE] Result appended | "
            f"results_count={len(self.experiment['results'])}"
        )

        # -------------------------------------------------
        # Update strategy bookkeeping
        # -------------------------------------------------
        self.experiment["strategies"][idx]["status"] = status
        self.experiment["strategies"][idx]["snapshot"] = None
        self.experiment["strategies"][idx]["personality_map"] = personality_map

        print("Finalized Strategy:", strategy)
        print(
            f"[FINALIZE] Strategy state updated | "
            f"strategy_status={status}"
        )

        # -------------------------------------------------
        # Advance experiment progress
        # -------------------------------------------------
        self.experiment["current_index"] = idx + 1

        if self.experiment["current_index"] >= len(self.experiment["strategies"]):
            self.experiment["status"] = "completed"
            self._experiment_completed = True
            print("[EXPERIMENT] All strategies completed")
        else:
            self.experiment["status"] = "in_progress"
            print(
                f"[EXPERIMENT] Moving to next strategy: "
                f"{self.experiment['current_index']}"
            )

        # -------------------------------------------------
        # Shutdown SLMs
        # -------------------------------------------------
        active_drones = getattr(self, "_active_drones", [])
        slm_users = [d for d in active_drones if getattr(d, "slm_enabled", False)]

        if slm_users:
            print("[SLM] Shutting down all drone SLM instances")

        for drone in slm_users:
            try:
                drone.stop_slm()
                print(f"[SLM] Stopped {drone.agent_id}")
            except Exception as e:
                print(f"[SLM] Failed to stop {drone.agent_id}: {e}")

        # -------------------------------------------------
        # Persist experiment state
        # -------------------------------------------------
        print("[FINALIZE] Persisting experiment state to disk")
        self._write_experiment_json()

    def _show_experiment_complete_screen(self, terrain, agents, config_name=None, duration: float = 3.0):
        """
        Final blocking screen shown once ALL strategies are completed.
        Shows current config name and next config info in sweep mode.
        """
        if self.fast_mode:
            print(f"[FAST] Config complete: {config_name or 'experiment'}")
            return

        start_time = time.time()

        # Determine if we're in sweep mode and get next config info
        in_sweep_mode = USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False)
        next_config_info = None

        if in_sweep_mode and self._sweep_queue:
            next_config = self._sweep_queue[0]  # Peek at next config without removing it
            next_agents = next_config.get("agents", "?")
            next_parcels = next_config.get("parcels", "?")
            next_config_info = f"A{next_agents}_P{next_parcels}"

        while time.time() - start_time < duration:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return

            self.screen.fill((192, 192, 192))

            # grid
            for x in range(0, self.screen_size[0], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
            for y in range(0, self.screen_size[1], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

            terrain.draw(self.screen)
            for drone in agents:
                drone.draw(self.screen, self.images)

            # Main completion message with config name
            if config_name:
                msg = self.large_font.render(f"EXPERIMENT {config_name} COMPLETE", True, (0, 180, 0))
            else:
                msg = self.large_font.render("EXPERIMENT COMPLETE", True, (0, 180, 0))
            self.screen.blit(msg, msg.get_rect(center=(self.screen_size[0] // 2, 120)))

            # Next action message with specific config
            if next_config_info:
                next_msg = self.font.render(
                    f"Starting next: {next_config_info} ({len(self._sweep_queue)} remaining)",
                    True,
                    (100, 100, 100)
                )
                self.screen.blit(next_msg, next_msg.get_rect(center=(self.screen_size[0] // 2, 170)))
            elif in_sweep_mode:
                final_msg = self.font.render("All sweep experiments complete", True, (60, 60, 60))
                self.screen.blit(final_msg, final_msg.get_rect(center=(self.screen_size[0] // 2, 170)))

            pygame.display.flip()
            self.clock.tick(30)

    def _show_post_run_inspection(
            self,
            *,
            terrain,
            agents,
            strategy_name: str,
            status: str,
            variant: Optional[str] = None,
            config: Optional[dict] = None,
            seconds: int = 5,
    ):
        """
        Show final strategy state briefly, then auto-advance to next strategy.
        """

        if self.fast_mode:
            print(f"[FAST] Finished: {strategy_name} → {status.upper()}")
            return

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            remaining = seconds - int(elapsed)

            if remaining <= 0:
                return  # auto-continue to next strategy

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return  # allow manual skip

            self.screen.fill((192, 192, 192))

            # grid
            for x in range(0, self.screen_size[0], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
            for y in range(0, self.screen_size[1], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

            terrain.draw(self.screen)
            for drone in agents:
                drone.draw(self.screen, self.images)

            title_text = strategy_name
            if variant:
                title_text = f"{strategy_name} ({variant})"

            title = self.large_font.render(
                f"Strategy Finished: {title_text}",
                True,
                (50, 50, 50),
            )

            status_text = self.font.render(
                f"Status: {status.upper()}",
                True,
                (80, 80, 80),
            )

            countdown_text = self.font.render(
                f"Next strategy in {remaining}s",
                True,
                (100, 100, 100),
            )

            self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, 80)))
            self.screen.blit(status_text, status_text.get_rect(center=(self.screen_size[0] // 2, 130)))
            self.screen.blit(countdown_text, countdown_text.get_rect(center=(self.screen_size[0] // 2, 160)))

            if config:
                mid_font = pygame.font.SysFont("Consolas", 32)
                y = 230
                for k, v in config.items():
                    txt = mid_font.render(f"{k}: {v}", True, (60, 60, 60))
                    self.screen.blit(txt, txt.get_rect(center=(self.screen_size[0] // 2, y)))
                    y += 50

            pygame.display.flip()
            self.clock.tick(30)

    def _show_strategy_countdown(
            self,
            *,
            terrain,
            agents,
            strategy_name: str,
            topology: str,
            seconds: int = 5,
            variant: Optional[str] = None,
            config: Optional[dict] = None
    ) -> bool:
        """
        Overlay countdown shown before a strategy starts.

        This screen is rendered directly on top of the active terrain
        to preserve spatial and experimental continuity.
        """

        if self.fast_mode:
            print(f"[FAST] Starting: {strategy_name}")
            return True

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            remaining = seconds - int(elapsed)

            if remaining <= 0:
                return True

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return False

            # -------------------------------------------------
            # Draw world
            # -------------------------------------------------
            self.screen.fill((192, 192, 192))

            for x in range(0, self.screen_size[0], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
            for y in range(0, self.screen_size[1], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

            terrain.draw(self.screen)
            for drone in agents:
                drone.draw(self.screen, self.images)

            # -------------------------------------------------
            # Overlay info
            # -------------------------------------------------
            overlay_y = 110

            title = self.large_font.render("NEXT STRATEGY", True, (80, 120, 200))
            self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, overlay_y)))

            overlay_y += 70

            name_text = strategy_name
            if variant:
                name_text = f"{strategy_name} ({variant})"

            name = self.title_font.render(name_text.upper(), True, (20, 20, 20))
            self.screen.blit(name, name.get_rect(center=(self.screen_size[0] // 2, overlay_y)))

            overlay_y += 50

            meta = [f"Topology: {topology}"]

            if config:
                for k, v in config.items():
                    meta.append(f"{k}: {v}")

            for line in meta:
                txt = self.font.render(line, True, (40, 40, 40))
                self.screen.blit(txt, txt.get_rect(center=(self.screen_size[0] // 2, overlay_y)))
                overlay_y += 34

            overlay_y += 30

            countdown = self.large_font.render(str(remaining), True, (200, 100, 40))
            self.screen.blit(countdown, countdown.get_rect(center=(self.screen_size[0] // 2, overlay_y)))

            hint = self.font.render("Press ESC to skip this strategy", True, (90, 90, 90))
            self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 100)))

            pygame.display.flip()
            self.clock.tick(30)

    def scan_completed_experiments(self, sweep_name: str) -> Set[Tuple[int, int, int]]:
        """Scan filesystem for sweep configs that should be skipped."""
        completed = set()

        if not EXPERIMENTS_DIR.exists():
            return completed

        sweep_dir = EXPERIMENTS_DIR / sweep_name
        if not sweep_dir.exists():
            return completed

        TERMINAL = {"completed", "aborted", "failed", "infeasible", "timeout", "stalled"}

        for scale_dir in sweep_dir.iterdir():
            if not scale_dir.is_dir() or not scale_dir.name.startswith("A"):
                continue

            try:
                parts = scale_dir.name.split("_")
                agents = int(parts[0][1:])  # A{agents}
                parcels = int(parts[1][1:])  # P{parcels}
            except (ValueError, IndexError):
                continue

            for exp_dir in scale_dir.iterdir():
                if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
                    continue

                exp_json = exp_dir / "experiment.json"
                if not exp_json.exists():
                    continue

                try:
                    exp_data = json.loads(exp_json.read_text(encoding="utf-8"))
                    strategies = exp_data.get("strategies", [])

                    # ---------------------------------------------
                    # Experiment is completed ONLY if all strategies
                    # are in terminal states
                    # ---------------------------------------------
                    if strategies and all(
                            s.get("status") in TERMINAL for s in strategies
                    ):
                        instance_id = strategies[0].get("instance_id", 0)
                        completed.add((agents, parcels, instance_id))

                except (json.JSONDecodeError, IOError):
                    continue

        print(f"[SWEEP] Found {len(completed)} completed experiment configs")
        return completed

    def prune_completed_sweeps(self, sweep_configs: List[Dict], completed: Set[Tuple[int, int, int]]) -> List[Dict]:
        """Remove completed configurations from sweep queue."""
        pruned = []

        for config in sweep_configs:
            agents = config.get("agents", 0)
            parcels = config.get("parcels", 0)
            instance_id = config.get("instance_id", 0)

            if (agents, parcels, instance_id) not in completed:
                pruned.append(config)
            else:
                print(f"[SWEEP] Skipping completed: A{agents}_P{parcels}_I{instance_id}")

        print(f"[SWEEP] Pruned {len(sweep_configs) - len(pruned)} completed configs")
        print(f"[SWEEP] Remaining: {len(pruned)} configs to run")
        return pruned

    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """
        Wrap text to fit within max_width pixels.

        Returns a list of lines that fit within the specified width.
        """
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            # Test if adding this word would exceed max_width
            test_line = current_line + (" " if current_line else "") + word
            test_width = font.size(test_line)[0]  # FIXED: use font.size() instead of font.get_rect()

            if test_width <= max_width:
                current_line = test_line
            else:
                # Current line is full, start a new line
                if current_line:
                    lines.append(current_line)
                current_line = word

                # Handle case where single word is too long
                while font.size(current_line)[0] > max_width and len(current_line) > 1:  # FIXED: use font.size()
                    # Split the word
                    split_point = len(current_line) // 2
                    lines.append(current_line[:split_point] + "-")
                    current_line = current_line[split_point:]

        # Add the last line
        if current_line:
            lines.append(current_line)

        return lines

    def _show_experiment_status_modal(
            self,
            *,
            sweep_name: str,
            agents: int,
            parcels: int,
            run_id: str,
            current_index: int,
            strategies: list,
            max_visible_strategies: int = None,
            experiment_dir: Optional[Path] = None,
            review: bool = False,
    ):
        """
        Strategy list of one experiment. ENTER runs a pending strategy, or replays a finished one
        (from its exported telemetry). SPACE marks finished strategies to replay side by side:
        they all ran on the same parcels and start cells. In review mode nothing can be run.
        """
        TERMINAL = {"completed", "failed", "infeasible", "timeout", "stalled"}
        dialog_message = None
        selected_index = current_index if not review else 0
        marked: Set[int] = set()
        icons = {"completed": "✓", "pending": "○", "infeasible": "◇", "failed": "✗", "aborted": "◇",
                 "timeout": "◷", "stalled": "◷"}

        def replayable(i):
            s_ = strategies[i]
            return (experiment_dir is not None and s_.get("status") in TERMINAL
                    and has_telemetry(experiment_dir, s_))

        while True:
            lines = []
            for i, s_ in enumerate(strategies):
                status = s_.get("status", "pending")
                tag = "[x]" if i in marked else ("[ ]" if replayable(i) else "   ")
                marker = "→" if i == selected_index else " "
                lines.append(f"{marker} {tag} {icons.get(status, '?')} [{status.upper():9}] {s_.get('name')}")

            hint = ("↑↓ Select | ENTER: Replay | SPACE: Mark to compare side by side | ESC: Back"
                    if review else
                    "↑↓ Select | ENTER: Run / Replay | SPACE: Mark to compare | ESC: Discard Experiment")
            key, selected_index = self._show_scroll_modal(
                title=f"{'Review' if review else 'Experiment Status'} | {agents}A {parcels}P | Run {run_id}",
                lines=lines,
                background=(28, 32, 40),
                title_color=(120, 200, 255),
                hint=hint,
                exit_keys={pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_SPACE},
                max_visible_lines=max_visible_strategies,
                dialog_message=dialog_message,
                initial_index=selected_index,
            )

            if dialog_message:
                dialog_message = None
                continue

            if key == pygame.K_ESCAPE:
                return None

            if key == pygame.K_SPACE:
                if replayable(selected_index):
                    marked.symmetric_difference_update({selected_index})
                else:
                    dialog_message = "Only finished strategies with exported telemetry can be marked."
                continue

            if key == pygame.K_RETURN:
                status = strategies[selected_index].get("status")

                if status in TERMINAL:
                    if not replayable(selected_index):
                        dialog_message = "No telemetry was exported for this strategy,\nso it cannot be replayed."
                        continue
                    chosen = sorted(marked) if selected_index in marked and marked else [selected_index]
                    ReplayViewer(self.screen, self.clock, self.font, self.large_font, self.images).show(
                        experiment_dir,
                        [strategies[i] for i in chosen],
                        title=f"Replay | {agents}A {parcels}P | Run {run_id}"
                              + (f" | {len(chosen)} strategies side by side" if len(chosen) > 1 else ""),
                    )
                    continue

                if review:
                    dialog_message = "This strategy has not been run yet.\n\nThere is nothing to replay."
                    continue

                return selected_index

    def _review_config(self, sweep_name: str, agents: int, parcels: int) -> None:
        """Replay entry for a configuration that already has runs: pick a run, then its strategies."""
        scale_dir = EXPERIMENTS_DIR / sweep_name / f"A{agents}_P{parcels}"
        runs = []
        for f in sorted(scale_dir.glob("experiment_*/experiment.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                runs.append((f.parent, json.loads(f.read_text(encoding="utf-8"))))
            except Exception:
                continue
        runs.sort(key=lambda r: ((r[1].get("provenance") or {}).get("replicate", 0), r[0].name))
        if not runs:
            self._show_modal(title="Nothing to replay", message=f"No runs found for A{agents} P{parcels}.",
                             background=(28, 32, 40), title_color=(220, 160, 120), hint="ESC = back",
                             exit_keys={pygame.K_ESCAPE, pygame.K_RETURN})
            return
        index = 0
        while True:
            if len(runs) > 1:
                lines = []
                for k, (d, data) in enumerate(runs):
                    n_done = sum(1 for x in data.get("strategies", []) if x.get("status") == "completed")
                    rep = (data.get("provenance") or {}).get("replicate")
                    lines.append(f"{'→' if k == index else ' '} replicate {rep if rep is not None else k} | "
                                 f"{d.name} | {n_done}/{len(data.get('strategies', []))} completed")
                key, index = self._show_scroll_modal(
                    title=f"Replay | A{agents} P{parcels} | choose a run ({len(runs)})",
                    lines=lines, background=(28, 32, 40), title_color=(120, 200, 255),
                    hint="↑↓ Select | ENTER: Open | ESC: Back", exit_keys={pygame.K_RETURN, pygame.K_ESCAPE},
                    initial_index=index)
                if key == pygame.K_ESCAPE:
                    return
            exp_dir, data = runs[index]
            self._show_experiment_status_modal(
                sweep_name=sweep_name, agents=agents, parcels=parcels, run_id=data.get("run_id", "unknown"),
                current_index=0, strategies=data.get("strategies", []), experiment_dir=exp_dir, review=True)
            if len(runs) == 1:
                return

    def _show_scroll_modal(
            self,
            *,
            title: str,
            lines: list[str],
            background=(30, 30, 30),
            title_color=(120, 200, 255),
            hint: str,
            exit_keys: set[int],
            max_visible_lines: int = None,
            dialog_message: Optional[str] = None,
            initial_index: int = 0,
    ):
        exit_keys = set(exit_keys)

        font_height = self.font.get_height()
        LINE_HEIGHT = font_height + 4

        TITLE_HEIGHT = 80
        HINT_HEIGHT = 60
        STATUS_HEIGHT = 30
        NAV_HEIGHT = 25
        PADDING = 40

        available_height = (
                self.screen_size[1]
                - TITLE_HEIGHT
                - HINT_HEIGHT
                - STATUS_HEIGHT
                - NAV_HEIGHT
                - PADDING
        )
        calculated_max_lines = max(1, available_height // LINE_HEIGHT)

        if max_visible_lines is None:
            MAX_VISIBLE_LINES = calculated_max_lines
        else:
            MAX_VISIBLE_LINES = min(max_visible_lines, calculated_max_lines)

        PANEL_WIDTH = int(self.screen_size[0] * 0.35)
        CONTENT_HEIGHT = MAX_VISIBLE_LINES * LINE_HEIGHT
        PANEL_HEIGHT = CONTENT_HEIGHT + 60

        available_vertical_space = (
                self.screen_size[1]
                - TITLE_HEIGHT
                - HINT_HEIGHT
                - STATUS_HEIGHT
                - NAV_HEIGHT
        )
        PANEL_Y = TITLE_HEIGHT + (available_vertical_space - PANEL_HEIGHT) // 2
        PANEL_X = (self.screen_size[0] - PANEL_WIDTH) // 2

        INNER_PADDING = 20
        CONTENT_X = PANEL_X + INNER_PADDING
        CONTENT_Y = PANEL_Y + 30
        CONTENT_WIDTH = PANEL_WIDTH - INNER_PADDING * 2

        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(self._wrap_text(line, self.font, CONTENT_WIDTH))

        total_lines = len(wrapped_lines)
        scroll = 0
        max_scroll = max(0, total_lines - MAX_VISIBLE_LINES)

        max_selection = max(0, total_lines - 1)
        selected_index = max(0, min(int(initial_index), max_selection))
        if selected_index >= MAX_VISIBLE_LINES:
            scroll = min(selected_index - MAX_VISIBLE_LINES + 1, max_scroll)

        viewport_rect = pygame.Rect(CONTENT_X, CONTENT_Y, CONTENT_WIDTH, CONTENT_HEIGHT)
        panel_rect = pygame.Rect(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT)

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()

                if e.type == pygame.KEYDOWN:
                    # Dialog has priority
                    if dialog_message and e.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        return e.key, selected_index

                    if e.key in exit_keys:
                        return e.key, selected_index

                    if dialog_message:
                        continue

                    if e.key == pygame.K_DOWN:
                        selected_index = min(selected_index + 1, max_selection)
                        if selected_index >= scroll + MAX_VISIBLE_LINES:
                            scroll = min(selected_index - MAX_VISIBLE_LINES + 1, max_scroll)

                    elif e.key == pygame.K_UP:
                        selected_index = max(selected_index - 1, 0)
                        if selected_index < scroll:
                            scroll = selected_index

            self.screen.fill(background)

            title_surf = self.large_font.render(title, True, title_color)
            self.screen.blit(title_surf, title_surf.get_rect(center=(self.screen_size[0] // 2, 40)))

            pygame.draw.rect(self.screen, (22, 26, 34), panel_rect, border_radius=12)
            pygame.draw.rect(self.screen, (90, 100, 120), panel_rect, 2, border_radius=12)

            self.screen.set_clip(viewport_rect)

            y = CONTENT_Y
            for i, line in enumerate(wrapped_lines[scroll:scroll + MAX_VISIBLE_LINES]):
                idx = scroll + i
                highlight = idx == selected_index

                if highlight:
                    pygame.draw.rect(
                        self.screen,
                        (60, 80, 120),
                        (CONTENT_X - 5, y - 2, CONTENT_WIDTH + 10, LINE_HEIGHT),
                        border_radius=4,
                    )

                txt = self.font.render(
                    ("→ " if highlight else "  ") + line,
                    True,
                    (255, 255, 255) if highlight else (230, 230, 230),
                )
                self.screen.blit(txt, (CONTENT_X, y))
                y += LINE_HEIGHT

            self.screen.set_clip(None)

            # -----------------------------
            # INLINE DIALOG OVERLAY
            # -----------------------------
            if dialog_message:
                overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                self.screen.blit(overlay, (0, 0))

                popup_w = int(self.screen_size[0] * 0.4)
                popup_h = 160
                popup_x = (self.screen_size[0] - popup_w) // 2
                popup_y = (self.screen_size[1] - popup_h) // 2

                rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
                pygame.draw.rect(self.screen, (32, 36, 46), rect, border_radius=10)
                pygame.draw.rect(self.screen, (200, 120, 80), rect, 2, border_radius=10)

                y = popup_y + 30
                for line in dialog_message.split("\n"):
                    txt = self.font.render(line, True, (230, 230, 230))
                    self.screen.blit(
                        txt,
                        txt.get_rect(center=(self.screen_size[0] // 2, y)),
                    )
                    y += 28

                hint_txt = self.font.render("ESC or ENTER to close", True, (160, 160, 160))
                self.screen.blit(
                    hint_txt,
                    hint_txt.get_rect(center=(self.screen_size[0] // 2, popup_y + popup_h - 28)),
                )

            hint_surf = self.font.render(hint, True, (180, 180, 180))
            self.screen.blit(
                hint_surf,
                hint_surf.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 30)),
            )

            pygame.display.flip()
            self.clock.tick(30)

    def show_sweep_selection_modal(self, configs, max_visible_configs: int = 20):
        dialog_message = None
        selected_index = 0
        sweep_name = (SCALING_CONFIG or {}).get("name", "sweep")

        while True:
            lines = []
            for i, cfg in enumerate(configs):
                agents = cfg["agents"]
                parcels = cfg["parcels"]
                state = cfg["state"]
                status_icon = {"completed": "✓", "partial": "◯", "new": "○"}.get(state, "?")
                lines.append(
                    f"[{i + 1:2d}] {status_icon} A{agents:2d} P{parcels:2d} [{state.upper():9}]"
                    + ("  (replay)" if state in ("completed", "partial") else "")
                )

            key, selected_index = self._show_scroll_modal(
                title=f"Sweep Configuration Selection ({len(configs)} total) | {sweep_name}",
                lines=lines,
                background=(30, 34, 42),
                title_color=(220, 220, 220),
                hint="↑↓ Select | ENTER: Run (completed: Replay) | R: Replay | ESC: Cancel",
                exit_keys={pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_r},
                max_visible_lines=max_visible_configs,
                dialog_message=dialog_message,
                initial_index=selected_index,
            )

            if dialog_message:
                dialog_message = None
                continue

            if key == pygame.K_ESCAPE:
                return None

            selected = configs[selected_index]

            if key == pygame.K_r or (key == pygame.K_RETURN and selected["state"] == "completed"):
                if selected["state"] == "new":
                    dialog_message = (f"A{selected['agents']} P{selected['parcels']} has no runs yet.\n\n"
                                      "There is nothing to replay.")
                    continue
                self._review_config(sweep_name, selected["agents"], selected["parcels"])
                continue

            if key == pygame.K_RETURN:
                return selected

    def _show_experiment_status_screen(self, strategy_name, steps, agents, terrain):
        """Show experiment status during fast execution mode"""
        self.screen.fill((40, 44, 52))

        # Title with current mode
        title = self.large_font.render("EXPERIMENT RUNNING", True, (100, 255, 100))
        self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, 120)))

        # Current mode indicator (more prominent)
        mode_text = "FAST MODE" if self.fast_mode else "NORMAL MODE"
        mode_color = (255, 200, 100) if self.fast_mode else (100, 255, 100)
        mode_indicator = self.large_font.render(mode_text, True, mode_color)
        self.screen.blit(mode_indicator, mode_indicator.get_rect(center=(self.screen_size[0] // 2, 170)))

        # Strategy info
        strategy_text = self.font.render(f"Strategy: {strategy_name}", True, (230, 230, 230))
        self.screen.blit(strategy_text, strategy_text.get_rect(center=(self.screen_size[0] // 2, 220)))

        # Step counter
        step_text = self.font.render(f"Step: {steps:,}", True, (230, 230, 230))
        self.screen.blit(step_text, step_text.get_rect(center=(self.screen_size[0] // 2, 260)))

        # Agent status
        alive_count = len([a for a in agents if not self.agent_dead(a)])
        total_count = len(agents)
        agent_text = self.font.render(f"Agents: {alive_count}/{total_count} alive", True, (230, 230, 230))
        self.screen.blit(agent_text, agent_text.get_rect(center=(self.screen_size[0] // 2, 300)))

        # Parcel status
        remaining_parcels = sum(1 for p in terrain.parcels if not p.delivered)
        total_parcels = len(terrain.parcels)
        parcel_text = self.font.render(f"Parcels: {total_parcels - remaining_parcels}/{total_parcels} delivered", True,
                                       (230, 230, 230))
        self.screen.blit(parcel_text, parcel_text.get_rect(center=(self.screen_size[0] // 2, 340)))

        # Toggle instructions
        hint = self.font.render("Press H to toggle between NORMAL/FAST rendering", True, (180, 180, 180))
        self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 100)))

    def _create_experiment(self, n_agents: int, n_parcels: int, selected_planner: str,
                           strategy_vault: List[Dict], num_instances: int,
                           experiment_dir: Optional[Path] = None, instance_offset: int = 0):
        """Build a new experiment schema (parcels, agents, station, strategy instances) and write it."""
        strategies_expanded = []

        for i in range(num_instances):
            for strategy in strategy_vault:
                instance = strategy.copy()

                variant = strategy.get("variant", "base")
                family = f"{strategy['name']}_{variant}"

                instance["name"] = f"{family}_instance_{instance_offset + i + 1}"
                instance["instance_id"] = instance_offset + i
                instance["variant"] = variant
                instance["family"] = family

                strategies_expanded.append(instance)

        strategies = strategies_expanded

        # Now use strategies_expanded instead of strategies
        self.experiment = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "scale": {
                "n_agents": n_agents,
                "n_parcels": n_parcels,
            },
            "parcels": [],
            "agents": [],
            "comm_config": self.default_comm_config.copy(),
            "current_index": 0,
            "status": "in_progress",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "results": [],
            "strategies": strategies,
        }

        # Generate parcel positions
        cols = self.screen_size[0] // GRID_SIZE
        rows = self.screen_size[1] // GRID_SIZE
        center_col = max(2, cols // 2 - 2)
        center_row = max(2, rows // 2 - 2)

        station_spec = {
            "col": center_col,
            "row": center_row,
            "w": 6,
            "h": 6,
        }
        self.experiment["station"] = station_spec

        s = self.experiment["station"]

        # Forbidden cells inside the station footprint
        forbidden_cells = []
        for r in range(s["row"], s["row"] + s["h"]):
            for c in range(s["col"], s["col"] + s["w"]):
                forbidden_cells.append((c, r))

        parcel_positions = deterministic_parcel_positions(
            self.seed, n_parcels, cols, rows, forbidden_cells
        )

        # Parcel weights are drawn once per experiment from the seed, so every
        # strategy and instance in this config faces identical payloads.
        weight_rng = random.Random(self.seed + 1)

        # Add parcels to experiment
        for i, (col, row) in enumerate(parcel_positions):
            # Generate a string ID for each parcel
            parcel_id = f"parcel_{i:03d}"  # e.g., "parcel_000", "parcel_001"
            # OR use UUID: parcel_id = str(uuid.uuid4())[:8]

            # Update the parcel creation to be more explicit:
            parcel_data = {
                "id": str(parcel_id),  # Explicitly cast to string
                "col": int(col),
                "row": int(row),
                "picked": bool(False),
                "delivered": bool(False),
                "weight": weight_rng.uniform(0.5, 2.0),
            }
            self.experiment["parcels"].append(parcel_data)
        # rebuild agents deterministically
        start_col = (self.screen_size[0] // GRID_SIZE) // 2
        start_row = (self.screen_size[1] // GRID_SIZE) // 2

        personality_cycle = list(SPPDEC_PERSONALITIES.keys())
        self.experiment["agents"] = [
            {
                "id": f"agent_{i}",
                "type": "drone",
                "planner": selected_planner,
                "start_col": start_col + i * 2,
                "start_row": start_row,
                "personality": personality_cycle[i % len(personality_cycle)],
            }
            for i in range(n_agents)
        ]

        # Create experiment directory
        if experiment_dir is not None:
            self.experiment_dir = experiment_dir / f"experiment_{self.run_id}"
        elif USE_SWEEP and SCALING_CONFIG:
            sweep_name = SCALING_CONFIG.get("name", "sweep")
            scale_dir = f"A{n_agents}_P{n_parcels}"
            self.experiment_dir = (
                    EXPERIMENTS_DIR
                    / sweep_name
                    / scale_dir
                    / f"experiment_{self.run_id}"
            )
        else:
            self.experiment_dir = EXPERIMENTS_DIR / f"experiment_{self.run_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_json_path = self.experiment_dir / "experiment.json"

        self._write_experiment_json()

    def run(self):
        # -------------------------------------------------
        # Splash + setup phase (single abort boundary)
        # -------------------------------------------------
        if not getattr(self, '_splash_shown', False) or not (USE_SWEEP and SCALING_CONFIG):
            self.show_splash(timeout=1.0)
            self._splash_shown = True

        # RESET RESUME FLAG FOR SWEEP PROCESSING (but preserve selection state)
        if USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False):
            self._resumed = False  # ← Only reset if sweep already initialized

        # Show sweep selection modal (GUI)
        if USE_SWEEP and SCALING_CONFIG and not getattr(self, '_sweep_initialized', False):
            # Generate sweeps if missing BEFORE building status
            if not SCALING_CONFIG.get("sweeps"):
                station = self.experiment.get("station")
                if station:
                    w, h = station["w"], station["h"]
                else:
                    w, h = 6, 6

                mode = SCALING_CONFIG.get("mode", "research")
                budget = SCALING_CONFIG.get("budget", 40)

                SCALING_CONFIG["sweeps"] = SCALING_CONFIG.get("sweeps") or generate_station_scaling(w, h, mode=mode, budget=budget)
                print(f"[SWEEP] Generated {len(SCALING_CONFIG['sweeps'])} configs")

            sweep_name = SCALING_CONFIG.get("name", "sweep")
            status_rows = self.build_sweep_status(sweep_name=sweep_name, sweep_configs=SCALING_CONFIG["sweeps"])

            selected = self.show_sweep_selection_modal(status_rows)
            if selected is None:
                print("[SWEEP] User aborted sweep selection")
                return False

            self._sweep_initialized = True
            print("[SWEEP] Selection made, proceeding with automated batch execution")

        if not getattr(self, "_resumed", False):
            self._experiment_completed = False

        # -------------------------------------------------
        # Initialize setup variables defensively
        # -------------------------------------------------
        n_agents = None
        n_parcels = None
        selected_planner = None
        # -------------------------------------------------
        # Fresh setup path (only if not resumed)
        # -------------------------------------------------
        if not getattr(self, "_resumed", False):

            # -------- SWEEP MODE --------
            if USE_SWEEP and SCALING_CONFIG:
                # Generate sweeps once if missing (existing logic)
                if not SCALING_CONFIG.get("sweeps"):
                    station = self.experiment.get("station")
                    if station:
                        w, h = station["w"], station["h"]
                    else:
                        w, h = 6, 6

                    mode = SCALING_CONFIG.get("mode", "research")
                    budget = SCALING_CONFIG.get("budget", 40)

                    SCALING_CONFIG["sweeps"] = generate_station_scaling(w, h, mode=mode, budget=budget)
                    print(f"[SWEEP] Generated {len(SCALING_CONFIG['sweeps'])} configs")

                # Initialize queue with persistence awareness
                if self._sweep_queue is None:
                    sweep_name = SCALING_CONFIG.get("name", "sweep")
                    completed = self.scan_completed_experiments(sweep_name)

                    # Prune completed configs from sweep
                    full_sweep = list(SCALING_CONFIG["sweeps"])
                    self._sweep_queue = self.prune_completed_sweeps(full_sweep, completed)

                # If empty after pruning, all done
                if not self._sweep_queue:
                    print("[SWEEP] All configurations completed.")
                    return

                # Show sweep selection modal only once at startup (not resumed)
                if not getattr(self, '_sweep_initialized', False):
                    sweep_name = SCALING_CONFIG.get("name", "sweep")
                    status_rows = self.build_sweep_status(
                        sweep_name=sweep_name,
                        sweep_configs=SCALING_CONFIG["sweeps"],
                    )

                    selected = self.show_sweep_selection_modal(status_rows)
                    if selected is None:
                        print("[SWEEP] User aborted sweep selection")
                        return False

                    # Optional: filter queue to start from selected config
                    # For now, we'll just mark as initialized and proceed with queue
                    self._sweep_initialized = True
                    print("[SWEEP] Selection made, proceeding with automated batch execution")

                # Process sweep queue automatically
                while self._sweep_queue:
                    cfg = self._sweep_queue[0]  # ← Peek at config, don't remove yet
                    # Reset flags for new config
                    self._strategy_initialized = False

                    n_agents = cfg["agents"]
                    n_parcels = cfg["parcels"]

                    sweep_name = SCALING_CONFIG.get("name", "sweep")

                    state = self.scan_sweep_config_state(
                        sweep_name=sweep_name,
                        agents=n_agents,
                        parcels=n_parcels,
                    )

                    if state["completed"]:
                        print(f"[SWEEP] A{n_agents}_P{n_parcels} already completed, skipping")
                        self._sweep_queue.pop(0)  # ← Remove the completed config
                        continue

                    scale_root = EXPERIMENTS_DIR / sweep_name / f"A{n_agents}_P{n_parcels}"

                    state = self._load_or_create_experiment(scale_root)

                    if state == ExperimentState.COMPLETED:
                        print(f"[SWEEP] A{n_agents}_P{n_parcels} completed during load, skipping")
                        self._sweep_queue.pop(0)  # ← Remove the completed config
                        continue

                    # If we reach here, we have a valid config to process
                    self._sweep_queue.pop(0)  # ← Remove the config we're about to process

                    if state == ExperimentState.RESUMABLE:
                        break

                    if state == ExperimentState.NEW:
                        break

                    break
                else:
                    print("[SWEEP] All configurations completed.")
                    return False

                if self._resumed:
                    print(f"[SWEEP] Resumed config: agents={n_agents}, parcels={n_parcels}")
                else:
                    print(f"[SWEEP] Running new config: agents={n_agents}, parcels={n_parcels}")

                # Enforce capacity
                try:
                    self.checker.enforce(n_agents)
                except AgentCapacityError as e:
                    print(f"[SWEEP] Skipping config {cfg}: {e}")
                    return

                if USE_SWEEP and SCALING_CONFIG:
                    selected_planner = self.planners[0]
                else:
                    selected_planner = self.show_setup_and_planner_select()
            # -------- GUI MODE --------
            else:
                # Only show GUI setup if not in an active sweep
                if not (USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False)):
                    while True:
                        try:
                            n_agents, n_parcels = self.show_experiment_setup(
                                initial_agents=1,
                                initial_parcels=self.parcels,
                                max_agents=self.system_limit,
                            )

                            self.parcels = n_parcels
                            selected_planner = self.show_setup_and_planner_select()
                            # Resume only after user input defines context
                            self._load_or_create_experiment(EXPERIMENTS_DIR)
                            try:
                                self.checker.enforce(n_agents)
                            except AgentCapacityError as e:
                                self._show_error_and_restart_setup(str(e))
                                continue
                            break

                        except RuntimeError:
                            continue

                        except SystemExit:
                            print("[EXPERIMENT] User aborted at setup")
                            return
                else:
                    # In sweep mode after initialization - values already set from sweep config
                    selected_planner = self.planners[0]  # Use default planner for sweep

        # -------------------------------------------------
        # Commit setup choices to experiment schema
        # -------------------------------------------------
        if not getattr(self, "_resumed", False):
            assert n_parcels is not None, "n_parcels must be set for new experiment"
            assert selected_planner is not None, "selected_planner must be set for new experiment"

            strategy_vault = build_strategy_vault(selected_planner)
            num_instances = SCALING_CONFIG.get("num_instances", 1)
            self._create_experiment(n_agents, n_parcels, selected_planner, strategy_vault, num_instances)

        # -------------------------------------------------
        # STRATEGY LOOP
        # -------------------------------------------------
        assert not (
                USE_SWEEP
                and getattr(self, "_experiment_completed", False)
                and int(self.experiment.get("current_index", 0)) < len(self.experiment["strategies"])
        ), "Experiment marked completed but strategies remain"

        # ONLY THEN check if current experiment has strategies
        current_index = int(self.experiment.get("current_index", 0))
        if current_index >= len(self.experiment["strategies"]):
            if USE_SWEEP and SCALING_CONFIG and self._sweep_queue:
                # More sweep configs exist, continue processing
                return True
            else:
                # No more sweep configs, exit
                print(
                    f"[EXPERIMENT] No remaining strategies to run ({current_index}/{len(self.experiment['strategies'])} processed)")
                return False

        remaining = len(self.experiment["strategies"]) - current_index
        print(
            f"[EXPERIMENT] Starting strategy {current_index + 1}/{len(self.experiment['strategies'])} ({remaining} remaining)")

        for idx, strategy in enumerate(self.experiment["strategies"]):
            # print("Strategies", self.experiment["strategies"])
            if idx < int(self.experiment.get("current_index", 0)):
                continue
            # print(f"self.experiment[\"strategies\"] {self.experiment['strategies']}")

            # Reset completion wait counter for this strategy
            if hasattr(self, '_completion_wait_counter'):
                del self._completion_wait_counter
            base_name = strategy.get("name", f"strategy_{idx}")
            variant = strategy.get("variant")

            if base_name == "sppdec" and variant:
                strategy_name = f"sppdec_{variant}"
            elif base_name == "centralized_cc" and variant:
                print("BASE_NAME", base_name)
                strategy_name = f"centralized_cc_{variant}"
            else:
                strategy_name = base_name

            strategy_type = strategy.get("strategy_type", "unknown")
            topology = strategy.get("topology", "unknown")
            planner_name = strategy["planner"]
            print(f"[EXPERIMENT] Running planner '{planner_name}' ({idx + 1}/{len(self.experiment['strategies'])})"
                  f"Current Strategy {strategy_name}"
                  f"Strategy mode {topology}")

            self._configure_planner(planner_name)

            snapshot = strategy.get("snapshot")

            # With the fixed timestep, seed the global RNG per run so strategies'
            # internal randomness (rollouts, tie-breaks) is reproducible and does
            # not depend on which other arms ran earlier in this process.
            if sim_clock.FIXED_TIMESTEP:
                random.seed(f"{self.experiment['seed']}:{strategy.get('name')}")
            # -------------------------------------------------
            # >>> FLIGHT RECORDER: instantiate once per strategy
            # -------------------------------------------------
            run_id = self.experiment["run_id"]

            metrics = MetricsAccumulator(
                run_id=self.experiment["run_id"],
                planner=planner_name,
                seed=self.experiment["seed"],
                n_drones=len(self.experiment["agents"]),
                n_parcels=len(self.experiment.get("parcels", [])),
                comm_reliability=self.experiment["comm_config"].get("reliability", 1.0),
                battery_capacity=100.0,  # safe default before agents exist
                episodic_interval=1.0,
            )
            flight_recorder = FlightRecorder(
                run_id=run_id,
                planner=planner_name,
                strategy_name=strategy_name,
                agents=self.experiment["agents"],
                parcels=self.experiment["parcels"],  # Still pass initial parcel definitions
                grid_size=GRID_SIZE,
                sim_dt=1.0 / 60.0,
                metrics=metrics
            )
            try:
                terrain, agents, pre_steps, all_parcels = self._setup_scene(maybe_snapshot=snapshot,
                                                                            strategy_name=strategy_name,
                                                                            strategy_spec=strategy,
                                                                            flight_recorder=flight_recorder
                                                                            )

                proceed = self._show_strategy_countdown(
                    terrain=terrain,
                    agents=agents,
                    strategy_name=strategy_name,
                    topology=topology,
                    variant=variant,
                    config=strategy.get("config")
                )

                if not proceed:
                    print("[EXPERIMENT] Strategy skipped by user")
                    continue

                from src.strategies.factory import build_strategy
                strategies = {}
                for drone in agents:
                    import copy
                    strategies[drone.agent_id] = build_strategy(
                        strategy_spec=copy.deepcopy(strategy),
                        terrain=terrain,
                        agents=agents,
                        physics=physics,
                        graphics_dir=GRAPHICS_DIR,
                        grid_size=GRID_SIZE,
                    )

            except AgentSpawnError as e:
                self._show_error_and_restart_setup(str(e))
                break
            # exit run(), caller restarts setup

            if terrain is None:
                return  # safely return to GUI loop

            # After creating flight_recorder:
            if snapshot:
                # Reconstruct history from snapshot
                flight_recorder.reconstruct_from_snapshot(snapshot)

                # Then register current parcels (they'll be skipped if already registered)
                for parcel in all_parcels:
                    flight_recorder.register_parcel(parcel)
            else:
                # New simulation
                for parcel in all_parcels:
                    flight_recorder.register_parcel(parcel)
            try:
                # controllers (one per agent)
                controllers = {}
                for i, drone in enumerate(agents):
                    drone.agent_id = self.experiment["agents"][i]["id"]
                # One sensor model for every drone of this run (all strategies see the same limited view), and
                # the start information the whole fleet shares.
                sensing = strategy.get("sensing") or DEFAULT_SENSING
                sensor = SensorModel(physics.manhattan_distance, radius=sensing.get("radius", float("inf")),
                                     noise_std=sensing.get("noise_std", 0.0), loss=sensing.get("loss", 0.0),
                                     parcel_radius=sensing.get("parcel_radius"), delay=sensing.get("delay", 0),
                                     blackout=sensing.get("blackout"))
                start_info = WorldPort.build_start_info(terrain)
                # One process per drone, each on its own core, for every strategy and every run (the only mode, forced in src/config.py). The centralised strategy's ground station is a
                # world-side entity; its drones talk to it over the radio like to any other drone.
                use_processes = True      # the only mode (src/config.py): one process per drone, each on its own core
                drone_cores = allocate_cores(len(agents)) if use_processes else [None] * len(agents)
                # The radio of the run: one profile for every link and every strategy (arm key "radio", or
                # RADIO_PROFILE in the environment; default "ideal", "submitted" reproduces the submitted setting).
                radio_profile = apply_profile(
                    strategy.get("radio") or os.environ.get("RADIO_PROFILE") or DEFAULT_PROFILE,
                    agents, terrain.command_center)
                print(f"[RADIO] profile: {radio_profile.name}", flush=True)
                print(f"[PROCESS] drone processes: {'on' if use_processes else 'off'}", flush=True)
                for i, drone in enumerate(agents):
                    agent_id = self.experiment["agents"][i]["id"]
                    drone.agent_id = agent_id
                    if terrain.command_center:
                        terrain.command_center.network.flight_recorder = flight_recorder
                        terrain.command_center.register_agent(drone)

                    if use_processes:
                        # The drone is its own process on its own core: the world sends it observations and carries
                        # out the commands it returns.
                        controllers[agent_id] = RemoteController(
                            drone=drone, terrain=terrain, strategy_spec=strategy, topology=topology, sensor=sensor,
                            start_info=start_info, run_seed=int(self.experiment.get("seed", 0) or 0),
                            core=drone_cores[i])
                        drone.controller = controllers[agent_id]   # the runner reads the drone's reported status here
                        if hasattr(drone, "network"):
                            drone.network.flight_recorder = flight_recorder
                        print(f"[PROCESS] drone {agent_id} runs as process {controllers[agent_id].pid} "
                              f"on core {drone_cores[i]}", flush=True)
                        continue

                    ai = AIAgentController(
                        drone=drone,
                        terrain=terrain,
                        strategy=strategies[agent_id],
                        flight_recorder=flight_recorder,
                        topology=topology,
                        sensor=sensor,
                        start_info=start_info,
                        run_seed=int(self.experiment.get("seed", 0) or 0),
                    )
                    drone.controller = ai
                    if hasattr(drone, "network"):
                        drone.network.flight_recorder = flight_recorder
                    strategies[agent_id].set_drone(ai.services)
                    if hasattr(strategies[agent_id], "flight_recorder"):
                        strategies[agent_id].flight_recorder = flight_recorder

                    print("Drone", drone)
                    print("Drone ID", drone.agent_id)

                    # Only create human controller and switcher in GUI mode
                    human = HumanAgentController(drone, terrain)
                    switcher = ControllerSwitcher([human, ai])
                    switcher.index = 1  # Start with AI controller
                    controllers[agent_id] = switcher

                from concurrent.futures import ThreadPoolExecutor
                executor = ThreadPoolExecutor(max_workers=len(agents))

                steps = int(pre_steps or 0)
                last_move_time = time.time()
                last_action_time = time.time()
                last_progress_step = steps  # last step any drone picked or dropped a parcel
                aborted = False
                strategy_finished = False

                prev_positions = {i: (a.pos.x, a.pos.y) for i, a in enumerate(agents)}

                # -------------------------------------------------
                # MAIN SIMULATION LOOP
                # -------------------------------------------------

                while steps < self.max_steps:
                    steps += 1
                    # Keep physics clock in sync with the simulation
                    physics.current_step = steps
                    sim_time = steps * (1.0 / 60.0)
                    sim_clock.set_time(sim_time)

                    # World event: a change in consumption part-way through (spec["fleet_profile"]["shift"] =
                    # {"step", "rate_scale", "weight_scale"}), e.g. wear or damage. Nothing tells the drones.
                    shift = (strategy.get("fleet_profile") or {}).get("shift")
                    if shift and steps == int(shift["step"]):
                        for a in agents:
                            a.energy_rate *= float(shift.get("rate_scale", 1.0))
                            a.weight_factor *= float(shift.get("weight_scale", 1.0))
                        print(f"[EXPERIMENT] consumption shift at step {steps}: {shift}")

                    # CENTRALIZED COGNITION TICK
                    if terrain.command_center:
                        terrain.command_center.update(
                            agents=agents,
                            parcels=[q for q in terrain.parcels if int(getattr(q, "release_step", 0)) <= steps],
                            now=sim_clock.now(),
                        )
                        if hasattr(terrain.command_center,
                                   "globally_infeasible") and terrain.command_center.globally_infeasible:
                            print("[EXPERIMENT] CommandCenter signaled global infeasibility")

                            self._finalize_strategy(
                                idx=idx,
                                status="aborted",
                                planner_name=planner_name,
                                strategy_name=strategy_name,
                                strategy_type=strategy_type,
                                steps=steps,
                                flight_recorder=flight_recorder,
                            )

                            aborted = True
                            break

                    # -------------------------------------------------
                    # RESET NETWORK BANDWIDTH FOR NEW STEP
                    # -------------------------------------------------
                    for drone in agents:
                        if hasattr(drone, "network"):
                            drone.network.reset_step()

                    if terrain.command_center and hasattr(terrain.command_center, "network"):
                        terrain.command_center.network.reset_step()

                    # Single event handling section
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            aborted = True
                            break

                        if e.type == pygame.KEYDOWN:
                            if e.key == pygame.K_ESCAPE:
                                aborted = True
                                break
                            elif e.key == pygame.K_h:  # H to toggle fast mode
                                self.fast_mode = not self.fast_mode
                                mode_text = "FAST" if self.fast_mode else "NORMAL"
                                print(f"[TOGGLE] Switched to {mode_text} mode")
                                continue

                        # Controller events
                        for switcher in controllers.values():
                            if hasattr(switcher, 'handle_event'):
                                switcher.handle_event(e)

                    # Fixed timestep: physics advances SIM_DT per step regardless of
                    # frame time, so slow frames (e.g. SLM calls) do not move drones further.
                    wall_dt = self.clock.tick(0 if sim_clock.FIXED_TIMESTEP else 60) / 1000.0
                    dt = sim_clock.step_dt(wall_dt)

                    # Rendering section (no more event handling here)
                    if not self.fast_mode:
                        # FULL RENDERING MODE
                        self.screen.fill((192, 192, 192))
                        for x in range(0, self.screen_size[0], GRID_SIZE):
                            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
                        for y in range(0, self.screen_size[1], GRID_SIZE):
                            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

                        terrain.draw(self.screen)
                        for drone in agents:
                            drone.draw(self.screen, self.images)
                            if topology in ("decentralized", "predictive_decentralized", "centralized"):
                                drone.draw_network_pulse(self.screen, min_cells=1, max_cells=3)
                    else:
                        # FAST MODE: Status display only
                        self._show_experiment_status_screen(strategy_name, steps, agents, terrain)

                    pygame.display.flip()
                    # ------------------------------
                    # UPDATE AGENTS
                    # ------------------------------
                    # ==============================
                    # One step. Every drone observes (the same instant), every drone thinks (in parallel, from its
                    # observation only), then the world carries out the commands in drone order and runs the physics.
                    # ==============================
                    from concurrent.futures import ThreadPoolExecutor

                    if not hasattr(self, "_executor"):
                        self._executor = ThreadPoolExecutor(max_workers=len(agents))

                    executor = self._executor
                    agent_ids = [self.experiment["agents"][i]["id"] for i in range(len(agents))]

                    observations = {aid: controllers[aid].observe(steps, sim_time) for aid in agent_ids}
                    thoughts = dict(zip(agent_ids, executor.map(
                        lambda aid: controllers[aid].think(observations[aid], dt, steps, sim_time), agent_ids)))

                    # The records the drones wrote while thinking go to the flight recorder, in drone order
                    for aid in agent_ids:
                        controllers[aid].replay(flight_recorder)

                    # A drone can tell the world that its strategy sees no way to go on (the run-level abort)
                    decisions = {aid: {"mode": "abort", "reason": sig["abort"]}
                                 for aid, (_, sig) in thoughts.items() if "abort" in sig}
                    # ==============================
                    # ABORT PROPAGATION CHECK
                    # ==============================
                    for agent_id, decision in decisions.items():
                        if isinstance(decision, dict) and decision.get("mode") == "abort":
                            reason = decision.get("reason", AbortReason.AGENT_ENERGY_INFEASIBLE)

                            print(f"[EXPERIMENT] Strategy requested abort (agent={agent_id}, reason={reason})")

                            # Resolve the actual agent object
                            agent = next(a for a in agents if a.agent_id == agent_id)

                            # Record abort inside FlightRecorder
                            flight_recorder.record_abort(
                                reason=reason,
                                agent_id=agent_id,
                                step=steps,
                                sim_time=sim_time,
                                metadata={
                                    "battery_pct": float(agent.power.percent()) if hasattr(agent, "power") else None,
                                    "battery": float(agent.power.level) if hasattr(agent, "power") else None,
                                    "position": (int(agent.col), int(agent.row)),
                                    "carrying": getattr(agent.carrying, "id", None),
                                    "topology": topology,
                                    "strategy": strategy_name,
                                }
                            )

                            # Then finalize as usual
                            self._finalize_strategy(
                                idx=idx,
                                status="aborted",
                                planner_name=planner_name,
                                strategy_name=strategy_name,
                                strategy_type=strategy_type,
                                steps=steps,
                                flight_recorder=flight_recorder,
                            )

                            strategy_finished = True
                            aborted = True
                            break

                    if aborted:
                        break

                    # The world carries out every drone's commands, in drone order
                    for aid in agent_ids:
                        controllers[aid].act(thoughts[aid][0], steps)

                    # ==============================
                    # Physics
                    # ==============================

                    for i, drone in enumerate(agents):
                        agent_id = self.experiment["agents"][i]["id"]

                        drone.update(
                            dt,
                            SPEED * drone.speed_scale,
                            ANIM_FPS,
                            rot_frames_with_parcel=self.images.get("drone_rot_with_parcel_frames"),
                            rot_frames=self.images.get("drone_rot_frames"),
                        )

                        px, py = prev_positions[i]
                        dx = drone.pos.x - px
                        dy = drone.pos.y - py
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist > 0.001:
                            last_move_time = time.time()
                            drone.airborne = True                       # a drone leaves the ground with its first movement
                        elif getattr(controllers.get(agent_id), "parked", False):
                            drone.airborne = False                      # a parked drone is on the ground
                        elif getattr(drone, "airborne", False) and HOVER_POWER_RATIO > 0.0:
                            drone.consume_hover(HOVER_POWER_RATIO, dt, SPEED)       # in the air and not moving: it hovers, which costs energy

                        flight_recorder.record_distance(dist, GRID_SIZE)
                        prev_positions[i] = (drone.pos.x, drone.pos.y)

                        # -------------------------------------------------
                        # >>> FLIGHT RECORDER: per-agent tick (state)
                        # -------------------------------------------------
                        flight_recorder.tick_agent(
                            agent_id=agent_id,
                            drone=drone,
                            step=steps,
                            sim_time=sim_time,
                        )
                        if HOVER_POWER_RATIO > 0.0 and agent_id in flight_recorder.agents:
                            flight_recorder.agents[agent_id]["hover_energy"] = round(getattr(drone, "hover_energy_used", 0.0), 4)

                        if hasattr(drone, "last_action") and drone.last_action:
                            kind, cell, parcel = drone.last_action
                            last_action_time = time.time()
                            last_progress_step = steps

                            # -------------------------------------------------
                            # >>> FLIGHT RECORDER: record action
                            # -------------------------------------------------
                            flight_recorder.record_action(
                                agent_id=agent_id,
                                kind=kind,
                                cell=cell,
                                parcel=parcel,
                                step=steps,
                                sim_time=sim_time,
                                world_time=time.time(),
                            )

                            if kind == "drop" and parcel and not getattr(parcel, "delivered", False):
                                parcel.delivered = True
                                parcel.picked = False
                                try:
                                    station = terrain.get_station_at(cell[0], cell[1])
                                    if station:
                                        try:
                                            station.register_delivery(cell)
                                        except TypeError:
                                            station.register_delivery()
                                except Exception:
                                    pass
                                flight_recorder.record_task_complete(1)

                            if kind in ("pick_failed", "drop_failed"):
                                flight_recorder.record_unsafe_failure(1)

                            self._append_event(
                                idx,
                                {"step": steps, "type": kind, "agent": agent_id, "cell": cell},
                            )

                            drone.last_action = None

                    # -------------------------------------------------
                    # >>> FLIGHT RECORDER: fleet-level checks
                    # -------------------------------------------------
                    flight_recorder.tick_fleet(
                        agents=agents,
                        step=steps,
                        sim_time=sim_time,
                    )

                    # -------------------------------------------------
                    # NETWORK DELIVERY PHASE
                    # -------------------------------------------------

                    # Deliver messages sent by drones
                    for drone in agents:
                        if hasattr(drone, "process_network"):
                            drone.process_network(steps)

                    # Deliver messages sent by CommandCenter
                    if terrain.command_center and hasattr(terrain.command_center, "process_network"):
                        terrain.command_center.process_network(steps)

                    # -------------------------------------------------
                    # TERMINATION CONDITIONS
                    # -------------------------------------------------

                    # ---------------------------------------------
                    # 1. Detect and log agent losses (NON-TERMINAL)
                    # ---------------------------------------------
                    dead_agents = [a for a in agents if self.agent_dead(a)]

                    for d in dead_agents:
                        # Log each death only once
                        if not getattr(d, "_death_logged", False):
                            flight_recorder.fleet_events.append({
                                "type": "agent_lost",
                                "agent": getattr(d, "agent_id", None),
                                "cell": (int(d.col), int(d.row)),
                                "battery": getattr(d.power, "level", None),
                                "time": {
                                    "step": steps,
                                    "sim_time": sim_time,
                                    "world_time": time.time(),
                                },
                            })
                            d._death_logged = True

                    # ---------------------------------------------
                    # 2. Determine remaining mission capability
                    # ---------------------------------------------
                    alive_agents = [a for a in agents if not self.agent_dead(a)]
                    active_controllers = [
                        a.controller for a in alive_agents
                        if a.controller is not None
                    ]

                    parked_controllers = [
                        c for c in active_controllers
                        if getattr(c, "parked", False)
                    ]

                    remaining = sum(1 for p in terrain.parcels if not p.delivered)
                    # -------------------------------------------------
                    # RUNNER DIAGNOSTIC TRACE (publication-grade)
                    # -------------------------------------------------
                    agent_modes = {}
                    for a in alive_agents:
                        controller = getattr(a, "controller", None)
                        if controller is None:
                            mode = "no_controller"
                        elif getattr(controller, "parked", False):
                            mode = "idle"
                        elif getattr(controller, "has_pending_task", False):
                            mode = "plan"
                        else:
                            mode = "uncertain"
                        agent_modes[a.agent_id] = mode

                    decision = "continue"
                    if alive_agents and len(parked_controllers) == len(active_controllers) and remaining > 0:
                        decision = "infeasible"
                    elif self.mission_exhausted(terrain, agents):
                        decision = "completed"
                    flight_recorder.fleet_events.append({
                        "type": "runner_state",
                        "step": steps,
                        "agent_modes": agent_modes,
                        "remaining_parcels": remaining,
                        "decision": decision,
                    })

                    # ---------------------------------------------
                    # DEBUG: Mission classification visibility
                    # ---------------------------------------------
                    agent_states = {
                        a.agent_id: {
                            "dead": self.agent_dead(a),
                            "at_station": self.agent_at_base(a, terrain),
                            "lost": self.agent_dead(a) and not self.agent_at_base(a, terrain),
                            "parked": getattr(a.controller, "parked", None) if a.controller else None,
                            "has_pending_task": getattr(a.controller, "has_pending_task",
                                                        None) if a.controller else None,
                            "battery": round(a.power.level, 2) if hasattr(a, "power") else None,
                        }
                        for a in agents
                    }
                    dead_total = sum(1 for s in agent_states.values() if s["dead"])
                    dead_at_station = sum(1 for s in agent_states.values() if s["dead"] and s["at_station"])
                    lost_in_field = sum(1 for s in agent_states.values() if s["lost"])
                    # ---------------------------------------------
                    # 3. HARD FAILURE: mission infeasible (behavioral)
                    # ---------------------------------------------
                    if alive_agents and len(parked_controllers) == len(active_controllers) and remaining > 0:
                        print("[EXPERIMENT] Mission infeasible: all agents exhausted and parked with remaining parcels")

                        flight_recorder.fleet_events.append({
                            "type": "mission_infeasible",
                            "step": steps,
                            "alive_agents": len(alive_agents),
                            "parked_agents": len(parked_controllers),
                            "remaining_parcels": remaining,
                            "agent_modes": agent_modes,
                            "decision": decision,
                        })

                        self._finalize_strategy(
                            idx=idx,
                            status="infeasible",
                            planner_name=planner_name,
                            strategy_name=strategy_name,
                            strategy_type=strategy_type,
                            steps=steps,
                            flight_recorder=flight_recorder,
                        )

                        self._show_post_run_inspection(
                            terrain=terrain,
                            agents=agents,
                            strategy_name=strategy_name,
                            status="infeasible",
                            variant=variant,
                            config={
                                "alive_agents": len(alive_agents),
                                "parked_agents": len(parked_controllers),
                                "remaining_parcels": remaining,
                                "reason": "All agents aborted planning due to insufficient battery and entered parking state"
                            }
                        )
                        print(
                            "[EXPERIMENT] Mission infeasible. "
                            "All remaining agents voluntarily entered idle with parcels remaining."
                        )

                        break

                    # ---------------------------------------------
                    # 4. HARD FAILURE: all agents lost
                    # ---------------------------------------------
                    if not alive_agents and remaining > 0:
                        self._finalize_strategy(
                            idx=idx,
                            status="failed",
                            planner_name=planner_name,
                            strategy_name=strategy_name,
                            strategy_type=strategy_type,
                            steps=steps,
                            flight_recorder=flight_recorder,
                        )

                        self._show_post_run_inspection(
                            terrain=terrain,
                            agents=agents,
                            strategy_name=strategy_name,
                            status="failed",
                            variant=variant,
                            config={
                                "alive_agents": 0,
                                "remaining_parcels": remaining,
                                "reason": "All drones lost before completing deliveries"
                            }
                        )
                        print(
                            "[EXPERIMENT] Mission failed. "
                            f"All agents exhausted. Lost in field={lost_in_field}, "
                            f"dead at station={dead_at_station}, remaining parcels={remaining}."
                        )

                        break

                    # ---------------------------------------------
                    # 4b. STALL: parcels remain but no parcel has been picked or dropped for
                    # STALL_STEPS. Movement alone is not progress: drones can circle without
                    # ever delivering (livelock). The longest single trip takes ~7 s simulated,
                    # so 60 s without any pick/drop means the mission cannot finish.
                    # ---------------------------------------------
                    if (sim_clock.FIXED_TIMESTEP and remaining > 0
                            and steps - last_progress_step >= STALL_STEPS):
                        print(f"[EXPERIMENT] Stalled: no pick/drop for {STALL_STEPS} steps "
                              f"with {remaining} parcels remaining; recording run as stalled")
                        self._finalize_strategy(
                            idx=idx,
                            status="stalled",
                            planner_name=planner_name,
                            strategy_name=strategy_name,
                            strategy_type=strategy_type,
                            steps=steps,
                            flight_recorder=flight_recorder,
                        )
                        strategy_finished = True
                        break

                    # ---------------------------------------------
                    # 5. SUCCESS CONDITION - NEW PARKING-AWARE CHECK
                    # ---------------------------------------------
                    if self.mission_exhausted(terrain, agents):
                        status = "completed"
                        # print(f"[DEBUG] Mission exhausted condition met at step {steps}")
                        # Add a brief wait to ensure all agents are settled at the station
                        if not hasattr(self, '_completion_wait_counter'):
                            self._completion_wait_counter = 0
                            print(f"[DEBUG] Starting completion wait...")

                        self._completion_wait_counter += 1

                        # Wait for 1 second (60 frames at 60 FPS) before finalizing
                        if self._completion_wait_counter < 60:
                            # print(f"[DEBUG] Waiting for agents to settle ({self._completion_wait_counter}/60)...")
                            continue  # Continue simulation for a bit longer

                        print(f"[DEBUG] All conditions confirmed. Marking strategy as completed.")

                        # Export flight data
                        self._finalize_strategy(
                            idx=idx,
                            status=status,
                            planner_name=planner_name,
                            strategy_name=strategy_name,
                            strategy_type=strategy_type,
                            steps=steps,
                            flight_recorder=flight_recorder,
                        )

                        # BREAK OUT OF SIMULATION LOOP IMMEDIATELY
                        strategy_finished = True
                        break

                # Step limit reached without completion or abort: record the run as a
                # timeout (makespan = step limit, delivered count as reached) instead of
                # leaving it unrecorded.
                if not strategy_finished and not aborted and steps >= self.max_steps:
                    print(f"[EXPERIMENT] Step limit {self.max_steps} reached; recording run as timeout")
                    self._finalize_strategy(
                        idx=idx,
                        status="timeout",
                        planner_name=planner_name,
                        strategy_name=strategy_name,
                        strategy_type=strategy_type,
                        steps=steps,
                        flight_recorder=flight_recorder,
                    )
                    strategy_finished = True

                if aborted:
                    print("[EXPERIMENT] Simulation loop exited due to abort")
                    break

                # -------------------------------------------------
                # POST-RUN UI (per strategy)
                # -------------------------------------------------
                if idx < len(self.experiment["strategies"]) - 1:
                    # There are more strategies to run
                    if not (USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False)):
                        self._show_post_run_inspection(
                            terrain=terrain,
                            agents=agents,
                            strategy_name=strategy_name,
                            status=self.experiment["strategies"][idx]["status"],
                            variant=variant,
                            config=strategy.get("config")
                        )
                    continue
                else:
                    # This was the last strategy
                    self._show_experiment_complete_screen(terrain, agents)
                    break

            finally:
                for _controller in (locals().get("controllers") or {}).values():   # end any drone processes
                    try:
                        _controller.close()
                    except Exception:
                        pass
                try:
                    flight_recorder.finalize()
                except Exception:
                    pass

        # -------------------------------------------------
        # SIGNAL TO CALLER WHETHER SWEEP SHOULD CONTINUE
        # -------------------------------------------------
        # At the very end of run() method, replace the final return logic:
        if USE_SWEEP and SCALING_CONFIG:
            if self._sweep_queue:
                print(f"[SWEEP] Moving to next config, {len(self._sweep_queue)} remaining")
                return True  # Continue to next sweep config
            else:
                print("[SWEEP] All sweep configs completed")
                return False
        else:
            return False


def main():
    runner = ExperimentRunnerGUI()
    try:
        while True:
            should_continue = runner.run()
            if not should_continue:
                break
    except SystemExit:
        pygame.quit()


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     import cProfile
#
#     profiler = cProfile.Profile()
#     profiler.enable()
#
#     main()
#
#     profiler.disable()
#     profiler.dump_stats("profile_results.prof")
