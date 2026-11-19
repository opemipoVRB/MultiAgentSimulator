# src/strategies/factory.py
import os
from typing import Dict
from src.strategies.marl import MARLStrategy
from src.strategies.single_task_bundle_cbba_decentralized_auction import SingleTaskBundlesCBBADecentralizedStrategy
from src.strategies.decentralized_auction import DecentralizedAuctionStrategy
from src.strategies.cbba_decentralized_auction import LoadBalancedCBBADecentralizedStrategy
from src.strategies.naive import NaiveStrategy
from src.strategies.centralised import CentralisedStrategy
from src.strategies.sppdec.strategy import SPPDECStrategy
from src.artifacts import CommandCenter
from src.sensor import DEFAULT_SENSING


def _build(
    *,
    strategy_spec: Dict,
    terrain,
    agents,
    physics,
    graphics_dir,
    grid_size,
):
    """
    Instantiate a strategy and wire any required world infrastructure.

    This is the ONLY place allowed to:
    - create CommandCenter
    - attach it to terrain
    - choose concrete Strategy classes
    """

    name = strategy_spec["name"]

    # -----------------------------
    # Centralized strategy
    # -----------------------------
    if name.startswith("centralized_cc"):
        station = terrain.stations[0]
        cc_col = station.col + station.w // 2
        cc_row = station.row + station.h // 2
        variant = strategy_spec.get("variant", "combined")

        command_center = CommandCenter(
            col=cc_col,
            row=cc_row,
            grid_size=grid_size,
            physics=physics,
            network_capacity=100,
            network_range=8,
            variant=variant,
            image_path=str(graphics_dir / "command_center.png"),
        )

        terrain.command_center = command_center
        command_center.bind_terrain(terrain)

        for drone in agents:
            command_center.register_agent(drone)

        return CentralisedStrategy(physics=physics)

    # -----------------------------
    # Communication-free MARL baseline (learned task choice; see .ignore/revision/marl_baseline_design.md)
    # -----------------------------
    if name.startswith("marl_ippo"):
        terrain.command_center = None
        weights = strategy_spec.get("marl_weights") or os.environ.get("MARL_WEIGHTS")
        if not weights:
            raise ValueError("marl_ippo needs the policy file: marl_weights in the arm or MARL_WEIGHTS in the environment")
        return MARLStrategy(
            physics=physics,
            weights=weights,
            sensing=strategy_spec.get("sensing") or DEFAULT_SENSING,
            trace_dir=os.environ.get("MARL_TRACE_DIR"),
        )

    # -----------------------------
    # SPPDEC strategy
    # -----------------------------
    if name.startswith("sppdec"):
        terrain.command_center = None
        # config_variant selects (beta, w-bar) when the arm name differs from it,
        # e.g. arbiter ablations run on the "combined" configuration.
        variant = strategy_spec.get("config_variant", strategy_spec.get("variant", "combined"))

        return SPPDECStrategy(
            physics=physics,
            variant=variant,
            arbiter=strategy_spec.get("arbiter", "slm"),
            slm_latency_steps=strategy_spec.get("slm_latency_steps", 0),
            slm_model=strategy_spec.get("slm_model"),
            prompt_version=strategy_spec.get("prompt_version"),
            symmetric_costs=strategy_spec.get("symmetric_costs", False),
            exclude_carrying_peers=strategy_spec.get("exclude_carrying_peers", False),
            no_slot_policy=strategy_spec.get("no_slot_policy", "greedy"),
            sensing=strategy_spec.get("sensing") or DEFAULT_SENSING,
            peer_rate_assumption=strategy_spec.get("peer_rate_assumption", "known"),
            conflict_policy=strategy_spec.get("conflict_policy", "trend"),
            coordination_model=strategy_spec.get("coordination_model", "hungarian"),
            slm_latency_mode=strategy_spec.get("slm_latency_mode", "blocking"),
            cbba_bundle_size=strategy_spec.get("cbba_bundle_size", 1),
            assumption_memory=strategy_spec.get("assumption_memory", False),
            observe_every_step=strategy_spec.get("observe_every_step", False),
            observe_gone=strategy_spec.get("observe_gone", True),
            observe_contested=strategy_spec.get("observe_contested", True),
            classifier_model=strategy_spec.get("classifier_model", "distilled"),
            use_self_check=strategy_spec.get("use_self_check", False),
            own_rate_known=strategy_spec.get("own_rate_known", True),
            peer_prior=strategy_spec.get("peer_prior", "like_self"),
            hardware=strategy_spec.get("hardware"),
            prompt_cache=strategy_spec.get("prompt_cache", False),
            world_view=strategy_spec.get("world_view", False),
            slm_suggest=strategy_spec.get("slm_suggest", True),
            plan_ahead=strategy_spec.get("plan_ahead", False),
            slm_think=strategy_spec.get("slm_think", False),
            rule_perturb=strategy_spec.get("rule_perturb", 0.0),
            freshness=strategy_spec.get("freshness", False),
            territory=strategy_spec.get("territory", False),
            territory_hint=strategy_spec.get("territory_hint"),
            ledger_complete=strategy_spec.get("ledger_complete", False),
        )

    # -----------------------------
    # Naive decentralized
    # -----------------------------
    if name.startswith("naive_decentralized"):
        terrain.command_center = None

        return NaiveStrategy(
            physics=physics,
            num_trials=10,
        )

    # -----------------------------
    # CBBA auction
    # -----------------------------

    # -----------------------------
    # Decentralized Auction (baseline)
    # -----------------------------

    if name.startswith("decentralized_auction"):
        return DecentralizedAuctionStrategy(physics=physics)

    # -----------------------------
    # CBBA variants
    # -----------------------------

    # -----------------------------
    # CBBA full bundle
    # -----------------------------
    if name.startswith("cbba_full"):
        strategy = LoadBalancedCBBADecentralizedStrategy(physics=physics)
        strategy.capability_bundle = bool(strategy_spec.get("capability_bundle", False))
        return strategy
    # -----------------------------
    # CBBA single-task (bundle size = 1)
    # -----------------------------
    if name.startswith("cbba_single"):
        return SingleTaskBundlesCBBADecentralizedStrategy(physics=physics)
    raise ValueError(f"Unknown strategy: {name}")


def build_strategy(*, strategy_spec: Dict, **kwargs):
    """Build the strategy for the arm and set the information-model options every strategy shares (station memory, ledger stop)."""
    strategy = _build(strategy_spec=strategy_spec, **kwargs)
    strategy.station_memory = strategy_spec.get("station_memory", "live")
    strategy.ledger_stop = bool(strategy_spec.get("ledger_stop", False))
    strategy.time_weight = float(strategy_spec.get("time_weight", 0.0))
    strategy.station_bids = bool(strategy_spec.get("station_bids", False))
    if hasattr(strategy, "graph"):
        # consensus: act on what Hungarian and CBBA agree on (see SPPDECLangGraph._consensus_decide)
        strategy.graph.consensus = bool(strategy_spec.get("consensus", False))
        strategy.graph.time_weight = strategy.time_weight
        strategy.graph.consensus_models = tuple(strategy_spec.get("consensus_models") or ("hungarian", "cbba_rounds"))
        strategy.graph.model_rule = strategy_spec.get("model_rule")
        strategy.graph.recheck = bool(strategy_spec.get("recheck", False))
        strategy.graph.no_gamble = bool(strategy_spec.get("no_gamble", False))
        strategy.graph.commitments = bool(strategy_spec.get("commitments", False))
        strategy.graph.count_first = bool(strategy_spec.get("count_first", False))
        strategy.graph.keep_schedule = bool(strategy_spec.get("keep_schedule", False))
        strategy.graph.sweep = dict(strategy_spec.get("sweep") or {})
        strategy.graph._apply_sweep()
        # The consensus method sizes bundles by each agent's share of the team's capability and prices flight time at each agent's declared speed; an arm sets either to False to ablate it
        # (the head-count rule ceil(open tasks / agents) is the same for agents of equal capability). The legacy, non-consensus arms keep their earlier behaviour.
        strategy.graph.speed_aware = bool(strategy_spec.get("speed_aware", strategy.graph.consensus))
        strategy.graph.completion_bid = bool(strategy_spec.get("completion_bid", False))
        strategy.graph.rolling = bool(strategy_spec.get("rolling", False))
        strategy.graph.equal_bid = bool(strategy_spec.get("equal_bid", False))
        strategy.graph.share_bundles = bool(strategy_spec.get("share_bundles", strategy.graph.consensus))
        strategy.graph.time_weight_hungarian = float(strategy_spec.get("time_weight_hungarian", 0.0))
    # Declared orders (mandatory): every order is declared to the whole fleet with its id, position and weight before the first step, as in the original CBBA, and
    # every strategy of every experiment sees it. Weights are not learned by sensing.
    strategy.declared_weights = True
    strategy.ledger_verify = bool(strategy_spec.get("ledger_verify", False))
    if getattr(strategy, "graph", None) is not None:
        strategy.graph.station_bids = strategy.station_bids
        strategy.graph.declared_weights = strategy.declared_weights
        strategy.graph.avoid_top_k = bool(strategy_spec.get("avoid_top_k", False))
        strategy.graph.avoid_width_fixed = strategy_spec.get("avoid_width")
        strategy.graph.avoid_beliefs = bool(strategy_spec.get("avoid_beliefs", False))
        strategy.graph.peer_model_shadow = bool(strategy_spec.get("peer_model_shadow", False))
        strategy.graph.station_model = bool(strategy_spec.get("station_model", False))
        strategy.graph.peer_personalities = bool(strategy_spec.get("peer_personalities", True))
        strategy.graph.peer_model_cost = bool(strategy_spec.get("peer_model_cost", False))
    return strategy
