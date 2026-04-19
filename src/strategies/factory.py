# src/strategies/factory.py
from typing import Dict
from src.strategies.single_task_bundle_cbba_decentralized_auction import SingleTaskBundlesCBBADecentralizedStrategy
from src.strategies.decentralized_auction import DecentralizedAuctionStrategy
from src.strategies.cbba_decentralized_auction import LoadBalancedCBBADecentralizedStrategy
from src.strategies.naive import NaiveStrategy
from src.strategies.centralised import CentralisedStrategy
from src.strategies.sppdec.strategy import SPPDECStrategy
from src.artifacts import CommandCenter


def build_strategy(
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
    # SPPDEC strategy
    # -----------------------------
    if name.startswith("sppdec"):
        terrain.command_center = None
        variant = strategy_spec.get("variant", "combined")

        return SPPDECStrategy(
            physics=physics,
            variant=variant,
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
        return LoadBalancedCBBADecentralizedStrategy(physics=physics)
    # -----------------------------
    # CBBA single-task (bundle size = 1)
    # -----------------------------
    if name.startswith("cbba_single"):
        return SingleTaskBundlesCBBADecentralizedStrategy(physics=physics)
    raise ValueError(f"Unknown strategy: {name}")
