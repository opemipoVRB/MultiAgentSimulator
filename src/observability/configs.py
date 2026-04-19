# src/observability/configs.py

import random
from typing import List, Dict


def generate_station_scaling(
        station_w: int,
        station_h: int,
        mode: str = "research",
        budget: int = 40,
        seed: int = 1337,
) -> List[Dict[str, int]]:
    """
    Generates agent vs parcel sweep configurations.

    Modes:
      - "full": exhaustive grid
      - "research": boundary + diagonal + representative midpoints
      - "budget": random but controlled number of samples

    Returns list of {"agents": int, "parcels": int}
    """

    capacity = station_w * station_h

    MIN_AGENTS = 2
    MAX_AGENTS = 20
    MIN_PARCELS = 4
    MAX_PARCELS = capacity

    sweeps = []

    # ---------------------------------
    # FULL GRID (very expensive)
    # ---------------------------------
    if mode == "full":
        for a in range(MIN_AGENTS, MAX_AGENTS + 1):
            for p in range(MIN_PARCELS, MAX_PARCELS + 1):
                sweeps.append({"agents": a, "parcels": p})

    # ---------------------------------
    # RESEARCH MODE (default, sparse but meaningful)
    # ---------------------------------
    elif mode == "research":
        sweeps = []

        # Capacity boundary
        capacity_levels = [4, capacity // 3, 2 * capacity // 3, capacity]

        agent_levels = [2, 4, 6, 8, 10, 12, 15, 20]

        # A. Fixed parcels, vary agents
        for p in capacity_levels:
            for a in agent_levels:
                sweeps.append({"agents": a, "parcels": p})

        # B. Fixed agents, vary parcels
        fixed_agents = [2, 6, 10, 15, 20]
        parcel_levels = [4, 8, 14, 20, 26, 30, capacity]

        for a in fixed_agents:
            for p in parcel_levels:
                sweeps.append({"agents": a, "parcels": p})

        # C. Balanced diagonal
        balanced = [
            (2, 6),
            (4, 10),
            (6, 14),
            (8, 18),
            (10, 22),
            (12, 26),
            (15, 30),
            (20, capacity),
        ]

        sweeps.extend({"agents": a, "parcels": p} for a, p in balanced)

    # ---------------------------------
    # BUDGET MODE (random but bounded)
    # ---------------------------------
    elif mode == "budget":
        rng = random.Random(seed)

        for _ in range(budget):
            a = rng.randint(MIN_AGENTS, MAX_AGENTS)
            p = rng.randint(MIN_PARCELS, MAX_PARCELS)
            sweeps.append({"agents": a, "parcels": p})

    else:
        raise ValueError(f"Unknown sweep mode: {mode}")

    # ---------------------------------
    # Cleanup and normalize
    # ---------------------------------
    filtered = []
    seen = set()

    for cfg in sweeps:
        a = max(MIN_AGENTS, min(MAX_AGENTS, cfg["agents"]))
        p = max(MIN_PARCELS, min(MAX_PARCELS, cfg["parcels"]))

        key = (a, p)
        if key not in seen:
            seen.add(key)
            filtered.append({"agents": a, "parcels": p})

    filtered.sort(key=lambda x: (x["parcels"], x["agents"]))
    return filtered


# ---------------------------------
# Sweep configuration used by runner
# ---------------------------------
"""
SCALING_CONFIG = {
    "name": "station_scaling",
    "num_instances": 10,

    # Choose mode here:
    # "research" gives around 25 to 40 configs
    # "budget" gives exactly budget configs
    # "full" is exhaustive and massive
    "mode": "research",

    # Used only when mode="budget"
    "budget": 40,

    # Generated dynamically by runner
    "sweeps": None,
}
"""
SCALING_CONFIG = {
    "name": "station_scaling",
    "num_instances": 1,  # 5 seeds per config is statistically solid
    "mode": "research",  # curated coverage
    "budget": 40,  # ignored in research mode
    "sweeps": [
        {"agents": 3, "parcels": 36},
        {"agents": 6, "parcels": 36},
        {"agents": 12, "parcels": 36},
        {"agents": 18, "parcels": 36},
    ],
    # "sweeps": None,
}

