# src/strategies/utils.py
from typing import Tuple
from artifacts import Drone


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def est_cost_cells(dist_cells: float, weight: float) -> float:
    """
    Battery cost estimate that mirrors Drone energy model.
    """
    return (
        dist_cells
        * Drone.BASE_COST_PER_CELL
        * (1.0 + weight * Drone.WEIGHT_FACTOR)
    )
