from typing import Tuple
from artifacts import Drone
from .base import PhysicsRules


class Grid2DPhysics(PhysicsRules):
    """
    Deterministic physics rules for discrete 2D grid environments.

    World assumptions and constraints:
    - Space is a uniform 2D grid with axis-aligned movement only (no diagonals)
    - Distance is measured using Manhattan (L1) distance: |x1 - x2| + |y1 - y2|
    - This corresponds to the minimum number of axis-aligned grid steps between points
    - Energy cost scales linearly with distance and payload weight
    - No inertia, momentum, or vertical motion is modeled
    - Obstacles are handled elsewhere (not encoded in this physics model)

    This model is intentionally simple and interpretable to serve as a consistent
    baseline physics layer for comparing coordination strategies fairly.

    IMPORTANT NOTE:
    - This is NOT Euclidean distance
    - This is NOT suitable for continuous or 3D environments
    - This class encodes *world constraints*, not agent behavior
    - Tight coupling with the Drone energy model ensures consistency between
      planning-time estimation and execution-time consumption
    """

    @staticmethod
    def manhattan_distance(
            a: Tuple[int, int],
            b: Tuple[int, int],
    ) -> int:
        """
        Compute Manhattan (L1) distance on a 2D grid.

        Defined as:
            |x1 - x2| + |y1 - y2|

        This corresponds to the minimum number of axis-aligned
        grid steps required to move from a to b.

        Parameters
        ----------
        a, b : Tuple[int, int]
            Grid cell coordinates.

        Returns
        -------
        int
            Number of grid cells traversed.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def distance(
            self,
            a: Tuple[int, int],
            b: Tuple[int, int],
    ) -> float:
        """
        Alias for Manhattan distance.

        Provided to satisfy the PhysicsRules interface.
        """
        return float(self.manhattan_distance(a, b))

    def movement_cost(
            self,
            distance: float,
            payload_weight: float,
    ) -> float:
        """
        Estimate energy cost for moving across grid cells.

        Mirrors the Drone energy model to ensure consistency between
        planning-time estimation and execution-time consumption.

        Energy model:
        - Linear in distance traveled
        - Payload weight increases cost multiplicatively

        Parameters
        ----------
        distance : float
            Manhattan distance in grid cells.
        payload_weight : float
            Payload weight.

        Returns
        -------
        float
            Estimated energy cost.
        """
        return (
                distance
                * Drone.BASE_COST_PER_CELL
                * (1.0 + payload_weight * Drone.WEIGHT_FACTOR)
        )

    def energy_cost(
            self,
            distance: float,
            weight: float,
    ) -> float:
        """
        Alias for movement_cost with consistent parameter naming.

        Provided for backward compatibility and interface flexibility.
        """
        return self.movement_cost(distance, weight)