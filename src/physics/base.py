# src/physics/base.py
from abc import ABC, abstractmethod
from typing import Tuple


class PhysicsRules(ABC):
    """
    Abstract physics contract.

    All strategies and controllers interact with the world
    exclusively through this interface.
    """

    @abstractmethod
    def distance(self, a, b) -> float:
        """
        Compute distance between two positions under this physics model.
        """
        raise NotImplementedError

    @abstractmethod
    def movement_cost(self, distance: float, payload_weight: float) -> float:
        """
        Compute energy or effort cost for moving a given distance
        while carrying a payload.
        """
        raise NotImplementedError
