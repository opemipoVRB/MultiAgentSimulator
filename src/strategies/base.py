# src/strategies/base.py
from __future__ import annotations
import time
from collections import deque
from typing import Any, Dict, List
import math

class BaseStrategy:
    """
    Minimal base strategy compatible with existing Coordinator.
    Subclasses must implement `compute_assignments(controllers, terrain)`.
    """

    strategy_name = None

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        self.tick_interval = float(tick_interval)
        self.max_assign_per_agent = int(max_assign_per_agent)
        self._last_tick = 0.0

    def should_run(self) -> bool:
        return (time.time() - self._last_tick) >= self.tick_interval

    def mark_ran(self):
        self._last_tick = time.time()

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        raise NotImplementedError
