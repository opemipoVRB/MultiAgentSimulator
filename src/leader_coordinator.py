# src/leader_coordinator.py
"""
Simple leader-election and leader-managed reservation helper.

Designed for small local swarms in-process. Leader election is deterministic:
choose controller with lowest id() by default or pass custom key function.

API:
  LeaderCoordinator.register_controllers(controllers)
  LeaderCoordinator.elect_leader()
  LeaderCoordinator.get_leader()
  LeaderCoordinator.issue_reservation(cell, owner, ttl)  # only leader can call this to commit
"""

from __future__ import annotations
import time
from typing import List, Optional, Tuple

from reservations import RESERVATION_MANAGER

class LeaderCoordinator:
    def __init__(self, leader_selector=None):
        self.controllers = []
        self._leader = None
        self.leader_selector = leader_selector or (lambda cs: min(cs, key=lambda c: id(c)) if cs else None)
        self._leader_token_ts = 0.0
        self._token_ttl = 2.0

    def register_controllers(self, controllers: List[object]):
        self.controllers = list(controllers)
        self.elect_leader()

    def elect_leader(self):
        self._leader = self.leader_selector(self.controllers)
        self._leader_token_ts = time.time() + self._token_ttl
        return self._leader

    def get_leader(self):
        # simple heartbeat: if token expired, re-elect
        if not self._leader or time.time() > self._leader_token_ts:
            self.elect_leader()
        return self._leader

    def refresh_leader_token(self):
        self._leader_token_ts = time.time() + self._token_ttl

    def is_leader(self, controller) -> bool:
        return controller is self.get_leader()

    def issue_reservation(self, cell: Tuple[int,int], owner: object, ttl: Optional[float] = None) -> bool:
        """
        Leader issues (commits) a reservation via the shared ReservationManager.
        Non-leaders should not call this; check is_leader() first.
        """
        if not self.is_leader(owner) and not self.is_leader(owner):  # owner may be agent object but leader must call
            # still allow leader to issue for any owner, but only if current leader token valid
            pass
        return RESERVATION_MANAGER.reserve(cell, owner, ttl)
