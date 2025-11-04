# src/reservations.py
"""
In-process reservation manager that provides atomic-style operations suitable
for single-process simulation and a clear API to adapt later (Redis, DB).
"""

from __future__ import annotations
import time
from typing import Dict, Tuple, Optional


class ReservationManager:
    """
    Manages reservations as a mapping cell -> (owner, expiry_ts).
    API:
      reserve(cell, owner_id, ttl) -> bool
      refresh(cell, owner_id, ttl) -> bool
      release(cell, owner_id) -> bool
      owner_of(cell) -> Optional[owner_id]
      list_reservations() -> Dict[cell, (owner, expiry_ts)]
      cleanup() -> None
    """

    def __init__(self):
        self._reservations: Dict[Tuple[int, int], Tuple[object, float]] = {}
        self._default_ttl = 8.0

    def reserve(self, cell: Tuple[int, int], owner: object, ttl: Optional[float] = None) -> bool:
        now = time.time()
        ttl = float(ttl) if ttl is not None else self._default_ttl
        existing = self._reservations.get(cell)
        if existing:
            ex_owner, ex_ts = existing
            if ex_ts > now and ex_owner is not owner:
                return False
        # set (owner, expiry)
        self._reservations[cell] = (owner, now + ttl)
        return True

    def refresh(self, cell: Tuple[int, int], owner: object, ttl: Optional[float] = None) -> bool:
        now = time.time()
        ttl = float(ttl) if ttl is not None else self._default_ttl
        existing = self._reservations.get(cell)
        if not existing:
            return False
        ex_owner, ex_ts = existing
        if ex_owner is not owner:
            return False
        self._reservations[cell] = (owner, now + ttl)
        return True

    def release(self, cell: Tuple[int, int], owner: object) -> bool:
        existing = self._reservations.get(cell)
        if not existing:
            return False
        ex_owner, _ = existing
        if ex_owner is not owner:
            return False
        self._reservations.pop(cell, None)
        return True

    def owner_of(self, cell: Tuple[int, int]) -> Optional[object]:
        self.cleanup()
        v = self._reservations.get(cell)
        return None if not v else v[0]

    def is_reserved(self, cell: Tuple[int, int]) -> bool:
        self.cleanup()
        return cell in self._reservations

    def list_reservations(self) -> Dict[Tuple[int, int], Tuple[object, float]]:
        self.cleanup()
        return dict(self._reservations)

    def cleanup(self) -> None:
        now = time.time()
        expired = [k for k, (_, ts) in self._reservations.items() if ts <= now]
        for k in expired:
            self._reservations.pop(k, None)


# Singleton instance
RESERVATION_MANAGER = ReservationManager()


# ===== Convenience legacy dict view for compatibility =====
# Many existing parts of your code look up a module-level RESERVATIONS dict.
# Provide a best-effort live view presenting the same structure: (owner, ts).
# It is read-only for external code but kept for backward compatibility.
def _legacy_dict_view():
    out = {}
    for k, (owner, ts) in RESERVATION_MANAGER.list_reservations().items():
        out[k] = (owner, ts)
    return out


# Expose alias (only snapshot; not live mutable dict). Use API above for atomic ops.
RESERVATIONS_SNAPSHOT = _legacy_dict_view
