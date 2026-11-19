# src/onboard.py
"""
What a strategy may use of its own drone: the onboard services.

Strategies used to receive the world's Drone object and reached through it into the world (the terrain, other
drones). They receive this object instead. It is built from what the drone has been told, never from the world:
  power       own battery level, capacity, hardware floor and usable energy, as of the latest observation
  inbox       radio messages delivered to this drone (filled from each observation; the strategy empties it)
  send, broadcast   queue a message for the radio; the world carries it after the drone has finished thinking
  fleet_ids, declared_spec, energy_rate   the fleet and each drone's declared spec, from the start information
  slm_*       the drone's onboard language model client
  personality, physics   the drone's own configuration
Anything else raises AttributeError, so a strategy cannot reach into the world by accident.
"""
from typing import Any, List, Optional, Tuple


class BatteryView:
    """Read-only view of the drone's own battery (its own sensor), as of the latest observation."""

    def __init__(self, services):
        self._services = services

    @property
    def level(self) -> float:
        return self._services._obs.battery

    @property
    def capacity(self) -> float:
        return self._services._obs.capacity

    @property
    def floor(self) -> float:
        """Hardware reserve floor of this drone (energy units)."""
        return self._services._obs.reserve_floor

    @property
    def usable(self) -> float:
        """Energy above the hardware floor."""
        return self.level - self.floor

    def percent(self) -> float:
        cap = self.capacity
        return 0.0 if cap <= 0 else max(0.0, min(100.0, (self.level / cap) * 100.0))


class OnboardServices:
    def __init__(self, drone, run_seed: int = 0, start_info=None):
        self._drone = drone                       # kept only to forward the onboard model client and configuration
        self.agent_id = drone.agent_id
        self.run_seed = int(run_seed)
        self.power = BatteryView(self)
        self.inbox: List[Any] = []
        self._outbox: List[Tuple[str, Any]] = []
        self._obs = None
        self._start_info = start_info              # what the whole fleet knows before the first step

    # -- fed by the controller on every step -------------------------------------------------------------------
    def take_in(self, obs) -> None:
        self._obs = obs
        if obs.start_info is not None:
            self._start_info = obs.start_info
        self.inbox.extend(obs.inbox)

    def take_outbox(self) -> List[Tuple[str, Any]]:
        out, self._outbox = self._outbox, []
        return out

    # -- radio ------------------------------------------------------------------------------------------------
    def fleet_ids(self) -> List[str]:
        return list(self._start_info.fleet) if self._start_info is not None else []

    def send(self, recipient_id: str, payload, step: int = 0) -> bool:
        self._outbox.append((recipient_id, payload))
        return True

    def broadcast(self, payload, step: int = 0) -> None:
        """Send `payload` to every other drone of the fleet as one radio transmission."""
        self._outbox.append(("*", payload))

    # -- own configuration --------------------------------------------------------------------------------------
    def declared_spec(self, agent_id: Optional[str] = None):
        """A drone's declared spec from the start information (the drone's own if no id is given)."""
        if self._start_info is None:
            return None
        return self._start_info.fleet.get(agent_id or self.agent_id)

    def _declared_rate(self, agent_id: str) -> float:
        spec = self.declared_spec(agent_id)
        if spec is None:
            return 1.0
        return round(spec.unloaded_per_cell / type(self._drone).BASE_COST_PER_CELL, 9)

    @property
    def energy_rate(self) -> float:
        """Own declared unloaded consumption relative to the nominal model."""
        return self._declared_rate(self.agent_id)

    @property
    def slm_model(self):
        return getattr(self._drone, "slm_model", None)

    @slm_model.setter
    def slm_model(self, model):
        """A strategy's choice of language model (an arm setting) reaches the drone's model client."""
        self._drone.slm_model = model

    @property
    def personality(self):
        return getattr(self._drone, "personality", None)

    @property
    def physics(self):
        return getattr(self._drone, "physics")

    def __getattr__(self, name):
        # onboard language model client only; everything else is not available to a strategy
        if name in ("slm_enabled", "slm_base_url", "slm_generate", "slm_last_usage"):
            return getattr(self._drone, name)
        raise AttributeError(f"{name!r} is not available to a strategy (onboard services only)")
