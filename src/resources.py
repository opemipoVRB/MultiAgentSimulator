# src/resources.py

import random
from collections import deque
from src.network.models import NetworkChannel, RadioEnergyModel


# -------------------------
# Resource classes
# -------------------------

class AgentResource:
    """Base resource class for agents (Power, Network, ...)."""

    def __init__(self, capacity: float):
        self.capacity = float(capacity)
        self.level = float(capacity)

    def percent(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.level / self.capacity) * 100.0))

    def is_depleted(self) -> bool:
        return self.level <= 0.0

    def recharge(self, amount: float):
        self.level = min(self.capacity, self.level + float(amount))

    def consume(self, amount: float):
        """Consume amount (not percent). Negative values will recharge."""
        self.level -= float(amount)
        if self.level < 0.0:
            self.level = 0.0


class PowerResource(AgentResource):
    """Power resource measured in 'energy units'. Default capacity = 100."""

    def __init__(self, capacity: float = 100.0):
        super().__init__(capacity)


class NetworkResource:
    """
    Concrete communication system used by agents.

    This class is the bridge between:
    - Environmental constraints (NetworkChannel)
    - Physical energy usage (RadioEnergyModel)
    - Message transport (outbox queue)
    - Real battery drain (PowerResource)

    This is where communication becomes physically real.
    """

    def __init__(
            self,
            capacity: int = 2.0,
            range_cells: int = 6,
            reliability: float = 0.9,
            delay_range=(1, 3),
            flight_recorder=None,
            channel: NetworkChannel = None,
            energy_model: RadioEnergyModel = None,
    ):
        # Use injected models or defaults
        self.channel = channel or NetworkChannel(
            bandwidth=capacity,
            range_cells=range_cells,
            reliability=reliability,
            delay_range=delay_range,
        )

        self.energy_model = energy_model or RadioEnergyModel()

        self.flight_recorder = flight_recorder

        # Messages waiting to be delivered
        # Each entry: (deliver_step, sender, receiver, payload)
        self.outbox = deque()

    # ------------------------------------------------------------
    # Simulation step lifecycle
    # ------------------------------------------------------------
    def reset_step(self):
        """Must be called once per simulation step."""
        self.channel.reset_step()

    @property
    def active(self):
        return bool(self.outbox)

    # ------------------------------------------------------------
    # Logging hook
    # ------------------------------------------------------------
    def _log(self, event: str, **data):
        if not self.flight_recorder:
            return
        try:
            self.flight_recorder.record_message({"type": event, **data})
        except Exception:
            pass

    # ------------------------------------------------------------
    # Transmission
    # ------------------------------------------------------------
    def send(self, sender, receiver, payload, current_step: int):
        """
        Attempt to transmit a message physically.

        This enforces:
        - Channel constraints
        - Energy availability
        - Delay
        - Logging
        """

        sender_id = getattr(sender, "agent_id", "command_center")
        receiver_id = getattr(receiver, "agent_id", None)

        # 1. Environmental feasibility
        allowed, reason = self.channel.can_transmit(sender, receiver)
        if not allowed:
            self._log("send_failed", sender=sender_id, receiver=receiver_id, reason=reason)
            return False, reason

        # 2. Compute energy cost
        energy_cost = self.energy_model.cost(sender, receiver)

        # 3. Drain sender battery if it has one
        sender_power = getattr(sender, "power", None)

        if sender_power is not None:
            if sender_power.level < energy_cost:
                self._log("send_failed", sender=sender_id, receiver=receiver_id, reason="depleted")
                return False, "depleted"

            sender_power.consume(energy_cost)

        # 4. Transmission succeeds physically
        self.channel.register_send()
        delay = self.channel.sample_delay()
        deliver_at = current_step + delay

        self.outbox.append((deliver_at, sender, receiver, payload))
        # Count real transmissions for metrics
        if self.flight_recorder and self.flight_recorder.metrics:
            self.flight_recorder.metrics.messages_sent += 1

        self._log(
            "send_enqueued",
            sender=sender_id,
            receiver=receiver_id,
            cost=energy_cost,
            delay=delay,
            step=current_step,
        )

        return True, "ok"

    # ------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------
    def deliver(self, current_step: int):
        """
        Deliver messages whose delay has elapsed.
        """
        delivered = []

        while self.outbox and self.outbox[0][0] <= current_step:
            _, sender, receiver, payload = self.outbox.popleft()
            delivered.append((receiver, payload))

            self._log(
                "delivered",
                sender=getattr(sender, "agent_id", None),
                receiver=getattr(receiver, "agent_id", None),
                step=current_step,
            )

        return delivered
