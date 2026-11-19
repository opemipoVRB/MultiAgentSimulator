# src/resources.py

import random
from collections import deque
from src.network.models import NetworkChannel, RadioEnergyModel


# Energy to receive one message, in units (SiK radio: about 0.7 mJ; 1 unit = 3.6 kJ). See .ignore/revision/simulated_hardware.md.
RADIO_RX_ENERGY = 1.9e-7


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

        # 1. Environmental feasibility (with acknowledged repeats when the profile sets retries > 1)
        energy_cost = self.energy_model.cost(sender, receiver)
        attempts = max(1, int(getattr(self.channel, "retries", 1)))
        waited = 0
        for attempt in range(attempts):
            allowed, reason = self.channel.can_transmit(sender, receiver, extra_steps=waited)
            if allowed or reason in ("capacity", "range"):
                break
            if attempts > 1 and attempt < attempts - 1:
                # a lost attempt still used the transmitter; the sender waits for the missing acknowledgement
                sp = getattr(sender, "power", None)
                if sp is not None and sp.level >= energy_cost:
                    sp.consume(energy_cost)
                if self.flight_recorder and self.flight_recorder.metrics:
                    self.flight_recorder.metrics.messages_sent += 1
                waited += 2 * int(self.channel.delay_range[1])
        if not allowed:
            self._log("send_failed", sender=sender_id, receiver=receiver_id, reason=reason)
            return False, reason

        # 2. Energy cost (computed above)

        # 3. Drain sender battery if it has one
        sender_power = getattr(sender, "power", None)

        if sender_power is not None:
            if sender_power.level < energy_cost:
                self._log("send_failed", sender=sender_id, receiver=receiver_id, reason="depleted")
                return False, "depleted"

            sender_power.consume(energy_cost)

        # 4. Transmission succeeds physically
        self.channel.register_send()
        delay = self.channel.sample_delay() + waited
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

    def send_broadcast(self, sender, receivers, payload, current_step: int):
        """
        One transmission heard by every receiver in range: one use of the channel and one energy cost (for the farthest
        receiver in range); each receiver gets or misses it independently, and all share one delay.
        """
        sender_id = getattr(sender, "agent_id", "command_center")
        if self.channel.sent_this_step >= self.channel.bandwidth:
            self._log("send_failed", sender=sender_id, receiver="*", reason="capacity")
            return False, "capacity"
        in_range = [r for r in receivers
                    if abs(sender.col - r.col) + abs(sender.row - r.row) <= self.channel.range_cells]
        if not in_range:
            return False, "range"
        retries = max(1, int(getattr(self.channel, "retries", 1)))
        if retries == 1 and self.channel.in_outage():
            self._log("send_failed", sender=sender_id, receiver="*", reason="outage")
            return False, "outage"
        energy_cost = max(self.energy_model.cost(sender, r) for r in in_range)
        sender_power = getattr(sender, "power", None)
        if sender_power is not None:
            if sender_power.level < energy_cost:
                self._log("send_failed", sender=sender_id, receiver="*", reason="depleted")
                return False, "depleted"
            sender_power.consume(energy_cost)
        self.channel.register_send()
        delay = self.channel.sample_delay()
        # each receiver hears the message at the first attempt that gets through (acknowledged repeat, up to `retries` attempts, a wait of twice the longest delay between attempts);
        # every extra attempt the sender makes costs transmit energy and counts as a message
        heard, attempts_used = [], 1
        for r in in_range:
            p = self.channel.reliability_at(abs(sender.col - r.col) + abs(sender.row - r.row))
            waited = 0
            for a in range(retries):
                if not self.channel.in_outage(waited) and random.random() <= p:
                    heard.append((r, waited))
                    attempts_used = max(attempts_used, a + 1)
                    break
                waited += 2 * int(self.channel.delay_range[1])
            else:
                attempts_used = retries
        for extra in range(attempts_used - 1):
            if sender_power is not None and sender_power.level >= energy_cost:
                sender_power.consume(energy_cost)
        for r, waited in heard:
            self.outbox.append((current_step + delay + waited, sender, r, payload))
        if self.flight_recorder and self.flight_recorder.metrics:
            self.flight_recorder.metrics.messages_sent += attempts_used
        self._log("send_enqueued", sender=sender_id, receiver="*", cost=energy_cost, delay=delay, step=current_step)
        return True, "ok"

    # ------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------
    def deliver(self, current_step: int):
        """
        Deliver messages whose delay has elapsed.
        """
        delivered = []

        # every message that is due, in the order sent (a slow message does not hold up a later, faster one)
        due = [m for m in self.outbox if m[0] <= current_step]
        if due:
            self.outbox = type(self.outbox)(m for m in self.outbox if m[0] > current_step)
        for _, sender, receiver, payload in due:
            delivered.append((receiver, payload))
            rx_power = getattr(receiver, "power", None)
            if RADIO_RX_ENERGY > 0.0 and rx_power is not None:
                rx_power.consume(min(RADIO_RX_ENERGY, rx_power.level))      # receiving costs energy too

            self._log(
                "delivered",
                sender=getattr(sender, "agent_id", None),
                receiver=getattr(receiver, "agent_id", None),
                step=current_step,
            )

        return delivered
