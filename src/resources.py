# src/resources.py


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


class NetworkResource(AgentResource):
    """Placeholder for future network/signal modeling."""

    def __init__(self, capacity: float = 100.0):
        super().__init__(capacity)
