# src/network/models.py
import random


class NetworkChannel:
    """
    Models the physical communication channel between agents.

    This class represents the environmental constraints of wireless communication,
    independent of energy, agents, or strategies.

    Concepts modeled:
    - Bandwidth: maximum messages per simulation timestep
    - Range: Manhattan distance constraint in grid cells
    - Reliability: probabilistic packet loss
    - Latency: stochastic delivery delay

    This class does NOT:
    - Know about energy
    - Know about strategies
    - Store messages
    - Interact with queues
    - Modify agents

    It only answers one question:
    "Is this transmission physically possible in the environment?"
    """

    def __init__(
            self,
            bandwidth: int = 2,
            range_cells: int = 6,
            reliability: float = 0.9,
            delay_range=(1, 3),
    ):
        """
        Parameters
        ----------
        bandwidth : int
            Maximum number of messages that can be transmitted per simulation step.
        range_cells : int
            Maximum Manhattan distance allowed between sender and receiver.
        reliability : float
            Probability that a transmission succeeds.
        delay_range : tuple[int, int]
            Inclusive range of delivery delays in simulation steps.
        """
        self.bandwidth = int(bandwidth)
        self.range_cells = int(range_cells)
        self.reliability = float(reliability)
        self.delay_range = delay_range

        # Tracks how many messages have been sent in the current step
        self.sent_this_step = 0

    def reset_step(self):
        """
        Reset per-step bandwidth usage.

        Must be called exactly once per simulation step.
        """
        self.sent_this_step = 0

    def can_transmit(self, sender, receiver):
        """
        Determine whether the environment permits this transmission.

        Evaluates only environmental constraints:
        - bandwidth availability
        - spatial range
        - probabilistic loss

        Parameters
        ----------
        sender : object
            Any object with col and row attributes.
        receiver : object
            Any object with col and row attributes.

        Returns
        -------
        (bool, str)
            allowed : whether transmission is physically possible
            reason : one of {"ok", "capacity", "range", "loss"}
        """

        # Bandwidth constraint
        if self.sent_this_step >= self.bandwidth:
            return False, "capacity"

        # Range constraint using Manhattan distance
        dist = abs(sender.col - receiver.col) + abs(sender.row - receiver.row)
        if dist > self.range_cells:
            return False, "range"

        # Probabilistic packet loss
        if random.random() > self.reliability:
            return False, "loss"

        return True, "ok"

    def register_send(self):
        """
        Register that a transmission has occurred.

        Must be called exactly once after a successful send.
        """
        self.sent_this_step += 1

    def sample_delay(self) -> int:
        """
        Sample a stochastic delivery delay.

        Returns
        -------
        int
            Number of simulation steps before delivery.
        """
        return random.randint(*self.delay_range)


class RadioEnergyModel:
    """
    Models energy consumption of radio transmission.

    This class is intentionally simple and interpretable.

    It computes energy cost only. It does NOT:
    - Drain batteries
    - Modify sender state
    - Know about queues or delivery
    - Interact with strategies

    You can later extend this model to include:
    - Distance based power loss curves
    - Payload size effects
    - Signal strength adaptation
    - Retransmission costs
    - Noise and interference modeling

    Currently:
    - Energy cost scales linearly with Manhattan distance
    """

    def __init__(self, base_cost: float = 0.05, distance_factor: float = 0.005):
        """
        Parameters
        ----------
        base_cost : float
            Minimum energy cost per transmission.
        distance_factor : float
            Additional energy cost per grid cell of distance.
        """
        self.base_cost = float(base_cost)
        self.distance_factor = float(distance_factor)

    def cost(self, sender, receiver) -> float:
        """
        Compute energy cost of transmitting from sender to receiver.

        Uses Manhattan distance to remain consistent with grid physics.

        Parameters
        ----------
        sender : object
            Must expose col and row attributes.
        receiver : object
            Must expose col and row attributes.

        Returns
        -------
        float
            Energy units required to perform this transmission.
        """
        dist = abs(sender.col - receiver.col) + abs(sender.row - receiver.row)
        return self.base_cost + dist * self.distance_factor
