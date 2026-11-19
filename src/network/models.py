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
            reliable_range=None,
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
        # Distance dependence: within `reliable_range` (cells) the link has its base reliability; beyond it the
        # reliability falls linearly to zero at `range_cells`. None: constant reliability up to the range (old model).
        self.reliable_range = reliable_range

        # Tracks how many messages have been sent in the current step
        self.sent_this_step = 0
        # Revision stress settings (set from the radio profile, the same for every link and strategy):
        #   outage   (period, length): the radio is dead for `length` steps in every `period` (all links together).
        #   retries  attempts per message (acknowledged repeat): each attempt costs transmit energy; after a failed one the sender waits
        #            twice the longest delay before the next. The acknowledgement itself is assumed never lost, which favours the baselines.
        self.outage = None
        self.retries = 1
        self.step = 0

    def reset_step(self):
        """
        Reset per-step bandwidth usage.

        Must be called exactly once per simulation step.
        """
        self.sent_this_step = 0
        self.step += 1

    def in_outage(self, extra_steps: int = 0) -> bool:
        if not self.outage:
            return False
        period, length = self.outage
        return ((self.step + extra_steps) % period) < length

    def can_transmit(self, sender, receiver, extra_steps: int = 0):
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

        if self.in_outage(extra_steps):
            return False, "outage"

        # Range constraint using Manhattan distance
        dist = abs(sender.col - receiver.col) + abs(sender.row - receiver.row)
        if dist > self.range_cells:
            return False, "range"

        # Probabilistic packet loss, worse toward the edge of the range
        if random.random() > self.reliability_at(dist):
            return False, "loss"

        return True, "ok"

    def reliability_at(self, distance: float) -> float:
        """Probability that a transmission over `distance` cells is received."""
        r0, r1 = self.reliable_range, self.range_cells
        if r0 is None or distance <= r0 or r1 <= r0:
            return self.reliability
        return self.reliability * max(0.0, 1.0 - (distance - r0) / (r1 - r0))

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
    - A fixed energy cost per transmission (distance term available, set to 0)
    """

    # Calibrated to real radio hardware (SiK telemetry radio, 32-byte message at 57,600 baud: about 3 mJ to send); one
    # energy unit = 3.6 kJ (100 units = a 100 Wh battery). Sources and calculation: .ignore/revision/simulated_hardware.md.
    TX_ENERGY_UNITS = 8.3e-7

    def __init__(self, base_cost: float = TX_ENERGY_UNITS, distance_factor: float = 0.0):
        """
        Parameters
        ----------
        base_cost : float
            Energy cost per transmission, in energy units.
        distance_factor : float
            Additional energy cost per grid cell of distance (0: the modelled radios send at fixed power).
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
