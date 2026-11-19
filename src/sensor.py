# src/sensor.py
"""
World-side sensor model: what a drone's sensors report about the parcels and the other drones.

One function for every strategy, so all of them see the same limited view. Settings (from the experiment's
sensing spec): radius in cells (default unlimited), position noise (standard deviation in cells) and the chance a
peer within range is missed on a given reading. The default (unlimited, no noise, no loss) reports everything
exactly as it is.

A parcel's state and true weight are reported only within the radius; a parcel outside it is reported as it was
when the drone started (location only), and the drone's own memory (drone side) holds what it saw last.
A reading is fixed per drone and per step (seeded from both), so looking twice in the same step gives the same
measurement and the noise does not depend on how often the code looks.
"""
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# What a drone senses unless an arm states its own setting: limited sensing (radius in cells; no noise: the study is about assumptions replacing communication). Unlimited, exact sensing is an unfair
# advantage over a real drone and does not exist as a setting; an arm that names no sensing runs on this.
DEFAULT_SENSING = {"radius": 8}


class SensorModel:
    def __init__(self, distance: Callable, radius: float = float("inf"), noise_std: float = 0.0, loss: float = 0.0,
                 parcel_radius: Optional[float] = None, delay: int = 0, blackout: Optional[Tuple[int, int]] = None):
        self.distance = distance
        self.radius = float(radius)
        # The range of parcel readings; by default the same as the range of peer readings (`radius`). Separate values
        # let one information channel be limited while the other is exact.
        self.parcel_radius = self.radius if parcel_radius is None else float(parcel_radius)
        self.noise_std = float(noise_std)
        self.loss = float(loss)
        # Revision stress settings, the same for every strategy that reads peers:
        #   delay     peer readings are `delay` steps old (the drone sees where the peer was).
        #   blackout  (period, length): each drone gets no peer readings for `length` steps in every `period`, at an offset seeded from its id (intermittent sensing).
        self.delay = int(delay)
        self.blackout = tuple(blackout) if blackout else None
        self._history: Dict[int, List[Dict]] = {}

    @property
    def active(self) -> bool:
        return (self.radius != float("inf") or self.parcel_radius != float("inf")
                or self.noise_std > 0.0 or self.loss > 0.0 or self.delay > 0 or self.blackout is not None)

    def sense(self, drone_id: str, step: int, pos: Tuple[int, int], parcels: Sequence[Dict], peers: Sequence[Dict]
              ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Returns (parcels in range with their true state, peers in range with noise and dropouts applied,
        ids of the parcels in range). Parcels out of range are not returned: the drone knows their locations
        from the start information and remembers what it last saw.
        """
        rng = random.Random(f"sensing-{drone_id}-{step}")
        in_range = [dict(p) for p in parcels if self.distance(pos, (p["col"], p["row"])) <= self.parcel_radius]
        seen_peers = []
        if self.delay > 0:
            # keep the first full list of each step (every drone's call in a step sees the same world) and read the one `delay` steps back
            if step not in self._history:
                self._history[step] = [dict(a) for a in peers]
                for old in [k for k in self._history if k < step - self.delay - 2]:
                    del self._history[old]
            peers = self._history.get(step - self.delay, [])
        if self.blackout is not None:
            period, length = self.blackout
            offset = random.Random(f"blackout-{drone_id}").randrange(period)
            if (step + offset) % period < length:
                peers = []
        for a in peers:
            if a["id"] == drone_id:
                continue
            if self.distance(pos, (a["col"], a["row"])) > self.radius:
                continue
            if self.loss > 0.0 and rng.random() < self.loss:
                continue
            b = dict(a)
            if self.noise_std > 0.0:
                b["col"] = int(round(a["col"] + rng.gauss(0.0, self.noise_std)))
                b["row"] = int(round(a["row"] + rng.gauss(0.0, self.noise_std)))
            seen_peers.append(b)
        return in_range, seen_peers, [p["id"] for p in in_range]
