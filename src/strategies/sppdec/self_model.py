# src/strategies/sppdec/self_model.py
"""
The drone's model of its own energy use, learned while it flies.

The physics charges a move of c cells carrying weight w:    energy = c * (a + b * w)
  a  energy per cell when empty      ("how does distance affect me")
  b  extra energy per cell per unit of weight   ("how does weight affect me")
The drone starts from the nominal values (or the true ones, in the reference arm) and corrects
a and b from what its battery actually does, one sample per step of motion, with a recursive
least-squares filter that forgets old samples slowly. From that it can answer:
  cost(d, w)       energy for d cells carrying w
  sensitivities()  d cost / d distance at a given weight, and d cost / d weight at a given distance
  std(d, w)        how unsure it still is about that cost (from the filter's covariance)
  trend()          how fast the estimates are moving, i.e. whether its own consumption is changing
and a change detector (CUSUM on the standardised prediction error) re-opens the filter when its
consumption shifts, for example after damage or a change of payload handling.

Pure arithmetic, no world access.
"""
from collections import deque
from math import sqrt
from typing import Deque, Dict, Tuple


class SelfEnergyModel:
    FORGETTING = 0.995         # weight kept per sample; ~200 samples of memory
    PRIOR_VARIANCE = 0.25      # relative uncertainty of the starting values (as a fraction squared)
    NOISE_FLOOR = 1e-6         # smallest residual variance assumed (the physics is nearly deterministic)
    CUSUM_SLACK = 0.5          # standardised error tolerated before it accumulates
    CUSUM_ALARM = 8.0          # accumulated standardised error that counts as a change
    TREND_WINDOW = 60          # samples between the estimates compared by trend()

    def __init__(self, base_per_cell: float, weight_per_cell: float):
        self.theta = [float(base_per_cell), float(weight_per_cell)]
        self.nominal = list(self.theta)
        # covariance of the estimates, in absolute units
        self.P = [[self.PRIOR_VARIANCE * self.theta[0] ** 2, 0.0],
                  [0.0, self.PRIOR_VARIANCE * self.theta[1] ** 2]]
        self.noise = self.NOISE_FLOOR
        self.samples = 0
        self.alarms = 0
        self._cusum = 0.0
        self._history: Deque[Tuple[float, float]] = deque(maxlen=self.TREND_WINDOW)

    # -- learning -------------------------------------------------------------------------
    def update(self, cells: float, weight: float, energy: float) -> None:
        """One step of motion: `cells` moved while carrying `weight` cost `energy`."""
        if cells <= 1e-9 or energy < 0:
            return
        x = [cells, cells * weight]
        Px = [self.P[0][0] * x[0] + self.P[0][1] * x[1], self.P[1][0] * x[0] + self.P[1][1] * x[1]]
        denom = self.FORGETTING * self.noise + x[0] * Px[0] + x[1] * Px[1]
        resid = energy - (self.theta[0] * x[0] + self.theta[1] * x[1])
        # change detection on the standardised error
        z = resid / sqrt(max(denom, 1e-18))
        self._cusum = max(0.0, self._cusum + abs(z) - self.CUSUM_SLACK)
        if self._cusum > self.CUSUM_ALARM and self.samples > 5:
            self.alarms += 1
            self._cusum = 0.0
            for i in range(2):  # forget what was learned: open the filter again
                self.P[i][i] = max(self.P[i][i], self.PRIOR_VARIANCE * self.nominal[i] ** 2)
        gain = [Px[0] / denom, Px[1] / denom]
        self.theta = [self.theta[0] + gain[0] * resid, self.theta[1] + gain[1] * resid]
        xP = [x[0] * self.P[0][0] + x[1] * self.P[1][0], x[0] * self.P[0][1] + x[1] * self.P[1][1]]
        self.P = [[(self.P[i][j] - gain[i] * xP[j]) / self.FORGETTING for j in range(2)] for i in range(2)]
        self.samples += 1
        self._history.append((self.theta[0], self.theta[1]))

    # -- questions the drone asks of itself -----------------------------------------------
    def cost(self, distance: float, weight: float) -> float:
        return max(0.0, distance * (self.theta[0] + self.theta[1] * weight))

    def sensitivities(self, distance: float, weight: float) -> Dict[str, float]:
        """Energy per extra cell at this weight, and per extra unit of weight over this distance."""
        return {"per_cell": self.theta[0] + self.theta[1] * weight, "per_weight": self.theta[1] * distance}

    def std(self, distance: float, weight: float) -> float:
        """Standard deviation of cost(distance, weight) implied by the filter's covariance."""
        x = [distance, distance * weight]
        var = (x[0] * (self.P[0][0] * x[0] + self.P[0][1] * x[1])
               + x[1] * (self.P[1][0] * x[0] + self.P[1][1] * x[1]))
        return sqrt(max(var, 0.0))

    def trend(self) -> Dict[str, float]:
        """Relative change of the two parameters over the last window of samples (0 when steady)."""
        if len(self._history) < self._history.maxlen:
            return {"per_cell": 0.0, "per_weight": 0.0}
        old, new = self._history[0], self._history[-1]
        return {"per_cell": (new[0] - old[0]) / max(abs(old[0]), 1e-9),
                "per_weight": (new[1] - old[1]) / max(abs(old[1]), 1e-9)}

    def snapshot(self) -> Dict:
        return {"per_cell": round(self.theta[0], 4), "per_weight": round(self.theta[1], 4),
                "samples": self.samples, "alarms": self.alarms}
