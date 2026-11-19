# src/energy.py
"""
The energy model a drone's strategy plans with: the drone's declared spec, measured in the pre-mission calibration.

Every strategy prices trips with this model, not with the nominal constants of the physics. For a drone whose declared
spec equals the nominal model it reproduces the physics exactly; for a drone that consumes more or less, or is more
sensitive to payload, it prices its trips accordingly. All other physics (distance and the rest) is delegated.
"""


class DeclaredPhysics:
    def __init__(self, physics, unloaded_per_cell: float, weight_sensitivity: float):
        self._physics = physics
        self.unloaded_per_cell = float(unloaded_per_cell)
        self.weight_sensitivity = float(weight_sensitivity)

    def movement_cost(self, distance: float, payload_weight: float) -> float:
        return distance * self.unloaded_per_cell * (1.0 + payload_weight * self.weight_sensitivity)

    def energy_cost(self, distance: float, weight: float) -> float:
        return self.movement_cost(distance, weight)

    def pick_drop_cost(self, weight: float) -> float:
        """Energy of one pick or one drop for this drone: the nominal action cost scaled by its declared consumption."""
        nominal_per_cell = self._physics.movement_cost(1.0, 0.0)
        rate = round(self.unloaded_per_cell / nominal_per_cell, 9)
        nominal_pd = self._physics.pick_drop_cost(0.0)
        return rate * nominal_pd * (1.0 + weight * self.weight_sensitivity)

    def __getattr__(self, name):
        return getattr(self._physics, name)
