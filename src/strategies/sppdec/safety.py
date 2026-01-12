# src/strategies/sppdec/safety.py

def estimate_energy_required(
        distance_to_target: float,
        distance_to_base: float,
        consumption_per_cell: float,
) -> float:
    return (distance_to_target + distance_to_base) * consumption_per_cell


def is_action_safe(
        battery_level: float,
        required_energy: float,
        safety_margin: float = 0.20,
) -> bool:
    return battery_level >= required_energy * (1.0 + safety_margin)
