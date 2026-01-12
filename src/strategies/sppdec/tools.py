# src/strategies/sppdec/tools.py


def estimate_task_cost(snapshot, task, physics) -> float:
    """
    Estimate the cost of executing a task under the active physics model.

    This function is physics-agnostic and delegates all world-specific
    assumptions (distance metric, energy scaling, payload effects)
    to the injected PhysicsRules instance.
    """
    agent = snapshot["agent"]
    pickup = task["pickup"]

    distance = physics.distance(
        (agent["col"], agent["row"]),
        pickup,
    )

    weight = task.get("weight", 1.0)
    return physics.energy_cost(distance, weight)
