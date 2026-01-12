"""
Physics Layer
=============

This module defines deterministic, environment-level rules that govern
movement, distance, energy consumption, and feasibility.

Design principles:
- Physics is strategy-agnostic.
- Physics is deterministic and side-effect free.
- Physics encodes *world laws*, not agent intent.
- Strategies may query physics, but never override it.

By isolating physics from decision-making, the system supports:
- Multiple environments (2D grid, 3D space, real-world robotics)
- Fair strategy comparison under identical physical constraints
- Verifiable safety and cost estimation

SPPDEC relies on physics as its tooling layer for feasibility checks,
conflict prediction, and energy validation.
"""

# and

"""
Physics layer for the Agentic Swarm Lab.

This package defines *environmental rules*, not strategies.

Physics modules answer questions like:
- How is distance computed?
- What does movement cost?
- What constraints are unbreakable?

Design principles:
- Physics is deterministic
- Physics is strategy-agnostic
- Physics is replaceable

Strategies (e.g. SPPDEC) MUST NOT assume a specific world geometry.
They query physics instead.

This allows the same coordination logic to operate across:
- 2D grid worlds
- Continuous 2D environments
- 3D airspace
- Simulated or real robotic platforms
"""
