from enum import Enum

class AbortReason(str, Enum):
    # ---------------------------
    # Task feasibility failures
    # ---------------------------
    GLOBALLY_INFEASIBLE = "globally_infeasible"
    # No agent in the system can complete remaining tasks

    AGENT_ENERGY_INFEASIBLE = "agent_energy_infeasible"
    # This agent cannot complete any available task safely

    NO_VALID_TASKS = "no_valid_tasks"
    # Tasks exist but violate constraints (e.g. weight, geometry, rules)

    # ---------------------------
    # Coordination failures
    # ---------------------------
    DEADLOCK = "deadlock"
    # Agents mutually block progress (e.g. circular waiting, contention)

    STARVATION = "starvation"
    # Agent repeatedly receives no work while work exists

    # ---------------------------
    # Execution failures
    # ---------------------------
    STALLED = "stalled"
    # Agent stopped making progress while assigned a task

    PATH_BLOCKED = "path_blocked"
    # No valid motion path exists (obstacles, geometry)

    # ---------------------------
    # Infrastructure failures
    # ---------------------------
    COMMUNICATION_FAILURE = "communication_failure"
    # Planner or agent failed due to network issues

    COMMAND_DROPPED = "command_dropped"
    # Message lost caused unrecoverable inconsistency

    # ---------------------------
    # System safety termination
    # ---------------------------
    SAFETY_VIOLATION = "safety_violation"
    # Unsafe behavior detected (near collision, power collapse, etc.)

    # ---------------------------
    # Fallback
    # ---------------------------
    UNKNOWN = "unknown"
