# src/system/topology.py
"""
Which logical CPUs share a physical core, restricted to the CPUs this process may use.

Two drones pinned to the two hyperthreads of one physical core would share its execution units, so a drone gets
a whole physical core: its first logical CPU.
"""
import os
from typing import List


def allowed_cpus() -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def physical_core_groups() -> List[List[int]]:
    """One list of logical CPU ids per physical core, ordered by the lowest id. Falls back to one group per
    logical CPU where the topology is not readable."""
    allowed = set(allowed_cpus())
    groups: List[List[int]] = []
    seen = set()
    for cpu in sorted(allowed):
        if cpu in seen:
            continue
        siblings = [cpu]
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            text = open(path).read().strip()
            ids = []
            for part in text.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    ids.extend(range(int(a), int(b) + 1))
                else:
                    ids.append(int(part))
            siblings = sorted(i for i in ids if i in allowed) or [cpu]
        except (OSError, ValueError):
            pass
        seen.update(siblings)
        groups.append(siblings)
    return groups
