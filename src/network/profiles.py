# src/network/profiles.py
"""
Radio profiles: the parameters of every radio link of a run, the same for every strategy.

  submitted  what the submitted experiments used: drone-to-drone messages perfect and instantaneous, unlimited range;
             the command center's radio lossy (95%), delayed (1-3 steps) and short ranged (8 cells), one way only.
  ideal      the primary comparison: every link perfect, instantaneous, whole-map range and symmetric, a broadcast one
             transmission. The communicating strategies get the strongest radio there is, which no real radio matches, so
             a result for SPPDEC (which sends nothing) cannot rest on handicapping a baseline.
  good       the best a real link does: every link symmetric, one step of latency (about 17 ms at 60 steps/s, the fast
             end of telemetry radios) and 0.3% loss (field-test figure inside the range). Kept to state, not hide, what
             the ideal radio gives the baselines (CBBA-Bundle, whose consensus rounds need messages to arrive, is
             about 29% slower on the golden set); not used for the headline comparison.
Sources and figures: .ignore/revision/real_systems_notes.md.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RadioProfile:
    name: str
    range_cells: int            # reach of any radio
    reliability: float          # probability a transmission is received (per receiver)
    delay_range: Tuple[int, int]
    symmetric: bool             # the command center and the drones share the same link parameters
    # the command center's own radio when the links are not symmetric
    reliable_range: Optional[int] = None   # cells within which the link has its full reliability (None: up to the range)
    cc_range_cells: int = 8
    cc_reliability: float = 0.95
    cc_delay_range: Tuple[int, int] = (1, 3)
    outage: Optional[Tuple[int, int]] = None   # (period, length) in steps: the radio is dead for `length` steps in every `period`
    retries: int = 1                           # attempts per message (acknowledged repeat; each attempt costs transmit energy)


PROFILES = {
    "submitted": RadioProfile("submitted", range_cells=1000, reliability=1.0, delay_range=(0, 0), symmetric=False),
    "ideal": RadioProfile("ideal", range_cells=1000, reliability=1.0, delay_range=(0, 0), symmetric=True),
    "good": RadioProfile("good", range_cells=1000, reliability=0.997, delay_range=(1, 1), symmetric=True),
    # stress profile for the revision (checklist 2): half of all transmissions lost, 1-3 steps of delay, every link alike
    "lossy50": RadioProfile("lossy50", range_cells=1000, reliability=0.5, delay_range=(1, 3), symmetric=True),
    # more stress profiles for the revision (every link alike; one radio profile per batch)
    "lossy80": RadioProfile("lossy80", range_cells=1000, reliability=0.2, delay_range=(1, 3), symmetric=True),
    "slow": RadioProfile("slow", range_cells=1000, reliability=1.0, delay_range=(10, 30), symmetric=True),       # 0.17 to 0.5 s at 60 steps a second
    "outage": RadioProfile("outage", range_cells=1000, reliability=1.0, delay_range=(1, 3), symmetric=True, outage=(600, 200)),   # dead a third of the time
    "lossy50_retx3": RadioProfile("lossy50_retx3", range_cells=1000, reliability=0.5, delay_range=(1, 3), symmetric=True, retries=3),
    "lossy80_retx5": RadioProfile("lossy80_retx5", range_cells=1000, reliability=0.2, delay_range=(1, 3), symmetric=True, retries=5),
}
DEFAULT_PROFILE = "ideal"


def apply_profile(profile_name: str, drones, command_center=None) -> RadioProfile:
    """Set the channel of every drone's radio (and the command center's) from the named profile."""
    profile = PROFILES[profile_name]
    for d in drones:
        ch = d.network.channel
        ch.range_cells, ch.reliability, ch.delay_range = profile.range_cells, profile.reliability, profile.delay_range
        ch.reliable_range = profile.reliable_range
        ch.outage, ch.retries = profile.outage, profile.retries
    if command_center is not None:
        ch = command_center.network.channel
        if profile.symmetric:
            ch.range_cells, ch.reliability, ch.delay_range = profile.range_cells, profile.reliability, profile.delay_range
            ch.reliable_range = profile.reliable_range
            ch.outage, ch.retries = profile.outage, profile.retries
        else:
            ch.range_cells, ch.reliability, ch.delay_range = (profile.cc_range_cells, profile.cc_reliability,
                                                              profile.cc_delay_range)
    return profile
