# Agentic AI Meets Multi-Robot Swarms  
### Episodic LLM Guidance for Decentralised Autonomy under Resource Constraints

## Overview
This repository provides a **PyGame-based simulation environment** designed to explore the intersection of **Agentic AI** and **multi-robot swarm coordination** under realistic resource limitations.  
Each drone (agent) operates within a grid world, tasked with locating, picking up, and delivering parcels — all while managing internal resources such as **power** and **network connectivity**.

The environment forms the foundation for investigating whether **episodic, high-level guidance from a Large Language Model (LLM)** can help resource-limited agents coordinate more effectively without continuous central control.

---

## Problem
In multi-agent systems, maintaining coordination across robot swarms is challenging when:
- Bandwidth is limited,
- Communication is intermittent or noisy,
- Agents operate with finite power reserves, and
- The environment changes dynamically.

Fully decentralised approaches are robust to failure but can become inefficient or disorganised.  
Fully centralised control allows global awareness but introduces bottlenecks and single points of failure.  

This work therefore investigates a **hybrid cognitive architecture**:  
- Decentralised agents maintain **local autonomy** and **peer prediction**, while  
- Periodic, **episodic guidance** from an LLM provides system-wide recalibration to improve coordination, **resource allocation**, and **energy efficiency**.

---

## Methodology

### Simulation Environment
The simulation is built in **PyGame**, modelling:
- A 2D **terrain grid** populated with parcels and a delivery station.
- **Autonomous drones** with internal resource states:
  - **Energy (battery)** that depletes based on distance travelled and parcel weight.  
  - **Network availability**, influencing how often agents can communicate or request LLM guidance.

Agents must **balance task efficiency** (delivering parcels quickly) with **resource conservation**, deciding when to:
- Return to station for recharge,
- Conserve bandwidth by operating offline, or
- Request LLM guidance when uncertainty is high.

### Agent Controllers
Two controllers are provided:
- **HumanController** — for direct manual testing.
- **ComputerController** — a heuristic or rule-based agent.
  
A **ControllerSwitcher** allows toggling between human and AI control at runtime (`TAB` key).

### Hybrid Architecture (Planned)
Two configurations are compared:

#### 1. Baseline Swarm
- Agents follow fixed heuristics (simple pickup/delivery rules).  
- Communication and energy management are purely local.

#### 2. Predictive Hybrid Swarm
- Agents integrate **Small Language Models (SLMs)** for local prediction of peer state.  
- When energy or prediction confidence drops below a threshold, agents query a **central LLM (Episodic Guidance Module)** for high-level strategic updates.  
- The LLM provides **coordination cues** such as resource redistribution, movement prioritisation, and energy-saving strategies.

---

## Resource Dynamics
Each agent manages:
| Resource | Description | Effect |
|-----------|--------------|--------|
| **Power** | Depletes with travel distance and carried load | Determines operation time before recharge |
| **Network** | Represents communication bandwidth or signal strength | Limits ability to share state or query LLM |
| **Task Load** | Tracks number of parcels handled concurrently | Affects decision fatigue and LLM request frequency |

When power or bandwidth is low, agents **reduce communication frequency** and rely on **peer prediction** instead of central coordination.  
The LLM is only consulted **episodically**, when uncertainty or energy imbalance rises system-wide.

---

## Framework Design
The long-term goal integrates:
- **LangChain** and **LangGraph** for orchestrating LLM reasoning, agent dialogue, and episodic state recalibration.  
- **Local autonomy** during network loss, with predictive continuation until the next episodic update.  
- **Dynamic adaptation**, allowing the swarm to gracefully degrade and self-stabilize under communication and energy constraints.

---

## Evaluation Metrics
| Metric | Description |
|---------|-------------|
| **Mission Success Rate** | % of parcels successfully delivered |
| **Energy Efficiency** | Energy consumed per successful delivery |
| **Communication Efficiency** | Messages or bandwidth used per mission |
| **Adaptability** | Performance under sudden bandwidth/power loss |
| **Prediction Accuracy** | Correctness of peer-state forecasts during disconnection |

---

## Expected Contribution
This project aims to:
1. Demonstrate how **episodic LLM guidance** can enhance swarm coordination under resource constraints.  
2. Present a **simulation framework** to evaluate energy-aware, communication-efficient swarm strategies.  
3. Provide empirical insights on **graceful degradation** and **autonomous recovery** in distributed multi-agent systems.

---

## Repository Structure
```

src/
├── graphics/                  # Drone and parcel sprites (visual assets)
├── strategies/                # Coordination strategy modules
│   ├── __init__.py
│   ├── auction.py             # Auction-based multi-agent coordination
│   ├── base.py                # BaseStrategy interface and abstract definitions
│   ├── centralized_greedy.py  # Centralized greedy coordination strategy
│   └── reservation.py         # Reservation table strategy (legacy/simple)
│
├── arbitration.py             # Deterministic scorer and assignment arbitration logic
├── artifacts.py               # Terrain, Drone, Parcel, and Station definitions
├── controllers.py             # Human and AI controller logic (uses Coordinator + LLM agent)
├── coordinator.py             # High-level Coordinator and Strategy registry
├── games.py                   # Main PyGame simulation loop and HUD rendering
├── leader_coordinator.py      # Leader election and authoritative reservation management
├── llm_agent.py               # Per-drone LLM wrapper with caching and rate limits
├── llm_client.py              # Core LLM API client utilities
├── llm_planner.py             # Planner client (LangChain/TinyLLM integration, fallback logic)
├── proposals.py               # Proposal schema, validation, and caching utilities
├── reservations.py            # Pluggable atomic reservation API (default in-process)
└── utils.py                   # Shared helper utilities (image loading, scaling, math)


````

---

## Controls
| Action | Key |
|--------|-----|
| Move (hold) | Arrow keys or diagonals |
| Pick up / Drop | `Space` |
| Switch Controller | `TAB` |
| Start timed session | `S` |
| Reset counters | `R` |
| Quit | `ESC` |
| Place parcel (mouse) | Left-click |

---

## Installation
```bash
git clone https://github.com/opemipoVRB/MultiAgentSimulator
cd MultiAgentSimulator/src
pip install pygame-ce
python game.py
````

---

## Next Steps

* Integrate **energy and bandwidth simulation** into the agent model.
* Connect the **Episodic Guidance Module** via LangChain/LangGraph.
* Extend to multi-agent scenarios for cooperative decision-making.
* Evaluate resource-aware task allocation strategies.

---

## Citation
> **Opemipo Durodola (2025)**
> *Agentic AI Meets Multi-Robot Swarms: Episodic LLM Guidance for Decentralised Autonomy under Resource Constraints*
> Simulation and prototype research framework, 2025.
---