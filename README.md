# Agentic AI Meets Multi-Robot Swarms  
### Episodic LLM Guidance for Decentralised Autonomy under Resource Constraints



## TL;DR

This repository contains an applied research simulation framework for studying how Small Language Models (SLMs) and episodic Large Language Models (LLMs) can be used as coordination, negotiation, and arbitration primitives in decentralised multi-robot systems operating under intermittent connectivity, limited bandwidth, and finite energy.

The goal is to empirically evaluate whether language-mediated orchestration improves coordination stability, convergence, and energy efficiency compared to classical decentralised and centralised baselines.
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

### Hybrid Coordination Architecture

Three coordination strategies are compared to study how different levels of structure and authority affect decentralised multi-agent performance under energy and communication constraints.

#### 1. Centralized Coordination
- A central coordinator assigns tasks based on a global view of agent state and task availability when communication permits.
- Agents execute assigned tasks autonomously but do not negotiate or reassign tasks among themselves.
- When communication with the central coordinator is unavailable, no new assignments are issued and coordination degrades.
- This strategy represents a classical centralized planning approach with partial connectivity.

#### 2. Decentralized Greedy Coordination
- Agents operate fully autonomously using local, greedy pickup and delivery heuristics.
- Task selection, movement, and energy management are handled purely locally.
- Agents do not model peer intent, anticipate conflicts, or exchange structured coordination signals.
- Communication, when available, does not influence planning decisions.
- This strategy serves as a decentralised lower bound, highlighting inefficiencies caused by uncoordinated autonomy.

#### 3. Structured Predictive Purpose-Driven Emergent Coordination (SPPDEC)
- Agents implement **Structured Predictive Purpose-Driven Emergent Coordination (SPPDEC)** as the proposed method.
- Each agent performs predictive local reasoning to evaluate task feasibility, energy safety, and potential conflicts.
- Coordination emerges dynamically through:
  - explicit purpose formation,
  - structured intent exchange when communication is available,
  - contextual emergence and transfer of purpose-scoped coordination authority,
  - conservative refusal of unsafe or inefficient commitments.
- A **Large Language Model (LLM)** is used episodically as a guidance and arbitration mechanism when local coordination fails, prediction confidence degrades, or persistent conflicts arise.
- The LLM provides high-level coordination cues such as task redistribution, movement prioritisation, conflict resolution, and energy-balancing strategies.
- The LLM does not issue continuous control commands, does not override local safety constraints, and is not required for ongoing execution.
- When communication or LLM access is unavailable, agents continue operating autonomously using predictive local reasoning, allowing coordination quality to degrade gracefully without loss of safety.

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
│
├─ graphics/                # Drone, parcel, station sprites
├─ strategies/              # Decision-making layer (NEW, core to paper)
│   ├─ __init__.py
│   ├─ base.py              # BaseStrategy (abstract protocol interface)
│   ├─ centralised.py       # Centralised planner strategy
│   ├─ decentralised_greedy.py  # Local greedy decentralised strategy
│   ├─ naive.py             # NaiveStrategy (current behaviour refactored)
│   └─ structured_decentralised_protocol.py  # Implements SPPDEC (Structured Predictive Purpose-Driven Emergent Coordination)
├─ artifacts.py            # Terrain, Drone, Parcel, Station definitions
├─ controllers.py          # HumanAgentController, AIAgentController
├─ flight_recorder.py       # Metrics, traces, trajectories, outcomes
├─ game.py                 # Main PyGame simulation loop
├─ run_experiment.py        # Experiment runner, batch execution, logging
├─ utils.py                # Helper utilities (image scaling, loading, etc.)
│
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