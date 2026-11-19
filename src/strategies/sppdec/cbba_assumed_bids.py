# src/strategies/sppdec/cbba_assumed_bids.py
"""
CBBA run inside one drone's head.

Standard CBBA (Choi, Brunet, How) alternates two phases among the agents: each agent builds a
bundle of tasks by marginal gain and records its bids (phase 1), then the agents exchange their
winner and bid tables and resolve conflicts (phase 2), until the tables agree. In the
communicating baseline phase 2 is real message exchange.

Here there are no messages. The drone runs both phases for the whole believed fleet, itself
included. Wherever the protocol would receive a peer's bids, it uses the bids it assumes the peer
would make, computed from its beliefs about that peer (position, energy, consumption rate).
The drone then acts on its own resulting bundle.

Pure functions, no world access: everything arrives as plain data, so the algorithm is unit-testable.
"""
from typing import Callable, Dict, List, Optional, Tuple

NEG_INF = float("-inf")


COMPLETION_TIE_BREAK = 1e-4      # steps per unit of energy: energy only separates bids that finish at the same step (a full battery is 100 units, so 0.01 step at most)


def _completion_time(agent: Dict, path: List[str], tasks: Dict[str, Dict], drop: Tuple[int, int], dist: Callable, energy: Callable, pick_drop: Callable) -> float:
    """
    The bid for the team objective "finish the last delivery soonest" (Lagoudakis et al. 2005: for the makespan the bid is the robot's whole path cost, here the step at which
    the agent would complete `path`): the steps until the agent is free (`free_in`, from where it will be then, `pos`), plus each trip's flight (empty leg to the pickup, loaded
    leg to the drop) at the agent's own speed. A parcel that may no longer be there (presence below 1) inflates the last trip. Energy enters only as a tie-break.
    """
    speed = float(agent.get("speed", 1.0)) or 1.0
    spc = float(agent["spc"])
    clock = float(agent.get("free_in", 0.0))
    cur = agent["pos"]
    joules = 0.0
    last = None
    for tid in path:
        t = tasks[tid]
        d = t.get("drop", drop)
        clock += (dist(cur, t["pickup"]) + dist(t["pickup"], d)) * spc / speed
        joules += energy(dist(cur, t["pickup"]), 0.0) + energy(dist(t["pickup"], d), t["weight"]) + 2.0 * pick_drop(t["weight"])
        cur, last = d, t
    if agent.get("t_ref") and agent.get("e_ref"):
        # both objectives count equally, each against its own ideal for the whole mission: the share of the team's ideal finish time this path ends at, plus the share of the team's ideal energy it uses
        return clock / last.get("presence", 1.0) / agent["t_ref"] + joules * agent.get("rate", 1.0) / agent["e_ref"]
    return clock / last.get("presence", 1.0) + COMPLETION_TIE_BREAK * joules * agent.get("rate", 1.0)


def _path_cost(agent: Dict, path: List[str], tasks: Dict[str, Dict], drop: Tuple[int, int],
               dist: Callable, energy: Callable, pick_drop: Callable = lambda w: 0.0, travel_cost: Callable = None, cumulative: bool = True,
               scale_presence: bool = True) -> float:
    """Energy for `agent` to serve `path` in order: to each pickup empty, then to the drop loaded, plus the pick and
    the drop of each parcel. With `travel_cost` (the energy-equivalent of the time a flight of that many cells takes) the time of the legs is added, charged at each task's completion time along the path
    (`cumulative`): a task later in a bundle then costs more, which is what lets the CBBA rounds converge; feasibility checks call this without it.
    `scale_presence` divides each task's cost by the chance the parcel is still there (it ranks the bids); a feasibility check passes False: the energy the trip takes does not depend on how long ago the parcel was seen."""
    if not path:
        return 0.0
    if agent.get("completion") and scale_presence:
        return _completion_time(agent, path, tasks, drop, dist, energy, pick_drop)
    cur = agent["pos"]
    speed = float(agent.get("speed", 1.0)) or 1.0           # the agent's flight speed relative to the nominal: a slower agent's flight takes proportionally longer
    time_cost = 0.0
    clock = 0.0                              # cells flown along the path so far (with `cumulative`, time is charged at each task's completion)
    total = agent.get("busy", 0.0)          # energy still to spend on the flight in progress (a carrying peer finishes its delivery first)
    for tid in path:
        t = tasks[tid]
        d = t.get("drop", drop)             # the drop cell this task would use (the station centre when the task names none)
        total += (energy(dist(cur, t["pickup"]), 0.0) + energy(dist(t["pickup"], d), t["weight"])
                  + 2.0 * pick_drop(t["weight"])) / (t.get("presence", 1.0) if scale_presence else 1.0)       # presence: the chance the parcel is still where it was last seen
        if travel_cost is not None:
            leg = dist(cur, t["pickup"]) + dist(t["pickup"], d)
            clock += leg
            time_cost += travel_cost((clock if cumulative else leg) / speed) / t.get("presence", 1.0)
        cur = d
    return total * agent.get("rate", 1.0) + time_cost


def bid_cost(agent: Dict, task: Dict, drop: Tuple[int, int], dist: Callable, energy: Callable,
             pick_drop: Callable = lambda w: 0.0, feasibility_margin: float = 0.0, travel_cost: Callable = None) -> float:
    """
    What `agent` is expected to bid, as a cost, for `task` alone: the whole trip (empty flight to the pickup, loaded flight to the drop, pick and drop)
    at the agent's rate, plus the agent's fairness penalty. This is the first-round bid of CBBA for that task, and it is the entry the Hungarian matrix
    holds, so both models work from the same values. Infinite when the agent may not bid (not eligible, or the trip does not fit its energy).
    """
    elig = agent.get("eligible")
    if elig is not None and task["id"] not in elig:
        return float("inf")
    trip = _path_cost(agent, [task["id"]], {task["id"]: task}, drop, dist, energy, pick_drop)
    rem = agent.get("remaining")
    if rem is not None and _path_cost(agent, [task["id"]], {task["id"]: task}, drop, dist, energy, pick_drop, scale_presence=False) + feasibility_margin > rem:
        return float("inf")
    if travel_cost is not None:
        trip = _path_cost(agent, [task["id"]], {task["id"]: task}, drop, dist, energy, pick_drop, travel_cost)
    return trip + agent.get("penalty", 0.0)


def capability_caps(agents: List[Dict], tasks: List[Dict], drop: Tuple[int, int], dist: Callable, energy: Callable, pick_drop: Callable, feasibility_margin: float = 0.0) -> Dict[str, int]:
    """
    Bundle length of each agent as one function of the team's head count and the agents' capabilities together: the agent's share of the open tasks is
    tasks * c_i / sum_j c_j, the sum running over the whole team, so the number of agents and their capabilities both set every share.
    The capability c_i is one number combining time and energy: the agent's flight speed times an energy factor f_i in (0, 1], the fraction of its speed-share of the work that its
    remaining battery can carry (trips of a typical energy, at its own consumption rate). An agent whose battery covers its share counts at its full speed; one that cannot counts for
    less. There is no absolute cap: the shares are renormalised over the team, so every open task has an owner (the solver's exact energy check still decides what is flown).
    The cap of an agent is its share rounded up, so the shares always cover the tasks. Agents of equal capability have equal factors and so equal shares, ceil(tasks / agents),
    whatever the batteries are.
    """
    import math
    ids = [a["id"] for a in agents]
    n = len(tasks)
    if not ids or n == 0:
        return {i: 1 for i in ids}
    ebar = st_mean([energy(dist(drop, t["pickup"]), 0.0) + energy(dist(t["pickup"], t.get("drop", drop)), t["weight"]) + 2.0 * pick_drop(t["weight"]) for t in tasks])
    speed = {a["id"]: float(a.get("speed", 1.0)) or 1.0 for a in agents}
    total_speed = sum(speed.values())
    c = {}
    for a in agents:
        i = a["id"]
        need = n * speed[i] / total_speed                                   # trips the agent would take on its speed share alone
        rem = a.get("remaining")
        room = float("inf") if rem is None else max(0.0, (rem - feasibility_margin) / max(1e-9, ebar * a.get("rate", 1.0)))
        c[i] = speed[i] * (1.0 if need <= 0 else min(1.0, room / need))
    total = sum(c.values())
    if total <= 0:
        c, total = dict(speed), total_speed
    return {i: max(1, int(math.ceil(n * c[i] / total - 1e-9))) for i in ids}


def st_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def hungarian_bundles(agents: List[Dict], tasks: List[Dict], drop: Tuple[int, int], dist: Callable, energy: Callable,
                      bundle_size: int = 3, feasibility_margin: float = 3.0,
                      pick_drop: Callable = lambda w: 0.0, travel_cost: Callable = None, matcher: str = "hungarian", caps: Optional[Dict[str, int]] = None) -> Dict:
    """
    (`matcher="greedy"`: each layer is matched cheapest pair first instead of by min-sum, on the same bids; ties go to the lower agent id, then the task's place in the list.)
    Hungarian bundles: the assignment solved in layers on the same bids CBBA works from. Layer 1 is the min-sum assignment of the agents to the tasks
    (the whole-trip bid of `bid_cost`). The assigned tasks are removed and every agent's next bid is priced from where it will be after its tasks so far
    (the marginal cost of adding the task to its path, with the energy of the path so far subtracted from its battery, as CBBA prices a second task).
    The layers repeat until a bundle holds `bundle_size` tasks or no task is left that anyone can serve. A bundle lists its tasks in layer order.
    Deterministic given the order of `agents` and `tasks` (the caller sorts both).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    ids = [a["id"] for a in agents]
    by_id = {a["id"]: a for a in agents}
    tk = {t["id"]: t for t in tasks}
    left = [t["id"] for t in tasks]
    path = {i: [] for i in ids}
    big = 1e9
    layers = max([int(bundle_size)] + [int(v) for v in (caps or {}).values()]) if caps else int(bundle_size)
    all_ids = ids
    for _ in range(max(1, layers)):
        ids = [i for i in all_ids if len(path[i]) < (caps.get(i, bundle_size) if caps else bundle_size)]       # an agent at its cap takes no part in the layer
        if not left or not ids:
            break
        cost = np.full((len(ids), len(left)), big)
        for r, aid in enumerate(ids):
            a = by_id[aid]
            elig = a.get("eligible")
            so_far = 0.0 if a.get("completion") else _path_cost(a, path[aid], tk, drop, dist, energy, pick_drop, travel_cost)     # completion bids are whole-path (the makespan rule), not marginal
            rem = a.get("remaining")
            for c, tid in enumerate(left):
                if elig is not None and tid not in elig:
                    continue
                if rem is not None and _path_cost(a, path[aid] + [tid], tk, drop, dist, energy, pick_drop, scale_presence=False) + feasibility_margin > rem:
                    continue
                whole = _path_cost(a, path[aid] + [tid], tk, drop, dist, energy, pick_drop, travel_cost)
                cost[r, c] = whole - so_far + a.get("penalty", 0.0)
        padded = cost
        if len(ids) > len(left):
            padded = np.hstack([cost, np.zeros((len(ids), len(ids) - len(left)))])
        if matcher == "greedy":
            pairs = sorted((cost[r, c], r, c) for r in range(len(ids)) for c in range(len(left)) if cost[r, c] < big)
            used_r, used_c, rows, cols = set(), set(), [], []
            for _, r, c in pairs:
                if r not in used_r and c not in used_c:
                    used_r.add(r); used_c.add(c); rows.append(r); cols.append(c)
        else:
            rows, cols = linear_sum_assignment(padded)
        given = []
        for r, c in zip(rows, cols):
            if c < len(left) and cost[r, c] < big:
                path[ids[r]].append(left[c])
                given.append(left[c])
        if not given:
            break
        left = [t for t in left if t not in given]
    return {"bundles": path}


def cbba_with_assumed_bids(
        agents: List[Dict],
        tasks: List[Dict],
        drop: Tuple[int, int],
        dist: Callable,
        energy: Callable,
        bundle_size: int = 3,
        feasibility_margin: float = 3.0,
        max_rounds: Optional[int] = None,
        pick_drop: Callable = lambda w: 0.0,
        travel_cost: Callable = None,
        caps: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    agents: [{"id", "pos", "rate", "penalty", "remaining" (energy left, or None for unlimited),
              "eligible" (set of task ids this agent may bid on, or None for all)}]
    tasks:  [{"id", "pickup", "weight"}]
    Returns {"bundles": {agent_id: [task ids in execution order]}, "rounds": int, "converged": bool}.
    Deterministic: ties break on the lower agent id.
    """
    ids = [a["id"] for a in agents]
    by_id = {a["id"]: a for a in agents}
    tk = {t["id"]: t for t in tasks}
    task_ids = [t["id"] for t in tasks]
    order = {tid: k for k, tid in enumerate(task_ids)}  # list position: ids are random per run
    if not tasks or not agents:
        return {"bundles": {i: [] for i in ids}, "rounds": 0, "converged": True, "settled": True}

    L = max(1, int(bundle_size))
    max_rounds = max_rounds or (len(task_ids) + len(ids) + 6)

    # Per agent state: y winners, z winning bids, bundle (order added), path (execution order)
    y = {i: {} for i in ids}
    z = {i: {} for i in ids}
    bundle = {i: [] for i in ids}
    path = {i: [] for i in ids}

    def score(aid: str, p: List[str]) -> float:
        a = by_id[aid]
        return -_path_cost(a, p, tk, drop, dist, energy, pick_drop, travel_cost)

    steal = {"on": True}

    def build(aid: str) -> bool:
        """Phase 1 for one agent. Returns True if its bundle changed."""
        a = by_id[aid]
        changed = False
        elig = a.get("eligible")
        while len(bundle[aid]) < (caps.get(aid, L) if caps else L):
            cur_score = 0.0 if a.get("completion") else score(aid, path[aid])      # completion bids are whole-path, not marginal
            best = None  # (gain, tid, position)
            for tid in task_ids:
                if tid in bundle[aid]:
                    continue
                if elig is not None and tid not in elig:
                    continue
                g_best, pos_best = None, None
                for pos in range(len(path[aid]) + 1):
                    cand = path[aid][:pos] + [tid] + path[aid][pos:]
                    g = score(aid, cand) - cur_score - a.get("penalty", 0.0)
                    if g_best is None or g > g_best + 1e-12:
                        g_best, pos_best = g, pos
                # budget: the whole path must fit in the energy this agent is believed to have
                rem = a.get("remaining")
                cand_path = path[aid][:pos_best] + [tid] + path[aid][pos_best:]
                if rem is not None and _path_cost(a, cand_path, tk, drop, dist, energy, pick_drop, scale_presence=False) + feasibility_margin > rem:
                    continue
                # h_ij: bid only if it beats the best known winning bid for this task
                zi = z[aid].get(tid, NEG_INF)
                yi = y[aid].get(tid)
                if yi is None or yi == aid:
                    pass
                elif not steal["on"]:
                    continue  # settling rounds: a task that has a winner stays with it
                elif not (g_best > zi + 1e-12 or (abs(g_best - zi) <= 1e-12 and aid < yi)):
                    continue
                if best is None or g_best > best[0] + 1e-12 or (abs(g_best - best[0]) <= 1e-12 and order[tid] < order[best[1]]):
                    best = (g_best, tid, pos_best)
            if best is None:
                break
            g, tid, pos = best
            bundle[aid].append(tid)
            path[aid].insert(pos, tid)
            y[aid][tid] = aid
            z[aid][tid] = g
            changed = True
        return changed

    def resolve() -> bool:
        """
        Phase 2: every agent receives every other agent's (assumed) tables and applies the CBBA
        update rule. With all tables visible, the winner of a task is the highest bid among the
        agents that still claim it themselves (a claim another agent holds about a third party
        is stale once that agent has released the task); ties go to the lower agent id. An agent
        that loses a task also releases the tasks it added after it.
        """
        changed = False
        claims_of = {i: dict(y[i]) for i in ids}
        zs_of = {i: dict(z[i]) for i in ids}
        for tid in task_ids:
            win_z, win_y = NEG_INF, None
            for k in ids:
                if claims_of[k].get(tid) == k:
                    zk = zs_of[k].get(tid, NEG_INF)
                    if win_y is None or zk > win_z + 1e-12 or (abs(zk - win_z) <= 1e-12 and k < win_y):
                        win_z, win_y = zk, k
            for i in ids:
                cur_y, cur_z = y[i].get(tid), z[i].get(tid, NEG_INF)
                if win_y is None:
                    if cur_y is not None:
                        changed = True
                        y[i].pop(tid, None)
                        z[i].pop(tid, None)
                    continue
                if cur_y != win_y or abs(cur_z - win_z) > 1e-12:
                    changed = True
                    lost = cur_y == i and win_y != i
                    y[i][tid], z[i][tid] = win_y, win_z
                    if lost and tid in bundle[i]:
                        cut = bundle[i].index(tid)
                        for rel in bundle[i][cut + 1:]:
                            if y[i].get(rel) == i:
                                y[i].pop(rel, None)
                                z[i].pop(rel, None)
                        bundle[i] = bundle[i][:cut]
                        path[i] = [t for t in path[i] if t in bundle[i]]
        return changed

    converged = False
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        c1 = False
        for i in ids:
            c1 = build(i) or c1
        c2 = resolve()
        if not c1 and not c2:
            converged = True
            break
    settled = converged
    if not converged:
        # Marginal gains here are not diminishing (a far agent's later tasks are cheaper than its
        # first), so plain CBBA can cycle. Finish with rounds in which a task that already has a
        # winner cannot be taken away; that always terminates.
        steal["on"] = False
        for extra in range(1, len(task_ids) * 2 + 4):
            c1 = False
            for i in ids:
                c1 = build(i) or c1
            c2 = resolve()
            rounds += 1
            if not c1 and not c2:
                settled = True
                break
    # tasks an agent still believes are its own but that no one else holds are kept as-is

    return {"bundles": {i: list(path[i]) for i in ids}, "rounds": rounds, "converged": converged,
            "settled": settled,
            "winners": {i: dict(y[i]) for i in ids}}
