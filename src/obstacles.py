# src/obstacles.py
"""
No-fly blocks on the map (revision: obstacles). Off unless the OBSTACLES environment variable names a layout, "blocks,width,height" (for example "4,3,2": four blocks of 3 x 2 cells).

The layout is a function of the run seed only, so every strategy flies in the same map, and it is part of the start information (the map is declared to every drone, like the station).
A drone flies straight when the straight line between two cells is clear; otherwise it follows the shortest path over free cells (eight neighbours, no cutting a corner of a block).
Planning and flight use the same routine (`distance`, `path`), so a strategy's cost estimate equals the distance flown. Parcels, the station (with a one-cell ring) and the drones' start cells are never inside a block,
and every free cell can be reached from the station.
"""
import heapq
import math
import os
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]
_BLOCKED: Set[Cell] = set()
_GRID: Tuple[int, int] = (0, 0)
_CACHE: Dict[Tuple[Cell, Cell], Tuple[float, List[Cell]]] = {}


def set_obstacles(cells: Iterable[Cell], grid: Tuple[int, int]) -> None:
    global _BLOCKED, _GRID
    _BLOCKED = {(int(c), int(r)) for c, r in cells}
    _GRID = (int(grid[0]), int(grid[1]))
    _CACHE.clear()


def active() -> bool:
    return bool(_BLOCKED)


def cells() -> List[Cell]:
    return sorted(_BLOCKED)


def blocked(cell: Cell) -> bool:
    return (int(cell[0]), int(cell[1])) in _BLOCKED


def segment_clear(a: Cell, b: Cell) -> bool:
    n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.2))
    for i in range(n + 1):
        t = i / n
        x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        if (int(round(x)), int(round(y))) in _BLOCKED:
            return False
    return True


def _grid_path(a: Cell, b: Cell) -> Tuple[float, List[Cell]]:
    cols, rows = _GRID
    open_heap = [(0.0, 0.0, a)]
    best = {a: 0.0}
    came: Dict[Cell, Optional[Cell]] = {a: None}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == b:
            path = []
            while cur != a:
                path.append(cur)
                cur = came[cur]
            return g, path[::-1]
        if g > best.get(cur, 1e18):
            continue
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (cur[0] + dc, cur[1] + dr)
            if not (0 <= n[0] < cols and 0 <= n[1] < rows) or n in _BLOCKED:
                continue
            if dc and dr and ((cur[0] + dc, cur[1]) in _BLOCKED or (cur[0], cur[1] + dr) in _BLOCKED):
                continue
            ng = g + (math.sqrt(2.0) if dc and dr else 1.0)
            if ng + 1e-9 < best.get(n, 1e18):
                best[n] = ng
                came[n] = cur
                heapq.heappush(open_heap, (ng + math.hypot(b[0] - n[0], b[1] - n[1]), ng, n))
    return float("inf"), []


def route(a: Cell, b: Cell) -> Tuple[float, List[Cell]]:
    """(distance in cells, waypoints after `a`) between two cells: straight when clear, else the shortest path over free cells."""
    a, b = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    if a == b:
        return 0.0, []
    key = (a, b)
    hit = _CACHE.get(key)
    if hit is None:
        if segment_clear(a, b):
            hit = (math.hypot(b[0] - a[0], b[1] - a[1]), [b])
        else:
            hit = _grid_path(a, b)
        _CACHE[key] = hit
    return hit


def distance(a: Cell, b: Cell) -> float:
    return route(a, b)[0]


def make_layout(seed, grid: Tuple[int, int], spec: str, keep_free: Iterable[Cell], station: Dict[str, int]) -> List[Cell]:
    """Blocks drawn from the seed: `spec` = "blocks,width,height"; `keep_free` = parcel and start cells; `station` = {col,row,w,h}."""
    n, bw, bh = (int(v) for v in spec.split(","))
    cols, rows = int(grid[0]), int(grid[1])
    free = {(int(c), int(r)) for c, r in keep_free}
    for c in range(station["col"] - 1, station["col"] + station["w"] + 1):
        for r in range(station["row"] - 1, station["row"] + station["h"] + 1):
            free.add((c, r))
    rng = random.Random(f"obstacles-{seed}-{spec}")
    for _ in range(200):
        blocks: Set[Cell] = set()
        for _b in range(n):
            w, h = (bw, bh) if rng.random() < 0.5 else (bh, bw)
            for _try in range(100):
                c0, r0 = rng.randrange(0, max(1, cols - w + 1)), rng.randrange(0, max(1, rows - h + 1))
                cand = {(c, r) for c in range(c0, c0 + w) for r in range(r0, r0 + h)}
                if not (cand & free) and not (cand & blocks):
                    blocks |= cand
                    break
        # every free cell must be reachable from the station
        start = (station["col"], station["row"])
        seen, stack = {start}, [start]
        while stack:
            c, r = stack.pop()
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (c + dc, r + dr)
                if 0 <= nb[0] < cols and 0 <= nb[1] < rows and nb not in blocks and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(seen) == cols * rows - len(blocks):
            return sorted(blocks)
    return []
