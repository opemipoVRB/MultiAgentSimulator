# src/controllers.py
import math
import pygame
import random
import time
from typing import Optional, Tuple, List, Dict

from llm_planner import PlannerClient

# create planner client: set use_llm=True if you installed LangChain and have credentials.
PLANNER = PlannerClient(use_llm=False)

# If True, AI will attempt trips even if it cannot guarantee a return-to-base.
# The drone will be allowed to attempt pickup/drop and may become lost if battery drains mid-route.
ALLOW_RISKY_TRIPS = True

# -------------------------
# Reservation (coordination) system (module-level)
# -------------------------
# This simple reservation dictionary allows controllers to "claim" a parcel
# (by cell coordinates) so other controllers will avoid driving to the same
# pickup. The value is a tuple: (owner_controller, timestamp_seconds).
#
# Important: this is intentionally simple and works in a single-threaded game
# loop context. If you later run controllers in threads/processes, you'll
# need to add synchronization primitives.
RESERVATIONS: Dict[Tuple[int, int], Tuple["AIAgentController", float]] = {}
# How long a reservation is considered valid (seconds). If the owner doesn't
# pick within this time another controller may claim it.
RESERVATION_TIMEOUT = 8.0

# How many pickups ahead to attempt to reserve from a new plan (keeps others from duplicating)
MAX_RESERVE_AHEAD = 2

# How often (seconds) to run the reservation cleanup (instead of random checks)
RESERVATION_CLEANUP_INTERVAL = 1.0


# -------------------------
# Base controller classes
# -------------------------
class BaseAgentController:
    """Base controller API used by the game loop."""
    def __init__(self, drone, terrain):
        self.drone = drone
        self.terrain = terrain
        # friendly status string we can show in HUD for debugging
        self.last_status: Optional[str] = None

    def handle_event(self, event):
        pass

    def update(self, dt: float):
        pass


class HumanAgentController(BaseAgentController):
    """Manual control for the drone (keyboard)."""
    def __init__(self, drone, terrain):
        super().__init__(drone, terrain)
        self._last_keys = pygame.key.get_pressed()

    def handle_event(self, event):
        pass

    def update(self, dt: float):
        keys = pygame.key.get_pressed()
        if self.drone.lost:
            return

        # Movement (only set a new target if drone isn't already moving)
        if not self.drone.moving:
            dx = 0
            dy = 0
            if keys[pygame.K_LEFT]:
                dx -= 1
            if keys[pygame.K_RIGHT]:
                dx += 1
            if keys[pygame.K_UP]:
                dy -= 1
            if keys[pygame.K_DOWN]:
                dy += 1

            if dx != 0 or dy != 0:
                new_col = self.drone.col + dx
                new_row = self.drone.row + dy
                # ensure energy to move and optionally return home
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    home_col = int(station.col + station.w // 2)
                    home_row = int(station.row + station.h // 2)
                else:
                    home_col, home_row = self.drone.col, self.drone.row

                # The human controller still uses a conservative check (so player doesn't instantly run out).
                if hasattr(self.drone, "can_reach_and_return") and not self.drone.can_reach_and_return(
                        new_col, new_row, home_col, home_row):
                    # insufficient energy for that move + return -> send home instead
                    self.drone.set_target_cell(home_col, home_row)
                else:
                    self.drone.set_target_cell(new_col, new_row)

        # Edge-detect SPACE for pick/drop (human triggered)
        if keys[pygame.K_SPACE] and not self._last_keys[pygame.K_SPACE]:
            if self.drone.carrying is None:
                p = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p:
                    success = self.drone.perform_pick(p)
                    if not success:
                        # pick failed due to battery -> pick_failed UI flash
                        self.drone._last_action = ("pick_failed", (self.drone.col, self.drone.row), None)
            else:
                # drop only if target cell has no non-picked parcel (includes delivered)
                if self.terrain.occupied_cell(self.drone.col, self.drone.row):
                    self.drone._last_action = ("drop_failed", (self.drone.col, self.drone.row), None)
                else:
                    self.drone.perform_drop(self.drone.carrying)

        self._last_keys = keys


# -------------------------
# AI controller with reservation logic
# -------------------------
class AIAgentController(BaseAgentController):
    """
    Planner-driven AI controller with an improved parcel reservation system and
    light-weight team coordination.

    Improvements relative to earlier version:
      * Refreshes reservation timestamps for owned reservations.
      * Reserves the first MAX_RESERVE_AHEAD pickups returned by planner so other
        AIs avoid wasting travel.
      * Centralized cleanup runs on a timer (less random overhead).
      * Planner requests are rate-limited to avoid flooding PLANNER.
      * Controller releases reservations on pick and on plan advancement.
      * Greedy fallback so idle drones actively claim unreserved parcels.
      * Aggressive greedy claiming when idle so all drones will work concurrently
        if they have sufficient power to attempt a pickup.
    """

    def __init__(self, drone, terrain, search_radius: Optional[int] = None, replanning_interval: float = 4.0):
        super().__init__(drone, terrain)
        self.state = "idle"
        self.plan: List[Dict] = []   # each item: {"pickup": (c,r), "dropoff": (c,r), "weight": float}
        self.plan_idx: int = 0
        self.cooldown = 0.0
        self.search_radius = search_radius
        self._last_plan_time = 0.0
        self._replanning_interval = replanning_interval
        self._last_action_time = 0.0
        self.last_narration: Optional[str] = None
        # used to bias away from repeating the very last chosen drop cell
        self._last_chosen_drop: Optional[Tuple[int, int]] = None

        # reservation cleanup bookkeeping
        self._last_reservation_cleanup = 0.0
        # planner backoff: avoid calling planner too often
        self._planner_backoff = 0.0

    # ---------------------------
    # Reservation helpers (coordination between multiple AIs)
    # ---------------------------
    @staticmethod
    def _parcel_key(col: int, row: int) -> Tuple[int, int]:
        """Canonical key for a parcel cell used in RESERVATIONS dict."""
        return (int(col), int(row))

    def _cleanup_reservations(self, force: bool = False):
        """Remove expired reservations so other controllers can claim them.

        Runs at most once per RESERVATION_CLEANUP_INTERVAL unless `force` is True.
        """
        now = time.time()
        if not force and (now - self._last_reservation_cleanup) < RESERVATION_CLEANUP_INTERVAL:
            return
        self._last_reservation_cleanup = now

        expired = []
        for key, (owner, ts) in list(RESERVATIONS.items()):
            # if reservation is older than timeout or owner no longer valid -> expire it
            if (now - ts) > RESERVATION_TIMEOUT:
                expired.append(key)
            else:
                # if owner controller's drone no longer exists or is lost, expire as well
                try:
                    if owner is None or getattr(owner, "drone", None) is None or getattr(owner.drone, "lost", False):
                        expired.append(key)
                except Exception:
                    expired.append(key)
        for k in expired:
            RESERVATIONS.pop(k, None)

    def _is_reserved_by_other(self, col: int, row: int) -> bool:
        """Return True if a valid (non-expired) reservation exists for (col,row) and is owned by someone else."""
        self._cleanup_reservations()
        k = self._parcel_key(col, row)
        return (k in RESERVATIONS) and (RESERVATIONS[k][0] is not self)

    def _reserved_by_me(self, col: int, row: int) -> bool:
        """Return True if I currently own the reservation for that cell."""
        self._cleanup_reservations()
        k = self._parcel_key(col, row)
        return (k in RESERVATIONS) and (RESERVATIONS[k][0] is self)

    def _try_reserve(self, col: int, row: int) -> bool:
        """
        Attempt to reserve the parcel cell.
        If already owned by me, refresh timestamp and return True.
        Returns True if reservation is acquired (or already owned by this controller),
        False if someone else holds it.
        """
        self._cleanup_reservations()
        k = self._parcel_key(col, row)

        if k in RESERVATIONS:
            owner, ts = RESERVATIONS[k]
            if owner is self:
                # refresh timestamp
                RESERVATIONS[k] = (self, time.time())
                return True
            else:
                # someone else owns it and it's not expired
                return False

        # Acquire reservation
        RESERVATIONS[k] = (self, time.time())
        # debug info
        self.last_status = f"reserved {k}"
        return True

    def _release_reservation(self, col: int, row: int):
        """Release reservation for a cell if owned by this controller."""
        k = self._parcel_key(col, row)
        if k in RESERVATIONS and RESERVATIONS[k][0] is self:
            RESERVATIONS.pop(k, None)

    def _reserve_plan_pickups(self):
        """Try to reserve the first few pickups from the current plan to avoid duplicate travel."""
        if not self.plan:
            return
        reserved = 0
        for step in self.plan[self.plan_idx: self.plan_idx + MAX_RESERVE_AHEAD]:
            if reserved >= MAX_RESERVE_AHEAD:
                break
            pickup = step.get("pickup")
            if not pickup:
                continue
            # if another already has it, we skip it (planner should reassign later)
            got = self._try_reserve(pickup[0], pickup[1])
            if got:
                reserved += 1
            else:
                # If we can't reserve the first pickup, don't fight — try to replan next cycle with force.
                self.last_status = f"pickup {pickup} reserved by other (plan reserve failed)"
                # small cooldown to avoid spamming planner
                self._planner_backoff = max(self._planner_backoff, 0.7)
                return

    # -------------
    # Helpers
    # -------------
    def _field_parcels_remaining(self) -> bool:
        """Return True if any parcel in the terrain is not delivered (still in the field)."""
        for p in self.terrain.parcels:
            # treat undelivered parcels in the world as "in the field". Picked parcels are not in the field.
            if not getattr(p, "delivered", False) and not getattr(p, "picked", False):
                return True
        return False

    # ---------- Greedy fallback helpers ----------
    def _euclid_dist_cells(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return math.hypot(dx, dy)

    def _energy_est_for_cells(self, n_cells: int, weight: float) -> float:
        """Wrapper: uses drone's energy model if available."""
        try:
            return float(self.drone.energy_needed_for_cells(n_cells, weight))
        except Exception:
            # fallback conservative constant
            base = getattr(self.drone, "BASE_COST_PER_CELL", 0.2)
            wfactor = getattr(self.drone, "WEIGHT_FACTOR", 0.5)
            return float(n_cells) * base * (1.0 + weight * wfactor)

    def _can_attempt_parcel(self, parcel) -> bool:
        """
        Determines if this drone has sufficient power to attempt the parcel.
        Logic:
          - If ALLOW_RISKY_TRIPS: require enough energy to *reach* the pickup (immediate leg).
          - Else: require enough energy to reach pickup and then reach nearest station (conservative).
        """
        if parcel is None:
            return False
        my_col, my_row = int(self.drone.col), int(self.drone.row)
        pc, pr = int(parcel.col), int(parcel.row)

        # Manhattan dist in cells
        dist_to_pick = abs(my_col - pc) + abs(my_row - pr)

        # estimate energy for leg to pickup (empty)
        needed = self._energy_est_for_cells(dist_to_pick, 0.0)
        if ALLOW_RISKY_TRIPS:
            return self.drone.power.level >= needed

        # conservative: also need energy from pickup to nearest station center (carrying)
        station = self.terrain.nearest_station(pc, pr)
        if station:
            sc = int(station.col + station.w // 2)
            sr = int(station.row + station.h // 2)
            dist_pick_to_station = abs(pc - sc) + abs(pr - sr)
            needed += self._energy_est_for_cells(dist_pick_to_station, getattr(parcel, "weight", 1.0))
        else:
            # no station: require only pickup leg (we can't guarantee drop)
            pass

        return self.drone.power.level >= needed

    def _find_nearest_unreserved_parcel(self) -> Optional[object]:
        """
        Return the nearest parcel object that is not picked, not delivered,
        and either unreserved or reserved by me (prefers truly unreserved).
        Additionally filters by _can_attempt_parcel so the drone only claims work it can do.
        """
        best = None
        best_d = None
        my_best = None
        my_best_d = None

        my_pos = (int(self.drone.col), int(self.drone.row))
        for p in self.terrain.parcels:
            if getattr(p, "picked", False) or getattr(p, "delivered", False):
                continue
            # skip parcels this drone cannot sensibly attempt
            try:
                if not self._can_attempt_parcel(p):
                    continue
            except Exception:
                # if energy check fails for some reason, be conservative and skip
                continue

            key = (int(p.col), int(p.row))
            # if reserved by other, skip it
            if key in RESERVATIONS and RESERVATIONS[key][0] is not self:
                continue
            d = self._euclid_dist_cells(my_pos, (p.col, p.row))
            # prefer completely unreserved parcels first
            if key not in RESERVATIONS:
                if best is None or d < best_d:
                    best = p
                    best_d = d
            else:
                # reserved by me
                if my_best is None or d < my_best_d:
                    my_best = p
                    my_best_d = d

        # if there are any unreserved parcels, return the nearest unreserved, else nearest mine
        return best if best is not None else my_best

    def _try_claim_and_go_to_parcel(self, parcel) -> bool:
        """
        Try to reserve the parcel, and if successful, set target and state to seek it.
        Returns True on success, False otherwise.
        """
        if parcel is None:
            return False
        key = (int(parcel.col), int(parcel.row))

        # If already reserved by me, just set the target
        if self._reserved_by_me(key[0], key[1]):
            if not self.drone.moving:
                self.drone.set_target_cell(key[0], key[1])
                self.state = "seeking"
                self.last_status = f"heading to reserved {key}"
            return True

        got = self._try_reserve(key[0], key[1])
        if not got:
            return False

        # Reservation acquired -> head there
        if not self.drone.moving:
            self.drone.set_target_cell(key[0], key[1])
            self.state = "seeking"
            self.last_status = f"reserved and heading to {key}"
        return True

    # ---------------------------
    # Planner interaction
    # ---------------------------
    def _make_snapshot(self) -> Dict:
        """
        Create a compact snapshot of the world state for the planner input.

        This snapshot includes:
          - agent location and battery_pct
          - carrying info (has, weight)
          - nearest station (col,row,w,h)
          - all parcels (col,row,weight,picked,delivered)
          - and reserved_cells so planner can avoid recommending already-claimed pickups
        """
        snap = {
            "agent": {
                "col": int(self.drone.col),
                "row": int(self.drone.row),
                "battery_pct": int(self.drone.power.percent()) if hasattr(self.drone, "power") else 0,
                # also include raw power level (energy units) if available
                "battery_level": float(self.drone.power.level) if hasattr(self.drone, "power") else None,
            },
            "carrying": {
                "has": bool(self.drone.carrying),
                "weight": getattr(self.drone.carrying, "weight", 0.0) if self.drone.carrying else 0.0
            },
            "nearest_station": None,
            "all_parcels": [],
            "timestamp": time.time(),
            # energy model exposed so planner and LLM use the same numbers:
            "energy_model": {
                "base_cost_per_cell": float(getattr(self.drone, "BASE_COST_PER_CELL", 0.2)),
                "weight_factor": float(getattr(self.drone, "WEIGHT_FACTOR", 0.5)),
                "pick_drop_cost": float(getattr(self.drone, "PICK_DROP_COST", 0.7)),
                "allow_diagonal": bool(getattr(self.drone, "allow_diagonal", True))
            },
            # grid info helpful for distance reasoning
            "grid": {
                "grid_size": int(self.terrain.grid_size),
                "cols": int(self.terrain.screen_size[0] // self.terrain.grid_size),
                "rows": int(self.terrain.screen_size[1] // self.terrain.grid_size),
            },
            # reservations exposed so planner can avoid already-claimed pickups
            "reserved_cells": [list(k) for k in RESERVATIONS.keys()]
        }

        # nearest station as before
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
        if station:
            snap["nearest_station"] = {"col": int(station.col), "row": int(station.row),
                                       "w": int(station.w), "h": int(station.h),
                                       # center cell used by controllers as "home"
                                       "center": (int(station.col + station.w // 2), int(station.row + station.h // 2))
                                       }

        # helper to compute euclidean distance in cell units
        def _euclid_cell_dist(a_col, a_row, b_col, b_row):
            dx = float(b_col - a_col)
            dy = float(b_row - a_row)
            return (dx * dx + dy * dy) ** 0.5

        # exposed energy calc helper (same formula as Drone.energy_needed_for_cells)
        base = snap["energy_model"]["base_cost_per_cell"]
        wfactor = snap["energy_model"]["weight_factor"]

        def est_energy_for_cells(n_cells, weight):
            return float(n_cells) * base * (1.0 + weight * wfactor)

        # For each parcel, include estimates so LLM doesn't have to guess mapping battery%->energy
        for p in self.terrain.parcels:
            parcel_info = {
                "col": int(p.col),
                "row": int(p.row),
                "weight": float(getattr(p, "weight", 1.0)),
                "picked": bool(getattr(p, "picked", False)),
                "delivered": bool(getattr(p, "delivered", False))
            }

            # compute distances & rough energy numbers (agent->pickup, pickup->nearest_station_center)
            adist = _euclid_cell_dist(self.drone.col, self.drone.row, p.col, p.row)
            parcel_info["dist_agent_cells_euclidean"] = round(adist, 3)

            if station:
                sc = int(station.col + station.w // 2)
                sr = int(station.row + station.h // 2)
                sdist = _euclid_cell_dist(p.col, p.row, sc, sr)
                parcel_info["dist_to_station_cells_euclidean"] = round(sdist, 3)
                # estimated energy to go agent->pickup (empty) then pickup->station (carrying)
                est = est_energy_for_cells(adist, 0.0) + est_energy_for_cells(sdist, parcel_info["weight"])
                parcel_info["est_energy_agent_pickup_and_deliver_to_station"] = round(est, 3)
            else:
                parcel_info["dist_to_station_cells_euclidean"] = None
                parcel_info["est_energy_agent_pickup_and_deliver_to_station"] = None

            # add a simpler conservative manhattan estimate (useful if prompt asks for it)
            md = abs(int(self.drone.col) - int(p.col)) + abs(int(self.drone.row) - int(p.row))
            parcel_info["dist_agent_cells_manhattan"] = int(md)

            snap["all_parcels"].append(parcel_info)

        return snap

    def _request_plan(self, force_refresh: bool = False):
        """Ask PLANNER for a plan given current snapshot. Handles parsing and some truncation for narration.

        This method also rate-limits planner calls via self._planner_backoff to avoid flooding the planner.
        """
        now = time.time()
        if self._planner_backoff > 0 and (now - self._last_plan_time) < self._planner_backoff and not force_refresh:
            # still backing off
            return

        snap = self._make_snapshot()
        try:
            plan_obj = PLANNER.request_plan(snap, force_refresh=force_refresh)
        except Exception as ex:
            # Planner failed: log and keep empty plan
            print("[PLANNER] request failed:", ex)
            self.plan = []
            self.plan_idx = 0
            self.last_narration = None
            self._last_plan_time = time.time()
            # set small backoff so we don't spam repeatedly
            self._planner_backoff = max(self._planner_backoff, 1.0)
            return

        plan_list = plan_obj.get("plan", []) if isinstance(plan_obj, dict) else []
        parsed = []
        for step in plan_list:
            try:
                pickup = (int(step["pickup"][0]), int(step["pickup"][1]))
                dropoff = (int(step["dropoff"][0]), int(step["dropoff"][1]))
                weight = float(step.get("weight", 1.0))
                parsed.append({"pickup": pickup, "dropoff": dropoff, "weight": weight})
            except Exception:
                continue

        self.plan = parsed
        self.plan_idx = 0
        self._last_plan_time = time.time()
        self._planner_backoff = 0.0  # reset backoff on success

        # narration trimming - keep it short for UI
        raw_n = plan_obj.get("narration", "") if isinstance(plan_obj, dict) else ""
        sentences = [s.strip() for s in raw_n.replace("\n", " ").split(".") if s.strip()]
        truncated = ". ".join(sentences[:3])
        if len(truncated) > 300:
            truncated = truncated[:297].rstrip() + "..."
        self.last_narration = truncated

        # try to reserve the first few pickups so other agents don't duplicate work
        try:
            self._reserve_plan_pickups()
        except Exception:
            pass

        # debug log so you can inspect planner decisions
        try:
            print("[PLANNER] plan received:", self.plan)
            print("[PLANNER] narration:", self.last_narration)
        except Exception:
            pass

    # ---------------------------
    # Plan helpers
    # ---------------------------
    def _current_step(self) -> Optional[Dict]:
        if 0 <= self.plan_idx < len(self.plan):
            return self.plan[self.plan_idx]
        return None

    def _advance_plan(self):
        # When we advance a plan, also release any reservation that we may have held
        step = self._current_step()
        if step:
            pickup = tuple(step["pickup"])
            self._release_reservation(pickup[0], pickup[1])

        self.plan_idx += 1
        if self.plan_idx >= len(self.plan):
            # plan exhausted
            self.plan = []
            self.plan_idx = 0
            self.state = "idle"

    def _ensure_energy_for_route(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int],
                                 carry_weight: float = 0.0, require_return: bool = False) -> bool:
        """
        Conservative energy check for the leg from_cell -> to_cell.
        If require_return=True it also checks ability to return home (not used when ALLOW_RISKY_TRIPS=True).
        When ALLOW_RISKY_TRIPS is True, the controller will allow immediate legs (bypass check).
        """
        # If risky mode and only checking immediate leg, allow the trip unconditionally.
        if ALLOW_RISKY_TRIPS and not require_return:
            # keep a debug trace but don't spam too often
            if random.random() < 0.02:
                print(f"[AI] bypassing energy check (risky mode) for leg {from_cell} -> {to_cell} (require_return={require_return})")
            return True

        # conservative Manhattan estimate for energy needed (controller-side)
        dist = abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])
        needed = self.drone.energy_needed_for_cells(dist, carry_weight)

        if require_return:
            # determine nearest station as "home" for return calculation
            station = self.terrain.nearest_station(from_cell[0], from_cell[1])
            if station:
                home_col = int(station.col + station.w // 2)
                home_row = int(station.row + station.h // 2)
                dist_back = abs(to_cell[0] - home_col) + abs(to_cell[1] - home_row)
                needed += self.drone.energy_needed_for_cells(dist_back, 0.0)

        # conservative check
        return (self.drone.power.level >= needed)

    # ---------------------------
    # Delivery cell chooser (unchanged)
    # ---------------------------
    def _choose_delivery_cell(self) -> Tuple[int, int]:
        """
        Choose a delivery cell. Preference order:
         1. least-used free cell inside nearest station (if any)
         2. other free station cell avoiding repeating the last chosen
         3. random free cell on map avoiding last chosen if possible
         4. fallback to current drone cell
        Stores chosen into self._last_chosen_drop for biasing future choices.
        """
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)

        def _set_and_return(cell: Tuple[int, int]) -> Tuple[int, int]:
            self._last_chosen_drop = cell
            return cell

        if station:
            # prefer least-used free cell (station helper provided in artifacts if implemented)
            try:
                least_used = station.least_used_free_cell(self.terrain)
            except Exception:
                least_used = None

            if least_used:
                return _set_and_return(least_used)

            # otherwise pick a free cell but avoid repeating last chosen if possible
            cells = [(c, r) for r in range(station.row, station.row + station.h)
                     for c in range(station.col, station.col + station.w)]
            free_cells = [c for c in cells if not self.terrain.occupied_cell(c[0], c[1])]
            if free_cells:
                choices = [c for c in free_cells if c != self._last_chosen_drop]
                if not choices:
                    choices = free_cells
                return _set_and_return(random.choice(choices))

            # no free cells -> pick some station cell avoiding repeats
            choices = [c for c in cells if c != self._last_chosen_drop]
            if choices:
                return _set_and_return(random.choice(choices))
            return _set_and_return(random.choice(cells))

        # No station present: sample free cells on the map
        cols = self.terrain.screen_size[0] // self.terrain.grid_size
        rows = self.terrain.screen_size[1] // self.terrain.grid_size
        attempts = 0
        chosen = None
        while attempts < 400:
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)
            if (c, r) == (self.drone.col, self.drone.row):
                attempts += 1
                continue
            if self.terrain.is_station_cell(c, r):
                attempts += 1
                continue
            if not self.terrain.occupied_cell(c, r) and (c, r) != self._last_chosen_drop:
                chosen = (c, r)
                break
            # relax after many attempts
            if attempts > 50 and not self.terrain.occupied_cell(c, r):
                chosen = (c, r)
                break
            attempts += 1

        if chosen is None:
            chosen = (self.drone.col, self.drone.row)
        return _set_and_return(chosen)

    # ---------------------------
    # Main update loop (modified to use reservations + greedy fallback)
    # ---------------------------
    def update(self, dt: float):
        # cooldown timer
        if self.cooldown > 0:
            self.cooldown -= dt

        # reservation cleanup on a timer
        try:
            self._cleanup_reservations()
        except Exception:
            pass

        # planner backoff decays over time
        if self._planner_backoff > 0 and (time.time() - self._last_plan_time) > self._planner_backoff:
            self._planner_backoff = 0.0

        if self.drone.lost:
            self.last_status = "lost"
            return

        now = time.time()

        # --- IMMEDIATE pick/drop while not moving ---
        if not self.drone.moving:
            # If not carrying, attempt local pickup if there is a parcel here
            if self.drone.carrying is None:
                p_here = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p_here:
                    # If we don't own the reservation, try to reserve it now before picking.
                    # If someone else has it, we will not pick.
                    if not self._reserved_by_me(p_here.col, p_here.row):
                        got = self._try_reserve(p_here.col, p_here.row)
                        if not got:
                            self.last_status = f"local parcel reserved by other {p_here.col,p_here.row}"
                            # back off briefly
                            self.cooldown = max(self.cooldown, 0.3)
                            # but continue to try greedy elsewhere below
                        else:
                            # we own it - but ensure we actually have energy to pick
                            if not self._can_attempt_parcel(p_here):
                                # release reservation - cannot attempt
                                self._release_reservation(p_here.col, p_here.row)
                                self.last_status = "local parcel but insufficient energy"
                                # fallthrough to greedy/plan logic
                            else:
                                # attempt pick immediately
                                self.last_status = f"arrived at parcel cell {(self.drone.col, self.drone.row)} picking"
                                success = self.drone.perform_pick(p_here)
                                if not success:
                                    # pick failed; mark UI flash via last_action is handled by perform_pick (returns False on lost)
                                    self.drone._last_action = ("pick_failed", (self.drone.col, self.drone.row), None)
                                    # release reservation since we couldn't pick
                                    self._release_reservation(p_here.col, p_here.row)
                                    # consider returning to station only when there are NO field parcels remaining
                                    station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                                    if station and not self._field_parcels_remaining():
                                        home = (station.col + station.w // 2, station.row + station.h // 2)
                                        self.drone.set_target_cell(*home)
                                        self.state = "returning"
                                    return
                                else:
                                    # picked successfully; release reservation (parcel now marked picked) and update state
                                    self._release_reservation(p_here.col, p_here.row)
                                    self.state = "carrying"
                                    self._last_action_time = now
                                    self.last_status = "picked parcel"
                                    return

            # If carrying, attempt immediate drop if we are on a station / drop cell and cell is free
            else:
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station and station.contains_cell(self.drone.col, self.drone.row):
                    # only drop if cell is not occupied by another undelivered parcel
                    if not self.terrain.occupied_cell(self.drone.col, self.drone.row):
                        self.last_status = f"arrived at station {(self.drone.col, self.drone.row)} dropping"
                        parcel_ref = self.drone.carrying
                        self.drone.perform_drop(parcel_ref)
                        self._last_chosen_drop = (self.drone.col, self.drone.row)
                        step = self._current_step()
                        if step and tuple(step["dropoff"]) == (self.drone.col, self.drone.row):
                            self._advance_plan()
                        # after drop, release any reservations referring to this parcel cell (defensive)
                        try:
                            RESERVATIONS.pop((int(parcel_ref.col), int(parcel_ref.row)), None)
                        except Exception:
                            pass
                        return
                # else: not in station or occupied -> don't force drop; let normal logic decide

        # ---- If we reach here we are either moving OR not moving but there's nothing to immediately pick/drop.
        # Now handle planning & navigation logic (request plan, follow plan, energy checks, etc.)

        # request plan if none or time to replan (with planner backoff)
        time_to_replan = (now - self._last_plan_time) > self._replanning_interval
        if (not self.plan) or time_to_replan:
            try:
                self._request_plan(force_refresh=False)
            except Exception:
                pass

        # if drone currently moving wait for arrival (arrival frame above handles pick/drop)
        if self.drone.moving:
            self.last_status = "moving"
            return

        # ---------- carrying branch: aim to drop according to plan ----------
        if self.drone.carrying:
            step = self._current_step()
            # deduce planned dropoff: prefer plan, else nearest station center or chooser
            if step:
                planned_drop = tuple(step["dropoff"])
            else:
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    try:
                        alt = station.least_used_free_cell(self.terrain)
                    except Exception:
                        alt = None
                    if alt:
                        planned_drop = alt
                    else:
                        planned_drop = (station.col + station.w // 2, station.row + station.h // 2) if station else (self.drone.col, self.drone.row)
                else:
                    planned_drop = (self.drone.col, self.drone.row)

            # If planned_drop is occupied or equals the last chosen drop, try alternatives
            if self.terrain.occupied_cell(planned_drop[0], planned_drop[1]) or \
                    (self._last_chosen_drop is not None and planned_drop == self._last_chosen_drop):
                station_for_drop = self.terrain.nearest_station(planned_drop[0], planned_drop[1])
                if station_for_drop:
                    try:
                        alt = station_for_drop.least_used_free_cell(self.terrain)
                    except Exception:
                        alt = None
                    if alt:
                        planned_drop = alt
                    else:
                        planned_drop = self._choose_delivery_cell()
                else:
                    planned_drop = self._choose_delivery_cell()

            # energy check: can we reach planned_drop while carrying?
            # When ALLOW_RISKY_TRIPS=True we only require energy to reach planned_drop (not to return)
            enough = self._ensure_energy_for_route(
                (self.drone.col, self.drone.row),
                planned_drop,
                carry_weight=getattr(self.drone.carrying, "weight", 0.0),
                require_return=(not ALLOW_RISKY_TRIPS)
            )

            if not enough:
                # Only auto-return to station if there are NO undelivered field parcels left.
                if not ALLOW_RISKY_TRIPS:
                    station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                    if station and not self._field_parcels_remaining():
                        home = (station.col + station.w // 2, station.row + station.h // 2)
                        self.drone.set_target_cell(*home)
                        self.state = "returning"
                        return
                    else:
                        # if parcels remain, do not force return; either replan or allow risky behaviour
                        if self._field_parcels_remaining():
                            # try to replan or attempt risky (if enabled)
                            if ALLOW_RISKY_TRIPS:
                                if not self.drone.moving:
                                    self.last_status = f"attempting risky drop to {planned_drop}"
                                    self.drone.set_target_cell(planned_drop[0], planned_drop[1])
                                return
                            else:
                                # conservative: request replan
                                try:
                                    self._request_plan(force_refresh=True)
                                except Exception:
                                    pass
                                return
                        # if no field parcels remain, fallback handled above
                else:
                    # ALLOW_RISKY_TRIPS -> attempt the trip anyway (drone may die en-route)
                    if not self.drone.moving:
                        self.last_status = f"attempting risky drop to {planned_drop}"
                        self.drone.set_target_cell(planned_drop[0], planned_drop[1])
                    return

            # if at planned drop cell, drop (if still free)
            if (self.drone.col, self.drone.row) == planned_drop:
                if not self.terrain.occupied_cell(planned_drop[0], planned_drop[1]):
                    parcel_ref = self.drone.carrying
                    self.drone.perform_drop(parcel_ref)
                    # record chosen drop to bias away in future
                    self._last_chosen_drop = planned_drop
                    # station accounting (station.register_delivery) handled by game loop when receiving last_action
                    if step and tuple(step["dropoff"]) == planned_drop:
                        self._advance_plan()
                    # After dropping, clear any reservation related to the parcel pickup cell as defensive cleanup
                    try:
                        RESERVATIONS.pop((int(parcel_ref.col), int(parcel_ref.row)), None)
                    except Exception:
                        pass
                    return
                else:
                    # occupied unexpectedly, replan
                    try:
                        self._request_plan(force_refresh=True)
                    except Exception:
                        pass
                    return

            # otherwise, set target to planned_drop
            if not self.drone.moving:
                self.drone.set_target_cell(planned_drop[0], planned_drop[1])
                self.last_status = f"heading to drop {planned_drop}"
            return

        # ---------- not carrying branch: attempt pickup steps ----------
        step = self._current_step()
        if not step:
            # No planner step — try greedy fallback to nearest unreserved parcel.
            self.state = "idle"

            # If we already have a reservation for some parcel, go to it
            my_reserved = None
            for (rc, rr), (owner, ts) in list(RESERVATIONS.items()):
                if owner is self:
                    my_reserved = (rc, rr)
                    break
            if my_reserved and not self.drone.moving:
                # set target to my reserved pickup
                self.drone.set_target_cell(my_reserved[0], my_reserved[1])
                self.state = "seeking"
                self.last_status = f"going to my reserved {my_reserved}"
                return

            # otherwise attempt to find and claim the nearest parcel that we can attempt now
            parcel = None
            try:
                parcel = self._find_nearest_unreserved_parcel()
            except Exception:
                parcel = None

            if parcel:
                success = self._try_claim_and_go_to_parcel(parcel)
                if success:
                    return
                else:
                    # failed to claim (some race). Try to replan next cycle (force) after a small cooldown
                    try:
                        self._request_plan(force_refresh=True)
                    except Exception:
                        pass
                    self.cooldown = max(self.cooldown, 0.2)
                    self.last_status = "greedy claim failed, will replan"
                    return

            # no suitable parcel found -> stay idle
            self.last_status = "idle (no available parcels I can reach)"
            return

        # If we have a planner step, continue with the normal planner-following behavior
        pickup = tuple(step["pickup"])

        # verify pickup still available (not picked, not delivered)
        pobj = self.terrain.parcel_at_cell(pickup[0], pickup[1])
        if pobj is None or getattr(pobj, "delivered", False):
            # advance plan and consider replanning
            # also release any reservation we might have held
            self._release_reservation(pickup[0], pickup[1])
            self._advance_plan()
            if (now - self._last_plan_time) > self._replanning_interval:
                try:
                    self._request_plan(force_refresh=True)
                except Exception:
                    pass
            return

        # If the planned pickup is reserved by someone else, skip ahead in plan or fallback to greedy immediately.
        if self._is_reserved_by_other(pickup[0], pickup[1]):
            # try to skip to next step in plan that is not reserved by other
            skipped = False
            for look_idx in range(self.plan_idx + 1, len(self.plan)):
                next_pick = tuple(self.plan[look_idx]["pickup"])
                if not self._is_reserved_by_other(next_pick[0], next_pick[1]):
                    self.plan_idx = look_idx
                    skipped = True
                    break
            if not skipped:
                # no future unreserved step in plan -> force a replan now (or greedy fallback)
                try:
                    self._request_plan(force_refresh=True)
                except Exception:
                    pass
                # also attempt greedy immediately if planner didn't produce anything useful
                try:
                    parcel = self._find_nearest_unreserved_parcel()
                    if parcel:
                        if self._try_claim_and_go_to_parcel(parcel):
                            return
                except Exception:
                    pass
                self.cooldown = max(self.cooldown, 0.2)
                self.last_status = f"planner step {pickup} reserved by other, replanned/greedy attempted"
                return

        # Before we commit to traveling to the pickup, attempt to reserve it.
        # If someone else already has the reservation, we should avoid traveling there.
        if not self._reserved_by_me(pickup[0], pickup[1]):
            # Try to reserve. If it fails, skip this pickup (advance or replan).
            got = self._try_reserve(pickup[0], pickup[1])
            if not got:
                # somebody else has claimed it — replan or skip
                self.last_status = f"pickup {pickup} reserved by other"
                # back off a little: force a replan to choose a different target next cycle
                if (now - self._last_plan_time) > self._replanning_interval:
                    try:
                        self._request_plan(force_refresh=True)
                    except Exception:
                        pass
                else:
                    # small cooldown so we don't spam planner
                    self.cooldown = max(self.cooldown, 0.3)
                # ALSO try greedy fallback immediately to encourage concurrency
                try:
                    parcel = self._find_nearest_unreserved_parcel()
                    if parcel and (int(parcel.col), int(parcel.row)) != pickup:
                        if self._try_claim_and_go_to_parcel(parcel):
                            return
                except Exception:
                    pass
                return
            else:
                # we successfully reserved it (timestamp refreshed inside _try_reserve)
                self.last_status = f"reserved pickup {pickup}"

        dropoff = tuple(step["dropoff"])
        # energy checks: current->pickup then pickup->dropoff (with weight)
        # NOTE: when ALLOW_RISKY_TRIPS=True we do not require enough energy to also return home.
        can_reach_pick = self._ensure_energy_for_route((self.drone.col, self.drone.row), pickup,
                                                       carry_weight=0.0,
                                                       require_return=False)
        can_reach_after_pick = self._ensure_energy_for_route(pickup, dropoff,
                                                             carry_weight=step.get("weight", 1.0),
                                                             require_return=(not ALLOW_RISKY_TRIPS))
        if not (can_reach_pick and can_reach_after_pick):
            # if conservative mode, return to station to recharge only if there are NO field parcels left
            station = self.terrain.nearest_station(self.drone.col, self.drone.row)
            if station and not ALLOW_RISKY_TRIPS and not self._field_parcels_remaining():
                home = (station.col + station.w // 2, station.row + station.h // 2)
                # release reservation since we're giving up this pickup for now
                self._release_reservation(pickup[0], pickup[1])
                self.drone.set_target_cell(*home)
                self.state = "returning"
                return
            else:
                # either replan or attempt risky pickup
                if ALLOW_RISKY_TRIPS:
                    if not self.drone.moving:
                        self.last_status = f"attempting risky pickup at {pickup}"
                        self.drone.set_target_cell(pickup[0], pickup[1])
                    return
                else:
                    # conservative and there are field parcels -> replan
                    # release reservation before asking for a replan, so others may take it
                    self._release_reservation(pickup[0], pickup[1])
                    try:
                        self._request_plan(force_refresh=True)
                    except Exception:
                        pass
                    return

        # if at pickup cell -> pick using drone.perform_pick (handled above but keep fallback)
        if (self.drone.col, self.drone.row) == pickup:
            success = self.drone.perform_pick(pobj)
            if success:
                # picked, now the carrying branch will run next frame
                # release reservation (parcel now picked)
                self._release_reservation(pickup[0], pickup[1])
                self.state = "carrying"
                self._last_action_time = now
                self.last_status = "picked (fallback)"
                return
            else:
                # pick failed (low battery) -> release reservation and act according to remaining parcels
                self._release_reservation(pickup[0], pickup[1])
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station and not self._field_parcels_remaining():
                    home = (station.col + station.w // 2, station.row + station.h // 2)
                    self.drone.set_target_cell(*home)
                    return
                else:
                    self.drone.lost = True
                    return

        # otherwise set target to pickup (we already reserved it above)
        if not self.drone.moving:
            # validate target parcel hasn't been taken between decision and set_target
            if pobj.picked or getattr(pobj, "delivered", False):
                # someone else took it; release reservation and advance plan next loop
                self._release_reservation(pickup[0], pickup[1])
                self._advance_plan()
                return
            self.drone.set_target_cell(pickup[0], pickup[1])
            self.state = "seeking"
            self.last_status = f"seeking pickup {pickup}"
            return


class ControllerSwitcher:
    """Simple switcher between controllers (Human/AI). TAB cycles."""
    def __init__(self, controllers: List[BaseAgentController]):
        assert controllers, "provide at least one controller"
        self.controllers = controllers
        self.index = 0

    @property
    def current(self) -> BaseAgentController:
        return self.controllers[self.index]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self.index = (self.index + 1) % len(self.controllers)
        else:
            self.current.handle_event(event)

    def update(self, dt: float):
        self.current.update(dt)
