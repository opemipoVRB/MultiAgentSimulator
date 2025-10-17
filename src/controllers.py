# src/controllers.py
import pygame
import random


class BaseAgentController:
    def __init__(self, drone, terrain):
        self.drone = drone
        self.terrain = terrain

    def handle_event(self, event):
        pass

    def update(self, dt):
        pass


class HumanAgentController(BaseAgentController):
    def __init__(self, drone, terrain):
        super().__init__(drone, terrain)
        self._last_keys = pygame.key.get_pressed()

    def handle_event(self, event):
        pass

    def update(self, dt):
        keys = pygame.key.get_pressed()

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
                self.drone.set_target_cell(new_col, new_row)

        # Edge-detect SPACE for pick/drop
        if keys[pygame.K_SPACE] and not self._last_keys[pygame.K_SPACE]:
            if self.drone.carrying is None:
                p = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p:
                    p.picked = True
                    self.drone.carrying = p
                    # announce pick action for UI/main loop
                    self.drone._last_action = ("pick", (self.drone.col, self.drone.row), p)
            else:
                existing = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if not existing:
                    # drop (same logic as AI uses)
                    self._do_drop(self.drone.col, self.drone.row)

        self._last_keys = keys

    def _do_drop(self, col, row):
        """Perform drop for human controller and announce to main loop.

        If the target cell already contains an undelivered parcel, abort the drop
        and signal a drop-failed action so the UI can flash/notify.
        """
        # check cell occupancy of undelivered/unpicked parcel
        existing = self.terrain.parcel_at_cell(col, row)
        if existing:
            # abort drop and announce failure (no counting)
            self.drone._last_action = ("drop_failed", (col, row), None)
            return

        # parcel being dropped
        parcel = self.drone.carrying
        if not parcel:
            return

        # place the parcel onto this cell
        parcel.col = col
        parcel.row = row
        parcel.pos = pygame.Vector2(col * self.drone.grid_size + self.drone.grid_size / 2,
                                    row * self.drone.grid_size + self.drone.grid_size / 2)
        parcel.picked = False

        # clear drone carry
        self.drone.carrying = None

        # announce drop and include parcel reference so main loop can mark it exactly once
        self.drone._last_action = ("drop", (col, row), parcel)


class AIAgentController(BaseAgentController):
    """Simple autonomous controller.

    Behavior:
      - If not carrying: find nearest unpicked, undelivered parcel and go to it.
      - If carrying: deliver to a free cell inside nearest station if present, otherwise choose random free cell.
      - Revalidates targets so it does not get stuck on occupied or stale targets.
    """

    def __init__(self, drone, terrain, search_radius=None):
        super().__init__(drone, terrain)
        self.state = "idle"  # 'seek', 'deliver', 'idle'
        self.target_parcel = None
        self.target_cell = None
        self.cooldown = 0.0
        self.search_radius = search_radius

    def handle_event(self, event):
        pass

    def update(self, dt):
        # cooldown to avoid spamming commands
        if self.cooldown > 0:
            self.cooldown -= dt

        # if drone actively moving, wait for arrival
        if self.drone.moving:
            return

        # If carrying, ensure target_cell is valid. If not, re-choose.
        if self.drone.carrying:
            if self.state != "deliver":
                self.state = "deliver"
                self._choose_delivery_cell()

            # re-validate target cell: if another undelivered parcel appeared there, pick another cell
            if self.target_cell is not None:
                occ = self.terrain.parcel_at_cell(self.target_cell[0], self.target_cell[1])
                if occ:
                    self._choose_delivery_cell()

            # If arrived on target cell, attempt drop (only if cell currently free of undelivered parcel)
            if (self.target_cell is not None) and (self.drone.col, self.drone.row) == self.target_cell:
                existing = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if not existing:
                    # perform drop using the same logic as human controller
                    self._do_drop(self.drone.col, self.drone.row)

                    # reset internal state so AI continues searching next parcel
                    self.state = "idle"
                    self.target_parcel = None
                    self.target_cell = None
                    self.cooldown = 0.15
                    return
                else:
                    # target cell unexpectedly occupied, re-choose and return to avoid loop
                    self._choose_delivery_cell()
                    return

            # otherwise, set target to delivery cell if not moving
            if self.target_cell is not None and not self.drone.moving:
                self.drone.set_target_cell(*self.target_cell)
                return

        # not carrying: seek nearest available parcel
        if not self.terrain.parcels:
            self.state = "idle"
            return

        # ignore parcels that are picked or delivered
        candidates = [p for p in self.terrain.parcels if not p.picked and not getattr(p, "delivered", False)]
        if not candidates:
            self.state = "idle"
            return

        # pick nearest by squared cell distance
        def cell_dist(p):
            dx = p.col - self.drone.col
            dy = p.row - self.drone.row
            return dx * dx + dy * dy

        candidates.sort(key=cell_dist)

        # validate previously chosen parcel
        if self.target_parcel and (self.target_parcel.picked or getattr(self.target_parcel, "delivered", False)):
            self.target_parcel = None

        parcel = self.target_parcel if self.target_parcel else candidates[0]
        self.target_parcel = parcel
        self.state = "seek"

        # if already on parcel cell -> pick it up immediately
        if (parcel.col, parcel.row) == (self.drone.col, self.drone.row):
            parcel.picked = True
            self.drone.carrying = parcel
            # announce pick
            self.drone._last_action = ("pick", (self.drone.col, self.drone.row), parcel)
            # switch to deliver state and choose a delivery cell
            self.state = "deliver"
            self._choose_delivery_cell()
            self.cooldown = 0.1
            return

        # otherwise move toward parcel (re-validate target just before issuing move)
        if not self.drone.moving:
            if parcel.picked or getattr(parcel, "delivered", False):
                # choose another parcel next frame
                self.target_parcel = None
                return
            self.drone.set_target_cell(parcel.col, parcel.row)

    def _do_drop(self, col, row):
        """
        Place the carried parcel onto the specified cell and announce drop.
        """
        # if target cell has an undelivered/unpicked parcel, do not drop here
        if self.terrain.parcel_at_cell(col, row):
            # choose another delivery cell and do not announce drop
            self._choose_delivery_cell()
            return

        parcel = self.drone.carrying
        if not parcel:
            return

        parcel.col = col
        parcel.row = row
        parcel.pos = pygame.Vector2(col * self.drone.grid_size + self.drone.grid_size / 2,
                                    row * self.drone.grid_size + self.drone.grid_size / 2)
        parcel.picked = False

        # clear drone carry but keep parcel reference to send to main loop
        self.drone.carrying = None

        # announce drop so main loop can flash and count deliveries, include parcel ref
        self.drone._last_action = ("drop", (col, row), parcel)

    def _choose_delivery_cell(self):
        """Choose a delivery cell. Prefer a free cell inside the nearest station; otherwise pick a free cell anywhere."""
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
        if station:
            # collect all cells inside station
            cells = []
            for r in range(station.row, station.row + station.h):
                for c in range(station.col, station.col + station.w):
                    cells.append((c, r))

            # prefer cells inside the station that are currently free of undelivered parcels
            free_cells = [cell for cell in cells if self.terrain.parcel_at_cell(cell[0], cell[1]) is None]

            if free_cells:
                self.target_cell = random.choice(free_cells)
                return
            else:
                # no completely free cells - pick a random station cell anyway to avoid stalling
                self.target_cell = random.choice(cells)
                return

        # no station: pick a random free cell on the map
        cols = self.terrain.screen_size[0] // self.terrain.grid_size
        rows = self.terrain.screen_size[1] // self.terrain.grid_size
        attempts = 0
        while attempts < 400:
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)
            if self.terrain.parcel_at_cell(c, r) is None and not self.terrain.is_station_cell(c, r) and (c, r) != (
            self.drone.col, self.drone.row):
                self.target_cell = (c, r)
                return
            attempts += 1
        # fallback: drop where we are
        self.target_cell = (self.drone.col, self.drone.row)


class ControllerSwitcher:
    def __init__(self, controllers):
        assert controllers, "provide at least one controller"
        self.controllers = controllers
        self.index = 0

    @property
    def current(self):
        return self.controllers[self.index]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self.index = (self.index + 1) % len(self.controllers)
        else:
            self.current.handle_event(event)

    def update(self, dt):
        self.current.update(dt)
