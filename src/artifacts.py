# src/artifacts.py
import pygame
import random


class Parcel:
    def __init__(self, col, row, grid_size):
        self.col = col
        self.row = row
        self.grid_size = grid_size
        self.pos = pygame.Vector2(col * grid_size + grid_size / 2, row * grid_size + grid_size / 2)
        self.picked = False
        self.delivered = False  # new flag

    def draw(self, surf, parcel_img=None, parcel_scale=0.7):
        # Always draw delivered parcels, but visually distinct
        if self.delivered:
            # Draw delivered parcel smaller and dimmer / semi-transparent
            if parcel_img:
                img = parcel_img.copy()
                # approximate dim by filling an alpha surface; simpler: blit with 160 alpha
                img.set_alpha(180)  # make it slightly transparent
                rect = img.get_rect(center=(int(self.pos.x), int(self.pos.y)))
                surf.blit(img, rect)
            else:
                s = int(self.grid_size * parcel_scale * 0.5)
                rect = (int(self.pos.x) - s // 2, int(self.pos.y) - s // 2, s, s)
                # draw darker/dim color to indicate delivered
                pygame.draw.rect(surf, (140, 120, 60), rect)
                pygame.draw.rect(surf, (100, 100, 100), rect, 2)
            return

        # normal (undelivered) parcels: only draw if not picked
        if self.picked:
            return
        if parcel_img:
            rect = parcel_img.get_rect(center=(int(self.pos.x), int(self.pos.y)))
            surf.blit(parcel_img, rect)
        else:
            s = int(self.grid_size * parcel_scale * 0.6)
            pygame.draw.rect(surf, (200, 160, 60), (int(self.pos.x) - s // 2, int(self.pos.y) - s // 2, s, s))


class Drone:
    """
    Lightweight Drone agent. Drawing is delegated to game code via an images dict.
    """

    def __init__(self, start_cell, grid_size, screen_size):
        self.col, self.row = start_cell
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.pos = pygame.Vector2(self.col * grid_size + grid_size / 2, self.row * grid_size + grid_size / 2)
        self.target = None
        self.moving = False
        self.carrying = None  # reference to Parcel
        self.anim_t = 0.0
        self.anim_frame = 0
        self._last_action = None

    def set_target_cell(self, col, row):
        max_col = (self.screen_size[0] // self.grid_size) - 1
        max_row = (self.screen_size[1] // self.grid_size) - 1
        col = max(0, min(col, max_col))
        row = max(0, min(row, max_row))
        if (col, row) == (self.col, self.row):
            return
        self.target = pygame.Vector2(col * self.grid_size + self.grid_size / 2, row * self.grid_size + self.grid_size / 2)
        self.moving = True

    def update(self, dt, speed, anim_fps, rot_frames_with_parcel, rot_frames):
        if self.moving and self.target:
            direction = self.target - self.pos
            dist = direction.length()
            if dist == 0:
                self.arrive()
            else:
                step = speed * dt
                if step >= dist:
                    self.pos = self.target
                    self.arrive()
                else:
                    self.pos += direction.normalize() * step

            self.anim_t += dt
            if self.anim_t >= 1 / anim_fps:
                self.anim_t = 0.0
                if self.carrying and rot_frames_with_parcel:
                    self.anim_frame = (self.anim_frame + 1) % len(rot_frames_with_parcel)
                elif rot_frames:
                    self.anim_frame = (self.anim_frame + 1) % len(rot_frames)
                else:
                    self.anim_frame = 0
        else:
            self.anim_frame = 0

    def arrive(self):
        self.col = int(self.pos.x // self.grid_size)
        self.row = int(self.pos.y // self.grid_size)
        self.target = None
        self.moving = False
        self.pos = pygame.Vector2(self.col * self.grid_size + self.grid_size / 2, self.row * self.grid_size + self.grid_size / 2)

    def draw(self, surf, images):
        """
        images: dict with optional keys:
          - drone_static
          - drone_rot_frames (list)
          - drone_static_with_parcel
          - drone_rot_with_parcel_frames (list)
          - parcel_img
        """
        x, y = int(self.pos.x), int(self.pos.y)
        img = None

        if self.carrying:
            if self.moving and images.get("drone_rot_with_parcel_frames"):
                frames = images["drone_rot_with_parcel_frames"]
                img = frames[self.anim_frame % len(frames)]
            elif not self.moving and images.get("drone_static_with_parcel"):
                img = images["drone_static_with_parcel"]
            elif self.moving and images.get("drone_rot_frames"):
                frames = images["drone_rot_frames"]
                img = frames[self.anim_frame % len(frames)]
            else:
                img = images.get("drone_static")
        else:
            if self.moving and images.get("drone_rot_frames"):
                frames = images["drone_rot_frames"]
                img = frames[self.anim_frame % len(frames)]
            else:
                img = images.get("drone_static")

        if img:
            rect = img.get_rect(center=(x, y))
            surf.blit(img, rect)
        else:
            pygame.draw.circle(surf, (3, 54, 96), (x, y), int(self.grid_size * 0.35))

        # overlay parcel if carrying and no dedicated with-parcel image
        if self.carrying and not images.get("drone_static_with_parcel") and not images.get("drone_rot_with_parcel_frames"):
            parcel_img = images.get("parcel_img")
            if parcel_img:
                rect = parcel_img.get_rect(center=(x, y + int(self.grid_size * 0.18)))
                surf.blit(parcel_img, rect)
            else:
                s = int(self.grid_size * 0.7 * 0.6)
                pygame.draw.rect(surf, (200, 160, 60), (x - s // 2, y + int(self.grid_size * 0.18) - s // 2, s, s))


class DeliveryStation:
    """Delivery station occupying a rectangular block of grid cells."""

    def __init__(self, col, row, grid_size, w=2, h=2):
        """
        col,row - top-left cell of the station block
        w,h - width and height in grid cells
        """
        self.col = col
        self.row = row
        self.w = max(1, int(w))
        self.h = max(1, int(h))
        self.grid_size = grid_size
        # center position for distance calculations
        center_col = col + (self.w - 1) / 2
        center_row = row + (self.h - 1) / 2
        self.pos = pygame.Vector2(center_col * grid_size + grid_size / 2, center_row * grid_size + grid_size / 2)
        self.delivered = 0  # counter for deliveries to this station

    def contains_cell(self, col, row):
        return (self.col <= col < self.col + self.w) and (self.row <= row < self.row + self.h)

    def draw(self, surf):
        # draw semi-transparent fill and bold border covering the station rectangle
        x = self.col * self.grid_size
        y = self.row * self.grid_size
        width = self.w * self.grid_size
        height = self.h * self.grid_size

        # translucent fill
        s = pygame.Surface((width, height), pygame.SRCALPHA)
        s.fill((30, 110, 200, 60))  # translucent blue
        surf.blit(s, (x, y))

        # border
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surf, (40, 120, 200), rect, 3)

        # draw delivered count in corner
        font = pygame.font.SysFont("Consolas", max(12, self.grid_size // 6))
        txt = font.render(str(self.delivered), True, (255, 255, 255))
        surf.blit(txt, (x + 6, y + 6))


class Terrain:
    def __init__(self, grid_size, screen_size, parcel_img=None, parcel_scale=0.7):
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.parcel_img = parcel_img
        self.parcel_scale = parcel_scale
        self.parcels = []
        self.stations = []

    def spawn_random(self, n):
        cols = self.screen_size[0] // self.grid_size
        rows = self.screen_size[1] // self.grid_size
        for _ in range(n):
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)
            self.add_parcel(c, r)

    def add_parcel(self, col, row):
        # do not add parcels on top of station blocks
        if self.parcel_at_cell(col, row) is None and not self.is_station_cell(col, row):
            self.parcels.append(Parcel(col, row, self.grid_size))

    def parcel_at_cell(self, col, row):
        # return an undelivered, unpicked parcel at the cell (or None)
        for p in self.parcels:
            if (not p.picked) and (not p.delivered) and p.col == col and p.row == row:
                return p
        return None

    def add_station(self, col, row, w=2, h=2):
        """
        Add a station with top-left cell (col,row) and width w, height h in cells.
        """
        # ensure station fits in screen bounds
        max_col = (self.screen_size[0] // self.grid_size) - 1
        max_row = (self.screen_size[1] // self.grid_size) - 1
        col = max(0, min(col, max_col))
        row = max(0, min(row, max_row))
        w = max(1, min(w, max_col - col + 1))
        h = max(1, min(h, max_row - row + 1))
        self.stations.append(DeliveryStation(col, row, self.grid_size, w=w, h=h))

    def is_station_cell(self, col, row):
        for s in self.stations:
            if s.contains_cell(col, row):
                return True
        return False

    def get_station_at(self, col, row):
        for s in self.stations:
            if s.contains_cell(col, row):
                return s
        return None

    def nearest_station(self, col, row):
        if not self.stations:
            return None
        best = None
        best_dist = None
        for s in self.stations:
            dx = s.pos.x - (col * self.grid_size + self.grid_size / 2)
            dy = s.pos.y - (row * self.grid_size + self.grid_size / 2)
            d = dx * dx + dy * dy
            if best is None or d < best_dist:
                best = s
                best_dist = d
        return best

    def draw(self, surf):
        for p in self.parcels:
            p.draw(surf, parcel_img=self.parcel_img, parcel_scale=self.parcel_scale)
        for s in self.stations:
            s.draw(surf)
