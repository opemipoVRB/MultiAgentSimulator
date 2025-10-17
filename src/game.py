import pygame
import sys
import random
from pathlib import Path

from artifacts import Terrain, Drone
from utils import load_image, scale_to_cell
from controllers import HumanAgentController, AIAgentController, ControllerSwitcher

pygame.init()
# Start fullscreen
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
SCREEN_W, SCREEN_H = screen.get_size()
pygame.display.set_caption("Drone Agent - Parcel Pickup/Drop (Fullscreen)")
clock = pygame.time.Clock()

# Config - increased cell size and larger drone, smaller parcels
GRID_SIZE = 100  # cells slightly bigger
SPEED = 420  # drone speed in pixels per second
DRONE_SCALE = 1.25  # make drone larger relative to cell
PARCEL_SCALE = 0.70  # parcels smaller relative to cell
ANIM_FPS = 12
FLASH_DURATION = 0.15  # seconds for cell flash feedback
# Assets live in the `graphics` folder next to this script (src/graphics)
ASSET_DIR = Path(__file__).parent / "graphics"


def load_and_scale(name, scale_factor):
    p = ASSET_DIR / name
    img = load_image(p)
    return scale_to_cell(img, GRID_SIZE, scale_factor) if img else None


# === Load images for drone states ===
drone_static = load_and_scale("drone_static.png", DRONE_SCALE)
# rotating frames (drone_rotating_0.png, ...)
drone_rot_frames = []
i = 0
while True:
    p = ASSET_DIR / f"drone_rotating_{i}.png"
    if p.exists():
        drone_rot_frames.append(scale_to_cell(pygame.image.load(str(p)).convert_alpha(), GRID_SIZE, DRONE_SCALE))
        i += 1
    else:
        break
if not drone_rot_frames:
    r = load_image(ASSET_DIR / "drone_rotating.png")
    if r:
        drone_rot_frames = [scale_to_cell(r, GRID_SIZE, DRONE_SCALE)]

# With parcel images

drone_static_with_parcel = load_and_scale("drone_static_with_parcel.png", DRONE_SCALE)
drone_rot_with_parcel_frames = []
i = 0
while True:
    p = ASSET_DIR / f"drone_rotating_with_parcel_{i}.png"
    if p.exists():
        drone_rot_with_parcel_frames.append(scale_to_cell(pygame.image.load(str(p)).convert_alpha(), GRID_SIZE, DRONE_SCALE))
        i += 1
    else:
        break
if not drone_rot_with_parcel_frames:
    r = load_image(ASSET_DIR / "drone_rotating_with_parcel.png")
    if r:
        drone_rot_with_parcel_frames = [scale_to_cell(r, GRID_SIZE, DRONE_SCALE)]

parcel_img = scale_to_cell(load_image(ASSET_DIR / "parcel.png"), GRID_SIZE, PARCEL_SCALE)

images = {
    "drone_static": drone_static,
    "drone_rot_frames": drone_rot_frames,
    "drone_static_with_parcel": drone_static_with_parcel,
    "drone_rot_with_parcel_frames": drone_rot_with_parcel_frames,
    "parcel_img": parcel_img,
}

USE_IMAGE = any(images.values())

# Create terrain and agents
terrain = Terrain(GRID_SIZE, (SCREEN_W, SCREEN_H), parcel_img=images["parcel_img"], parcel_scale=PARCEL_SCALE)
start_col = (SCREEN_W // GRID_SIZE) // 2
start_row = (SCREEN_H // GRID_SIZE) // 2
# spawn parcels
terrain.spawn_random(15)

# Add exactly one delivery station in the center with area 4x4
cols = SCREEN_W // GRID_SIZE
rows = SCREEN_H // GRID_SIZE
center_col = max(2, cols // 2 - 2)
center_row = max(2, rows // 2 - 2)
terrain.add_station(center_col, center_row, w=4, h=4)

drone = Drone((start_col, start_row), GRID_SIZE, (SCREEN_W, SCREEN_H))

# Controllers
human = HumanAgentController(drone, terrain)
ai = AIAgentController(drone, terrain)
switcher = ControllerSwitcher([human, ai])

font = pygame.font.SysFont("Consolas", 20)

# Flash state for cell feedback when picking or dropping
flash_timer = 0.0
flash_color = None
flash_cell = None

# Delivery counters and timed session state
total_delivered = 0
session_delivered = 0
session_active = False
session_time_left = 0.0
SESSION_DEFAULT_SECONDS = 60.0

running = True
dt = 0
while running:
    # events: always let switcher see the event first so TAB works reliably
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

        # give controllers a first chance to process the event (TAB handled here)
        switcher.handle_event(e)

        # global keys that should not block controller switching
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_s:
                # start a timed delivery session
                session_active = True
                session_time_left = SESSION_DEFAULT_SECONDS
                session_delivered = 0
            elif e.key == pygame.K_r:
                # reset counts and session
                total_delivered = 0
                session_delivered = 0
                session_active = False
                session_time_left = 0.0

        # Mouse parcel placement
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            mx, my = e.pos
            col, row = int(mx // GRID_SIZE), int(my // GRID_SIZE)
            # prevent placing inside station area
            if not terrain.is_station_cell(col, row):
                terrain.add_parcel(col, row)

    # update controller logic (human or AI)
    switcher.update(dt)

    # after controllers run, check if drone reported an action for flash feedback
    if hasattr(drone, "_last_action") and drone._last_action:
        kind = drone._last_action[0]
        if kind == "pick":
            _, cell, parcel = drone._last_action
            flash_color = (60, 220, 100, 160)
            flash_cell = cell
            flash_timer = FLASH_DURATION
        elif kind == "drop":
            _, cell, parcel = drone._last_action
            flash_color = (240, 120, 120, 160)
            flash_cell = cell
            flash_timer = FLASH_DURATION

            # only mark and count delivery if the parcel reference is valid and not already delivered
            if parcel and not getattr(parcel, "delivered", False):
                # mark parcel delivered
                parcel.delivered = True
                parcel.picked = False

                # station accounting
                station = terrain.get_station_at(cell[0], cell[1])
                if station:
                    station.delivered += 1

                total_delivered += 1
                if session_active:
                    session_delivered += 1

        # clear the action so we do not process it again
        drone._last_action = None

    # update session timer
    if session_active:
        session_time_left -= dt
        if session_time_left <= 0:
            session_active = False
            session_time_left = 0.0

    # update drone physics/animation
    drone.update(dt, SPEED, ANIM_FPS, images.get("drone_rot_with_parcel_frames"), images.get("drone_rot_frames"))

    # draw
    screen.fill((192, 192, 192))

    # grid
    cols = SCREEN_W // GRID_SIZE
    rows = SCREEN_H // GRID_SIZE
    for x in range(0, SCREEN_W, GRID_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, SCREEN_H))
    for y in range(0, SCREEN_H, GRID_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (SCREEN_W, y))

    # draw parcels via terrain
    terrain.draw(screen)

    # draw stations (they draw a multi-cell highlighted area)
    for s in terrain.stations:
        s.draw(surf=screen)

    # draw flash cell under drone if active
    if flash_timer > 0 and flash_cell is not None:
        cell_x = flash_cell[0] * GRID_SIZE
        cell_y = flash_cell[1] * GRID_SIZE
        s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
        s.fill(flash_color)
        screen.blit(s, (cell_x, cell_y))
        flash_timer -= dt
        if flash_timer <= 0:
            flash_timer = 0
            flash_cell = None
            flash_color = None

    # draw drone
    drone.draw(screen, images)

    # HUD
    hud_lines = [
        f"Controller: {switcher.current.__class__.__name__} | Cells: {SCREEN_W // GRID_SIZE} x {SCREEN_H // GRID_SIZE}   Drone cell: ({drone.col}, {drone.row})   carrying: {drone.carrying is not None}",
        f"Total delivered: {total_delivered}    Session delivered: {session_delivered}    Session time left: {int(session_time_left)}s",
        "Controls: Hold arrow keys for continuous movement (diagonals allowed). Space = pick up or drop. TAB to switch controller. S = start session (60s). R = reset counts. ESC to quit.",
    ]
    for i, line in enumerate(hud_lines):
        txt = font.render(line, True, (10, 10, 10))
        screen.blit(txt, (12, 12 + i * 26))

    pygame.display.flip()
    dt = clock.tick(60) / 1000.0

pygame.quit()
sys.exit()
