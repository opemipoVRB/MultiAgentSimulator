# src/games.py
import pygame
import sys
import random
import time
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

# Logging for lost drones
LOST_LOG_PATH = Path(__file__).parent / "lost_drones.log"
lost_drones = []  # list of dicts with lost info

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
        drone_rot_with_parcel_frames.append(
            scale_to_cell(pygame.image.load(str(p)).convert_alpha(), GRID_SIZE, DRONE_SCALE))
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

# fonts
font = pygame.font.SysFont("Consolas", 20)
large_font = pygame.font.SysFont("Consolas", 48)
title_font = pygame.font.SysFont("Consolas", 72)


def show_splash(timeout=4.0, splash_name="drone_static.png"):
    """Display splash screen with optional image.
    Press any key to continue, or wait `timeout` seconds.
    """
    splash_img = None
    p = ASSET_DIR / splash_name
    if p.exists():
        raw = load_image(p)
        if raw:
            # scale so width <= 60% screen width and height <= 40% screen height
            max_w = int(SCREEN_W * 0.6)
            max_h = int(SCREEN_H * 0.4)
            w, h = raw.get_size()
            scale = min(1.0, max_w / w, max_h / h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            splash_img = pygame.transform.smoothscale(raw, new_size)

    start = pygame.time.get_ticks() / 1000.0
    while True:
        now = pygame.time.get_ticks() / 1000.0
        elapsed = now - start
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN or e.type == pygame.MOUSEBUTTONDOWN:
                return

        screen.fill((24, 32, 48))

        if splash_img:
            img_rect = splash_img.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 80))
            screen.blit(splash_img, img_rect)
            text_y = img_rect.bottom + 28
        else:
            text_y = SCREEN_H // 2 - 60

        title = title_font.render("Agentic Swarm Lab", True, (220, 230, 240))
        subtitle = large_font.render("Episodic LLM Guidance", True, (180, 200, 220))
        hint = font.render("Press any key to continue or wait...", True, (180, 180, 200))

        screen.blit(title, title.get_rect(center=(SCREEN_W // 2, text_y)))
        screen.blit(subtitle, subtitle.get_rect(center=(SCREEN_W // 2, text_y + 64)))
        screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, text_y + 140)))

        pygame.display.flip()
        clock.tick(60)
        if elapsed >= timeout:
            return


def show_setup(initial=15, min_val=0, max_val=500):
    """
    Setup screen to choose number of parcels.
    - Up/Down to change
    - Type digits to enter number
    - Enter to accept
    - ESC to quit
    Returns selected integer.
    """
    value = initial
    typing = ""
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif e.key == pygame.K_RETURN or e.key == pygame.K_KP_ENTER:
                    # finalize typed value if any
                    if typing:
                        try:
                            v = int(typing)
                            value = max(min_val, min(max_val, v))
                        except ValueError:
                            pass
                    return value
                elif e.key == pygame.K_BACKSPACE:
                    typing = typing[:-1]
                elif e.key == pygame.K_UP:
                    value = min(max_val, value + 1)
                    typing = ""
                elif e.key == pygame.K_DOWN:
                    value = max(min_val, value - 1)
                    typing = ""
                elif e.unicode.isdigit():
                    typing += e.unicode
                    # clamp visible typed number
                    try:
                        v = int(typing)
                        if v > max_val:
                            typing = str(max_val)
                    except ValueError:
                        typing = ""
        screen.fill((40, 44, 52))
        header = large_font.render("Simulation Setup", True, (230, 230, 230))
        prompt = font.render("Enter number of parcels to spawn (Up/Down or type digits):", True, (210, 210, 210))
        value_display = title_font.render(str(value) if not typing else typing, True, (240, 240, 200))
        hint = font.render("Enter to start | ESC to quit", True, (200, 200, 200))
        screen.blit(header, header.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 140)))
        screen.blit(prompt, prompt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 60)))
        screen.blit(value_display, value_display.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 20)))
        screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 140)))
        pygame.display.flip()
        clock.tick(60)


def run_game(initial_parcels):
    # Create terrain and agents
    terrain = Terrain(GRID_SIZE, (SCREEN_W, SCREEN_H), parcel_img=images["parcel_img"], parcel_scale=PARCEL_SCALE)
    start_col = (SCREEN_W // GRID_SIZE) // 2
    start_row = (SCREEN_H // GRID_SIZE) // 2
    # Add exactly one delivery station in the center with area 4x4
    cols = SCREEN_W // GRID_SIZE
    rows = SCREEN_H // GRID_SIZE
    center_col = max(2, cols // 2 - 2)
    center_row = max(2, rows // 2 - 2)
    terrain.add_station(center_col, center_row, w=4, h=4)
    # spawn parcels based on setup entry
    terrain.spawn_random(initial_parcels)

    drone = Drone((start_col, start_row), GRID_SIZE, (SCREEN_W, SCREEN_H))

    # Controllers
    human = HumanAgentController(drone, terrain)
    ai = AIAgentController(drone, terrain)
    switcher = ControllerSwitcher([human, ai])

    font_local = pygame.font.SysFont("Consolas", 20)
    narration_font = pygame.font.SysFont("Consolas", 22)

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
                        station.register_delivery(cell)
                        # now station.usage tracks counts per cell
                    total_delivered += 1
                    if session_active:
                        session_delivered += 1

            elif kind == "pick_failed":
                _, cell, _ = drone._last_action
                flash_color = (240, 200, 80, 160)  # yellow-ish
                flash_cell = cell
                flash_timer = FLASH_DURATION

            elif kind == "drop_failed":
                _, cell, _ = drone._last_action
                flash_color = (240, 200, 80, 160)  # yellow-ish
                flash_cell = cell
                flash_timer = FLASH_DURATION

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

        # ------ detect newly lost drones and log/report them ------
        if getattr(drone, "lost", False) and not getattr(drone, "_reported_lost", False):
            info = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "col": int(drone.col),
                "row": int(drone.row),
                "x": float(drone.pos.x),
                "y": float(drone.pos.y),
                "battery_pct": int(drone.power.percent()) if hasattr(drone, "power") else 0,
                "carrying": getattr(drone.carrying, "col", None) is not None
            }
            lost_drones.append(info)
            # append to log file
            try:
                with open(LOST_LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.write(
                        f"{info['time']} - LOST at cell ({info['col']},{info['row']}) pos ({info['x']:.1f},{info['y']:.1f}) battery {info['battery_pct']}% carrying {info['carrying']}\n")
            except Exception:
                pass
            drone._reported_lost = True

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

        # # draw stations (they draw a multi-cell highlighted area)
        # for s in terrain.stations:
        #     # defensive draw call - some station implementations expect surf kw
        #     try:
        #         s.draw(surf=screen)
        #     except TypeError:
        #         s.draw(screen)

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

        # draw drone (drone.draw now shows battery icon and lost X if Drone implements it)
        drone.draw(screen, images)

        # ---------------------------
        # HUD panel on the left
        # ---------------------------
        #
        # This draws a translucent HUD panel on the left. To adjust:
        # - hud_w controls width (we set it to 28% of screen width).
        # - hud_h is computed from number of lines, but you may fix it if you want.
        # - left_margin controls horizontal offset from left edge.
        # - top_margin controls vertical offset from top edge.
        #
        hud_lines = [
            f"Controller: {switcher.current.__class__.__name__} | Cells: {SCREEN_W // GRID_SIZE} x {SCREEN_H // GRID_SIZE}",
            f"Drone cell: ({drone.col}, {drone.row})   carrying: {drone.carrying is not None}   Battery: {int(drone.power.percent()) if hasattr(drone, 'power') else 0}%",
            f"Total delivered: {total_delivered}    Session delivered: {session_delivered}    Session time left: {int(session_time_left)}s",
            f"Lost drones: {len(lost_drones)}"
        ]
        if lost_drones:
            last = lost_drones[-1]
            hud_lines.append(
                f"Last lost: {last['time']} @ ({last['col']},{last['row']}) battery {last['battery_pct']}%")
        hud_lines.append(
            f"Controls: Hold arrow keys for continuous movement (diagonals allowed).\n"
            f"Space = pick up or drop.\nTAB to switch controller.\nS = start session (60s).\nR = reset counts.\nESC to quit.")

        # panel layout params
        hud_w = int(SCREEN_W * 0.35)  # width 35% of screen; increase for wider HUD
        left_margin = 12  # distance from left edge
        top_margin = 12  # distance from top edge
        line_h = 26
        padding = 12

        # compute HUD height dynamically from lines (with some padding)
        hud_h = padding * 2 + len(hud_lines) * line_h

        # hud_h = int(SCREEN_W * 0.095)  # padding * 2 + max(len(hud_lines), 1) * line_h

        # create translucent surface for HUD
        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill((245, 245, 250, 180))  # RGBA, last is alpha for translucency
        try:
            pygame.draw.rect(hud_surf, (30, 30, 30), hud_surf.get_rect(), 2, border_radius=6)
        except TypeError:
            pygame.draw.rect(hud_surf, (30, 30, 30), hud_surf.get_rect(), 2)

        # render lines onto HUD surface
        for i, line in enumerate(hud_lines):
            txt = font_local.render(line, True, (10, 10, 10))
            hud_surf.blit(txt, (padding, padding + i * line_h))

        # blit HUD surface to screen
        screen.blit(hud_surf, (left_margin, top_margin))

        # ---------------------------
        # Narration panel on the right
        # ---------------------------
        narration = None
        try:
            narration = getattr(switcher.current, "last_narration", None)
        except Exception:
            narration = None

        if narration:
            # right-side narration panel defaults
            box_w = int(SCREEN_W * 0.28)  # width 28% of screen; adjust for longer lines
            box_h = int(SCREEN_H * 0.10)  # height 10% of screen; increase to show more lines
            right_margin = 12  # distance from right edge; increase to move the box left
            box_x = SCREEN_W - box_w - right_margin
            box_y = 12  # small top margin; change to move vertical position

            # create surface with alpha for slight translucency
            sbox = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            sbox.fill((250, 250, 252, 230))
            try:
                pygame.draw.rect(sbox, (34, 34, 34), sbox.get_rect(), 3, border_radius=6)
            except TypeError:
                pygame.draw.rect(sbox, (34, 34, 34), sbox.get_rect(), 3)

            # header
            hdr_txt = "Plan narration"
            hdr = font_local.render(hdr_txt, True, (10, 10, 10))
            sbox.blit(hdr, (12, 10))

            # cheap word wrap
            words = narration.split()
            lines = []
            cur = ""
            # approx chars per line based on width; tweak multiplier if fonts change
            approx_chars_per_line = max(50, (box_w // 16))
            for w in words:
                if len(cur) + len(w) + 1 > approx_chars_per_line:
                    lines.append(cur.strip())
                    cur = w + " "
                else:
                    cur += w + " "
            if cur:
                lines.append(cur.strip())

            # clamp lines to available vertical space
            line_h = 26
            max_lines = max(4, (box_h - 60) // line_h)  # dynamic max lines based on height
            truncated = False
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                truncated = True

            # draw lines
            start_y = 44
            for idx, ln in enumerate(lines):
                txt = narration_font.render(ln, True, (28, 28, 28))
                sbox.blit(txt, (12, start_y + idx * line_h))

            if truncated:
                ell = narration_font.render("... (truncated)", True, (120, 120, 120))
                sbox.blit(ell, (12, start_y + max_lines * line_h))

            # blit the narration box at computed position
            screen.blit(sbox, (box_x, box_y))

        pygame.display.flip()
        dt = clock.tick(60) / 1000.0

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    # splash then setup flow
    show_splash(timeout=4.0)
    parcels = show_setup(initial=15, min_val=0, max_val=1000)
    run_game(parcels)
