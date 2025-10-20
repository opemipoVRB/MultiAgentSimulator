# src/games.py
import pygame
import sys
import random
import time
from pathlib import Path

from artifacts import Terrain, Drone
from utils import load_image, scale_to_cell
# Import controllers and the shared RESERVATIONS mapping so HUD/debug can inspect it.
from controllers import HumanAgentController, AIAgentController, ControllerSwitcher, RESERVATIONS

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


def show_setup(initial_parcels=15, initial_drones=1, min_val=0, max_val=500):
    """
    Setup screen to choose number of parcels and number of drones.
    - Up/Down keys to increase or decrease values
    - Type digits directly to enter a number
    - Press Enter to confirm and move to next step
    - Press ESC to quit the game
    Returns: (num_parcels, num_drones)
    """

    # Helper inner function to collect a single integer input interactively.
    def get_number(prompt_text, initial_value):
        value = initial_value
        typing = ""

        while True:
            # Poll for all pygame events like keyboard and quit.
            for e in pygame.event.get():

                # Allow window close or ESC to exit the program entirely.
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if e.type == pygame.KEYDOWN:
                    # ESC aborts entirely
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    # ENTER confirms current value (either typed or the shown value)
                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
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
                    elif getattr(e, "unicode", "").isdigit():
                        # accumulate typed digits
                        typing += e.unicode
                        try:
                            v = int(typing)
                            if v > max_val:
                                typing = str(max_val)
                        except ValueError:
                            typing = ""

            # Draw the setup screen contents (simple, centered UI)
            screen.fill((40, 44, 52))
            header = large_font.render("Simulation Setup", True, (230, 230, 230))
            prompt = font.render(prompt_text, True, (210, 210, 210))
            value_display = title_font.render(
                str(value) if not typing else typing, True, (240, 240, 200)
            )
            hint = font.render("Enter to confirm | ESC to quit", True, (200, 200, 200))

            screen.blit(header, header.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 140)))
            screen.blit(prompt, prompt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 60)))
            screen.blit(value_display, value_display.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 20)))
            screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 140)))

            pygame.display.flip()
            clock.tick(60)

    # Ask the user for number of parcels
    num_parcels = get_number("Enter number of parcels to spawn:", initial_parcels)

    # Ask the user for number of drones to deploy
    num_drones = get_number("Enter number of drones to deploy:", initial_drones)

    return num_parcels, num_drones

def run_game(initial_parcels, num_drones):
    """
    Main simulation loop for multiple drones.
    - Spawns terrain, parcels, and multiple drones.
    - Each drone is controlled by its own AI controller.
    - HUD shows per-drone battery/status plus overall stats and reservations.
    """

    # -----------------------------
    # 1) Create the world and populate parcels
    # -----------------------------
    terrain = Terrain(GRID_SIZE, (SCREEN_W, SCREEN_H), parcel_img=images["parcel_img"], parcel_scale=PARCEL_SCALE)
    terrain.spawn_random(initial_parcels)
    print(f"Spawned {initial_parcels} parcels")

    # -----------------------------
    # 2) Add a delivery station in the center of the map
    # -----------------------------
    cols = SCREEN_W // GRID_SIZE
    rows = SCREEN_H // GRID_SIZE
    center_col = max(2, cols // 2 - 2)
    center_row = max(2, rows // 2 - 2)
    terrain.add_station(center_col, center_row, w=4, h=4)

    # -----------------------------
    # 3) Create multiple Drone objects and AI controllers
    # -----------------------------
    drones = []
    ai_controllers = []

    for i in range(num_drones):
        # Distribute drones around the center to avoid stacking exactly on top of each other.
        # Compute a small offset from the center and clamp to bounds.
        offset = (i - (num_drones - 1) / 2.0) * 2  # spreads drones horizontally
        start_col = int(min(max(0, cols // 2 + round(offset)), cols - 1))
        start_row = int(min(max(0, rows // 2), rows - 1))

        drone = Drone((start_col, start_row), GRID_SIZE, (SCREEN_W, SCREEN_H))
        drones.append(drone)

        # Give each drone an AI controller
        ai = AIAgentController(drone, terrain)
        ai_controllers.append(ai)

    print(f"Deployed {len(drones)} drone(s)")

    # Pre-warm planner and reservations so drones begin acting quickly
    for ai in ai_controllers:
        try:
            ai._cleanup_reservations(force=True)
        except Exception:
            pass
        try:
            ai._request_plan(force_refresh=True)
        except Exception:
            # If planner is unavailable, controllers will fall back to greedy behavior
            pass

    # -----------------------------
    # 4) Initialize fonts and counters
    # -----------------------------
    font_local = pygame.font.SysFont("Consolas", 20)
    narration_font = pygame.font.SysFont("Consolas", 22)
    flash_timer = 0.0
    flash_color = None
    flash_cell = None
    total_delivered = 0
    session_delivered = 0
    session_active = False
    session_time_left = 0.0
    lost_drones.clear()

    # -----------------------------
    # 5) Simulation timing
    # -----------------------------
    running = True
    dt = 0.0

    # -----------------------------
    # 6) Main loop
    # -----------------------------
    while running:
        # --- event handling (global) ---
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_s:
                    # start a timed delivery session (optional)
                    session_active = True
                    session_time_left = 60.0
                    session_delivered = 0
                elif e.key == pygame.K_r:
                    total_delivered = 0
                    session_delivered = 0
                    session_active = False
                    session_time_left = 0.0
            # allow placing parcels with mouse left click (same rule: don't place inside station)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                ccol, crow = int(mx // GRID_SIZE), int(my // GRID_SIZE)
                if not terrain.is_station_cell(ccol, crow):
                    terrain.add_parcel(ccol, crow)

        # --- update AI controllers first (they decide targets / pick/drop) ---
        for ai in ai_controllers:
            try:
                ai.update(dt)
            except Exception:
                # defensive: don't let one controller crash the loop
                print("AI controller update error", flush=True)

        # If planners are down or plans are empty, let idle AIs try greedy claim immediately.
        # This encourages concurrency: every idle AI will attempt to claim the nearest unreserved parcel.
        for ai in ai_controllers:
            try:
                # if not moving, not carrying and no plan, greedy-claim
                if (not ai.drone.moving) and (ai.drone.carrying is None) and (not ai.plan):
                    parcel = None
                    try:
                        parcel = ai._find_nearest_unreserved_parcel()
                    except Exception:
                        parcel = None
                    if parcel:
                        try:
                            claimed = ai._try_claim_and_go_to_parcel(parcel)
                            if claimed:
                                # refresh reservations for HUD
                                ai._cleanup_reservations()
                        except Exception:
                            pass
            except Exception:
                pass

        # --- advance drone physics and energy consumption ---
        for drone in drones:
            drone.update(dt, SPEED, ANIM_FPS, images.get("drone_rot_with_parcel_frames"), images.get("drone_rot_frames"))

        # --- process per-drone _last_action (pick/drop/pick_failed/drop_failed) ---
        # This mirrors the single-drone logic but runs for each drone.
        for drone in drones:
            if not hasattr(drone, "_last_action") or drone._last_action is None:
                continue

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

                # If parcel is valid and not already counted as delivered, mark delivered and register with station.
                if parcel and not getattr(parcel, "delivered", False):
                    parcel.delivered = True
                    parcel.picked = False

                    # Station registration so station usage bookkeeping is correct
                    station = terrain.get_station_at(cell[0], cell[1])
                    if station:
                        station.register_delivery(cell)

                    # increment counters
                    total_delivered += 1
                    if session_active:
                        session_delivered += 1

                    # Defensive: remove any reservation entries that match this parcel's original cell or drop cell
                    try:
                        # remove exact match keys (pickup cell might match parcel.col/row before move)
                        pickup_key = (int(parcel.col), int(parcel.row))
                        RESERVATIONS.pop(pickup_key, None)
                    except Exception:
                        pass
                    # Additionally, clear any reservation that references the drop cell
                    RESERVATIONS.pop((cell[0], cell[1]), None)

            elif kind == "pick_failed":
                _, cell, _ = drone._last_action
                flash_color = (240, 200, 80, 160)
                flash_cell = cell
                flash_timer = FLASH_DURATION

            elif kind == "drop_failed":
                _, cell, _ = drone._last_action
                flash_color = (240, 200, 80, 160)
                flash_cell = cell
                flash_timer = FLASH_DURATION

            # After handling, clear drone._last_action so we don't double-count
            drone._last_action = None

        # --- detect newly lost drones and log/report them ---
        for drone in drones:
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
                try:
                    with open(LOST_LOG_PATH, "a", encoding="utf-8") as fh:
                        fh.write(
                            f"{info['time']} - LOST at cell ({info['col']},{info['row']}) pos ({info['x']:.1f},{info['y']:.1f}) battery {info['battery_pct']}% carrying {info['carrying']}\n")
                except Exception:
                    pass
                drone._reported_lost = True

        # --- drawing ---
        screen.fill((192, 192, 192))

        # grid lines
        for x in range(0, SCREEN_W, GRID_SIZE):
            pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, SCREEN_H))
        for y in range(0, SCREEN_H, GRID_SIZE):
            pygame.draw.line(screen, (200, 200, 200), (0, y), (SCREEN_W, y))

        # draw parcels and stations
        terrain.draw(screen)
        for s in terrain.stations:
            try:
                s.draw(surf=screen)
            except TypeError:
                s.draw(screen)

        # flash cell under drone if active
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

        # draw every drone
        for drone in drones:
            drone.draw(screen, images)

        # ---------------------------
        # HUD: show summary + per-drone lines for debugging
        # ---------------------------
        # count of field parcels remaining (not picked, not delivered)
        field_remaining = sum(1 for p in terrain.parcels if not getattr(p, "delivered", False) and not getattr(p, "picked", False))

        hud_lines = [
            f"Drones: {len(drones)} | Parcels (objects): {len(terrain.parcels)} | Field remaining: {field_remaining} | Delivered: {total_delivered}",
            f"Avg battery: {int(sum(d.power.percent() for d in drones) / len(drones)) if drones else 0}%",
            f"Lost drones: {len(lost_drones)}"
        ]

        # Per-drone details (index, battery %, carrying, target, last_status)
        for idx, d in enumerate(drones):
            carrying = "Y" if d.carrying else "-"
            lost_flag = " LOST" if getattr(d, "lost", False) else ""
            # get controller status if available
            ai_status = ""
            try:
                ai_status = ai_controllers[idx].last_status or ""
            except Exception:
                ai_status = ""
            # compute current target cell (either drone.target in world coords -> convert to cell,
            # or the next A* cell via drone.current_target_cell)
            tgt_cell = None
            try:
                ct = d.current_target_cell()
                if ct:
                    tgt_cell = ct
                elif d.target:
                    tgt_cell = (int(d.target.x // d.grid_size), int(d.target.y // d.grid_size))
            except Exception:
                tgt_cell = None
            tgt_str = f"{tgt_cell}" if tgt_cell else "-"
            hud_lines.append(f"#{idx} cell:({d.col},{d.row}) bat:{int(d.power.percent())}% carry:{carrying} tgt:{tgt_str} status:{ai_status}{lost_flag}")

        # Reservation overview (debug): show reserved pickup cells and which AI index owns them
        if RESERVATIONS:
            hud_lines.append("Reservations:")
            for k, (owner, ts) in list(RESERVATIONS.items()):
                owner_idx = None
                for i, a in enumerate(ai_controllers):
                    if a is owner:
                        owner_idx = i
                        break
                owner_label = f"AI{owner_idx}" if owner_idx is not None else "unknown"
                hud_lines.append(f"  {k} -> {owner_label}")

        hud_lines.append("ESC to quit")

        # draw HUD surface
        hud_w = int(SCREEN_W * 0.35)
        hud_h = int(max(140, 26 * len(hud_lines) + 20))
        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill((245, 245, 250, 180))
        try:
            pygame.draw.rect(hud_surf, (30, 30, 30), hud_surf.get_rect(), 2, border_radius=6)
        except TypeError:
            pygame.draw.rect(hud_surf, (30, 30, 30), hud_surf.get_rect(), 2)

        for i, line in enumerate(hud_lines):
            txt = font_local.render(line, True, (10, 10, 10))
            hud_surf.blit(txt, (12, 12 + i * 26))

        screen.blit(hud_surf, (12, 12))

        # narration box: show last narration for first AI (optional)
        narration = None
        if ai_controllers:
            narration = getattr(ai_controllers[0], "last_narration", None)
        if narration:
            box_w = int(SCREEN_W * 0.28)
            box_h = int(SCREEN_H * 0.10)
            right_margin = 12
            box_x = SCREEN_W - box_w - right_margin
            box_y = 12
            sbox = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            sbox.fill((250, 250, 252, 230))
            try:
                pygame.draw.rect(sbox, (34, 34, 34), sbox.get_rect(), 3, border_radius=6)
            except TypeError:
                pygame.draw.rect(sbox, (34, 34, 34), sbox.get_rect(), 3)

            hdr = font_local.render("Plan narration (AI0)", True, (10, 10, 10))
            sbox.blit(hdr, (12, 10))

            words = narration.split()
            lines = []
            cur = ""
            approx_chars_per_line = max(50, (box_w // 16))
            for w in words:
                if len(cur) + len(w) + 1 > approx_chars_per_line:
                    lines.append(cur.strip())
                    cur = w + " "
                else:
                    cur += w + " "
            if cur:
                lines.append(cur.strip())

            line_h = 26
            max_lines = max(4, (box_h - 60) // line_h)
            truncated = False
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                truncated = True

            start_y = 44
            for idx, ln in enumerate(lines):
                txt = narration_font.render(ln, True, (28, 28, 28))
                sbox.blit(txt, (12, start_y + idx * line_h))
            if truncated:
                ell = narration_font.render("... (truncated)", True, (120, 120, 120))
                sbox.blit(ell, (12, start_y + max_lines * line_h))

            screen.blit(sbox, (box_x, box_y))

        pygame.display.flip()
        dt = clock.tick(60) / 1000.0

    # Clean up
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    show_splash(timeout=4.0)
    parcels, drones = show_setup(initial_parcels=15, initial_drones=1, min_val=0, max_val=1000)
    run_game(parcels, drones)
