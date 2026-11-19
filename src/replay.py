# src/replay.py
"""
Replay of exported runs inside the GUI.

Reads telemetry only (experiment.json, agents.json): no strategy, controller or simulation code
runs. Any number of strategies from the same experiment are drawn side by side on one shared clock;
they ran on the same parcels and start cells, so they can be compared directly.

The runner's menus call `ReplayViewer.show`. Controls are listed at the bottom of the screen.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import pygame

GRID_SIZE = 100          # pixels per cell in the exported positions
TRAIL_STEPS = 45
RADIO_STEPS = 45     # how long a transmission's ring stays on screen (steps)
FLASH_STEPS = 30
STEPS_PER_SECOND = 60    # simulation steps shown per second at 1x

AGENT_COLORS = [
    (230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49), (145, 30, 180), (14, 154, 167),
    (240, 50, 230), (128, 128, 0), (154, 99, 36), (128, 0, 0), (70, 153, 144), (0, 0, 117),
    (160, 111, 255), (192, 160, 0), (210, 105, 30), (46, 139, 87), (178, 34, 34), (95, 158, 160),
    (199, 21, 133), (85, 107, 47),
]


def _plain_name(arm_id: str) -> str:
    """Plain name of an arm from the generated analysis/revision/arm_names.json, if it is there."""
    try:
        path = Path(__file__).resolve().parents[1] / "analysis" / "revision" / "arm_names.json"
        from src.strategies.sppdec.legacy_ids import modernise
        return json.loads(path.read_text(encoding="utf-8")).get(modernise(arm_id), arm_id)
    except Exception:
        return arm_id


# ----------------------------------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------------------------------
def strategy_dir(experiment_dir: Path, entry: Dict) -> Optional[Path]:
    """Directory holding the exported telemetry of one strategy entry, or None."""
    candidates = []
    if entry.get("name"):
        candidates.append(experiment_dir / str(entry["name"]))  # exported folders carry the entry name
    for c in candidates:
        if (c / "agents.json").exists():
            return c
    return None


def has_telemetry(experiment_dir: Path, entry: Dict) -> bool:
    return strategy_dir(experiment_dir, entry) is not None


class RunData:
    """One strategy's run, reduced to what the replay draws."""

    def __init__(self, experiment_dir: Path, entry: Dict, parcels: List[Dict]):
        d = strategy_dir(experiment_dir, entry)
        agents = json.loads((d / "agents.json").read_text(encoding="utf-8"))
        self.label = _plain_name(str(entry.get("family") or entry.get("name")))
        self.status = str(entry.get("status", ""))
        self.parcels = [(int(p["col"]), int(p["row"])) for p in parcels]
        self.n_agents = len(agents)
        self.steps = max((len(a["trajectory"]) for a in agents), default=0)
        self.pos = []        # per agent: list of (x_px, y_px)
        self.bat = []        # per agent: list of battery levels
        self.carry = []      # per agent: list of (from_step, to_step)
        self.targets = []    # per agent: list of (step, col, row) decided pickups
        self.picked_at = {}  # (col, row) -> step the parcel was picked
        self.drops = []      # step of each delivery
        self.events = []     # (step, kind, col, row): interrupts and vanished targets
        self.collisions = []  # (step, col, row)
        self.messages = []   # (step, agent index, to, kind): radio transmissions
        sens = entry.get("sensing") or {}
        r = sens.get("radius")
        self.sense_radius = float(r) if r is not None and float(r) != float("inf") else None     # cells (Manhattan); None = sees everything
        picks, drops = [], []
        for i, a in enumerate(agents):
            self.pos.append([(t["pos"][0], t["pos"][1]) for t in a["trajectory"]])
            self.bat.append([e["battery"] for e in a["energy"]])
            self.targets.append(sorted(
                (da["time"]["step"], da["decided_cell"][0], da["decided_cell"][1])
                for da in a.get("decision_actions", []) if da.get("decided_cell")))
            for ac in a["actions"]:
                if ac["type"] == "pick":
                    picks.append((ac["time"]["step"], i, ac["cell"][0], ac["cell"][1]))
                elif ac["type"] == "drop":
                    drops.append((ac["time"]["step"], i))
            for po in a.get("plan_outcomes", []):
                kind = po.get("outcome", "")
                if kind.startswith("interrupt") or kind == "target_gone":
                    cell = po.get("planned_cell") or po.get("cell")
                    if cell:
                        self.events.append((po["time"]["step"], kind, cell[0], cell[1]))
            for m in a.get("message_events", []):
                self.messages.append((m["time"]["step"], i, m.get("to", "*"), m.get("kind", "")))
            for c in a.get("collisions", []):
                self.collisions.append((c["time"]["step"], c["cell"][0], c["cell"][1]))
        self.carry = [[] for _ in range(self.n_agents)]
        for s, i, c, r in sorted(picks):
            self.picked_at[(c, r)] = s
            end = next((ds for ds, di in sorted(drops) if di == i and ds > s), 10 ** 9)
            self.carry[i].append((s, end))
        self.drops = sorted(ds for ds, _ in drops)
        self.collisions = sorted(set(self.collisions))
        self.messages.sort()
        self.last_drop = self.drops[-1] if self.drops else 0
        wall = [t["time"].get("world_time") for a in agents for t in (a["trajectory"][:1] + a["trajectory"][-1:])]
        wall = [w for w in wall if w is not None]
        self.wall_seconds = (max(wall) - min(wall)) if wall else None

    def carrying(self, agent: int, step: int) -> bool:
        return any(a <= step < b for a, b in self.carry[agent])

    def target(self, agent: int, step: int):
        cur = None
        for t in self.targets[agent]:
            if t[0] <= step:
                cur = t
            else:
                break
        if cur is None or self.carrying(agent, step):
            return None
        picked = self.picked_at.get((cur[1], cur[2]))
        if picked is not None and picked < step:
            return None
        return cur[1], cur[2]

    def delivered(self, step: int) -> int:
        return sum(1 for s in self.drops if s <= step)


# ----------------------------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------------------------
def _layout(n: int, W: int = 1920, H: int = 1080, grid=(25, 16), top: int = 40, bottom: int = 118):
    """(columns, rows) that gives the largest map for `n` panels on a W x H screen."""
    best, best_scale = (n, 1), -1.0
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        pw = (W - 12 * (cols + 1)) / cols
        ph = (H - top - bottom - 12 * (rows - 1)) / rows - 30
        scale = min(pw / (grid[0] * GRID_SIZE), ph / (grid[1] * GRID_SIZE))
        if scale > best_scale + 1e-9:
            best, best_scale = (cols, rows), scale
    return best


class ReplayViewer:
    def __init__(self, screen, clock, font, large_font, images: Dict):
        self.screen = screen
        self.clock = clock
        self.font = font
        self.large_font = large_font
        self.images = images or {}
        self.small = pygame.font.SysFont("Consolas", 14)
        self._sprite_cache = {}

    # -- data ------------------------------------------------------------------------------
    def _load(self, experiment_dir: Path, entries: List[Dict], parcels: List[Dict]) -> List[RunData]:
        runs = []
        W, H = self.screen.get_size()
        for k, entry in enumerate(entries):
            self.screen.fill((24, 28, 36))
            msg = self.large_font.render("Loading replay data...", True, (230, 230, 230))
            self.screen.blit(msg, msg.get_rect(center=(W // 2, H // 2 - 30)))
            sub = self.font.render(f"{k + 1}/{len(entries)}  {entry.get('family') or entry.get('name')}",
                                   True, (170, 170, 170))
            self.screen.blit(sub, sub.get_rect(center=(W // 2, H // 2 + 20)))
            pygame.display.flip()
            pygame.event.pump()
            runs.append(RunData(experiment_dir, entry, parcels))
        return runs

    def _sprite(self, key: str, scale: float):
        img = self.images.get(key)
        if img is None:
            return None
        ck = (key, round(scale, 3))
        if ck not in self._sprite_cache:
            w, h = img.get_size()
            self._sprite_cache[ck] = pygame.transform.smoothscale(img, (max(2, int(w * scale)), max(2, int(h * scale))))
        return self._sprite_cache[ck]

    # -- drawing ---------------------------------------------------------------------------
    def _draw_panel(self, rect: pygame.Rect, run: RunData, step: int, grid, station, opts, index: int = 0):
        scr = self.screen
        pygame.draw.rect(scr, (30, 35, 44), rect, border_radius=6)
        head_h = 26
        title = self.small.render(f"[{index + 1}] {run.label}", True, (235, 235, 235))
        scr.blit(title, (rect.x + 8, rect.y + 5))
        i = min(step, max(run.steps - 1, 0))
        done = step >= run.steps - 1
        n_par = len(run.parcels)
        if done:
            stat = (f"{run.delivered(i)}/{n_par} delivered | {len(run.messages)} messages | last delivery {run.last_drop / 60:.1f} s | "
                    f"ended {run.steps / 60:.1f} s ({run.status})")
            colr = (150, 210, 150) if run.status == "completed" else (235, 180, 110)
        else:
            stat = f"{run.delivered(i)}/{n_par} delivered | t {i / 60:.1f} s | {sum(1 for m in run.messages if m[0] <= i)} messages"
            colr = (190, 200, 215)
        st = self.small.render(stat, True, colr)
        scr.blit(st, (rect.right - st.get_width() - 8, rect.y + 5))

        area = pygame.Rect(rect.x + 4, rect.y + head_h, rect.w - 8, rect.h - head_h - 4)
        gw, gh = grid
        s = min(area.w / (gw * GRID_SIZE), area.h / (gh * GRID_SIZE))
        ox = area.x + (area.w - gw * GRID_SIZE * s) / 2
        oy = area.y + (area.h - gh * GRID_SIZE * s) / 2
        cell = GRID_SIZE * s
        clip = scr.get_clip()
        scr.set_clip(area)

        # grid
        for c in range(gw + 1):
            pygame.draw.line(scr, (44, 50, 62), (ox + c * cell, oy), (ox + c * cell, oy + gh * cell))
        for r in range(gh + 1):
            pygame.draw.line(scr, (44, 50, 62), (ox, oy + r * cell), (ox + gw * cell, oy + r * cell))
        # station, drawn like the simulator's station
        if station:
            sr = pygame.Rect(ox + station["col"] * cell, oy + station["row"] * cell, station["w"] * cell, station["h"] * cell)
            ov = pygame.Surface((sr.w, sr.h), pygame.SRCALPHA)
            ov.fill((30, 110, 200, 60))
            scr.blit(ov, sr.topleft)
            pygame.draw.rect(scr, (40, 120, 200), sr, 3)
            cnt = self.small.render(str(run.delivered(i)), True, (255, 255, 255))
            scr.blit(cnt, (sr.x + 5, sr.y + 4))
        # parcels still on the ground
        psp = self._sprite("parcel_img", s)
        for (c, r) in run.parcels:
            t = run.picked_at.get((c, r))
            if t is not None and t <= i:
                continue
            cx, cy = ox + (c + 0.5) * cell, oy + (r + 0.5) * cell
            if psp:
                scr.blit(psp, psp.get_rect(center=(cx, cy)))
            else:
                q = cell * 0.3
                pygame.draw.rect(scr, (200, 160, 60), (cx - q / 2, cy - q / 2, q, q))
        # trails
        if opts["trails"]:
            for k in range(run.n_agents):
                col = AGENT_COLORS[k % len(AGENT_COLORS)]
                pts = [(ox + p[0] * s, oy + p[1] * s) for p in run.pos[k][max(0, i - TRAIL_STEPS):i + 1]]
                if len(pts) > 1:
                    pygame.draw.lines(scr, col, False, pts, 2)
        # targets
        if opts["targets"]:
            for k in range(run.n_agents):
                tgt = run.target(k, i)
                if not tgt:
                    continue
                col = AGENT_COLORS[k % len(AGENT_COLORS)]
                p = run.pos[k][min(i, len(run.pos[k]) - 1)]
                a = (ox + p[0] * s, oy + p[1] * s)
                b = (ox + (tgt[0] + 0.5) * cell, oy + (tgt[1] + 0.5) * cell)
                n = max(1, int(math.dist(a, b) // 10))
                for j in range(0, n, 2):
                    p0 = (a[0] + (b[0] - a[0]) * j / n, a[1] + (b[1] - a[1]) * j / n)
                    p1 = (a[0] + (b[0] - a[0]) * (j + 1) / n, a[1] + (b[1] - a[1]) * (j + 1) / n)
                    pygame.draw.line(scr, col, p0, p1, 1)
        # events
        if opts["events"]:
            for (es, kind, c, r) in run.events:
                if es <= i < es + FLASH_STEPS:
                    f = 1 - (i - es) / FLASH_STEPS
                    pygame.draw.circle(scr, (255, int(140 * f + 60), 0), (ox + (c + 0.5) * cell, oy + (r + 0.5) * cell),
                                       int(cell * (0.5 + 0.6 * (1 - f))), 3)
            for (es, c, r) in run.collisions:
                if es <= i < es + FLASH_STEPS:
                    x, y = ox + (c + 0.5) * cell, oy + (r + 0.5) * cell
                    pygame.draw.line(scr, (220, 50, 50), (x - 6, y - 6), (x + 6, y + 6), 2)
                    pygame.draw.line(scr, (220, 50, 50), (x + 6, y - 6), (x - 6, y + 6), 2)
        # perception: the boundary of what each drone can sense (Manhattan range, so a diamond), breathing slowly
        if opts.get("sense", True) and run.sense_radius:
            breathe = 0.5 + 0.5 * math.sin(i / 14.0)
            for k in range(run.n_agents):
                q = run.pos[k][min(i, len(run.pos[k]) - 1)]
                cx, cy = ox + q[0] * s, oy + q[1] * s
                rr = run.sense_radius * cell
                col = AGENT_COLORS[k % len(AGENT_COLORS)]
                shade = tuple(int(c * (0.35 + 0.45 * breathe)) for c in col)
                pygame.draw.polygon(scr, shade, [(cx, cy - rr), (cx + rr, cy), (cx, cy + rr), (cx - rr, cy)], 1)
        # radio: a ring spreads from the sender for a moment after each transmission
        if opts.get("radio", True):
            for (ms, k, to, kind) in run.messages:
                if ms <= i < ms + RADIO_STEPS:
                    f = (i - ms) / RADIO_STEPS
                    q = run.pos[k][min(i, len(run.pos[k]) - 1)]
                    pygame.draw.circle(scr, (60 + int(120 * (1 - f)), 200, 255), (ox + q[0] * s, oy + q[1] * s),
                                       int(cell * (0.6 + 3.2 * f)), 2)
        # drones
        for k in range(run.n_agents):
            col = AGENT_COLORS[k % len(AGENT_COLORS)]
            p = run.pos[k][min(i, len(run.pos[k]) - 1)]
            x, y = ox + p[0] * s, oy + p[1] * s
            carrying = run.carrying(k, i)
            sp = self._sprite("drone_static_with_parcel" if carrying else "drone_static", s)
            if sp:
                scr.blit(sp, sp.get_rect(center=(x, y)))
            else:
                pygame.draw.circle(scr, col, (x, y), max(4, int(cell * 0.32)), 0 if carrying else 3)
            pygame.draw.circle(scr, col, (x, y), max(3, int(cell * 0.12)))
            lab = self.small.render(str(k), True, (255, 255, 255))
            scr.blit(lab, lab.get_rect(center=(x, y - cell * 0.42)))
            b = max(0.0, min(100.0, run.bat[k][min(i, len(run.bat[k]) - 1)]))
            bw = cell * 0.6
            pygame.draw.rect(scr, (60, 66, 78), (x - bw / 2, y + cell * 0.40, bw, 3))
            pygame.draw.rect(scr, (210, 60, 60) if b < 25 else (60, 190, 90), (x - bw / 2, y + cell * 0.40, bw * b / 100, 3))
        scr.set_clip(clip)

    # -- main loop -------------------------------------------------------------------------
    def show(self, experiment_dir: Path, entries: List[Dict], title: str = "Replay") -> None:
        experiment_dir = Path(experiment_dir)
        exp = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
        runs = self._load(experiment_dir, entries, exp.get("parcels", []))
        if not runs:
            return
        station = exp.get("station")
        prov = (exp.get("provenance") or {}).get("grid")
        if prov:
            grid = (int(prov[0]), int(prov[1]))
        else:
            W, H = pygame.display.get_surface().get_size()
            grid = (max(1, W // GRID_SIZE), max(1, H // GRID_SIZE))
        total = max(r.steps for r in runs)
        cols, rows = _layout(len(runs), *self.screen.get_size(), grid)
        step, playing, speed, acc = 0, True, 1.0, 0.0
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        opts = {"trails": True, "targets": True, "events": True, "radio": True, "sense": True}
        timeline = None
        dragging = False

        while True:
            timeline = self._draw_frame(runs, step, grid, station, opts, title, playing, speed, total, cols, rows)
            W, H = self.screen.get_size()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN:
                    shift = e.mod & pygame.KMOD_SHIFT
                    ctrl = e.mod & pygame.KMOD_CTRL
                    jump = 300 if ctrl else 30 if shift else 1
                    if e.key == pygame.K_ESCAPE:
                        return
                    elif e.key == pygame.K_SPACE:
                        playing = not playing
                        if step >= total - 1:
                            step = 0
                    elif e.key == pygame.K_RIGHT:
                        step = min(total - 1, step + jump)
                        playing = False
                    elif e.key == pygame.K_LEFT:
                        step = max(0, step - jump)
                        playing = False
                    elif e.key == pygame.K_UP:
                        speed = speeds[min(len(speeds) - 1, speeds.index(speed) + 1)]
                    elif e.key == pygame.K_DOWN:
                        speed = speeds[max(0, speeds.index(speed) - 1)]
                    elif e.key == pygame.K_HOME:
                        step = 0
                    elif e.key == pygame.K_END:
                        step = total - 1
                    elif e.key == pygame.K_t:
                        opts["trails"] = not opts["trails"]
                    elif e.key == pygame.K_g:
                        opts["targets"] = not opts["targets"]
                    elif e.key == pygame.K_e:
                        opts["events"] = not opts["events"]
                    elif e.key == pygame.K_r:
                        opts["radio"] = not opts["radio"]
                    elif e.key == pygame.K_p:
                        opts["sense"] = not opts["sense"]
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and timeline.inflate(0, 20).collidepoint(e.pos):
                    dragging = True
                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    dragging = False
                if dragging and e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                    frac = (e.pos[0] - timeline.x) / max(1, timeline.w)
                    step = int(max(0.0, min(1.0, frac)) * (total - 1))
                    playing = False

            if playing:
                acc += speed * STEPS_PER_SECOND / 30.0
                n = int(acc)
                acc -= n
                step += n
                if step >= total - 1:
                    step, playing = total - 1, False

            pygame.display.flip()
            self.clock.tick(30)

    def _draw_frame(self, runs, step, grid, station, opts, title, playing, speed, total, cols, rows) -> pygame.Rect:
        """Draw one full replay frame (all panels, timeline, hints). Returns the timeline rectangle."""
        W, H = self.screen.get_size()
        top, bottom = 40, 118
        pw, ph = (W - 12 * (cols + 1)) / cols, (H - top - bottom - 12 * (rows - 1)) / rows
        timeline = pygame.Rect(40, H - bottom + 22, W - 80, 14)
        self.screen.fill((20, 24, 30))
        head = self.font.render(title, True, (120, 200, 255))
        self.screen.blit(head, (14, 6))
        for k, run in enumerate(runs):
            cx, cy = k % cols, k // cols
            rect = pygame.Rect(12 + cx * (pw + 12), top + cy * (ph + 12), pw, ph)
            self._draw_panel(rect, run, step, grid, station, opts, k)
        pygame.draw.rect(self.screen, (60, 66, 78), timeline, border_radius=6)
        fill = pygame.Rect(timeline.x, timeline.y, max(4, timeline.w * step / max(1, total - 1)), timeline.h)
        pygame.draw.rect(self.screen, (90, 150, 230), fill, border_radius=6)
        # where each strategy ended, on the shared timeline
        for k, run in enumerate(runs):
            x = timeline.x + timeline.w * (run.steps - 1) / max(1, total - 1)
            pygame.draw.line(self.screen, AGENT_COLORS[k % len(AGENT_COLORS)], (x, timeline.y - 5), (x, timeline.bottom + 5), 3)
            lab = self.small.render(str(k + 1), True, (235, 235, 235))
            self.screen.blit(lab, lab.get_rect(center=(x, timeline.y - 13)))
        # shared clock and how long each strategy ran
        clock_txt = self.large_font.render(f"{step / 60:7.1f} s", True, (235, 235, 235))
        self.screen.blit(clock_txt, (W - clock_txt.get_width() - 14, 0))
        info = self.font.render(
            f"step {step} / {total - 1}   speed {speed:g}x   {'playing' if playing else 'paused'}", True, (225, 225, 225))
        self.screen.blit(info, (40, H - bottom + 44))
        ran = "   ".join(
            f"[{k + 1}] {r.steps / 60:.1f} s" + (f" (computed in {r.wall_seconds:.1f} s)" if r.wall_seconds else "")
            for k, r in enumerate(runs))
        ran_surf = self.small.render("ran for: " + ran, True, (200, 205, 215))
        self.screen.blit(ran_surf, (40, H - bottom + 72))
        hint = self.small.render(
            "SPACE play/pause | LEFT/RIGHT step (SHIFT 30, CTRL 300) | UP/DOWN speed | HOME/END | click timeline | "
            "T trails | G targets | E events | R radio | P perception | ESC back", True, (160, 165, 175))
        self.screen.blit(hint, (40, H - 22))
        return timeline
