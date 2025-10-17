# utils.py
from pathlib import Path
import pygame


def load_image(path: Path):
    if path.exists():
        return pygame.image.load(str(path)).convert_alpha()
    return None


def scale_to_cell(img, grid_size: int, scale_factor: float = 1.0):
    if img is None:
        return None
    target = int(grid_size * scale_factor)
    w, h = img.get_size()
    scale = target / max(w, h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return pygame.transform.smoothscale(img, new_size)


def cell_center_from_grid(col: int, row: int, grid_size: int):
    x = col * grid_size + grid_size / 2
    y = row * grid_size + grid_size / 2
    return pygame.Vector2(x, y)


def grid_from_pos(pos: pygame.Vector2, grid_size: int):
    col = int(pos.x // grid_size)
    row = int(pos.y // grid_size)
    return col, row
