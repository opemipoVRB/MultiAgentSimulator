# src/utils.py
import pygame
from pathlib import Path
from typing import Optional, Union

SurfaceOrPath = Union[pygame.Surface, Path, str]

def load_image(path: Union[Path, str, None]) -> Optional[pygame.Surface]:
    """
    Load an image via pygame and return a Surface or None if loading failed.
    Accepts Path, string, or None.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = pygame.image.load(str(p))
        # prefer alpha if available, fall back if convert_alpha fails
        try:
            return img.convert_alpha()
        except Exception:
            return img.convert()
    except Exception:
        return None


def scale_to_cell(img: Optional[pygame.Surface], grid_size: int, scale_factor: float) -> Optional[pygame.Surface]:
    """
    Scale `img` so it fits in a square area ~ grid_size * scale_factor while preserving aspect ratio.
    Returns a new Surface or None if img is None.
    - grid_size: pixels per cell
    - scale_factor:  e.g. 1.0 => full cell, 0.7 => 70% of cell size
    """
    if img is None:
        return None

    try:
        w, h = img.get_size()
    except Exception:
        return None

    # target maximum dimension in pixels (fit into a square of this size)
    target = max(1, int(grid_size * float(scale_factor)))

    # compute scale ratio preserving aspect ratio based on the larger dimension
    if max(w, h) == 0:
        return img
    scale_ratio = target / float(max(w, h))

    new_w = max(1, int(round(w * scale_ratio)))
    new_h = max(1, int(round(h * scale_ratio)))

    try:
        return pygame.transform.smoothscale(img, (new_w, new_h))
    except Exception:
        try:
            return pygame.transform.scale(img, (new_w, new_h))
        except Exception:
            return img


def cell_center_from_grid(col: int, row: int, grid_size: int):
    x = col * grid_size + grid_size / 2
    y = row * grid_size + grid_size / 2
    return pygame.Vector2(x, y)


def grid_from_pos(pos: pygame.Vector2, grid_size: int):
    col = int(pos.x // grid_size)
    row = int(pos.y // grid_size)
    return col, row
