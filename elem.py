"""
hypothesis_space.py

Elementary function library for ARC-style reasoning systems.

Each function is a candidate hypothesis primitive:
- spatial/geometric transforms
- object-level operations
- color/logical rules
- pattern completion operators
- grid metadata features
"""

from copy import deepcopy
from collections import deque
import numpy as np

Grid = list[list[int]]

# ============================================================
# Spatial & Geometric Transformations
# ============================================================

def rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def rotate180(grid: Grid) -> Grid:
    return rotate90(rotate90(grid))


def rotate270(grid: Grid) -> Grid:
    return rotate90(rotate180(grid))


def reflect_horizontal(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def reflect_vertical(grid: Grid) -> Grid:
    return grid[::-1]


def translate(grid: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    H, W = len(grid), len(grid[0])
    out = [[fill for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr][nc] = grid[r][c]
    return out


def crop(grid: Grid, r0: int, r1: int, c0: int, c1: int) -> Grid:
    return [row[c0:c1] for row in grid[r0:r1]]


def tile(grid: Grid, hr: int, hc: int) -> Grid:
    return [row * hc for row in grid] * hr


# ============================================================
# Object-Based Operations
# ============================================================

def components(grid: Grid):
    H, W = len(grid), len(grid[0])
    vis = [[False]*W for _ in range(H)]
    comps = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0 or vis[r][c]:
                continue

            color = grid[r][c]
            q = deque([(r, c)])
            cells = []

            while q:
                x, y = q.popleft()
                if not (0 <= x < H and 0 <= y < W):
                    continue
                if vis[x][y] or grid[x][y] != color:
                    continue

                vis[x][y] = True
                cells.append((x, y))

                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    q.append((x+dx, y+dy))

            comps.append({"color": color, "cells": cells})

    return comps


def bounding_box(obj):
    rs = [r for r, c in obj["cells"]]
    cs = [c for r, c in obj["cells"]]
    return min(rs), max(rs), min(cs), max(cs)


def sort_by_size(objects):
    return sorted(objects, key=lambda o: len(o["cells"]))


def detect_outliers(objects):
    sizes = np.array([len(o["cells"]) for o in objects])
    if len(sizes) == 0:
        return []
    mean, std = sizes.mean(), sizes.std() + 1e-6
    return [o for o in objects if abs(len(o["cells"]) - mean) > 2*std]


# ============================================================
# Color & Logical Rules
# ============================================================

def recolor(grid: Grid, old: int, new: int) -> Grid:
    return [[new if x == old else x for x in row] for row in grid]


def color_map(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(x, x) for x in row] for row in grid]


def background(grid: Grid):
    vals = [x for row in grid for x in row]
    return max(set(vals), key=vals.count)


def flood_fill(grid: Grid, sr: int, sc: int, new_color: int) -> Grid:
    H, W = len(grid), len(grid[0])
    target = grid[sr][sc]
    if target == new_color:
        return grid

    out = deepcopy(grid)
    q = deque([(sr, sc)])

    while q:
        r, c = q.popleft()
        if not (0 <= r < H and 0 <= c < W):
            continue
        if out[r][c] != target:
            continue

        out[r][c] = new_color

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            q.append((r+dr, c+dc))

    return out


# ============================================================
# Pattern Recognition
# ============================================================

def symmetry_horizontal(grid: Grid):
    return grid == grid[::-1]


def symmetry_vertical(grid: Grid):
    return all(row == row[::-1] for row in grid)


def line_extrapolate(row: list[int]):
    if len(row) < 2:
        return row
    diff = row[-1] - row[-2]
    return row + [row[-1] + diff]


def inpaint_missing(grid: Grid, fill=0):
    return [[fill if x is None else x for x in row] for row in grid]


def find_intersections(grid: Grid):
    H, W = len(grid), len(grid[0])
    inter = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                inter.append((r, c))
    return inter


def path_exists(grid: Grid, start, goal, passable=0):
    H, W = len(grid), len(grid[0])
    q = deque([start])
    seen = set()

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        if (r, c) in seen:
            continue
        seen.add((r, c))

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == passable:
                q.append((nr, nc))

    return False


# ============================================================
# Grid Metadata
# ============================================================

def grid_size(grid: Grid):
    return len(grid), len(grid[0])


def complexity_score(grid: Grid):
    # heuristic: number of connected components + color diversity
    comps = components(grid)
    colors = len(set(x for row in grid for x in row))
    return len(comps) + colors


def size_change_flag(g1: Grid, g2: Grid):
    return grid_size(g1) != grid_size(g2)


# ============================================================
# Hypothesis registry
# ============================================================

HYPOTHESIS_SPACE = [
    # spatial
    rotate90,
    rotate180,
    rotate270,
    reflect_horizontal,
    reflect_vertical,
    translate,
    crop,
    tile,

    # object
    components,
    sort_by_size,
    detect_outliers,

    # color/logical
    recolor,
    color_map,
    flood_fill,
    background,

    # pattern
    symmetry_horizontal,
    symmetry_vertical,
    line_extrapolate,
    find_intersections,
    path_exists,

    # metadata
    grid_size,
    complexity_score,
]
