"""
Minimal ARC hypothesis engine:
- grid transformations
- object extraction
- compositional hypothesis space
- brute + beam-style evaluation scaffold
"""

from copy import deepcopy
from collections import deque
from itertools import product

Grid = list[list[int]]

# ----------------------------
# Geometry primitives
# ----------------------------

def rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def mirror_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def mirror_v(grid: Grid) -> Grid:
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def shift(grid: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    H, W = len(grid), len(grid[0])
    out = [[fill for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr][nc] = grid[r][c]
    return out


def recolor(grid: Grid, old: int, new: int) -> Grid:
    return [[new if x == old else x for x in row] for row in grid]


def color_map(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(x, x) for x in row] for row in grid]


# ----------------------------
# Object extraction
# ----------------------------

def components(grid: Grid):
    H, W = len(grid), len(grid[0])
    vis = [[False] * W for _ in range(H)]
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
                    q.append((x + dx, y + dy))

            comps.append({"color": color, "cells": cells})

    return comps


def bbox(obj):
    rs = [r for r, c in obj["cells"]]
    cs = [c for r, c in obj["cells"]]
    return min(rs), max(rs), min(cs), max(cs)


def place_object(grid: Grid, obj_cells, value: int):
    out = deepcopy(grid)
    H, W = len(grid), len(grid[0])
    for r, c in obj_cells:
        if 0 <= r < H and 0 <= c < W:
            out[r][c] = value
    return out


# ----------------------------
# Composition system (hypotheses)
# ----------------------------

def compose(*fns):
    def H(x):
        for f in fns:
            x = f(x)
        return x
    return H


BASE_HYPOTHESES = [
    rotate90,
    mirror_h,
    mirror_v,
    transpose,
    lambda g: shift(g, 1, 0),
    lambda g: shift(g, 0, 1),
]


def generate_hypotheses(base, depth=2):
    hyps = base[:]
    for d in range(2, depth + 1):
        for combo in product(base, repeat=d):
            hyps.append(compose(*combo))
    return hyps


# ----------------------------
# Evaluation
# ----------------------------

def grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(ar == br for ar, br in zip(a, b))


def score_hypothesis(H, train_pairs):
    for x, y in train_pairs:
        try:
            if not grids_equal(H(x), y):
                return float("inf")
        except Exception:
            return float("inf")
    return 0


# ----------------------------
# Solver loop (brute baseline)
# ----------------------------

def find_best_hypothesis(train_pairs, depth=2):
    hyps = generate_hypotheses(BASE_HYPOTHESES, depth=depth)

    for H in hyps:
        if score_hypothesis(H, train_pairs) == 0:
            return H

    return None


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # toy example
    train = [
        (
            [[1, 0], [0, 2]],
            [[0, 2], [1, 0]]
        )
    ]

    H = find_best_hypothesis(train)

    print("Found hypothesis:", H)

    if H:
        print("Test:", H([[1, 0], [0, 2]]))
