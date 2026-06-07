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
import pandas as pd
from pathlib import Path

import json
import math

from elem import *
from plot import *


Grid = list[list[int]]

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

def compose(*fns):
    def H(x):
        for f in fns:
            x = f(x)
        return x
    return H

def generate_hypotheses(base, depth=2):
    hyps = base[:]
    for d in range(2, depth + 1):
        for combo in product(base, repeat=d):
            #print(combo[1])
            hyps.append(compose(*combo))
    return hyps

def grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(ar == br for ar, br in zip(a, b))

def score_hypothesis(H, train_pairs, test_input, test_label) -> int:
    """
    Checks if a hypothesis fits all training data AND perfectly
    predicts the unseen test target grid.
    Returns 1 if a perfect match is found, 0 otherwise.
    """
    # 1. Verify it satisfies all training pairs first
    for pair in train_pairs:
        x = pair['input']
        y = pair['output']
        try:
            if not grids_equal(H(x), y):
                return 0
        except Exception:
            return 0

    # 2. Test accuracy matching (The core evaluation rule)
    try:
        prediction = H(test_input)
        if grids_equal(prediction, test_label):
            return 1  # Perfect match!
    except Exception:
        pass

    return 0

def find_best_hypothesis(train_pairs, test_input, test_label, depth=2):
    hyps = generate_hypotheses(HYPOTHESIS_SPACE, depth=depth)

    if hyps is not None:
        for H in hyps:
            score = score_hypothesis(H, train_pairs, test_input, test_label)
            if score == 0:
                print(f"Score: {score}")
                return H

    return None

if __name__ == "__main__":
    N_tasks = 100
    for i in range(N_tasks):
        # Load challenges using standard json module
        with open(Path("arc-agi_training_challenges.json"), "r", encoding="utf-8") as f:
            chal_dict = json.load(f)
        chal_df = pd.DataFrame.from_dict(chal_dict, orient="index")

        # Load solutions using standard json module
        with open(Path("arc-agi_training_solutions.json"), "r", encoding="utf-8") as f:
            sol_dict = json.load(f)
        sol_df = pd.DataFrame.from_dict(sol_dict, orient="index")

        train_pairs = chal_df["train"].iloc[i]
        task_dict = chal_df["test"].iloc[i][0]["input"]  # Get the first item in the list
        solution = sol_df.iloc[i][0]

        #plot_arc_task(train_pairs, task_dict)

        # toy example
        train = [([[1, 0], [0, 2]],
                [[0, 2], [1, 0]])]

        H = find_best_hypothesis(train_pairs, task_dict, solution, depth=2)
        print("Found hypothesis:", H)

        if H:
            predicted_test_output = H(task_dict)

        print("Found hypothesis:", H)
        
