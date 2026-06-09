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

def compute_accuracy_score(H, train_pairs):
    """
    Counts the total number of correct pixels across all training pairs.
    Returns: (correct_pixels, total_pixels)
    """
    total_correct = 0
    total_pixels = 0

    for pair in train_pairs:
        x = pair['input']
        y = pair['output']

        try:
            pred = H(x)

            # If dimensions do not match target, 0 pixels are correct for this pair
            if len(pred) != len(y) or len(pred[0]) != len(y[0]):
                total_pixels += len(y) * len(y[0])
                continue

            # Count matches row by row, pixel by pixel
            for r in range(len(y)):
                for c in range(len(y[0])):
                    total_pixels += 1
                    if pred[r][c] == y[r][c]:
                        total_correct += 1

        except Exception:
            # If the function crashes, it gets 0 correct pixels for this pair
            total_pixels += len(y) * len(y[0])
            continue

    return total_correct, total_pixels

def find_best_hypothesis_by_score(train_pairs):
    """
    Loops through the primitive HYPOTHESIS_SPACE, tests each one,
    and returns the function that achieves the highest pixel accuracy.
    """
    best_H = None
    best_pixel_count = -1

    print(f"\nEvaluating {len(HYPOTHESIS_SPACE)} primitive functions...")

    for H in HYPOTHESIS_SPACE:
        fn_name = H.__name__ if hasattr(H, '__name__') else "unknown_function"

        correct, total = compute_accuracy_score(H, train_pairs)
        accuracy_pct = (correct / total * 100) if total > 0 else 0

        #print(f"-> {fn_name:<25} | Correct Pixels: {correct}/{total} ({accuracy_pct:.1f}%)")

        # Track the function that got the most pixels correct
        if correct > best_pixel_count:
            best_pixel_count = correct
            best_H = H

    print(f"\n🏆 Winner chosen: {best_H.__name__ if best_H else 'None'} with {best_pixel_count} correct pixels.")
    return best_H


if __name__ == "__main__":
    N_tasks = 10
    for i in range(N_tasks):
        # Load challenges using standard json module
        with open(Path("../data/arc-agi_training_challenges.json"), "r", encoding="utf-8") as f:
            chal_dict = json.load(f)
        chal_df = pd.DataFrame.from_dict(chal_dict, orient="index")

        # Load solutions using standard json module
        with open(Path("../data/arc-agi_training_solutions.json"), "r", encoding="utf-8") as f:
            sol_dict = json.load(f)
        sol_df = pd.DataFrame.from_dict(sol_dict, orient="index")

        train_pairs = chal_df["train"].iloc[i]
        task_dict = chal_df["test"].iloc[i][0]["input"]  # Get the first item in the list
        solution = sol_df.iloc[i][0]

        #plot_arc_task(train_pairs, task_dict)

        # toy example
        train = [([[1, 0], [0, 2]],
                [[0, 2], [1, 0]])]

        H = find_best_hypothesis_by_score(train_pairs)#, task_dict, solution)
        print("Found hypothesis:", H)

        if H:
            predicted_test_output = H(task_dict)

        print("Found hypothesis:", H)
