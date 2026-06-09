import json
import os
import numpy as np

def run_arc_eda(split_name, challenges_file, solutions_file):
    """Runs a combined structural and analytical EDA on a matching pair of ARC files."""
    if not (os.path.exists(challenges_file) and os.path.exists(solutions_file)):
        print(f"Error: Missing files for {split_name}. Ensure both challenges and solutions exist.")
        return

    with open(challenges_file, "r", encoding="utf-8") as f:
        challenges = json.load(f)
    with open(solutions_file, "r", encoding="utf-8") as f:
        solutions = json.load(f)

    # Trackers for basic structural statistics
    total_tasks = len(challenges)
    total_train_pairs = 0
    total_test_queries = 0
    all_dimensions = []

    # Trackers for behavior and color metrics
    same_size_count = 0
    scaled_size_count = 0
    dynamic_size_count = 0
    all_colors = []
    unique_colors_per_task = []

    for task_id, blocks in challenges.items():
        train_list = blocks.get("train", [])
        test_list = blocks.get("test", [])
        test_outputs = solutions.get(task_id, [])

        total_train_pairs += len(train_list)
        total_test_queries += len(test_list)

        # 1. Collect Grid Dimensions from Train Split
        for pair in train_list:
            for grid_type in ["input", "output"]:
                grid = pair.get(grid_type, [])
                if grid:
                    all_dimensions.append((len(grid), len(grid[0])))

        # 2. Collect Grid Dimensions from Test Inputs
        for pair in test_list:
            grid = pair.get("input", [])
            if grid:
                all_dimensions.append((len(grid), len(grid[0])))

        # 3. Analyze Spatial Shifts (Input vs Output Grid Size Transformations)
        for idx, test_in in enumerate(test_list):
            in_grid = test_in.get("input", [])
            out_grid = test_outputs[idx] if idx < len(test_outputs) else []

            if in_grid and out_grid:
                in_shape = (len(in_grid), len(in_grid[0]))
                out_shape = (len(out_grid), len(out_grid[0]))

                if in_shape == out_shape:
                    same_size_count += 1
                elif out_shape[0] % in_shape[0] == 0 and out_shape[1] % in_shape[1] == 0:
                    scaled_size_count += 1
                else:
                    dynamic_size_count += 1

        # 4. Analyze Color Profiles
        task_colors = set()
        for pair in train_list:
            for grid in [pair.get("input", []), pair.get("output", [])]:
                for row in grid:
                    all_colors.extend(row)
                    task_colors.update(row)
        unique_colors_per_task.append(len(task_colors))

    # Calculate extremes and aggregates safely
    min_rows = min(d[0] for d in all_dimensions) if all_dimensions else 0
    max_rows = max(d[0] for d in all_dimensions) if all_dimensions else 0
    min_cols = min(d[1] for d in all_dimensions) if all_dimensions else 0
    max_cols = max(d[1] for d in all_dimensions) if all_dimensions else 0

    colors_arr = np.array(all_colors)
    bg_percentage = (np.sum(colors_arr == 0) / len(colors_arr)) * 100 if len(colors_arr) > 0 else 0
    avg_unique_colors = np.mean(unique_colors_per_task) if unique_colors_per_task else 0

    # Print Clean Combined Summary Report
    print("=" * 55)
    print(f"       ARC-AGI COMPREHENSIVE EDA: {split_name.upper()}     ")
    print("=" * 55)
    print(f"• Total Unique Tasks Found:          {total_tasks}")
    print(f"• Total Train Demonstration Pairs:   {total_train_pairs}")
    print(f"• Total Test Prediction Queries:     {total_test_queries}")
    print(f"• Average Train Pairs per Task:      {total_train_pairs / total_tasks:.1f}")
    print(f"• Grid Dimension Bounds (Row x Col): {min_rows}x{min_cols} to {max_rows}x{max_cols}")
    print(f"• Background (0) Cell Density:       {bg_percentage:.1f}% of all pixels")
    print(f"• Avg Unique Colors Used Per Task:   {avg_unique_colors:.1f} colors")
    print("\n--- TEST GRID DIMENSION TRANSFORMATIONS ---")
    print(f"• Output keeps exact Input size:     {same_size_count} tasks")
    print(f"• Output scales by integer multiple: {scaled_size_count} tasks")
    print(f"• Output changes size dynamically:   {dynamic_size_count} tasks")
    print("=" * 55)
    print("\n")

# Execution block
if __name__ == "__main__":
    # Run 1: Training Set
    run_arc_eda(
        split_name="Training Dataset",
        challenges_file="arc-agi_training_challenges.json",
        solutions_file="arc-agi_training_solutions.json"
    )

    # Run 2: Evaluation Set
    run_arc_eda(
        split_name="Evaluation Dataset",
        challenges_file="arc-agi_evaluation_challenges.json",
        solutions_file="arc-agi_evaluation_solutions.json"
    )
