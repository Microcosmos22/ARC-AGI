import os
import json
from pathlib import Path

"""
This script reads your generated submission.json (the exact format required by the Kaggle ARC Prize Competition) and compares it directly
against a local folder containing the ground-truth ARC JSON files (like the public evaluation set).

"""

def load_json_file(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_arc_submission(submission_json_path: str, solution_folder_path: str):
    """
    Evaluates an ARC-AGI submission file against actual target ground truth tasks.
    Mimics Kaggle's top-2 attempt strict matching evaluation logic.
    """
    sub_path = Path(submission_json_path)
    sol_dir = Path(solution_folder_path)

    if not sub_path.exists():
        print(f"Error: Submission file not found at {submission_json_path}")
        return
    if not sol_dir.exists():
        print(f"Error: Solution tasks folder not found at {solution_folder_path}")
        return

    # Load submission predictions
    predictions = load_json_file(sub_path)

    total_tasks = 0
    solved_tasks = 0
    missing_tasks = 0

    print(f"\n--- ARC-AGI Evaluation Summary ---")
    print(f"{'Task ID':<12} | {'Test Pair':<10} | {'Status':<10}")
    print("-" * 40)

    # Iterate through ground truth solutions in the designated directory
    for sol_file in sol_dir.glob("*.json"):
        task_id = sol_file.stem  # e.g., "007bbfb7"
        task_data = load_json_file(sol_file)

        # Pull the 'test' list containing unseen puzzle outputs
        test_pairs = task_data.get("test", [])

        # If this task isn't in your submission, flag it as a failure
        if task_id not in predictions:
            missing_tasks += len(test_pairs)
            total_tasks += len(test_pairs)
            for i in range(len(test_pairs)):
                print(f"{task_id:<12} | {i:<10} | MISSING")
            continue

        task_predictions = predictions[task_id]  # Expected to be a list of predictions

        for index, actual_pair in enumerate(test_pairs):
            total_tasks += 1
            ground_truth_grid = actual_pair["output"]

            # Extract attempts for this specific test sub-index
            # Kaggle submission layout handles multiple test items per file: [{"attempt_1": [...], "attempt_2": [...]}]
            try:
                pred_item = task_predictions[index]
                attempt_1 = pred_item.get("attempt_1", None)
                attempt_2 = pred_item.get("attempt_2", None)
            except (IndexError, KeyError):
                attempt_1, attempt_2 = None, None

            # Check for a perfect matrix match against either attempt
            is_correct = (attempt_1 == ground_truth_grid) or (attempt_2 == ground_truth_grid)

            if is_correct:
                solved_tasks += 1
                status = "SUCCESS"
            else:
                status = "FAILED"

            print(f"{task_id:<12} | {index:<10} | {status}")

    # Final calculations
    print("-" * 40)
    if total_tasks == 0:
        print("No tasks were processed. Check your solution folder path.")
        return

    accuracy = (solved_tasks / total_tasks) * 100
    print(f"Total Test Tasks Checked: {total_tasks}")
    print(f"Successfully Solved:      {solved_tasks}")
    print(f"Failed / Wrong Answer:    {total_tasks - solved_tasks - missing_tasks}")
    print(f"Missing from Submission:  {missing_tasks}")
    print(f"\n👉 Final Accuracy Score:  {accuracy:.2f}%")

# --- EXECUTION ---
if __name__ == "__main__":
    # Path to the submission file your model produced
    MY_SUBMISSION_FILE = "submission.json"

    # Path to the directory holding your evaluation tasks (e.g. data/evaluation/)
    GROUND_TRUTH_FOLDER = "./arc-agi-tasks"

    evaluate_arc_submission(MY_SUBMISSION_FILE, GROUND_TRUTH_FOLDER)
