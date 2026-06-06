import os
import json
import csv

# Set your folder path here (e.g., "C:/Users/Name/Documents/ARC_Tasks")
folder_path = "." 
output_csv = "arc_tasks_summary.csv"

# Define the columns for the Excel-friendly CSV
headers = ["File Name", "Task ID", "Split (train/test)", "Example Index", "Input Grid", "Output Grid"]

def format_grid(grid):
    """Converts a 2D list grid into a readable string block for Excel."""
    if not grid:
        return ""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(headers)

    # Scan folder for JSON files
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Iterate through each Task ID in the JSON file
                for task_id, task_data in data.items():
                    for split in ["train", "test"]:
                        if split in task_data:
                            for idx, example in enumerate(task_data[split]):
                                input_grid = format_grid(example.get("input", []))
                                output_grid = format_grid(example.get("output", []))

                                # Write row data
                                writer.writerow([
                                    file_name,
                                    task_id,
                                    split,
                                    idx + 1,
                                    input_grid,
                                    output_grid
                                ])
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

print(f"Success! '{output_csv}' has been created and is ready for Excel.")
