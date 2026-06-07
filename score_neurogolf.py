import os
import math
from pathlib import Path
import onnx

def calculate_neurogolf_points(mem_bytes: int, params: int) -> float:
    """
    Computes the exact per-task score based on NeuroGolf formula:
    Points = max(1.0, 25.0 - log(Mem_bytes + Params))
    """
    cost = mem_bytes + params
    if cost <= 0:
        return 25.0 # Max boundary safety

    score = 25.0 - math.log(cost)
    return max(1.0, score)

def profile_onnx_file(file_path: Path):
    """
    Statically inspects an ONNX model to extract total parameter count
    and estimates the graph's footprint memory bytes.
    """
    try:
        model = onnx.load(str(file_path))
        graph = model.graph

        # 1. Count Parameters from Initializers (Weights/Biases)
        total_params = 0
        for initializer in graph.initializer:
            # Multiply dimensions to get total elements in tensor
            dims = initializer.dims
            elements = 1
            for d in dims:
                elements *= d
            total_params += elements

        # 2. Approximate Activation / Value Info Footprint Bytes
        # (Emulates NeuroGolf scorer's static evaluation pass)
        total_mem_bytes = 0
        for value_info in graph.value_info:
            tensor_type = value_info.type.tensor_type
            if tensor_type.HasField("shape"):
                elements = 1
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        elements *= dim.dim_value

                # Assuming standard Float32 tensors (4 bytes per element)
                total_mem_bytes += (elements * 4)

        # Handle fallback for ultra-tiny graphs
        if total_mem_bytes == 0:
            total_mem_bytes = total_params * 4

        return total_mem_bytes, total_params

    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

def evaluate_submission_folder(folder_path: str):
    """
    Scans a folder of ONNX files, calculates their individual scores,
    and aggregates them to display your cumulative leaderboard score.
    """
    target_dir = Path(folder_path)
    if not target_dir.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    onnx_files = list(target_dir.glob("*.onnx"))
    print(f"Found {len(onnx_files)} ONNX solution files.\n")
    print(f"{'Task File':<30} | {'Memory (B)':<12} | {'Params':<10} | {'Points':<8}")
    print("-" * 70)

    total_competition_score = 0.0
    solved_count = 0

    for file in onnx_files:
        metrics = profile_onnx_file(file)
        if metrics is None:
            continue

        mem_bytes, params = metrics
        points = calculate_neurogolf_points(mem_bytes, params)

        total_competition_score += points
        solved_count += 1

        print(f"{file.name:<30} | {mem_bytes:<12,} | {params:<10,} | {points:.3f}")

    print("-" * 70)
    print(f"Total Tasks Evaluated: {solved_count}")
    print(f"Predicted Cumulative Leaderboard Score: {total_competition_score:.3f} Points")

# --- EXECUTION ---
if __name__ == "__main__":
    # Replace with the path to the folder holding your compiled .onnx files
    SUBMISSION_DIR = "./my_onnx_submissions"

    # Create mock folder if it doesn't exist yet for testing
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    evaluate_submission_folder(SUBMISSION_DIR)
