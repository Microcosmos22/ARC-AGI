import torch, json, os, contextlib, logging
from copy import deepcopy
from peft import LoraConfig
from transformers import logging as hf_logging

import numpy as np

from datasets import Dataset
import warnings
"""
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
os.environ["DATASETS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)"""

challenges_file = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
solutions_file = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json"

def hflip(g): return [row[::-1] for row in g]
def vflip(g): return g[::-1]
def tpose(g): return list(map(list, zip(*g)))
AUGS = [lambda x:x, hflip, vflip, tpose]

def convert_grid_to_string(grid) -> str:
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += " "
    return text.strip()

def load_task(j: int):
    """ Loads the j-th task, contains input-output pairs as well as the test task """
    inputs = []
    outputs = []
    test_task = 0

    with open(challenges_file, "r", encoding="utf-8") as f:
        challenges = json.load(f)
    with open(solutions_file, "r", encoding="utf-8") as f:
        solutions = json.load(f)


    for task_id, blocks in challenges.items():
        train_list = blocks.get("train", [])
        test_list = blocks.get("test", [])


    for task_id, blocks in solutions.items():
        test_label = blocks[0]

    for pair in train_list: # Load input-output pairs
        for grid_type in ["input", "output"]:
            grid = pair.get(grid_type, [])

            if grid_type == "input":
                inputs.append(convert_grid_to_string(grid))
            else:
                outputs.append(convert_grid_to_string(grid))

    for pair in test_list: # Load test task
        test_task = convert_grid_to_string(pair.get("input", []))

    return inputs, outputs, test_task, test_label

# =========================
# Config
# =========================

EPOCHS = 8
max_seq_length = 2048
BASE_MODEL = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
log_path = "/kaggle/working/results.txt"

lora_cfg = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)


from unsloth import UnslothTrainer, UnslothTrainingArguments

def encode(grid):
    return "".join("".join(map(str, row)) for row in grid)

def build_text(train_in, train_out, test_in, test_out):

    text = ""
    system_prompt = """<|im_start|>system\n You are an ARC-AGI reasoning system.
    Given several input-output examples, infer the transformation rule.Then apply the same rule to the test input.
    Return only the output grid. <|im_end|>"""

    text + (
        "<|im_start|>user\n"
        f"{encode(train_in)}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{encode(train_out)}\n"
        "<|im_end|>\n"
    )

    text += (
        "<|im_start|>user\n"
        f"{encode(test_in)}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{encode(test_out)}\n"
        "<|im_end|>"
    )
    return text

def exact_match_score(pred_grid, true_grid):
    """
    Returns 1 if grids are identical, else 0.
    Works for lists-of-lists or numpy
    arrays.
    """

    if pred_grid is None or true_grid is None:
        return 0

    # convert numpy -> list if needed
    if hasattr(pred_grid, "tolist"):
        pred_grid = pred_grid.tolist()
    if hasattr(true_grid, "tolist"):
        true_grid = true_grid.tolist()

    if len(pred_grid) != len(true_grid):
        return 0

    for r1, r2 in zip(pred_grid, true_grid):
        if r1 != r2:
            return 0

    return 1

from unsloth import FastLanguageModel
from datasets import Dataset
import torch

# Load base model ONCE
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=True,
    max_seq_length=2048,
    dtype=torch.float16,
    attn_implementation="eager",
)

if __name__ == "__main__":

    for j in range(200):
        print(j)

        # fresh LoRA adapter per task (your current design choice preserved)
        model = FastLanguageModel.get_peft_model(base_model, **lora_cfg)

        # load task
        train_in, train_out, test_in, test_label = load_task(j)

        rows = []

        for f in AUGS:
            for x_in, x_out in zip(train_in, train_out):
                rows.append({
                    "text": build_text(
                        f(x_in),f(x_out),test_in,test_label)})

        ds = Dataset.from_list(rows)

        print(len(ds))
        print(ds[:2])
        print(type(ds[0]["text"]))

        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            args=UnslothTrainingArguments(
                max_steps=EPOCHS,
                per_device_train_batch_size=1,
                learning_rate=2e-4,
                report_to="none",
            ),
        )

        trainer.train()

        # inference
        model = FastLanguageModel.for_inference(model)

        prompt = build_inference_text(train_in, train_out, test_in)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True)

        print("PRED:", pred)
        print(f"Score for task {j}: {exact_match_score(pred, test_label)}")
