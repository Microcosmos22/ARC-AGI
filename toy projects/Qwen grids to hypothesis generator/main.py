from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

input_grid = [
    [0,1,0],
    [0,1,0],
    [0,1,0]
]

output_grid = [
    [0,0,0],
    [1,1,1],
    [0,0,0]
]

prompt = f"""
You are an ARC reasoning system.

Input:
{input_grid}

Output:
{output_grid}

Suggest 5 possible symbolic transformations.

Return JSON only.

Example:

{{
    "hypotheses":[
        "rotate90",
        "reflect_horizontal"
    ]
}}
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(text)
