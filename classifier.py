import pandas as pd
from pathlib import Path

df = pd.read_json(Path("task_class")/ Path("arc_primitives.json")).T

# Optional: Reset index to turn "task001", "task002" into an explicit column
df = df.reset_index().rename(columns={"index": "Task_ID"})

print(df["Primary_Category"])

df["count"] = df.groupby("Primary_Category")["Primary_Category"].transform("count")

print(df)
