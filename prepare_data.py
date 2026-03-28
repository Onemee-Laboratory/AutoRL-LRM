"""
prepare_data.py — Dataset Preparation for AutoRL-LRM
Run once: python prepare_data.py
"""

import os
import json
import random
import re
from pathlib import Path
from datasets import load_dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_DIR   = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TRAIN_FILE = DATA_DIR / "math_train.jsonl"
VAL_FILE   = DATA_DIR / "math_val.jsonl"
SEED       = 42

random.seed(SEED)

SUBJECTS = ["algebra", "counting_and_probability", "geometry",
            "number_theory", "prealgebra", "precalculus"]

def extract_answer(solution: str) -> str:
    boxed = re.findall(r'\\boxed\{([^}]+)\}', solution)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r'-?\d+\.?\d*', solution)
    return numbers[-1] if numbers else solution.strip()[-20:]

def prepare():
    all_train, all_test = [], []

    for subject in SUBJECTS:
        print(f"Loading {subject}...")
        ds = load_dataset("EleutherAI/hendrycks_math", subject)
        for item in ds["train"]:
            all_train.append({
                "problem":  item["problem"],
                "solution": item["solution"],
                "answer":   extract_answer(item["solution"]),
                "level":    item["level"],
                "type":     item["type"],
            })
        for item in ds["test"]:
            all_test.append({
                "problem":  item["problem"],
                "solution": item["solution"],
                "answer":   extract_answer(item["solution"]),
                "level":    item["level"],
                "type":     item["type"],
            })

    random.shuffle(all_train)
    random.shuffle(all_test)

    with open(TRAIN_FILE, "w") as f:
        for item in all_train:
            f.write(json.dumps(item) + "\n")

    with open(VAL_FILE, "w") as f:
        for item in all_test:
            f.write(json.dumps(item) + "\n")

    print(f"Train: {len(all_train)} items → {TRAIN_FILE}")
    print(f"Val:   {len(all_test)} items → {VAL_FILE}")
    print("Done!")

if __name__ == "__main__":
    prepare()
