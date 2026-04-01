"""
prepare_data.py — Dataset Preparation for AutoRL-LRM
Run once: python prepare_data.py
"""

import os
import sys
import json
import random
import re
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Paths — from env.sh, no hardcode
# ---------------------------------------------------------------------------

try:
    DATA_DIR = Path(os.environ["AUTORL_DATA"])
except KeyError:
    print("ERROR: AUTORL_DATA is not set. Did you run 'source env.sh'?")
    sys.exit(1)

DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = DATA_DIR / "math_train.jsonl"
VAL_FILE   = DATA_DIR / "math_val.jsonl"

SEED = 42
random.seed(SEED)

SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "number_theory", "prealgebra", "precalculus",
]

# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

MIN_LINES   = 100
SAMPLE_SIZE = 20


def is_valid(path: Path) -> tuple[bool, str]:
    """
    Returns (True, "") if file is valid, (False, reason) otherwise.
    Checks: exists, min line count, JSON parseable, required fields.
    """
    if not path.exists():
        return False, "file missing"

    lines = path.read_text().strip().splitlines()

    if len(lines) < MIN_LINES:
        return False, f"only {len(lines)} lines (expected >= {MIN_LINES})"

    # sample first, middle, last + random interior
    indices = set([0, len(lines) // 2, len(lines) - 1])
    indices.update(range(min(SAMPLE_SIZE, len(lines))))
    for i in indices:
        try:
            record = json.loads(lines[i])
        except json.JSONDecodeError:
            return False, f"invalid JSON at line {i}"
        for field in ("problem", "answer"):
            if field not in record:
                return False, f"missing field '{field}' at line {i}"

    return True, ""


def get_hf_file_size(dataset_id: str, filename: str) -> int:
    """
    Query HuggingFace API for a specific parquet file size (Content-Length).
    Returns size in bytes, or -1 if unavailable.
    """
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    url = f"{hf_endpoint}/api/datasets/{dataset_id}"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "AutoRL-LRM/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            for sibling in data.get("siblings", []):
                if sibling.get("rfilename", "") == filename:
                    return sibling.get("size", -1)
    except Exception:
        pass
    return -1


def check_and_clean(path: Path, label: str) -> bool:
    """
    Check file integrity. If invalid, delete and return False.
    If valid, print summary and return True.
    """
    ok, reason = is_valid(path)
    if ok:
        n = sum(1 for _ in path.open())
        print(f"  ✅ {label}: {n} lines — valid")
        return True
    else:
        print(f"  ⚠️  {label}: {reason} — will re-download")
        if path.exists():
            path.unlink()
        return False


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(solution: str) -> str:
    boxed = re.findall(r'\\boxed\{([^}]+)\}', solution)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r'-?\d+\.?\d*', solution)
    return numbers[-1] if numbers else solution.strip()[-20:]


# ---------------------------------------------------------------------------
# Download + write
# ---------------------------------------------------------------------------

def prepare():
    print(f"DATA_DIR: {DATA_DIR}")
    print("")

    # --- check existing files ---
    train_ok = check_and_clean(TRAIN_FILE, "math_train.jsonl")
    val_ok   = check_and_clean(VAL_FILE,   "math_val.jsonl")

    if train_ok and val_ok:
        print("\nData already complete and valid — nothing to do.")
        sys.exit(0)

    # --- query HF for expected sizes (early signal before download) ---
    print("\nQuerying HuggingFace for expected dataset size ...")
    hf_size = get_hf_file_size("EleutherAI/hendrycks_math", "README.md")
    if hf_size > 0:
        print(f"  HF dataset info reachable (README: {hf_size} bytes) ✅")
    else:
        print("  HF dataset info unavailable — proceeding anyway")

    # --- download ---
    all_train, all_val = [], []

    for subject in SUBJECTS:
        if train_ok and not val_ok:
            # only val missing — still need to rebuild both from source
            # (we cannot easily split per-subject after the fact)
            pass
        print(f"  Loading {subject} ...")
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
            all_val.append({
                "problem":  item["problem"],
                "solution": item["solution"],
                "answer":   extract_answer(item["solution"]),
                "level":    item["level"],
                "type":     item["type"],
            })

    random.shuffle(all_train)
    random.shuffle(all_val)

    # --- write only files that need it ---
    if not train_ok:
        with open(TRAIN_FILE, "w") as f:
            for item in all_train:
                f.write(json.dumps(item) + "\n")
        print(f"\n  Wrote {len(all_train)} train items → {TRAIN_FILE}")

    if not val_ok:
        with open(VAL_FILE, "w") as f:
            for item in all_val:
                f.write(json.dumps(item) + "\n")
        print(f"  Wrote {len(all_val)} val items → {VAL_FILE}")

    # --- final integrity check ---
    print("\nFinal integrity check ...")
    train_ok = check_and_clean(TRAIN_FILE, "math_train.jsonl")
    val_ok   = check_and_clean(VAL_FILE,   "math_val.jsonl")

    if train_ok and val_ok:
        print("\n✅ Done.")
    else:
        print("\n❌ Integrity check failed after download. Check your HF connection.")
        sys.exit(1)


if __name__ == "__main__":
    prepare()
