"""
eval_rl.py — Fixed Evaluation Harness for AutoRL-LRM
=====================================================
DO NOT MODIFY THIS FILE.
Consistent evaluation across all experiments.

Metrics:
  pass@1        — greedy decoding exact match
  pass@8        — best-of-8 sampling exact match
  escape_radius — mean KL divergence from base model (leash metric)
"""

import os, re, json, math, torch, numpy as np
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer, AutoModelForCausalLM

AUTORL_HOME = os.environ.get("AUTORL_HOME", ".")
AUTORL_CHECKPOINTS = os.path.join(AUTORL_HOME, "checkpoints")

BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
VAL_PATH       = "data/math_val.jsonl"
CHECKPOINT_DIR = Path(os.environ.get("AUTORL_CHECKPOINTS", "checkpoints")) / "latest"
N_EVAL         = 200
K_SAMPLES      = 8
MAX_NEW_TOKENS = 512
KL_SAMPLE_SIZE = 50

SYSTEM_PROMPT = (
    "You are a mathematics expert. "
    "Solve problems step by step, "
    "then state the final answer as \\boxed{answer}."
)


def make_prompt(problem):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n\n<|im_start|>user\n{problem}\n\n<|im_start|>assistant\n"
    )
