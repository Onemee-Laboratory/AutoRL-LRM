"""
train_rl.py — RLVR Training Loop for AutoRL-LRM
Uses trl library for proper GRPO/PPO implementation.
This file is modified by the AI agent each experiment.
Only the === AGENT MODIFIES THESE === section should be changed.
"""

import os
import re
import time
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# === AGENT MODIFIES THESE ===
# ============================================================

LR                  = 1e-6          # learning rate
KL_COEFF            = 0.01          # KL penalty weight
REWARD_SHAPING      = "binary"      # binary | dense | confidence_gated
DEVIATION_BONUS     = 0.0           # CGDB bonus coefficient (0 = disabled)
DEVIATION_THRESHOLD = 0.5           # CGDB: min reward to trigger bonus
TEMPERATURE         = 0.8           # sampling temperature
N_SAMPLES           = 8             # samples per prompt (GRPO group size)
TRAIN_STEPS         = 100           # training steps
GRAD_ACCUM          = 4             # gradient accumulation steps
MAX_NEW_TOKENS      = 256           # max tokens to generate
LR_SCHEDULER        = "cosine"      # cosine | constant | linear
MAX_PROMPT_LENGTH   = 256           # max prompt length

# ============================================================
# === DO NOT MODIFY BELOW THIS LINE ===
# ============================================================

AUTORL_HOME      = os.environ.get("AUTORL_HOME", "/home/oz/workspace/working/RL/autorl-lrm")
Autorl_DATA      = AUTORL_HOME + "/data"
AUTORL_CHECKPOINTS = AUTORL_HOME + "/checkpoints"
BASE_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PATH       = os.environ.get("AUTORL_DATA", AUTORL_HOME + "/data") + "/math_train.jsonl"
RESULTS_FILE     = "results_rl.tsv"
SEED             = 42


def extract_answer(text: str):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None


def make_reward_fn(reward_shaping: str, base_model=None, tokenizer=None):
    """Factory for reward functions"""

    def binary_reward(completions, prompts=None, answer=None, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            pred = extract_answer(text)
            correct = (pred is not None and
                      answer is not None and
                      str(pred).strip() == str(answer[0]).strip())
            rewards.append(1.0 if correct else 0.0)
        return rewards

    def dense_reward(completions, prompts=None, answer=None, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            pred = extract_answer(text)
            if pred is None:
                rewards.append(0.0)
                continue
            correct = str(pred).strip() == str(answer[0]).strip()
            if correct:
                rewards.append(1.0)
            else:
                try:
                    pred_val = float(pred)
                    true_val = float(answer[0])
                    diff = abs(pred_val - true_val)
                    rewards.append(max(0.0, 0.3 - diff / (abs(true_val) + 1e-8)))
                except ValueError:
                    rewards.append(0.1)
        return rewards

    def confidence_gated_reward(completions, prompts=None, answer=None, **kwargs):
        """CGDB: Confidence-Gated Deviation Bonus"""
        base_rewards = binary_reward(completions, prompts=prompts,
                                     answer=answer, **kwargs)
        rewards = []
        for i, (completion, base_r) in enumerate(zip(completions, base_rewards)):
            if base_r >= DEVIATION_THRESHOLD and DEVIATION_BONUS > 0:
                # Add deviation bonus for correct low-probability responses
                bonus = DEVIATION_BONUS * base_r
                rewards.append(base_r + bonus)
            else:
                rewards.append(base_r)
        return rewards

    if reward_shaping == "binary":
        return binary_reward
    elif reward_shaping == "dense":
        return dense_reward
    elif reward_shaping == "confidence_gated":
        return confidence_gated_reward
    else:
        return binary_reward


def load_train_dataset(path: str) -> Dataset:
    """Load and format dataset for GRPOTrainer"""
    items = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            items.append({
                "prompt": (
                    f"Solve this math problem step by step.\n\n"
                    f"Problem: {item['problem']}\n\n"
                    f"Solution:"
                ),
                "answer": item["answer"],
            })
    return Dataset.from_list(items)


def train():
    print(f"lr:              {LR}")
    print(f"kl_coeff:        {KL_COEFF}")
    print(f"reward_shaping:  {REWARD_SHAPING}")
    print(f"deviation_bonus: {DEVIATION_BONUS}")
    print(f"n_samples:       {N_SAMPLES}")
    print(f"train_steps:     {TRAIN_STEPS}")

    start_time = time.time()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GRPO config
    config = GRPOConfig(
        output_dir=os.environ.get("AUTORL_CHECKPOINTS", AUTORL_HOME + "/checkpoints") + "/latest",
        learning_rate=LR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=TRAIN_STEPS,
        num_generations=N_SAMPLES,
        temperature=TEMPERATURE,
        beta=KL_COEFF,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        logging_steps=10,
        save_steps=TRAIN_STEPS,
        report_to="none",
        bf16=True,
    )

    # Load dataset
    dataset = load_train_dataset(TRAIN_PATH)

    # Reward function
    reward_fn = make_reward_fn(REWARD_SHAPING)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    trainer.train()

    training_seconds = time.time() - start_time

    print(f"\n---")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"train_steps:      {TRAIN_STEPS}")
    print(f"reward_shaping:   {REWARD_SHAPING}")
    print(f"lr:               {LR}")
    print("Model saved to checkpoints/latest")


if __name__ == "__main__":
    train()
