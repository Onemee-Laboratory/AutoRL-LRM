"""
eval_rl.py — Fixed Evaluation Harness for AutoRL-LRM
DO NOT MODIFY THIS FILE.
"""

import os, re, json, torch, numpy as np
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
VAL_PATH       = os.environ.get("AUTORL_DATA", "data") + "/math_val.jsonl"
CHECKPOINT_DIR = sorted(Path(os.environ.get("AUTORL_CHECKPOINTS", "checkpoints")).glob("latest/checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))[-1]
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
    return f"<|im_start|>system\n{SYSTEM_PROMPT}\n\n<|im_start|>user\n{problem}\n\n<|im_start|>assistant\n"

def extract_answer(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None

def load_val(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def eval_pass_at_1(model, tokenizer, dataset, device):
    correct = 0
    model.eval()
    with torch.no_grad():
        for item in dataset[:N_EVAL]:
            prompt = make_prompt(item["problem"])
            inputs = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            pred = extract_answer(response)
            if pred and str(pred).strip() == str(item["answer"]).strip():
                correct += 1
    return correct / min(N_EVAL, len(dataset))

def eval_pass_at_k(model, tokenizer, dataset, device, k=8):
    solved = 0
    model.eval()
    with torch.no_grad():
        for item in dataset[:N_EVAL]:
            prompt = make_prompt(item["problem"])
            inputs = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.8,
                num_return_sequences=k,
                pad_token_id=tokenizer.eos_token_id
            )
            responses = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            if any(extract_answer(r) and
                   str(extract_answer(r)).strip() == str(item["answer"]).strip()
                   for r in responses):
                solved += 1
    return solved / min(N_EVAL, len(dataset))

def compute_escape_radius(model, base_model, tokenizer, dataset, device):
    kl_values = []
    model.eval()
    base_model.eval()
    with torch.no_grad():
        for item in dataset[:KL_SAMPLE_SIZE]:
            prompt = make_prompt(item["problem"])
            inputs = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=256).to(device)
            trained_logits = model(**inputs).logits[0, -1, :]
            base_logits = base_model(**inputs).logits[0, -1, :]
            trained_probs = torch.softmax(trained_logits, dim=-1)
            base_log_probs = torch.log_softmax(base_logits, dim=-1)
            kl = torch.nn.functional.kl_div(
                base_log_probs, trained_probs, reduction='sum'
            ).item()
            kl_values.append(kl)
    return float(np.mean(kl_values))

def evaluate():
    print("=" * 60)
    print("AutoRL-LRM — Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {CHECKPOINT_DIR}")

    if not CHECKPOINT_DIR.exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT_DIR}")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(CHECKPOINT_DIR), dtype=torch.bfloat16, device_map="auto"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    dataset = load_val(VAL_PATH)
    print(f"Evaluating on {min(N_EVAL, len(dataset))} problems...")

    pass_at_1 = eval_pass_at_1(model, tokenizer, dataset, device)
    pass_at_k = eval_pass_at_k(model, tokenizer, dataset, device, k=K_SAMPLES)
    escape = compute_escape_radius(model, base_model, tokenizer, dataset, device)

    print(f"\n---")
    print(f"pass@1:          {pass_at_1:.6f}")
    print(f"pass@{K_SAMPLES}:          {pass_at_k:.6f}")
    print(f"escape_radius:   {escape:.6f}")

if __name__ == "__main__":
    evaluate()
