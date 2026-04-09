"""
train_rl.py — RLVR Training Loop for AutoRL-LRM
================================================
Uses trl library for proper GRPO/PPO implementation.
This file is modified by the AI agent each experiment.
Only the === AGENT MODIFIES THESE === section should be changed.
"""

import os
import re
import time
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# === AGENT MODIFIES THESE ===
# ============================================================

LR                  = 1e-6          # learning rate
KL_COEFF            = 0.02          # KL penalty weight
REWARD_SHAPING      = "cgdb_asymmetric"  # binary | dense | confidence_gated | cgdb_asymmetric
DEVIATION_BONUS     = 0.0           # CGDB bonus coefficient (0 = disabled)
DEVIATION_THRESHOLD = 0.5           # CGDB: min reward to trigger bonus
TEMPERATURE         = 1.2           # sampling temperature
N_SAMPLES           = 16            # samples per prompt (GRPO group size)
TRAIN_STEPS         = 1000          # training steps
GRAD_ACCUM          = 8             # gradient accumulation steps
MAX_NEW_TOKENS      = 384           # max tokens to generate
LR_SCHEDULER        = "cosine"      # cosine | constant | linear
MAX_PROMPT_LENGTH   = 256           # max prompt length

# --- Balcan et al. (2026) arXiv:2603.03538: soundness/completeness asymmetry ---
# Active when REWARD_SHAPING = "cgdb_asymmetric"
ALPHA_S             = 1.0           # soundness penalty (hard: penalise incoherent CoT)
ALPHA_C             = 0.4           # completeness penalty (soft: partial credit for near-miss)

# --- Hessian / Invisible Leash tracking ---
HESSIAN_TRACKING    = True          # track spectral norm via Lanczos HVP
HESSIAN_EVERY       = 50            # compute every N steps (keep >= 25)

# --- Power sampling baseline (Karan & Du, 2025) ---
# Run BEFORE GRPO to record training-free upper bound in results_rl.tsv
POWER_SAMPLING_BASELINE = False     # set True once; then False to skip
POWER_ALPHA         = 2.0           # sharpening exponent alpha
POWER_N_MCMC        = 30            # MH iterations
POWER_BASELINE_N    = 50            # problems to evaluate

# ============================================================
# === DO NOT MODIFY BELOW THIS LINE ===
# ============================================================

AUTORL_HOME        = os.environ.get("AUTORL_HOME", str(Path(__file__).resolve().parent))
AUTORL_DATA        = os.environ.get("AUTORL_DATA",        os.path.join(AUTORL_HOME, "data"))
AUTORL_CHECKPOINTS = os.environ.get("AUTORL_CHECKPOINTS", os.path.join(AUTORL_HOME, "checkpoints"))
BASE_MODEL         = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PATH         = os.path.join(AUTORL_DATA, "math_train.jsonl")
RESULTS_FILE     = "results_rl.tsv"
SEED             = 42


def extract_answer(text: str):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None


# ============================================================
# === CGDB REWARD (Balcan et al. 2026, arXiv:2603.03538) ===
# ============================================================

def _cot_soundness(text: str, pred: str) -> float:
    """
    Heuristic soundness probe: is the CoT internally consistent?
    Returns 1.0 (sound) to 0.0 (incoherent).
    Soundness error = rewarding wrong CoT → reward hacking.
    """
    if pred and pred in text:
        return 1.0
    return 0.5  # neutral when undetermined


def _cot_completeness(text: str) -> float:
    """
    Heuristic completeness probe: does the CoT contain reasoning steps?
    Returns 0.0–1.0. Completeness error = penalising valid CoT → kills exploration.
    """
    has_steps = bool(re.search(
        r"(step|therefore|because|since|thus|so|=>|→|=)", text, re.IGNORECASE
    ))
    has_math = bool(re.search(r"[\+\-\*\/\=\^]", text))
    return 0.5 * has_steps + 0.5 * has_math


def make_reward_fn(reward_shaping: str, base_model=None, tokenizer=None):
    """Factory for reward functions."""

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
        """Original CGDB: confidence-gated deviation bonus."""
        base_rewards = binary_reward(completions, prompts=prompts,
                                     answer=answer, **kwargs)
        rewards = []
        for completion, base_r in zip(completions, base_rewards):
            if base_r >= DEVIATION_THRESHOLD and DEVIATION_BONUS > 0:
                rewards.append(base_r + DEVIATION_BONUS * base_r)
            else:
                rewards.append(base_r)
        return rewards

    def cgdb_asymmetric(completions, prompts=None, answer=None, **kwargs):
        """
        Enhanced CGDB with soundness/completeness asymmetry.
        Ref: Balcan et al. (2026) arXiv:2603.03538

        Soundness error  (α_s): CoT incoherent despite correct answer → reward hacking
        Completeness error (α_c): CoT correct but scored wrong → kills exploration

        On correct answer: penalise low soundness (hard, weight ALPHA_S)
        On wrong answer:   partial credit for high completeness (soft, weight ALPHA_C)
        + original CGDB deviation bonus on top
        """
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            pred = extract_answer(text)
            correct = (pred is not None and
                       answer is not None and
                       str(pred).strip() == str(answer[0]).strip())

            s = _cot_soundness(text, pred or "")
            c = _cot_completeness(text)

            if correct:
                r = 1.0
                # soundness penalty: even a correct answer loses reward if CoT is incoherent
                r -= ALPHA_S * (1.0 - s)
                # CGDB deviation bonus
                if r >= DEVIATION_THRESHOLD and DEVIATION_BONUS > 0:
                    r += DEVIATION_BONUS * r
            else:
                r = 0.0
                # completeness partial credit: plausible CoT deserves soft signal
                r += ALPHA_C * c

            rewards.append(float(r))
        return rewards

    dispatch = {
        "binary":            binary_reward,
        "dense":             dense_reward,
        "confidence_gated":  confidence_gated_reward,
        "cgdb_asymmetric":   cgdb_asymmetric,
    }
    return dispatch.get(reward_shaping, binary_reward)


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


# ============================================================
# === HESSIAN / INVISIBLE LEASH TRACKING ===
# ============================================================

class HessianTracker:
    """
    Tracks top eigenvalue of the Hessian via Lanczos HVP.
    No full Hessian materialisation — memory-safe on H100.

    Connects spectral norm to escape_radius to test the
    Invisible Leash hypothesis: λ_max ↑  ↔  escape_radius ↑
    """

    def __init__(self, every_n: int = 50, n_lanczos: int = 15):
        self.every_n   = every_n
        self.n_lanczos = n_lanczos
        self.history   = {"step": [], "lambda_max": [], "escape_radius": []}

    def _hvp(self, loss, params, v_list):
        """Hessian-vector product H @ v via double backprop."""
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True
        )
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        gv = sum((g * v).sum() for g, v in zip(grads, v_list))
        hvps = torch.autograd.grad(gv, params, retain_graph=True, allow_unused=True)
        return [h.detach() if h is not None else torch.zeros_like(p)
                for h, p in zip(hvps, params)]

    def compute_lambda_max(self, loss, model) -> float:
        params = [p for p in model.parameters()
                  if p.requires_grad and p.grad_fn is not None]
        if not params:
            return float("nan")
        try:
            # random unit vector
            v = [torch.randn_like(p) for p in params]
            norm = math.sqrt(sum((vi**2).sum().item() for vi in v) + 1e-30)
            v = [vi / norm for vi in v]

            alphas, betas, v_prev = [], [], [torch.zeros_like(vi) for vi in v]
            for j in range(self.n_lanczos):
                w = self._hvp(loss, params, v)
                alpha = sum((wi * vi).sum().item() for wi, vi in zip(w, v))
                alphas.append(alpha)
                w = [wi - alpha * vi - (betas[-1] if betas else 0.0) * vp
                     for wi, vi, vp in zip(w, v, v_prev)]
                beta = math.sqrt(sum((wi**2).sum().item() for wi in w) + 1e-30)
                if beta < 1e-10:
                    break
                betas.append(beta)
                v_prev = v
                v = [wi / beta for wi in w]

            T = np.diag(alphas)
            for i, b in enumerate(betas):
                T[i, i + 1] = b
                T[i + 1, i] = b
            eigvals = np.linalg.eigvalsh(T)
            return float(eigvals.max())
        except Exception as e:
            print(f"[Hessian] error: {e}")
            return float("nan")

    def step(self, loss, model, global_step: int, escape_radius: float = float("nan")):
        if global_step % self.every_n != 0:
            return

        # Compute grad_norm to detect zero-gradient cases
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            print(f"[Hessian] step={global_step:4d}  no trainable params → λ_max=nan")
            self.history["step"].append(global_step)
            self.history["lambda_max"].append(float("nan"))
            self.history["escape_radius"].append(escape_radius)
            return

        grad_norm_sq = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None)
        grad_norm = math.sqrt(grad_norm_sq) if grad_norm_sq > 0 else 0.0

        if grad_norm < 1e-8:
            print(f"[Hessian] step={global_step:4d}  grad_norm={grad_norm:.2e} → skipping Hessian (zero gradient)")
            self.history["step"].append(global_step)
            self.history["lambda_max"].append(float("nan"))
            self.history["escape_radius"].append(escape_radius)
            return

        lam = self.compute_lambda_max(loss, model)
        self.history["step"].append(global_step)
        self.history["lambda_max"].append(lam)
        self.history["escape_radius"].append(escape_radius)
        print(f"  [Hessian] step={global_step:4d}  λ_max={lam:.4f}  ε={escape_radius:.4f}")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Hessian] history saved → {path}")

    def pearson_r(self) -> float:
        """r(λ_max, escape_radius): core empirical claim of AutoRL-LRM."""
        lam = np.array(self.history["lambda_max"])
        er  = np.array(self.history["escape_radius"])
        mask = ~(np.isnan(lam) | np.isnan(er))
        if mask.sum() < 3:
            return float("nan")
        return float(np.corrcoef(lam[mask], er[mask])[0, 1])


# ============================================================
# === POWER SAMPLING BASELINE (Karan & Du, 2025) ===
# ============================================================

def _seq_logprob(model, input_ids):
    """Sum of log-probs for all tokens in input_ids."""
    with torch.no_grad():
        out = model(input_ids=input_ids)
    lp = torch.log_softmax(out.logits, dim=-1)   # (1, T, V)
    tok_lp = lp[0, :-1, :].gather(
        1, input_ids[0, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return tok_lp.sum().item()


def power_sample_one(model, tokenizer, prompt: str, device) -> str:
    """One draw from p^alpha via block-wise Metropolis-Hastings."""
    enc = tokenizer(prompt, return_tensors="pt",
                    truncation=True, max_length=MAX_PROMPT_LENGTH).to(device)
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        cur_ids = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    cur_lp = _seq_logprob(model, cur_ids)
    block_size = 32

    for _ in range(POWER_N_MCMC):
        gen_len = cur_ids.shape[1] - prompt_len
        bs = min(block_size, max(1, gen_len))
        start = 0 if gen_len <= bs else torch.randint(0, gen_len - bs, (1,)).item()
        prefix = cur_ids[:, :prompt_len + start]
        with torch.no_grad():
            prop_ids = model.generate(
                prefix, max_new_tokens=MAX_NEW_TOKENS - start,
                do_sample=True, temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        prop_lp = _seq_logprob(model, prop_ids)
        if math.log(max(torch.rand(1).item(), 1e-12)) < POWER_ALPHA * (prop_lp - cur_lp):
            cur_ids = prop_ids
            cur_lp  = prop_lp

    return tokenizer.decode(cur_ids[0, prompt_len:], skip_special_tokens=True)


def run_power_sampling_baseline(model, tokenizer, device) -> dict:
    """
    Run power sampling on the val set and log to results_rl.tsv.
    Returns {"pass@1": float, "escape_radius": "N/A"}.
    """
    print("\n" + "=" * 60)
    print("Power Sampling Baseline (Karan & Du, 2025)")
    print(f"  alpha={POWER_ALPHA}  n_mcmc={POWER_N_MCMC}  n={POWER_BASELINE_N}")
    print("=" * 60)
    val_path = os.environ.get("AUTORL_DATA", AUTORL_HOME + "/data") + "/math_val.jsonl"
    dataset  = []
    with open(val_path) as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    dataset = dataset[:POWER_BASELINE_N]

    model.eval()
    correct = 0
    for i, item in enumerate(dataset):
        prompt = (
            f"Solve this math problem step by step.\n\n"
            f"Problem: {item['problem']}\n\nSolution:"
        )
        resp = power_sample_one(model, tokenizer, prompt, device)
        pred = extract_answer(resp)
        if pred and str(pred).strip() == str(item["answer"]).strip():
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{POWER_BASELINE_N}]  acc_so_far={correct/(i+1):.3f}")

    acc = correct / len(dataset)
    print(f"\nPower sampling pass@1 = {acc:.6f}")

    row = f"power_sampling\t{acc:.6f}\tN/A\tN/A\tbaseline\t" \
          f"training-free power sampling alpha={POWER_ALPHA} n_mcmc={POWER_N_MCMC}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(row)
    print(f"Logged to {RESULTS_FILE}")
    return {"pass@1": acc}


def train():
    print(f"lr:                    {LR}")
    print(f"kl_coeff:              {KL_COEFF}")
    print(f"reward_shaping:        {REWARD_SHAPING}")
    print(f"deviation_bonus:       {DEVIATION_BONUS}")
    print(f"alpha_s/alpha_c:       {ALPHA_S} / {ALPHA_C}")
    print(f"n_samples:             {N_SAMPLES}")
    print(f"train_steps:           {TRAIN_STEPS}")
    print(f"hessian_tracking:      {HESSIAN_TRACKING}")
    print(f"power_sampling_baseline: {POWER_SAMPLING_BASELINE}")

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- Power sampling baseline (run before any training) ---
    if POWER_SAMPLING_BASELINE:
        run_power_sampling_baseline(model, tokenizer, device)

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
        generation_batch_size=N_SAMPLES,
    )

    # Load dataset
    dataset = load_train_dataset(TRAIN_PATH)

    # Reward function
    reward_fn = make_reward_fn(REWARD_SHAPING)

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # --- Hessian tracker: patch training_step ---
    tracker = HessianTracker(every_n=HESSIAN_EVERY) if HESSIAN_TRACKING else None
    if tracker is not None:
        _orig_step = trainer.training_step

        def _patched_step(mdl, inputs, num_items_in_batch=None):
            loss = _orig_step(mdl, inputs, num_items_in_batch)
            step = trainer.state.global_step
            # escape_radius not available mid-step; pass nan (filled post-eval)
            tracker.step(loss, mdl, step, escape_radius=float("nan"))
            return loss

        trainer.training_step = _patched_step

    trainer.train()

    training_seconds = time.time() - start_time

    # --- Save Hessian history + Pearson r ---
    if tracker is not None:
        h_path = os.path.join(
            os.environ.get("AUTORL_CHECKPOINTS", AUTORL_HOME + "/checkpoints"),
            "hessian_history.json"
        )
        tracker.save(h_path)
        r = tracker.pearson_r()
        print(f"\n[Hessian] Pearson r(λ_max, ε_t) = {r:.4f}")
        print("  Interpretation: r > 0.6 supports Invisible Leash hypothesis")

    print(f"\n---")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"train_steps:      {TRAIN_STEPS}")
    print(f"reward_shaping:   {REWARD_SHAPING}")
    print(f"lr:               {LR}")
    print("Model saved to checkpoints/latest")


if __name__ == "__main__":
    train()
