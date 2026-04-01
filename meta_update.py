"""
meta_update.py — Meta-agent for AutoRL-LRM self-improvement loop
=================================================================
Reads new_papers.md + results_rl.tsv, then uses the Claude API to:
  1. Identify actionable new ideas not yet in program_rl.md
  2. Propose new parameters or reward functions for train_trl.py
  3. Update program_rl.md with new experiment steps and paper connections
  4. Optionally patch train_trl.py with new reward function stubs

Designed to run after scan_papers.py, before the next experiment loop.

Usage:
    python meta_update.py                        # dry run (print only)
    python meta_update.py --apply                # write changes to disk
    python meta_update.py --apply --patch-train  # also patch train_trl.py
"""

import os
import json
import argparse
import datetime
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to AUTORL_HOME, no hardcodes)
# ---------------------------------------------------------------------------

AUTORL_HOME        = os.environ.get("AUTORL_HOME", str(Path(__file__).resolve().parent))
NEW_PAPERS_PATH    = os.path.join(AUTORL_HOME, "new_papers.md")
PROGRAM_RL_PATH    = os.path.join(AUTORL_HOME, "program_rl.md")
TRAIN_TRL_PATH     = os.path.join(AUTORL_HOME, "train_trl.py")
RESULTS_TSV_PATH   = os.path.join(AUTORL_HOME, "results_rl.tsv")
META_LOG_PATH      = os.path.join(AUTORL_HOME, "meta_update_log.md")

# Ollama model — override via env var AUTORL_META_MODEL
# e.g. export AUTORL_META_MODEL=qwen2.5-coder:32b
MODEL       = os.environ.get("AUTORL_META_MODEL", "qwen3-coder-next")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST",       "http://localhost:11434")
MAX_TOKENS  = 4096


# ---------------------------------------------------------------------------
# Ollama API call (no SDK, pure urllib, no extra deps)
# Uses /api/chat with system + user messages (same as OpenAI chat format).
# ---------------------------------------------------------------------------

def call_claude(system: str, user: str) -> str:
    """Name kept as call_claude for minimal diff; calls Ollama internally."""
    payload = json.dumps({
        "model":   MODEL,
        "stream":  False,
        "options": {"num_predict": MAX_TOKENS, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_HOST}. "
            f"Is \'ollama serve\' running? Error: {e}"
        )

    # Ollama /api/chat response: {"message": {"role": "assistant", "content": "..."}}
    return data["message"]["content"]


# ---------------------------------------------------------------------------
# Load context files
# ---------------------------------------------------------------------------

def load_file(path: str, max_chars: int = 8000) -> str:
    if not os.path.exists(path):
        return f"[file not found: {path}]"
    with open(path) as f:
        content = f.read()
    if len(content) > max_chars:
        # keep tail (most recent entries most relevant)
        content = "...[truncated]...\n" + content[-max_chars:]
    return content


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROGRAM_UPDATE = """
You are a research assistant for the AutoRL-LRM project.
Your job is to update program_rl.md with new actionable experiments
derived from recently scanned papers.

STRICT RULES:
- Only add experiments that are directly implementable by modifying
  the === AGENT MODIFIES THESE === section of train_trl.py, OR by
  adding a new training function or reward function stub to that file.
- Do NOT suggest architectural changes, new models, or new datasets.
- Do NOT remove or overwrite existing content — only append or extend.
- Each new experiment step must follow the existing format.
- For reward/training trick papers, follow this connection section format:
  ## Connection to "<Paper Title>" Paper
  **Reference:** ...
  **Core claim:** ...
  **How it connects:** ...
  **Agent decision rule:** ...
- For algorithm papers (new RL algorithms), follow this format instead:
  ## Connection to "<Algorithm Name>" Algorithm
  **Reference:** ...
  **Core claim:** ...
  **How it connects to this experiment loop:**
  - `ALGORITHM=<name>` implements this algorithm in train_trl.py
  - Key hyperparameters: ...
  - Expected behaviour vs GRPO: ...
  **Agent decision rule:**
  - If pass@1 ↑ vs GRPO baseline → keep; add to ALGORITHM=auto pool
  - If pass@1 ↓ → discard; note in results_rl.tsv; do not retry unless new evidence
  - If escape_radius changes significantly → note in description field
- Output ONLY the new content to append (no preamble, no markdown fence).
  Start directly with the new section headers.
- If there is nothing genuinely new to add, output exactly: NO_UPDATE
"""

SYSTEM_TRAIN_PATCH = """
You are a Python expert helping extend train_trl.py for the AutoRL-LRM project.

Your job: given a new paper idea, either:
  A) Write a new reward function stub to add inside make_reward_fn(), OR
  B) Write a new training function stub for a new RL algorithm

STRICT RULES:
- Reward functions must follow the exact signature:
  def new_reward_name(completions, prompts=None, answer=None, **kwargs) -> list[float]
- Algorithm training functions must follow this signature:
  def train_<algorithm_name>(model, tokenizer, dataset, config, reward_fn) -> None
- New parameters must use ALL_CAPS names and include a comment with range.
- Output ONLY valid Python — the function body + the parameter lines.
  No prose, no markdown fences.
- If the idea is not implementable, output: NO_PATCH
"""


def build_program_update_prompt(new_papers: str, program_rl: str, results: str) -> str:
    return f"""
## Current program_rl.md
{program_rl}

## Recent results (results_rl.tsv)
{results}

## Newly scanned papers (new_papers.md)
{new_papers}

---
Task: Identify papers in new_papers.md that propose reward shaping strategies,
new RL algorithms, training tricks, or theoretical insights NOT yet reflected
in program_rl.md.

For each such paper:
1. Add new experiment steps to "Recommended Experiment Order"
2. Add a new "Connection to <Paper/Algorithm Name>" section
   - Use "Paper" suffix for reward/theory papers
   - Use "Algorithm" suffix for new RL algorithm papers
3. If the paper suggests concrete hyperparameters, add them to the parameter table
4. If the paper proposes a new RL algorithm, add it to the ALGORITHM parameter options

Focus on papers tagged: algorithm, reward_shaping, kl_regularisation, diversity, curriculum, verifier.
Ignore papers tagged: general (unless score >= 4).

Output only the new content to append to program_rl.md, or NO_UPDATE.
"""


def build_train_patch_prompt(paper_summary: str, train_trl: str) -> str:
    return f"""
## Current make_reward_fn() in train_trl.py
{train_trl}

## New paper idea
{paper_summary}

Task: If this paper proposes a concrete reward function not yet in train_trl.py,
write the Python implementation stub and parameter additions.
Output only valid Python or NO_PATCH.
"""


# ---------------------------------------------------------------------------
# Apply updates
# ---------------------------------------------------------------------------

def append_to_program_rl(new_content: str):
    with open(PROGRAM_RL_PATH, "a") as f:
        f.write(f"\n\n<!-- meta_update: {datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M UTC')} -->\n")
        f.write(new_content)
    print(f"[meta_update] Appended {len(new_content)} chars to program_rl.md")


def patch_train_trl(new_code: str):
    """Insert new reward function into make_reward_fn dispatch table."""
    with open(TRAIN_TRL_PATH) as f:
        src = f.read()

    # Find the dispatch dict and insert before it
    dispatch_marker = '    dispatch = {'
    if dispatch_marker not in src:
        print("[meta_update] Could not find dispatch dict — skipping train_trl.py patch")
        return

    new_src = src.replace(dispatch_marker, new_code + "\n\n    " + dispatch_marker.strip())
    with open(TRAIN_TRL_PATH, "w") as f:
        f.write(new_src)
    print(f"[meta_update] Patched train_trl.py with new reward function stub")


def log_update(content: str, applied: bool):
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M UTC")
    status = "APPLIED" if applied else "DRY_RUN"
    with open(META_LOG_PATH, "a") as f:
        f.write(f"\n\n---\n## {timestamp} [{status}]\n\n{content}\n")


# ---------------------------------------------------------------------------
# Extract high-value papers for train patch
# ---------------------------------------------------------------------------

def extract_high_value_papers(new_papers: str, min_score: int = 3) -> list[str]:
    """
    Return paragraph blocks from new_papers.md tagged as
    reward_shaping or algorithm with relevance score >= min_score.
    These are candidates for train_trl.py patching.
    """
    import re
    blocks = []
    current_block = []
    in_target_section = False

    target_categories = {"reward_shaping", "algorithm"}

    for line in new_papers.split("\n"):
        if any(f"### Category: {cat}" in line for cat in target_categories):
            in_target_section = True
        elif line.startswith("### Category:"):
            if in_target_section and current_block:
                blocks.append("\n".join(current_block))
            in_target_section = False
            current_block = []
        elif in_target_section:
            if line.startswith("### ") and current_block:
                blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)

    if in_target_section and current_block:
        blocks.append("\n".join(current_block))

    high_score = []
    for block in blocks:
        m = re.search(r"Relevance score:\*\*\s*(\d+)", block)
        if m and int(m.group(1)) >= min_score:
            high_score.append(block)

    return high_score[:3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(apply: bool = False, patch_train: bool = False):
    print(f"[meta_update] Loading context files ...")
    new_papers = load_file(NEW_PAPERS_PATH, max_chars=10000)
    program_rl = load_file(PROGRAM_RL_PATH, max_chars=8000)
    results    = load_file(RESULTS_TSV_PATH, max_chars=3000)
    train_trl  = load_file(TRAIN_TRL_PATH,  max_chars=5000)

    if "[file not found" in new_papers:
        print("[meta_update] new_papers.md not found — run scan_papers.py first")
        return

    # --- Step 1: Update program_rl.md ---
    print("\n[meta_update] Querying ollama for program_rl.md updates ...")
    prompt = build_program_update_prompt(new_papers, program_rl, results)
    try:
        update_content = call_claude(SYSTEM_PROGRAM_UPDATE, prompt)
    except Exception as e:
        print(f"[meta_update] ollama API error: {e}")
        return

    if update_content.strip() == "NO_UPDATE":
        print("[meta_update] No new content to add to program_rl.md")
        log_update("NO_UPDATE", applied=False)
    else:
        print("\n--- Proposed program_rl.md additions ---")
        print(update_content[:1500])
        if len(update_content) > 1500:
            print(f"... [{len(update_content)-1500} more chars]")
        print("---\n")

        log_update(update_content, applied=apply)

        if apply:
            append_to_program_rl(update_content)
        else:
            print("[meta_update] Dry run — use --apply to write changes")

    # --- Step 2: Patch train_trl.py with new reward functions ---
    if patch_train:
        high_value = extract_high_value_papers(new_papers)
        if not high_value:
            print("[meta_update] No high-relevance reward_shaping or algorithm papers found for train patch")
        else:
            for i, paper_block in enumerate(high_value):
                print(f"\n[meta_update] Querying ollama for train_trl.py patch ({i+1}/{len(high_value)}) ...")
                patch_prompt = build_train_patch_prompt(paper_block, train_trl)
                try:
                    patch_code = call_claude(SYSTEM_TRAIN_PATCH, patch_prompt)
                except Exception as e:
                    print(f"[meta_update] ollama API error on patch {i+1}: {e}")
                    continue

                if patch_code.strip() == "NO_PATCH":
                    print(f"  → NO_PATCH for paper {i+1}")
                else:
                    print(f"\n--- Proposed train_trl.py patch ---")
                    print(patch_code[:800])
                    print("---")
                    if apply:
                        patch_train_trl(patch_code)
                    else:
                        print("[meta_update] Dry run — use --apply --patch-train to write")

    print("\n[meta_update] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply",       action="store_true", help="Write changes to disk")
    parser.add_argument("--patch-train", action="store_true", help="Also patch train_trl.py")
    args = parser.parse_args()
    run(apply=args.apply, patch_train=args.patch_train)
