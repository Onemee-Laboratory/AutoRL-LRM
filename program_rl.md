# AutoRL-LRM Research Program

## Overview
You are an AI research agent investigating RLVR (Reinforcement Learning from
Verifiable Rewards) algorithms for Large Reasoning Models (LRMs).

Your task: autonomously experiment with RL training configurations to improve
mathematical reasoning, measured by **pass@1** on the MATH benchmark.

Secondary goal: maximize **escape_radius** — how far the trained model moves
from the base model's distribution (the "leash" metric).

## Baseline
```
Pass@1:                (established on first run)
pass@8:                (established on first run)
escape_radius:         (established on first run)
power_sampling_pass@1: (established after POWER_SAMPLING_BASELINE=True run)
```
Update this section after the first experiment and after the power sampling baseline run.

## Self-Evolution Protocol

When scan_papers.py finds a high-score algorithm paper (score >= 4):

### Step 1 — Clone and run
- git clone the paper's repo into $AUTORL_BASELINES/<paper_id>/
- Install deps: uv pip install -r requirements.txt
- Run their baseline on $AUTORL_DATA/math_val.jsonl
- Record their pass@1 in results_rl.tsv as commit=<paper_id>_baseline

### Step 2 — Analyse
- Read their core training loop
- Identify the key difference from current ALGORITHM in train_trl.py
- Write analysis to $AUTORL_LOG/<paper_id>_analysis.md

### Step 3 — Implement variant
- Add new training function to train_trl.py following existing signature
- Set ALGORITHM=<new_name> and run with baseline config
- Compare pass@1 and escape_radius vs GRPO baseline

### Step 4 — Evolve
- If pass@1 improves: keep, add to ALGORITHM=auto pool, sweep hyperparameters
- If pass@1 drops: analyse why, modify one component, retry (max 3 attempts)
- If 3 attempts all fail: discard, log to results_rl.tsv, move to next paper

### STRICT RULES for self-evolution
- NEVER clone into AUTORL_WORKSPACE — only AUTORL_BASELINES/
- NEVER run untrusted code as root
- ALWAYS record every attempt in results_rl.tsv
- ALWAYS compare against GRPO baseline, not just previous experiment
- Cap at 3 evolution attempts per paper before moving on

## What You Can Modify in train_trl.py
**Only modify the section marked `=== AGENT MODIFIES THESE ===`.**

| Parameter | Default | Range / Options |
|---|---|---|
| **ALGORITHM** | grpo | grpo, reinforce, rloo, dapo, dr_grpo, auto |
| LR | 1e-6 | 1e-7 to 5e-6 |
| KL_COEFF | 0.01 | 0.0 to 0.1 |
| REWARD_SHAPING | binary | binary, dense, confidence_gated, cgdb_asymmetric |
| DEVIATION_BONUS | 0.0 | 0.0 to 0.5 (CGDB coefficient) |
| DEVIATION_THRESHOLD | 0.5 | 0.1 to 0.9 |
| TEMPERATURE | 0.8 | 0.5 to 1.2 |
| N_SAMPLES | 8 | 4 to 16 |
| TRAIN_STEPS | 200 | 100 to 500 |
| GRAD_ACCUM | 4 | 1 to 8 |
| MAX_NEW_TOKENS | 512 | 256 to 1024 |
| LR_SCHEDULER | cosine | cosine, constant, linear |
| WEIGHT_DECAY | 0.01 | 0.0 to 0.1 |
| **ALPHA_S** | 1.0 | 0.5 to 2.0 (soundness penalty, cgdb_asymmetric only) |
| **ALPHA_C** | 0.4 | 0.0 to 1.0 (completeness credit, cgdb_asymmetric only) |
| **HESSIAN_TRACKING** | True | True / False |
| **HESSIAN_EVERY** | 50 | 25 to 200 |
| **POWER_SAMPLING_BASELINE** | False | True (run once only, then set False) |
| **POWER_ALPHA** | 2.0 | 1.5 to 4.0 |
| **POWER_N_MCMC** | 30 | 10 to 100 |

## Recommended Experiment Order
1. Establish baseline (default config, do not change anything)
2. **Run power sampling baseline**: set POWER_SAMPLING_BASELINE=True, run once, then set False. Log result as `power_sampling` commit in results_rl.tsv. This is the training-free upper bound (Karan & Du, 2025).
3. Compare LR values: 5e-7, 1e-6, 3e-6
4. Try KL_COEFF=0.0 vs 0.01 vs 0.05
5. Try REWARD_SHAPING=dense
6. Try REWARD_SHAPING=confidence_gated with DEVIATION_BONUS=0.1
7. Try N_SAMPLES=16
8. Try TEMPERATURE=1.0 for more exploration
9. CGDB sweep: DEVIATION_BONUS in [0.05, 0.1, 0.2, 0.5]
10. Try TRAIN_STEPS=400
11. **Try REWARD_SHAPING=cgdb_asymmetric** (Balcan et al. 2026, soundness/completeness asymmetry)
12. **ALPHA_S sweep** (cgdb_asymmetric): [0.5, 1.0, 1.5, 2.0]
13. **ALPHA_C sweep** (cgdb_asymmetric): [0.1, 0.2, 0.4, 0.8]
14. **Best config + ALPHA_S/ALPHA_C combined with DEVIATION_BONUS**
15. **Implement and run REINFORCE** — simplest policy gradient baseline
16. **Implement and run RLOO** (Leave-One-Out baseline)
17. **Implement and run DAPO** — if paper found by scan_papers.py
18. **Implement and run Dr. GRPO** — if paper found by scan_papers.py
19. **Algorithm comparison** — run best config from each algorithm; set ALGORITHM=auto
20. **New algorithms from scan_papers.py** — implement and run with baseline config first
21. **After each experiment, check if cited papers are missing from watchlist — add via watchlist_manager.py**
22. **If a new algorithm is found by scan_papers.py with score >= 4, add to news_papers.md via watchlist_manager.py**

## STRICT RULES — NEVER VIOLATE
- NEVER modify eval_rl.py
- NEVER modify prepare_data.py
- NEVER change BASE_MODEL, VAL_PATH, N_EVAL, or K_SAMPLES in eval_rl.py
- Make EXACTLY ONE change per experiment (exception: ALPHA_S and ALPHA_C may be changed together)
- Always compare pass@1 against the baseline
- If pass@1 improves OR escape_radius improves significantly: status=keep
- If both metrics are worse: status=discard, revert to previous config
- Log every experiment to results_rl.tsv with clear description
- **POWER_SAMPLING_BASELINE must be set to False after the first run**
- **HESSIAN_TRACKING=True by default** — only set False if training is too slow (>2× baseline time)
- After every run with HESSIAN_TRACKING=True, read checkpoints/hessian_history.json and note λ_max in results_rl.tsv
- **cgdb_asymmetric supersedes confidence_gated** — do not run both unless explicitly comparing
- The agent MAY implement new RL algorithms by adding new training functions below === DO NOT MODIFY BELOW THIS LINE === and registering in ALGORITHM options
- When ALGORITHM=auto, select the algorithm with best pass@1 so far
- After implementing a new algorithm, always run with baseline config first
- The agent MAY modify watchlist_manager.py to add dynamic watchlists, add papers, update scoring — output format MUST match scan_papers.py bullet block format
- NEVER clone into AUTORL_WORKSPACE — only AUTORL_BASELINES/
- When adding a new paper to the watchlist via watchlist_manager.py, always use
  the same format as scan_papers.py output (bullet blocks, not markdown tables).
  See $AUTORL_WORKSPACE/new_papers.md for the canonical format.
- The agent MAY modify watchlist_manager.py to:
  - Add new dynamic watchlists based on research needs
  - Add papers discovered during experiments to appropriate watchlists
  - Update scoring logic based on relevance to current research goals
  - Output format MUST always match scan_papers.py bullet block format
- The agent MUST NOT modify the core watchlist structure (CORE_WATCHLISTS)
  without a clear reason logged in results_rl.tsv
  
## Key Research Questions
1. Does higher KL_COEFF reduce escape_radius (stronger leash)?
2. Does CGDB (confidence_gated + deviation_bonus) increase escape_radius?
3. Is there a tradeoff between pass@1 and escape_radius?
4. What configuration maximizes both pass@1 AND escape_radius?
5. Does cgdb_asymmetric outperform confidence_gated on pass@1 and escape_radius?
6. Does λ_max (Hessian spectral norm) correlate with escape_radius across experiments?
7. What is the gap between power sampling pass@1 and the best GRPO config?
8. Does cgdb_asymmetric close the gap to power sampling better than vanilla CGDB?
9. Which algorithm (GRPO, REINFORCE, RLOO, DAPO, Dr. GRPO) achieves best pass@1?
10. Does escape_radius differ significantly across algorithms?
11. Can a newly discovered algorithm from scan_papers.py outperform all existing ones?
12. **Can the agent discover and add relevant papers to the watchlist automatically
    based on scan_papers.py output and current experiment results?**

## Template: Connection to New Algorithm (added by meta_update.py)
## Connection to "<Algorithm Name>" Algorithm
**Reference:** Authors — Institution, arXiv:XXXX.XXXXX (year)
**Core claim:** ...
**How it connects:**
- ALGORITHM=<n> implements this in train_trl.py
- Key hyperparameters: ...
- Expected behaviour vs GRPO: ...
**Agent decision rule:**
- pass@1 up vs GRPO → keep; add to ALGORITHM=auto pool
- pass@1 down → discard; note in results_rl.tsv
- escape_radius changes significantly → note in description field

## results_rl.tsv Format
Columns: commit | pass@1 | pass@8 | escape_radius | status | description
- For watchlist updates: note which papers were added and to which watchlist

Extended description field should include:
- lambda_max (from hessian_history.json, final step value)
- For cgdb_asymmetric runs: note ALPHA_S and ALPHA_C values used
- For algorithm runs: note ALGORITHM value and whether it beats GRPO baseline
- For watchlist updates: note which papers were added and to which watchlist

## Theory
See $AUTORL_THEORY (theory.md) for paper connections, hypotheses, and agent decision rules.