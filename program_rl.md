# AutoRL-LRM Agent Instructions

## Overview
You are an AI research agent investigating RLVR (Reinforcement Learning from
Verifiable Rewards) algorithms for Large Reasoning Models (LRMs).

Your task: autonomously experiment with RL training configurations to improve
mathematical reasoning, measured by **pass@1** on the MATH benchmark.

Secondary goal: maximize **escape_radius** — how far the trained model moves
from the base model's distribution (the "leash" metric).


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

## Reference
Read theory.md for theoretical background before making decisions.

## Code Discipline — Absolute Rules

### No Hardcodes — Zero Tolerance
Every path, filename, threshold, model name, and URL must come from
environment variables defined in env.sh. No exceptions.

**Forbidden patterns — agent must never write these:**
- `os.environ.get("VAR", "some/default/path")`  ← default is a hardcode
- `os.path.join(AUTORL_HOME, "watchlist")`       ← subpath is a hardcode
- `"new_papers.md"` anywhere in Python           ← filename is a hardcode
- `"http://localhost:11434"` anywhere in Python  ← URL is a hardcode
- `Path(__file__).resolve().parent`              ← fallback is a hardcode

**Required pattern:**
```python
def _require(var: str) -> str:
    val = os.environ.get(var, "").strip()
    if not val:
        print(f"ERROR: {var} not set. Run 'source env.sh'.")
        sys.exit(1)
    return val
```

Every variable must use `_require()`. If a variable does not exist in
env.sh yet, add it to env.sh first, then read it with `_require()`.

### Self-Check Before Committing
Before writing any Python file, agent must grep its own output:
```bash
grep -n '["'"'"'][./a-zA-Z_]*\.md["'"'"']\|["'"'"']http\|os.environ.get.*,.*["'"'"']' <file.py>
```
If this grep returns any matches, fix them before committing.

### Violation Response
If agent detects a hardcode in existing code during any task,
it must fix the hardcode in that same commit before proceeding.


## Diagnostic Rules

After every training run, check the log for these signals:

| Signal | Threshold | Action |
|---|---|---|
| clipped_ratio | = 1.0 for 3+ steps | double MAX_NEW_TOKENS |
| frac_reward_zero_std | > 0.5 | increase N_SAMPLES or diversify prompts |
| learning_rate | < 1e-10 | switch LR_SCHEDULER to constant_with_warmup |
| λ_max | = nan | grep codebase for HessianTracker call site; fix params filter |
| Pearson r | = nan | escape_radius not passed to hessian_tracker.step(); fix call site |
| reward | flat for 5+ steps | change one parameter per experiment queue order |

## What You Can Modify in train_trl.py

Only modify the section marked `=== AGENT MODIFIES THESE ===`.

| Parameter | Default | Range |
|---|---|---|
| ALGORITHM | grpo | grpo, reinforce, rloo, dapo, dr_grpo, auto |
| LR | 1e-6 | 1e-7 to 5e-6 |
| KL_COEFF | 0.01 | 0.0 to 0.1 |
| REWARD_SHAPING | binary | binary, dense, confidence_gated, cgdb_asymmetric |
| DEVIATION_BONUS | 0.0 | 0.0 to 0.5 |
| DEVIATION_THRESHOLD | 0.5 | 0.1 to 0.9 |
| TEMPERATURE | 0.8 | 0.5 to 1.2 |
| N_SAMPLES | 8 | 4 to 16 |
| TRAIN_STEPS | 200 | 100 to 500 |
| GRAD_ACCUM | 4 | 1 to 8 |
| MAX_NEW_TOKENS | 512 | 256 to 1024 |
| LR_SCHEDULER | cosine | cosine, constant, constant_with_warmup, linear |
| WEIGHT_DECAY | 0.01 | 0.0 to 0.1 |
| ALPHA_S | 1.0 | 0.5 to 2.0 |
| ALPHA_C | 0.4 | 0.0 to 1.0 |
| HESSIAN_TRACKING | True | True / False |
| HESSIAN_EVERY | 50 | 25 to 200 |
| POWER_SAMPLING_BASELINE | False | True once only, then False |
| POWER_ALPHA | 2.0 | 1.5 to 4.0 |
| POWER_N_MCMC | 30 | 10 to 100 |

## Experiment Queue

Make exactly one change per experiment.
ALPHA_S and ALPHA_C may be changed together — they are a pair.

- [ ] 1. Establish baseline — default config, no changes
- [ ] 2. Power sampling baseline — POWER_SAMPLING_BASELINE=True, run once, set False
- [ ] 3. LR sweep — 5e-7, 1e-6, 3e-6
- [ ] 4. KL_COEFF sweep — 0.0, 0.01, 0.05
- [ ] 5. REWARD_SHAPING=dense
- [ ] 6. REWARD_SHAPING=confidence_gated, DEVIATION_BONUS=0.1
- [ ] 7. N_SAMPLES=16
- [ ] 8. TEMPERATURE=1.0
- [ ] 9. DEVIATION_BONUS sweep — 0.05, 0.1, 0.2, 0.5
- [ ] 10. TRAIN_STEPS=400
- [ ] 11. REWARD_SHAPING=cgdb_asymmetric
- [ ] 12. ALPHA_S sweep — 0.5, 1.0, 1.5, 2.0
- [ ] 13. ALPHA_C sweep — 0.1, 0.2, 0.4, 0.8
- [ ] 14. Best config + combined ALPHA_S/ALPHA_C + DEVIATION_BONUS
- [ ] 15. ALGORITHM=reinforce — baseline config first
- [ ] 16. ALGORITHM=rloo — baseline config first
- [ ] 17. ALGORITHM=dapo — if found by scan_papers.py
- [ ] 18. ALGORITHM=dr_grpo — if found by scan_papers.py
- [ ] 19. Algorithm comparison — best config each; set ALGORITHM=auto on winner
- [ ] 20. New algorithms from scan_papers.py score >= 4 — baseline config first

After each experiment:
- Record in results_rl.tsv
- Check hessian_history.json for λ_max trend
- Note Pearson r if available
- Check if any cited papers are missing from watchlist


## Strict Rules

- NEVER modify eval_rl.py or prepare_data.py
- NEVER change BASE_MODEL, VAL_PATH, N_EVAL, K_SAMPLES in eval_rl.py
- NEVER clone repos into AUTORL_WORKSPACE — only AUTORL_BASELINES
- NEVER run untrusted code as root
- ALWAYS compare pass@1 against the GRPO baseline, not just previous experiment
- ALWAYS record every attempt in results_rl.tsv
- POWER_SAMPLING_BASELINE → set False immediately after first run
- HESSIAN_TRACKING=True by default — set False only if training exceeds 2× baseline time
- cgdb_asymmetric supersedes confidence_gated — do not run both unless explicitly comparing
- New RL algorithms go below `=== DO NOT MODIFY BELOW THIS LINE ===`
  following existing function signature
- When ALGORITHM=auto → select algorithm with best pass@1 so far
- New algorithm → always run baseline config first before tuning

## Baseline
pass@1:                (fill after first run)
pass@8:                (fill after first run)
escape_radius:         (fill after first run)
power_sampling_pass@1: (fill after power sampling run)


## results_rl.tsv Format

Columns: commit | pass@1 | pass@8 | escape_radius | status | description

Description field must include:
- λ_max final value from hessian_history.json
- Pearson r if computed
- ALPHA_S and ALPHA_C for cgdb_asymmetric runs
- ALGORITHM value for algorithm runs
- Papers added to watchlist if any

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
12. **ALPHA_S sweep** (cgdb_asymmetric): [0.5, 1.0, 1.5, 2.0] — controls soundness strictness
13. **ALPHA_C sweep** (cgdb_asymmetric): [0.1, 0.2, 0.4, 0.8] — controls completeness leniency
14. **Best config + ALPHA_S/ALPHA_C combined with DEVIATION_BONUS** — maximise both pass@1 and escape_radius
15. **Implement and run REINFORCE** — simplest policy gradient baseline; compare pass@1 and escape_radius vs GRPO
16. **Implement and run RLOO** (Leave-One-Out baseline) — lower variance than REINFORCE; compare vs GRPO
17. **Implement and run DAPO** — if paper found by scan_papers.py; follow paper's recommended hyperparameters for first run
18. **Implement and run Dr. GRPO** — if paper found by scan_papers.py; follow paper's recommended hyperparameters for first run
19. **Algorithm comparison** — run best config from each algorithm; pick winner by pass@1; set ALGORITHM=auto going forward
20. **New algorithms from scan_papers.py** — if meta_update.py adds a new algorithm to this list, implement and run it with baseline config first
21. **After each experiment, check if any cited papers in results_rl.tsv
    description are missing from watchlist — add them via watchlist_manager.py**
22. **If a new algorithm is found by scan_papers.py with score >= 4,
    automatically add it to news_papers.md via watchlist_manager.py**

## STRICT RULES — NEVER VIOLATE
- NEVER modify eval_rl.py
- NEVER modify prepare_data.py
- NEVER change BASE_MODEL, VAL_PATH, N_EVAL, or K_SAMPLES in eval_rl.py
- Make EXACTLY ONE change per experiment (exception: ALPHA_S and ALPHA_C may be changed together since they are a pair)
- Always compare pass@1 against the baseline
- If pass@1 improves OR escape_radius improves significantly: status=keep
- If both metrics are worse: status=discard, revert to previous config
- Log every experiment to results_rl.tsv with clear description
- **POWER_SAMPLING_BASELINE must be set to False after the first run** — it is a one-shot baseline, not a training mode
- **HESSIAN_TRACKING=True by default** — only set False if training is too slow (>2× baseline time)
- After every run with HESSIAN_TRACKING=True, read `checkpoints/hessian_history.json` and note λ_max trend in results_rl.tsv description field
- **cgdb_asymmetric supersedes confidence_gated** — do not run both in the same sweep unless explicitly comparing them
- The agent MAY implement new RL algorithms (REINFORCE, RLOO, DAPO, Dr. GRPO, or any discovered via scan_papers.py) by adding new training functions anywhere below `=== DO NOT MODIFY BELOW THIS LINE ===` and registering the algorithm name in the ALGORITHM parameter options. New functions must follow the same signature as the existing GRPO training block.
- When ALGORITHM=auto, the agent selects the algorithm with the best pass@1 so far and runs the next experiment with that algorithm.
- After implementing a new algorithm, always run it with the baseline config first (default LR, KL_COEFF, REWARD_SHAPING) before tuning its hyperparameters.
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
  
## Template: Connection to New Algorithm (added by meta_update.py)
When scan_papers.py finds a new RL algorithm paper, meta_update.py appends
a section using this template:

## Connection to "<Algorithm Name>" Algorithm
**Reference:** Authors — Institution, arXiv:XXXX.XXXXX (year)

**Core claim:** ...

**How it connects to this experiment loop:**
- `ALGORITHM=<name>` implements this algorithm in train_trl.py
- Key hyperparameters: ...
- Expected behaviour vs GRPO: ...

**Agent decision rule:**
- If pass@1 ↑ vs GRPO baseline → keep; add to ALGORITHM=auto pool
- If pass@1 ↓ → discard; note in results_rl.tsv; do not retry unless new evidence
- If escape_radius changes significantly → note in description field

