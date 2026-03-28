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
pass@1:         (established on first run)
pass@8:         (established on first run)
escape_radius:  (established on first run)
```
Update this section after the first experiment.

## What You Can Modify in train_rl.py
**Only modify the section marked `=== AGENT MODIFIES THESE ===`.**

| Parameter | Default | Range / Options |
|---|---|---|
| LR | 1e-6 | 1e-7 to 5e-6 |
| KL_COEFF | 0.01 | 0.0 to 0.1 |
| REWARD_SHAPING | binary | binary, dense, confidence_gated |
| DEVIATION_BONUS | 0.0 | 0.0 to 0.5 (CGDB coefficient) |
| DEVIATION_THRESHOLD | 0.5 | 0.1 to 0.9 |
| TEMPERATURE | 0.8 | 0.5 to 1.2 |
| N_SAMPLES | 8 | 4 to 16 |
| TRAIN_STEPS | 200 | 100 to 500 |
| GRAD_ACCUM | 4 | 1 to 8 |
| MAX_NEW_TOKENS | 512 | 256 to 1024 |
| LR_SCHEDULER | cosine | cosine, constant, linear |
| WEIGHT_DECAY | 0.01 | 0.0 to 0.1 |

## Recommended Experiment Order
1. Establish baseline (default config, do not change anything)
2. Compare LR values: 5e-7, 1e-6, 3e-6
3. Try KL_COEFF=0.0 vs 0.01 vs 0.05
4. Try REWARD_SHAPING=dense
5. Try REWARD_SHAPING=confidence_gated with DEVIATION_BONUS=0.1
6. Try N_SAMPLES=16
7. Try TEMPERATURE=1.0 for more exploration
8. CGDB sweep: DEVIATION_BONUS in [0.05, 0.1, 0.2, 0.5]
9. Try TRAIN_STEPS=400

## STRICT RULES — NEVER VIOLATE
- NEVER modify eval_rl.py
- NEVER modify prepare_data.py
- NEVER change BASE_MODEL, VAL_PATH, N_EVAL, or K_SAMPLES in eval_rl.py
- Make EXACTLY ONE change per experiment
- Always compare pass@1 against the baseline
- If pass@1 improves OR escape_radius improves significantly: status=keep
- If both metrics are worse: status=discard, revert to previous config
- Log every experiment to results_rl.tsv with clear description

## Key Research Questions
1. Does higher KL_COEFF reduce escape_radius (stronger leash)?
2. Does CGDB (confidence_gated + deviation_bonus) increase escape_radius?
3. Is there a tradeoff between pass@1 and escape_radius?
4. What configuration maximizes both pass@1 AND escape_radius?

## Connection to "Invisible Leash" Paper
- escape_radius directly measures RLVR leash strength
- CGDB reward shaping is designed to extend the leash
- This framework empirically validates the leash hypothesis
- Results from this loop are the experimental section of the paper

## results_rl.tsv Format
Columns: commit | pass@1 | pass@8 | escape_radius | status | description
