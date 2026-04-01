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

## What You Can Modify in train_trl.py
**Only modify the section marked `=== AGENT MODIFIES THESE ===`.**

| Parameter | Default | Range / Options |
|---|---|---|
| **ALGORITHM** | grpo | grpo, reinforce, rloo, dapo, dr_grpo, auto |
| LR | 1e-6 | 1e-7 to 5e-6 |
| KL_COEFF | 0.01 | 0.0 to 0.1 |
| REWARD_SHAPING | binary | binary, dense, confidence_gated, **cgdb_asymmetric** |
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
12. **ALPHA_S sweep** (cgdb_asymmetric): [0.5, 1.0, 1.5, 2.0] — controls soundness strictness
13. **ALPHA_C sweep** (cgdb_asymmetric): [0.1, 0.2, 0.4, 0.8] — controls completeness leniency
14. **Best config + ALPHA_S/ALPHA_C combined with DEVIATION_BONUS** — maximise both pass@1 and escape_radius
15. **Implement and run REINFORCE** — simplest policy gradient baseline; compare pass@1 and escape_radius vs GRPO
16. **Implement and run RLOO** (Leave-One-Out baseline) — lower variance than REINFORCE; compare vs GRPO
17. **Implement and run DAPO** — if paper found by scan_papers.py; follow paper's recommended hyperparameters for first run
18. **Implement and run Dr. GRPO** — if paper found by scan_papers.py; follow paper's recommended hyperparameters for first run
19. **Algorithm comparison** — run best config from each algorithm; pick winner by pass@1; set ALGORITHM=auto going forward
20. **New algorithms from scan_papers.py** — if meta_update.py adds a new algorithm to this list, implement and run it with baseline config first

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

## Key Research Questions
1. Does higher KL_COEFF reduce escape_radius (stronger leash)?
2. Does CGDB (confidence_gated + deviation_bonus) increase escape_radius?
3. Is there a tradeoff between pass@1 and escape_radius?
4. What configuration maximizes both pass@1 AND escape_radius?
5. **Does cgdb_asymmetric outperform confidence_gated on pass@1 and escape_radius?** (Balcan et al. 2026)
6. **Does λ_max (Hessian spectral norm) correlate with escape_radius across experiments?** (Invisible Leash hypothesis — Pearson r printed after each run)
7. **What is the gap between power sampling pass@1 and the best GRPO config?** (Karan & Du 2025 — training-free upper bound)
8. **Does cgdb_asymmetric close the gap to power sampling better than vanilla CGDB?**
9. **Which algorithm (GRPO, REINFORCE, RLOO, DAPO, Dr. GRPO) achieves the best pass@1 under the same compute budget?**
10. **Does escape_radius differ significantly across algorithms — does the choice of algorithm affect leash strength?**
11. **Can a newly discovered algorithm from scan_papers.py outperform all existing ones?**

## Connection to "Invisible Leash" Paper
- escape_radius directly measures RLVR leash strength
- CGDB reward shaping is designed to extend the leash
- **cgdb_asymmetric (soundness/completeness) provides theoretical grounding via Balcan et al. 2026**
- **Hessian spectral norm λ_max is tracked per run; Pearson r(λ_max, ε_t) is the paper's key empirical claim**
- **Power sampling pass@1 is the training-free ceiling; GRPO should exceed it to justify training cost**
- Results from this loop are the experimental section of the paper

## Connection to "CoT Verifier Learnability" Paper
**Reference:** Balcan, Blum, Fragkia, Li, Sharma — CMU + TTIC, arXiv:2603.03538 (2026)

**Core claim:** Online learnability of CoT verifiers is governed by a Littlestone dimension bound,
and the two failure modes — soundness errors (rewarding wrong CoT) and completeness errors
(penalising correct CoT) — have asymmetric costs that must be reflected in the reward signal.

**How it connects to this experiment loop:**
- `REWARD_SHAPING=cgdb_asymmetric` directly implements the soundness/completeness asymmetry
- `ALPHA_S` controls the soundness penalty weight (hard): higher → stricter against reward hacking
- `ALPHA_C` controls the completeness credit weight (soft): higher → more tolerant of unusual-but-correct CoT
- **Hypothesis to test:** cgdb_asymmetric with ALPHA_S > ALPHA_C should improve pass@1 over
  confidence_gated by reducing reward hacking while preserving exploration
- **Leash connection:** soundness errors are the mechanism by which the policy escapes the leash
  (the model learns to produce incoherent but reward-triggering CoT); ALPHA_S directly suppresses this
- **What to watch:** if escape_radius increases with cgdb_asymmetric but pass@1 also increases,
  this suggests the leash is being extended in a *useful* direction (genuine capability), not reward hacking

**Agent decision rule for cgdb_asymmetric experiments:**
- If pass@1 ↑ and escape_radius ↑ → soundness penalty is working; try increasing ALPHA_S further
- If pass@1 ↑ but escape_radius ↓ → model is being constrained; try reducing ALPHA_S or increasing ALPHA_C
- If pass@1 ↓ and escape_radius ↑ → completeness errors dominating; reduce ALPHA_C

## Connection to "Power Sampling" Paper
**Reference:** Karan & Du — Harvard, arXiv:2510.14901 (2025)

**Core claim:** Base models already contain latent reasoning capability. Sampling from the sharpened
distribution p^α via block-wise Metropolis-Hastings (power sampling) achieves pass@1 on par with
GRPO — without any training. Crucially, power sampling preserves pass@k diversity that GRPO destroys.

**How it connects to this experiment loop:**
- `POWER_SAMPLING_BASELINE=True` runs power sampling on the val set and logs pass@1 to results_rl.tsv
  before any GRPO training; this is the **training-free ceiling**
- **Primary benchmark question:** does our best GRPO config exceed power sampling pass@1?
  If not, GRPO is not adding value beyond distribution sharpening
- **Secondary question:** does our pass@8 exceed power sampling pass@8?
  Power sampling explicitly preserves diversity; GRPO (especially with KL_COEFF=0) collapses it.
  cgdb_asymmetric's completeness credit (ALPHA_C) is designed to mitigate this collapse.
- **Escape radius interpretation:** power sampling does not move the base model weights, so its
  escape_radius = 0 by definition. Any positive escape_radius in our GRPO runs represents
  *additional* movement beyond what pure distribution sharpening achieves. This is what the
  Invisible Leash paper calls novel learned behavior.

**Agent decision rule relative to power sampling:**
- If GRPO pass@1 < power sampling pass@1 → current config is suboptimal; the training is not
  justified; try increasing TRAIN_STEPS, DEVIATION_BONUS, or switching to cgdb_asymmetric
- If GRPO pass@1 > power sampling pass@1 AND escape_radius > 0 → genuine learning beyond
  base model capability; this is the target regime
- If GRPO pass@8 < power sampling pass@8 → diversity collapse; try KL_COEFF > 0 or ALPHA_C > 0.4
- Always record power sampling pass@1 in the Baseline section at the top of this file after first run

## results_rl.tsv Format
Columns: commit | pass@1 | pass@8 | escape_radius | status | description

Extended description field should include:
- λ_max (from hessian_history.json, final step value)
- Whether Pearson r(λ_max, ε_t) was computed (note: ε_t filled post-eval only)
- For cgdb_asymmetric runs: note ALPHA_S and ALPHA_C values used
- For algorithm runs: note ALGORITHM value and whether it beats GRPO baseline

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