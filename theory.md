# AutoRL-LRM Theory

## Invisible Leash Hypothesis
λ_max of the Hessian correlates with escape_radius of the MCMC sampler.
Higher spectral norm → wider exploration basin → better reasoning.
Empirical claim: Pearson r(λ_max, ε_t) > 0.6

## CGDB Asymmetric Reward
Soundness penalty weighted by ALPHA_S.
Completeness credit weighted by ALPHA_C.
Asymmetry justified by Balcan et al. 2026.

## Block-wise Metropolis-Hastings
From Karan & Du 2025. escape_radius computed per block.
Passed to HessianTracker for Pearson correlation.

## GRPO Training Dynamics
frac_reward_zero_std > 0.5 → groups too homogeneous → no learning.
clipped_ratio = 1.0 → max_completion_length too short.
learning_rate < 1e-10 → scheduler dead → model frozen.

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
12. **Can the agent discover and add relevant papers to the watchlist automatically
    based on scan_papers.py output and current experiment results?**
    
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

Agent decision rule relative to power sampling:

If GRPO pass@1 < power sampling pass@1 → config suboptimal; try increasing TRAIN_STEPS,
DEVIATION_BONUS, or switching to cgdb_asymmetric
If GRPO pass@1 > power sampling pass@1 AND escape_radius > 0 → genuine learning; target regime
If GRPO pass@8 < power sampling pass@8 → diversity collapse; try KL_COEFF > 0 or ALPHA_C > 0.4
Always record power sampling pass@1 in the Baseline section of program_rl.md after first run

Template: Connection to New Algorithm (added automatically by meta_update.py)
Connection to "<Algorithm Name>" Algorithm
Reference: Authors — Institution, arXiv:XXXX.XXXXX (year)
Core claim: ...
How it connects to this experiment loop:

ALGORITHM=<name> implements this algorithm in train_trl.py
Key hyperparameters: ...
Expected behaviour vs GRPO: ...

Agent decision rule:

If pass@1 up vs GRPO baseline → keep; add to ALGORITHM=auto pool
If pass@1 down → discard; note in results_rl.tsv; do not retry unless new evidence
If escape_radius changes significantly → note in description field