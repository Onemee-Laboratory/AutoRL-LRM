# AutoRL-LRM

Automated RL Algorithm Search for Large Reasoning Models.

An autonomous research loop that discovers, implements, and evaluates RL algorithms
to improve mathematical reasoning, guided by two metrics: **pass@1** and
**escape_radius** (the "Invisible Leash"). The loop self-improves by scanning
new papers nightly and patching its own instructions and code.

---

## Architecture

```
                     +--------------------+
                     |  autoloop_meta.sh  |
                     |  (Meta loop)       |
                     |                    |
                     | scan_papers.py     |
                     | arxiv / S2 ──────→ new_papers.md
                     | meta_update.py     |
                     | Ollama/Qwen ──────→ program_rl.md (new experiments)
                     |                 ──→ train_trl.py  (new algorithms)
                     +--------------------+
                                │
                                │ Updates
                                ▼
                     +--------------------+
                     |  program_rl.md     | ◄───┐
                     |  train_trl.py      |     │
                     +--------------------+     │
                                │               │
                                │ Reads & uses  │
                     +--------------------+     │
                     |  autoloop_rl.sh    |     │
                     |  (RL execution)    |     │
                     +--------------------+     │
                                │               │
        ┌───────────────────────┼───────────────┼──────────────┐
        │                       │               │              │
        ▼                       ▼               ▼              ▼
+---------------+     +----------------+ +-------------+ +------------+
| Train Model   |     | Evaluate Model | | Extract     | | Clean logs |
| train_trl.py  |     | eval_rl.py     | | pass@1      | | prepare    |
| ALGORITHM=    |     |                | | pass@8      | | context    |
| grpo/reinforce|     |                | | escape_radius| |            |
| rloo/dapo/... |     |                | | λ_max        | |            |
+---------------+     +----------------+ +-------------+ +------------+
        │                       │               │              │
        └───────────────────────┴───────────────┴──────────────┘
                                │
                                ▼
                     +--------------------+
                     | Ollama/Qwen        |
                     | Propose next edit  |
                     | for train_trl.py   |
                     +--------------------+
                                │
                                ▼
                     +--------------------+
                     | Commit changes     |
                     | train_trl.py       |
                     | results_rl.tsv     |
                     +--------------------+
                                │
                                ▼
                     +--------------------+
                     | Next RL iteration  |
                     +--------------------+
```

**Two loops:**
- **Inner loop** (`autoloop_rl.sh`) — train → evaluate → extract metrics → Ollama proposes next edit → commit → repeat
- **Outer loop** (`autoloop_meta.sh`) — scan arxiv/S2 → Ollama reads papers → patches `program_rl.md` with new experiments and `train_trl.py` with new algorithm stubs

---

## Directory Structure

```
AUTORL_WORKSPACE/               # code — this repo
├── env.sh                      # environment setup (source first)
├── program_rl.md               # agent instructions + experiment order
├── train_trl.py                # training (agent modifies this)
├── eval_rl.py                  # evaluation harness (do not modify)
├── prepare_data.py             # dataset download (run once)
├── autoloop_rl.sh              # inner experiment loop
├── autoloop_meta.sh            # outer literature + meta-update loop
├── scan_papers.py              # arxiv + Semantic Scholar scanner
├── meta_update.py              # Ollama-based program_rl.md updater
├── results_rl.tsv              # experiment results log
├── new_papers.md               # scanned papers output
└── pyproject.toml

AUTORL_HOME/                    # runtime — set by user on first run
├── data/                       # math_train.jsonl, math_val.jsonl
├── checkpoints/                # model checkpoints
├── log/                        # training and meta-update logs
└── venv/                       # Python virtual environment
    └── .venv/
```

---

## Setup

**1. Clone and configure environment:**

```bash
git clone <repo>
cd autorl-lrm
source env.sh
# Enter AUTORL_HOME when prompted (e.g. /home/oz/.autorl)
```

**2. Create runtime directories:**

```bash
mkdir -p $AUTORL_HOME/data $AUTORL_HOME/checkpoints $AUTORL_HOME/log
```

**3. Create virtual environment:**

```bash
uv sync
```

**4. Download dataset (run once):**

```bash
python prepare_data.py
```

---

## Usage

### Inner loop — GRPO experiments

```bash
bash autoloop_rl.sh
```

The agent reads `program_rl.md`, modifies `train_trl.py`, trains, evaluates,
and logs results to `results_rl.tsv` automatically.

### Outer loop — literature scan + self-update

```bash
# Option A: scan only (no writes, no LLM call)
bash autoloop_meta.sh --scan-only

# Option B: full meta-update (Ollama + program_rl.md + train_trl.py update)
bash autoloop_meta.sh --apply

# Option B + patch train_trl.py with new algorithm/reward stubs
bash autoloop_meta.sh --apply --patch-train
```

### Cron (nightly meta-update):

```bash
0 2 * * * cd $AUTORL_WORKSPACE && source env.sh && \
          bash autoloop_meta.sh --apply >> $AUTORL_LOG/autoloop_meta.log 2>&1
```

---

## Key Parameters (agent-controlled in `train_trl.py`)

| Parameter | Default | Description |
|---|---|---|
| `ALGORITHM` | grpo | `grpo`, `reinforce`, `rloo`, `dapo`, `dr_grpo`, `auto` |
| `LR` | 1e-6 | Learning rate |
| `KL_COEFF` | 0.0 | KL penalty (leash strength) |
| `REWARD_SHAPING` | dense | `binary`, `dense`, `confidence_gated`, `cgdb_asymmetric` |
| `ALPHA_S` | 1.0 | Soundness penalty weight (Balcan et al. 2026) |
| `ALPHA_C` | 0.4 | Completeness credit weight (Balcan et al. 2026) |
| `N_SAMPLES` | 8 | Rollouts per prompt |
| `TEMPERATURE` | 0.8 | Sampling temperature |
| `HESSIAN_TRACKING` | True | Track λ_max via Lanczos HVP |
| `POWER_SAMPLING_BASELINE` | False | Run training-free baseline once |

`ALGORITHM=auto` — agent selects the best-performing algorithm based on `results_rl.tsv`.

---

## Self-Improvement Loop

New algorithms and reward functions are discovered automatically:

1. `scan_papers.py` queries arxiv + Semantic Scholar nightly
2. Papers tagged `algorithm` or `reward_shaping` are passed to `meta_update.py`
3. Ollama/Qwen reads the papers and patches `program_rl.md` with new experiment steps
4. With `--patch-train`, it also adds implementation stubs to `train_trl.py`
5. The inner loop picks up the new steps on the next iteration

---

## Metrics

| Metric | Description |
|---|---|
| `pass@1` | Single-shot accuracy on MATH val set |
| `pass@8` | Best-of-8 accuracy (diversity measure) |
| `escape_radius` | KL divergence from base model (leash metric) |
| `λ_max` | Hessian spectral norm (Invisible Leash correlation) |

Power sampling `pass@1` (Karan & Du 2025) serves as the training-free ceiling —
GRPO and other algorithms must exceed it to justify training cost.

---

## Related Papers

- **Invisible Leash** — escape_radius as RLVR leash metric
- **Balcan et al. (2026)** arXiv:2603.03538 — CoT verifier soundness/completeness asymmetry
- **Karan & Du (2025)** arXiv:2510.14901 — Power sampling training-free baseline

---

## Requirements

- Python ≥ 3.10
- CUDA matched to driver (auto-detected via `nvidia-smi`)
- Ollama running locally (`ollama serve`) for meta-update loop
- H100 or equivalent GPU recommended