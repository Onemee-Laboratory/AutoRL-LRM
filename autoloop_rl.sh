#!/bin/bash
# autoloop_rl.sh — Autonomous RL Research Loop
# first set AUTORL_HOME in by "bash env.sh"

_this_dir="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
source "$_this_dir/env.sh"

AUTORL_LOG=$AUTORL_HOME/log

RESULTS="results_rl.tsv"

# Initialize results file
if [ ! -f "$RESULTS" ]; then
    echo -e "commit\tpass@1\tpass@8\tescape_radius\tstatus\tdescription" > $RESULTS
fi

EXPERIMENT=0

while true; do
    EXPERIMENT=$((EXPERIMENT + 1))
    echo "=========================================="
    echo "Experiment #$EXPERIMENT starting at $(date)"
    echo "=========================================="

    # Step 1 — train
    echo "--- TRAINING ---"
    python train_trl.py 2>&1 | tee $AUTORL_LOG/train_rl.log

    # Step 2 — evaluate
    echo "--- EVALUATION ---"
    python eval_rl.py 2>&1 | tee $AUTORL_LOG/eval_rl.log

    # Step 3 — extract metrics
    PASS1=$(grep -oP 'pass@1:\s*\K[\d.]+' $AUTORL_LOG/eval_rl.log | tail -1)
    PASS8=$(grep -oP 'pass@8:\s*\K[\d.]+' $AUTORL_LOG/eval_rl.log | tail -1)
    ESCAPE=$(grep -oP 'escape_radius:\s*\K[\d.]+' $AUTORL_LOG/eval_rl.log | tail -1)
    COMMIT=$(git rev-parse --short HEAD)

    echo "Result: pass@1=$PASS1 pass@8=$PASS8 escape_radius=$ESCAPE"

    # Step 4 — log to results.tsv
    if [ ! -z "$PASS1" ]; then
        echo -e "$COMMIT\t$PASS1\t$PASS8\t$ESCAPE\tpending\texperiment_$EXPERIMENT" >> $RESULTS
    fi

    # Step 5 — clean logs
    CLEAN_LOG=$(cat $AUTORL_LOG/train_rl.log $AUTORL_LOG/eval_rl.log | grep -v "https://" | grep -v "http://" | tail -50)

    # Step 6 — aider proposes next experiment
    OLLAMA_API_BASE=http://localhost:11434 OLLAMA_REQUEST_TIMEOUT=1200 aider \
        --model ollama/qwen3-coder-next \
        --read program_rl.md \
        train_trl.py results_rl.tsv \
        --yes \
        --no-auto-commits \
        --message "$CLEAN_LOG
---
pass@1=$PASS1 pass@8=$PASS8 escape_radius=$ESCAPE
Baseline: pass@1=0.280 pass@8=0.430 escape_radius=0.000
Update results_rl.tsv status (keep if pass@1 > 0.280, else discard).
Then make ONE small change to train_trl.py to improve pass@1.
Follow STRICT RULES in program_rl.md."

    # Step 7 — commit
    git add train_trl.py results_rl.tsv
    git commit -m "rl-exp-$EXPERIMENT: pass@1=$PASS1 escape=$ESCAPE" \
               --allow-empty 2>/dev/null

    # Step 8 - push
    #git push origin main --force

    echo "Experiment #$EXPERIMENT done."
    echo ""
done
