#!/bin/bash
#
# A3C GPU: Weight Averaging (Federated Learning Style)
# =====================================================
#
# Usage:
#   ./run_a3c_gpu.sh [workers] [total_rounds] [sync_interval] [eval_window] [eps_start] [eps_end] [eps_decay_rounds]
#
# Examples:
#   ./run_a3c_gpu.sh                                    # Default: 4 workers, 50000 rounds
#   ./run_a3c_gpu.sh 8 100000                           # 8 workers, 100k rounds
#   ./run_a3c_gpu.sh 8 100000 1 200 0.9 0.0 2000       # Custom epsilon: 0.9→0.0 over 2000 rounds
#
# Parameters:
#   workers          - Number of parallel workers (default: 4)
#   total_rounds     - Maximum training rounds (default: 50000)
#   sync_interval    - Weight sync frequency (default: 1)
#   eval_window      - Performance evaluation window (default: 200)
#   eps_start        - Initial epsilon for teacher guidance (default: 0.95)
#   eps_end          - Final epsilon for teacher guidance (default: 0.1)
#   eps_decay_rounds - Rounds to decay epsilon (default: 1500)
#
# Architecture:
#   - Each worker has independent local model
#   - Workers update locally, send weights periodically
#   - Main averages weights and distributes
#
# This is better than gradient collection because:
#   - No gradient synchronization issues
#   - More stable learning
#   - True asynchronous updates
#

set -e

WORKERS=${1:-4}
TOTAL_ROUNDS=${2:-50000}
SYNC_INTERVAL=${3:-1}
EVAL_WINDOW=${4:-200}

# PPO Epsilon-greedy parameters (for teacher model guidance)
PPO_EPS_START=${5:-0.95}      # Start epsilon (high = more teacher guidance)
PPO_EPS_END=${6:-0.2}        # End epsilon (keep some teacher guidance, was 0.1)
PPO_EPS_DECAY_ROUNDS=${7:-400000}  # Decay over N rounds (slower decay, was 1500)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  A3C GPU: Weight Averaging (Federated Learning Style)            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: ✓ $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "GPU: ✗ (CPU mode)"
fi
echo ""

echo "Architecture:"
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  Each Worker: Independent Local Model              │"
echo "  │  - Collects experiences                            │"
echo "  │  - Updates locally via PPO                         │"
echo "  │  - Sends weights periodically                      │"
echo "  │                                                     │"
echo "  │  Main Process: Weight Averaging                    │"
echo "  │  - Collects weights from all workers              │"
echo "  │  - Averages weights → Global Model                 │"
echo "  │  - Distributes to workers                          │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

echo "Configuration:"
echo "  Workers: $WORKERS"
echo "  Max Rounds: $TOTAL_ROUNDS"
echo "  Sync Interval: $SYNC_INTERVAL batches"
echo "  Eval Window: $EVAL_WINDOW rounds"
echo ""

echo "Stage Progression (Team A win rate thresholds):"
echo "  ┌──────────────────────────────────────────────────────────────────┐"
echo "  │ Stage 1 │ Team A vs random, peaceful      │ 60% → advance       │"
echo "  │ Stage 2 │ Team A vs peaceful, coin_coll   │ 65% → advance       │"
echo "  │ Stage 3 │ Team A vs coin_coll, rule_based │ 70% → advance       │"
echo "  │ Stage 4 │ Team A vs rule_based, aggr_tchr │ 75% → advance       │"
echo "  │ Stage 5 │ SELF-PLAY: Team A vs Team B     │ same model, 2v2     │"
echo "  └──────────────────────────────────────────────────────────────────┘"
echo ""

echo "PPO Epsilon-Greedy (Teacher Model Guidance):"
echo "  Start: $PPO_EPS_START (teacher action probability)"
echo "  End: $PPO_EPS_END (teacher action probability)"
echo "  Decay: Over $PPO_EPS_DECAY_ROUNDS rounds"
echo "  (Epsilon linearly decreases from $PPO_EPS_START to $PPO_EPS_END)"
echo ""

echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Export environment variables for PPO agent
export PPO_EPS_START="$PPO_EPS_START"
export PPO_EPS_END="$PPO_EPS_END"
export PPO_EPS_DECAY_ROUNDS="$PPO_EPS_DECAY_ROUNDS"

python3 a3c_gpu_train.py \
    --num-workers "$WORKERS" \
    --total-rounds "$TOTAL_ROUNDS" \
    --sync-interval "$SYNC_INTERVAL" \
    --eval-window "$EVAL_WINDOW"

echo ""
echo "Done!"
