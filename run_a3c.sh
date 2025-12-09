#!/bin/bash
# A3C (Asynchronous Advantage Actor-Critic) Training Script
# Multiple workers asynchronously collect experience and update shared model

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  A3C TRAINING - Asynchronous Advantage Actor-Critic         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Default: use number of CPU cores / 2
DEFAULT_WORKERS=$(($(nproc) / 2))
if [ $DEFAULT_WORKERS -lt 2 ]; then
    DEFAULT_WORKERS=2
fi

WORKERS=${1:-$DEFAULT_WORKERS}
ROUNDS=${2:-2000}

echo "Configuration:"
echo "  Workers: ${WORKERS} (parallel environments)"
echo "  Total Rounds: ${ROUNDS}"
echo "  Opponent: aggressive_teacher_agent"
echo ""
echo "This will run ${WORKERS} game environments simultaneously,"
echo "each collecting experience and updating the shared model."
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

python3 a3c_train.py \
    --num-workers ${WORKERS} \
    --total-rounds ${ROUNDS} \
    --opponent aggressive_teacher_agent \
    --selfplay \
    --selfplay-ratio 0.2

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  A3C TRAINING COMPLETE!                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Check results: ls -la results/a3c_*/"
echo "Test model: python3 main.py play --agents ppo_agent aggressive_teacher_agent"
echo ""

