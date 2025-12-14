#!/bin/bash
# A3C Training with Planning (TRM-focused)
# ==========================================
#
# Policy Model: TRM만 학습 (ViT, Policy head, Value head는 frozen)
# Env Model: ViT + Policy head + Value head만 학습 (TRM은 frozen, 추론에도 사용 안함)
# Planning: Deep Supervision 적용 (n_sup 단계의 recursive reasoning)

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  A3C TRAINING WITH PLANNING (TRM-focused)                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Default: use number of CPU cores / 2
DEFAULT_WORKERS=$(($(nproc) / 2))
if [ $DEFAULT_WORKERS -lt 2 ]; then
    DEFAULT_WORKERS=2
fi

WORKERS=${1:-$DEFAULT_WORKERS}
ROUNDS=${2:-100000}

echo "Configuration:"
echo "  Workers: ${WORKERS}"
echo "  Total Rounds: ${ROUNDS}"
echo "  Policy: TRM만 학습 (ViT, Heads frozen)"
echo "  Env Model: ViT + Heads만 학습 (TRM frozen, 추론에도 사용 안함)"
echo "  Planning: Deep Supervision 적용"
echo ""

# Phase 2 policy model path (required)
PHASE2_MODEL=${3:-"data/policy_models/policy_phase2.pt"}

if [ ! -f "$PHASE2_MODEL" ]; then
    echo "❌ Error: Phase 2 model not found: $PHASE2_MODEL"
    echo "   Please run Phase 2 training first:"
    echo "   python3 train_phase2.py --train-policy --num-epochs 100"
    exit 1
fi

echo "✓ Phase 2 model found: $PHASE2_MODEL"
echo ""

# Set environment variables
export BOMBER_USE_TRM=1
export BOMBER_TRM_RECURRENT=0  # Non-recurrent (z always zero)
export BOMBER_FROZEN_VIT=1     # TRM만 학습

# Planning 활성화
export BOMBER_USE_PLANNING=1
export BOMBER_PLANNING_EPISODES=${BOMBER_PLANNING_EPISODES:-5}  # Planning episodes per round
export BOMBER_PLANNING_HORIZON=${BOMBER_PLANNING_HORIZON:-8}    # Planning rollout horizon
export BOMBER_PLANNING_MIN_BUFFER=${BOMBER_PLANNING_MIN_BUFFER:-200}  # Min visited states for planning

# Model configuration (should match policy_phase2.pt)
export BOMBER_VIT_DIM=256
export BOMBER_VIT_DEPTH=2
export BOMBER_VIT_HEADS=4
export BOMBER_TRM_N=4
export BOMBER_TRM_N_SUP=8

# Model path
export PPO_MODEL_PATH=${PPO_MODEL_PATH:-"data/policy_models/a3c_model.pt"}

# Copy Phase 2 model
echo "Copying Phase 2 model to: $PPO_MODEL_PATH"
mkdir -p "$(dirname $PPO_MODEL_PATH)"
cp "$PHASE2_MODEL" "$PPO_MODEL_PATH"
echo "✓ Model ready"
echo ""

echo "Planning Configuration:"
echo "  Episodes per round: ${BOMBER_PLANNING_EPISODES}"
echo "  Horizon: ${BOMBER_PLANNING_HORIZON}"
echo "  Min buffer: ${BOMBER_PLANNING_MIN_BUFFER}"
echo ""

echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Run A3C training
python3 a3c_gpu_train.py \
    --num-workers ${WORKERS} \
    --total-rounds ${ROUNDS} \
    --model-path ${PPO_MODEL_PATH} \
    --rounds-per-batch 5 \
    --sync-interval 40 \
    --global-weight-ratio 0.1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  A3C TRAINING COMPLETE!                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Model saved to: $PPO_MODEL_PATH"
echo ""

