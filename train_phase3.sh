#!/bin/bash
# Phase 3: Recurrent RL Training Script
# TRM with recurrent latent z across timesteps

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: RECURRENT RL TRAINING                              ║"
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
echo "  Model: TRM with Recurrent z"
echo ""

# Phase 2 policy model path (if exists, will be loaded)
PHASE2_MODEL=${3:-"data/policy_models/policy_phase2.pt"}

# Set environment variables for TRM recurrent mode
export BOMBER_USE_TRM=1
export BOMBER_TRM_RECURRENT=1
# Default to frozen ViT (train TRM + value while keeping ViT features fixed).
# Override by exporting BOMBER_FROZEN_VIT=0 if you want full fine-tuning.
export BOMBER_FROZEN_VIT=${BOMBER_FROZEN_VIT:-1}
export BOMBER_TRM_N=6
export BOMBER_TRM_T=3
export BOMBER_TRM_N_SUP=16
# Note: z_dim is always equal to embed_dim (not configurable separately)
# export BOMBER_TRM_Z_DIM=256  # Deprecated: z_dim = embed_dim automatically

# ViT configuration (matches policy_phase2.pt checkpoint)
export BOMBER_VIT_DIM=256
export BOMBER_VIT_DEPTH=2
export BOMBER_VIT_HEADS=4

# Model path
export PPO_MODEL_PATH=${PPO_MODEL_PATH:-"agent_code/ppo_agent/ppo_model_phase3.pt"}

# Copy Phase 2 model if exists
if [ -f "$PHASE2_MODEL" ]; then
    echo "Loading Phase 2 model from: $PHASE2_MODEL"
    mkdir -p "$(dirname $PPO_MODEL_PATH)"
    cp "$PHASE2_MODEL" "$PPO_MODEL_PATH"
    echo "  → Copied to: $PPO_MODEL_PATH"
else
    echo "⚠️  Phase 2 model not found: $PHASE2_MODEL"
    echo "   Starting with random initialization"
fi

echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

python3 a3c_gpu_train.py \
    --num-workers ${WORKERS} \
    --total-rounds ${ROUNDS} \
    --opponent aggressive_teacher_agent \
    --selfplay \
    --selfplay-ratio 0.2

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3 TRAINING COMPLETE!                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Model saved to: $PPO_MODEL_PATH"
echo "Test model: python3 main.py play --agents ppo_agent aggressive_teacher_agent"
echo ""

