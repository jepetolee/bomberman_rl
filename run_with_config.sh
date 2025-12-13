#!/bin/bash
# Run training with YAML configuration
# ====================================
# This script loads configuration from YAML and runs the appropriate training phase

set -e

CONFIG_FILE="${1:-config/trm_config.yaml}"
PRESET="${2:-}"
PHASE="${3:-all}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TRM Training with YAML Configuration                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Config file: $CONFIG_FILE"
[ -n "$PRESET" ] && echo "Preset: $PRESET"
echo "Phase: $PHASE"
echo ""

# Load configuration and apply to environment
if [ -n "$PRESET" ]; then
    python3 config/load_config.py --config "$CONFIG_FILE" --preset "$PRESET" --apply-env
else
    python3 config/load_config.py --config "$CONFIG_FILE" --apply-env
fi

# Run the specified phase
case "$PHASE" in
    phase1|1)
        echo "Running Phase 1: Teacher Data Collection"
        python3 collect_teacher_data.py
        ;;
    
    phase2|2)
        echo "Running Phase 2: Dyna-Q + DeepSupervision"
        python3 train_phase2.py --train-env-model
        python3 train_phase2.py --train-policy --use-planning
        ;;
    
    phase3|3)
        echo "Running Phase 3: Recurrent RL"
        bash train_phase3.sh
        ;;
    
    all|*)
        echo "Running all phases..."
        echo ""
        echo "Phase 1: Teacher Data Collection"
        python3 collect_teacher_data.py
        echo ""
        echo "Phase 2: Environment Model + Policy Training"
        python3 train_phase2.py --train-env-model
        python3 train_phase2.py --train-policy --use-planning
        echo ""
        echo "Phase 3: Recurrent RL"
        bash train_phase3.sh
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

