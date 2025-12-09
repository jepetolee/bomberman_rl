#!/bin/bash
# Self-Play Training Script
# A3C-style distributed learning + AlphaZero-style self-play

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SELF-PLAY TRAINING SYSTEM                                   ║"
echo "║  A3C + AlphaZero Style                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Default settings
TEACHER_ROUNDS=1000
SELFPLAY_ROUNDS=2000
WIN_THRESHOLD=0.25

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            TEACHER_ROUNDS=200
            SELFPLAY_ROUNDS=500
            echo "Quick mode: ${TEACHER_ROUNDS} teacher + ${SELFPLAY_ROUNDS} self-play rounds"
            shift
            ;;
        --standard)
            TEACHER_ROUNDS=1000
            SELFPLAY_ROUNDS=2000
            echo "Standard mode: ${TEACHER_ROUNDS} teacher + ${SELFPLAY_ROUNDS} self-play rounds"
            shift
            ;;
        --long)
            TEACHER_ROUNDS=2000
            SELFPLAY_ROUNDS=5000
            echo "Long mode: ${TEACHER_ROUNDS} teacher + ${SELFPLAY_ROUNDS} self-play rounds"
            shift
            ;;
        --help)
            echo "Usage: $0 [--quick|--standard|--long]"
            echo ""
            echo "Options:"
            echo "  --quick     Quick training (200 + 500 rounds)"
            echo "  --standard  Standard training (1000 + 2000 rounds)"
            echo "  --long      Long training (2000 + 5000 rounds)"
            echo ""
            echo "Or run directly with custom settings:"
            echo "  python3 self_play_train.py --teacher-rounds 1000 --selfplay-rounds 2000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Phase 1 (Teacher): ${TEACHER_ROUNDS} rounds vs aggressive_teacher_agent"
echo "  Phase 2 (Self-Play): ${SELFPLAY_ROUNDS} rounds"
echo "  Win threshold: ${WIN_THRESHOLD} (25%)"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Run self-play training
python3 self_play_train.py \
    --teacher-rounds ${TEACHER_ROUNDS} \
    --selfplay-rounds ${SELFPLAY_ROUNDS} \
    --win-threshold ${WIN_THRESHOLD}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TRAINING COMPLETE!                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Check results: ls -la results/self_play_*/"
echo "  2. View charts: eog results/self_play_*/*.png"
echo "  3. Test model: python3 main.py play --agents ppo_agent aggressive_teacher_agent"
echo ""

