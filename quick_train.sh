#!/bin/bash
# Quick Training Script for Bomberman RL
# 교사 모델 기반 PPO 에이전트 학습 파이프라인

set -e  # Exit on error

echo "========================================="
echo "Bomberman RL - Quick Training Pipeline"
echo "========================================="
echo ""

# 1. Test teacher agent
echo "Step 1/5: Testing aggressive teacher agent..."
python3 main.py play \
  --agents aggressive_teacher_agent aggressive_teacher_agent rule_based_agent rule_based_agent \
  --n-rounds 20 \
  --no-gui \
  --save-stats results/teacher_test.json

echo "Generating teacher test charts..."
python3 plot_results.py results/teacher_test.json
echo "✓ Teacher test complete! Check results/teacher_test_*.png"
echo ""

# 2. Basic PPO training
echo "Step 2/5: Basic PPO training (100 rounds vs rule_based)..."
python3 main.py play \
  --agents ppo_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --no-gui \
  --n-rounds 100 \
  --save-stats results/ppo_basic.json

echo "Generating basic training charts..."
python3 plot_results.py results/ppo_basic.json
echo "✓ Basic training complete! Check results/ppo_basic_*.png"
echo ""

# 3. Advanced training with teacher model
echo "Step 3/5: Advanced PPO training (200 rounds vs teacher agent)..."
python3 main.py play \
  --agents ppo_agent ppo_agent aggressive_teacher_agent aggressive_teacher_agent \
  --train 2 \
  --no-gui \
  --n-rounds 200 \
  --save-stats results/ppo_vs_teacher.json

echo "Generating advanced training charts..."
python3 plot_results.py results/ppo_vs_teacher.json --include ppo_agent --rolling 20
echo "✓ Advanced training complete! Check results/ppo_vs_teacher_*.png"
echo ""

# 4. Progressive training (교사 모델을 최종 상대로 설정)
echo "Step 4/5: Progressive training (500 rounds, adaptive difficulty -> teacher agent)..."
python3 main.py play \
  --agents ppo_agent ppo_agent \
  --train 2 \
  --dynamic-opponents \
  --opponent-pool peaceful_agent coin_collector_agent random_agent \
  --rb-agent aggressive_teacher_agent \
  --rb-prob-start 0.05 \
  --rb-prob-end 0.7 \
  --n-rounds 500 \
  --no-gui \
  --save-stats results/ppo_progressive_all.json

echo "Generating progressive training charts..."
python3 plot_results.py results/ppo_progressive_all.json --include ppo_agent --rolling 50
echo "Analyzing matchups..."
python3 analyze_matchups.py results/ppo_progressive_all.json
echo "✓ Progressive training complete! Check results/ppo_progressive_*.png"
echo ""

# 5. Final evaluation
echo "Step 5/5: Final evaluation (100 rounds vs mixed opponents)..."
python3 main.py play \
  --agents ppo_agent ppo_agent rule_based_agent aggressive_teacher_agent \
  --no-gui \
  --n-rounds 100 \
  --save-stats results/final_evaluation.json

echo "Generating evaluation charts..."
python3 plot_results.py results/final_evaluation.json
echo "✓ Evaluation complete! Check results/final_evaluation_*.png"
echo ""

echo "========================================="
echo "Training Pipeline Complete!"
echo "========================================="
echo ""
echo "Summary of generated files:"
echo "  results/teacher_test_*.png       - Teacher agent performance"
echo "  results/ppo_basic_*.png          - Basic PPO training results"
echo "  results/ppo_vs_teacher_*.png     - Advanced training vs teacher"
echo "  results/ppo_progressive_*.png    - Progressive training results"
echo "  results/final_evaluation_*.png   - Final performance evaluation"
echo ""
echo "Model saved at: agent_code/ppo_agent/ppo_model.pt"
echo ""
echo "Next steps:"
echo "  1. Review charts in results/ directory"
echo "  2. Test with GUI: python3 main.py play --agents ppo_agent aggressive_teacher_agent"
echo "  3. Continue training if needed"
echo ""

