#!/bin/bash
#
# 전체 파이프라인 실행 스크립트
# ==============================
# 1. 사전 학습된 모델 확인
# 2. 커리큘럼 러닝 실행
# 3. 보상 정책 확인
#

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 기본 설정
PRETRAINED_MODEL="${1:-ppo_model.pt}"  # 첫 번째 인자 또는 기본값
CONFIG_FILE="${2:-config/trm_config.yaml}"  # 두 번째 인자 또는 기본값
RESULTS_DIR="${3:-results/curriculum}"  # 세 번째 인자 또는 기본값
NUM_WORKERS="${4:-4}"  # 네 번째 인자 또는 기본값
TOTAL_ROUNDS="${5:-50000}"  # 다섯 번째 인자 또는 기본값

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}전체 파이프라인 실행${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "사전 학습 모델: ${GREEN}${PRETRAINED_MODEL}${NC}"
echo -e "설정 파일: ${GREEN}${CONFIG_FILE}${NC}"
echo -e "결과 디렉토리: ${GREEN}${RESULTS_DIR}${NC}"
echo -e "워커 수: ${GREEN}${NUM_WORKERS}${NC}"
echo -e "총 라운드: ${GREEN}${TOTAL_ROUNDS}${NC}"
echo ""

# ============== 1단계: 사전 학습 모델 확인 ==============
echo -e "${YELLOW}[1단계] 사전 학습 모델 확인${NC}"
echo "----------------------------------------"

if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo -e "${RED}✗ 사전 학습 모델을 찾을 수 없습니다: ${PRETRAINED_MODEL}${NC}"
    echo -e "${YELLOW}다음 중 하나를 실행하세요:${NC}"
    echo "  1. train_phase2.py로 사전 학습:"
    echo "     python3 train_phase2.py --train-policy --num-epochs 100"
    echo ""
    echo "  2. 또는 기존 모델 경로 지정:"
    echo "     $0 /path/to/pretrained_model.pt"
    exit 1
fi

echo -e "${GREEN}✓ 사전 학습 모델 확인: ${PRETRAINED_MODEL}${NC}"

# 모델 정보 출력
if command -v python3 &> /dev/null; then
    echo -e "${BLUE}모델 정보 확인 중...${NC}"
    python3 << EOF
import torch
import os
model_path = "$PRETRAINED_MODEL"
if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        print(f"  - 파라미터 수: {sum(p.numel() for p in state_dict.values()):,}")
        print(f"  - 레이어 수: {len(state_dict)}")
        print(f"  - 파일 크기: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"  - 모델 로드 실패: {e}")
EOF
fi

echo ""

# ============== 2단계: 커리큘럼 러닝 실행 ==============
echo -e "${YELLOW}[2단계] 커리큘럼 러닝 실행${NC}"
echo "----------------------------------------"

# 결과 디렉토리 생성
mkdir -p "$RESULTS_DIR"

# 커리큘럼 러닝 실행 (a3c_gpu_train.py 사용)
echo -e "${BLUE}커리큘럼 러닝 시작...${NC}"
echo -e "${YELLOW}주의: 이 과정은 시간이 오래 걸릴 수 있습니다.${NC}"
echo ""

# 환경 변수 설정
export PPO_MODEL_PATH="$PRETRAINED_MODEL"
export BOMBER_USE_TRM=1  # TRM recurrent mode 활성화

# 커리큘럼 러닝 실행
python3 a3c_gpu_train.py \
    --num-workers "$NUM_WORKERS" \
    --total-rounds "$TOTAL_ROUNDS" \
    --rounds-per-batch 5 \
    --sync-interval 40 \
    --results-dir "$RESULTS_DIR" \
    --model-path "$PRETRAINED_MODEL" \
    2>&1 | tee "${RESULTS_DIR}/curriculum_training.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ 커리큘럼 러닝 실패 (종료 코드: $TRAINING_EXIT_CODE)${NC}"
    echo -e "${YELLOW}로그 확인: ${RESULTS_DIR}/curriculum_training.log${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 커리큘럼 러닝 완료${NC}"
echo ""

# 최종 모델 경로 확인
FINAL_MODEL="$PRETRAINED_MODEL"
if [ -f "${RESULTS_DIR}/ppo_model.pt" ]; then
    FINAL_MODEL="${RESULTS_DIR}/ppo_model.pt"
    echo -e "${BLUE}최종 모델: ${FINAL_MODEL}${NC}"
fi

# ============== 3단계: 보상 정책 확인 ==============
echo -e "${YELLOW}[3단계] 보상 정책 확인${NC}"
echo "----------------------------------------"

EVAL_DIR="${RESULTS_DIR}/evaluation"
mkdir -p "$EVAL_DIR"

echo -e "${BLUE}모델 평가 시작...${NC}"

# 평가 실행
python3 evaluate_model.py \
    --model-path "$FINAL_MODEL" \
    --rounds 50 \
    --results-dir "$EVAL_DIR" \
    2>&1 | tee "${EVAL_DIR}/evaluation.log"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ 평가 실패 (종료 코드: $EVAL_EXIT_CODE)${NC}"
    echo -e "${YELLOW}로그 확인: ${EVAL_DIR}/evaluation.log${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 평가 완료${NC}"
echo ""

# ============== 보상 정책 분석 ==============
echo -e "${YELLOW}[4단계] 보상 정책 분석${NC}"
echo "----------------------------------------"

python3 << EOF
import json
import os
import sys

eval_dir = "$EVAL_DIR"
summary_file = os.path.join(eval_dir, "evaluation_summary.json")

if not os.path.exists(summary_file):
    print("✗ 평가 요약 파일을 찾을 수 없습니다.")
    sys.exit(1)

with open(summary_file, 'r') as f:
    summary = json.load(f)

print("\n📊 보상 정책 분석 결과")
print("=" * 70)
print(f"\n모델: {summary['model_path']}")
print(f"평가 날짜: {summary['evaluation_date']}")
print(f"전체 승률: {summary['overall_win_rate']:.1f}%")
print(f"\n상세 결과:")
print(f"{'상대':<40} {'승리':<6} {'점수 차이':<12} {'킬 차이':<12}")
print("-" * 70)

for opp_name, stats in summary['results'].items():
    win_str = "✓" if stats['win'] else "✗"
    print(f"{opp_name:<40} {win_str:<6} {stats['score_diff']:>+10.1f}  {stats['kill_diff']:>+10}")

# 보상 정책 추론
print("\n🎯 보상 정책 추론:")
print("-" * 70)

# 킬 중심 전략 확인
total_kill_diff = sum(s['kill_diff'] for s in summary['results'].values())
if total_kill_diff > 0:
    print("✓ 공격적 전략 (킬 중심)")
else:
    print("✗ 공격적 전략 미흡")

# 생존 전략 확인
total_deaths = sum(s['ppo_deaths'] for s in summary['results'].values())
if total_deaths < len(summary['results']) * 10:  # 상대적으로 적은 사망
    print("✓ 생존 전략 우수")
else:
    print("✗ 생존 전략 개선 필요")

# 다양한 상대 대응 확인
win_count = sum(1 for s in summary['results'].values() if s['win'])
if win_count >= len(summary['results']) * 0.7:
    print("✓ 다양한 상대에 대한 적응력 우수")
else:
    print("✗ 다양한 상대에 대한 적응력 개선 필요")

print(f"\n📁 상세 결과: {summary_file}")
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}전체 파이프라인 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "결과 디렉토리: ${BLUE}${RESULTS_DIR}${NC}"
echo -e "  - 학습 로그: ${RESULTS_DIR}/curriculum_training.log"
echo -e "  - 평가 결과: ${EVAL_DIR}/evaluation_summary.json"
echo -e "  - 최종 모델: ${FINAL_MODEL}"
echo ""

