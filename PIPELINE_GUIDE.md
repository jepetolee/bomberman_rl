# 전체 파이프라인 실행 가이드

사전 학습된 모델을 사용하여 커리큘럼 러닝을 실행하고, 보상 정책을 확인하는 방법입니다.

## 📋 전체 프로세스

```
1. 사전 학습 모델 확인
   ↓
2. 커리큘럼 러닝 실행 (A3C)
   ↓
3. 보상 정책 확인 및 분석
```

---

## 🚀 빠른 시작

### 방법 1: 자동 스크립트 (권장)

```bash
# 기본 실행 (ppo_model.pt 사용)
./run_full_pipeline.sh

# 커스텀 모델 경로 지정
./run_full_pipeline.sh /path/to/pretrained_model.pt

# 모든 옵션 지정
./run_full_pipeline.sh \
  ppo_model.pt \                    # 사전 학습 모델
  config/trm_config.yaml \          # 설정 파일
  results/curriculum \              # 결과 디렉토리
  4 \                                # 워커 수
  50000                              # 총 라운드
```

### 방법 2: 단계별 실행

#### 1단계: 사전 학습 모델 확인

```bash
# 모델이 있는지 확인
ls -lh ppo_model.pt

# 모델 정보 확인
python3 -c "
import torch
state_dict = torch.load('ppo_model.pt', map_location='cpu', weights_only=True)
print(f'파라미터 수: {sum(p.numel() for p in state_dict.values()):,}')
print(f'레이어 수: {len(state_dict)}')
"
```

**모델이 없으면:**
```bash
# Phase 2 사전 학습 실행
python3 train_phase2.py --train-policy --num-epochs 100 --batch-size 512
```

#### 2단계: 커리큘럼 러닝 실행

```bash
# A3C 커리큘럼 러닝 (기본)
python3 a3c_gpu_train.py \
  --num-workers 4 \
  --total-rounds 50000 \
  --rounds-per-batch 5 \
  --sync-interval 40 \
  --results-dir results/curriculum \
  --model-path ppo_model.pt

# 또는 Planning 포함 (환경 모델 필요)
python3 a3c_planning_train.py \
  --num-workers 4 \
  --total-rounds 50000 \
  --planning-steps 100 \
  --env-model-path data/env_models/env_model.pt \
  --model-path ppo_model.pt
```

**커리큘럼 단계:**
- Stage 1: Easy (random_agent, peaceful_agent) - 승률 60% 이상
- Stage 2: Medium (peaceful_agent, coin_collector_agent) - 승률 65% 이상
- Stage 3: Hard (coin_collector_agent, rule_based_agent) - 승률 70% 이상
- Stage 4: Expert (team_teacher_agent, aggressive_teacher_agent) - 승률 75% 이상
- Stage 5: Self-Play (같은 모델끼리 대전)

#### 3단계: 보상 정책 확인

```bash
# 보상 정책 분석
python3 check_reward_policy.py \
  --model-path ppo_model.pt \
  --rounds 100 \
  --samples 1000

# 전체 평가 (다양한 상대와 대전)
python3 evaluate_model.py \
  --model-path ppo_model.pt \
  --rounds 50 \
  --results-dir results/evaluation
```

---

## 📊 보상 정책 분석 결과 해석

### 행동 분포 분석

```
행동 분포:
  BOMB    25.3% ████████████████████
  RIGHT   18.7% ██████████████
  UP      16.2% ████████████
  DOWN    15.8% ████████████
  LEFT    14.1% ██████████
  WAIT    9.9%  ███████
```

**의미:**
- **BOMB 비율이 높음 (25%)**: 공격적 전략, 킬 중심
- **WAIT 비율이 낮음 (<10%)**: 수동적 행동 최소화
- **이동 행동 균형**: 전략적 위치 이동

### 게임 통계 분석

```
게임 통계:
  킬: 45
  사망: 12
  킬/사망 비율: 3.75
  코인 수집: 23
  폭탄 사용: 67
```

**의미:**
- **킬/사망 비율 > 1.5**: 공격적 전략 성공
- **폭탄 사용 많음**: 적극적인 전투 참여
- **코인 수집**: 보조 목표도 수행

### 전략 패턴

```
전략 패턴:
  ✓ 공격적 전략 (킬 중심)
  ✓ 폭탄 활용 전략
  ✓ 수집 중심 전략
```

**의미:**
- **공격적 전략**: 킬을 우선시하는 행동
- **폭탄 활용**: 전투에서 폭탄 적극 사용
- **수집 중심**: 코인 수집도 병행

---

## ⚙️ 고급 설정

### 커리큘럼 러닝 파라미터 조정

```bash
# 더 빠른 학습 (적은 라운드)
python3 a3c_gpu_train.py \
  --num-workers 8 \
  --total-rounds 20000 \
  --rounds-per-batch 10 \
  --sync-interval 20

# 더 안정적인 학습 (많은 라운드)
python3 a3c_gpu_train.py \
  --num-workers 4 \
  --total-rounds 100000 \
  --rounds-per-batch 5 \
  --sync-interval 50
```

### 보상 정책 상세 분석

```bash
# 더 많은 샘플로 정확한 분석
python3 check_reward_policy.py \
  --model-path ppo_model.pt \
  --rounds 200 \
  --samples 5000

# 특정 상대와만 평가
python3 evaluate_model.py \
  --model-path ppo_model.pt \
  --opponents aggressive_teacher_agent,rule_based_agent \
  --rounds 100
```

---

## 📁 결과 파일 구조

```
results/
├── curriculum/                    # 커리큘럼 러닝 결과
│   ├── curriculum_training.log   # 학습 로그
│   ├── ppo_model.pt              # 최종 모델
│   └── worker_*.json             # 워커별 통계
│
├── evaluation/                    # 평가 결과
│   ├── evaluation_summary.json   # 평가 요약
│   └── eval_*.json               # 상대별 상세 결과
│
└── reward_policy_check/           # 보상 정책 분석
    └── policy_check.json         # 정책 분석 결과
```

---

## 🔍 문제 해결

### 모델을 찾을 수 없음

```bash
# Phase 2 사전 학습 실행
python3 train_phase2.py --train-policy --num-epochs 100

# 또는 기존 모델 경로 확인
find . -name "*.pt" -type f
```

### CUDA Out of Memory

```bash
# 워커 수 줄이기
python3 a3c_gpu_train.py --num-workers 2

# 또는 배치 크기 줄이기
python3 train_phase2.py --batch-size 256
```

### 커리큘럼이 진행되지 않음

```bash
# 승률 확인
grep "win_rate" results/curriculum/curriculum_training.log

# 더 많은 라운드 실행
python3 a3c_gpu_train.py --total-rounds 100000
```

---

## 📈 성능 모니터링

### 실시간 모니터링

```bash
# 학습 로그 실시간 확인
tail -f results/curriculum/curriculum_training.log

# GPU 사용량 확인
watch -n 1 nvidia-smi
```

### 성능 지표

- **승률**: 각 단계별 승률 (60% → 65% → 70% → 75%)
- **킬/사망 비율**: 공격 효율성 (목표: > 2.0)
- **코인 수집**: 보조 목표 달성도
- **평균 점수**: 전체 성능 지표

---

## 🎯 다음 단계

1. **Self-Play 강화**: Stage 5에서 더 많은 라운드 실행
2. **하이퍼파라미터 튜닝**: 학습률, 배치 크기 조정
3. **모델 앙상블**: 여러 모델 조합으로 성능 향상
4. **전문가 상대**: 더 강한 상대와 대전하여 실력 향상

---

## 📝 요약

```bash
# 전체 파이프라인 한 번에 실행
./run_full_pipeline.sh ppo_model.pt

# 또는 단계별 실행
python3 a3c_gpu_train.py --num-workers 4 --total-rounds 50000
python3 check_reward_policy.py --model-path ppo_model.pt --rounds 100
python3 evaluate_model.py --model-path ppo_model.pt --rounds 50
```

**예상 소요 시간:**
- 커리큘럼 러닝: 4-8시간 (워커 수와 라운드에 따라)
- 보상 정책 확인: 5-10분
- 전체 평가: 10-20분

