# Frozen ViT 학습 가이드

ViT 백본을 고정하고, Value Network와 TRM만 강화학습을 수행하는 방법입니다.

## 🎯 목적

사전 학습된 ViT 백본의 특징 추출 능력을 유지하면서, Value Network와 TRM만 RL로 미세 조정합니다.

## 📋 설정 방법

### 1. 환경 변수 설정

```bash
# Frozen ViT 모드 활성화
export BOMBER_FROZEN_VIT=1

# 사전 학습 모델 경로 지정
export PPO_MODEL_PATH=data/policy_models/policy_phase2.pt

# TRM recurrent 모드 활성화 (선택)
export BOMBER_USE_TRM=1
```

### 2. 실행

```bash
# A3C 커리큘럼 러닝 (Frozen ViT 모드)
BOMBER_FROZEN_VIT=1 \
PPO_MODEL_PATH=data/policy_models/policy_phase2.pt \
python3 a3c_gpu_train.py \
  --num-workers 4 \
  --total-rounds 50000 \
  --model-path data/policy_models/policy_phase2.pt
```

## 🔧 동작 원리

### 학습되는 파라미터

1. **Value Network** (`model.v_head`)
   - 가치 예측을 위한 네트워크
   - RL 환경에서 보상 신호를 통해 학습

2. **TRM (Tiny Recursive Model)**
   - `trm_patch_proj`: TRM용 패치 임베딩
   - `trm_pos_embed`: 위치 임베딩
   - `trm_net`: 재귀 추론 네트워크

3. **Policy Head** (`model.pi_head`)
   - TRM과 연결된 정책 출력 레이어

### 고정되는 파라미터

1. **ViT 백본** (`model.vit`)
   - 모든 ViT 레이어 (패치 임베딩, 트랜스포머 블록, 정규화)
   - 사전 학습된 특징 추출 능력 유지

## 📊 보상 정책 확인 결과

### COIN_FOUND 이벤트

- **발생 조건**: 크레이트를 파괴하여 코인이 드러날 때
- **보상**: 0.1 (낮은 보상)
- **위치**: `environment.py`의 `update_bombs()` 함수

### BFS 맵과 보상

**현재 상태:**
- BFS 맵이 특징 맵에 포함되지 않음
- 거리 기반 shaping reward 없음
- 적의 코인과 아군의 코인을 구분하지 않음

**특징 맵 구조:**
```
grid[0]: 벽
grid[1]: 크레이트
grid[2]: 빈 공간
grid[3]: 코인 (모든 코인, 구분 없음)
grid[4]: 폭탄
grid[5]: 폭발
grid[6]: 자신
grid[7]: 아군
grid[8]: 적
grid[9]: 위험 맵
```

**개선 제안:**
1. 적의 코인까지의 BFS 거리를 특징 맵에 추가 (grid[10])
2. 거리 기반 shaping reward 추가
3. 적의 코인에 더 높은 보상 부여

## 🚀 실행 예제

### 전체 파이프라인

```bash
# 1. 사전 학습 모델 확인
ls -lh data/policy_models/policy_phase2.pt

# 2. Frozen ViT 모드로 커리큘럼 러닝
BOMBER_FROZEN_VIT=1 \
PPO_MODEL_PATH=data/policy_models/policy_phase2.pt \
python3 a3c_gpu_train.py \
  --num-workers 4 \
  --total-rounds 50000 \
  --results-dir results/frozen_vit

# 3. 보상 정책 확인
python3 check_coin_reward.py --check-events --check-reward

# 4. 모델 평가
python3 evaluate_model.py \
  --model-path results/frozen_vit/ppo_model.pt \
  --rounds 50
```

## 📈 예상 효과

1. **학습 안정성**: ViT 백본 고정으로 특징 공간 안정화
2. **학습 속도**: 학습 파라미터 감소로 빠른 수렴
3. **전이 학습**: 사전 학습된 특징 활용

## ⚠️ 주의사항

1. **모델 호환성**: `PolicyValueViT_TRM_Hybrid` 모델만 지원
2. **사전 학습 필수**: ViT 백본이 사전 학습되어 있어야 함
3. **TRM 활성화**: RL 단계에서는 `use_trm=True`로 설정

## 🔍 디버깅

### Frozen ViT 모드 확인

```python
# 학습 중 로그 확인
[Frozen ViT] ViT 백본 고정, Value Network와 TRM만 학습
[Frozen ViT Optimizer] Trainable parameters: X,XXX
[Frozen ViT Optimizer] Total parameters: XX,XXX
[Frozen ViT Optimizer] Frozen (ViT): XX,XXX
```

### 파라미터 확인

```python
import torch
from agent_code.ppo_agent.models.vit_trm import PolicyValueViT_TRM_Hybrid

model = PolicyValueViT_TRM_Hybrid(...)

# ViT 파라미터 확인 (고정되어야 함)
for param in model.vit.parameters():
    print(f"ViT param requires_grad: {param.requires_grad}")  # False

# TRM 파라미터 확인 (학습되어야 함)
for param in model.trm_net.parameters():
    print(f"TRM param requires_grad: {param.requires_grad}")  # True
```

## 📝 요약

- **Frozen ViT 모드**: `BOMBER_FROZEN_VIT=1` 환경 변수로 활성화
- **학습 파라미터**: Value Network + TRM + Policy Head
- **고정 파라미터**: ViT 백본 전체
- **보상 정책**: COIN_FOUND는 0.1, BFS 맵은 현재 미사용

