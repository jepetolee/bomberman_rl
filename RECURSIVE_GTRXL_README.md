# Recursive GTrXL 구현 가이드

## 🏛️ 아키텍처 개요

**Recursive GTrXL**은 GTrXL의 강력한 기억 능력과 TRM의 파라미터 효율성을 결합한 설계입니다.

### 핵심 아이디어: "One Block, Many Thoughts"

- **기존 GTrXL**: 12개의 레이어를 쌓아 올린 구조
- **Recursive GTrXL**: **단 1개의 GTrXL 블록을 K번 재귀적으로 통과**시켜 깊은 추론 구현

이 방식의 장점:
- 파라미터 수: 기존 대비 ~1/K 수준 (예: K=4면 1/4)
- 추론 깊이: K층짜리 모델과 유사한 성능
- 안정성: GTrXL의 Gating 메커니즘 덕분에 재귀적으로 돌아도 신호 안정성 유지

## 📋 구조

```
Input (Game Grid)
    ↓
CNN Backbone (EfficientNetB0-style)
    ↓
Feature Projection + Positional Encoding
    ↓
[Recursive Core]
    ├─ Step Embedding (k=0)
    ├─ Shared GTrXL Block (1st pass)
    ├─ Step Embedding (k=1)
    ├─ Shared GTrXL Block (2nd pass)
    ├─ ...
    ├─ Step Embedding (k=K-1)
    └─ Shared GTrXL Block (Kth pass)
    ↓
Final LayerNorm
    ↓
Policy Head + Value Head
```

## 🔧 설정 (YAML)

`config/trm_config.yaml`에서 다음과 같이 설정:

```yaml
model:
  type: "recursive_gtrxl"  # 모델 타입 지정
  embed_dim: 256
  in_channels: 10
  num_actions: 6
  img_size: [17, 17]
  
  recursive_gtrxl:
    # CNN backbone
    cnn_base_channels: 32
    cnn_width_mult: 1.0
    
    # Recursive GTrXL
    n_layers_simulated: 4  # 재귀 횟수 (K) - 추론 깊이
                           # 권장: 4~8, 처음엔 4로 시작
    num_heads: 8
    memory_size: 128       # 각 재귀 단계마다 별도 메모리 슬롯
```

## 💻 사용 방법

### 1. 모델 생성 (자동)

YAML 설정을 읽어서 자동으로 모델이 생성됩니다:

```python
from config.load_config import load_config, create_model_from_config
import torch

config = load_config("config/trm_config.yaml")
model = create_model_from_config(config, device=torch.device("cuda"))

# 사용
x = torch.randn(1, 10, 17, 17)  # [B, C, H, W]
logits, value = model(x)  # Policy logits, Value
```

### 2. 메모리와 함께 사용 (Transformer-XL 스타일)

```python
# 초기 메모리 없음
logits, value = model(x)

# 메모리 업데이트와 함께
logits, value, new_memory = model.forward_with_memory(x, memory=None)
# new_memory: [n_layers_simulated, B, T, D] - 각 재귀 단계별 메모리

# 다음 스텝에서 메모리 사용
logits2, value2, new_memory2 = model.forward_with_memory(x2, memory=new_memory)
```

## 🔬 구현 세부사항

### Step Embedding

각 재귀 단계(k)에 고유한 임베딩을 더해 모델이 현재 추론 단계를 인식:

```python
step_signal = self.step_embedding(torch.tensor(k, device=device))
h = h + step_signal  # [B, T, D]
```

### 메모리 관리

물리적으로는 레이어가 1개지만, 논리적으로는 K층이므로 **메모리도 K개의 슬롯**을 가집니다:

- `memory=None`: 모든 단계에서 메모리 없음
- `memory=[B, M, D]`: 모든 단계에서 동일한 메모리 사용
- `memory=[K, B, M, D]`: 각 단계별로 별도 메모리 (권장)

### Gating 안정성

GTrXL의 Gated Residual 연결 덕분에:
- 재귀적으로 여러 번 통과해도 값이 발산하거나 소실되지 않음
- Gate가 입력값과 출력값을 적절히 섞어줌
- TRM 구현의 핵심 메커니즘

## 📊 성능 비교 (예상)

| 모델 | 파라미터 수 | 추론 깊이 | 메모리 |
|------|------------|----------|--------|
| GTrXL (K=4) | N | 4 | O(4) |
| Recursive GTrXL (K=4) | N/4 | 4 | O(4) |

## ⚙️ 하이퍼파라미터 권장값

- **n_layers_simulated**: 처음엔 **4**로 시작, 필요시 6~8로 증가
- **embed_dim**: 256 (작은 모델) 또는 512 (큰 모델)
- **num_heads**: 8
- **memory_size**: 128 (작은 모델) 또는 256 (큰 모델)

## 🚀 향후 확장 가능성

1. **ACT (Adaptive Computation Time)**: 상황에 따라 재귀 횟수를 동적으로 조절
2. **Early Exit**: 충분히 추론이 끝났으면 조기 종료
3. **Multi-Scale Recursion**: 단계별로 다른 attention 범위 사용

## 📝 참고

- 원 논문: "Stabilizing Transformers for RL" (Parisotto et al., 2019)
- GTrXL의 Gating 메커니즘이 재귀 구조의 안정성을 보장
- Bomberman RL에서 장기 인과관계 (폭탄 → 적 가둠 → 터짐) 학습에 유용
