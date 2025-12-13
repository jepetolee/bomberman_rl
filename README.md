## Bomberman RL (TRM + ViT Hybrid)

이 레포는 Bomberman 환경에서 **Vision Transformer(ViT)** 기반 정책에 **TRM(Tiny Recursive Model)** 분기를 결합하고, 다음 파이프라인으로 학습합니다.

- **Phase 1**: teacher agent로 episode 데이터 수집 (`data/teacher_episodes/episode_*.pt`)
- **Phase 2**: teacher 데이터로 (선택)환경모델 + **DeepSupervision 사전학습** (`data/policy_models/policy_phase2.pt`)
- **Phase 3**: A3C 스타일 커리큘럼 RL (원하면 recurrent TRM latent `z`)

중요: **BFS는 “보상 shaping”에만 사용**합니다. 관측(feature/embedding)에는 포함하지 않습니다.

---

## 프로젝트 현재 스펙 (중요)

- **관측 채널 수**: `in_channels=10`
  - `agent_code/ppo_agent/callbacks.py: state_to_features()`가 10채널을 생성합니다.
- **Reward-only BFS coin shaping**: `agent_code/ppo_agent/train.py`
  - `_get_distance_to_enemy_coins()`로 계산한 거리 변화에 따라 shaping reward를 부여합니다.
  - (관측에 BFS 맵을 넣지 않습니다.)
- **Hybrid 모델**: `agent_code/ppo_agent/models/vit_trm.py: PolicyValueViT_TRM_Hybrid`
  - ViT backbone + TRM residual을 더해 policy/value head로 전달

---

## 설치

### 필수

```bash
pip install -r requirements.txt
```

### YAML 설정 사용 시

```bash
pip install pyyaml
```

---

## 데이터/모델 경로

- **Teacher episodes**: `data/teacher_episodes/episode_*.pt`
  - 디렉토리가 매우 클 수 있습니다(수십 GB). `.gitignore`로 커밋에서 제외합니다.
- **Phase 2 policy 모델**: `data/policy_models/policy_phase2.pt`

---

## 실행(공식 엔트리포인트)

이 레포에서 “공식”으로 유지하는 `.sh`는 2개만 남깁니다.

- `run_with_config.sh`: YAML 기반으로 Phase1~3 실행
- `train_phase3.sh`: Phase3 RL 실행(옵션)

---

## 설정(YAML)

- 설정 파일: `config/trm_config.yaml`
- 로더/적용: `config/load_config.py`

### 1) 설정을 환경변수로 적용만 하기

```bash
python3 config/load_config.py --config config/trm_config.yaml --apply-env
```

### 2) Phase별 실행 (권장)

```bash
# Phase 1: teacher episode 수집
bash run_with_config.sh config/trm_config.yaml "" phase1

# Phase 2: DeepSupervision 사전학습 (+ planning 옵션은 config에서)
bash run_with_config.sh config/trm_config.yaml "" phase2

# Phase 3: RL
bash run_with_config.sh config/trm_config.yaml "" phase3
```

---

## Phase 1: Teacher 데이터 수집

```bash
python3 collect_teacher_data.py
```

출력:
- `data/teacher_episodes/episode_XXXXXX.pt`

---

## Phase 2: DeepSupervision 사전학습

### Policy만 학습(기본)

```bash
python3 train_phase2.py --train-policy
```

YAML에서 아래가 사용됩니다:
- `phase2.data_dir`
- `phase2.deepsupervision.*`
- `phase2.deepsupervision.policy_model_path` (기본: `data/policy_models/policy_phase2.pt`)

### (선택) planning 사용

planning을 쓰려면 환경 모델도 함께 쓰게 됩니다(설정에서 on/off).

```bash
python3 train_phase2.py --train-env-model
python3 train_phase2.py --train-policy --use-planning
```

---

## Phase 3: A3C 커리큘럼 RL

### 기본 실행

```bash
python3 a3c_gpu_train.py --num-workers 4 --total-rounds 50000 --model-path data/policy_models/policy_phase2.pt
```

### (옵션) recurrent TRM z 모드

`train_phase3.sh`는 TRM recurrent 관련 환경변수 설정 + A3C 실행을 묶어둔 스크립트입니다.

```bash
bash train_phase3.sh
```

---

## Frozen ViT 모드 (옵션)

ViT 백본은 고정하고, **TRM + value head(및 필요한 head)**만 RL로 미세조정하고 싶다면:

```bash
BOMBER_FROZEN_VIT=1 PPO_MODEL_PATH=data/policy_models/policy_phase2.pt \
python3 a3c_gpu_train.py --num-workers 4 --total-rounds 50000 --model-path data/policy_models/policy_phase2.pt
```

---

## 평가/분석

### 모델 성능 평가

```bash
python3 evaluate_model.py --model-path data/policy_models/policy_phase2.pt --rounds 50
```

### “보상 정책” 성향 체크(행동 분포 등)

```bash
python3 check_reward_policy.py --model-path data/policy_models/policy_phase2.pt --rounds 100 --samples 1000
```

### 결과 플롯

```bash
python3 plot_results.py results/some_stats.json
```

---

## 보상 정책(핵심)

### COIN_FOUND

`environment.py`에서 크레이트가 터지며 코인이 “드러날 때” 이벤트가 발생합니다.

### Reward-only BFS coin shaping

- **목적**: 적 주변 코인에 더 빨리 접근하도록 shaping
- **중요**: BFS 거리/맵은 **관측(feature)에 넣지 않고**, `train.py`에서 **보상 계산에만 사용**합니다.

코드: `agent_code/ppo_agent/train.py`

---

## 트러블슈팅

### `ModuleNotFoundError: No module named 'yaml'`

```bash
pip install pyyaml
```

### CUDA OOM

- Phase2는 teacher 데이터가 크면 메모리 사용량이 커집니다.
- `config/trm_config.yaml`의 `phase2.deepsupervision.batch_size`를 줄이거나, `train_phase2.py`에서 학습 에피소드 샘플링을 조정하세요.

---

## Git 커밋 / .gitignore

대용량 산출물(teacher 데이터, 모델 체크포인트, 결과 폴더 등)은 `.gitignore`로 제외되어 있습니다.
설정 파일(`config/*.yaml`)은 커밋됩니다.

### 체크포인트 구조

학습 중 자동으로 저장되는 체크포인트:

```
agent_code/ppo_agent/checkpoints/
├── pre_selfplay_20241201_123456.pt   # Phase 1 완료 후
├── gen_0001.pt                        # 1세대
├── gen_0002.pt                        # 2세대
├── gen_0003.pt                        # 3세대
├── ...
└── final_20241201_134567.pt          # 최종 모델
```

---

## 5. 분산 학습 (Multi-GPU)

[[memory:6591693]]

```bash
# 4개 GPU로 분산 학습
torchrun --nproc_per_node 4 test.py

# 레퍼런스 서버 시작
python test.py ref
```

---

## 6. 저장된 모델 평가

```bash
# 학습된 PPO 에이전트 vs 교사 모델 (GUI로 확인)
python main.py play \
  --agents ppo_agent aggressive_teacher_agent aggressive_teacher_agent rule_based_agent \
  --n-rounds 10

# GUI 없이 빠른 평가
python main.py play \
  --agents ppo_agent ppo_agent aggressive_teacher_agent aggressive_teacher_agent \
  --no-gui \
  --n-rounds 100 \
  --save-stats results/evaluation.json

python plot_results.py results/evaluation.json
```

---

## 7. 에이전트 설명

### 기본 제공 에이전트

| 에이전트 | 설명 | 난이도 |
|---------|------|--------|
| `random_agent` | 랜덤 행동 | ⭐ |
| `peaceful_agent` | 폭탄 안 놓고 코인만 수집 | ⭐ |
| `coin_collector_agent` | 코인 수집 중심 | ⭐⭐ |
| `rule_based_agent` | 규칙 기반, BFS 탐색 | ⭐⭐⭐ |
| `aggressive_teacher_agent` | **A* 추적, 전략적 공격, 팀 협동** | ⭐⭐⭐⭐ |
| `ppo_agent` | 강화학습 (PPO) | (학습 필요) |

### 팀 구성

- 에이전트 이름에 `_0`, `_1` 등의 suffix가 붙으면 같은 prefix끼리 한 팀
- 예: `aggressive_teacher_agent_0`, `aggressive_teacher_agent_1` = 한 팀
- 팀원끼리는 서로 공격하지 않음

---

## 8. PPO 에이전트 커스터마이징

### 모델 구조 변경

`agent_code/ppo_agent/models/vit.py` 또는 새 모델 파일을 수정하세요.

### 보상 함수 조정

`agent_code/ppo_agent/train.py`의 `game_events_occurred()` 함수에서 보상을 커스터마이징:

```python
# 예시: 킬 보상 증가
if e.KILLED_OPPONENT in events:
    reward += 10  # 기본값 5에서 증가
```

### 하이퍼파라미터

`agent_code/ppo_agent/train.py`의 `setup_training()`:
- `learning_rate`
- `gamma` (할인율)
- `gae_lambda`
- `clip_epsilon`

---

## 9. 노트

### 모델 저장
- PPO 에이전트는 각 라운드 종료 시 `agent_code/ppo_agent/ppo_model.pt`에 자동 저장
- 학습 재개 시 자동으로 로드

### 로그
- 각 에이전트별 로그: `logs/<agent_name>.log`
- 디버그 정보 확인 가능

### 리플레이 저장

```bash
# 리플레이 저장
python main.py play --agents ppo_agent aggressive_teacher_agent --save-replay --n-rounds 1

# 리플레이 재생
python main.py replay <replay_file.pt>
```

---

## 10. 추천 학습 파이프라인

**초보자:**
```bash
# 1. 기본 학습 (200 라운드)
python main.py play --agents ppo_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --no-gui --n-rounds 200 --save-stats results/step1.json

# 2. 결과 확인
python plot_results.py results/step1.json
```

**중급자:**
```bash
# 1. Progressive 학습 (1000 라운드)
python main.py play --agents ppo_agent ppo_agent --train 2 --progressive-opponents --n-rounds 1000 --no-gui --save-stats results/step2_all.json

# 2. 시각화
python plot_results.py results/step2_all.json --rolling 50

# 3. 매치업 분석
python analyze_matchups.py results/step2_all.json
```

**고급자:**
```bash
# 1. 교사 모델 기반 집중 학습 (500 라운드)
python main.py play --agents ppo_agent ppo_agent aggressive_teacher_agent aggressive_teacher_agent --train 2 --no-gui --n-rounds 500 --save-stats results/step3.json

# 2. 평가 (다양한 상대)
python main.py play --agents ppo_agent ppo_agent rule_based_agent aggressive_teacher_agent --no-gui --n-rounds 100 --save-stats results/eval.json

# 3. 분석
python plot_results.py results/step3.json
python plot_results.py results/eval.json
```

---

## Troubleshooting

**Q: 학습이 너무 느려요**
- `--no-gui` 옵션 사용
- `--skip-frames` 옵션 추가 (GUI 사용 시)
- 라운드 수 줄이기

**Q: PPO 점수가 안 오르는 것 같아요**
- 더 많은 라운드 학습 (1000+)
- 보상 함수 조정 (`train.py`)
- Progressive/Curriculum 학습 사용
- 학습률 조정

**Q: 메모리 부족 오류**
- 한 번에 학습하는 에이전트 수 줄이기 (`--train` 값 감소)
- 배치 사이즈 조정

**Q: 교사 모델이 너무 강해요**
- 처음엔 `rule_based_agent`로 시작
- Progressive 모드로 점진적 난이도 증가
- Curriculum 학습 활용

---

## License & Credits

Bomberman RL framework + Aggressive Teacher Agent implementation
