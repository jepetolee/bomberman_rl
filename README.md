# Bomberman RL

Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Quick Start

### 1. 교사 모델 테스트 (Aggressive Teacher Agent)

새로 구현된 공격적 교사 모델을 테스트해 보세요:

```bash
# 교사 모델 vs rule_based (GUI 없이)
python main.py play --agents aggressive_teacher_agent aggressive_teacher_agent rule_based_agent rule_based_agent --n-rounds 30 --no-gui --save-stats results/teacher_test.json

# 결과 시각화
python plot_results.py results/teacher_test.json
```

**교사 모델 특징:**
- ✅ A* 경로 탐색으로 적극적인 적 추적
- ✅ 팀 협동 (2개 에이전트가 한 팀)
- ✅ 전략적 폭탄 배치 (탈출 경로 보장)
- ✅ 코너 감지 및 적 가두기
- ✅ 팀원 보호 (아군은 절대 공격하지 않음)

**성능 (50 라운드 기준):**
- Kills: 23 vs 10 (rule_based 대비 **2.3배**)
- Suicides: 11 vs 77 (rule_based 대비 **7배 더 안전**)

---

## 2. PPO 에이전트 학습

### 기본 학습 (Rule-based 상대)

```bash
# 단일 PPO 에이전트 학습 (GUI 없이, 200 라운드)
python main.py play --agents ppo_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --no-gui --n-rounds 200 --save-stats results/ppo_basic.json
```

### 교사 모델 기반 학습 (권장)

공격적인 교사 모델을 상대로 학습하면 더 강한 에이전트를 만들 수 있습니다:

```bash
# PPO 에이전트 vs 교사 모델 (2명)
python main.py play \
  --agents ppo_agent ppo_agent aggressive_teacher_agent aggressive_teacher_agent \
  --train 2 \
  --no-gui \
  --n-rounds 500 \
  --save-stats results/ppo_vs_teacher.json
```

### 커리큘럼 학습

점진적으로 난이도를 높이는 학습:

```bash
# Phase 1: random 상대 (200 라운드)
# Phase 2: rule_based 상대 (300 라운드)
python main.py play \
  --agents ppo_agent ppo_agent ppo_agent ppo_agent \
  --train 4 \
  --curriculum \
  --phase1-rounds 200 \
  --phase2-rounds 300 \
  --phase1-opponent random_agent \
  --phase2-opponent aggressive_teacher_agent \
  --no-gui \
  --save-stats results/ppo_curriculum.json
```

### 동적 상대 스케줄링

라운드마다 랜덤하게 상대 선택 (점차 어려워짐):

```bash
# Progressive 모드: fail -> peaceful -> random -> coin -> rule로 점진적 전환
python main.py play \
  --agents ppo_agent ppo_agent \
  --train 2 \
  --progressive-opponents \
  --no-fail \
  --n-rounds 1000 \
  --no-gui \
  --save-stats results/ppo_progressive_all.json
```

```bash
# Dynamic 모드: rule_based 확률을 5% -> 60%로 점진적 증가
python main.py play \
  --agents ppo_agent ppo_agent \
  --train 2 \
  --dynamic-opponents \
  --opponent-pool random_agent coin_collector_agent peaceful_agent \
  --rb-agent aggressive_teacher_agent \
  --rb-prob-start 0.05 \
  --rb-prob-end 0.60 \
  --n-rounds 1000 \
  --no-gui \
  --save-stats results/ppo_dynamic_all.json
```

---

## 3. 결과 분석 및 시각화

### 차트 생성

```bash
# 기본 차트 (점수, 통계, 라운드별 진행)
python plot_results.py results/ppo_vs_teacher.json

# 생성되는 차트:
# - ppo_vs_teacher_scores.png         : 에이전트별 총 점수
# - ppo_vs_teacher_agent_stats.png    : 코인/킬/자살 통계
# - ppo_vs_teacher_round_steps.png    : 라운드당 스텝 수
# - ppo_vs_teacher_combined_score.png : 라운드별 누적 점수 (PPO만)
# - ppo_vs_teacher_schedule.png       : 상대 스케줄 (dynamic/progressive 모드)
```

### 커스텀 시각화

```bash
# PPO 에이전트만 포함, 50 라운드 rolling average
python plot_results.py results/ppo_progressive_all.json \
  --include ppo_agent \
  --rolling 50
```

### 대결 분석

```bash
# 상대별 승/무/패 통계
python analyze_matchups.py results/ppo_dynamic_all.json
```

출력 예시:
```
Opponents                                      W      D      L   Win%  Rounds
random_agent, random_agent                    45      2      3   90.0      50
coin_collector_agent, peaceful_agent          38      5      7   76.0      50
aggressive_teacher_agent, rule_based_agent    15     10     25   30.0      50
```

---

## 4. Self-Play 학습 (A3C + AlphaZero 스타일)

### 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-PLAY TRAINING SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Teacher Training                                      │
│  ┌─────────────┐         ┌─────────────────────┐               │
│  │  PPO Agent  │ ──vs──▶ │ aggressive_teacher  │               │
│  └─────────────┘         └─────────────────────┘               │
│       │                                                         │
│       │ Win rate >= 25%                                         │
│       ▼                                                         │
│  Phase 2: Self-Play (AlphaZero Style)                          │
│  ┌─────────────┐         ┌─────────────────────┐               │
│  │  PPO Agent  │ ──vs──▶ │ Past PPO Versions   │               │
│  │  (Current)  │         │ (Checkpoints)       │               │
│  └─────────────┘         └─────────────────────┘               │
│       │                                                         │
│       │ Every 200 rounds                                        │
│       ▼                                                         │
│  Save new generation checkpoint                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 빠른 실행

```bash
# 빠른 테스트 (약 30분)
./run_selfplay.sh --quick

# 표준 학습 (약 2시간)
./run_selfplay.sh --standard

# 장기 학습 (약 5시간)
./run_selfplay.sh --long
```

### 커스텀 설정

```bash
python3 self_play_train.py \
    --teacher-rounds 1000 \
    --selfplay-rounds 2000 \
    --win-threshold 0.25
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--teacher-rounds` | 1000 | Phase 1 학습 라운드 수 |
| `--selfplay-rounds` | 2000 | Phase 2 학습 라운드 수 |
| `--win-threshold` | 0.25 | Phase 2로 넘어가는 승률 기준 |
| `--skip-teacher` | - | Phase 1 건너뛰기 |
| `--eval-only` | - | 평가만 실행 |

### 결과 확인

```bash
# 결과 디렉토리 확인
ls -la results/self_play_*/

# 차트 확인
python3 plot_results.py results/self_play_*/all_results.json

# 체크포인트 확인
ls -la agent_code/ppo_agent/checkpoints/
```

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
