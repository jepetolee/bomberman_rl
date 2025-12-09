# Bomberman RL 빠른 시작 가이드

## 🚀 5분 만에 시작하기

### 1. 교사 모델 테스트

```bash
# 자동 학습 스크립트 실행
python3 example_train.py --mode teacher

# 또는 직접 명령어 실행
python3 main.py play \
  --agents aggressive_teacher_agent aggressive_teacher_agent rule_based_agent rule_based_agent \
  --n-rounds 30 --no-gui --save-stats results/teacher_test.json

# 결과 시각화
python3 plot_results.py results/teacher_test.json
```

생성되는 차트:
- `teacher_test_scores.png` - 에이전트별 총 점수
- `teacher_test_agent_stats.png` - 코인/킬/자살 통계
- `teacher_test_round_steps.png` - 라운드당 스텝 수

### 2. 기본 PPO 학습

```bash
# 자동 학습 (200 라운드)
python3 example_train.py --mode basic

# 결과 확인
ls -lh results/example_basic_*.png
```

### 3. 고급 PPO 학습 (교사 모델 상대)

```bash
# 300 라운드 학습
python3 example_train.py --mode advanced

# 차트 확인
eog results/example_advanced_*.png  # Linux
# 또는 파일 탐색기에서 results/ 폴더 열기
```

### 4. 최고 성능 학습 (Progressive 모드)

```bash
# 500 라운드, 점진적 난이도 증가
python3 example_train.py --mode progressive

# 매치업 분석 결과 확인
# "상대별 승/무/패 통계" 출력됨
```

### 5. 학습된 모델 평가

```bash
python3 example_train.py --mode evaluate
```

---

## 📊 차트 해석 방법

### Scores (점수)
- 높을수록 좋음
- PPO가 rule_based보다 높으면 학습 성공

### Agent Stats (통계)
- **Coins** (초록색): 코인 수집 개수
- **Kills** (주황색): 적 처치 개수 ⭐
- **Suicides** (빨간색): 자살 횟수 (낮을수록 좋음)

### Combined Score (누적 점수)
- 시간에 따른 점수 추이
- 우상향하면 학습 중 ✅
- 평평하면 더 학습 필요

---

## 🎮 GUI로 직접 플레이

```bash
# PPO vs 교사 모델 (GUI)
python3 main.py play --agents ppo_agent aggressive_teacher_agent

# 속도 조절
python3 main.py play --agents ppo_agent aggressive_teacher_agent --update-interval 0.3

# 턴제 모드 (키보드로 한 스텝씩)
python3 main.py play --agents ppo_agent aggressive_teacher_agent --turn-based
```

**조작법:**
- `↑↓←→`: 이동
- `Space`: 폭탄 설치
- `Enter`: 대기
- `Q` or `ESC`: 라운드 건너뛰기

---

## 📈 학습 전략 추천

### 초보자
1. `python3 example_train.py --mode basic` (200 라운드)
2. 차트 확인
3. GUI로 결과 확인: `python3 main.py play --agents ppo_agent rule_based_agent`

### 중급자
1. `python3 example_train.py --mode advanced` (300 라운드)
2. 추가 학습: 위 명령어 여러 번 실행
3. 평가: `python3 example_train.py --mode evaluate`

### 고급자
1. `python3 example_train.py --mode progressive` (500 라운드)
2. 매치업 분석 결과 검토
3. 커스텀 보상 함수 조정 (`agent_code/ppo_agent/train.py`)
4. 추가 학습 반복

---

## 🔧 커스터마이징

### 보상 함수 수정

`agent_code/ppo_agent/train.py` 파일 열기:

```python
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # 킬 보상 증가
    if e.KILLED_OPPONENT in events:
        reward += 20  # 기본값 5 -> 20으로 증가
    
    # 코인 보상 감소 (공격 중심 학습)
    if e.COIN_COLLECTED in events:
        reward += 0.5  # 기본값 1 -> 0.5로 감소
```

### 학습률 조정

`agent_code/ppo_agent/train.py` 파일의 `setup_training()`:

```python
self.optimizer = optim.Adam(
    self.model.parameters(), 
    lr=0.0001  # 기본값 0.0003에서 감소 (더 안정적)
)
```

---

## 🎯 성능 목표

학습이 잘 되고 있는지 확인하는 기준:

### 200 라운드 후 (vs rule_based)
- ✅ Score > 30
- ✅ Kills > 5
- ✅ Suicides < 10

### 500 라운드 후 (progressive)
- ✅ Score > 80
- ✅ Kills > 15
- ✅ Suicides < 15
- ✅ Win rate > 40% (vs rule_based)

### 1000 라운드 후 (vs teacher)
- ✅ Score > 100
- ✅ Kills > 20
- ✅ Suicides < 20
- ✅ Win rate > 30% (vs teacher)

---

## 🐛 문제 해결

### "학습이 안 되는 것 같아요"
```bash
# 더 많은 라운드 학습
python3 main.py play --agents ppo_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 --no-gui --n-rounds 1000 --save-stats results/long_train.json

# 진행 상황 확인
python3 plot_results.py results/long_train.json --rolling 100
```

### "너무 느려요"
```bash
# --no-gui 반드시 사용
# 라운드 수를 줄여서 테스트
python3 main.py play --agents ppo_agent rule_based_agent \
  --train 1 --no-gui --n-rounds 50
```

### "교사 모델이 너무 강해요"
```bash
# progressive 모드 사용 (점진적 난이도)
python3 example_train.py --mode progressive
```

### "차트가 안 보여요"
```bash
# matplotlib 설치 확인
pip3 install matplotlib

# 차트 재생성
python3 plot_results.py results/ppo_basic.json
```

---

## 📁 결과 파일 구조

```
bomberman_rl/
├── agent_code/
│   ├── ppo_agent/
│   │   └── ppo_model.pt          ← 학습된 모델 (자동 저장)
│   └── aggressive_teacher_agent/
│       └── callbacks.py           ← 교사 모델 코드
├── results/
│   ├── example_basic.json         ← 학습 결과 데이터
│   ├── example_basic_scores.png   ← 점수 차트
│   ├── example_basic_agent_stats.png
│   └── ...
└── logs/
    ├── ppo_agent.log              ← 디버그 로그
    └── aggressive_teacher_agent.log
```

---

## 🚀 전체 파이프라인 실행

완전 자동 학습 (약 30분 소요):

```bash
# 모든 단계 자동 실행
./quick_train.sh

# 또는 Python 버전
python3 example_train.py --mode basic
python3 example_train.py --mode advanced  
python3 example_train.py --mode progressive
python3 example_train.py --mode evaluate
```

---

## 💡 팁

1. **정기적으로 차트 확인** - 100 라운드마다 `plot_results.py` 실행
2. **모델 백업** - `ppo_model.pt` 파일 주기적으로 복사
3. **여러 전략 실험** - 보상 함수 다르게 해서 여러 모델 학습
4. **팀 구성** - 2개 PPO 에이전트가 협력하게 하면 더 강함
5. **로그 확인** - 문제 발생 시 `logs/ppo_agent.log` 확인

---

## 📚 더 배우기

- `README.md` - 전체 문서
- `agent_code/ppo_agent/train.py` - 학습 로직
- `agent_code/ppo_agent/models/` - 네트워크 구조
- `agent_code/aggressive_teacher_agent/callbacks.py` - A* 알고리즘, 전략

---

## 🎓 고급 기능

### 분산 학습 (Multi-GPU)
```bash
torchrun --nproc_per_node 4 test.py
```

### 리플레이 저장 및 재생
```bash
# 저장
python3 main.py play --agents ppo_agent aggressive_teacher_agent \
  --save-replay --n-rounds 1

# 재생
python3 main.py replay replays/<filename>.pt
```

### 커리큘럼 학습
```bash
python3 main.py play --agents ppo_agent ppo_agent ppo_agent ppo_agent \
  --train 4 --curriculum \
  --phase1-rounds 200 --phase1-opponent random_agent \
  --phase2-rounds 300 --phase2-opponent aggressive_teacher_agent \
  --no-gui
```

---

**즐거운 학습 되세요!** 🎮🤖

