# A3C 학습 방식 설명

## 📚 A3C (Asynchronous Advantage Actor-Critic) 개요

A3C는 **비동기적으로 여러 worker가 동시에 환경을 탐험**하며, **공유 모델을 업데이트**하는 분산 강화학습 알고리즘입니다.

---

## 🏗️ 현재 프로젝트의 A3C 구조

### 1. 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│         GPU: Shared PPO Model (share_memory())           │
│         모든 worker가 동시에 접근                        │
└──────────────┬──────────────────┬───────────────────────┘
               │                  │
        ┌──────┴──────┐    ┌──────┴──────┐
        │  Worker 0   │    │  Worker N   │
        │  (CPU Env)  │    │  (CPU Env)  │
        └─────────────┘    └─────────────┘
```

### 2. 핵심 구성 요소

#### **Shared Model (공유 모델)**
```python
# a3c_gpu_train.py
shared_model = create_shared_model(config['model_path'], device)
shared_model.share_memory()  # ← 모든 프로세스가 같은 GPU 메모리 공유
```

- **1개의 PPO 모델**이 GPU에 저장
- `share_memory()`로 모든 worker 프로세스가 동시 접근
- 모든 worker가 **같은 모델 파라미터**를 사용

#### **Worker 프로세스**
```python
# 각 worker는 독립적으로 게임 실행
for i in range(num_workers):
    Process(target=worker_loop, args=(...))
```

- **병렬로 게임 실행** (각 worker가 독립적인 환경)
- 게임 결과를 수집하여 메인 프로세스로 전송
- **비동기적으로** 경험 수집

---

## 🔄 학습 프로세스

### Step 1: 경험 수집 (Experience Collection)

```
Worker 0: [게임 실행] → 경험 수집 → 결과 전송
Worker 1: [게임 실행] → 경험 수집 → 결과 전송
Worker 2: [게임 실행] → 경험 수집 → 결과 전송
Worker 3: [게임 실행] → 경험 수집 → 결과 전송
```

각 worker는:
1. **현재 공유 모델**을 사용하여 행동 선택
2. 게임을 실행 (예: 5라운드)
3. 결과를 메인 프로세스로 전송

### Step 2: PPO 업데이트 (Policy Update)

```python
# agent_code/ppo_agent/train.py

def end_of_round(...):
    # 모든 PPO 에이전트가 라운드 종료 시
    if count >= len(SHARED.instance_ids):
        _ppo_update()  # ← PPO 업데이트 실행
```

**PPO 업데이트 과정:**

1. **GAE (Generalized Advantage Estimation) 계산**
   ```python
   adv, returns = _compute_gae(rewards, values, dones, ...)
   # Advantage = 실제 리턴 - 예측 가치
   # Returns = Advantage + 예측 가치
   ```

2. **Policy Loss 계산**
   ```python
   ratio = exp(new_log_prob - old_log_prob)
   surr1 = ratio * advantage
   surr2 = clamp(ratio, 0.8, 1.2) * advantage
   policy_loss = -min(surr1, surr2)  # PPO Clipping
   ```

3. **Value Loss 계산**
   ```python
   value_loss = MSE(predicted_value, actual_return)
   ```

4. **Entropy Loss (탐험 장려)**
   ```python
   entropy = -sum(p * log(p))  # 행동 분포의 엔트로피
   ```

5. **최종 Loss**
   ```python
   loss = policy_loss + 0.5 * value_loss - 0.1 * entropy
   ```

6. **Gradient 업데이트**
   ```python
   loss.backward()
   clip_grad_norm_(parameters, 0.5)  # Gradient Clipping
   optimizer.step()
   ```

### Step 3: 모델 저장 및 Worker 동기화

```python
# a3c_gpu_train.py
if perf_tracker.is_best(current_round):
    torch.save(shared_model.state_dict(), 'ppo_model.pt')
    # ★ 최고 성능일 때만 저장
```

**중요: Worker가 새 모델을 불러오는 방식**

현재 구현은 `subprocess`를 사용하므로, 각 worker는 **독립적인 프로세스**입니다.

```python
# agent_code/ppo_agent/callbacks.py
def ensure_policy(...):
    # 모델 파일의 수정 시간을 추적
    if os.path.isfile(self.model_path):
        current_mtime = os.path.getmtime(self.model_path)
        if current_mtime > self._last_model_mtime:
            # 모델이 업데이트되었으면 다시 로드!
            state = torch.load(self.model_path, ...)
            self.policy.load_state_dict(state)
            self._last_model_mtime = current_mtime
```

**동작 방식:**
1. Main 프로세스가 모델 업데이트 → `ppo_model.pt` 파일 저장
2. 각 worker의 subprocess가 새 게임 시작
3. `setup()` 호출 시 `ensure_policy()` 실행
4. 모델 파일의 수정 시간 확인
5. 업데이트되었으면 **자동으로 새 모델 로드** ✅

**결과:** 각 worker가 새로운 에피소드 실행 전에 최신 모델을 사용!

---

## 🎯 현재 구현의 특징

### ✅ A3C 스타일
- **병렬 worker**: 여러 프로세스가 동시에 게임 실행
- **공유 모델**: GPU 메모리 공유로 효율적
- **비동기 업데이트**: 각 worker가 독립적으로 경험 수집

### ⚠️ 전통적 A3C와의 차이

**전통적 A3C:**
```
Worker 0: 경험 수집 → Gradient 계산 → Global 모델 업데이트
Worker 1: 경험 수집 → Gradient 계산 → Global 모델 업데이트
(각 worker가 독립적으로 업데이트)
```

**현재 구현 (PPO + A3C 스타일):**
```
Worker 0: 경험 수집 → 결과 전송
Worker 1: 경험 수집 → 결과 전송
Main: 모든 경험 수집 → PPO 업데이트 (중앙 집중식)
```

**장점:**
- PPO의 안정성 유지 (중앙 집중식 업데이트)
- A3C의 병렬화 이점 활용 (빠른 경험 수집)

---

## 📊 학습 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│  Main Process                                           │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Shared PPO Model (GPU)                          │ │
│  │  - share_memory()                                 │ │
│  │  - 모든 worker가 접근                             │ │
│  └───────────────────────────────────────────────────┘ │
│                                                        │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Performance Tracker                              │ │
│  │  - 최근 200라운드 성능 추적                        │ │
│  │  - Best model만 저장                              │ │
│  └───────────────────────────────────────────────────┘ │
│                                                        │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Adaptive Curriculum                              │ │
│  │  - 승률 기반 스테이지 진행                         │ │
│  │  - Self-Play 진입                                 │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │ Worker 0│         │ Worker 1│   ...   │ Worker N│
    │         │         │         │         │         │
    │ 1. 게임 실행      │ 1. 게임 실행      │ 1. 게임 실행
    │ 2. 경험 수집      │ 2. 경험 수집      │ 2. 경험 수집
    │ 3. 결과 전송      │ 3. 결과 전송      │ 3. 결과 전송
    └─────────┘         └─────────┘         └─────────┘
```

---

## 🔑 핵심 개념

### 1. **Advantage (어드밴티지)**
```
Advantage = 실제 리턴 - 예측 가치

예:
  - 예측 가치: 5.0
  - 실제 리턴: 10.0
  - Advantage: +5.0 (좋은 행동!)
  
  - 예측 가치: 5.0
  - 실제 리턴: 2.0
  - Advantage: -3.0 (나쁜 행동)
```

### 2. **PPO Clipping**
```python
ratio = new_prob / old_prob

if ratio > 1.2:  # 너무 많이 변하면
    use_clipped = 1.2 * advantage  # 제한
else:
    use_actual = ratio * advantage
```

**목적:** 정책이 너무 급격히 변하지 않도록 제한

### 3. **GAE (Generalized Advantage Estimation)**
```python
# 시간 t에서의 Advantage
δ_t = reward_t + γ * V(s_{t+1}) - V(s_t)

# GAE는 여러 스텝의 δ를 조합
GAE_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
```

**장점:** 낮은 분산, 높은 편향의 균형

---

## 🎮 실제 학습 예시

### Round 1-5 (Worker 0)
```
PPO Agent: [상태1] → 행동1 → 보상+1
PPO Agent: [상태2] → 행동2 → 보상+5 (킬!)
PPO Agent: [상태3] → 행동3 → 보상-3 (죽음)
...
→ 경험 버퍼에 저장
```

### Round 6-10 (Worker 1)
```
PPO Agent: [상태1] → 행동4 → 보상+2
...
→ 경험 버퍼에 저장
```

### 업데이트 시점
```
모든 worker가 라운드 완료
→ PPO 업데이트 실행
→ 공유 모델 업데이트
→ 모든 worker가 새 모델 사용
```

---

## 📈 성능 향상 요소

### 1. **병렬화**
- 4개 worker = **4배 빠른 경험 수집**
- GPU 공유로 메모리 효율적

### 2. **커리큘럼 학습**
- 쉬운 상대 → 어려운 상대
- 승률 기반 자동 진행

### 3. **Self-Play**
- 같은 모델끼리 경쟁
- 무한 진화 가능

---

## 🔧 하이퍼파라미터

```python
# agent_code/ppo_agent/train.py

gamma = 0.99              # 할인율
gae_lambda = 0.95         # GAE 파라미터
clip_range = 0.2          # PPO Clipping 범위
ent_coef = 0.1            # 엔트로피 계수 (탐험)
vf_coef = 0.5             # Value Loss 가중치
max_grad_norm = 0.5       # Gradient Clipping
update_epochs = 4         # 업데이트 반복 횟수
batch_size = 256          # 배치 크기
learning_rate = 3e-4      # 학습률
```

---

## 💡 요약

1. **병렬 경험 수집**: 여러 worker가 동시에 게임 실행
2. **공유 모델**: GPU 메모리 공유로 효율적
3. **PPO 업데이트**: 중앙 집중식으로 안정적 학습
4. **커리큘럼 학습**: 승률 기반 자동 진행
5. **Self-Play**: 같은 모델끼리 경쟁하며 진화

**결과:** 빠르고 안정적인 강화학습! 🚀

