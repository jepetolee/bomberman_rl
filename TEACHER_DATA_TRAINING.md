# Teacher Data 학습 가이드

## 개요

이 문서는 Teacher Data를 수집하고 학습하는 전체 과정을 설명합니다.
3단계 파이프라인: Teacher Data Collection → Dyna-Q Planning + DeepSupervision → Recurrent RL

---

## Phase 1: Teacher Data Collection

### 목적

Teacher agent (`aggressive_teacher_agent`)의 실제 행동으로부터 20,000 episode의 데이터를 수집합니다.
이 데이터는 Phase 2에서 환경 모델 학습과 Policy 학습에 사용됩니다.

### 실행 방법

```bash
python collect_teacher_data.py --episodes 20000 --workers 4
```

### 수집 과정

1. **멀티 워커 실행**
   - 여러 프로세스가 병렬로 게임 실행
   - 각 워커는 독립적으로 episode 수집

2. **Episode 구조**
   - Teacher agent와 다양한 opponent 조합으로 대전
   - Opponents: `random_agent`, `peaceful_agent`, `coin_collector_agent`, `rule_based_agent`, `team_teacher_agent`

3. **데이터 저장 형식**
   ```python
   EpisodeData = {
       'states': [T, C, H, W],      # State sequence (9 channels, 17x17)
       'teacher_actions': [T],      # Teacher가 선택한 action indices
       'rewards': [T],              # Reward sequence
       'dones': [T],                # Terminal flags
       'game_states': [T, ...],     # Full game state dicts (환경 모델 학습용)
   }
   ```

4. **저장 위치**
   - `data/teacher_episodes/episode_XXXXXX.pt` (각 episode별 파일)
   - 총 20,000개 episode 목표

### 데이터 필터링

- 최소 episode 길이: 10 steps (환경변수로 조정 가능)
- 짧은 episode는 제외하여 학습 품질 향상

---

## Phase 2: Dyna-Q Planning + DeepSupervision

Phase 2는 두 단계로 구성됩니다:

### 2-1. 환경 모델 학습 (Environment Model Training)

**목적**: 수집된 데이터로 환경이 어떻게 동작하는지 학습

**학습 과정**:

1. **데이터 로드**
   ```python
   # 모든 episode를 메모리로 로드
   episodes = load_all_episodes('data/teacher_episodes')
   # (state, action) → (next_state, reward) transition 추출
   states, actions, rewards, next_states = extract_transitions(episodes)
   ```

2. **모델 구조**
   - 입력: `(state [B,9,17,17], action [B])`
   - 출력: `(next_state_pred [B,9,17,17], reward_pred [B])`
   - CNN 기반 encoder-decoder 구조
   - State transition predictor + Reward predictor

3. **학습 루프**
   ```python
   for epoch in range(num_epochs):
       for batch_states, batch_actions, batch_rewards, batch_next_states in dataloader:
           # Forward
           next_state_pred, reward_pred = env_model(batch_states, batch_actions)
           
           # Loss
           state_loss = MSE(next_state_pred, batch_next_states)
           reward_loss = MSE(reward_pred, batch_rewards)
           loss = state_loss + reward_loss
           
           # Backward
           loss.backward()
           optimizer.step()
   ```

4. **실행 명령**
   ```bash
   python train_phase2.py --train-env-model --num-epochs 50
   ```

5. **저장 위치**
   - `data/env_models/env_model.pt`

---

### 2-2. Dyna-Q Planning + DeepSupervision

**목적**: 환경 모델을 사용하여 가상 경험을 생성하고, DeepSupervision으로 Policy 학습

#### Dyna-Q Planning 과정

1. **Visited States Buffer 구축**
   ```python
   # 실제 episode에서 방문한 (state, action) 쌍 저장
   visited_states = VisitedStatesBuffer(max_size=10000)
   for episode in episodes:
       for state, action in zip(episode['states'], episode['teacher_actions']):
           visited_states.add(state, action)
   ```

2. **Planning: 가상 경험 생성**
   ```python
   planner = DynaPlanner(env_model, visited_states)
   
   for _ in range(n_planning_steps=100):
       # 1. 방문했던 상태 중 하나를 샘플링
       s_sim = visited_states.sample_state()
       
       # 2. 해당 상태에서 수행했던 행동 중 하나를 샘플링
       a_sim = visited_states.sample_action_for_state(s_sim)
       
       # 3. 환경 모델로 예측
       s_next_pred, r_pred = env_model(s_sim, a_sim)
       
       # 4. 가상 경험 저장
       simulated_experiences.append((s_sim, a_sim, r_pred, s_next_pred))
   ```

#### DeepSupervision 학습

**핵심 아이디어**: 같은 state를 N_sup번 반복하여 Policy가 점진적으로 개선되도록 학습

1. **TRM Recursive Reasoning**
   ```python
   # 같은 state에 대해 반복적으로 추론
   z = torch.zeros(B, z_dim)  # Initial latent state
   x_embed = patch_embed(state)  # Patch embeddings
   
   for sup_step in range(n_sup=16):
       # Recursive reasoning: z를 개선
       z = trm_net(x_embed, z)
       
       if sup_step == n_sup - 1:
           # 마지막 step에서만 loss 계산 (논문 방식)
           logits = policy_head(z)
           loss = cross_entropy(logits, teacher_action)
   ```

2. **학습 전략**
   - **"last" 전략**: 마지막 supervision step에서만 loss 계산
   - **"all" 전략**: 모든 step에서 loss 계산 후 평균
   - **"weighted" 전략**: 각 step에 가중치를 주어 평균 (나중 step에 더 높은 weight)

3. **Real + Simulated Experiences**
   ```python
   # 실제 경험과 가상 경험을 모두 사용
   real_loss = train_policy_deepsup(model, real_states, real_actions, ...)
   sim_loss = train_policy_deepsup(model, sim_states, sim_actions, ...)
   
   # 가상 경험은 더 낮은 weight로 학습
   total_loss = real_loss * 1.0 + sim_loss * 0.5
   ```

4. **Value Network 제외**
   ```python
   # Policy Network만 학습 (Value Network는 gradient 비활성화)
   for param in model.v_head.parameters():
       param.requires_grad = False
   
   for param in model.pi_head.parameters():
       param.requires_grad = True
   ```

5. **실행 명령**
   ```bash
   python train_phase2.py --train-policy --use-planning \
       --planning-steps 100 --n-sup 16
   ```

6. **저장 위치**
   - `data/policy_models/policy_phase2.pt`

---

## Phase 3: Recurrent RL

**목적**: Value Network가 학습되면, recurrent 환경에서 이전 timestep의 latent z를 활용한 강화학습

### Recurrent z 활용

1. **Latent State 전파**
   ```python
   # Timestep t
   z_prev = None  # 첫 timestep은 zeros
   
   # Forward
   logits, value, z_new = model.forward_with_z(obs_t, z_prev=z_prev)
   
   # 다음 timestep에 z_new 사용
   z_prev = z_new  # Timestep t+1에서 사용
   ```

2. **Round 시작 시 리셋**
   ```python
   # Round가 시작되면 z를 None으로 리셋
   if current_round != last_round:
       z_prev = None  # 초기화
   ```

3. **실행 명령**
   ```bash
   bash train_phase3.sh [workers] [rounds] [phase2_model_path]
   ```

4. **환경변수 설정**
   ```bash
   export BOMBER_USE_TRM=1
   export BOMBER_TRM_RECURRENT=1
   export BOMBER_TRM_N=6
   export BOMBER_TRM_T=3
   export BOMBER_TRM_N_SUP=16
   ```

---

## 학습 파이프라인 요약

```
Phase 1: Teacher Data Collection
  ↓
  [20,000 episodes collected]
  ↓
Phase 2-1: Environment Model Training
  ↓
  [env_model.pt saved]
  ↓
Phase 2-2: Dyna-Q Planning + DeepSupervision
  ↓
  [policy_phase2.pt saved]
  ↓
Phase 3: Recurrent RL
  ↓
  [ppo_model_phase3.pt - final model]
```

---

## 주요 하이퍼파라미터

### Patch Configuration
- `BOMBER_TRM_PATCH_SIZE`: Patch 크기 (기본값: 1)
- `BOMBER_TRM_PATCH_STRIDE`: Patch stride (기본값: patch_size)
  - `stride < patch_size`: Overlapping patches (더 많은 정보)
  - `stride = patch_size`: Non-overlapping patches

### TRM Configuration
- `BOMBER_TRM_N`: Latent recursion steps (기본값: 6)
- `BOMBER_TRM_T`: Deep recursion steps (기본값: 3)
- `BOMBER_TRM_N_SUP`: Supervision steps (기본값: 16)
- `BOMBER_TRM_Z_DIM`: Latent dimension (기본값: embed_dim)

### Phase 2 Configuration
- `--planning-steps`: Planning step 수 (기본값: 100)
- `--n-sup`: DeepSupervision step 수 (기본값: 16)
- `--num-epochs`: Training epochs (기본값: 100)

---

## 예제: 전체 파이프라인 실행

```bash
# 1. Teacher 데이터 수집
python collect_teacher_data.py --episodes 20000 --workers 4

# 2. 환경 모델 학습
python train_phase2.py --train-env-model --num-epochs 50

# 3. Policy 학습 (Planning + DeepSupervision)
python train_phase2.py --train-policy --planning-steps 100 --n-sup 16

# 4. Recurrent RL 학습
bash train_phase3.sh 4 2000 data/policy_models/policy_phase2.pt
```

---

## 참고사항

1. **데이터 품질**: Teacher agent의 행동이 최적이 아닐 수 있으므로, 다양한 opponent와 대전하여 다양성 확보

2. **환경 모델 정확도**: 모델이 부정확하면 Planning이 해로울 수 있음 → 모델 정확도 모니터링 필요

3. **DeepSupervision 비용**: N_sup × (n+1) forward passes 필요 → 계산 비용 고려

4. **Phase 전환**: Phase 2에서 Phase 3로 전환 시점은 Value network 성능 기준으로 결정

