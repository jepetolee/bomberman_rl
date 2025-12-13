# 교사 데이터셋 로딩 및 훈련 코드 명세

## 개요

이 문서는 교사(Teacher) 데이터셋을 불러오고 훈련하는 전체 과정의 코드 동작을 상세히 설명합니다.

---

## 1. 데이터 수집 (Phase 1)

### 1.1 데이터 구조

```python
# collect_teacher_data.py

class EpisodeData:
    """단일 episode의 데이터 구조"""
    def __init__(self):
        self.states = []           # List of np.ndarray [C, H, W]
        self.teacher_actions = []  # List of int (action indices)
        self.rewards = []          # List of float
        self.dones = []            # List of bool
        self.game_states = []      # List of dict (full game state)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (저장용)"""
        return {
            'states': np.array(self.states),           # [T, C, H, W]
            'teacher_actions': np.array(self.teacher_actions),  # [T]
            'rewards': np.array(self.rewards),         # [T]
            'dones': np.array(self.dones),             # [T]
            'game_states': self.game_states,           # List[dict]
        }
```

### 1.2 데이터 저장

```python
# collect_teacher_data.py - worker_collect_data()

# Episode 수집 후
episode = EpisodeData()
# ... episode에 데이터 추가 ...

# 유효성 검사
if episode.is_valid():  # 최소 길이 체크
    # PyTorch pickle 형식으로 저장
    episode_file = os.path.join(data_dir, f'episode_{episode_id:06d}.pt')
    episode_dict = episode.to_dict()
    
    with open(episode_file, 'wb') as f:
        pickle.dump(episode_dict, f)
```

**저장 위치**: `data/teacher_episodes/episode_XXXXXX.pt`

**파일 구조**:
- 각 episode는 별도 `.pt` 파일로 저장
- 파일명: `episode_000000.pt`, `episode_000001.pt`, ...

---

## 2. 데이터 로딩

### 2.1 TeacherDataset 클래스

```python
# train_phase2.py

class TeacherDataset(Dataset):
    """PyTorch Dataset for teacher episodes"""
    
    def __init__(self, data_dir: str, max_episodes: int = None):
        self.data_dir = Path(data_dir)
        # 모든 episode 파일 찾기
        self.episode_files = sorted(self.data_dir.glob('episode_*.pt'))
        
        if max_episodes is not None:
            self.episode_files = self.episode_files[:max_episodes]
    
    def __len__(self):
        return len(self.episode_files)
    
    def __getitem__(self, idx):
        """단일 episode 로드"""
        with open(self.episode_files[idx], 'rb') as f:
            episode = pickle.load(f)  # Dict 형식으로 로드
        return episode
```

**동작 방식**:
1. `data_dir`에서 `episode_*.pt` 파일들을 찾음
2. 정렬하여 인덱스로 접근 가능하게 함
3. `__getitem__()` 호출 시 pickle로 로드

### 2.2 전체 Episode 로딩

```python
# train_phase2.py

def load_all_episodes(data_dir: str, max_episodes: int = None) -> List[Dict]:
    """모든 episode를 메모리로 로드"""
    dataset = TeacherDataset(data_dir, max_episodes)
    episodes = []
    
    # 모든 episode를 리스트로 로드
    for i in range(len(dataset)):
        episodes.append(dataset[i])
    
    return episodes
```

**반환 형식**: `List[Dict]`
- 각 Dict는 하나의 episode
- Dict 키: `'states'`, `'teacher_actions'`, `'rewards'`, `'dones'`, `'game_states'`

---

## 3. Transition 추출

### 3.1 extract_transitions 함수

```python
# train_phase2.py

def extract_transitions(episodes: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """
    Episode에서 (state, action, reward, next_state) transition 추출
    
    Args:
        episodes: List[Dict] - 각 Dict는 하나의 episode
    
    Returns:
        states: [N, C, H, W] - 모든 state
        actions: [N] - 모든 action
        rewards: [N] - 모든 reward
        next_states: [N, C, H, W] - 모든 next state
    """
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    
    for episode in episodes:
        states = episode['states']          # [T, C, H, W]
        actions = episode['teacher_actions']  # [T]
        rewards = episode['rewards']        # [T]
        
        # Transition 생성: (s_t, a_t, r_t, s_{t+1})
        for t in range(len(states) - 1):
            all_states.append(states[t])           # s_t
            all_actions.append(actions[t])         # a_t
            all_rewards.append(rewards[t])         # r_t
            all_next_states.append(states[t + 1])  # s_{t+1}
    
    # NumPy 배열로 변환 후 PyTorch Tensor로 변환
    states_tensor = torch.from_numpy(np.array(all_states)).float()
    actions_tensor = torch.from_numpy(np.array(all_actions)).long()
    rewards_tensor = torch.from_numpy(np.array(all_rewards)).float()
    next_states_tensor = torch.from_numpy(np.array(all_next_states)).float()
    
    return states_tensor, actions_tensor, rewards_tensor, next_states_tensor
```

**처리 과정**:
1. 각 episode를 순회
2. Episode 내 timestep t에 대해 `(s_t, a_t, r_t, s_{t+1})` transition 생성
3. 모든 episode의 transition을 하나로 합침
4. NumPy 배열 → PyTorch Tensor 변환

**결과**: 
- 총 N개의 transition (모든 episode의 총 step 수 - episode 수)
- 각 tensor의 shape:
  - `states`: [N, 9, 17, 17]
  - `actions`: [N]
  - `rewards`: [N]
  - `next_states`: [N, 9, 17, 17]

---

## 4. 환경 모델 학습 (Phase 2-1)

### 4.1 학습 함수

```python
# train_phase2.py

def train_env_model(
    data_dir: str,
    model_path: str,
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device = None,
):
    """환경 모델 학습: (state, action) → (next_state, reward)"""
    
    # 1. 데이터 로드
    episodes = load_all_episodes(data_dir)
    if len(episodes) == 0:
        raise ValueError(f"No episodes found in {data_dir}")
    
    # 2. Transition 추출
    states, actions, rewards, next_states = extract_transitions(episodes)
    print(f"Loaded {len(states)} transitions")
    
    # 3. 모델 생성
    model = EnvironmentModel(
        state_channels=9,
        state_height=17,
        state_width=17,
        num_actions=6,
        hidden_dim=256,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_state = nn.MSELoss()
    criterion_reward = nn.MSELoss()
    
    # 4. DataLoader 생성
    dataset = torch.utils.data.TensorDataset(
        states, actions, rewards, next_states
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True  # 각 epoch마다 셔플
    )
    
    # 5. 학습 루프
    model.train()
    for epoch in range(num_epochs):
        total_state_loss = 0.0
        total_reward_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_actions, batch_rewards, batch_next_states in dataloader:
            # Device로 이동
            batch_states = batch_states.to(device)         # [B, 9, 17, 17]
            batch_actions = batch_actions.to(device)       # [B]
            batch_rewards = batch_rewards.to(device)       # [B]
            batch_next_states = batch_next_states.to(device)  # [B, 9, 17, 17]
            
            # Forward: (state, action) → (next_state_pred, reward_pred)
            next_state_pred, reward_pred = model(
                batch_states, 
                batch_actions
            )
            
            # Loss 계산
            state_loss = criterion_state(next_state_pred, batch_next_states)
            reward_loss = criterion_reward(reward_pred, batch_rewards)
            loss = state_loss + reward_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 통계
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            num_batches += 1
        
        # Epoch 통계 출력
        if (epoch + 1) % 10 == 0:
            avg_state_loss = total_state_loss / num_batches
            avg_reward_loss = total_reward_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"State Loss: {avg_state_loss:.4f} | "
                  f"Reward Loss: {avg_reward_loss:.4f}")
    
    # 6. 모델 저장
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved environment model to {model_path}")
```

**학습 과정 요약**:
1. Episode 로드 → Transition 추출
2. `TensorDataset` + `DataLoader` 생성 (배치 처리)
3. 각 배치에서:
   - `(state, action)` 입력
   - `(next_state_pred, reward_pred)` 예측
   - 실제 `(next_state, reward)`와 비교하여 loss 계산
4. Gradient 업데이트
5. 모델 저장

---

## 5. Dyna-Q Planning + DeepSupervision 학습 (Phase 2-2)

### 5.1 Visited States Buffer 구축

```python
# train_phase2.py - train_policy_with_planning()

# 환경 모델 로드
env_model = create_env_model(..., model_path=env_model_path, device=device)

# Dyna planner 생성
visited_states = VisitedStatesBuffer(max_size=10000)
planner = DynaPlanner(env_model, visited_states, device=device)

# 실제 episode에서 방문한 (state, action) 쌍을 buffer에 저장
for episode in episodes:
    states = episode['states']        # [T, C, H, W]
    actions = episode['teacher_actions']  # [T]
    
    for state, action in zip(states, actions):
        state_tensor = torch.from_numpy(state).float()
        planner.add_experience(state_tensor, int(action))

print(f"Visited states buffer size: {len(visited_states)}")
```

**VisitedStatesBuffer 구조**:
- 방문했던 `(state, action)` 쌍을 저장
- Planning 단계에서 샘플링 가능

### 5.2 Planning: 가상 경험 생성

```python
# agent_code/ppo_agent/dyna_planning.py

class DynaPlanner:
    def plan(self, n_planning_steps: int = 100, batch_size: int = 32):
        """가상 경험 생성"""
        simulated_experiences = []
        
        # 배치 단위로 처리
        for step in range(0, n_planning_steps, batch_size):
            batch_states = []
            batch_actions = []
            
            # 샘플링
            for _ in range(batch_size):
                state, _ = self.visited_states.sample_state()
                action = self.visited_states.sample_action_for_state(state_hash)
                batch_states.append(state)
                batch_actions.append(action)
            
            # 배치 예측
            state_batch = torch.stack(batch_states).to(device)
            action_batch = torch.tensor(batch_actions, device=device)
            
            with torch.no_grad():
                next_state_pred, reward_pred = self.env_model(
                    state_batch, 
                    action_batch
                )
            
            # 가상 경험 저장
            for i in range(len(batch_states)):
                simulated_experiences.append((
                    batch_states[i].cpu(),      # Original state
                    batch_actions[i],           # Action
                    float(reward_pred[i].item()),  # Predicted reward
                    next_state_pred[i].cpu(),   # Predicted next state
                ))
        
        return simulated_experiences
```

**Planning 과정**:
1. Visited states에서 `(state, action)` 샘플링
2. 환경 모델로 `(next_state, reward)` 예측
3. 가상 경험 리스트 반환

### 5.3 DeepSupervision 학습

```python
# train_phase2.py - train_policy_with_planning()

# Policy 모델 생성
model = PolicyValueViT_TRM(...).to(device)
optimizer = create_policy_optimizer(model, lr=lr)

# 실제 경험 추출
states, actions, _, _ = extract_transitions(episodes)
states = states.to(device)  # [N, 9, 17, 17]
actions = actions.to(device)  # [N]

# 학습 루프
for epoch in range(num_epochs):
    # 1. Planning으로 가상 경험 생성
    simulated_experiences = planner.plan(n_planning_steps=n_planning_steps)
    
    if len(simulated_experiences) > 0:
        sim_states = torch.stack([s for s, _, _, _ in simulated_experiences]).to(device)
        sim_actions = torch.tensor([a for _, a, _, _ in simulated_experiences], device=device)
    else:
        sim_states = torch.empty(0, 9, 17, 17).to(device)
        sim_actions = torch.empty(0, dtype=torch.long).to(device)
    
    # 2. 배치 샘플링
    num_real = min(batch_size // 2, len(states))
    num_sim = batch_size - num_real
    
    real_indices = torch.randperm(len(states))[:num_real]
    real_batch_states = states[real_indices]
    real_batch_actions = actions[real_indices]
    
    if len(sim_states) > 0:
        sim_indices = torch.randperm(len(sim_states))[:num_sim]
        sim_batch_states = sim_states[sim_indices]
        sim_batch_actions = sim_actions[sim_indices]
    else:
        sim_batch_states = torch.empty(0, 9, 17, 17).to(device)
        sim_batch_actions = torch.empty(0, dtype=torch.long).to(device)
    
    # 3. DeepSupervision 학습
    real_loss, sim_loss = train_with_simulated_experiences(
        model=model,
        real_states=real_batch_states,
        real_actions=real_batch_actions,
        simulated_states=sim_batch_states,
        simulated_actions=sim_batch_actions,
        optimizer=optimizer,
        n_sup=n_sup,
        strategy="last",
        real_weight=1.0,
        sim_weight=0.5,
        device=device,
    )
```

### 5.4 DeepSupervision 함수 상세

```python
# agent_code/ppo_agent/train_deepsup.py

def train_policy_deepsup(
    model: PolicyValueViT_TRM,
    states: torch.Tensor,      # [B, C, H, W]
    actions: torch.Tensor,     # [B]
    optimizer: torch.optim.Optimizer,
    n_sup: int = 16,
    strategy: str = "last",
    device: torch.device = None,
) -> float:
    """DeepSupervision으로 Policy 학습"""
    
    model.train()
    
    # Value head gradient 비활성화
    for param in model.v_head.parameters():
        param.requires_grad = False
    
    # Policy head + TRM network gradient 활성화
    for param in model.pi_head.parameters():
        param.requires_grad = True
    for param in model.trm_net.parameters():
        param.requires_grad = True
    
    B = states.shape[0]
    states = states.to(device)
    actions = actions.to(device)
    
    # 초기 z (zeros)
    z = torch.zeros(B, model.z_dim, device=device)
    
    # Patch embedding (모든 supervision step에서 공유)
    x_embed = model._patch_embed(states)  # [B, N, D]
    x_embed = x_embed + model.pos_embed
    
    # Strategy: "last" (논문 방식)
    if strategy == "last":
        # N_sup - 1번: no gradient
        with torch.no_grad():
            for _ in range(n_sup - 1):
                z = model._latent_recursion(x_embed, z, model.n_latent)
                z = model._deep_recursion(x_embed, z, model.n_latent, model.T)[0]
        
        # 마지막 1번: with gradient
        z = model._latent_recursion(x_embed, z, model.n_latent)
        z = model.trm_net(x_embed, z)
        
        # Policy loss
        logits = model.pi_head(z)  # [B, num_actions]
        loss = F.cross_entropy(logits, actions)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Value head gradient 재활성화
    for param in model.v_head.parameters():
        param.requires_grad = True
    
    return loss.item()
```

**DeepSupervision 과정**:
1. 같은 state에 대해 `n_sup`번 반복
2. 각 반복마다 `z` (latent state) 개선
3. 마지막 step에서만 policy loss 계산
4. Value head는 gradient 비활성화 (Policy만 학습)

---

## 전체 학습 흐름 요약

```
1. 데이터 수집 (collect_teacher_data.py)
   ├─ Episode 실행
   ├─ EpisodeData 수집
   └─ episode_XXXXXX.pt로 저장

2. 데이터 로딩 (train_phase2.py)
   ├─ TeacherDataset: episode 파일 목록
   ├─ load_all_episodes(): 모든 episode 메모리 로드
   └─ extract_transitions(): (s, a, r, s') 추출

3. 환경 모델 학습 (train_env_model)
   ├─ DataLoader 생성
   ├─ (state, action) → (next_state, reward) 학습
   └─ env_model.pt 저장

4. Planning + DeepSupervision (train_policy_with_planning)
   ├─ Visited states buffer 구축
   ├─ Planning: 가상 경험 생성
   ├─ DeepSupervision: Policy 학습
   └─ policy_phase2.pt 저장
```

---

## 주요 함수 호출 순서

```python
# Phase 2 전체 실행 예시

# 1. 환경 모델 학습
train_env_model(
    data_dir="data/teacher_episodes",
    model_path="data/env_models/env_model.pt",
    num_epochs=50,
    batch_size=128,
    lr=1e-3,
)

# 2. Policy 학습 (Planning + DeepSupervision)
train_policy_with_planning(
    data_dir="data/teacher_episodes",
    env_model_path="data/env_models/env_model.pt",
    policy_model_path="data/policy_models/policy_phase2.pt",
    num_epochs=100,
    batch_size=64,
    n_planning_steps=100,
    n_sup=8,  # config에서 설정한 값
    lr=1e-4,
)
```

---

## 데이터 흐름 다이어그램

```
Episode Files (.pt)
    ↓
load_all_episodes()
    ↓
List[Dict] (episodes)
    ↓
extract_transitions()
    ↓
(states, actions, rewards, next_states) Tensors
    ↓
┌─────────────────────────────────────┐
│  Environment Model Training         │
│  (state, action) → (next_state, r) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Visited States Buffer              │
│  (state, action) pairs              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Planning                           │
│  Sample → Predict → Simulated exp   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  DeepSupervision Training           │
│  Real + Simulated experiences       │
│  N_sup iterations per state         │
└─────────────────────────────────────┘
    ↓
policy_phase2.pt
```

---

이 명세서는 교사 데이터셋의 로딩부터 학습까지의 전체 과정을 코드 레벨에서 상세히 설명합니다.

