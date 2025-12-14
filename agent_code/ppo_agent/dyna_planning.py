import random
from typing import List, Tuple, Optional
import torch
import numpy as np
from collections import defaultdict

from .models.environment_model import EnvironmentModel


class VisitedStatesBuffer:
    """Buffer to store visited states for Dyna-Q planning"""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states = []  # List of state tensors
        self.state_actions = defaultdict(list)  # state_hash -> list of actions taken
        self.state_indices = {}  # state_hash -> index in self.states
        self.last_seen = {}  # (state_hash, action) -> timestep last seen
        self.timestep = 0    # counts real experiences added via add()
        
    def _hash_state(self, state: torch.Tensor) -> str:
        """Create a hash for state (used as key)"""
        # Use a simple hash based on state values
        state_np = state.cpu().numpy()
        return str(hash(state_np.tobytes()))
    
    def add(self, state: torch.Tensor, action: int):
        """
        Add a visited state-action pair
        
        Args:
            state: [C, H, W] state tensor
            action: action index
        """
        state_hash = self._hash_state(state)
        
        if state_hash not in self.state_indices:
            # Add new state
            if len(self.states) >= self.max_size:
                # Remove oldest
                oldest_hash = self._hash_state(self.states[0])
                del self.state_indices[oldest_hash]
                self.states.pop(0)
            
            self.state_indices[state_hash] = len(self.states)
            self.states.append(state.clone())
        
        # Add action to this state's action list
        idx = self.state_indices[state_hash]
        if action not in self.state_actions[state_hash]:
            self.state_actions[state_hash].append(action)
        
        # Track last seen timestep for Dyna-Q+ style bonus
        self.last_seen[(state_hash, action)] = self.timestep
        self.timestep += 1
    
    def sample_state(self) -> Tuple[torch.Tensor, str]:
        """
        Sample a random visited state
        
        Returns:
            state: [C, H, W] state tensor
            state_hash: hash of the state
        """
        if len(self.states) == 0:
            raise ValueError("No states in buffer")
        
        idx = random.randint(0, len(self.states) - 1)
        state = self.states[idx]
        state_hash = self._hash_state(state)
        return state, state_hash
    
    def sample_action_for_state(self, state_hash: str) -> int:
        """Sample a random action that was taken in this state"""
        if state_hash not in self.state_actions or len(self.state_actions[state_hash]) == 0:
            raise ValueError(f"No actions recorded for state hash {state_hash}")
        return random.choice(self.state_actions[state_hash])
    
    def get_bonus(self, state_hash: str, action: int, kappa: float) -> float:
        """
        Dyna-Q+ style exploration bonus: bonus = kappa / tau, where tau is
        steps since (state, action) was last seen. If never seen, use current
        timestep as tau to encourage trying it.
        """
        if kappa <= 0:
            return 0.0
        last = self.last_seen.get((state_hash, action), -1)
        if last < 0:
            tau = max(1, self.timestep)
        else:
            tau = max(1, self.timestep - last)
        return kappa / float(tau)
    
    def __len__(self):
        return len(self.states)


class DynaPlanner:
    """
    Dyna-Q style planner that generates simulated experiences using environment model
    """
    def __init__(
        self,
        env_model: EnvironmentModel,
        visited_states: VisitedStatesBuffer,
        device: torch.device = None,
    ):
        self.env_model = env_model
        self.visited_states = visited_states
        self.device = device if device is not None else torch.device('cpu')
        self.env_model.eval()  # Planning uses model in eval mode
    
    def plan(
        self,
        n_planning_steps: int = 100,
        batch_size: int = 32,
        kappa: float = 0.1,
    ) -> List[Tuple[torch.Tensor, int, float, torch.Tensor]]:
        """
        Generate simulated experiences through planning
        
        Args:
            n_planning_steps: Number of planning steps to perform
            batch_size: Batch size for parallel prediction
        Returns:
            List of (state, action, reward, next_state) tuples
        """
        if len(self.visited_states) == 0:
            return []
        
        simulated_experiences = []
        
        # Process in batches for efficiency
        for step in range(0, n_planning_steps, batch_size):
            batch_steps = min(batch_size, n_planning_steps - step)
            batch_states = []
            batch_actions = []
            batch_hashes = []
            
            # Sample states and actions
            for _ in range(batch_steps):
                try:
                    state, state_hash = self.visited_states.sample_state()
                    action = self.visited_states.sample_action_for_state(state_hash)
                    batch_states.append(state)
                    batch_actions.append(action)
                    batch_hashes.append(state_hash)
                except (ValueError, KeyError):
                    # Skip if no valid state/action
                    continue
            
            if len(batch_states) == 0:
                continue
            
            # Batch prediction
            state_batch = torch.stack(batch_states).to(self.device)  # [B, C, H, W]
            action_batch = torch.tensor(batch_actions, device=self.device)  # [B]
            
            with torch.no_grad():
                try:
                    next_state_pred, reward_pred = self.env_model(state_batch, action_batch)
                    
                    # Check for NaN or Inf values
                    if torch.isnan(next_state_pred).any() or torch.isnan(reward_pred).any():
                        # Skip this batch if NaN detected
                        continue
                    if torch.isinf(next_state_pred).any() or torch.isinf(reward_pred).any():
                        # Skip this batch if Inf detected
                        continue
                    
                    # Add to simulated experiences
                    for i in range(len(batch_states)):
                        base_reward = float(reward_pred[i].item())
                        bonus = self.visited_states.get_bonus(batch_hashes[i], batch_actions[i], kappa)
                        reward_with_bonus = base_reward + bonus
                        # Clamp predictions to reasonable ranges
                        next_state_clamped = torch.clamp(next_state_pred[i], 0.0, 1.0)
                        reward_clamped = float(torch.clamp(torch.tensor(reward_with_bonus), -200.0, 200.0).item())
                        simulated_experiences.append((
                            batch_states[i].cpu(),  # Original state
                            batch_actions[i],       # Action
                            reward_clamped,         # Reward with Dyna-Q+ bonus
                            next_state_clamped.cpu(),  # Predicted next state
                        ))
                except Exception as e:
                    # Skip this batch if prediction fails
                    continue
        
        return simulated_experiences
    
    def add_experience(self, state: torch.Tensor, action: int):
        """Add a real experience to visited states buffer"""
        self.visited_states.add(state, action)


def create_dyna_planner(
    env_model: EnvironmentModel,
    visited_states_buffer_size: int = 10000,
    device: torch.device = None,
) -> DynaPlanner:
    """Factory function to create DynaPlanner"""
    visited_states = VisitedStatesBuffer(max_size=visited_states_buffer_size)
    return DynaPlanner(env_model, visited_states, device=device)

