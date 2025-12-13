import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EnvironmentModel(nn.Module):
    """
    Environment model that predicts (next_state, reward) given (state, action)
    
    This is used for Dyna-Q style planning where we generate simulated experiences
    to augment real training data.
    """
    def __init__(
        self,
        state_channels: int = 9,
        state_height: int = 17,
        state_width: int = 17,
        num_actions: int = 6,
        hidden_dim: int = 256,
        use_deterministic: bool = True,
    ):
        super().__init__()
        self.state_channels = state_channels
        self.state_height = state_height
        self.state_width = state_width
        self.num_actions = num_actions
        self.use_deterministic = use_deterministic
        
        # Input: state [B, C, H, W] + action embedding
        state_size = state_channels * state_height * state_width
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, hidden_dim // 4)
        
        # State encoder (CNN-based)
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # [B, 256, 4, 4]
        )
        encoder_output_size = 256 * 4 * 4
        
        # Combined feature size
        combined_size = encoder_output_size + hidden_dim // 4
        
        # State transition predictor (predict state directly)
        self.transition_net = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_size),
            nn.Sigmoid(),  # Output in [0, 1] range for state prediction
        )

        # Reward predictor
        self.reward_net = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward
        
        Args:
            state: [B, C, H, W] current state
            action: [B] action indices
        Returns:
            next_state_pred: [B, C, H, W] predicted next state
            reward_pred: [B] predicted reward
        """
        B = state.shape[0]
        device = state.device
        
        # Encode state
        state_feat = self.state_encoder(state)  # [B, 256, 4, 4]
        state_feat_flat = state_feat.view(B, -1)  # [B, 256*4*4]
        
        # Encode action
        action_emb = self.action_embed(action)  # [B, hidden_dim//4]
        
        # Combine features
        combined = torch.cat([state_feat_flat, action_emb], dim=-1)  # [B, combined_size]
        
        # Predict reward
        reward_pred = self.reward_net(combined).squeeze(-1)  # [B]
        
        # Predict next state directly
        next_state_pred = self.transition_net(combined)  # [B, state_size]
        next_state_pred = next_state_pred.view(B, self.state_channels, self.state_height, self.state_width)  # Reshape to [B, C, H, W]
        
        # If deterministic, return directly; otherwise add noise
        if not self.use_deterministic and self.training:
            # Add small noise for exploration during training
            noise = torch.randn_like(next_state_pred) * 0.01
            next_state_pred = next_state_pred + noise
            next_state_pred = torch.clamp(next_state_pred, 0, 1)
        
        return next_state_pred, reward_pred

    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Convenience method to get only next state"""
        next_state, _ = self.forward(state, action)
        return next_state

    def predict_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Convenience method to get only reward"""
        _, reward = self.forward(state, action)
        return reward


def create_env_model(
    state_channels: int = 9,
    state_height: int = 17,
    state_width: int = 17,
    num_actions: int = 6,
    hidden_dim: int = 256,
    model_path: str = None,
    device: torch.device = None,
) -> EnvironmentModel:
    """Factory function to create and optionally load environment model"""
    model = EnvironmentModel(
        state_channels=state_channels,
        state_height=state_height,
        state_width=state_width,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
    )
    
    if device is not None:
        model = model.to(device)
    
    if model_path is not None:
        import os
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle architecture changes
            except Exception as e:
                # Silently fail if model can't be loaded (e.g., architecture mismatch)
                # This is expected if planning is not being used
                pass
    
    return model

