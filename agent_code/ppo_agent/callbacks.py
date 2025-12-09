import importlib
import os
import random
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def _team_tag(agent_name: str) -> str:
    if not agent_name:
        return ""
    name = str(agent_name)
    return name.split('_')[0]


def _to_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from .models.vit import PolicyValueViT


class _Shared:
    def __init__(self):
        self.device = _to_device()
        # Allow overriding model path via env; default to local file
        self.model_path = os.environ.get("PPO_MODEL_PATH", "ppo_model.pt")
        self.policy: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Checkpointing controls
        try:
            self.save_every_rounds = int(os.environ.get("PPO_SAVE_EVERY", "0"))  # 0=only latest
        except Exception:
            self.save_every_rounds = 0
        self.checkpoint_dir = os.environ.get("PPO_CHECKPOINT_DIR", None)
        self.save_round_counter = 0

        # Epsilon-greedy exploration schedule (rule-based assisted)
        # Higher epsilon = more teacher guidance, faster learning of aggressive behavior
        try:
            self.eps_start = float(os.environ.get("PPO_EPS_START", "0.7"))
        except Exception:
            self.eps_start = 0.7
        try:
            self.eps_end = float(os.environ.get("PPO_EPS_END", "0.2"))  # Keep some teacher guidance
        except Exception:
            self.eps_end = 0.2  # Default: keep 20% teacher guidance even at end
        try:
            self.eps_decay_rounds = int(os.environ.get("PPO_EPS_DECAY_ROUNDS", "400000"))
        except Exception:
            self.eps_decay_rounds = 1500

        # Training curriculum/suppression controls
        try:
            self.suppress_suicide_until = int(os.environ.get("PPO_SUPPRESS_SUICIDE_UNTIL", "0"))
        except Exception:
            self.suppress_suicide_until = 0
        # Count of finished rounds across all agents (incremented when a round fully wraps up)
        self.completed_rounds = 0

        # Suicide penalty schedule: start high, then reduce after a switch round
        try:
            self.suicide_penalty_high = float(os.environ.get("PPO_SUICIDE_HIGH", "-15"))
        except Exception:
            self.suicide_penalty_high = -15.0
        try:
            self.suicide_penalty_after = float(os.environ.get("PPO_SUICIDE_AFTER", "nan"))  # NaN -> use default
        except Exception:
            self.suicide_penalty_after = float("nan")
        try:
            self.suicide_penalty_switch_round = int(os.environ.get("PPO_SUICIDE_SWITCH", "200"))
        except Exception:
            self.suicide_penalty_switch_round = 200

        # Registered agent instances (for multi-agent shared control)
        self._next_instance_id = 0
        self.instance_ids = {}

        # Per-round coordination: count how many of our agents reported end_of_round
        self.round_done_counts = {}

        # Experience buffers shared across our controlled agents
        self.reset_buffers()

        # Hyperparameters (filled in setup_training)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.update_epochs = 4
        self.batch_size = 64
        self.learning_rate = 1e-4

        # Cached rule-based module for exploration support
        self.rule_module = None
        
        # Statistics for teacher model usage
        self.teacher_action_count = 0
        self.teacher_invalid_count = 0
        self.total_action_count = 0
        self.last_teacher_stats = None  # For main process to read
        self._stats_file = None  # Path to stats file

    def ensure_policy(self, train: bool, logger) -> None:
        # Set stats file path when model path is known
        if self._stats_file is None and self.model_path:
            model_dir = os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.'
            self._stats_file = os.path.join(model_dir, 'teacher_stats.json')
        
        if self.policy is None:
            mixer = os.environ.get("BOMBER_VIT_MIXER", "attn")
            embed_dim = int(os.environ.get("BOMBER_VIT_DIM", 64))
            depth = int(os.environ.get("BOMBER_VIT_DEPTH", 2))
            num_heads = int(os.environ.get("BOMBER_VIT_HEADS", 4))
            self.policy = PolicyValueViT(
                in_channels=9,
                num_actions=len(ACTIONS),
                img_size=(s.COLS, s.ROWS),
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=4.0,
                drop=0.0,
                attn_drop=0.0,
                patch_size=1,
                use_cls_token=False,
                mixer=mixer,
            ).to(self.device)
            # Load checkpoint for both train/eval if available, unless explicitly reset
            reset_requested = os.environ.get("PPO_RESET", "0") == "1"
            if (not reset_requested) and os.path.isfile(self.model_path):
                try:
                    logger.info(f"Loading PPO model from '{self.model_path}'.")
                    state = torch.load(self.model_path, map_location=self.device,weights_only=True)
                    self.policy.load_state_dict(state)
                    # Track model file modification time for A3C workers
                    self._last_model_mtime = os.path.getmtime(self.model_path)
                except Exception as ex:
                    logger.warning(f"Failed to load PPO model from '{self.model_path}': {ex}")
            else:
                logger.info("Starting with randomly initialized PPO weights.")
        
        # For A3C: Always reload model if file was updated (each subprocess gets latest)
        # This ensures workers use the most recent model after PPO updates
        if not hasattr(self, '_last_model_mtime'):
            self._last_model_mtime = 0
        
        if os.path.isfile(self.model_path):
            try:
                current_mtime = os.path.getmtime(self.model_path)
                if current_mtime > self._last_model_mtime:
                    # Model file was updated, reload it
                    state = torch.load(self.model_path, map_location=self.device, weights_only=True)
                    self.policy.load_state_dict(state)
                    self._last_model_mtime = current_mtime
                    logger.debug(f"Reloaded updated PPO model (mtime: {current_mtime})")
            except Exception as ex:
                pass  # Silently ignore reload errors to avoid spam

    def register_instance(self, self_obj) -> int:
        if id(self_obj) in self.instance_ids:
            return self.instance_ids[id(self_obj)]
        inst_id = self._next_instance_id
        self._next_instance_id += 1
        self.instance_ids[id(self_obj)] = inst_id
        return inst_id

    def reset_buffers(self):
        self.buf_states = []
        self.buf_actions = []
        self.buf_logps = []
        self.buf_values = []
        self.buf_rewards = []
        self.buf_dones = []

    def current_epsilon(self) -> float:
        if self.eps_decay_rounds <= 0:
            return float(self.eps_end)
        span = max(1, self.eps_decay_rounds)
        
        # ALWAYS use global round count from file - ignore local completed_rounds completely
        # This ensures all workers use the same global total, not individual counts
        global_rounds = None
        
        # Try to get results_dir from environment (set by A3C worker)
        results_dir = os.environ.get('A3C_RESULTS_DIR', None)
        if results_dir:
            file_path = os.path.join(results_dir, 'global_round_count.txt')
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        global_rounds = int(f.read().strip())
                except:
                    pass
        
        # If not found, search in results directory
        if global_rounds is None:
            try:
                results_base = os.path.join(os.getcwd(), 'results')
                if os.path.exists(results_base):
                    # Find most recent results directory
                    subdirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
                    if subdirs:
                        # Sort by modification time, most recent first
                        subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(results_base, d)), reverse=True)
                        for subdir in subdirs:
                            file_path = os.path.join(results_base, subdir, 'global_round_count.txt')
                            if os.path.exists(file_path):
                                try:
                                    with open(file_path, 'r') as f:
                                        global_rounds = int(f.read().strip())
                                    break
                                except:
                                    continue
            except:
                pass
        
        # MUST use global rounds - if not found, return start epsilon (conservative)
        if global_rounds is None:
            return float(self.eps_start)
        
        # Use global round count - this is the TOTAL across all workers
        frac = min(1.0, max(0.0, global_rounds / span))
        epsilon = float(self.eps_start + (self.eps_end - self.eps_start) * frac)
        return epsilon

    def ensure_rule_module(self):
        if self.rule_module is None:
            # Use aggressive_teacher_agent for epsilon-greedy exploration
            # This helps PPO learn aggressive attacking behavior faster!
            teacher_module = os.environ.get("PPO_TEACHER_MODULE", "agent_code.aggressive_teacher_agent.callbacks")
            self.rule_module = importlib.import_module(teacher_module)
        return self.rule_module

    def build_rule_helper(self, logger):
        module = self.ensure_rule_module()
        helper = SimpleNamespace(logger=logger)
        module.setup(helper)
        return helper


# Global shared singleton
SHARED = _Shared()


def setup(self):
    # Shared model/optimizer state across multiple instances of this agent code
    self.device = SHARED.device
    SHARED.ensure_policy(self.train, self.logger)
    self.policy = SHARED.policy  # convenience alias

    # Assign a stable instance id for multi-agent coordination
    self.instance_id = SHARED.register_instance(self)

    # Optional rule-based guidance for epsilon-greedy exploration
    self._explore_module = None
    self._explore_agent = None
    self._last_epsilon = 0.0
    if self.train:
        try:
            self._explore_module = SHARED.ensure_rule_module()
            self._explore_agent = SHARED.build_rule_helper(self.logger)
        except Exception as ex:
            self._explore_module = None
            self._explore_agent = None
            try:
                self.logger.warning(f"Failed to initialize rule-based explorer: {ex}")
            except Exception:
                pass

    # Runtime scratch vars used between act() and game_events_occurred()
    self._last_state_tensor: Optional[torch.Tensor] = None
    self._last_action_idx: Optional[int] = None
    self._last_log_prob: Optional[float] = None
    self._last_value: Optional[float] = None


def act(self, game_state: dict) -> str:
    self.policy.eval()
    obs = state_to_features(game_state)
    x = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # [1, C, H, W]
    with torch.no_grad():
        logits, value = self.policy(x)
        # No action masking - let teacher model and PPO decide freely
        # Invalid actions will be handled by the game engine (INVALID_ACTION event)
        cat = torch.distributions.Categorical(logits=logits)

        action_idx: Optional[int] = None
        used_rule = False
        epsilon = 0.0
        if self.train and self._explore_agent is not None and self._explore_module is not None:
            epsilon = SHARED.current_epsilon()
            self._last_epsilon = epsilon
            SHARED.total_action_count += 1
            if epsilon > 0.0 and random.random() < epsilon:
                try:
                    rb_action = self._explore_module.act(self._explore_agent, game_state)
                    if rb_action in ACTIONS:
                        candidate_idx = ACTIONS.index(rb_action)
                        # Use teacher action directly without any validity check
                        # Game engine will handle invalid actions
                        action_idx = candidate_idx
                        used_rule = True
                        SHARED.teacher_action_count += 1
                    else:
                        # Teacher returned invalid action name
                        SHARED.teacher_invalid_count += 1
                except Exception as ex:
                    SHARED.teacher_invalid_count += 1
                    try:
                        self.logger.debug(f"Rule-based exploration fallback failed: {ex}")
                    except Exception:
                        pass
        else:
            self._last_epsilon = 0.0

        if action_idx is None:
            if self.train:
                action_idx = int(cat.sample().item())
            else:
                action_idx = int(torch.argmax(logits, dim=-1).item())

        action_tensor = torch.tensor(action_idx, device=self.device)
        log_prob = float(cat.log_prob(action_tensor).item())

        # Debug trace: step, pos, chosen action, top-2 probs, value
        try:
            step = game_state.get('step')
            _, _, _, pos = game_state.get('self')
            probs_soft = torch.softmax(logits, dim=-1).squeeze(0)
            top_probs, top_inds = torch.topk(probs_soft, k=min(2, probs_soft.numel()))
            top_info = ", ".join([f"{ACTIONS[int(i)]}:{float(p):.2f}" for p, i in zip(top_probs.tolist(), top_inds.tolist())])
            extra = f" eps={epsilon:.2f} rule={int(used_rule)}"
            self.logger.debug(f"[step {step}] pos={pos} action={ACTIONS[action_idx]} val={float(value.squeeze(0)):.2f}{extra} top=[{top_info}]")
        except Exception:
            pass

    # Stash for training callback
    self._last_state_tensor = x.detach().cpu()
    self._last_action_idx = action_idx
    self._last_log_prob = log_prob
    self._last_value = float(value.squeeze(0).item())

    return ACTIONS[action_idx]


def state_to_features(game_state: dict) -> np.ndarray:
    if game_state is None:
        return np.zeros((9, s.COLS, s.ROWS), dtype=np.float32)

    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state.get('explosion_map')
    self_info = game_state['self']
    others = game_state['others']

    grid = np.zeros((9, s.COLS, s.ROWS), dtype=np.float32)

    grid[0] = (field == -1).astype(np.float32)  # walls
    grid[1] = (field == 1).astype(np.float32)   # crates
    grid[2] = (field == 0).astype(np.float32)   # free

    for x, y in coins:
        grid[3, x, y] = 1.0

    for (x, y), timer in bombs:
        grid[4, x, y] = max(grid[4, x, y], float(timer) / max(1.0, float(s.BOMB_TIMER)))

    if explosion_map is not None:
        norm = max(1.0, float(s.BOMB_TIMER + s.EXPLOSION_TIMER))
        grid[5] = np.clip(explosion_map.astype(np.float32) / norm, 0.0, 1.0)

    self_name, _, _, (sx, sy) = self_info
    self_tag = _team_tag(self_name)
    grid[6, sx, sy] = 1.0

    for other_name, _, _, (ox, oy) in others:
        tag = _team_tag(other_name)
        if tag == self_tag:
            grid[7, ox, oy] = 1.0
        else:
            grid[8, ox, oy] = 1.0

    return grid.astype(np.float32)


