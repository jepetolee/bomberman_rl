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
from .models.vit_trm import PolicyValueViT_TRM, PolicyValueViT_TRM_Hybrid


class _Shared:
    def __init__(self):
        self.device = _to_device()
        # Allow overriding model path via env; default to local file
        # Default to policy_phase2.pt if available, otherwise fallback to ppo_model.pt
        default_model_path = "data/policy_models/policy_phase2.pt" if os.path.exists("data/policy_models/policy_phase2.pt") else "ppo_model.pt"
        self.model_path = os.environ.get("PPO_MODEL_PATH", default_model_path)
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
            self.eps_start = 0.4
        try:
            self.eps_end = float(os.environ.get("PPO_EPS_END", "0.2"))  # Keep some teacher guidance
        except Exception:
            self.eps_end = 0.2  # Default: keep 20% teacher guidance even at end
        try:
            self.eps_decay_rounds = int(os.environ.get("PPO_EPS_DECAY_ROUNDS", "10000"))
        except Exception:
            self.eps_decay_rounds = 1500

        # Training curriculum/suppression controls
        try:
            self.suppress_suicide_until = int(os.environ.get("PPO_SUPPRESS_SUICIDE_UNTIL", "0"))
        except Exception:
            self.suppress_suicide_until = 0
        # Count of finished rounds across all agents (incremented when a round fully wraps up)
        self.completed_rounds = 0

        # Suicide penalty schedule: start high, keep high to prevent suicide
        try:
            self.suicide_penalty_high = float(os.environ.get("PPO_SUICIDE_HIGH", "-30"))
        except Exception:
            self.suicide_penalty_high = -30.0  # Increased from -15.0 to strongly discourage suicide
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
            use_trm = os.environ.get("BOMBER_USE_TRM", "0") == "1"
            use_frozen_vit = os.environ.get("BOMBER_FROZEN_VIT", "0") == "1"
            
            # Try to load config from YAML first and use create_model_from_config if available
            try:
                from config.load_config import load_config, get_model_config, create_model_from_config
                cfg_path = os.environ.get("BOMBER_CONFIG_PATH", "config/trm_config.yaml")
                config = load_config(cfg_path)
                model_cfg = get_model_config(config)
                model_type = model_cfg.get('type', 'vit').lower()
                
                # If YAML specifies efficient_gtrxl, use create_model_from_config
                # strict_yaml=True: YAML 설정을 엄격하게 따르고, 기본값을 사용하지 않음
                if model_type == 'efficient_gtrxl':
                    self.policy = create_model_from_config(config, device=self.device, strict_yaml=True)
                    try:
                        print(f"[PPO] Config type={model_cfg.get('type','unknown')}, class={self.policy.__class__.__name__}")
                    except Exception:
                        pass
                    # Load checkpoint strictly; if mismatch, ignore and keep new model
                    reset_requested = os.environ.get("PPO_RESET", "0") == "1"
                    if (not reset_requested) and os.path.isfile(self.model_path):
                        try:
                            logger.info(f"Loading PPO model from '{self.model_path}' (strict).")
                            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
                            missing_keys, unexpected_keys = self.policy.load_state_dict(state, strict=True)
                            if missing_keys or unexpected_keys:
                                logger.warning(f"Checkpoint mismatch: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}. Starting fresh.")
                                self.policy = create_model_from_config(config, device=self.device, strict_yaml=True)
                            else:
                                logger.info(f"Loaded checkpoint '{self.model_path}' as {self.policy.__class__.__name__}")
                        except Exception as e:
                            logger.warning(f"Failed to load checkpoint strictly: {e}. Starting from scratch.")
                    return
                trm_cfg = model_cfg.get('trm', {})
                patch_cfg = model_cfg.get('patch', {})
                vit_cfg = model_cfg.get('vit', {})
                
                # Use YAML values with env var override
                embed_dim = int(os.environ.get("BOMBER_VIT_DIM", model_cfg.get('embed_dim', 512)))  # Increased default
                z_dim = embed_dim  # z_dim always equals embed_dim
                n_latent = int(os.environ.get("BOMBER_TRM_N", trm_cfg.get('n_latent', 4)))
                n_sup = int(os.environ.get("BOMBER_TRM_N_SUP", trm_cfg.get('n_sup', 8)))
                T = int(os.environ.get("BOMBER_TRM_T", trm_cfg.get('T', 3)))
                use_ema = os.environ.get("BOMBER_TRM_EMA", "0") != "0" if not trm_cfg.get('use_ema', True) else True
                ema_decay = float(os.environ.get("BOMBER_TRM_EMA", str(trm_cfg.get('ema_decay', 0.999))))
                patch_size = int(os.environ.get("BOMBER_TRM_PATCH_SIZE", patch_cfg.get('size', 1)))
                patch_stride = int(os.environ.get("BOMBER_TRM_PATCH_STRIDE", patch_cfg.get('stride', 1)))
                vit_depth = int(os.environ.get("BOMBER_VIT_DEPTH", vit_cfg.get('depth', 6)))  # Increased default
                vit_heads = int(os.environ.get("BOMBER_VIT_HEADS", vit_cfg.get('num_heads', 8)))  # Increased default
            except Exception:
                # Fallback to env vars only
                embed_dim = int(os.environ.get("BOMBER_VIT_DIM", 512))  # Increased default
                z_dim = int(os.environ.get("BOMBER_TRM_Z_DIM", str(embed_dim)))
                n_latent = int(os.environ.get("BOMBER_TRM_N", "6"))
                n_sup = int(os.environ.get("BOMBER_TRM_N_SUP", "16"))
                T = int(os.environ.get("BOMBER_TRM_T", "3"))
                use_ema = os.environ.get("BOMBER_TRM_EMA", "0.999") != "0"
                ema_decay = float(os.environ.get("BOMBER_TRM_EMA", "0.999"))
                patch_size = int(os.environ.get("BOMBER_TRM_PATCH_SIZE", "1"))  # Default to 1
                patch_stride = int(os.environ.get("BOMBER_TRM_PATCH_STRIDE", "1"))
                vit_depth = int(os.environ.get("BOMBER_VIT_DEPTH", 6))  # Increased default
                vit_heads = int(os.environ.get("BOMBER_VIT_HEADS", 8))  # Increased default
            
            if use_trm:
                # Always use Hybrid model (ViT backbone + TRM residual)
                # This matches the model saved by train_phase2.py (policy_phase2.pt)
                # Frozen ViT mode (BOMBER_FROZEN_VIT=1): Only TRM is trained
                # Non-frozen mode (BOMBER_FROZEN_VIT=0): All parameters are trained
                self.policy = PolicyValueViT_TRM_Hybrid(
                    in_channels=10,
                    num_actions=len(ACTIONS),
                    img_size=(s.COLS, s.ROWS),
                    embed_dim=embed_dim,
                    vit_depth=vit_depth,
                    vit_heads=vit_heads,
                    vit_mlp_ratio=4.0,
                    vit_patch_size=patch_size,  # Use YAML patch_size
                    trm_n_latent=n_latent,
                    trm_mlp_ratio=4.0,
                    trm_drop=0.0,
                    trm_patch_size=patch_size,
                    trm_patch_stride=patch_stride,
                    use_ema=use_ema,
                    ema_decay=ema_decay,
                ).to(self.device)
            else:
                # Standard ViT model
                mixer = os.environ.get("BOMBER_VIT_MIXER", "attn")
                # Use YAML embed_dim if available, otherwise default to 256
                # Note: embed_dim may already be defined if use_trm was True, but depth/num_heads are not
                try:
                    from config.load_config import load_config, get_model_config
                    cfg_path = os.environ.get("BOMBER_CONFIG_PATH", "config/trm_config.yaml")
                    config = load_config(cfg_path)
                    model_cfg = get_model_config(config)
                    vit_cfg = model_cfg.get('vit', {})
                    if 'embed_dim' not in locals():
                        embed_dim = int(os.environ.get("BOMBER_VIT_DIM", model_cfg.get('embed_dim', 512)))  # Increased default
                    depth = int(os.environ.get("BOMBER_VIT_DEPTH", vit_cfg.get('depth', 6)))  # Increased default
                    num_heads = int(os.environ.get("BOMBER_VIT_HEADS", vit_cfg.get('num_heads', 8)))  # Increased default
                except Exception:
                    if 'embed_dim' not in locals():
                        embed_dim = int(os.environ.get("BOMBER_VIT_DIM", 512))  # Increased default
                    depth = int(os.environ.get("BOMBER_VIT_DEPTH", 6))  # Increased default
                    num_heads = int(os.environ.get("BOMBER_VIT_HEADS", 8))  # Increased default
                self.policy = PolicyValueViT(
                    in_channels=10,
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
                    state = torch.load(self.model_path, map_location=self.device, weights_only=True)
                    
                    # Try strict load first
                    try:
                        missing_keys, unexpected_keys = self.policy.load_state_dict(state, strict=False)
                        if missing_keys or unexpected_keys:
                            logger.info(f"Partial load: {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
                            if missing_keys and len(missing_keys) < 50:  # Only log if not too many
                                logger.debug(f"Missing keys (first 10): {missing_keys[:10]}")
                            if unexpected_keys and len(unexpected_keys) < 50:
                                logger.debug(f"Unexpected keys (first 10): {unexpected_keys[:10]}")
                    except Exception as e:
                        # If strict=False also fails, log and continue with random init
                        logger.warning(f"Failed to load state_dict (partial): {e}")
                        logger.info("Continuing with randomly initialized weights.")
                    
                    # Track model file modification time for A3C workers
                    self._last_model_mtime = os.path.getmtime(self.model_path)
                except Exception as ex:
                    logger.warning(f"Failed to load PPO model from '{self.model_path}': {ex}")
                    logger.info("Continuing with randomly initialized weights.")
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
                    # Use strict=False for reloads to handle architecture changes
                    missing_keys, unexpected_keys = self.policy.load_state_dict(state, strict=False)
                    self._last_model_mtime = current_mtime
                    if missing_keys or unexpected_keys:
                        logger.debug(f"Reloaded updated PPO model (mtime: {current_mtime}, {len(missing_keys)} missing, {len(unexpected_keys)} unexpected)")
                    else:
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
        # Optional recurrent latent buffer (z_prev per step). Used by planning and recurrent PPO.
        self.buf_z_prev = []

    def current_epsilon(self) -> float:
        if self.eps_decay_rounds <= 0:
            return float(self.eps_end)
        span = max(1, self.eps_decay_rounds)
        
        # Force teacher-only phase for early rounds (kick-start)
        try:
            force_rounds = int(os.environ.get("PPO_FORCE_TEACHER_ROUNDS", "0"))
        except Exception:
            force_rounds = 0
        
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
        # 하지만 force_rounds가 설정되어 있으면 라운드 파일이 없어도 teacher만 사용
        if global_rounds is None:
            if force_rounds > 0:
                return 1.0
            return float(self.eps_start)
        
        # Force teacher-only phase for early global rounds
        if force_rounds > 0 and global_rounds < force_rounds:
            return 1.0
        
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
        # 일부 rule 모듈이 helper.build_rule_helper를 기대하므로 self 참조를 연결
        try:
            helper.build_rule_helper = self.build_rule_helper  # compatibility
        except Exception:
            pass
        module.setup(helper)
        return helper


# Global shared singleton
SHARED = _Shared()


def setup(self):
    # Shared model/optimizer state across multiple instances of this agent code
    self.device = SHARED.device
    SHARED.ensure_policy(self.train, self.logger)
    self.policy = SHARED.policy  # convenience alias

    # helper.build_rule_helper 를 요구하는 rule 모듈 호환을 위해 self에도 연결
    try:
        import types
        self.build_rule_helper = types.MethodType(build_rule_helper, self)
    except Exception:
        pass

    # Assign a stable instance id for multi-agent coordination
    self.instance_id = SHARED.register_instance(self)

    # Recurrent TRM latent state management
    self._current_z: Optional[torch.Tensor] = None
    self._last_round = -1
    # Disable recurrent carry unless explicitly enabled
    self._use_recurrent_z = os.environ.get("BOMBER_TRM_RECURRENT", "0") == "1"

    # Runtime scratch vars used between act() and game_events_occurred()
    self._last_state_tensor: Optional[torch.Tensor] = None
    self._last_action_idx: Optional[int] = None
    self._last_log_prob: Optional[float] = None
    self._last_value: Optional[float] = None
    self._last_z_prev: Optional[torch.Tensor] = None


def act(self, game_state: dict) -> str:
    import sys
    self.policy.eval()
    
    # Check if new round started (reset z)
    current_round = game_state.get('round', -1)
    step = game_state.get('step', 999)
    
    # Debug: Log to file to avoid subprocess buffering issues
    try:
        debug_log = os.path.join(os.path.dirname(SHARED.model_path), 'act_debug.log')
        if step <= 3:
            with open(debug_log, 'a') as f:
                f.write(f"[ACT CALLED] Round={current_round} Step={step} Train={self.train}\n")
                f.flush()
    except Exception:
        pass
    
    if current_round != self._last_round:
        self._current_z = None
        self._last_round = current_round
    
    obs = state_to_features(game_state)
    x = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # [1, C, H, W]
    
    # Danger mask: avoid stepping into tiles about to explode (timer <=1 or explosion active)
    def _mask_unsafe_logits(logits: torch.Tensor) -> torch.Tensor:
        bombs = game_state.get('bombs', [])
        explosion_map = game_state.get('explosion_map', np.zeros((s.COLS, s.ROWS)))
        _, _, _, (cx, cy) = game_state['self']
        danger = set()
        # existing explosions
        for ix in range(explosion_map.shape[0]):
            for iy in range(explosion_map.shape[1]):
                if explosion_map[ix, iy] > 0:
                    danger.add((ix, iy))
        # bombs with timer<=1
        for (bx, by), t in bombs:
            if t <= 1:
                for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2),(3,0),(-3,0),(0,3),(0,-3)]:
                    nx, ny = bx+dx, by+dy
                    if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                        danger.add((nx, ny))
        # action deltas
        deltas = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0), 'WAIT': (0,0)}
        logits = logits.clone()
        for idx, act in enumerate(ACTIONS):
            if act == 'BOMB':
                continue
            dx, dy = deltas.get(act, (0,0))
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in danger:
                logits[:, idx] = -1e9
        return logits

    with torch.no_grad():
        # Check if policy supports recurrent z (TRM model)
        if hasattr(self.policy, 'forward_with_z'):
            if self._use_recurrent_z:
                # True recurrent mode: carry z across timesteps in the same round
                z_prev = self._current_z
                logits, value, z_new = self.policy.forward_with_z(x, z_prev=z_prev)
                self._current_z = z_new  # Store for next timestep
                self._last_z_prev = None if z_prev is None else z_prev.detach().cpu()
            else:
                # Non-recurrent mode: ALWAYS start from zero (z_prev=None) every step
                self._current_z = None
                logits, value, _ = self.policy.forward_with_z(x, z_prev=None)
                self._last_z_prev = None
        else:
            logits, value = self.policy(x)
            self._last_z_prev = None
        
        # Mask unsafe actions (tiles about to explode)
        logits = _mask_unsafe_logits(logits)
        cat = torch.distributions.Categorical(logits=logits)

        if self.train:
            print(f"[PPO Branch] entering train branch", flush=True)
            # 강제 teacher 사용: 무조건 teacher 호출, 반환되면 마스크 무시하고 그대로 적용
            eps = SHARED.current_epsilon()  # 로그용
            round_num = game_state.get('round', 0)
            # 확실한 위치에 남긴다: 파일 기준 (agent_code/ppo_agent/act_debug.log)
            log_path = os.path.join(os.path.dirname(__file__), "act_debug.log")
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            except Exception:
                pass
            if step <= 5:
                try:
                    with open(log_path, 'a') as f:
                        f.write(f"[PPO Debug] Round={round_num} Step={step} Eps={eps:.3f} Train={self.train}\n")
                        f.flush()
                except Exception:
                    pass
                # 콘솔에도 한 줄
                print(f"[PPO Debug] R={round_num} S={step} eps={eps:.3f} train={self.train}", flush=True)

            teacher_action_idx = None
            teacher_action = None
            try:
                rule_module = SHARED.ensure_rule_module()
                # Build helper with dummy logger and build_rule_helper attr for compatibility
                class _DummyLogger:
                    def debug(self,*a,**k): pass
                    def info(self,*a,**k): pass
                    def warning(self,*a,**k): pass
                    def error(self,*a,**k): pass
                helper = SimpleNamespace(logger=self.logger or _DummyLogger())
                helper.build_rule_helper = lambda *_args, **_kwargs: helper
                try:
                    rule_module.setup(helper)
                except Exception:
                    pass
                teacher_action = rule_module.act(helper, game_state)
                try:
                    with open(log_path, 'a') as f:
                        f.write(f"[PPO Teacher] Got action: {teacher_action}\n")
                        f.flush()
                except Exception:
                    pass
                print(f"[PPO Teacher] Got action: {teacher_action}", flush=True)
                if teacher_action in ACTIONS:
                    teacher_action_idx = ACTIONS.index(teacher_action)
            except Exception as e:
                import traceback
                err_txt = f"[PPO Teacher] Error: {e}\n{traceback.format_exc()}"
                try:
                    with open(log_path, 'a') as f:
                        f.write(err_txt + "\n")
                        f.flush()
                except Exception:
                    pass
                print(err_txt, flush=True)
                teacher_action_idx = None

            if teacher_action_idx is not None:
                # teacher가 준 액션이면 마스크 무시하고 그대로 사용
                action_idx = int(teacher_action_idx)
                try:
                    with open(log_path, 'a') as f:
                        f.write(f"[PPO Agent] Using teacher action={teacher_action} (forced, ignore mask)\n")
                        f.flush()
                except Exception:
                    pass
                print(f"[PPO Agent] Using teacher action={teacher_action} (forced)", flush=True)
            else:
                # teacher 실패: 마스크된 유효 액션 중 랜덤 -> 없으면 폴리시 샘플
                valid_idxs = [i for i, v in enumerate(logits.squeeze(0)) if v > -1e8]
                if valid_idxs:
                    action_idx = int(random.choice(valid_idxs))
                    try:
                        with open(log_path, 'a') as f:
                            f.write("[PPO Agent] Teacher invalid -> random valid action\n")
                            f.flush()
                    except Exception:
                        pass
                    print("[PPO Agent] Teacher invalid -> random valid action", flush=True)
                else:
                    action_idx = int(cat.sample().item())
                    try:
                        with open(log_path, 'a') as f:
                            f.write("[PPO Agent] No valid actions -> policy sample\n")
                            f.flush()
                    except Exception:
                        pass
                    print("[PPO Agent] No valid actions -> policy sample", flush=True)
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
            self.logger.debug(f"[step {step}] pos={pos} action={ACTIONS[action_idx]} val={float(value.squeeze(0)):.2f} top=[{top_info}]")
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
        return np.zeros((10, s.COLS, s.ROWS), dtype=np.float32)

    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state.get('explosion_map', np.zeros((s.COLS, s.ROWS)))
    self_info = game_state['self']
    others = game_state['others']

    grid = np.zeros((10, s.COLS, s.ROWS), dtype=np.float32)

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

    danger_map = np.zeros((s.COLS, s.ROWS), dtype=np.float32)

    blast_power = s.BOMB_POWER

    for (bx, by), timer in bombs:
        if timer > (s.BOMB_TIMER - 1):
            risk = 0.1
        else:
            risk = 1.0 - (timer / float(s.BOMB_TIMER))

        danger_map[bx, by] = max(danger_map[bx, by], risk)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for i in range(1, blast_power + 1):
                nx, ny = bx + dx * i, by + dy * i
                
                if not (0 <= nx < s.COLS and 0 <= ny < s.ROWS):
                    break
                
                if field[nx, ny] == -1:
                    break
                
                danger_map[nx, ny] = max(danger_map[nx, ny], risk)

    if explosion_map is not None:
        active_explosions = (explosion_map > 0).astype(np.float32)
        danger_map = np.maximum(danger_map, active_explosions)

    grid[9] = danger_map

    return grid.astype(np.float32)


