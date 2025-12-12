from typing import List
import os
import json
from pathlib import Path
import numpy as np
from collections import deque
import settings as s

import torch
import torch.nn as nn
import torch.optim as optim

import events as e
from .callbacks import state_to_features, ACTIONS, SHARED

def get_distance_to_nearest_enemy(game_state):
    if game_state is None:
        return float('inf')

    field = game_state['field']
    _, _, _, (sx, sy) = game_state['self']
    others = game_state['others']
    bombs = game_state['bombs']

    self_name = game_state['self'][0]
    self_team = self_name.split('_')[0] if self_name else ""
    
    enemy_positions = []
    for o_n, o_s, o_b, (ox, oy) in others:
        other_team = o_n.split('_')[0] if o_n else ""
        if other_team != self_team:
            enemy_positions.append((ox, oy))

    if not enemy_positions:
        return float('inf')

    dist_map = np.full((s.COLS, s.ROWS), float('inf'))
    queue = deque()

    for ex, ey in enemy_positions:
        dist_map[ex, ey] = 0
        queue.append((ex, ey, 0))

    bomb_locs = { (bx, by) for (bx, by), t in bombs }

    while queue:
        x, y, d = queue.popleft()

        if x == sx and y == sy:
            return d

        next_dist = d + 1
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < s.COLS and 0 <= ny < s.ROWS:
                if dist_map[nx, ny] == float('inf') and field[nx, ny] == 0 and (nx, ny) not in bomb_locs:
                    dist_map[nx, ny] = next_dist
                    queue.append((nx, ny, next_dist))
    
    return float('inf')

def setup_training(self):
    # Share hyperparameters and optimizer across agent instances
    SHARED.gamma = 0.99
    SHARED.gae_lambda = 0.95
    SHARED.clip_range = 0.2
    SHARED.ent_coef = 0.3            # Higher entropy for more exploration (was 0.2)
    SHARED.vf_coef = 0.5
    SHARED.max_grad_norm = 0.5
    SHARED.update_epochs = 6         # More update epochs for better learning (was 4)
    SHARED.batch_size = 128          # Smaller batch for more frequent updates (was 256)
    SHARED.learning_rate = 5e-4      # Slightly higher learning rate (was 3e-4)

    if SHARED.optimizer is None:
        SHARED.optimizer = optim.Adam(SHARED.policy.parameters(), lr=SHARED.learning_rate)


def _reward_from_events(self, events: List[str]) -> float:
    # Aggressive reward shaping - HEAVILY prioritize kills!
    game_rewards = {
        # Primary objectives - VERY HIGH rewards for kills
        e.KILLED_OPPONENT: 50.0,      # MASSIVE reward for kills! (was 10.0)
        e.COIN_COLLECTED: 1.0,
        e.CRATE_DESTROYED: 0.3,       # Encourage destruction
        e.SURVIVED_ROUND: 1.0,
        
        # Encourage bomb usage - critical for kills!
        e.BOMB_DROPPED: 5.0,          # Much higher reward for dropping bombs (was 2.0)
        e.COIN_FOUND: 0.1,
        
        # Penalties
        e.KILLED_TEAMMATE: -10.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -10.0,          # Higher penalty for dying (was -3.0)
        e.INVALID_ACTION: -0.1,       # Stronger penalty for invalid
        
        # Movement - encourage activity
        e.WAITED: -0.2,               # Stronger penalty for waiting (was -0.1)
        e.MOVED_UP: 0.05,             # More reward for movement (was 0.02)
        e.MOVED_DOWN: 0.05,
        e.MOVED_LEFT: 0.05,
        e.MOVED_RIGHT: 0.05,
    }
    # Configure suicide penalty schedule
    try:
        completed = getattr(SHARED, 'completed_rounds', 0)
        suppress_until = getattr(SHARED, 'suppress_suicide_until', 0)
        switch_round = getattr(SHARED, 'suicide_penalty_switch_round', 800)
        high_penalty = getattr(SHARED, 'suicide_penalty_high', -15.0)
        # Default 'after' penalty equals negative magnitude of opponent kill reward
        default_after = abs(game_rewards[e.KILLED_OPPONENT])
        after_penalty = getattr(SHARED, 'suicide_penalty_after', default_after)
        if not (after_penalty == after_penalty):  # NaN check
            after_penalty = default_after

        if completed < suppress_until:
            suicide_penalty = 0.0
        else:
            suicide_penalty = high_penalty if completed < switch_round else after_penalty
        game_rewards[e.KILLED_SELF] = float(suicide_penalty)
    except Exception:
        pass
    reward = 0.0
    for ev in events:
        reward += game_rewards.get(ev, 0.0)
    return float(reward)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    # We rely on values cached in callbacks.act()
    if self._last_state_tensor is None or self._last_action_idx is None:
        return

    reward = _reward_from_events(self, events)

    if old_game_state is not None and new_game_state is not None:
        dist_old = get_distance_to_nearest_enemy(old_game_state)
        dist_new = get_distance_to_nearest_enemy(new_game_state)
        
        if dist_old != float('inf') and dist_new != float('inf'):
            if dist_new < dist_old:
                shaping_reward = 0.5 # Approaching reward
                reward += shaping_reward
                self.logger.debug(f"Approaching enemy: +{shaping_reward} (dist {dist_old} -> {dist_new})")
            
            elif dist_new > dist_old:
                shaping_penalty = -0.2 # Retreating reward
                reward += shaping_penalty
                self.logger.debug(f"Retreating: {shaping_penalty} (dist {dist_old} -> {dist_new})")

    done = False  # terminal handled in end_of_round

    # Store step into shared buffers
    SHARED.buf_states.append(self._last_state_tensor.squeeze(0))  # [C,H,W]
    SHARED.buf_actions.append(self._last_action_idx)
    SHARED.buf_logps.append(self._last_log_prob)
    SHARED.buf_values.append(self._last_value)
    SHARED.buf_rewards.append(reward)
    SHARED.buf_dones.append(done)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Final reward for last step
    final_reward = _reward_from_events(self, events)
    
    # Add win/loss penalty based on final score comparison
    # This encourages winning and penalizes losing
    if last_game_state is not None:
        try:
            # Get our team's total score
            self_name, self_score, _, _ = last_game_state.get('self', (None, 0, 0, (0, 0)))
            others_data = last_game_state.get('others', [])
            
            # Calculate team scores
            self_tag = str(self_name).split('_')[0] if self_name else ""
            our_team_score = self_score
            enemy_team_score = 0
            
            for other_name, other_score, _, _ in others_data:
                other_tag = str(other_name).split('_')[0] if other_name else ""
                if other_tag == self_tag:
                    our_team_score += other_score
                else:
                    enemy_team_score += other_score
            
            # Add win/loss reward - MUCH HIGHER to emphasize winning
            if our_team_score > enemy_team_score:
                # Win bonus - significantly increased
                final_reward += 50.0  # Increased from 5.0
            elif our_team_score < enemy_team_score:
                # Loss penalty - significantly increased to discourage losing
                final_reward -= 50.0  # Increased from 10.0
            # Draw: no extra reward/penalty
        except Exception:
            pass  # Silently ignore errors in score calculation
    
    if SHARED.buf_rewards:
        # Fold terminal bonus/penalty into the final transition instead of creating a duplicate entry
        SHARED.buf_rewards[-1] += final_reward
        SHARED.buf_dones[-1] = True
    elif self._last_state_tensor is not None and self._last_action_idx is not None:
        # Fallback for edge cases where nothing was buffered (e.g., agent never acted)
        SHARED.buf_states.append(self._last_state_tensor.squeeze(0))
        SHARED.buf_actions.append(self._last_action_idx)
        SHARED.buf_logps.append(self._last_log_prob)
        SHARED.buf_values.append(self._last_value)
        SHARED.buf_rewards.append(final_reward)
        SHARED.buf_dones.append(True)

    # Coordinate update: run once when all our shared instances have finished this round
    round_id = last_game_state['round'] if last_game_state is not None else None
    if round_id is not None:
        count = SHARED.round_done_counts.get(round_id, 0) + 1
        SHARED.round_done_counts[round_id] = count
        if count >= len(SHARED.instance_ids):
            if len(SHARED.buf_states) > 1:
                _ppo_update()
                try:
                    torch.save(SHARED.policy.state_dict(), SHARED.model_path)
                    # Optional periodic checkpoints
                    SHARED.save_round_counter += 1
                    if getattr(SHARED, 'save_every_rounds', 0) and SHARED.save_round_counter % max(1, SHARED.save_every_rounds) == 0:
                        ckpt_dir = SHARED.checkpoint_dir or os.path.join(os.path.dirname(SHARED.model_path) or '.', 'checkpoints')
                        try:
                            os.makedirs(ckpt_dir, exist_ok=True)
                        except Exception:
                            pass
                        ckpt_path = os.path.join(ckpt_dir, f"ckpt_round_{SHARED.save_round_counter:06d}.pt")
                        torch.save(SHARED.policy.state_dict(), ckpt_path)
                        try:
                            self.logger.info(f"Saved PPO checkpoint: {ckpt_path}")
                        except Exception:
                            pass
                except Exception as ex:
                    self.logger.warning(f"Failed to save PPO model: {ex}")
            # One full round completed across all our instances
            try:
                SHARED.completed_rounds += 1
                # Calculate teacher stats every round if we have data (for faster feedback)
                if SHARED.total_action_count > 0:
                    teacher_usage = SHARED.teacher_action_count / SHARED.total_action_count * 100
                    current_eps = SHARED.current_epsilon()
                    invalid_rate = SHARED.teacher_invalid_count / max(1, SHARED.teacher_action_count + SHARED.teacher_invalid_count) * 100
                    # Store in shared state
                    SHARED.last_teacher_stats = {
                        'round': round_id,
                        'usage': teacher_usage,
                        'epsilon': current_eps,
                        'invalid_rate': invalid_rate,
                        'total_actions': SHARED.total_action_count,
                        'teacher_actions': SHARED.teacher_action_count
                    }
                    # Save to file every round (so worker can always read latest)
                    # Save to ONE location only: A3C results_dir if available, otherwise model directory
                    try:
                        # Priority 1: A3C results directory (set by worker) - MUST be absolute path
                        stats_file = None
                        a3c_results_dir = os.environ.get('A3C_RESULTS_DIR', None)
                        if a3c_results_dir:
                            # Convert to absolute path to avoid working directory issues
                            a3c_results_dir = os.path.abspath(a3c_results_dir)
                            stats_file = os.path.join(a3c_results_dir, 'teacher_stats.json')
                        else:
                            # Priority 2: Model directory
                            stats_file = SHARED._stats_file or os.path.join(
                                os.path.dirname(SHARED.model_path) if os.path.dirname(SHARED.model_path) else '.',
                                'teacher_stats.json'
                            )
                            # Also convert to absolute path
                            stats_file = os.path.abspath(stats_file)
                        
                        # Ensure directory exists
                        stats_dir = os.path.dirname(stats_file) or '.'
                        os.makedirs(stats_dir, exist_ok=True)
                        with open(stats_file, 'w') as f:
                            json.dump(SHARED.last_teacher_stats, f)
                        
                        # Log first time to confirm it's working
                        if SHARED.completed_rounds == 1:
                            try:
                                self.logger.info(f"📚 Teacher stats tracking started: {teacher_usage:.1f}% (eps={current_eps:.3f}, total={SHARED.total_action_count})")
                                self.logger.info(f"📚 Stats saved to: {stats_file}")
                            except:
                                pass
                    except Exception as e:
                        try:
                            if SHARED.completed_rounds <= 10:  # Log first few errors
                                self.logger.warning(f"Failed to save teacher stats: {e}")
                        except:
                            pass
            except Exception:
                pass
            # Clear for next round
            SHARED.reset_buffers()
            # Cleanup count to avoid growth
            try:
                del SHARED.round_done_counts[round_id]
            except KeyError:
                pass


def _compute_gae(rewards, values, dones, *, gamma: float, gae_lambda: float):
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if (t == T - 1) else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(T)]
    return torch.tensor(adv, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def _ppo_update():
    SHARED.policy.train()

    device = SHARED.device

    states = torch.stack(SHARED.buf_states).to(device)           # [T, C, H, W]
    actions = torch.tensor(SHARED.buf_actions, dtype=torch.long, device=device)  # [T]
    old_logps = torch.tensor(SHARED.buf_logps, dtype=torch.float32, device=device)  # [T]
    values = torch.tensor(SHARED.buf_values, dtype=torch.float32, device=device)    # [T]
    rewards = torch.tensor(SHARED.buf_rewards, dtype=torch.float32, device=device)  # [T]
    dones = torch.tensor(SHARED.buf_dones, dtype=torch.bool, device=device)         # [T]

    adv, returns = _compute_gae(
        rewards.tolist(), values.tolist(), dones.tolist(),
        gamma=SHARED.gamma, gae_lambda=SHARED.gae_lambda
    )
    adv = adv.to(device)
    returns = returns.to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    T = states.size(0)
    idxs = torch.arange(T)

    for _ in range(SHARED.update_epochs):
        perm = torch.randperm(T)
        for start in range(0, T, SHARED.batch_size):
            mb_idx = perm[start:start + SHARED.batch_size]
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logps = old_logps[mb_idx]
            mb_adv = adv[mb_idx]
            mb_returns = returns[mb_idx]

            logits, values_pred = SHARED.policy(mb_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logps - mb_old_logps)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - SHARED.clip_range, 1.0 + SHARED.clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values_pred, mb_returns)

            loss = policy_loss + SHARED.vf_coef * value_loss - SHARED.ent_coef * entropy

            SHARED.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(SHARED.policy.parameters(), SHARED.max_grad_norm)
            SHARED.optimizer.step()

