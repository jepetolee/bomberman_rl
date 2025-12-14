"""
Aggressive Team Teacher Agent for Bomberman RL
- Team cooperation (2 agents per team)
- A* pathfinding for hunting enemies
- Priority-stack system for action selection (like rule_based)
- Aggressive enemy targeting with teammate protection
"""

from collections import deque
from random import shuffle
from typing import List, Tuple, Optional
import heapq
import numpy as np

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# ==================== TEAM UTILITIES ====================

def _team_tag(name: str) -> str:
    """Extract team tag from agent name."""
    if not name:
        return ""
    return str(name).split('_')[0]


def _agent_suffix(name: str) -> int:
    """Get agent's role number within team."""
    try:
        return int(str(name).split('_', 1)[1])
    except Exception:
        return 0


def get_team_info(self_name: str, others: List[Tuple]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Separate teammates and enemies."""
    self_tag = _team_tag(self_name)
    teammates = []
    enemies = []
    
    for name, _score, _bombs_left, pos in others:
        if _team_tag(name) == self_tag:
            teammates.append(pos)
        else:
            enemies.append(pos)
    
    return teammates, enemies


def teammate_in_blast(pos: Tuple[int, int], teammates: List[Tuple[int, int]], 
                       power: int = s.BOMB_POWER) -> bool:
    """Check if any teammate would be in blast zone."""
    if not teammates:
        return False
    
    cx, cy = pos
    for tx, ty in teammates:
        if tx == cx and abs(ty - cy) <= power:
            return True
        if ty == cy and abs(tx - cx) <= power:
            return True
    return False


def partition_coins(coins: List[Tuple[int, int]], teammates: List[Tuple[int, int]], 
                    self_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Assign coins to closest agent."""
    if not coins or not teammates:
        return coins
    
    sx, sy = self_pos
    my_coins = []
    
    for cx, cy in coins:
        my_dist = abs(cx - sx) + abs(cy - sy)
        teammate_dists = [abs(cx - tx) + abs(cy - ty) for tx, ty in teammates]
        if not teammate_dists or my_dist <= min(teammate_dists):
            my_coins.append((cx, cy))
    
    return my_coins if my_coins else coins


# ==================== PATHFINDING ====================

def look_for_targets(free_space: np.ndarray, start: Tuple[int, int], 
                      targets: List[Tuple[int, int]], logger=None) -> Optional[Tuple[int, int]]:
    """BFS to find first step towards nearest target."""
    if len(targets) == 0:
        return None
    
    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    
    while len(frontier) > 0:
        current = frontier.pop(0)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    
    if logger:
        logger.debug(f'Target found at {best}')
    
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def a_star_to_target(start: Tuple[int, int], goal: Tuple[int, int], 
                      arena: np.ndarray, bomb_map: np.ndarray,
                      others: List[Tuple[int, int]], bomb_xys: set) -> Optional[Tuple[int, int]]:
    """A* pathfinding avoiding dangerous tiles."""
    if start == goal:
        return start
    
    counter = 0
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    open_set = [(h(start), counter, start)]
    came_from = {start: None}
    g_score = {start: 0}
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path[1] if len(path) > 1 else path[0]
        
        cx, cy = current
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            
            if not (0 < nx < arena.shape[0] - 1 and 0 < ny < arena.shape[1] - 1):
                continue
            if arena[nx, ny] != 0:
                continue
            if neighbor in others or neighbor in bomb_xys:
                continue
            
            # High cost for dangerous tiles
            move_cost = 1
            if bomb_map[nx, ny] < 5:
                move_cost += 50
            
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                counter += 1
                heapq.heappush(open_set, (tentative_g + h(neighbor), counter, neighbor))
                came_from[neighbor] = current
    
    return None


# ==================== MAIN AGENT ====================

def setup(self):
    """Initialize agent."""
    self.logger.debug('Aggressive team teacher setup')
    np.random.seed()
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0
    self.current_round = 0


def reset_self(self):
    """Reset for new round."""
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0


def act(self, game_state):
    """Main action logic using priority stack system."""
    # Round reset
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    
    # Extract game state
    arena = game_state['field']
    self_name, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    bomb_xys_set = set(bomb_xys)
    others_data = game_state['others']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    
    # Team info
    teammates, enemies = get_team_info(self_name, others_data)
    all_others = [pos for (_, _, _, pos) in others_data]
    
    # Role - ALL agents are attackers for maximum aggression!
    role = _agent_suffix(self_name)
    # 전략 완화: 모든 에이전트를 무조건 공격자로 두지 않는다.
    # 짝수 에이전트만 공격, 홀수는 보수적 탐색/코인 수집에 집중
    is_attacker = (_agent_suffix(self_name) % 2 == 0)
    
    # Create bomb danger map (like rule_based)
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    
    # Loop detection (EXACTLY like rule_based_agent)
    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))
    
    # ===== DETERMINE VALID TILES/ACTIONS (like rule_based) =====
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    
    for d in directions:
        if ((arena[d] == 0) and
            (explosion_map[d] < 1) and
            (bomb_map[d] > 0) and
            (d not in all_others) and
            (d not in bomb_xys)):
            valid_tiles.append(d)
    
    if (x - 1, y) in valid_tiles:
        valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles:
        valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles:
        valid_actions.append('UP')
    if (x, y + 1) in valid_tiles:
        valid_actions.append('DOWN')
    if (x, y) in valid_tiles:
        valid_actions.append('WAIT')
    if (bombs_left > 0) and (x, y) not in self.bomb_history:
        valid_actions.append('BOMB')
    
    # ===== SAFETY CHECK FUNCTIONS =====
    def has_safe_escape_route(bomb_pos: Tuple[int, int]) -> bool:
        """폭탄 설치 후 탈출 경로가 있는지 확인"""
        # 폭탄이 터질 시간(보통 3-4 스텝) 후에도 안전한 타일이 있는지 확인
        safe_time = 2  # 2스텝 이상 여유가 있어야 함
        
        # 현재 위치에서 도달 가능한 안전한 타일 확인
        escape_candidates = []
        for tile in valid_tiles:
            # 폭탄 범위 밖이거나 충분히 시간 여유가 있는 타일
            tx, ty = tile
            if bomb_map[tx, ty] > safe_time and tile != bomb_pos:
                escape_candidates.append(tile)
        
        # 탈출 후보가 있으면 탈출 가능
        return len(escape_candidates) > 0
    
    def bomb_safe_for_team() -> bool:
        if not enemies:
            return False
        return not teammate_in_blast((x, y), teammates)
    
    # ===== BUILD ACTION IDEAS (stack - last added = highest priority) =====
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)
    
    # Compile targets
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[1] - 1)
    dead_ends = [(ix, iy) for ix in cols for iy in rows if (arena[ix, iy] == 0)
                 and ([arena[ix + 1, iy], arena[ix - 1, iy], arena[ix, iy + 1], arena[ix, iy - 1]].count(0) == 1)]
    crates = [(ix, iy) for ix in cols for iy in rows if (arena[ix, iy] == 1)]
    
    # Build target list - OPTIMIZED for coin collection efficiency
    # Priority: coins > crates > dead_ends > enemies (very conservative)
    targets = []
    
    # OPTIMIZATION: Use ALL coins without partition for maximum collection speed
    # coin_collector_agent doesn't partition - both agents compete naturally
    # BFS will find closest coin to each agent automatically
    targets.extend(coins)     # ALL coins - ABSOLUTE highest priority
    targets.extend(crates)    # Crates second
    targets.extend(dead_ends) # Dead ends for positioning
    
    # VERY CONSERVATIVE: Only add enemies if NO coins/crates available
    # This is more conservative than rule_based_agent for better coin collection
    # Matches coin_collector_agent behavior (never targets enemies)
    if (len(crates) + len(coins) == 0) and enemies:
        targets.extend(enemies)
    
    # Exclude targets on bombs
    targets = [t for t in targets if t not in bomb_xys]
    
    # Navigate to targets - OPTIMIZED for survival and coin collection
    # Balance between safety and efficiency (like team_teacher_agent)
    free_space = arena == 0
    
    # Only block others when stuck in loop - allows faster coin collection
    if self.ignore_others_timer > 0:
        for o in all_others:
            free_space[o] = False
    # CRITICAL: Don't block teammates - allows better coordination
    # Team members can coordinate for faster coin collection
    for tm in teammates:
        if tm in all_others:
            free_space[tm] = True  # Allow teammates to overlap
    
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1):
        action_ideas.append('UP')
    if d == (x, y + 1):
        action_ideas.append('DOWN')
    if d == (x - 1, y):
        action_ideas.append('LEFT')
    if d == (x + 1, y):
        action_ideas.append('RIGHT')
    if d is None:
        action_ideas.append('WAIT')
    
    # ===== BOMB PLACEMENT LOGIC =====
    # Very conservative bomb placement - only when absolutely safe and effective
    
    # Bomb at dead end (OPTIMIZED: check safety first for better survival)
    # Only bomb if we have time to potentially escape (bomb_map shows some safety)
    if (x, y) in dead_ends:
        # Check if we have at least some time (bomb_map > 1 means not immediate danger)
        # This prevents bombing when we're about to die anyway
        if bomb_map[x, y] > 1:
            action_ideas.append('BOMB')
    
    # Bomb next to crate (OPTIMIZED: check safety first)
    # Only bomb if we're not in immediate danger
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        # Check if we have escape route (safe tiles nearby)
        safe_nearby = any(bomb_map[nx, ny] > 2 for nx, ny in valid_tiles if (nx, ny) != (x, y))
        # Only bomb if safe or not in immediate danger
        if safe_nearby or bomb_map[x, y] > 2:
            action_ideas.append('BOMB')
    
    # Bomb when touching enemy (OPTIMIZED: check safety first)
    # Only bomb if safe to do so (team_teacher_agent style - more conservative)
    if enemies:
        min_enemy_dist = min(abs(ex - x) + abs(ey - y) for ex, ey in enemies)
        # Only bomb if enemy is directly adjacent (touching, distance <= 1)
        if min_enemy_dist <= 1:
            # Check teammate safety AND have escape route for better survival
            if not teammate_in_blast((x, y), teammates):
                # Only bomb if we have a safe escape route or touching (mutual kill acceptable)
                safe_escape = any(bomb_map[nx, ny] > 2 for nx, ny in valid_tiles if (nx, ny) != (x, y))
                # Allow if touching (mutual kill risk acceptable) or if safe escape exists
                if safe_escape or min_enemy_dist == 0:
                    action_ideas.append('BOMB')
    
    # REMOVED: Predictive bombing - too risky, causes suicides
    
    # REMOVED: Complex attack positioning - too risky, focus on survival first
    
    # ===== ESCAPE FROM BOMBS (HIGHEST PRIORITY - SURVIVAL FIRST) =====
    # OPTIMIZED: More aggressive escape detection for better survival
    # Check bomb_map for immediate danger and escape early
    bomb_escape_actions = []
    
    # CRITICAL: Check if current position is in danger zone
    # More sensitive detection: <= 3 means we should start moving immediately
    # This helps match team_teacher_agent's survival rate
    current_in_danger = bomb_map[x, y] <= 3  # Start escaping earlier
    
    for (xb, yb), t in bombs:
        # Same column - run vertically away (EXACTLY like rule_based: < 4)
        if (xb == x) and (abs(yb - y) < 4):
            # Run away - HIGHEST priority if in danger
            if (yb > y) and 'UP' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('UP')  # Add to high priority list
                else:
                    action_ideas.append('UP')
            if (yb < y) and 'DOWN' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('DOWN')
                else:
                    action_ideas.append('DOWN')
            # Turn corner to escape
            if 'LEFT' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('LEFT')
                else:
                    action_ideas.append('LEFT')
            if 'RIGHT' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('RIGHT')
                else:
                    action_ideas.append('RIGHT')
        # Same row - run horizontally away
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x) and 'LEFT' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('LEFT')
                else:
                    action_ideas.append('LEFT')
            if (xb < x) and 'RIGHT' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('RIGHT')
                else:
                    action_ideas.append('RIGHT')
            # Turn corner to escape
            if 'UP' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('UP')
                else:
                    action_ideas.append('UP')
            if 'DOWN' in valid_actions:
                if current_in_danger:
                    bomb_escape_actions.append('DOWN')
                else:
                    action_ideas.append('DOWN')
    
    # On top of bomb - escape ANY direction immediately (CRITICAL for survival)
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            # Add all valid movement actions to escape list
            for move in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                if move in valid_actions:
                    bomb_escape_actions.append(move)
    
    # Add escape actions at the END (highest priority) - they will be checked last
    # This ensures escape takes precedence over everything else
    action_ideas.extend(bomb_escape_actions)
    
    # ===== SELECT ACTION (last valid wins) =====
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        
        # Extra safety check for BOMB action - OPTIMIZED for survival
        # Additional checks to prevent suicidal bombs
        if a == 'BOMB':
            # 팀원 안전 확인
            if not bomb_safe_for_team():
                continue
            
            # OPTIMIZATION: Don't bomb if we're in immediate danger
            # Check if current position will be hit by explosion soon
            if bomb_map[x, y] <= 2:  # More conservative: <= 2 means very little time
                # We're about to die, don't waste bomb
                continue
            
            # OPTIMIZATION: Check if we have at least one safe escape route
            # This prevents most suicidal bombs while still allowing strategic bombing
            # More conservative: require bomb_map > 3 for safe escape
            safe_escape_exists = any(bomb_map[nx, ny] > 3 for nx, ny in valid_tiles if (nx, ny) != (x, y))
            if not safe_escape_exists:
                # No safe escape - only allow if touching enemy (mutual kill acceptable)
                min_enemy_dist = min(abs(ex - x) + abs(ey - y) for ex, ey in enemies) if enemies else 999
                if min_enemy_dist > 0:  # Not touching enemy
                    continue  # Skip bomb if no escape and not touching enemy
        
        if a in valid_actions:
            if a == 'BOMB':
                self.bomb_history.append((x, y))
            return a
    
    return 'WAIT'
