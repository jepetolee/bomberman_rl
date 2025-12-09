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
    is_attacker = True  # Changed: All agents attack (was: role % 2 == 0)
    
    # Create bomb danger map (like rule_based)
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    
    # Loop detection
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
    
    # ===== BUILD ACTION IDEAS (stack - last added = highest priority) =====
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)
    
    # Compile targets
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[1] - 1)
    dead_ends = [(ix, iy) for ix in cols for iy in rows if (arena[ix, iy] == 0)
                 and ([arena[ix + 1, iy], arena[ix - 1, iy], arena[ix, iy + 1], arena[ix, iy - 1]].count(0) == 1)]
    crates = [(ix, iy) for ix in cols for iy in rows if (arena[ix, iy] == 1)]
    
    # Partition coins with teammates
    my_coins = partition_coins(coins, teammates, (x, y))
    
    # Build target list based on role
    targets = []
    
    # AGGRESSIVE: ALWAYS prioritize enemies first!
    if enemies:
        # Put enemies first in target list - maximum aggression
        targets.extend(enemies)
        targets.extend(enemies)  # Double priority for enemies
        targets.extend(my_coins)
        targets.extend(crates)
    else:
        targets.extend(my_coins)
        targets.extend(dead_ends)
        targets.extend(crates)
    
    # Exclude targets on bombs
    targets = [t for t in targets if t not in bomb_xys]
    
    # Navigate to targets
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in all_others:
            free_space[o] = False
    
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
    
    # Bomb at dead end (trap crates)
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    
    # Bomb next to crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')
    
    # AGGRESSIVE: Bomb when enemy is within 5 tiles (very aggressive range)
    if enemies:
        min_enemy_dist = min(abs(ex - x) + abs(ey - y) for ex, ey in enemies)
        if min_enemy_dist <= 3:
            # Check if enemy is in line of sight
            for ex, ey in enemies:
                if (ex == x or ey == y) and min_enemy_dist <= 5:
                    blocked = False
                    if ex == x:
                        for dy in range(min(y, ey) + 1, max(y, ey)):
                            if arena[x, dy] != 0:
                                blocked = True
                                break
                    else:
                        for dx in range(min(x, ex) + 1, max(x, ex)):
                            if arena[dx, y] != 0:
                                blocked = True
                                break
                    if not blocked and not teammate_in_blast((x, y), teammates):
                        action_ideas.append('BOMB')
                        action_ideas.append('BOMB')  # Higher priority
                        break
    
    # AGGRESSIVE: Bomb when touching enemy or very close!
    if enemies:
        min_enemy_dist = min(abs(ex - x) + abs(ey - y) for ex, ey in enemies)
        # EXTREMELY aggressive: bomb if enemy is within 4 tiles (was 2)
        if min_enemy_dist <= 4:
            if not teammate_in_blast((x, y), teammates):
                action_ideas.append('BOMB')
                action_ideas.append('BOMB')  # Higher priority
                action_ideas.append('BOMB')  # Even higher priority for aggression
                action_ideas.append('BOMB')  # Maximum priority
    
    # AGGRESSIVE: Bomb when enemy is in blast range
    if enemies:
        for ex, ey in enemies:
            if (ex == x and abs(ey - y) <= s.BOMB_POWER) or \
               (ey == y and abs(ex - x) <= s.BOMB_POWER):
                blocked = False
                if ex == x:
                    for dy in range(min(y, ey) + 1, max(y, ey)):
                        if arena[x, dy] != 0:
                            blocked = True
                            break
                else:
                    for dx in range(min(x, ex) + 1, max(x, ex)):
                        if arena[dx, y] != 0:
                            blocked = True
                            break
                
                if not blocked and not teammate_in_blast((x, y), teammates):
                    action_ideas.append('BOMB')
                    action_ideas.append('BOMB')  # Higher priority
                    break
    
    # AGGRESSIVE: Predict enemy movement and bomb ahead
    if enemies:
        for ex, ey in enemies:
            # Predict: if enemy is moving, bomb where they might go
            # Check if enemy is near a corner or dead end
            enemy_near_corner = False
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = ex + dx, ey + dy
                if (0 < nx < arena.shape[0] - 1 and 0 < ny < arena.shape[1] - 1):
                    if arena[nx, ny] != 0:  # Wall or crate
                        enemy_near_corner = True
                        break
            
            # If enemy is near corner, bomb to trap them
            if enemy_near_corner:
                # Find attack position that can hit enemy even if they move
                for attack_dist in range(1, s.BOMB_POWER + 1):
                    for pos in [(ex + attack_dist, ey), (ex - attack_dist, ey), 
                               (ex, ey + attack_dist), (ex, ey - attack_dist)]:
                        px, py = pos
                        if (0 < px < arena.shape[0] - 1 and 0 < py < arena.shape[1] - 1 and
                            arena[px, py] == 0 and pos not in all_others and pos not in bomb_xys):
                            # Check if we're close to this position
                            dist_to_pos = abs(px - x) + abs(py - y)
                            if dist_to_pos <= 3:  # Close enough to attack
                                # Check line of sight
                                blocked = False
                                if px == ex:
                                    for check_y in range(min(py, ey) + 1, max(py, ey)):
                                        if arena[ex, check_y] != 0:
                                            blocked = True
                                            break
                                else:
                                    for check_x in range(min(px, ex) + 1, max(px, ex)):
                                        if arena[check_x, ey] != 0:
                                            blocked = True
                                            break
                                if not blocked and not teammate_in_blast((px, py), teammates):
                                    # Navigate to this position
                                    next_step = a_star_to_target((x, y), pos, arena, bomb_map, all_others, bomb_xys_set)
                                    if next_step:
                                        nx, ny = next_step
                                        if nx == x - 1:
                                            action_ideas.append('LEFT')
                                        elif nx == x + 1:
                                            action_ideas.append('RIGHT')
                                        elif ny == y - 1:
                                            action_ideas.append('UP')
                                        elif ny == y + 1:
                                            action_ideas.append('DOWN')
                                    # If already at position, bomb
                                    if dist_to_pos <= 1:
                                        action_ideas.append('BOMB')
                                    break
                if action_ideas and action_ideas[-1] in ['LEFT', 'RIGHT', 'UP', 'DOWN', 'BOMB']:
                    break
    
    # AGGRESSIVE: Navigate towards attack position on enemy (ALL agents attack)
    if enemies:  # Removed is_attacker check - all agents hunt
        closest_enemy = min(enemies, key=lambda e: abs(e[0] - x) + abs(e[1] - y))
        ex, ey = closest_enemy
        enemy_dist = abs(ex - x) + abs(ey - y)
        
        # Check if enemy is cornered (limited escape routes)
        enemy_escape_routes = 0
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = ex + dx, ey + dy
            if (0 < nx < arena.shape[0] - 1 and 0 < ny < arena.shape[1] - 1 and
                arena[nx, ny] == 0 and (nx, ny) not in all_others and (nx, ny) not in bomb_xys):
                enemy_escape_routes += 1
        
        enemy_is_cornered = enemy_escape_routes <= 2
        
        # TEAM COORDINATION: Check if teammate is also attacking same enemy
        teammate_attacking = False
        if teammates:
            for tx, ty in teammates:
                teammate_enemy_dist = min(abs(ex - tx) + abs(ey - ty) for ex, ey in enemies) if enemies else 999
                if teammate_enemy_dist <= 5:  # Teammate is also close to enemy
                    teammate_attacking = True
                    break
        
        # If enemy is nearby or cornered, or teammate is attacking, try to get into attack position
        # Increased range for more aggressive hunting (was 8, now 15)
        # Also attack if teammate is nearby (coordination)
        if enemy_dist <= 15 or enemy_is_cornered or teammate_attacking:
            attack_positions = []
            
            # Check all positions that can hit the enemy
            for i in range(1, s.BOMB_POWER + 1):
                for pos in [(ex + i, ey), (ex - i, ey), (ex, ey + i), (ex, ey - i)]:
                    px, py = pos
                    if (0 < px < arena.shape[0] - 1 and 0 < py < arena.shape[1] - 1 and
                        arena[px, py] == 0 and pos not in all_others and pos not in bomb_xys):
                        # Check path is clear
                        blocked = False
                        if px == ex:  # Same column
                            for check_y in range(min(py, ey) + 1, max(py, ey)):
                                if arena[ex, check_y] != 0:
                                    blocked = True
                                    break
                        else:  # Same row
                            for check_x in range(min(px, ex) + 1, max(px, ex)):
                                if arena[check_x, ey] != 0:
                                    blocked = True
                                    break
                        if not blocked:
                            dist = abs(px - x) + abs(py - y)
                            # Prefer positions where enemy is cornered
                            priority = dist - (10 if enemy_is_cornered else 0)
                            attack_positions.append((pos, priority))
            
            if attack_positions:
                attack_positions.sort(key=lambda ap: ap[1])
                target_pos = attack_positions[0][0]
                
                next_step = a_star_to_target((x, y), target_pos, arena, bomb_map, all_others, bomb_xys_set)
                if next_step:
                    nx, ny = next_step
                    if nx == x - 1:
                        action_ideas.append('LEFT')
                    elif nx == x + 1:
                        action_ideas.append('RIGHT')
                    elif ny == y - 1:
                        action_ideas.append('UP')
                    elif ny == y + 1:
                        action_ideas.append('DOWN')
        
        # If at attack position and enemy is cornered, definitely bomb
        if enemy_is_cornered and enemy_dist <= s.BOMB_POWER:
            if (ex == x or ey == y):
                blocked = False
                if ex == x:
                    for dy in range(min(y, ey) + 1, max(y, ey)):
                        if arena[x, dy] != 0:
                            blocked = True
                            break
                else:
                    for dx in range(min(x, ex) + 1, max(x, ex)):
                        if arena[dx, y] != 0:
                            blocked = True
                            break
                if not blocked and not teammate_in_blast((x, y), teammates):
                    action_ideas.append('BOMB')
                    action_ideas.append('BOMB')  # Very high priority
        
        # AGGRESSIVE: If teammate is attacking same enemy, coordinate attack
        if teammate_attacking and enemy_dist <= s.BOMB_POWER + 1:
            # Coordinate: if teammate is on one side, we attack from other side
            if (ex == x or ey == y):
                blocked = False
                if ex == x:
                    for dy in range(min(y, ey) + 1, max(y, ey)):
                        if arena[x, dy] != 0:
                            blocked = True
                            break
                else:
                    for dx in range(min(x, ex) + 1, max(x, ex)):
                        if arena[dx, y] != 0:
                            blocked = True
                            break
                if not blocked and not teammate_in_blast((x, y), teammates):
                    action_ideas.append('BOMB')
                    action_ideas.append('BOMB')  # High priority for coordinated attack
    
    # ===== ESCAPE FROM BOMBS (HIGHEST PRIORITY) =====
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Same column - run vertically away
            if (yb > y):
                action_ideas.append('UP')
            if (yb < y):
                action_ideas.append('DOWN')
            # Turn corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Same row - run horizontally away
            if (xb > x):
                action_ideas.append('LEFT')
            if (xb < x):
                action_ideas.append('RIGHT')
            # Turn corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    
    # On top of bomb - escape any direction
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])
    
    # ===== TEAMMATE BOMB SAFETY CHECK =====
    def bomb_safe_for_team() -> bool:
        if not enemies:
            return False
        return not teammate_in_blast((x, y), teammates)
    
    # ===== SELECT ACTION (last valid wins) =====
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        
        # Extra safety check for BOMB action - MAXIMUM AGGRESSION
        if a == 'BOMB':
            # Only check teammate safety - NO escape route check for maximum kills!
            if not bomb_safe_for_team():
                continue
            # REMOVED all escape route checks - be extremely aggressive!
            # If there's an enemy nearby, bomb regardless of escape route
            # This maximizes kill opportunities
        
        if a in valid_actions:
            if a == 'BOMB':
                self.bomb_history.append((x, y))
            return a
    
    return 'WAIT'
