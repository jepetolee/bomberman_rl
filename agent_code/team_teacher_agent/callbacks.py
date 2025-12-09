from collections import deque
from random import shuffle
from typing import List, Tuple

import numpy as np

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def _team_tag(name: str) -> str:
    if not name:
        return ""
    return str(name).split('_')[0]


def _agent_suffix(name: str) -> int:
    try:
        return int(str(name).split('_', 1)[1])
    except Exception:
        return 0


def look_for_targets(free_space, start, targets, logger=None):
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


def setup(self):
    self.logger.debug('Team teacher setup')
    np.random.seed()
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0
    self.current_round = 0


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0


def _teammate_positions(self_tag: str, others: List[Tuple]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    teammates = []
    enemies = []
    for other in others:
        name, _score, _bombs_left, pos = other
        if _team_tag(name) == self_tag:
            teammates.append(pos)
        else:
            enemies.append(pos)
    return teammates, enemies


def _coin_partition(coins: List[Tuple[int, int]], teammates: List[Tuple[int, int]], self_pos: Tuple[int, int]):
    if not coins or not teammates:
        return coins
    sx, sy = self_pos
    keep = []
    for cx, cy in coins:
        self_dist = abs(cx - sx) + abs(cy - sy)
        teammate_best = min(abs(cx - tx) + abs(cy - ty) for tx, ty in teammates)
        if self_dist <= teammate_best:
            keep.append((cx, cy))
    return keep if keep else coins


def _teammate_in_blast(center: Tuple[int, int], teammates: List[Tuple[int, int]], radius: int = s.BOMB_POWER) -> bool:
    if not teammates:
        return False
    cx, cy = center
    for tx, ty in teammates:
        if tx == cx and abs(ty - cy) <= radius:
            return True
        if ty == cy and abs(tx - cx) <= radius:
            return True
    return False


def act(self, game_state):
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    arena = game_state['field']
    self_name, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, _timer) in bombs]
    others = game_state['others']
    coins = game_state['coins']

    self_tag = _team_tag(self_name)
    teammates, enemies = _teammate_positions(self_tag, others)

    role = _agent_suffix(self_name)
    is_support = role % 2 == 1

    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    occupied = [pos for _n, _s, _b, pos in others]
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (d not in occupied) and
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

    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[1] - 1)
    dead_ends = [(ix, iy) for ix in cols for iy in rows if (arena[ix, iy] == 0) and
                 ([arena[ix + 1, iy], arena[ix - 1, iy], arena[ix, iy + 1], arena[ix, iy - 1]].count(0) == 1)]
    crates = [(ix, iy) for ix in cols for iy in rows if (arena[ix, iy] == 1)]

    teammate_positions = teammates
    filtered_coins = _coin_partition(coins, teammate_positions, (x, y))

    targets: List[Tuple[int, int]] = []
    targets.extend(filtered_coins)
    targets.extend(dead_ends)
    targets.extend(crates)

    if not is_support or len(filtered_coins) + len(crates) == 0:
        targets.extend([pos for pos in enemies])

    targets = [t for t in targets if t not in bomb_xys]

    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for pos in occupied:
            free_space[pos] = False
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

    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    if enemies:
        if min(abs(ex - x) + abs(ey - y) for ex, ey in enemies) <= 1:
            action_ideas.append('BOMB')
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            if (yb > y):
                action_ideas.append('UP')
            if (yb < y):
                action_ideas.append('DOWN')
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            if (xb > x):
                action_ideas.append('LEFT')
            if (xb < x):
                action_ideas.append('RIGHT')
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    def bomb_safe_for_team() -> bool:
        if not enemies:
            return False
        return not _teammate_in_blast((x, y), teammate_positions)

    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a == 'BOMB' and not bomb_safe_for_team():
            continue
        if a in valid_actions:
            if a == 'BOMB':
                self.bomb_history.append((x, y))
            return a

    return 'WAIT'

