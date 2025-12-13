import pickle
from typing import Tuple, List

import numpy as np

import settings as s
from agents import Agent
from environment import GenericWorld, WorldArgs
from fallbacks import pygame
from items import Coin


class ReplayWorld(GenericWorld):
    def __init__(self, args: WorldArgs):
        super().__init__(args)

        replay_file = args.replay
        self.logger.info(f'Loading replay file "{replay_file}"')
        self.replay_file = replay_file
        with open(replay_file, 'rb') as f:
            self.loaded_replay = pickle.load(f)
        if not 'n_steps' in self.loaded_replay:
            self.loaded_replay['n_steps'] = s.MAX_STEPS

        pygame.display.set_caption(f'{replay_file}')

        # Recreate the agents
        agents = []
        for name, _, b, xy in self.loaded_replay["agents"]:
            avatar_sprite_desc = bomb_sprite_desc = self.colors.pop()
            if "display_names" in self.loaded_replay:
                display_name = self.loaded_replay["display_names"][name]
                if name in self.loaded_replay["avatars"]:
                    avatar_sprite_desc = self.loaded_replay["avatars"][name]
                if name in self.loaded_replay["bombs"]:
                    bomb_sprite_desc = self.loaded_replay["bombs"][name]
            else:
                display_name = name
            agents.append(ReplayAgent(name, display_name, avatar_sprite_desc, bomb_sprite_desc))
        self.agents = agents

    def build_arena(self) -> Tuple[np.array, List[Coin], List[Agent]]:
        arena = np.array(self.loaded_replay['arena'])

        coins = []
        for xy in self.loaded_replay['coins']:
            if arena[xy] == 0:
                coins.append(Coin(xy, True))
            else:
                coins.append(Coin(xy, False))

        agents = []
        for i, agent in enumerate(self.agents):
            agents.append(agent)
            agent.x, agent.y = self.loaded_replay['agents'][i][-1]

        return arena, coins, agents

    def poll_and_run_agents(self):
        # Perform recorded agent actions
        perm = self.loaded_replay['permutations'][self.step - 1]
        self.replay['permutations'].append(perm)
        
        # Build a name-to-agent mapping for safer access
        # (in case agents died and active_agents length changed)
        active_agent_names = {a.name: a for a in self.active_agents}
        
        for i in perm:
            # Check if index is valid (agents may have died)
            if i >= len(self.active_agents):
                # Index out of range - try to find agent by name from original replay
                # Find agent name from original step's actions
                step_actions = {}
                for agent_name in self.loaded_replay['actions'].keys():
                    if self.step - 1 < len(self.loaded_replay['actions'][agent_name]):
                        step_actions[agent_name] = self.loaded_replay['actions'][agent_name][self.step - 1]
                
                # Skip this iteration - agent probably died before this step
                continue
            
            a = self.active_agents[i]

            # Double-check agent is still alive and in active_agent_names
            if a.name not in active_agent_names or a.dead:
                continue

            # Check if action exists for this step
            agent_actions = self.loaded_replay['actions'].get(a.name, [])
            if self.step - 1 >= len(agent_actions):
                # No more actions for this agent (probably died earlier)
                continue

            self.logger.debug(f'Repeating action from agent <{a.name}>')
            action = agent_actions[self.step - 1]
            self.logger.info(f'Agent <{a.name}> chose action {action}.')
            self.replay['actions'][a.name].append(action)
            self.perform_agent_action(a, action)

    def time_to_stop(self):
        time_to_stop = super().time_to_stop()
        if self.step == self.loaded_replay['n_steps']:
            self.logger.info('Replay ends here, wrap up round')
            time_to_stop = True
        return time_to_stop
    
    def get_state_for_agent(self, agent: Agent):
        """Get game state for an agent (copied from BombeRLeWorld)"""
        if agent.dead:
            return None

        state = {
            'round': self.round,
            'step': self.step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
            'user_input': getattr(self, 'user_input', None),
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            if exp.is_dangerous():
                for (x, y) in exp.blast_coords:
                    explosion_map[x, y] = max(explosion_map[x, y], exp.timer - 1)
        state['explosion_map'] = explosion_map

        return state


class ReplayAgent(Agent):
    """
    Agents class firing off a predefined sequence of actions.
    """

    def __init__(self, name, display_name, avatar_sprite_desc, bomb_sprite_desc):
        """Recreate the agent as it was at the beginning of the original game."""
        super().__init__(name, None, display_name, False, None, avatar_sprite_desc, bomb_sprite_desc)

    def setup(self):
        pass

    def act(self, game_state):
        pass

    def wait_for_act(self):
        return 0, self.actions.popleft()
