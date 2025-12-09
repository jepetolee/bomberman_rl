import os
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time
from tqdm import tqdm
import json

import settings as s
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld
from copy import deepcopy

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class Timekeeper:
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or time() >= self.next_time

    def note(self):
        self.next_time = time() + self.interval

    def wait(self):
        if not self.is_due():
            duration = self.next_time - time()
            sleep(duration)


def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, make_video, update_interval):
    if make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()

    gui_timekeeper = Timekeeper(update_interval)

    def render(wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            gui_timekeeper.wait()

        if gui_timekeeper.is_due():
            gui_timekeeper.note()
            # Render (which takes time)
            gui.render()
            pygame.display.flip()

    user_input = None
    for _ in tqdm(range(n_rounds)):
        world.new_round()
        while world.running:
            # Only render when the last frame is not too old
            if gui is not None:
                render(every_step)

                # Check GUI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in ESCAPE_KEYS:
                            world.end_round()
                        elif key_pressed in s.INPUT_MAP:
                            user_input = s.INPUT_MAP[key_pressed]

            # Advances step (for turn based: only if user input is available)
            if world.running and not (turn_based and user_input is None):
                world.do_step(user_input)
                user_input = None
            else:
                # Might want to wait
                pass

        # Save video of last game
        if make_video:
            gui.make_video()

        # Render end screen until next round is queried
        if gui is not None:
            do_continue = False
            while not do_continue:
                render(True)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in s.INPUT_MAP or key_pressed in ESCAPE_KEYS:
                            do_continue = True

    return world.end()


def main(argv = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
    group.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

    # Curriculum options
    play_parser.add_argument("--curriculum", default=False, action="store_true", help="Enable two-phase curriculum training")
    play_parser.add_argument("--phase1-rounds", type=int, default=200, help="Rounds for phase 1 (vs random)")
    play_parser.add_argument("--phase2-rounds", type=int, default=300, help="Rounds for phase 2 (vs rule-based)")
    play_parser.add_argument("--phase1-opponent", type=str, default="random_agent", help="Opponent code for phase 1")
    play_parser.add_argument("--phase2-opponent", type=str, default="rule_based_agent", help="Opponent code for phase 2")
    play_parser.add_argument("--train-agent", type=str, default="ppo_agent", help="Agent code to train (repeated train times)")

    # Dynamic opponents (epsilon-greedy style scheduling of rule_based)
    play_parser.add_argument("--dynamic-opponents", default=False, action="store_true", help="Per-round random opponents with annealed rule_based probability")
    play_parser.add_argument("--opponent-pool", type=str, nargs="+",
                             default=["coin_collector_agent", "peaceful_agent", "random_agent"],
                             help="Pool of non-rule-based opponent agent codes")
    play_parser.add_argument("--rb-agent", type=str, default="rule_based_agent", help="Rule-based opponent agent code")
    play_parser.add_argument("--rb-prob-start", type=float, default=0.05, help="Initial probability to pick rule_based per opponent slot")
    play_parser.add_argument("--rb-prob-end", type=float, default=0.6, help="Final probability to pick rule_based per opponent slot")
    play_parser.add_argument("--rb-prob-anneal-rounds", type=int, default=None, help="Rounds over which to anneal from start to end; defaults to n-rounds")

    # Progressive curriculum: fail -> peaceful -> coin -> rule (smooth mixture)
    play_parser.add_argument("--progressive-opponents", default=False, action="store_true",
                             help="Use smooth mixture scheduling from fail->peaceful->coin->rule over rounds")
    play_parser.add_argument("--no-fail", default=False, action="store_true",
                             help="Exclude fail_agent from progressive scheduling")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")

    # Initialize environment and agents
    if args.command_name == "play":
        # Dynamic opponents per round (epsilon-greedy style scheduling for rule_based)
        if args.dynamic_opponents or args.progressive_opponents:
            if args.train == 0 and not args.continue_without_training:
                args.continue_without_training = True

            total_rounds = args.n_rounds
            anneal_rounds = args.rb_prob_anneal_rounds or total_rounds

            def rb_prob_for_round(r_idx: int) -> float:
                # Linear annealing from start to end
                frac = min(1.0, max(0.0, r_idx / max(1, anneal_rounds - 1)))
                return args.rb_prob_start + (args.rb_prob_end - args.rb_prob_start) * frac

            # Build GUI once to avoid window flicker
            gui = None
            if has_gui:
                # Create a tiny dummy world to initialize GUI; replaced each loop
                tmp_agents = [(args.train_agent, i < args.train) for i in range(s.MAX_AGENTS)]
                tmp_world = BombeRLeWorld(args, tmp_agents)
                gui = GUI(tmp_world)

            combined_results = []
            aggregate_by_agent = {}
            round_logs = []

            for r in range(total_rounds):
                # Choose opponents for this round
                num_opponents = max(0, s.MAX_AGENTS - args.train)
                chosen_opponents = []
                match_suffix = ""

                if args.progressive_opponents:
                    # Smoothly transition through fail -> peaceful -> random -> coin -> rule anchors
                    t = 0.0 if total_rounds <= 1 else min(1.0, max(0.0, r / float(max(1, total_rounds - 1))))
                    labels = ["fail_agent", "peaceful_agent", "random_agent", "coin_collector_agent", args.rb_agent]
                    if args.no_fail:
                        labels = [lab for lab in labels if lab != "fail_agent"]

                    num_anchors = len(labels)
                    if num_anchors == 0:
                        labels = [args.rb_agent]
                        num_anchors = 1

                    if num_anchors == 1:
                        weights = [1.0]
                    else:
                        stage_pos = min(1.0, max(0.0, t)) * (num_anchors - 1)
                        seg_idx = min(num_anchors - 2, int(stage_pos))
                        frac = stage_pos - seg_idx
                        weights = [0.0] * num_anchors
                        weights[seg_idx] = 1.0 - frac
                        weights[seg_idx + 1] = frac

                    for _ in range(num_opponents):
                        rnd = int.from_bytes(os.urandom(2), 'little') / 65535.0
                        cum = 0.0
                        pick = labels[-1]
                        for w, lab in zip(weights, labels):
                            cum += w
                            if rnd <= cum:
                                pick = lab
                                break
                        chosen_opponents.append(pick)
                    match_suffix = f"prog_t{t:.2f}"
                    schedule_info = {"mode": "progressive", "t": float(f"{t:.4f}")}
                else:
                    # Dynamic epsilon-greedy scheduling for rule-based
                    p_rb = rb_prob_for_round(r)
                    for _ in range(num_opponents):
                        if os.urandom(8)[0] / 255.0 < p_rb:
                            chosen_opponents.append(args.rb_agent)
                        else:
                            pool = [p for p in args.opponent_pool if p != "fail_agent"]
                            if not pool:
                                pool = ["random_agent"]
                            idx = os.urandom(1)[0] % max(1, len(pool))
                            chosen_opponents.append(pool[idx])
                    match_suffix = f"rb{p_rb:.2f}"
                    schedule_info = {"mode": "dynamic", "p_rb": float(f"{p_rb:.4f}")}

                # Compose agent list: first 'train' are training agents; rest are opponents
                agents = []
                for i in range(args.train):
                    agents.append((args.train_agent, True))
                for opp in chosen_opponents:
                    agents.append((opp, False))

                round_args = deepcopy(args)
                round_args.n_rounds = 1
                round_args.silence_errors = True
                # Annotate match name with round and current schedule state for traceability
                if round_args.match_name:
                    round_args.match_name = f"{round_args.match_name}-r{r+1:05d}-{match_suffix}"
                else:
                    round_args.match_name = f"r{r+1:05d}-{match_suffix}"

                world = BombeRLeWorld(round_args, agents)
                every_step = not args.skip_frames
                result = world_controller(world, round_args.n_rounds,
                                 gui=gui, every_step=every_step, turn_based=round_args.turn_based,
                                 make_video=round_args.make_video, update_interval=round_args.update_interval)
                # Collect results
                if isinstance(result, dict):
                    # attach schedule summary and opponents used this round
                    try:
                        result['_schedule'] = schedule_info
                        result['_opponents'] = list(chosen_opponents)
                    except Exception:
                        pass
                    combined_results.append(result)
                    # Aggregate by_agent
                    for agent_name, stats in result.get('by_agent', {}).items():
                        agg = aggregate_by_agent.setdefault(agent_name, {})
                        for k, v in stats.items():
                            try:
                                agg[k] = agg.get(k, 0) + int(v)
                            except Exception:
                                pass
                # Also log a compact per-round record
                try:
                    round_logs.append({
                        'round': r + 1,
                        'opponents': list(chosen_opponents),
                        **schedule_info
                    })
                except Exception:
                    pass
            # Done
            # Write combined stats if requested
            if args.save_stats is not False:
                if args.save_stats is not True:
                    file_name = args.save_stats
                elif args.match_name is not None:
                    file_name = f'results/{args.match_name}_all.json'
                else:
                    file_name = f'results/dynamic_all.json'
                name = Path(file_name)
                if not name.parent.exists():
                    name.parent.mkdir(parents=True)
                with open(name, 'w') as f:
                    json.dump({
                        'round_results': combined_results,
                        'aggregate_by_agent': aggregate_by_agent,
                        'meta': {
                            'total_rounds': total_rounds,
                            'rb_prob_start': args.rb_prob_start,
                            'rb_prob_end': args.rb_prob_end,
                            'anneal_rounds': anneal_rounds,
                            'mode': 'progressive' if args.progressive_opponents else 'dynamic',
                        }
                    }, f, indent=4, sort_keys=True)
                # Also write a simple schedule log (JSON and CSV)
                try:
                    base_stem = name.stem[:-4] if name.stem.endswith('_all') else name.stem
                    sched_json = name.with_name(base_stem + '_schedule.json')
                    with open(sched_json, 'w') as sf:
                        json.dump({'rounds': round_logs}, sf, indent=4, sort_keys=True)
                    sched_csv = name.with_name(base_stem + '_schedule.csv')
                    with open(sched_csv, 'w') as cf:
                        cf.write('round,mode,param,opponent_1,opponent_2\n')
                        for rec in round_logs:
                            mode = rec.get('mode', '')
                            param = rec.get('t', rec.get('p_rb', ''))
                            opps = rec.get('opponents', [])
                            o1 = opps[0] if len(opps) > 0 else ''
                            o2 = opps[1] if len(opps) > 1 else ''
                            cf.write(f"{rec.get('round','')},{mode},{param},{o1},{o2}\n")
                except Exception:
                    pass
            return

        # Curriculum mode orchestrates multiple worlds sequentially
        if args.curriculum:
            if args.train == 0 and not args.continue_without_training:
                args.continue_without_training = True

            phases = [
                (args.phase1_rounds, args.phase1_opponent, "phase1"),
                (args.phase2_rounds, args.phase2_opponent, "phase2"),
            ]

            for n_rounds_phase, opponent_code, phase_tag in phases:
                phase_args = deepcopy(args)
                phase_args.n_rounds = n_rounds_phase
                # Tag match name to distinguish phases
                if phase_args.match_name:
                    phase_args.match_name = f"{phase_args.match_name}-{phase_tag}"
                else:
                    phase_args.match_name = phase_tag

                # Build agents: first args.train are training instances, rest are opponents
                agents = []
                for i in range(s.MAX_AGENTS):
                    if i < args.train:
                        agents.append((args.train_agent, True))
                    else:
                        agents.append((opponent_code, False))

                world = BombeRLeWorld(phase_args, agents)
                every_step = not args.skip_frames
                gui = GUI(world) if has_gui else None
                world_controller(world, phase_args.n_rounds,
                                 gui=gui, every_step=every_step, turn_based=phase_args.turn_based,
                                 make_video=phase_args.make_video, update_interval=phase_args.update_interval)
            return

        agents = []
        if args.train == 0 and not args.continue_without_training:
            args.continue_without_training = True
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        world = BombeRLeWorld(args, agents)
        every_step = not args.skip_frames
    elif args.command_name == "replay":
        world = ReplayWorld(args)
        every_step = True
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    # Launch GUI
    if has_gui:
        gui = GUI(world)
    else:
        gui = None
    world_controller(world, args.n_rounds,
                     gui=gui, every_step=every_step, turn_based=args.turn_based,
                     make_video=args.make_video, update_interval=args.update_interval)


if __name__ == '__main__':
    main()
