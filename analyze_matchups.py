"""Summarize head-to-head results from BombeRLeWorld *_all.json outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _detect_train_agents(round_results: List[Dict]) -> Iterable[str]:
    if not round_results:
        return []
    first = round_results[0]
    by_agent = first.get("by_agent") or {}
    # Assume training agents are the PPO ones that start with ppo_agent
    # Adjust here if your naming differs.
    return [name for name in by_agent.keys() if str(name).startswith("ppo_agent")]


def _result(team_score: int, opp_score: int) -> str:
    if team_score > opp_score:
        return "win"
    if team_score < opp_score:
        return "loss"
    return "draw"


def summarize(results_json: Path) -> Dict[str, Dict[str, int]]:
    data = json.loads(results_json.read_text(encoding="utf-8"))
    round_results = data.get("round_results") or []
    train_agents = set(_detect_train_agents(round_results))

    summary = defaultdict(lambda: {"win": 0, "draw": 0, "loss": 0, "rounds": 0})

    for rr in round_results:
        opponents = tuple(rr.get("_opponents", []))
        key = ", ".join(opponents) if opponents else "unknown"
        by_agent: Dict[str, Dict[str, int]] = rr.get("by_agent", {})

        team_score = sum((by_agent.get(a, {}) or {}).get("score", 0) for a in train_agents)
        opp_score = sum((stats or {}).get("score", 0) for agent, stats in by_agent.items() if agent not in train_agents)

        outcome = _result(team_score, opp_score)
        entry = summary[key]
        entry[outcome] += 1
        entry["rounds"] += 1

    return summary


def format_summary(summary: Dict[str, Dict[str, int]]) -> str:
    lines = []
    lines.append(f"{'Opponents':40s}  {'W':>5s}  {'D':>5s}  {'L':>5s}  {'Win%':>6s}  Rounds")
    for opponents, rec in sorted(summary.items(), key=lambda item: item[0]):
        total = rec['rounds'] or 1
        win_pct = 100.0 * rec['win'] / total
        lines.append(
            f"{opponents or 'unknown':40s}  {rec['win']:5d}  {rec['draw']:5d}  {rec['loss']:5d}  {win_pct:6.1f}  {rec['rounds']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate head-to-head results from *_all.json")
    parser.add_argument("results_json", type=Path, help="Path to results JSON (e.g. results_curriculum_all.json)")
    args = parser.parse_args()

    if not args.results_json.exists():
        parser.error(f"File not found: {args.results_json}")

    summary = summarize(args.results_json)
    print(format_summary(summary))


if __name__ == "__main__":
    main()


