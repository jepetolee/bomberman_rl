"""Plot graphs from results JSON produced by BombeRLeWorld --save-stats.

Usage:
  python -m bomberman_rl.plot_results /path/to/results.json

Outputs PNGs next to the JSON file:
  - <name>_scores.png         : Total score per agent (bar)
  - <name>_agent_stats.png    : Coins/Kills/Suicides per agent (grouped bar)
  - <name>_round_steps.png    : Steps per round (line)
  - <name>_combined_score.png : Combined score over rounds for selected agents (line, only for *_all.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List
import re

import matplotlib

# Use non-interactive backend for headless/SSH environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _aggregate_from_round_jsons(target: Path) -> Dict[str, Any]:
    # If a directory is passed, aggregate all *-r*.json within it
    if target.exists() and target.is_dir():
        directory = target
        base = None
    else:
        # Infer base name: if *_all.json requested, strip suffix; else use stem as base
        base = target.stem
        if base.endswith("_all"):
            base = base[:-4]
        directory = target.parent if target.parent.exists() else Path(".")

    # Collect files like base-r*.json; if none, fall back to all *-r*.json
    files = []
    if base is not None:
        files = sorted(directory.glob(f"{base}r*.json"))
    if not files:
        files = sorted(directory.glob("*r*.json"))
    round_results: List[Dict[str, Any]] = []
    aggregate_by_agent: Dict[str, Dict[str, int]] = {}
    synthetic_by_round: Dict[str, Dict[str, int]] = {}

    for i, fp in enumerate(files, start=1):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        round_results.append(data)

        # Aggregate by_agent totals
        for agent_name, stats in data.get("by_agent", {}).items():
            agg = aggregate_by_agent.setdefault(agent_name, {})
            for k, v in stats.items():
                try:
                    agg[k] = agg.get(k, 0) + int(v)
                except Exception:
                    pass

        # Synthesize by_round steps metric
        try:
            # Expect exactly one entry per file in by_round
            br = list(data.get("by_round", {}).values())[0]
            steps_val = int(br.get("steps", 0))
        except Exception:
            steps_val = 0
        synthetic_by_round[f"Round {i}"] = {"steps": steps_val}

    return {
        "round_results": round_results,
        "aggregate_by_agent": aggregate_by_agent,
        "by_agent": aggregate_by_agent,
        "by_round": synthetic_by_round,
    }


def load_results(path: Path) -> Dict[str, Any]:
    # Allow directory argument to aggregate all rounds in that folder
    if path.exists() and path.is_dir():
        return _aggregate_from_round_jsons(path)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback: try aggregating from per-round files and save for future runs
        aggregated = _aggregate_from_round_jsons(path)
        if aggregated.get("round_results"):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(aggregated, f, indent=4, sort_keys=True)
            except Exception:
                pass
            return aggregated
        raise


def short_agent_name(name: str) -> str:
    n = str(name)
    if n.endswith("_agent"):
        n = n[:-6]
    n = n.replace("rule_based", "rule")
    n = n.replace("coin_collector", "coin")
    n = n.replace("peaceful", "peace")
    # ppo variants
    if n.startswith("ppo_agent_"):
        n = "ppo" + n[len("ppo_agent_"):]
    elif n == "ppo_agent":
        n = "ppo"
    n = re.sub(r"_+", "_", n).strip("_")
    if len(n) > 12:
        n = n[:10] + "…"
    return n


def plot_scores(path: Path, data: Dict[str, Any]) -> Path:
    by_agent = data.get("by_agent", {})
    agents = list(by_agent.keys())
    agents_short = [short_agent_name(a) for a in agents]
    scores = [by_agent[a].get("score", 0) for a in agents]

    fig_width = max(8, 0.6 * max(1, len(agents_short)))
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(range(len(agents_short)), scores, color="#4C78A8")
    ax.set_title("Total Score per Agent")
    ax.set_ylabel("Score")
    ax.set_xlabel("Agent")
    ax.set_xticks(list(range(len(agents_short))))
    ax.set_xticklabels(agents_short, rotation=45, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    out = path.with_name(path.stem + "_scores.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_agent_stats(path: Path, data: Dict[str, Any]) -> Path:
    by_agent = data.get("by_agent", {})
    agents = list(by_agent.keys())
    agents_short = [short_agent_name(a) for a in agents]
    metrics = ["coins", "kills", "suicides"]
    values = {m: [by_agent[a].get(m, 0) for a in agents] for m in metrics}

    x = range(len(agents))
    width = 0.25
    fig_width = max(9, 0.7 * max(1, len(agents_short)))
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    colors = {"coins": "#59A14F", "kills": "#F28E2B", "suicides": "#E15759"}
    for i, m in enumerate(metrics):
        ax.bar([xi + (i - 1) * width for xi in x], values[m], width, label=m.capitalize(), color=colors[m])

    ax.set_xticks(list(x))
    ax.set_xticklabels(agents_short, rotation=45, ha="right")
    ax.set_title("Agent Stats (Totals)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Agent")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out = path.with_name(path.stem + "_agent_stats.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_round_steps(path: Path, data: Dict[str, Any]) -> Path:
    by_round = data.get("by_round", {})
    # Sort rounds by their numeric order if possible, else by key
    # Keys look like "Round 01 (2025-...)"; we'll extract the number best-effort
    def round_index(k: str) -> int:
        try:
            parts = k.split()
            for i, p in enumerate(parts):
                if p.lower().startswith("round") and i + 1 < len(parts):
                    return int(parts[i + 1])
        except Exception:
            pass
        return 0

    items = sorted(by_round.items(), key=lambda kv: round_index(kv[0]))
    if not items:
        return path.with_name(path.stem + "_round_steps.png")

    rounds = [k for k, _ in items]
    steps = [v.get("steps", 0) for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(steps) + 1), steps, marker="o", color="#B07AA1")
    ax.set_title("Steps per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Steps")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()

    out = path.with_name(path.stem + "_round_steps.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _includes_any(name: str, substrings: Iterable[str]) -> bool:
    lname = name.lower()
    return any(sub.lower() in lname for sub in substrings)


def _rolling_mean(xs: List[float], k: int) -> List[float]:
    if k is None or k <= 1:
        return xs
    k = min(k, len(xs))
    out: List[float] = []
    window_sum = sum(xs[:k])
    out.append(window_sum / k)
    for i in range(k, len(xs)):
        window_sum += xs[i] - xs[i - k]
        out.append(window_sum / k)
    # Pad the beginning to align lengths
    return [out[0]] * (k - 1) + out


def plot_combined_score_over_rounds(path: Path, data: Dict[str, Any], include_substrings: Iterable[str], rolling: int) -> Path:
    round_results = data.get("round_results")
    if not isinstance(round_results, list) or not round_results:
        # Nothing to plot
        return path.with_name(path.stem + "_combined_score.png")

    combined: List[float] = []
    for rr in round_results:
        by_agent = rr.get("by_agent", {})
        s = 0.0
        for agent_name, stats in by_agent.items():
            if _includes_any(agent_name, include_substrings):
                s += float(stats.get("score", 0))
        combined.append(s)

    smoothed = _rolling_mean(combined, rolling)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(combined) + 1), combined, color="#4C78A8", alpha=0.35, label="raw")
    if rolling and rolling > 1:
        ax.plot(range(1, len(smoothed) + 1), smoothed, color="#F28E2B", label=f"rolling mean (k={rolling})")
    ax.set_title("Combined score over rounds (selected agents)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Combined score")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out = path.with_name(path.stem + "_combined_score.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _load_schedule_records(path: Path, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Prefer inlined schedule in round_results
    rounds = []
    rr = data.get("round_results")
    if isinstance(rr, list) and rr and isinstance(rr[0], dict):
        for i, item in enumerate(rr, start=1):
            sch = item.get("_schedule") or {}
            opps = item.get("_opponents") or []
            if opps or sch:
                rounds.append({
                    'round': i,
                    'mode': sch.get('mode'),
                    'param': sch.get('t', sch.get('p_rb')),
                    'opponents': opps,
                })
    if rounds:
        return rounds

    # Else, try sibling schedule json: <stem>_schedule.json
    base = path
    directory = base.parent if base.suffix else base
    if base.suffix:
        stem = base.stem
        if stem.endswith('_all'):
            stem = stem[:-4]
        candidates = [directory / f"{stem}_schedule.json"]
    else:
        candidates = sorted(directory.glob("*_schedule.json"))
    for fp in candidates:
        try:
            with open(fp, 'r') as f:
                loaded = json.load(f)
            rounds = loaded.get('rounds') or []
            if rounds:
                return rounds
        except Exception:
            continue
    return []


def plot_opponent_schedule(path: Path, data: Dict[str, Any]) -> Path:
    records = _load_schedule_records(path, data)
    out = path.with_name(path.stem + "_schedule.png")
    if not records:
        # Nothing to plot; create empty placeholder
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, 'No schedule data', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # Map opponents to categories
    def norm_name(n: str) -> str:
        if not n:
            return ''
        return short_agent_name(n)

    # Define palette for known labels
    palette = {
        'fail': '#E15759',
        'peace': '#59A14F',
        'coin': '#76B7B2',
        'rule': '#4C78A8',
        'random': '#B07AA1',
        'tpl': '#FF9DA7',
        'user': '#9C755F',
        'ppo': '#F28E2B',
    }

    # Build category list from data
    cats = []
    for rec in records:
        opps = rec.get('opponents', [])
        for o in opps[:2]:
            nm = norm_name(o)
            if nm and nm not in cats:
                cats.append(nm)
    if not cats:
        cats = ['random', 'rule']
    # Keep stable order with known keys first
    preferred = ['fail', 'peace', 'coin', 'random', 'rule', 'ppo', 'tpl', 'user']
    cats = [c for c in preferred if c in cats] + [c for c in cats if c not in preferred]

    # Color list aligned to cats
    colors = [palette.get(c, '#AAAAAA') for c in cats]
    cat_to_idx = {c: i for i, c in enumerate(cats)}

    # Build 2 x T matrix of indices (slot0, slot1)
    T = len(records)
    import numpy as np
    mat = np.full((2, T), fill_value=-1, dtype=int)
    for i, rec in enumerate(records):
        opps = rec.get('opponents', [])
        for row in range(2):
            if row < len(opps):
                nm = norm_name(opps[row])
                mat[row, i] = cat_to_idx.get(nm, -1)

    # Plot as image
    fig_height = 1.5 + 0.2 * len(cats)
    fig, ax = plt.subplots(figsize=(max(10, T / 40), fig_height))
    # Map -1 to white
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = list(range(len(cats) + 1))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['slot1', 'slot2'])
    ax.set_xticks([])
    ax.set_title('Opponent schedule (per round)')

    # Colorbar with category labels
    cbar = fig.colorbar(im, ax=ax, ticks=[i + 0.5 for i in range(len(cats))], orientation='vertical')
    cbar.ax.set_yticklabels(cats)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot graphs from BombeRLeWorld results JSON")
    parser.add_argument("results_json", type=Path, help="Path to results JSON produced by --save-stats")
    parser.add_argument("--include", nargs="+", default=["ppo_agent"], help="Agent name substrings to include for combined score plot")
    parser.add_argument("--rolling", type=int, default=20, help="Rolling window for smoothing combined score (0/1 to disable)")
    args = parser.parse_args()

    data = load_results(args.results_json)
    out1 = plot_scores(args.results_json, data)
    out2 = plot_agent_stats(args.results_json, data)
    out3 = plot_round_steps(args.results_json, data)
    out4 = plot_combined_score_over_rounds(args.results_json, data, args.include, args.rolling)
    out5 = plot_opponent_schedule(args.results_json, data)

    print("Saved:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)
    print(out5)


if __name__ == "__main__":
    main()


