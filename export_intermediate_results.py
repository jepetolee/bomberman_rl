import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge_aggregate(round_results):
    agg = defaultdict(lambda: defaultdict(int))
    for r in round_results:
        by_agent = (r or {}).get("by_agent") or {}
        for agent, stats in by_agent.items():
            for k, v in stats.items():
                try:
                    agg[agent][k] += int(v)
                except Exception:
                    pass
    return {a: dict(m) for a, m in agg.items()}


def main():
    parser = argparse.ArgumentParser(description="Export per-round JSONs and N-round snapshots from *_all.json")
    parser.add_argument("all_json", type=Path, help="Path to results/<name>_all.json")
    parser.add_argument("--interval", type=int, default=100, help="Snapshot interval in rounds (default: 100)")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory (default: same as all_json parent)")
    args = parser.parse_args()

    if not args.all_json.exists():
        raise FileNotFoundError(f"all_json not found: {args.all_json}")

    data = json.loads(args.all_json.read_text(encoding="utf-8"))
    round_results = data.get("round_results") or []
    if not round_results:
        print("No round_results in all_json; nothing to export.")
        return

    outdir = args.outdir or args.all_json.parent
    outdir.mkdir(parents=True, exist_ok=True)

    stem = args.all_json.stem
    if stem.endswith("_all"):
        stem = stem[:-4]

    # 1) Per-round JSON files: <stem>-r00001.json, ...
    for i, rr in enumerate(round_results, start=1):
        (outdir / f"{stem}-r{i:05d}.json").write_text(
            json.dumps(rr, indent=4, sort_keys=True), encoding="utf-8"
        )

    # 2) N-round snapshot files: <stem>_all_until_r000100.json, ...
    total = len(round_results)
    interval = max(1, int(args.interval))
    meta = data.get("meta") or {}
    for upto in range(interval, total + 1, interval):
        snap_path = outdir / f"{stem}_all_until_r{upto:05d}.json"
        snap_data = {
            "round_results": round_results[:upto],
            "aggregate_by_agent": merge_aggregate(round_results[:upto]),
            "meta": meta,
        }
        snap_path.write_text(json.dumps(snap_data, indent=4, sort_keys=True), encoding="utf-8")

    print(f"Exported per-round files and snapshots to: {outdir}")


if __name__ == "__main__":
    main()



