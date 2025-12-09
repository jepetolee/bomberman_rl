import argparse, json
from pathlib import Path
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("all_json", type=Path, help="results/<name>_all.json")
    p.add_argument("--interval", type=int, default=100, help="snapshot interval (N rounds)")
    p.add_argument("--outdir", type=Path, default=None, help="output directory (default: same as all_json)")
    args = p.parse_args()

    data = json.loads(Path(args.all_json).read_text(encoding="utf-8"))
    round_results = data.get("round_results", [])
    if not round_results:
        print("No round_results in all_json.")
        return

    outdir = args.outdir or args.all_json.parent
    outdir.mkdir(parents=True, exist_ok=True)

    stem = args.all_json.stem
    if stem.endswith("_all"):
        stem = stem[:-4]
    base = outdir / stem

    # 1) 라운드별 JSON
    for i, rr in enumerate(round_results, start=1):
        (outdir / f"{stem}-r{i:05d}.json").write_text(json.dumps(rr, indent=4, sort_keys=True), encoding="utf-8")

    # 2) N라운드 단위 스냅샷(_all_until_rXXXXX.json)
    def merge_aggregate(rs):
        agg = defaultdict(lambda: defaultdict(int))
        for r in rs:
            for agent, stats in (r.get("by_agent") or {}).items():
                for k, v in stats.items():
                    try:
                        agg[agent][k] += int(v)
                    except Exception:
                        pass
        return {a: dict(m) for a, m in agg.items()}

    total = len(round_results)
    interval = max(1, args.interval)
    for upto in range(interval, total + 1, interval):
        snap_path = outdir / f"{stem}_all_until_r{upto:05d}.json"
        snap = {
            "round_results": round_results[:upto],
            "aggregate_by_agent": merge_aggregate(round_results[:upto]),
            "meta": data.get("meta", {}),
        }
        snap_path.write_text(json.dumps(snap, indent=4, sort_keys=True), encoding="utf-8")

    print(f"Exported per-round and snapshots to: {outdir}")

if __name__ == "__main__":
    main()