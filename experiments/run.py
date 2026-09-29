"""Run experiments from config files, and compare what has been run.

    python experiments/run.py --config experiments/configs/heuristic_depth1.json
    python experiments/run.py --config experiments/configs/*.json
    python experiments/run.py --config base.json --set num_games=10 seed=7
    python experiments/run.py --config sweep.json --profile
    python experiments/run.py --dry-run --config experiments/configs/heuristic_depth1.json
    python experiments/run.py --list                 # every run so far, as a table
    python experiments/run.py --list --csv out.csv   # ... and as CSV

Each run writes `run-<id>.json` (the whole result: config, stats, timing, the
machine, every game) into `--data-dir`, and appends one line to `runs.jsonl`
inside it. The table printed at the end is built from those rows -- by
`--list` it is built from every run in the directory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional  # noqa: UP035

# Running this file directly puts `experiments/` on sys.path, not the project
# root, so the package has to be reachable by hand.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris.config import ExperimentConfig, parse_override  # noqa: E402
from tetris.experiment import compare_runs, run_experiment  # noqa: E402

#: Where results go unless `--data-dir` says otherwise.
DEFAULT_DATA_DIR = ROOT / "experiments" / "results"


def load_configs(paths: List[str], overrides: Optional[List[str]] = None):
    """Every config in every given file, with the command-line overrides applied.

    A file may hold one config or a list of them. Every override is applied to
    every config, which is the simple rule to remember when sweeping.
    """
    applied = dict(parse_override(text) for text in (overrides or []))
    configs = []
    for path in paths:
        for config in ExperimentConfig.load_all(path):
            configs.append(config.with_overrides(**applied) if applied else config)
    return configs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="extend", nargs="+", default=[], metavar="PATH",
                        help="one or more config files (`configs/*.json` works); each may "
                             "hold one config or a list of them")
    parser.add_argument("--set", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override a config field, e.g. --set num_games=10 seed=7")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                        help=f"where results are written (default {DEFAULT_DATA_DIR})")
    parser.add_argument("--profile", action="store_true",
                        help="also write a cProfile report for each run")
    parser.add_argument("--list", action="store_true",
                        help="do not run anything: print every run in --data-dir")
    parser.add_argument("--csv", type=pathlib.Path, metavar="PATH",
                        help="with --list, also write the table to CSV")
    parser.add_argument("--log-every", type=int, default=None, metavar="N",
                        help="print progress every N games (default: about every 10%%)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would run, then exit")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per run instead of per-game progress")
    args = parser.parse_args(argv)

    if args.list:
        if not args.data_dir.exists():
            print(f"nothing to list: {args.data_dir} does not exist")
            return 0
        print(compare_runs(args.data_dir, save_csv=args.csv))
        if args.csv:
            print(f"\nwrote {args.csv}")
        return 0

    if not args.config:
        parser.error("--config is required (or use --list)")

    try:
        configs = load_configs(args.config, args.set)
    except (ValueError, KeyError) as problem:
        print(f"bad configuration: {problem}", file=sys.stderr)
        return 1
    if not configs:
        print("no configs to run", file=sys.stderr)
        return 1

    if args.dry_run:
        for config in configs:
            print(f"{config.config_id}  {config.describe()}")
        return 0

    results = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.describe()}", flush=True)
        result = run_experiment(
            config,
            save_dir=args.data_dir,
            profile=args.profile,
            log_every=args.log_every,
            logger=None if args.quiet else print,
        )
        print(f"    {result.timing['games_per_second']:.2f} games/s, "
              f"{result.timing['elapsed_seconds']:.1f}s -> {result.result_path}",
              flush=True)
        results.append(result)

    print()
    print(compare_runs(results))
    print(f"\nresults in {args.data_dir} "
          f"({len(results)} new run(s); `--list` shows all of them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
