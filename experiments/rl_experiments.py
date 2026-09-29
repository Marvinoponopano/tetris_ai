"""Train, evaluate, compare, plot and diagnose the DQN agent, in that order.

    python experiments/rl_experiments.py plan
    python experiments/rl_experiments.py run --experiment A --parallel 5
    python experiments/rl_experiments.py run --experiment B --seeds 0 1 2 3 4
    python experiments/rl_experiments.py aggregate
    python experiments/rl_experiments.py compare --experiments A B C D --games 20
    python experiments/rl_experiments.py plot --all
    python experiments/rl_experiments.py diagnose --checkpoint experiments/rl/C/seed0/best/<file>.pt
    python experiments/rl_experiments.py diagnose --untrained

The experiments are defined in `experiments/rl_plan.py`; everything they write
lands under `experiments/rl/<experiment>/`, one directory per seed. Each command
is a thin wrapper around a function in `tetris.rl`:

    run        tetris.rl.campaign.Campaign.run          training + periodic evaluation
    aggregate  tetris.rl.campaign.Campaign.aggregate    first vs last, over seeds
    compare    tetris.rl.compare.compare_agents         against the baselines, same seeds
    plot       tetris.rl.plots.plot_campaign            the figures, raw + smoothed
    diagnose   tetris.rl.diagnose.diagnose_checkpoint   why the policy plays like that

Nothing here trains by accident: `plan` and `aggregate` only read, and `run`
always says which experiment it is about to start and how much it costs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Dict, List, Optional  # noqa: UP035

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "experiments" / "rl"


def load_plan():
    """Load `experiments/rl_plan.py` from its path (`experiments/` is not a package)."""
    spec = importlib.util.spec_from_file_location("rl_plan", HERE / "rl_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load_plan()


def _campaign_dir(data_dir: pathlib.Path, name: str) -> pathlib.Path:
    return pathlib.Path(data_dir) / str(name).upper()


def _load_aggregate(directory: pathlib.Path) -> Optional[Dict]:
    path = pathlib.Path(directory) / "aggregate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_table(documents: List[Dict]) -> str:
    """One block per experiment: the trends, with the per-seed deltas behind them."""
    if not documents:
        return "nothing to aggregate yet"
    header = (f"{'exp':>4} {'episodes':>9} {'seeds':>6} {'metric':>12} "
              f"{'first(ep0)':>12} {'tail':>12} {'delta':>10} {'sd':>9} "
              f"{'improved':>9}  verdict")
    lines = [header, "-" * len(header)]
    for document in documents:
        metric = document.get("metric", "score_mean")
        trend = (document.get("metrics") or {}).get(metric, {})
        lines.append(
            f"{document.get('experiment', '?'):>4} {document.get('episodes', 0):>9,} "
            f"{len(document.get('seeds', [])):>6} {metric:>12} "
            f"{trend.get('first_mean', 0.0):>12,.1f} {trend.get('tail_mean', 0.0):>12,.1f} "
            f"{trend.get('delta_mean', 0.0):>+10,.1f} {trend.get('delta_sd', 0.0):>9,.1f} "
            f"{str(trend.get('improved_seeds', 0)) + '/' + str(trend.get('n_seeds', 0)):>9}  "
            f"{trend.get('verdict', '?')}"
        )
    lines.append("")
    lines.append("other metrics (tail - first, mean over seeds):")
    for document in documents:
        for name, trend in (document.get("metrics") or {}).items():
            if name == document.get("metric"):
                continue
            lines.append(
                f"  {document.get('experiment', '?'):>4} {name:<22} "
                f"{trend.get('first_mean', 0.0):>12,.3f} -> {trend.get('tail_mean', 0.0):>12,.3f}"
                f"   delta {trend.get('delta_mean', 0.0):>+10,.3f}"
                f"   improved {trend.get('improved_seeds', 0)}/{trend.get('n_seeds', 0)}"
            )
    return "\n".join(lines)


def command_plan(args) -> int:
    print("the planned experiments (experiments/rl_plan.py)")
    print()
    print(plan.table(args.experiment or None))
    print()
    print(f"results directory: {args.data_dir}")
    print("each experiment runs every seed it lists and evaluates on schedule;")
    print("`run` writes experiments/rl/<experiment>/seed<k>/{metrics.jsonl,summary.json,best/}")
    return 0


def command_run(args) -> int:
    for name in args.experiment:
        campaign = plan.campaign(
            name,
            seeds=args.seeds,
            output_dir=args.data_dir,
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
        )
        result = campaign.run(parallel=args.parallel, logger=None if args.quiet else print)
        if args.quiet:
            print(result.summary())
    return 0


def command_aggregate(args) -> int:
    documents = []
    for name in args.experiment or list(plan.EXPERIMENTS):
        document = _load_aggregate(_campaign_dir(args.data_dir, name))
        if document is None:
            print(f"{name}: no aggregate.json yet (run it first)", file=sys.stderr)
            continue
        documents.append(document)
    print(aggregate_table(documents))
    if args.save:
        target = pathlib.Path(args.save)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(documents, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {target}")
    return 0


def command_compare(args) -> int:
    from tetris.rl.compare import ComparisonProtocol, best_checkpoints, compare_agents

    checkpoints: Dict[str, pathlib.Path] = {}
    for name in args.experiment or []:
        directory = _campaign_dir(args.data_dir, name)
        if not directory.exists():
            print(f"{name}: nothing under {directory} yet", file=sys.stderr)
            continue
        for label, path in best_checkpoints(directory, prefix=str(name).upper()).items():
            checkpoints[label] = path

    for entry in args.checkpoint or []:
        label, _, path = entry.partition("=")
        if not path:
            print(f"--checkpoint needs label=path, got {entry!r}", file=sys.stderr)
            return 2
        checkpoints[label] = pathlib.Path(path)

    if not checkpoints and not args.baselines_only:
        print("no checkpoints to compare: pass --experiment NAME or --checkpoint label=path "
              "(or --baselines-only for the baselines alone)", file=sys.stderr)
        return 2

    protocol = ComparisonProtocol(
        num_games=args.games, seed=args.seed, max_pieces=args.max_pieces
    )
    print(f"comparison protocol: {protocol.describe()}")
    print(f"agents: {', '.join(sorted(checkpoints)) or '(baselines only)'}")
    if "lookahead" in args.baselines:
        print("note: the lookahead agent is ~30 s per game at this cap -- expect minutes")
    payload = compare_agents(
        checkpoints=checkpoints,
        protocol=protocol,
        output_dir=args.output_dir,
        baselines=tuple(args.baselines),
        **({"primary_baseline": args.primary} if args.primary else {}),
        logger=print,
    )
    print()
    print(payload["table"])
    print(f"\nwrote {pathlib.Path(args.output_dir) / 'comparison.json'}")
    return 0


def command_plot(args) -> int:
    from tetris.rl.plots import plot_campaign

    if args.campaign:
        targets = [(pathlib.Path(path).name, pathlib.Path(path)) for path in args.campaign]
    else:
        names = args.experiment or (list(plan.EXPERIMENTS) if args.all else [])
        targets = [(name, _campaign_dir(args.data_dir, name)) for name in names]
    if not targets:
        print("nothing to plot: pass --experiment NAME, --all or --campaign PATH",
              file=sys.stderr)
        return 2

    baselines = _load_baselines(args.comparison)
    for name, directory in targets:
        if not directory.exists():
            print(f"{name}: nothing under {directory} yet", file=sys.stderr)
            continue
        written = plot_campaign(
            directory,
            output_dir=args.output_dir,
            window=args.window,
            baselines=baselines,
            title=str(name).upper(),
        )
        for path in written:
            print(f"  {path}")
    if args.comparison and pathlib.Path(args.comparison).exists():
        from tetris.rl.plots import plot_comparison

        target = pathlib.Path(args.comparison).with_name("05_baselines.png")
        print(f"  {plot_comparison(args.comparison, target)}")
    return 0


def _load_baselines(comparison_path: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
    """Reference values for the figures: the baselines' stats, if they exist yet."""
    if not comparison_path:
        return None
    path = pathlib.Path(comparison_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        label: {
            "score_mean": (entry.get("stats") or {}).get("score_mean", 0.0),
            "lines_mean": (entry.get("stats") or {}).get("lines_mean", 0.0),
        }
        for label, entry in (payload.get("agents") or {}).items()
        if ":" not in label
    }


def command_diagnose(args) -> int:
    from tetris.rl.diagnose import diagnose_checkpoint, diagnose_untrained

    if args.untrained:
        diagnosis = diagnose_untrained(
            label="untrained", games=args.games, seed=args.seed,
            max_pieces=args.max_pieces, lookahead_decisions=args.lookahead_decisions,
        )
    else:
        if not args.checkpoint:
            print("pass --checkpoint PATH (or --untrained)", file=sys.stderr)
            return 2
        diagnosis = diagnose_checkpoint(
            args.checkpoint, games=args.games, seed=args.seed,
            max_pieces=args.max_pieces, lookahead_decisions=args.lookahead_decisions,
        )
    print(diagnosis.summary())
    if args.save:
        print(f"\nwrote {diagnosis.save(args.save)}")
    return 0


def command_report(args) -> int:
    """Regenerate the plots and print the aggregate table for everything present."""
    from tetris.rl.plots import plot_campaign

    documents = []
    for name in args.experiment or list(plan.EXPERIMENTS):
        directory = _campaign_dir(args.data_dir, name)
        document = _load_aggregate(directory)
        if document is None:
            continue
        documents.append(document)
        baselines = _load_baselines(args.data_dir / "comparison" / "comparison.json")
        for path in plot_campaign(directory, window=args.window, baselines=baselines,
                                  title=str(name).upper()):
            print(f"  {path}")
    print()
    print(aggregate_table(documents))
    comparison = args.data_dir / "comparison" / "comparison.json"
    if comparison.exists():
        from tetris.rl.compare import format_comparison

        payload = json.loads(comparison.read_text(encoding="utf-8"))
        print()
        print(format_comparison(list((payload.get("agents") or {}).values())))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                        help=f"where campaigns live (default {DEFAULT_DATA_DIR})")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="print the planned experiments")
    plan_parser.add_argument("--experiment", nargs="+", metavar="NAME")
    plan_parser.set_defaults(handler=command_plan)

    run_parser = sub.add_parser("run", help="run one or more experiments")
    run_parser.add_argument("--experiment", nargs="+", required=True, metavar="NAME")
    run_parser.add_argument("--seeds", nargs="+", type=int,
                            help="override the plan's seeds, e.g. --seeds 0 1 2")
    run_parser.add_argument("--episodes", type=int, help="override the episode budget")
    run_parser.add_argument("--eval-every", type=int, dest="eval_every",
                            help="override how often training pauses to evaluate")
    run_parser.add_argument("--eval-games", type=int, dest="eval_games",
                            help="games per evaluation")
    run_parser.add_argument("--parallel", type=int, default=1,
                            help="seeds to run at once, in separate processes")
    run_parser.add_argument("--quiet", action="store_true", help="only the summaries")
    run_parser.set_defaults(handler=command_run)

    aggregate_parser = sub.add_parser("aggregate", help="aggregate what has been run")
    aggregate_parser.add_argument("--experiment", nargs="+", metavar="NAME")
    aggregate_parser.add_argument("--save", metavar="PATH")
    aggregate_parser.set_defaults(handler=command_aggregate)

    compare_parser = sub.add_parser("compare", help="the agent vs the baselines")
    compare_parser.add_argument("--experiment", nargs="+", metavar="NAME",
                                help="use the best checkpoint of every seed of these")
    compare_parser.add_argument("--checkpoint", action="append", metavar="LABEL=PATH")
    compare_parser.add_argument("--baselines", nargs="+",
                                default=["random", "heuristic", "lookahead"],
                                choices=["random", "heuristic", "lookahead"])
    compare_parser.add_argument("--baselines-only", action="store_true", dest="baselines_only")
    compare_parser.add_argument(
        "--primary", choices=("random", "heuristic", "lookahead"), default=None,
        help="the baseline every other agent is paired against (default: random; "
             "heuristic is the one worth pairing a learning agent against)")
    compare_parser.add_argument("--games", type=int, default=20)
    compare_parser.add_argument("--seed", type=int, default=20260928)
    compare_parser.add_argument("--max-pieces", type=int, default=1000, dest="max_pieces")
    compare_parser.add_argument("--output-dir", type=pathlib.Path,
                                default=DEFAULT_DATA_DIR / "comparison", dest="output_dir")
    compare_parser.set_defaults(handler=command_compare)

    plot_parser = sub.add_parser("plot", help="the figures for a campaign")
    plot_parser.add_argument("--experiment", nargs="+", metavar="NAME")
    plot_parser.add_argument("--campaign", nargs="+", type=pathlib.Path)
    plot_parser.add_argument("--all", action="store_true")
    plot_parser.add_argument("--window", type=int, default=20,
                             help="moving-average window in episodes (default 20)")
    plot_parser.add_argument("--output-dir", type=pathlib.Path, dest="output_dir")
    plot_parser.add_argument("--comparison", metavar="PATH",
                             default=str(DEFAULT_DATA_DIR / "comparison" / "comparison.json"))
    plot_parser.set_defaults(handler=command_plot)

    diagnose_parser = sub.add_parser("diagnose", help="why the agent plays like that")
    diagnose_parser.add_argument("--checkpoint", metavar="PATH")
    diagnose_parser.add_argument("--untrained", action="store_true",
                                 help="diagnose a fresh agent instead of a checkpoint")
    diagnose_parser.add_argument("--games", type=int, default=3)
    diagnose_parser.add_argument("--seed", type=int, default=20260928)
    diagnose_parser.add_argument("--max-pieces", type=int, default=200, dest="max_pieces")
    diagnose_parser.add_argument("--lookahead-decisions", type=int, default=30,
                                 dest="lookahead_decisions")
    diagnose_parser.add_argument("--save", metavar="PATH")
    diagnose_parser.set_defaults(handler=command_diagnose)

    report_parser = sub.add_parser("report", help="plots + aggregate table for everything")
    report_parser.add_argument("--experiment", nargs="+", metavar="NAME")
    report_parser.add_argument("--window", type=int, default=20)
    report_parser.set_defaults(handler=command_report)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except (ValueError, KeyError, FileNotFoundError) as problem:
        print(f"{args.command} failed: {problem}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
