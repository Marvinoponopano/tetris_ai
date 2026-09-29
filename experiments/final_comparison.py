"""The final table: Random / Heuristic / Lookahead / Initial RL / Improved RL.

Same protocol as the campaign's own evaluations -- seed 20260928, 8 games on
`game i = seed + i`, piece cap 1000, greedy -- so a row here and a row in
`aggregate.json` are the same measurement, and the table in `REPORT2.md` can be
reproduced with::

    python experiments/final_comparison.py

Checkpoint selection is stated rather than implicit, because it is a place where a
comparison can be flattered: *every* RL arm is represented by the checkpoint its
own campaign scored highest (`best/best.json`), including the round-1 arms, so the
selection rule is identical for all of them. It is still a selection over 25
evaluations x 5 seeds, which makes these numbers optimistic in absolute terms --
the *unselected* numbers are the campaign tails in `aggregate.json`, and
`~/.hermes/cache/scratch/arm_table.py` prints them side by side.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris.rl.compare import ComparisonProtocol, compare_agents  # noqa: E402
from tetris.policy_controller import best_records  # noqa: E402

#: label -> arm. "Improved RL" is K: the only round-2 arm that beat G on the
#: campaign protocol (+31.5 over 960 games, 4/5 seeds). M is shown next to it
#: because its *single best checkpoint* scores higher on these 8 games -- that is
#: a selection effect, not arm-level evidence, and the report says so.
ARMS = {
    "initial RL (C)": "C",
    "round-1 RL (G)": "G",
    "improved RL (K)": "K",
    "combination (M)": "M",
}


def best_checkpoint_for(arm: str):
    """The arm's highest-scoring checkpoint, from the records it wrote itself."""
    records = [record for record in best_records(ROOT / "experiments" / "rl")
               if record.source.split("/")[0] == arm]
    return records[0].path if records else None


def main() -> int:
    checkpoints = {}
    for label, arm in ARMS.items():
        path = best_checkpoint_for(arm)
        if path is None:
            print(f"{label}: no best checkpoint for arm {arm}")
            continue
        checkpoints[label] = path
        print(f"{label:>18} <- {pathlib.Path(path).relative_to(ROOT)}")

    result = compare_agents(
        checkpoints=checkpoints,
        protocol=ComparisonProtocol(seed=20260928, num_games=8, max_pieces=1000),
        output_dir=ROOT / "experiments" / "rl" / "comparison",
        baselines=("random", "heuristic", "lookahead"),
        primary_baseline="heuristic",
        logger=lambda message: print(message, flush=True),
    )
    print()
    print(result["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
