"""The A/B/C/D/E experiment plan, and how each one is run.

Kept apart from the CLI so the plan can be imported (and checked) on its own,
and so the numbers that define an experiment live in one readable place.

    python -c "import experiments.rl_plan as p; print(p.table())"
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

from tetris.rl.campaign import DEFAULT_METRIC, Campaign
from tetris.rl.config import TrainingConfig
from tetris.rl.evaluation import DEFAULT_EVAL_GAMES, DEFAULT_EVAL_SEED
from tetris.evaluate import DEFAULT_MAX_PIECES

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "rl"

#: What every experiment shares: the agent, the replay setup and the episode
#: length. Only the *size* of the run and the schedule that has to fit inside it
#: change from one experiment to the next.
COMMON = dict(
    learning_rate=1e-3,
    batch_size=64,
    buffer_size=50_000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    hidden_sizes=(256, 256),
    max_pieces=200,
    device="cpu",
)

#: The evaluation protocol every experiment and every baseline shares. The
#: in-training evaluations use the first `eval_games` of these same seeds, so a
#: training curve and the final comparison table are the same experiment.
EVAL = dict(eval_games=DEFAULT_EVAL_GAMES, eval_seed=DEFAULT_EVAL_SEED,
            eval_max_pieces=DEFAULT_MAX_PIECES)

#: name -> the experiment. `training` is the TrainingConfig (so a `reward` entry
#: is part of it), `variant` names an agent variant from `tetris.rl.variants`
#: (None = the reference DQN). A/B/C are sizes of the same run; D and E are the
#: targeted changes the diagnosis of A-C asked for, each differing from C in
#: exactly one knob.
EXPERIMENTS: Dict[str, Dict[str, object]] = {
    "A": {
        "purpose": "small: does anything move at all, and is the plumbing honest?",
        "episodes": 300,
        "training": dict(COMMON, min_buffer_size=500, target_update_steps=250,
                         epsilon_decay_steps=5_000, checkpoint_every=100, log_every=50),
        "eval_every": 50,
        "variant": None,
        "seeds": (0, 1, 2, 3, 4),
    },
    "B": {
        "purpose": "moderate: enough episodes for the schedule to anneal and the "
                   "score to have a chance to separate from random",
        "episodes": 2_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=40_000, checkpoint_every=500, log_every=100),
        "eval_every": 100,
        "variant": None,
        "seeds": (0, 1, 2, 3, 4),
    },
    "C": {
        "purpose": "long: the same agent with a budget in the tens of thousands of "
                   "gradient steps, in case the small runs were simply too short",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=120_000, checkpoint_every=1_000, log_every=200),
        "eval_every": 250,
        "variant": None,
        "seeds": (0, 1, 2, 3, 4),
    },
    "D": {
        "purpose": "targeted change 1 from the diagnosis: Double Q-learning targets "
                   "(every other knob identical to C)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=120_000, checkpoint_every=1_000, log_every=200),
        "eval_every": 250,
        "variant": "double-dqn",
        "seeds": (0, 1, 2, 3, 4),
    },
    "E": {
        "purpose": "targeted change 2 from the diagnosis: pay per_piece for every "
                   "piece survived, so playing on is worth more than dying and a "
                   "line clear becomes the only way to survive past ~40 pieces "
                   "(every other knob identical to C)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=120_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": None,
        "seeds": (0, 1, 2, 3, 4),
    },
    "F": {
        "purpose": "targeted change 3 from the diagnosis: anneal epsilon inside the "
                   "run (20k steps, not 120k) so most of the training happens with "
                   "the policy it will be evaluated as, not with a mostly random one "
                   "(every other knob identical to C)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200),
        "eval_every": 250,
        "variant": None,
        "seeds": (0, 1, 2, 3, 4),
    },
    "G": {
        "purpose": "the two changes the diagnosis asked for, together: an annealed "
                   "schedule AND per_piece=5 (each alone was run as E and F; this is "
                   "the pair, and the only two-knob arm here)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": None,
        "seeds": (0, 1, 2, 3, 4),
    },
    # --- the second round: each of H..L is G with exactly one thing changed ------
    #
    # G is the control for all of them, because G is the best agent this project
    # has and the round-1 diagnosis of *its* policy is what these targets. Every
    # arm below keeps G's episodes, seeds, schedule and evaluation protocol, so a
    # difference between two rows is the one change and nothing else.
    "H": {
        "purpose": "reward ratio: per_line 10 -> 50 and per_piece 5 -> 1. G's income is "
                   "89% per-piece and 7% per-line, and a line clear (+10) nets about "
                   "the same as the two pieces of survival it costs (+10, minus ~2 "
                   "of holes and height), so max-Q has no reason to prefer clearing. "
                   "This makes a line worth 50 pieces of survival and survival worth "
                   "little, so the only way to earn is to clear. Predicts fewer "
                   "pieces and score but more lines per 100 pieces; if it only loses "
                   "score and clears no more, the objective was never the binding "
                   "constraint (everything else identical to G)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_line": 50.0, "per_piece": 1.0}),
        "eval_every": 250,
        "variant": None,
        "seeds": (0, 1, 2, 3, 4),
    },
    "I": {
        "purpose": "credit assignment: three-step returns (n=3), so the reward a "
                   "line clear pays arrives inside the target of the move that "
                   "started building the row instead of having to travel back "
                   "through four bootstrapped estimates. Targets the delay "
                   "between the stack-building placements and the clear they "
                   "make possible (everything else identical to G)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": "n-step",
        "seeds": (0, 1, 2, 3, 4),
    },
    "J": {
        "purpose": "representation: five state planes instead of three -- column "
                   "heights and the hole mask, from the same tetris.metrics "
                   "functions the heuristic agent scores with. Targets the gap "
                   "that G has to learn geometry from raw cells with ~86k "
                   "gradient steps while the heuristic is handed it "
                   "(everything else identical to G)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": "feature-planes",
        "seeds": (0, 1, 2, 3, 4),
    },
    "K": {
        "purpose": "architecture: dueling head (Q = V + A - mean A), because the "
                   "diagnosis measured a mean best Q of 3.63 against only 1.61 "
                   "between best and second-best -- most of the value is the "
                   "position, not the placement, and 40 output copies of it is "
                   "what the reference net has (everything else identical to G)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": "dueling",
        "seeds": (0, 1, 2, 3, 4),
    },
    "L": {
        "purpose": "replay: sample transitions by TD error with importance "
                   "weights (alpha 0.6, beta annealed 0.4 -> 1.0), because a run "
                   "makes ~86k decisions of which line clears are ~0.5%, so "
                   "uniform sampling spends almost every batch on placements the "
                   "network already knows (everything else identical to G)",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": "prioritized",
        "seeds": (0, 1, 2, 3, 4),
    },
    "M": {
        "purpose": "the combination, and the only arm allowed two changes: dueling "
                   "(the one arm of the five that improved the agent: +31.5 mean "
                   "score, 4/5 seeds) with the five-plane encoding (the one that "
                   "made the heuristic's ranking easier to learn: 45.0% supervised "
                   "agreement against 32.7% on three planes). They attack different "
                   "halves -- what the value is worth, and what the network can see "
                   "-- so this arm is the honest test of whether the round-2 gains "
                   "add up or overlap.\n\n"
                   "Predicts: at or above K. If it lands at K's level the two do "
                   "overlap; if it lands above, they compose.",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": "dueling-feature-planes",
        "seeds": (0, 1, 2, 3, 4),
    },
    "N": {
        "purpose": "lookahead: show the network the piece after next (queue[1], "
                   "which the encoding has always thrown away). Read this arm "
                   "against J, not against G: it is the five feature planes plus "
                   "one queue plane, so the difference is exactly the queue. If "
                   "planning for a piece two moves away is what the agent lacks, "
                   "this is where it shows; the reward, schedule, head, buffer "
                   "and episodes are G's/J's",
        "episodes": 6_000,
        "training": dict(COMMON, min_buffer_size=1_000, target_update_steps=500,
                         epsilon_decay_steps=20_000, checkpoint_every=1_000, log_every=200,
                         reward={"per_piece": 5.0}),
        "eval_every": 250,
        "variant": "queue-planes",
        "seeds": (0, 1, 2, 3, 4),
    },
}


def experiment(name: str):
    """The raw dictionary for one experiment (raises for an unknown name)."""
    key = str(name).upper()
    if key not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}")
    return key, EXPERIMENTS[key]


def training_config(name: str, **overrides) -> TrainingConfig:
    """The `TrainingConfig` of one experiment, with optional overrides on top."""
    _, spec = experiment(name)
    payload = dict(spec["training"])
    payload["episodes"] = int(spec["episodes"])
    payload["label"] = f"{str(name).upper()}: {spec['purpose']}"
    payload.update(overrides)
    return TrainingConfig.from_dict(payload)


def campaign(
    name: str,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[pathlib.Path] = None,
    episodes: Optional[int] = None,
    eval_every: Optional[int] = None,
    eval_games: Optional[int] = None,
    **overrides,
) -> Campaign:
    """The `Campaign` for one experiment, ready to `run()`.

    `seeds`, `episodes`, `eval_every` and `eval_games` shrink or redirect the
    plan (a smoke run, a single seed); anything else in `**overrides` goes to
    the `TrainingConfig` (and is rejected there if it is not a real knob).
    """
    key, spec = experiment(name)
    config = training_config(
        key, **({"episodes": int(episodes)} if episodes else {}), **overrides
    )
    return Campaign(
        name=key,
        config=config,
        seeds=tuple(spec["seeds"] if seeds is None else seeds),
        eval_every=int(eval_every or spec["eval_every"]),
        eval_games=int(eval_games or EVAL["eval_games"]),
        metric=DEFAULT_METRIC,
        output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else output_dir,
        variant=spec["variant"],
        log_every=int(config.log_every),
        note=str(spec["purpose"]),
    )


def _reward_caption(training: Dict[str, object]) -> str:
    """The reward weights an experiment changes, or `default`."""
    reward = training.get("reward")
    if not reward:
        return "default"
    return "[" + ", ".join(f"{name}={value:g}" for name, value in sorted(reward.items())) + "]"


def table(names: Optional[List[str]] = None) -> str:
    """The plan as a table: what each experiment is, and what it will cost."""
    chosen = [key for key in EXPERIMENTS if names is None or key in
              {str(name).upper() for name in names}]
    header = (f"{'exp':>3} {'episodes':>9} {'seeds':>7} {'eps decay':>10} "
              f"{'target':>7} {'min buf':>8} {'eval every':>10} {'variant':>11} "
              f"{'reward':>22}  purpose")
    rows = [header, "-" * len(header)]
    for key in chosen:
        spec = EXPERIMENTS[key]
        training = dict(spec["training"])
        rows.append(
            f"{key:>3} {spec['episodes']:>9,} {len(spec['seeds']):>7} "
            f"{training['epsilon_decay_steps']:>10,} {training['target_update_steps']:>7,} "
            f"{training['min_buffer_size']:>8,} {spec['eval_every']:>10} "
            f"{str(spec['variant'] or 'dqn'):>11} {_reward_caption(training):>22}  "
            f"{spec['purpose']}"
        )
    return "\n".join(rows)


__all__ = [
    "COMMON",
    "DEFAULT_OUTPUT_DIR",
    "EVAL",
    "EXPERIMENTS",
    "campaign",
    "experiment",
    "table",
    "training_config",
]
