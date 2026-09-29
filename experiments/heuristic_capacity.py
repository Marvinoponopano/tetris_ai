"""Can this network rank placements the way the heuristic does?

Every arm in `REPORT2.md` lands on the same plateau: ~620-700 score and 0.0-0.47
lines per 100 pieces, against the heuristic's 39.76. Four different fixes --
estimator, objective, credit assignment, representation -- all move the agent
inside that plateau and none move it out. That leaves two candidate explanations,
and they have very different fixes:

1. **The network cannot represent a clearing policy.** The heuristic decides with
   column heights, holes and bumpiness; a 600x256x256x40 MLP over three planes
   may simply be unable to express that ranking, in which case the encoder is the
   wall and no amount of RL tuning helps.
2. **The network could, but Q-learning does not find it.** In that case the
   ranking is reachable and the wall is the learning process -- exploration,
   reward scale, or simply the ~86k gradient decisions one run affords.

The way to tell them apart is to take the reinforcement learning out of the
question. Collect states, label every legal placement with the *heuristic's own*
score, and train the same architecture on the same planes by supervised
regression. Supervised learning has no exploration problem, no bootstrapping and
no reward: if the network cannot imitate the heuristic with labels handed to it,
it is not going to discover it from reward either. If it can imitate well, then
the architecture is fine and the failure is in the learning process.

Two probes are run, differing only in the input planes, because that is exactly
the J arm's change:

    three planes  `encode_state`                 (what G saw)
    five planes   `encode_state_with_features`   (what J saw)

Usage::

    python experiments/heuristic_capacity.py --games 2 --steps 3000
    python experiments/heuristic_capacity.py --games 2 --steps 3000 --out capacity.json

The agreement number is comparable to `diagnose`'s "agreement with the heuristic":
both ask "on what fraction of states does the argmax of the network's scores equal
the heuristic's best placement", so the DQN's 0.102 and this probe's numbers are
the same measurement on different score functions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris.agents import HeuristicAgent  # noqa: E402
from tetris.env import TetrisEnv  # noqa: E402
from tetris.rl.encoding import (  # noqa: E402
    ACTION_SIZE,
    STATE_SHAPE,
    encode_state,
    encode_state_with_features,
    index_for_placement,
)
from tetris.rl.network import QNetwork  # noqa: E402

#: The two encoders the probe compares, named the way the report names them.
ENCODERS = {
    "three planes": (encode_state, STATE_SHAPE),
    "five planes": (encode_state_with_features, (5,) + STATE_SHAPE[1:]),
}


def collect(games: int, seed: int, max_pieces: int = 1000):
    """Play `games` games with the depth-0 heuristic, recording what it ranked.

    Returns the states, the heuristic's score vector per state (`-inf` where a
    placement is illegal) and the action it picked. `HeuristicAgent.choose_action`
    fills `last_scores` with a score per valid placement, in the same order, which
    is the teacher signal this probe regresses.
    """
    teacher = HeuristicAgent(lookahead_depth=0)
    states, vectors, choices = [], [], []
    for game in range(games):
        env = TetrisEnv(seed=seed + game)
        state = env.reset()
        while True:
            actions = env.get_valid_actions()
            if not actions or env.pieces_placed >= max_pieces:
                break
            chosen = teacher.choose_action(state, actions)
            scores = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
            # `last_scores` is one (placement, score) pair per action, in the
            # order the actions were offered
            for placement, score in teacher.last_scores:
                scores[index_for_placement(placement)] = score
            # the teacher's own choice is the label, and at depth 0 it is by
            # definition the argmax of its scores -- assert rather than assume,
            # because the whole probe rests on that alignment
            # the label is what the heuristic *played*, tie-breaks included --
            # the same definition `diagnose` uses, so the numbers compare
            label = index_for_placement(chosen)
            states.append(state)
            vectors.append(scores)
            choices.append(label)
            state, _, done, _ = env.step(chosen)
            if done:
                break
    return np.asarray(states), np.asarray(vectors), np.asarray(choices)


def fit_probe(
    states: np.ndarray,
    vectors: np.ndarray,
    choices: np.ndarray,
    encoder,
    state_shape: Tuple[int, ...],
    hidden_sizes: Tuple[int, ...] = (256, 256),
    steps: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    valid_fraction: float = 0.8,
) -> Dict[str, float]:
    """Regress the heuristic's scores; report how often the argmax matches.

    The split is by *state*, and the held-out states are the last 20% of the
    games, so the probe is scored on positions from games it never trained on --
    imitating a policy on states you trained on measures memorisation.
    """
    encoded = np.stack([encoder(state) for state in states]).astype(np.float32)
    split = int(len(encoded) * valid_fraction)
    train_x, test_x = encoded[:split], encoded[split:]
    train_y, test_y = vectors[:split], vectors[split:]
    train_choice, test_choice = choices[:split], choices[split:]
    legal_test = np.isfinite(test_y)

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    network = QNetwork(state_shape=state_shape, action_size=ACTION_SIZE,
                       hidden_sizes=hidden_sizes)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    x_all = torch.as_tensor(train_x)
    y_all = torch.as_tensor(train_y, dtype=torch.float32)
    legal = torch.as_tensor(np.isfinite(train_y))
    y_all = torch.where(legal, y_all, torch.zeros_like(y_all))
    started = time.perf_counter()
    losses: List[float] = []
    curve: List[Tuple[int, float]] = []
    eval_every = max(1, steps // 20)
    test_x_tensor = torch.as_tensor(test_x)
    for step in range(steps):
        index = rng.integers(0, len(x_all), size=batch_size)
        x = x_all[index]
        y = y_all[index]
        mask = legal[index]
        predicted = network(x)
        # only legal placements carry a target: masked mean squared error
        squared = (predicted - y) ** 2 * mask
        loss = squared.sum() / mask.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % eval_every == 0:
            with torch.no_grad():
                snapshot = np.where(legal_test, network(test_x_tensor).numpy(), -np.inf)
            curve.append((step + 1, float(np.mean(snapshot.argmax(axis=1) == test_choice))))
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        predicted = network(torch.as_tensor(test_x)).numpy()
    predicted = np.where(legal_test, predicted, -np.inf)
    picked = predicted.argmax(axis=1)
    agreement = float(np.mean(picked == test_choice))
    # Tie-tolerant agreement: the probe's pick scores as well as the teacher's
    # best, by the teacher's own scores. `HeuristicAgent` breaks ties by search
    # order, so the exact number above charges the probe for a pick that is
    # equally good by the measure it was trained on.
    teacher_best = np.array([np.max(test_y[i][legal_test[i]]) for i in range(len(test_x))])
    agreement_best = float(np.mean([
        test_y[i][picked[i]] >= teacher_best[i] - 1e-9 for i in range(len(test_x))
    ]))
    tied_labels = float(np.mean([
        (np.sum(np.abs(test_y[i][legal_test[i]] - teacher_best[i]) < 1e-9) > 1)
        for i in range(len(test_x))
    ]))
    both_legal = legal_test & np.isfinite(test_y)
    correlations = [
        float(np.corrcoef(predicted[row][both_legal[row]], test_y[row][both_legal[row]])[0, 1])
        for row in range(len(test_x))
        if both_legal[row].sum() > 2 and np.std(test_y[row][both_legal[row]]) > 1e-9
    ]
    # what the heuristic itself achieves as a ceiling: its argmax is the label
    return {
        "states": int(len(encoded)),
        "train_states": int(split),
        "test_states": int(len(test_x)),
        "agreement": agreement,
        "agreement_as_good_as_best": agreement_best,
        "label_tied_fraction": tied_labels,
        "score_correlation": statistics.mean(correlations) if correlations else float("nan"),
        "final_loss": losses[-1],
        "agreement_curve": curve,
        "loss_first_100": statistics.mean(losses[:100]),
        "steps": steps,
        "seconds": round(elapsed, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2, help="heuristic games to learn from")
    parser.add_argument("--steps", type=int, default=3000, help="supervised gradient steps")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    print(f"collecting states from {args.games} heuristic games (seed {args.seed}) ...")
    states, vectors, choices = collect(args.games, args.seed)
    print(f"  {len(states)} decision states, "
          f"{statistics.mean(np.isfinite(v).sum() for v in vectors):.1f} legal placements each")

    report = {"games": args.games, "seed": args.seed, "probes": {}}
    for name, (encoder, shape) in ENCODERS.items():
        print(f"fitting the same MLP on {name} ({args.steps} steps) ...")
        result = fit_probe(states, vectors, choices, encoder, shape, steps=args.steps,
                           seed=args.seed)
        report["probes"][name] = result
        thin = result["agreement_curve"][::max(1, len(result["agreement_curve"]) // 6)]
        print("  curve " + "  ".join(f"{s}:{a:.2f}" for s, a in thin))
        print(f"  tensor {tuple(shape)}: agreement {result['agreement']:.3f}  "
              f"(tie-tolerant {result['agreement_as_good_as_best']:.3f})  "
              f"correlation {result['score_correlation']:+.3f}  loss "
              f"{result['loss_first_100']:.2f} -> {result['final_loss']:.2f}  "
              f"({result['seconds']}s)")

    print()
    print("agreement is the same measurement `diagnose` reports for the DQN:")
    print("  the DQN (G, greedy) agrees with the heuristic on 10.2% of decisions")
    print("  chance is 1/19 = 5.3%")
    print("  the tie-tolerant column counts a pick that scores equally well as agreement")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
