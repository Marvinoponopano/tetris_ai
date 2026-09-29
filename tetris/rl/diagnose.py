"""Why is the agent playing like that? Measured, not guessed.

When a training run fails to improve, the useful next step is not a bigger run:
it is finding out *which* of the parts is broken. This module plays a loaded
agent greedily on real games and instruments every part of the pipeline, so each
candidate explanation either shows up in the numbers or is ruled out.

What is measured, and what each measurement can rule in or out
-------------------------------------------------------------

**Reward** -- the episode reward is decomposed into its own terms
(``per_line`` / ``hole`` / ``height`` / ``game_over`` / ``per_piece``), exactly as
`RewardConfig` computes it, and each term's share of the total is reported. The
share is the whole story: if 99% of the reward is the height penalty and the
``per_line`` term is zero for every episode, the agent has never *experienced*
the reward that line clearing pays, so nothing in the data can teach it that
clearing lines is good. No amount of extra training fixes that; the reward or
the curriculum has to change.

**Survival** -- pieces per game, deaths vs cap, peak stack height, and how many
episodes ever cleared a line. Distinguishes "never learns to stack flat" (dies
in 20 pieces) from "stacks flat but never clears anything" (dies at the cap with
0 lines).

**Actions** -- the distribution over the 40 placement indices: how many distinct
actions the greedy policy ever chooses, the share taken by its favourite, the
entropy, and the rotation/column histograms. A policy that always picks the same
column is not learning a board-dependent decision at all. Illegal selections are
counted too, and must be zero: the mask is the thing that keeps the action space
honest.

**Q-values** -- the spread between the best and second-best legal Q-value. A
network whose Q-values are all within a hair of each other is not ranking the
options; a network with a huge spread but a bad score has learned something
confident and wrong (overestimation is exactly what `DoubleDQNAgent` is for).

**Agreement with the heuristic** -- the fraction of decisions where the greedy
action equals what `HeuristicAgent` would have played, the mean regret against
that agent's own scoring, and the correlation between the Q-values and the
heuristic's scores over the same legal moves. This is the most direct "does the
network know anything about Tetris?" measurement available without more
training, and it costs one board simulation per legal move.

**State** -- the encoded planes are checked to be what the encoding documents:
the board plane, the current piece drawn where it actually is, the next piece
preview. An all-zero piece plane would mean the network cannot tell which piece
it is placing, which caps every policy at the quality of a piece-agnostic one.

**Exploration** -- the epsilon schedule at the checkpoint's decision count. A
run that ends with epsilon still at 0.5 has not yet been asked to play its own
policy, and its evaluation curve is then a curve of a partly random agent.

The output is a plain dict (saved as JSON) plus a readable summary, so a
conclusion can be quoted with the numbers that support it.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: UP035

import numpy as np

from ..agents import HeuristicAgent
from ..env import RewardConfig, TetrisEnv
from ..metrics import max_column_height
from .config import TrainingConfig
from .dqn import DQNAgent, load_agent_for_evaluation
from .encoding import encode_state, mask_for_state
from .evaluation import DEFAULT_EVAL_SEED

#: How many decisions the (expensive) one-piece-lookahead agreement is measured
#: over. Depth-1 scoring is ~1000 board simulations per decision, so this is
#: deliberately a sample rather than the whole game.
LOOKAHEAD_DECISIONS = 30

#: How many distinct placement indices the action space has (4 rotations x 10).
ACTION_SIZE = 40
WIDTH = 10


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector, in bits (0 = one action always)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """Pearson correlation, or None when it is not defined for this sample."""
    if left.size < 3 or right.size < 3:
        return None
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


@dataclass
class Diagnosis:
    """Everything one diagnostic rollout measured (see the module docstring)."""

    label: str
    checkpoint: Optional[str] = None
    episode: Optional[int] = None
    #: Decisions this diagnostic played (what the action and Q-value blocks count).
    decisions: int = 0
    #: The *training* counters the agent was carrying when it was diagnosed.
    agent_decisions: int = 0
    learn_steps: int = 0
    games: int = 0
    seed: int = DEFAULT_EVAL_SEED
    max_pieces: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    reward: Dict[str, Any] = field(default_factory=dict)
    survival: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    q_values: Dict[str, Any] = field(default_factory=dict)
    heuristic: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    exploration: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint,
            "episode": self.episode,
            "decisions": self.decisions,
            "agent_decisions": self.agent_decisions,
            "learn_steps": self.learn_steps,
            "games": self.games,
            "seed": self.seed,
            "max_pieces": self.max_pieces,
            "config": self.config,
            "reward": self.reward,
            "survival": self.survival,
            "actions": self.actions,
            "q_values": self.q_values,
            "heuristic": self.heuristic,
            "state": self.state,
            "exploration": self.exploration,
        }

    def save(self, path) -> str:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return str(target)

    def summary(self) -> str:
        """The diagnosis as a readable block: what was measured, and what it means."""
        reward, survival = self.reward, self.survival
        actions, q_values = self.actions, self.q_values
        heuristic, state = self.heuristic, self.state
        lines = [
            f"diagnosis: {self.label}",
            f"  checkpoint     {self.checkpoint or '(none)'}"
            + (f"   episode {self.episode}" if self.episode else ""),
            f"  training       {self.agent_decisions:,} decisions, {self.learn_steps:,} "
            f"gradient steps",
            f"  this diagnosis {self.decisions:,} decisions over {self.games} game(s) at "
            f"seed {self.seed}, cap {self.max_pieces}",
            "",
            "  reward -- where the episode reward actually comes from",
            f"    mean episode reward      {reward.get('episode_mean', 0.0):>12,.2f}",
        ]
        for name, share in (reward.get("shares") or {}).items():
            lines.append(
                f"    {name:<24} {reward['components'].get(name, 0.0):>12,.2f}"
                f"   {share * 100:6.2f}% of the total"
            )
        lines += [
            f"    episodes with a line clear   {survival.get('episodes_with_a_line_clear', 0)}"
            f"/{self.games}",
            "",
            "  survival",
            f"    pieces per game          {survival.get('pieces_mean', 0.0):>12,.2f}"
            f"   max {survival.get('pieces_max', 0.0):.0f}",
            f"    games that topped out    {survival.get('deaths', 0):>12}"
            f"/{self.games}   (the rest hit the piece cap)",
            f"    lines per 100 pieces     {survival.get('lines_per_100_pieces', 0.0):>12,.3f}",
            f"    peak column height       {survival.get('peak_column_mean', 0.0):>12,.2f}"
            f"   (20 = the stack reached the top)",
            "",
            "  actions",
            f"    distinct actions used    {actions.get('distinct_actions', 0):>12}/{ACTION_SIZE}",
            f"    favourite action share   {actions.get('top_action_share', 0.0) * 100:>12.1f}%"
            f"   (action {actions.get('top_action')} = rotation "
            f"{actions.get('top_action_rotation')}, column {actions.get('top_action_column')})",
            f"    action entropy           {actions.get('entropy_bits', 0.0):>12.2f} bits",
            f"    illegal selections       {actions.get('illegal_selections', 0):>12}",
            "",
            "  Q-values (legal actions only)",
            f"    mean best Q              {q_values.get('best_mean', 0.0):>12,.3f}",
            f"    best - second best       {q_values.get('spread_mean', 0.0):>12,.3f}"
            f"   (sd {q_values.get('spread_sd', 0.0):.3f})",
            f"    decisions with a tie     {q_values.get('tied_decisions', 0):>12}"
            f"/{self.decisions}",
            "",
            "  against the heuristic agent (no lookahead)",
            f"    agreement                {heuristic.get('agreement_depth0', 0.0) * 100:>12.1f}%"
            f"   of decisions (chance "
            f"{heuristic.get('agreement_chance', 0.0) * 100:.1f}%)",
            f"    mean regret              {heuristic.get('regret_mean', 0.0):>12,.3f}"
            f"   heuristic score units per move",
            f"    moves that end the game  {heuristic.get('suicidal_choices', 0):>12}"
            f"   taken while another move survived"
            f" (forced deaths {heuristic.get('forced_deaths', 0)})",
            f"    Q vs heuristic score r   {_number(heuristic.get('q_score_correlation')):>12}",
            f"    agreement (1 lookahead)  "
            f"{_number(heuristic.get('agreement_depth1'), percent=True):>12}"
            f"   over the first {heuristic.get('lookahead_decisions', 0)} decisions",
            "",
            "  state encoding",
            f"    mean plane sums          {state.get('plane_sums_mean', [])}",
            f"    states with no piece     {state.get('states_without_a_piece', 0):>12}",
            f"    empty action masks       {state.get('empty_masks', 0):>12}",
            "",
            "  exploration",
            f"    epsilon at the checkpoint {self.exploration.get('epsilon_at_checkpoint', 0.0):>11.3f}"
            f"   (decisions {self.agent_decisions:,} of a schedule that ends at "
            f"{self.exploration.get('epsilon_decay_steps', 0):,})",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _number(value: Optional[float], percent: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if percent else f"{value:.3f}"


def diagnose_agent(
    agent,
    label: str = "agent",
    checkpoint: Optional[str] = None,
    games: int = 3,
    seed: int = DEFAULT_EVAL_SEED,
    max_pieces: Optional[int] = 200,
    lookahead_decisions: int = LOOKAHEAD_DECISIONS,
) -> Diagnosis:
    """Play `games` greedy games with `agent` and instrument every part.

    Nothing here trains or explores: it is the same policy the evaluation
    measures, watched more closely. `max_pieces` caps the diagnostic games (the
    question is about the first few hundred pieces, not about endurance).
    """
    config = getattr(agent, "config", None)
    reward_config = config.reward_config() if isinstance(config, TrainingConfig) else None
    reward_config = reward_config or RewardConfig()
    env = TetrisEnv(seed=seed, reward=reward_config)
    critic = HeuristicAgent(lookahead_depth=0)
    lookahead_agent = HeuristicAgent(lookahead_depth=1) if lookahead_decisions > 0 else None

    components = {"per_line": 0.0, "hole": 0.0, "height": 0.0, "game_over": 0.0,
                  "per_piece": 0.0}
    episode_rewards: List[float] = []
    action_counts = np.zeros(ACTION_SIZE, dtype=np.int64)
    rotation_counts = np.zeros(4, dtype=np.int64)
    column_counts = np.zeros(WIDTH, dtype=np.int64)
    plane_sums: List[np.ndarray] = []
    best_values: List[float] = []
    spreads: List[float] = []
    correlations: List[float] = []
    agreements, lookahead_agreements = 0, 0
    lookahead_seen = 0
    regrets: List[float] = []
    regret_shares: List[float] = []
    chance_agreements: List[float] = []
    suicidal_choices = 0
    forced_deaths = 0
    tied_decisions = 0
    illegal = 0
    empty_masks = 0
    states_without_a_piece = 0
    pieces_per_game: List[int] = []
    lines_per_game: List[int] = []
    peaks: List[float] = []
    decisions = 0

    for game in range(int(games)):
        state = env.reset(seed=seed + game)
        total = 0.0
        game_decisions = 0
        while not env.is_game_over() and env.pieces_placed < max_pieces:
            valid = env.get_valid_actions()
            mask = mask_for_state(state)
            planes = encode_state(state)
            plane_sums.append(planes.sum(axis=(1, 2)))
            if not mask.any():
                empty_masks += 1
            if not planes[1].any():          # the current-piece plane
                states_without_a_piece += 1

            action = agent.choose_action(state, valid)
            index = int(action.rotation) * WIDTH + int(action.column)
            if not mask[index]:
                illegal += 1
            action_counts[index] += 1
            rotation_counts[int(action.rotation)] += 1
            column_counts[int(action.column)] += 1
            decisions += 1
            game_decisions += 1

            decision = getattr(agent, "last_decision", None)
            q_row = np.zeros(ACTION_SIZE, dtype=float)
            if decision is not None:
                q_row = np.asarray(decision.q_values, dtype=float)
                legal_values = q_row[mask]
                if legal_values.size:
                    ordered = np.sort(legal_values)
                    best_values.append(float(ordered[-1]))
                    if ordered.size > 1:
                        spreads.append(float(ordered[-1] - ordered[-2]))
                        if ordered[-1] - ordered[-2] <= 1e-9:
                            tied_decisions += 1
            else:
                legal_values = np.zeros(int(mask.sum()))

            # --- the critic: what would the heuristic have played, and how good
            # --- was the move the network picked by that agent's own scoring?
            #
            # `HeuristicAgent.last_scores` marks a placement with -inf when it
            # ends the game, so three separate things are worth counting: an
            # ordinary regret (both moves survive), a *suicidal* choice (the
            # agent picked the move that dies while another move survived), and
            # a forced death (every legal move ends the game).
            critic.choose_action(state, valid)
            scores = {placement: score for placement, score in critic.last_scores}
            finite = {placement: score for placement, score in scores.items()
                      if math.isfinite(score)}
            chance_agreements.append(1.0 / max(1, len(valid)))
            chosen_score = next(
                (score for placement, score in critic.last_scores
                 if (placement.rotation, placement.column) == (action.rotation, action.column)),
                None,
            )
            if scores:
                if chosen_score is not None and not math.isfinite(chosen_score):
                    if finite:
                        suicidal_choices += 1          # died on purpose, with a way out
                    else:
                        forced_deaths += 1             # nothing would have survived
                if chosen_score is not None and math.isfinite(chosen_score) and finite:
                    best_score = max(finite.values())
                    regrets.append(float(best_score - chosen_score))
                    span = max(1e-9, abs(best_score) if best_score else 1.0)
                    regret_shares.append(float((best_score - chosen_score) / span))
                if critic.last_decision is not None and critic.last_decision.action == action:
                    agreements += 1
            if q_row.any() and finite:
                # The same legal moves in the same order: the network's Q-values
                # against the heuristic's own score for each of them.
                indices = sorted(finite, key=lambda placement: placement[0] * WIDTH + placement[1])
                heuristic_scores = np.array([finite[placement] for placement in indices], dtype=float)
                chosen_q = np.array(
                    [q_row[placement[0] * WIDTH + placement[1]] for placement in indices],
                    dtype=float,
                )
                correlation = _correlation(chosen_q, heuristic_scores)
                if correlation is not None:
                    correlations.append(correlation)

            if lookahead_agent is not None and lookahead_seen < lookahead_decisions:
                lookahead_agent.choose_action(state, valid)
                if lookahead_agent.last_decision is not None and lookahead_agent.last_decision.action == action:
                    lookahead_agreements += 1
                lookahead_seen += 1

            # --- the reward, in its own terms (identical to RewardConfig.__call__)
            next_state, reward, done, info = env.step(action)
            components["per_line"] += reward_config.per_line * info["lines_cleared"]
            components["hole"] += reward_config.hole * max(0, info["holes_created"])
            components["height"] += reward_config.height * max(0, info["height_added"])
            if info["game_over"]:
                components["game_over"] += reward_config.game_over
            else:
                components["per_piece"] += reward_config.per_piece
            total += float(reward)
            peaks.append(float(max_column_height(next_state.board)))
            state = next_state

        episode_rewards.append(total)
        pieces_per_game.append(int(env.pieces_placed))
        lines_per_game.append(int(env.game.lines_cleared))

    total_reward = sum(episode_rewards)
    shares = {
        name: (value / total_reward if abs(total_reward) > 1e-9 else 0.0)
        for name, value in components.items()
    }
    counts = action_counts
    top_action = int(np.argmax(counts)) if counts.sum() else -1
    heuristic_block: Dict[str, Any] = {
        "agreement_depth0": agreements / decisions if decisions else 0.0,
        # What agreement means without any skill: pick a legal move at random and
        # you would agree with the heuristic this often. Below it, the policy is
        # not merely unskilled -- it is anti-correlated with a good player's.
        "agreement_chance": float(np.mean(chance_agreements)) if chance_agreements else 0.0,
        "regret_mean": float(np.mean(regrets)) if regrets else 0.0,
        "regret_share_mean": float(np.mean(regret_shares)) if regret_shares else 0.0,
        "q_score_correlation": float(np.mean(correlations)) if correlations else None,
        "suicidal_choices": suicidal_choices,
        "forced_deaths": forced_deaths,
        "lookahead_decisions": lookahead_seen,
        "agreement_depth1": (lookahead_agreements / lookahead_seen) if lookahead_seen else None,
    }

    exploration: Dict[str, Any] = {}
    agent_decisions_count = int(getattr(agent, "decisions", 0))
    if isinstance(config, TrainingConfig):
        exploration = {
            "epsilon_start": config.epsilon_start,
            "epsilon_end": config.epsilon_end,
            "epsilon_decay_steps": config.epsilon_decay_steps,
            "epsilon_at_checkpoint": config.epsilon_for_step(int(getattr(agent, "decisions", 0))),
            "schedule": {
                int(step): round(config.epsilon_for_step(int(step)), 6)
                for step in (0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000)
            },
        }

    return Diagnosis(
        label=label,
        checkpoint=checkpoint,
        episode=int(getattr(agent, "episode", 0)) or None,
        decisions=decisions,
        agent_decisions=int(getattr(agent, "decisions", 0)),
        learn_steps=int(getattr(agent, "steps", 0)),
        games=int(games),
        seed=int(seed),
        max_pieces=max_pieces,
        config=config.to_dict() if isinstance(config, TrainingConfig) else {},
        reward={
            "episode_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episode_sd": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "components": {name: round(value, 4) for name, value in components.items()},
            "shares": {name: round(value, 6) for name, value in shares.items()},
            "per_step_mean": total_reward / decisions if decisions else 0.0,
        },
        survival={
            "pieces_mean": float(np.mean(pieces_per_game)) if pieces_per_game else 0.0,
            "pieces_max": float(max(pieces_per_game)) if pieces_per_game else 0.0,
            "deaths": int(sum(1 for pieces in pieces_per_game
                              if max_pieces is None or pieces < max_pieces)),
            "lines_total": int(sum(lines_per_game)),
            "lines_mean": float(np.mean(lines_per_game)) if lines_per_game else 0.0,
            "lines_per_100_pieces": round(
                100 * sum(lines_per_game) / max(1, sum(pieces_per_game)), 4
            ),
            "episodes_with_a_line_clear": int(sum(1 for lines in lines_per_game if lines > 0)),
            "peak_column_mean": float(np.mean(peaks)) if peaks else 0.0,
            "peak_column_max": float(max(peaks)) if peaks else 0.0,
        },
        actions={
            "distinct_actions": int(np.count_nonzero(counts)),
            "top_action": top_action,
            "top_action_rotation": top_action // WIDTH if top_action >= 0 else None,
            "top_action_column": top_action % WIDTH if top_action >= 0 else None,
            "top_action_share": float(counts[top_action] / counts.sum()) if counts.sum() else 0.0,
            "entropy_bits": _entropy(counts),
            "illegal_selections": illegal,
            "rotation_counts": rotation_counts.tolist(),
            "column_counts": column_counts.tolist(),
            "counts": counts.tolist(),
        },
        q_values={
            "best_mean": float(np.mean(best_values)) if best_values else 0.0,
            "best_min": float(np.min(best_values)) if best_values else 0.0,
            "best_max": float(np.max(best_values)) if best_values else 0.0,
            "spread_mean": float(np.mean(spreads)) if spreads else 0.0,
            "spread_sd": float(np.std(spreads)) if spreads else 0.0,
            "tied_decisions": tied_decisions,
        },
        heuristic=heuristic_block,
        state={
            "plane_sums_mean": [round(float(value), 4) for value in
                                (np.mean(plane_sums, axis=0) if plane_sums else np.zeros(3))],
            "states_without_a_piece": states_without_a_piece,
            "empty_masks": empty_masks,
            "states": len(plane_sums),
        },
        exploration=exploration,
    )


def diagnose_checkpoint(path, label: Optional[str] = None, **kwargs) -> Diagnosis:
    """Load a checkpoint and diagnose it (greedy, no exploration)."""
    agent = load_agent_for_evaluation(path)
    return diagnose_agent(
        agent,
        label=label or pathlib.Path(path).stem,
        checkpoint=str(path),
        **kwargs,
    )


def diagnose_untrained(
    config: Optional[TrainingConfig] = None,
    label: str = "untrained",
    seed: int = 0,
    **kwargs,
) -> Diagnosis:
    """Diagnose a fresh, untrained agent -- the floor a run has to rise above."""
    config = config or TrainingConfig()
    agent = DQNAgent(config=config, seed=seed)
    agent.set_epsilon(0.0)
    return diagnose_agent(agent, label=label, **kwargs)


__all__ = [
    "ACTION_SIZE",
    "LOOKAHEAD_DECISIONS",
    "Diagnosis",
    "diagnose_agent",
    "diagnose_checkpoint",
    "diagnose_untrained",
]
