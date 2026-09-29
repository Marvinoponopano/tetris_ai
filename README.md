# Tetris AI — engine, window, and RL environment

A clean, dependency-light Tetris engine with three front ends on top of one
shared core: a terminal view, a pygame window, and a reinforcement-learning
environment. The core is pure Python (no keyboard/graphics coupling) and the
board is a plain `list[list[int]]` that converts to NumPy with `np.array(...)`,
so the same rules drive a human at the keyboard, a random baseline, an
evaluation harness, and a heuristic agent that plans every piece before it
lands.

## Layout

    tetris_ai/
    ├── play.py              # the launcher: human / AI-train / AI-viz modes
    ├── evaluate.py          # run an agent over many games, headless
    ├── train.py             # train the DQN: episodes in, log + checkpoints out
    ├── main.py              # the terminal demo (same engine, no pygame)
    ├── requirements.txt
    ├── requirements-gui.txt # pygame, only for the window
    ├── requirements-rl.txt  # torch + matplotlib, only for the learning layer
    ├── experiments/         # running and comparing experiments (see below)
    │   ├── run.py           # the CLI: config in, saved results + table out
    │   ├── configs/         # example config files (JSON)
    │   ├── results/         # what runs produced: run-*.json + runs.jsonl
    │   ├── lookahead.py     # depths 0/1/2 on the same seeds -> JSON
    │   ├── rl_plan.py       # what experiment A..G is, and what it costs
    │   ├── rl_experiments.py# the CLI: plan / run / aggregate / plot / compare / diagnose
    │   └── rl/              # campaigns: metrics, checkpoints, aggregate, plots
    │                        #   + REPORT.md, the write-up of the results
    ├── tetris/              # the engine (pure Python, no I/O)
    │   ├── actions.py       # the Action enum
    │   ├── pieces.py        # tetromino shapes + rotation
    │   ├── board.py         # the 10x20 grid + collision
    │   ├── state.py         # GameState snapshot for the AI
    │   ├── game.py          # Game — the environment API
    │   ├── env.py           # TetrisEnv — the RL environment (placements)
    │   ├── metrics.py       # board features (heights, holes, bumpiness, wells)
    │   ├── heuristic.py     # simulate_placement, evaluate_board, the search
    │   ├── agents.py        # the agent interface, RandomAgent, HeuristicAgent
    │   ├── evaluate.py      # evaluate_agent() — many games, JSON/CSV output
    │   ├── config.py        # ExperimentConfig — what to run, on what machine
    │   ├── rl/              # the DQN: encoding, network, replay, agent, training,
    │   │                    #   campaign, evaluation, variants, plots, compare,
    │   │                    #   diagnose (the only place torch is imported)
    │   ├── experiment.py    # run_experiment(), the run log, compare_runs()
    │   ├── app.py           # TetrisApp — timing/pause/restart, no pygame
    │   ├── render.py        # the view: colored blocks in the terminal
    │   └── pygame_view.py   # the view: a window + keyboard input
    └── tests/               # pytest tests for the engine, the views and the env

## Environment API

    game = Game()
    state = game.get_state()              # snapshot of the board
    action = agent.choose_action(state)   # (the AI part, later)
    state, reward, done, info = game.step(action)

`step()` applies exactly one action and returns a gym-style tuple. Gravity is
separate: `game.apply_gravity()` falls one cell (used by the human demo loop).

That is the *frame-level* API: the human window and the terminal demo drive it
with `Action.LEFT`, `Action.ROTATE_CW`, ... For training there is a second,
higher-level API below.

## Reinforcement-learning environment (TetrisEnv)

    from tetris.env import TetrisEnv

    env = TetrisEnv(seed=42)
    state = env.reset()
    while not done:
        action = agent.choose_action(state, env.get_valid_actions())
        state, reward, done, info = env.step(action)

No pygame, no window, no terminal -- `tetris/env.py` imports NumPy and the
engine and nothing else, so episodes can be run in a bare loop.

**Actions are placements** One action places the current piece
and the environment does the rest (rotate, move, hard-drop, lock, clear lines,
spawn the next piece):

    env.step((1, 5))                          # (rotation, column)
    env.step(Placement(rotation=1, column=5))
    env.step({"rotation": 1, "column": 5})

`column` is the board column the piece's *leftmost* cell lands in, so
`range(WIDTH - piece_width + 1)` is the whole range for each orientation.
`get_valid_actions()` returns every one of them for the current piece (9 for an
O, 17 for an I, 32 for a T/J/L), and `step()` raises `ValueError` on anything
outside the walls -- the environment never plays a move you did not ask for.

**State** (`env.get_state()`) is a `TetrisState` of NumPy arrays:

    board               (20, 10) float32   locked cells, 1.0 = filled
    board_with_piece    (20, 10) float32   the same, plus the falling piece
    current_piece       (4, 2)   int8      its cells, board (row, col)
    current_piece_type  (7,)     float32   one-hot piece id
    current_rotation    (4,)     float32   one-hot rotation 0..3
    current_column      int                leftmost column (-1 = no piece)
    next_piece_type     (7,)     float32   one-hot id of the next piece
    next_piece          (4, 2)   int8      its cells in spawn orientation
    queue               (n,)     int8      upcoming ids, queue[0] == next
    score, lines_cleared, level, game_over

`print(state)` prints exactly that table (shapes and values);
`state.feature_vector()` is a flat 219-float observation for a plain MLP.

**Reward** is one small config, easy to change:

    RewardConfig(per_line=10.0, hole=-1.0, height=-0.5, game_over=-100.0,
                 per_piece=0.0)

i.e. `per_line * lines + hole * holes_created + height * cells_grown +
per_piece + game_over` on the last step. Pass different weights, or your own
function: `TetrisEnv(reward=lambda outcome: ...)`, where `outcome` is a
`StepOutcome` with `lines_cleared`, `score_gained`, `holes_before/after`,
`height_before/after`, `game_over` and `pieces_placed`.

`per_piece` is 0 by default, so with no arguments the reward is exactly what it
always was. It pays for *staying alive* (once per piece placed, on the steps
that do not end the episode) and exists because every other per-step term is a
penalty: with `per_piece=0` an agent that has not yet learned to clear a line
earns *less* for surviving longer, and the cheapest episode it can play is the
shortest one. `height` is measured on the *tallest* column (`board_height`), so
only growth of the peak is charged, not the stack's total mass.

**`info`** carries `lines_cleared`, `score`, `score_gained`, `game_over`,
`holes`, `height`, `holes_created`, `height_added`, `placement`, `piece`,
`pieces_placed`, `valid_actions` and `reward`.

**Determinism**: `TetrisEnv(seed=42)` replays the same piece sequence on every
`reset()`, so two agents can be compared on identical episodes. `reset(seed=k)`
switches episodes; an unseeded env is random each time.

**Speed** on this machine: ~5,600 placements/s, ~290 random-policy games/s,
23k state vectors/s -- the drop itself (one row at a time, in the engine, so
every rule stays in one place) is most of the cost.

## Baseline agent and evaluation

`tetris/agents.py` holds the agent interface, the random baseline and the
heuristic agent (below):

    class Agent(Protocol):
        def choose_action(self, state, valid_actions) -> Placement: ...

`RandomAgent(seed=0)` picks uniformly from `env.get_valid_actions()`. It ignores
the state entirely -- it is the floor to beat, and the control that shows what
"no skill at all" scores on a given set of piece sequences. Agents may also
define `reset(seed=None)` (the random one does), which the harness calls before
every game so a whole run is reproducible even when the same agent object is
reused.

`tetris/evaluate.py` plays complete games headless:

    from tetris.agents import RandomAgent
    from tetris.evaluate import evaluate_agent

    result = evaluate_agent(RandomAgent(seed=0), num_games=100, seed=123)
    print(result)                        # small table
    result.stats["score_mean"]           # the numbers as a dict
    result.save("results.json")          # or results.csv (one row per game)

`stats` carries `score_mean/median/min/max/stdev`, `lines_mean/max/total`,
`pieces_mean/max`, `game_length_mean/max`, `games_capped`. Every value in it is
fixed by the seeds -- no wall-clock -- so two runs of the same agent on the same
seed compare equal and results can be diffed. (`game_length_*` equals
`pieces_*` here: one action places one piece, so a game's length is its piece
count; the two figures only diverge in an environment with frame-level control.)

`max_pieces` (default 1000) is a safety cap so a *good* agent -- Tetris can be
played forever -- cannot hang a run; games that hit it are flagged `capped`.
`log_every` controls progress output ("Game 10/100") and `logger=None` silences
it. From the shell:

    python evaluate.py                              # 100 games, random agent
    python evaluate.py --agent heuristic --games 100   # the planning agent
    python evaluate.py --games 1000 --seed 123      # 3.4s, ~300 games/s
    python evaluate.py --save results.json          # or results.csv, or --quiet

    evaluating 'random' over 100 games (seed 123)
    Game 10/100
    ...
    Game 100/100
    agent 'random' -- 100 games, seed 123
      score          mean    471.0   median    469   min   284   max    658   sd   83.1
      lines cleared  mean     0.00   max         0
      pieces         mean     19.1   max        28
      game length    mean     19.1   max        28   (pieces = game length here)
      ran in 0.34s (293 games/s)

## The experiment framework

`evaluate.py` answers "how did this agent do?". `tetris/config.py`,
`tetris/experiment.py` and `experiments/run.py` answer "how did these agents do
on identical games, what did it cost, and can I reproduce it?" -- from a file,
so a result can be quoted instead of remembered.

### Configuration (`tetris/config.py`)

A config is JSON, and it holds exactly what changes the numbers:

    {
      "agent": "heuristic",
      "num_games": 30,
      "seed": 123,
      "max_pieces": 400,
      "lookahead_depth": 1,
      "beam": 4,
      "weights": {"holes": -8.0},
      "label": "one piece of lookahead"
    }

    from tetris.config import ExperimentConfig

    config = ExperimentConfig(num_games=30, seed=123, lookahead_depth=1)
    config.save("experiments/configs/mine.json")          # JSON out
    config = ExperimentConfig.load("experiments/configs/mine.json")
    smaller = config.with_overrides(num_games=5)          # a new, re-validated config
    config.config_id                                      # "same experiment?"

`config_id` hashes everything that *can* change a result (`agent`, `num_games`,
`seed`, `max_pieces`, `lookahead_depth`, `beam`, `weights`) and nothing that
cannot (`label`), so re-running an experiment keeps its id and two different
experiments never share one. Unknown keys are an error rather than being
ignored -- a typo that silently runs a different experiment is exactly what a
config file is supposed to prevent. Weights are stored complete, defaults filled
in, so a saved config never depends on what the defaults were at the time.

### Running and logging (`tetris/experiment.py`)

    from tetris.experiment import run_experiment, compare_runs

    result = run_experiment(config, save_dir="experiments/results")
    print(result)                                  # the run, as a table
    result.stats["score_mean"]                     # the deterministic numbers
    result.timing["games_per_second"]              # ... and the speed
    print(compare_runs("experiments/results/runs.jsonl"))

`run_experiment(config, agent=None, ...)` builds the agent from the config's
`agent` name through the registry in `tetris/agents.py`, **or takes any agent
object you pass in** -- that is the seam a learning agent plugs into later, since
nothing in the framework cares how the agent chooses. `evaluate_agent(agent, ...)`
stays the low-level entry point for an agent with no config. From the shell:

    python experiments/run.py --config experiments/configs/heuristic_depth1.json
    python experiments/run.py --config experiments/configs/*.json
    python experiments/run.py --config base.json --set num_games=10 seed=7
    python experiments/run.py --config x.json --profile        # + a cProfile report
    python experiments/run.py --dry-run --config x.json        # what would run
    python experiments/run.py --list [--csv compared.csv]      # every run so far

Each run writes to `--data-dir` (default `experiments/results`):

    run-<id>.json     the config, the stats, the timing, the environment, and one
                      row per game -- everything a later plot or pairing needs
    runs.jsonl        one JSON object per run: the summary line the tables read
    profile-<id>.txt  with --profile: a cProfile report, top 20 by cumulative time

The run id is the start time plus the config id, so runs never collide and a
result can always be traced back to the config that produced it. The comparison
columns are the ones a comparison has to have: agent, games, seed, score
mean/median/max, lines mean/max, pieces mean, games per second, seconds.

### Reproducibility

`stats` and `diagnostics` are **deterministic**: they depend only on the config
and the seed, so two runs of one config give identical numbers (there is a test
that asserts exactly that, and another that recomputes the saved stats from the
saved per-game rows), and `timing` is the only part allowed to move. Game *i* of
a run is played on `TetrisEnv(seed=seed + i)`, so the same seed means the same
piece sequences for *every* agent -- which is what makes comparisons paired.
Every result also records what produced it: Python version and implementation,
executable, platform, machine, and the versions of numpy and pygame, read from
the installed distributions so an optional library is never imported just to
describe it.

### Throughput

Every run reports `games_per_second`, plus `pieces_per_second` (= decisions per
second, since one action places one piece) and, for an agent that searches,
boards scored per second and per decision. `Metered` -- the wrapper
`run_experiment` puts around every agent -- counts them, forwarding the agent's
own decisions untouched and banking a peak-stack height per game. This is the
number that will matter most once reinforcement learning starts: it is the wall
between an idea and an idea measured over 10,000 games.

### What it looks like

    $ python experiments/run.py --config experiments/configs/random_baseline.json \
        experiments/configs/heuristic_depth0.json experiments/configs/heuristic_depth1.json \
        --quiet
    [1/3] random, 100 games, seed 123, cap 1000
        563.01 games/s, 0.2s -> experiments/results/run-20260925T165943943200-35c5aa358af6.json
    [2/3] heuristic, depth 0, default weights, 30 games, seed 123, cap 400
        3.42 games/s, 8.8s -> experiments/results/run-20260925T165944145878-037b3e25853f.json
    [3/3] heuristic, depth 1, default weights, 30 games, seed 123, cap 400
        0.17 games/s, 172.1s -> experiments/results/run-20260925T165952915011-015f1d0bfa3d.json

         agent  games  seed    cap   score mean     median          max lines mean lines max  pieces  games/s    secs
    -----------------------------------------------------------------------------------------------------------------
        random    100   123   1000        471.0        469          658       0.00         0    19.1   563.01     0.2
     heuristic     30   123    400    158,082.7    158,563      167,412     156.90       159   400.0     3.42     8.8
     heuristic     30   123    400    161,106.5    160,777      168,014     158.60       160   400.0     0.17   172.1

    results in experiments/results (3 new run(s); `--list` shows all of them)

Those score means are the lookahead experiment's numbers to the decimal
(158,082.7 and 161,106.5 at depths 0 and 1) -- the same agents, on the same
seeds, now producing records instead of a terminal scrollback. Only the `timing`
block is allowed to differ (depth 1 came out at 14.3 ms per decision here,
against the 14.0 ms measured on an idle machine), because the statistics are the
part that has to repeat, and they do.

The throughput each agent actually managed, from the saved `timing` and
`diagnostics` (this is the number the framework exists to make visible):

| agent | games/s | seconds/game | pieces/s | boards scored | boards/s | per decision | peak column |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random (100 games) | 563.01 | 0.002 | 10,753 | 0 | -- | 0 | 18.78 |
| heuristic, depth 0 (30) | 3.42 | 0.292 | 1,369 | 277,770 | 31,684 | 23.1 | 10.17 |
| heuristic, depth 1 (30) | 0.17 | 5.738 | 70 | 6,540,727 | 37,997 | 545.1 | 6.53 |

Two things to read off it. Depth 1 scores 6.5 million boards for those 30 games
and still sustains ~38,000 boards/s -- 26 µs per board, the same cost the
lookahead section measured, so the search's price is entirely the branching
factor. And a game of Tetris costs the random agent 2 ms but the depth-1 agent
5.7 s: 2,900x. That is what "10,000 games" means for each agent (under a minute
versus 16 hours), and it is why the throughput line belongs in every record.

## The heuristic agent

`HeuristicAgent` is the first agent that plays well. It has no model, no
training and no randomness: for the current piece it simulates *every* legal
placement, scores the board each one would leave behind, and plays the best.

    board  ->  all legal placements  ->  simulate each  ->  features  ->  weighted score  ->  argmax

    from tetris.agents import HeuristicAgent

    agent = HeuristicAgent()                       # default weights, no lookahead
    agent = HeuristicAgent({"holes": -20.0})       # emphasise one feature
    agent = HeuristicAgent(lookahead_depth=1)      # plan the next piece as well
    action = agent.choose_action(state, valid_actions)

`agent.last_scores` holds `[(action, score), ...]` for the decision just made,
so a move can always be explained after the fact, and `agent.last_decision`
holds the whole search result (see *Looking ahead* below).

### The six features (`tetris/metrics.py`)

`extract_features(board)` returns all six for one board; each also has a
function of its own (`aggregate_height`, `max_column_height`, `count_holes`,
`bumpiness`, `completed_lines`, `wells`) so it can be read and tested alone.
Every one of them takes a board in any shape the project uses — the engine's
`list[list[int]]`, an occupancy array, or `state.board`.

| feature | what it measures | why it matters |
| --- | --- | --- |
| `aggregate_height` | sum of all column heights | the stack's total mass: how much of the well is filled |
| `max_height` | the tallest column | where the game ends; one 16-high column is fatal even if the average is 4 |
| `holes` | empty cells with a block above them | unreachable while the stack stands — every one costs future pieces |
| `bumpiness` | total step between neighbouring columns | flat surfaces accept every piece; jagged ones force bad placements |
| `completed_lines` | full rows on the board | the game's own objective, and the thing that makes holes go away |
| `wells` | depth of one-wide slots below their neighbours | usually a liability; a 4-deep slot next to a wall is the exception (an I-piece home) |

### The weights (`tetris/heuristic.py`)

| weight | default | per |
| --- | --- | --- |
| `lines_cleared` | +4.0 | row completed by this placement |
| `holes` | −8.0 | hole on the resulting board |
| `aggregate_height` | −1.0 | cell of total stack height |
| `max_height` | −1.0 | cell of the tallest column |
| `bumpiness` | −1.0 | cell of surface unevenness |
| `wells` | −1.0 | cell of well depth |

    from tetris.heuristic import HeuristicWeights, evaluate_board

    evaluate_board(board)                        # one number, default weights
    evaluate_board(board, {"wells": -4.0})       # a dict of overrides
    HeuristicWeights(holes=-8.0, bumpiness=-0.4) # or the dataclass
    HeuristicWeights().as_dict()                 # {'lines_cleared': 4.0, ...}

A dict and the dataclass are interchangeable everywhere weights are accepted.
An unknown key raises `ValueError` rather than being silently ignored, and the
whole policy is those six numbers — change one and the agent's play changes.

### Simulating without touching the game

`simulate_placement(board, piece_type, rotation, column)` returns
`Simulation(board, cleared)`: the occupancy the placement would leave, and how
many rows it would clear. It copies the occupancy (`occupancy()` always returns
a fresh array), drops the piece to its resting row — computed straight from the
column tops, so no walking the piece down the board — and clears full rows the
way `Board.clear_lines` does. The board handed in is never modified, and
`agent.choose_action` cannot disturb the environment it is asked about.

That is only worth anything if it predicts reality, so
`tests/test_heuristic.py` checks it against the engine: for random boards and
every piece type, each legal placement is simulated *and* played for real, and
the resulting boards must match cell for cell, along with the line counts.

### Looking ahead (`lookahead_depth`)

Depth 0 scores the board a placement leaves behind, but cannot tell that some of
those boards are dead ends. `lookahead_depth` lets the agent plan the next piece
too:

    current piece -> placement -> next piece -> its best placement

    HeuristicAgent()                                  # depth 0: the greedy planner
    HeuristicAgent(lookahead_depth=1)                 # current piece + the next one
    HeuristicAgent(lookahead_depth=2, beam=4)         # + the one after that

For every legal move of the current piece the search simulates the move, then
searches the next piece's moves on the board it would leave, and adds the best
continuation to the move's own score. A move's value is therefore the sum of the
evaluations along the best line it can see. The move that gets played is still
chosen from *every* legal placement — the `beam` only limits the continuations
looked at below the choice, and only from depth 2 on (depth 1 is exhaustive).

What that costs, on one real mid-game position (a T with a J to follow, 34 legal
placements for each) — a board costs ~25 µs whatever the depth, so the price of a
decision is exactly the number of boards it scores:

| depth | boards scored | time per decision |
| --- | --- | --- |
| 0 | 34 | 0.8 ms |
| 1 | 1,190 (= 34 + 34×34) | 30 ms |
| 2, beam 1 | 2,346 | 59 ms |
| 2, beam 2 | 3,502 | 88 ms |
| 2, beam 4 | 5,814 | 146 ms |
| 2, beam 8 | 10,438 | 262 ms |

(Timings move with the machine: the same measurements came out about twice as
slow while something else was running. The board counts never move.)

The branching factor is the whole story: a piece has 9 legal placements if it is
an O, 17 if it is an I, S or Z, and 34 if it is a T, J or L. Each extra ply
multiplies the work by roughly that number again.

Two kinds of line are not scored at all, because the game would have ended:

* a **lock-out** — the placement settles entirely above the board, so nothing is
  placed and the engine stops the game;
* a board the next piece **cannot appear on**. Pieces always arrive centred, so
  a tower in the middle is fatal even with both side columns wide open. That
  predicate is `spawn_blocked(board, piece_type)`, and it is checked against the
  engine itself, the same way `simulate_placement` is.

Both score `DEAD` (`-inf`), so a move that leaves no room for the piece that
follows is never played while any alternative exists. In the position below a Z
in play and an S to come, the greedy agent plays the upright Z at column 3 (the
best-looking board right now) and the S then has nowhere to appear; depth 1 sees
that, scores it `DEAD`, and drops the Z into the gap between the towers instead.

    ...#..#...        current Z, next S
    ..##..##..        depth 0: (1, 3)  -160.0
    ..##..##..        depth 1: (1, 6)  -328.0   (the depth-0 move: DEAD)
    ..######..        12 safe moves, 5 fatal, 221 boards scored
    ##########

### The lookahead experiment (`experiments/lookahead.py`)

30 games per depth, 400 pieces each, seed 123 — every depth plays the *same*
piece sequences (see *Reproducibility* above), so the numbers compare game by
game, not just on average:

    python experiments/lookahead.py --games 30 --pieces 400 --seed 123

| depth | score mean | score max | lines mean | lines max | lines / 100 pieces | peak column | ms / decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 158,083 | 167,412 | 156.90 | 159 | 39.23 | 10.2 | 0.7 |
| 1 | 161,107 | 168,014 | 158.60 | 160 | 39.65 | 6.5 | 14.0 |
| 2 (beam 2) | 161,291 | 167,794 | 158.70 | 159 | 39.67 | 5.5 | 40.8 |

(`ms / decision` here is averaged over a whole game, so it is much lower than the
worst-case position in the cost table above: the average piece has ~23 legal
placements, not 34, and pieces with 9 moves are cheap.)

Average game length was the 400-piece cap for all three depths: nothing topped
out, so game length does not separate them here and the comparison rests on lines
per piece and on how high the stack had to be played.

What the numbers say:

* **Efficiency was already near its ceiling.** A piece is four cells, a line is
  ten, so no policy can clear more than 40 lines per 100 pieces without wasting
  cells. Depth 0 is at 39.23 (98% of that), depth 1 at 39.65 (99%): there is
  almost nothing left to win by looking ahead.
* **Depth 1 wins that little bit, reliably.** Per game it cleared more lines than
  depth 0 in 20 of 30 games (2 worse, 8 equal), mean +1.70 lines (sd 2.47), and
  its *worst* game (157 lines) beat depth 0's worst (149).
* **What lookahead really buys is room to breathe.** Playing the same pieces,
  depth 1 ran its stack lower than depth 0 in **29 of 30 games** (equal in 1,
  never higher), by 3.63 rows on average; the running peak fell from 10.2 rows to
  6.5, and where depth 0 reached 8+ rows in 29 of 30 games, depth 1 did in 1.
  That headroom is exactly the margin a game dies from — and it is free of any
  charge to efficiency.
* **Depth 2 does not pay for itself.** Over depth 1 it gained 0.10 lines a game
  on average (better in 9 games, worse in 7, equal in 14) for 3x the runtime. It
  did run lower again (5.5 vs 6.5 rows, lower in 21 of 30 games), but nothing in
  these numbers pays for 41 ms per decision.

So: `lookahead_depth=1` to play, depth 0 for bulk evaluation (20x faster, within
1% of the lines), depth 2 for research rather than results.

This script (`experiments/lookahead.py`) is the one that produced the numbers
below, with its own per-game watching and its own JSON. New experiments need not
imitate it: the framework above does the same job from a config file, and running
its depth 0 and depth 1 configs reproduces these two score means to the decimal
(`experiments/configs/heuristic_depth*.json`).

### Watching the agent decide

The search can explain itself: every decision is kept as `agent.last_decision`,
and `debug=True` prints it as it happens.

    agent = HeuristicAgent(lookahead_depth=1, debug=True)
    agent.choose_action(state, valid_actions)

    AI selected:
    Rotation: 1
    Column: 6
    Evaluation: -328.0
      (depth 1 lookahead, 221 boards scored, 17 legal moves, next: S)
      runners-up: (1, 0) -337.0, (1, 8) -340.0, (1, 4) -346.0

`Evaluation` is the total for the searched line, so its size grows with depth and
is not comparable between depths — the *ranking* is what the agent uses, and
`runners-up` shows the margin. `agent.last_scores` still holds every legal move
and its score, in the order the actions were given.

From the command line, without leaving the terminal:

    python evaluate.py --agent heuristic --lookahead 1 --games 5 --trace 2 --trace-board

`--trace N` prints the first N decisions of game 1 (the game the evaluation then
plays), and `--trace-board` draws the position each one was chosen on with the
same renderer the terminal game uses.

### Choosing the weights (`tetris/heuristic.py`)

The defaults above are not a guess and not an optimum either. A scratch script
(not shipped in the package) played 1000-piece games with eleven candidate
weight vectors, including the published Fahey and El-Tetris sets, on seeds
1/600, then re-ran the survivors on 900/901:

* **Efficiency barely moves.** Every sane vector cleared ~38.7–39.8 lines per
  100 pieces. There is no weight-fiddling path to a much better agent here.
* **Survival does.** Peak stack height ran from 9.8 to 17.8 rows and deaths
  from 0/8 to 4/8 games. That is the axis the weights actually control.
* The shipped set (all four penalty weights at −1.0, lines at +4.0) and a
  close relative tied on efficiency while holding the stack lowest.
* The published Fahey/El-Tetris weights transferred *worse* than the tuned ones
  (1–2 deaths in 8 games at 1000 pieces). They were fitted to different
  features — landing height and row/column transitions — and a different,
  frame-level environment.

Re-running that comparison is a loop, not a tool:

    from tetris.agents import HeuristicAgent
    from tetris.evaluate import evaluate_agent

    for weights in ({"holes": -8.0}, {"holes": -20.0}, {"wells": -4.0}):
        agent = HeuristicAgent(weights)
        result = evaluate_agent(agent, num_games=20, seed=1, max_pieces=1000, logger=None)
        print(weights, result.stats["lines_mean"], result.stats["pieces_mean"])

Judge candidates on `lines_mean` **and** on how often they top out — at 1000
pieces a whole class of vectors clears the same lines and only some survive.

### RandomAgent vs HeuristicAgent

Same 100 piece sequences for both (`evaluate_agent(..., num_games=100, seed=123)`,
`max_pieces=1000` — a *good* agent cannot be allowed to run forever):

| | `RandomAgent` | `HeuristicAgent` |
| --- | --- | --- |
| score mean | 471 | 919,005 |
| score median | 469 | 923,387 |
| score min / max | 284 / 658 | 492,580 / 956,890 |
| lines cleared mean / median | 0.00 / 0 | 395.74 / 397 |
| lines cleared max | 0 | 399 |
| lines cleared total | 0 | 39,574 |
| pieces survived mean | 19.1 | 997.4 |
| games that hit the cap | 0/100 | 99/100 |
| score per piece | 24.7 | 921.4 |

Read the medians, not one game: the heuristic's *worst* run scored 492,580
(≈1,000× the random mean) and 1 of the 100 games did top out before the cap.
Uncapped, three games on seeds 500–502 ran **5000 pieces each** (1997–1998 lines,
~22.2M points, 39.94 lines per 100 pieces — 99.9% of the theoretical maximum —
with the peak column riding at 13–15 of 20 rows) without dying: the run has to be
stopped, not the agent. One decision — scoring every legal placement and picking
the best — costs ~0.7 ms, so the 100-game evaluation takes ~70 s.

    python evaluate.py --agent heuristic --games 100 --seed 123
    python evaluate.py --agent random    --games 100 --seed 123   # same pieces
    python evaluate.py --agent heuristic --lookahead 1 --games 100 --seed 123

(The heuristic numbers here are the *default* agent, depth 0. `--lookahead 1`
plans the next piece as well: see *The lookahead experiment* below for what that
buys and what it costs.)

## Training the DQN agent

`tetris/rl/` is the learning layer, and the only place PyTorch is imported:
`pip install -r requirements-rl.txt`. PyTorch is optional: the engine, the
environment, the agents, the harness and the tests all work without it, and the
tests skip these modules when torch is missing.

    tetris/rl/encoding.py    state -> (3, 20, 10) binary planes; placement -> 0..39
    tetris/rl/network.py     the Q-network: 600 inputs -> 40 Q-values
    tetris/rl/replay.py      the replay ring, sampled at random
    tetris/rl/config.py      TrainingConfig: the knobs, validated, JSON round-trip
    tetris/rl/dqn.py         DQNAgent: epsilon-greedy acting, the Bellman update,
                             the target network, checkpoints
    tetris/rl/training.py    the episode loop, its log and its checkpoints

    tetris/rl/evaluation.py  greedy (epsilon 0) evaluation on fixed seeds
    tetris/rl/campaign.py    one config x N seeds, evaluated as it trains
    tetris/rl/variants.py    agent variants (a diagnosis becomes a subclass)
    tetris/rl/plots.py       the figures
    tetris/rl/compare.py     the same games for dqn vs the three baselines
    tetris/rl/diagnose.py    why the policy plays the way it does

### One run

    python train.py --episodes 300 --seed 0 --checkpoint-dir checkpoints/A

writes `train_log.csv` (one row per episode), `checkpoint_<episode>.pt`,
`training_summary.json` and `config.json`. Every episode is seeded
(`reset(seed=seed + i)`), so a run is reproducible; learning starts after
`min_buffer_size` transitions; and hitting `max_pieces` truncates an episode
without it counting as a terminal state. Resuming continues from `--resume`.

### An experiment is a campaign, not a run

One training run is an anecdote: the same config on a different seed gives a
different curve. A **campaign** is one config across N seeds, evaluated on the
same schedule while it trains:

    for every seed:      train campaign.eval_every episodes
                           -> greedy evaluation on eval_games fixed seeds
                           -> write the metrics, keep the run if it is the best
                         repeat until episodes run out
    then:                aggregate over seeds (mean, sd, per-seed deltas)

    training
       ↓                       (every N episodes)
    evaluation on fixed seeds   greedy, exploration frozen
       ↓
    save metrics  ─────────────> metrics.jsonl
       ↓
    continue training

    experiments/rl/<EXP>/seed<k>/metrics.jsonl   one JSON object per line
                    seed<k>/summary.json         that seed's run and evaluations
                    seed<k>/best/best_*.pt       the best checkpoint so far
                    seed<k>/training/            what train.py always writes
                    aggregate.json               the cross-seed summary
                    plots/                       the figures

Every training row carries **episode, reward, score, lines, pieces, loss_mean,
epsilon and the wall time**; every evaluation row carries the same score /
lines / pieces on fixed seeded games, plus `lines_per_100_pieces` (the
cap-independent one). Rows are flushed as they are produced, so a run that dies
at episode 900 still leaves 900 usable rows.

### The evaluation protocol

Evaluation runs the policy *greedily* (`epsilon=0`) on `eval_games` games with
`TetrisEnv(seed=eval_seed + i)`, which are exactly the games
`tetris.evaluate` and the comparison below use. Training reward is measured
while exploring and is noisy; the evaluation is the number to trust. The
untrained policy is measured the same way at episode 0, which is the only
honest "before".

### Checkpoints

The best model is kept automatically, by a stated metric: **`score_mean` on the
evaluation games** (configurable through `--metric`, and the tracker knows how
to minimise loss instead). A run only replaces its `best/` checkpoint when the
metric improves; every improvement is written under a new name that contains the
episode and the metric (`best_ep001500_score_mean_608.500.pt`), so no
checkpoint is overwritten and the whole progression is on disk afterwards.
`best.json` in the same directory says which one is current and why.

### The plan, and the five commands

`experiments/rl_plan.py` is the plan: what each experiment is, how long, on
which seeds, and what it changes. The CLI runs it:

    python experiments/rl_experiments.py plan                        # what will run
    python experiments/rl_experiments.py run --experiment C --parallel 5
    python experiments/rl_experiments.py aggregate --experiment C    # + --save
    python experiments/rl_experiments.py plot --experiment C --comparison experiments/rl/comparison.json
    python experiments/rl_experiments.py compare --experiment C --seed 20260928 --games 8 \
        --primary heuristic
    python experiments/rl_experiments.py diagnose --checkpoint foo.pt --games 3

`run --seeds 0 --episodes 20 --eval-every 5 --eval-games 1` shrinks any
experiment for a smoke test; the plan itself never changes.

The plan is three things. **A/B/C** are the size ladder (300 / 2,000 / 6,000
episodes, otherwise identical) that answers "does more training help". **D-G** are
the targeted changes that the first diagnosis turned into single-knob experiments
against C: `D` Double Q-learning, `E` a reward that pays per piece survived, `F` an
epsilon schedule that anneals inside the run, and `G`, the one deliberate
two-knob arm, `E + F`. Adding an arm is a dict in `rl_plan.py`; a test pins that
D/E/F differ from C in exactly the knob they claim to, so "targeted" stays true.

**H-N** are round two, and each is **G with exactly one thing moved** (G, not C,
because G is the best agent here and its own diagnosis is what these target):

    H  reward ratio:   per_line 10 -> 50, per_piece 5 -> 1     (the swap rate)
    I  n-step returns: n = 3                                   (credit assignment)
    J  state planes:   3 -> 5, adding column heights and holes  (representation)
    K  dueling head:   Q = V + A - mean(A)                     (architecture)
    L  prioritised replay: alpha 0.6, beta 0.4 -> 1.0          (sample efficiency)
    N  queue plane:    + queue[1], the piece after next        (lookahead info)

Read `N` against `J`, not against G: N is J plus the queue plane, so that pair
answers "does the queue help" with one plane between them. A test pins the same
single-change property for H-N that D/E/F have against C.

The machinery those arms needed lives in `tetris/rl/`:

    nstep.py       turn a stream of one-step transitions into n-step ones
    prioritized.py sum-tree replay, sampled by TD error, with importance weights
    registry.py    which agent class wrote a checkpoint, so it loads back as itself

and the training loop reaches the encoder and the buffer through methods on the
agent (`encode`, `build_buffer`, `remember`, `end_episode`, `sample_minibatch`)
rather than calling them directly. The reference agent's defaults are exactly the
calls the loop used to make, which is checked two ways: `tests/test_rl_seams.py`
asserts each against its old call, and the 250-episode replay in
`~/.hermes/cache/scratch/seam_regression.py` diffs every field of every episode
row of G/seed0 against the log that run wrote before the change.

### The figures

`plot` writes four figures per campaign, all of them with the measurements
visible:

    01_training_reward.png    reward per episode
    02_lines_per_episode.png  lines cleared per episode
    03_evaluation_score.png   greedy score on fixed seeds (error bars = SE over
                              games) + lines per 100 pieces, with the baselines
                              from --comparison drawn as reference lines
    04_loss.png               Huber loss per episode
    05_baselines.png          every agent on the same protocol (from compare)

Every time series is drawn **raw first**: the light grey line is one measurement
per episode, and the bold line is a *trailing* mean (the last 20 measurements at
each point, never centred, so no point is influenced by measurements that had
not happened yet). The caption says how many raw points are behind it, so a
figure cannot be quoted without that being obvious.

A baseline that is too far outside the curves to draw (the heuristic agent
scores ~930,000 against a DQN's ~550) is **named in the corner with its value**
rather than drawn: a line at 930,000 would flatten the very curve the figure
exists to show. The comparison figure (`05_baselines.png`) plots them properly,
with a log axis when the agents span orders of magnitude.

### Comparison against the baselines

`compare` plays the same seeded games (`game i = seed + i`) for the DQN
checkpoints and for `random`, `heuristic` (depth 0) and the `lookahead`
heuristic (depth 1), then pairs them game by game: score/lines/pieces deltas,
wins-losses-ties and an exact sign-test p-value. `--primary` chooses the
baseline the others are paired against (default random; for a learning agent the
heuristic is the interesting one, because beating random proves little). The protocol in the comparison
is the campaign's own evaluation protocol, so a training curve and the table are
the same measurement.

### Diagnosis

`diagnose` answers "why does it play like that" with measurements rather than
opinions: the reward decomposed into its terms (and each term's share of the
total), how the pieces were distributed over the 40 actions, the Q-value spread
and how often it is tied, how often the greedy move agrees with the heuristic
agent (against the *chance* agreement, `1/legal moves`), the regret by the
heuristic's own scoring, how often it chose a move that ends the game while
another move survived, the replay buffer's contents, and the epsilon it was
actually exploring at. `--save` writes the whole thing as JSON.

### What came of it

`experiments/rl/REPORT.md` is the write-up: the numbers of each experiment, what
the diagnosis found, which targeted changes were tried and whether they helped.
It is the file to read before believing any single curve in `experiments/rl/`.

**Round two** (`experiments/rl/REPORT2.md`) asks the harder question — what stops
the agent from being *good* — with six more one-change arms against G, and it is
the round where the answer turned out to be inconvenient:

    arm                              change vs G            score tail   seeds better
    H  reward ratio                  per_line 10->50,         624.1         0/5
                                     per_piece 5->1          (-74.3)
    I  n-step returns (n=3)                                   679.4         1/5
                                                            (-19.1)
    J  five state planes             +column heights,         702.4         2/5
                                     +holes                  (+3.9)
    K  dueling head                  Q = V + A - mean(A)      730.0         4/5   <-- the one that worked
                                                            (+31.5)
    L  prioritised replay            alpha 0.6, beta          539.4         2/5
                                     0.4->1.0               (-159.0)  2/5 seeds collapsed
    N  queue plane                   +queue[1]                685.3         2/5
                                                            (-13.1)
    M  dueling + five planes         (the combination)        713.8         3/5
                                                            (+15.4)

K is the improved agent: +31.5 mean score, 4/5 seeds, better on every tracked
metric, and steadier than G (median 809 against 720 on the final comparison).
Everything else was neutral or worse, and that is the point of reporting it.

`experiments/heuristic_capacity.py` then answers *why* the DQN stays far behind
the heuristic, by removing reinforcement learning from the question: label every
legal placement with the heuristic's own score and fit the same architecture by
supervision. Agreement with the heuristic goes 10.2% (the trained DQN) → 32.7%
(three planes) → **45.0% (five planes)**, with the teacher's score ordering matched
at +0.86 correlation. The network is not the wall; the learning signal and its
budget are. `experiments/final_comparison.py` prints the fixed-seed table over
random / heuristic / lookahead / initial RL / improved RL, and
`checkpoints/best.pt` points at K so `python play_ai.py` shows the improved agent.


## Running the tests

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -r requirements-gui.txt   # only needed for the window
    pytest

## Playing in a window (pygame)

    pip install -r requirements-gui.txt
    python play.py

A 568x672 window with the 10x20 well, the falling piece, the locked blocks in
their tetromino colors, a ghost outline of where the piece will land, and a
panel with score, lines, level and the next three pieces. Game over and pause
are drawn over the well.

    left / a        move left          right / d   move right
    down / s        soft drop         space       hard drop
    up / w / x      rotate clockwise  z           rotate counter-clockwise
    p               pause             r           restart
    g               ghost piece       esc / q     quit

Useful options: `--seed 42` (the same seed always deals the same pieces, and
`r` replays them), `--fps 60`, `--cell 24` (window scale), `--grid none|dots|lines`,
`--no-ghost`, `--headless` (SDL's dummy driver, no window), `--frames 120` (quit
after N frames) and `--screenshot frame.png`.

The three modes share one engine and differ only in *who decides* and *whether
there is a view*:

    python play.py                  # human: keystrokes -> Action -> engine
    python play.py --mode train     # AI policy, no view (part 2)
    python play.py --mode viz       # AI policy, drawn in the window (part 2)

`viz` is wired: an AI plays in the same window a human plays in. `train` has no
window by design -- training happens through the campaign CLI, which runs seeds in
parallel and evaluates them as they go (`python experiments/rl_experiments.py run
--experiment G --parallel 5`), so the mode refuses rather than pretending.

### Watching the agent play

    python play.py --mode viz --seed 5              # the best trained checkpoint
    python play.py --mode viz --agent heuristic     # the hand-written baseline
    python play.py --mode viz --agent random        # the floor
    python play.py --mode viz --agent heuristic --lookahead 1

The AI-specific options:

    --agent {dqn,heuristic,random}   who plays (default dqn)
    --checkpoint PATH                a specific .pt; default is the best on disk
    --think-ms MS                    pause between moves (default 80; 0 is instant)
    --gravity                        let the piece fall on its own between moves
    --until-game-over                stop when the game ends, not after --frames
    --verbose                        log every decision to the terminal
    --record DIR                     one PNG per frame (--record-every N to thin)

`play.py --mode viz` pairs the policy with `PolicyController`, which is worth
knowing about because the two sides speak different languages: a policy answers
with a **whole placement** (rotation, column -- one decision per piece), while the
engine takes **one action per frame** (rotate, slide, hard drop). The controller
holds the plan for the piece in flight and emits the moves that carry it out, so
you see the piece rotate and slide rather than teleport. It reads the state it
decides from through `TetrisEnv.get_state()`/`get_valid_actions()` -- the same
calls the evaluation harness makes -- so there is one source of truth for what is
legal, and no second simulation to drift. A move the stack blocks is detected as a
no-op and abandoned (the piece is dropped where it stands) instead of retrying
forever; the counters at the end of a run report how often that happened.

`experiments/rl/demo/dqn-g-best-seed5.mp4` is a recording of one such game (the
trained agent, seed 5: 29 pieces, score 550, no lines cleared, tops out), and
`experiments/rl/demo/still_filmstrip.png` is a five-panel summary of the same game.
Regenerate the frames with `--record` and join them at 30 fps:

    python play.py --mode viz --headless --seed 5 --until-game-over \
        --cell 32 --think-ms 80 --record /tmp/frames
    ffmpeg -framerate 30 -i /tmp/frames/frame_%05d.png -c:v libx264 \
        -pix_fmt yuv420p experiments/rl/demo/dqn-g-best-seed5.mp4

The capture is real time (the loop runs at `--fps`), so the frame rate you encode
at is a speed factor: this recording was captured at ~49 fps and encoded at 30, so
it plays about 1.6x slower than life -- slow enough to see the slides.

### `play_ai.py`: the full AI viewer

`play.py --mode viz` is the minimal way to watch an agent in the game's own
window. `play_ai.py` is the dedicated viewer: a bigger board, a panel about the
*AI* rather than the player, speed control that outruns the frame rate, and an
optional debug overlay.

    python play_ai.py                              # the best checkpoint on disk
    python play_ai.py --model checkpoints/best.pt  # a specific saved model
    python play_ai.py --agent heuristic --lookahead 1
    python play_ai.py --speed very-fast --debug
    python play_ai.py --config play_ai.json        # the same options, in a file

With no `--model` the viewer looks up the best checkpoint itself (`best_checkpoint()`
reads the campaigns' `best.json` records), which is why the first form is usually
enough. `checkpoints/best.pt` is a relative symlink to that same model, so the
second form works as written and still resolves into the campaigns -- the viewer
then knows which campaign trained it and reports that campaign's reward weights
rather than falling back to the defaults. A path that does not exist gets an error
listing the checkpoints that do; `checkpoints/README.md` explains re-pointing the
link when a different seed wins.

The window shows the board, the falling piece and its landing preview, the locked
blocks, and a panel with these rows -- game numbers on the left, AI numbers below:

    STATUS / SPEED / ACTION / AGENT / MODEL     what the AI is doing right now
    SCORE / LINES / LEVEL / PIECES / EPSILON    the game, and the agent's own counters
    NEXT                                        the next three pieces
    LAST MOVE                                   the piece just locked: its shape
                                                (with a colour swatch), the chosen
                                                rotation and column, the agent's own
                                                evaluation of that choice, the reward
                                                it earned, and the board it produced

`ACTION` is the engine move in flight (`rotate_cw`, `left`, `drop`, ...), so you can
see the plan being carried out one frame at a time rather than a piece teleporting.

Keys (the AI moves the pieces, so the keyboard is the spectator's):

    SPACE      pause / resume          +  /  -    faster / slower
    R          restart                 G          toggle the landing preview
    D          debug overlay           ESC / Q    quit

Speed is a table, and the `+`/`-` keys move up and down it:

    Slow        400 ms/move
    Normal      120 ms/move
    Fast         20 ms/move
    Very fast    no wait, 6 moves per painted frame

`Very fast` is the interesting one: it takes several engine moves per painted frame,
so the AI is not capped by the frame rate (a 30-piece game in a couple of seconds).
The panel names whichever level is in use.

`--debug` overlays the same numbers on the board, in the form they are usually asked
for: chosen rotation, chosen column, Q-value (or search score), reward, lines cleared.

**The AI is not a re-implementation.** The viewer builds the same agent object the
training loop and the evaluation harness use and calls the same
`choose_action(state, valid_actions)`, with the state from `TetrisEnv.get_state()`.
The only difference from a headless run is that a clock and a canvas are attached:

    training        headless env   ->  as fast as the CPU allows
    viz (play.py)   same env       ->  human-readable pace
    play_ai.py      same env       ->  the same, plus speed control and the AI panel

`tests/test_play_ai.py` pins that down: the same checkpoint and seed are played twice,
once through `TetrisEnv.step()` and once through the window, and the placements *and
the per-piece rewards* have to come out identical -- so if the viewer ever grew its own
decision-making, or its own idea of the reward, the test would fail.

**The reward on the panel is the training reward.** It is computed with the
environment's own `count_holes` / `board_height` / `occupancy` and the environment's
own `RewardConfig`, whose weights the viewer reads back from the campaign that trained
the model (`experiments/rl/G/campaign.json` records `reward.per_piece = 5.0`); override
them with `--reward-per-piece`. That matters, because `per_piece` is the difference
between "board filled up: -40 reward" and the quantity the agent was actually
optimising.

**The model is only ever read.** The checkpoint is opened read-only, and the viewer
fingerprints it (size, mtime, sha256) before and after the run and prints whether it is
unchanged -- so closing the window, or killing it, cannot corrupt a model.

Verifying the eight things this viewer promises:

| promise | test |
| --- | --- |
| the model loads | `test_the_default_model_is_the_best_checkpoint_on_disk`, `test_an_explicit_model_is_loaded_and_named` |
| the AI plays automatically | `test_the_ai_plays_without_any_input` |
| pause works | `test_space_pauses_the_ai_and_pauses_it_again_to_resume` |
| restart works | `test_r_starts_a_fresh_game_and_the_ai_keeps_playing` |
| speed changes | `test_the_speed_keys_step_up_and_down_the_table`, `test_very_fast_moves_more_than_once_per_frame` |
| game over works | `test_the_game_runs_to_game_over_and_the_run_can_stop_there` |
| the same model evaluates headlessly | `test_the_viewer_plays_the_same_game_as_the_headless_environment` |
| closing the viewer cannot corrupt the model | `test_the_viewer_leaves_the_model_file_untouched`, `test_a_quit_event_closes_the_viewer_without_an_error` |

    .venv/bin/python -m pytest tests/test_play_ai.py -q

## Playing in the terminal

    python main.py

The same game, drawn with colored half-block characters (see below) and driven
by single keystrokes: `a`/`d` move, `s` soft drop, `w`/`x` rotate, `z` rotate
counter-clockwise, `space` hard drop, `r` restart, `p` pause, `q` quit.

### How the blocks are drawn

`tetris/render.py` is the only module that does output. A character cell is
about twice as tall as it is wide, and the half-block glyphs (`▀`/`▄`) let one
character show two stacked pixels, so two characters side by side are a 2x2
pixel square — exactly one board cell. The well is therefore 20 columns x 20
lines, the real 1:2 aspect of a 10x20 board, with square blocks. Each block
gets a 24-bit color and a bevel (light top-left, dark bottom-right) so
neighbouring blocks stay separate.

Rendering is usable from code too: `render.frame(game)` gives a whole screen,
`render.board(game)` just the framed playfield, and `Renderer(color=False)` is
a plain ASCII fallback for logs and dumb terminals. `Game.render()` is still
the engine's minimal ASCII debug view.

### How the layers fit together

    tetris/game.py        the engine: board, pieces, scoring, gravity. Pure
                          logic -- no pygame, no keyboard, no printing.
          ^
          |  Game API: step / apply_gravity / reset / is_game_over / get_state
          |
    tetris/app.py         TetrisApp: the frame loop and timing (gravity, pause,
          ^               restart, game over). Still no pygame.
          |
    controller            where decisions come from: HumanController (pygame
    .choose_action(game)  keys -> Action), or PolicyController wrapping an
    view .draw(app)       agent's placement. Where pixels come from: PygameView,
                          render.py, or nothing at all.

`pygame_view.py` contains no game rules: it only calls `app.tick()`,
`app.restart()`, `app.toggle_pause()` and the read-only `Game` queries the
painting needs. So an agent takes over by implementing one method --

    class AIPolicy:
        def choose_action(self, state, valid_actions):   # env.get_state() is
            return (rotation, column)                    # already shaped for it

-- and `play.py --mode viz` is then `TetrisApp(game, PolicyController(AIPolicy()))`
plus the same `PygameView`; `--mode train` is the same thing with no view attached
and `gravity=False`, so the policy drives every move as fast as the CPU allows.
`PolicyController` is the adapter described under "Watching the agent play": it
turns one placement into the rotate/slide/drop moves the engine understands, in
whatever order that piece needs them.

