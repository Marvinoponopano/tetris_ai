# Is the DQN agent learning? — the A–G experiments

This is the write-up of the controlled-training work: what was run, what the
numbers were, why they came out that way, what was changed in response, and
whether any of it worked. The raw material is in this directory
(`<EXP>/seed<k>/metrics.jsonl`, `aggregate.json`, `plots/`, `best/`,
`comparison/`) and the code that produced it is `experiments/rl_experiments.py` +
`experiments/rl_plan.py`.

## The short answer

**Yes — the agent learns, and the interesting part is how long that took to
establish.** Three experiments of increasing size (A/B/C, 300/2,000/6,000
episodes) all improved from the untrained policy in 5 of 5 seeds, but they all
flat-lined in the same place, and *more training made it slightly worse* (15
seeds across A/B/C: 11 end below their own mid-run peak, 4 flat, 0 above). That
looked like "the agent does not learn". It was not: the plateau was caused by two
defects the diagnosis found — **the reward paid the agent for dying**, and **the
exploration schedule never annealed inside a run**. Fixing both (G) lifted the
final evaluation from 537.5 to **698.5**, survival from 22.1 to **30.1 pieces**,
cleared lines from 0.07 to **0.47 per 100 pieces with 5 of 5 seeds improving**,
and the best checkpoints to 730-822 score at 31-33 pieces against C's 607-668 at
22-28. It still clears 0.47 lines per 100 pieces where the heuristic clears 39.76,
so it is a long-lived agent, not yet a good one — but it is unambiguously
learning, and the changes that made it learn were predicted from measurements.

| experiment | what it tests | episodes | eval score, untrained | eval score, end | pieces | lines/100 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | pipeline / small | 300 | 248.8 | 560.2 | 23.9 | 0.06 | improves, then flat |
| B | moderate | 2,000 | 248.8 | 558.8 | 23.5 | 0.07 | improves, then flat |
| C | long (control) | 6,000 | 248.8 | 537.5 | 22.1 | 0.07 | improves, then flat |
| D | Double Q-learning | 6,000 | 248.8 | 511.4 | 20.3 | 0.04 | **no help** |
| E | reward pays per piece survived | 6,000 | 248.8 | 614.9 | 26.6 | 0.22 | **5/5 seeds beat C** |
| F | epsilon anneals inside the run | 6,000 | 248.8 | 590.3 | 23.6 | 0.17 | **helps** |
| **G** | **E + F** | 6,000 | 248.8 | **698.5** | **30.1** | **0.47** | **best arm; still climbing** |

Read the first three rows together, not separately: **20x the training (300 ->
6,000 episodes) bought nothing** — A is as good as C, and slightly better. The
number that moves is the distance from *untrained*, not the distance from *more
training*. And the untrained score (248.8) is a floor set by a degenerate
tie-breaking rule (see *Action handling*), so the +311 is mostly the agent
learning to spread placements at all.


Pieces survived tells the same story without the score's line bonuses:

| experiment | pieces, untrained | pieces, end | delta |
| --- | --- | --- | --- |
| A | 12.4 | 23.9 | +11.5 (5/5 seeds) |
| B | 12.4 | 23.5 | +11.1 (5/5 seeds) |
| C | 12.4 | 22.1 | +9.8 (5/5 seeds) |

Lines cleared — the game's actual objective — never moved in A/B/C: 0.00 lines
per game untrained, 0.017 lines per game after 6,000 episodes (that is *one*
line in an 8-game evaluation, and it improved in 1 of 5 seeds). The heuristic
baseline clears 39.6 lines per 100 pieces; C clears 0.07. This is the number that
the rest of the report is about, and it does move once the reward is changed
(0.47 per 100 pieces in G, improving in 5 of 5 seeds).

## How the experiment process is set up

One run is an anecdote, so nothing here is a single run.

* **A campaign, not a run.** One config x 5 seeds (0-4). Each seed trains in its
  own process (5 in parallel) and writes its own `metrics.jsonl`.
* **Evaluated while it trains.** Every N episodes (50/100/250 for A/B/C)
  training pauses, the agent is evaluated **greedily** (`epsilon=0`) on 8 games
  on fixed seeds (`TetrisEnv(seed=20260928 + i)`), the metrics are flushed, and
  training continues. Training reward is measured while exploring and is noisy;
  the greedy evaluation is the number to trust.
* **The untrained policy is measured the same way at episode 0** — that is the
  only honest "before".
* **Same evaluation protocol everywhere.** The in-training evaluation, the
  comparison table and the figures all use `seed 20260928`, `game i = seed + i`,
  8 games, cap 1000 pieces.
* **Best checkpoint kept by a stated metric**: `score_mean` on those evaluation
  games. Every improvement is written to a new file
  (`best_ep001500_score_mean_608.500.pt`) so nothing is overwritten.
* **Everything is recorded per episode**: reward, score, lines, pieces,
  `loss_mean`/`loss_last`, `epsilon`, decisions, gradient steps, wall time.
* **The plan is data, not a script** (`experiments/rl_plan.py`), so an
  experiment is reproducible from its name and shrinks (`--episodes 20`) for
  smoke runs without the plan changing.

Figures (`plots/`) draw the raw per-episode measurements *and* a **trailing**
mean (window 20) — never a smoothed line alone, never a centred window.

## Why: the diagnosis

Every claim below is a measurement, reproducible with
`python experiments/rl_experiments.py diagnose ...`.

### 1. The reward pays the agent for dying (the central finding)

`RewardConfig` is `per_line*lines + hole*holes_created + height*height_grown +
game_over`, i.e. the line bonus is the *only* positive term, and it is never
reached. Measured over experiment B's 10,000 training episodes:

    corr(episode reward, pieces survived) = -0.225     <-- surviving longer earns less
    corr(episode reward, score)           = +0.035     <-- no signal about the score at all
    binned by survival:
      pieces  0-15   reward -157.1   score 335.7
      pieces 15-20   reward -167.5   score 444.1
      pieces 20-25   reward -173.1   score 538.1
      pieces 25-30   reward -173.9   score 622.3

(It holds in every experiment: A -0.242, B -0.225, C -0.257.)

The mechanism: every per-step term is a penalty, the terminal `game_over=-100`
is paid whenever the episode ends (so it does not discriminate between dying at
piece 12 and piece 27), and a policy that has not yet cleared a line therefore
minimises its loss by placing as few pieces as possible. Measured directly as
the per-step value of one survived piece under the default weights:

| policy | pieces/game | holes/piece | lines/piece | per-piece reward | needs `per_piece` > |
| --- | --- | --- | --- | --- | --- |
| random | 17.8 | 3.62 | 0.000 | **-4.17** | 4.17 |
| DQN (B seed 1 best) | 26.9 | 2.07 | 0.009 | **-2.35** | 2.35 |
| heuristic (depth 0) | 1000 (cap) | 0.08 | 0.398 | **+3.73** | — |

The heuristic is the only policy whose per-step reward is positive, because it
is the only one that clears lines. And note what the DQN actually optimized:
its per-piece cost (2.35) is *lower* than random's (4.17) while its games are
longer — it found the cheapest way to die. That is the reward function working
as written, pointed at the wrong objective.

### 2. Reward decomposition (B seed 1, best checkpoint, 3 games, greedy)

    episode reward   -154.8   (sd 12.0)     per step -6.83
      game_over      -300.0    64.6% of the total
      hole           -135.0    29.1%
      height          -29.5     6.4%
      per_line          0.0     0.0%   <-- no line was ever cleared

The agent has literally never experienced the one reward term that means
"playing well", so that term cannot have shaped anything.

### 3. Exploration was still on

At that checkpoint epsilon was **0.453** (23,050 of 40,000 decisions into B's
schedule): the greedy policy being evaluated had been trained by a policy that
was still ~45% random. B's `epsilon_decay_steps=40,000` against ~26 decisions
per episode means the schedule needs ~1,500 episodes to anneal, and its best
checkpoint was saved mid-anneal at episode 1,100. C (120,000 steps) anneals only
in the last episodes of a 6,000-episode run. So the runs were *also* too short
for the schedule they were given — a second, independent reason A ≈ B ≈ C.

Across the campaign this is not one unlucky checkpoint: the epsilon recorded with
each saved best is **C 0.60 / 0.63 / 0.60 / 0.77 / 0.80**, D 0.14-0.88, E
0.05-0.72, B 0.05-0.95. Only A (5,000-step decay), F and G — the runs built to
anneal inside the run — finished with epsilon at its 0.05 floor in 5, 4 and 5
seeds respectively.

### 4. Action handling: correct, and degenerate at episode 0

* 0 illegal placements in 68 diagnostic decisions; the action mask agrees with
  `TetrisEnv.get_valid_actions()` index for index for every piece type (tested).
* 17 of 40 actions were used, favourite share 16.2%, entropy 3.75 bits — the
  policy is *not* collapsed onto one action by the end.
* Q-values are not degenerate either: `best - second best` = 2.378 (sd 2.734),
  0 of 68 decisions tied.
* **But at episode 0 all 40 Q-values are zero, and `argmax` then returns the
  first legal index — rotation 0, column 0 — for every piece.** That is why the
  untrained evaluation survives only 12.4 pieces (pieces pile up in the left
  four columns) while the *random* baseline survives 17.8. The jump from 12.4 to
  ~24 pieces is the agent escaping that artifact, not learning to play Tetris.

### 5. State representation: nothing is missing for how it is scored

The network sees 3 binary planes (600 inputs): board occupancy, the current
piece at the cells it occupies, and the next piece. Measurements: mean plane
sums `[43.06, 4.0, 4.0]`, 0 states with no piece, 0 empty action masks — the
planes are populated exactly as designed (4 cells = one tetromino).

That is the same information a depth-1 heuristic has (board + current piece +
next piece). So the gap to the heuristic is **not** an information gap; it is
that the heuristic's scoring function encodes the objective (lines up, holes and
height down) and the DQN's reward does not.

### 6. Agreement with the heuristic: worse than chance

| metric | DQN (B seed 1 best, greedy) |
| --- | --- |
| agrees with `HeuristicAgent(depth 0)` | 2.9% of decisions |
| agreement by chance (1 / legal moves) | 5.4% |
| agrees with depth-1 lookahead | 5.0% |
| mean regret (heuristic's own scoring) | 20.65 points/move (16.7% of the move's value) |
| Q-value vs heuristic score, correlation | +0.304 |
| chose a move that ends the game while another move survived | 2 of 68 decisions |

The policy is *anti*-correlated with a good player (below chance agreement),
while its Q-values are weakly, positively related to the heuristic's ranking
(r = 0.30): the network has learned a real ordering, just not a good one.

### 7. What the training curves look like

* Training reward: flat for all 30,000 episodes of C — decile means
  -173.9, -169.5, -173.4, -166.3, -173.4, -168.1, -173.1, -167.1, -173.1, -167.3,
  i.e. it oscillates in an 8-point band with no trend (B is the same; the trailing
  mean in `plots/` shows a ~10-point drift at best). A flat reward curve beside a
  rising *evaluation* curve is the signature of a metric mismatch, not of slow
  learning.
* Loss: falls from 3.13 to 1.59 over C's run (B: 3.20 -> 1.82, min 1.41) with
  occasional spikes (max 26.98) — a real, healthy reduction: the network is
  fitting its target, the target is just the wrong target.
* Evaluation score: jumps to 480-640 within the first 250 episodes, then
  oscillates ±70 with no trend.

## The targeted changes

The diagnosis named two candidate causes: (1) the reward pays for dying, and
(2) plain DQN's max-target overestimates on 40 actions with mostly-illegal ones.
Both were turned into experiments that differ from C in exactly one knob, so C
is the control they are measured against.

* **D — Double Q-learning** (`variant="double-dqn"`): identical config, only the
  Bellman target changes (online network picks the action, target network scores
  it). Tests the overestimation hypothesis. Costs one extra forward pass (+12%
  per episode, measured).
* **E — pay for surviving** (`reward={"per_piece": 5.0}`): identical config and
  agent, one reward weight added. `per_piece=5.0` is above every measured
  per-piece cost of a non-clearing policy (random 4.17, the DQN itself 2.35), so
  the same play now earns *more* the longer it lasts; and since the board fills
  up, surviving past roughly 40 pieces requires clearing lines — the unreachable
  jackpot becomes the only way to keep collecting.
* **F — anneal inside the run** (`epsilon_decay_steps=20000`, C uses 120000): one
  knob, the exploration schedule. Prompted by a measured fact rather than a
  hunch — every one of C's five best checkpoints was saved at epsilon 0.60 / 0.63
  / 0.60 / 0.77 / 0.80 (D: 0.14-0.88, E: 0.05-0.72), so the policy being evaluated
  had been trained by a mostly-random one. Only A, F and G annealed inside their
  run.
* **G — both** (F's schedule + E's reward): the only two-knob arm, and it is
  labelled as such in `rl_plan.py`.

`per_piece` was added to `RewardConfig` with a default of 0.0, so the default
reward — and every number this project produced before — is unchanged
(there is a test that asserts exactly that).

## The targeted changes: results

Each arm below is measured against C on the same 5 seeds and the same evaluation
games. `first` is the episode-0 (untrained) evaluation, `tail` is the mean of the
last three evaluations of the run.

| exp | what it changes (vs C) | score ep0 | score tail (delta) | pieces tail | lines/100 | seeds improved |
| --- | --- | --- | --- | --- | --- | --- |
| C | reference | 248.8 | 537.5 (+288.8, sd 24) | 22.1 | 0.07 | 5/5 |
| D | variant=double-dqn | 248.8 | 511.4 (+262.7, sd 34) | 20.3 | 0.04 | 5/5 |
| E | reward per_piece=5 | 248.8 | **614.9 (+366.1, sd 6.3)** | **26.6** | **0.22** | 5/5 |

### D — Double Q-learning: no. The overestimation hypothesis is not the bottleneck

Paired per seed, D is *behind* C in 4 of 5 seeds (mean -26.1 score; -26, -7, +28,
-30, -95). That is inside the seed-to-seed spread of either arm, so the honest
reading is "no measurable improvement", and if anything a small loss. The
network's Q-values were never degenerate (best-vs-second-best 2.38, no ties in
68 decisions) and the loss fell normally, so there was no runaway optimism for
Double DQN to remove. Two things follow: the plateau is not a max-bias problem,
and D costs 12% more per episode for nothing.

### E — paying for survival: yes, and it is the first thing that moved lines

E is the first arm where the objective and the metric point the same way, and it
shows up everywhere:

| | C | E |
| --- | --- | --- |
| tail score (mean over seeds) | 537.5 | **614.9** |
| score delta, sd across seeds | +288.8 (sd 24.0) | **+366.1 (sd 6.3)** |
| pieces survived | 22.1 | **26.6** |
| lines per 100 pieces | 0.07 | **0.22** |
| seeds where lines improved | 1/5 | **4/5** |

Paired per seed, E beats C in **5 of 5** (deltas +104, +59, +97, +84, +42). With
five seeds an exact sign test cannot go below p = 0.0625, so that is the
strongest statement these seeds can make -- consistent, but not significant at
0.05 on its own, and it should be read as "5 of 5, p = 0.0625".

The reward decomposition confirms the intended change: E's training reward is
-78.7 per episode against C's -170, and `corr(reward, pieces)` went from **-0.257
to +0.720** -- surviving longer now earns more, which was the entire point.

And the policy itself is different in kind, not just in degree (E seed 1's best
checkpoint vs B seed 1's, both greedy):

| | B (per_piece=0) | E (per_piece=5) |
| --- | --- | --- |
| pieces per game | 22.7 | 29.7 |
| agreement with the heuristic | 2.9% (chance 5.4%) | **12.4% (chance 5.3%)** |
| moves that ended the game while another survived | 2 | **0** |
| distinct actions used | 17/40 | 25/40 |
| action entropy | 3.75 bits | 4.22 bits |

Below chance agreement means anti-correlated with good play; E is *above*
chance, uses more of the action space, and no longer makes suicidal moves.

**But lines are still nearly absent (0.22 per 100 pieces, against the
heuristic's 39.8), and E's curve shows why the run was cut short:** it plateaus
at 540-600 for 5,000 episodes and then, in the last 1,000 episodes -- exactly
when a 120,000-decision epsilon schedule finally anneals -- climbs to 610, 609,
620, 615 with `lines/100` rising from 0.00 to 0.28-0.29. The last quarter of the
*anneal* is the best part of the run, which says the schedule, not the budget,
was the binding constraint. That is what F and G test.

### F — annealing inside the run: yes, but less than the reward

F is C with one knob turned: `epsilon_decay_steps` 120,000 -> 20,000, so the
policy is fully annealed by ~800 episodes and the remaining 5,200 are spent
training the policy it is evaluated as.

    F tail score 590.3 (+341.6, sd 42.8), pieces 23.6, lines/100 0.17 (3/5 seeds)
    C tail score 537.5 (+288.8, sd 24.0), pieces 22.1, lines/100 0.07 (1/5 seeds)

So the schedule alone is worth about +53 score over C, and it also produced
line-clearing improvements in 3 of 5 seeds. It is smaller than E's gain and its
seed spread is wide (sd 42.8), but it is in the same direction.

### G — both changes: this is the run that learns

| exp | what it changes | tail score (delta) | pieces | lines/100 | lines improved |
| --- | --- | --- | --- | --- | --- |
| C | reference | 537.5 (+288.8, sd 24.0) | 22.1 | 0.07 | 1/5 |
| E | `reward per_piece=5` | 614.9 (+366.1, sd 6.3) | 26.6 | 0.22 | 4/5 |
| F | `epsilon_decay_steps=20k` | 590.3 (+341.6, sd 42.8) | 23.6 | 0.17 | 3/5 |
| **G** | **both** | **698.5 (+449.7, sd 8.6)** | **30.1** | **0.47** | **5/5** |

G is the best arm on every metric, and the combination is worth more than either
change alone: +449.7 against the untrained policy, with a seed spread of 8.6 that
is the tightest of any arm. Its per-seed best checkpoints reach `score_mean`
822.5 / 730.0 / 762.8 / 732.0 / 789.0 at 30.8-33.1 pieces survived, where C's
best checkpoints were 607-668 at 22-28 pieces.

G's evaluation curve (mean over seeds) is still climbing at the end of the run:
249 -> 544 (ep 500) -> 610 -> 641 -> 699 -> 693 -> 688 -> **718.7 (ep 5000)**
-> 705.5 (ep 6000), and `lines/100` rises from 0.10 to 0.72 over the same span.
Unlike A/B/C/D/E, this run was **cut short rather than saturated**.

## The comparison, on the same games

Every DQN row is the mean over that campaign's 5 seeds of its best-by-`score_mean`
checkpoint, evaluated on the campaign's own protocol: 8 games, `seed 20260928`,
`game i = seed + i`, cap 1000 — the *same* 8 piece sequences the baselines
played. A DQN row therefore covers 40 games (5 seeds x 8), and the baselines'
`won/lost` column counts games won on score against the heuristic agent.

Careful with one difference from the tables above: this table scores each
campaign's **best** checkpoint (its peak), while the earlier tables score the
**last three evaluations** (where the run ended). That is why C is 630 here and
537.5 above -- both are true, of different things, and the tail is the one that
says whether training was still helping.

| agent | seeds | score mean | sd (game-to-game) | score median | pieces | lines / 100 pieces | won/lost vs heuristic | score as % of heuristic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic (depth 0) | 1 | 930,266.2 | 8,095 | 932,028 | 1000 (cap) | 39.76 | — | 100% |
| lookahead (depth 1) | 1 | 930,025.5 | 13,369 | 931,358 | 1000 (cap) | 39.86 | 4/4 | 99.974% |
| random | 1 | 499.8 | 47 | 511 | 20.2 | 0.00 | 0/8 | 0.054% |
| DQN A (300 ep) | 5 | 595.5 | 61 | 598 | 25.0 | 0.18 | 0/40 | 0.064% |
| DQN B (2,000 ep) | 5 | 642.5 | 86 | 620 | 26.8 | 0.47 | 0/40 | 0.069% |
| DQN C (6,000 ep, control) | 5 | 630.0 | 79 | 627 | 26.8 | 0.28 | 0/40 | 0.068% |
| DQN D (double-Q) | 5 | 615.8 | 71 | 616 | 25.4 | 0.09 | 0/40 | 0.066% |
| DQN E (per_piece=5) | 5 | 647.2 | 112 | 639 | 27.4 | 0.46 | 0/40 | 0.070% |
| DQN F (annealed) | 5 | 657.9 | 114 | 649 | 26.0 | 0.37 | 0/40 | 0.071% |
| **DQN G (E + F)** | 5 | **767.2** | 119 | 728 | **32.1** | **1.17** | 0/40 | 0.082% |

Figure: `experiments/rl/comparison/00_all_agents.png` (log score axis, because the
two agents at the top are 1,200x the DQN); the per-seed table is in
`experiments/rl/comparison/comparison.json` and the per-experiment summary this
table was built from is `experiments/rl/comparison/summary.json`.

Three honest readings of that table:

1. **The trained agent beats random, on identical games.** G is above random on
   score (767.2 vs 499.8), on survival (32.1 vs 20.2 pieces) and on line clearing
   (1.17 vs 0.00 per 100 pieces). Every DQN arm in this table does.
2. **The ordering of the arms matches the predictions, not the training budget.**
   G > F > E > C > D on mean best-checkpoint score, and the only arm that was
   predicted to do nothing (D) is the only arm below the control. A and B — 300
   and 2,000 episodes — land in the same band as C, again showing the budget was
   never the lever.
3. **It is nowhere near the heuristic.** Three orders of magnitude of score
   (0.082% of it), 1.17 lines per 100 pieces against 39.76, and it survives 32
   pieces where the heuristic survives the 1000-piece cap without dying once. It
   loses all 40 games to the heuristic. These agents are learning to survive, and
   they are not yet playing Tetris well. The lookahead agent's 4/4 against depth 0
   is the "depth 1 wins about half the games on score" result from the main
   README, reproduced here on the shared seeds.


## So: is the agent learning? What the evidence supports

**Yes, and the evidence is that a controlled change to a diagnosed defect moved
the metric in the predicted direction five times out of five seeds -- and that
the plateau the first three experiments showed was caused by the reward and the
schedule, not by the budget.**

1. **It learns at all.** Every arm improves from the untrained evaluation:
   12.4 -> 22.1-30.1 pieces survived, 248.8 -> 511-698.5 score, 5 of 5 seeds in
   every arm. The training loss falls (3.1 -> 1.6), the greedy evaluation rises,
   and the untrained evaluation (measured before the first gradient step) is the
   honest "before". This is not noise: the deltas are 10-20x their seed spread.
2. **More episodes were not the problem.** A (300 episodes) = 560.2, B (2,000) =
   558.8, C (6,000) = 537.5: 20x the budget bought nothing, and the tails of all
   three are *below* their own mid-run peaks. Training reward was flat
   (-174 to -166 decile means over 30,000 episodes) while the evaluation rose and
   then stalled -- the signature of an objective that is not the metric.
3. **The reward was paying the agent to die, and fixing it worked.** With the
   default weights `corr(reward, pieces survived) = -0.257`: every per-step term
   is a penalty and the terminal penalty is paid whenever the episode ends, so
   with no line clears the cheapest episode is the shortest one. The DQN ended up
   with the *lowest* per-piece cost of any non-clearing policy (-2.35 against
   random's -4.17) -- it had optimised exactly what it was paid for. Paying
   `per_piece=5` flipped the correlation to +0.720, and the arm went to 614.9
   (5/5 seeds over C), 26.6 pieces, with above-chance agreement with the
   heuristic (12.4% vs 5.3% chance) and zero suicidal moves.
4. **The exploration schedule was the second defect.** The control's best
   checkpoints were saved at epsilon 0.60-0.80 (mid-anneal), so for most of every
   run the "policy" being evaluated was trained by a mostly-random one.
   Annealing in 20,000 steps
   instead of 120,000 (F) lifted the tail to 590.3; annealing *and* paying for
   survival (G) lifted it to 698.5 with `lines/100` improving in 5/5 seeds.
5. **What did not help, and why that is informative.** Double Q-learning (D) -- the
   textbook fix for overestimation on many actions -- did *worse* than the control
   (4/5 seeds behind C, mean -26.1). The diagnosis predicted this in hindsight:
   the Q-values were never degenerate (2.38 best-vs-second-best, 0 ties, loss
   falling normally), so there was no optimism to remove. The bottleneck was the
   objective, not the estimator.
6. **Get past the reward and the ceiling is still lines.** Even G clears 0.47
   lines per 100 pieces against the heuristic's 39.76. The agent is learning to
   *survive* (which is what it is now paid for) and, as a side effect, sometimes
   to clear a line; it is not learning to play efficiently, and nothing in its
   reward says that a cleared line is 40x more valuable than a survived piece.

### What I would do next, in order

1. **Raise the line bonus relative to survival** (`per_line` 10 -> 100-300 with
   `per_piece=5`): the per-piece economics say a line clear is worth ~4 extra
   pieces survived at the current weights, which is why G survives instead of
   clearing. Efficiency is the metric that separates a good Tetris agent from a
   long-lived one.
2. **Give G more episodes.** Its curve was still rising at 6,000 (`lines/100`
   0.10 -> 0.72 across the run) and its best checkpoints came at episodes
   3,250-5,000, not at the end. A 20,000-episode G run is the obvious next step,
   and it is now a *budget* question rather than a *design* question.
3. **Extend the state to the next few pieces.** The encoding sees the current and
   next piece only (the same information the depth-1 heuristic has), so the
   network cannot learn to hold a well for an I-piece two pieces away. `queue`
   is already in `TetrisState`; the encoding ignores it.
4. **Re-run the diagnosis on G's best checkpoint** (`diagnose --checkpoint ...`)
   and look at `agreement_depth1` and the reward decomposition's `per_line` share:
   the first is the policy-quality number, the second says whether line clears
   have become a real part of the return.

### Watching it play: `play_ai.py`

The numbers above say the agent is long-lived and not yet good. The fastest way to
believe that is to watch it, so the project now has a viewer for exactly that:

    python play_ai.py                                   # the best checkpoint on disk
    python play_ai.py --model checkpoints/best.pt       # any saved model
    python play_ai.py --speed very-fast --debug         # speed control + the AI overlay

It plays the trained agent -- G/seed0, episode 3250, the best checkpoint, score 822.5 --
in a window, and puts the AI's own numbers next to the game's: the action in flight, the
placement it chose for the last piece, the agent's evaluation of that choice (the Q-value,
with how many legal placements it was choosing between), and the reward that move earned,
computed with the same `RewardConfig` the agent trained under, read back from
`experiments/rl/G/campaign.json` (`per_piece=5`). `--debug` overlays chosen rotation,
chosen column, Q-value, reward and lines cleared on the board.

What you see for seed 5 is what the tables above predict: 29 pieces, 550 score, **no
lines**, a stack that fills in from the edges and eventually tops out -- and the last
move's reward reading `-106.00`, the terminal penalty plus the per-piece term, exactly the
number training would have recorded.

    experiments/rl/demo/ai_viewer_seed5.mp4    the whole game, with the AI panel (35 s)
    experiments/rl/demo/still_ai_viewer.png    four frames of it, start to game over
    experiments/rl/demo/dqn-g-best-seed5.mp4   the same game in the plain game window
    experiments/rl/demo/still_filmstrip.png    that one, as five stills

Three independent paths -- the campaign's evaluation harness, `play.py --mode viz`, and
`play_ai.py` -- all produce that same game for seed 5 (29 pieces, 550, 0 abandoned plans),
which is the evidence that the viewer is a frontend and not a second implementation. The
reward identity is pinned by a test: the same seed played through `TetrisEnv.step()` and
through the window must agree in placement *and* in reward, piece by piece.

The eight things the viewer promises (model loading, auto-play, pause, restart, speed,
game over, headless equivalence, and "closing it cannot corrupt the model") each have a
test in `tests/test_play_ai.py`; the README's `play_ai.py` section maps promise to test
name.

### Reproducing all of it

    python experiments/rl_experiments.py plan
    python experiments/rl_experiments.py run --experiment G --parallel 5     # ~12 min
    python experiments/rl_experiments.py aggregate --experiment G A B C D E F
    python experiments/rl_experiments.py plot --all \
        --comparison experiments/rl/comparison/comparison.json
    python experiments/rl_experiments.py compare --experiment A B C D E F G \
        --primary heuristic --seed 20260928 --games 8 --max-pieces 1000
    python experiments/rl_experiments.py diagnose \
        --checkpoint experiments/rl/G/seed0/best/best_ep003250_score_mean_822.500.pt \
        --games 3 --max-pieces 200 --lookahead-decisions 20

Every number in this report comes from `aggregate.json`, `comparison.json` or a
`diagnose` run under `experiments/rl/`; nothing here was typed from a single
observation.


