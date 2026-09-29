# Can the DQN be made *good*? — the H–N experiments

`REPORT.md` answers "is the agent learning". It is: G, the best arm from that
round, beats random decisively (767.25 against 499.75 mean score over 40 games),
and the two defects that made it look broken were found and fixed — a reward that
paid it to die, and an exploration schedule that never annealed inside the run.

It is also, by every measure that matters, not *good*. This report is the second
question: **what is the actual bottleneck, and does fixing it make the agent
play Tetris?** The brief for it was explicit — investigate the state
representation, the action representation, the reward, the network, the
exploration schedule, stability, replay, target updates, episode termination,
piece distribution and lookahead; use the heuristic as the reference; change one
thing at a time; do not implement techniques blindly; and do not dress up a poor
result as a good one.

---

## The starting point, measured properly

G is the control for everything below. On the campaign's own protocol (5 seeds,
greedy evaluation on 8 fixed games, `seed 20260928`, `game i = seed + i`, cap
1000 pieces, tail = mean of the last 3 evaluations):

| | G |
|---|---|
| score mean | 698.5 (± 8.6 across seeds) |
| score median | 694.9 |
| best single game | 1,116 |
| pieces per game | 30.1 |
| lines per 100 pieces | 0.467 |
| max lines in any single game | **2** |

That last row is the one that re-framed everything. Over **1000 games** (125
evaluations × 8 games, all five seeds), the best game any DQN in this project has
ever played cleared **two lines**, and the mean is 0.012–0.078 lines per game
depending on the arm: roughly one line every 13 to 80 games. The heuristic agent
clears ~40 lines per 100 pieces, i.e. about 12 per game — a factor of 150–1000.

So the honest description of G is not "the DQN clears fewer lines". It is **the
DQN does not clear lines at all.** It survives ~30 pieces by spreading them
flat, which is a genuinely non-trivial skill (random dies at 20.25 pieces), and
the handful of clears it does get are incidents, not plans. Everything this round
tries is aimed at that gap.

## Method

**The control is G, not C.** Round one's arms were each one change against C.
This round's are each one change against **G**, because G is the best agent the
project has and it is G's policy — not C's — that has to be fixed. Every arm
keeps G's reward (`per_piece=5`), schedule (epsilon annealed over 20,000 steps),
episode budget (6,000), seeds (0–4), evaluation schedule and evaluation protocol.
A difference between two rows is therefore the one declared change.

**One thing at a time, and pre-registered.** Each arm's target and prediction was
written into `experiments/rl_plan.py` *before* it ran, including what would count
as failure. `tests/test_rl_experiments_cli.py` pins that H changes only the reward
ratio and that I–N change only the agent while leaving the training config
identical to G, so "targeted" is checked by a test rather than asserted in prose.

| arm | the one change | targets a measured defect |
|---|---|---|
| **H** | reward ratio: `per_line` 10 → 50, `per_piece` 5 → 1 | income is 89% per-piece, 7% per-line; a clear (+10) nets ≈ what the two pieces of survival it costs (+10, minus ~2 of holes and height) do |
| **I** | 3-step returns | a clear's payoff arrives on the fifth placement; 1-step TD must propagate it back through four bootstrapped estimates |
| **J** | 5 state planes (adds column heights, holes) | the network must infer geometry from raw cells in ~86k gradient steps, while the heuristic is *handed* it by `tetris.metrics` |
| **K** | dueling head `Q = V + A − mean(A)` | mean best Q 3.63 with only 1.61 between best and second-best: ~44% of the reported value is action-independent, and the reference net stores 40 copies of it |
| **L** | prioritised replay (α 0.6, β 0.4→1.0) | ~86k decisions per run, of which line clears are ~0.5%; uniform sampling spends almost every batch on placements the network already knows |
| **N** | 6 planes: adds `queue[1]`, the piece after next | the encoding has always thrown the queue away, so the agent cannot plan for a piece it has not been dealt |

`N` is read against `J` (it is J plus one plane), not against G.

**The reference agent is provably untouched by the machinery.** I, J, K, L and N
needed seams the loop did not have: an encoder the agent owns, a buffer the agent
builds, a transition the agent can rewrite, and an end-of-episode notification.
Those are checked, not argued:

* `tests/test_rl_seams.py` asserts each seam against the exact call it replaced
  (`agent.encode(state) == encode_state(state)`, the buffer's arrays, the sampled
  batch for a fixed RNG);
* `~/.hermes/cache/scratch/seam_regression.py` replays the first **250 episodes
  of G/seed0** through the new loop and diffs every field of every episode row
  against the log that run wrote *before* the change — score, pieces, lines,
  reward, epsilon, decisions, gradient steps, mean and last loss. 250 × 12 fields
  agree exactly, so the seams are behaviour-preserving rather than
  behaviour-similar.

**Arm checkpoints load as themselves.** A dueling head is not an MLP and a
six-plane input is not a three-plane one, so a variant checkpoint cannot be
rebuilt as a plain `DQNAgent`. Each class registers itself in
`tetris/rl/registry.py` under the name it writes into the checkpoint, and
`load_agent_for_evaluation` dispatches on that name — which is what lets the
comparison table, the diagnosis and the viewer load an arm's checkpoint without
knowing which arm it was.

## Looking ahead, honestly

The brief asks about lookahead. It is worth being precise about why it is not an
arm of its own here: the action space is *whole placements*, so "one move ahead"
means scoring all legal placements and then every placement reachable from each —
up to 40 × 40 network evaluations per decision, or ~1600 forward passes where the
policy currently does one. That is affordable at play time and not at training
time, and it is a *policy* change rather than a learning change, so it would have
measured "how much does search help a mediocre Q-function" rather than "can the
agent learn to see".

Arm **N** is the cheap version of the same idea: give the network the information
a lookahead would use — the piece after next — and let the 600 extra input weights
decide what it is worth. That keeps the comparison a learning experiment, and it
costs one plane instead of 1600 forward passes.

## Results

### The table, and what each arm was predicted to do

Tail = mean of the last three evaluations (episode 5,500–6,000), mean over the
five seeds, on the same 8 fixed games. `max` is the best *single game* within those
evaluations, averaged over seeds (not the best evaluation mean, which is a different
and kinder number). Every arm's own start (episode 0 tail) is
identical — 248.8 — because at episode 0 they are all the same untrained network.

| arm | score mean | median | max | lines mean | lines/100 | pieces mean | vs G | seeds better than G |
|---|---|---|---|---|---|---|---|---|
| **C** (round-1 control) | 537.5 ± 24.0 | 541.1 | 661.9 | 0.017 | 0.072 | 22.1 ± 0.4 | −160.9 | 0/5 |
| **G** (control) | 698.5 ± 8.6 | 694.9 | 856.5 | 0.142 | 0.467 | 30.1 ± 0.6 | — | — |
| **H** reward ratio | 624.1 ± 38.1 | 634.3 | 781.6 | 0.083 | 0.314 | 25.9 ± 1.5 | −74.3 | **0/5** |
| **I** 3-step returns | 679.4 ± 29.7 | 670.8 | 839.1 | 0.117 | 0.419 | 27.5 ± 1.3 | −19.1 | **1/5** |
| **J** 5 state planes | 702.4 ± 30.3 | 720.9 | 874.3 | 0.125 | 0.418 | 29.8 ± 1.1 | +3.9 | **2/5** |
| **K** dueling head | **730.0 ± 30.2** | **725.1** | **876.4** | **0.167** | **0.526** | **31.0 ± 0.9** | **+31.5** | **4/5** |
| **L** prioritized replay | 539.4 ± **284.8** | 541.7 | 653.5 | 0.108 | 0.338 | 23.1 ± 11.8 | −159.0 | **2/5** |
| **N** queue plane | 685.3 ± **14.8** | 687.8 | 848.8 | 0.108 | 0.380 | 29.2 ± 0.8 | −13.1 | **2/5** |
| **M** dueling + 5 planes | 713.8 ± 39.7 | 716.3 | 867.5 | 0.108 | 0.346 | 30.1 ± 1.2 | +15.4 | **3/5** |

Per-seed tails, so a mean cannot hide a bad seed:

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| G | 707 | 693 | 709 | 690 | 693 |
| H | 679− | 596− | 588− | 610− | 647− |
| I | 642− | 679− | 682− | 669− | 725+ |
| J | 706− | 689− | 731+ | 658− | 728+ |
| K | **738+** | **707+** | **744+** | **768+** | 693− |
| L | **787+** | 211− | 254− | 666− | **779+** |
| N | 673− | 701+ | 669− | 700+ | 684− |
| M | **738+** | 678− | 672− | **765+** | **715+** |

Against what was predicted before the runs:

| arm | prediction | outcome |
|---|---|---|
| H | fewer pieces and score, **more** lines per 100 | 2 of 3: fewer pieces (25.9), lower score (624.1) — but **fewer lines** (0.314 against 0.467), 0/5 seeds |
| I | the delay, not the estimator, is the obstacle | no: 679.4, 1/5 seeds, and slightly *more* holes per piece |
| J | flat mean, better line-clearing where it works | flat mean (+3.9, 2/5) — but the widest lines spread in the project (± 0.41) with seeds at 0.97 and 0.71 lines/100 |
| K | the value is mostly the position, so share it | yes: +31.5, 4/5 seeds, better on every tracked metric, 5/5 improving from its own start |
| L | ~0.5% of decisions are line clears, so sample them | no, and destructively: 2 of 5 seeds fell to ~211 and stayed there |
| N | planning for a piece two moves away is what the agent lacks | no: 685.3 (−13.1), 2/5 seeds — but the tightest spread of any arm (± 14.8) |
| M | at or above K (they compose); at K's level (they overlap) | **overlap**: 713.8 (+15.4, 3/5) is better than G and worse than K's +31.5 |

### H — the objective was not the binding constraint

This was the arm with the strongest prior: the diagnosis showed 89% of the agent's
income was survival and 7% was lines, and one clear (+10) was worth about what the
two pieces of survival it costs (+10, minus ~2 of holes and height) were worth. So
make a line worth 50 and survival worth 1, and clearing becomes the only way to
earn.

The agent still learned — 248.8 → 624.1, improving in 5/5 seeds — and it still
never cleared. Re-measured on its best checkpoint:

```
H (seed0, best ep5500, 3 games, greedy)
  mean episode reward   -139.67        (G: +9.33)
  per_piece              +78.00   -18.6%   <- 26 pieces at 1.0
  per_line                 0.00   -0.0%
  hole                  -167.00    39.9%
  height                 -30.00     7.2%
  game_over             -300.00    71.6%
  episodes with a clear     0/3
  holes per piece          -2.06   (G: -1.62)
```

Two things follow. First, the reward change did what it said: income is no longer
survival-dominated, and the *episode return went negative* — the agent still takes
the −100 terminal penalty and no longer collects enough per-piece income to pay
for it. Second, and this is the finding: **it did not start clearing, and its
placements got worse, not better** — more holes per piece than G. The incentive
was changed and the behaviour followed the incentive only in the sense of giving
up income. Clearing is not something this policy is failing to *want*.

### I — credit assignment is not the obstacle either

n=3, so the clear that arrives on the fifth placement is inside the target of the
first. It did not help: 679.4 (−19.1), 1/5 seeds, holes per piece −1.96 (worse
than G's −1.62), 1 clear in 3 games where G's checkpoint managed 4. A longer
horizon also means a longer path for the reward that *is* there — the per-piece
payment and the terminal penalty — to arrive averaged over three steps instead of
one, and on this evidence that is what dominated the arm.

### J — the representation moved the ceiling, not the mean

The mean is flat (+3.9, 2/5 seeds) and by the campaign's own rule that is not an
improvement. What did change is the shape of the distribution:

```
lines per 100 pieces, tail, per seed
  G   0.693  0.419  0.527  0.284  0.414      <- all five clustered near the mean
  J   0.712  0.000  0.267  0.139  0.971      <- two seeds far above it, three below
```

J produced the two highest line-clearing seeds in the project (0.971 and 0.712
lines per 100 pieces) *and* a seed that never cleared a line at all, and the
widest lines spread of any arm (± 0.41 against G's ± 0.15). Its tail's best single
game averages 874.3, second only to K's 876.4. So the five planes do not raise the average
policy — they change which policy the run lands on, with a better ceiling.

That reading is supported by the capacity probe below: the two extra planes make
the heuristic's own ranking measurably easier for this architecture to learn.

### K — the one arm that improved the agent

Dueling is the only change of the five that raised the level rather than the
spread:

```
score mean   698.5 -> 730.0     (+31.5, 4/5 seeds)
score median 694.9 -> 725.1
lines mean   0.142 -> 0.167
lines/100    0.467 -> 0.526
pieces mean   30.1 -> 31.0
```

and per seed it is outside G's range rather than inside it: G occupies 690–709,
K occupies 693–768 with four seeds above G's best. It is also the only arm that
improved **5/5 seeds against its own start on every tracked metric** — score,
median, max, lines, lines per 100, pieces.

Why it should help is the measurement that motivated it: the diagnosis found a
mean best Q of 3.63 with only 1.61 between best and second-best, i.e. roughly 44%
of the reported value is the *position* rather than the placement. The reference
network stores that shared part 40 times and learns it 40 times; the dueling head
stores it once as `V(s)` and learns only the differences as `A(s,a)`. On a task
where the useful signal per decision is small, that is the difference between
+31.5 and noise.

### L — prioritized replay, honestly reported: it broke

This arm is a failure and it is reported as one. Four of five seeds showed the
dips you would expect while the priorities churn, but two of them never recovered:

```
L, tail score per seed:  787   211   254   666   779
L, seed1 eval curve:     249  620  585  248  211  211  211     <- permanent
L, seed2 eval curve:     249  637  613  382  525  502  216
L, seed0 eval curve:     249  620  658  601  426  690  780     <- recovered
```

211 is below random (499.8) and corresponds to dying at ~12 pieces. The pattern —
collapse onto a bad policy and stay there — is the classic prioritised-replay
failure mode: sampling with replacement proportional to |TD error|^0.6 concentrates
the batch on a small set of high-error transitions, and the highest-error
transitions in this game are the ones next to a board that has already failed, so
the network trains on, and then reproduces, a losing position. The importance
weights (β annealed 0.4 → 1.0) are supposed to correct the bias this introduces in
the *gradient*; they do not stop the *distribution* from collapsing.

Two honest caveats. This is one configuration (α = 0.6, β 0.4 → 1.0, buffer 50k);
a lower α or a stronger ε would likely behave differently, and I did not spend
further arms tuning it. And L's *best* checkpoint is fine (1,029, two lines) —
which is precisely the trap this campaign's selection rule would have fallen into
if it only ever looked at the best checkpoint per arm.

### N — seeing the piece after next does not help either

The encoding has always thrown the queue away, so arm N added one plane showing
`queue[1]`, the piece after next -- the same information a one-ply lookahead would
have before committing. It made the agent *more consistent* and no better:

```
score mean   698.5 -> 685.3    (-13.1, 2/5 seeds)
lines/100    0.467 -> 0.380
pieces mean   30.1 ->  29.2
spread       +/- 8.6 -> +/- 14.8   (the tightest arm in the round: 673 701 669 700 684)
```

That is a useful negative for the lookahead question. Two readings, and the data
supports the second: either the agent cannot use the extra piece, or -- more
likely, and consistent with the capacity probe below -- the policy it converges to
does not *need* the next piece, because spreading flat is a decision that is nearly
independent of what comes next. A lookahead would only pay off for a policy that is
already trying to build rows.

### M — the two changes overlap rather than compose

The one two-change arm, and the honest test of whether K's +31.5 and J's extra
planes add up. They do not:

```
                 score mean   lines/100   per-seed tails
G                 698.5        0.467      707  693  709  690  693
K (dueling)       730.0        0.526      738  707  744  768  693
J (5 planes)      702.4        0.418      706  689  731  658  728
M (both)          713.8        0.346      738  678  672  765  715      <- 3/5 seeds
```

M beats G (+15.4, 3/5 seeds) and loses to K (+31.5, 4/5), and its lines/100 is
*worse* than both parents. That is a real result and not noise: two arms whose
individual effects are +31.5 and +3.9 combine to +15.4, i.e. the dueling head
already captures most of what the extra planes were giving the campaign, even
though the planes clearly help a *supervised* fit (see below). What the five
planes change is which policy a run lands on — M's seed 3 hits 0.917 lines per 100
pieces, the second-best seed in the project — but the average policy of the combined
arm is not better than the head change alone.

So the recommended model from this round is **K: dueling, five fewer moving parts
than M and a better mean**.

## Why the neural network does worse than the heuristic — the capacity probe

The brief asks the right question: if the network is far behind the heuristic,
find out *why* rather than assume. The arms above rule things out. So this round
of it ends with the sharper question, which has two very different answers:

1. the network **cannot represent** a clearing policy, in which case the encoder
   is the wall; or
2. the network **could**, and Q-learning from reward at this scale does not find
   it, in which case the wall is the learning process.

`experiments/heuristic_capacity.py` separates them by removing reinforcement
learning from the question entirely. It plays three games with the depth-0
heuristic, records 3,000 decision states, labels **every** legal placement with the
heuristic's own score, and trains the *same* 600→256→256→40 architecture by
supervised regression on those labels — no exploration, no bootstrapping, no
reward, no delayed credit. Agreement is defined exactly as `diagnose` defines it
(the argmax of the network's scores equals what the heuristic played), so the
numbers are directly comparable. The held-out states are the last 20% of the
games, so this measures imitation and not memorisation.

| input planes | agreement | tie-tolerant | score correlation | loss |
|---|---|---|---|---|
| three (`encode_state`) — what G saw | **32.7%** | 35.5% | **+0.804** | 1,998 → 18 |
| five (`encode_state_with_features`) — what J saw | **45.0%** | 47.7% | **+0.860** | 1,746 → 9 |
| the DQN (G, greedy) | 10.2% | — | — | — |
| chance | 5.3% | — | — | — |

and the curves, which are still climbing at 20,000 steps:

```
three planes  0.07  0.17  0.21  0.26  0.29  0.29  0.29  0.32  0.32  0.33
five planes   0.06  0.20  0.31  0.37  0.41  0.44  0.43  0.45  0.45
```

Three conclusions, in order of how much they are worth:

1. **The architecture is not the wall.** Handed the heuristic's own labels, the
   same network on the same planes reaches 32.7% agreement and +0.80 correlation
   with the teacher's scores — from 2,400 training states and 20,000 gradient
   steps. Its *reward-free* imitation is **3.2× the DQN's reward-trained agreement**
   and 6× chance.
2. **The reward signal is what is missing the ranking.** Correlation +0.80 with
   the teacher's score ordering, but only 32.7% top-1 agreement, means the network
   gets the *value* roughly right and then loses the decision among near-ties. That
   is exactly the signature of a Q-function trained on a small, noisy, sparse
   reward: the ordering is learned, the top of the ordering is not.
3. **The five planes are the encoder that makes it learnable** — +12.3 points of
   agreement and +0.056 of correlation for two extra planes, the same planes J
   added. It is the clearest evidence in this round that the representation change
   is *in the right direction*, even though J's campaign mean is flat.

Everything in that list points the same way: **the limiter is the learning
signal, not the network.** The next thing to try is not another architecture. It is
to put the heuristic's knowledge into the learning process itself — pretrain on
the heuristic's decisions (optionally with a margin loss against its scores, or
DQfD-style with the demonstrations kept in the buffer for the whole run) and only
then fine-tune with Q-learning. The probe says plainly that this is a fit this
architecture can make.

## What actually improved, and why

One change out of six moved the agent, and it moved it by about 4.5%:

```
score mean   698.5 -> 730.0      +31.5   (4 of 5 seeds; the other is level)
score median 694.9 -> 725.1
score max    856.5 -> 876.4
lines mean   0.142 -> 0.167
lines/100    0.467 -> 0.526
pieces mean   30.1 ->  31.0
```

**Dueling worked, and the reason it worked is the measurement that picked it.** The
diagnosis found a mean best Q of 3.63 with only 1.61 between best and second-best.
Roughly 44% of the value the network predicts is the same for every placement in a
state -- it is a property of the *position*, not of the action. The reference
network represents that shared part with 40 separate output weights and has to
learn it 40 times from a reward signal that arrives once per decision. The dueling
head factors it into one `V(s)` and learns only `A(s,a)` for the differences. That
is a small, cheap change that removes redundant work exactly where this task has
little signal to spare, and it is the only arm whose gain survives the per-seed
check (4/5 seeds, and four of its five seeds are above G's *best*).

**Everything else was neutral or worse, and each ruled something out:**

- the objective (H): making a line worth 50 pieces of survival and survival worth 1
  did not produce a single extra clear *per piece* — 0.314 against 0.467 — and made
  placements worse (2.06 holes per piece against 1.62). The policy does not fail
  because it does not want to clear;
- credit assignment (I): three-step returns did not help either, and the longer
  horizon mostly diluted the dense part of the reward;
- the encoding alone (J): flat mean, better ceiling and worse floor — the planes
  change which policy the run finds, not how good the average one is;
- replay (L): prioritised sampling destabilised 2 of 5 seeds into a permanent
  policy collapse (tail 211, below random), which is worth knowing before it is used
  on a task where the highest-error transitions are the already-lost ones;
- lookahead information (N): the piece after next, handed over for free, made the
  agent slightly worse and much more consistent;
- and the combination (M): the changes overlap rather than compose.

**And the capacity probe explains the size of the whole plateau.** Six designs, one
change each, and every arm lands inside 620–730 score and 0.31–0.53 lines per 100
pieces, while the heuristic sits at 39.76. The probe says why that ceiling is where
it is: supervised on the heuristic's own scores, the same architecture reaches 32.7%
(three planes) / 45.0% (five planes) agreement and +0.80 / +0.86 score correlation
from 2,400 states, against the DQN's 10.2% agreement from ~86,000 reward-driven
decisions per run. The network can learn the heuristic's ordering; the reward signal
this setup supplies is not teaching it. So the plateau is a property of the learning
signal and its budget, not of the architecture, the encoder, the objective, the
estimator or the buffer — all of which have now been tested one at a time.

## What I would do next, in order

1. **Imitate the heuristic, then fine-tune.** The probe says this is a fit the
   architecture can make: supervised agreement 45.0% on five planes from 2,400
   states. Pretrain the five-plane network on the heuristic's placements (a
   cross-entropy or margin loss against its score vector), then run the *same*
   round-two protocol from that warm start. This is the one lever with measured
   headroom behind it, and it is the cheapest way to find out whether the plateau is
   the signal or the search.
2. **Give the run a real budget.** ~86k gradient decisions per run is small for
   DQN; G was still improving at 6,000 episodes in round one. With a warm start,
   20,000 episodes is now worth the wall clock in a way it was not from scratch.
3. **Keep dueling, keep the five planes.** K is the best agent here; J is the arm
   whose ceiling (0.97 lines/100) and supervised learnability (+12.3 points) say it
   is the right encoder. They are also the two that survive being combined (M, 3/5
   seeds, above G).
4. **Then, and only then, revisit search.** A one-ply lookahead costs ~1,600 forward
   passes per decision and can be applied at play time to any checkpoint. It is only
   worth measuring against a policy that is trying to build rows; N shows that
   handing the information over is not the same as using it.
5. **Report lines per 100 pieces, never score alone.** `score_mean` is dominated by
   drop points and therefore by survival, which is why H can look respectable at 624
   while never clearing a line in three games, and why L's broken arm still has a
   1,029 best checkpoint. The skill metric is lines per 100 pieces, and the second
   metric to watch is the *worst* seed rather than the best.

## Final comparison

Generated by `python experiments/final_comparison.py` on the campaign's own
protocol (seed 20260928, 8 games, `game i = seed + i`, cap 1000, greedy), so every
row is measured on identical piece sequences. **Every RL row is that arm's best
checkpoint** by its own campaign record -- the same selection rule for all of them,
including the round-1 arms.

| agent | avg score | median | max score | avg lines | max lines | game length (pieces) |
|---|---|---|---|---|---|---|
| Random | 499.8 | 511 | 566 | 0.00 | 0 | 20.2 |
| Heuristic (depth 0) | 930,266.2 | 932,028 | 938,972 | 397.62 | 399 | 1,000 (the cap) |
| Lookahead heuristic (depth 1) | 930,025.5 | 931,358 | 952,524 | 398.62 | 399 | 1,000 (the cap) |
| **Initial RL** (C) | 668.2 | 668 | 796 | 0.12 | 1 | 28.0 |
| *round-1 RL* (G, for reference) | 822.5 | 720 | 1,072 | 0.75 | 2 | 31.6 |
| **Improved RL** (K, dueling) | 825.5 | **809** | 996 | 0.38 | 2 | 35.4 |
| *combination* (M, dueling + planes) | **849.8** | 743 | **1,256** | 0.50 | 2 | 33.6 |

Lines per 100 pieces, the metric that does not reward survival: random 0.00,
C 0.45, G 2.37, K 1.06, M 1.49, heuristic 39.76, lookahead 39.86.

Four things this table says, including the inconvenient ones:

1. **Against the reference agents the DQN is still nowhere.** Improved RL scores
   825.5 where the heuristic scores 930,266, and clears 0.38 lines in eight games
   where the heuristic clears 397.6; it survives 35.4 pieces against the heuristic's
   1,000-piece cap. Nothing in this round changes that. The agent got *better at
   surviving*, not good at Tetris.
2. **K is steadier than G, which is most of what "+31.5" buys in practice.** On
   these 8 games their means are level (825.5 against 822.5), but K's median is 809
   against 720 and its spread is half as wide (sd 106.5 against 187.9). G's single
   best game is higher (1,072 against 996); K is the one you would rather watch,
   because it does not occasionally collapse.
3. **Two measurements disagree about K and M, and both are reported.** By the
   campaign protocol K is the best arm (+31.5 over 960 games, 4/5 seeds). On this
   8-game table M's *best checkpoint* out-scores everything (849.8, with a
   1,256-point game -- the best single game any DQN here has played). Eight games is
   ~280 pieces, so one lucky clear moves `lines/100` by ~0.35 and one long game moves
   the mean by hundreds of points. K is called the improved agent here because it is
   the arm-level result on the largest sample, not the checkpoint-level result on the
   smallest.
4. **The lookahead heuristic is not meaningfully better than the heuristic.** Depth
   1 scores 930,025 against 930,266 (-240.8, winning 4/4 paired games by tiny
   margins, p = 1.000) and clears 39.86 lines per 100 pieces against 39.76. On this
   action space -- whole placements scored by a feature sum -- one ply of search buys
   almost nothing, which is worth knowing *before* paying ~1,600 forward passes per
   decision to give a network the same idea.

### The one-line answer

Initial RL (C) survives 22 pieces and clears 0.07 lines per 100. Improved RL (K)
survives 31, clears 0.53, scores 4.5% higher with four of five seeds improved, and is
the only change of six that did anything. Both are still ~0.1% of the heuristic, because
the wall is not the network -- it is that this learning signal, at this budget, teaches
survival and not clearing. The next experiment that has evidence behind it is imitation
pretraining on the heuristic, not another architecture.

