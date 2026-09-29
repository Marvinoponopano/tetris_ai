# `checkpoints/`

`best.pt` is a **pointer**, not a copy: a relative symlink to the best model the
campaigns left behind, so the example command from the README works as written:

```bash
python play_ai.py --model checkpoints/best.pt
```

It currently points at `experiments/rl/K/seed2/best/best_ep004000_score_mean_825.500.pt`
— campaign K (dueling head, the one round-two change that improved the agent:
+31.5 mean score, 4/5 seeds), seed 2, episode 4000, mean evaluation score 825.5.
Before that it pointed at G, the round-one winner; `python play_ai.py` therefore
shows the best agent the project has, not the first one.

Why a symlink rather than a copy:

- **Nothing is duplicated.** The campaigns keep every seed's best model and the
  `best.json` that records what it scored; this directory holds no second copy to
  fall out of date.
- **The model's identity survives.** The viewer works out which reward weights a
  model was trained under by asking which campaign it belongs to. Resolving this
  link lands on the real file, so `play_ai.py` still reports
  `per_piece=5` from `experiments/rl/G/campaign.json` instead of falling back to
  the defaults (whose `per_piece=0` is a *different* quantity — it is what made
  the A–G campaign necessary in the first place).

You do not need this file at all: `python play_ai.py` with no `--model` looks up
the best checkpoint itself. `--model` is for pinning a specific one, e.g. an
earlier campaign's:

```bash
ls experiments/rl/*/seed*/best/*.pt              # everything that exists
python play_ai.py --model experiments/rl/E/seed1/best/*.pt
```

If you re-train and a different checkpoint wins, re-point the link — but **pick by
arm, not by "the highest `score_mean` on disk"**. The global best is currently
`experiments/rl/L/seed4/...score_mean_847.000.pt`, and L is the arm that collapsed
on 2 of its 5 seeds: a peak from an unstable run is not a model to point at. Point
it at the best checkpoint of an arm you trust instead (K and M are the two round-two
arms that beat G):

```bash
# the best checkpoint of a named arm (here: K, the improved agent)
python - <<'PY'
import json, pathlib
records = [json.loads(p.read_text())["best"]
           for p in pathlib.Path("experiments/rl/K").glob("seed*/best/best.json")]
best = max(records, key=lambda record: record["score_mean"])
print(f'ln -sfn "../{best["path"]}" checkpoints/best.pt')
PY
```

**Score is not the whole story when choosing.** `score_mean` is dominated by drop
points and therefore by how long the agent survived; the metric that separates a
good agent from a long-lived one is `lines_per_100_pieces` in each seed's
`best.json` and `metrics.jsonl`.
