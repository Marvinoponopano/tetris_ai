"""Tests for the AI controller that drives the window (`play.py --mode viz`).

The controller is the translation layer between the two languages in this
project: a policy answers with a whole placement, the engine takes one action per
frame. Everything here runs without a display and without pygame -- the module is
pure logic, which is what makes the demo testable at all.
"""

import os
import subprocess
import sys

import pytest

from tetris.actions import Action
from tetris.env import TetrisEnv, placements_for
from tetris.game import Game
from tetris.pieces import PieceType
from tetris.policy_controller import (
    MOVE_LIMIT,
    PolicyController,
    STUCK_LIMIT,
    best_checkpoint,
    best_records,
    build_agent,
)


class ScriptedPolicy:
    """An agent that always asks for one placement (the plan under test)."""

    name = "scripted"

    def __init__(self, rotation: int, column: int):
        self.rotation = rotation
        self.column = column
        self.calls = 0
        self.saw_state = None
        self.saw_valid = None

    def choose_action(self, state, valid_actions):
        self.calls += 1
        self.saw_state = state
        self.saw_valid = list(valid_actions)
        requested = (self.rotation, self.column)
        assert requested in valid_actions, (
            f"{requested} is not legal for this piece: {valid_actions}"
        )
        return requested


def build(rotation: int, column: int, seed: int = 3, **kwargs):
    """A controller, its policy, and the game it will play -- all deterministic."""
    policy = ScriptedPolicy(rotation, column)
    controller = PolicyController(policy, **kwargs)
    game = Game(__import__("random").Random(seed))
    return controller, policy, game


def play_until_locked(controller, game, limit: int = MOVE_LIMIT * 3):
    """Run `choose_action` + the engine until one piece has been locked."""
    actions = []
    for _ in range(limit):
        action = controller.choose_action(game)
        if action is None:
            continue
        actions.append(action)
        game.step(action)
        if action is Action.DROP:
            return actions
    raise AssertionError("the piece never landed")


def test_a_placement_becomes_rotate_then_slide_then_drop():
    """The whole point: the plan is carried out as engine moves, in order."""
    # A T piece in its second distinct orientation, landing left of centre.
    controller, policy, game = build(rotation=1, column=2)
    actions = play_until_locked(controller, game)

    assert actions[-1] is Action.DROP
    assert all(action in (Action.ROTATE_CW, Action.LEFT, Action.RIGHT, Action.DROP)
               for action in actions)
    assert Action.ROTATE_CW in actions, "the piece had to rotate to the plan"

    assert policy.calls == 1, "one decision per piece, not one per frame"
    assert controller.decisions == 1
    assert controller.pieces == 1


def test_the_piece_lands_at_the_requested_column():
    """The engine's own idea of where the piece went, not the controller's."""
    controller, _policy, game = build(rotation=0, column=6)
    play_until_locked(controller, game)

    occupied = [(row, col) for row, line in enumerate(game.board.as_list())
                for col, cell in enumerate(line) if cell]
    assert occupied, "the piece must have been locked onto the board"
    assert min(col for _, col in occupied) == 6


def test_a_left_placement_uses_the_left_action_and_a_right_one_the_right():
    """The slide direction is decided per frame from where the piece actually is."""
    probe = Game(__import__("random").Random(3))
    legal = [column for rotation, column in placements_for(probe.current.type)
             if rotation == 0]
    leftmost, rightmost = min(legal), max(legal)

    left_controller, _p, left_game = build(rotation=0, column=leftmost)
    left_actions = play_until_locked(left_controller, left_game)

    right_controller, _p, right_game = build(rotation=0, column=rightmost)
    right_actions = play_until_locked(right_controller, right_game)

    assert Action.LEFT in left_actions and Action.RIGHT not in left_actions
    assert Action.RIGHT in right_actions and Action.LEFT not in right_actions


def test_the_policy_is_asked_once_per_piece():
    controller, policy, game = build(rotation=0, column=3)
    for _ in range(3):
        play_until_locked(controller, game)

    assert policy.calls == 3
    assert controller.pieces == 3
    assert controller.decisions == 3


def test_the_policy_sees_the_envs_own_state_and_legal_placements():
    """No second simulation: the state comes from the env the harness would use."""
    controller, policy, game = build(rotation=0, column=4)
    controller.choose_action(game)

    assert policy.saw_valid == list(placements_for(game.current.type))
    assert policy.saw_state.current_column == min(
        col for _, col in game.current.absolute_cells()
    )
    assert policy.saw_state.board.shape == (20, 10)


def test_thinking_is_paced_by_elapsed_time():
    """`think_ms` gates actions, so one move per frame is opt-in, not forced."""
    controller, _policy, game = build(rotation=0, column=2, seed=3, think_ms=100.0)

    controller.feed([], 30.0)
    assert controller.choose_action(game) is None, "not enough time has passed"
    controller.feed([], 30.0)
    assert controller.choose_action(game) is None
    controller.feed([], 50.0)
    assert controller.choose_action(game) is not None, "100 ms have passed"

    # And with no think time every frame acts.
    instant = PolicyController(ScriptedPolicy(0, 0), think_ms=0.0)
    assert instant.choose_action(game) is not None


def test_with_no_think_time_every_call_acts():
    controller, _policy, game = build(rotation=0, column=0, seed=5)
    first = controller.choose_action(game)
    game.step(first)
    second = controller.choose_action(game)
    assert first is not None and second is not None


def test_a_blocked_move_is_given_up_on_instead_of_looped_forever():
    """A blocked slide must degrade into a drop, not an infinite retry.

    The engine ignores a move that would collide, so nothing changes from one
    frame to the next -- which is exactly the case a demo would hang on.
    """
    controller, policy, game = build(rotation=0, column=0, seed=1)
    piece = game.current
    cells = set(piece.absolute_cells())
    # Wall the piece in on every row it occupies: no horizontal move can land,
    # while the space below stays free so the drop still works.
    for row, col in [(row, col) for row in {r for r, _ in cells}
                     for col in range(game.board.width)]:
        if (row, col) not in cells:
            game.board.grid[row][col] = 1

    # Ask for a different column, so a slide is actually attempted and blocked.
    legal = [column for rotation, column in placements_for(piece.type) if rotation == 0]
    here = min(col for _, col in cells)
    policy.column = next(column for column in legal if column != here)

    actions = play_until_locked(controller, game, limit=MOVE_LIMIT + 4)

    assert actions[-1] is Action.DROP
    assert controller.abandoned >= 1, "the blocked plan must be reported"
    assert len(actions) <= STUCK_LIMIT + 2, f"took {len(actions)} actions to give up"


def test_an_out_of_moves_plan_is_abandoned_by_the_move_limit():
    """The hard budget: even a pathological plan cannot exceed MOVE_LIMIT moves."""
    controller, _policy, game = build(rotation=0, column=0, seed=2)
    controller.choose_action(game)          # plan the piece first ...
    controller._moves = MOVE_LIMIT          # ... then spend the whole budget
    action = controller.choose_action(game)
    assert action is Action.DROP
    assert controller.abandoned == 1


def test_a_finished_game_asks_for_nothing():
    controller, _policy, game = build(rotation=0, column=0)
    game.game_over = True
    assert controller.choose_action(game) is None
    assert controller.decisions == 0


def test_no_legal_placement_drops_the_piece_rather_than_crashing():
    """`get_valid_actions` returns [] for a piece with nowhere to go."""
    controller, _policy, game = build(rotation=0, column=0)
    game.current = None                      # what a finished game looks like
    assert controller.choose_action(game) is None


def test_restart_forgets_the_piece_in_flight():
    controller, policy, game = build(rotation=1, column=1)
    controller.choose_action(game)
    assert controller._plan is not None

    controller.clear()
    assert controller._plan is None
    assert controller._elapsed_ms == 0.0

    controller.choose_action(game)
    assert policy.calls == 2, "a restart must not reuse the old plan"


def test_the_counters_describe_a_whole_run():
    controller, _policy, game = build(rotation=0, column=2)
    for _ in range(2):
        play_until_locked(controller, game)

    assert controller.pieces == 2
    assert controller.decisions == 2
    assert controller.actions >= 2
    assert controller.name.startswith("PolicyController(")
    assert "pieces 2" in controller.describe()


def test_the_controller_reads_the_game_it_is_handed():
    """Restart builds a new Game; the controller must follow it, not a stale one."""
    controller, policy, game = build(rotation=0, column=0, seed=1)
    controller.choose_action(game)
    fresh = Game(__import__("random").Random(9))
    controller.clear()
    controller.choose_action(fresh)

    assert controller.env.game is fresh
    assert policy.calls == 2


def test_they_keys_a_spectator_needs_still_work():
    """pause/restart/ghost/quit stay available while the AI plays."""
    pygame = pytest.importorskip("pygame")
    controller, _policy, game = build(rotation=0, column=0)
    events = [
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r),
    ]
    assert controller.feed(events, 16.7) == ["pause", "ghost", "restart"]
    assert controller.feed(
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)], 16.7
    ) == ["quit"]
    assert controller.feed([], 16.7) == []


def test_the_module_can_be_picked_up_without_pygame_or_torch():
    """The controller must not drag a display or a training stack into an import."""
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, tetris.policy_controller; "
         "print('pygame' in sys.modules, 'torch' in sys.modules)"],
        cwd=project, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


# ------------------------------------------------------------------ the agents

def test_every_agent_name_builds_something_that_can_play():
    for name in ("random", "heuristic"):
        agent = build_agent(name)
        env = TetrisEnv(seed=4)
        state = env.get_state()
        placement = agent.choose_action(state, env.get_valid_actions())
        assert placement in env.get_valid_actions()


def test_an_unknown_agent_name_is_rejected():
    with pytest.raises(ValueError, match="unknown agent"):
        build_agent("oracle")


def test_the_heuristic_agent_can_look_ahead():
    agent = build_agent("heuristic", lookahead=1)
    assert agent.lookahead_depth == 1


# --------------------------------------------------------------- checkpoints

def write_best(root, campaign, seed, name, score):
    """A `best.json` the way the campaign writes it (plus its checkpoint file)."""
    directory = root / campaign / f"seed{seed}" / "best"
    directory.mkdir(parents=True)
    (directory / name).write_bytes(b"not a real checkpoint")
    (directory / "best.json").write_text(
        '{"metric": "score_mean", "higher_is_better": true, "best": {'
        f'"episode": 100, "score_mean": {score}, "checkpoint": "{name}", '
        f'"path": "/somewhere/else/{name}"}}, "improvements": []}}'
    )
    return directory / name


def test_the_best_checkpoint_is_the_highest_score_on_disk(tmp_path):
    write_best(tmp_path, "A", 0, "best_ep000100_score_mean_300.000.pt", 300.0)
    winner = write_best(tmp_path, "B", 3, "best_ep000900_score_mean_822.500.pt", 822.5)
    write_best(tmp_path, "B", 1, "best_ep000500_score_mean_700.000.pt", 700.0)

    found = best_checkpoint(str(tmp_path))

    assert found.path == str(winner)
    assert found.score_mean == 822.5
    assert found.episode == 100
    assert found.source == "B/seed3"
    assert "822.5" in str(found)


def test_a_checkpoint_is_resolved_next_to_its_own_record(tmp_path):
    """`path` in best.json is absolute and goes stale; the file name does not."""
    winner = write_best(tmp_path, "G", 0, "best_ep003250_score_mean_822.500.pt", 822.5)
    found = best_checkpoint(str(tmp_path))
    assert found.path == str(winner)
    assert os.path.exists(found.path)


def test_a_first_run_checkpoint_is_kept(tmp_path):
    """A campaign that has only evaluated episode 0 is still a valid answer."""
    only = write_best(tmp_path, "A", 0, "best_ep000000_score_mean_248.750.pt", 248.75)
    assert best_checkpoint(str(tmp_path)).path == str(only)


def test_without_any_checkpoint_the_error_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="rl_experiments"):
        best_checkpoint(str(tmp_path))


def test_a_broken_record_does_not_break_the_lookup(tmp_path):
    write_best(tmp_path, "A", 0, "best_ep000100_score_mean_300.000.pt", 300.0)
    broken = tmp_path / "B" / "seed0" / "best"
    broken.mkdir(parents=True)
    (broken / "best.json").write_text("{ this is not json")
    assert best_checkpoint(str(tmp_path)).score_mean == 300.0


def test_a_record_whose_weights_are_not_in_the_checkout_is_not_a_model(tmp_path):
    """The repository ignores 880 of the 882 training checkpoints (3.4 GB, all
    regenerable), so a fresh clone has records describing files that are not
    there. The lookup must skip them instead of answering with a model that
    cannot be loaded -- and `best_checkpoint` must fall through to the weights
    that *are* present, even when a missing one scored higher.
    """
    present = write_best(tmp_path, "K", 2, "best_ep004000_score_mean_825.500.pt", 825.5)
    gone = write_best(tmp_path, "M", 1, "best_ep004500_score_mean_849.750.pt", 849.75)
    os.remove(gone)                                   # not shipped in the repo

    found = best_records(str(tmp_path))

    assert [record.path for record in found] == [str(present)]
    assert best_checkpoint(str(tmp_path)).path == str(present)
    listed = best_records(str(tmp_path), include_missing=True)
    assert [record.path for record in listed] == [str(gone), str(present)]


def test_the_real_project_has_a_trained_agent_to_play():
    """The demo's default is really discoverable in this checkout."""
    if not os.path.isdir(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "rl"
    )):
        pytest.skip("no experiments directory")
    found = best_checkpoint()
    assert os.path.exists(found.path), found.path
    assert found.score_mean > 0


def test_piece_type_coverage_is_not_a_typo():
    """Guard the enum import used above against a rename."""
    assert PieceType.I in PieceType and STUCK_LIMIT >= 1
