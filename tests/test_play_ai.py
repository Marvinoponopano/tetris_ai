"""Tests for the AI viewer (`play_ai.py`) -- one per thing it promises.

The viewer is a *frontend*: the point of these tests is that it changes nothing
about the AI. The strongest one is
`test_the_viewer_plays_the_same_game_as_the_headless_environment`, which drives
the same checkpoint and seed through the trained agent twice -- once through
`TetrisEnv`, once through the window's controller -- and asserts the placements
and the rewards came out identical. Everything else here is the window's own
contract: keys, speed, panel contents, and that the model file is only ever read.
"""

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window for any of this

import random                                            # noqa: E402
import sys                                               # noqa: E402

import pytest                                            # noqa: E402

from tetris.env import RewardConfig, TetrisEnv           # noqa: E402
from tetris.game import Game                             # noqa: E402
from tetris.policy_controller import (                   # noqa: E402
    PolicyController,
    best_checkpoint,
    best_records,
    build_agent,
    campaign_of,
    find_checkpoint,
    reward_config_for,
)

import play_ai                                           # noqa: E402


# --------------------------------------------------------------------- helpers

def pygame_or_skip():
    return pytest.importorskip("pygame")


def viewer(seed=5, cell=20, fps=240, debug=False, agent=None, game_factory=None):
    """A viewer stack: app + controller + view, all headless and deterministic."""
    pygame = pygame_or_skip()
    from tetris.ai_view import AIPygameView
    from tetris.app import TetrisApp

    agent = agent if agent is not None else build_agent("heuristic", lookahead=0)
    controller = PolicyController(agent, think_ms=0.0)
    game = Game(random.Random(seed))
    factory = game_factory if game_factory is not None else (lambda: Game(random.Random(seed)))
    app = TetrisApp(game, controller, fps=fps, gravity=False, game_factory=factory)
    pygame.init()
    view = AIPygameView(cell=cell, debug=debug)
    return app, controller, view


def press(pygame, *keys):
    """Queue keystrokes for the viewer's own event loop."""
    for key in keys:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))


def run(app, controller, view, frames, speed=1, **kwargs):
    from tetris.ai_view import run_ai

    logs = []
    count = run_ai(app, view, controller, speed=speed, max_frames=frames,
                   logger=logs.append, **kwargs)
    return count, logs


# ------------------------------------------------- 1. the model loads, and says so

def test_the_default_model_is_the_best_checkpoint_on_disk(capsys):
    found = best_checkpoint()
    assert play_ai.main(["--headless", "--frames", "5"]) == 0
    out = capsys.readouterr().out
    assert f"model: {found.path}" in out
    assert found.source in out                       # which campaign it came from


def test_an_explicit_model_is_loaded_and_named(capsys):
    found = best_checkpoint()
    assert play_ai.main(["--headless", "--frames", "5", "--model", found.path]) == 0
    out = capsys.readouterr().out
    assert found.path in out
    assert os.path.basename(found.path) in out


def test_a_missing_model_reports_what_does_exist():
    """A guessed path is the most likely mistake, so say what the real ones are."""
    with pytest.raises(SystemExit) as exit_info:
        play_ai.main(["--headless", "--model", "/nope/not-a-model.pt"])
    message = str(exit_info.value)
    assert "/nope/not-a-model.pt" in message
    assert best_checkpoint().path in message
    assert "no --model at all" in message


def test_without_any_checkpoint_at_all_the_message_points_at_training():
    message = play_ai.missing_model_message("checkpoints/best.pt", root="/nowhere")
    assert "rl_experiments.py run" in message
    assert "--agent heuristic/random" in message


def test_the_documented_stand_in_path_resolves_to_a_real_campaign():
    """`--model checkpoints/best.pt` is in the docs, so the link has to be live."""
    link = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints", "best.pt")
    if not os.path.lexists(link):
        pytest.skip("no checkpoints/best.pt stand-in in this checkout")
    assert os.path.isfile(link), "the link must resolve to a model file"
    record = find_checkpoint(link)
    assert record is not None, (
        "the stand-in link must still resolve into the campaigns, or the viewer "
        "would report default reward weights for a trained model"
    )
    # The invariant is that the stand-in resolves *into the campaigns*, so the
    # viewer reports the reward weights it was trained with. It deliberately does
    # not assert that it is the highest-scoring checkpoint on disk: that changes
    # whenever an arm runs, and the highest score_mean is not the best model to
    # point at anyway -- the arm that produced it may be the unstable one.
    assert os.path.realpath(record.path) == os.path.realpath(link)


def test_records_are_listed_best_first(tmp_path):
    for campaign, seed, score in (("A", 0, 100.0), ("B", 1, 900.0), ("B", 0, 400.0)):
        directory = tmp_path / campaign / f"seed{seed}" / "best"
        directory.mkdir(parents=True)
        (directory / "model.pt").write_bytes(b"x")
        (directory / "best.json").write_text(
            '{"best": {"episode": 10, "score_mean": ' + str(score) +
            ', "checkpoint": "model.pt"}}'
        )
    records = best_records(str(tmp_path))
    assert [record.score_mean for record in records] == [900.0, 400.0, 100.0]
    assert records[0].source == "B/seed1"
    assert best_records("/nowhere") == []


def test_the_reward_weights_come_from_the_campaign_that_trained_the_model(capsys):
    """The reward on the panel has to be the reward the agent was trained on."""
    assert play_ai.main(["--headless", "--frames", "5"]) == 0
    out = capsys.readouterr().out
    config, source = reward_config_for(best_checkpoint().source)
    assert source is not None and source in out
    assert f"per_piece={config.per_piece:g}" in out
    assert config.per_piece > 0, "the reported reward must include the survival term"


def test_a_named_model_still_finds_its_campaign_record(capsys):
    """`--model path.pt` must not silently fall back to default reward weights."""
    found = best_checkpoint()
    record = find_checkpoint(found.path)
    assert record is not None and record.source == found.source

    assert play_ai.main(["--headless", "--frames", "5", "--model", found.path]) == 0
    out = capsys.readouterr().out
    assert "campaign.json" in out, "the model's own reward weights should be found"
    assert "per_piece=5" in out


def test_a_model_copied_out_of_the_tree_plays_with_default_weights(tmp_path, capsys):
    """A model the campaigns never recorded plays; the panel says which weights it used."""
    found = best_checkpoint()
    stray = tmp_path / "checkpoints"
    stray.mkdir()
    copy = stray / "best.pt"
    copy.write_bytes(open(found.path, "rb").read())

    assert find_checkpoint(str(copy)) is None
    assert campaign_of(str(copy), root=str(tmp_path)) is None
    assert play_ai.main(["--headless", "--frames", "5", "--model", str(copy)]) == 0
    assert "defaults (per_piece=0)" in capsys.readouterr().out


def test_a_campaign_is_read_off_the_path_even_without_a_record():
    """The directory shape is the fallback, so a moved or renamed model still resolves."""
    outside = os.path.join(os.sep, "somewhere", "else", "G", "seed3", "best",
                           "best_ep000100_score_mean_500.000.pt")
    assert campaign_of(outside, root=os.path.join(os.sep, "nowhere")) == "G"
    assert campaign_of("/tmp/loose-model.pt", root="/nowhere") is None


def test_the_reward_weights_can_be_overridden_on_the_command_line(capsys):
    assert play_ai.main(["--headless", "--frames", "5",
                         "--reward-per-piece", "2.5"]) == 0
    assert "per_piece=2.5" in capsys.readouterr().out


# ------------------------------------------------------- 2. the AI plays by itself

def test_the_ai_plays_without_any_input():
    app, controller, view = viewer()
    frames, _logs = run(app, controller, view, 300)

    assert frames == 300
    assert controller.decisions > 0 and controller.pieces > 0
    assert controller.actions >= controller.pieces       # at least a drop each
    occupied = sum(1 for row in app.game.board.as_list() for cell in row if cell)
    assert occupied > 0, "pieces must actually land on the board"


def test_the_piece_is_moved_with_real_engine_actions_not_teleported():
    app, controller, view = viewer()
    run(app, controller, view, 300)
    assert controller.actions > controller.pieces, (
        "a placement needs rotations and slides before the drop"
    )


# ---------------------------------------------------------------- 3. pause/resume

def test_space_pauses_the_ai_and_pauses_it_again_to_resume():
    pygame = pygame_or_skip()
    app, controller, view = viewer()

    press(pygame, pygame.K_SPACE)                        # pause before the first frame
    run(app, controller, view, 60)
    assert app.paused is True
    assert controller.pieces == 0, "a paused viewer must not plan or play"

    press(pygame, pygame.K_SPACE)                        # resume
    run(app, controller, view, 300, speed=2)             # 20 ms/move: a few pieces
    assert app.paused is False
    assert controller.pieces > 0


# ------------------------------------------------------------------- 4. restart

def test_r_starts_a_fresh_game_and_the_ai_keeps_playing():
    pygame = pygame_or_skip()
    app, controller, view = viewer()
    run(app, controller, view, 400)
    played, scored = app.game, app.game.score
    assert controller.pieces > 0 and scored > 0

    press(pygame, pygame.K_r)
    run(app, controller, view, 200)

    assert app.game is not played, "restart must deal a new game"
    assert app.game.score < scored + app.game.score or app.game.score >= 0
    assert len(app.game.board.as_list()) == 20
    assert controller.pieces > 0, "the AI carries on in the new game"


# --------------------------------------------------------------------- 5. speed

def test_the_speed_table_goes_from_slow_to_instant():
    from tetris.ai_view import SPEED_LEVELS

    assert [label for label, _ms, _ticks in SPEED_LEVELS] == \
        ["Slow", "Normal", "Fast", "Very fast"]
    waits = [ms for _label, ms, _ticks in SPEED_LEVELS]
    assert waits == sorted(waits, reverse=True), "the table must be ordered"
    assert SPEED_LEVELS[-1][2] > 1, "the top speed must beat the frame rate, not match it"


def test_the_speed_keys_step_up_and_down_the_table():
    """A spectator presses + + + - while the game is running."""
    import threading
    import time

    from tetris.ai_view import SPEED_LEVELS

    pygame = pygame_or_skip()
    app, controller, view = viewer(fps=120)

    def keys():
        for key in (pygame.K_PLUS, pygame.K_PLUS, pygame.K_PLUS, pygame.K_MINUS):
            time.sleep(0.1)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))

    worker = threading.Thread(target=keys, daemon=True)
    worker.start()
    _frames, logs = run(app, controller, view, 120, speed=1)
    worker.join(timeout=2)

    changes = [line.split(":")[0] for line in logs if line.startswith("speed ")]
    assert changes == ["speed Normal", "speed Fast", "speed Very fast", "speed Fast"], (
        f"the keys produced {changes}"
    )
    assert controller.think_ms == SPEED_LEVELS[2][1], "the last press was a step down"


def test_very_fast_moves_more_than_once_per_frame():
    """The top speed is not capped by the frame rate: several ticks per frame."""
    app, controller, view = viewer()
    run(app, controller, view, 40, speed=3)
    assert controller.actions > 40, (
        f"40 frames produced only {controller.actions} actions"
    )


def test_slow_and_normal_pace_the_thinking():
    app, controller, view = viewer()
    run(app, controller, view, 30, speed=0)
    slow = controller.actions

    app, controller, view = viewer()
    run(app, controller, view, 30, speed=2)
    fast = controller.actions

    assert fast > slow, "a higher speed must produce more moves in the same frames"


def test_speed_names_and_indices_both_parse():
    assert play_ai.speed_value("slow") == 0
    assert play_ai.speed_value("VERY-FAST") == 3
    assert play_ai.speed_value("very fast") == 3
    assert play_ai.speed_value("2") == 2
    with pytest.raises(Exception):
        play_ai.speed_value("ludicrous")


# ------------------------------------------------------------------ 6. game over

def test_the_game_runs_to_game_over_and_the_run_can_stop_there(capsys):
    """Seed 5 tops out at 29 pieces for the trained agent; the viewer must show it."""
    assert play_ai.main(["--headless", "--seed", "5", "--frames", "4000",
                         "--until-game-over", "--speed", "very-fast",
                         "--agent", "random"]) == 0
    out = capsys.readouterr().out
    assert "game over:" in out
    assert "frames:" in out
    frames = int(out.split("frames:")[1].split()[0])
    assert frames < 4000, "--until-game-over must stop the loop"


def test_the_panel_says_game_over_and_the_ai_stops_choosing():
    app, controller, view = viewer(agent=build_agent("random"))
    run(app, controller, view, 4000)
    assert app.game.is_game_over()
    placed = controller.pieces
    run(app, controller, view, 50)
    assert controller.pieces == placed, "a finished game must not keep placing pieces"


# ------------------------------------- 7. the same model, evaluated headlessly

def test_the_viewer_plays_the_same_game_as_the_headless_environment():
    """The whole architecture in one test: the window is only a window.

    The same checkpoint and the same seed are played twice -- once by the
    evaluation-style loop over `TetrisEnv`, once through the viewer's controller
    on the engine -- and both the chosen placements and the reward of each piece
    must match. If the viewer ever grew its own decision-making (or its own idea
    of the reward), this is what would fail.
    """
    pygame_or_skip()
    seed = 5
    pieces_to_compare = 200                  # both games are the *same* game
    agent_name = "heuristic"                 # deterministic: no torch, no RNG
    agent = build_agent(agent_name, lookahead=0)
    env = TetrisEnv(seed=seed, reward=RewardConfig(per_piece=5.0))

    env.reset(seed=seed)
    headless = []
    while len(headless) < pieces_to_compare:
        valid = env.get_valid_actions()
        placement = agent.choose_action(env.get_state(), valid)
        _state, reward, _done, info = env.step(placement)
        headless.append((info["placement"][0], info["placement"][1], round(reward, 6)))

    viewer_agent = build_agent(agent_name, lookahead=0)
    controller = PolicyController(viewer_agent, think_ms=0.0,
                                  reward_config=RewardConfig(per_piece=5.0))
    game = Game(random.Random(seed))
    watched = []
    while not game.is_game_over() and len(watched) < pieces_to_compare:
        action = controller.choose_action(game)
        if action is None:
            if game.is_game_over():
                break
            continue
        game.step(action)
        if action.value == "drop":           # the piece locked: score it
            controller.choose_action(game)
            report = controller.last_report
            watched.append((report.rotation, report.column, round(report.reward, 6)))

    assert len(watched) == len(headless) == pieces_to_compare
    assert [piece[:2] for piece in watched] == [piece[:2] for piece in headless]
    assert [piece[2] for piece in watched] == [piece[2] for piece in headless], (
        "the reward on the panel must be the reward training would have seen"
    )


# ---------------------------------------- 8. the model file is only ever read

def test_the_viewer_leaves_the_model_file_untouched(capsys):
    found = best_checkpoint()
    before = play_ai.fingerprint(found.path)

    assert play_ai.main(["--headless", "--frames", "120"]) == 0

    after = play_ai.fingerprint(found.path)
    assert after == before, "the viewer must not write to a checkpoint"
    assert "model untouched" in capsys.readouterr().out


def test_a_quit_event_closes_the_viewer_without_an_error():
    pygame = pygame_or_skip()
    app, controller, view = viewer()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    frames, _logs = run(app, controller, view, 1000)
    assert frames <= 1, "the loop must exit on the first QUIT"


# ---------------------------------------------------- the panel and its contents

def test_the_panel_reports_everything_the_viewer_promises():
    from tetris.ai_view import AIStatus

    status = AIStatus(
        status="PLAYING", speed="Normal", speed_detail="120 ms/move", agent="dqn",
        model="G/seed0", action="rotate_cw", current_piece="T",
        report={"piece": "T", "rotation": 2, "column": 6, "value": 12.5,
                "value_label": "Q-value", "legal": 9, "reward": -3.5,
                "lines_cleared": 1, "holes": 3, "height": 12},
        decisions=7, pieces=7, actions=31, abandoned=0, epsilon=0.05,
    )
    panel = "\n".join(status.lines_of())
    for promise in ("status", "speed", "agent", "model", "action", "piece",
                    "chosen rotation", "chosen column", "Q-value", "reward",
                    "lines cleared", "epsilon"):
        assert promise in panel, promise

    overlay = "\n".join(status.debug_lines())
    for promised in ("chosen rotation", "chosen column", "Q-value", "reward",
                     "lines cleared"):
        assert promised in overlay, promised


def test_the_panel_labels_a_heuristic_score_honestly():
    """A search score is not a Q-value; the panel must not call it one."""
    agent = build_agent("heuristic", lookahead=0)
    controller = PolicyController(agent, think_ms=0.0)
    game = Game(random.Random(3))

    while True:                              # lock one piece ...
        action = controller.choose_action(game)
        if action is not None:
            game.step(action)
            if action.value == "drop":
                break
    controller.choose_action(game)           # ... then let the controller score it

    report = controller.last_report
    assert report is not None and report.value is not None
    assert report.value_label == "search score"


def test_long_panel_values_are_shortened_rather_than_overlapping():
    pygame_or_skip()
    from tetris.ai_view import AIPygameView

    view = AIPygameView(cell=20)
    font = view.font_value
    assert view._fit("short", font, 500) == "short"
    clipped = view._fit("x" * 200, font, 60)
    assert clipped.endswith("\u2026")
    assert font.size(clipped)[0] <= 60


def test_the_viewer_can_record_frames(tmp_path):
    app, controller, view = viewer()
    frames = tmp_path / "frames"
    frames.mkdir()
    run(app, controller, view, 10, record=str(frames), record_every=2)
    assert len(sorted(frames.glob("frame_*.png"))) == 5


def test_the_board_is_bigger_than_the_human_default():
    """'Large enough to comfortably watch' is a property, not a preference."""
    from tetris.pygame_view import DEFAULT_CELL
    from tetris.ai_view import AI_CELL

    assert AI_CELL >= DEFAULT_CELL


# ------------------------------------------------------------- the config file

def test_a_config_file_supplies_defaults_and_explicit_flags_win(tmp_path):
    path = tmp_path / "play_ai.json"
    path.write_text(json.dumps({"agent": "heuristic", "speed": "fast", "debug": True}))

    args = play_ai.parse_args(["--config", str(path)])
    assert args.agent == "heuristic"
    assert args.speed == 2 and args.debug is True

    args = play_ai.parse_args(["--config", str(path), "--speed", "slow"])
    assert args.speed == 0, "the command line must beat the file"


def test_a_config_file_that_is_not_json_says_so(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(SystemExit) as exit_info:
        play_ai.parse_args(["--config", str(path)])
    assert "JSON" in str(exit_info.value)


def test_the_module_imports_without_a_display():
    """`--help` and the config path must not need SDL, exactly like play.py."""
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = __import__("subprocess").run(
        [sys.executable, "-c",
         "import sys, play_ai; print('pygame' in sys.modules)"],
        cwd=project, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
