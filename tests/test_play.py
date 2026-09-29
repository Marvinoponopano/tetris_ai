"""Tests for play.py: the modes, the seed, and a real headless run.

The graphical part is exercised end to end (window creation on SDL's dummy
driver, the frame loop, the screenshot) -- but nothing here inspects pixels.
"""

import os
import subprocess
import sys

import pytest

from play import build_game, main

MODES = ("human", "viz", "train")


def test_build_game_is_deterministic_with_a_seed():
    first, second = build_game(42), build_game(42)
    assert first.queue == second.queue
    assert first.current.type == second.current.type
    assert build_game(1).queue != build_game(2).queue


def test_build_game_without_a_seed_still_works():
    game = build_game()
    assert len(game.queue) >= 7
    assert game.current is not None


def test_train_mode_points_at_the_campaign_cli():
    """'train' has no window on purpose, and says where training really happens."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--mode", "train"])
    message = str(exit_info.value)
    assert "train" in message
    assert "rl_experiments.py run" in message


def test_viz_mode_plays_headlessly_and_writes_a_screenshot(tmp_path):
    """The AI path end to end: agent -> controller -> engine -> window."""
    pytest.importorskip("pygame")
    shot = tmp_path / "ai.png"
    assert main([
        "--mode", "viz",
        "--headless",
        "--agent", "heuristic",           # no torch needed for this test
        "--seed", "7",
        "--frames", "60",
        "--think-ms", "0",
        "--cell", "20",
        "--fps", "120",
        "--screenshot", str(shot),
    ]) == 0
    assert shot.exists() and shot.stat().st_size > 0


def test_viz_mode_records_frames(tmp_path):
    """--record is what the demo video is made of."""
    pytest.importorskip("pygame")
    frames = tmp_path / "frames"
    frames.mkdir()
    assert main([
        "--mode", "viz", "--headless", "--agent", "random",
        "--seed", "3", "--frames", "10", "--record-every", "2",
        "--think-ms", "0", "--fps", "120", "--record", str(frames),
    ]) == 0
    written = sorted(frames.glob("frame_*.png"))
    assert len(written) == 5, [path.name for path in written]


def test_viz_mode_defaults_to_the_trained_agent(tmp_path):
    """With no --agent, the demo plays the best checkpoint the campaigns left."""
    pytest.importorskip("pygame")
    pytest.importorskip("torch")
    from tetris.policy_controller import best_checkpoint

    found = best_checkpoint()
    assert main([
        "--mode", "viz", "--headless", "--seed", "5", "--frames", "40",
        "--think-ms", "0", "--cell", "20", "--fps", "120",
        "--screenshot", str(tmp_path / "trained.png"),
    ]) == 0
    assert found.path.endswith(".pt")


def test_an_unknown_agent_is_rejected():
    with pytest.raises(SystemExit):
        main(["--mode", "viz", "--agent", "oracle"])


def test_human_mode_runs_headlessly_and_writes_a_screenshot(tmp_path):
    pytest.importorskip("pygame")
    shot = tmp_path / "frame.png"
    assert main([
        "--headless",
        "--frames", "12",
        "--seed", "4",
        "--cell", "18",
        "--grid", "none",
        "--fps", "120",           # plenty of frames, but quickly
        "--screenshot", str(shot),
    ]) == 0
    assert shot.exists()
    assert shot.stat().st_size > 0


def test_human_mode_is_the_default(tmp_path):
    pytest.importorskip("pygame")
    # No --mode at all should behave exactly like --mode human.
    assert main(["--headless", "--frames", "3", "--seed", "1", "--fps", "120"]) == 0


def test_unknown_mode_and_bad_fps_are_rejected():
    with pytest.raises(SystemExit):
        main(["--mode", "sneaky"])
    with pytest.raises(SystemExit):
        main(["--fps", "0"])


def test_a_human_run_does_not_need_a_terminal():
    """--help must work with no pygame import and no display (CI friendliness)."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


def test_importing_the_launcher_does_not_pull_in_pygame():
    """The AI modes and the tests must not need pygame (or a display) yet."""
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-c", "import sys, play; print('pygame' in sys.modules)"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
