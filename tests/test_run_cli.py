"""Tests for the experiment command line (`experiments/run.py`).

The script is loaded from its path rather than imported as a module, because
`experiments/` is a folder of scripts and data, not a package. `main(argv)`
takes its arguments as a list, so these tests need no `sys.argv` patching.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_PY = ROOT / "experiments" / "run.py"
CONFIGS = ROOT / "experiments" / "configs"


def load_run_cli():
    spec = importlib.util.spec_from_file_location("experiments_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_cli = load_run_cli()

TINY = {"agent": "random", "num_games": 2, "seed": 1, "max_pieces": 8}


def write_config(tmp_path, payload, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# ------------------------------------------------------------ loading configs


def test_load_configs_reads_one_file_one_config(tmp_path):
    path = write_config(tmp_path, TINY)
    configs = run_cli.load_configs([str(path)])
    assert len(configs) == 1
    assert configs[0].num_games == 2


def test_load_configs_reads_a_list_from_one_file(tmp_path):
    path = write_config(tmp_path, [TINY, {**TINY, "seed": 2}])
    assert [config.seed for config in run_cli.load_configs([str(path)])] == [1, 2]


def test_load_configs_applies_overrides_to_every_config(tmp_path):
    path = write_config(tmp_path, [TINY, {**TINY, "seed": 2}])
    configs = run_cli.load_configs([str(path)], ["num_games=1", "weights=null"])
    assert [config.num_games for config in configs] == [1, 1]


def test_load_configs_merges_several_files(tmp_path):
    one = write_config(tmp_path, TINY, "one.json")
    two = write_config(tmp_path, {**TINY, "seed": 9}, "two.json")
    assert len(run_cli.load_configs([str(one), str(two)])) == 2


def test_several_paths_can_follow_one_config_flag(tmp_path, capsys):
    """`--config a.json b.json` is what people type first: it must work."""
    one = write_config(tmp_path, TINY, "one.json")
    two = write_config(tmp_path, {**TINY, "seed": 2}, "two.json")

    code = run_cli.main(["--dry-run", "--config", str(one), str(two)])

    assert code == 0
    assert capsys.readouterr().out.count("\n") == 2


def test_the_config_flag_can_also_be_repeated(tmp_path, capsys):
    one = write_config(tmp_path, TINY, "one.json")
    two = write_config(tmp_path, {**TINY, "seed": 2}, "two.json")

    code = run_cli.main(["--dry-run", "--config", str(one), "--config", str(two)])

    assert code == 0
    assert capsys.readouterr().out.count("\n") == 2


def test_a_glob_of_configs_works(tmp_path, capsys):
    write_config(tmp_path, TINY, "a_one.json")
    write_config(tmp_path, {**TINY, "seed": 2}, "b_two.json")
    paths = sorted(str(path) for path in tmp_path.glob("*.json"))

    code = run_cli.main(["--dry-run", "--config", *paths])

    assert code == 0
    assert capsys.readouterr().out.count("\n") == 2


# --------------------------------------------------------------------- commands


def test_dry_run_prints_what_would_run_and_writes_nothing(tmp_path, capsys):
    path = write_config(tmp_path, [TINY, {**TINY, "agent": "heuristic", "max_pieces": 5}])
    data = tmp_path / "data"

    code = run_cli.main(["--dry-run", "--config", str(path), "--data-dir", str(data)])

    printed = capsys.readouterr().out
    assert code == 0
    assert printed.count("\n") == 2
    assert "random" in printed and "heuristic" in printed
    assert not data.exists()


def test_running_a_config_writes_a_result_and_a_log(tmp_path, capsys):
    path = write_config(tmp_path, TINY)
    data = tmp_path / "data"

    code = run_cli.main(["--config", str(path), "--data-dir", str(data), "--quiet"])

    printed = capsys.readouterr().out
    assert code == 0
    assert (data / "runs.jsonl").exists()
    assert len(list(data.glob("run-*.json"))) == 1
    assert "games/s" in printed                      # the comparison table
    assert "results in" in printed

    row = json.loads((data / "runs.jsonl").read_text().splitlines()[0])
    assert row["agent"] == "random"
    assert row["stats"]["score_mean"] >= 0


def test_overrides_change_what_is_saved(tmp_path):
    path = write_config(tmp_path, TINY)
    data = tmp_path / "data"

    run_cli.main(["--config", str(path), "--data-dir", str(data), "--quiet",
                  "--set", "num_games=1", "--set", "seed=42"])

    row = json.loads((data / "runs.jsonl").read_text().splitlines()[0])
    assert row["num_games"] == 1
    assert row["seed"] == 42


def test_running_several_configs_lists_them_all(tmp_path, capsys):
    path = write_config(tmp_path, [TINY, {**TINY, "agent": "heuristic", "max_pieces": 5}])
    data = tmp_path / "data"

    code = run_cli.main(["--config", str(path), "--data-dir", str(data), "--quiet"])

    printed = capsys.readouterr().out
    assert code == 0
    assert "[1/2]" in printed and "[2/2]" in printed
    assert len(json.loads("[" + ",".join((data / "runs.jsonl").read_text().splitlines()) + "]")) == 2


def test_profiling_leaves_a_report_behind(tmp_path):
    path = write_config(tmp_path, {**TINY, "agent": "heuristic", "max_pieces": 5})
    data = tmp_path / "data"

    code = run_cli.main(["--config", str(path), "--data-dir", str(data), "--quiet",
                         "--profile"])

    reports = list(data.glob("profile-*.txt"))
    assert code == 0
    assert len(reports) == 1
    assert "choose_action" in reports[0].read_text()


def test_list_prints_every_run_in_the_directory(tmp_path, capsys):
    path = write_config(tmp_path, TINY)
    data = tmp_path / "data"
    run_cli.main(["--config", str(path), "--data-dir", str(data), "--quiet"])
    run_cli.main(["--config", str(path), "--data-dir", str(data), "--quiet"])
    capsys.readouterr()

    code = run_cli.main(["--list", "--data-dir", str(data)])

    printed = capsys.readouterr().out
    assert code == 0
    assert "score mean" in printed
    assert len([line for line in printed.splitlines() if "random" in line]) == 2


def test_list_can_write_a_csv(tmp_path, capsys):
    path = write_config(tmp_path, TINY)
    data = tmp_path / "data"
    run_cli.main(["--config", str(path), "--data-dir", str(data), "--quiet"])
    capsys.readouterr()

    csv_path = tmp_path / "compared.csv"
    code = run_cli.main(["--list", "--data-dir", str(data), "--csv", str(csv_path)])

    assert code == 0
    assert csv_path.exists()
    assert len(csv_path.read_text().splitlines()) == 2       # header + one run


def test_list_on_an_empty_directory_says_so(tmp_path, capsys):
    code = run_cli.main(["--list", "--data-dir", str(tmp_path / "nothing")])
    assert code == 0
    assert "nothing to list" in capsys.readouterr().out


# ------------------------------------------------------------------- failures


def test_a_bad_config_file_exits_nonzero_with_a_message(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"agent": "random", "number_of_games": 5}')      # wrong key

    code = run_cli.main(["--config", str(path), "--data-dir", str(tmp_path / "data")])

    captured = capsys.readouterr()
    assert code == 1
    assert "bad configuration" in captured.err
    assert "number_of_games" in captured.err


def test_a_bad_override_exits_nonzero(tmp_path, capsys):
    path = write_config(tmp_path, TINY)
    code = run_cli.main(["--config", str(path), "--set", "nope=1",
                         "--data-dir", str(tmp_path / "data")])
    assert code == 1
    assert "bad configuration" in capsys.readouterr().err


def test_config_is_required_without_list(tmp_path):
    with pytest.raises(SystemExit) as problem:
        run_cli.main(["--data-dir", str(tmp_path / "data")])
    assert problem.value.code == 2


# ------------------------------------------------------------- shipped configs


def test_every_shipped_example_config_loads():
    from tetris.config import ExperimentConfig

    files = sorted(CONFIGS.glob("*.json"))
    assert files, "the example configs are part of the documentation"

    loaded = [(path.name, config) for path in files
              for config in ExperimentConfig.load_all(path)]
    assert any(config.agent == "random" for _, config in loaded)
    assert any(config.agent == "heuristic" for _, config in loaded)
    assert len({config.config_id for _, config in loaded}) == len(loaded)
