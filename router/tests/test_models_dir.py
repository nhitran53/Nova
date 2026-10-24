"""Trained checkpoints live outside the repo: $NOVA_MODELS_DIR, default /home/isaach/models/Luna/router."""
from pathlib import Path

import pytest

from router import demo, eval_sets, evaluate, pipeline, train
from router.core import DEFAULT_MODELS_DIR, models_dir

HERE = Path(__file__).parent.parent


def test_default_models_dir_is_outside_repo(monkeypatch):
    monkeypatch.delenv("NOVA_MODELS_DIR", raising=False)
    assert DEFAULT_MODELS_DIR == "/home/isaach/models/Luna/router"
    assert models_dir() == Path(DEFAULT_MODELS_DIR)
    assert HERE not in models_dir().parents


def test_env_var_overrides_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_MODELS_DIR", str(tmp_path))
    assert models_dir() == tmp_path


@pytest.mark.parametrize("module, name", [(train, "synthetic-v0"), (pipeline, "v1"), (evaluate, "v1"),
                                          (eval_sets, "v1"), (demo, "synthetic-v0")])
def test_cli_defaults_follow_models_dir(module, name, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NOVA_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "400")
    argv = ["--text", "x", "--help"] if module is pipeline else ["--help"]
    with pytest.raises(SystemExit):
        module.main(argv)
    out = "".join(capsys.readouterr().out.split())
    assert str(tmp_path / name) in out


def test_no_checkpoints_inside_repo():
    assert not (HERE / "checkpoints").exists()
