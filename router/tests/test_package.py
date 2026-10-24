"""The editable flat package must be importable from either documented directory."""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("cwd", [PROJECT, PROJECT.parent], ids=["project", "parent"])
def test_router_cli_module_importable(cwd):
    result = subprocess.run(
        [sys.executable, "-m", "router.cli", "--help"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--text" in result.stdout


def test_router_public_api_and_labels_packaged():
    from importlib.resources import files

    from router import Router, load_labels

    assert Router.__name__ == "Router"
    assert load_labels(files("router").joinpath("labels.json"))
