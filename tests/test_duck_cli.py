"""Unit tests for unified MicroDuck CLI (duck.py)."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_UTF8 = dict(os.environ, PYTHONIOENCODING="utf-8")


def test_duck_cli_help():
    """Verify duck.py exists and returns help message."""
    duck_py = REPO_ROOT / "duck.py"
    assert duck_py.exists(), "duck.py must exist at repository root"

    res = subprocess.run([sys.executable, str(duck_py), "--help"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0
    assert "status" in res.stdout
    assert "verify" in res.stdout
    assert "train" in res.stdout
    assert "view" in res.stdout
    assert "export" in res.stdout


def test_duck_cli_status():
    """Verify duck.py status outputs checkpoint and curriculum info."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run([sys.executable, str(duck_py), "status"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0
    assert "Stage" in res.stdout or "stage" in res.stdout
    assert "MicroDuck" in res.stdout


def test_duck_cli_verify_help():
    """Verify duck.py verify --help works."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run([sys.executable, str(duck_py), "verify", "--help"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0


