"""PyTest shared fixtures and configuration for duckV1 test suite."""

import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_isolated_env(tmp_path: Path, base_env: dict = None) -> dict:
    """Create isolated subprocess environment redirecting checkpoint writes away from tracked models/ and pipelines/."""
    site_file = tmp_path / "sitecustomize.py"
    repo_root = REPO_ROOT.resolve().as_posix()
    tmp_dir = tmp_path.resolve().as_posix()

    site_code = f"""import os, sys
from pathlib import Path

try:
    import torch
    _orig = torch.save
    _repo_root = Path(r"{repo_root}").resolve()
    _models_dir = _repo_root / "models"
    _pipelines_dir = _repo_root / "pipelines"
    _tmp_dir = Path(r"{tmp_dir}").resolve()

    def _safe_save(obj, f, *a, **k):
        try:
            target_p = Path(str(f)).resolve()
            if target_p.is_relative_to(_models_dir) or target_p.is_relative_to(_pipelines_dir):
                dest = _tmp_dir / target_p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                return _orig(obj, str(dest), *a, **k)
        except Exception:
            pass
        return _orig(obj, f, *a, **k)

    torch.save = _safe_save

    _orig_replace = os.replace

    def _safe_replace(src, dst, *a, **k):
        try:
            src_p = Path(str(src)).resolve()
            dst_p = Path(str(dst)).resolve()
            if dst_p.is_relative_to(_models_dir) or dst_p.is_relative_to(_pipelines_dir):
                safe_dst = _tmp_dir / dst_p.name
                safe_dst.parent.mkdir(parents=True, exist_ok=True)
                if src_p.is_relative_to(_models_dir) or src_p.is_relative_to(_pipelines_dir):
                    safe_src = _tmp_dir / src_p.name
                    if safe_src.exists():
                        return _orig_replace(str(safe_src), str(safe_dst), *a, **k)
                return _orig_replace(src, str(safe_dst), *a, **k)
        except Exception:
            pass
        return _orig_replace(src, dst, *a, **k)

    os.replace = _safe_replace
except Exception:
    pass
"""
    site_file.write_text(site_code, encoding="utf-8")

    env = dict(base_env if base_env is not None else os.environ)
    existing_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{existing_py_path}" if existing_py_path else str(tmp_path)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@pytest.fixture
def isolated_cli_env(tmp_path: Path) -> dict:
    """Fixture providing isolated subprocess environment for CLI training and export tests."""
    return make_isolated_env(tmp_path)
