"""Unit tests for unified MicroDuck CLI (duck.py)."""

import os
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ENV_UTF8 = dict(os.environ, PYTHONIOENCODING="utf-8")


@pytest.mark.parametrize("target,prefix", [("rpi5", "microduck_rpi5"), ("pizero2w", "microduck_pizero2w"), ("esp32", "microduck_esp32"), ("v2", "cyberduck_v2")])
def test_audit_checkpoint_namespace(monkeypatch, target, prefix):
    import duck
    from types import SimpleNamespace
    commands = []
    monkeypatch.setattr(duck.subprocess, "run", lambda cmd: commands.append(cmd) or SimpleNamespace(returncode=0))
    args = duck.build_parser().parse_args(["train", "--target", target, "--resume"])
    assert duck.cmd_train(args) == 0
    cmd = commands[0]
    assert Path(cmd[cmd.index("--save-path") + 1]).name == prefix + "_ppo.pt"
    assert Path(cmd[cmd.index("--best-save-path") + 1]).name == prefix + "_best.pt"
    assert "--resume" in cmd


def test_audit_export_all_routes_distinct_checkpoints_and_preflights_missing(tmp_path, monkeypatch):
    import duck
    import scripts.export_policy as exporter
    monkeypatch.setattr(duck, "REPO_ROOT", tmp_path)
    directory = tmp_path / "models/checkpoints"
    directory.mkdir(parents=True)
    paths = {t: directory / (duck.CHECKPOINT_PREFIXES[t] + "_ppo.pt") for t in ("rpi5", "pizero2w", "esp32")}
    for path in paths.values():
        path.touch()
    calls = []
    monkeypatch.setattr(exporter, "prepare_target_exports", lambda checkpoints, *args: checkpoints)
    monkeypatch.setattr(exporter, "export_onnx", lambda checkpoint, **kw: calls.append(Path(checkpoint)) or tmp_path / "policy.onnx")
    monkeypatch.setattr(exporter, "export_esp32_headers", lambda checkpoint, **kw: calls.append(Path(checkpoint)) or (tmp_path / "w.h", tmp_path / "c.h"))
    args = duck.build_parser().parse_args(["export", "--target", "all"])
    assert duck.cmd_export(args) == 0
    assert calls == list(paths.values())
    paths["esp32"].unlink()
    calls.clear()
    assert duck.cmd_export(args) == 1
    assert not calls


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
    """Verify duck.py status outputs checkpoint, curriculum info, and 3 hardware targets."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run([sys.executable, str(duck_py), "status"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0
    assert "Stage" in res.stdout or "stage" in res.stdout
    assert "MicroDuck" in res.stdout
    assert "Pi Zero 2 W" in res.stdout
    assert "microduck_pizero2w_policy.onnx" in res.stdout
    assert "RPi 5" in res.stdout
    assert "ESP32-S3" in res.stdout


def test_duck_cli_verify_help():
    """Verify duck.py verify --help works."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run([sys.executable, str(duck_py), "verify", "--help"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0


def test_duck_cli_status_cyberduck_v2():
    """Verify duck.py status outputs CyberDuck V2 Pro information and checkpoint status."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run([sys.executable, str(duck_py), "status"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0
    assert "CYBERDUCK V2 PRO" in res.stdout
    assert "cyberduck_v2.xml" in res.stdout
    assert "cyberduck_v2_best.pt" in res.stdout


def test_duck_cli_train_v2_resume_dry_run(tmp_path, isolated_cli_env):
    """Verify duck.py train --target v2 --resume --dry-run executes cleanly."""
    duck_py = REPO_ROOT / "duck.py"
    from scripts.train_ppo import train_ppo
    checkpoint = tmp_path / "v2_ppo.pt"
    train_ppo(model_path=REPO_ROOT / "duck_v2/cyberduck_v2.xml", num_envs=1,
        rollout_steps=2, max_iterations=1, num_epochs=1, save_path=checkpoint, verbose=False, device="cpu")
    res = subprocess.run(
        [sys.executable, str(duck_py), "train", "--target", "v2", "--resume", "--dry-run",
         "--save-path", str(checkpoint), "--best-save-path", str(tmp_path / "v2_best.pt")],
        capture_output=True,
        encoding="utf-8",
        env=isolated_cli_env,
    )
    assert res.returncode == 0
    assert "cyberduck_v2_ppo.pt" in res.stdout or "Dry-run mode enabled" in res.stdout


def test_duck_cli_view_v2_headless():
    """Verify duck.py view --v2 --headless executes cleanly."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run([sys.executable, str(duck_py), "view", "--v2", "--headless"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0
    assert "CyberDuck V2 Pro" in res.stdout or "Mô hình" in res.stdout


def test_duck_cli_train_v2_dry_run(tmp_path, isolated_cli_env):
    """Verify duck.py train --target v2 --dry-run executes 1 iteration cleanly."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run(
        [sys.executable, str(duck_py), "train", "--target", "v2", "--dry-run",
         "--save-path", str(tmp_path / "fresh_v2_ppo.pt")],
        capture_output=True,
        encoding="utf-8",
        env=isolated_cli_env,
    )
    assert res.returncode == 0
    assert "Dry-run mode enabled" in res.stdout or "PPO" in res.stdout


def test_duck_cli_status_advanced_gaits_and_vision():
    """Verify duck.py status outputs Advanced Gaits and AI Vision readiness."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run([sys.executable, str(duck_py), "status"], capture_output=True, encoding="utf-8", env=ENV_UTF8)
    assert res.returncode == 0
    assert "KỸ NĂNG VẬN ĐỘNG NÂNG CAO & AI VISION" in res.stdout
    assert "Waddling Gait" in res.stdout
    assert "Spot Turn" in res.stdout
    assert "Fall Recovery" in res.stdout
    assert "AI Vision Head Tracking" in res.stdout


def test_duck_cli_view_gait_and_track_headless():
    """Verify duck.py view with --gait and --track in headless mode."""
    duck_py = REPO_ROOT / "duck.py"
    res = subprocess.run(
        [sys.executable, str(duck_py), "view", "--gait", "waddle", "--track", "--headless"],
        capture_output=True,
        encoding="utf-8",
        env=ENV_UTF8,
    )
    assert res.returncode == 0
    assert "CHẾ ĐỘ HEADLESS" in res.stdout


def test_viewer_3d_html_single_active_tab_and_this_contract():
    """Verify viewer_3d.html has single active tab and passes 'this' in showTab(tabId, this)."""
    html_path = REPO_ROOT / "viewer_3d.html"
    assert html_path.exists(), "viewer_3d.html must exist"
    html = html_path.read_text(encoding="utf-8")

    # (a) Exactly 1 active tab pane and 1 active tab button
    assert html.count('class="tab-pane active"') == 1, "Must have exactly 1 active tab pane"
    assert html.count('class="tab-btn active"') == 1, "Must have exactly 1 active tab button"

    # All tab buttons must pass 'this'
    import re
    onclick_matches = re.findall(r'onclick="showTab\(([^)]+)\)"', html)
    assert len(onclick_matches) >= 3, "Expected multiple showTab calls in buttons"
    for args_str in onclick_matches:
        assert "this" in args_str, f"showTab must receive 'this', got: {args_str}"

    # showTab function definition must accept (tabId, btn)
    assert "function showTab(tabId, btn)" in html or "function showTab(tabId, btn = null)" in html


def test_readme_export_commands_parse_cleanly_via_duck_cli():
    """Verify every `python duck.py export ...` command extracted from README.md parses cleanly."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")

    import re
    import shlex
    import importlib.util

    export_cmds = re.findall(r'python duck\.py export\s+([^\r\n]+)', content)
    assert len(export_cmds) >= 3, f"Expected at least 3 export commands in README.md, found: {export_cmds}"

    duck_py = REPO_ROOT / "duck.py"
    spec = importlib.util.spec_from_file_location("duck", str(duck_py))
    duck_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(duck_mod)

    parser_fn = getattr(duck_mod, "build_parser", None)
    if parser_fn is None:
        for cmd_args_str in export_cmds:
            args_list = shlex.split(cmd_args_str)
            res = subprocess.run([sys.executable, str(duck_py), "export"] + args_list, capture_output=True, encoding="utf-8")
            assert res.returncode == 0, f"Command failed: python duck.py export {cmd_args_str}\n{res.stderr}"
    else:
        p = parser_fn()
        targets_found = set()
        for cmd_args_str in export_cmds:
            args_list = shlex.split(cmd_args_str)
            parsed = p.parse_args(["export"] + args_list)
            assert parsed.command == "export"
            targets_found.add(getattr(parsed, "target", None))
        assert "rpi5" in targets_found
        assert "pizero2w" in targets_found
        assert "esp32" in targets_found


def test_duck_cli_export_target_routing_isolated(tmp_path, isolated_cli_env):
    """Test duck.py export target routing without dirtying models/ or pipelines/."""
    duck_py = REPO_ROOT / "duck.py"
    import torch
    from scripts.train_ppo import ActorCritic
    from training.policy_contract import observation_schema
    ckpt_path = tmp_path / "export.pt"
    model = ActorCritic(hidden_dim=16)
    checkpoint = dict(model_state_dict=model.state_dict(), model_name="microduck_rpi5_8dof.xml",
        observation_schema=observation_schema(42, True, 0.02))
    torch.save(checkpoint, ckpt_path)

    # Test rpi5 export to tmp_path
    out_rpi5 = tmp_path / "test_rpi5.onnx"
    res = subprocess.run(
        [sys.executable, str(duck_py), "export", "--target", "rpi5", "--checkpoint", str(ckpt_path), "--output", str(out_rpi5)],
        capture_output=True,
        encoding="utf-8",
        env=isolated_cli_env,
    )
    assert res.returncode == 0, f"Export rpi5 failed:\n{res.stderr}\n{res.stdout}"
    assert out_rpi5.exists() and out_rpi5.stat().st_size > 1000

    # Test pizero2w export to tmp_path
    checkpoint["model_name"] = "microduck_pizero2w_8dof.xml"
    torch.save(checkpoint, ckpt_path)
    out_pizero = tmp_path / "test_pizero.onnx"
    res = subprocess.run(
        [sys.executable, str(duck_py), "export", "--target", "pizero2w", "--checkpoint", str(ckpt_path), "--output", str(out_pizero)],
        capture_output=True,
        encoding="utf-8",
        env=isolated_cli_env,
    )
    assert res.returncode == 0, f"Export pizero2w failed:\n{res.stderr}\n{res.stdout}"
    assert out_pizero.exists() and out_pizero.stat().st_size > 1000

    # Test esp32 export to tmp_path
    checkpoint["model_name"] = "microduck_esp32_8dof.xml"
    torch.save(checkpoint, ckpt_path)
    out_esp32_w = tmp_path / "test_esp32.h"
    out_esp32_c = tmp_path / "test_esp32_cfg.h"
    res = subprocess.run(
        [
            sys.executable,
            str(duck_py),
            "export",
            "--target",
            "esp32",
            "--checkpoint",
            str(ckpt_path),
            "--esp32-weights-output",
            str(out_esp32_w),
            "--esp32-config-output",
            str(out_esp32_c),
        ],
        capture_output=True,
        encoding="utf-8",
        env=isolated_cli_env,
    )
    assert res.returncode == 0, f"Export esp32 failed:\n{res.stderr}\n{res.stdout}"
    assert out_esp32_w.exists() and out_esp32_w.stat().st_size > 1000
    assert out_esp32_c.exists() and out_esp32_c.stat().st_size > 100


def test_pizero2w_runner_walk_min_z_and_no_duplicate_info():
    """Verify ParallelSimRunner uses walk_min_z parameter/attribute and has no duplicate timeout loop."""
    import inspect
    from pipelines.pizero2w_stack.training.run_parallel_sim import ParallelSimRunner
    sig = inspect.signature(ParallelSimRunner.__init__)
    assert "walk_min_z" in sig.parameters, "ParallelSimRunner.__init__ must have walk_min_z parameter"
    assert sig.parameters["walk_min_z"].default == 0.060

    runner_src_path = REPO_ROOT / "pipelines" / "pizero2w_stack" / "training" / "run_parallel_sim.py"
    src = runner_src_path.read_text(encoding="utf-8")
    assert "self.walk_min_z" in src, "ParallelSimRunner must store self.walk_min_z"
    assert "root_z < self.walk_min_z" in src or "root_z < getattr(self, 'walk_min_z'" in src, (
        "fallen check must use walk_min_z rather than hardcoded 0.08"
    )

    # Check for duplicate timeout loop
    timeout_loop_count = src.count('infos[i]["timeout"] = bool(timeouts[i])')
    assert timeout_loop_count == 1, f"Expected exactly 1 timeout assignment, found {timeout_loop_count}"


def test_no_orphan_scripts_and_no_machine_hardcoded_paths():
    """Verify orphan debug scripts are removed and no developer hardcoded paths remain."""
    assert not (REPO_ROOT / "scripts" / "find_url.py").exists(), "scripts/find_url.py should be removed"
    assert not (REPO_ROOT / "scripts" / "check_pollen_repo.py").exists(), "scripts/check_pollen_repo.py should be removed"

    checked_scripts = [
        REPO_ROOT / "scripts" / "render_model_3d.py",
        REPO_ROOT / "scripts" / "render_exploded_view.py",
        REPO_ROOT / "duck_v2" / "scripts" / "export_cyberduck_v2_cad.py",
        REPO_ROOT / "scripts" / "inspect_parts.py",
    ]
    for p in checked_scripts:
        assert p.exists(), f"Missing script {p}"
        content = p.read_text(encoding="utf-8")
        assert "LEGION" not in content, f"Hardcoded developer path in {p}"
        assert "eac089d5" not in content, f"Hardcoded conversation path in {p}"
