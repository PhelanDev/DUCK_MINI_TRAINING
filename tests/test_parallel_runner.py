"""Tests for Batched Parallel Simulation Runner and FPS Benchmark.

Verifies parallel execution, environment observation/action integration,
CLI flags, error handling, and benchmark metrics.
"""

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import MicroDuck8DoFConfig
from training.run_parallel_sim import ParallelSimRunner, run_benchmark


def test_parallel_sim_benchmark_cli():
    """Verify CLI execution and output formatting."""
    res = subprocess.run(
        [
            sys.executable,
            "training/run_parallel_sim.py",
            "--num_envs",
            "16",
            "--steps",
            "50",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Sim failed with stderr:\n{res.stderr}"
    assert "Parallel Simulation Benchmark" in res.stdout
    assert "Steps Per Second (FPS):" in res.stdout
    assert "Real-time Factor:" in res.stdout
    assert "Environments: 16" in res.stdout
    assert "Total Steps:" in res.stdout


def test_parallel_sim_runner_api():
    """Verify Python API of ParallelSimRunner directly."""
    cfg = MicroDuck8DoFConfig()
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=8,
        cfg=cfg,
        decimation=5,
    )

    try:
        assert runner.num_envs == 8
        assert runner.obs_dim == 37
        assert runner.action_dim == 8

        # Reset environments
        obs = runner.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8, 37)

        # Step environments with random normalized actions [-1, 1]
        actions = np.random.uniform(-1.0, 1.0, size=(8, 8)).astype(np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)

        assert next_obs.shape == (8, 37)
        assert rewards.shape == (8,)
        assert dones.shape == (8,)
        assert len(infos) == 8
        assert np.all(np.isfinite(next_obs))
        assert np.all(np.isfinite(rewards))

        # Test partial reset
        sub_obs = runner.reset([0, 3])
        assert sub_obs.shape == (8, 37)
    finally:
        runner.close()


def test_parallel_sim_runner_action_shape_check():
    """Verify action shape defensive check."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=4,
    )
    try:
        with pytest.raises(AssertionError, match="Expected action shape"):
            runner.step(np.zeros((4, 7), dtype=np.float32))
    finally:
        runner.close()


def test_parallel_sim_runner_invalid_model_path():
    """Verify error raised on non-existent model path."""
    with pytest.raises(FileNotFoundError, match="MuJoCo model not found"):
        ParallelSimRunner(model_path="non_existent_model.xml", num_envs=2)


def test_parallel_sim_benchmark_threading():
    """Verify benchmark runs with multiple worker threads."""
    metrics = run_benchmark(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=16,
        steps=20,
        num_threads=2,
    )

    assert metrics["num_envs"] == 16
    assert metrics["steps"] == 20
    assert metrics["total_steps"] == 320
    assert metrics["fps"] > 0
    assert metrics["real_time_factor"] > 0
    assert metrics["elapsed_time"] > 0


def test_parallel_sim_cli_invalid_path():
    """Verify CLI returns non-zero code on invalid model path."""
    res = subprocess.run(
        [
            sys.executable,
            "training/run_parallel_sim.py",
            "--model_path",
            "non_existent.xml",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "Error during parallel simulation" in res.stderr


def test_parallel_sim_body_frame_velocity_tracking():
    """Verify linear velocity is projected to body frame so yaw rotation tracks correctly."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
    )
    try:
        runner.reset()
        # Env 0: Robot facing world +X (identity quat), moving forward along +X at 0.5 m/s
        # Target command: [0.5, 0.0, 0.0]
        runner.commands[0] = [0.5, 0.0, 0.0]
        runner.datas[0].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        runner.datas[0].qvel[0:3] = [0.5, 0.0, 0.0]

        # Env 1: Robot rotated 90 deg (yaw +pi/2, facing world +Y), moving along world +Y at 0.5 m/s
        # In body frame, this is forward velocity vx = 0.5 m/s!
        runner.commands[1] = [0.5, 0.0, 0.0]
        runner.datas[1].qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        runner.datas[1].qvel[0:3] = [0.0, 0.5, 0.0]

        # Step with zero action (hold)
        actions = np.zeros((2, 8), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)

        # Both environments must have matching tracking reward, invariant to yaw orientation!
        r0 = infos[0]["tracking_lin_vel"]
        r1 = infos[1]["tracking_lin_vel"]
        assert r0 > 0.5, f"Env 0 tracking reward should be > 0.5, got {r0}"
        assert r1 > 0.5, f"Env 1 (yaw 90) tracking reward should be > 0.5, got {r1}"
        np.testing.assert_allclose(r0, r1, atol=1e-4, err_msg=f"Yaw-invariant rewards must match: {r0} vs {r1}")
    finally:
        runner.close()


def test_parallel_sim_foot_contact_binarization():
    """Verify foot contact in observations is binarized to 0.0/1.0 matching hardware."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
    )
    try:
        obs = runner.reset()
        foot_contacts = obs[:, 33:37]
        # Should be strictly 0.0 or 1.0 (binary)
        assert np.all((foot_contacts == 0.0) | (foot_contacts == 1.0))
        # Padded elements [35:37] must be 0.0
        assert np.all(foot_contacts[:, 2:] == 0.0)
    finally:
        runner.close()
