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
        assert runner.obs_dim in (37, 42)
        assert runner.action_dim == 8

        # Reset environments
        obs = runner.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8, runner.obs_dim)

        # Step environments with random normalized actions [-1, 1]
        actions = np.random.uniform(-1.0, 1.0, size=(8, 8)).astype(np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)

        assert next_obs.shape == (8, runner.obs_dim)
        assert rewards.shape == (8,)
        assert dones.shape == (8,)
        assert len(infos) == 8
        assert np.all(np.isfinite(next_obs))
        assert np.all(np.isfinite(rewards))

        # Test partial reset
        sub_obs = runner.reset([0, 3])
        assert sub_obs.shape == (8, runner.obs_dim)
    finally:
        runner.close()


def test_parallel_sim_stair_ascent_info():
    """Verify ParallelSimRunner includes stair_ascent in step infos when on stairs."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
    )
    try:
        runner.reset()
        # Teleport env 0 into stair zone at elevated height
        runner.datas[0].qpos[0] = 1.45  # x >= 1.35m
        runner.datas[0].qpos[2] = 0.190  # z > 0.170m

        actions = np.zeros((2, 8), dtype=np.float32)
        _, _, _, infos = runner.step(actions)

        assert "stair_ascent" in infos[0]
        # Env 0 is on stairs elevated, should receive positive stair_ascent
        assert infos[0]["stair_ascent"] > 0.0
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
        # Synchronize joints between env 0 and 1 so velocity tracking reward is strictly yaw-invariant
        runner.datas[1].qpos[7:15] = runner.datas[0].qpos[7:15].copy()
        runner.last_targets[1] = runner.last_targets[0].copy()

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


def test_parallel_sim_domain_randomization_and_velocity_commands():
    """Verify reset_env applies domain randomization on joint angles and commands."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=4,
    )
    try:
        assert hasattr(runner, "max_episode_steps")
        assert runner.max_episode_steps == 500
        assert hasattr(runner, "episode_steps")
        assert hasattr(runner, "rng")
        assert runner.episode_steps.shape == (4,)
        assert np.all(runner.episode_steps == 0)

        # Check domain randomization on commands [vx: 0.0 to 0.25, vy: -0.08 to 0.08, wz: -0.50 to 0.50]
        cmds = runner.commands
        assert np.all(cmds[:, 0] >= 0.0) and np.all(cmds[:, 0] <= 0.25)
        assert np.all(cmds[:, 1] >= -0.08) and np.all(cmds[:, 1] <= 0.08)
        assert np.all(cmds[:, 2] >= -0.50) and np.all(cmds[:, 2] <= 0.50)

        # Check joint perturbation in qpos[7:15] around nominal within [-0.03, 0.03]
        for d in runner.datas:
            joint_angles = d.qpos[7:15]
            diff = joint_angles - runner.cfg.nominal_qpos
            assert np.all(np.abs(diff) <= 0.03 + 1e-6)
            # Knees must be >= 0 (left_knee_pitch: index 11, right_knee_pitch: index 14)
            assert d.qpos[11] >= 0.0
            assert d.qpos[14] >= 0.0

        # last_targets must match randomized joint angles
        for i in range(runner.num_envs):
            np.testing.assert_allclose(
                runner.last_targets[i],
                runner.datas[i].qpos[7:15].astype(np.float32),
            )
    finally:
        runner.close()


def test_parallel_sim_episode_steps_and_truncation():
    """Verify episode_steps increments each step and triggers done when reaching max_episode_steps."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
    )
    try:
        actions = np.zeros((2, 8), dtype=np.float32)

        # Step 3 times
        for s in range(1, 4):
            next_obs, rewards, dones, infos = runner.step(actions)
            assert np.all(runner.episode_steps == s)
            assert not np.any(dones)

        # Fast forward env 0 to max_episode_steps - 1
        runner.episode_steps[0] = runner.max_episode_steps - 1
        runner.episode_steps[1] = 10

        next_obs, rewards, dones, infos = runner.step(actions)
        assert dones[0] == True, "Env 0 must trigger done on timeout (500 steps)"
        assert dones[1] == False, "Env 1 must not trigger done"
        # Check auto-reset of episode_steps on env 0
        assert runner.episode_steps[0] == 0, "Env 0 episode_steps must be reset to 0 after done"
        assert runner.episode_steps[1] == 11
        assert infos[0].get("timeout") == True
        assert infos[1].get("timeout") == False
    finally:
        runner.close()


def test_parallel_sim_aligned_next_obs_on_done():
    """Verify next_obs returned by step() is aligned with the reset observation for done envs."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
    )
    try:
        # Cause env 0 to fall over / tilt upside down
        runner.datas[0].qpos[3:7] = [0.0, 1.0, 0.0, 0.0]
        runner.datas[0].qpos[2] = 0.05

        actions = np.zeros((2, 8), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)

        assert dones[0] == True
        # Env 0 was reset inside step: its data has root_z ~ 0.170
        assert runner.datas[0].qpos[2] > 0.15
        # next_obs[0] must match the observation of the reset state, NOT the fallen 0.02 state!
        expected_reset_obs = runner._extract_obs(0)
        np.testing.assert_allclose(next_obs[0], expected_reset_obs, atol=1e-5)
    finally:
        runner.close()


def test_parallel_sim_fall_penalty_applied():
    """Verify that fall_penalty is applied to environments where the robot falls."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=1,
    )
    try:
        runner.reset()
        runner.datas[0].qpos[3:7] = [0.0, 1.0, 0.0, 0.0]
        runner.datas[0].qpos[2] = 0.05
        actions = np.zeros((1, 8), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)
        assert dones[0] == True
        assert infos[0]["fallen"] == True
        assert rewards[0] < -3.0
    finally:
        runner.close()


def test_parallel_sim_reset_nominal_stance():
    """Verify reset_env initializes qpos around NOMINAL_JOINT_POS with knee bent and within limits."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=1,
    )
    try:
        runner.reset()
        qpos_joints = runner.datas[0].qpos[7:15]
        assert 0.25 <= qpos_joints[4] <= 0.45, f"Left knee expected ~0.35, got {qpos_joints[4]}"
        assert 0.25 <= qpos_joints[7] <= 0.45, f"Right knee expected ~0.35, got {qpos_joints[7]}"
        assert -0.25 <= qpos_joints[3] <= -0.05, f"Left hip expected ~-0.15, got {qpos_joints[3]}"
        assert -0.25 <= qpos_joints[6] <= -0.05, f"Right hip expected ~-0.15, got {qpos_joints[6]}"
        assert 0.155 <= runner.datas[0].qpos[2] <= 0.170, f"CoM z expected ~0.165, got {runner.datas[0].qpos[2]}"
    finally:
        runner.close()


def test_parallel_sim_stand_still_mode():
    """Verify that when stand_still=True, all velocity commands remain zero across resets."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=4,
        stand_still=True,
    )
    try:
        assert np.all(runner.commands == 0.0)
        runner.reset()
        assert np.all(runner.commands == 0.0)
        # Test toggling stand_still off
        runner.set_stand_still(False)
        runner.reset()
        # When stand_still is False, commands should be randomized
        assert np.any(runner.commands != 0.0)
    finally:
        runner.close()



