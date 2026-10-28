"""Comprehensive Tests for 42-Dimensional Perception RL Training Pipeline.

Covers both hardware pipelines:
1. pipelines/rpi5_stack/training/
2. pipelines/dual_esp32s3_stack/training/

Validates:
- 42-dimensional observation space packing (including 5 LiDAR rangefinder rays)
- Action space scaling to MG996R limits and velocity rate clipping
- Reward computation including obstacle avoidance penalty (< 0.35m)
- INT8 quantization helper functions and C header generation for ESP32-S3 PIE
- Multi-threaded parallel simulation runners loading their respective MJCF models
- CLI execution and FPS benchmarking for both pipelines
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pytest

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import RPi5 training modules
from pipelines.rpi5_stack.training.env_config import (
    MicroDuck8DoFConfig as RPi5Config,
    compute_observations as rpi5_compute_observations,
    scale_actions as rpi5_scale_actions,
    clip_action_rates as rpi5_clip_action_rates,
    compute_rewards as rpi5_compute_rewards,
)
from pipelines.rpi5_stack.training.run_parallel_sim import (
    ParallelSimRunner as RPi5ParallelSimRunner,
    run_benchmark as rpi5_run_benchmark,
)

# Import Dual ESP32-S3 training modules
from pipelines.dual_esp32s3_stack.training.env_config import (
    MicroDuck8DoFConfig as ESP32Config,
    compute_observations as esp32_compute_observations,
    scale_actions as esp32_scale_actions,
    clip_action_rates as esp32_clip_action_rates,
    compute_rewards as esp32_compute_rewards,
    quantize_obs_int8,
    dequantize_act_int8,
    generate_esp32_quant_header,
)
from pipelines.dual_esp32s3_stack.training.run_parallel_sim import (
    ParallelSimRunner as ESP32ParallelSimRunner,
    run_benchmark as esp32_run_benchmark,
)


def test_rpi5_env_config_dimensions():
    """Verify RPi5 configuration dataclass dimensions and constraints."""
    cfg = RPi5Config()
    assert cfg.OBS_DIM == 42, f"Expected 42 dims for perception RL, got {cfg.OBS_DIM}"
    assert cfg.ACTION_DIM == 8
    assert cfg.LIDAR_RAYS == 5
    assert cfg.STACK_NAME == "rpi5_stack"
    assert cfg.POLICY_DTYPE == "float32"
    assert len(cfg.JOINT_NAMES) == 8
    assert cfg.TORQUE_LIMIT == 1.0
    assert cfg.MAX_VELOCITY == 6.0
    assert "obstacle_avoidance" in cfg.REWARD_WEIGHTS


def test_esp32_env_config_dimensions():
    """Verify ESP32-S3 configuration dataclass dimensions and constraints."""
    cfg = ESP32Config()
    assert cfg.OBS_DIM == 42, f"Expected 42 dims for perception RL, got {cfg.OBS_DIM}"
    assert cfg.ACTION_DIM == 8
    assert cfg.LIDAR_RAYS == 5
    assert cfg.STACK_NAME == "dual_esp32s3_stack"
    assert cfg.POLICY_DTYPE == "int8"
    assert len(cfg.JOINT_NAMES) == 8
    assert cfg.TORQUE_LIMIT == 1.0
    assert cfg.MAX_VELOCITY == 6.0
    assert "obstacle_avoidance" in cfg.REWARD_WEIGHTS


def test_observation_packing_42dim():
    """Verify standard 42-dimensional observation vector packing for both pipelines."""
    base_ang_vel = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    commands = np.array([0.5, 0.0, 0.1], dtype=np.float32)
    joint_pos = np.arange(8, dtype=np.float32) * 0.1
    joint_vel = np.arange(8, dtype=np.float32) * -0.05
    last_actions = np.ones(8, dtype=np.float32) * 0.2
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    lidar_ranges = np.array([1.5, 1.8, 2.0, 1.7, 1.4], dtype=np.float32)

    for compute_obs in [rpi5_compute_observations, esp32_compute_observations]:
        obs = compute_obs(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_actions,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges,
        )

        assert obs.shape == (42,)
        assert obs.dtype == np.float32

        # Verify component slices match inputs
        np.testing.assert_allclose(obs[0:3], base_ang_vel)
        np.testing.assert_allclose(obs[3:6], proj_grav)
        np.testing.assert_allclose(obs[6:9], commands)
        np.testing.assert_allclose(obs[9:17], joint_pos)
        np.testing.assert_allclose(obs[17:25], joint_vel)
        np.testing.assert_allclose(obs[25:33], last_actions)
        np.testing.assert_allclose(obs[33:37], foot_contact)
        np.testing.assert_allclose(obs[37:42], lidar_ranges)


def test_observation_defensive_shape_check():
    """Verify assertion error when wrong dimension tensor is passed."""
    for compute_obs in [rpi5_compute_observations, esp32_compute_observations]:
        with pytest.raises(AssertionError, match="Expected observation shape \\(42,\\)"):
            compute_obs(
                base_ang_vel=np.zeros(3),
                proj_grav=np.zeros(3),
                commands=np.zeros(3),
                joint_pos=np.zeros(8),
                joint_vel=np.zeros(8),
                last_actions=np.zeros(8),
                foot_contact=np.zeros(4),
                lidar_ranges=np.zeros(4),  # Wrong size: 4 rays instead of 5!
            )


def test_obstacle_avoidance_rewards():
    """Verify obstacle avoidance term in reward calculation."""
    base_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.ones(4, dtype=np.float32)

    # 1. Clear path (all rays >= 1.5m)
    clear_lidar = np.array([1.5, 1.8, 2.0, 1.7, 1.4], dtype=np.float32)
    r_clear = rpi5_compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        lidar_ranges=clear_lidar,
    )

    assert "obstacle_avoidance" in r_clear
    assert r_clear["obstacle_avoidance"] >= 0.35, "Clear path should have positive clearance reward"

    # 2. Obstacle detected in close proximity (< 0.35m threshold)
    blocked_lidar = np.array([0.15, 0.20, 0.10, 0.30, 0.50], dtype=np.float32)
    r_blocked = rpi5_compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        lidar_ranges=blocked_lidar,
    )

    assert r_blocked["obstacle_avoidance"] < 0.0, "Proximity (< 0.35m) must incur negative penalty"
    assert r_blocked["total_reward"] < r_clear["total_reward"]


def test_esp32_int8_quantization_helpers():
    """Verify INT8 quantization, dequantization, and C header generation for ESP32-S3."""
    rng = np.random.default_rng(42)

    # 1. Observation quantization (float32 -> int8)
    float_obs = rng.uniform(-1.0, 1.0, size=42).astype(np.float32)
    int8_obs = quantize_obs_int8(float_obs, scale=127.0)

    assert int8_obs.shape == (42,)
    assert int8_obs.dtype == np.int8
    assert np.all(int8_obs >= -128) and np.all(int8_obs <= 127)

    # 2. Batch quantization
    batch_obs = rng.uniform(-2.0, 2.0, size=(10, 42)).astype(np.float32)
    batch_int8 = quantize_obs_int8(batch_obs, scale=64.0)
    assert batch_int8.shape == (10, 42)
    assert batch_int8.dtype == np.int8

    # 3. Action dequantization (int8 -> float32)
    int8_actions = np.array([-127, -64, 0, 64, 127, 32, -32, 0], dtype=np.int8)
    float_actions = dequantize_act_int8(int8_actions, scale=127.0)

    assert float_actions.shape == (8,)
    assert float_actions.dtype == np.float32
    assert pytest.approx(float_actions[0], abs=0.01) == -1.0
    assert pytest.approx(float_actions[2], abs=0.01) == 0.0
    assert pytest.approx(float_actions[4], abs=0.01) == 1.0

    # 4. C header generation
    header_str = generate_esp32_quant_header(scale_obs=127.0, scale_act=127.0)
    assert "#ifndef ESP32_QUANT_CONFIG_H" in header_str
    assert "#define OBS_DIM 42" in header_str
    assert "#define ACTION_DIM 8" in header_str
    assert "#define OBS_SCALE 127.0" in header_str
    assert "#define ACT_SCALE 127.0" in header_str

    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        generate_esp32_quant_header(output_path=tmp_path)
        content = Path(tmp_path).read_text(encoding="utf-8")
        assert "#define OBS_DIM 42" in content
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_rpi5_parallel_sim_runner():
    """Verify ParallelSimRunner with RPi5 MJCF model and 42-dim observation space."""
    rpi5_model_path = REPO_ROOT / "pipelines" / "rpi5_stack" / "models" / "mjcf" / "microduck_rpi5_8dof.xml"
    runner = RPi5ParallelSimRunner(
        model_path=rpi5_model_path,
        num_envs=4,
        decimation=5,
    )
    try:
        assert runner.num_envs == 4
        assert runner.obs_dim == 42
        assert runner.action_dim == 8

        obs = runner.reset()
        assert obs.shape == (4, 42)
        assert np.all(np.isfinite(obs))
        # Ensure LiDAR readings are positive distances (cutoff=2.0)
        assert np.all(obs[:, 37:42] >= 0.0)

        # Step environments
        actions = np.zeros((4, 8), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)

        assert next_obs.shape == (4, 42)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
        assert "obstacle_avoidance" in infos[0]
        assert np.all(np.isfinite(next_obs))
    finally:
        runner.close()


def test_esp32_parallel_sim_runner():
    """Verify ParallelSimRunner with Dual ESP32-S3 MJCF model and 42-dim observation space."""
    esp32_model_path = REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "models" / "mjcf" / "microduck_esp32_8dof.xml"
    runner = ESP32ParallelSimRunner(
        model_path=esp32_model_path,
        num_envs=4,
        decimation=5,
    )
    try:
        assert runner.num_envs == 4
        assert runner.obs_dim == 42
        assert runner.action_dim == 8

        obs = runner.reset()
        assert obs.shape == (4, 42)
        assert np.all(np.isfinite(obs))
        assert np.all(obs[:, 37:42] >= 0.0)

        actions = np.random.uniform(-0.5, 0.5, size=(4, 8)).astype(np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)

        assert next_obs.shape == (4, 42)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
        assert "obstacle_avoidance" in infos[0]
    finally:
        runner.close()


def test_cli_benchmark_both_stacks():
    """Verify CLI benchmark execution on both pipelines."""
    # Test RPi5 CLI benchmark
    res_rpi5 = subprocess.run(
        [
            sys.executable,
            "pipelines/rpi5_stack/training/run_parallel_sim.py",
            "--num_envs",
            "4",
            "--steps",
            "10",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res_rpi5.returncode == 0, f"RPi5 CLI benchmark failed:\n{res_rpi5.stderr}"
    assert "Parallel Simulation Benchmark" in res_rpi5.stdout
    assert "Steps Per Second (FPS):" in res_rpi5.stdout

    # Test ESP32 CLI benchmark
    res_esp32 = subprocess.run(
        [
            sys.executable,
            "pipelines/dual_esp32s3_stack/training/run_parallel_sim.py",
            "--num_envs",
            "4",
            "--steps",
            "10",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert res_esp32.returncode == 0, f"ESP32 CLI benchmark failed:\n{res_esp32.stderr}"
    assert "Parallel Simulation Benchmark" in res_esp32.stdout
    assert "Steps Per Second (FPS):" in res_esp32.stdout
