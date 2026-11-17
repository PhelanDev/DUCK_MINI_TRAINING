"""Tests for Stage 1 slalom navigation and speed fixes, including LiDAR filtering."""

from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.run_parallel_sim import ParallelSimRunner
from training.env_config import compute_rewards, LocomotionMode, MicroDuck8DoFConfig


def test_lidar_d_min_dot_ignores_miss_acquisition_and_bounds_closing_speed():
    """Verify d_min_dot filters ray transitions (reset, miss-acquisition, hit-to-miss) and bounds speed to [-1, 1]."""
    sim = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=1,
        stage=1,
        terrain_mode="hurdles",
        randomize_pillars=False,
    )
    try:
        actions = np.zeros((1, sim.action_dim), dtype=np.float32)

        # 1. Reset state with prev_d_min == 0.0 -> step() must produce d_min_dot == 0.0
        sim.reset()
        sim.prev_d_min[:] = 0.0
        sim.step(actions)
        assert sim.d_min_dot[0] == pytest.approx(0.0)

        # 2. Pillar acquisition transition: prev_d_min == 2.0 (miss/max range), next step sees pillar (curr_d_min < 1.99)
        # Without filter, (curr_d_min - 2.0) / 0.02 produces large negative spikes. With filter, must be 0.0.
        sim.prev_d_min[:] = 2.0
        sim.step(actions)
        assert sim.d_min[0] < 1.99, "Expected obstacle within LiDAR range"
        assert sim.d_min_dot[0] == pytest.approx(0.0)

        # 3. Ray miss transition: prev was tracking an obstacle (prev_d_min = 0.5), next step loses ray to 2.0 (curr_d_min >= 1.99)
        # Move robot lateral out of corridor so all LiDAR rays return cutoff 2.0
        sim.datas[0].qpos[1] = 3.0
        mujoco.mj_forward(sim.model, sim.datas[0])
        sim.prev_d_min[:] = 0.50
        sim.step(actions)
        assert sim.d_min[0] >= 1.99
        assert sim.d_min_dot[0] == pytest.approx(0.0)

        # 4. Rapid ray variation between steps during continuous tracking: bounded to [-1.0, 1.0]
        # Move robot near pillar at x = 0.5, curr_d_min is ~0.495m.
        # If prev_d_min was 0.80m, raw_dot would be (0.495 - 0.80) / 0.02 = -15.25 m/s.
        # Clamped physical bound must produce -1.0 m/s.
        sim.datas[0].qpos[0] = 0.50
        sim.datas[0].qpos[1] = 0.0
        mujoco.mj_forward(sim.model, sim.datas[0])
        sim.prev_d_min[:] = 0.80
        sim.step(actions)
        assert sim.d_min[0] < 1.99
        assert sim.d_min_dot[0] == pytest.approx(-1.0)
        assert np.all(np.abs(sim.d_min_dot) <= 1.0)
    finally:
        sim.close()


def test_hurdles_wz_lim_scales_by_curriculum_stage():
    """Verify hurdles wz_lim scales smoothly across curriculum stages (0.04, 0.06, 0.08, 0.10)."""
    expected_wz_lims = {
        0: 0.04,
        1: 0.06,
        2: 0.08,
        3: 0.10,
    }
    for stage, expected_lim in expected_wz_lims.items():
        sim = ParallelSimRunner(
            model_path="duck_v2/cyberduck_v2.xml",
            num_envs=256,
            stage=stage,
            terrain_mode="hurdles",
            randomize_pillars=False,
        )
        try:
            sim.reset()
            wz_cmds = sim.commands[:, 2]
            assert np.all(np.abs(wz_cmds) <= expected_lim + 1e-6), (
                f"Stage {stage} cmd_wz exceeded {expected_lim}: max={np.max(np.abs(wz_cmds))}"
            )
            if stage == 1:
                assert np.any(np.abs(wz_cmds) > 0.05), (
                    "Stage 1 must deterministically produce commands with |cmd_wz| > 0.05"
                )
        finally:
            sim.close()


def test_heading_penalty_relaxes_near_close_obstacle():
    """Verify heading_penalty is relaxed near close obstacles and restored when clear."""
    base_inputs = {
        "base_lin_vel": np.array([0.15, 0.0, 0.0], dtype=np.float32),
        "cmd_lin_vel": np.array([0.20, 0.0, 0.0], dtype=np.float32),
        "base_ang_vel": np.zeros(3, dtype=np.float32),
        "cmd_ang_vel": 0.0,
        "proj_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "joint_torques": np.zeros(8, dtype=np.float32),
        "joint_vel": np.zeros(8, dtype=np.float32),
        "foot_contact": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    base_penalty = -2.0 * (0.4 ** 2)  # -0.32

    # 1. Scalar tests
    scalar_cases = [
        (0.30, 0.0),
        (0.575, 0.5 * base_penalty),
        (0.90, base_penalty),
        (None, base_penalty),
    ]
    for d_val, expected in scalar_cases:
        res = compute_rewards(
            **base_inputs,
            current_yaw=0.4,
            target_yaw=0.0,
            cmd_mode=LocomotionMode.WALK,
            d_min=d_val,
        )
        assert isinstance(res["heading_penalty"], float)
        assert res["heading_penalty"] == pytest.approx(expected, abs=1e-5)

    # 2. Batched tests
    n_envs = 4
    batch_inputs = {
        "base_lin_vel": np.tile(base_inputs["base_lin_vel"], (n_envs, 1)),
        "cmd_lin_vel": np.tile(base_inputs["cmd_lin_vel"], (n_envs, 1)),
        "base_ang_vel": np.tile(base_inputs["base_ang_vel"], (n_envs, 1)),
        "cmd_ang_vel": np.zeros((n_envs, 3), dtype=np.float32),
        "proj_gravity": np.tile(base_inputs["proj_gravity"], (n_envs, 1)),
        "joint_torques": np.tile(base_inputs["joint_torques"], (n_envs, 1)),
        "joint_vel": np.tile(base_inputs["joint_vel"], (n_envs, 1)),
        "foot_contact": np.tile(base_inputs["foot_contact"], (n_envs, 1)),
    }
    batch_d_min = np.array([0.30, 0.575, 0.90, 1.50], dtype=np.float32)
    expected_batch = np.array([0.0, 0.5 * base_penalty, base_penalty, base_penalty], dtype=np.float32)

    res_batch = compute_rewards(
        **batch_inputs,
        current_yaw=np.full(n_envs, 0.4, dtype=np.float32),
        target_yaw=np.zeros(n_envs, dtype=np.float32),
        cmd_mode=np.zeros(n_envs, dtype=np.int32),
        d_min=batch_d_min,
    )
    assert isinstance(res_batch["heading_penalty"], np.ndarray)
    assert res_batch["heading_penalty"].shape == (n_envs,)
    np.testing.assert_allclose(res_batch["heading_penalty"], expected_batch, atol=1e-5)

    # 3. Batched with d_min=None
    res_batch_none = compute_rewards(
        **batch_inputs,
        current_yaw=np.full(n_envs, 0.4, dtype=np.float32),
        target_yaw=np.zeros(n_envs, dtype=np.float32),
        cmd_mode=np.zeros(n_envs, dtype=np.int32),
        d_min=None,
    )
    assert isinstance(res_batch_none["heading_penalty"], np.ndarray)
    np.testing.assert_allclose(
        res_batch_none["heading_penalty"],
        np.full(n_envs, base_penalty, dtype=np.float32),
        atol=1e-5,
    )


def test_speed_gradient_reward_gain_doubles():
    """Verify that tracking_lin_vel weight doubles velocity gradient gain in total_reward.

    For commanded speed 0.21 m/s, moving at 0.21 m/s vs 0.11 m/s produces an unweighted
    tracking_lin_vel delta of 2.0 * (0.21 - 0.11) = 0.20. With tracking_lin_vel weight 3.0,
    the total_reward difference must equal exactly 3.0 * 2.0 * 0.10 = 0.60.
    """
    base_inputs = {
        "cmd_lin_vel": np.array([0.21, 0.0, 0.0], dtype=np.float32),
        "base_ang_vel": np.zeros(3, dtype=np.float32),
        "cmd_ang_vel": 0.0,
        "proj_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "joint_torques": np.zeros(8, dtype=np.float32),
        "joint_vel": np.zeros(8, dtype=np.float32),
        "foot_contact": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }

    # Low speed (0.11 m/s)
    res_011 = compute_rewards(
        base_lin_vel=np.array([0.11, 0.0, 0.0], dtype=np.float32),
        **base_inputs,
    )
    # Target commanded speed (0.21 m/s)
    res_021 = compute_rewards(
        base_lin_vel=np.array([0.21, 0.0, 0.0], dtype=np.float32),
        **base_inputs,
    )

    # Raw tracking_lin_vel component remains 2.0 * vx
    assert res_011["tracking_lin_vel"] == pytest.approx(2.0 * 0.11, abs=1e-5)
    assert res_021["tracking_lin_vel"] == pytest.approx(2.0 * 0.21, abs=1e-5)

    # total_reward difference must reflect weight 3.0: 3.0 * 0.20 = 0.60
    total_delta = res_021["total_reward"] - res_011["total_reward"]
    assert total_delta == pytest.approx(0.60, abs=1e-5)


def test_cyberduck_foot_clearance_target_matches_8cm_leg():
    """Verify target_foot_height in default config matches 8cm leg geometry (0.015m) and eliminates clearance penalty at apex."""
    cfg = MicroDuck8DoFConfig()
    assert "target_foot_height" in cfg.REWARD_WEIGHTS
    assert cfg.REWARD_WEIGHTS["target_foot_height"] == pytest.approx(0.015)

    base_inputs = {
        "base_lin_vel": np.array([0.20, 0.0, 0.0], dtype=np.float32),
        "cmd_lin_vel": np.array([0.20, 0.0, 0.0], dtype=np.float32),
        "base_ang_vel": np.zeros(3, dtype=np.float32),
        "cmd_ang_vel": 0.0,
        "proj_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "joint_torques": np.zeros(8, dtype=np.float32),
        "joint_vel": np.zeros(8, dtype=np.float32),
        "foot_contact": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }

    # Left foot (index 0) in swing (fz = 0.0 < 1.0 N, moving at 0.30 m/s)
    # Right foot (index 1) in stance (fz = 5.0 >= 1.0 N, stationary)
    foot_vel_xy = np.array([[0.30, 0.0], [0.0, 0.0]], dtype=np.float32)
    foot_contact_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)

    # 1. Apex swing at target height 0.015m receives zero clearance penalty with default weights
    foot_z_apex = np.array([0.015, 0.0], dtype=np.float32)
    res_apex = compute_rewards(
        **base_inputs,
        foot_z=foot_z_apex,
        foot_vel_xy=foot_vel_xy,
        foot_contact_forces=foot_contact_forces,
    )
    assert res_apex["foot_clearance"] == pytest.approx(0.0, abs=1e-6)

    # 2. Dragging swing at low height 0.006m receives strictly negative clearance penalty
    foot_z_low = np.array([0.006, 0.0], dtype=np.float32)
    res_low = compute_rewards(
        **base_inputs,
        foot_z=foot_z_low,
        foot_vel_xy=foot_vel_xy,
        foot_contact_forces=foot_contact_forces,
    )
    assert res_low["foot_clearance"] < 0.0
    assert res_apex["foot_clearance"] > res_low["foot_clearance"]
