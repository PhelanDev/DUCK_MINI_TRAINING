"""Regression tests for CyberDuck V2 touch sensors, gait anti-bobbing locks, and Stage 0 LiDAR filtering."""

from pathlib import Path
import sys
import numpy as np
import mujoco
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import LocomotionMode, compute_rewards
from training.run_parallel_sim import ParallelSimRunner


def _reward_inputs(**overrides):
    base = {
        "base_lin_vel": np.array([0.15, 0.0, 0.0], dtype=np.float32),
        "cmd_lin_vel": np.array([0.20, 0.0, 0.0], dtype=np.float32),
        "base_ang_vel": np.zeros(3, dtype=np.float32),
        "cmd_ang_vel": 0.0,
        "proj_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "joint_torques": np.zeros(8, dtype=np.float32),
        "joint_vel": np.zeros(8, dtype=np.float32),
        "foot_contact": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "clock_sin": 0.8,
        "cmd_mode": LocomotionMode.WALK,
    }
    base.update(overrides)
    return base


def test_cyberduck_v2_foot_touch_sensors_and_double_support_dwell():
    """CyberDuck V2 standing on flat floor must report touch > 0.1 N on both feet and accumulate dwell."""
    runner = ParallelSimRunner(model_path="duck_v2/cyberduck_v2.xml", num_envs=2, stage=0)
    try:
        runner.reset()
        actions = np.zeros((2, 8), dtype=np.float32)
        last_infos = None
        for _ in range(25):
            obs, _, _, last_infos = runner.step(actions)

        # Standing on both soles must read positive <touch> forces (> 0.1 N) and obs[33:35] == 1.0
        for d in runner.datas:
            assert float(d.sensordata[10]) > 0.1, f"left_foot_touch blind: {d.sensordata[10]}"
            assert float(d.sensordata[11]) > 0.1, f"right_foot_touch blind: {d.sensordata[11]}"
        np.testing.assert_allclose(obs[:, 33:35], 1.0)
        assert float(runner.double_support_time.min()) > 0.20
        assert last_infos[0]["double_support_penalty"] == pytest.approx(-0.5)

        # Lifted off the floor must read 0.0 N on both <touch> sensors
        for d in runner.datas:
            d.qpos[2] = 0.35
            mujoco.mj_forward(runner.model, d)
            assert float(d.sensordata[10]) == 0.0
            assert float(d.sensordata[11]) == 0.0
    finally:
        runner.close()


def test_foot_contact_reward_ignores_phase_clock_channels():
    """r_foot_contact must inspect only foot channels [:2], never counting clock_sin/clock_cos."""
    # Single +X vel with ZERO actual foot contact, but 1 positive clock channel
    r_air = compute_rewards(
        **_reward_inputs(foot_contact=np.array([0.0, 0.0, 0.9, -0.4], dtype=np.float32))
    )
    assert r_air["foot_contact"] == pytest.approx(0.0)

    # Stationary stand command with both feet on floor AND positive clock channels -> 0.5
    r_stand = compute_rewards(
        **_reward_inputs(
            base_lin_vel=np.zeros(3, dtype=np.float32),
            cmd_lin_vel=np.zeros(3, dtype=np.float32),
            foot_contact=np.ones(4, dtype=np.float32),
        )
    )
    assert r_stand["foot_contact"] == pytest.approx(0.5)


def test_gait_sync_blocks_in_place_bobbing_airborne_and_dwell():
    """gait_sync must stay 0.0 when vel_x <= 0.02, airborne [0,0], or dwelling (dst > 0.20s)."""
    # 1. Stationary / slow in-place bobbing (vx <= 0.02) -> 0.0
    for vx in (0.0, 0.005, 0.02):
        r_bob = compute_rewards(**_reward_inputs(base_lin_vel=np.array([vx, 0.0, 0.0], dtype=np.float32)))
        assert r_bob["gait_sync"] == pytest.approx(0.0), f"Expected 0.0 gait_sync at vx={vx}"

    # 2. Both feet off floor [0.0, 0.0] -> 0.0 even when moving forward
    r_no_touch = compute_rewards(
        **_reward_inputs(foot_contact=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    )
    assert r_no_touch["gait_sync"] == pytest.approx(0.0)

    # 3. Double-support dwell (dst > 0.20s) -> 0.0 even when shuffling at vx > 0.02
    r_dwell = compute_rewards(
        **_reward_inputs(
            foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            double_support_time=0.25,
        )
    )
    assert r_dwell["gait_sync"] == pytest.approx(0.0)

    # 4. Batched check matching 1D behavior
    batch_vel = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.15, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=np.float32)
    batch_fc = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    r_batch = compute_rewards(
        base_lin_vel=batch_vel,
        cmd_lin_vel=np.tile(np.array([0.20, 0.0, 0.0], dtype=np.float32), (4, 1)),
        base_ang_vel=np.zeros((4, 3), dtype=np.float32),
        cmd_ang_vel=np.zeros(4, dtype=np.float32),
        proj_gravity=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (4, 1)),
        joint_torques=np.zeros((4, 8), dtype=np.float32),
        joint_vel=np.zeros((4, 8), dtype=np.float32),
        foot_contact=batch_fc,
        double_support_time=np.array([0.0, 0.0, 0.25, 0.0], dtype=np.float32),
        clock_sin=np.full(4, 0.8, dtype=np.float32),
        cmd_mode=np.full(4, int(LocomotionMode.WALK), dtype=np.int32),
    )
    np.testing.assert_allclose(r_batch["gait_sync"], [0.0, 0.0, 0.0, 2.4], atol=1e-5)


def test_stage0_flat_lidar_filters_floor_hits():
    """Pitching forward on flat floor must filter phantom floor strikes; inactive obstacles report d_min=None."""
    runner = ParallelSimRunner(model_path="duck_v2/cyberduck_v2.xml", num_envs=2, stage=0, terrain_mode="hurdles")
    try:
        runner.reset()
        theta = np.deg2rad(15.0)
        for d in runner.datas:
            d.qpos[3:7] = [np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]
            mujoco.mj_forward(runner.model, d)
        obs, _, _, infos = runner.step(np.zeros((2, 8), dtype=np.float32))
        np.testing.assert_allclose(obs[:, 37:42], 2.0, atol=1e-5)
        assert infos[0]["d_min"] == pytest.approx(2.0)
        assert infos[0]["obstacle_avoidance"] == pytest.approx(0.0)

        # When all obstacle pillars are submerged below floor (z <= 0), d_min becomes None
        for gid in runner.obstacle_geom_ids:
            runner.model.geom_pos[gid, 2] = -0.50
        runner.reset()
        for d in runner.datas:
            d.qpos[3:7] = [np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]
            mujoco.mj_forward(runner.model, d)
        obs2, _, _, infos2 = runner.step(np.zeros((2, 8), dtype=np.float32))
        np.testing.assert_allclose(obs2[:, 37:42], 2.0, atol=1e-5)
        assert infos2[0]["d_min"] is None
        assert infos2[0]["obstacle_avoidance"] == pytest.approx(0.0)
    finally:
        runner.close()
