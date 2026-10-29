"""Tests for obstacle anti-stuck rewards: command relaxation and approach-velocity penalty."""

from pathlib import Path
import sys
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import (
    ang_vel_mask,
    avoid_vel_penalty,
    compute_rewards,
    MicroDuck8DoFConfig,
)
from training.run_parallel_sim import DuckParallelSim


def test_ang_mask_values():
    """Verify angular velocity relaxation mask values across distance thresholds."""
    # d_danger = 0.25, d_safe = 0.55
    d_samples = np.array([0.60, 0.55, 0.40, 0.25, 0.10], dtype=np.float32)
    masks = ang_vel_mask(d_samples)

    assert pytest.approx(float(masks[0]), abs=1e-5) == 1.0
    assert pytest.approx(float(masks[1]), abs=1e-5) == 1.0
    assert pytest.approx(float(masks[2]), abs=1e-5) == 0.5
    assert pytest.approx(float(masks[3]), abs=1e-5) == 0.1
    assert pytest.approx(float(masks[4]), abs=1e-5) == 0.1


def test_avoid_penalty_zero_when_receding():
    """Verify that avoid_vel_penalty is zero when moving away from obstacles (d_min_dot > 0)."""
    d_min = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    d_min_dot = np.array([0.1, 0.3, 0.5], dtype=np.float32)  # Positive = moving away

    penalties = avoid_vel_penalty(d_min, d_min_dot, sigma_obs=0.3)
    assert np.allclose(penalties, 0.0)


def test_avoid_penalty_stronger_when_closer():
    """Verify that approaching obstacles at the same speed is penalized more severely closer to the obstacle."""
    d_close = np.array([0.2], dtype=np.float32)
    d_far = np.array([0.8], dtype=np.float32)
    d_dot = np.array([-0.3], dtype=np.float32)  # Approaching speed 0.3 m/s

    pen_close = avoid_vel_penalty(d_close, d_dot, sigma_obs=0.3)
    pen_far = avoid_vel_penalty(d_far, d_dot, sigma_obs=0.3)

    assert pen_close[0] < 0.0
    assert pen_far[0] < 0.0
    # Penalty closer to obstacle is more negative (larger magnitude)
    assert pen_close[0] < pen_far[0]
    assert abs(pen_close[0]) > abs(pen_far[0])


def test_compute_rewards_backward_compatible():
    """Verify compute_rewards backward compatibility when d_min and d_min_dot are omitted."""
    cfg = MicroDuck8DoFConfig()
    base_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    cmd_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    base_ang = np.array([0.0, 0.0, 0.1], dtype=np.float32)
    cmd_ang = 0.1
    grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)
    f_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    lidar = np.ones(5, dtype=np.float32) * 1.5

    # Run without d_min, d_min_dot
    out_orig = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=cmd_ang,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        lidar_ranges=lidar,
        weights=cfg.REWARD_WEIGHTS,
    )

    # Run with d_min=None, d_min_dot=None explicitly
    out_explicit = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=cmd_ang,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        lidar_ranges=lidar,
        weights=cfg.REWARD_WEIGHTS,
        d_min=None,
        d_min_dot=None,
    )

    assert pytest.approx(out_orig["total_reward"]) == out_explicit["total_reward"]
    assert pytest.approx(out_orig["obstacle_avoidance"]) == out_explicit["obstacle_avoidance"]
    assert pytest.approx(out_orig["tracking_ang_vel"]) == out_explicit["tracking_ang_vel"]


def test_parallel_sim_tracks_d_min_and_d_min_dot():
    """Verify DuckParallelSim calculates d_min and d_min_dot across steps."""
    sim = DuckParallelSim(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
        decimation=1,
    )
    sim.reset()
    assert hasattr(sim, "prev_d_min")
    assert sim.prev_d_min is not None
    assert sim.prev_d_min.shape == (2,)

    actions = np.zeros((2, 8), dtype=np.float32)
    _, _, _, infos = sim.step(actions)
    assert hasattr(sim, "d_min")
    assert hasattr(sim, "d_min_dot")
    assert sim.d_min.shape == (2,)
    assert sim.d_min_dot.shape == (2,)


def test_lidar_pomdp_d_min_ignores_rear_obstacle():
    """Verify that when LiDAR is active, d_min is derived from LiDAR perception rather than rear obstacles."""
    import mujoco

    model_path = REPO_ROOT / "pipelines" / "rpi5_stack" / "models" / "mjcf" / "microduck_rpi5_8dof.xml"
    sim = DuckParallelSim(
        model_path=str(model_path),
        num_envs=1,
        decimation=1,
        stage=2,
    )
    # Move front/lateral obstacles/pillars/stairs far away (> 5m), keeping rear obstacle at x = -0.75m
    front_obstacle_names = [
        "obstacle_front", "obstacle_left", "obstacle_right",
        "pillar_l1_left", "pillar_l1_right", "pillar_l2_center", "pillar_l3_left", "pillar_l3_right",
        "stair_step_1", "stair_step_2", "stair_step_3", "stair_platform",
    ]
    for name in front_obstacle_names:
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            sim.model.geom_pos[gid, 0] = 10.0
            sim.model.geom_pos[gid, 1] = 10.0

    obs = sim.reset()
    # At reset: rear obstacle is at distance ~0.67m, but LiDAR facing forward sees nothing closer than 2.0m
    # sim.d_min must reflect LiDAR range (2.0m) and NOT rear obstacle distance (~0.67m)
    assert sim.obs_dim == 42
    assert sim.d_min[0] > 1.5, f"d_min was {sim.d_min[0]}, should ignore rear obstacle at 0.67m"
    assert pytest.approx(float(sim.d_min[0]), abs=0.05) == 2.0

    # Step simulation with zero actions
    actions = np.zeros((1, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)
    assert sim.d_min[0] > 1.5, f"d_min after step was {sim.d_min[0]}, should ignore rear obstacle"
    assert pytest.approx(float(sim.d_min[0]), abs=0.05) == 2.0


def test_stair_ascent_reward_calculation():
    """Verify stair ascent elevation reward calculation and zone thresholding."""
    cfg = MicroDuck8DoFConfig()
    assert "stair_ascent" in cfg.REWARD_WEIGHTS
    assert cfg.REWARD_WEIGHTS["stair_ascent"] == 1.5

    base_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    cmd_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    base_ang = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)
    f_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # 1. Single env: Before stairs (x = 1.0m < 1.35m) -> 0 reward even if elevated
    rew_before = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        weights=cfg.REWARD_WEIGHTS,
        root_x=1.0,
        root_z=0.200,
    )
    assert "stair_ascent" in rew_before
    assert pytest.approx(rew_before["stair_ascent"]) == 0.0

    # 2. Single env: On stairs (x = 1.40m >= 1.35m), elevated by 0.020m (z = 0.190m)
    # elevation_gain = 0.190 - 0.170 = 0.020m -> r_stair = 0.020 * 10.0 = 0.20
    rew_on_stairs = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        weights=cfg.REWARD_WEIGHTS,
        root_x=1.40,
        root_z=0.190,
    )
    assert pytest.approx(rew_on_stairs["stair_ascent"]) == 0.20
    # Difference in total reward should reflect weight * r_stair_ascent = 1.5 * 0.20 = 0.30
    diff_total = rew_on_stairs["total_reward"] - rew_before["total_reward"]
    assert pytest.approx(diff_total, abs=1e-5) == 0.30

    # 3. Single env: Elevation clipping at max 0.050m (z = 0.250m)
    rew_clipped = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        weights=cfg.REWARD_WEIGHTS,
        root_x=1.50,
        root_z=0.250,
    )
    assert pytest.approx(rew_clipped["stair_ascent"]) == 0.50

    # 4. Batched envs:
    b_lin = np.zeros((3, 3), dtype=np.float32)
    b_cmd = np.zeros((3, 3), dtype=np.float32)
    b_ang = np.zeros((3, 3), dtype=np.float32)
    b_grav = np.tile(grav, (3, 1))
    b_torques = np.zeros((3, 8), dtype=np.float32)
    b_j_vel = np.zeros((3, 8), dtype=np.float32)
    b_f_contact = np.tile(f_contact, (3, 1))
    root_x_batch = np.array([1.0, 1.40, 1.64], dtype=np.float32)
    root_z_batch = np.array([0.200, 0.185, 0.230], dtype=np.float32)

    rew_batch = compute_rewards(
        base_lin_vel=b_lin,
        cmd_lin_vel=b_cmd,
        base_ang_vel=b_ang,
        cmd_ang_vel=0.0,
        proj_gravity=b_grav,
        joint_torques=b_torques,
        joint_vel=b_j_vel,
        foot_contact=b_f_contact,
        weights=cfg.REWARD_WEIGHTS,
        root_x=root_x_batch,
        root_z=root_z_batch,
    )
    assert isinstance(rew_batch["stair_ascent"], np.ndarray)
    assert rew_batch["stair_ascent"].shape == (3,)
    # Env 0: x < 1.35 -> 0.0
    # Env 1: x >= 1.35, z = 0.185 -> (0.185 - 0.170) * 10 = 0.15
    # Env 2: x >= 1.35, z = 0.230 -> min(0.060, 0.050) * 10 = 0.50
    np.testing.assert_allclose(rew_batch["stair_ascent"], [0.0, 0.15, 0.50], atol=1e-5)


def test_compute_d_min_with_slalom_radii():
    """Verify DuckParallelSim._compute_d_min subtracts individual pillar radii accurately."""
    import mujoco

    sim = DuckParallelSim(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=1,
        decimation=1,
    )
    slalom_names = [
        "pillar_l1_left", "pillar_l1_right", "pillar_l2_center",
        "pillar_l3_left", "pillar_l3_right", "pillar_back",
    ]
    for name in slalom_names:
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid != -1, f"Missing geom {name} in model"
        assert gid in sim.obstacle_geom_ids, f"{name} (gid={gid}) not in sim.obstacle_geom_ids"

    # Test pillar_l2_center (pos=[0.80, 0.08, 0.15], radius=0.07)
    l2_gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l2_center")
    l2_pos = sim.model.geom_pos[l2_gid, :2].copy()
    l2_radius = sim.model.geom_size[l2_gid, 0]
    assert pytest.approx(l2_radius, abs=1e-4) == 0.07

    # Position robot 0.20m away from pillar_l2_center center along X
    # Move all other obstacles far away to isolate pillar_l2_center
    for gid in sim.obstacle_geom_ids:
        if gid != l2_gid:
            sim.model.geom_pos[gid, 0] = 50.0
            sim.model.geom_pos[gid, 1] = 50.0

    sim.datas[0].qpos[0] = l2_pos[0] + 0.20
    sim.datas[0].qpos[1] = l2_pos[1]

    d_min_single = sim._compute_d_min(env_idx=0)
    # Expected: 0.20 - 0.07 = 0.13
    assert pytest.approx(d_min_single, abs=1e-4) == 0.13

    # Test batched version
    d_min_batch = sim._compute_d_min()
    assert pytest.approx(float(d_min_batch[0]), abs=1e-4) == 0.13

    # Inside cylinder: distance to center 0.03 < radius 0.07 -> should clip to 0.0
    sim.datas[0].qpos[0] = l2_pos[0] + 0.03
    sim.datas[0].qpos[1] = l2_pos[1]
    assert pytest.approx(sim._compute_d_min(env_idx=0), abs=1e-4) == 0.0



