import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import (
    MicroDuck8DoFConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
    compute_rewards,
    quat_rotate_inverse,
    _DEFAULT_CFG,
)


def test_env_config_dimensions():
    cfg = MicroDuck8DoFConfig()
    assert cfg.OBS_DIM == 37
    assert cfg.ACTION_DIM == 8
    assert len(cfg.JOINT_NAMES) == 8
    assert cfg.TORQUE_LIMIT == 1.0
    assert cfg.MAX_VELOCITY == 6.0

    expected_joints = [
        "head_yaw",
        "head_pitch",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee_pitch",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee_pitch",
    ]
    assert cfg.JOINT_NAMES == expected_joints

    # Verify joint limits are present for all joints
    for name in expected_joints:
        assert name in cfg.JOINT_LIMITS
        low, high = cfg.JOINT_LIMITS[name]
        assert low < high


def test_observation_packing():
    base_ang_vel = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    commands = np.array([0.5, 0.0, 0.1], dtype=np.float32)
    joint_pos = np.arange(8, dtype=np.float32) * 0.1
    joint_vel = np.arange(8, dtype=np.float32) * -0.05
    last_actions = np.ones(8, dtype=np.float32) * 0.2
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    obs = compute_observations(
        base_ang_vel=base_ang_vel,
        proj_grav=proj_grav,
        commands=commands,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        last_actions=last_actions,
        foot_contact=foot_contact,
    )

    assert obs.shape == (37,)
    assert obs.dtype == np.float32

    # Verify component slices match inputs
    np.testing.assert_allclose(obs[0:3], base_ang_vel)
    np.testing.assert_allclose(obs[3:6], proj_grav)
    np.testing.assert_allclose(obs[6:9], commands)
    np.testing.assert_allclose(obs[9:17], joint_pos)
    np.testing.assert_allclose(obs[17:25], joint_vel)
    np.testing.assert_allclose(obs[25:33], last_actions)
    np.testing.assert_allclose(obs[33:37], foot_contact)


def test_observation_defensive_shape_check():
    # Provide wrong size joint positions (7 instead of 8)
    with pytest.raises(AssertionError, match="Expected observation shape \\(37,\\)"):
        compute_observations(
            base_ang_vel=np.zeros(3),
            proj_grav=np.zeros(3),
            commands=np.zeros(3),
            joint_pos=np.zeros(7),  # Wrong size!
            joint_vel=np.zeros(8),
            last_actions=np.zeros(8),
            foot_contact=np.zeros(4),
        )


def test_action_scaling():
    cfg = MicroDuck8DoFConfig()

    # -1 should map to low limits, +1 to high limits, 0 to midpoints
    raw_min = -np.ones(8, dtype=np.float32)
    scaled_min = scale_actions(raw_min, cfg)
    raw_max = np.ones(8, dtype=np.float32)
    scaled_max = scale_actions(raw_max, cfg)
    raw_mid = np.zeros(8, dtype=np.float32)
    scaled_mid = scale_actions(raw_mid, cfg)

    for i, name in enumerate(cfg.JOINT_NAMES):
        low, high = cfg.JOINT_LIMITS[name]
        assert np.isclose(scaled_min[i], low)
        assert np.isclose(scaled_max[i], high)
        assert np.isclose(scaled_mid[i], 0.5 * (low + high))
        assert low <= scaled_mid[i] <= high

    # Out of range actions should be safely clipped to physical limits
    raw_overflow = np.array([-2.0, 3.5, 0.0, -10.0, 5.0, -1.5, 1.5, 0.0])
    scaled_overflow = scale_actions(raw_overflow, cfg)
    for i, name in enumerate(cfg.JOINT_NAMES):
        low, high = cfg.JOINT_LIMITS[name]
        assert low <= scaled_overflow[i] <= high

    # Default cached configuration test (no cfg passed)
    scaled_default = scale_actions(raw_min)
    np.testing.assert_allclose(scaled_default, _DEFAULT_CFG.low_limits)


def test_clip_action_rates():
    current_targets = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    last_targets = np.zeros(8, dtype=np.float32)
    dt = 0.02
    max_velocity = 6.0
    max_delta = max_velocity * dt  # 0.12 rad per step

    clipped = clip_action_rates(current_targets, last_targets, dt=dt, max_velocity=max_velocity)
    delta = clipped - last_targets
    assert np.all(np.abs(delta) <= max_delta + 1e-6)
    np.testing.assert_allclose(clipped, np.ones(8) * max_delta)

    # Within velocity limit should not be clipped
    small_step = last_targets + 0.05
    clipped_small = clip_action_rates(small_step, last_targets, dt=dt, max_velocity=max_velocity)
    np.testing.assert_allclose(clipped_small, small_step)


def test_reward_computation():
    base_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cmd_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    rewards = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
    )

    assert isinstance(rewards, dict)
    assert "total_reward" in rewards
    assert not np.isnan(rewards["total_reward"])
    assert "tracking_lin_vel" in rewards
    assert "tracking_ang_vel" in rewards
    assert "upright" in rewards
    assert "torque_penalty" in rewards
    assert "joint_vel_penalty" in rewards
    assert "action_rate_penalty" in rewards
    assert rewards["action_rate_penalty"] == 0.0

    # When tracking is perfect, upright is vertical, torques are 0, reward should be high
    perfect_total = rewards["total_reward"]

    # When robot falls (gravity horizontal), reward should decrease
    bad_rewards = compute_rewards(
        base_lin_vel=np.zeros(3),
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=np.array([1.0, 0.0, 0.0]),  # on its side
        joint_torques=np.ones(8) * 1.0,  # high torque
        joint_vel=np.ones(8) * 5.0,  # high vel
        foot_contact=np.zeros(4),
    )
    assert bad_rewards["total_reward"] < perfect_total


def test_reward_action_rate_penalty():
    base_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.ones(4, dtype=np.float32)

    # Identical actions -> penalty is 0.0
    actions_same = np.ones(8, dtype=np.float32) * 0.5
    last_actions_same = np.ones(8, dtype=np.float32) * 0.5
    r_same = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        actions=actions_same,
        last_actions=last_actions_same,
    )
    assert r_same["action_rate_penalty"] == 0.0

    # Large action shift -> non-zero penalty reduces total reward
    actions_diff = np.ones(8, dtype=np.float32) * 1.0
    last_actions_diff = -np.ones(8, dtype=np.float32) * 1.0
    # diff = 2.0 each, diff^2 = 4.0 * 8 = 32.0
    r_diff = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        actions=actions_diff,
        last_actions=last_actions_diff,
    )
    assert np.isclose(r_diff["action_rate_penalty"], 32.0)
    assert r_diff["total_reward"] < r_same["total_reward"]
    # Verify expected math: difference is -0.01 * 32.0 = -0.32
    assert np.isclose(r_same["total_reward"] - r_diff["total_reward"], 0.32)


def test_quat_rotate_inverse_frame_transformation():
    """Verify transforming world frame velocity to body frame across orientations."""
    # 1. Identity quaternion -> no rotation
    q_ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_world = np.array([1.0, 0.5, -0.2], dtype=np.float32)
    np.testing.assert_allclose(quat_rotate_inverse(q_ident, v_world), v_world, atol=1e-6)

    # 2. Robot yawed 90 degrees (+Z axis): facing +Y in world frame
    # When moving along +Y in world frame, velocity in robot body frame must be forward (+X)
    q_yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=np.float32)
    v_world_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_body = quat_rotate_inverse(q_yaw90, v_world_y)
    np.testing.assert_allclose(v_body, [1.0, 0.0, 0.0], atol=1e-6)

    # 3. Robot yawed 180 degrees (+Z axis): facing -X in world frame
    # When moving along -X in world frame, velocity in robot body frame must be forward (+X)
    q_yaw180 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    v_world_nx = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    v_body_180 = quat_rotate_inverse(q_yaw180, v_world_nx)
    np.testing.assert_allclose(v_body_180, [1.0, 0.0, 0.0], atol=1e-6)

    # 4. Batched orientation test (shape (N, 3))
    q_batch = np.stack([q_ident, q_yaw90, q_yaw180])
    v_batch = np.stack([v_world, v_world_y, v_world_nx])
    v_batch_body = quat_rotate_inverse(q_batch, v_batch)
    assert v_batch_body.shape == (3, 3)
    np.testing.assert_allclose(v_batch_body[1], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(v_batch_body[2], [1.0, 0.0, 0.0], atol=1e-6)


def test_batched_compute_rewards_equivalence():
    """Verify batched compute_rewards produces identical values to single-env loop."""
    num_envs = 5
    base_lin = np.random.uniform(-1.0, 1.0, size=(num_envs, 3)).astype(np.float32)
    cmd_lin = np.random.uniform(-1.0, 1.0, size=(num_envs, 3)).astype(np.float32)
    base_ang = np.random.uniform(-1.0, 1.0, size=(num_envs, 3)).astype(np.float32)
    cmd_ang = np.random.uniform(-1.0, 1.0, size=(num_envs,)).astype(np.float32)
    grav = np.random.uniform(-1.0, 1.0, size=(num_envs, 3)).astype(np.float32)
    torques = np.random.uniform(-1.0, 1.0, size=(num_envs, 8)).astype(np.float32)
    j_vel = np.random.uniform(-1.0, 1.0, size=(num_envs, 8)).astype(np.float32)
    f_contact = np.random.uniform(0.0, 1.0, size=(num_envs, 4)).astype(np.float32)
    actions = np.random.uniform(-1.0, 1.0, size=(num_envs, 8)).astype(np.float32)
    last_act = np.random.uniform(-1.0, 1.0, size=(num_envs, 8)).astype(np.float32)

    # Batched execution
    batch_rewards = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=cmd_ang,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        actions=actions,
        last_actions=last_act,
    )

    # Sequential execution
    for i in range(num_envs):
        single_r = compute_rewards(
            base_lin_vel=base_lin[i],
            cmd_lin_vel=cmd_lin[i],
            base_ang_vel=base_ang[i],
            cmd_ang_vel=cmd_ang[i],
            proj_gravity=grav[i],
            joint_torques=torques[i],
            joint_vel=j_vel[i],
            foot_contact=f_contact[i],
            actions=actions[i],
            last_actions=last_act[i],
        )
        assert np.isclose(batch_rewards["total_reward"][i], single_r["total_reward"], atol=1e-5)
        assert np.isclose(batch_rewards["tracking_lin_vel"][i], single_r["tracking_lin_vel"], atol=1e-5)
        assert np.isclose(batch_rewards["upright"][i], single_r["upright"], atol=1e-5)


def test_pd_gains_config():
    """Verify KP and KD are standardized in MicroDuck8DoFConfig."""
    cfg = MicroDuck8DoFConfig()
    assert hasattr(cfg, "KP") and cfg.KP == 25.0
    assert hasattr(cfg, "KD") and cfg.KD == 0.8
