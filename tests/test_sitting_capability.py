"""Tests for Sitting Locomotion Subsystem and Adaptive Fall Detection (Task 2).

Covers:
- LocomotionMode IntEnum (WALK = 0, SIT = 1)
- SIT_NOMINAL_QPOS and TARGET_SIT_HEIGHT constants
- MicroDuck8DoFConfig.REWARD_WEIGHTS["sitting_pose"]
- compute_rewards behavior under WALK vs SIT modes
  (r_sit reward, standstill_penalty suppression, velocity error elimination)
- Batched compute_rewards with mixed locomotion modes
- ParallelSimRunner adaptive fall detection:
  z < 0.08 for WALK, z < 0.05 for SIT.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import (
    MicroDuck8DoFConfig,
    LocomotionMode,
    SIT_NOMINAL_QPOS,
    TARGET_SIT_HEIGHT,
    compute_rewards,
)
from training.run_parallel_sim import ParallelSimRunner


def test_locomotion_mode_enum_and_constants():
    """Verify LocomotionMode enum, SIT_NOMINAL_QPOS, and TARGET_SIT_HEIGHT."""
    assert hasattr(LocomotionMode, "WALK")
    assert hasattr(LocomotionMode, "SIT")
    assert LocomotionMode.WALK == 0
    assert LocomotionMode.SIT == 1
    assert len(LocomotionMode) >= 2

    assert TARGET_SIT_HEIGHT == 0.075

    # SIT_NOMINAL_QPOS: 8-DOF target joint angles (float32)
    assert isinstance(SIT_NOMINAL_QPOS, np.ndarray)
    assert SIT_NOMINAL_QPOS.shape == (8,)
    assert SIT_NOMINAL_QPOS.dtype == np.float32

    expected_qpos = np.array(
        [0.0, 0.0, 0.0, 0.70, 1.35, 0.0, 0.70, 1.35],
        dtype=np.float32,
    )
    np.testing.assert_allclose(SIT_NOMINAL_QPOS, expected_qpos)

    # REWARD_WEIGHTS must contain sitting_pose weight = 2.0
    cfg = MicroDuck8DoFConfig()
    assert "sitting_pose" in cfg.REWARD_WEIGHTS
    assert cfg.REWARD_WEIGHTS["sitting_pose"] == 2.0


def test_compute_rewards_sitting_mode_active():
    """Verify compute_rewards gives full r_sit, zeros standstill penalty and vel error when SIT."""
    base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.25, 0.0, 0.0], dtype=np.float32)  # Commanded to walk
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = 0.0
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # 1. Perfect sitting posture and height
    joint_pos_sit = SIT_NOMINAL_QPOS.copy()
    root_z_sit = TARGET_SIT_HEIGHT

    res_sit = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=joint_pos_sit,
        root_z=root_z_sit,
        cmd_mode=LocomotionMode.SIT,
    )

    assert "r_sit" in res_sit or "sitting_pose" in res_sit
    sit_val = res_sit.get("r_sit", res_sit.get("sitting_pose"))
    assert np.isclose(sit_val, 1.0)
    # Standstill penalty must be 0.0 even though cmd_lin_vel[0] > 0.05 and vel_x < 0.05
    assert np.isclose(res_sit["standstill_penalty"], 0.0)
    # Forward lin vel tracking must be 0.0 (no velocity error penalty)
    assert np.isclose(res_sit["tracking_lin_vel"], 0.0)

    # 2. Deviated sitting posture and height
    # pose error: ||q_legs - q_sit_legs||^2 = 0.20 -> r_sit_pose = exp(-1.0)
    # height error: (root_z - 0.075)^2 = 0.005 -> r_sit_height = exp(-1.0)
    deviated_joint_pos = SIT_NOMINAL_QPOS.copy()
    deviated_joint_pos[3] += np.sqrt(0.20)
    deviated_root_z = TARGET_SIT_HEIGHT + np.sqrt(0.005)

    res_deviated = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=deviated_joint_pos,
        root_z=deviated_root_z,
        cmd_mode=1,  # int mode
    )
    sit_dev_val = res_deviated.get("r_sit", res_deviated.get("sitting_pose"))
    expected_dev = float(np.exp(-1.0))
    assert np.isclose(sit_dev_val, expected_dev, atol=1e-4)


def test_compute_rewards_walk_mode_no_sit_reward():
    """Verify compute_rewards does not award r_sit and retains standstill penalty when WALK."""
    base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.25, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = 0.0
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # Robot is placed in sitting posture and height, but command mode is WALK (0)
    res_walk = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=SIT_NOMINAL_QPOS,
        root_z=TARGET_SIT_HEIGHT,
        cmd_mode=LocomotionMode.WALK,
    )

    sit_val = res_walk.get("r_sit", res_walk.get("sitting_pose"))
    assert np.isclose(sit_val, 0.0)
    # Standstill penalty must be applied (-0.5) because robot is commanded to walk but stopped
    cfg = MicroDuck8DoFConfig()
    expected_penalty = cfg.REWARD_WEIGHTS.get("standstill_penalty", -0.5)
    assert np.isclose(res_walk["standstill_penalty"], expected_penalty)


def test_compute_rewards_batched_modes():
    """Verify compute_rewards vectorized calculation with mixed locomotion modes."""
    batch_size = 2
    base_lin_vel = np.zeros((batch_size, 3), dtype=np.float32)
    cmd_lin_vel = np.array([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    base_ang_vel = np.zeros((batch_size, 3), dtype=np.float32)
    cmd_ang_vel = np.zeros(batch_size, dtype=np.float32)
    proj_gravity = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    joint_torques = np.zeros((batch_size, 8), dtype=np.float32)
    joint_vel = np.zeros((batch_size, 8), dtype=np.float32)
    foot_contact = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    joint_pos = np.tile(SIT_NOMINAL_QPOS, (batch_size, 1))
    root_z = np.array([TARGET_SIT_HEIGHT, TARGET_SIT_HEIGHT], dtype=np.float32)

    # env 0 is WALK, env 1 is SIT
    cmd_modes = np.array([0, 1], dtype=np.int32)

    res_batch = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=joint_pos,
        root_z=root_z,
        cmd_mode=cmd_modes,
    )

    r_sit = res_batch.get("r_sit", res_batch.get("sitting_pose"))
    assert np.isclose(r_sit[0], 0.0)
    assert np.isclose(r_sit[1], 1.0)

    # env 0 has standstill penalty, env 1 has zero standstill penalty
    cfg = MicroDuck8DoFConfig()
    standstill_w = cfg.REWARD_WEIGHTS.get("standstill_penalty", -0.5)
    assert np.isclose(res_batch["standstill_penalty"][0], standstill_w)
    assert np.isclose(res_batch["standstill_penalty"][1], 0.0)


def test_parallel_sim_runner_adaptive_fall_detection():
    """Verify ParallelSimRunner adaptive fall threshold (0.08 for WALK vs 0.05 for SIT)."""
    sim = ParallelSimRunner(num_envs=2, decimation=1)
    assert hasattr(sim, "cmd_modes")
    assert hasattr(sim, "set_cmd_modes")
    assert sim.cmd_modes.shape == (2,)

    # Configure env 0 as WALK (0) and env 1 as SIT (1)
    sim.set_cmd_modes([0, 1])
    assert sim.cmd_modes[0] == 0
    assert sim.cmd_modes[1] == 1

    # Place both robots at height z = 0.075m (upright, nominal leg joint angles)
    for env_idx in range(2):
        d = sim.datas[env_idx]
        d.qpos[0:3] = [0.0, 0.0, 0.075]  # z = 0.075m
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright quaternion
        d.qpos[7:15] = SIT_NOMINAL_QPOS
        d.qvel[:] = 0.0
        # Sync forward kinematics
        import mujoco
        mujoco.mj_forward(sim.model, d)

    # Step with zero actions
    actions = np.zeros((2, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)

    # Env 0 (WALK) at z=0.075m < 0.08m: MUST FALL
    assert dones[0] == True
    assert infos[0]["fallen"] == True

    # Env 1 (SIT) at z=0.075m >= 0.05m: MUST NOT FALL
    assert dones[1] == False
    assert infos[1]["fallen"] == False


def test_parallel_sim_runner_sit_mode_very_low_fall():
    """Verify that even in SIT mode, dropping below 0.05m triggers a fall."""
    sim = ParallelSimRunner(num_envs=1, decimation=1)
    sim.set_cmd_modes(LocomotionMode.SIT)

    d = sim.datas[0]
    d.qpos[0:3] = [0.0, 0.0, 0.040]  # z = 0.04m < 0.05m
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:15] = SIT_NOMINAL_QPOS
    d.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(sim.model, d)

    actions = np.zeros((1, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)

    assert dones[0] == True
    assert infos[0]["fallen"] == True
