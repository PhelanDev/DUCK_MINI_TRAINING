"""Tests for Lying Locomotion Subsystem and Adaptive Fall Detection (Task 1).

Covers:
- LocomotionMode IntEnum (WALK = 0, SIT = 1, LIE = 2)
- LIE_NOMINAL_QPOS and TARGET_LIE_HEIGHT constants
- MicroDuck8DoFConfig.REWARD_WEIGHTS["lying_pose"]
- compute_rewards behavior under WALK vs LIE modes:
  (r_lie reward, standstill_penalty suppression, velocity error elimination)
- Batched compute_rewards with mixed locomotion modes (WALK, SIT, LIE)
- ParallelSimRunner adaptive fall detection:
  z < 0.080 for WALK, z < 0.050 for SIT, z < 0.025 for LIE.
  tilt_thresh: grav_z > -0.707 for WALK/SIT, grav_z > 0.0 for LIE.
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
try:
    from training.env_config import LIE_NOMINAL_QPOS, TARGET_LIE_HEIGHT
except ImportError:
    LIE_NOMINAL_QPOS = None
    TARGET_LIE_HEIGHT = None

from training.run_parallel_sim import ParallelSimRunner


def test_locomotion_mode_enum_and_constants():
    """Verify LocomotionMode enum, LIE_NOMINAL_QPOS, and TARGET_LIE_HEIGHT."""
    assert hasattr(LocomotionMode, "WALK")
    assert hasattr(LocomotionMode, "SIT")
    assert hasattr(LocomotionMode, "LIE")
    assert LocomotionMode.WALK == 0
    assert LocomotionMode.SIT == 1
    assert LocomotionMode.LIE == 2

    assert TARGET_LIE_HEIGHT == 0.040

    # LIE_NOMINAL_QPOS: 8-DOF target joint angles (float32)
    assert isinstance(LIE_NOMINAL_QPOS, np.ndarray)
    assert LIE_NOMINAL_QPOS.shape == (8,)
    assert LIE_NOMINAL_QPOS.dtype == np.float32

    expected_qpos = np.array(
        [0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05],
        dtype=np.float32,
    )
    np.testing.assert_allclose(LIE_NOMINAL_QPOS, expected_qpos)

    # REWARD_WEIGHTS must contain lying_pose weight = 2.0
    cfg = MicroDuck8DoFConfig()
    assert "lying_pose" in cfg.REWARD_WEIGHTS
    assert cfg.REWARD_WEIGHTS["lying_pose"] == 2.0


def test_compute_rewards_lying_mode_active():
    """Verify compute_rewards gives full r_lie, zeros standstill penalty and vel error when LIE."""
    base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.25, 0.0, 0.0], dtype=np.float32)  # Commanded to walk
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = 0.0
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # 1. Perfect lying posture and height
    joint_pos_lie = np.array([0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05], dtype=np.float32)
    root_z_lie = 0.040

    res_lie = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=joint_pos_lie,
        root_z=root_z_lie,
        cmd_mode=LocomotionMode.LIE if hasattr(LocomotionMode, "LIE") else 2,
    )

    assert "r_lie" in res_lie or "lying_pose" in res_lie
    lie_val = res_lie.get("r_lie", res_lie.get("lying_pose"))
    assert np.isclose(lie_val, 1.0)
    # Standstill penalty must be 0.0 even though cmd_lin_vel[0] > 0.05 and vel_x < 0.05
    assert np.isclose(res_lie["standstill_penalty"], 0.0)
    # Forward lin vel tracking must be 0.0 (no velocity error penalty)
    assert np.isclose(res_lie["tracking_lin_vel"], 0.0)

    # 2. Deviated lying posture and height
    # pose error: ||q_legs - q_lie_legs||^2 = 0.20 -> r_lie_pose = exp(-1.0)
    # height error: (root_z - 0.040)^2 = 0.003 -> r_lie_height = exp(-1.0)
    deviated_joint_pos = joint_pos_lie.copy()
    deviated_joint_pos[3] += np.sqrt(0.20)
    deviated_root_z = 0.040 + np.sqrt(0.003)

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
        cmd_mode=2,  # int mode
    )
    lie_dev_val = res_deviated.get("r_lie", res_deviated.get("lying_pose"))
    expected_dev = float(np.exp(-1.0))
    assert np.isclose(lie_dev_val, expected_dev, atol=1e-4)


def test_compute_rewards_walk_mode_no_lie_reward():
    """Verify compute_rewards does not award r_lie and retains standstill penalty when WALK."""
    base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.25, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = 0.0
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    joint_pos_lie = np.array([0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05], dtype=np.float32)

    # Robot is placed in lying posture and height, but command mode is WALK (0)
    res_walk = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=joint_pos_lie,
        root_z=0.040,
        cmd_mode=LocomotionMode.WALK,
    )

    lie_val = res_walk.get("r_lie", res_walk.get("lying_pose", 0.0))
    assert np.isclose(lie_val, 0.0)
    # Standstill penalty must be applied (-0.5) because robot is commanded to walk but stopped
    cfg = MicroDuck8DoFConfig()
    expected_penalty = cfg.REWARD_WEIGHTS.get("standstill_penalty", -0.5)
    assert np.isclose(res_walk["standstill_penalty"], expected_penalty)


def test_compute_rewards_batched_modes():
    """Verify compute_rewards vectorized calculation with mixed locomotion modes (WALK, SIT, LIE)."""
    batch_size = 3
    base_lin_vel = np.zeros((batch_size, 3), dtype=np.float32)
    cmd_lin_vel = np.array([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    base_ang_vel = np.zeros((batch_size, 3), dtype=np.float32)
    cmd_ang_vel = np.zeros(batch_size, dtype=np.float32)
    proj_gravity = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    joint_torques = np.zeros((batch_size, 8), dtype=np.float32)
    joint_vel = np.zeros((batch_size, 8), dtype=np.float32)
    foot_contact = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    joint_pos_lie = np.array([0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05], dtype=np.float32)
    joint_pos = np.tile(joint_pos_lie, (batch_size, 1))
    root_z = np.array([0.040, TARGET_SIT_HEIGHT, 0.040], dtype=np.float32)

    # env 0 is WALK (0), env 1 is SIT (1), env 2 is LIE (2)
    cmd_modes = np.array([0, 1, 2], dtype=np.int32)

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

    r_lie = res_batch.get("r_lie", res_batch.get("lying_pose"))
    assert np.isclose(r_lie[0], 0.0)
    assert np.isclose(r_lie[1], 0.0)
    assert np.isclose(r_lie[2], 1.0)

    # env 0 has standstill penalty, env 1 and 2 have zero standstill penalty
    cfg = MicroDuck8DoFConfig()
    standstill_w = cfg.REWARD_WEIGHTS.get("standstill_penalty", -0.5)
    assert np.isclose(res_batch["standstill_penalty"][0], standstill_w)
    assert np.isclose(res_batch["standstill_penalty"][1], 0.0)
    assert np.isclose(res_batch["standstill_penalty"][2], 0.0)


def test_parallel_sim_runner_adaptive_fall_detection_lie():
    """Verify ParallelSimRunner adaptive fall threshold: 0.080 for WALK, 0.025 for LIE."""
    sim = ParallelSimRunner(num_envs=2, decimation=1)
    assert hasattr(sim, "cmd_modes")
    assert hasattr(sim, "set_cmd_modes")

    # Configure env 0 as WALK (0) and env 1 as LIE (2)
    sim.set_cmd_modes([0, 2])
    assert sim.cmd_modes[0] == 0
    assert sim.cmd_modes[1] == 2

    lie_nominal = np.array([0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05], dtype=np.float32)

    # Place both robots at height z = 0.040m (upright quaternion, nominal lying joint angles)
    for env_idx in range(2):
        d = sim.datas[env_idx]
        d.qpos[0:3] = [0.0, 0.0, 0.040]  # z = 0.040m
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright quaternion
        d.qpos[7:15] = lie_nominal
        d.qvel[:] = 0.0
        import mujoco
        mujoco.mj_forward(sim.model, d)

    # Step with zero actions
    actions = np.zeros((2, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)

    # Env 0 (WALK) at z=0.040m < 0.080m: MUST FALL and receive fall_penalty
    assert dones[0] == True
    assert infos[0]["fallen"] == True

    # Env 1 (LIE) at z=0.040m >= 0.025m: MUST NOT FALL
    assert dones[1] == False
    assert infos[1]["fallen"] == False


def test_parallel_sim_runner_lie_mode_very_low_fall():
    """Verify that even in LIE mode, dropping below 0.025m triggers a fall."""
    sim = ParallelSimRunner(num_envs=1, decimation=1)
    sim.set_cmd_modes(2)

    lie_nominal = np.array([0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05], dtype=np.float32)

    d = sim.datas[0]
    d.qpos[0:3] = [0.0, 0.0, 0.020]  # z = 0.020m < 0.025m
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:15] = lie_nominal
    d.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(sim.model, d)

    actions = np.zeros((1, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)

    assert dones[0] == True
    assert infos[0]["fallen"] == True


def test_parallel_sim_runner_lie_mode_tilt_fall():
    """Verify that in LIE mode, robot only falls when upside down (grav_z > 0.0)."""
    sim = ParallelSimRunner(num_envs=2, decimation=1)
    # env 0 is WALK (0), env 1 is LIE (2)
    sim.set_cmd_modes([0, 2])

    lie_nominal = np.array([0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05], dtype=np.float32)

    # Rotate 60 degrees around Y axis:
    # cos(30 deg) = 0.866, sin(30 deg) = 0.5. q = [0.866, 0.0, 0.5, 0.0]
    # In world frame, grav = [0, 0, -1].
    # Body Z will have grav_z = -(2*(w^2 + z^2) - 1) = -(2*(0.866^2) - 1) = -(2*0.75 - 1) = -0.5
    # For WALK: grav_z = -0.5 > -0.707 -> falls!
    # For LIE: grav_z = -0.5 <= 0.0 -> does NOT fall!
    import mujoco
    q_tilt = np.array([np.cos(np.pi / 6), 0.0, np.sin(np.pi / 6), 0.0], dtype=np.float64)

    for env_idx in range(2):
        d = sim.datas[env_idx]
        d.qpos[0:3] = [0.0, 0.0, 0.100]  # height is 0.100m, above all z thresholds
        d.qpos[3:7] = q_tilt
        d.qpos[7:15] = lie_nominal
        d.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, d)

    actions = np.zeros((2, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)

    # WALK: tilted > 45 deg (-0.5 > -0.707) -> falls
    assert dones[0] == True
    assert infos[0]["fallen"] == True

    # LIE: tilted 60 deg but not upside down (-0.5 <= 0.0) -> does NOT fall
    assert dones[1] == False
    assert infos[1]["fallen"] == False

    # Now test upside down (180 deg pitch) for LIE mode: q = [0, 0, 1, 0] -> grav_z = +1.0 > 0.0
    sim_upside = ParallelSimRunner(num_envs=1, decimation=1)
    sim_upside.set_cmd_modes(2)
    d = sim_upside.datas[0]
    d.qpos[0:3] = [0.0, 0.0, 0.100]
    d.qpos[3:7] = [0.0, 0.0, 1.0, 0.0]  # upside down
    d.qpos[7:15] = lie_nominal
    d.qvel[:] = 0.0
    mujoco.mj_forward(sim_upside.model, d)

    actions = np.zeros((1, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim_upside.step(actions)
    assert dones[0] == True
    assert infos[0]["fallen"] == True
