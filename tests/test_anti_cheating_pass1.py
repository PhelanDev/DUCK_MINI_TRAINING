"""Tests for Anti-Cheating & Exploit Hardening (Pass 1).

Validates fixes for 4 critical reward hacking and physical exploits:
1. Strict tilt termination (grav_z > -0.707 triggers fallen=True).
2. Trampling / marching-in-place exploit guard (feet_air_time reward = 0.0 when v_x < 0.05).
3. Speed diving clip on tracking_lin_vel (effective v_x capped at cmd_v_x + 0.05).
4. Slalom corridor bypass check (|y| > 0.55m in x in [0.2, 1.2]m during hurdles mode triggers corridor_fallen=True).
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import mujoco

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import compute_rewards
from training.run_parallel_sim import ParallelSimRunner


def test_strict_tilt_termination():
    """Verify that a 50-degree tilt (grav_z = -0.6428 > -0.707) terminates episode as fallen."""
    runner = ParallelSimRunner(
        num_envs=1,
        terrain_mode="flat",
    )
    try:
        runner.reset()
        # Ensure robot is at standing height (root_z = 0.20 >= 0.08)
        runner.datas[0].qpos[2] = 0.20

        # Rotate robot 50 degrees around pitch (Y-axis)
        # Quaternion: [cos(theta/2), 0, sin(theta/2), 0]
        # For 50 deg: cos(25 deg) ~ 0.9063078, sin(25 deg) ~ 0.4226183
        # Resulting grav_z: 2*(x^2 + y^2) - 1 = 2*(sin^2(25 deg)) - 1 = -cos(50 deg) ~ -0.642788
        # Since -0.6428 > -0.707, strict tilt threshold MUST trigger fallen!
        theta = np.deg2rad(50.0)
        w = np.cos(theta / 2.0)
        y = np.sin(theta / 2.0)
        runner.datas[0].qpos[3:7] = np.array([w, 0.0, y, 0.0], dtype=np.float32)

        # Step physics one control step with zero action
        mujoco.mj_forward(runner.model, runner.datas[0])
        actions = np.zeros((1, 8), dtype=np.float32)
        _, _, dones, infos = runner.step(actions)

        assert bool(infos[0]["fallen"]) is True, "Expected robot tilted 50 deg to be marked as fallen"
        assert bool(dones[0]) is True, "Expected episode to terminate when tilted 50 deg"
    finally:
        runner.close()


def test_trampling_in_place_no_feet_air_reward():
    """Verify that feet_air_time reward is 0.0 when robot tramples in place (v_x < 0.05 m/s)."""
    cmd_lin_vel = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Foot 0 had 0.20s swing air time and just touched down
    feet_air_time = np.array([0.20, 0.0], dtype=np.float32)
    feet_touchdown = np.array([True, False], dtype=bool)

    # 1. Trampling in-place: v_x = 0.01 m/s (< 0.05 m/s) -> MUST receive 0.0 feet_air_time reward
    vel_trample = np.array([0.01, 0.0, 0.0], dtype=np.float32)
    rew_trample = compute_rewards(
        base_lin_vel=vel_trample,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        feet_air_time=feet_air_time,
        feet_touchdown=feet_touchdown,
    )
    assert rew_trample["feet_air_time"] == 0.0, "Trampling in-place must receive 0 feet_air_time reward"

    # 2. Moving forward: v_x = 0.15 m/s (>= 0.05 m/s) -> receives valid feet_air_time reward
    vel_walk = np.array([0.15, 0.0, 0.0], dtype=np.float32)
    rew_walk = compute_rewards(
        base_lin_vel=vel_walk,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        feet_air_time=feet_air_time,
        feet_touchdown=feet_touchdown,
    )
    assert rew_walk["feet_air_time"] > 0.0, "Forward walking must receive positive feet_air_time reward"
    w_feet_air = 2.0
    assert pytest.approx(rew_walk["feet_air_time"], abs=1e-5) == w_feet_air * (0.20 - 0.08)

    # 3. Batched verification: env 0 trampling, env 1 walking
    batch_base_lin = np.array([[0.01, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=np.float32)
    batch_cmd_lin = np.array([[0.20, 0.0, 0.0], [0.20, 0.0, 0.0]], dtype=np.float32)
    batch_air_time = np.array([[0.20, 0.0], [0.20, 0.0]], dtype=np.float32)
    batch_touchdown = np.array([[True, False], [True, False]], dtype=bool)
    rew_batch = compute_rewards(
        base_lin_vel=batch_base_lin,
        cmd_lin_vel=batch_cmd_lin,
        base_ang_vel=np.zeros((2, 3), dtype=np.float32),
        cmd_ang_vel=np.zeros(2, dtype=np.float32),
        proj_gravity=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        joint_torques=np.zeros((2, 8), dtype=np.float32),
        joint_vel=np.zeros((2, 8), dtype=np.float32),
        foot_contact=np.tile(foot_contact, (2, 1)),
        feet_air_time=batch_air_time,
        feet_touchdown=batch_touchdown,
    )
    assert rew_batch["feet_air_time"][0] == 0.0
    assert rew_batch["feet_air_time"][1] > 0.0


def test_speed_diving_clipped():
    """Verify that forward velocity tracking reward is clipped at cmd_v_x + 0.05."""
    cmd_lin_vel = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.ones(4, dtype=np.float32)

    # 1. Robot speed diving at v_x = 0.80 m/s when commanded 0.20 m/s
    vel_diving = np.array([0.80, 0.0, 0.0], dtype=np.float32)
    rew_diving = compute_rewards(
        base_lin_vel=vel_diving,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
    )

    # 2. Robot moving exactly at cap v_x = 0.25 m/s (cmd_x + 0.05)
    vel_capped = np.array([0.25, 0.0, 0.0], dtype=np.float32)
    rew_capped = compute_rewards(
        base_lin_vel=vel_capped,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
    )

    # Tracking reward for speed diving MUST NOT exceed reward at 0.25 m/s
    assert pytest.approx(rew_diving["tracking_lin_vel"], abs=1e-5) == rew_capped["tracking_lin_vel"]
    assert pytest.approx(rew_diving["tracking_lin_vel"], abs=1e-5) == 2.0 * 0.25

    # 3. Batched verification
    batch_base_lin = np.array([[0.80, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32)
    batch_cmd_lin = np.array([[0.20, 0.0, 0.0], [0.20, 0.0, 0.0]], dtype=np.float32)
    rew_batch = compute_rewards(
        base_lin_vel=batch_base_lin,
        cmd_lin_vel=batch_cmd_lin,
        base_ang_vel=np.zeros((2, 3), dtype=np.float32),
        cmd_ang_vel=np.zeros(2, dtype=np.float32),
        proj_gravity=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        joint_torques=np.zeros((2, 8), dtype=np.float32),
        joint_vel=np.zeros((2, 8), dtype=np.float32),
        foot_contact=np.tile(foot_contact, (2, 1)),
    )
    assert pytest.approx(rew_batch["tracking_lin_vel"][0], abs=1e-5) == rew_batch["tracking_lin_vel"][1]


def test_slalom_corridor_bypass_penalty():
    """Verify that exiting the slalom corridor (|y| > 0.55m) in hurdles mode triggers corridor_fallen and penalty."""
    runner = ParallelSimRunner(
        num_envs=2,
        terrain_mode="hurdles",
    )
    try:
        runner.reset()
        # Env 0: In pillar zone (x = 0.60m) but bypassed corridor (y = 0.60m > 0.55m)
        runner.datas[0].qpos[0] = 0.60
        runner.datas[0].qpos[1] = 0.60
        runner.datas[0].qpos[2] = 0.20

        # Env 1: In pillar zone (x = 0.60m) and inside corridor (y = 0.20m <= 0.55m)
        runner.datas[1].qpos[0] = 0.60
        runner.datas[1].qpos[1] = 0.20
        runner.datas[1].qpos[2] = 0.20

        mujoco.mj_forward(runner.model, runner.datas[0])
        mujoco.mj_forward(runner.model, runner.datas[1])

        actions = np.zeros((2, 8), dtype=np.float32)
        _, rewards, dones, infos = runner.step(actions)

        # Env 0 must be corridor_fallen
        assert "corridor_fallen" in infos[0], "Expected 'corridor_fallen' key in infos"
        assert bool(infos[0]["corridor_fallen"]) is True
        assert bool(infos[0]["fallen"]) is True
        assert bool(dones[0]) is True

        # Env 1 must NOT be corridor_fallen
        assert bool(infos[1]["corridor_fallen"]) is False
        assert bool(infos[1]["fallen"]) is False
        assert bool(dones[1]) is False

        # Env 0 must receive fall penalty (-5.0) relative to non-fallen baseline
        fall_penalty = float(runner.cfg.REWARD_WEIGHTS.get("fall_penalty", -5.0))
        assert rewards[0] < rewards[1] + fall_penalty + 1.0

        # Outside pillar zone (x = 0.10m < 0.20m), y = 0.60m should NOT trigger corridor_fallen
        runner.datas[0].qpos[0] = 0.10
        runner.datas[0].qpos[1] = 0.60
        runner.datas[0].qpos[2] = 0.20
        mujoco.mj_forward(runner.model, runner.datas[0])
        _, _, _, infos_outside = runner.step(actions)
        assert bool(infos_outside[0]["corridor_fallen"]) is False
    finally:
        runner.close()
