"""Tests for Anti-Cheating & Exploit Hardening (Pass 2).

Validates fixes for 4 critical reward hacking and physical exploits:
1. Cohort success rate in curriculum: next_stage requires success_rate >= 0.70.
2. Joint limit dwelling penalty: qpos near bounds (< 5% margin) penalized.
3. Runner get_success_rate calculation.
4. Domain randomization on floor friction in [1.1, 1.6].
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import mujoco

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    JOINT_LIMITS,
    compute_rewards,
)
from training.terrain_curriculum import next_stage, next_stair_stage
from training.run_parallel_sim import ParallelSimRunner


def test_curriculum_requires_cohort_success_rate():
    """Verify that next_stage and next_stair_stage require cohort success_rate >= 0.70."""
    stage = 0
    hold = 0

    # Even with high ep_len (450) and high mean_distance_x (2.0m),
    # if success_rate is 0.50 (< 0.70), curriculum MUST NOT accumulate hold or advance!
    for _ in range(15):
        stage, hold = next_stage(
            stage=stage,
            ep_len_mean=450.0,
            rew_lin_vel_ratio=0.85,
            hold_count=hold,
            hold_iters=10,
            mean_distance_x=2.0,
            success_rate=0.50,
        )
        assert stage == 0, "Curriculum must not advance when cohort success_rate < 0.70"
        assert hold == 0, "Hold count must remain 0 when success_rate condition fails"

    # When success_rate is 0.75 (>= 0.70), hold count accumulates and advances after 10 iters
    for _ in range(9):
        stage, hold = next_stage(
            stage=stage,
            ep_len_mean=450.0,
            rew_lin_vel_ratio=0.85,
            hold_count=hold,
            hold_iters=10,
            mean_distance_x=2.0,
            success_rate=0.75,
        )
        assert stage == 0

    assert hold == 9
    stage, hold = next_stage(
        stage=stage,
        ep_len_mean=450.0,
        rew_lin_vel_ratio=0.85,
        hold_count=hold,
        hold_iters=10,
        mean_distance_x=2.0,
        success_rate=0.75,
    )
    assert stage == 1
    assert hold == 0

    # Verify same behavior on next_stair_stage
    stair_stage = 0
    stair_hold = 0
    stair_stage, stair_hold = next_stair_stage(
        stage=stair_stage,
        ep_len_mean=450.0,
        rew_vx_ratio=0.85,
        hold_count=stair_hold,
        hold_iters=10,
        mean_distance_x=2.0,
        success_rate=0.40,
    )
    assert stair_stage == 0
    assert stair_hold == 0


def test_joint_limit_dwelling_penalty():
    """Verify that joint dwelling near physical limits (< 5% margin) is penalized."""
    cmd_lin_vel = np.array([0.15, 0.0, 0.0], dtype=np.float32)
    base_lin_vel = np.array([0.15, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # 1. Nominal standing joints: knee = 0.15 (safe from 0.0 and 1.48)
    nominal_qpos = np.copy(_DEFAULT_CFG.nominal_qpos)
    rew_nominal = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=nominal_qpos,
    )
    assert "joint_limit_penalty" in rew_nominal, "joint_limit_penalty must be present in reward dict"
    assert rew_nominal["joint_limit_penalty"] == 0.0, "Nominal posture must incur zero limit penalty"

    # 2. Joint nearing upper physical limit: left_knee_pitch = 1.47 (limit is 1.48, 5% margin is 0.074)
    extreme_qpos = np.copy(_DEFAULT_CFG.nominal_qpos)
    extreme_qpos[4] = 1.47  # Left knee near physical upper stop
    rew_extreme = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=extreme_qpos,
    )
    assert rew_extreme["joint_limit_penalty"] < 0.0, "Dwelling near joint limit must produce negative penalty"
    assert rew_extreme["total_reward"] < rew_nominal["total_reward"], "Total reward must be reduced by joint limit penalty"

    # 3. Batch evaluation
    batch_qpos = np.stack([nominal_qpos, extreme_qpos], axis=0)
    batch_base_lin = np.tile(base_lin_vel, (2, 1))
    batch_cmd_lin = np.tile(cmd_lin_vel, (2, 1))
    batch_base_ang = np.tile(base_ang_vel, (2, 1))
    batch_grav = np.tile(proj_gravity, (2, 1))
    batch_torques = np.tile(joint_torques, (2, 1))
    batch_vel = np.tile(joint_vel, (2, 1))
    batch_contact = np.tile(foot_contact, (2, 1))

    rew_batch = compute_rewards(
        base_lin_vel=batch_base_lin,
        cmd_lin_vel=batch_cmd_lin,
        base_ang_vel=batch_base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=batch_grav,
        joint_torques=batch_torques,
        joint_vel=batch_vel,
        foot_contact=batch_contact,
        joint_pos=batch_qpos,
    )
    assert np.isclose(rew_batch["joint_limit_penalty"][0], 0.0)
    assert rew_batch["joint_limit_penalty"][1] < 0.0


def test_runner_get_success_rate():
    """Verify that ParallelSimRunner.get_success_rate calculates the cohort success rate accurately."""
    runner = ParallelSimRunner(num_envs=4, terrain_mode="flat")
    try:
        runner.reset()

        # Populate completed_distances_x: 3 successes (>= 1.5m), 1 failure (< 1.5m)
        runner.completed_distances_x.clear()
        runner.completed_distances_x.extend([1.6, 2.1, 0.8, 1.55])
        rate = runner.get_success_rate(min_dist=1.5)
        assert pytest.approx(rate, abs=1e-4) == 0.75, "Expected 3/4 = 0.75 success rate"

        # All successes
        runner.completed_distances_x.clear()
        runner.completed_distances_x.extend([1.5, 1.8, 2.2])
        assert pytest.approx(runner.get_success_rate(1.5), abs=1e-4) == 1.0

        # All failures
        runner.completed_distances_x.clear()
        runner.completed_distances_x.extend([0.2, 0.4, 1.49])
        assert pytest.approx(runner.get_success_rate(1.5), abs=1e-4) == 0.0
    finally:
        runner.close()


def test_domain_randomization_friction():
    """Verify that reset_env applies domain randomization on floor friction in [1.1, 1.6]."""
    runner = ParallelSimRunner(num_envs=1, terrain_mode="flat")
    try:
        frictions = set()
        for _ in range(20):
            runner.reset_env(0)
            floor_gid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            assert floor_gid != -1, "Floor geom must exist"
            mu = float(runner.model.geom_friction[floor_gid, 0])
            assert 1.1 <= mu <= 1.6, f"Friction {mu} out of expected range [1.1, 1.6]"
            frictions.add(round(mu, 3))

        # With 20 resets, we expect multiple distinct values
        assert len(frictions) > 5, "Floor friction should vary across resets due to domain randomization"
    finally:
        runner.close()
