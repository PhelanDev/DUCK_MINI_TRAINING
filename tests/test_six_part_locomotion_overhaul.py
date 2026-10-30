"""Tests for Six-Part Locomotion Surgical Overhaul.

Covers:
Part 1: Heading yaw extraction (quat_to_yaw) & Body-Frame velocity projection (quat_rotate_inverse)
Part 2: Gait phase clock (compute_gait_phase) & observation embedding (compute_observations with phase_clock)
"""

import os
import sys
import numpy as np
import pytest
import mujoco

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.env_config import (
    quat_to_yaw,
    quat_rotate_inverse,
    compute_gait_phase,
    compute_observations,
    scale_actions,
    WALK_JOINT_LIMITS,
    DEFAULT_NOMINAL_QPOS,
    LocomotionMode,
    compute_rewards,
    MicroDuck8DoFConfig,
)
from training.run_parallel_sim import ParallelSimRunner
from training.terrain_curriculum import next_stage, next_stair_stage
from scripts.interactive_viewer import (
    extract_viewer_observation,
    get_robot_state,
)


class TestPart1YawAndBodyFrameProjection:
    """Part 1: quat_to_yaw and quat_rotate_inverse unit tests."""

    def test_quat_to_yaw_identity(self):
        """Identity quaternion [1, 0, 0, 0] must give yaw = 0.0."""
        q_ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        yaw = quat_to_yaw(q_ident)
        assert yaw == pytest.approx(0.0, abs=1e-5)

    def test_quat_to_yaw_plus_90_deg(self):
        """Rotation +90 deg around Z: q = [cos(pi/4), 0, 0, sin(pi/4)] -> +pi/2."""
        q_90 = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=np.float32)
        yaw = quat_to_yaw(q_90)
        assert yaw == pytest.approx(np.pi / 2.0, abs=1e-5)

    def test_quat_to_yaw_minus_45_deg(self):
        """Rotation -45 deg around Z: q = [cos(-pi/8), 0, 0, sin(-pi/8)] -> -pi/4."""
        q_minus_45 = np.array(
            [np.cos(-np.pi / 8.0), 0.0, 0.0, np.sin(-np.pi / 8.0)], dtype=np.float32
        )
        yaw = quat_to_yaw(q_minus_45)
        assert yaw == pytest.approx(-np.pi / 4.0, abs=1e-5)

    def test_quat_to_yaw_batched(self):
        """Batched quaternion array of shape (N, 4) -> returns shape (N,)."""
        quats = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)],
                [np.cos(-np.pi / 8.0), 0.0, 0.0, np.sin(-np.pi / 8.0)],
            ],
            dtype=np.float32,
        )
        yaws = quat_to_yaw(quats)
        assert yaws.shape == (3,)
        assert yaws[0] == pytest.approx(0.0, abs=1e-5)
        assert yaws[1] == pytest.approx(np.pi / 2.0, abs=1e-5)
        assert yaws[2] == pytest.approx(-np.pi / 4.0, abs=1e-5)

    def test_quat_rotate_inverse_body_frame_velocity_projection(self):
        """Crab-walking test: Duck rotated +90 deg around Z moving along World +X.

        Facing World +Y:
        world_lin_vel = [0.2, 0.0, 0.0]
        Body Frame velocity must be [0.0, -0.2, 0.0] (pure lateral drift to duck's right!).
        This proves crab-walking sideways at 90 deg yaw yields 0.0 forward velocity in Body Frame.
        """
        q = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=np.float32)
        world_lin_vel = np.array([0.2, 0.0, 0.0], dtype=np.float32)

        body_vel = quat_rotate_inverse(q, world_lin_vel)
        assert body_vel[0] == pytest.approx(0.0, abs=1e-5)
        assert body_vel[1] == pytest.approx(-0.2, abs=1e-5)
        assert body_vel[2] == pytest.approx(0.0, abs=1e-5)


class TestPart2GaitPhaseClockAndObservation:
    """Part 2: compute_gait_phase and observation phase_clock embedding unit tests."""

    def test_compute_gait_phase_step_0(self):
        """At step 0: phase == 0.0, clock_sin == 0.0, clock_cos == 1.0."""
        phase, clock_sin, clock_cos = compute_gait_phase(episode_steps=0, dt_control=0.02, period=0.8)
        assert phase == pytest.approx(0.0, abs=1e-5)
        assert clock_sin == pytest.approx(0.0, abs=1e-5)
        assert clock_cos == pytest.approx(1.0, abs=1e-5)

    def test_compute_gait_phase_step_10(self):
        """At step 10: 10 * 0.02 / 0.8 = 0.25 (quarter period) -> sin=1.0, cos=0.0."""
        phase, clock_sin, clock_cos = compute_gait_phase(episode_steps=10, dt_control=0.02, period=0.8)
        assert phase == pytest.approx(0.25, abs=1e-5)
        assert clock_sin == pytest.approx(1.0, abs=1e-5)
        assert clock_cos == pytest.approx(0.0, abs=1e-5)

    def test_compute_gait_phase_step_20(self):
        """At step 20: 20 * 0.02 / 0.8 = 0.50 (half period) -> sin=0.0, cos=-1.0."""
        phase, clock_sin, clock_cos = compute_gait_phase(episode_steps=20, dt_control=0.02, period=0.8)
        assert phase == pytest.approx(0.50, abs=1e-5)
        assert clock_sin == pytest.approx(0.0, abs=1e-5)
        assert clock_cos == pytest.approx(-1.0, abs=1e-5)

    def test_compute_gait_phase_batched(self):
        """Batched input array (N,) -> returns phase, sin, cos of shape (N,)."""
        steps = np.array([0, 10, 20], dtype=np.int32)
        phases, sins, coss = compute_gait_phase(episode_steps=steps, dt_control=0.02, period=0.8)
        assert phases.shape == (3,)
        assert sins.shape == (3,)
        assert coss.shape == (3,)
        np.testing.assert_allclose(phases, [0.0, 0.25, 0.50], atol=1e-5)
        np.testing.assert_allclose(sins, [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(coss, [1.0, 0.0, -1.0], atol=1e-5)

    def test_compute_observations_preserves_default_when_phase_clock_none(self):
        """When phase_clock is None, existing 37-dim observation behavior is 100% preserved."""
        base_ang_vel = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        commands = np.array([0.2, 0.0, 0.0], dtype=np.float32)
        joint_pos = np.zeros(8, dtype=np.float32)
        joint_vel = np.zeros(8, dtype=np.float32)
        last_actions = np.zeros(8, dtype=np.float32)
        foot_contact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        obs = compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_actions,
            foot_contact=foot_contact,
            phase_clock=None,
        )

        assert obs.shape == (37,)
        np.testing.assert_allclose(obs[33:37], [1.0, 0.0, 0.0, 0.0])

    def test_compute_observations_with_phase_clock(self):
        """When phase_clock is provided, embeds [clock_sin, clock_cos] at obs[35:37].

        Observation dimension remains strictly (37,) (or (42,) with LiDAR).
        """
        base_ang_vel = np.zeros(3, dtype=np.float32)
        proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        commands = np.array([0.2, 0.0, 0.0], dtype=np.float32)
        joint_pos = np.zeros(8, dtype=np.float32)
        joint_vel = np.zeros(8, dtype=np.float32)
        last_actions = np.zeros(8, dtype=np.float32)
        foot_contact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        phase_clock = np.array([0.8, -0.6], dtype=np.float32)

        obs = compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_actions,
            foot_contact=foot_contact,
            phase_clock=phase_clock,
        )

        assert obs.shape == (37,)
        # Left/Right contact at [33:35]
        np.testing.assert_allclose(obs[33:35], [1.0, 0.0])
        # Phase clock [sin, cos] embedded at [35:37]
        np.testing.assert_allclose(obs[35:37], [0.8, -0.6])

    def test_compute_observations_with_phase_clock_and_lidar(self):
        """Obs shape with LiDAR (5 rays) + phase_clock remains (42,)."""
        base_ang_vel = np.zeros(3, dtype=np.float32)
        proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        commands = np.array([0.2, 0.0, 0.0], dtype=np.float32)
        joint_pos = np.zeros(8, dtype=np.float32)
        joint_vel = np.zeros(8, dtype=np.float32)
        last_actions = np.zeros(8, dtype=np.float32)
        foot_contact = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        lidar = np.ones(5, dtype=np.float32) * 1.5
        phase_clock = np.array([1.0, 0.0], dtype=np.float32)

        obs = compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_actions,
            foot_contact=foot_contact,
            lidar_ranges=lidar,
            phase_clock=phase_clock,
        )

        assert obs.shape == (42,)
        np.testing.assert_allclose(obs[33:35], [0.0, 1.0])
        np.testing.assert_allclose(obs[35:37], [1.0, 0.0])
        np.testing.assert_allclose(obs[37:42], lidar)


class TestWalkJointLimitsAndClamping:
    """Task 1 / Part 5: WALK_JOINT_LIMITS and scale_actions walk_clamp unit tests."""

    def test_walk_joint_limits_structure(self):
        """WALK_JOINT_LIMITS must be shape (8, 2) with tighter bounds than full JOINT_LIMITS."""
        cfg = MicroDuck8DoFConfig()
        assert WALK_JOINT_LIMITS.shape == (8, 2)
        # Left hip roll [-0.25, 0.25]
        assert WALK_JOINT_LIMITS[2, 0] == -0.25
        assert WALK_JOINT_LIMITS[2, 1] == 0.25
        # Left hip pitch [-0.50, 0.50]
        assert WALK_JOINT_LIMITS[3, 0] == -0.50
        assert WALK_JOINT_LIMITS[3, 1] == 0.50
        # Left knee pitch [0.0, 0.85]
        assert WALK_JOINT_LIMITS[4, 0] == 0.0
        assert WALK_JOINT_LIMITS[4, 1] == 0.85

    def test_scale_actions_with_walk_clamp(self):
        """scale_actions with walk_clamp=True must clamp to walk limits, while False uses full limits."""
        actions_max = np.ones(8, dtype=np.float32)
        actions_min = -np.ones(8, dtype=np.float32)

        # Full limits
        scaled_full_max = scale_actions(actions_max, walk_clamp=False)
        scaled_full_min = scale_actions(actions_min, walk_clamp=False)
        assert scaled_full_max[4] == pytest.approx(1.48)  # knee full limit
        assert scaled_full_max[3] == pytest.approx(0.87)  # hip pitch full limit

        # Walk clamped limits
        scaled_walk_max = scale_actions(actions_max, walk_clamp=True)
        scaled_walk_min = scale_actions(actions_min, walk_clamp=True)
        assert scaled_walk_max[4] == pytest.approx(0.85)  # knee walk clamped limit
        assert scaled_walk_max[3] == pytest.approx(0.50)  # hip pitch walk clamped limit
        assert scaled_walk_min[3] == pytest.approx(-0.50)


class TestPart3AntiCheatingRewards:
    """Part 3: 9-component anti-cheating reward structure in compute_rewards."""

    @staticmethod
    def _make_base_inputs():
        return {
            "base_lin_vel": np.array([0.15, 0.0, 0.0], dtype=np.float32),
            "cmd_lin_vel": np.array([0.20, 0.0, 0.0], dtype=np.float32),
            "base_ang_vel": np.zeros(3, dtype=np.float32),
            "cmd_ang_vel": 0.0,
            "proj_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "joint_torques": np.zeros(8, dtype=np.float32),
            "joint_vel": np.zeros(8, dtype=np.float32),
            "foot_contact": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }

    def test_reward_weights_contain_new_components(self):
        """MicroDuck8DoFConfig.REWARD_WEIGHTS must contain heading_penalty, leg_pose_penalty, and gait_sync."""
        cfg = MicroDuck8DoFConfig()
        assert "heading_penalty" in cfg.REWARD_WEIGHTS
        assert cfg.REWARD_WEIGHTS["heading_penalty"] == -2.0
        assert "leg_pose_penalty" in cfg.REWARD_WEIGHTS
        assert cfg.REWARD_WEIGHTS["leg_pose_penalty"] == -0.4
        assert "gait_sync" in cfg.REWARD_WEIGHTS
        assert cfg.REWARD_WEIGHTS["gait_sync"] == 0.8

    def test_heading_penalty_walk_mode(self):
        """In WALK mode, yaw error = 0.5 rad gives heading_penalty == -2.0 * 0.25 = -0.50."""
        inputs = self._make_base_inputs()
        rewards = compute_rewards(
            **inputs,
            current_yaw=0.5,
            target_yaw=0.0,
            cmd_mode=LocomotionMode.WALK,
        )
        assert "heading_penalty" in rewards
        assert rewards["heading_penalty"] == pytest.approx(-2.0 * 0.25)

    def test_heading_penalty_sit_and_lie_mode(self):
        """In SIT or LIE mode, heading_penalty is zeroed out (0.0)."""
        inputs = self._make_base_inputs()
        rewards_sit = compute_rewards(
            **inputs,
            current_yaw=0.5,
            target_yaw=0.0,
            cmd_mode=LocomotionMode.SIT,
        )
        assert rewards_sit["heading_penalty"] == pytest.approx(0.0)

        rewards_lie = compute_rewards(
            **inputs,
            current_yaw=0.5,
            target_yaw=0.0,
            cmd_mode=LocomotionMode.LIE,
        )
        assert rewards_lie["heading_penalty"] == pytest.approx(0.0)

    def test_gait_sync_positive_sin_left_swing_right_stance(self):
        """When clock_sin = 0.8 > 0 (desired: left swing 0.0, right stance 1.0):

        - Perfect contact [0.0, 1.0, ...] & vx = 0.15 -> maximal 0.8 * 2.0 = 1.6
        - Out-of-phase [1.0, 0.0, ...] -> 0.0
        - Slow/stationary vx = 0.01 <= 0.03 -> 0.0
        """
        inputs = self._make_base_inputs()

        # In-phase & moving
        r_in_phase = compute_rewards(
            **inputs,
            clock_sin=0.8,
            cmd_mode=LocomotionMode.WALK,
        )
        assert "gait_sync" in r_in_phase
        assert r_in_phase["gait_sync"] == pytest.approx(1.6)

        # Out-of-phase & moving
        inputs_out = inputs.copy()
        inputs_out["foot_contact"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_out_phase = compute_rewards(
            **inputs_out,
            clock_sin=0.8,
            cmd_mode=LocomotionMode.WALK,
        )
        assert r_out_phase["gait_sync"] == pytest.approx(0.0)

        # In-phase but stationary (gated by vx > 0.03)
        inputs_slow = inputs.copy()
        inputs_slow["base_lin_vel"] = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        r_slow = compute_rewards(
            **inputs_slow,
            clock_sin=0.8,
            cmd_mode=LocomotionMode.WALK,
        )
        assert r_slow["gait_sync"] == pytest.approx(0.0)

    def test_gait_sync_negative_sin_left_stance_right_swing(self):
        """When clock_sin = -0.8 < 0 (desired: left stance 1.0, right swing 0.0):

        - Contact [1.0, 0.0, ...] & vx = 0.15 -> maximal 0.8 * 2.0 = 1.6
        """
        inputs = self._make_base_inputs()
        inputs["foot_contact"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_neg = compute_rewards(
            **inputs,
            clock_sin=-0.8,
            cmd_mode=LocomotionMode.WALK,
        )
        assert r_neg["gait_sync"] == pytest.approx(1.6)

    def test_gait_sync_sit_and_lie_mode(self):
        """In SIT or LIE mode, gait_sync is strictly 0.0."""
        inputs = self._make_base_inputs()
        r_sit = compute_rewards(
            **inputs,
            clock_sin=0.8,
            cmd_mode=LocomotionMode.SIT,
        )
        assert r_sit["gait_sync"] == pytest.approx(0.0)

        r_lie = compute_rewards(
            **inputs,
            clock_sin=0.8,
            cmd_mode=LocomotionMode.LIE,
        )
        assert r_lie["gait_sync"] == pytest.approx(0.0)

    def test_leg_pose_penalty_nominal_and_deviated(self):
        """When leg joints deviate from DEFAULT_NOMINAL_QPOS[2:8], leg_pose_penalty < 0.0 (-0.4 * diff_sq).

        When exactly nominal, leg_pose_penalty == 0.0.
        When SIT or LIE, leg_pose_penalty == 0.0.
        """
        inputs = self._make_base_inputs()

        # Exactly nominal
        q_nom = DEFAULT_NOMINAL_QPOS.copy()
        r_nom = compute_rewards(
            **inputs,
            joint_pos=q_nom,
            cmd_mode=LocomotionMode.WALK,
        )
        assert "leg_pose_penalty" in r_nom
        assert r_nom["leg_pose_penalty"] == pytest.approx(0.0)

        # Deviated on left hip roll (index 2) by 0.5 rad -> diff_sq = 0.25 -> penalty = -0.4 * 0.25 = -0.10
        q_dev = DEFAULT_NOMINAL_QPOS.copy()
        q_dev[2] += 0.5
        r_dev = compute_rewards(
            **inputs,
            joint_pos=q_dev,
            cmd_mode=LocomotionMode.WALK,
        )
        assert r_dev["leg_pose_penalty"] == pytest.approx(-0.4 * 0.25)

        # SIT or LIE mode suppresses leg pose penalty
        r_sit = compute_rewards(
            **inputs,
            joint_pos=q_dev,
            cmd_mode=LocomotionMode.SIT,
        )
        assert r_sit["leg_pose_penalty"] == pytest.approx(0.0)

        r_lie = compute_rewards(
            **inputs,
            joint_pos=q_dev,
            cmd_mode=LocomotionMode.LIE,
        )
        assert r_lie["leg_pose_penalty"] == pytest.approx(0.0)

    def test_conditional_alive_bonus(self):
        """In WALK mode with cmd_vx > 0.05 and clock_sin or current_yaw provided:

        - vx <= 0.02 -> alive_bonus == 0.0
        - vx > 0.02 -> alive_bonus > 0.0 (0.05)
        - when clock_sin and current_yaw are None (legacy call) -> alive_bonus == 0.05 (backward compatible)
        """
        inputs = self._make_base_inputs()

        # Moving slow / standing still with clock_sin provided
        inputs_still = inputs.copy()
        inputs_still["base_lin_vel"] = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        r_still = compute_rewards(
            **inputs_still,
            clock_sin=0.8,
            cmd_mode=LocomotionMode.WALK,
        )
        assert r_still["alive_bonus"] == pytest.approx(0.0)

        # Moving forward with clock_sin provided
        inputs_walk = inputs.copy()
        inputs_walk["base_lin_vel"] = np.array([0.15, 0.0, 0.0], dtype=np.float32)
        r_walk = compute_rewards(
            **inputs_walk,
            clock_sin=0.8,
            cmd_mode=LocomotionMode.WALK,
        )
        assert r_walk["alive_bonus"] == pytest.approx(0.05)

        # Backward compatibility when clock_sin and current_yaw are None
        r_legacy_still = compute_rewards(
            **inputs_still,
            cmd_mode=LocomotionMode.WALK,
        )
        assert r_legacy_still["alive_bonus"] == pytest.approx(0.05)

    def test_batched_rewards_vectorized(self):
        """Vector of 2 envs with different yaws, velocities, modes, and phase clocks."""
        base_lin_vel = np.array(
            [[0.15, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        cmd_lin_vel = np.array(
            [[0.20, 0.0, 0.0], [0.20, 0.0, 0.0]], dtype=np.float32
        )
        base_ang_vel = np.zeros((2, 3), dtype=np.float32)
        cmd_ang_vel = np.zeros((2,), dtype=np.float32)
        proj_gravity = np.tile([0.0, 0.0, -1.0], (2, 1)).astype(np.float32)
        joint_torques = np.zeros((2, 8), dtype=np.float32)
        joint_vel = np.zeros((2, 8), dtype=np.float32)
        foot_contact = np.array(
            [[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )

        q_dev = np.tile(DEFAULT_NOMINAL_QPOS, (2, 1)).astype(np.float32)
        q_dev[:, 2] += 0.5  # diff_sq = 0.25

        current_yaw = np.array([0.5, 0.5], dtype=np.float32)
        target_yaw = np.array([0.0, 0.0], dtype=np.float32)
        clock_sin = np.array([0.8, 0.8], dtype=np.float32)
        cmd_mode = np.array([int(LocomotionMode.WALK), int(LocomotionMode.SIT)], dtype=np.int32)

        rewards = compute_rewards(
            base_lin_vel=base_lin_vel,
            cmd_lin_vel=cmd_lin_vel,
            base_ang_vel=base_ang_vel,
            cmd_ang_vel=cmd_ang_vel,
            proj_gravity=proj_gravity,
            joint_torques=joint_torques,
            joint_vel=joint_vel,
            foot_contact=foot_contact,
            joint_pos=q_dev,
            current_yaw=current_yaw,
            target_yaw=target_yaw,
            clock_sin=clock_sin,
            cmd_mode=cmd_mode,
        )

        assert isinstance(rewards["total_reward"], np.ndarray)
        assert rewards["total_reward"].shape == (2,)

        # Env 0: WALK mode -> heading_penalty = -0.5, gait_sync = 1.6, leg_pose_penalty = -0.1, alive_bonus = 0.05
        assert rewards["heading_penalty"][0] == pytest.approx(-0.50, abs=1e-5)
        assert rewards["gait_sync"][0] == pytest.approx(1.6, abs=1e-5)
        assert rewards["leg_pose_penalty"][0] == pytest.approx(-0.10, abs=1e-5)
        assert rewards["alive_bonus"][0] == pytest.approx(0.05, abs=1e-5)

        # Env 1: SIT mode -> heading_penalty = 0.0, gait_sync = 0.0, leg_pose_penalty = 0.0, alive_bonus = 0.05 (not walk cmd)
        assert rewards["heading_penalty"][1] == pytest.approx(0.0, abs=1e-5)
        assert rewards["gait_sync"][1] == pytest.approx(0.0, abs=1e-5)
        assert rewards["leg_pose_penalty"][1] == pytest.approx(0.0, abs=1e-5)
        assert rewards["alive_bonus"][1] == pytest.approx(0.05, abs=1e-5)


class TestPart3EarlyTerminationAndCurriculumGate:
    """Part 3: Strict Early Termination (35-deg tilt, yaw > 1.0 rad), Body-Frame velocity & 3-Condition Curriculum Gate."""

    def test_strict_tilt_35deg_walk_mode(self):
        """In WALK mode, 40-deg pitch tilt (grav_z ~ -0.766 > -0.81915) triggers fallen=True and done=True. Upright does not."""
        runner = ParallelSimRunner(num_envs=2, terrain_mode="flat")
        try:
            runner.reset()
            # Env 0: WALK mode, pitch tilt 40 deg
            theta_40 = np.deg2rad(40.0)
            runner.datas[0].qpos[2] = 0.20
            runner.datas[0].qpos[3:7] = [np.cos(theta_40 / 2.0), 0.0, np.sin(theta_40 / 2.0), 0.0]
            runner.cmd_modes[0] = int(LocomotionMode.WALK)
            mujoco.mj_forward(runner.model, runner.datas[0])

            # Env 1: WALK mode, upright (tilt 0 deg)
            runner.datas[1].qpos[2] = 0.20
            runner.datas[1].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            runner.cmd_modes[1] = int(LocomotionMode.WALK)
            mujoco.mj_forward(runner.model, runner.datas[1])

            actions = np.zeros((2, 8), dtype=np.float32)
            _, _, dones, infos = runner.step(actions)

            assert bool(infos[0]["fallen"]) is True, "40 deg pitch tilt in WALK mode must trigger fallen"
            assert bool(dones[0]) is True, "40 deg pitch tilt in WALK mode must terminate episode"
            assert bool(infos[1]["fallen"]) is False, "Upright robot in WALK mode must not fall"
            assert bool(dones[1]) is False, "Upright robot in WALK mode must not terminate"
        finally:
            runner.close()

    def test_stage0_yaw_spin_termination(self):
        """In Stage 0 with cmd_wz = 0.0, robot rotated > 60 deg (|yaw| > 1.0 rad) triggers fallen=True."""
        runner = ParallelSimRunner(num_envs=2, stage=0, terrain_mode="flat")
        try:
            runner.reset()
            runner.commands[:, :] = 0.0
            runner.commands[:, 0] = 0.20  # forward walk command, cmd_wz = 0.0

            # Env 0: yaw rotation 65 deg (~1.134 rad > 1.0 rad)
            theta_65 = np.deg2rad(65.0)
            runner.datas[0].qpos[2] = 0.20
            runner.datas[0].qpos[3:7] = [np.cos(theta_65 / 2.0), 0.0, 0.0, np.sin(theta_65 / 2.0)]
            runner.cmd_modes[0] = int(LocomotionMode.WALK)
            mujoco.mj_forward(runner.model, runner.datas[0])

            # Env 1: yaw rotation 20 deg (~0.349 rad <= 1.0 rad)
            theta_20 = np.deg2rad(20.0)
            runner.datas[1].qpos[2] = 0.20
            runner.datas[1].qpos[3:7] = [np.cos(theta_20 / 2.0), 0.0, 0.0, np.sin(theta_20 / 2.0)]
            runner.cmd_modes[1] = int(LocomotionMode.WALK)
            mujoco.mj_forward(runner.model, runner.datas[1])

            actions = np.zeros((2, 8), dtype=np.float32)
            _, _, dones, infos = runner.step(actions)

            assert bool(infos[0]["fallen"]) is True, "65 deg yaw spin in Stage 0 must trigger fallen"
            assert bool(dones[0]) is True, "65 deg yaw spin in Stage 0 must terminate episode"
            assert bool(infos[1]["fallen"]) is False, "20 deg yaw heading in Stage 0 must not fall"
            assert bool(dones[1]) is False, "20 deg yaw heading in Stage 0 must not terminate"
        finally:
            runner.close()

    def test_curriculum_lateral_drift_gate(self):
        """Test that excessive lateral drift (|mean_distance_y| >= max_lateral_drift_y) blocks stage advance."""
        # 1. Lateral drift = 0.30 >= 0.25 -> must NOT advance, hold_count resets to 0
        st, hold = next_stage(
            stage=0,
            ep_len_mean=410.0,
            mean_distance_x=1.5,
            mean_distance_y=0.30,
            max_lateral_drift_y=0.25,
            hold_count=5,
        )
        assert st == 0
        assert hold == 0

        # 2. Negative lateral drift = -0.28 -> |drift| >= 0.25 -> must NOT advance
        st, hold = next_stage(
            stage=0,
            ep_len_mean=410.0,
            mean_distance_x=1.5,
            mean_distance_y=-0.28,
            max_lateral_drift_y=0.25,
            hold_count=5,
        )
        assert st == 0
        assert hold == 0

        # 3. Small lateral drift = 0.10 < 0.25 -> increments hold_count
        st, hold = next_stage(
            stage=0,
            ep_len_mean=410.0,
            mean_distance_x=1.5,
            mean_distance_y=0.10,
            max_lateral_drift_y=0.25,
            hold_count=0,
            hold_iters=10,
        )
        assert st == 0
        assert hold == 1

        # 4. Successive hold reaches hold_iters -> advances to stage 1
        st, hold = next_stage(
            stage=0,
            ep_len_mean=410.0,
            mean_distance_x=1.5,
            mean_distance_y=0.10,
            max_lateral_drift_y=0.25,
            hold_count=9,
            hold_iters=10,
        )
        assert st == 1
        assert hold == 0

        # 5. Same test on next_stair_stage
        st_stair, hold_stair = next_stair_stage(
            stage=0,
            ep_len_mean=410.0,
            mean_distance_x=1.5,
            mean_distance_y=0.30,
            max_lateral_drift_y=0.25,
            hold_count=5,
        )
        assert st_stair == 0
        assert hold_stair == 0

    def test_parallel_sim_runner_reward_and_phase_integration(self):
        """Verify that step() computes phase, heading_penalty, gait_sync, leg_pose_penalty, episode_distance_y, and obs phase_clock."""
        runner = ParallelSimRunner(num_envs=1, terrain_mode="flat")
        try:
            obs = runner.reset()
            # Step 0 phase clock check in obs[35:37]
            _, exp_sin0, exp_cos0 = compute_gait_phase(0, dt_control=runner.control_dt, period=0.8)
            assert obs[0, 35] == pytest.approx(exp_sin0, abs=1e-4)
            assert obs[0, 36] == pytest.approx(exp_cos0, abs=1e-4)

            actions = np.zeros((1, 8), dtype=np.float32)
            next_obs, rewards, dones, infos = runner.step(actions)

            # Check that diagnostic info dictionary contains the 9-component overhaul keys
            info = infos[0]
            assert "heading_penalty" in info, "info must contain heading_penalty"
            assert "gait_sync" in info, "info must contain gait_sync"
            assert "leg_pose_penalty" in info, "info must contain leg_pose_penalty"
            assert "episode_distance_y" in info, "info must contain episode_distance_y"
            assert info["episode_distance_y"] == pytest.approx(float(runner.datas[0].qpos[1]), abs=1e-5)

            # Step 1 phase clock check in next_obs[35:37]
            _, exp_sin1, exp_cos1 = compute_gait_phase(1, dt_control=runner.control_dt, period=0.8)
            assert next_obs[0, 35] == pytest.approx(exp_sin1, abs=1e-4)
            assert next_obs[0, 36] == pytest.approx(exp_cos1, abs=1e-4)
        finally:
            runner.close()


class TestPart4TrainViewerSynchronization:
    """Part 4: 100% Train <-> Viewer Synchronization & Fall Detection."""

    def test_train_viewer_observation_and_action_sync(self):
        """Set up single env in ParallelSimRunner and load model in viewer to test parity."""
        runner = ParallelSimRunner(num_envs=1, decimation=10, terrain_mode="flat")
        try:
            runner.reset()
            model_path = os.path.join(PROJECT_ROOT, "models", "mjcf", "microduck_8dof.xml")
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)

            # Synchronize state
            data.qpos[:] = runner.datas[0].qpos[:]
            data.qvel[:] = runner.datas[0].qvel[:]
            mujoco.mj_forward(model, data)

            runner.episode_steps[0] = 15
            runner.commands[0] = np.array([0.25, -0.05, 0.10], dtype=np.float32)
            runner.last_actions[0] = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.05, 0.2, -0.1], dtype=np.float32)

            viewer_obs = extract_viewer_observation(
                model,
                data,
                cmd_vel=runner.commands[0],
                last_actions=runner.last_actions[0],
                obs_dim=37,
                episode_step=15,
                control_dt=0.02,
            )
            runner_obs = runner._extract_obs(0)

            # Check all 37 dimensions match to within 1e-5
            assert viewer_obs.shape == (37,)
            assert runner_obs.shape == (37,)
            assert np.allclose(viewer_obs, runner_obs, atol=1e-5)

            # Specifically check obs[35:37] phase clock at step 15
            _, exp_sin, exp_cos = compute_gait_phase(15, dt_control=0.02, period=0.8)
            assert viewer_obs[35] == pytest.approx(exp_sin, abs=1e-5)
            assert viewer_obs[36] == pytest.approx(exp_cos, abs=1e-5)
            assert runner_obs[35] == pytest.approx(exp_sin, abs=1e-5)
            assert runner_obs[36] == pytest.approx(exp_cos, abs=1e-5)

            # Test action scaling and rate clipping parity
            test_act = np.array([0.5, -0.3, 0.2, -0.4, 0.6, -0.1, 0.3, -0.5], dtype=np.float32)
            cfg = runner.cfg
            scaled_runner = scale_actions(test_act.reshape(1, 8), cfg, nominal_centered=True)[0]
            from training.env_config import _DEFAULT_CFG, clip_action_rates
            scaled_viewer = scale_actions(test_act, _DEFAULT_CFG, nominal_centered=True)
            assert np.allclose(scaled_viewer, scaled_runner, atol=1e-6)

            last_target = runner.last_targets[0]
            targets_runner = clip_action_rates(
                scaled_runner.reshape(1, 8), runner.last_targets, dt=runner.control_dt, max_velocity=cfg.MAX_VELOCITY
            )[0]
            targets_viewer = clip_action_rates(
                scaled_viewer, last_target, dt=0.02, max_velocity=_DEFAULT_CFG.MAX_VELOCITY
            )
            assert np.allclose(targets_viewer, targets_runner, atol=1e-6)
        finally:
            runner.close()

    def test_viewer_fall_detection_and_auto_reset(self):
        """Test that base height drop to z=0.057m or pitch tilt > 35 deg marks is_stable=False."""
        model_path = os.path.join(PROJECT_ROOT, "models", "mjcf", "microduck_8dof.xml")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # 1. Normal upright standing: z=0.18m, upright quat [1, 0, 0, 0] -> is_stable == True
        data.qpos[0:3] = [0.0, 0.0, 0.18]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        state_upright = get_robot_state(model, data)
        assert state_upright["is_stable"] is True
        assert state_upright["grav_z"] == pytest.approx(-1.0, abs=1e-4)

        # 2. Base height drops to z = 0.057m (duck collapsed/lying flat) -> is_stable == False
        data.qpos[0:3] = [0.0, 0.0, 0.057]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        state_dropped = get_robot_state(model, data)
        assert state_dropped["is_stable"] is False, "Base height z=0.057m must be unstable"

        # 3. Base height normal z = 0.18m, but pitch tilts 40 deg (> 35 deg, grav_z > -0.81915) -> is_stable == False
        pitch_40 = np.deg2rad(40.0)
        q_pitch_40 = np.array([np.cos(pitch_40 / 2.0), 0.0, np.sin(pitch_40 / 2.0), 0.0], dtype=np.float32)
        data.qpos[0:3] = [0.0, 0.0, 0.18]
        data.qpos[3:7] = q_pitch_40
        mujoco.mj_forward(model, data)
        state_tilted = get_robot_state(model, data)
        assert state_tilted["grav_z"] > -0.81915
        assert state_tilted["is_stable"] is False, "Pitch tilt 40 deg (> 35 deg) must be unstable"

        # 4. Small pitch tilt 20 deg (< 35 deg, grav_z = -cos(20 deg) = -0.9396 < -0.81915) -> is_stable == True
        pitch_20 = np.deg2rad(20.0)
        q_pitch_20 = np.array([np.cos(pitch_20 / 2.0), 0.0, np.sin(pitch_20 / 2.0), 0.0], dtype=np.float32)
        data.qpos[0:3] = [0.0, 0.0, 0.18]
        data.qpos[3:7] = q_pitch_20
        mujoco.mj_forward(model, data)
        state_small_tilt = get_robot_state(model, data)
        assert state_small_tilt["grav_z"] < -0.81915
        assert state_small_tilt["is_stable"] is True, "Pitch tilt 20 deg (< 35 deg) must remain stable"




