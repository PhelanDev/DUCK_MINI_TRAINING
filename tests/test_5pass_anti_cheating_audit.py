"""Tests for 5-Pass Zero-Tolerance Anti-Cheating & Vulnerability Audit.

Validates all 12 critical anti-cheating, reward-hacking, and physical integrity rules:
1. Strict tilt 45 deg (grav_z > -0.707 triggers fall in WALK/SIT).
2. Trampling in-place guard (v_x < 0.05 -> feet_air_time == 0).
3. Speed diving clip (v_x capped at cmd_vx + 0.05).
4. Alive bonus <= 0.05 (no standing survival reward milking).
5. Curriculum cohort success rate >= 70% mandatory.
6. Stair edge-fall detection (|y| > 0.44m).
7. Slalom corridor bounds (|y| > 0.55m in x in [0.2, 1.2]m triggers corridor_fallen).
8. Joint limit dwelling penalty for joints near 5% bounds.
9. Rate limiting |delta_q| <= 0.10 rad/step.
10. Multi-mode unauthorized pose (WALK collapsing to sit or lie triggers fall; standstill penalty active in WALK, zero in SIT/LIE).
11. Recovery lock & finite loop (SelfRightingController locks after 3 failures).
12. Voice intent desync & CRC16 validation.
13. 4 Golden Hoops anti-vibration & locomotion integrity certification.
14. Anti-backward velocity penalty (-4.0*vx) & forward gait lock (vx > 0.02).
15. Six-part locomotion overhaul (body-frame velocity, phase clock, gait sync, leg pose penalty, strict fall & viewer sync).
"""

from pathlib import Path
import sys
import time
import numpy as np
import pytest
import mujoco

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    LocomotionMode,
    EmotionState,
    compute_rewards,
    clip_action_rates,
    scale_actions,
    quat_to_yaw,
    quat_rotate_inverse,
    compute_gait_phase,
    compute_observations,
    DEFAULT_NOMINAL_QPOS,
    SIT_NOMINAL_QPOS,
    LIE_NOMINAL_QPOS,
    TARGET_SIT_HEIGHT,
    TARGET_LIE_HEIGHT,
    JOINT_LIMITS,
)
from training.run_parallel_sim import ParallelSimRunner
from training.terrain_curriculum import next_stage, next_stair_stage
from scripts.interactive_viewer import extract_viewer_observation
from training.self_righting import (
    SelfRightingController,
    FallClassifier,
    FallType,
    RecoveryState,
)
from training.voice_commander import (
    VoiceCommander,
    VoiceIntent,
    parse_text,
    dispatch_intent,
    pack_uart_voice_packet,
    unpack_uart_voice_packet,
    calc_crc16,
)


def test_audit_1_strict_tilt_45deg():
    """Audit 1: Verify 45-degree tilt threshold (grav_z > -0.707) terminates episode as fallen."""
    runner = ParallelSimRunner(num_envs=3, terrain_mode="flat")
    try:
        runner.reset()
        # Env 0: WALK mode, tilted 50 deg around pitch (grav_z ~ -0.6428 > -0.707)
        runner.cmd_modes[0] = int(LocomotionMode.WALK)
        runner.datas[0].qpos[2] = 0.20
        theta_50 = np.deg2rad(50.0)
        runner.datas[0].qpos[3:7] = [np.cos(theta_50 / 2.0), 0.0, np.sin(theta_50 / 2.0), 0.0]

        # Env 1: SIT mode, tilted 50 deg around pitch (grav_z ~ -0.6428 > -0.707)
        runner.cmd_modes[1] = int(LocomotionMode.SIT)
        runner.datas[1].qpos[2] = TARGET_SIT_HEIGHT
        runner.datas[1].qpos[3:7] = [np.cos(theta_50 / 2.0), 0.0, np.sin(theta_50 / 2.0), 0.0]

        # Env 2: WALK mode, upright (tilt 0 deg -> grav_z = -1.0 <= -0.707)
        runner.cmd_modes[2] = int(LocomotionMode.WALK)
        runner.datas[2].qpos[2] = 0.20
        runner.datas[2].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        for d in runner.datas:
            mujoco.mj_forward(runner.model, d)

        actions = np.zeros((3, 8), dtype=np.float32)
        _, _, dones, infos = runner.step(actions)

        # Env 0 & 1 must be fallen and done
        assert bool(infos[0]["fallen"]) is True, "50 deg tilt in WALK must trigger fallen"
        assert bool(dones[0]) is True
        assert bool(infos[1]["fallen"]) is True, "50 deg tilt in SIT must trigger fallen"
        assert bool(dones[1]) is True

        # Env 2 upright must not be fallen
        assert bool(infos[2]["fallen"]) is False, "Upright pose must not be marked fallen"
        assert bool(dones[2]) is False
    finally:
        runner.close()


def test_audit_2_trampling_in_place_guard():
    """Audit 2: Verify trampling in place (v_x < 0.05 m/s) receives zero feet_air_time reward."""
    cmd_lin_vel = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    feet_air_time = np.array([0.25, 0.0], dtype=np.float32)
    feet_touchdown = np.array([True, False], dtype=bool)

    # 1. Trampling at v_x = 0.02 (< 0.05) -> Feet air time reward MUST BE 0.0
    vel_trample = np.array([0.02, 0.0, 0.0], dtype=np.float32)
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
    assert rew_trample["feet_air_time"] == 0.0, "Trampling must yield zero feet_air_time reward"

    # 2. Genuine locomotion at v_x = 0.15 (>= 0.05) -> receives valid positive feet_air_time reward
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
    assert rew_walk["feet_air_time"] > 0.0, "Genuine forward walking must receive positive air-time reward"


def test_audit_3_speed_diving_clip():
    """Audit 3: Verify speed diving forward velocity reward is clipped at cmd_vx + 0.05."""
    cmd_lin_vel = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.ones(4, dtype=np.float32)

    # Robot diving at v_x = 0.85 m/s when commanded 0.20 m/s
    vel_diving = np.array([0.85, 0.0, 0.0], dtype=np.float32)
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

    # Robot at threshold v_x = 0.25 m/s (cmd_x + 0.05)
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

    assert pytest.approx(rew_diving["tracking_lin_vel"], abs=1e-5) == rew_capped["tracking_lin_vel"], (
        "Diving velocity reward must be clipped to cap value"
    )


def test_audit_4_alive_bonus_and_standstill_penalty():
    """Audit 4: Verify alive bonus <= 0.05 and standing still is penalized when walk commanded."""
    cfg = MicroDuck8DoFConfig()
    alive_weight = float(cfg.REWARD_WEIGHTS.get("alive_bonus", 0.05))
    assert alive_weight <= 0.05, f"Alive bonus {alive_weight} must be <= 0.05 to prevent standing reward milking"

    # Standing still under WALK command:
    # v_x = 0.0, cmd_x = 0.20 -> standstill penalty (-0.50) + alive bonus (+0.05) = net -0.45
    cmd_lin_vel = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    vel_still = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rew_still = compute_rewards(
        base_lin_vel=vel_still,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.ones(4, dtype=np.float32),
        cmd_mode=LocomotionMode.WALK,
    )

    # Active walking at target speed
    vel_walk = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    rew_walk = compute_rewards(
        base_lin_vel=vel_walk,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.ones(4, dtype=np.float32),
        cmd_mode=LocomotionMode.WALK,
    )

    # Standing still must result in negative net alive/standstill contribution and lower total reward
    assert rew_still["total_reward"] < rew_walk["total_reward"] - 0.5


def test_audit_5_curriculum_cohort_success_rate():
    """Audit 5: Verify curriculum requires cohort success_rate >= 0.70 to advance."""
    # 1. next_stage with success_rate = 0.65 (< 0.70): cannot advance
    stage, hold = next_stage(
        stage=0,
        ep_len_mean=450.0,
        rew_lin_vel_ratio=0.85,
        hold_count=0,
        hold_iters=5,
        mean_distance_x=2.0,
        success_rate=0.65,
    )
    assert stage == 0
    assert hold == 0, "Curriculum must not accumulate hold count when success rate < 0.70"

    # 2. next_stage with success_rate = 0.75 (>= 0.70): advances after hold_iters
    stage = 0
    hold = 0
    for _ in range(5):
        stage, hold = next_stage(
            stage=stage,
            ep_len_mean=450.0,
            rew_lin_vel_ratio=0.85,
            hold_count=hold,
            hold_iters=5,
            mean_distance_x=2.0,
            success_rate=0.75,
        )
    assert stage == 1, "Curriculum must advance after sustained high success rate"
    assert hold == 0

    # 3. next_stair_stage: requires success_rate >= 0.70
    stair_stage, stair_hold = next_stair_stage(
        stage=0,
        ep_len_mean=450.0,
        rew_vx_ratio=0.85,
        hold_count=0,
        hold_iters=5,
        mean_distance_x=2.0,
        success_rate=0.50,
    )
    assert stair_stage == 0
    assert stair_hold == 0

    # 4. ParallelSimRunner get_success_rate calculation
    runner = ParallelSimRunner(num_envs=1, terrain_mode="flat")
    try:
        runner.completed_distances_x.clear()
        runner.completed_distances_x.extend([1.6, 1.8, 1.5, 0.8, 0.4])  # 3/5 = 0.60
        assert pytest.approx(runner.get_success_rate(min_dist=1.5), abs=1e-4) == 0.60
    finally:
        runner.close()


def test_audit_6_stair_edge_fall_detection():
    """Audit 6: Verify stair edge-fall detection triggers when |y| > 0.44m in stair zone."""
    runner = ParallelSimRunner(num_envs=2, terrain_mode="stairs")
    try:
        runner.reset()
        # Env 0: In stair zone (x = 1.50m in [1.34, 2.02]m) but fell off side edge (y = 0.46m > 0.44m)
        runner.datas[0].qpos[0] = 1.50
        runner.datas[0].qpos[1] = 0.46
        runner.datas[0].qpos[2] = 0.20

        # Env 1: In stair zone (x = 1.50m) and safely centered (y = 0.15m <= 0.44m)
        runner.datas[1].qpos[0] = 1.50
        runner.datas[1].qpos[1] = 0.15
        runner.datas[1].qpos[2] = 0.20

        for d in runner.datas:
            mujoco.mj_forward(runner.model, d)

        actions = np.zeros((2, 8), dtype=np.float32)
        _, rewards, dones, infos = runner.step(actions)

        # Env 0 must be edge_fallen, fallen, and done
        assert bool(infos[0]["edge_fallen"]) is True, "Expected edge_fallen=True when |y| > 0.44m"
        assert bool(infos[0]["fallen"]) is True
        assert bool(dones[0]) is True

        # Env 1 safely centered must not be edge_fallen
        assert bool(infos[1]["edge_fallen"]) is False
        assert bool(infos[1]["fallen"]) is False
        assert bool(dones[1]) is False

        # Env 0 penalized by fall penalty (-5.0)
        fall_penalty = float(runner.cfg.REWARD_WEIGHTS.get("fall_penalty", -5.0))
        assert pytest.approx(rewards[0], abs=1e-5) == infos[0]["total_reward"] + fall_penalty

        # Outside stair zone (x = 0.50m < 1.34m), y = 0.46m should NOT trigger edge_fallen
        runner.datas[0].qpos[0] = 0.50
        runner.datas[0].qpos[1] = 0.46
        runner.datas[0].qpos[2] = 0.20
        mujoco.mj_forward(runner.model, runner.datas[0])
        _, _, _, infos_outside = runner.step(actions)
        assert bool(infos_outside[0]["edge_fallen"]) is False
    finally:
        runner.close()


def test_audit_7_slalom_corridor_bounds():
    """Audit 7: Verify slalom corridor bypass (|y| > 0.55m in x in [0.2, 1.2]m) triggers corridor_fallen."""
    runner = ParallelSimRunner(num_envs=2, terrain_mode="hurdles")
    try:
        runner.reset()
        # Env 0: In pillar zone (x = 0.70m) and bypassed boundary (y = 0.58m > 0.55m)
        runner.datas[0].qpos[0] = 0.70
        runner.datas[0].qpos[1] = 0.58
        runner.datas[0].qpos[2] = 0.20

        # Env 1: In pillar zone (x = 0.70m) and within corridor (y = 0.20m <= 0.55m)
        runner.datas[1].qpos[0] = 0.70
        runner.datas[1].qpos[1] = 0.20
        runner.datas[1].qpos[2] = 0.20

        for d in runner.datas:
            mujoco.mj_forward(runner.model, d)

        actions = np.zeros((2, 8), dtype=np.float32)
        _, _, dones, infos = runner.step(actions)

        assert bool(infos[0]["corridor_fallen"]) is True, "Exiting corridor must trigger corridor_fallen"
        assert bool(infos[0]["fallen"]) is True
        assert bool(dones[0]) is True

        assert bool(infos[1]["corridor_fallen"]) is False
        assert bool(infos[1]["fallen"]) is False
        assert bool(dones[1]) is False
    finally:
        runner.close()


def test_audit_8_joint_limit_dwelling_penalty():
    """Audit 8: Verify joints dwelling near 5% bounds of physical range receive penalty."""
    base_lin_vel = np.array([0.15, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.15, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.ones(4, dtype=np.float32)

    # 1. Nominal posture: safe from bounds -> penalty == 0.0
    nom_qpos = DEFAULT_NOMINAL_QPOS.copy()
    rew_nom = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=nom_qpos,
    )
    assert rew_nom["joint_limit_penalty"] == 0.0, "Nominal posture must have 0 joint_limit_penalty"

    # 2. Joint near upper limit: left knee at 1.47 (physical limit is 1.48)
    near_limit_qpos = DEFAULT_NOMINAL_QPOS.copy()
    near_limit_qpos[4] = 1.47
    rew_limit = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=0.0,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        joint_pos=near_limit_qpos,
    )
    assert rew_limit["joint_limit_penalty"] < 0.0, "Dwelling near joint limit must produce negative penalty"
    assert rew_limit["total_reward"] < rew_nom["total_reward"]


def test_audit_9_rate_limiting_delta_q():
    """Audit 9: Verify rate limiting enforces |delta_q| <= 0.10 rad/step."""
    # 1. In clip_action_rates (dt=0.02, max_velocity=5.0 -> max_delta = 0.10)
    last_targets = np.zeros(8, dtype=np.float32)
    large_targets = np.full(8, 1.5, dtype=np.float32)
    clipped = clip_action_rates(large_targets, last_targets, dt=0.02, max_velocity=5.0)
    deltas = np.abs(clipped - last_targets)
    assert np.all(deltas <= 0.10 + 1e-6), f"clip_action_rates exceeded 0.10 rad/step: {np.max(deltas)}"

    # 2. In SelfRightingController
    ctrl = SelfRightingController(max_rate=0.10)
    curr_qpos = np.zeros(8, dtype=np.float32)
    ctrl.trigger(FallType.PRONE, curr_qpos)
    last = curr_qpos.copy()
    for _ in range(50):
        target, _ = ctrl.step(last, root_z=0.05, proj_gravity=np.array([0, 0, -0.9], dtype=np.float32))
        step_delta = np.abs(target - last)
        assert np.all(step_delta <= 0.10 + 1e-6), f"SelfRightingController exceeded 0.10 rad/step: {np.max(step_delta)}"
        last = target.copy()


def test_audit_10_multimode_unauthorized_pose():
    """Audit 10: Verify multi-mode unauthorized poses are terminated and reward-restricted."""
    # 1. In ParallelSimRunner: WALK mode collapsing to sit (z=0.065m) or lie (z=0.035m) triggers fall
    runner = ParallelSimRunner(num_envs=3, terrain_mode="flat", decimation=1)
    try:
        runner.reset()
        # Env 0: WALK mode, collapsed to sitting height (z = 0.065m < 0.080m) with sitting legs
        runner.cmd_modes[0] = int(LocomotionMode.WALK)
        runner.datas[0].qpos[0:3] = [0.0, 0.0, 0.065]
        runner.datas[0].qpos[7:15] = SIT_NOMINAL_QPOS
        runner.datas[0].qvel[:] = 0.0

        # Env 1: WALK mode, collapsed to lying height (z = 0.035m < 0.080m) with lying legs
        runner.cmd_modes[1] = int(LocomotionMode.WALK)
        runner.datas[1].qpos[0:3] = [0.0, 0.0, 0.035]
        runner.datas[1].qpos[7:15] = LIE_NOMINAL_QPOS
        runner.datas[1].qvel[:] = 0.0

        # Env 2: WALK mode, valid standing height (z = 0.20m >= 0.080m)
        runner.cmd_modes[2] = int(LocomotionMode.WALK)
        runner.datas[2].qpos[0:3] = [0.0, 0.0, 0.20]
        runner.datas[2].qvel[:] = 0.0

        for d in runner.datas:
            mujoco.mj_forward(runner.model, d)

        actions = np.zeros((3, 8), dtype=np.float32)
        _, _, dones, infos = runner.step(actions)

        assert bool(infos[0]["fallen"]) is True, "Sitting height under WALK command must trigger fallen"
        assert bool(dones[0]) is True
        assert bool(infos[1]["fallen"]) is True, "Lying height under WALK command must trigger fallen"
        assert bool(dones[1]) is True
        assert bool(infos[2]["fallen"]) is False, "Standing height under WALK command must not trigger fallen"
    finally:
        runner.close()

    # 2. In compute_rewards: WALK command denies sitting/lying rewards and activates standstill penalty
    rew_walk = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.array([0.20, 0, 0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0, 0, -1], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.ones(4, dtype=np.float32),
        joint_pos=SIT_NOMINAL_QPOS,
        root_z=TARGET_SIT_HEIGHT,
        cmd_mode=LocomotionMode.WALK,
    )
    assert rew_walk["r_sit"] == 0.0, "r_sit must be zero when commanded WALK"
    assert rew_walk["r_lie"] == 0.0, "r_lie must be zero when commanded WALK"
    assert rew_walk["standstill_penalty"] == -0.5, "Standstill must be active under WALK command"

    # In SIT or LIE mode: standstill penalty is suppressed (0.0)
    rew_sit = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.array([0.20, 0, 0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0, 0, -1], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.ones(4, dtype=np.float32),
        joint_pos=SIT_NOMINAL_QPOS,
        root_z=TARGET_SIT_HEIGHT,
        cmd_mode=LocomotionMode.SIT,
    )
    assert rew_sit["standstill_penalty"] == 0.0, "Standstill penalty must be 0 in SIT mode"
    assert rew_sit["r_sit"] > 0.0, "r_sit must be positive in SIT mode"


def test_audit_11_self_righting_recovery_lock():
    """Audit 11: Verify SelfRightingController safely locks into FAILED_LOCKED after 3 failures."""
    ctrl = SelfRightingController(max_retries=3, phase_timeout=15)
    curr_qpos = np.zeros(8, dtype=np.float32)
    stuck_z = 0.040
    stuck_grav = np.array([0.0, 0.0, -0.40], dtype=np.float32)

    ctrl.trigger(FallType.PRONE, curr_qpos)
    target = curr_qpos.copy()

    # Step through cycles until lock is achieved
    for _ in range(250):
        target, is_done = ctrl.step(target, stuck_z, stuck_grav)
        if ctrl.state == SelfRightingController.FAILED_LOCKED:
            break

    assert ctrl.state == SelfRightingController.FAILED_LOCKED, "Expected controller to enter FAILED_LOCKED"
    assert ctrl.is_failed is True
    assert ctrl.attempts >= 3, "Controller must attempt 3 retries before locking"

    # Once locked, further steps remain locked (no infinite loop)
    target_locked, is_done_locked = ctrl.step(target, stuck_z, stuck_grav)
    assert ctrl.state == SelfRightingController.FAILED_LOCKED
    assert is_done_locked is False


def test_audit_12_voice_intent_crc16_integrity():
    """Audit 12: Verify VoiceIntent binary UART frame integrity and CCITT-16 CRC validation."""
    # 1. Spotting & Dispatch check
    intent = parse_text("nằm xuống")
    assert intent == VoiceIntent.LIE
    dispatched = dispatch_intent(intent)
    assert dispatched["cmd_mode"] == LocomotionMode.LIE

    # 2. Pack UART packet
    raw_packet = pack_uart_voice_packet(VoiceIntent.SIT, seq_id=42)
    assert len(raw_packet) == 22, f"Expected 22 bytes, got {len(raw_packet)}"
    assert raw_packet[0] == 0xAA and raw_packet[1] == 0x55, "Header sync bytes invalid"
    assert raw_packet[2] == 0x06, "Packet type must be 0x06"

    # 3. Successful decode
    unpacked = unpack_uart_voice_packet(raw_packet)
    assert unpacked["cmd_mode"] == LocomotionMode.SIT
    assert unpacked["seq_id"] == 42

    # 4. Corruption rejection (CRC mismatch)
    corrupted_payload = bytearray(raw_packet)
    corrupted_payload[7] ^= 0xFF  # Flip bits in payload
    with pytest.raises(ValueError, match="CRC mismatch"):
        unpack_uart_voice_packet(bytes(corrupted_payload))

    # 5. Header sync corruption rejection
    corrupted_sync = bytearray(raw_packet)
    corrupted_sync[0] = 0x00
    with pytest.raises(ValueError, match="Invalid sync bytes"):
        unpack_uart_voice_packet(bytes(corrupted_sync))

    # 6. Truncated packet rejection
    with pytest.raises(ValueError, match="Incomplete packet"):
        unpack_uart_voice_packet(raw_packet[:12])

    # 7. Non-blocking throughput verification (> 1000 pkt/s)
    t0 = time.perf_counter()
    for _ in range(500):
        pkt = pack_uart_voice_packet(VoiceIntent.STAND_WALK)
        _ = unpack_uart_voice_packet(pkt)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.050, f"500 packet pack+unpack cycles took {elapsed:.4f}s (exceeded 50ms budget)"


def test_audit_13_four_golden_hoops_anti_vibration():
    """Audit 13: 4 Golden Hoops Anti-Vibration & Locomotion Integrity Certification."""
    from training.env_config import _DEFAULT_CFG, compute_rewards

    # 1. Hoop 1: Single support & double support dwell threshold
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("double_support_penalty") == -0.5
    r_dwell = compute_rewards(
        base_lin_vel=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        double_support_time=0.25,
    )
    assert r_dwell["tracking_lin_vel"] == 0.0, "Dwell > 0.20s must zero linear velocity reward"
    assert r_dwell["double_support_penalty"] == -0.5, "Dwell > 0.20s must apply -0.5 penalty"

    # 2. Hoop 2: Action rate (-0.10) & Joint acceleration (-0.01)
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("action_rate_penalty") == -0.10
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("joint_acc_penalty") == -0.01
    r_acc = compute_rewards(
        base_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.ones(8, dtype=np.float32) * 5.0,
        last_joint_vel=-np.ones(8, dtype=np.float32) * 5.0,
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert "joint_acc_penalty" in r_acc
    assert np.isclose(r_acc["joint_acc_penalty"], 800.0)

    # 3. Hoop 3: Orientation tilt (-2.0) & Head pos penalty (-1.0)
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("upright") == 1.5
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("orientation_penalty") == -2.0
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("head_pos_penalty") == -1.0
    r_tilt = compute_rewards(
        base_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.5, 0.0, -0.866], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.isclose(r_tilt["orientation_penalty"], 0.50)

    # 4. Hoop 4: Floor friction >= 1.5 & Foot slip penalty (-0.8)
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("foot_slip_penalty") == -0.8
    xml_path = Path(__file__).resolve().parent.parent / "models" / "mjcf" / "microduck_8dof.xml"
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()
    assert 'friction="1.5' in xml_text
    r_slip = compute_rewards(
        base_lin_vel=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        foot_vel_xy=np.array([[0.3, 0.1], [0.0, 0.0]], dtype=np.float32),
        foot_contact_forces=np.array([[0.0, 0.0, 3.5], [0.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert np.isclose(r_slip["foot_slip_penalty"], 0.10, atol=1e-4)



def test_audit_14_anti_backward_and_forward_gait_lock():
    """Audit 14: Anti-Backward Velocity Penalty (-4.0 * vx) & Forward Gait Lock (vx > 0.02).
    Certifies that:
    1. Backward movement is penalized 4.0 * vx (twice the forward tracking rate).
    2. Double support dwell preserves negative backward penalty while zeroing forward reward.
    3. Gait rewards (foot_contact & feet_air_time) are strictly gated by (vel_x > 0.02).
    4. MuJoCo XML kinematics confirm +X forward orientation and hip_pitch/knee_pitch sign conventions.
    """
    cmd_lin = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    base_ang = np.zeros(3, dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)
    one_foot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # 1. Forward vs Backward velocity rewards
    r_fwd = compute_rewards(
        base_lin_vel=np.array([0.20, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=one_foot,
    )
    assert np.isclose(r_fwd["tracking_lin_vel"], 0.40, atol=1e-5), "Forward vx=0.20 must yield +0.40"

    r_bwd = compute_rewards(
        base_lin_vel=np.array([-0.20, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=one_foot,
    )
    assert np.isclose(r_bwd["tracking_lin_vel"], -0.80, atol=1e-5), "Backward vx=-0.20 must yield -0.80 penalty"

    # 2. Dwell preservation: backward penalty must NOT be wiped by dwell
    r_bwd_dwell = compute_rewards(
        base_lin_vel=np.array([-0.20, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        double_support_time=0.25,
    )
    assert np.isclose(r_bwd_dwell["tracking_lin_vel"], -0.80, atol=1e-5), "Dwell must preserve backward penalty"

    # 3. Gait lock: vx <= 0.02 locks foot_contact and feet_air_time to 0.0
    air_t = np.array([0.25, 0.0], dtype=np.float32)
    td = np.array([True, False], dtype=bool)
    r_still_gait = compute_rewards(
        base_lin_vel=np.array([0.01, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=one_foot,
        feet_air_time=air_t,
        feet_touchdown=td,
    )
    assert r_still_gait["foot_contact"] == 0.0, "vx <= 0.02 must receive 0.0 foot_contact"
    assert r_still_gait["feet_air_time"] == 0.0, "vx <= 0.02 must receive 0.0 feet_air_time"

    r_fwd_gait = compute_rewards(
        base_lin_vel=np.array([0.05, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=one_foot,
        feet_air_time=air_t,
        feet_touchdown=td,
    )
    assert r_fwd_gait["foot_contact"] == 1.0, "vx=0.05 must receive 1.0 foot_contact"
    assert r_fwd_gait["feet_air_time"] > 0.0, "vx=0.05 must receive >0.0 feet_air_time"

    # 4. Kinematics verification
    xml_path = Path(__file__).resolve().parent.parent / "models" / "mjcf" / "microduck_8dof.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    beak_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "beak_geom")

    assert data.geom_xpos[beak_geom_id][0] > data.xpos[base_id][0], "Beak must point along +X"

    hip_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")
    knee_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_pitch")
    hip_qadr = model.jnt_qposadr[hip_jnt_id]
    knee_qadr = model.jnt_qposadr[knee_jnt_id]

    data.qpos[hip_qadr] = -0.35
    data.qpos[knee_qadr] = 0.0
    mujoco.mj_forward(model, data)
    assert (data.xpos[foot_id][0] - data.xpos[base_id][0]) > 0.03, "Hip pitch < 0 swings foot forward (+X)"

    data.qpos[hip_qadr] = 0.0
    data.qpos[knee_qadr] = 0.50
    mujoco.mj_forward(model, data)
    assert (data.xpos[foot_id][0] - data.xpos[base_id][0]) < -0.02, "Knee pitch > 0 folds foot backward (-X)"


def test_audit_15_six_part_locomotion_overhaul():
    """Audit 15: Six-Part Locomotion Overhaul & Kinodynamic Anti-Cheating Certification.

    Validates:
    1. Body-Frame velocity projection: rotating +90 deg around Z and translating along
       World +X yields vx_local == 0.0 (crab-walking exploits eliminated). Also verify
       heading_penalty (-2.0 * yaw^2) in WALK mode and zero in SIT/LIE.
    2. Gait phase clock (T=0.8s) embeds [clock_sin, clock_cos] at obs[35:37], and gait_sync
       is strictly 0.0 when vx_local <= 0.03 while rewarding phase-synchronized foot contacts
       when vx_local > 0.03.
    3. leg_pose_penalty penalizes split stance deviating from DEFAULT_NOMINAL_QPOS[2:8] with
       weight -0.4, but remains 0.0 when nominal or in SIT/LIE modes.
    4. Strict early termination: WALK mode tilt > 35 deg triggers fallen == True, and Stage 0
       yaw deviation > 60 deg (1.0 rad) triggers fallen == True; curriculum progression is
       blocked when |mean_distance_y| >= 0.25m.
    5. 100% observation and control parity between ParallelSimRunner._extract_obs() and
       interactive_viewer.extract_viewer_observation().
    """
    zero_foot = np.zeros(4, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Check 1: Body-Frame Velocity Projection & Heading Penalty
    # -------------------------------------------------------------------------
    q_90 = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=np.float32)
    world_lin_vel = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    body_vel = quat_rotate_inverse(q_90, world_lin_vel)
    assert np.isclose(body_vel[0], 0.0, atol=1e-5), "Crab-walking at 90 deg yaw must yield vx_local == 0.0"
    assert np.isclose(body_vel[1], -0.2, atol=1e-5), "Crab-walking at 90 deg yaw must be pure lateral drift vy_local == -0.2"

    # Heading penalty in WALK mode: -2.0 * yaw^2
    r_heading_walk = compute_rewards(
        base_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=zero_foot,
        current_yaw=0.5,
        target_yaw=0.0,
        cmd_mode=LocomotionMode.WALK,
    )
    assert np.isclose(r_heading_walk["heading_penalty"], -2.0 * (0.5 ** 2), atol=1e-5), "Heading penalty in WALK must be -2.0 * yaw^2"

    # Heading penalty in SIT and LIE modes must be 0.0
    r_heading_sit = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=zero_foot,
        current_yaw=0.5,
        target_yaw=0.0,
        cmd_mode=LocomotionMode.SIT,
    )
    assert r_heading_sit["heading_penalty"] == 0.0, "SIT mode must have 0.0 heading_penalty"

    r_heading_lie = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=zero_foot,
        current_yaw=0.5,
        target_yaw=0.0,
        cmd_mode=LocomotionMode.LIE,
    )
    assert r_heading_lie["heading_penalty"] == 0.0, "LIE mode must have 0.0 heading_penalty"

    # -------------------------------------------------------------------------
    # Check 2: Gait Phase Clock (T=0.8s) & Phase-Synchronized Gait Sync
    # -------------------------------------------------------------------------
    phase, clock_sin, clock_cos = compute_gait_phase(episode_steps=10, dt_control=0.02, period=0.8)
    assert np.isclose(phase, 0.25, atol=1e-5), "10 steps at 0.02s with T=0.8s must yield phase == 0.25"
    assert np.isclose(clock_sin, 1.0, atol=1e-5), "phase 0.25 must yield clock_sin == 1.0"
    assert np.isclose(clock_cos, 0.0, atol=1e-5), "phase 0.25 must yield clock_cos == 0.0"

    obs = compute_observations(
        base_ang_vel=np.zeros(3, dtype=np.float32),
        proj_grav=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        commands=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        joint_pos=DEFAULT_NOMINAL_QPOS,
        joint_vel=np.zeros(8, dtype=np.float32),
        last_actions=np.zeros(8, dtype=np.float32),
        foot_contact=np.zeros(4, dtype=np.float32),
        phase_clock=np.array([clock_sin, clock_cos], dtype=np.float32),
    )
    assert obs.shape == (37,), "Base observation dimension must be 37"
    assert np.isclose(obs[35], clock_sin, atol=1e-5), "obs[35] must store clock_sin"
    assert np.isclose(obs[36], clock_cos, atol=1e-5), "obs[36] must store clock_cos"

    # gait_sync is strictly 0.0 when vx_local <= 0.03
    r_gait_still = compute_rewards(
        base_lin_vel=np.array([0.02, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        clock_sin=1.0,
        cmd_mode=LocomotionMode.WALK,
    )
    assert r_gait_still["gait_sync"] == 0.0, "vx <= 0.03 must strictly lock gait_sync to 0.0"

    # gait_sync rewards phase-synchronized foot contacts when vx_local > 0.03
    r_gait_sync = compute_rewards(
        base_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        clock_sin=1.0,
        cmd_mode=LocomotionMode.WALK,
    )
    assert r_gait_sync["gait_sync"] > 1.0, "Synchronized foot contact with vx > 0.03 must reward gait_sync"

    # -------------------------------------------------------------------------
    # Check 3: leg_pose_penalty Penalizes Split Stance Deviating from Nominal
    # -------------------------------------------------------------------------
    r_nom = compute_rewards(
        base_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=zero_foot,
        joint_pos=DEFAULT_NOMINAL_QPOS,
        cmd_mode=LocomotionMode.WALK,
    )
    assert r_nom["leg_pose_penalty"] == 0.0, "Nominal leg pose must incur 0.0 leg_pose_penalty"

    split_qpos = DEFAULT_NOMINAL_QPOS.copy()
    split_qpos[2] += 0.5  # left hip pitch
    split_qpos[5] -= 0.5  # right hip pitch
    # dev_sq = 0.5^2 + (-0.5)^2 = 0.50. Weight is -0.4 -> expected = -0.20
    r_split_walk = compute_rewards(
        base_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=zero_foot,
        joint_pos=split_qpos,
        cmd_mode=LocomotionMode.WALK,
    )
    assert np.isclose(r_split_walk["leg_pose_penalty"], -0.20, atol=1e-5), "Split leg pose must be penalized with weight -0.4"

    r_split_sit = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=zero_foot,
        joint_pos=split_qpos,
        cmd_mode=LocomotionMode.SIT,
    )
    assert r_split_sit["leg_pose_penalty"] == 0.0, "SIT mode must have 0.0 leg_pose_penalty"

    r_split_lie = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=zero_foot,
        joint_pos=split_qpos,
        cmd_mode=LocomotionMode.LIE,
    )
    assert r_split_lie["leg_pose_penalty"] == 0.0, "LIE mode must have 0.0 leg_pose_penalty"

    # -------------------------------------------------------------------------
    # Check 4: Strict Early Termination & Curriculum Lateral Drift Gate
    # -------------------------------------------------------------------------
    runner = ParallelSimRunner(num_envs=2, terrain_mode="flat")
    try:
        runner.reset()
        # Env 0: WALK mode with 40-deg pitch tilt (> 35 deg, grav_z ~ -0.766 > -0.81915)
        runner.cmd_modes[0] = int(LocomotionMode.WALK)
        runner.datas[0].qpos[2] = 0.20
        pitch_40 = np.deg2rad(40.0)
        runner.datas[0].qpos[3:7] = [np.cos(pitch_40 / 2.0), 0.0, np.sin(pitch_40 / 2.0), 0.0]

        # Env 1: Stage 0 yaw deviation > 60 deg (1.134 rad > 1.0 rad) with cmd_wz = 0.0
        runner.cmd_modes[1] = int(LocomotionMode.WALK)
        runner.commands[1] = [0.2, 0.0, 0.0]
        runner.datas[1].qpos[2] = 0.20
        yaw_65 = np.deg2rad(65.0)
        runner.datas[1].qpos[3:7] = [np.cos(yaw_65 / 2.0), 0.0, 0.0, np.sin(yaw_65 / 2.0)]
        runner.curriculum_stage = 0

        for d in runner.datas:
            mujoco.mj_forward(runner.model, d)

        actions = np.zeros((2, 8), dtype=np.float32)
        _, _, dones, infos = runner.step(actions)

        assert bool(infos[0]["fallen"]) is True, "40 deg tilt (>35 deg) in WALK must trigger fallen == True"
        assert bool(dones[0]) is True
        assert bool(infos[1]["fallen"]) is True, "Stage 0 yaw > 60 deg (1.0 rad) must trigger fallen == True"
        assert bool(dones[1]) is True
    finally:
        runner.close()

    # Curriculum lateral drift gate: |mean_distance_y| >= 0.25m blocks progression
    stage_drift, hold_drift = next_stage(
        stage=0,
        ep_len_mean=450.0,
        mean_distance_x=1.5,
        mean_distance_y=0.28,
        hold_count=5,
    )
    assert stage_drift == 0 and hold_drift == 0, "Curriculum progression must be blocked when |mean_distance_y| >= 0.25m"

    stage_ok, hold_ok = next_stage(
        stage=0,
        ep_len_mean=450.0,
        mean_distance_x=1.5,
        mean_distance_y=0.10,
        hold_count=9,
        hold_iters=10,
    )
    assert stage_ok == 1 and hold_ok == 0, "Clean forward corridor walk (|mean_distance_y| < 0.25m) must advance stage"

    # -------------------------------------------------------------------------
    # Check 5: 100% Observation & Control Parity
    # -------------------------------------------------------------------------
    runner_sync = ParallelSimRunner(num_envs=1, terrain_mode="flat", decimation=10)
    try:
        runner_sync.reset()
        runner_sync.episode_steps[0] = 7
        runner_sync.commands[0] = np.array([0.20, -0.05, 0.0], dtype=np.float32)
        runner_sync.last_actions[0] = np.array([0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.1, -0.1], dtype=np.float32)

        model = runner_sync.model
        data = runner_sync.datas[0]
        mujoco.mj_forward(model, data)

        sim_obs = runner_sync._extract_obs(0)
        viewer_obs = extract_viewer_observation(
            model=model,
            data=data,
            cmd_vel=runner_sync.commands[0],
            last_actions=runner_sync.last_actions[0],
            obs_dim=37,
            episode_step=7,
            control_dt=runner_sync.control_dt,
        )

        assert sim_obs.shape == viewer_obs.shape == (37,), "Observation dimensions must be (37,)"
        assert np.allclose(sim_obs, viewer_obs, atol=1e-5), "Observation mismatch between ParallelSimRunner and interactive_viewer"
        assert np.isclose(sim_obs[35], viewer_obs[35], atol=1e-5)
        assert np.isclose(sim_obs[36], viewer_obs[36], atol=1e-5)
        assert not (np.isclose(sim_obs[35], 0.0, atol=1e-4) and np.isclose(sim_obs[36], 0.0, atol=1e-4))

        # Action scaling & clipping parity
        test_act = np.array([0.5, -0.3, 0.2, -0.4, 0.6, -0.1, 0.3, -0.5], dtype=np.float32)
        scaled_runner = scale_actions(test_act.reshape(1, 8), runner_sync.cfg, nominal_centered=True)[0]
        scaled_viewer = scale_actions(test_act, _DEFAULT_CFG, nominal_centered=True)
        assert np.allclose(scaled_viewer, scaled_runner, atol=1e-6), "scale_actions parity failed"

        last_target = runner_sync.last_targets[0]
        targets_runner = clip_action_rates(
            scaled_runner.reshape(1, 8), runner_sync.last_targets, dt=runner_sync.control_dt, max_velocity=runner_sync.cfg.MAX_VELOCITY
        )[0]
        targets_viewer = clip_action_rates(
            scaled_viewer, last_target, dt=0.02, max_velocity=_DEFAULT_CFG.MAX_VELOCITY
        )
        assert np.allclose(targets_viewer, targets_runner, atol=1e-6), "clip_action_rates parity failed"
    finally:
        runner_sync.close()

