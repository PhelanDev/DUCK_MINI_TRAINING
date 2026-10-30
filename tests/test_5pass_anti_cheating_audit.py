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
    DEFAULT_NOMINAL_QPOS,
    SIT_NOMINAL_QPOS,
    LIE_NOMINAL_QPOS,
    TARGET_SIT_HEIGHT,
    TARGET_LIE_HEIGHT,
    JOINT_LIMITS,
)
from training.run_parallel_sim import ParallelSimRunner
from training.terrain_curriculum import next_stage, next_stair_stage
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
