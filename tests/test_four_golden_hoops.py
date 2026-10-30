"""Tests for 4 Golden Hoops Anti-Vibration & Locomotion Enforcement.

Task 1: Single-support gate & double-support dwell penalty (> 0.2s).
Task 2: Action rate penalty (-0.10) & joint acceleration penalty.
Task 3: Orientation penalty (-2.0) & head position penalty (-1.0).
Task 4: High floor friction (1.5) & foot slip penalty (-0.8).
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.env_config import compute_rewards, LocomotionMode, _DEFAULT_CFG


def test_hoop1_single_support_and_double_support_dwell():
    """Vòng Kim Cô 1: Thưởng nhấc 1 chân (single_support) và cắt thưởng vận tốc + phạt nếu dính 2 chân > 0.2s."""
    base_lin = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    cmd_lin = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    base_ang = np.zeros(3, dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)

    # 1. Single support: 1 chân chạm sàn, 1 chân nhấc -> Nhận đủ reward_vel và foot_contact = 1.0
    r_single = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        double_support_time=0.0,
    )
    assert r_single["tracking_lin_vel"] > 0.5, "Single support must receive full linear velocity reward"
    assert r_single["foot_contact"] == 1.0, "Single support must yield foot_contact reward of 1.0"
    assert r_single.get("double_support_penalty", 0.0) == 0.0, "No double support penalty during single support"

    # 2. Double support dwell > 0.2s: cả 2 chân dính sàn quá 0.2s -> Cắt sạch tracking_lin_vel về 0 và phạt double_support_penalty
    r_double_dwell = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        double_support_time=0.22,  # > 0.20s threshold
    )
    assert r_double_dwell["tracking_lin_vel"] == 0.0, "Two feet sticking to ground > 0.2s must zero out tracking_lin_vel"
    assert "double_support_penalty" in r_double_dwell
    assert r_double_dwell["double_support_penalty"] < -0.4, "Must incur negative double_support_penalty (-0.5)"
    assert r_double_dwell["total_reward"] < r_single["total_reward"]

    # 3. Lệnh đứng yên (cmd_x <= 0.05): Được phép chạm 2 chân mà KHÔNG bị cắt điểm hoặc phạt
    cmd_stationary = np.zeros(3, dtype=np.float32)
    r_standstill = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=cmd_stationary,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        double_support_time=1.0,
    )
    assert r_standstill.get("double_support_penalty", 0.0) == 0.0, "Stationary command must NOT penalize double support"

    # 4. Chế độ Ngồi (SIT) hoặc Nằm (LIE): Không bị phạt double_support_penalty
    r_sit = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        cmd_mode=LocomotionMode.SIT,
        double_support_time=0.5,
    )
    assert r_sit.get("double_support_penalty", 0.0) == 0.0, "Sitting mode must NOT penalize double support"


def test_parallel_sim_runner_double_support_time_tracking():
    """Verify that ParallelSimRunner accumulates double_support_time when both feet touch."""
    from training.run_parallel_sim import ParallelSimRunner
    sim = ParallelSimRunner(num_envs=2)
    obs = sim.reset()
    assert hasattr(sim, "double_support_time")
    assert sim.double_support_time.shape == (2,)
    # Run multiple steps standing still on two feet
    actions = np.zeros((2, 8), dtype=np.float32)
    for _ in range(5):
        sim.step(actions)
    # Both feet on ground in stand pose -> double_support_time should be > 0
    assert np.all(sim.double_support_time > 0.0)
    # Reset env 0 -> env 0 resets to 0.0, env 1 retains
    prev_1 = sim.double_support_time[1]
    sim.reset_env(0)
    assert sim.double_support_time[0] == 0.0
    assert sim.double_support_time[1] == prev_1




def test_hoop2_action_rate_and_joint_acc_penalty():
    """Vòng Kim Cô 2: Phạt nặng giật khớp liên thanh với action_rate_penalty (-0.10) và joint_acc_penalty (-0.01)."""
    assert _DEFAULT_CFG.REWARD_WEIGHTS["action_rate_penalty"] == -0.10, "action_rate_penalty weight must be -0.10"
    assert "joint_acc_penalty" in _DEFAULT_CFG.REWARD_WEIGHTS
    assert _DEFAULT_CFG.REWARD_WEIGHTS["joint_acc_penalty"] == -0.01, "joint_acc_penalty weight must be -0.01"

    base_lin = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    cmd_lin = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    f_contact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # 1. Smooth actions vs rapid alternating actions
    smooth_act = np.ones(8, dtype=np.float32) * 0.5
    jitter_act_curr = np.ones(8, dtype=np.float32) * 1.0
    jitter_act_prev = -np.ones(8, dtype=np.float32) * 1.0

    r_smooth = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=f_contact,
        actions=smooth_act,
        last_actions=smooth_act,
    )

    r_jitter = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=f_contact,
        actions=jitter_act_curr,
        last_actions=jitter_act_prev,
    )

    # diff = 2.0, squared sum = 32.0. Penalty difference = 32.0 * 0.10 = 3.20
    assert np.isclose(r_smooth["total_reward"] - r_jitter["total_reward"], 3.20), "Action rate penalty must reduce reward by 3.20"

    # 2. Joint acceleration penalty (sudden velocity spike/inversion)
    j_vel_spike = np.ones(8, dtype=np.float32) * 5.0
    j_vel_prev = -np.ones(8, dtype=np.float32) * 5.0

    r_acc = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel_spike,
        foot_contact=f_contact,
        last_joint_vel=j_vel_prev,
    )

    assert "joint_acc_penalty" in r_acc
    # delta_vel = 10.0, sq sum = 800.0, penalty = -0.01 * 800.0 = -8.00
    assert np.isclose(r_acc["joint_acc_penalty"], 800.0)
    assert r_acc["total_reward"] < r_smooth["total_reward"]


def test_hoop3_orientation_and_head_pos_penalty():
    """Vòng Kim Cô 3: Phạt gập người chúi mỏ (orientation_penalty = -2.0) và lệch cổ (head_pos_penalty = -1.0)."""
    # 1. Kiểm tra trọng số mặc định
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("upright") == 1.5, "upright weight must be 1.5"
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("orientation_penalty") == -2.0, "orientation_penalty weight must be -2.0"
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("head_pos_penalty") == -1.0, "head_pos_penalty weight must be -1.0"

    base_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    cmd_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    base_ang = np.zeros(3, dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)
    f_contact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # 2. Upright posture vs Pitch tilt (chúi mỏ về phía trước)
    # Pitch tilt: gx = 0.5, gy = 0.0, gz = -0.866
    r_vertical = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        joint_pos=np.zeros(8, dtype=np.float32),
    )

    r_nose_dive = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.5, 0.0, -0.866], dtype=np.float32),
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        joint_pos=np.zeros(8, dtype=np.float32),
    )
    # orientation_penalty = 2.0 * gx^2 + 1.0 * gy^2 = 2.0 * 0.25 = 0.50
    assert "orientation_penalty" in r_nose_dive
    assert np.isclose(r_nose_dive["orientation_penalty"], 0.50)
    assert r_nose_dive["total_reward"] < r_vertical["total_reward"]
    # Total penalty = -2.0 * 0.50 = -1.00
    assert r_vertical["total_reward"] - r_nose_dive["total_reward"] >= 1.00

    # 3. Chế độ Nằm (LIE): Được phép ngả người mà không bị phạt orientation_penalty
    r_lie = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        cmd_mode=LocomotionMode.LIE,
    )
    assert r_lie.get("orientation_penalty", 0.0) == 0.0, "Lie mode must NOT incur orientation penalty"

    # 4. Head position penalty: cổ nhìn thẳng (0.0, 0.0) vs lệch cổ/chúi đầu (0.3, 0.4)
    # Head penalty = 0.3^2 + 2.0 * 0.4^2 = 0.09 + 0.32 = 0.41
    j_slouch = np.zeros(8, dtype=np.float32)
    j_slouch[0] = 0.3
    j_slouch[1] = 0.4

    r_slouch = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        joint_pos=j_slouch,
    )
    assert "head_pos_penalty" in r_slouch
    assert np.isclose(r_slouch["head_pos_penalty"], 0.41)
    assert r_slouch["total_reward"] < r_vertical["total_reward"]
    assert np.isclose(r_vertical["total_reward"] - r_slouch["total_reward"], 0.41)


def test_hoop4_floor_friction_and_foot_slip_penalty():
    """Vòng Kim Cô 4: Tăng ma sát sàn XML (>= 1.5) và phạt trượt bàn chân (foot_slip_penalty = -0.8)."""
    # 1. Trọng số mặc định
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("foot_slip_penalty") == -0.8, "foot_slip_penalty weight must be -0.8"

    # 2. Kiểm tra XML ma sát floor và foot
    xml_path = os.path.join(PROJECT_ROOT, "models", "mjcf", "microduck_8dof.xml")
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()
    assert 'name="floor"' in xml_content
    # Friction attribute should have >= 1.5
    assert 'friction="1.5' in xml_content or 'friction="1.6' in xml_content, "Floor/foot friction must be >= 1.5 in XML"

    base_lin = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    cmd_lin = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)

    # 3. Chân chịu tải chạm sàn mà trượt -> Bị phạt foot_slip_penalty
    # Chân trái chịu tải Fz = 3.5N, vận tốc trượt vx=0.3, vy=0.1 -> v_xy^2 = 0.09 + 0.01 = 0.10
    forces_contact = np.array([[0.0, 0.0, 3.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    vel_slip = np.array([[0.3, 0.1], [0.0, 0.0]], dtype=np.float32)

    r_slip = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        foot_vel_xy=vel_slip,
        foot_contact_forces=forces_contact,
    )
    assert "foot_slip_penalty" in r_slip
    assert np.isclose(r_slip["foot_slip_penalty"], 0.10, atol=1e-4)
    # Phạt -0.8 * 0.10 = -0.08

    # 4. Chân nhấc trên không (Fz = 0.0N, f_contact = 0.0) vung nhanh về phía trước -> KHÔNG bị phạt foot_slip
    forces_air = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.5]], dtype=np.float32)
    vel_swing = np.array([[0.5, 0.0], [0.0, 0.0]], dtype=np.float32)  # vung nhanh 0.5 m/s

    r_swing = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        foot_vel_xy=vel_swing,
        foot_contact_forces=forces_air,
    )
    assert "foot_slip_penalty" in r_swing
    assert r_swing["foot_slip_penalty"] == 0.0, "Swing foot in air must NOT incur foot slip penalty"


def test_hoop5_anti_backward_walking_and_forward_gate():
    """Vòng Kim Cô 5: Cấm tuyệt đối đi giật lùi ăn điểm dương; khoá thưởng tư thế/nhấc chân khi không tiến."""
    assert _DEFAULT_CFG.REWARD_WEIGHTS.get("backward_vel_penalty") == -2.0

    cmd_lin = np.array([0.25, 0.0, 0.0], dtype=np.float32)
    base_ang = np.zeros(3, dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Thẳng lưng tuyệt đối
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)
    single_foot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Nhấc 1 chân chuẩn

    # 1. Đi lùi (vx = -0.12 m/s) dù thẳng lưng + nhấc 1 chân -> total_reward PHẢI ÂM NẶNG (< -1.0)
    r_backward = compute_rewards(
        base_lin_vel=np.array([-0.12, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=single_foot,
    )
    assert "backward_vel_penalty" in r_backward
    assert r_backward["backward_vel_penalty"] < -2.0, "Walking backward must trigger heavy backward_vel_penalty"
    assert r_backward["foot_contact"] == 0.0, "Walking backward must zero out single-support foot_contact reward"
    assert r_backward["total_reward"] < -1.0, (
        f"Walking backward (vx=-0.12) must yield negative total_reward (< -1.0), got {r_backward['total_reward']}"
    )

    # 2. Đứng im tại chỗ (vx = 0.0 m/s) khi có lệnh đi -> total_reward PHẢI <= 0.0
    r_stand = compute_rewards(
        base_lin_vel=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    assert r_stand["total_reward"] <= 0.0, (
        f"Standing still under walk command must NOT yield positive total_reward, got {r_stand['total_reward']}"
    )

    # 3. Bước tiến thực sự (vx = +0.25 m/s) -> Mở khoá toàn bộ phần thưởng (> +2.5)
    r_forward = compute_rewards(
        base_lin_vel=np.array([0.25, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=single_foot,
    )
    assert r_forward["total_reward"] > 2.5


def test_parallel_sim_always_commands_forward_walking():
    """Đảm bảo khi stand_still=False, 100% môi trường nhận lệnh tiến cmd_x >= 0.10 m/s (không có lệnh đứng im <= 0.05)."""
    from training.run_parallel_sim import ParallelSimRunner
    sim = ParallelSimRunner(num_envs=16, stand_still=False)
    try:
        for _ in range(5):
            sim.reset()
            assert np.all(sim.commands[:, 0] >= 0.10), f"Found non-walking cmd_x < 0.10: {sim.commands[:, 0]}"
            assert np.all(sim.commands[:, 0] <= 0.25)
    finally:
        sim.close()






