#!/usr/bin/env python3
"""Interactive 3D Visualizer and Pose Controller for MicroDuck 8-DOF.

Visualizes the MicroDuck 8-DOF robot using `mujoco.viewer` with interactive
joint manipulation, preset poses (standing, squatting, stepping, bowing),
dynamic walking gait demonstration, and headless/CLI fallback support for
headless environments or remote servers.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Reconfigure stdout/stderr encoding to UTF-8 on Windows for safe console printing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False


# Joint names matching MuJoCo MJCF and URDF specifications
JOINT_NAMES = [
    "head_yaw",
    "head_pitch",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
]

# Physical joint range limits for MG996R servo actuators (in radians)
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "head_yaw": (-1.57, 1.57),          # ±90 deg
    "head_pitch": (-0.52, 0.78),        # -30 to +45 deg
    "left_hip_roll": (-0.44, 0.44),     # ±25 deg
    "left_hip_pitch": (-0.87, 0.87),    # ±50 deg
    "left_knee_pitch": (0.0, 1.48),     # 0 to 85 deg
    "right_hip_roll": (-0.44, 0.44),    # ±25 deg
    "right_hip_pitch": (-0.87, 0.87),   # ±50 deg
    "right_knee_pitch": (0.0, 1.48),    # 0 to 85 deg
}

# Standard pose presets for testing stability and kinodynamics
POSE_PRESETS: Dict[str, List[float]] = {
    "stand": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "squat": [0.0, 0.0, 0.0, -0.20, 0.40, 0.0, -0.20, 0.40],
    "left_step": [0.0, 0.0, -0.10, 0.30, 0.40, 0.05, -0.20, 0.10],
    "right_step": [0.0, 0.0, 0.05, -0.20, 0.10, 0.10, 0.30, 0.40],
    "bow": [0.0, 0.35, 0.0, -0.30, 0.30, 0.0, -0.30, 0.30],
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line interface argument parser."""
    parser = argparse.ArgumentParser(
        description="MicroDuck 8-DOF Interactive 3D Viewer & Kinematic Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/mjcf/microduck_8dof.xml",
        help="Path to MicroDuck 8-DOF MJCF model XML file",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="stand",
        choices=list(POSE_PRESETS.keys()),
        help="Nominal pose preset to initialize robot",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation run duration in seconds (None runs indefinitely)",
    )
    parser.add_argument(
        "--gait-demo",
        action="store_true",
        help="Execute dynamic periodic walking gait demonstration",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless simulation mode without opening a GUI window",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal command prompt to manipulate joints",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Physics simulation timestep in seconds",
    )
    return parser


def compute_gait_targets(t: float, frequency: float = 1.2) -> np.ndarray:
    """Generate periodic joint angles for bipedal walking demonstration.
    
    Generates coordinated hip roll (lateral weight shifting), hip pitch (sagittal leg swing),
    and knee pitch (foot ground clearance) trajectories.
    """
    omega = 2.0 * math.pi * frequency
    phase = omega * t

    # Weight shift: lateral hip roll
    roll_amp = 0.12
    l_roll = roll_amp * math.sin(phase)
    r_roll = roll_amp * math.sin(phase)

    # Leg swing: hip pitch
    pitch_amp = 0.25
    l_hip_pitch = pitch_amp * math.sin(phase)
    r_hip_pitch = -pitch_amp * math.sin(phase)

    # Knee flexion: raise swing foot
    knee_bias = 0.20
    knee_amp = 0.25
    l_knee = max(0.0, knee_bias + knee_amp * math.sin(phase))
    r_knee = max(0.0, knee_bias - knee_amp * math.sin(phase))

    # Head counter-motion
    head_yaw = -0.15 * math.sin(phase)
    head_pitch = 0.05 * math.cos(phase * 2.0)

    raw_targets = np.array([
        head_yaw,
        head_pitch,
        l_roll,
        l_hip_pitch,
        l_knee,
        r_roll,
        r_hip_pitch,
        r_knee,
    ], dtype=np.float32)

    # Enforce physical MG996R servo limits
    clamped_targets = np.empty_like(raw_targets)
    for i, jname in enumerate(JOINT_NAMES):
        low, high = JOINT_LIMITS[jname]
        clamped_targets[i] = np.clip(raw_targets[i], low, high)

    return clamped_targets


def apply_pd_control(
    model: Any,
    data: Any,
    target_positions: np.ndarray,
    kp: float = 25.0,
    kd: float = 0.8,
    torque_limit: float = 1.0,
) -> np.ndarray:
    """Compute and apply PD torques to simulate MG996R position servos."""
    current_positions = data.qpos[7:15]
    current_velocities = data.qvel[6:14]

    torques = kp * (target_positions - current_positions) - kd * current_velocities
    clamped_torques = np.clip(torques, -torque_limit, torque_limit)
    data.ctrl[:] = clamped_torques
    return clamped_torques


def get_robot_state(model: Any, data: Any) -> Dict[str, Any]:
    """Extract comprehensive kinodynamic state from MuJoCo data structure."""
    root_pos = np.array(data.qpos[0:3], dtype=np.float32)
    root_quat = np.array(data.qpos[3:7], dtype=np.float32)
    joint_angles = np.array(data.qpos[7:15], dtype=np.float32)
    joint_vels = np.array(data.qvel[6:14], dtype=np.float32)
    joint_torques = np.array(data.ctrl[:], dtype=np.float32)

    # Foot touch contact sensors
    left_touch = float(data.sensordata[10]) if len(data.sensordata) > 10 else 0.0
    right_touch = float(data.sensordata[11]) if len(data.sensordata) > 11 else 0.0

    # Projected gravity
    w, x, y, z = root_quat
    grav_z = 2.0 * (x * x + y * y) - 1.0

    is_stable = bool(
        root_pos[2] > 0.08
        and grav_z < -0.3
        and np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.qvel))
    )

    return {
        "root_pos": root_pos,
        "root_quat": root_quat,
        "grav_z": grav_z,
        "joint_angles": joint_angles,
        "joint_vels": joint_vels,
        "joint_torques": joint_torques,
        "left_touch": left_touch,
        "right_touch": right_touch,
        "is_stable": is_stable,
    }


def run_simulation_headless(
    model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
    pose: str = "stand",
    duration: float = 1.0,
    gait_demo: bool = False,
    dt: float = 0.002,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute a headless simulation without GUI window for verification and testing."""
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo library is not available in current environment.")

    model_path = str(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)

    # Initialize nominal standing pose
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.170]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    target_pose = np.array(POSE_PRESETS.get(pose, POSE_PRESETS["stand"]), dtype=np.float32)
    data.qpos[7:15] = target_pose
    mujoco.mj_forward(model, data)

    total_steps = int(max(1, duration / dt))
    sim_time = 0.0

    for step_idx in range(total_steps):
        if gait_demo:
            targets = compute_gait_targets(sim_time)
        else:
            targets = target_pose

        apply_pd_control(model, data, targets)
        mujoco.mj_step(model, data)
        sim_time += dt

    state = get_robot_state(model, data)
    return {
        "completed_steps": total_steps,
        "sim_time": sim_time,
        "final_root_z": float(state["root_pos"][2]),
        "final_joint_angles": state["joint_angles"],
        "is_stable": state["is_stable"],
        "left_touch": state["left_touch"],
        "right_touch": state["right_touch"],
    }


def run_interactive_viewer(
    model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
    pose: str = "stand",
    duration: Optional[float] = None,
    gait_demo: bool = False,
    dt: float = 0.002,
):
    """Launch passive 3D MuJoCo viewer or fall back to headless/CLI mode."""
    if not _MUJOCO_AVAILABLE:
        print("[LỖI] Không tìm thấy thư viện mujoco. Vui lòng cài đặt: pip install mujoco")
        return

    model_path = str(model_path)
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy tệp mô hình: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)

    # Reset to nominal standing pose
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.170]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    target_qpos = np.array(POSE_PRESETS.get(pose, POSE_PRESETS["stand"]), dtype=np.float32)
    data.qpos[7:15] = target_qpos
    mujoco.mj_forward(model, data)

    print("=" * 70)
    print("  MICRODUCK 8-DOF (MG996R) - TRỰC QUAN HÓA 3D & KIỂM THỬ TƯ THẾ")
    print("=" * 70)
    print(f"  Mô hình      : {model_path}")
    print(f"  Tư thế gốc   : {pose}")
    print(f"  Chế độ bước  : {'Gait Demo (chu kỳ đi bộ sinh động)' if gait_demo else 'Static Pose'}")
    print(f"  Số khớp      : {len(JOINT_NAMES)} (2 đầu, 6 chân)")
    print("  Điều khiển   : PD position control mô phỏng servo MG996R")
    print("=" * 70)

    # Attempt to launch passive GUI viewer
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_wall_time = time.time()
            step_count = 0

            while viewer.is_running():
                step_start = time.time()
                elapsed = time.time() - start_wall_time

                if duration is not None and elapsed >= duration:
                    print(f"\n[THÔNG BÁO] Đã hoàn thành thời gian chạy mô phỏng ({duration}s).")
                    break

                if gait_demo:
                    targets = compute_gait_targets(elapsed)
                else:
                    targets = target_qpos

                apply_pd_control(model, data, targets)
                mujoco.mj_step(model, data)
                viewer.sync()

                step_count += 1
                if step_count % 250 == 0:
                    state = get_robot_state(model, data)
                    status_str = "ỔN ĐỊNH" if state["is_stable"] else "MẤT THĂNG BẰNG"
                    print(
                        f"\r[T={elapsed:5.1f}s] Chiều cao z={state['root_pos'][2]:.3f}m | "
                        f"Trạng thái: {status_str} | Chạm sàn: L={state['left_touch']:.1f} N, R={state['right_touch']:.1f} N",
                        end="",
                        flush=True,
                    )

                # Regulate to real-time playback
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            print("\n[THÔNG BÁO] Đã đóng cửa sổ 3D viewer.")

    except Exception as e:
        print(f"\n[LƯU Ý] Không thể mở giao diện đồ họa 3D ({e}).")
        print("[CHUYỂN ĐỔI] Tự động chuyển sang chế độ mô phỏng không giao diện (Headless CLI Mode)...")
        test_duration = duration if duration is not None else 3.0
        metrics = run_simulation_headless(
            model_path=model_path,
            pose=pose,
            duration=test_duration,
            gait_demo=gait_demo,
            dt=dt,
        )
        print(f"  Tổng bước chạy      : {metrics['completed_steps']}")
        print(f"  Thời gian mô phỏng  : {metrics['sim_time']:.2f} s")
        print(f"  Độ cao thân cuối    : {metrics['final_root_z']:.4f} m (mức chuẩn ~0.17m)")
        print(f"  Độ vững vật lý      : {'ĐẠT (Thăng bằng)' if metrics['is_stable'] else 'KHÔNG ĐẠT'}")
        print(f"  Lực tiếp xúc sàn    : Chân trái = {metrics['left_touch']:.2f} N, Chân phải = {metrics['right_touch']:.2f} N")


def run_interactive_cli(
    model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
    dt: float = 0.002,
):
    """Run interactive text-based terminal prompt for manual joint testing."""
    if not _MUJOCO_AVAILABLE:
        print("[LỖI] Thư viện mujoco chưa được cài đặt.")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = dt
    data = mujoco.MjData(model)

    # Nominal reset
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.170]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    current_targets = np.zeros(8, dtype=np.float32)

    print("\n" + "=" * 60)
    print("  MICRODUCK 8-DOF - ĐIỀU KHIỂN KHỚP TƯƠNG TÁC DÒNG LỆNH")
    print("=" * 60)
    print("Các lệnh khả dụng:")
    print("  pose <stand|squat|left_step|right_step|bow> : Đặt tư thế mẫu")
    print("  set <tên_khớp> <góc_radian>                 : Đặt góc cho khớp")
    print("  status                                      : In trạng thái CoM và các khớp")
    print("  step <số_bước>                              : Chạy n bước mô phỏng")
    print("  reset                                       : Đặt lại vị trí robot")
    print("  help                                        : In trợ giúp")
    print("  quit / exit                                 : Thoát chương trình")
    print("=" * 60)

    while True:
        try:
            cmd_line = input("\n[MicroDuck-CLI] >> ").strip()
            if not cmd_line:
                continue

            parts = cmd_line.split()
            cmd = parts[0].lower()

            if cmd in ["quit", "exit", "q"]:
                print("Tạm biệt!")
                break

            elif cmd == "pose":
                if len(parts) < 2 or parts[1] not in POSE_PRESETS:
                    print(f"Tư thế hợp lệ: {', '.join(POSE_PRESETS.keys())}")
                    continue
                preset_name = parts[1]
                current_targets = np.array(POSE_PRESETS[preset_name], dtype=np.float32)
                for _ in range(100):
                    apply_pd_control(model, data, current_targets)
                    mujoco.mj_step(model, data)
                print(f"[OK] Đã chuyển sang tư thế '{preset_name}'.")

            elif cmd == "set":
                if len(parts) < 3:
                    print("Cú pháp: set <tên_khớp> <góc_radian>")
                    continue
                jname = parts[1]
                if jname not in JOINT_NAMES:
                    print(f"Khớp không hợp lệ. Danh sách: {', '.join(JOINT_NAMES)}")
                    continue
                try:
                    val = float(parts[2])
                except ValueError:
                    print("Giá trị góc phải là số thực.")
                    continue

                low, high = JOINT_LIMITS[jname]
                if not (low <= val <= high):
                    print(f"[CẢNH BÁO] Góc {val:.3f} vượt quá giới hạn [{low}, {high}]. Đang kẹp lại.")
                    val = float(np.clip(val, low, high))

                idx = JOINT_NAMES.index(jname)
                current_targets[idx] = val
                for _ in range(100):
                    apply_pd_control(model, data, current_targets)
                    mujoco.mj_step(model, data)
                print(f"[OK] Đã đặt khớp '{jname}' = {val:.3f} rad ({math.degrees(val):.1f}°).")

            elif cmd == "status":
                state = get_robot_state(model, data)
                print("-" * 50)
                print(f"Tọa độ gốc (root CoM) : x={state['root_pos'][0]:.3f}, y={state['root_pos'][1]:.3f}, z={state['root_pos'][2]:.3f} m")
                print(f"Độ vững              : {'ỔN ĐỊNH' if state['is_stable'] else 'MẤT THĂNG BẰNG'}")
                print(f"Tiếp xúc sàn         : Trái={state['left_touch']:.1f} N | Phải={state['right_touch']:.1f} N")
                print("Góc các khớp hiện tại:")
                for i, name in enumerate(JOINT_NAMES):
                    ang = state["joint_angles"][i]
                    tgt = current_targets[i]
                    print(f"  - {name:<17}: góc={ang:+.3f} rad ({math.degrees(ang):+5.1f}°) [mục tiêu={tgt:+.3f}]")
                print("-" * 50)

            elif cmd == "step":
                steps = int(parts[1]) if len(parts) > 1 else 100
                for _ in range(steps):
                    apply_pd_control(model, data, current_targets)
                    mujoco.mj_step(model, data)
                print(f"[OK] Đã mô phỏng {steps} bước ({steps * dt:.3f}s).")

            elif cmd == "reset":
                mujoco.mj_resetData(model, data)
                data.qpos[0:3] = [0.0, 0.0, 0.170]
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                current_targets = np.zeros(8, dtype=np.float32)
                mujoco.mj_forward(model, data)
                print("[OK] Đã đặt lại toàn bộ trạng thái robot về mặc định.")

            elif cmd == "help":
                print("Lệnh: pose <tên>, set <khớp> <rad>, status, step <n>, reset, quit")

            else:
                print(f"Không nhận diện được lệnh '{cmd}'. Gõ 'help' để xem danh sách lệnh.")

        except (KeyboardInterrupt, EOFError):
            print("\nĐã hủy.")
            break


def main():
    """Main CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.interactive:
        run_interactive_cli(model_path=args.model, dt=args.dt)
    elif args.headless:
        print("[CHẾ ĐỘ HEADLESS] Đang chạy kiểm thử mô phỏng không giao diện...")
        run_duration = args.duration if args.duration is not None else 2.0
        metrics = run_simulation_headless(
            model_path=args.model,
            pose=args.pose,
            duration=run_duration,
            gait_demo=args.gait_demo,
            dt=args.dt,
        )
        print("Kết quả mô phỏng:")
        print(f"  - Tổng số bước      : {metrics['completed_steps']}")
        print(f"  - Thời gian mô phỏng: {metrics['sim_time']:.2f} s")
        print(f"  - Độ cao CoM cuối   : {metrics['final_root_z']:.4f} m")
        print(f"  - Trạng thái vững   : {'ỔN ĐỊNH' if metrics['is_stable'] else 'MẤT CÂN BẰNG'}")
        print(f"  - Lực tiếp xúc chân : Trái={metrics['left_touch']:.2f} N, Phải={metrics['right_touch']:.2f} N")
    else:
        run_interactive_viewer(
            model_path=args.model,
            pose=args.pose,
            duration=args.duration,
            gait_demo=args.gait_demo,
            dt=args.dt,
        )


if __name__ == "__main__":
    main()
