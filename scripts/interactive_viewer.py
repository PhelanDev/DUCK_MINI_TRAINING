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

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from training.env_config import (
        _DEFAULT_CFG,
        compute_observations,
        scale_actions,
        clip_action_rates,
    )
    DEFAULT_KP = _DEFAULT_CFG.KP
    DEFAULT_KD = _DEFAULT_CFG.KD
    DEFAULT_TORQUE_LIMIT = _DEFAULT_CFG.TORQUE_LIMIT
except Exception:
    _DEFAULT_CFG = None
    DEFAULT_KP = 25.0
    DEFAULT_KD = 0.8
    DEFAULT_TORQUE_LIMIT = 1.0

try:
    from training.terrain_curriculum import apply_stage, STAGES
except Exception:
    apply_stage = None
    STAGES = []

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
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to trained RL policy checkpoint (.pt or .onnx)",
    )
    parser.add_argument(
        "--cmd-vel",
        type=float,
        nargs=3,
        default=[0.2, 0.0, 0.0],
        metavar=("VX", "VY", "WZ"),
        help="Target velocity command [vx (m/s), vy (m/s), wz (rad/s)] for policy",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Terrain curriculum stage (0=flat/wide, 1=low hurdles, 2=med hurdles, 3=high hurdles)",
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
    kp: float = DEFAULT_KP,
    kd: float = DEFAULT_KD,
    torque_limit: float = DEFAULT_TORQUE_LIMIT,
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


def extract_viewer_observation(
    model: Any,
    data: Any,
    cmd_vel: Union[Tuple[float, float, float], List[float], np.ndarray] = (0.2, 0.0, 0.0),
    last_actions: Optional[np.ndarray] = None,
    obs_dim: int = 42,
) -> np.ndarray:
    """Extract observation vector (37 or 42 dimensions) from current MuJoCo simulation data.

    Matches the observation specification of MicroDuck 8-DOF RL environment:
    - Base angular velocity [wx, wy, wz] (3) from gyro sensor (sensordata 3:6)
    - Projected gravity vector [gx, gy, gz] (3) from orientation quaternion (sensordata 6:10)
    - Velocity commands [vx, vy, wz] (3)
    - Joint positions [q1..q8] (8) from encoder sensors (sensordata 12:20)
    - Joint velocities [dq1..dq8] (8) from encoder velocity sensors (sensordata 20:28)
    - Previous actions [a1..a8] (8)
    - Feet contact sensors (4) from touch sensors (sensordata 10, 11, zero padded)
    - Optional LiDAR ranges (5) from rangefinder sensors (sensordata 28:33, clipped to max range)
    """
    if last_actions is None:
        last_actions = np.zeros(8, dtype=np.float32)
    else:
        last_actions = np.asarray(last_actions, dtype=np.float32)

    # 1. Base angular velocity (sensordata 3:6)
    base_ang_vel = np.array(data.sensordata[3:6], dtype=np.float32)

    # 2. Projected gravity from quaternion (sensordata 6:10)
    w, x, y, z = data.sensordata[6:10]
    proj_grav = np.array(
        [
            2.0 * (w * y - x * z),
            -2.0 * (w * x + y * z),
            2.0 * (x * x + y * y) - 1.0,
        ],
        dtype=np.float32,
    )

    # 3. Target velocity commands [vx, vy, wz]
    cmd = np.asarray(cmd_vel, dtype=np.float32)

    # 4. Joint positions & velocities (sensordata 12:28)
    joint_pos = np.array(data.sensordata[12:20], dtype=np.float32)
    joint_vel = np.array(data.sensordata[20:28], dtype=np.float32)

    # 5. Foot contact sensors: left (10), right (11) binarized > 0.1 N, padded to 4 dims
    left_touch = float(data.sensordata[10]) if len(data.sensordata) > 10 else 0.0
    right_touch = float(data.sensordata[11]) if len(data.sensordata) > 11 else 0.0
    foot_contact = np.array(
        [1.0 if left_touch > 0.1 else 0.0, 1.0 if right_touch > 0.1 else 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    # 6. LiDAR ranges (sensordata 28:33)
    if len(data.sensordata) >= 33:
        raw_lidar = np.array(data.sensordata[28:33], dtype=np.float32)
        lidar_ranges = np.where(raw_lidar < 0.0, 2.0, raw_lidar)
    elif obs_dim == 42:
        lidar_ranges = np.full(5, 2.0, dtype=np.float32)
    else:
        lidar_ranges = None

    try:
        return compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_actions,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges if obs_dim == 42 else None,
        )
    except Exception:
        parts = [
            base_ang_vel.ravel(),
            proj_grav.ravel(),
            cmd.ravel(),
            joint_pos.ravel(),
            joint_vel.ravel(),
            last_actions.ravel(),
            foot_contact.ravel(),
        ]
        if obs_dim == 42 and lidar_ranges is not None:
            parts.append(lidar_ranges.ravel())
        return np.concatenate(parts, axis=-1).astype(np.float32)


def load_viewer_policy(
    policy_path: Union[str, Path],
    obs_dim: int = 42,
    action_dim: int = 8,
) -> Any:
    """Load trained RL policy from PyTorch (.pt) or ONNX (.onnx) file.

    Returns:
        torch.nn.Module (eval mode) for PyTorch checkpoints,
        or onnxruntime.InferenceSession for ONNX models.
    """
    p_path = Path(policy_path).resolve()
    if not p_path.exists():
        raise FileNotFoundError(f"Policy file not found at {p_path}")

    suffix = p_path.suffix.lower()
    if suffix == ".onnx":
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required to load .onnx policies. Run: pip install onnxruntime"
            ) from e
        session = ort.InferenceSession(str(p_path), providers=["CPUExecutionProvider"])
        return session

    # Default to PyTorch checkpoint (.pt, .pth, etc.)
    try:
        from scripts.export_policy import load_actor_from_checkpoint
        actor = load_actor_from_checkpoint(
            p_path, obs_dim=obs_dim, action_dim=action_dim
        )
        actor.eval()
        return actor
    except Exception:
        import torch
        from scripts.train_ppo import ActorCritic
        ckpt = torch.load(str(p_path), map_location="cpu", weights_only=False)
        obs_d = ckpt.get("obs_dim", obs_dim) if isinstance(ckpt, dict) else obs_dim
        act_d = ckpt.get("action_dim", action_dim) if isinstance(ckpt, dict) else action_dim
        hid_d = ckpt.get("hidden_dim", 256) if isinstance(ckpt, dict) else 256
        model = ActorCritic(obs_dim=obs_d, action_dim=act_d, hidden_dim=hid_d)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif "actor_state_dict" in ckpt:
                model.actor.load_state_dict(ckpt["actor_state_dict"])
            else:
                model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)
        model.eval()
        return model.actor


def policy_step(policy: Any, obs: np.ndarray) -> np.ndarray:
    """Execute single inference forward pass with observation vector.

    Args:
        policy: Loaded policy (torch.nn.Module or onnxruntime.InferenceSession).
        obs: Observation vector of shape (obs_dim,) or (1, obs_dim).

    Returns:
        np.ndarray: Normalized action vector of shape (8,) bounded in [-1.0, 1.0].
    """
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim == 1:
        obs_arr = obs_arr.reshape(1, -1)

    # Check for ONNX Runtime session
    if hasattr(policy, "run") and hasattr(policy, "get_inputs"):
        input_name = policy.get_inputs()[0].name
        output = policy.run(None, {input_name: obs_arr})[0]
        actions = np.asarray(output, dtype=np.float32).ravel()
    else:
        # PyTorch model
        import torch
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_arr)
            if hasattr(policy, "actor"):
                out = policy.actor(obs_tensor)
            else:
                out = policy(obs_tensor)
            actions = out.cpu().numpy().ravel().astype(np.float32)

    return np.clip(actions, -1.0, 1.0).astype(np.float32)


def run_simulation_headless(
    model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
    pose: str = "stand",
    duration: float = 1.0,
    gait_demo: bool = False,
    policy_path: Optional[Union[str, Path]] = None,
    cmd_vel: Union[Tuple[float, float, float], List[float], np.ndarray] = (0.2, 0.0, 0.0),
    dt: float = 0.002,
    verbose: bool = False,
    stage: int = 0,
) -> Dict[str, Any]:
    """Execute a headless simulation without GUI window for verification and testing."""
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo library is not available in current environment.")

    model_path = str(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = dt
    if apply_stage is not None and stage > 0:
        apply_stage(model, stage)
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

    policy = load_viewer_policy(policy_path) if policy_path else None
    control_dt = 0.02
    decimation = max(1, int(round(control_dt / dt)))
    last_actions = np.zeros(8, dtype=np.float32)
    last_targets = np.array(data.qpos[7:15], dtype=np.float32)
    targets = target_pose

    for step_idx in range(total_steps):
        if policy is not None:
            if step_idx % decimation == 0:
                obs = extract_viewer_observation(
                    model, data, cmd_vel=cmd_vel, last_actions=last_actions, obs_dim=42
                )
                actions = policy_step(policy, obs)
                scaled_targets = scale_actions(actions, _DEFAULT_CFG)
                max_vel = _DEFAULT_CFG.MAX_VELOCITY if _DEFAULT_CFG else 6.0
                targets = clip_action_rates(
                    scaled_targets, last_targets, dt=control_dt, max_velocity=max_vel
                )
                last_targets = targets
                last_actions = actions
        elif gait_demo:
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


def configure_viewer_camera(
    viewer_cam: Any,
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.15),
    distance: float = 1.2,
    azimuth: float = 135.0,
    elevation: float = -20.0,
) -> None:
    """Configure default viewer camera view angles to prevent black screen on hybrid GPUs.

    Explicitly sets focus point, distance, azimuth, and elevation for MuJoCo passive viewer.
    """
    viewer_cam.lookat[:] = list(lookat)
    viewer_cam.distance = distance
    viewer_cam.azimuth = azimuth
    viewer_cam.elevation = elevation


def run_interactive_viewer(
    model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
    pose: str = "stand",
    duration: Optional[float] = None,
    gait_demo: bool = False,
    policy_path: Optional[Union[str, Path]] = None,
    cmd_vel: Union[Tuple[float, float, float], List[float], np.ndarray] = (0.2, 0.0, 0.0),
    dt: float = 0.002,
    stage: int = 0,
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
    if apply_stage is not None and stage > 0:
        apply_stage(model, stage)
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
    if stage > 0 and STAGES and stage < len(STAGES):
        print(f"  Curriculum   : {STAGES[stage].description}")
    elif stage > 0:
        print(f"  Curriculum   : Stage {stage}")
    if policy_path:
        print(f"  Chế độ điều khiển: RL Policy Brain ({policy_path})")
        print(f"  Lệnh vận tốc     : vx={cmd_vel[0]:.2f} m/s, vy={cmd_vel[1]:.2f} m/s, wz={cmd_vel[2]:.2f} rad/s")
    else:
        print(f"  Chế độ bước  : {'Gait Demo (chu kỳ đi bộ sinh động)' if gait_demo else 'Static Pose'}")
    print(f"  Số khớp      : {len(JOINT_NAMES)} (2 đầu, 6 chân)")
    print("  Điều khiển   : PD position control mô phỏng servo MG996R")
    print("=" * 70)

    # Attempt to launch passive GUI viewer
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set explicit camera orientation to prevent black screen / uninitialized view on hybrid GPUs
            configure_viewer_camera(viewer.cam)
            # Ensure geom groups 0, 1, 2, 3 are all active for rendering
            for g_idx in range(4):
                viewer.opt.geomgroup[g_idx] = 1
            viewer.sync()

            start_wall_time = time.time()
            step_count = 0

            policy = load_viewer_policy(policy_path) if policy_path else None
            control_dt = 0.02
            decimation = max(1, int(round(control_dt / dt)))
            last_actions = np.zeros(8, dtype=np.float32)
            last_targets = np.array(data.qpos[7:15], dtype=np.float32)
            targets = target_qpos

            while viewer.is_running():
                step_start = time.time()
                elapsed = time.time() - start_wall_time

                if duration is not None and elapsed >= duration:
                    print(f"\n[THÔNG BÁO] Đã hoàn thành thời gian chạy mô phỏng ({duration}s).")
                    break

                if policy is not None:
                    if step_count % decimation == 0:
                        obs = extract_viewer_observation(
                            model, data, cmd_vel=cmd_vel, last_actions=last_actions, obs_dim=42
                        )
                        actions = policy_step(policy, obs)
                        scaled_targets = scale_actions(actions, _DEFAULT_CFG)
                        max_vel = _DEFAULT_CFG.MAX_VELOCITY if _DEFAULT_CFG else 6.0
                        targets = clip_action_rates(
                            scaled_targets, last_targets, dt=control_dt, max_velocity=max_vel
                        )
                        last_targets = targets
                        last_actions = actions
                elif gait_demo:
                    targets = compute_gait_targets(elapsed)
                else:
                    targets = target_qpos

                apply_pd_control(model, data, targets)
                mujoco.mj_step(model, data)
                step_count += 1
                if step_count % 10 == 0:
                    viewer.sync()
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
            policy_path=policy_path,
            cmd_vel=cmd_vel,
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


def main() -> int:
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
            policy_path=args.policy,
            cmd_vel=tuple(args.cmd_vel),
            dt=args.dt,
            stage=args.stage,
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
            policy_path=args.policy,
            cmd_vel=tuple(args.cmd_vel),
            dt=args.dt,
            stage=args.stage,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
