"""Reinforcement Learning Environment Configuration for MicroDuck 8-DOF (Dual ESP32-S3 Stack).

Defines 42-dimensional perception observation space, 8-DOF action scaling,
rate limiting, obstacle avoidance rewards, and INT8 quantization utilities
for running lightweight policies on ESP32-S3 Xtensa LX7 PIE vector hardware.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class MicroDuck8DoFConfig:
    """Configuration dataclass for MicroDuck 8-DOF on Dual ESP32-S3 Stack."""

    STACK_NAME: str = "dual_esp32s3_stack"
    TARGET_MASS: float = 0.780  # kg (~780g total weight)
    POLICY_DTYPE: str = "int8"  # Quantized for ESP32-S3 PIE vector instructions

    OBS_DIM: int = 42
    ACTION_DIM: int = 8
    LIDAR_RAYS: int = 5
    LIDAR_MIN_RANGE: float = 0.05
    LIDAR_MAX_RANGE: float = 2.0
    OBSTACLE_THRESHOLD: float = 0.35  # meters

    JOINT_NAMES: List[str] = field(
        default_factory=lambda: [
            "head_yaw",
            "head_pitch",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee_pitch",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee_pitch",
        ]
    )

    # Physical limits for MG996R servo joints (radians)
    JOINT_LIMITS: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "head_yaw": (-1.57, 1.57),
            "head_pitch": (-0.52, 0.78),
            "left_hip_roll": (-0.44, 0.44),
            "left_hip_pitch": (-0.87, 0.87),
            "left_knee_pitch": (0.0, 1.48),
            "right_hip_roll": (-0.44, 0.44),
            "right_hip_pitch": (-0.87, 0.87),
            "right_knee_pitch": (0.0, 1.48),
        }
    )

    # MG996R specifications
    TORQUE_LIMIT: float = 1.0  # N.m (stall torque rating)
    MAX_VELOCITY: float = 6.0  # rad/s (~344 deg/s)
    DEFAULT_DT: float = 0.02  # 50 Hz control loop frequency

    # PD controller gains for simulated MG996R position servos
    KP: float = 25.0  # Joint position proportional gain (N*m/rad)
    KD: float = 0.8  # Joint velocity derivative gain (N*m*s/rad)

    # Balanced reward weights
    REWARD_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "upright": 1.0,
            "torque_penalty": -0.001,
            "joint_vel_penalty": -0.0001,
            "action_rate_penalty": -0.01,
            "foot_contact": 0.1,
            "obstacle_avoidance": 0.2,
            "stair_ascent": 1.5,
        }
    )

    def __post_init__(self):
        self.low_limits = np.array(
            [self.JOINT_LIMITS[name][0] for name in self.JOINT_NAMES],
            dtype=np.float32,
        )
        self.high_limits = np.array(
            [self.JOINT_LIMITS[name][1] for name in self.JOINT_NAMES],
            dtype=np.float32,
        )


# Module-level cached configuration for simulation steps
_DEFAULT_CFG = MicroDuck8DoFConfig()


def compute_observations(
    base_ang_vel: np.ndarray,
    proj_grav: np.ndarray,
    commands: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_actions: np.ndarray,
    foot_contact: np.ndarray,
    lidar_ranges: np.ndarray,
) -> np.ndarray:
    """Pack raw state tensors into standard 42-dimensional observation vector.

    Dimensions:
    - Base angular velocity (3)
    - Projected gravity vector (3)
    - Velocity commands [vx, vy, wz] (3)
    - Joint positions (8)
    - Joint velocities (8)
    - Previous actions (8)
    - Feet contact sensors (4)
    - 5-ray LiDAR rangefinder distances (5)
    Total = 42 dimensions.
    """
    obs = np.concatenate(
        [
            np.asarray(base_ang_vel, dtype=np.float32).ravel(),
            np.asarray(proj_grav, dtype=np.float32).ravel(),
            np.asarray(commands, dtype=np.float32).ravel(),
            np.asarray(joint_pos, dtype=np.float32).ravel(),
            np.asarray(joint_vel, dtype=np.float32).ravel(),
            np.asarray(last_actions, dtype=np.float32).ravel(),
            np.asarray(foot_contact, dtype=np.float32).ravel(),
            np.asarray(lidar_ranges, dtype=np.float32).ravel(),
        ],
        axis=-1,
    ).astype(np.float32)

    assert obs.shape == (42,), f"Expected observation shape (42,), got {obs.shape}"
    return obs


def scale_actions(
    raw_actions: np.ndarray,
    cfg: Optional[MicroDuck8DoFConfig] = None,
) -> np.ndarray:
    """Scale raw normalized actions [-1, 1] into physical joint target angles.

    Safely clips outputs to respect the hardware limits of each MG996R joint.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG

    actions = np.asarray(raw_actions, dtype=np.float32)
    clipped_actions = np.clip(actions, -1.0, 1.0)

    # Linear mapping: raw -1.0 -> low_limit, +1.0 -> high_limit
    scaled = cfg.low_limits + (clipped_actions + 1.0) * 0.5 * (
        cfg.high_limits - cfg.low_limits
    )
    return np.clip(scaled, cfg.low_limits, cfg.high_limits).astype(np.float32)


def clip_action_rates(
    current_targets: np.ndarray,
    last_targets: np.ndarray,
    dt: float = 0.02,
    max_velocity: float = 6.0,
) -> np.ndarray:
    """Limit rate of angle target change to prevent exceeding max servo velocity.

    max_delta = max_velocity * dt.
    """
    current = np.asarray(current_targets, dtype=np.float32)
    last = np.asarray(last_targets, dtype=np.float32)

    max_delta = float(max_velocity * dt)
    delta = np.clip(current - last, -max_delta, max_delta)
    return (last + delta).astype(np.float32)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v from World Frame into Body Frame using quaternion q = [w, x, y, z].

    Supports both single vector/quaternion (shape (3,) / (4,)) and
    batched arrays (shape (N, 3) / (N, 4)).
    """
    q_arr = np.asarray(q, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)

    qw = q_arr[..., 0:1]
    qv = -q_arr[..., 1:4]

    t = 2.0 * np.cross(qv, v_arr)
    return (v_arr + qw * t + np.cross(qv, t)).astype(np.float32)


def compute_rewards(
    base_lin_vel: np.ndarray,
    cmd_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    cmd_ang_vel: Union[float, np.ndarray],
    proj_gravity: np.ndarray,
    joint_torques: np.ndarray,
    joint_vel: np.ndarray,
    foot_contact: np.ndarray,
    lidar_ranges: Optional[np.ndarray] = None,
    actions: Optional[np.ndarray] = None,
    last_actions: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    root_x: Optional[Union[float, np.ndarray]] = None,
    root_z: Optional[Union[float, np.ndarray]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute balanced reward terms and total reward for 42-dim RL training.

    Supports both single environment (1D arrays -> float rewards) and
    batched environments (2D arrays of shape (N, ...) -> 1D arrays of shape (N,)).
    """
    if weights is None:
        weights = _DEFAULT_CFG.REWARD_WEIGHTS

    base_lin = np.asarray(base_lin_vel, dtype=np.float32)
    cmd_lin = np.asarray(cmd_lin_vel, dtype=np.float32)
    base_ang = np.asarray(base_ang_vel, dtype=np.float32)
    grav = np.asarray(proj_gravity, dtype=np.float32)
    torques = np.asarray(joint_torques, dtype=np.float32)
    j_vel = np.asarray(joint_vel, dtype=np.float32)
    f_contact = np.asarray(foot_contact, dtype=np.float32)

    is_batch = base_lin.ndim > 1

    # Target yaw rate
    if is_batch:
        if isinstance(cmd_ang_vel, np.ndarray) and cmd_ang_vel.ndim > 1:
            target_yaw = cmd_ang_vel[:, 2] if cmd_ang_vel.shape[-1] >= 3 else cmd_ang_vel[:, 0]
        elif isinstance(cmd_ang_vel, np.ndarray) and cmd_ang_vel.shape[0] == base_lin.shape[0]:
            target_yaw = cmd_ang_vel
        else:
            target_yaw = float(cmd_ang_vel)
    else:
        if hasattr(cmd_ang_vel, "__len__"):
            cmd_ang_arr = np.asarray(cmd_ang_vel, dtype=np.float32).ravel()
            target_yaw = float(cmd_ang_arr[2] if len(cmd_ang_arr) >= 3 else cmd_ang_arr[0])
        else:
            target_yaw = float(cmd_ang_vel)

    # 1. Linear velocity tracking in XY plane (exponential kernel)
    lin_err_sq = np.sum(np.square(base_lin[..., :2] - cmd_lin[..., :2]), axis=-1)
    r_track_lin_vel = np.exp(-lin_err_sq / 0.25)

    # 2. Angular velocity tracking around Z axis
    ang_err_sq = np.square(base_ang[..., 2] - target_yaw)
    r_track_ang_vel = np.exp(-ang_err_sq / 0.25)

    # 3. Upright posture reward (gravity projection along body Z should be -1.0)
    grav_err_sq = (
        np.square(grav[..., 0])
        + np.square(grav[..., 1])
        + np.square(grav[..., 2] - (-1.0))
    )
    r_upright = np.exp(-grav_err_sq / 0.2)

    # 4. Energy and torque minimization penalty
    r_torque_penalty = np.sum(np.square(torques), axis=-1)

    # 5. Joint velocity smoothing penalty
    r_joint_vel_penalty = np.sum(np.square(j_vel), axis=-1)

    # 6. Action rate penalty (protects MG996R gears against jerky policy outputs)
    if actions is not None and last_actions is not None:
        act = np.asarray(actions, dtype=np.float32)
        last_act = np.asarray(last_actions, dtype=np.float32)
        r_action_rate_penalty = np.sum(np.square(act - last_act), axis=-1)
    else:
        r_action_rate_penalty = (
            np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0
        )

    # 7. Foot contact reward (khuyến khích nhấc chân luân phiên khi di chuyển)
    num_contacts = np.sum(f_contact > 0.0, axis=-1)
    cmd_speed = np.linalg.norm(cmd_lin[..., :2], axis=-1) + np.abs(target_yaw)
    is_moving_cmd = cmd_speed > 0.05
    r_foot_contact = np.where(
        is_moving_cmd,
        (num_contacts == 1).astype(np.float32),
        (num_contacts == 2).astype(np.float32) * 0.5,
    ).astype(np.float32)

    # 8. Obstacle avoidance reward / penalty
    if lidar_ranges is not None:
        ranges = np.asarray(lidar_ranges, dtype=np.float32)
        min_dist = np.min(ranges, axis=-1)
        r_obstacle = np.where(
            min_dist < _DEFAULT_CFG.OBSTACLE_THRESHOLD,
            min_dist - _DEFAULT_CFG.OBSTACLE_THRESHOLD,
            0.0,
        ).astype(np.float32)
    else:
        r_obstacle = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    # 9. Stair ascent elevation reward (x >= 1.35m)
    if root_x is not None and root_z is not None:
        rx = np.asarray(root_x, dtype=np.float32)
        rz = np.asarray(root_z, dtype=np.float32)
        stair_mask = (rx >= 1.35).astype(np.float32)
        elevation_gain = np.clip(rz - 0.170, 0.0, 0.050)
        r_stair_ascent = stair_mask * elevation_gain * 10.0
    else:
        r_stair_ascent = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    total = (
        weights.get("tracking_lin_vel", 1.0) * r_track_lin_vel
        + weights.get("tracking_ang_vel", 0.5) * r_track_ang_vel
        + weights.get("upright", 1.0) * r_upright
        + weights.get("torque_penalty", -0.001) * r_torque_penalty
        + weights.get("joint_vel_penalty", -0.0001) * r_joint_vel_penalty
        + weights.get("action_rate_penalty", -0.01) * r_action_rate_penalty
        + weights.get("foot_contact", 0.1) * r_foot_contact
        + weights.get("obstacle_avoidance", 0.2) * r_obstacle
        + weights.get("stair_ascent", 0.0) * r_stair_ascent
    )

    if not is_batch:
        return {
            "total_reward": float(total),
            "tracking_lin_vel": float(r_track_lin_vel),
            "tracking_ang_vel": float(r_track_ang_vel),
            "upright": float(r_upright),
            "torque_penalty": float(r_torque_penalty),
            "joint_vel_penalty": float(r_joint_vel_penalty),
            "action_rate_penalty": float(r_action_rate_penalty),
            "foot_contact": float(r_foot_contact),
            "obstacle_avoidance": float(r_obstacle),
            "stair_ascent": float(r_stair_ascent),
        }

    return {
        "total_reward": total.astype(np.float32),
        "tracking_lin_vel": r_track_lin_vel.astype(np.float32),
        "tracking_ang_vel": r_track_ang_vel.astype(np.float32),
        "upright": r_upright.astype(np.float32),
        "torque_penalty": r_torque_penalty.astype(np.float32),
        "joint_vel_penalty": r_joint_vel_penalty.astype(np.float32),
        "action_rate_penalty": r_action_rate_penalty.astype(np.float32),
        "foot_contact": r_foot_contact.astype(np.float32),
        "obstacle_avoidance": r_obstacle.astype(np.float32),
        "stair_ascent": r_stair_ascent.astype(np.float32),
    }


# ==============================================================================
# INT8 Quantization Utilities for ESP32-S3 (Xtensa LX7 PIE Vector Accelerator)
# ==============================================================================

# Physical maximum normalization bounds for 42-dimensional observation vector
# Prevents clipping distortion / saturation when physical values exceed 1.0
DEFAULT_OBS_BOUNDS: np.ndarray = np.array(
    [
        # [0..2] Base angular velocity (rad/s, max ~6.0)
        6.0, 6.0, 6.0,
        # [3..5] Projected gravity (unit vector in [-1, 1])
        1.0, 1.0, 1.0,
        # [6..8] Commanded velocity [vx, vy, wz] (m/s, rad/s)
        1.0, 1.0, 1.0,
        # [9..16] Joint positions (rad) - physical travel limits of 8 MG996R servos
        1.57, 0.78, 0.44, 0.87, 1.48, 0.44, 0.87, 1.48,
        # [17..24] Joint velocities (rad/s, max ~6.0)
        6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
        # [25..32] Last actions (normalized target in [-1, 1])
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        # [33..36] Foot contact sensors (binary 0 or 1)
        1.0, 1.0, 1.0, 1.0,
        # [37..41] 5-ray LiDAR rangefinder distances (m, max range 2.0m)
        2.0, 2.0, 2.0, 2.0, 2.0,
    ],
    dtype=np.float32,
)

DEFAULT_OBS_SCALES: np.ndarray = (127.0 / DEFAULT_OBS_BOUNDS).astype(np.float32)


def quantize_obs_int8(
    obs: np.ndarray,
    scale: Union[float, np.ndarray] = 127.0,
    obs_bounds: Optional[Union[float, np.ndarray]] = DEFAULT_OBS_BOUNDS,
    normalize: bool = True,
) -> np.ndarray:
    """Quantize float32 observation vector(s) into symmetric INT8 [-128, 127].

    To prevent saturation/clipping distortion on physical quantities exceeding 1.0
    (e.g. angular velocity up to 6.0 rad/s, LiDAR up to 2.0m, joint angles up to 1.57 rad),
    each observation component is normalized by its physical bound before scaling.

    Supports both 1D (42,) and 2D (N, 42) arrays.
    """
    obs_float = np.asarray(obs, dtype=np.float32)
    if normalize and obs_bounds is not None:
        bounds = np.asarray(obs_bounds, dtype=np.float32)
        norm_obs = np.clip(obs_float / bounds, -1.0, 1.0)
    else:
        norm_obs = obs_float
    scaled = np.round(norm_obs * scale)
    return np.clip(scaled, -128.0, 127.0).astype(np.int8)


def dequantize_obs_int8(
    obs_int8: np.ndarray,
    scale: Union[float, np.ndarray] = 127.0,
    obs_bounds: Optional[Union[float, np.ndarray]] = DEFAULT_OBS_BOUNDS,
) -> np.ndarray:
    """Dequantize INT8 observation vector(s) back to physical float32 values."""
    arr = np.asarray(obs_int8, dtype=np.float32)
    norm = arr / scale
    if obs_bounds is not None:
        bounds = np.asarray(obs_bounds, dtype=np.float32)
        return (norm * bounds).astype(np.float32)
    return norm.astype(np.float32)


def dequantize_act_int8(
    act_int8: np.ndarray,
    scale: float = 127.0,
) -> np.ndarray:
    """Dequantize INT8 policy actions [-128, 127] back to float32 targets in [-1.0, 1.0].

    Supports both 1D (8,) and 2D (N, 8) arrays.
    """
    act_arr = np.asarray(act_int8, dtype=np.float32)
    return np.clip(act_arr / scale, -1.0, 1.0).astype(np.float32)


def generate_esp32_quant_header(
    scale_obs: float = 127.0,
    scale_act: float = 127.0,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate C/C++ header for ESP32-S3 firmware with dimensions and quantization scales."""
    bounds_formatted = ", ".join(f"{b:.2f}f" for b in DEFAULT_OBS_BOUNDS)
    scales_formatted = ", ".join(f"{s:.4f}f" for s in (float(scale_obs) / DEFAULT_OBS_BOUNDS))
    header_content = f"""#ifndef ESP32_QUANT_CONFIG_H
#define ESP32_QUANT_CONFIG_H

/**
 * MicroDuck 8-DOF (Dual ESP32-S3 Stack) - INT8 Quantization & Dimension Config
 * Automatically generated for Xtensa LX7 PIE Vector Accelerator.
 */

#define OBS_DIM 42
#define ACTION_DIM 8
#define LIDAR_RAYS 5

#define OBS_SCALE {float(scale_obs):.1f}f
#define ACT_SCALE {float(scale_act):.1f}f
#define OBS_INV_SCALE (1.0f / {float(scale_obs):.1f}f)
#define ACT_INV_SCALE (1.0f / {float(scale_act):.1f}f)

// Physical maximum normalization bounds for 42 observation dimensions
static const float OBS_MAX_BOUNDS[OBS_DIM] = {{
    {bounds_formatted}
}};

// Effective INT8 scale per observation channel: (scale_obs / OBS_MAX_BOUNDS[i])
static const float OBS_CHANNEL_SCALES[OBS_DIM] = {{
    {scales_formatted}
}};

// Servo Joint Angle Boundaries for MG996R (Radians)
#define HEAD_YAW_MIN -1.57f
#define HEAD_YAW_MAX 1.57f
#define HEAD_PITCH_MIN -0.52f
#define HEAD_PITCH_MAX 0.78f
#define LEFT_HIP_ROLL_MIN -0.44f
#define LEFT_HIP_ROLL_MAX 0.44f
#define LEFT_HIP_PITCH_MIN -0.87f
#define LEFT_HIP_PITCH_MAX 0.87f
#define LEFT_KNEE_PITCH_MIN 0.0f
#define LEFT_KNEE_PITCH_MAX 1.48f
#define RIGHT_HIP_ROLL_MIN -0.44f
#define RIGHT_HIP_ROLL_MAX 0.44f
#define RIGHT_HIP_PITCH_MIN -0.87f
#define RIGHT_HIP_PITCH_MAX 0.87f
#define RIGHT_KNEE_PITCH_MIN 0.0f
#define RIGHT_KNEE_PITCH_MAX 1.48f

#endif // ESP32_QUANT_CONFIG_H
"""
    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(header_content, encoding="utf-8")

    return header_content
