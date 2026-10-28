"""Reinforcement Learning Environment Configuration for MicroDuck 8-DOF (MG996R).

This module defines observation/action space dimensions, joint boundaries,
observation serialization, action scaling, velocity rate limiting,
and balanced reward calculations for bipedal locomotion.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class MicroDuck8DoFConfig:
    """Configuration dataclass for MicroDuck 8-DOF robot."""

    OBS_DIM: int = 37
    ACTION_DIM: int = 8

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

    # Obstacle avoidance parameters
    OBSTACLE_THRESHOLD: float = 0.35  # meters
    LIDAR_MAX_RANGE: float = 2.0  # meters

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
            "sigma_obs": 0.3,
            "foot_clearance": 2.0,
            "stumble": 0.5,
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


# Module-level cached configuration for high-frequency simulation steps
_DEFAULT_CFG = MicroDuck8DoFConfig()


def compute_observations(
    base_ang_vel: np.ndarray,
    proj_grav: np.ndarray,
    commands: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_actions: np.ndarray,
    foot_contact: np.ndarray,
    lidar_ranges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pack raw state tensors into observation vector (37-dim nominal, or 42-dim with 5-ray LiDAR).

    Dimensions:
    - Base angular velocity (3)
    - Projected gravity vector (3)
    - Velocity commands [vx, vy, wz] (3)
    - Joint positions (8)
    - Joint velocities (8)
    - Previous actions (8)
    - Feet contact sensors (4)
    - Optional LiDAR rangefinder rays (5)
    Total = 37 dims (or 42 dims).
    """
    parts = [
        np.asarray(base_ang_vel, dtype=np.float32).ravel(),
        np.asarray(proj_grav, dtype=np.float32).ravel(),
        np.asarray(commands, dtype=np.float32).ravel(),
        np.asarray(joint_pos, dtype=np.float32).ravel(),
        np.asarray(joint_vel, dtype=np.float32).ravel(),
        np.asarray(last_actions, dtype=np.float32).ravel(),
        np.asarray(foot_contact, dtype=np.float32).ravel(),
    ]
    if lidar_ranges is not None:
        parts.append(np.asarray(lidar_ranges, dtype=np.float32).ravel())

    obs = np.concatenate(parts, axis=-1).astype(np.float32)

    expected_dim = 42 if lidar_ranges is not None else 37
    assert obs.shape == (expected_dim,), f"Expected observation shape ({expected_dim},), got {obs.shape}"
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

    # Conjugate quaternion: q_inv = [w, -x, -y, -z]
    qw = q_arr[..., 0:1]
    qv = -q_arr[..., 1:4]

    # t = 2 * (qv x v)
    t = 2.0 * np.cross(qv, v_arr)
    # v_body = v + qw * t + (qv x t)
    return (v_arr + qw * t + np.cross(qv, t)).astype(np.float32)


def ang_vel_mask(d_min: Union[float, np.ndarray]) -> np.ndarray:
    """Mask tracking_ang_vel when close to obstacles to allow robot to turn away freely.

    mask = clip((d_min - 0.25) / (0.55 - 0.25), 0.1, 1.0)
    """
    d = np.asarray(d_min, dtype=np.float32)
    return np.clip((d - 0.25) / (0.55 - 0.25), 0.1, 1.0).astype(np.float32)


def avoid_vel_penalty(
    d_min: Union[float, np.ndarray],
    d_min_dot: Union[float, np.ndarray],
    sigma_obs: float = 0.3,
) -> np.ndarray:
    """Penalize approaching obstacles with velocity towards them.

    avoid_vel_penalty = -max(0, -d_min_dot) * exp(-d_min / sigma_obs)
    """
    d = np.asarray(d_min, dtype=np.float32)
    d_dot = np.asarray(d_min_dot, dtype=np.float32)
    approach_speed = np.maximum(0.0, -d_dot)
    penalty = -approach_speed * np.exp(-d / sigma_obs)
    return penalty.astype(np.float32)


def compute_foot_clearance_and_stumble(
    foot_z: np.ndarray,
    foot_vel_xy: np.ndarray,
    foot_contact_forces: np.ndarray,
    target_height: float = 0.06,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute foot clearance reward (ETH style) and stumble penalty.

    Args:
        foot_z: Height of feet [left, right], shape (..., 2)
        foot_vel_xy: Horizontal velocity of feet, shape (..., 2, 2)
        foot_contact_forces: 3D contact forces [Fx, Fy, Fz], shape (..., 2, 3)
        target_height: Desired swing apex height (m), default 0.06

    Returns:
        rew_clearance: Clearance reward (<= 0, penalizes low swing foot height), shape (...)
        pen_stumble: Stumble penalty (<= 0, penalizes excessive lateral impact force), shape (...)
    """
    z = np.asarray(foot_z, dtype=np.float32)
    vel = np.asarray(foot_vel_xy, dtype=np.float32)
    forces = np.asarray(foot_contact_forces, dtype=np.float32)

    is_batch = z.ndim > 1

    # Normal contact force magnitude along Z
    fz = np.abs(forces[..., 2])
    is_swing = (fz < 1.0).astype(np.float32)

    # Foot horizontal speed clipped at 1.0 m/s
    vel_xy_norm = np.clip(np.linalg.norm(vel, axis=-1), 0.0, 1.0)

    # ETH clearance penalty: - sum_feet (z - target_height)^2 * sqrt(||v_xy||) * is_swing
    height_err_sq = np.square(z - target_height)
    rew_clearance = -np.sum(height_err_sq * np.sqrt(vel_xy_norm) * is_swing, axis=-1)

    # Stumble detection: lateral impact force f_xy > 2 * |f_z| AND f_xy > 2.0 N
    f_xy = np.linalg.norm(forces[..., :2], axis=-1)
    stumble_mask = (f_xy > 2.0 * fz) & (f_xy > 2.0)
    pen_stumble = -np.sum(stumble_mask.astype(np.float32), axis=-1)

    if not is_batch:
        return float(rew_clearance), float(pen_stumble)
    return rew_clearance.astype(np.float32), pen_stumble.astype(np.float32)


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
    d_min: Optional[Union[float, np.ndarray]] = None,
    d_min_dot: Optional[Union[float, np.ndarray]] = None,
    foot_z: Optional[np.ndarray] = None,
    foot_vel_xy: Optional[np.ndarray] = None,
    foot_contact_forces: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute balanced reward terms and total reward for RL training.

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
    if d_min is not None:
        r_track_ang_vel = r_track_ang_vel * ang_vel_mask(d_min)

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
    if d_min is not None and d_min_dot is not None:
        sigma = weights.get("sigma_obs", 0.3)
        r_obstacle = avoid_vel_penalty(d_min, d_min_dot, sigma_obs=sigma)
    elif lidar_ranges is not None:
        ranges = np.asarray(lidar_ranges, dtype=np.float32)
        min_dist = np.min(ranges, axis=-1)
        r_obstacle = np.where(
            min_dist < _DEFAULT_CFG.OBSTACLE_THRESHOLD,
            min_dist - _DEFAULT_CFG.OBSTACLE_THRESHOLD,
            0.0,
        ).astype(np.float32)
    else:
        r_obstacle = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    # 9. Foot clearance reward & stumble penalty
    if foot_z is not None and foot_vel_xy is not None and foot_contact_forces is not None:
        target_h = weights.get("target_foot_height", 0.06)
        r_clearance, r_stumble = compute_foot_clearance_and_stumble(
            foot_z=foot_z,
            foot_vel_xy=foot_vel_xy,
            foot_contact_forces=foot_contact_forces,
            target_height=target_h,
        )
    else:
        r_clearance = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0
        r_stumble = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    total = (
        weights.get("tracking_lin_vel", 1.0) * r_track_lin_vel
        + weights.get("tracking_ang_vel", 0.5) * r_track_ang_vel
        + weights.get("upright", 1.0) * r_upright
        + weights.get("torque_penalty", -0.001) * r_torque_penalty
        + weights.get("joint_vel_penalty", -0.0001) * r_joint_vel_penalty
        + weights.get("action_rate_penalty", -0.01) * r_action_rate_penalty
        + weights.get("foot_contact", 0.1) * r_foot_contact
        + weights.get("obstacle_avoidance", 0.2) * r_obstacle
        + weights.get("foot_clearance", 2.0) * r_clearance
        + weights.get("stumble", 0.5) * r_stumble
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
            "foot_clearance": float(r_clearance),
            "rew_foot_clearance": float(r_clearance),
            "stumble": float(r_stumble),
            "pen_stumble": float(r_stumble),
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
        "foot_clearance": r_clearance.astype(np.float32),
        "rew_foot_clearance": r_clearance.astype(np.float32),
        "stumble": r_stumble.astype(np.float32),
        "pen_stumble": r_stumble.astype(np.float32),
    }
