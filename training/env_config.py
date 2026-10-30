"""Reinforcement Learning Environment Configuration for MicroDuck 8-DOF (MG996R).

This module defines observation/action space dimensions, joint boundaries,
observation serialization, action scaling, velocity rate limiting,
and balanced reward calculations for bipedal locomotion.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class EmotionState(IntEnum):
    """10-state emotion model for MicroDuck 8-DOF head expressions."""

    NEUTRAL = 0
    HAPPY = 1
    SAD = 2
    ANGRY = 3
    SCARED = 4
    CURIOUS = 5
    SURPRISED = 6
    SLEEPY = 7
    PLAYFUL = 8
    PROUD = 9


class LocomotionMode(IntEnum):
    """Locomotion operational mode for MicroDuck 8-DOF."""

    WALK = 0
    SIT = 1
    LIE = 2


TARGET_SIT_HEIGHT: float = 0.075
TARGET_LIE_HEIGHT: float = 0.040

# Nominal sitting joint angles: (head_yaw, head_pitch, l_hip_roll, l_hip_pitch, l_knee_pitch, r_hip_roll, r_hip_pitch, r_knee_pitch)
SIT_NOMINAL_QPOS: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 0.70, 1.35, 0.0, 0.70, 1.35],
    dtype=np.float32,
)

# Nominal lying joint angles: (head_yaw, head_pitch, l_hip_roll, l_hip_pitch, l_knee_pitch, r_hip_roll, r_hip_pitch, r_knee_pitch)
LIE_NOMINAL_QPOS: np.ndarray = np.array(
    [0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05],
    dtype=np.float32,
)


NUM_EMOTIONS: int = 10

# Target (head_yaw, head_pitch) in radians for each expressive emotion state
EMOTION_TARGET_HEAD_POS: Dict[Union[EmotionState, int], Tuple[float, float]] = {
    EmotionState.NEUTRAL: (0.0, 0.0),
    EmotionState.HAPPY: (0.0, 0.25),
    EmotionState.SAD: (0.0, -0.30),
    EmotionState.ANGRY: (0.0, -0.15),
    EmotionState.SCARED: (0.0, 0.45),
    EmotionState.CURIOUS: (0.40, 0.10),
    EmotionState.SURPRISED: (0.0, 0.60),
    EmotionState.SLEEPY: (0.0, -0.40),
    EmotionState.PLAYFUL: (-0.35, 0.20),
    EmotionState.PROUD: (0.0, 0.35),
}

# Pre-computed lookup array for fast vectorized reward computations: shape (10, 2)
EMOTION_TARGET_HEAD_ARRAY: np.ndarray = np.array(
    [EMOTION_TARGET_HEAD_POS[EmotionState(i)] for i in range(NUM_EMOTIONS)],
    dtype=np.float32,
)


def get_emotion_one_hot(state: Union[EmotionState, int]) -> np.ndarray:
    """Generate 10-dimensional one-hot vector for given emotion state."""
    idx = int(state)
    if not (0 <= idx < NUM_EMOTIONS):
        raise ValueError(f"Emotion state index {idx} out of range [0, {NUM_EMOTIONS - 1}]")
    one_hot = np.zeros(NUM_EMOTIONS, dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot


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

    # Walking-specific joint limits for stable gait (prevents over-extension)
    WALK_JOINT_LIMITS: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "head_yaw": (-1.57, 1.57),
            "head_pitch": (-0.52, 0.78),
            "left_hip_roll": (-0.25, 0.25),
            "left_hip_pitch": (-0.50, 0.50),
            "left_knee_pitch": (0.0, 0.85),
            "right_hip_roll": (-0.25, 0.25),
            "right_hip_pitch": (-0.50, 0.50),
            "right_knee_pitch": (0.0, 0.85),
        }
    )

    # Nominal standing joint angles (radians) for natural balanced posture
    # Note: MicroDuck 8-DOF has no active ankles; foot is fixed perpendicular to shin.
    # Therefore hip_pitch + knee_pitch MUST equal 0.0 (-0.15 + 0.15 = 0.0) so feet rest perfectly flat on floor!
    NOMINAL_JOINT_POS: Dict[str, float] = field(
        default_factory=lambda: {
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.15,
            "left_knee_pitch": 0.15,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.15,
            "right_knee_pitch": 0.15,
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
            "upright": 1.5,
            "orientation_penalty": -2.0,
            "head_pos_penalty": -1.0,
            "torque_penalty": -0.001,
            "joint_vel_penalty": -0.0001,
            "joint_acc_penalty": -0.01,
            "action_rate_penalty": -0.10,
            "foot_slip_penalty": -0.8,
            "backward_vel_penalty": -2.0,
            "foot_contact": 1.0,
            "double_support_penalty": -0.5,
            "obstacle_avoidance": 0.2,
            "sigma_obs": 0.3,
            "foot_clearance": 2.0,
            "stumble": 0.5,
            "w_centering": 1.5,
            "stair_ascent": 1.5,
            "alive_bonus": 0.05,
            "fall_penalty": -5.0,
            "standstill_penalty": -0.5,
            "feet_air_time": 2.0,
            "emotion_tracking": 0.5,
            "sitting_pose": 2.0,
            "lying_pose": 2.0,
            "joint_limit_penalty": -0.2,
            "heading_penalty": -2.0,
            "leg_pose_penalty": -0.4,
            "gait_sync": 0.8,
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
        self.walk_low_limits = np.array(
            [self.WALK_JOINT_LIMITS[name][0] for name in self.JOINT_NAMES],
            dtype=np.float32,
        )
        self.walk_high_limits = np.array(
            [self.WALK_JOINT_LIMITS[name][1] for name in self.JOINT_NAMES],
            dtype=np.float32,
        )
        self.nominal_qpos = np.array(
            [self.NOMINAL_JOINT_POS.get(name, 0.0) for name in self.JOINT_NAMES],
            dtype=np.float32,
        )


# Module-level cached configuration for high-frequency simulation steps
_DEFAULT_CFG = MicroDuck8DoFConfig()

# Global physical joint limit bounds (8, 2): column 0 = lower bound, column 1 = upper bound
JOINT_LIMITS: np.ndarray = np.stack(
    [_DEFAULT_CFG.low_limits, _DEFAULT_CFG.high_limits], axis=-1
).astype(np.float32)

# Global walking joint limit bounds (8, 2)
WALK_JOINT_LIMITS: np.ndarray = np.stack(
    [_DEFAULT_CFG.walk_low_limits, _DEFAULT_CFG.walk_high_limits], axis=-1
).astype(np.float32)

DEFAULT_NOMINAL_JOINT_POS: Dict[str, float] = {
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.15,
    "left_knee_pitch": 0.15,
    "right_hip_roll": 0.0,
    "right_hip_pitch": -0.15,
    "right_knee_pitch": 0.15,
}

DEFAULT_NOMINAL_QPOS: np.ndarray = _DEFAULT_CFG.nominal_qpos


def compute_observations(
    base_ang_vel: np.ndarray,
    proj_grav: np.ndarray,
    commands: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_actions: np.ndarray,
    foot_contact: np.ndarray,
    lidar_ranges: Optional[np.ndarray] = None,
    emotion_one_hot: Optional[np.ndarray] = None,
    phase_clock: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pack raw state tensors into observation vector (37-dim nominal, 42-dim with 5-ray LiDAR, 47-dim or 52-dim with emotions).

    Dimensions:
    - Base angular velocity (3)
    - Projected gravity vector (3)
    - Velocity commands [vx, vy, wz] (3)
    - Joint positions (8)
    - Joint velocities (8)
    - Previous actions (8)
    - Feet contact sensors (4) [or feet contact (2) + phase_clock [sin, cos] (2) when phase_clock is provided]
    - Optional LiDAR rangefinder rays (5)
    - Optional Emotion state one-hot vector (10)
    Total = 37, 42, 47, or 52 dims.
    """
    fc_arr = np.asarray(foot_contact, dtype=np.float32).ravel().copy()
    if phase_clock is not None:
        p_arr = np.asarray(phase_clock, dtype=np.float32).ravel()
        if fc_arr.shape[0] >= 4:
            fc_arr[2:4] = p_arr[:2]
        elif fc_arr.shape[0] == 2:
            fc_arr = np.concatenate([fc_arr, p_arr[:2]], axis=0)

    parts = [
        np.asarray(base_ang_vel, dtype=np.float32).ravel(),
        np.asarray(proj_grav, dtype=np.float32).ravel(),
        np.asarray(commands, dtype=np.float32).ravel(),
        np.asarray(joint_pos, dtype=np.float32).ravel(),
        np.asarray(joint_vel, dtype=np.float32).ravel(),
        np.asarray(last_actions, dtype=np.float32).ravel(),
        fc_arr,
    ]
    if lidar_ranges is not None:
        parts.append(np.asarray(lidar_ranges, dtype=np.float32).ravel())
    if emotion_one_hot is not None:
        emo_arr = np.asarray(emotion_one_hot, dtype=np.float32).ravel()
        assert emo_arr.shape == (NUM_EMOTIONS,), f"Expected emotion_one_hot shape ({NUM_EMOTIONS},), got {emo_arr.shape}"
        parts.append(emo_arr)

    obs = np.concatenate(parts, axis=-1).astype(np.float32)

    expected_dim = 37 + (5 if lidar_ranges is not None else 0) + (NUM_EMOTIONS if emotion_one_hot is not None else 0)
    assert obs.shape == (expected_dim,), f"Expected observation shape ({expected_dim},), got {obs.shape}"
    return obs


def scale_actions(
    raw_actions: np.ndarray,
    cfg: Optional[MicroDuck8DoFConfig] = None,
    nominal_centered: bool = False,
    walk_clamp: bool = False,
) -> np.ndarray:
    """Scale raw normalized actions [-1, 1] into physical joint target angles.

    If nominal_centered=True, action 0.0 corresponds to nominal_qpos.
    If nominal_centered=False (default), linear mapping: -1.0 -> low_limit, +1.0 -> high_limit.
    Safely clips outputs to respect the hardware limits of each MG996R joint.
    If walk_clamp=True, outputs are clamped to WALK_JOINT_LIMITS.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG

    actions = np.asarray(raw_actions, dtype=np.float32)
    clipped_actions = np.clip(actions, -1.0, 1.0)

    if not nominal_centered:
        # Linear mapping: raw -1.0 -> low_limit, +1.0 -> high_limit
        scaled = cfg.low_limits + (clipped_actions + 1.0) * 0.5 * (
            cfg.high_limits - cfg.low_limits
        )
    else:
        nom = cfg.nominal_qpos
        scaled = np.where(
            clipped_actions >= 0.0,
            nom + clipped_actions * (cfg.high_limits - nom),
            nom + clipped_actions * (nom - cfg.low_limits),
        )
    low = cfg.walk_low_limits if walk_clamp else cfg.low_limits
    high = cfg.walk_high_limits if walk_clamp else cfg.high_limits
    return np.clip(scaled, low, high).astype(np.float32)


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


def quat_to_yaw(q: np.ndarray) -> np.ndarray:
    """Extract heading yaw angle (radians in [-pi, pi]) from quaternion q = [w, x, y, z].

    Supports single quaternion (shape (4,)) or batched array (shape (..., 4)).
    """
    q_arr = np.asarray(q, dtype=np.float32)
    w = q_arr[..., 0]
    x = q_arr[..., 1]
    y = q_arr[..., 2]
    z = q_arr[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)


def compute_gait_phase(
    episode_steps: Union[int, float, np.ndarray],
    dt_control: float = 0.02,
    period: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gait phase in [0, 1) and (clock_sin, clock_cos) for period T=0.8s (1.25 Hz)."""
    steps = np.asarray(episode_steps, dtype=np.float32)
    phase = np.mod(steps * float(dt_control) / float(period), 1.0).astype(np.float32)
    two_pi_phase = 2.0 * np.pi * phase
    clock_sin = np.sin(two_pi_phase).astype(np.float32)
    clock_cos = np.cos(two_pi_phase).astype(np.float32)
    return phase, clock_sin, clock_cos


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
    root_x: Optional[Union[float, np.ndarray]] = None,
    root_y: Optional[np.ndarray] = None,
    root_z: Optional[Union[float, np.ndarray]] = None,
    terrain_mode: str = "hurdles",
    feet_air_time: Optional[np.ndarray] = None,
    feet_touchdown: Optional[np.ndarray] = None,
    current_emotion: Optional[Union[int, np.ndarray, EmotionState]] = None,
    qpos_head: Optional[np.ndarray] = None,
    joint_pos: Optional[np.ndarray] = None,
    cmd_mode: Optional[Union[int, np.ndarray, LocomotionMode]] = 0,
    double_support_time: Optional[Union[float, np.ndarray]] = None,
    last_joint_vel: Optional[np.ndarray] = None,
    current_yaw: Optional[Union[float, np.ndarray]] = None,
    target_yaw: Union[float, np.ndarray] = 0.0,
    clock_sin: Optional[Union[float, np.ndarray]] = None,
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

    # Target yaw rate (around Z axis in rad/s)
    if is_batch:
        if isinstance(cmd_ang_vel, np.ndarray) and cmd_ang_vel.ndim > 1:
            cmd_yaw_rate = cmd_ang_vel[:, 2] if cmd_ang_vel.shape[-1] >= 3 else cmd_ang_vel[:, 0]
        elif isinstance(cmd_ang_vel, np.ndarray) and cmd_ang_vel.shape[0] == base_lin.shape[0]:
            cmd_yaw_rate = cmd_ang_vel
        else:
            cmd_yaw_rate = float(cmd_ang_vel)
    else:
        if hasattr(cmd_ang_vel, "__len__"):
            cmd_ang_arr = np.asarray(cmd_ang_vel, dtype=np.float32).ravel()
            cmd_yaw_rate = float(cmd_ang_arr[2] if len(cmd_ang_arr) >= 3 else cmd_ang_arr[0])
        else:
            cmd_yaw_rate = float(cmd_ang_vel)

    # Locomotion mode resolution (WALK = 0, SIT = 1, LIE = 2)
    if cmd_mode is None:
        cmd_mode = 0
    if is_batch:
        mode_arr = np.asarray(cmd_mode, dtype=np.int32)
        if mode_arr.ndim == 0:
            is_sit = np.full(base_lin.shape[0], mode_arr == int(LocomotionMode.SIT), dtype=bool)
            is_lie = np.full(base_lin.shape[0], mode_arr == int(LocomotionMode.LIE), dtype=bool)
        else:
            is_sit = (mode_arr == int(LocomotionMode.SIT))
            is_lie = (mode_arr == int(LocomotionMode.LIE))
        walk_mode_mask = (~is_sit) & (~is_lie)
    else:
        is_sit = (int(cmd_mode) == int(LocomotionMode.SIT))
        is_lie = (int(cmd_mode) == int(LocomotionMode.LIE))
        walk_mode_mask = (not is_sit) and (not is_lie)

    # 1. Forward linear velocity reward: direct multiplication reward_vel = 2.0 * base_lin_vel_x
    # minus lateral velocity error penalty to prevent crab-walking.
    # Speed diving exploit patch: clip rewarded forward velocity at cmd_x + 0.05.
    vel_x = base_lin[..., 0]
    vel_y = base_lin[..., 1]
    cmd_x = cmd_lin[..., 0]
    cmd_y = cmd_lin[..., 1]
    if is_batch:
        vel_x_eff = np.minimum(vel_x, cmd_x + 0.05)
        fwd_reward = np.where(vel_x >= 0.0, 2.0 * vel_x_eff, 4.0 * vel_x)
        r_track_lin_vel = fwd_reward - 0.5 * np.square(vel_y - cmd_y)
        r_track_lin_vel = np.where(is_sit | is_lie, 0.0, r_track_lin_vel).astype(np.float32)
    else:
        vel_x_eff = min(float(vel_x), float(cmd_x) + 0.05)
        fwd_reward = 2.0 * vel_x_eff if float(vel_x) >= 0.0 else 4.0 * float(vel_x)
        r_track_lin_vel = fwd_reward - 0.5 * float((vel_y - cmd_y) ** 2)
        if is_sit or is_lie:
            r_track_lin_vel = 0.0

    # Standstill penalty: if commanded forward (cmd_lin[..., 0] > 0.05) and base_lin_vel_x < 0.05 m/s
    # penalize standing still (-0.5 per step) to force locomotion (disabled when sitting or lying)
    is_walking_cmd = cmd_x > 0.05
    is_standstill = (vel_x < 0.05) & is_walking_cmd
    r_standstill = np.where(is_standstill, 1.0, 0.0).astype(np.float32)
    if is_batch:
        r_standstill = np.where(is_sit | is_lie, 0.0, r_standstill).astype(np.float32)
    elif is_sit or is_lie:
        r_standstill = 0.0

    # Backward velocity penalty: heavy penalty if agent walks backward while commanded to walk forward
    if is_batch:
        is_backward = (vel_x < -0.01) & is_walking_cmd & (~is_sit) & (~is_lie)
        r_backward_vel = np.where(is_backward, 1.0 + 5.0 * np.abs(vel_x), 0.0).astype(np.float32)
        active_walk = is_walking_cmd & (~is_sit) & (~is_lie)
        gate_val = np.where(vel_x < 0.0, 0.0, np.where(vel_x < 0.05, 0.2, 1.0))
        forward_gate = np.where(active_walk, gate_val, 1.0).astype(np.float32)
    else:
        is_backward = (vel_x < -0.01) and bool(is_walking_cmd) and (not is_sit) and (not is_lie)
        r_backward_vel = float(1.0 + 5.0 * abs(vel_x)) if is_backward else 0.0
        active_walk = bool(is_walking_cmd) and (not is_sit) and (not is_lie)
        if active_walk:
            if vel_x < 0.0:
                forward_gate = 0.0
            elif vel_x < 0.05:
                forward_gate = 0.2
            else:
                forward_gate = 1.0
        else:
            forward_gate = 1.0

    # Double support dwell penalty (> 0.20s): if walking commanded and both feet stick to floor > 0.2s,
    # zero out forward linear velocity reward and apply double_support_penalty (-0.5 per step)
    if is_batch:
        if double_support_time is not None:
            dst = np.asarray(double_support_time, dtype=np.float32)
            is_dwell = (dst > 0.20) & is_walking_cmd & (~is_sit) & (~is_lie)
            r_track_lin_vel = np.where(is_dwell, np.minimum(0.0, r_track_lin_vel), r_track_lin_vel).astype(np.float32)
            r_double_support = np.where(is_dwell, 1.0, 0.0).astype(np.float32)
        else:
            r_double_support = np.zeros(base_lin.shape[0], dtype=np.float32)
    else:
        if double_support_time is not None:
            dst = float(double_support_time)
            is_dwell = (dst > 0.20) and bool(is_walking_cmd) and (not is_sit) and (not is_lie)
            if is_dwell:
                r_track_lin_vel = min(0.0, r_track_lin_vel)
                r_double_support = 1.0
            else:
                r_double_support = 0.0
        else:
            r_double_support = 0.0

    # Feet air time reward: reward swing duration on touchdown (cyclic gait step reward)
    # Trampling guard patch: require actual forward movement vel_x >= 0.05 m/s
    if feet_air_time is not None and feet_touchdown is not None:
        air_t = np.asarray(feet_air_time, dtype=np.float32)
        td = np.asarray(feet_touchdown, dtype=np.float32)
        # Reward air time between 0.08s and 0.40s upon touchdown
        air_reward = np.clip(air_t - 0.08, 0.0, 0.40) * td
        # Active only when commanded to move and actually moving forward (vel_x >= 0.05)
        cmd_active = (np.linalg.norm(cmd_lin[..., :2], axis=-1) > 0.05).astype(np.float32)
        if is_batch:
            forward_active = (vel_x > 0.02).astype(np.float32)
            r_feet_air_time = np.sum(air_reward, axis=-1) * cmd_active * forward_active
        else:
            forward_active = float(vel_x > 0.02)
            r_feet_air_time = float(np.sum(air_reward) * cmd_active * forward_active)
    else:
        r_feet_air_time = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    # 2. Angular velocity tracking around Z axis
    ang_err_sq = np.square(base_ang[..., 2] - cmd_yaw_rate)
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

    # Orientation tilt penalty (heavily penalizes pitching forward/backward or rolling)
    r_orientation_penalty = 2.0 * np.square(grav[..., 0]) + 1.0 * np.square(grav[..., 1])
    if is_batch:
        r_orientation_penalty = np.where(is_lie, 0.0, r_orientation_penalty).astype(np.float32)
    elif is_lie:
        r_orientation_penalty = 0.0

    # 4. Energy and torque minimization penalty
    r_torque_penalty = np.sum(np.square(torques), axis=-1)

    # 5. Joint velocity smoothing penalty
    r_joint_vel_penalty = np.sum(np.square(j_vel), axis=-1)

    # Joint acceleration penalty (suppresses rapid back-and-forth oscillations)
    if last_joint_vel is not None:
        last_j_vel = np.asarray(last_joint_vel, dtype=np.float32)
        r_joint_acc_penalty = np.sum(np.square(j_vel - last_j_vel), axis=-1)
    else:
        r_joint_acc_penalty = (
            np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0
        )

    # 6. Action rate penalty (protects MG996R gears against jerky policy outputs)
    if actions is not None and last_actions is not None:
        act = np.asarray(actions, dtype=np.float32)
        last_act = np.asarray(last_actions, dtype=np.float32)
        r_action_rate_penalty = np.sum(np.square(act - last_act), axis=-1)
    else:
        r_action_rate_penalty = (
            np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0
        )

    # 7. Foot contact reward (khuyến khích nhấc chân luân phiên khi di chuyển TIẾN)
    num_contacts = np.sum(f_contact > 0.0, axis=-1)
    cmd_speed = np.linalg.norm(cmd_lin[..., :2], axis=-1) + np.abs(cmd_yaw_rate)
    is_moving_cmd = cmd_speed > 0.05
    forward_gait_gate = (vel_x > 0.02).astype(np.float32)
    r_foot_contact = np.where(
        is_moving_cmd,
        (num_contacts == 1).astype(np.float32) * forward_gait_gate,
        (num_contacts == 2).astype(np.float32) * 0.5,
    ).astype(np.float32)
    if is_batch:
        r_foot_contact = np.where(
            is_moving_cmd & (vel_x < -1e-4),
            0.0,
            r_foot_contact,
        ).astype(np.float32)
    else:
        if is_moving_cmd and vel_x < -1e-4:
            r_foot_contact = 0.0

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

    # Foot slip penalty: penalize horizontal velocity of foot while in contact with floor
    if foot_vel_xy is not None:
        f_vel_xy = np.asarray(foot_vel_xy, dtype=np.float32)
        if foot_contact_forces is not None:
            f_forces = np.asarray(foot_contact_forces, dtype=np.float32)
            in_contact = np.abs(f_forces[..., 2]) > 1.0
        else:
            in_contact = f_contact[..., :2] > 0.5
        vel_xy_sq = np.sum(np.square(f_vel_xy), axis=-1)
        r_foot_slip_raw = np.sum(vel_xy_sq * in_contact.astype(np.float32), axis=-1)
        if is_batch:
            r_foot_slip = np.asarray(r_foot_slip_raw, dtype=np.float32)
        else:
            r_foot_slip = float(r_foot_slip_raw)
    else:
        r_foot_slip = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    r_centering = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0
    if terrain_mode == "stairs" and root_y is not None:
        y_pos = np.asarray(root_y, dtype=np.float32)
        r_centering = -2.0 * np.abs(y_pos) - 1.0 * np.abs(base_lin[..., 1])

    # 10. Stair ascent elevation reward (x >= 1.35m)
    if root_x is not None and root_z is not None:
        rx = np.asarray(root_x, dtype=np.float32)
        rz = np.asarray(root_z, dtype=np.float32)
        stair_mask = (rx >= 1.35).astype(np.float32)
        elevation_gain = np.clip(rz - 0.170, 0.0, 0.050)
        r_stair_ascent = stair_mask * elevation_gain * 10.0
    else:
        r_stair_ascent = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    # 11. Emotion head posture tracking reward & head position deviation penalty
    head = None
    if qpos_head is not None:
        head = np.asarray(qpos_head, dtype=np.float32)
    elif joint_pos is not None:
        head = np.asarray(joint_pos, dtype=np.float32)[..., :2]

    r_emotion = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0
    r_head_pos_penalty = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    if head is not None:
        if current_emotion is not None:
            if is_batch:
                emo_indices = np.asarray(current_emotion, dtype=np.int32).ravel()
                target_head = EMOTION_TARGET_HEAD_ARRAY[emo_indices]
                err_sq = np.sum(np.square(head[..., :2] - target_head), axis=-1)
                r_emotion = np.exp(-err_sq / 0.15).astype(np.float32)
            else:
                emo_idx = int(current_emotion)
                target_head = EMOTION_TARGET_HEAD_ARRAY[emo_idx]
                err_sq = np.sum(np.square(head.ravel()[:2] - target_head))
                r_emotion = float(np.exp(-err_sq / 0.15))
        else:
            if is_batch:
                target_head = np.zeros((head.shape[0], 2), dtype=np.float32)
            else:
                target_head = np.zeros(2, dtype=np.float32)

        diff_yaw = head[..., 0] - target_head[..., 0]
        diff_pitch = head[..., 1] - target_head[..., 1]
        pen_head = np.square(diff_yaw) + 2.0 * np.square(diff_pitch)
        if is_batch:
            r_head_pos_penalty = pen_head.astype(np.float32)
        else:
            r_head_pos_penalty = float(pen_head)

    # 12. Sitting posture and height reward
    if joint_pos is not None:
        q_legs = np.asarray(joint_pos, dtype=np.float32)[..., 2:8]
        sit_legs = SIT_NOMINAL_QPOS[2:8]
        pose_err_sq = np.sum(np.square(q_legs - sit_legs), axis=-1)
        r_sit_pose = np.exp(-pose_err_sq / 0.20)
    else:
        r_sit_pose = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    if root_z is not None:
        rz = np.asarray(root_z, dtype=np.float32)
        height_err_sq = np.square(rz - TARGET_SIT_HEIGHT)
        r_sit_height = np.exp(-height_err_sq / 0.005)
    else:
        r_sit_height = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    r_sit_raw = 0.5 * r_sit_pose + 0.5 * r_sit_height
    if is_batch:
        r_sit = np.where(is_sit, r_sit_raw, 0.0).astype(np.float32)
    else:
        r_sit = float(r_sit_raw) if is_sit else 0.0

    # 13. Lying posture and height reward
    if joint_pos is not None:
        q_legs = np.asarray(joint_pos, dtype=np.float32)[..., 2:8]
        lie_legs = LIE_NOMINAL_QPOS[2:8]
        lie_pose_err_sq = np.sum(np.square(q_legs - lie_legs), axis=-1)
        r_lie_pose = np.exp(-lie_pose_err_sq / 0.20)
    else:
        r_lie_pose = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    if root_z is not None:
        rz = np.asarray(root_z, dtype=np.float32)
        lie_height_err_sq = np.square(rz - TARGET_LIE_HEIGHT)
        r_lie_height = np.exp(-lie_height_err_sq / 0.003)
    else:
        r_lie_height = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    r_lie_raw = 0.5 * r_lie_pose + 0.5 * r_lie_height
    if is_batch:
        r_lie = np.where(is_lie, r_lie_raw, 0.0).astype(np.float32)
    else:
        r_lie = float(r_lie_raw) if is_lie else 0.0

    # 14. Joint limit dwelling penalty (< 5% margin from physical stop)
    r_joint_limit = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0
    if joint_pos is not None:
        q = np.asarray(joint_pos, dtype=np.float32)
        low = JOINT_LIMITS[:, 0]
        high = JOINT_LIMITS[:, 1]
        rng = high - low
        margin = 0.05 * rng
        low_thresh = low + margin
        high_thresh = high - margin
        pen_low = np.maximum(0.0, low_thresh - q) / margin
        pen_high = np.maximum(0.0, q - high_thresh) / margin
        pen = pen_low + pen_high
        r_joint_limit_raw = -np.sum(np.square(pen), axis=-1)
        if is_batch:
            r_joint_limit = r_joint_limit_raw.astype(np.float32)
        else:
            r_joint_limit = float(r_joint_limit_raw)

    if clock_sin is not None or current_yaw is not None:
        if is_batch:
            is_walk_cmd = walk_mode_mask & (cmd_x > 0.05)
            r_alive = np.where(is_walk_cmd & (vel_x <= 0.02), 0.0, 1.0).astype(np.float32)
        else:
            is_walk_cmd = walk_mode_mask and (float(cmd_x) > 0.05)
            r_alive = 0.0 if (is_walk_cmd and float(vel_x) <= 0.02) else 1.0
    else:
        r_alive = np.ones(base_lin.shape[0], dtype=np.float32) if is_batch else 1.0

    # Heading penalty (locks heading yaw to target_yaw in WALK mode)
    if current_yaw is not None:
        yaw_cur = np.asarray(current_yaw, dtype=np.float32)
        yaw_tgt = np.asarray(target_yaw, dtype=np.float32)
        yaw_diff = yaw_cur - yaw_tgt
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
        w_heading = weights.get("heading_penalty", -2.0)
        if is_batch:
            r_heading = (w_heading * np.square(yaw_diff) * walk_mode_mask).astype(np.float32)
        else:
            r_heading = float(w_heading * np.square(yaw_diff)) if walk_mode_mask else 0.0
    else:
        r_heading = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    # Leg pose penalty (penalizes deviation from DEFAULT_NOMINAL_QPOS[2:8] in WALK mode)
    if joint_pos is not None:
        q_pos = np.asarray(joint_pos, dtype=np.float32)
        q_legs = q_pos[..., 2:8]
        nom_legs = DEFAULT_NOMINAL_QPOS[2:8]
        leg_err_sq = np.sum(np.square(q_legs - nom_legs), axis=-1)
        w_leg_pose = weights.get("leg_pose_penalty", -0.4)
        if is_batch:
            r_leg_pose = (w_leg_pose * leg_err_sq * walk_mode_mask).astype(np.float32)
        else:
            r_leg_pose = float(w_leg_pose * leg_err_sq) if walk_mode_mask else 0.0
    else:
        r_leg_pose = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    # Gait sync reward (rewards phase-synchronized alternating foot contact gated by forward motion)
    eff_clock_sin = clock_sin
    if eff_clock_sin is None and f_contact.shape[-1] >= 4:
        if is_batch:
            if np.any(f_contact[..., 2] != 0.0) or np.any(f_contact[..., 3] != 0.0):
                eff_clock_sin = f_contact[..., 2]
        else:
            if f_contact[2] != 0.0 or f_contact[3] != 0.0:
                eff_clock_sin = f_contact[2]

    if eff_clock_sin is not None:
        c_sin = np.asarray(eff_clock_sin, dtype=np.float32)
        left_c = np.clip(f_contact[..., 0], 0.0, 1.0)
        right_c = np.clip(f_contact[..., 1], 0.0, 1.0)
        desired_L = np.where(c_sin < 0.0, 1.0, 0.0)
        desired_R = np.where(c_sin >= 0.0, 1.0, 0.0)
        sync_score = (1.0 - np.abs(left_c - desired_L)) + (1.0 - np.abs(right_c - desired_R))
        w_gait_sync = weights.get("gait_sync", 0.8)
        if is_batch:
            fwd_gate = (vel_x > 0.03).astype(np.float32)
            mode_gate = walk_mode_mask.astype(np.float32)
            r_gait_sync = (w_gait_sync * sync_score * fwd_gate * mode_gate).astype(np.float32)
        else:
            fwd_gate = 1.0 if float(vel_x) > 0.03 else 0.0
            mode_gate = 1.0 if walk_mode_mask else 0.0
            r_gait_sync = float(w_gait_sync * sync_score * fwd_gate * mode_gate)
    else:
        r_gait_sync = np.zeros(base_lin.shape[0], dtype=np.float32) if is_batch else 0.0

    w_limit = weights.get("joint_limit_penalty", -0.2)
    limit_scale = -w_limit if w_limit < 0 else w_limit

    total = (
        weights.get("tracking_lin_vel", 1.0) * r_track_lin_vel
        + forward_gate * (
            weights.get("tracking_ang_vel", 0.5) * r_track_ang_vel
            + weights.get("upright", 1.5) * r_upright
            + weights.get("foot_contact", 1.0) * r_foot_contact
            + weights.get("emotion_tracking", 0.5) * r_emotion
        )
        + weights.get("orientation_penalty", -2.0) * r_orientation_penalty
        + weights.get("head_pos_penalty", -1.0) * r_head_pos_penalty
        + weights.get("torque_penalty", -0.001) * r_torque_penalty
        + weights.get("joint_vel_penalty", -0.0001) * r_joint_vel_penalty
        + weights.get("joint_acc_penalty", -0.01) * r_joint_acc_penalty
        + weights.get("action_rate_penalty", -0.10) * r_action_rate_penalty
        + weights.get("backward_vel_penalty", -2.0) * r_backward_vel
        + weights.get("foot_slip_penalty", -0.8) * r_foot_slip
        + weights.get("obstacle_avoidance", 0.2) * r_obstacle
        + weights.get("foot_clearance", 2.0) * r_clearance
        + weights.get("stumble", 0.5) * r_stumble
        + (weights.get("w_centering", 1.5) * r_centering if terrain_mode == "stairs" else 0.0)
        + weights.get("stair_ascent", 0.0) * r_stair_ascent
        + weights.get("alive_bonus", 0.05) * r_alive
        + weights.get("standstill_penalty", -0.5) * r_standstill
        + weights.get("double_support_penalty", -0.5) * r_double_support
        + weights.get("feet_air_time", 2.0) * r_feet_air_time
        + weights.get("sitting_pose", 2.0) * r_sit
        + weights.get("lying_pose", 2.0) * r_lie
        + limit_scale * r_joint_limit
        + r_heading
        + r_leg_pose
        + r_gait_sync
    )

    if not is_batch:
        return {
            "total_reward": float(total),
            "tracking_lin_vel": float(r_track_lin_vel),
            "tracking_ang_vel": float(r_track_ang_vel),
            "upright": float(r_upright),
            "orientation_penalty": float(r_orientation_penalty),
            "head_pos_penalty": float(r_head_pos_penalty),
            "torque_penalty": float(r_torque_penalty),
            "joint_vel_penalty": float(r_joint_vel_penalty),
            "joint_acc_penalty": float(r_joint_acc_penalty),
            "action_rate_penalty": float(r_action_rate_penalty),
            "backward_vel_penalty": float(weights.get("backward_vel_penalty", -2.0) * r_backward_vel),
            "foot_contact": float(r_foot_contact),
            "foot_slip_penalty": float(r_foot_slip),
            "double_support_penalty": float(weights.get("double_support_penalty", -0.5) * r_double_support),
            "obstacle_avoidance": float(r_obstacle),
            "foot_clearance": float(r_clearance),
            "rew_foot_clearance": float(r_clearance),
            "stumble": float(r_stumble),
            "pen_stumble": float(r_stumble),
            "stair_ascent": float(r_stair_ascent),
            "alive_bonus": float(weights.get("alive_bonus", 0.05) * r_alive),
            "standstill_penalty": float(weights.get("standstill_penalty", -0.5) * r_standstill),
            "feet_air_time": float(weights.get("feet_air_time", 2.0) * r_feet_air_time),
            "emotion_tracking": float(r_emotion),
            "sitting_pose": float(r_sit),
            "r_sit": float(r_sit),
            "lying_pose": float(r_lie),
            "r_lie": float(r_lie),
            "joint_limit_penalty": float(r_joint_limit),
            "heading_penalty": float(r_heading),
            "leg_pose_penalty": float(r_leg_pose),
            "gait_sync": float(r_gait_sync),
        }

    return {
        "total_reward": total.astype(np.float32),
        "tracking_lin_vel": r_track_lin_vel.astype(np.float32),
        "tracking_ang_vel": r_track_ang_vel.astype(np.float32),
        "upright": r_upright.astype(np.float32),
        "orientation_penalty": r_orientation_penalty.astype(np.float32),
        "head_pos_penalty": r_head_pos_penalty.astype(np.float32),
        "torque_penalty": r_torque_penalty.astype(np.float32),
        "joint_vel_penalty": r_joint_vel_penalty.astype(np.float32),
        "joint_acc_penalty": r_joint_acc_penalty.astype(np.float32),
        "action_rate_penalty": r_action_rate_penalty.astype(np.float32),
        "backward_vel_penalty": (weights.get("backward_vel_penalty", -2.0) * r_backward_vel).astype(np.float32),
        "foot_contact": r_foot_contact.astype(np.float32),
        "foot_slip_penalty": r_foot_slip.astype(np.float32),
        "double_support_penalty": (weights.get("double_support_penalty", -0.5) * r_double_support).astype(np.float32),
        "obstacle_avoidance": r_obstacle.astype(np.float32),
        "foot_clearance": r_clearance.astype(np.float32),
        "rew_foot_clearance": r_clearance.astype(np.float32),
        "stumble": r_stumble.astype(np.float32),
        "pen_stumble": r_stumble.astype(np.float32),
        "stair_ascent": r_stair_ascent.astype(np.float32),
        "alive_bonus": (weights.get("alive_bonus", 0.05) * r_alive).astype(np.float32),
        "standstill_penalty": (weights.get("standstill_penalty", -0.5) * r_standstill).astype(np.float32),
        "feet_air_time": (weights.get("feet_air_time", 2.0) * r_feet_air_time).astype(np.float32),
        "emotion_tracking": r_emotion.astype(np.float32),
        "sitting_pose": r_sit.astype(np.float32),
        "r_sit": r_sit.astype(np.float32),
        "lying_pose": r_lie.astype(np.float32),
        "r_lie": r_lie.astype(np.float32),
        "joint_limit_penalty": r_joint_limit.astype(np.float32),
        "heading_penalty": r_heading.astype(np.float32),
        "leg_pose_penalty": r_leg_pose.astype(np.float32),
        "gait_sync": r_gait_sync.astype(np.float32),
    }
