"""Reinforcement Learning Environment Configuration for MicroDuck 8-DOF (RPi Zero 2 W Stack).

Thin parameterized wrapper delegating common training environment logic to training.env_config.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from training.env_config import (
    MicroDuck8DoFConfig as BaseConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
    quat_rotate_inverse,
    compute_rewards,
)


@dataclass
class MicroDuck8DoFConfig(BaseConfig):
    """Configuration dataclass for MicroDuck 8-DOF on Raspberry Pi Zero 2 W Stack."""

    STACK_NAME: str = "pizero2w_stack"
    TARGET_MASS: float = 0.649  # kg (~649g total weight)
    POLICY_DTYPE: str = "float32"

    OBS_DIM: int = 42
    ACTION_DIM: int = 8
    LIDAR_RAYS: int = 5
    LIDAR_MIN_RANGE: float = 0.05
    LIDAR_MAX_RANGE: float = 2.0
    OBSTACLE_THRESHOLD: float = 0.35  # meters

    NOMINAL_JOINT_POS: Dict[str, float] = field(
        default_factory=lambda: {
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.15,
            "left_knee_pitch": 0.35,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.15,
            "right_knee_pitch": 0.35,
        }
    )


# Module-level cached configuration for simulation steps
_DEFAULT_CFG = MicroDuck8DoFConfig()

DEFAULT_NOMINAL_JOINT_POS: Dict[str, float] = _DEFAULT_CFG.NOMINAL_JOINT_POS
DEFAULT_NOMINAL_QPOS: np.ndarray = _DEFAULT_CFG.nominal_qpos

DEFAULT_OBS_BOUNDS: np.ndarray = np.array(
    [
        5.0, 5.0, 5.0,  # Base angular velocity [wx, wy, wz] (rad/s)
        1.0, 1.0, 1.0,  # Projected gravity vector [gx, gy, gz]
        1.0, 1.0, 2.0,  # Velocity command targets [vx, vy, wz] (m/s, rad/s)
        1.57, 0.78, 0.44, 0.87, 1.48, 0.44, 0.87, 1.48,  # Joint angles (rad)
        6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,  # Joint velocities (rad/s)
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Previous normalized actions [-1, 1]
        1.0, 1.0, 1.0, 1.0,  # Feet contact state indicators [0 or 1]
        2.0, 2.0, 2.0, 2.0, 2.0,  # 5-ray LiDAR rangefinder distances (meters)
    ],
    dtype=np.float32,
)
