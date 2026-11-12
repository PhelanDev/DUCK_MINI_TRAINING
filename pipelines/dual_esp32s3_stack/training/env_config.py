"""Reinforcement Learning Environment Configuration for MicroDuck 8-DOF (Dual ESP32-S3 Stack).

Thin parameterized wrapper delegating common training environment logic to training.env_config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from training.env_config import (
    MicroDuck8DoFConfig as BaseConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
    quat_rotate_inverse,
    compute_rewards,
    DEFAULT_OBS_BOUNDS,
    DEFAULT_OBS_SCALES,
    quantize_obs_int8,
    dequantize_obs_int8,
    dequantize_act_int8,
    generate_esp32_quant_header,
)


@dataclass
class MicroDuck8DoFConfig(BaseConfig):
    """Configuration dataclass for MicroDuck 8-DOF on Dual ESP32-S3 Stack."""

    STACK_NAME: str = "dual_esp32s3_stack"
    TARGET_MASS: float = 0.684  # kg (~684g total weight)
    POLICY_DTYPE: str = "int8"  # Quantized for ESP32-S3 PIE vector instructions

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
