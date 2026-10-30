"""MicroDuck 8-DOF reinforcement learning training package."""

from training.env_config import (
    MicroDuck8DoFConfig,
    WALK_JOINT_LIMITS,
    compute_observations,
    scale_actions,
    clip_action_rates,
    quat_rotate_inverse,
    quat_to_yaw,
    compute_gait_phase,
    compute_rewards,
)

__all__ = [
    "MicroDuck8DoFConfig",
    "WALK_JOINT_LIMITS",
    "compute_observations",
    "scale_actions",
    "clip_action_rates",
    "quat_rotate_inverse",
    "quat_to_yaw",
    "compute_gait_phase",
    "compute_rewards",
]
