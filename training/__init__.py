"""MicroDuck 8-DOF reinforcement learning training package."""

from training.env_config import (
    MicroDuck8DoFConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
    compute_rewards,
)

__all__ = [
    "MicroDuck8DoFConfig",
    "compute_observations",
    "scale_actions",
    "clip_action_rates",
    "compute_rewards",
]
