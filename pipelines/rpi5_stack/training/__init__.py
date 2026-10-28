"""Training module for MicroDuck 8-DOF Raspberry Pi 5 Stack."""

from .env_config import (
    MicroDuck8DoFConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
    compute_rewards,
)
from .run_parallel_sim import ParallelSimRunner, run_benchmark

__all__ = [
    "MicroDuck8DoFConfig",
    "compute_observations",
    "scale_actions",
    "clip_action_rates",
    "compute_rewards",
    "ParallelSimRunner",
    "run_benchmark",
]
