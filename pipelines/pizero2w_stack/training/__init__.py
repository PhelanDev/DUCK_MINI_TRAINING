"""Training module for MicroDuck 8-DOF Raspberry Pi Zero 2 W Stack."""

from .env_config import (
    DEFAULT_NOMINAL_JOINT_POS,
    DEFAULT_NOMINAL_QPOS,
    DEFAULT_OBS_BOUNDS,
    MicroDuck8DoFConfig,
    clip_action_rates,
    compute_observations,
    compute_rewards,
    quat_rotate_inverse,
    scale_actions,
)

__all__ = [
    "MicroDuck8DoFConfig",
    "compute_observations",
    "scale_actions",
    "clip_action_rates",
    "compute_rewards",
    "quat_rotate_inverse",
    "DEFAULT_NOMINAL_JOINT_POS",
    "DEFAULT_NOMINAL_QPOS",
    "DEFAULT_OBS_BOUNDS",
]

try:
    from .run_parallel_sim import ParallelSimRunner, run_benchmark

    __all__.extend(["ParallelSimRunner", "run_benchmark"])
except ImportError:
    pass
