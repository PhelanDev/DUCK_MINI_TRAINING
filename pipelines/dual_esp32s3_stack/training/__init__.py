"""Training module for MicroDuck 8-DOF Dual ESP32-S3 Stack."""

from .env_config import (
    MicroDuck8DoFConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
    compute_rewards,
    quantize_obs_int8,
    dequantize_act_int8,
    generate_esp32_quant_header,
)
from .run_parallel_sim import ParallelSimRunner, run_benchmark

__all__ = [
    "MicroDuck8DoFConfig",
    "compute_observations",
    "scale_actions",
    "clip_action_rates",
    "compute_rewards",
    "quantize_obs_int8",
    "dequantize_act_int8",
    "generate_esp32_quant_header",
    "ParallelSimRunner",
    "run_benchmark",
]
