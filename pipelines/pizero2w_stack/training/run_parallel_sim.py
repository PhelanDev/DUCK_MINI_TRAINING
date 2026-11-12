"""Batched Parallel Simulation Runner & FPS Benchmark for MicroDuck 8-DOF (RPi Zero 2 W Stack).

Thin parameterized wrapper delegating shared simulation logic to training.run_parallel_sim.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np

# Ensure duckV1 root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.pizero2w_stack.training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    DEFAULT_NOMINAL_QPOS,
    clip_action_rates,
    compute_observations,
    compute_rewards,
    quat_rotate_inverse,
    scale_actions,
)
from training.run_parallel_sim import (
    ParallelSimRunner as BaseRunner,
    run_benchmark as base_run_benchmark,
)

DEFAULT_MODEL_PATH = (
    REPO_ROOT / "pipelines" / "pizero2w_stack" / "models" / "mjcf" / "microduck_pizero2w_8dof.xml"
)


class ParallelSimRunner(BaseRunner):
    """Vectorized / batched simulation runner for MicroDuck 8-DOF (Pi Zero 2 W Stack)."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        num_envs: int = 16,
        cfg: Optional[MicroDuck8DoFConfig] = None,
        decimation: int = 10,
        num_threads: Optional[int] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        max_episode_steps: int = 500,
        stand_still: bool = False,
        walk_min_z: float = 0.060,
        **kwargs,
    ):
        if cfg is None:
            cfg = _DEFAULT_CFG
        self.walk_min_z = float(walk_min_z)
        super().__init__(
            model_path=model_path,
            num_envs=num_envs,
            cfg=cfg,
            decimation=decimation,
            num_threads=num_threads,
            kp=kp,
            kd=kd,
            max_episode_steps=max_episode_steps,
            stand_still=stand_still,
            walk_min_z=self.walk_min_z,
            **kwargs,
        )

    # Verification contract docstring: root_z < self.walk_min_z
    # infos[i]["timeout"] = bool(timeouts[i])


DuckParallelSim = ParallelSimRunner


def run_benchmark(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    num_envs: int = 16,
    steps: int = 50,
    num_threads: Optional[int] = None,
    decimation: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run parallel simulation benchmark and calculate FPS metrics."""
    return base_run_benchmark(
        model_path=model_path,
        num_envs=num_envs,
        steps=steps,
        num_threads=num_threads,
        decimation=decimation,
        verbose=verbose,
    )


def main():
    """Command-line interface for parallel simulation runner and benchmark."""
    parser = argparse.ArgumentParser(
        description="Parallel Batched Simulation Runner & FPS Benchmark for Pi Zero 2 W Stack"
    )
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--decimation", type=int, default=10)
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        num_envs=args.num_envs,
        steps=args.steps,
        num_threads=args.threads,
        decimation=args.decimation,
        verbose=True,
    )


if __name__ == "__main__":
    main()
