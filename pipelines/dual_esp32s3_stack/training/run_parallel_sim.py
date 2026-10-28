"""Batched Parallel Simulation Runner & FPS Benchmark for MicroDuck 8-DOF (Dual ESP32-S3 Stack).

Executes vectorized MuJoCo simulation instances sharing a single mjModel across
multiple environments using multi-threaded stepping or batched loops.
Extracts 42-dimensional perception observations (including 5 LiDAR rangefinder rays)
and benchmarks stepping throughput for Reinforcement Learning training.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.dual_esp32s3_stack.training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    clip_action_rates,
    compute_observations,
    compute_rewards,
    quat_rotate_inverse,
    scale_actions,
)

DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "pipelines"
    / "dual_esp32s3_stack"
    / "models"
    / "mjcf"
    / "microduck_esp32_8dof.xml"
)


class ParallelSimRunner:
    """Vectorized / batched simulation runner for MicroDuck 8-DOF (Dual ESP32-S3 Stack)."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        num_envs: int = 16,
        cfg: Optional[MicroDuck8DoFConfig] = None,
        decimation: int = 10,
        num_threads: Optional[int] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found at {self.model_path}")

        self.cfg = cfg if cfg is not None else _DEFAULT_CFG
        self.num_envs = int(num_envs)
        self.decimation = max(1, int(decimation))
        self.kp = float(kp) if kp is not None else float(self.cfg.KP)
        self.kd = float(kd) if kd is not None else float(self.cfg.KD)

        # Load shared MuJoCo model and instantiate per-env data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_envs)]

        # Time tracking
        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = self.sim_dt * self.decimation

        # Threading configuration
        if num_threads is None:
            cpu_count = os.cpu_count() or 4
            self.num_threads = min(cpu_count, max(1, self.num_envs // 4))
        else:
            self.num_threads = max(1, int(num_threads))

        self.executor = (
            ThreadPoolExecutor(max_workers=self.num_threads)
            if self.num_threads > 1 and self.num_envs > 1
            else None
        )

        # Tracking state buffers (42-dim obs, 8-dim act)
        self.obs_dim = self.cfg.OBS_DIM
        self.action_dim = self.cfg.ACTION_DIM

        self.last_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.last_targets = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
        # Default forward velocity command [vx=0.5, vy=0.0, wz=0.0]
        self.commands[:, 0] = 0.5

        # Initialize environments
        self.reset()

    def reset_env(self, env_idx: int) -> np.ndarray:
        """Reset a single environment to nominal standing pose."""
        d = self.datas[env_idx]
        mujoco.mj_resetData(self.model, d)

        # Default nominal root position (standing height ~0.17m)
        d.qpos[0:3] = [0.0, 0.0, 0.170]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion (upright)
        d.qpos[7:15] = 0.0  # joint angles at zero
        d.qvel[:] = 0.0

        # Step forward kinematics and sensor update
        mujoco.mj_forward(self.model, d)

        self.last_actions[env_idx] = 0.0
        self.last_targets[env_idx] = 0.0
        return self._extract_obs(env_idx)

    def reset(self, env_ids: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """Reset specified environments or all if None. Returns array of observations."""
        if env_ids is None:
            indices = range(self.num_envs)
        else:
            indices = env_ids

        for i in indices:
            self.reset_env(i)

        return self.get_observations()

    def _extract_obs(self, env_idx: int) -> np.ndarray:
        """Extract 42-dimensional perception observation vector from env data."""
        d = self.datas[env_idx]

        # Base angular velocity from gyroscope sensor
        base_ang_vel = np.array(d.sensordata[3:6], dtype=np.float32)

        # Projected gravity vector from base quaternion
        w, x, y, z = d.sensordata[6:10]
        proj_grav = np.array(
            [
                2.0 * (w * y - x * z),
                -2.0 * (w * x + y * z),
                2.0 * (x * x + y * y) - 1.0,
            ],
            dtype=np.float32,
        )

        cmd = self.commands[env_idx]
        joint_pos = np.array(d.sensordata[12:20], dtype=np.float32)
        joint_vel = np.array(d.sensordata[20:28], dtype=np.float32)
        last_act = self.last_actions[env_idx]

        # Foot contact sensors: left (10), right (11), padded to 4 dimensions
        left_touch = float(d.sensordata[10])
        right_touch = float(d.sensordata[11])
        foot_contact = np.array([left_touch, right_touch, 0.0, 0.0], dtype=np.float32)

        # 5-ray LiDAR rangefinder readings (indices 28:33)
        # MuJoCo rangefinder returns -1.0 if no geom is hit within cutoff distance
        # Replace negative values with max range (2.0m) to indicate clear path
        raw_lidar = np.array(d.sensordata[28:33], dtype=np.float32)
        lidar_ranges = np.where(raw_lidar < 0.0, self.cfg.LIDAR_MAX_RANGE, raw_lidar)

        return compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_act,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges,
        )

    def get_observations(self) -> np.ndarray:
        """Extract all current observations as a (num_envs, 42) numpy array."""
        obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.num_envs):
            obs[i] = self._extract_obs(i)
        return obs

    def _step_subrange(
        self,
        indices: np.ndarray,
        target_positions: np.ndarray,
    ):
        """Worker function to simulate a subset of environments."""
        for idx in indices:
            d = self.datas[idx]
            target_pos = target_positions[idx]
            for _ in range(self.decimation):
                current_pos = d.qpos[7:15]
                current_vel = d.qvel[6:14]
                # PD controller for MG996R servo position tracking
                torques = self.kp * (target_pos - current_pos) - self.kd * current_vel
                d.ctrl[:] = np.clip(
                    torques, -self.cfg.TORQUE_LIMIT, self.cfg.TORQUE_LIMIT
                )
                mujoco.mj_step(self.model, d)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments by one control step with given normalized actions.

        Returns: (next_observations, rewards, dones, infos)
        """
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Expected action shape ({self.num_envs}, {self.action_dim}), got {actions.shape}"

        # 1. Scale actions to joint limits
        scaled_targets = np.empty_like(actions)
        for i in range(self.num_envs):
            scaled_targets[i] = scale_actions(actions[i], self.cfg)

        # 2. Rate-limit target joint changes
        target_positions = np.empty_like(scaled_targets)
        for i in range(self.num_envs):
            target_positions[i] = clip_action_rates(
                scaled_targets[i],
                self.last_targets[i],
                dt=self.control_dt,
                max_velocity=self.cfg.MAX_VELOCITY,
            )
        self.last_targets[:] = target_positions

        # 3. Step physics simulation (either multi-threaded or sequential)
        if self.executor is not None and self.num_threads > 1:
            chunks = np.array_split(np.arange(self.num_envs), self.num_threads)
            futures = [
                self.executor.submit(self._step_subrange, chunk, target_positions)
                for chunk in chunks
                if len(chunk) > 0
            ]
            for f in futures:
                f.result()
        else:
            self._step_subrange(np.arange(self.num_envs), target_positions)

        # 4. Extract observations, compute rewards, and check done flags
        next_obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = []

        for i in range(self.num_envs):
            d = self.datas[i]
            next_obs[i] = self._extract_obs(i)

            # Extract terms for reward calculation
            base_lin_vel = np.array(d.qvel[0:3], dtype=np.float32)
            cmd_lin_vel = np.array([self.commands[i, 0], self.commands[i, 1], 0.0], dtype=np.float32)
            base_ang_vel = np.array(d.sensordata[3:6], dtype=np.float32)
            cmd_ang_vel = float(self.commands[i, 2])
            proj_grav = next_obs[i, 3:6]
            joint_torques = np.array(d.ctrl[:], dtype=np.float32)
            joint_vel = np.array(d.sensordata[20:28], dtype=np.float32)
            foot_contact = next_obs[i, 33:37]
            lidar_ranges = next_obs[i, 37:42]

            reward_dict = compute_rewards(
                base_lin_vel=base_lin_vel,
                cmd_lin_vel=cmd_lin_vel,
                base_ang_vel=base_ang_vel,
                cmd_ang_vel=cmd_ang_vel,
                proj_gravity=proj_grav,
                joint_torques=joint_torques,
                joint_vel=joint_vel,
                foot_contact=foot_contact,
                lidar_ranges=lidar_ranges,
                actions=actions[i],
                last_actions=self.last_actions[i],
                weights=self.cfg.REWARD_WEIGHTS,
            )
            rewards[i] = reward_dict["total_reward"]
            infos.append(reward_dict)

            # Check termination conditions: fall over, excessive tilt, or numerical divergence
            root_z = float(d.qpos[2])
            grav_z = float(proj_grav[2])
            if (
                root_z < 0.08
                or grav_z > -0.3
                or not np.all(np.isfinite(d.qpos))
                or not np.all(np.isfinite(d.qvel))
            ):
                dones[i] = True
                self.reset_env(i)

        self.last_actions[:] = actions
        if np.any(dones):
            self.last_actions[dones] = 0.0
        return next_obs, rewards, dones, infos

    def close(self):
        """Clean up thread executor resources."""
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        self.close()


def run_benchmark(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    num_envs: int = 16,
    steps: int = 50,
    num_threads: Optional[int] = None,
    decimation: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run parallel simulation benchmark and calculate FPS metrics."""
    runner = ParallelSimRunner(
        model_path=model_path,
        num_envs=num_envs,
        decimation=decimation,
        num_threads=num_threads,
    )

    try:
        # Warm-up step
        warmup_actions = np.zeros((num_envs, runner.action_dim), dtype=np.float32)
        runner.step(warmup_actions)

        # Generate action batches for benchmark
        rng = np.random.default_rng(seed=42)

        start_time = time.perf_counter()
        for _ in range(steps):
            step_actions = rng.uniform(-1.0, 1.0, size=(num_envs, runner.action_dim)).astype(np.float32)
            runner.step(step_actions)
        elapsed_time = time.perf_counter() - start_time

        total_steps = num_envs * steps
        fps = total_steps / elapsed_time if elapsed_time > 0 else 0.0
        real_time_factor = (total_steps * runner.control_dt) / elapsed_time if elapsed_time > 0 else 0.0

        metrics = {
            "stack": "dual_esp32s3_stack",
            "num_envs": num_envs,
            "steps": steps,
            "total_steps": total_steps,
            "elapsed_time": elapsed_time,
            "fps": fps,
            "real_time_factor": real_time_factor,
            "sim_dt": runner.sim_dt,
            "control_dt": runner.control_dt,
            "decimation": runner.decimation,
            "threads": runner.num_threads,
            "obs_dim": runner.obs_dim,
            "action_dim": runner.action_dim,
        }

        if verbose:
            print("=" * 65)
            print("Parallel Simulation Benchmark: MicroDuck 8-DOF (Dual ESP32-S3)")
            print("=" * 65)
            print(f"Model: {model_path}")
            print(f"Perception Obs Space: {runner.obs_dim} dimensions (includes 5 LiDAR rays)")
            print(f"Environments: {num_envs}")
            print(f"Steps per Env: {steps}")
            print(f"Total Steps: {total_steps}")
            print(
                f"Decimation: {runner.decimation} (Physics dt: {runner.sim_dt:.4f}s, Control dt: {runner.control_dt:.4f}s)"
            )
            print(f"Threads: {runner.num_threads}")
            print(f"Elapsed Time: {elapsed_time:.4f}s")
            print(f"Steps Per Second (FPS): {fps:.2f}")
            print(f"Real-time Factor: {real_time_factor:.2f}x")
            print("=" * 65)

        return metrics
    finally:
        runner.close()


def main():
    """Command-line interface for parallel simulation runner and benchmark."""
    parser = argparse.ArgumentParser(
        description="Parallel Simulation Runner & FPS Benchmark for MicroDuck 8-DOF (Dual ESP32-S3 Stack)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to robot MuJoCo MJCF XML model",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments to simulate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of simulation steps per environment",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of worker threads (default: auto)",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=10,
        help="Control step decimation factor (substeps)",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            model_path=args.model_path,
            num_envs=args.num_envs,
            steps=args.steps,
            num_threads=args.threads,
            decimation=args.decimation,
            verbose=True,
        )
        sys.exit(0)
    except Exception as exc:
        print(f"Error during parallel simulation: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
