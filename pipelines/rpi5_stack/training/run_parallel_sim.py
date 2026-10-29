"""Batched Parallel Simulation Runner & FPS Benchmark for MicroDuck 8-DOF (RPi 5 Stack).

Executes vectorized MuJoCo simulation instances sharing a single mjModel across
multiple environments using multi-threaded stepping or batched loops.
Extracts 42-dimensional perception observations (including 5 LiDAR rangefinder rays)
and benchmarks stepping throughput for Reinforcement Learning training.
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

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.rpi5_stack.training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    DEFAULT_NOMINAL_QPOS,
    clip_action_rates,
    compute_observations,
    compute_rewards,
    quat_rotate_inverse,
    scale_actions,
)

DEFAULT_MODEL_PATH = (
    REPO_ROOT / "pipelines" / "rpi5_stack" / "models" / "mjcf" / "microduck_rpi5_8dof.xml"
)


class ParallelSimRunner:
    """Vectorized / batched simulation runner for MicroDuck 8-DOF (RPi 5 Stack)."""

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

        self.max_episode_steps = int(max_episode_steps)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_ep_lens: deque = deque(maxlen=100)
        self.rng = np.random.default_rng(seed=42)

        # Obstacle avoidance and approach-velocity tracking state
        self.obstacle_geom_ids: List[int] = []
        _SLALOM_PILLAR_NAMES = [
            "pillar_l1_left", "pillar_l1_right",
            "pillar_l2_center",
            "pillar_l3_left", "pillar_l3_right",
            "pillar_back",
        ]
        _LEGACY_OBSTACLE_NAMES = [
            "obstacle_front", "obstacle_left", "obstacle_right", "obstacle_back",
        ]
        obstacle_names = _SLALOM_PILLAR_NAMES + _LEGACY_OBSTACLE_NAMES
        for name in obstacle_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self.obstacle_geom_ids.append(gid)

        self.prev_d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min_dot = np.zeros(self.num_envs, dtype=np.float32)

        # Initialize environments
        self.reset()

    def _compute_d_min(self, env_idx: Optional[int] = None) -> Optional[Union[float, np.ndarray]]:
        """Compute minimum distance to obstacle cylinders (minus individual cylinder radius)."""
        if not self.obstacle_geom_ids:
            return None
        obs_xy = np.array([self.model.geom_pos[gid, :2] for gid in self.obstacle_geom_ids], dtype=np.float32)
        obs_radii = np.array([self.model.geom_size[gid, 0] for gid in self.obstacle_geom_ids], dtype=np.float32)
        if env_idx is not None:
            base_xy = np.array(self.datas[env_idx].qpos[0:2], dtype=np.float32)
            diff = base_xy - obs_xy
            dist = np.linalg.norm(diff, axis=-1) - obs_radii
            return float(np.clip(np.min(dist), 0.0, None))
        else:
            base_xy = np.array([d.qpos[0:2] for d in self.datas], dtype=np.float32)
            diff = base_xy[:, None, :] - obs_xy[None, :, :]
            dist = np.linalg.norm(diff, axis=-1) - obs_radii[None, :]
            return np.clip(np.min(dist, axis=-1), 0.0, None).astype(np.float32)

    def reset_env(self, env_idx: int) -> np.ndarray:
        """Reset a single environment with domain randomization and random velocity commands."""
        d = self.datas[env_idx]
        mujoco.mj_resetData(self.model, d)

        d.qpos[0:3] = [0.0, 0.0, 0.165]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Khởi tạo góc khớp ban đầu quanh tư thế danh định đứng cân bằng (chùng gối nhẹ)
        nominal = getattr(self.cfg, "nominal_qpos", DEFAULT_NOMINAL_QPOS)
        d.qpos[7:15] = nominal + self.rng.uniform(-0.02, 0.02, size=self.action_dim)
        d.qpos[11] = np.clip(
            d.qpos[11],
            self.cfg.JOINT_LIMITS["left_knee_pitch"][0],
            self.cfg.JOINT_LIMITS["left_knee_pitch"][1],
        )
        d.qpos[14] = np.clip(
            d.qpos[14],
            self.cfg.JOINT_LIMITS["right_knee_pitch"][0],
            self.cfg.JOINT_LIMITS["right_knee_pitch"][1],
        )
        d.qvel[:] = 0.0

        # Ngẫu nhiên hóa lệnh vận tốc mục tiêu [vx, vy, wz]
        self.commands[env_idx, 0] = self.rng.uniform(0.0, 0.25)
        self.commands[env_idx, 1] = self.rng.uniform(-0.08, 0.08)
        self.commands[env_idx, 2] = self.rng.uniform(-0.50, 0.50)

        mujoco.mj_forward(self.model, d)

        self.last_actions[env_idx] = 0.0
        self.last_targets[env_idx] = d.qpos[7:15].astype(np.float32)
        self.episode_steps[env_idx] = 0
        obs = self._extract_obs(env_idx)
        if self.obs_dim == 42 and len(obs) >= 42:
            d_init = float(np.min(obs[37:42]))
        else:
            d_init = self._compute_d_min(env_idx)
        if d_init is not None:
            self.prev_d_min[env_idx] = d_init
            self.d_min[env_idx] = d_init
            self.d_min_dot[env_idx] = 0.0
        return obs

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
        # Binarize normal contact forces (> 0.1 N) to match hardware deployment
        left_touch = float(d.sensordata[10])
        right_touch = float(d.sensordata[11])
        foot_contact = np.array(
            [1.0 if left_touch > 0.1 else 0.0, 1.0 if right_touch > 0.1 else 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

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

        # 1. Scale actions to joint limits (vectorized)
        scaled_targets = scale_actions(actions, self.cfg)

        # 2. Rate-limit target joint changes (vectorized)
        target_positions = clip_action_rates(
            scaled_targets,
            self.last_targets,
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
        for i in range(self.num_envs):
            next_obs[i] = self._extract_obs(i)

        # Vectorized extraction of kinodynamic tensors
        base_lin_vel_world = np.array([d.qvel[0:3] for d in self.datas], dtype=np.float32)
        base_quat = np.array([d.qpos[3:7] for d in self.datas], dtype=np.float32)
        # Project linear velocity from World Frame into Body Frame for tracking reward
        base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel_world)

        cmd_lin_vel = np.zeros((self.num_envs, 3), dtype=np.float32)
        cmd_lin_vel[:, :2] = self.commands[:, :2]

        base_ang_vel = np.array([d.sensordata[3:6] for d in self.datas], dtype=np.float32)
        cmd_ang_vel = self.commands[:, 2]
        proj_grav = next_obs[:, 3:6]
        joint_torques = np.array([d.ctrl[:] for d in self.datas], dtype=np.float32)
        joint_vel = np.array([d.sensordata[20:28] for d in self.datas], dtype=np.float32)
        foot_contact = next_obs[:, 33:37]
        lidar_ranges = next_obs[:, 37:42] if self.obs_dim == 42 else None

        # Obstacle avoidance tracking: compute distance and approach velocity
        if lidar_ranges is not None:
            curr_d_min = np.min(lidar_ranges, axis=-1)
        else:
            curr_d_min = self._compute_d_min()

        if curr_d_min is not None:
            self.d_min[:] = curr_d_min
            self.d_min_dot[:] = (curr_d_min - self.prev_d_min) / self.control_dt
            self.prev_d_min[:] = curr_d_min

        root_x = np.array([d.qpos[0] for d in self.datas], dtype=np.float32)
        root_z = np.array([d.qpos[2] for d in self.datas], dtype=np.float32)

        # Vectorized reward computation across all environments
        reward_dict = compute_rewards(
            base_lin_vel=base_lin_vel_body,
            cmd_lin_vel=cmd_lin_vel,
            base_ang_vel=base_ang_vel,
            cmd_ang_vel=cmd_ang_vel,
            proj_gravity=proj_grav,
            joint_torques=joint_torques,
            joint_vel=joint_vel,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges,
            actions=actions,
            last_actions=self.last_actions,
            weights=self.cfg.REWARD_WEIGHTS,
            root_x=root_x,
            root_z=root_z,
        )
        rewards = reward_dict["total_reward"]

        # Vectorized termination check: fall over, excessive tilt, or numerical divergence
        grav_z = proj_grav[:, 2]
        diverged = np.array([
            (not np.all(np.isfinite(d.qpos))) or (not np.all(np.isfinite(d.qvel)))
            for d in self.datas
        ], dtype=bool)

        self.episode_steps += 1
        timeouts = self.episode_steps >= self.max_episode_steps
        fallen = (root_z < 0.08) | (grav_z > -0.3) | diverged
        fall_penalty = float(self.cfg.REWARD_WEIGHTS.get("fall_penalty", -5.0))
        if np.any(fallen):
            rewards = rewards.copy()
            rewards[fallen] += fall_penalty

        dones = fallen | timeouts

        # Maintain per-env info dictionary list for diagnostic logging / evaluation
        infos: List[Dict[str, Any]] = [
            {k: float(v[i]) for k, v in reward_dict.items()}
            for i in range(self.num_envs)
        ]
        for i in range(self.num_envs):
            infos[i]["timeout"] = bool(timeouts[i])
            infos[i]["time_outs"] = bool(timeouts[i] and not fallen[i])
            infos[i]["fallen"] = bool(fallen[i])
        for i in range(self.num_envs):
            infos[i]["timeout"] = bool(timeouts[i])

        self.last_actions[:] = actions
        if np.any(dones):
            for i in np.where(dones)[0]:
                self.completed_ep_lens.append(int(self.episode_steps[i]))
                infos[i]["episode_length"] = int(self.episode_steps[i])
                infos[i]["terminal_obs"] = next_obs[i].copy()
                next_obs[i] = self.reset_env(i)

        return next_obs, rewards, dones, infos

    def get_mean_episode_length(self, default: float = 0.0) -> float:
        """Return average episode length from completed episodes window."""
        if len(self.completed_ep_lens) == 0:
            return float(default)
        return float(np.mean(self.completed_ep_lens))

    def close(self):
        """Clean up thread executor resources."""
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        self.close()


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
            "stack": "rpi5_stack",
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
            print("Parallel Simulation Benchmark: MicroDuck 8-DOF (RPi 5 Stack)")
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
        description="Parallel Simulation Runner & FPS Benchmark for MicroDuck 8-DOF (RPi 5 Stack)"
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
