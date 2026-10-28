"""Batched Parallel Simulation Runner and FPS Benchmark for MicroDuck 8-DOF.

Executes vectorized MuJoCo simulation instances sharing a single mjModel across
multiple environments using multi-threaded stepping or batched loops.
Integrates with training.env_config for observation packing, action scaling,
rate limiting, and reward computation.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    clip_action_rates,
    compute_observations,
    compute_rewards,
    quat_rotate_inverse,
    scale_actions,
)
from training.terrain_curriculum import apply_stage


class ParallelSimRunner:
    """Vectorized / batched simulation runner for MicroDuck 8-DOF."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
        num_envs: int = 16,
        cfg: Optional[MicroDuck8DoFConfig] = None,
        decimation: int = 10,
        num_threads: Optional[int] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        max_episode_steps: int = 500,
        stage: int = 0,
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
            # Auto choose reasonable thread count
            cpu_count = os.cpu_count() or 4
            self.num_threads = min(cpu_count, max(1, self.num_envs // 4))
        else:
            self.num_threads = max(1, int(num_threads))

        self.executor = (
            ThreadPoolExecutor(max_workers=self.num_threads)
            if self.num_threads > 1 and self.num_envs > 1
            else None
        )

        # Tracking state buffers
        if len(self.datas[0].sensordata) >= 33 or self.cfg.OBS_DIM == 42:
            self.obs_dim = 42
        else:
            self.obs_dim = self.cfg.OBS_DIM
        self.action_dim = self.cfg.ACTION_DIM

        self.last_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.last_targets = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
        # Default forward velocity command [vx=0.5, vy=0.0, wz=0.0]
        self.commands[:, 0] = 0.5

        self.max_episode_steps = int(max_episode_steps)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.rng = np.random.default_rng(seed=42)

        # Obstacle avoidance and approach-velocity tracking state
        self.control_dt = float(self.model.opt.timestep * self.decimation)
        self.obstacle_geom_ids: List[int] = []
        for name in ["obstacle_front", "obstacle_left", "obstacle_right", "obstacle_back"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self.obstacle_geom_ids.append(gid)

        self.prev_d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min_dot = np.zeros(self.num_envs, dtype=np.float32)

        # Foot bodies and geoms for clearance reward and stumble detection
        self.left_foot_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
        self.right_foot_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
        self.left_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        self.right_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")

        # Curriculum stage
        self.curriculum_stage = int(stage)
        self.set_stage(self.curriculum_stage)

        # Initialize environments
        self.reset()

    def set_stage(self, stage: int) -> None:
        """Update terrain and obstacle curriculum stage."""
        self.curriculum_stage = int(stage)
        apply_stage(self.model, self.curriculum_stage)

    def _extract_foot_dynamics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract foot heights, horizontal velocities, and 3D contact forces in world frame.

        Returns:
            foot_z: (num_envs, 2)
            foot_vel_xy: (num_envs, 2, 2)
            foot_contact_forces: (num_envs, 2, 3)
        """
        foot_z = np.zeros((self.num_envs, 2), dtype=np.float32)
        foot_vel_xy = np.zeros((self.num_envs, 2, 2), dtype=np.float32)
        foot_contact_forces = np.zeros((self.num_envs, 2, 3), dtype=np.float32)

        has_feet = (
            self.left_foot_bid != -1
            and self.right_foot_bid != -1
            and self.left_foot_gid != -1
            and self.right_foot_gid != -1
        )
        if not has_feet:
            return foot_z, foot_vel_xy, foot_contact_forces

        v_buf = np.zeros(6, dtype=np.float64)
        c_force = np.zeros(6, dtype=np.float64)

        for env_idx, d in enumerate(self.datas):
            # 1. Foot Heights
            foot_z[env_idx, 0] = d.xpos[self.left_foot_bid][2]
            foot_z[env_idx, 1] = d.xpos[self.right_foot_bid][2]

            # 2. Foot Velocities in World Frame (flg_local=0)
            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.left_foot_bid, v_buf, 0)
            foot_vel_xy[env_idx, 0] = v_buf[3:5]

            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.right_foot_bid, v_buf, 0)
            foot_vel_xy[env_idx, 1] = v_buf[3:5]

            # 3. 3D Contact Forces on feet transformed to world frame
            for c_idx in range(d.ncon):
                contact = d.contact[c_idx]
                g1 = contact.geom1
                g2 = contact.geom2

                is_left = (g1 == self.left_foot_gid) or (g2 == self.left_foot_gid)
                is_right = (g1 == self.right_foot_gid) or (g2 == self.right_foot_gid)
                if not (is_left or is_right):
                    continue

                mujoco.mj_contactForce(self.model, d, c_idx, c_force)
                r_c2w = contact.frame.reshape(3, 3).T
                f_world = r_c2w @ c_force[0:3]

                if is_left:
                    foot_contact_forces[env_idx, 0] += f_world
                if is_right:
                    foot_contact_forces[env_idx, 1] += f_world

        return foot_z, foot_vel_xy, foot_contact_forces

    def _compute_d_min(self, env_idx: Optional[int] = None) -> Optional[Union[float, np.ndarray]]:
        """Compute minimum distance to obstacle cylinders (minus cylinder radius 0.08m)."""
        if not self.obstacle_geom_ids:
            return None
        obs_xy = np.array([self.model.geom_pos[gid, :2] for gid in self.obstacle_geom_ids], dtype=np.float32)
        if env_idx is not None:
            base_xy = np.array(self.datas[env_idx].qpos[0:2], dtype=np.float32)
            diff = base_xy - obs_xy
            dist = np.linalg.norm(diff, axis=-1) - 0.08
            return float(np.clip(np.min(dist), 0.0, None))
        else:
            base_xy = np.array([d.qpos[0:2] for d in self.datas], dtype=np.float32)
            diff = base_xy[:, None, :] - obs_xy[None, :, :]
            dist = np.linalg.norm(diff, axis=-1) - 0.08
            return np.clip(np.min(dist, axis=-1), 0.0, None).astype(np.float32)

    def reset_env(self, env_idx: int) -> np.ndarray:
        """Reset a single environment with domain randomization and random velocity commands."""
        d = self.datas[env_idx]
        mujoco.mj_resetData(self.model, d)

        d.qpos[0:3] = [0.0, 0.0, 0.170]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Thêm nhiễu nhỏ vào góc khớp ban đầu để đa dạng hóa trạng thái xuất phát
        d.qpos[7:15] = self.rng.uniform(-0.03, 0.03, size=self.action_dim)
        d.qpos[11] = max(0.0, d.qpos[11])  # left_knee_pitch >= 0
        d.qpos[14] = max(0.0, d.qpos[14])  # right_knee_pitch >= 0
        d.qvel[:] = 0.0

        # Ngẫu nhiên hóa lệnh vận tốc mục tiêu [vx, vy, wz]
        self.commands[env_idx, 0] = self.rng.uniform(0.0, 0.25)
        self.commands[env_idx, 1] = self.rng.uniform(-0.08, 0.08)
        self.commands[env_idx, 2] = self.rng.uniform(-0.50, 0.50)

        mujoco.mj_forward(self.model, d)

        self.last_actions[env_idx] = 0.0
        self.last_targets[env_idx] = d.qpos[7:15].astype(np.float32)
        self.episode_steps[env_idx] = 0
        d_init = self._compute_d_min(env_idx)
        if d_init is not None:
            self.prev_d_min[env_idx] = d_init
            self.d_min[env_idx] = d_init
            self.d_min_dot[env_idx] = 0.0
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
        """Extract 37-dimensional observation vector from env data."""
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

        # 5-ray LiDAR rangefinder readings (sensordata indices 28:33) if available
        # MuJoCo rangefinder returns -1.0 if no geom is hit within cutoff distance
        if len(d.sensordata) >= 33:
            raw_lidar = np.array(d.sensordata[28:33], dtype=np.float32)
            lidar_max = getattr(self.cfg, "LIDAR_MAX_RANGE", 2.0)
            lidar_ranges = np.where(raw_lidar < 0.0, lidar_max, raw_lidar)
        elif self.obs_dim == 42:
            lidar_max = getattr(self.cfg, "LIDAR_MAX_RANGE", 2.0)
            lidar_ranges = np.full(5, lidar_max, dtype=np.float32)
        else:
            lidar_ranges = None

        return compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_act,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges if self.obs_dim == 42 else None,
        )

    def get_observations(self) -> np.ndarray:
        """Extract all current observations as a (num_envs, 37) numpy array."""
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
        # Crucial: Project linear velocity from World Frame into Body Frame for tracking reward!
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

        # Compute obstacle distances and approach velocity
        curr_d_min = self._compute_d_min()
        if curr_d_min is not None:
            self.d_min[:] = curr_d_min
            self.d_min_dot[:] = (curr_d_min - self.prev_d_min) / self.control_dt
            self.prev_d_min[:] = curr_d_min
            reward_d_min = self.d_min
            reward_d_min_dot = self.d_min_dot
        else:
            reward_d_min = None
            reward_d_min_dot = None

        # Extract foot kinematics and 3D contact forces for clearance and stumble rewards
        foot_z, foot_vel_xy, foot_contact_forces = self._extract_foot_dynamics()

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
            d_min=reward_d_min,
            d_min_dot=reward_d_min_dot,
            foot_z=foot_z,
            foot_vel_xy=foot_vel_xy,
            foot_contact_forces=foot_contact_forces,
        )
        rewards = reward_dict["total_reward"]

        # Vectorized termination check: fall over, excessive tilt, numerical divergence, or timeout
        root_z = np.array([d.qpos[2] for d in self.datas], dtype=np.float32)
        grav_z = proj_grav[:, 2]
        diverged = np.array([
            (not np.all(np.isfinite(d.qpos))) or (not np.all(np.isfinite(d.qvel)))
            for d in self.datas
        ], dtype=bool)

        self.episode_steps += 1
        timeouts = self.episode_steps >= self.max_episode_steps
        fallen = (root_z < 0.08) | (grav_z > -0.3) | diverged
        dones = fallen | timeouts

        # Maintain per-env info dictionary list for diagnostic logging / evaluation
        infos: List[Dict[str, Any]] = [
            {k: float(v[i]) for k, v in reward_dict.items()}
            for i in range(self.num_envs)
        ]
        for i in range(self.num_envs):
            infos[i]["timeout"] = bool(timeouts[i])
            infos[i]["time_outs"] = bool(timeouts[i] and not fallen[i])

        self.last_actions[:] = actions
        if np.any(dones):
            for i in np.where(dones)[0]:
                infos[i]["terminal_obs"] = next_obs[i].copy()
                next_obs[i] = self.reset_env(i)

        return next_obs, rewards, dones, infos

    def close(self):
        """Clean up thread executor resources."""
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        self.close()


DuckParallelSim = ParallelSimRunner


def run_benchmark(
    model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
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
        # Real-time factor = total simulated environment time / elapsed real time
        real_time_factor = (total_steps * runner.control_dt) / elapsed_time if elapsed_time > 0 else 0.0

        metrics = {
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
        }

        if verbose:
            print("=" * 60)
            print("Parallel Simulation Benchmark: MicroDuck 8-DOF")
            print("=" * 60)
            print(f"Model: {model_path}")
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
            print("=" * 60)

        return metrics
    finally:
        runner.close()


def main():
    """Command-line interface for parallel simulation runner and benchmark."""
    parser = argparse.ArgumentParser(
        description="Parallel Batched Simulation Runner & FPS Benchmark for MicroDuck 8-DOF"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/mjcf/microduck_8dof.xml",
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
