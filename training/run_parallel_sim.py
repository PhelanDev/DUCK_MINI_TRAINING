"""Batched Parallel Simulation Runner and FPS Benchmark for MicroDuck 8-DOF.

Executes vectorized MuJoCo simulation instances sharing a single mjModel across
multiple environments using multi-threaded stepping or batched loops.
Integrates with training.env_config for observation packing, action scaling,
rate limiting, and reward computation.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    DEFAULT_NOMINAL_QPOS,
    EmotionState,
    NUM_EMOTIONS,
    LocomotionMode,
    SIT_NOMINAL_QPOS,
    TARGET_SIT_HEIGHT,
    LIE_NOMINAL_QPOS,
    TARGET_LIE_HEIGHT,
    get_emotion_one_hot,
    clip_action_rates,
    compute_observations,
    compute_rewards,
    compute_gait_phase,
    quat_rotate_inverse,
    quat_to_yaw,
    scale_actions,
)
from training.terrain_curriculum import apply_stage, apply_stair_stage, STAIR_STAGES, randomize_pillars
from training.self_righting import FallClassifier, SelfRightingController
from training.nav_planner import LidarWaypointPlanner, NavPlannerConfig


def has_obstacle_contact(data: Any, obstacle_gid_set: set) -> bool:
    """Check if any active contact involves a geom in obstacle_gid_set."""
    if not obstacle_gid_set or getattr(data, "ncon", 0) == 0:
        return False
    for c_idx in range(data.ncon):
        c = data.contact[c_idx]
        if c.geom1 in obstacle_gid_set or c.geom2 in obstacle_gid_set:
            return True
    return False


def _find_first_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, candidates: Tuple[str, ...]) -> int:
    """Find ID of first existing named object from candidate names."""
    for name in candidates:
        obj_id = mujoco.mj_name2id(model, obj_type, name)
        if obj_id != -1:
            return obj_id
    return -1


LIDAR_SITE_NAMES: Tuple[str, ...] = (
    "site_lidar_m30",
    "site_lidar_m15",
    "site_lidar_0",
    "site_lidar_p15",
    "site_lidar_p30",
)


def extract_filtered_lidar_ranges(
    model: Optional[mujoco.MjModel],
    data: Any,
    obs_dim: int = 42,
    lidar_max: float = 2.0,
    site_ids: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """Extract 5-ray LiDAR readings in meters with floor-hit filtering (`z_hit <= 0.012`)."""
    if len(data.sensordata) >= 33:
        raw_lidar = np.array(data.sensordata[28:33], dtype=np.float32)
        lidar_ranges = np.where(raw_lidar < 0.0, lidar_max, raw_lidar)
        if site_ids is None and model is not None:
            site_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
                for sname in LIDAR_SITE_NAMES
            ]
        if site_ids and hasattr(data, "site_xpos") and hasattr(data, "site_xmat"):
            for k, sid in enumerate(site_ids):
                if sid != -1 and 0.0 <= raw_lidar[k] < lidar_max:
                    z_hit = float(data.site_xpos[sid, 2]) + float(raw_lidar[k]) * float(data.site_xmat[sid, 8])
                    if z_hit <= 0.012:
                        lidar_ranges[k] = lidar_max
        return lidar_ranges
    if obs_dim == 42:
        return np.full(5, lidar_max, dtype=np.float32)
    return None


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
        terrain_mode: str = "hurdles",
        stand_still: bool = False,
        include_emotions: bool = False,
        enable_auto_recovery: bool = False,
        walk_min_z: float = 0.060,
        friction_range: Tuple[float, float] = (1.1, 1.6),
        embed_phase_clock: Optional[bool] = None,
        randomize_pillars: bool = False,
        pillar_density: float = 1.0,
        pillar_jitter_xy: float = 0.06,
        pillar_jitter: Optional[float] = None,
        use_nav_planner: bool = False,
        nav_goal_xy: Tuple[float, float] = (2.0, 0.0),
        nav_planner_cfg: Optional[NavPlannerConfig] = None,
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found at {self.model_path}")

        self.terrain_mode = str(terrain_mode)
        self.randomize_pillars_enabled = bool(randomize_pillars)
        self.pillar_density = float(pillar_density)
        self.pillar_jitter_xy = float(pillar_jitter if pillar_jitter is not None else pillar_jitter_xy)
        self.include_emotions = bool(include_emotions)
        self.walk_min_z = float(walk_min_z)
        self.friction_range = (float(friction_range[0]), float(friction_range[1]))
        self.cfg = cfg if cfg is not None else _DEFAULT_CFG
        if embed_phase_clock is not None:
            self.embed_phase_clock = bool(embed_phase_clock)
        else:
            self.embed_phase_clock = (getattr(self.cfg, "STACK_NAME", "microduck_8dof") == "microduck_8dof")
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
        if self.include_emotions:
            self.base_obs_dim = 42 if self.cfg.OBS_DIM in (37, 42) else self.cfg.OBS_DIM
            self.obs_dim = self.base_obs_dim + NUM_EMOTIONS
        elif len(self.datas[0].sensordata) >= 33 or self.cfg.OBS_DIM == 42:
            self.base_obs_dim = 42
            self.obs_dim = 42
        else:
            self.base_obs_dim = self.cfg.OBS_DIM
            self.obs_dim = self.cfg.OBS_DIM
        self.privileged_obs_dim = self.obs_dim + 11
        self.action_dim = self.cfg.ACTION_DIM

        self.last_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.last_targets = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.stand_still = bool(stand_still)
        if not self.stand_still:
            # Default forward velocity command [vx=0.5, vy=0.0, wz=0.0]
            self.commands[:, 0] = 0.5

        self.max_episode_steps = int(max_episode_steps)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_ep_lens: deque = deque(maxlen=100)
        self.completed_distances_x: deque = deque(maxlen=100)
        self.completed_distances_y: deque = deque(maxlen=100)
        self.feet_air_time = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.last_feet_contact = np.ones((self.num_envs, 2), dtype=bool)
        self.double_support_time = np.zeros(self.num_envs, dtype=np.float32)
        self.last_joint_vel = np.zeros((self.num_envs, 8), dtype=np.float32)
        self.rng = np.random.default_rng(seed=42)
        self.current_emotions = self.rng.integers(0, NUM_EMOTIONS, size=self.num_envs, dtype=np.int32)
        self.cmd_modes = np.zeros(self.num_envs, dtype=np.int32)
        self.tilt_exceed_count = np.zeros(self.num_envs, dtype=np.int32)

        # Obstacle avoidance and approach-velocity tracking state
        self.control_dt = float(self.model.opt.timestep * self.decimation)
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

        self.lidar_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sname)
            for sname in ("site_lidar_m30", "site_lidar_m15", "site_lidar_0", "site_lidar_p15", "site_lidar_p30")
        ]

        self.prev_d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min_dot = np.zeros(self.num_envs, dtype=np.float32)

        # Foot bodies and geoms for clearance reward and stumble detection (supporting V1 & V2 MJCF)
        self.left_foot_gid = _find_first_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ("left_foot_geom", "left_sole_geom"))
        self.right_foot_gid = _find_first_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ("right_foot_geom", "right_sole_geom"))
        self.left_foot_bid = _find_first_id(self.model, mujoco.mjtObj.mjOBJ_BODY, ("left_foot_link", "left_foot"))
        self.right_foot_bid = _find_first_id(self.model, mujoco.mjtObj.mjOBJ_BODY, ("right_foot_link", "right_foot"))

        # Floor geom and base body for domain randomization (friction and torso mass)
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.base_body_id = _find_first_id(self.model, mujoco.mjtObj.mjOBJ_BODY, ("base_link", "trunk_base"))
        self.default_base_mass = float(self.model.body_mass[self.base_body_id]) if self.base_body_id != -1 else 0.32
        self._randomize_model_dynamics()

        # Curriculum stage
        self.curriculum_stage = int(stage)
        self.set_stage(self.curriculum_stage)

        # Determine staircase bounding box for edge-fall detection
        self.stair_min_x = 0.08
        self.stair_max_x = 1.10
        self.stair_half_w = 0.40
        stair_gids = []
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and ("stair" in gname or "platform" in gname):
                stair_gids.append(gid)
        self.terrain_gids = stair_gids
        if stair_gids:
            self.stair_min_x = float(min(self.model.geom_pos[g, 0] - self.model.geom_size[g, 0] for g in stair_gids))
            self.stair_max_x = float(max(self.model.geom_pos[g, 0] + self.model.geom_size[g, 0] for g in stair_gids))
            self.stair_half_w = float(min(self.model.geom_size[g, 1] for g in stair_gids))

        # Self-righting recovery controllers
        self.enable_auto_recovery = bool(enable_auto_recovery)
        if self.enable_auto_recovery:
            self.recovery_controllers: List[SelfRightingController] = [
                SelfRightingController() for _ in range(self.num_envs)
            ]
        else:
            self.recovery_controllers = []

        # Hierarchical waypoint navigation planner (L2)
        self.use_nav_planner = bool(use_nav_planner)
        self.nav_goal_xy = tuple(nav_goal_xy)
        if self.use_nav_planner:
            self.nav_planner = LidarWaypointPlanner(nav_planner_cfg)
        else:
            self.nav_planner = None

        # Initialize environments
        self.reset()

    def set_stand_still(self, stand_still: bool) -> None:
        """Toggle stationary standing mode (zero velocity commands)."""
        self.stand_still = bool(stand_still)
        if self.stand_still:
            self.commands[:, :] = 0.0

    def set_emotions(self, emotions: Union[int, EmotionState, np.ndarray, List[int]]) -> None:
        """Set emotion state across environments."""
        if isinstance(emotions, (int, EmotionState)):
            self.current_emotions[:] = int(emotions)
        else:
            self.current_emotions[:] = np.asarray(emotions, dtype=np.int32)

    def set_cmd_modes(self, modes: Union[int, List[int], np.ndarray, LocomotionMode]) -> None:
        """Set locomotion mode (WALK=0, SIT=1, LIE=2) across environments."""
        if isinstance(modes, (int, LocomotionMode)):
            self.cmd_modes[:] = int(modes)
        else:
            self.cmd_modes[:] = np.asarray(modes, dtype=np.int32)

    def set_cmd_mode(self, mode: Union[int, List[int], np.ndarray, LocomotionMode]) -> None:
        """Alias for set_cmd_modes."""
        self.set_cmd_modes(mode)

    def set_stage(self, stage: int) -> None:
        """Update terrain and obstacle curriculum stage."""
        prev_stage = getattr(self, "curriculum_stage", None)
        self.curriculum_stage = int(stage)
        if self.terrain_mode == "stairs":
            apply_stair_stage(self.model, self.curriculum_stage)
        else:
            apply_stage(self.model, self.curriculum_stage)
        if prev_stage is not None and self.curriculum_stage != prev_stage:
            self.completed_ep_lens.clear()
            self.completed_distances_x.clear()
            self.completed_distances_y.clear()

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

    def _get_lidar_ranges(self, env_idx: int) -> Optional[np.ndarray]:
        """Extract 5-ray LiDAR readings in meters with floor hit filtering."""
        lidar_max = float(getattr(self.cfg, "LIDAR_MAX_RANGE", 2.0))
        return extract_filtered_lidar_ranges(
            self.model,
            self.datas[env_idx],
            obs_dim=self.base_obs_dim,
            lidar_max=lidar_max,
            site_ids=self.lidar_site_ids,
        )

    def update_nav_commands(
        self,
        env_indices: Optional[Union[List[int], np.ndarray, range]] = None,
        lidar_obs: Optional[np.ndarray] = None,
    ) -> None:
        """Update self.commands for specified environments using LidarWaypointPlanner.

        Extracts current robot pose (qpos[:2], qpos[3:7]) and 5 LiDAR rays.
        """
        if not getattr(self, "use_nav_planner", False) or getattr(self, "nav_planner", None) is None:
            return

        if env_indices is None:
            indices = list(range(self.num_envs))
        else:
            indices = [int(i) for i in env_indices]

        goal_xy = np.asarray(self.nav_goal_xy, dtype=np.float32)[:2]

        for idx_k, env_idx in enumerate(indices):
            d = self.datas[env_idx]
            pos_xy = np.asarray(d.qpos[:2], dtype=np.float32)
            quat_wxyz = np.asarray(d.qpos[3:7], dtype=np.float32)

            if lidar_obs is not None:
                lidar_arr = np.asarray(lidar_obs, dtype=np.float32)
                if lidar_arr.ndim == 1:
                    raw_ray = lidar_arr[-5:] if lidar_arr.shape[0] >= 5 else None
                elif lidar_arr.ndim == 2:
                    k = idx_k if lidar_arr.shape[0] == len(indices) else env_idx
                    raw_ray = lidar_arr[k, 37:42] if lidar_arr.shape[1] >= 42 else lidar_arr[k, :5]
                else:
                    raw_ray = None
            else:
                raw_ray = self._get_lidar_ranges(env_idx)

            cmd = self.nav_planner.compute_command_from_quat(
                pos_xy=pos_xy,
                quat_wxyz=quat_wxyz,
                goal_xy=goal_xy,
                lidar_ranges=raw_ray,
            )
            self.commands[env_idx, :3] = cmd

    def _has_active_obstacles(self) -> bool:
        """Return True if at least one obstacle geom is raised above the floor (z > 0.0)."""
        return any(float(self.model.geom_pos[gid, 2]) > 0.0 for gid in self.obstacle_geom_ids)

    def _compute_d_min(self, env_idx: Optional[int] = None) -> Optional[Union[float, np.ndarray]]:
        """Compute minimum distance to obstacle cylinders (minus individual cylinder radius)."""
        if not self.obstacle_geom_ids:
            return None
        active_gids = [gid for gid in self.obstacle_geom_ids if self.model.geom_pos[gid, 2] > 0.0]
        if not active_gids:
            return None
        obs_xy = np.array([self.model.geom_pos[gid, :2] for gid in active_gids], dtype=np.float32)
        obs_radii = np.array([self.model.geom_size[gid, 0] for gid in active_gids], dtype=np.float32)
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

    def randomize_stage_pillars(self) -> List[str]:
        """Synchronously randomize active slalom pillars on shared model outside worker threads."""
        if self.randomize_pillars_enabled and self.terrain_mode == "hurdles" and self.curriculum_stage >= 1:
            return randomize_pillars(
                self.model,
                stage=self.curriculum_stage,
                rng=self.rng,
                density_rho=self.pillar_density,
                jitter_xy=self.pillar_jitter_xy,
            )
        return []

    def _randomize_model_dynamics(self):
        """Synchronously randomize floor friction and torso mass on shared model outside worker threads."""
        if self.floor_gid != -1:
            self.model.geom_friction[self.floor_gid, 0] = float(
                self.rng.uniform(self.friction_range[0], self.friction_range[1])
            )
        if self.base_body_id != -1:
            self.model.body_mass[self.base_body_id] = float(
                self.default_base_mass * self.rng.uniform(0.95, 1.05)
            )

    def reset_env(self, env_idx: int) -> np.ndarray:
        """Reset a single environment with random velocity commands."""
        d = self.datas[env_idx]
        mujoco.mj_resetData(self.model, d)

        # Domain randomization on shared self.model is performed synchronously in
        # reset() / _randomize_model_dynamics() outside worker thread execution to avoid data races.
        if self.num_envs == 1:
            self._randomize_model_dynamics()

        if self.include_emotions:
            self.current_emotions[env_idx] = self.rng.integers(0, NUM_EMOTIONS)

        d.qpos[0:3] = [0.0, 0.0, 0.128]
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

        # Ngẫu nhiên hóa lệnh vận tốc mục tiêu [vx, vy, wz] (hoặc giữ đứng yên nếu stand_still)
        if self.stand_still:
            self.commands[env_idx, :] = 0.0
        elif self.terrain_mode in ("hurdles", "stairs"):
            self.commands[env_idx, 0] = self.rng.uniform(0.18, 0.25)
            self.commands[env_idx, 1] = self.rng.uniform(-0.03, 0.03)
            wz_lim = min(0.10, 0.04 + 0.02 * getattr(self, "curriculum_stage", 0))
            self.commands[env_idx, 2] = self.rng.uniform(-wz_lim, wz_lim)
        else:
            self.commands[env_idx, 0] = self.rng.uniform(0.10, 0.25)
            self.commands[env_idx, 1] = self.rng.uniform(-0.08, 0.08)
            self.commands[env_idx, 2] = self.rng.uniform(-0.50, 0.50)

        mujoco.mj_forward(self.model, d)

        self.last_actions[env_idx] = 0.0
        self.last_targets[env_idx] = d.qpos[7:15].astype(np.float32)
        self.last_joint_vel[env_idx] = 0.0
        self.double_support_time[env_idx] = 0.0
        self.episode_steps[env_idx] = 0
        self.feet_air_time[env_idx] = 0.0
        self.last_feet_contact[env_idx] = True
        if self.enable_auto_recovery and env_idx < len(self.recovery_controllers):
            self.recovery_controllers[env_idx].reset()
        if self.use_nav_planner:
            self.update_nav_commands(env_indices=[env_idx])
        obs = self._extract_obs(env_idx)
        if not self._has_active_obstacles():
            d_init = None
        elif self.base_obs_dim == 42 and len(obs) >= 42:
            d_init = float(np.min(obs[37:42]))
        else:
            d_init = self._compute_d_min(env_idx)
        if d_init is not None:
            self.prev_d_min[env_idx] = d_init
            self.d_min[env_idx] = d_init
            self.d_min_dot[env_idx] = 0.0
        else:
            self.prev_d_min[env_idx] = 0.0
            self.d_min[env_idx] = 0.0
            self.d_min_dot[env_idx] = 0.0
        return obs

    def reset(self, env_ids: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """Reset specified environments or all if None. Returns array of observations."""
        self._randomize_model_dynamics()
        if env_ids is None:
            self.randomize_stage_pillars()
            indices = range(self.num_envs)
        else:
            indices = env_ids

        for i in indices:
            self.reset_env(i)

        if self.use_nav_planner:
            self.update_nav_commands(env_indices=indices)

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
        lidar_ranges = self._get_lidar_ranges(env_idx)

        emotion_one_hot = (
            get_emotion_one_hot(self.current_emotions[env_idx])
            if self.include_emotions
            else None
        )

        if self.embed_phase_clock:
            _, clock_sin, clock_cos = compute_gait_phase(
                self.episode_steps[env_idx], dt_control=self.control_dt, period=0.8
            )
            phase_clock = np.array([clock_sin, clock_cos], dtype=np.float32)
        else:
            phase_clock = None

        return compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_act,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges if self.base_obs_dim == 42 else None,
            emotion_one_hot=emotion_one_hot,
            phase_clock=phase_clock,
        )

    def get_observations(self) -> np.ndarray:
        """Extract all current observations as a (num_envs, 37) numpy array."""
        obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.num_envs):
            obs[i] = self._extract_obs(i)
        return obs

    def _get_terrain_height_at_xy(self, x: float, y: float) -> float:
        """Query top elevation of terrain / stairs at world coordinate (x, y)."""
        h = 0.0
        terrain_gids = getattr(self, "terrain_gids", None)
        if terrain_gids:
            for gid in terrain_gids:
                gx, gy, gz = self.model.geom_pos[gid]
                sx, sy, sz = self.model.geom_size[gid]
                top_z = float(gz + sz)
                if top_z > h and abs(x - gx) <= sx and abs(y - gy) <= sy:
                    h = top_z
        return float(h)

    def get_privileged_obs(
        self,
        raw_obs: Optional[np.ndarray] = None,
        env_indices: Optional[Union[List[int], np.ndarray]] = None,
    ) -> np.ndarray:
        """Extract privileged observation tensor of shape (N, obs_dim + 11).

        Components:
        - raw_obs (obs_dim): standard actor observation vector
        - base_lin_vel (3): linear velocity in base body frame
        - contact_forces_z (2): vertical contact forces on left/right foot soles
        - terrain_height_at_feet (2): terrain elevation directly under left/right foot
        - nearest_obstacle_dist (1): distance to nearest obstacle surface
        - nearest_obstacle_rel_xy (2): relative 2D offset to nearest obstacle in robot body frame
        - current_stage_norm (1): curriculum stage normalized to [0, 1]
        """
        if env_indices is None:
            indices = list(range(self.num_envs))
        else:
            indices = [int(i) for i in env_indices]

        n_envs = len(indices)
        if n_envs == 0:
            return np.empty((0, self.privileged_obs_dim), dtype=np.float32)

        if raw_obs is None:
            actor_obs = np.array([self._extract_obs(i) for i in indices], dtype=np.float32)
        else:
            raw_arr = np.asarray(raw_obs, dtype=np.float32)
            if raw_arr.shape[0] == self.num_envs and len(indices) < self.num_envs:
                actor_obs = raw_arr[indices]
            else:
                actor_obs = raw_arr

        active_obstacle_gids = [
            gid for gid in self.obstacle_geom_ids if float(self.model.geom_pos[gid, 2]) > 0.0
        ]
        has_obs = len(active_obstacle_gids) > 0
        if has_obs:
            obs_xy = np.array([self.model.geom_pos[gid, :2] for gid in active_obstacle_gids], dtype=np.float32)
            obs_radii = np.array([self.model.geom_size[gid, 0] for gid in active_obstacle_gids], dtype=np.float32)

        stage_norm = float(getattr(self, "curriculum_stage", 0)) / 3.0
        priv_extra = np.zeros((n_envs, 11), dtype=np.float32)

        for k, idx in enumerate(indices):
            d = self.datas[idx]
            # 1. Base linear velocity in body frame (3)
            base_lin_vel_world = d.qvel[0:3].astype(np.float32)
            base_quat = d.qpos[3:7].astype(np.float32)
            base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel_world)
            priv_extra[k, 0:3] = base_lin_vel_body

            # 2. Foot contact forces Z (2)
            if len(d.sensordata) >= 12:
                priv_extra[k, 3] = float(d.sensordata[10])
                priv_extra[k, 4] = float(d.sensordata[11])

            # 3. Terrain height at feet (2)
            if self.left_foot_bid != -1 and self.right_foot_bid != -1:
                priv_extra[k, 5] = self._get_terrain_height_at_xy(float(d.xpos[self.left_foot_bid][0]), float(d.xpos[self.left_foot_bid][1]))
                priv_extra[k, 6] = self._get_terrain_height_at_xy(float(d.xpos[self.right_foot_bid][0]), float(d.xpos[self.right_foot_bid][1]))

            # 4. Nearest obstacle dist (1) & rel XY in body frame (2)
            if has_obs:
                base_xy = d.qpos[0:2].astype(np.float32)
                diff = obs_xy - base_xy
                dist = np.linalg.norm(diff, axis=-1) - obs_radii
                min_i = int(np.argmin(dist))
                priv_extra[k, 7] = float(np.clip(dist[min_i], 0.0, None))
                yaw = float(quat_to_yaw(base_quat))
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                dx_w, dy_w = float(diff[min_i, 0]), float(diff[min_i, 1])
                priv_extra[k, 8] = float(cos_y * dx_w + sin_y * dy_w)
                priv_extra[k, 9] = float(-sin_y * dx_w + cos_y * dy_w)
            else:
                priv_extra[k, 7] = 2.0
                priv_extra[k, 8] = 2.0
                priv_extra[k, 9] = 0.0

            # 5. Curriculum stage norm (1)
            priv_extra[k, 10] = stage_norm

        return np.concatenate([actor_obs, priv_extra], axis=-1).astype(np.float32)

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
        self, actions: np.ndarray, fast: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Step all environments by one control step with given normalized actions.

        Returns: (next_observations, rewards, dones, infos)
        """
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Expected action shape ({self.num_envs}, {self.action_dim}), got {actions.shape}"

        # 1. Scale actions to joint limits around nominal stance (vectorized)
        scaled_targets = scale_actions(actions, self.cfg, nominal_centered=True)

        if self.enable_auto_recovery:
            for i in range(self.num_envs):
                ctrl = self.recovery_controllers[i]
                if ctrl.is_recovering:
                    qpos = self.datas[i].qpos[7:15]
                    root_z_val = float(self.datas[i].qpos[2])
                    w, x, y, z = self.datas[i].sensordata[6:10]
                    proj_grav_val = np.array([
                        2.0 * (w * y - x * z),
                        -2.0 * (w * x + y * z),
                        2.0 * (x * x + y * y) - 1.0,
                    ], dtype=np.float32)
                    target_q, is_done = ctrl.step(qpos, root_z_val, proj_grav_val)
                    scaled_targets[i] = target_q
                    if ctrl.is_push_completed or (
                        ctrl.state == SelfRightingController.PHASE3_PUSH_TO_STAND
                        and ctrl.phase_steps >= 15
                    ):
                        d = self.datas[i]
                        d.qpos[2] = max(float(d.qpos[2]), 0.128)
                        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                        d.qvel[:] = 0.0
                        mujoco.mj_forward(self.model, d)

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

        self.episode_steps += 1

        # 4. Extract observations, compute rewards, and check done flags
        if self.use_nav_planner:
            self.update_nav_commands()

        next_obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.num_envs):
            next_obs[i] = self._extract_obs(i)

        # Vectorized extraction of kinodynamic tensors
        base_lin_vel_world = np.array([d.qvel[0:3] for d in self.datas], dtype=np.float32)
        base_quat = np.array([d.qpos[3:7] for d in self.datas], dtype=np.float32)
        # Crucial: Project linear velocity from World Frame into Body Frame!
        base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel_world)

        cmd_lin_vel = np.zeros((self.num_envs, 3), dtype=np.float32)
        cmd_lin_vel[:, :2] = self.commands[:, :2]

        base_ang_vel_raw = np.array([d.sensordata[3:6] for d in self.datas], dtype=np.float32)
        # MuJoCo <gyro> sensor is already mounted on torso in body frame matching _extract_obs.
        base_ang_vel_body = base_ang_vel_raw
        cmd_ang_vel = self.commands[:, 2]
        proj_grav = next_obs[:, 3:6]
        joint_torques = np.array([d.ctrl[:] for d in self.datas], dtype=np.float32)
        joint_vel = np.array([d.sensordata[20:28] for d in self.datas], dtype=np.float32)
        joint_pos = np.array([d.sensordata[12:20] for d in self.datas], dtype=np.float32)
        foot_contact = next_obs[:, 33:37]
        lidar_ranges = next_obs[:, 37:42] if self.base_obs_dim == 42 else None
        qpos_head = np.array([d.sensordata[12:14] for d in self.datas], dtype=np.float32)

        current_yaw = quat_to_yaw(base_quat)
        _, clock_sin, clock_cos = compute_gait_phase(
            self.episode_steps, dt_control=self.control_dt, period=0.8
        )

        # Compute obstacle distances and approach velocity
        if not self._has_active_obstacles():
            curr_d_min = None
        elif lidar_ranges is not None:
            curr_d_min = np.min(lidar_ranges, axis=-1)
        else:
            curr_d_min = self._compute_d_min()

        if curr_d_min is not None:
            self.d_min[:] = curr_d_min
            raw_dot = (curr_d_min - self.prev_d_min) / self.control_dt
            invalid_trans = (
                (self.prev_d_min <= 0.0)
                | (self.prev_d_min >= 1.99)
                | (curr_d_min >= 1.99)
            )
            self.d_min_dot[:] = np.where(invalid_trans, 0.0, np.clip(raw_dot, -1.0, 1.0))
            self.prev_d_min[:] = curr_d_min
            reward_d_min = self.d_min
            reward_d_min_dot = self.d_min_dot
        else:
            self.d_min[:] = 0.0
            self.d_min_dot[:] = 0.0
            self.prev_d_min[:] = 0.0
            reward_d_min = None
            reward_d_min_dot = None

        # Extract foot kinematics and 3D contact forces for clearance and stumble rewards
        foot_z, foot_vel_xy, foot_contact_forces = self._extract_foot_dynamics()
        root_x = np.array([d.qpos[0] for d in self.datas], dtype=np.float32)
        root_y = np.array([d.qpos[1] for d in self.datas], dtype=np.float32)
        root_z = np.array([d.qpos[2] for d in self.datas], dtype=np.float32)

        # Feet air time tracking: left foot (0) and right foot (1)
        curr_contact = (foot_contact[:, :2] > 0.5)
        touchdown = curr_contact & (~self.last_feet_contact)
        self.feet_air_time[~curr_contact] += self.control_dt

        both_contact = curr_contact[:, 0] & curr_contact[:, 1]
        fwd_moving = np.abs(base_lin_vel_body[:, 0]) >= 0.05
        dt_accum = np.where(fwd_moving, 0.5 * self.control_dt, self.control_dt).astype(np.float32)
        self.double_support_time = np.where(both_contact, self.double_support_time + dt_accum, 0.0).astype(np.float32)

        # Vectorized reward computation across all environments
        reward_dict = compute_rewards(
            base_lin_vel=base_lin_vel_body,
            cmd_lin_vel=cmd_lin_vel,
            base_ang_vel=base_ang_vel_body,
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
            root_x=root_x,
            root_y=root_y,
            root_z=root_z,
            terrain_mode=self.terrain_mode,
            feet_air_time=self.feet_air_time,
            feet_touchdown=touchdown,
            current_emotion=self.current_emotions if self.include_emotions else None,
            qpos_head=qpos_head,
            joint_pos=joint_pos,
            cmd_mode=self.cmd_modes,
            double_support_time=self.double_support_time,
            last_joint_vel=self.last_joint_vel,
            current_yaw=current_yaw,
            target_yaw=0.0,
            clock_sin=clock_sin,
        )
        rewards = reward_dict["total_reward"]
        self.last_joint_vel[:] = joint_vel

        # Reset air time for feet in contact and update last contact history
        self.feet_air_time[curr_contact] = 0.0
        self.last_feet_contact[:] = curr_contact

        # Vectorized termination check: fall over, excessive tilt, numerical divergence, or timeout
        grav_z = proj_grav[:, 2]
        diverged = np.array([
            (not np.all(np.isfinite(d.qpos))) or (not np.all(np.isfinite(d.qvel)))
            for d in self.datas
        ], dtype=bool)

        timeouts = self.episode_steps >= self.max_episode_steps

        is_walk = (self.cmd_modes == int(LocomotionMode.WALK))
        is_sit = (self.cmd_modes == int(LocomotionMode.SIT))
        is_lie = (self.cmd_modes == int(LocomotionMode.LIE))

        is_sitting_legs = (joint_pos[:, 4] > 0.9) | (joint_pos[:, 7] > 0.9)
        walk_moving = is_walk & (self.episode_steps > 1) & (base_lin_vel_body[:, 0] >= 0.03) & (~is_sitting_legs)
        configured_min_z = getattr(self, "walk_min_z", 0.060)
        sit_collapse_z = np.where(is_walk & is_sitting_legs, 0.080, configured_min_z)
        walk_z_thresh = np.where(walk_moving, min(0.055, configured_min_z), sit_collapse_z)
        z_fall_threshold = np.where(
            is_lie,
            0.025,
            np.where(is_sit, 0.050, walk_z_thresh),
        )

        tilt_thresh = np.where(is_lie, 0.0, -0.707)
        exceeding_tilt = (grav_z > tilt_thresh)
        severe_tilt = (grav_z > -0.50)

        self.tilt_exceed_count = np.where(exceeding_tilt, self.tilt_exceed_count + 1, 0)
        tilt_fallen = exceeding_tilt & (
            severe_tilt
            | (self.episode_steps <= 1)
            | (~is_walk)
            | (self.tilt_exceed_count >= 2)
        )

        z_fallen = (root_z < z_fall_threshold)

        # Stage 0 heading yaw termination in WALK mode (when straight walking is commanded)
        yaw_fallen = np.zeros(self.num_envs, dtype=bool)
        if self.curriculum_stage == 0:
            straight_cmd = np.abs(self.commands[:, 2]) < 0.05
            yaw_fallen = is_walk & straight_cmd & (np.abs(current_yaw) > 1.0)

        fallen = z_fallen | tilt_fallen | yaw_fallen | diverged

        edge_fallen = np.zeros(self.num_envs, dtype=bool)
        if self.terrain_mode == "stairs":
            stair_cfg = STAIR_STAGES[max(0, min(self.curriculum_stage, len(STAIR_STAGES) - 1))]
            half_w = getattr(self, "stair_half_w", stair_cfg.stair_width / 2.0)
            min_x = getattr(self, "stair_min_x", 0.08)
            max_x = getattr(self, "stair_max_x", 1.10)
            in_stair_x = (root_x >= min_x) & (root_x <= max_x)
            edge_fallen = in_stair_x & (np.abs(root_y) > (half_w + 0.04))
            fallen = fallen | edge_fallen

        corridor_fallen = np.zeros(self.num_envs, dtype=bool)
        if self.terrain_mode == "hurdles":
            in_pillar_zone = (root_x >= 0.20) & (root_x <= 1.20)
            corridor_fallen = in_pillar_zone & (np.abs(root_y) > 0.55)
            fallen = fallen | corridor_fallen

        if self.enable_auto_recovery:
            for i in range(self.num_envs):
                ctrl = self.recovery_controllers[i]
                if diverged[i]:
                    fallen[i] = True
                elif fallen[i]:
                    if ctrl.state == SelfRightingController.IDLE:
                        fall_type = FallClassifier.classify(root_z[i], proj_grav[i], self.cmd_modes[i])
                        ctrl.trigger(fall_type, self.datas[i].qpos[7:15])
                        fallen[i] = False
                    elif ctrl.is_recovering:
                        fallen[i] = False
                    elif ctrl.state == SelfRightingController.DONE:
                        fallen[i] = False
                        ctrl.reset()
                    elif ctrl.state == SelfRightingController.FAILED_LOCKED:
                        fallen[i] = True
                else:
                    if ctrl.state == SelfRightingController.DONE:
                        ctrl.reset()

        fall_penalty = float(self.cfg.REWARD_WEIGHTS.get("fall_penalty", -5.0))
        if np.any(fallen):
            rewards = rewards.copy()
            rewards[fallen] += fall_penalty

        dones = fallen | timeouts
        time_outs = timeouts & ~fallen

        self.last_actions[:] = actions

        if fast:
            if np.any(time_outs):
                timeout_indices = np.where(time_outs)[0]
                terminal_obs = next_obs[time_outs].copy()
                terminal_critic_obs = self.get_privileged_obs(raw_obs=next_obs, env_indices=timeout_indices)
            else:
                terminal_obs = np.empty((0, self.obs_dim), dtype=np.float32)
                terminal_critic_obs = np.empty((0, self.privileged_obs_dim), dtype=np.float32)

            if np.any(dones):
                for i in np.where(dones)[0]:
                    self.completed_ep_lens.append(int(self.episode_steps[i]))
                    self.completed_distances_x.append(float(root_x[i]))
                    self.completed_distances_y.append(float(root_y[i]))
                    next_obs[i] = self.reset_env(i)
            fast_info = {
                "time_outs": time_outs,
                "tracking_lin_vel": float(np.mean(reward_dict["tracking_lin_vel"])) if "tracking_lin_vel" in reward_dict else 0.0,
                "terminal_obs": terminal_obs,
                "terminal_critic_obs": terminal_critic_obs,
            }
            return next_obs, rewards, dones, fast_info

        # Maintain per-env info dictionary list for diagnostic logging / evaluation
        infos: List[Dict[str, Any]] = [
            {k: float(v[i]) for k, v in reward_dict.items()}
            for i in range(self.num_envs)
        ]
        active_obstacle_gids = {
            gid for gid in self.obstacle_geom_ids if float(self.model.geom_pos[gid, 2]) > 0.0
        }
        for i in range(self.num_envs):
            infos[i]["timeout"] = bool(timeouts[i])
            infos[i]["time_outs"] = bool(time_outs[i])
            infos[i]["edge_fallen"] = bool(edge_fallen[i])
            infos[i]["corridor_fallen"] = bool(corridor_fallen[i])
            infos[i]["yaw_fallen"] = bool(yaw_fallen[i])
            infos[i]["tilt_fallen"] = bool(tilt_fallen[i])
            infos[i]["fallen"] = bool(fallen[i])
            infos[i]["distance_x"] = float(root_x[i])
            infos[i]["episode_distance_y"] = float(root_y[i])
            infos[i]["yaw"] = float(current_yaw[i])
            infos[i]["d_min"] = float(self.d_min[i]) if curr_d_min is not None else None
            infos[i]["obstacle_contact"] = bool(has_obstacle_contact(self.datas[i], active_obstacle_gids))
            if self.enable_auto_recovery:
                infos[i]["recovering"] = bool(self.recovery_controllers[i].is_recovering)
                infos[i]["recovery_state"] = int(self.recovery_controllers[i].state)

        if np.any(dones):
            for i in np.where(dones)[0]:
                self.completed_ep_lens.append(int(self.episode_steps[i]))
                self.completed_distances_x.append(float(root_x[i]))
                self.completed_distances_y.append(float(root_y[i]))
                infos[i]["episode_length"] = int(self.episode_steps[i])
                infos[i]["distance_x"] = float(root_x[i])
                infos[i]["episode_distance_y"] = float(root_y[i])
                infos[i]["terminal_obs"] = next_obs[i].copy()
                infos[i]["terminal_critic_obs"] = self.get_privileged_obs(raw_obs=next_obs[i:i+1], env_indices=[i])[0]
                next_obs[i] = self.reset_env(i)

        return next_obs, rewards, dones, infos

    def step_arrays(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fast vectorized step returning NumPy arrays directly without per-env dict allocations."""
        return self.step(actions, fast=True)

    def get_mean_episode_length(self) -> float:
        """Return average episode length from completed episodes window (or current steps if empty)."""
        if len(self.completed_ep_lens) > 0:
            return float(np.mean(self.completed_ep_lens))
        return float(self.episode_steps.mean())

    def get_mean_distance_x(self) -> float:
        """Return average forward distance along X from completed episodes window (or current positions if empty)."""
        if len(self.completed_distances_x) > 0:
            return float(np.mean(self.completed_distances_x))
        return float(np.mean([d.qpos[0] for d in self.datas]))

    def get_mean_distance_y(self) -> float:
        """Return average lateral distance along Y from completed episodes window (or current positions if empty)."""
        if len(self.completed_distances_y) > 0:
            return float(np.mean(self.completed_distances_y))
        return float(np.mean([d.qpos[1] for d in self.datas]))

    def get_success_rate(self, min_dist: float = 1.5) -> float:
        """Return proportion of completed episodes that achieved at least min_dist along X."""
        if len(self.completed_distances_x) > 0:
            return float(sum(1 for d in self.completed_distances_x if d >= min_dist) / len(self.completed_distances_x))
        if self.num_envs == 0:
            return 0.0
        cur_dists = [float(d.qpos[0]) for d in self.datas]
        return float(sum(1 for d in cur_dists if d >= min_dist) / len(cur_dists))

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
