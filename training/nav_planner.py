"""High-level 2D waypoint & 5-ray LiDAR gap steering planner for biped robot."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from training.env_config import quat_to_yaw

# 5-Ray angles [m30, m15, 0, p15, p30] matching MuJoCo rangefinder fan
RAY_ANGLES = np.array(
    [-np.pi / 6, -np.pi / 12, 0.0, np.pi / 12, np.pi / 6], dtype=np.float32
)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class NavPlannerConfig:
    max_vx: float = 0.30
    max_vy: float = 0.06
    max_wz: float = 0.55
    safe_dist: float = 0.45
    goal_tol: float = 0.15
    max_range: float = 2.0


class LidarWaypointPlanner:
    """Computes [vx_cmd, vy_cmd, wz_cmd] for 2D waypoint navigation with gap steering."""

    def __init__(self, cfg: Optional[NavPlannerConfig] = None):
        self.cfg = cfg or NavPlannerConfig()

    def is_goal_reached(self, pos_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
        """Check if 2D position is within goal_tol radius of target waypoint."""
        diff = np.asarray(goal_xy, dtype=np.float32)[:2] - np.asarray(pos_xy, dtype=np.float32)[:2]
        return bool(float(np.linalg.norm(diff)) <= self.cfg.goal_tol)

    def compute_command(
        self,
        pos_xy: np.ndarray,
        yaw: float,
        goal_xy: np.ndarray,
        lidar_ranges: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute navigation command [vx, vy, wz] towards goal with 5-ray gap steering."""
        if self.is_goal_reached(pos_xy, goal_xy):
            return np.zeros(3, dtype=np.float32)

        p = np.asarray(pos_xy, dtype=np.float32)[:2]
        g = np.asarray(goal_xy, dtype=np.float32)[:2]
        delta = g - p
        dist = float(np.linalg.norm(delta))
        goal_bearing = wrap_to_pi(float(np.arctan2(delta[1], delta[0])) - float(yaw))

        target_steer = goal_bearing
        fwd_clearance = self.cfg.max_range

        if lidar_ranges is not None:
            flat_lidar = np.asarray(lidar_ranges, dtype=np.float32).ravel()
            if len(flat_lidar) >= 5:
                # Handle MuJoCo -1 / NaN (no obstacle) and clip to [0, max_range]
                ranges = np.where(np.isnan(flat_lidar[:5]) | (flat_lidar[:5] < 0.0), self.cfg.max_range, flat_lidar[:5])
                ranges = np.clip(ranges, 0.0, self.cfg.max_range)
                fwd_clearance = float(np.min(ranges[1:4]))  # center 3 rays (-15, 0, +15)

                if np.any(ranges < self.cfg.safe_dist):
                    # Balance free space vs goal direction; penalize blocked rays (< safe_dist)
                    norm_ranges = ranges / max(self.cfg.max_range, 1e-4)
                    ang_diffs = np.array([abs(wrap_to_pi(float(a - goal_bearing))) for a in RAY_ANGLES], dtype=np.float32)
                    scores = norm_ranges - ang_diffs - 2.0 * (ranges < self.cfg.safe_dist)
                    best_idx = int(np.argmax(scores))
                    target_steer = float(RAY_ANGLES[best_idx])

        # Velocity scaling: slow down before tight gaps or only within final braking radius
        clearance_scale = float(np.clip(fwd_clearance / max(self.cfg.safe_dist, 1e-4), 0.35, 1.0))
        heading_scale = max(0.0, float(np.cos(target_steer)))
        brake_dist = max(2.0 * self.cfg.goal_tol, 0.25)
        dist_scale = min(1.0, dist / brake_dist)

        vx = float(np.clip(self.cfg.max_vx * clearance_scale * heading_scale * dist_scale, 0.0, self.cfg.max_vx))
        vy = 0.0  # shortcut: biped forward-walking primary, lateral crab-walk disabled
        wz = float(np.clip(target_steer * 1.5, -self.cfg.max_wz, self.cfg.max_wz))

        return np.array([vx, vy, wz], dtype=np.float32)

    def compute_command_from_quat(
        self,
        pos_xy: np.ndarray,
        quat_wxyz: np.ndarray,
        goal_xy: np.ndarray,
        lidar_ranges: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute navigation command extracting heading yaw from quaternion [w, x, y, z]."""
        yaw = float(quat_to_yaw(quat_wxyz))
        return self.compute_command(pos_xy, yaw, goal_xy, lidar_ranges)
