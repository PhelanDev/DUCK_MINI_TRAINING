"""Tests for obstacle anti-stuck rewards: command relaxation and approach-velocity penalty."""

from pathlib import Path
import sys
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import (
    ang_vel_mask,
    avoid_vel_penalty,
    compute_rewards,
    MicroDuck8DoFConfig,
)
from training.run_parallel_sim import DuckParallelSim


def test_ang_mask_values():
    """Verify angular velocity relaxation mask values across distance thresholds."""
    # d_danger = 0.25, d_safe = 0.55
    d_samples = np.array([0.60, 0.55, 0.40, 0.25, 0.10], dtype=np.float32)
    masks = ang_vel_mask(d_samples)

    assert pytest.approx(float(masks[0]), abs=1e-5) == 1.0
    assert pytest.approx(float(masks[1]), abs=1e-5) == 1.0
    assert pytest.approx(float(masks[2]), abs=1e-5) == 0.5
    assert pytest.approx(float(masks[3]), abs=1e-5) == 0.1
    assert pytest.approx(float(masks[4]), abs=1e-5) == 0.1


def test_avoid_penalty_zero_when_receding():
    """Verify that avoid_vel_penalty is zero when moving away from obstacles (d_min_dot > 0)."""
    d_min = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    d_min_dot = np.array([0.1, 0.3, 0.5], dtype=np.float32)  # Positive = moving away

    penalties = avoid_vel_penalty(d_min, d_min_dot, sigma_obs=0.3)
    assert np.allclose(penalties, 0.0)


def test_avoid_penalty_stronger_when_closer():
    """Verify that approaching obstacles at the same speed is penalized more severely closer to the obstacle."""
    d_close = np.array([0.2], dtype=np.float32)
    d_far = np.array([0.8], dtype=np.float32)
    d_dot = np.array([-0.3], dtype=np.float32)  # Approaching speed 0.3 m/s

    pen_close = avoid_vel_penalty(d_close, d_dot, sigma_obs=0.3)
    pen_far = avoid_vel_penalty(d_far, d_dot, sigma_obs=0.3)

    assert pen_close[0] < 0.0
    assert pen_far[0] < 0.0
    # Penalty closer to obstacle is more negative (larger magnitude)
    assert pen_close[0] < pen_far[0]
    assert abs(pen_close[0]) > abs(pen_far[0])


def test_compute_rewards_backward_compatible():
    """Verify compute_rewards backward compatibility when d_min and d_min_dot are omitted."""
    cfg = MicroDuck8DoFConfig()
    base_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    cmd_lin = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    base_ang = np.array([0.0, 0.0, 0.1], dtype=np.float32)
    cmd_ang = 0.1
    grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)
    f_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    lidar = np.ones(5, dtype=np.float32) * 1.5

    # Run without d_min, d_min_dot
    out_orig = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=cmd_ang,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        lidar_ranges=lidar,
        weights=cfg.REWARD_WEIGHTS,
    )

    # Run with d_min=None, d_min_dot=None explicitly
    out_explicit = compute_rewards(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=cmd_ang,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        lidar_ranges=lidar,
        weights=cfg.REWARD_WEIGHTS,
        d_min=None,
        d_min_dot=None,
    )

    assert pytest.approx(out_orig["total_reward"]) == out_explicit["total_reward"]
    assert pytest.approx(out_orig["obstacle_avoidance"]) == out_explicit["obstacle_avoidance"]
    assert pytest.approx(out_orig["tracking_ang_vel"]) == out_explicit["tracking_ang_vel"]


def test_parallel_sim_tracks_d_min_and_d_min_dot():
    """Verify DuckParallelSim calculates d_min and d_min_dot across steps."""
    sim = DuckParallelSim(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
        decimation=1,
    )
    sim.reset()
    assert hasattr(sim, "prev_d_min")
    assert sim.prev_d_min is not None
    assert sim.prev_d_min.shape == (2,)

    actions = np.zeros((2, 8), dtype=np.float32)
    _, _, _, infos = sim.step(actions)
    assert hasattr(sim, "d_min")
    assert hasattr(sim, "d_min_dot")
    assert sim.d_min.shape == (2,)
    assert sim.d_min_dot.shape == (2,)
