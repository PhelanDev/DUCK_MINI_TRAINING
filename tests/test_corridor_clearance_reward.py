"""Tests for corridor_clearance_reward and its compute_rewards anti-cheating integration."""

from pathlib import Path
import sys
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import (
    MicroDuck8DoFConfig,
    LocomotionMode,
    compute_rewards,
    corridor_clearance_reward,
)


def test_corridor_clearance_prefers_centered_open_gap():
    """Verify corridor_clearance_reward gives higher score when centered vs hugging,

    0 in open space (all >= 1.2m), and handles 1D, 2D batch, and < 5 rays.
    """
    # Centered corridor: front wide open (2.0), left & right symmetric (0.65)
    centered = np.array([0.65, 0.70, 2.0, 0.70, 0.65], dtype=np.float32)
    # Hugging right pillar: right side dangerously close (0.20), left open (1.80)
    hugging = np.array([0.20, 0.25, 0.90, 1.50, 1.80], dtype=np.float32)
    # Open space (all >= 1.2m): must stay zero so open-floor reward is untouched
    open_space = np.ones(5, dtype=np.float32) * 1.5

    # 1D single evaluation
    r_center = corridor_clearance_reward(centered)
    r_hug = corridor_clearance_reward(hugging)
    r_open = corridor_clearance_reward(open_space)

    assert isinstance(r_center, float), f"Expected float for 1D input, got {type(r_center)}"
    assert isinstance(r_hug, float)
    assert isinstance(r_open, float)
    assert r_center > r_hug, f"Centered ({r_center}) should score higher than hugging ({r_hug})"
    assert r_open == pytest.approx(0.0, abs=1e-6), "Open space must yield 0.0"

    # 2D batch evaluation
    batch = np.stack([centered, hugging, open_space], axis=0)
    r_batch = corridor_clearance_reward(batch)

    assert isinstance(r_batch, np.ndarray), f"Expected np.ndarray for batch, got {type(r_batch)}"
    assert r_batch.dtype == np.float32
    assert r_batch.shape == (3,)
    assert r_batch[0] > r_batch[1]
    assert r_batch[2] == pytest.approx(0.0, abs=1e-6)
    assert r_batch[0] == pytest.approx(r_center, abs=1e-6)
    assert r_batch[1] == pytest.approx(r_hug, abs=1e-6)

    # Fallback on fewer than 5 rays (< 5 rays fallback == 0.0)
    r_few_1d = corridor_clearance_reward(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
    assert isinstance(r_few_1d, float)
    assert r_few_1d == 0.0

    r_few_batch = corridor_clearance_reward(np.zeros((4, 3), dtype=np.float32))
    assert isinstance(r_few_batch, np.ndarray)
    assert r_few_batch.shape == (4,)
    assert r_few_batch.dtype == np.float32
    assert np.all(r_few_batch == 0.0)

    # Empty array fallback
    r_empty = corridor_clearance_reward(np.array([], dtype=np.float32))
    assert r_empty == 0.0


def test_corridor_clearance_inactive_in_stairs_mode():
    """Verify corridor_clearance is zero when terrain_mode is 'stairs'."""
    cfg = MicroDuck8DoFConfig()
    lidar = np.array([0.25, 0.35, 1.4, 0.9, 1.0], dtype=np.float32)

    # 1D call
    out_stairs = compute_rewards(
        base_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        lidar_ranges=lidar,
        weights=cfg.REWARD_WEIGHTS,
        terrain_mode="stairs",
    )
    assert out_stairs["corridor_clearance"] == pytest.approx(0.0, abs=1e-6)

    # Batched call
    num_envs = 3
    out_stairs_batch = compute_rewards(
        base_lin_vel=np.tile(np.array([0.2, 0.0, 0.0], dtype=np.float32), (num_envs, 1)),
        cmd_lin_vel=np.tile(np.array([0.2, 0.0, 0.0], dtype=np.float32), (num_envs, 1)),
        base_ang_vel=np.zeros((num_envs, 3), dtype=np.float32),
        cmd_ang_vel=np.zeros(num_envs, dtype=np.float32),
        proj_gravity=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (num_envs, 1)),
        joint_torques=np.zeros((num_envs, 8), dtype=np.float32),
        joint_vel=np.zeros((num_envs, 8), dtype=np.float32),
        foot_contact=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_envs, 1)),
        lidar_ranges=np.tile(lidar, (num_envs, 1)),
        weights=cfg.REWARD_WEIGHTS,
        terrain_mode="stairs",
    )
    assert np.all(out_stairs_batch["corridor_clearance"] == pytest.approx(0.0, abs=1e-6))


def test_corridor_clearance_gated_by_forward_walk_anti_cheating():
    """Verify corridor_clearance is strictly gated by (vel_x > 0.02) & is_walking_cmd & walk_mode_mask.

    Zero velocity, backward velocity, zero command, sit, and lie modes must all yield 0.0.
    """
    cfg = MicroDuck8DoFConfig()
    lidar = np.array([0.65, 0.70, 1.8, 0.70, 0.65], dtype=np.float32)

    def _call_rewards(vel_x: float, cmd_vx: float, mode: LocomotionMode):
        return compute_rewards(
            base_lin_vel=np.array([vel_x, 0.0, 0.0], dtype=np.float32),
            cmd_lin_vel=np.array([cmd_vx, 0.0, 0.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            cmd_ang_vel=0.0,
            proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            joint_torques=np.zeros(8, dtype=np.float32),
            joint_vel=np.zeros(8, dtype=np.float32),
            foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            lidar_ranges=lidar,
            weights=cfg.REWARD_WEIGHTS,
            terrain_mode="hurdles",
            cmd_mode=mode,
        )

    # 1. Forward walking at 0.20 m/s with 0.20 m/s cmd in WALK mode -> positive reward
    out_active = _call_rewards(vel_x=0.20, cmd_vx=0.20, mode=LocomotionMode.WALK)
    assert out_active["corridor_clearance"] > 0.0, "Active forward walking must receive clearance reward"

    # 2. Standing still (vel_x = 0.0) -> must be 0.0
    out_still = _call_rewards(vel_x=0.0, cmd_vx=0.20, mode=LocomotionMode.WALK)
    assert out_still["corridor_clearance"] == pytest.approx(0.0, abs=1e-6)

    # 3. Creeping below gate (vel_x = 0.01 <= 0.02) -> must be 0.0
    out_creep = _call_rewards(vel_x=0.01, cmd_vx=0.20, mode=LocomotionMode.WALK)
    assert out_creep["corridor_clearance"] == pytest.approx(0.0, abs=1e-6)

    # 4. Walking backward (vel_x = -0.10) -> must be 0.0
    out_bwd = _call_rewards(vel_x=-0.10, cmd_vx=0.20, mode=LocomotionMode.WALK)
    assert out_bwd["corridor_clearance"] == pytest.approx(0.0, abs=1e-6)

    # 5. Zero command (cmd_vx = 0.0 <= 0.05, even with vel_x = 0.20) -> must be 0.0
    out_no_cmd = _call_rewards(vel_x=0.20, cmd_vx=0.0, mode=LocomotionMode.WALK)
    assert out_no_cmd["corridor_clearance"] == pytest.approx(0.0, abs=1e-6)

    # 6. Sitting command mode -> must be 0.0
    out_sit = _call_rewards(vel_x=0.20, cmd_vx=0.20, mode=LocomotionMode.SIT)
    assert out_sit["corridor_clearance"] == pytest.approx(0.0, abs=1e-6)

    # 7. Lying command mode -> must be 0.0
    out_lie = _call_rewards(vel_x=0.20, cmd_vx=0.20, mode=LocomotionMode.LIE)
    assert out_lie["corridor_clearance"] == pytest.approx(0.0, abs=1e-6)

    # 8. Batched multi-condition verification
    # env 0: forward walk (active)
    # env 1: standing still
    # env 2: backward walk
    # env 3: sit mode
    # env 4: lie mode
    base_lin_batch = np.array([
        [0.20, 0.0, 0.0],
        [0.00, 0.0, 0.0],
        [-0.10, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.20, 0.0, 0.0],
    ], dtype=np.float32)
    cmd_lin_batch = np.array([
        [0.20, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.20, 0.0, 0.0],
    ], dtype=np.float32)
    modes_batch = np.array([
        int(LocomotionMode.WALK),
        int(LocomotionMode.WALK),
        int(LocomotionMode.WALK),
        int(LocomotionMode.SIT),
        int(LocomotionMode.LIE),
    ], dtype=np.int32)
    lidar_batch = np.tile(lidar, (5, 1))

    out_batch = compute_rewards(
        base_lin_vel=base_lin_batch,
        cmd_lin_vel=cmd_lin_batch,
        base_ang_vel=np.zeros((5, 3), dtype=np.float32),
        cmd_ang_vel=np.zeros(5, dtype=np.float32),
        proj_gravity=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (5, 1)),
        joint_torques=np.zeros((5, 8), dtype=np.float32),
        joint_vel=np.zeros((5, 8), dtype=np.float32),
        foot_contact=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (5, 1)),
        lidar_ranges=lidar_batch,
        weights=cfg.REWARD_WEIGHTS,
        terrain_mode="hurdles",
        cmd_mode=modes_batch,
    )

    batch_clearance = out_batch["corridor_clearance"]
    assert batch_clearance[0] > 0.0
    assert batch_clearance[1] == pytest.approx(0.0, abs=1e-6)
    assert batch_clearance[2] == pytest.approx(0.0, abs=1e-6)
    assert batch_clearance[3] == pytest.approx(0.0, abs=1e-6)
    assert batch_clearance[4] == pytest.approx(0.0, abs=1e-6)
