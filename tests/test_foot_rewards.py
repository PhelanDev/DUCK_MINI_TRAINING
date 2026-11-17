"""Tests for foot clearance reward and stumble penalty."""

from pathlib import Path
import sys
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import compute_foot_clearance_and_stumble, compute_rewards
from training.run_parallel_sim import DuckParallelSim


def test_stance_foot_zero_clearance_penalty():
    """Verify that stance foot (force >= 1.0 N) receives zero clearance penalty even if height is 0."""
    foot_z = np.array([0.0, 0.0], dtype=np.float32)
    foot_vel_xy = np.zeros((2, 2), dtype=np.float32)
    # Stance force: Fz = 3.0 N (> 1.0 N)
    forces = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]], dtype=np.float32)

    rew_clearance, pen_stumble = compute_foot_clearance_and_stumble(
        foot_z=foot_z,
        foot_vel_xy=foot_vel_xy,
        foot_contact_forces=forces,
        target_height=0.06,
    )
    assert rew_clearance == 0.0
    assert pen_stumble == 0.0


def test_swing_dragging_penalized_more_than_lifted():
    """Verify that dragging foot during swing is penalized more than foot at target height."""
    # Foot 0 is in swing (force = 0), foot 1 is stance (force = 3.0)
    forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    vel = np.array([[0.5, 0.0], [0.0, 0.0]], dtype=np.float32)

    # Case A: foot 0 at target height 0.06 m
    z_lifted = np.array([0.06, 0.0], dtype=np.float32)
    rew_lifted, _ = compute_foot_clearance_and_stumble(
        foot_z=z_lifted,
        foot_vel_xy=vel,
        foot_contact_forces=forces,
        target_height=0.06,
    )

    # Case B: foot 0 dragging on ground (z = 0.0 m)
    z_drag = np.array([0.0, 0.0], dtype=np.float32)
    rew_drag, _ = compute_foot_clearance_and_stumble(
        foot_z=z_drag,
        foot_vel_xy=vel,
        foot_contact_forces=forces,
        target_height=0.06,
    )

    assert pytest.approx(rew_lifted, abs=1e-6) == 0.0
    assert rew_drag < 0.0
    assert rew_drag < rew_lifted  # Dragging is strictly more negative


def test_stumble_triggers_on_lateral_force():
    """Verify stumble penalty triggers on high lateral impact force (f_xy > 2 * |fz| and f_xy > 2.0 N)."""
    foot_z = np.array([0.02, 0.0], dtype=np.float32)
    vel = np.zeros((2, 2), dtype=np.float32)
    # Foot 0 has lateral force Fx = 5.0, Fz = 1.0 (5.0 > 2.0 and 5.0 > 2.0 * 1.0)
    forces = np.array([[5.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=np.float32)

    _, pen_stumble = compute_foot_clearance_and_stumble(
        foot_z=foot_z,
        foot_vel_xy=vel,
        foot_contact_forces=forces,
    )
    assert pytest.approx(pen_stumble) == -1.0


def test_no_stumble_on_small_noise():
    """Verify stumble penalty does NOT trigger on small noise (f_xy <= 2.0 N)."""
    foot_z = np.array([0.02, 0.0], dtype=np.float32)
    vel = np.zeros((2, 2), dtype=np.float32)
    # Foot 0: Fx = 1.5 N, Fz = 0.1 N (1.5 > 2 * 0.1, but 1.5 <= 2.0 N threshold)
    forces = np.array([[1.5, 0.0, 0.1], [0.0, 0.0, 3.0]], dtype=np.float32)

    _, pen_stumble = compute_foot_clearance_and_stumble(
        foot_z=foot_z,
        foot_vel_xy=vel,
        foot_contact_forces=forces,
    )
    assert pen_stumble == 0.0


def test_vectorized_shapes():
    """Verify batched inputs of shape (N, 2, ...) produce output tensors of shape (N,)."""
    N = 4
    foot_z = np.zeros((N, 2), dtype=np.float32)
    vel = np.ones((N, 2, 2), dtype=np.float32) * 0.2
    forces = np.zeros((N, 2, 3), dtype=np.float32)
    # Env 0 has stumble on left foot
    forces[0, 0] = [4.0, 0.0, 0.5]

    rew_clearance, pen_stumble = compute_foot_clearance_and_stumble(
        foot_z=foot_z,
        foot_vel_xy=vel,
        foot_contact_forces=forces,
    )
    assert isinstance(rew_clearance, np.ndarray)
    assert isinstance(pen_stumble, np.ndarray)
    assert rew_clearance.shape == (N,)
    assert pen_stumble.shape == (N,)
    assert pen_stumble[0] == -1.0
    assert pen_stumble[1] == 0.0


def test_parallel_sim_extracts_foot_dynamics():
    """Verify DuckParallelSim extracts foot z, vel, and forces, providing rew_foot_clearance and pen_stumble in infos."""
    sim = DuckParallelSim(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
        decimation=1,
    )
    sim.reset()
    actions = np.zeros((2, 8), dtype=np.float32)
    _, _, _, infos = sim.step(actions)

    for info in infos:
        assert "rew_foot_clearance" in info
        assert "pen_stumble" in info
