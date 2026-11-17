"""Unit tests for Advanced Gait & Behavior Engine (training/advanced_gaits.py)."""

import os
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import JOINT_LIMITS, DEFAULT_NOMINAL_QPOS
from training.advanced_gaits import (
    JOINT_NAMES,
    GaitPattern,
    BipedalGaitGenerator,
    AdvancedBehaviorController,
)
from training.self_righting import FallType, RecoveryState


def test_gait_pattern_enum():
    """Verify all gait patterns exist with correct string representations."""
    assert GaitPattern.WALK.value == "walk"
    assert GaitPattern.WADDLE.value == "waddle"
    assert GaitPattern.SPOT_TURN_LEFT.value == "spot_turn_left"
    assert GaitPattern.SPOT_TURN_RIGHT.value == "spot_turn_right"
    assert GaitPattern.HEAD_DANCE.value == "head_dance"


def test_bipedal_gait_generator_joint_limits():
    """Verify that every gait pattern strictly respects MG996R servo limits over multiple cycles."""
    gen = BipedalGaitGenerator(default_freq=1.0)
    patterns = [
        GaitPattern.WALK,
        GaitPattern.WADDLE,
        GaitPattern.SPOT_TURN_LEFT,
        GaitPattern.SPOT_TURN_RIGHT,
        GaitPattern.HEAD_DANCE,
    ]

    for pat in patterns:
        for step in range(50):
            t = step * 0.04  # 2.0 seconds total
            targets = gen.compute_gait(t, pat)
            assert targets.shape == (8,), f"Pattern {pat} returned wrong shape {targets.shape}"
            assert targets.dtype == np.float32

            # Check individual joint boundaries
            for i, jname in enumerate(JOINT_NAMES):
                low, high = JOINT_LIMITS[i, 0], JOINT_LIMITS[i, 1]
                assert low <= targets[i] <= high, (
                    f"Joint {jname} limit exceeded in pattern {pat}: {targets[i]:.4f} not in [{low}, {high}]"
                )


def test_waddle_gait_characteristics():
    """Verify that Waddling gait has larger lateral sway and coordinated head nodding than walk."""
    gen = BipedalGaitGenerator(default_freq=1.0)
    t_samples = np.linspace(0.0, 1.0, 50)

    walk_sways = []
    waddle_sways = []
    waddle_head_pitches = []

    for t in t_samples:
        walk_q = gen.compute_walk(t)
        waddle_q = gen.compute_waddle(t)

        walk_sways.append(walk_q[2])  # left_hip_roll
        waddle_sways.append(waddle_q[2])
        waddle_head_pitches.append(waddle_q[1])  # head_pitch

    # Waddling hip sway must be significantly larger than nominal walk
    assert max(waddle_sways) - min(waddle_sways) > max(walk_sways) - min(walk_sways)
    # Head pitch should nod actively
    assert max(waddle_head_pitches) - min(waddle_head_pitches) > 0.03


def test_spot_turn_directionality():
    """Verify that spot turn left and right generate differential pitch and gaze directions."""
    gen = BipedalGaitGenerator(default_freq=1.0)
    t = 0.25

    turn_left = gen.compute_spot_turn(t, direction=1)
    turn_right = gen.compute_spot_turn(t, direction=-1)

    # Looking left (positive yaw) vs looking right (negative yaw)
    assert turn_left[0] > 0.15, "Head should look left during spot turn left"
    assert turn_right[0] < -0.15, "Head should look right during spot turn right"

    # Opposing hip pitch directions
    assert np.sign(turn_left[3] - DEFAULT_NOMINAL_QPOS[3]) != np.sign(turn_right[3] - DEFAULT_NOMINAL_QPOS[3])


def test_head_dance_leaves_legs_stable():
    """Verify head dance keeps legs at nominal standing pose while moving head."""
    gen = BipedalGaitGenerator(default_freq=1.0)
    for t in [0.0, 0.2, 0.5, 0.8]:
        q = gen.compute_head_dance(t)
        # Legs (indices 2..7) must match DEFAULT_NOMINAL_QPOS
        np.testing.assert_allclose(q[2:], DEFAULT_NOMINAL_QPOS[2:], atol=1e-5)


def test_advanced_behavior_controller_fall_recovery_transition():
    """Verify that AdvancedBehaviorController catches a fall, triggers recovery, and returns to gait."""
    ctrl = AdvancedBehaviorController(default_pattern=GaitPattern.WADDLE, enable_fall_recovery=True)
    assert not ctrl.is_recovering

    # 1. Normal step while upright
    curr_qpos = DEFAULT_NOMINAL_QPOS.copy()
    upright_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    targets, recovering = ctrl.step(0.02, curr_qpos, 0.194, upright_grav)
    assert not recovering
    assert not ctrl.is_recovering

    # 2. Robot falls prone on floor (z = 0.05m)
    fallen_grav = np.array([0.0, 0.0, -0.95], dtype=np.float32)
    targets, recovering = ctrl.step(0.02, curr_qpos, 0.050, fallen_grav)
    assert recovering, "Controller should intercept fall and activate recovery"
    assert ctrl.is_recovering

    # 3. Step through recovery until standing is restored
    step_count = 0
    while ctrl.is_recovering and step_count < 100:
        targets, recovering = ctrl.step(0.02, targets, 0.050, fallen_grav)
        step_count += 1

    # Simulate reaching upright standing in Phase 3
    targets, recovering = ctrl.step(0.02, targets, 0.194, upright_grav)
    assert not recovering
    assert not ctrl.is_recovering


def test_bipedal_gait_generator_quadrature_forward_walk_and_waddle():
    """Verify compute_walk and compute_waddle advance forward >0.015m and survive 250 steps."""
    from training.run_parallel_sim import ParallelSimRunner
    from training.env_config import unscale_actions

    gen = BipedalGaitGenerator(default_freq=1.25)
    runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=1,
        decimation=10,
        stage=0,
    )
    dt = float(runner.control_dt)

    try:
        # 1. Walk gait
        runner.reset()
        steps_walk = 0
        for s in range(250):
            q = gen.compute_walk(s * dt, freq=1.25)
            act = unscale_actions(q, runner.cfg, nominal_centered=True)
            _, _, dones, _ = runner.step(act[None, :], fast=True)
            steps_walk += 1
            if dones[0]:
                break
        final_x_walk = float(runner.datas[0].qpos[0])
        assert steps_walk == 250, f"Walk fell early at step {steps_walk}/250"
        assert final_x_walk > 0.015, f"Walk final_x too low: {final_x_walk:.4f}m <= 0.015m"

        # 2. Waddle gait
        runner.reset()
        steps_waddle = 0
        for s in range(250):
            q = gen.compute_waddle(s * dt, freq=1.25)
            act = unscale_actions(q, runner.cfg, nominal_centered=True)
            _, _, dones, _ = runner.step(act[None, :], fast=True)
            steps_waddle += 1
            if dones[0]:
                break
        final_x_waddle = float(runner.datas[0].qpos[0])
        assert steps_waddle == 250, f"Waddle fell early at step {steps_waddle}/250"
        assert final_x_waddle > 0.015, f"Waddle final_x too low: {final_x_waddle:.4f}m <= 0.015m"
    finally:
        runner.close()
