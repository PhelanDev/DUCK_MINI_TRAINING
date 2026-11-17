"""Behavioral coverage for locomotion progress and swing reward gates."""

import mujoco
import numpy as np
import pytest

from training.env_config import DEFAULT_NOMINAL_QPOS, compute_rewards
from training.run_parallel_sim import ParallelSimRunner


def _reward_inputs(**overrides):
    inputs = dict(
        base_lin_vel=np.array([0.15, 0.0, 0.0]),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0]),
        base_ang_vel=np.zeros(3),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0]),
        joint_torques=np.zeros(8),
        joint_vel=np.zeros(8),
        foot_contact=np.array([0.0, 1.0, 0.0, 0.0]),
        clock_sin=1.0,
        delta_root_x=0.003,
        standstill_time=0.0,
    )
    inputs.update(overrides)
    return inputs


@pytest.mark.parametrize("batch", [False, True])
def test_swing_rewards_require_the_matching_airborne_foot(batch):
    def reward(contact, swing):
        args = _reward_inputs(foot_contact=np.asarray(contact), swing_valid=np.asarray(swing))
        if batch:
            args = {k: np.stack([v, v]) if isinstance(v, np.ndarray) else v for k, v in args.items()}
        return compute_rewards(**args)

    valid = reward([0, 1, 0, 0], [True, False])
    assert np.all(np.asarray(valid["foot_contact"]) > 0)
    assert np.all(np.asarray(valid["gait_sync"]) > 0)
    for contact, swing in (
        ([0, 1, 0, 0], [False, False]),
        ([0, 1, 0, 0], [False, True]),
        ([1, 1, 0, 0], [True, False]),
        ([0, 0, 0, 0], [True, False]),
    ):
        rejected = reward(contact, swing)
        np.testing.assert_array_equal(rejected["foot_contact"], 0)
        np.testing.assert_array_equal(rejected["gait_sync"], 0)


@pytest.mark.parametrize("batch", [False, True])
def test_leg_pose_relaxation_requires_real_forward_progress(batch):
    joint_pos = DEFAULT_NOMINAL_QPOS.copy()
    joint_pos[3] += 0.2

    def penalty(dx, body_vx):
        args = _reward_inputs(joint_pos=joint_pos, delta_root_x=dx,
                              base_lin_vel=np.array([body_vx, 0.0, 0.0]))
        if batch:
            args = {k: np.stack([v, v]) if isinstance(v, np.ndarray) else v for k, v in args.items()}
        return compute_rewards(**args)["leg_pose_penalty"]

    full_penalty = penalty(0.0, 0.15)
    assert np.all(np.asarray(full_penalty) < 0)
    np.testing.assert_allclose(penalty(0.003, 0.15), 0.25 * full_penalty)
    np.testing.assert_allclose(penalty(-0.003, 0.15), full_penalty)
    np.testing.assert_allclose(penalty(0.003, 0.05), full_penalty)


@pytest.mark.parametrize("decimation", [5, 10, 20])
def test_consistent_large_displacement_is_clipped_for_control_timestep(monkeypatch, decimation):
    runner = ParallelSimRunner("duck_v2/cyberduck_v2.xml", num_envs=1,
                               num_threads=1, decimation=decimation)
    try:
        d = runner.datas[0]
        d.qvel[0] = 2.0
        runner.commands[0] = [0.2, 0.0, 0.0]
        mujoco.mj_forward(runner.model, d)

        def advance(indices, targets):
            d.qpos[0] += 2.0 * runner.control_dt
            mujoco.mj_forward(runner.model, d)

        monkeypatch.setattr(runner, "_step_subrange", advance)
        _, _, _, info = runner.step_arrays(np.zeros((1, 8)))
        assert info["invalid_displacement_count"] == 0
        assert info["delta_root_x_mean"] == pytest.approx(runner.control_dt)
        assert info["reward_components"]["forward_progress"] == pytest.approx(40 * runner.control_dt)
    finally:
        runner.close()


def test_turning_acceleration_keeps_world_progress_and_body_forward_gates_separate(monkeypatch):
    runner = ParallelSimRunner("duck_v2/cyberduck_v2.xml", num_envs=1, num_threads=1)
    try:
        d = runner.datas[0]
        runner.commands[0] = [0.2, 0.0, 0.2]  # allow the scripted turn at stage 0

        def advance(indices, targets):
            d.qpos[0] += 0.1 * runner.control_dt
            d.qvel[0] = 0.2
            yaw = 2 * np.pi / 3
            d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(runner.model, d)

        monkeypatch.setattr(runner, "_step_subrange", advance)
        _, _, _, info = runner.step_arrays(np.zeros((1, 8)))
        # The displacement matches world-frame acceleration; the body faces
        # against the displacement, so this is not positive walking progress.
        assert info["invalid_displacement_count"] == 0
        assert info["delta_root_x_mean"] == pytest.approx(0.1 * runner.control_dt)
        assert info["reward_components"]["forward_progress"] == 0
    finally:
        runner.close()


@pytest.mark.parametrize("fast", [False, True])
def test_terminal_fall_penalty_is_in_weighted_reward_snapshot(monkeypatch, fast):
    runner = ParallelSimRunner("duck_v2/cyberduck_v2.xml", num_envs=1, num_threads=1)
    try:
        d = runner.datas[0]

        def fall(indices, targets):
            d.qpos[2] = 0.01
            mujoco.mj_forward(runner.model, d)

        monkeypatch.setattr(runner, "_step_subrange", fall)
        _, rewards, dones, info = runner.step(np.zeros((1, 8)), fast=fast)
        terms = info["reward_components"] if fast else info[0]["reward_components"]
        assert dones[0]
        assert terms["fall_penalty"] == runner.cfg.REWARD_WEIGHTS["fall_penalty"]
        assert sum(terms.values()) == pytest.approx(float(rewards[0]), abs=1e-5)
        assert runner.standstill_time[0] == 0  # reset must not erase the snapshot
        assert (info["standstill_time_mean"] if fast else info[0]["standstill_time"]) > 0
    finally:
        runner.close()
