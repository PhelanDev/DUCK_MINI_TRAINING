"""Tests for Episode Length Collapse, Suicide Lunge Fix, and Double-Support Transfer Window.

Validates Task 1 of plan 261030-2005-fix-eplen-collapse-and-valloss:
1. 16-step suicide lunge exploit is net negative (sum(rewards) < 0.0).
2. Leg pose penalty (leg_pose_penalty < 0.0 when deviating from nominal).
3. Gait sync reward (gait_sync > 0.0 when alternating stance matches clock_sin with vel_x > 0.03).
4. Double support transfer window accumulates at half-rate (0.5 * control_dt) when walking forward (vel_x >= 0.05).
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import mujoco

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    LocomotionMode,
    compute_rewards,
    DEFAULT_NOMINAL_QPOS,
)
from training.run_parallel_sim import ParallelSimRunner


def test_suicide_lunge_17_steps_is_net_negative():
    """Verify that lunging forward for 15 steps and falling on step 16 yields a strictly negative total episode return."""
    runner = ParallelSimRunner(num_envs=1)
    try:
        runner.reset()
        total_ep_reward = 0.0
        # Simulate 15 steps of forward progress
        for _ in range(15):
            _, rew, done, _ = runner.step(np.zeros((1, 8), dtype=np.float32))
            total_ep_reward += float(rew[0])
            assert not done[0], "Runner should not terminate prematurely during normal forward steps"

        # Force a severe fall on step 16 (tilt > 60 deg, grav_z > -0.50)
        theta_65 = np.deg2rad(65.0)
        runner.datas[0].qpos[3:7] = [np.cos(theta_65 / 2.0), 0.0, np.sin(theta_65 / 2.0), 0.0]
        mujoco.mj_forward(runner.model, runner.datas[0])
        _, rew_fall, done_fall, info_fall = runner.step(np.zeros((1, 8), dtype=np.float32))
        total_ep_reward += float(rew_fall[0])

        assert bool(info_fall[0]["fallen"]) is True, "Expected fallen=True on severe tilt"
        assert bool(done_fall[0]) is True, "Expected done=True on fall"
        assert total_ep_reward < 0.0, (
            f"16-step suicide lunge exploit must yield net negative episode reward, got {total_ep_reward:+.2f}"
        )
    finally:
        runner.close()


def test_leg_pose_penalty():
    """Verify that deviating from nominal leg pose incurs negative leg_pose_penalty in WALK mode, but is zero in SIT/LIE."""
    nominal_q = DEFAULT_NOMINAL_QPOS.copy()

    # 1. Nominal standing pose -> 0.0 penalty
    rew_nominal = compute_rewards(
        base_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=nominal_q,
        cmd_mode=LocomotionMode.WALK,
    )
    assert "leg_pose_penalty" in rew_nominal, "rew_nominal missing 'leg_pose_penalty'"
    assert pytest.approx(rew_nominal["leg_pose_penalty"], abs=1e-5) == 0.0

    # 2. Split leg pose (legs splayed forward and back) -> negative penalty
    split_q = nominal_q.copy()
    split_q[3] += 0.35  # left_hip_pitch
    split_q[6] -= 0.35  # right_hip_pitch
    rew_split = compute_rewards(
        base_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=split_q,
        cmd_mode=LocomotionMode.WALK,
    )
    assert rew_split["leg_pose_penalty"] < 0.0, f"Expected negative leg_pose_penalty, got {rew_split['leg_pose_penalty']}"
    assert rew_split["total_reward"] < rew_nominal["total_reward"]

    # 3. In SIT or LIE mode -> leg_pose_penalty is inactive (0.0)
    rew_sit = compute_rewards(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=split_q,
        cmd_mode=LocomotionMode.SIT,
    )
    assert pytest.approx(rew_sit["leg_pose_penalty"], abs=1e-5) == 0.0


def test_gait_sync_reward():
    """Verify that alternating foot contacts matching clock_sin yield positive gait_sync when vel_x > 0.03."""
    # When clock_sin < 0 (left stance, right swing): desired_L = 1.0, desired_R = 0.0
    fc_match = np.array([1.0, 0.0, -0.8, 0.6], dtype=np.float32)
    rew_match = compute_rewards(
        base_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=fc_match,
        cmd_mode=LocomotionMode.WALK,
    )
    assert "gait_sync" in rew_match, "rew_match missing 'gait_sync'"
    assert rew_match["gait_sync"] > 0.5, f"Expected positive gait_sync > 0.5, got {rew_match['gait_sync']}"

    # Opposite stance (wrong foot on ground) -> gait_sync is 0.0
    fc_opposite = np.array([0.0, 1.0, -0.8, 0.6], dtype=np.float32)
    rew_opposite = compute_rewards(
        base_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=fc_opposite,
        cmd_mode=LocomotionMode.WALK,
    )
    assert pytest.approx(rew_opposite["gait_sync"], abs=1e-5) == 0.0
    assert rew_opposite["gait_sync"] < rew_match["gait_sync"]

    # When vel_x <= 0.03 (standing still or near-zero motion), gait_sync is gated to 0.0
    rew_slow = compute_rewards(
        base_lin_vel=np.array([0.02, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=fc_match,
        cmd_mode=LocomotionMode.WALK,
    )
    assert pytest.approx(rew_slow["gait_sync"], abs=1e-5) == 0.0


def test_double_support_transfer_during_walking():
    """Verify that walking forward at vel_x >= 0.05 accumulates double_support_time at half-rate (0.5 * control_dt),
    while standing still (vel_x = 0) accumulates at full control_dt."""
    runner = ParallelSimRunner(num_envs=2)
    try:
        runner.reset()
        control_dt = runner.control_dt

        # Env 0: Standing still (both feet on ground, zero velocity)
        # Env 1: Walking forward (both feet on ground, vel_x = 0.15 m/s)
        runner.datas[0].qvel[:] = 0.0
        runner.datas[1].qvel[0] = 0.15
        for d in runner.datas:
            mujoco.mj_forward(runner.model, d)

        runner.step(np.zeros((2, 8), dtype=np.float32))

        # Env 0 (stationary) accumulated at full control_dt
        assert pytest.approx(runner.double_support_time[0], abs=1e-5) == control_dt, (
            f"Stationary double_support_time should accumulate at full rate {control_dt}, got {runner.double_support_time[0]}"
        )
        # Env 1 (moving forward) accumulated at half-rate 0.5 * control_dt
        assert pytest.approx(runner.double_support_time[1], abs=1e-5) == 0.5 * control_dt, (
            f"Moving forward double_support_time should accumulate at half rate {0.5 * control_dt}, got {runner.double_support_time[1]}"
        )
    finally:
        runner.close()


def test_transient_one_step_dynamic_tilt_does_not_kill_walking_episode():
    """Verify that during walking (episode_steps > 1), a 1-step transient 47-deg tilt (grav_z ~ -0.682 in (-0.707, -0.50])
    does NOT trigger done[0], but 2 consecutive steps of 47-deg tilt DOES trigger fallen=True and done=True."""
    runner = ParallelSimRunner(num_envs=1, terrain_mode="flat")
    try:
        runner.reset()
        # Step 1: normal upright step so episode_steps becomes 1
        runner.step(np.zeros((1, 8), dtype=np.float32))
        assert runner.episode_steps[0] == 1

        # Step 2: transient 47-deg tilt (grav_z ~ -0.682 in (-0.707, -0.50))
        theta_47 = np.deg2rad(47.0)
        runner.datas[0].qpos[2] = 0.18
        runner.datas[0].qpos[3:7] = [np.cos(theta_47 / 2.0), 0.0, np.sin(theta_47 / 2.0), 0.0]
        mujoco.mj_forward(runner.model, runner.datas[0])
        _, _, dones, infos = runner.step(np.zeros((1, 8), dtype=np.float32))
        assert bool(infos[0]["fallen"]) is False, "1-step transient 47-deg tilt during walking should not trigger fallen"
        assert bool(dones[0]) is False, "1-step transient 47-deg tilt during walking should not trigger done"

        # Step 3: sustained 47-deg tilt on second consecutive step -> MUST terminate!
        runner.datas[0].qpos[2] = 0.18
        runner.datas[0].qpos[3:7] = [np.cos(theta_47 / 2.0), 0.0, np.sin(theta_47 / 2.0), 0.0]
        mujoco.mj_forward(runner.model, runner.datas[0])
        _, _, dones2, infos2 = runner.step(np.zeros((1, 8), dtype=np.float32))
        assert bool(infos2[0]["fallen"]) is True, "2 consecutive steps of 47-deg tilt must trigger fallen"
        assert bool(dones2[0]) is True, "2 consecutive steps of 47-deg tilt must trigger done"
    finally:
        runner.close()


def test_dynamic_knee_flexion_height_threshold():
    """Verify that when walk_min_z=0.065 is configured, knee flexion at root_z = 0.072m
    is NOT marked fallen. But under default walk_min_z=0.080, root_z = 0.072m IS marked fallen."""
    # 1. Configured with lower walk_min_z = 0.065
    runner_flex = ParallelSimRunner(num_envs=1, terrain_mode="flat", decimation=1, walk_min_z=0.065)
    try:
        runner_flex.reset()
        runner_flex.cmd_modes[:] = int(LocomotionMode.WALK)
        runner_flex.datas[0].qpos[0:3] = [0.0, 0.0, 0.072]
        runner_flex.datas[0].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        runner_flex.datas[0].qvel[:] = 0.0
        mujoco.mj_forward(runner_flex.model, runner_flex.datas[0])
        _, _, dones_flex, infos_flex = runner_flex.step(np.zeros((1, 8), dtype=np.float32))
        assert bool(infos_flex[0]["fallen"]) is False, "Knee flexion at z=0.072m with walk_min_z=0.065 should not fall"
        assert bool(dones_flex[0]) is False
    finally:
        runner_flex.close()

    # 2. Default walk_min_z = 0.080 -> z=0.072m must fall
    runner_def = ParallelSimRunner(num_envs=1, terrain_mode="flat", decimation=1)
    try:
        runner_def.reset()
        runner_def.cmd_modes[:] = int(LocomotionMode.WALK)
        runner_def.datas[0].qpos[0:3] = [0.0, 0.0, 0.072]
        runner_def.datas[0].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        runner_def.datas[0].qvel[:] = 0.0
        mujoco.mj_forward(runner_def.model, runner_def.datas[0])
        _, _, dones_def, infos_def = runner_def.step(np.zeros((1, 8), dtype=np.float32))
        assert bool(infos_def[0]["fallen"]) is True, "z=0.072m under default walk_min_z=0.080 must fall"
        assert bool(dones_def[0]) is True
    finally:
        runner_def.close()


def test_stack_runners_success_rate_and_distance_methods():
    """Verify ParallelSimRunner in both pipelines/rpi5_stack and pipelines/dual_esp32s3_stack
    have completed_distances_x, get_mean_distance_x(), and get_success_rate(), and use
    calibrated grav_z > -0.707 instead of -0.3."""
    from pipelines.rpi5_stack.training.run_parallel_sim import ParallelSimRunner as RPi5Runner
    from pipelines.dual_esp32s3_stack.training.run_parallel_sim import ParallelSimRunner as ESP32Runner

    for RunnerCls, name in [(RPi5Runner, "RPi5"), (ESP32Runner, "ESP32-S3")]:
        runner = RunnerCls(num_envs=1)
        try:
            assert hasattr(runner, "completed_distances_x"), f"{name} runner missing completed_distances_x buffer"
            assert callable(getattr(runner, "get_mean_distance_x", None)), f"{name} runner missing get_mean_distance_x"
            assert callable(getattr(runner, "get_success_rate", None)), f"{name} runner missing get_success_rate"

            runner.reset()
            # Test calibrated tilt threshold: 50 deg tilt (-0.64 > -0.707) should cause fall on stack runners
            theta_50 = np.deg2rad(50.0)
            runner.datas[0].qpos[2] = 0.20
            runner.datas[0].qpos[3:7] = [np.cos(theta_50 / 2.0), 0.0, np.sin(theta_50 / 2.0), 0.0]
            mujoco.mj_forward(runner.model, runner.datas[0])
            _, _, dones, infos = runner.step(np.zeros((1, 8), dtype=np.float32))

            # With calibrated grav_z > -0.707, 50 deg tilt triggers fall
            assert bool(infos[0]["fallen"]) is True, f"{name} runner should fall on 50-deg tilt (grav_z ~ -0.64 > -0.707)"
            assert bool(dones[0]) is True

            # Verify distance tracking
            assert len(runner.completed_distances_x) >= 1
            mean_dist = runner.get_mean_distance_x()
            succ_rate = runner.get_success_rate(min_dist=1.5)
            assert isinstance(mean_dist, float)
            assert isinstance(succ_rate, float)
        finally:
            runner.close()


def test_initial_policy_rollout_eplen_and_valloss_stability():
    """Verify that a freshly initialized ActorCritic does not die in 15 steps from wild noise
    and keeps initial action std <= 0.40."""
    import torch
    from scripts.train_ppo import ActorCritic, DEFAULT_MODEL_PATH

    model = ActorCritic(obs_dim=42, action_dim=8)
    std = torch.exp(model.log_std).detach().numpy()
    assert np.all(std <= 0.40), f"Initial action std should be <= 0.40 to prevent joint slamming, got {std}"

    runner = ParallelSimRunner(model_path=DEFAULT_MODEL_PATH, num_envs=8)
    try:
        obs = runner.reset()
        first_done_step = np.full(8, 100, dtype=np.int32)
        for s in range(1, 101):
            with torch.no_grad():
                act, _, _, _ = model.get_action_and_value(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, dones, _ = runner.step(np.clip(act.numpy(), -1.0, 1.0))
            for i in range(8):
                if dones[i] and first_done_step[i] == 100:
                    first_done_step[i] = s
        mean_initial_survival = float(np.mean(first_done_step))
        assert mean_initial_survival >= 20.0, (
            f"Fresh policy with calibrated init_noise_std must survive >= 20 steps under full noise (was ~14), got {mean_initial_survival:.1f}"
        )
    finally:
        runner.close()


def test_ppo_iteration_1_valloss_not_exploding():
    """Verify that 1 iteration of train_ppo produces value_loss < 50.0 (preventing ValLoss = 3390 spike)."""
    from scripts.train_ppo import train_ppo
    res = train_ppo(
        num_envs=4,
        rollout_steps=64,
        max_iterations=1,
        verbose=False,
        dry_run=False,
    )
    v_loss = res["metrics"]["value_loss"]
    assert v_loss < 50.0, f"Iteration 1 value_loss should be < 50.0, got {v_loss}"


def test_interactive_viewer_observation_phase_clock_parity():
    """Verify extract_viewer_observation embeds [clock_sin, clock_cos] at obs[35:37],
    matching ParallelSimRunner._extract_obs()."""
    from scripts.interactive_viewer import extract_viewer_observation
    from training.run_parallel_sim import ParallelSimRunner
    from training.env_config import compute_gait_phase

    runner = ParallelSimRunner(num_envs=1, terrain_mode="flat")
    try:
        runner.reset()
        runner.episode_steps[0] = 10
        obs_runner = runner._extract_obs(0)

        # Expected phase clock at step 10
        _, expected_sin, expected_cos = compute_gait_phase(10, dt_control=runner.control_dt, period=0.8)

        obs_viewer = extract_viewer_observation(
            runner.model,
            runner.datas[0],
            cmd_vel=runner.commands[0],
            last_actions=runner.last_actions[0],
            obs_dim=runner.obs_dim,
            sim_step=10,
            control_dt=runner.control_dt,
        )

        assert pytest.approx(obs_viewer[35], abs=1e-4) == expected_sin
        assert pytest.approx(obs_viewer[36], abs=1e-4) == expected_cos
        assert pytest.approx(obs_runner[35], abs=1e-4) == expected_sin
        assert pytest.approx(obs_runner[36], abs=1e-4) == expected_cos
    finally:
        runner.close()


