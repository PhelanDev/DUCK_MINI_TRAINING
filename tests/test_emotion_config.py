"""Tests for 10 Emotion States & Observation Space Extension (Task 1).

Covers:
- EmotionState enum (10 emotions, 0..9)
- NUM_EMOTIONS and EMOTION_TARGET_HEAD_POS
- get_emotion_one_hot vector serialization
- compute_observations extending to 47 and 52 dimensions
- compute_rewards emotion tracking reward
- ParallelSimRunner with include_emotions=True returning 52-dim observations
"""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import (
    MicroDuck8DoFConfig,
    EmotionState,
    NUM_EMOTIONS,
    EMOTION_TARGET_HEAD_POS,
    get_emotion_one_hot,
    compute_observations,
    compute_rewards,
)
from training.run_parallel_sim import ParallelSimRunner


def test_emotion_enum_and_constants():
    """Verify EmotionState enum, NUM_EMOTIONS, and target head angles."""
    assert NUM_EMOTIONS == 10
    assert len(EmotionState) == 10

    expected_mapping = {
        "NEUTRAL": 0,
        "HAPPY": 1,
        "SAD": 2,
        "ANGRY": 3,
        "SCARED": 4,
        "CURIOUS": 5,
        "SURPRISED": 6,
        "SLEEPY": 7,
        "PLAYFUL": 8,
        "PROUD": 9,
    }

    for name, value in expected_mapping.items():
        assert hasattr(EmotionState, name)
        state = getattr(EmotionState, name)
        assert state.value == value
        assert state in EMOTION_TARGET_HEAD_POS or value in EMOTION_TARGET_HEAD_POS

    # Verify head limits and specific target values
    cfg = MicroDuck8DoFConfig()
    yaw_min, yaw_max = cfg.JOINT_LIMITS["head_yaw"]
    pitch_min, pitch_max = cfg.JOINT_LIMITS["head_pitch"]

    expected_targets = {
        EmotionState.NEUTRAL: (0.0, 0.0),
        EmotionState.HAPPY: (0.0, 0.25),
        EmotionState.SAD: (0.0, -0.30),
        EmotionState.ANGRY: (0.0, -0.15),
        EmotionState.SCARED: (0.0, 0.45),
        EmotionState.CURIOUS: (0.40, 0.10),
        EmotionState.SURPRISED: (0.0, 0.60),
        EmotionState.SLEEPY: (0.0, -0.40),
        EmotionState.PLAYFUL: (-0.35, 0.20),
        EmotionState.PROUD: (0.0, 0.35),
    }

    for state, (expected_yaw, expected_pitch) in expected_targets.items():
        yaw, pitch = EMOTION_TARGET_HEAD_POS[state]
        assert np.isclose(yaw, expected_yaw)
        assert np.isclose(pitch, expected_pitch)
        assert yaw_min <= yaw <= yaw_max
        assert pitch_min <= pitch <= pitch_max


def test_get_emotion_one_hot():
    """Verify get_emotion_one_hot generates accurate 10-dim one-hot vectors."""
    for i in range(10):
        # By int
        vec_int = get_emotion_one_hot(i)
        assert vec_int.shape == (10,)
        assert vec_int.dtype == np.float32
        assert vec_int[i] == 1.0
        assert np.sum(vec_int) == 1.0

        # By enum
        state = EmotionState(i)
        vec_enum = get_emotion_one_hot(state)
        assert np.array_equal(vec_int, vec_enum)

    # Invalid emotion indices raise error
    with pytest.raises((ValueError, IndexError)):
        get_emotion_one_hot(-1)

    with pytest.raises((ValueError, IndexError)):
        get_emotion_one_hot(10)


def test_compute_observations_with_emotions():
    """Verify observation packing with emotion one-hot vector (47 and 52 dims)."""
    base_ang_vel = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    commands = np.array([0.5, 0.0, 0.1], dtype=np.float32)
    joint_pos = np.arange(8, dtype=np.float32) * 0.1
    joint_vel = np.arange(8, dtype=np.float32) * -0.05
    last_actions = np.ones(8, dtype=np.float32) * 0.2
    foot_contact = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    lidar_ranges = np.array([1.0, 1.2, 0.8, 1.5, 2.0], dtype=np.float32)
    emotion_one_hot = get_emotion_one_hot(EmotionState.HAPPY)

    # 1. Without lidar, without emotion -> 37 dims (backward compatibility)
    obs_37 = compute_observations(
        base_ang_vel=base_ang_vel,
        proj_grav=proj_grav,
        commands=commands,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        last_actions=last_actions,
        foot_contact=foot_contact,
    )
    assert obs_37.shape == (37,)

    # 2. With lidar, without emotion -> 42 dims
    obs_42 = compute_observations(
        base_ang_vel=base_ang_vel,
        proj_grav=proj_grav,
        commands=commands,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        last_actions=last_actions,
        foot_contact=foot_contact,
        lidar_ranges=lidar_ranges,
    )
    assert obs_42.shape == (42,)

    # 3. Without lidar, with emotion -> 47 dims
    obs_47 = compute_observations(
        base_ang_vel=base_ang_vel,
        proj_grav=proj_grav,
        commands=commands,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        last_actions=last_actions,
        foot_contact=foot_contact,
        emotion_one_hot=emotion_one_hot,
    )
    assert obs_47.shape == (47,)
    np.testing.assert_allclose(obs_47[:37], obs_37)
    np.testing.assert_allclose(obs_47[37:47], emotion_one_hot)

    # 4. With lidar, with emotion -> 52 dims
    obs_52 = compute_observations(
        base_ang_vel=base_ang_vel,
        proj_grav=proj_grav,
        commands=commands,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        last_actions=last_actions,
        foot_contact=foot_contact,
        lidar_ranges=lidar_ranges,
        emotion_one_hot=emotion_one_hot,
    )
    assert obs_52.shape == (52,)
    np.testing.assert_allclose(obs_52[:42], obs_42)
    np.testing.assert_allclose(obs_52[42:52], emotion_one_hot)

    # Defensive check: incorrect emotion vector size
    with pytest.raises(AssertionError):
        compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=commands,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_actions,
            foot_contact=foot_contact,
            emotion_one_hot=np.zeros(5, dtype=np.float32),
        )


def test_compute_rewards_emotion_tracking():
    """Verify emotion head posture tracking reward calculation."""
    base_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    cmd_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = 0.0
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Target for HAPPY is (0.0, 0.25)
    target_happy = np.array([0.0, 0.25], dtype=np.float32)

    # 1. Exact match with qpos_head
    r_perfect = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        current_emotion=EmotionState.HAPPY,
        qpos_head=target_happy,
    )
    assert "emotion_tracking" in r_perfect
    assert np.isclose(r_perfect["emotion_tracking"], 1.0)

    # 2. Match with joint_pos[:2]
    joint_pos_happy = np.zeros(8, dtype=np.float32)
    joint_pos_happy[:2] = target_happy
    r_joint_pos = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        current_emotion=EmotionState.HAPPY,
        joint_pos=joint_pos_happy,
    )
    assert np.isclose(r_joint_pos["emotion_tracking"], 1.0)

    # 3. Deviated head posture: ||q - q_target||^2 = 0.15 -> r_emotion = exp(-1.0)
    deviated_head = np.array([0.0, 0.25 + np.sqrt(0.15)], dtype=np.float32)
    r_deviated = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        current_emotion=EmotionState.HAPPY,
        qpos_head=deviated_head,
    )
    assert np.isclose(r_deviated["emotion_tracking"], np.exp(-1.0), atol=1e-5)
    # Total reward should be higher for perfect match than deviated match
    assert r_perfect["total_reward"] > r_deviated["total_reward"]

    # 4. Without current_emotion -> emotion_tracking is 0.0
    r_none = compute_rewards(
        base_lin_vel=base_lin_vel,
        cmd_lin_vel=cmd_lin_vel,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        qpos_head=target_happy,
    )
    assert np.isclose(r_none["emotion_tracking"], 0.0)

    # 5. Batched compute_rewards with emotions
    num_envs = 3
    base_lin_batch = np.tile(base_lin_vel, (num_envs, 1))
    cmd_lin_batch = np.tile(cmd_lin_vel, (num_envs, 1))
    base_ang_batch = np.tile(base_ang_vel, (num_envs, 1))
    cmd_ang_batch = np.zeros(num_envs, dtype=np.float32)
    proj_grav_batch = np.tile(proj_gravity, (num_envs, 1))
    torques_batch = np.tile(joint_torques, (num_envs, 1))
    vel_batch = np.tile(joint_vel, (num_envs, 1))
    contact_batch = np.tile(foot_contact, (num_envs, 1))
    emotions_batch = np.array([EmotionState.NEUTRAL, EmotionState.HAPPY, EmotionState.SAD], dtype=np.int32)
    qpos_head_batch = np.array(
        [
            EMOTION_TARGET_HEAD_POS[EmotionState.NEUTRAL],
            EMOTION_TARGET_HEAD_POS[EmotionState.HAPPY],
            [0.0, 0.0],  # Deviated from SAD (0.0, -0.30)
        ],
        dtype=np.float32,
    )

    r_batch = compute_rewards(
        base_lin_vel=base_lin_batch,
        cmd_lin_vel=cmd_lin_batch,
        base_ang_vel=base_ang_batch,
        cmd_ang_vel=cmd_ang_batch,
        proj_gravity=proj_grav_batch,
        joint_torques=torques_batch,
        joint_vel=vel_batch,
        foot_contact=contact_batch,
        current_emotion=emotions_batch,
        qpos_head=qpos_head_batch,
    )
    assert np.isclose(r_batch["emotion_tracking"][0], 1.0)
    assert np.isclose(r_batch["emotion_tracking"][1], 1.0)
    assert r_batch["emotion_tracking"][2] < 1.0


def test_parallel_sim_runner_include_emotions():
    """Verify ParallelSimRunner with include_emotions=True produces 52-dim observations."""
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=4,
        decimation=5,
        include_emotions=True,
    )
    try:
        assert runner.include_emotions is True
        assert runner.obs_dim == 52
        assert runner.current_emotions.shape == (4,)
        assert runner.current_emotions.dtype == np.int32

        # Reset observations shape
        obs = runner.reset()
        assert obs.shape == (4, 52)

        # Step observations shape and info
        actions = np.zeros((4, runner.action_dim), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)
        assert next_obs.shape == (4, 52)
        assert len(infos) == 4
        assert "emotion_tracking" in infos[0]

        # Verify default runner remains backward-compatible (obs_dim in (37, 42), include_emotions False)
        runner_default = ParallelSimRunner(
            model_path="models/mjcf/microduck_8dof.xml",
            num_envs=2,
            decimation=5,
        )
        assert getattr(runner_default, "include_emotions", False) is False
        assert runner_default.obs_dim in (37, 42)
        obs_default = runner_default.reset()
        assert obs_default.shape == (2, runner_default.obs_dim)
        # Also test with LiDAR perception model (RPi5 model)
        rpi5_model_path = Path("pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml")
        if rpi5_model_path.exists():
            runner_rpi5 = ParallelSimRunner(
                model_path=rpi5_model_path,
                num_envs=2,
                decimation=5,
                include_emotions=True,
            )
            assert runner_rpi5.obs_dim == 52
            obs_rpi5 = runner_rpi5.reset()
            assert obs_rpi5.shape == (2, 52)
            if hasattr(runner_rpi5, "executor") and runner_rpi5.executor:
                runner_rpi5.executor.shutdown(wait=False)
    finally:
        if hasattr(runner, "executor") and runner.executor:
            runner.executor.shutdown(wait=False)
        if "runner_default" in locals() and hasattr(runner_default, "executor") and runner_default.executor:
            runner_default.executor.shutdown(wait=False)
