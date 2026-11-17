import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import (
    MicroDuck8DoFConfig,
    scale_actions,
    unscale_actions,
    _DEFAULT_CFG,
)


def test_pd_calibration_values():
    cfg = MicroDuck8DoFConfig()
    assert cfg.KP == 13.5
    assert cfg.KD == 0.9


def test_residual_action_scales_spec():
    cfg = MicroDuck8DoFConfig()
    expected_scales = {
        "head_yaw": 0.15,
        "head_pitch": 0.15,
        "left_hip_roll": 0.25,
        "left_hip_pitch": 0.45,
        "left_knee_pitch": 0.50,
        "right_hip_roll": 0.25,
        "right_hip_pitch": 0.45,
        "right_knee_pitch": 0.50,
    }
    assert hasattr(cfg, "RESIDUAL_ACTION_SCALE")
    assert cfg.RESIDUAL_ACTION_SCALE == expected_scales
    assert hasattr(cfg, "action_scales")
    assert cfg.action_scales.shape == (8,)
    for i, name in enumerate(cfg.JOINT_NAMES):
        assert np.isclose(cfg.action_scales[i], expected_scales[name])


def test_symmetric_residual_action_scale_derivatives_and_bounds():
    cfg = MicroDuck8DoFConfig()
    eps = 1e-4

    # 1. Action 0.0 strictly maps to nominal_qpos
    zero_act = np.zeros(8, dtype=np.float32)
    scaled_zero = scale_actions(zero_act, cfg, nominal_centered=True)
    np.testing.assert_allclose(scaled_zero, cfg.nominal_qpos, atol=1e-6)

    # 2. Symmetric derivative across 0.0: d(scaled)/d(act) == cfg.action_scales on both sides
    pos_step = scale_actions(np.full(8, eps, dtype=np.float32), cfg, nominal_centered=True)
    neg_step = scale_actions(np.full(8, -eps, dtype=np.float32), cfg, nominal_centered=True)

    d_pos = (pos_step - cfg.nominal_qpos) / eps
    d_neg = (cfg.nominal_qpos - neg_step) / eps
    np.testing.assert_allclose(d_pos, cfg.action_scales, rtol=1e-3)
    np.testing.assert_allclose(d_neg, cfg.action_scales, rtol=1e-3)
    np.testing.assert_allclose(d_pos, d_neg, atol=1e-4)

    # 3. Head clamp check at +/- 1.0 action: head_pitch and head_yaw in [-0.15, 0.15]
    pos_max = scale_actions(np.ones(8, dtype=np.float32), cfg, nominal_centered=True)
    neg_max = scale_actions(-np.ones(8, dtype=np.float32), cfg, nominal_centered=True)
    assert np.isclose(pos_max[0], 0.15)
    assert np.isclose(pos_max[1], 0.15)
    assert np.isclose(neg_max[0], -0.15)
    assert np.isclose(neg_max[1], -0.15)

    # 4. Unclamped round-trip consistency
    # Compute safe range for each joint that doesn't clip against low_limits or high_limits
    safe_low = np.maximum(-0.95, (cfg.low_limits - cfg.nominal_qpos) / cfg.action_scales + 1e-3)
    safe_high = np.minimum(0.95, (cfg.high_limits - cfg.nominal_qpos) / cfg.action_scales - 1e-3)

    rng = np.random.default_rng(123)
    safe_acts = rng.uniform(safe_low, safe_high, size=(20, 8)).astype(np.float32)
    scaled_targets = scale_actions(safe_acts, cfg, nominal_centered=True)
    recovered_acts = unscale_actions(scaled_targets, cfg, nominal_centered=True)
    np.testing.assert_allclose(recovered_acts, safe_acts, atol=1e-5)


def test_pipeline_subconfigs_nominal_knee_pitch_flat_feet():
    from pipelines.dual_esp32s3_stack.training.env_config import (
        MicroDuck8DoFConfig as DualESPConfig,
        DEFAULT_NOMINAL_JOINT_POS as DUAL_NOM_DICT,
        DEFAULT_NOMINAL_QPOS as DUAL_NOM_QPOS,
    )
    from pipelines.rpi5_stack.training.env_config import (
        MicroDuck8DoFConfig as RPi5Config,
        DEFAULT_NOMINAL_JOINT_POS as RPI5_NOM_DICT,
        DEFAULT_NOMINAL_QPOS as RPI5_NOM_QPOS,
    )
    from pipelines.pizero2w_stack.training.env_config import (
        MicroDuck8DoFConfig as PiZeroConfig,
        DEFAULT_NOMINAL_JOINT_POS as PIZERO_NOM_DICT,
        DEFAULT_NOMINAL_QPOS as PIZERO_NOM_QPOS,
    )

    for cfg_cls, nom_dict, nom_qpos in [
        (DualESPConfig, DUAL_NOM_DICT, DUAL_NOM_QPOS),
        (RPi5Config, RPI5_NOM_DICT, RPI5_NOM_QPOS),
        (PiZeroConfig, PIZERO_NOM_DICT, PIZERO_NOM_QPOS),
    ]:
        cfg = cfg_cls()
        # Flat feet constraint: hip_pitch (-0.15) + knee_pitch (0.15) == 0.0
        assert cfg.NOMINAL_JOINT_POS["left_knee_pitch"] == 0.15
        assert cfg.NOMINAL_JOINT_POS["right_knee_pitch"] == 0.15
        assert nom_dict["left_knee_pitch"] == 0.15
        assert nom_dict["right_knee_pitch"] == 0.15
        # Index 4 (left_knee_pitch) and 7 (right_knee_pitch) in nominal_qpos
        assert np.isclose(nom_qpos[4], 0.15)
        assert np.isclose(nom_qpos[7], 0.15)
        assert np.isclose(cfg.nominal_qpos[4], 0.15)
        assert np.isclose(cfg.nominal_qpos[7], 0.15)


def test_parallel_sim_gyro_invariance_under_tilt(monkeypatch):
    """Verify that sensordata[3:6] (<gyro>) is fed directly to compute_rewards without double rotation."""
    from training.run_parallel_sim import ParallelSimRunner
    import training.run_parallel_sim as rps_mod

    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=1,
        decimation=1,
    )
    try:
        runner.reset()

        # Tilt robot pitch by 45 degrees: qpos[3:7] = [cos(pi/8), 0, sin(pi/8), 0]
        pitch_angle = np.pi / 4.0
        half_pitch = pitch_angle / 2.0
        tilted_quat = np.array([np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0], dtype=np.float32)
        runner.datas[0].qpos[3:7] = tilted_quat

        # Known body-frame gyro reading: pure z-axis angular velocity = 1.5 rad/s
        runner.datas[0].qvel[5] = 1.5
        known_gyro = np.array([0.0, 0.0, 1.5], dtype=np.float32)

        captured_base_ang_vel = []

        orig_compute_rewards = rps_mod.compute_rewards

        def mock_compute_rewards(**kwargs):
            captured_base_ang_vel.append(kwargs["base_ang_vel"].copy())
            return orig_compute_rewards(**kwargs)

        monkeypatch.setattr(rps_mod, "compute_rewards", mock_compute_rewards)

        # Step runner
        actions = np.zeros((1, 8), dtype=np.float32)
        runner.step(actions)

        assert len(captured_base_ang_vel) == 1
        fed_ang_vel = captured_base_ang_vel[0]

        # The angular velocity passed to compute_rewards must be the true body-frame gyro reading
        # NOT rotated by quat_rotate_inverse(tilted_quat, known_gyro)
        np.testing.assert_allclose(
            fed_ang_vel[0],
            known_gyro,
            atol=1e-5,
            err_msg="Gyro data was double-rotated instead of passed directly in body frame!",
        )
    finally:
        runner.close()
