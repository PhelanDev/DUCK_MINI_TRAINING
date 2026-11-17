import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_test_helpers import (
    EXPECTED_JOINTS,
    EXPECTED_LIDAR_RAYS,
    assert_mjcf_structure_and_joints,
    assert_mjcf_perception_suite,
    assert_mujoco_compilation_and_mass,
    assert_urdf_structure_and_joint_limits,
    assert_slalom_pillars_and_stairs,
    assert_sole_clearance,
)

MJCF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipelines", "pizero2w_stack", "models", "mjcf", "microduck_pizero2w_8dof.xml"
)
URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipelines", "pizero2w_stack", "models", "urdf", "microduck_pizero2w_8dof.urdf"
)


def test_pizero2w_mjcf_structure_and_joints():
    assert_mjcf_structure_and_joints(MJCF_PATH, "microduck_pizero2w_8dof")


def test_pizero2w_mjcf_perception_suite():
    assert_mjcf_perception_suite(MJCF_PATH)


def test_pizero2w_mujoco_compilation_and_mass_budget():
    assert_mujoco_compilation_and_mass(
        MJCF_PATH,
        min_mass=0.615,
        max_mass=0.685,
        base_mass_min=0.260,
        base_mass_max=0.290,
        n_steps=50,
    )


def test_pizero2w_urdf_structure_and_inertials():
    assert_urdf_structure_and_joint_limits(
        URDF_PATH,
        robot_name="microduck_pizero2w_8dof",
        min_total_mass=0.615,
        max_total_mass=0.685,
        min_base_mass=0.260,
        max_base_mass=0.290,
    )


def test_pizero2w_mjcf_obstacles():
    assert_slalom_pillars_and_stairs(MJCF_PATH)


def test_pizero2w_infer_script_import_and_mock():
    """Verify Pi Zero 2 W inference runtime import, angle-to-PWM mapping, and mock mode."""
    from pipelines.pizero2w_stack.deployment.infer_pizero2w import (
        PiZero2WInferenceEngine,
        angle_to_pwm,
        pack_pizero2w_observation,
    )

    assert angle_to_pwm(0.0) == 1500
    assert 500 <= angle_to_pwm(-2.0, min_rad=-1.57, max_rad=1.57) <= 600
    assert 2400 <= angle_to_pwm(2.0, min_rad=-1.57, max_rad=1.57) <= 2500
    assert angle_to_pwm(0.0, min_rad=0.0, max_rad=1.48) == 1500
    assert angle_to_pwm(-0.5, min_rad=0.0, max_rad=1.48) == 1500

    obs = pack_pizero2w_observation(
        base_ang_vel=[0.0, 0.0, 0.0],
        proj_grav=[0.0, 0.0, -1.0],
        commands=[0.2, 0.0, 0.0],
        joint_pos=[0.0] * 8,
        joint_vel=[0.0] * 8,
        last_actions=[0.0] * 8,
        foot_contact=[1.0, 1.0, 0.0, 0.0],
        lidar_ranges=[1.5] * 5,
    )
    assert obs.shape == (42,)

    engine = PiZero2WInferenceEngine(model_path=None, mock=True, freq=50.0)
    assert engine.mock is True
    assert engine.servo_controller is not None

    targets, pwms = engine.step(obs)
    assert len(targets) == 8
    assert len(pwms) == 8
    assert all(500 <= p <= 2500 for p in pwms)


def test_pizero2w_infer_cli(tmp_path):
    """Verify CLI flags --mock, --mock-pca, --freq, --rate work properly."""
    import subprocess

    script_path = REPO_ROOT / "pipelines" / "pizero2w_stack" / "deployment" / "infer_pizero2w.py"
    assert script_path.exists(), f"Missing script: {script_path}"
    import json
    from training.policy_contract import observation_schema
    from scripts.train_ppo import ActorCritic
    from scripts.export_policy import export_onnx
    policy = ActorCritic(hidden_dim=16)
    policy.observation_schema = observation_schema(42, False, 0.02)
    model_path = export_onnx(policy, tmp_path / "pizero_test.onnx")

    for mock_flag in ["--mock", "--mock-pca"]:
        cmd = [
            sys.executable,
            str(script_path),
            mock_flag,
            "--rate",
            "50",
            "--steps",
            "5",
            "--model", str(model_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI with {mock_flag} failed with error: {res.stderr}\nOutput: {res.stdout}"


def test_pizero2w_eyes_symmetric_and_all_mjcf_sole_clearance():
    """Verify right_eye size/pos symmetry and sole clearance >= -0.001 across all 4 MJCF models."""
    import mujoco
    import numpy as np

    m_pizero = mujoco.MjModel.from_xml_path("pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml")
    re_id = mujoco.mj_name2id(m_pizero, mujoco.mjtObj.mjOBJ_GEOM, "right_eye")
    le_id = mujoco.mj_name2id(m_pizero, mujoco.mjtObj.mjOBJ_GEOM, "left_eye")
    assert re_id != -1 and le_id != -1
    assert np.allclose(m_pizero.geom_size[re_id], m_pizero.geom_size[le_id]), (
        f"Eye size mismatch: right={m_pizero.geom_size[re_id]}, left={m_pizero.geom_size[le_id]}"
    )
    assert m_pizero.geom_pos[re_id][1] == pytest.approx(-m_pizero.geom_pos[le_id][1], abs=1e-4)
    assert m_pizero.geom_pos[re_id][0] == pytest.approx(m_pizero.geom_pos[le_id][0], abs=1e-4)
    assert m_pizero.geom_pos[re_id][2] == pytest.approx(m_pizero.geom_pos[le_id][2], abs=1e-4)

    for path in [
        "models/mjcf/microduck_8dof.xml",
        "pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml",
        "pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        "pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml",
    ]:
        assert_sole_clearance(path)


def test_pizero2w_target_mass_sync():
    """Verify TARGET_MASS in pizero2w env_config matches MuJoCo total mass."""
    import mujoco
    from pipelines.pizero2w_stack.training.env_config import _DEFAULT_CFG
    m = mujoco.MjModel.from_xml_path("pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml")
    total_mass = float(mujoco.mj_getTotalmass(m))
    assert _DEFAULT_CFG.TARGET_MASS == pytest.approx(total_mass, abs=1e-3), (
        f"PiZero2W TARGET_MASS {_DEFAULT_CFG.TARGET_MASS} != MuJoCo mass {total_mass}"
    )
