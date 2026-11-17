import os
import sys
from pathlib import Path

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
    assert_urdf_mujoco_compilation,
    assert_slalom_pillars_and_stairs,
    assert_sole_clearance_and_target_mass,
)

MJCF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipelines", "dual_esp32s3_stack", "models", "mjcf", "microduck_esp32_8dof.xml"
)
URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipelines", "dual_esp32s3_stack", "models", "urdf", "microduck_esp32_8dof.urdf"
)


def test_esp32_mjcf_structure_and_joints():
    assert_mjcf_structure_and_joints(MJCF_PATH, "microduck_esp32_8dof")


def test_esp32_mjcf_perception_suite():
    assert_mjcf_perception_suite(MJCF_PATH)


def test_esp32_mujoco_compilation_and_mass_budget():
    assert_mujoco_compilation_and_mass(
        MJCF_PATH,
        min_mass=0.650,
        max_mass=0.720,
        base_mass_min=0.295,
        base_mass_max=0.336,
        base_ipos_z_max=0.002,
        n_steps=50,
    )


def test_esp32_urdf_structure_and_inertials():
    assert_urdf_structure_and_joint_limits(
        URDF_PATH,
        robot_name="microduck_esp32_8dof",
        min_total_mass=0.650,
        max_total_mass=0.720,
        min_base_mass=0.295,
        max_base_mass=0.336,
        base_inertial_z_max=0.002,
    )


def test_esp32_urdf_mujoco_compilation():
    assert_urdf_mujoco_compilation(URDF_PATH)


def test_esp32_mjcf_obstacles():
    assert_slalom_pillars_and_stairs(MJCF_PATH)


def test_esp32_sole_clearance_and_target_mass():
    from pipelines.dual_esp32s3_stack.training.env_config import _DEFAULT_CFG
    assert_sole_clearance_and_target_mass(MJCF_PATH, _DEFAULT_CFG.TARGET_MASS)
