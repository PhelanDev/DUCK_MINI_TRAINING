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
    os.path.dirname(__file__), "..", "pipelines", "rpi5_stack", "models", "mjcf", "microduck_rpi5_8dof.xml"
)
URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipelines", "rpi5_stack", "models", "urdf", "microduck_rpi5_8dof.urdf"
)


def test_rpi5_mjcf_structure_and_joints():
    assert_mjcf_structure_and_joints(MJCF_PATH, "microduck_rpi5_8dof")


def test_rpi5_mjcf_perception_suite():
    assert_mjcf_perception_suite(MJCF_PATH)


def test_rpi5_mujoco_compilation_and_mass_budget():
    assert_mujoco_compilation_and_mass(
        MJCF_PATH,
        min_mass=0.725,
        max_mass=0.805,
        base_mass_min=0.370,
        base_mass_max=0.410,
        base_ipos_z_min=0.008,
        n_steps=50,
    )


def test_rpi5_urdf_structure_and_inertials():
    assert_urdf_structure_and_joint_limits(
        URDF_PATH,
        robot_name="microduck_rpi5_8dof",
        min_total_mass=0.725,
        max_total_mass=0.805,
        min_base_mass=0.370,
        max_base_mass=0.410,
    )


def test_rpi5_urdf_mujoco_compilation():
    assert_urdf_mujoco_compilation(URDF_PATH)


def test_rpi5_mjcf_obstacles():
    assert_slalom_pillars_and_stairs(MJCF_PATH)


def test_rpi5_sole_clearance_and_target_mass():
    from pipelines.rpi5_stack.training.env_config import _DEFAULT_CFG
    assert_sole_clearance_and_target_mass(MJCF_PATH, _DEFAULT_CFG.TARGET_MASS)
