import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_test_helpers import (
    assert_mjcf_structure_and_joints,
    assert_slalom_pillars_and_stairs,
    assert_sole_clearance,
)

XML_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "microduck_8dof.xml")


def test_mjcf_xml_structure_and_joints():
    assert_mjcf_structure_and_joints(XML_PATH, "microduck_8dof")


def test_mjcf_sensors_and_defaults():
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    sensor_elem = root.find("sensor")
    assert sensor_elem is not None, "Missing <sensor> section"
    assert sensor_elem.find("accelerometer") is not None, "Missing accelerometer sensor"
    assert sensor_elem.find("gyro") is not None, "Missing gyro sensor"
    assert sensor_elem.find("framequat") is not None, "Missing framequat sensor"

    touch_sensors = list(sensor_elem.iter("touch"))
    assert len(touch_sensors) >= 2, "Expected at least 2 touch sensors for feet"


def test_mujoco_compilation():
    try:
        import mujoco
    except ImportError:
        pytest.skip("MuJoCo python package not available")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    total_mass = sum(model.body_mass)
    assert 0.620 <= total_mass <= 0.740, f"Expected total mass ~0.684kg, got {total_mass:.3f}kg"


def test_slalom_pillars():
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None
    pillar_names = {
        "pillar_l1_left", "pillar_l1_right", "pillar_l2_center",
        "pillar_l3_left", "pillar_l3_right", "pillar_back",
    }
    geoms = {g.attrib.get("name"): g for g in worldbody.findall("geom")}
    for name in pillar_names:
        assert name in geoms, f"Missing {name} slalom pillar in worldbody"
        geom = geoms[name]
        assert geom.attrib.get("type") == "cylinder"
        assert geom.attrib.get("contype") == "1"
        assert geom.attrib.get("conaffinity") == "1"


def test_staircase_and_platform():
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None
    stair_names = {"stair_step_1", "stair_step_2", "stair_step_3", "stair_platform"}
    geoms = {g.attrib.get("name"): g for g in worldbody.findall("geom")}
    for name in stair_names:
        assert name in geoms, f"Missing {name} in worldbody"
        geom = geoms[name]
        assert geom.attrib.get("type") == "box"
        assert geom.attrib.get("contype") == "1"
        assert geom.attrib.get("conaffinity") == "1"
        assert geom.attrib.get("group") == "3"


def test_slalom_pillars_and_stairs_exist():
    assert_slalom_pillars_and_stairs(XML_PATH)


def test_microduck_8dof_sole_clearance():
    assert_sole_clearance(XML_PATH)


def test_inspect_parts_script_runs():
    import subprocess
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "inspect_parts.py")
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert res.returncode == 0, f"inspect_parts.py failed with: {res.stderr}\nOutput: {res.stdout}"
