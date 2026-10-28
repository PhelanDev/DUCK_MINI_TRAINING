import xml.etree.ElementTree as ET
import os
import pytest

def test_mjcf_xml_structure_and_joints():
    xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "microduck_8dof.xml")
    assert os.path.exists(xml_path), f"File {xml_path} does not exist"
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.tag == "mujoco"
    assert root.attrib.get("model") == "microduck_8dof"
    
    # Check 8 joints
    expected_joints = {
        "head_yaw": [-1.57, 1.57],
        "head_pitch": [-0.52, 0.78],
        "left_hip_roll": [-0.44, 0.44],
        "left_hip_pitch": [-0.87, 0.87],
        "left_knee_pitch": [0.0, 1.48],
        "right_hip_roll": [-0.44, 0.44],
        "right_hip_pitch": [-0.87, 0.87],
        "right_knee_pitch": [0.0, 1.48],
    }
    joints = {j.attrib["name"]: j for j in root.iter("joint") if j.attrib.get("name") in expected_joints}
    assert len(joints) == 8, f"Expected 8 joints, found {len(joints)}"
    
    for name, expected_range in expected_joints.items():
        joint_range = [float(x) for x in joints[name].attrib["range"].split()]
        assert pytest.approx(joint_range[0], abs=0.01) == expected_range[0]
        assert pytest.approx(joint_range[1], abs=0.01) == expected_range[1]
    
    # Check actuators
    actuators = list(root.iter("motor"))
    assert len(actuators) == 8, f"Expected 8 motor actuators, found {len(actuators)}"
    for act in actuators:
        assert float(act.attrib["ctrlrange"].split()[1]) == 1.0
        assert float(act.attrib["forcerange"].split()[1]) == 1.0

def test_mjcf_sensors_and_defaults():
    xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "microduck_8dof.xml")
    if not os.path.exists(xml_path):
        pytest.fail(f"File {xml_path} does not exist")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Check sensors
    sensor_elem = root.find("sensor")
    assert sensor_elem is not None, "Missing <sensor> section"
    accelerometer = sensor_elem.find("accelerometer")
    gyro = sensor_elem.find("gyro")
    framequat = sensor_elem.find("framequat")
    assert accelerometer is not None, "Missing accelerometer sensor"
    assert gyro is not None, "Missing gyro sensor"
    assert framequat is not None, "Missing framequat sensor"

    # Check foot contact touch sensors
    touch_sensors = list(sensor_elem.iter("touch"))
    assert len(touch_sensors) >= 2, "Expected at least 2 touch sensors for feet"

def test_mujoco_compilation():
    xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "microduck_8dof.xml")
    if not os.path.exists(xml_path):
        pytest.fail(f"File {xml_path} does not exist")
    try:
        import mujoco
    except ImportError:
        pytest.skip("MuJoCo python package not available")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    
    # Total mass ~ 780g (0.78 kg) +/- 10%
    total_mass = sum(model.body_mass)
    assert 0.70 <= total_mass <= 0.85, f"Expected total mass ~0.78kg, got {total_mass:.3f}kg"
