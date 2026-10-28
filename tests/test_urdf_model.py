import os
import xml.etree.ElementTree as ET
import pytest

def test_urdf_structure_and_actuators():
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "models", "urdf", "microduck_8dof.urdf")
    assert os.path.exists(urdf_path), f"File {urdf_path} does not exist"
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib.get("name") == "microduck_8dof"
    
    expected_joints = {
        "head_yaw": {"axis": [0, 0, 1], "lower": -1.57, "upper": 1.57},
        "head_pitch": {"axis": [0, 1, 0], "lower": -0.52, "upper": 0.78},
        "left_hip_roll": {"axis": [1, 0, 0], "lower": -0.44, "upper": 0.44},
        "left_hip_pitch": {"axis": [0, 1, 0], "lower": -0.87, "upper": 0.87},
        "left_knee_pitch": {"axis": [0, 1, 0], "lower": 0.0, "upper": 1.48},
        "right_hip_roll": {"axis": [1, 0, 0], "lower": -0.44, "upper": 0.44},
        "right_hip_pitch": {"axis": [0, 1, 0], "lower": -0.87, "upper": 0.87},
        "right_knee_pitch": {"axis": [0, 1, 0], "lower": 0.0, "upper": 1.48},
    }
    
    revolute_joints = [j for j in root.iter("joint") if j.attrib.get("type") == "revolute"]
    assert len(revolute_joints) == 8, f"Expected 8 revolute joints, found {len(revolute_joints)}"
    
    joints_by_name = {j.attrib["name"]: j for j in revolute_joints}
    assert set(joints_by_name.keys()) == set(expected_joints.keys())
    
    for name, exp in expected_joints.items():
        j = joints_by_name[name]
        limit = j.find("limit")
        assert limit is not None, f"Joint {name} missing <limit>"
        assert float(limit.attrib["effort"]) == 1.0
        assert float(limit.attrib["velocity"]) == 6.0
        assert pytest.approx(float(limit.attrib["lower"]), abs=0.01) == exp["lower"]
        assert pytest.approx(float(limit.attrib["upper"]), abs=0.01) == exp["upper"]
        
        dynamics = j.find("dynamics")
        assert dynamics is not None, f"Joint {name} missing <dynamics>"
        assert pytest.approx(float(dynamics.attrib.get("damping", 0.0)), abs=0.01) == 0.15
        
        axis = j.find("axis")
        assert axis is not None, f"Joint {name} missing <axis>"
        axis_xyz = [float(x) for x in axis.attrib["xyz"].split()]
        assert axis_xyz == exp["axis"]

def test_urdf_links_and_inertials():
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "models", "urdf", "microduck_8dof.urdf")
    assert os.path.exists(urdf_path), f"File {urdf_path} does not exist"
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    expected_links = [
        "base_link",
        "neck_link",
        "head_link",
        "left_hip_roll_link",
        "left_thigh_link",
        "left_shin_link",
        "left_foot_link",
        "right_hip_roll_link",
        "right_thigh_link",
        "right_shin_link",
        "right_foot_link",
    ]
    
    links_by_name = {l.attrib["name"]: l for l in root.iter("link")}
    for exp_link in expected_links:
        assert exp_link in links_by_name, f"Missing link {exp_link}"
    
    total_mass = 0.0
    for name, link in links_by_name.items():
        inertial = link.find("inertial")
        assert inertial is not None, f"Link {name} missing <inertial>"
        mass_elem = inertial.find("mass")
        assert mass_elem is not None, f"Link {name} missing <mass>"
        m = float(mass_elem.attrib["value"])
        assert m > 0, f"Link {name} mass must be positive"
        total_mass += m
        
        inertia = inertial.find("inertia")
        assert inertia is not None, f"Link {name} missing <inertia>"
        ixx = float(inertia.attrib["ixx"])
        iyy = float(inertia.attrib["iyy"])
        izz = float(inertia.attrib["izz"])
        assert ixx > 0 and iyy > 0 and izz > 0, f"Link {name} principal inertias must be positive"
        # Triangle inequality for valid rigid body
        assert ixx + iyy >= izz - 1e-9, f"Link {name} violates triangle inequality (ixx+iyy < izz)"
        assert ixx + izz >= iyy - 1e-9, f"Link {name} violates triangle inequality (ixx+izz < iyy)"
        assert iyy + izz >= ixx - 1e-9, f"Link {name} violates triangle inequality (iyy+izz < ixx)"
        
        # Check visual and collision
        visual = link.find("visual")
        collision = link.find("collision")
        assert visual is not None, f"Link {name} missing <visual>"
        assert collision is not None, f"Link {name} missing <collision>"

    assert 0.70 <= total_mass <= 0.85, f"Expected total mass ~0.78kg, got {total_mass:.3f}kg"

def test_urdf_mujoco_compilation():
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "models", "urdf", "microduck_8dof.urdf")
    assert os.path.exists(urdf_path), f"File {urdf_path} does not exist"
    
    try:
        import mujoco
    except ImportError:
        pytest.skip("MuJoCo python package not available")
        
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    
    # 8 revolute joints
    assert model.nu == 8 or model.njnt >= 8
