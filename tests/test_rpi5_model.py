import math
import os
import xml.etree.ElementTree as ET
import pytest

MJCF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipelines", "rpi5_stack", "models", "mjcf", "microduck_rpi5_8dof.xml"
)
URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pipelines", "rpi5_stack", "models", "urdf", "microduck_rpi5_8dof.urdf"
)

EXPECTED_JOINTS = {
    "head_yaw": [-1.57, 1.57],
    "head_pitch": [-0.52, 0.78],
    "left_hip_roll": [-0.44, 0.44],
    "left_hip_pitch": [-0.87, 0.87],
    "left_knee_pitch": [0.0, 1.48],
    "right_hip_roll": [-0.44, 0.44],
    "right_hip_pitch": [-0.87, 0.87],
    "right_knee_pitch": [0.0, 1.48],
}

EXPECTED_LIDAR_RAYS = [
    "lidar_ray_m30",
    "lidar_ray_m15",
    "lidar_ray_0",
    "lidar_ray_p15",
    "lidar_ray_p30",
]


def test_rpi5_mjcf_structure_and_joints():
    assert os.path.exists(MJCF_PATH), f"MJCF model not found: {MJCF_PATH}"
    tree = ET.parse(MJCF_PATH)
    root = tree.getroot()
    assert root.tag == "mujoco"
    assert root.attrib.get("model") == "microduck_rpi5_8dof"

    joints = {j.attrib["name"]: j for j in root.iter("joint") if j.attrib.get("name") in EXPECTED_JOINTS}
    assert len(joints) == 8, f"Expected 8 joints, found {len(joints)}"

    for name, expected_range in EXPECTED_JOINTS.items():
        joint_range = [float(x) for x in joints[name].attrib["range"].split()]
        assert pytest.approx(joint_range[0], abs=0.01) == expected_range[0]
        assert pytest.approx(joint_range[1], abs=0.01) == expected_range[1]

    # Check actuators (MG996R torque limit 1.0 N.m)
    actuators = list(root.iter("motor"))
    assert len(actuators) == 8, f"Expected 8 motor actuators, found {len(actuators)}"
    for act in actuators:
        assert float(act.attrib["ctrlrange"].split()[1]) == 1.0
        assert float(act.attrib["forcerange"].split()[1]) == 1.0


def test_rpi5_mjcf_perception_suite():
    assert os.path.exists(MJCF_PATH), f"MJCF model not found: {MJCF_PATH}"
    tree = ET.parse(MJCF_PATH)
    root = tree.getroot()

    sensor_elem = root.find("sensor")
    assert sensor_elem is not None, "Missing <sensor> section"

    # Base IMU
    assert sensor_elem.find("accelerometer") is not None, "Missing accelerometer"
    assert sensor_elem.find("gyro") is not None, "Missing gyro"
    assert sensor_elem.find("framequat") is not None, "Missing framequat"

    # Foot touch sensors
    touch_sensors = list(sensor_elem.iter("touch"))
    assert len(touch_sensors) >= 2, "Expected at least 2 touch sensors for feet"

    # 5-ray LiDAR sensors
    rangefinders = {rf.attrib["name"]: rf for rf in sensor_elem.iter("rangefinder")}
    for ray_name in EXPECTED_LIDAR_RAYS:
        assert ray_name in rangefinders, f"Missing rangefinder sensor: {ray_name}"
        rf = rangefinders[ray_name]
        cutoff = float(rf.attrib.get("cutoff", 2.0))
        assert pytest.approx(cutoff, abs=0.01) == 2.0, f"Rangefinder {ray_name} cutoff should be 2.0m"

    # Check sites for LiDAR rays in head_link
    sites = {s.attrib["name"]: s for s in root.iter("site")}
    for ray_name in EXPECTED_LIDAR_RAYS:
        site_name = rangefinders[ray_name].attrib.get("site")
        assert site_name in sites, f"LiDAR site {site_name} not found in model"
        site = sites[site_name]
        zaxis = [float(x) for x in site.attrib.get("zaxis", "1 0 0").split()]
        angle_deg = math.degrees(math.atan2(zaxis[1], zaxis[0]))
        expected_deg = {
            "lidar_ray_m30": -30.0,
            "lidar_ray_m15": -15.0,
            "lidar_ray_0": 0.0,
            "lidar_ray_p15": 15.0,
            "lidar_ray_p30": 30.0,
        }[ray_name]
        assert pytest.approx(angle_deg, abs=1.0) == expected_deg, (
            f"LiDAR ray {ray_name} angle expected {expected_deg}°, got {angle_deg}°"
        )

    # Camera
    cameras = {c.attrib["name"]: c for c in root.iter("camera")}
    assert "head_camera" in cameras, "Missing head_camera"
    head_cam = cameras["head_camera"]
    cam_pos = [float(x) for x in head_cam.attrib["pos"].split()]
    assert pytest.approx(cam_pos, abs=0.005) == [0.045, 0.0, 0.030]
    assert float(head_cam.attrib["fovy"]) == 60.0

    # Camera lens visual geom
    geoms = {g.attrib["name"]: g for g in root.iter("geom") if "name" in g.attrib}
    assert "cam_lens" in geoms, "Missing cam_lens visual geom"

    # Microphone site and visual geom
    assert "mic_site" in sites, "Missing mic_site"
    assert "mic_geom" in geoms, "Missing mic_geom visual geom"


def test_rpi5_mujoco_compilation_and_mass_budget():
    assert os.path.exists(MJCF_PATH), f"MJCF model not found: {MJCF_PATH}"
    try:
        import mujoco
    except ImportError:
        pytest.skip("MuJoCo python package not available")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    # Total mass ~860g (+/- 5%)
    total_mass = sum(model.body_mass)
    assert 0.817 <= total_mass <= 0.903, f"Expected total mass ~0.860kg (+/-5%), got {total_mass:.3f}kg"

    # base_link mass ~390g (+/- 5%)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert base_body_id >= 0, "base_link not found"
    base_mass = model.body_mass[base_body_id]
    assert 0.370 <= base_mass <= 0.410, f"Expected base_link mass ~0.390kg (+/-5%), got {base_mass:.3f}kg"

    # CoM of base_link elevated by ~10mm (z >= 0.008)
    base_ipos_z = model.body_ipos[base_body_id][2]
    assert base_ipos_z >= 0.008, f"Expected elevated CoM (z >= 0.008m), got {base_ipos_z:.4f}m"

    # Step 50 steps without NaN
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert not any(math.isnan(q) for q in data.qpos)
    assert not any(math.isnan(q) for q in data.qvel)


def test_rpi5_urdf_structure_and_inertials():
    assert os.path.exists(URDF_PATH), f"URDF model not found: {URDF_PATH}"
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib.get("name") == "microduck_rpi5_8dof"

    revolute_joints = [j for j in root.iter("joint") if j.attrib.get("type") == "revolute"]
    assert len(revolute_joints) == 8, f"Expected 8 revolute joints, found {len(revolute_joints)}"
    joints_by_name = {j.attrib["name"]: j for j in revolute_joints}
    assert set(joints_by_name.keys()) == set(EXPECTED_JOINTS.keys())

    for name, exp_range in EXPECTED_JOINTS.items():
        j = joints_by_name[name]
        limit = j.find("limit")
        assert limit is not None
        assert float(limit.attrib["effort"]) == 1.0
        assert float(limit.attrib["velocity"]) == 6.0
        assert pytest.approx(float(limit.attrib["lower"]), abs=0.01) == exp_range[0]
        assert pytest.approx(float(limit.attrib["upper"]), abs=0.01) == exp_range[1]

        dynamics = j.find("dynamics")
        assert dynamics is not None
        assert pytest.approx(float(dynamics.attrib.get("damping", 0.0)), abs=0.01) == 0.15

    # Check perception frames/links in URDF
    links_by_name = {l.attrib["name"]: l for l in root.iter("link")}
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
        "head_camera_link",
        "mic_link",
    ] + [f"{ray}_link" for ray in EXPECTED_LIDAR_RAYS]

    for link_name in expected_links:
        assert link_name in links_by_name, f"Missing URDF link {link_name}"

    # Masses & Triangle Inequality
    total_mass = 0.0
    for name, link in links_by_name.items():
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        if mass_elem is None:
            continue
        m = float(mass_elem.attrib["value"])
        total_mass += m

        inertia = inertial.find("inertia")
        if inertia is not None:
            ixx = float(inertia.attrib["ixx"])
            iyy = float(inertia.attrib["iyy"])
            izz = float(inertia.attrib["izz"])
            assert ixx > 0 and iyy > 0 and izz > 0, f"Link {name} principal inertias must be positive"
            assert ixx + iyy >= izz - 1e-9, f"Link {name} violates triangle inequality (ixx+iyy < izz)"
            assert ixx + izz >= iyy - 1e-9, f"Link {name} violates triangle inequality (ixx+izz < iyy)"
            assert iyy + izz >= ixx - 1e-9, f"Link {name} violates triangle inequality (iyy+izz < ixx)"

    assert 0.817 <= total_mass <= 0.903, f"Expected total URDF mass ~0.860kg, got {total_mass:.3f}kg"
    base_mass = float(links_by_name["base_link"].find("inertial").find("mass").attrib["value"])
    assert 0.370 <= base_mass <= 0.410, f"Expected URDF base_link mass ~0.390kg, got {base_mass:.3f}kg"


def test_rpi5_urdf_mujoco_compilation():
    assert os.path.exists(URDF_PATH), f"URDF model not found: {URDF_PATH}"
    try:
        import mujoco
    except ImportError:
        pytest.skip("MuJoCo python package not available")

    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert model.nu == 8 or model.njnt >= 8
