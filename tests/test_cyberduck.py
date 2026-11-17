import os
import sys
import struct
import pytest
import mujoco
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

V2_DIR = os.path.join(os.path.dirname(__file__), "..", "duck_v2")
MESH_DIR = os.path.join(V2_DIR, "meshes")
CAD_DIR = os.path.join(V2_DIR, "cad")
MJCF_PATH = os.path.join(V2_DIR, "cyberduck_v2.xml")

REQUIRED_MESHES = [
    "cyber_helm.stl",
    "cyber_visor.stl",
    "cyber_beak.stl",
    "cyber_neck.stl",
    "cyber_chest.stl",
    "cyber_reactor.stl",
    "cyber_thruster_pack.stl",
    "cyber_hip_left.stl",
    "cyber_hip_right.stl",
    "cyber_thigh_left.stl",
    "cyber_thigh_right.stl",
    "cyber_shin_left.stl",
    "cyber_shin_right.stl",
    "cyber_foot_left.stl",
    "cyber_foot_right.stl",
    "cyber_sole_left.stl",
    "cyber_sole_right.stl",
]

def count_stl_triangles(path):
    assert os.path.exists(path), f"File {path} does not exist"
    data = open(path, "rb").read()
    assert len(data) >= 84, "Invalid binary STL header"
    (n,) = struct.unpack_from("<I", data, 80)
    assert len(data) == 84 + n * 50, f"STL file size mismatch for {path}"
    return n

def test_cyberduck_v2_meshes_exist_and_high_def():
    """Verify all 13 custom CyberDuck V2 3D meshes exist and have high polygon counts (>1,000 triangles)."""
    assert os.path.isdir(MESH_DIR), f"Directory {MESH_DIR} not found"
    for mesh_name in REQUIRED_MESHES:
        mesh_path = os.path.join(MESH_DIR, mesh_name)
        tri_count = count_stl_triangles(mesh_path)
        assert tri_count >= 1000, f"Mesh {mesh_name} should be high-definition, but has only {tri_count} triangles"

def test_cyberduck_v2_mjcf_validity_and_kinematics():
    """Verify the CyberDuck V2 MuJoCo model loads cleanly and satisfies 8-DOF physical constraints."""
    assert os.path.exists(MJCF_PATH), f"MJCF model {MJCF_PATH} not found"
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Must have exactly 8 actuated joints
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert len(actuator_names) == 8, f"Expected 8 actuators, got {len(actuator_names)}"

    # Must have low, balanced center of mass
    com = data.subtree_com[model.body("trunk_base").id]
    assert abs(com[1]) < 0.015, f"CoM Y lateral asymmetry too large: {com[1]}"

    # Check ground contact
    geom_z_mins = []
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "sole" in geom_name:
            geom_z_mins.append(data.geom_xpos[i][2])

    assert len(geom_z_mins) >= 2, "Model missing sole collision geoms"
    for z in geom_z_mins:
        assert abs(z - 0.005) < 0.01, f"Sole contact height not at ground level: {z}"

def test_cyberduck_v2_cad_exports():
    """Verify CAD OBJ, MTL, and ZIP exports for V2 (both outer assembly and internal mechanical skeleton) are present."""
    obj_path = os.path.join(CAD_DIR, "cyberduck_v2_assembly.obj")
    mtl_path = os.path.join(CAD_DIR, "cyberduck_v2_assembly.mtl")
    zip_path = os.path.join(CAD_DIR, "cyberduck_v2_stl_parts.zip")

    assert os.path.exists(obj_path), "Assembly OBJ not found"
    assert os.path.getsize(obj_path) > 1000000, "Assembly OBJ file too small"
    assert os.path.exists(mtl_path), "MTL file not found"
    assert os.path.exists(zip_path), "STL ZIP file not found"

    # Mechanical Skeleton Chassis CAD files
    skel_obj = os.path.join(CAD_DIR, "cyberduck_v2_skeleton.obj")
    skel_mtl = os.path.join(CAD_DIR, "cyberduck_v2_skeleton.mtl")
    skel_zip = os.path.join(CAD_DIR, "cyberduck_v2_skeleton_parts.zip")
    assert os.path.exists(skel_obj), "Mechanical Skeleton OBJ not found"
    assert os.path.getsize(skel_obj) > 100000, "Mechanical Skeleton OBJ file too small"
    assert os.path.exists(skel_mtl), "Mechanical Skeleton MTL file not found"
    assert os.path.exists(skel_zip), "Mechanical Skeleton ZIP file not found"

def test_cyberduck_v2_sensor_suite_and_camera():
    """Verify that CyberDuck V2 contains the full sensor suite and head camera matching 8-DOF RL."""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)

    # Check sensor count (21 standard + 5 lidar rangefinder = 26 sensors, 33 scalar sensordata)
    sensor_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]
    required_sensors = [
        "base_accel", "base_gyro", "base_quat",
        "left_foot_touch", "right_foot_touch",
        "head_yaw_pos", "head_pitch_pos",
        "left_hip_roll_pos", "left_hip_pitch_pos", "left_knee_pitch_pos",
        "right_hip_roll_pos", "right_hip_pitch_pos", "right_knee_pitch_pos",
        "head_yaw_vel", "head_pitch_vel",
        "left_hip_roll_vel", "left_hip_pitch_vel", "left_knee_pitch_vel",
        "right_hip_roll_vel", "right_hip_pitch_vel", "right_knee_pitch_vel",
        "lidar_ray_m30", "lidar_ray_m15", "lidar_ray_0", "lidar_ray_p15", "lidar_ray_p30"
    ]
    assert len(sensor_names) == 26, f"Expected 26 sensors, got {len(sensor_names)}"
    for sname in required_sensors:
        assert sname in sensor_names, f"Missing required sensor {sname} in CyberDuck V2"

    # Check head camera
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    assert cam_id != -1, "CyberDuck V2 missing 'head_camera'"

    # Check foot touch sites
    l_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_touch_site")
    r_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_touch_site")
    assert l_site != -1 and r_site != -1, "CyberDuck V2 missing foot touch sites"

def test_cyberduck_v2_parallel_sim_integration():
    """Verify CyberDuck V2 runs in ParallelSimRunner and outputs standard (num_envs, 42) observations."""
    from training.run_parallel_sim import ParallelSimRunner
    sim = ParallelSimRunner(num_envs=2, model_path=MJCF_PATH, decimation=1)
    obs = sim.get_observations()
    assert obs.shape == (2, 42), f"Expected obs shape (2, 42), got {obs.shape}"

    actions = np.zeros((2, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)
    assert next_obs.shape == (2, 42)
    assert len(rewards) == 2
    assert len(dones) == 2

def test_cyberduck_v2_sites_are_visible_and_functional():
    """Verify all sensory sites are visible and cleanly mounted on the robot body without floating mid-air, while sensordata is valid."""
    assert os.path.exists(MJCF_PATH), f"MJCF model {MJCF_PATH} not found"
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Check that all sensory sites are visible (alpha > 0) and belong to visual site group 2
    sensory_site_names = [
        "imu", "imu_site", "camera_site", "site_lidar_m30", "site_lidar_m15",
        "site_lidar_0", "site_lidar_p15", "site_lidar_p30",
        "left_foot_touch_site", "right_foot_touch_site"
    ]
    for sname in sensory_site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        assert site_id != -1, f"Missing site {sname}"
        site_rgba = model.site_rgba[site_id]
        assert site_rgba[3] > 0.0, f"Site {sname} has zero alpha, making sensor indicator completely invisible!"
        assert model.site_group[site_id] == 2, f"Site {sname} must belong to group 2, got {model.site_group[site_id]}"

    # Ensure LiDAR sites are located on helmet surface, not floating > 10mm outside
    for lname in ["site_lidar_m30", "site_lidar_m15", "site_lidar_0", "site_lidar_p15", "site_lidar_p30"]:
        lidar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, lname)
        pos_x = model.site_pos[lidar_id][0]
        assert pos_x <= 0.028, f"LiDAR site {lname} is floating outside helmet: X={pos_x} > 0.028"

    # Verify sensordata is fully valid and finite (IMU accel, gyro, quat, foot touch, lidar)
    assert not np.any(np.isnan(data.sensordata)), "Sensordata contains NaN values!"
    assert len(data.sensordata) == 33, f"Expected 33 sensordata values, got {len(data.sensordata)}"

def test_cyberduck_v2_lighting_and_material_quality():
    """Verify that CyberDuck V2 visual config has headlight fill, high shadow quality, matte gray armor, and bronze accents."""
    assert os.path.exists(MJCF_PATH), f"MJCF model {MJCF_PATH} not found"
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)

    # Check visual quality shadow size
    assert model.vis.quality.shadowsize >= 4096, "Shadow size should be at least 4096 for crisp shading"

    # Check headlight ambient
    headlight_amb = model.vis.headlight.ambient
    assert headlight_amb[0] >= 0.40, f"Headlight ambient too low ({headlight_amb[0]}), causes dark dingy shadows"

    # Check mecha armor material (sleek matte gray / xám nhám)
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "mecha_white")
    assert mat_id != -1, "mecha_white material missing"
    rgba = model.mat_rgba[mat_id]
    assert 0.28 <= rgba[0] <= 0.45, f"mecha armor color should be sleek matte gray, got R={rgba[0]}"
    assert model.mat_specular[mat_id] <= 0.35, "matte armor should have low specular reflection"

    # Check bronze accent material (vàng đồng)
    gold_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "mecha_gold")
    assert gold_id != -1, "mecha_gold material missing"
    gold_rgba = model.mat_rgba[gold_id]
    assert gold_rgba[0] > gold_rgba[1] > gold_rgba[2], "mecha_gold should have warm bronze color (R > G > B)"
    assert model.mat_specular[gold_id] >= 0.60, "mecha_gold should have metallic specular shine"

    # Check that only 1 light casts shadow to prevent shadow acne
    cast_shadow_count = sum(1 for i in range(model.nlight) if model.light_castshadow[i])
    assert cast_shadow_count == 1, f"Expected exactly 1 shadow-casting light to prevent acne, got {cast_shadow_count}"

def test_cyberduck_v2_collision_geoms_transparent():
    """Verify that collision geometry class sets alpha=0 to prevent yellow overlay boxes."""
    assert os.path.exists(MJCF_PATH), f"MJCF model {MJCF_PATH} not found"
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)

    # Check left and right sole geom rgba and group
    for name in ["left_sole_geom", "right_sole_geom"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid != -1, f"Missing geom {name}"
        assert model.geom_rgba[gid][3] == 0.0, f"Collision geom {name} alpha must be 0, got {model.geom_rgba[gid][3]}"
        assert model.geom_group[gid] == 3, f"Collision geom {name} group must be 3, got {model.geom_group[gid]}"


def test_cyberduck_v2_foot_dynamics_and_heading_turn_gate():
    from training.run_parallel_sim import ParallelSimRunner
    from training.env_config import compute_rewards

    runner = ParallelSimRunner(model_path="duck_v2/cyberduck_v2.xml", num_envs=1)
    try:
        assert runner.left_foot_gid != -1 and runner.right_foot_gid != -1
        assert runner.left_foot_bid != -1 and runner.right_foot_bid != -1
        runner.reset()
        _, _, _, infos = runner.step(np.zeros((1, 8), dtype=np.float32))
        fz, fv, ff = runner._extract_foot_dynamics()
        assert np.any(np.abs(ff[..., 2]) > 0.1), "V2 foot contact forces must be non-zero on ground"
    finally:
        runner.close()

    rew_turn = compute_rewards(
        base_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.array([0.0, 0.0, 0.4], dtype=np.float32),
        cmd_ang_vel=0.4,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        current_yaw=0.8,
        target_yaw=0.0,
    )
    assert rew_turn["heading_penalty"] == 0.0


def test_cyberduck_v2_lightweight_collisions_curriculum_and_ground_clearance():
    """Verify CyberDuck V2 uses primitive colliders, contact exclusions, curriculum terrain, and ground clearance."""
    from training.terrain_curriculum import apply_stage, apply_stair_stage
    from training.run_parallel_sim import ParallelSimRunner

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 1. No active mesh colliders slowing down 512-env parallel physics
    active_mesh_cols = [
        i for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and (model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0)
    ]
    assert active_mesh_cols == [], f"Mesh collision geoms degrade FPS: {len(active_mesh_cols)}"

    # 2. Contact exclusions
    assert model.nexclude >= 10, f"Expected at least 10 contact exclusions, got {model.nexclude}"

    # 3. Curriculum terrain obstacle geoms exist
    curriculum_geoms = [
        "pillar_l1_left", "pillar_l1_right", "pillar_l2_center",
        "pillar_l3_left", "pillar_l3_right", "pillar_back",
        "stair_step_1", "stair_step_2", "stair_step_3",
        "stair_platform", "stair_dn_1", "stair_dn_2", "floor_exit"
    ]
    for name in curriculum_geoms:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1, f"Missing curriculum geom {name}"

    # 4. Soles at or above ground level (soles_bottom >= 0.0)
    for sole in ("left_sole_geom", "right_sole_geom"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, sole)
        assert gid != -1, f"Missing {sole}"
        sole_bottom = data.geom_xpos[gid][2] - model.geom_size[gid][2]
        assert sole_bottom >= -1e-4, f"{sole} bottom submerged: {sole_bottom:.5f} < -1e-4"

    # 5. Rangefinder cutoff == 2.0
    for lname in ("lidar_ray_m30", "lidar_ray_m15", "lidar_ray_0", "lidar_ray_p15", "lidar_ray_p30"):
        cutoff = model.sensor(lname).cutoff[0]
        assert abs(cutoff - 2.0) < 1e-4, f"Rangefinder {lname} cutoff should be 2.0, got {cutoff}"

    # 6. Actuator forcerange standardized to [-1.0, 1.0]
    assert np.allclose(model.actuator_forcerange, [[-1.0, 1.0]] * model.nu), (
        f"Actuator forcerange mismatch: {model.actuator_forcerange}"
    )

    # 7. Curriculum apply_stage and ParallelSimRunner integration
    apply_stage(model, 0)
    runner = ParallelSimRunner(model_path=MJCF_PATH, num_envs=2)
    try:
        assert len(runner.obstacle_geom_ids) == 6, f"Expected 6 obstacle geom IDs, got {len(runner.obstacle_geom_ids)}"
        runner.set_stage(1)
        apply_stair_stage(runner.model, 2)
    finally:
        runner.close()


def test_cyberduck_v2_joint_frictionloss_damping_and_sole_calibration():
    """Verify calibrated joint frictionloss (0.008), damping (0.25), and widened sole/touch_site geometry."""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)

    # Actuated joint DoFs (indices 6:14, after freejoint 0:6)
    np.testing.assert_allclose(
        model.dof_frictionloss[6:14],
        0.008,
        atol=1e-5,
        err_msg="Joint frictionloss must be tuned to 0.008 Nm (low static stick-slip)",
    )
    np.testing.assert_allclose(
        model.dof_damping[6:14],
        0.25,
        atol=1e-5,
        err_msg="Joint damping must be calibrated to 0.25 Nms/rad for smooth PD tracking",
    )
    np.testing.assert_allclose(
        model.dof_armature[6:14],
        0.005,
        atol=1e-5,
        err_msg="Joint armature must be 0.005 kg*m^2",
    )

    # Sole and touch_site dimensions
    expected_sole_size = np.array([0.025, 0.018, 0.004])
    expected_site_size = np.array([0.027, 0.019, 0.005])

    for sname in ("left_sole_geom", "right_sole_geom"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, sname)
        assert gid != -1, f"Missing geom {sname}"
        np.testing.assert_allclose(
            model.geom_size[gid],
            expected_sole_size,
            atol=1e-5,
            err_msg=f"{sname} size must be {expected_sole_size}",
        )

    for tname in ("left_foot_touch_site", "right_foot_touch_site"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tname)
        assert sid != -1, f"Missing site {tname}"
        np.testing.assert_allclose(
            model.site_size[sid],
            expected_site_size,
            atol=1e-5,
            err_msg=f"{tname} size must be {expected_site_size}",
        )

    # Touch site must strictly enclose sole in all 3 half-dimensions
    assert np.all(expected_site_size > expected_sole_size), "Touch site must strictly surround sole geometry"


def test_cyberduck_v2_static_stand_zero_self_collisions_and_roll_stability():
    """Static stand test with PD targeting nominal pose: 0 self-collisions, max(|hip_roll|) <= 0.02, max(|ctrl|) < 0.25."""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll")
    right_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll")
    left_roll_qpos_adr = model.jnt_qposadr[left_roll_id]
    right_roll_qpos_adr = model.jnt_qposadr[right_roll_id]

    nominal = np.array([0.0, 0.0, 0.0, -0.15, 0.15, 0.0, -0.15, 0.15], dtype=np.float64)
    data.qpos[7:15] = nominal
    mujoco.mj_forward(model, data)

    kp = 13.5
    kd = 0.9
    hip_rolls = []
    ctrl_magnitudes = []
    self_collisions = 0

    for _ in range(100):
        error = nominal - data.qpos[7:15]
        d_error = - data.qvel[6:14]
        ctrl = np.clip(kp * error + kd * d_error, -1.0, 1.0)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        hip_rolls.append(abs(data.qpos[left_roll_qpos_adr]))
        hip_rolls.append(abs(data.qpos[right_roll_qpos_adr]))
        ctrl_magnitudes.append(np.max(np.abs(data.ctrl)))

        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 != floor_gid and c.geom2 != floor_gid:
                self_collisions += 1

    assert self_collisions == 0, f"Expected 0 self-collisions during static stand, got {self_collisions}"
    max_hip_roll = max(hip_rolls)
    assert max_hip_roll <= 0.02, f"Hip roll deviation too large during static stand: {max_hip_roll:.4f} > 0.02 rad"
    max_ctrl = max(ctrl_magnitudes)
    assert max_ctrl < 0.25, f"Static stand required excessive control torque: {max_ctrl:.4f} >= 0.25 Nm"


def test_cyberduck_v2_random_walking_poses_zero_trunk_collisions():
    """Sweep 500 random walking poses across realistic joint angles: 0 collisions involving trunk_base with thighs or head."""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    lt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_thigh")
    rt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_thigh")
    head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    target_bodies = {lt_bid, rt_bid, head_bid}

    rng = np.random.RandomState(42)
    collisions = 0

    for _ in range(500):
        # head_yaw [-0.5, 0.5], head_pitch [-0.3, 0.4]
        # left_hip_roll [-0.25, 0.25], left_hip_pitch [-0.5, 0.5], left_knee_pitch [0.0, 0.85]
        # right_hip_roll [-0.25, 0.25], right_hip_pitch [-0.5, 0.5], right_knee_pitch [0.0, 0.85]
        q = np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.3, 0.4),
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.0, 0.85),
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.0, 0.85),
        ], dtype=np.float64)

        data.qpos[7:15] = q
        data.qpos[2] = 0.5  # lift in air to isolate self-collisions from floor
        mujoco.mj_forward(model, data)

        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if (b1 == trunk_bid and b2 in target_bodies) or (b2 == trunk_bid and b1 in target_bodies):
                collisions += 1

    assert collisions == 0, f"Expected 0 contacts between trunk_base and thighs/head across 500 poses, got {collisions}"
