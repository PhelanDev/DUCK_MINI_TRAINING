import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import compute_rewards as root_compute_rewards, _DEFAULT_CFG as root_cfg
from pipelines.rpi5_stack.training.env_config import compute_rewards as rpi5_compute_rewards, _DEFAULT_CFG as rpi5_cfg
from pipelines.dual_esp32s3_stack.training.env_config import compute_rewards as esp32_compute_rewards, _DEFAULT_CFG as esp32_cfg


@pytest.mark.parametrize(
    "compute_fn,tag",
    [
        (root_compute_rewards, "root"),
        (rpi5_compute_rewards, "rpi5"),
        (esp32_compute_rewards, "esp32"),
    ],
)
def test_backward_velocity_double_penalty(compute_fn, tag):
    """Verify linear velocity reward awards forward velocity (2.0 * vel_x_eff),
    harshly penalizes backward velocity (4.0 * vel_x), and preserves negative
    penalty when double-support dwell occurs.
    """
    cmd_lin = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    cmd_ang_vel = 0.0
    proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_torques = np.zeros(8, dtype=np.float32)
    joint_vel = np.zeros(8, dtype=np.float32)
    foot_contact = np.ones(4, dtype=np.float32)

    # 1. Forward velocity test: vel_x = +0.20 m/s -> reward = 2.0 * 0.20 = +0.40
    base_lin_forward = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    r_fwd = compute_fn(
        base_lin_vel=base_lin_forward,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        double_support_time=0.0,
    )
    assert np.isclose(r_fwd["tracking_lin_vel"], 0.40, atol=1e-5), (
        f"[{tag}] Expected forward reward 0.40 for vel_x=0.20, got {r_fwd['tracking_lin_vel']}"
    )

    # 2. Backward velocity test: vel_x = -0.20 m/s -> penalty = 4.0 * (-0.20) = -0.80
    base_lin_backward = np.array([-0.20, 0.0, 0.0], dtype=np.float32)
    r_bwd = compute_fn(
        base_lin_vel=base_lin_backward,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        double_support_time=0.0,
    )
    assert np.isclose(r_bwd["tracking_lin_vel"], -0.80, atol=1e-5), (
        f"[{tag}] Expected backward penalty -0.80 for vel_x=-0.20, got {r_bwd['tracking_lin_vel']}"
    )

    # 3. Dwell preserves backward penalty:
    # a) Forward + dwell (dst > 0.20s): positive reward is zeroed out
    r_fwd_dwell = compute_fn(
        base_lin_vel=base_lin_forward,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        double_support_time=0.25,
    )
    assert np.isclose(r_fwd_dwell["tracking_lin_vel"], 0.0, atol=1e-5), (
        f"[{tag}] Expected forward reward zeroed out during dwell, got {r_fwd_dwell['tracking_lin_vel']}"
    )

    # b) Backward + dwell (dst > 0.20s): negative penalty MUST BE PRESERVED (min(0.0, -0.80) = -0.80)
    r_bwd_dwell = compute_fn(
        base_lin_vel=base_lin_backward,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang_vel,
        cmd_ang_vel=cmd_ang_vel,
        proj_gravity=proj_gravity,
        joint_torques=joint_torques,
        joint_vel=joint_vel,
        foot_contact=foot_contact,
        double_support_time=0.25,
    )
    assert np.isclose(r_bwd_dwell["tracking_lin_vel"], -0.80, atol=1e-5), (
        f"[{tag}] Backward penalty must be preserved during dwell! Expected -0.80, got {r_bwd_dwell['tracking_lin_vel']}"
    )


@pytest.mark.parametrize(
    "compute_fn,tag",
    [
        (root_compute_rewards, "root"),
        (rpi5_compute_rewards, "rpi5"),
        (esp32_compute_rewards, "esp32"),
    ],
)
def test_batched_backward_velocity_and_dwell(compute_fn, tag):
    """Verify batched operation computes correct double backward penalty and preserves dwell penalties."""
    base_lin = np.array(
        [
            [0.20, 0.0, 0.0],   # env 0: forward -> +0.40
            [-0.20, 0.0, 0.0],  # env 1: backward -> -0.80
            [0.20, 0.0, 0.0],   # env 2: forward + dwell -> 0.0
            [-0.20, 0.0, 0.0],  # env 3: backward + dwell -> -0.80
        ],
        dtype=np.float32,
    )
    cmd_lin = np.array([[0.20, 0.0, 0.0]] * 4, dtype=np.float32)
    base_ang = np.zeros((4, 3), dtype=np.float32)
    cmd_ang = np.zeros(4, dtype=np.float32)
    grav = np.array([[0.0, 0.0, -1.0]] * 4, dtype=np.float32)
    torques = np.zeros((4, 8), dtype=np.float32)
    j_vel = np.zeros((4, 8), dtype=np.float32)
    f_contact = np.ones((4, 4), dtype=np.float32)
    dst = np.array([0.0, 0.0, 0.25, 0.25], dtype=np.float32)

    res = compute_fn(
        base_lin_vel=base_lin,
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=cmd_ang,
        proj_gravity=grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=f_contact,
        double_support_time=dst,
    )

    expected = np.array([0.40, -0.80, 0.0, -0.80], dtype=np.float32)
    np.testing.assert_allclose(
        res["tracking_lin_vel"],
        expected,
        atol=1e-5,
        err_msg=f"[{tag}] Batched tracking_lin_vel does not match expected values",
    )


@pytest.mark.parametrize(
    "compute_fn,tag",
    [
        (root_compute_rewards, "root"),
        (rpi5_compute_rewards, "rpi5"),
        (esp32_compute_rewards, "esp32"),
    ],
)
def test_gait_and_air_time_locked_by_forward_velocity(compute_fn, tag):
    """Verify alternating foot contact (and feet_air_time for root) is rewarded
    ONLY when actively moving forward (vel_x > 0.02). Stationary or backward
    stepping receives 0.0 gait reward."""
    cmd_lin = np.array([0.25, 0.0, 0.0], dtype=np.float32)
    base_ang = np.zeros(3, dtype=np.float32)
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    torques = np.zeros(8, dtype=np.float32)
    j_vel = np.zeros(8, dtype=np.float32)
    one_foot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    air_time = np.array([0.25, 0.0], dtype=np.float32)
    touchdown = np.array([True, False], dtype=bool)

    # 1. Negative or insufficient velocity (vx <= 0.02) -> Zero gait reward
    for vx_bad in [0.0, 0.01, 0.02, -0.05, -0.20]:
        kwargs = dict(
            base_lin_vel=np.array([vx_bad, 0.0, 0.0], dtype=np.float32),
            cmd_lin_vel=cmd_lin,
            base_ang_vel=base_ang,
            cmd_ang_vel=0.0,
            proj_gravity=proj_grav,
            joint_torques=torques,
            joint_vel=j_vel,
            foot_contact=one_foot,
        )
        if True:  # All stacks support feet_air_time
            kwargs["feet_air_time"] = air_time
            kwargs["feet_touchdown"] = touchdown

        res_bad = compute_fn(**kwargs)
        assert res_bad["foot_contact"] == 0.0, (
            f"[{tag}] foot_contact must be 0.0 when vx={vx_bad} <= 0.02, got {res_bad['foot_contact']}"
        )
        if True:  # All stacks support feet_air_time
            assert res_bad["feet_air_time"] == 0.0, (
                f"[{tag}] feet_air_time must be 0.0 when vx={vx_bad} <= 0.02, got {res_bad['feet_air_time']}"
            )

    # 2. Genuine forward step (vx = 0.05 > 0.02) -> Full gait reward
    kwargs_good = dict(
        base_lin_vel=np.array([0.05, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=cmd_lin,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=one_foot,
    )
    if True:  # All stacks support feet_air_time
        kwargs_good["feet_air_time"] = air_time
        kwargs_good["feet_touchdown"] = touchdown

    res_good = compute_fn(**kwargs_good)
    assert res_good["foot_contact"] == 1.0, (
        f"[{tag}] foot_contact must be 1.0 when vx=0.05 > 0.02, got {res_good['foot_contact']}"
    )
    if True:  # All stacks support feet_air_time
        assert res_good["feet_air_time"] > 0.0, (
            f"[{tag}] feet_air_time must be > 0.0 when vx=0.05 > 0.02, got {res_good['feet_air_time']}"
        )

    # 3. Stationary command (cmd_speed <= 0.05) double support standing reward is preserved
    cmd_still = np.zeros(3, dtype=np.float32)
    two_feet = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    kwargs_stand = dict(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=cmd_still,
        base_ang_vel=base_ang,
        cmd_ang_vel=0.0,
        proj_gravity=proj_grav,
        joint_torques=torques,
        joint_vel=j_vel,
        foot_contact=two_feet,
    )
    res_stand = compute_fn(**kwargs_stand)
    assert res_stand["foot_contact"] == 0.5, (
        f"[{tag}] Stationary standing reward must be 0.5, got {res_stand['foot_contact']}"
    )


def test_mjcf_forward_axis_and_hip_pitch_kinematics():
    """Verify MuJoCo XML kinematics:
    1. Forward +X axis alignment: beak and slalom pillars lie strictly on +X.
    2. Hip pitch < 0 swings thigh & foot forward (+X).
    3. Knee pitch > 0 folds shin & foot backward (-X).
    Explaining physical kinematics why knee flexion alone without hip flexion
    causes backward stepping."""
    import mujoco

    xml_path = Path(__file__).resolve().parent.parent / "models" / "mjcf" / "microduck_8dof.xml"
    assert xml_path.exists(), f"MJCF model missing at {xml_path}"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    beak_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "beak_geom")
    pillar_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")

    base_x = data.xpos[base_id][0]
    beak_x = data.geom_xpos[beak_geom_id][0]
    pillar_x = data.geom_xpos[pillar_geom_id][0]

    # 1. Beak and pillars are in front along +X
    assert beak_x > base_x, f"Beak (x={beak_x}) must be ahead of base (x={base_x}) along +X"
    assert pillar_x > base_x, f"Slalom pillar (x={pillar_x}) must be in front of robot along +X"

    hip_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")
    knee_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_pitch")
    hip_qadr = model.jnt_qposadr[hip_jnt_id]
    knee_qadr = model.jnt_qposadr[knee_jnt_id]

    # 2. hip_pitch = -0.35 swings foot forward (+X)
    data.qpos[hip_qadr] = -0.35
    data.qpos[knee_qadr] = 0.0
    mujoco.mj_forward(model, data)
    rel_x_hip = data.xpos[foot_id][0] - data.xpos[base_id][0]
    assert rel_x_hip > 0.03, f"hip_pitch < 0 must move foot forward (+X), got {rel_x_hip:.4f}m"

    # 3. knee_pitch = +0.50 retracts foot backward (-X)
    data.qpos[hip_qadr] = 0.0
    data.qpos[knee_qadr] = 0.50
    mujoco.mj_forward(model, data)
    rel_x_knee = data.xpos[foot_id][0] - data.xpos[base_id][0]
    assert rel_x_knee < -0.02, f"knee_pitch > 0 must retract foot backward (-X), got {rel_x_knee:.4f}m"
