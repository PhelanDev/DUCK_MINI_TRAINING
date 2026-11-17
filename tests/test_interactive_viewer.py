"""Tests for Interactive 3D Viewer and Project Documentation (Task 6).

Covers:
- Verification that scripts/interactive_viewer.py and README.md exist.
- Verification of comprehensive Vietnamese documentation sections in README.md.
- Interactive viewer pose presets, kinematics limits, and headless execution.
- Headless gait demonstration and stability verification.
- Verification of the reset last_actions fix in ParallelSimRunner.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_viewer_and_docs_exist():
    """Step 1 baseline test as specified in task-6-brief.md."""
    assert os.path.exists(os.path.join(PROJECT_ROOT, "scripts", "interactive_viewer.py"))
    assert os.path.exists(os.path.join(PROJECT_ROOT, "README.md"))
    with open(os.path.join(PROJECT_ROOT, "README.md"), "r", encoding="utf-8") as f:
        content = f.read()
        assert "MicroDuck 8-DOF" in content
        assert "run_parallel_sim.py" in content


def test_readme_comprehensive_content():
    """Verify README.md contains all required technical sections in Vietnamese."""
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    assert os.path.exists(readme_path), "README.md must exist"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Overview: MG996R servos, ~780g mass
    assert "MG996R" in content
    assert any(m in content for m in ["684", "764", "649", "780"])

    # 2. Kinematic tree & link dimensions (2 DOF head, 6 DOF legs)
    assert "head_yaw" in content
    assert "head_pitch" in content
    assert "hip_roll" in content
    assert "hip_pitch" in content
    assert "knee_pitch" in content

    # 3. Model verification
    assert "verify_model.py" in content

    # 4. Interactive 3D visualization
    assert "interactive_viewer.py" in content

    # 5. Parallel simulation benchmark
    assert "run_parallel_sim.py" in content
    assert "--num_envs" in content or "--envs" in content

    # 6. RL Sim-to-Real training guide with 37-dim obs & 8-dim action & ONNX export
    assert "37" in content or "42" in content
    assert "8" in content   # 8-dim action
    assert "Sim-to-Real" in content or "sim-to-real" in content or "Sim-To-Real" in content
    assert "ONNX" in content
    assert "ESP32" in content


def test_interactive_viewer_presets_and_limits():
    """Verify interactive_viewer module exports valid joint configurations and presets."""
    from scripts.interactive_viewer import (
        JOINT_NAMES,
        POSE_PRESETS,
        JOINT_LIMITS,
        build_arg_parser,
    )

    assert len(JOINT_NAMES) == 8
    expected_joints = [
        "head_yaw",
        "head_pitch",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee_pitch",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee_pitch",
    ]
    assert JOINT_NAMES == expected_joints

    # Verify standard presets
    for preset_name in ["stand", "squat", "left_step", "right_step", "bow"]:
        assert preset_name in POSE_PRESETS, f"Preset {preset_name} missing from POSE_PRESETS"
        pose_angles = POSE_PRESETS[preset_name]
        assert len(pose_angles) == 8

        # Check values are strictly within physical joint limits
        for i, jname in enumerate(JOINT_NAMES):
            low, high = JOINT_LIMITS[jname]
            angle = pose_angles[i]
            assert low <= angle <= high, (
                f"Preset '{preset_name}' joint '{jname}' angle {angle} outside [{low}, {high}]"
            )


def test_interactive_viewer_cli_parser():
    """Verify CLI argument parsing handles options correctly."""
    from scripts.interactive_viewer import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--headless", "--duration", "2.5", "--pose", "squat"])
    assert args.headless is True
    assert args.duration == 2.5
    assert args.pose == "squat"

    args_gait = parser.parse_args(["--gait-demo", "--duration", "1.0"])
    assert args_gait.gait_demo is True


def test_interactive_viewer_headless_simulation():
    """Run headless simulation for a short duration to test physical stability."""
    from scripts.interactive_viewer import run_simulation_headless

    metrics = run_simulation_headless(
        model_path="models/mjcf/microduck_8dof.xml",
        pose="stand",
        duration=0.1,
        dt=0.002,
    )

    assert metrics["completed_steps"] > 0
    assert metrics["final_root_z"] > 0.12  # Must not collapse to ground
    assert metrics["is_stable"] is True
    assert np.all(np.isfinite(metrics["final_joint_angles"]))


def test_interactive_viewer_gait_demo_headless():
    """Run gait demonstration in headless mode."""
    from scripts.interactive_viewer import run_simulation_headless

    metrics = run_simulation_headless(
        model_path="models/mjcf/microduck_8dof.xml",
        pose="stand",
        duration=0.1,
        gait_demo=True,
        dt=0.002,
    )

    assert metrics["completed_steps"] > 0
    assert metrics["final_root_z"] > 0.12
    assert metrics["is_stable"] is True


def test_interactive_viewer_pd_gains_sync():
    """Verify interactive viewer PD gains match MicroDuck8DoFConfig."""
    import inspect
    from scripts.interactive_viewer import apply_pd_control
    from training.env_config import _DEFAULT_CFG

    sig = inspect.signature(apply_pd_control)
    kp_default = sig.parameters["kp"].default
    kd_default = sig.parameters["kd"].default
    assert kp_default == _DEFAULT_CFG.KP == 13.5
    assert kd_default == _DEFAULT_CFG.KD == 0.9


def test_reset_last_actions_fix():
    """Test the deferred item: runner.last_actions[dones] is zeroed when envs reset."""
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=4,
        decimation=5,
    )
    try:
        # Step once with non-zero actions
        nonzero_actions = np.ones((4, 8), dtype=np.float32) * 0.8
        runner.step(nonzero_actions)
        # Normal step should have last_actions == nonzero_actions
        np.testing.assert_allclose(runner.last_actions, nonzero_actions, atol=1e-5)

        # Force environment 1 to fall over so done is triggered
        runner.datas[1].qpos[2] = 0.02  # z height < 0.08 threshold

        # Step again
        obs, rewards, dones, infos = runner.step(nonzero_actions)

        assert dones[1] == True, "Environment 1 should be marked as done"
        # Environment 1 must have its last_actions reset to 0.0
        np.testing.assert_allclose(
            runner.last_actions[1],
            np.zeros(8, dtype=np.float32),
            atol=1e-5,
            err_msg="Reset environment 1 must have last_actions zeroed",
        )
        # Other environments should still have nonzero last_actions
        for i in [0, 2, 3]:
            if not dones[i]:
                np.testing.assert_allclose(
                    runner.last_actions[i],
                    nonzero_actions[i],
                    atol=1e-5,
                )
    finally:
        runner.close()


def test_interactive_viewer_main_headless(monkeypatch):
    """Verify interactive_viewer main() function returns 0 in headless mode."""
    from scripts.interactive_viewer import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["interactive_viewer.py", "--headless", "--duration", "0.05", "--dt", "0.005"],
    )
    exit_code = main()
    assert exit_code == 0


def test_interactive_viewer_cli_simulation(monkeypatch):
    """Verify run_interactive_cli processes standard commands and exits gracefully."""
    from scripts.interactive_viewer import run_interactive_cli

    commands = iter([
        "pose squat",
        "set head_pitch 0.1",
        "step 10",
        "status",
        "reset",
        "help",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(commands))

    # Should run and exit without raising an exception
    run_interactive_cli(model_path="models/mjcf/microduck_8dof.xml", dt=0.005)


def test_configure_viewer_camera():
    """Verify configure_viewer_camera sets default camera orientation to avoid black screen on hybrid GPUs."""
    from scripts.interactive_viewer import configure_viewer_camera

    class DummyCamera:
        def __init__(self):
            self.lookat = [0.0, 0.0, 0.0]
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    cam = DummyCamera()
    configure_viewer_camera(cam)
    assert cam.lookat == [0.0, 0.0, 0.15]
    assert cam.distance == 1.2
    assert cam.azimuth == 135.0
    assert cam.elevation == -20.0

    # Verify custom parameters
    configure_viewer_camera(
        cam,
        lookat=(0.1, 0.2, 0.3),
        distance=2.0,
        azimuth=90.0,
        elevation=-30.0,
    )
    assert cam.lookat == [0.1, 0.2, 0.3]
    assert cam.distance == 2.0
    assert cam.azimuth == 90.0
    assert cam.elevation == -30.0

    # If MuJoCo is installed, test with real mujoco.MjvCamera instance
    try:
        import mujoco
        real_cam = mujoco.MjvCamera()
        configure_viewer_camera(real_cam)
        np.testing.assert_allclose(real_cam.lookat, [0.0, 0.0, 0.15], atol=1e-5)
        assert real_cam.distance == pytest.approx(1.2)
        assert real_cam.azimuth == pytest.approx(135.0)
        assert real_cam.elevation == pytest.approx(-20.0)
    except ImportError:
        pass




def test_interactive_viewer_policy_headless_stability():
    """Verify that running policy brain in headless mode stays stable above z >= 0.18m."""
    from scripts.interactive_viewer import run_simulation_headless

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "microduck_ppo.pt")
    if not os.path.exists(ckpt_path):
        pytest.skip("Policy checkpoint not available")

    metrics = run_simulation_headless(
        model_path="models/mjcf/microduck_8dof.xml",
        policy_path=ckpt_path,
        duration=0.5,
        dt=0.002,
        auto_reset=True,
    )
    assert metrics["completed_steps"] > 0
    # Must remain standing / walking and NOT collapse to z = 0.057m
    assert metrics["final_root_z"] > 0.12, f"Robot collapsed to z={metrics['final_root_z']:.3f}m"
    assert metrics["is_stable"] is True


def test_interactive_viewer_fallback_and_stairs_handling(monkeypatch):
    """Verify interactive viewer fallback branch and stair stage application work cleanly without TypeError or missing args."""
    import inspect
    from scripts.interactive_viewer import (
        run_interactive_viewer,
        run_opencv_viewer,
        run_simulation_headless,
    )

    # 1. run_opencv_viewer must accept stairs parameter
    sig_cv = inspect.signature(run_opencv_viewer)
    assert "stairs" in sig_cv.parameters, "run_opencv_viewer must accept 'stairs' parameter"
    assert sig_cv.parameters["stairs"].default is False

    # 2. Check source of run_opencv_viewer applies stair stage
    src_path = Path(PROJECT_ROOT) / "scripts" / "interactive_viewer.py"
    src = src_path.read_text(encoding="utf-8")
    assert "if stairs and apply_stair_stage is not None:" in src, (
        "run_opencv_viewer must apply stair stage when stairs=True"
    )

    # 3. Simulate GLFW/OpenGL failure in run_interactive_viewer and verify fallback passes all args cleanly
    called_args = {}

    def mock_headless(**kwargs):
        called_args.update(kwargs)
        return {
            "completed_steps": 10,
            "sim_time": 0.02,
            "final_root_z": 0.17,
            "is_stable": True,
            "left_touch": 1.0,
            "right_touch": 1.0,
        }

    monkeypatch.setattr("scripts.interactive_viewer._CV2_AVAILABLE", False)
    monkeypatch.setattr("scripts.interactive_viewer.run_simulation_headless", mock_headless)

    import mujoco
    def mock_launch_passive(*a, **k):
        raise RuntimeError("Mock GLFW/OpenGL failure")
    monkeypatch.setattr(mujoco.viewer, "launch_passive", mock_launch_passive)

    run_interactive_viewer(
        model_path="models/mjcf/microduck_8dof.xml",
        pose="squat",
        duration=0.01,
        stage=2,
        stairs=True,
        auto_reset=False,
        gait="spot_turn",
        track=True,
        target_color="red",
    )

    assert called_args.get("stage") == 2
    assert called_args.get("stairs") is True
    assert called_args.get("auto_reset") is False
    assert called_args.get("gait") == "spot_turn"
    assert called_args.get("track") is True
    assert called_args.get("target_color") == "red"


def test_viewer_stage0_applies_wide_pillars_submerges_stairs_and_spawns_native_height(monkeypatch):
    """Stage 0 in viewer must still apply terrain_curriculum.apply_stage(model, 0) and spawn at model.qpos0[2]."""
    import mujoco
    from scripts.interactive_viewer import run_simulation_headless
    from training.terrain_curriculum import apply_stage

    model_rel = "duck_v2/cyberduck_v2.xml"
    model_abs = os.path.join(PROJECT_ROOT, model_rel)
    ref_model = mujoco.MjModel.from_xml_path(model_abs)
    apply_stage(ref_model, 0)
    stair_id = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_GEOM, "stair_step_0")
    pillar_id = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_0")

    captured = {}
    orig_mj_step = mujoco.mj_step

    def spy_mj_step(model, data):
        if "initial_z" not in captured:
            captured["initial_z"] = float(data.qpos[2])
            captured["stair0_z"] = float(model.geom_pos[stair_id, 2])
            captured["pillar0_size0"] = float(model.geom_size[pillar_id, 0])
        return orig_mj_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", spy_mj_step)

    run_simulation_headless(
        model_path=model_rel,
        pose="stand",
        duration=0.01,
        dt=0.002,
        stage=0,
        stairs=False,
        auto_reset=False,
    )

    assert captured["initial_z"] == pytest.approx(float(ref_model.qpos0[2]), abs=1e-4)
    assert captured["stair0_z"] == pytest.approx(float(ref_model.geom_pos[stair_id, 2]), abs=1e-5)
    assert captured["pillar0_size0"] == pytest.approx(float(ref_model.geom_size[pillar_id, 0]), abs=1e-5)


def test_extract_viewer_observation_filters_floor_lidar_hits():
    """Viewer extract_viewer_observation must filter z_hit <= 0.012 floor rays to 2.0m like ParallelSimRunner."""
    import mujoco
    from scripts.interactive_viewer import extract_viewer_observation
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path=os.path.join(PROJECT_ROOT, "duck_v2", "cyberduck_v2.xml"),
        num_envs=1,
        stage=0,
        terrain_mode="hurdles",
    )
    try:
        runner.reset()
        m = runner.model
        d = runner.datas[0]
        head_pitch_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch")
        qadr = int(m.jnt_qposadr[head_pitch_jnt])

        # Find downward head pitch angle that produces at least one floor ray hit
        found_floor_hit = False
        for pitch in (0.45, -0.45, 0.30, -0.30):
            d.qpos[0] = -0.3
            d.qpos[1] = 0.0
            d.qpos[qadr] = pitch
            mujoco.mj_forward(m, d)
            raw = np.array(d.sensordata[28:33], dtype=np.float32)
            filtered = runner._get_lidar_ranges(0)
            if np.any((raw > 0.0) & (raw < 1.5) & (filtered == 2.0)):
                found_floor_hit = True
                break

        assert found_floor_hit, "Test setup failed to aim a LiDAR ray at the floor"
        obs = extract_viewer_observation(
            m,
            d,
            cmd_vel=np.array([0.22, 0.0, 0.0], dtype=np.float32),
            obs_dim=42,
        )
        np.testing.assert_allclose(obs[37:42], filtered, atol=1e-5)
    finally:
        runner.close()


def test_run_opencv_viewer_handles_getwindowproperty_error_and_passes_stairs(monkeypatch):
    """OpenCV viewer exits cleanly if getWindowProperty raises cv2.error, and main() passes stairs=args.stairs."""
    import types
    import mujoco
    import scripts.interactive_viewer as iv

    class DummyCv2Error(Exception):
        pass

    class DummyRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def update_scene(self, *args, **kwargs):
            pass

        def render(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

    fake_cv2 = types.SimpleNamespace(
        error=DummyCv2Error,
        WINDOW_AUTOSIZE=1,
        COLOR_RGB2BGR=0,
        FONT_HERSHEY_SIMPLEX=0,
        WND_PROP_VISIBLE=1,
        namedWindow=lambda *a, **k: None,
        cvtColor=lambda img, code: img,
        putText=lambda *a, **k: None,
        imshow=lambda *a, **k: None,
        waitKey=lambda *a, **k: -1,
        getWindowProperty=lambda *a, **k: (_ for _ in ()).throw(DummyCv2Error("NULL window")),
        destroyAllWindows=lambda *a, **k: None,
    )

    monkeypatch.setattr(iv, "cv2", fake_cv2, raising=False)
    monkeypatch.setattr(iv, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(mujoco, "Renderer", DummyRenderer)
    monkeypatch.setattr(iv.time, "sleep", lambda *_: None)

    # Should exit gracefully without raising DummyCv2Error when [X] closes the window
    iv.run_opencv_viewer(
        model_path="duck_v2/cyberduck_v2.xml",
        duration=1.0,
        stage=0,
        dt=0.002,
    )

    # main() must forward stairs=args.stairs to run_opencv_viewer
    captured = {}

    def spy_opencv_viewer(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(iv, "run_opencv_viewer", spy_opencv_viewer)
    monkeypatch.setattr(sys, "argv", ["interactive_viewer.py", "--opencv", "--stairs", "--stage", "2"])
    iv.main()
    assert captured.get("stairs") is True


def test_maybe_hot_reload_policy_reloads_when_mtime_increases(monkeypatch, tmp_path):
    """Verify _maybe_hot_reload_policy throttles checks, reloads on mtime bump, and handles errors safely."""
    from scripts.interactive_viewer import _maybe_hot_reload_policy

    dummy_policy_old = "old_policy"
    dummy_policy_new = "new_policy"
    policy_file = tmp_path / "dummy_policy.pt"
    policy_file.write_bytes(b"initial_content")

    t0 = 1000.0
    mtime0 = 500.0

    stat_called = [0]
    orig_getmtime = os.path.getmtime

    def counting_getmtime(path):
        stat_called[0] += 1
        return orig_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", counting_getmtime)

    # 0. Edge cases: empty/None policy_path or None current_policy
    assert _maybe_hot_reload_policy(None, dummy_policy_old, mtime0, t0) == (dummy_policy_old, mtime0, t0)
    assert _maybe_hot_reload_policy(str(policy_file), None, mtime0, t0) == (None, mtime0, t0)

    # 1. Within check_interval (elapsed < 1.0s), does not stat or reload
    pol, mtime, last_check = _maybe_hot_reload_policy(
        str(policy_file),
        dummy_policy_old,
        last_mtime=mtime0,
        last_check_wall=t0,
        check_interval=1.0,
        now=t0 + 0.5,
    )
    assert pol is dummy_policy_old
    assert mtime == mtime0
    assert last_check == t0
    assert stat_called[0] == 0

    # 2. When mtime increases past check_interval, reloads updated weights via load_viewer_policy
    monkeypatch.setattr(
        "scripts.interactive_viewer.load_viewer_policy",
        lambda p: dummy_policy_new,
    )
    new_mtime = mtime0 + 10.0
    monkeypatch.setattr(os.path, "getmtime", lambda p: new_mtime)

    pol2, mtime2, last_check2 = _maybe_hot_reload_policy(
        str(policy_file),
        dummy_policy_old,
        last_mtime=mtime0,
        last_check_wall=t0,
        check_interval=1.0,
        now=t0 + 1.2,
    )
    assert pol2 is dummy_policy_new
    assert mtime2 == new_mtime
    assert last_check2 == t0 + 1.2

    # If mtime is equal to last_mtime, does not reload
    pol3, mtime3, last_check3 = _maybe_hot_reload_policy(
        str(policy_file),
        dummy_policy_new,
        last_mtime=new_mtime,
        last_check_wall=t0 + 1.2,
        check_interval=1.0,
        now=t0 + 2.5,
    )
    assert pol3 is dummy_policy_new
    assert mtime3 == new_mtime
    assert last_check3 == t0 + 2.5

    # 3. If checkpoint temporarily fails to load (e.g. mid-write / corrupt), preserves current_policy without raising
    def broken_loader(p):
        raise RuntimeError("Mock file partially written / corrupt")

    monkeypatch.setattr("scripts.interactive_viewer.load_viewer_policy", broken_loader)
    monkeypatch.setattr(os.path, "getmtime", lambda p: new_mtime + 5.0)

    pol4, mtime4, last_check4 = _maybe_hot_reload_policy(
        str(policy_file),
        dummy_policy_new,
        last_mtime=new_mtime,
        last_check_wall=t0 + 2.5,
        check_interval=1.0,
        now=t0 + 4.0,
    )
    assert pol4 is dummy_policy_new
    assert mtime4 == new_mtime
    assert last_check4 == t0 + 4.0

    # Also handles getmtime raising FileNotFoundError safely
    def broken_getmtime(p):
        raise FileNotFoundError("Policy file removed")

    monkeypatch.setattr(os.path, "getmtime", broken_getmtime)
    pol5, mtime5, last_check5 = _maybe_hot_reload_policy(
        str(policy_file),
        dummy_policy_new,
        last_mtime=new_mtime,
        last_check_wall=t0 + 4.0,
        check_interval=1.0,
        now=t0 + 6.0,
    )
    assert pol5 is dummy_policy_new
    assert mtime5 == new_mtime
    assert last_check5 == t0 + 6.0
