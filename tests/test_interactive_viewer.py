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
    assert "780" in content or "780g" in content or "780 g" in content

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
    assert "--num_envs" in content

    # 6. RL Sim-to-Real training guide with 37-dim obs & 8-dim action & ONNX export
    assert "37" in content  # 37-dim observation
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
    assert kp_default == _DEFAULT_CFG.KP == 25.0
    assert kd_default == _DEFAULT_CFG.KD == 0.8


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
