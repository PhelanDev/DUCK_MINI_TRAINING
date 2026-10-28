"""Tests for scripts/verify_model.py physical model verification script."""

import os
import subprocess
import sys
import pytest


def test_verify_script_execution():
    """Verify that verify_model.py runs successfully and reports success."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "verify_model.py")
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed with code {res.returncode}:\n{res.stderr}"
    assert "Verification SUCCESS" in res.stdout
    assert "Total Mass:" in res.stdout


def test_verify_script_cli_custom_paths():
    """Verify that CLI flags for --mjcf and --urdf work."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "verify_model.py")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mjcf_path = os.path.join(root_dir, "models", "mjcf", "microduck_8dof.xml")
    urdf_path = os.path.join(root_dir, "models", "urdf", "microduck_8dof.urdf")

    res = subprocess.run(
        [sys.executable, script_path, "--mjcf", mjcf_path, "--urdf", urdf_path],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Verification SUCCESS" in res.stdout


def test_verify_script_cli_invalid_path():
    """Verify that providing an invalid path causes the script to fail with non-zero exit code."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "verify_model.py")
    res = subprocess.run(
        [sys.executable, script_path, "--mjcf", "nonexistent_model.xml"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0


def test_verify_script_functions():
    """Verify unit-level functions of verify_model."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts.verify_model import (
        verify_joint_consistency,
        verify_mass_and_inertials,
        verify_center_of_mass,
        run_simulation_check,
    )

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mjcf_path = os.path.join(root_dir, "models", "mjcf", "microduck_8dof.xml")
    urdf_path = os.path.join(root_dir, "models", "urdf", "microduck_8dof.urdf")

    # Joint consistency
    joint_results = verify_joint_consistency(mjcf_path, urdf_path)
    assert len(joint_results) == 8
    for j_name, data in joint_results.items():
        assert data["match"] is True

    # Mass & inertia
    mass_info = verify_mass_and_inertials(urdf_path)
    assert 0.70 <= mass_info["total_mass"] <= 0.85
    assert mass_info["all_inertias_valid"] is True

    # CoM check
    com_info = verify_center_of_mass(mjcf_path)
    assert com_info["com_low_and_centered"] is True

    # Simulation check
    sim_ok = run_simulation_check(mjcf_path, urdf_path, num_steps=50)
    assert sim_ok is True
