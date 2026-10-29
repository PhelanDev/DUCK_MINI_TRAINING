"""Tests for terrain and obstacle curriculum management."""

from pathlib import Path
import sys
import numpy as np
import pytest
import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.terrain_curriculum import (
    STAGES,
    Stage,
    apply_stage,
    next_stage,
)


def test_stage0_hides_hurdles_below_floor():
    """Verify that in Stage 0, top surfaces of all hurdles are strictly at or below floor level (z <= 0)."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")
    apply_stage(model, stage=0)

    for h_name in ["hurdle_1", "hurdle_2", "hurdle_back"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, h_name)
        assert gid != -1, f"Missing geom {h_name} in model"
        pos_z = model.geom_pos[gid, 2]
        half_z = model.geom_size[gid, 2]
        top_z = pos_z + half_z
        assert top_z <= 0.0, f"{h_name} top surface {top_z:.4f} is not submerged below floor in Stage 0"


def test_apply_stage_scales_pillars():
    """Verify that apply_stage correctly scales pillar XY positions relative to XML baseline."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")
    gid_front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_front")
    assert gid_front != -1

    # Stage 0: pillar_scale = 2.0 -> front x = 0.85 * 2.0 = 1.70
    apply_stage(model, stage=0)
    assert pytest.approx(model.geom_pos[gid_front, 0], abs=1e-4) == 0.85 * 2.00

    # Stage 1: pillar_scale = 1.30 -> front x = 0.85 * 1.30 = 1.105
    apply_stage(model, stage=1)
    assert pytest.approx(model.geom_pos[gid_front, 0], abs=1e-4) == 0.85 * 1.30

    # Stage 2: pillar_scale = 1.00 -> front x = 0.85
    apply_stage(model, stage=2)
    assert pytest.approx(model.geom_pos[gid_front, 0], abs=1e-4) == 0.85


def test_apply_stage_idempotent():
    """Verify that calling apply_stage repeatedly does not multiply positions cumulatively."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")
    gid_front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_front")

    apply_stage(model, stage=1)
    x1 = float(model.geom_pos[gid_front, 0])

    apply_stage(model, stage=1)
    x2 = float(model.geom_pos[gid_front, 0])

    assert pytest.approx(x1) == x2


def test_next_stage_requires_hold():
    """Verify that advancing to the next stage requires satisfying threshold for hold_iters consecutive steps."""
    stage = 0
    hold = 0

    # 9 consecutive good iterations (ep_len=450, vel_ratio=0.85)
    for _ in range(9):
        stage, hold = next_stage(stage, ep_len_mean=450.0, rew_lin_vel_ratio=0.85, hold_count=hold, hold_iters=10)
        assert stage == 0

    assert hold == 9

    # 10th iteration satisfies hold -> advances to stage 1
    stage, hold = next_stage(stage, ep_len_mean=450.0, rew_lin_vel_ratio=0.85, hold_count=hold, hold_iters=10)
    assert stage == 1
    assert hold == 0


def test_next_stage_never_regresses():
    """Verify that next_stage never regresses to a lower stage when performance drops."""
    stage = 2
    hold = 5

    # Significant performance drop
    stage, hold = next_stage(stage, ep_len_mean=150.0, rew_lin_vel_ratio=0.2, hold_count=hold, hold_iters=10)
    assert stage == 2
    assert hold == 0  # Hold count resets, but stage never decreases

