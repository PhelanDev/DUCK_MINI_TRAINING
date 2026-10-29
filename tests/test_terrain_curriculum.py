"""Tests for terrain and obstacle curriculum management (Slalom Pillars & Staircase)."""

from pathlib import Path
import sys
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

STAIR_GEOMS = ["stair_step_1", "stair_step_2", "stair_step_3", "stair_platform"]
PILLAR_GEOMS = [
    "pillar_l1_left",
    "pillar_l1_right",
    "pillar_l2_center",
    "pillar_l3_left",
    "pillar_l3_right",
    "pillar_back",
]


def test_stage_dataclass_fields():
    """Verify that Stage dataclass contains the exact specified fields without obsolete hurdle fields."""
    s = STAGES[0]
    assert hasattr(s, "stage_id")
    assert hasattr(s, "pillar_scale")
    assert hasattr(s, "stair_step_h")
    assert hasattr(s, "description")
    assert not hasattr(s, "hurdle_z")
    assert len(STAGES) == 4


def test_stage0_submerges_stairs_below_floor():
    """Verify that in Stage 0, top surfaces of all stairs are strictly at or below floor level (z <= 0)."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")
    apply_stage(model, stage=0)

    for s_name in STAIR_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, s_name)
        assert gid != -1, f"Missing geom {s_name} in model"
        pos_z = model.geom_pos[gid, 2]
        half_z = model.geom_size[gid, 2]
        top_z = pos_z + half_z
        assert top_z <= 0.0, f"{s_name} top surface {top_z:.4f} is not submerged below floor in Stage 0"


def test_apply_stage_scales_slalom_pillars():
    """Verify that apply_stage correctly scales slalom pillar XY positions across curriculum stages."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")
    gid_l1_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
    gid_l2_center = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l2_center")
    assert gid_l1_left != -1
    assert gid_l2_center != -1

    # Stage 0: pillar_scale = 1.80
    apply_stage(model, stage=0)
    assert pytest.approx(model.geom_pos[gid_l1_left, 0], abs=1e-4) == 0.45 * 1.80
    assert pytest.approx(model.geom_pos[gid_l1_left, 1], abs=1e-4) == 0.35 * 1.80
    assert pytest.approx(model.geom_pos[gid_l2_center, 0], abs=1e-4) == 0.80 * 1.80
    assert pytest.approx(model.geom_pos[gid_l2_center, 1], abs=1e-4) == 0.08 * 1.80

    # Stage 1: pillar_scale = 1.35
    apply_stage(model, stage=1)
    assert pytest.approx(model.geom_pos[gid_l1_left, 0], abs=1e-4) == 0.45 * 1.35
    assert pytest.approx(model.geom_pos[gid_l1_left, 1], abs=1e-4) == 0.35 * 1.35

    # Stage 2: pillar_scale = 1.10
    apply_stage(model, stage=2)
    assert pytest.approx(model.geom_pos[gid_l1_left, 0], abs=1e-4) == 0.45 * 1.10
    assert pytest.approx(model.geom_pos[gid_l1_left, 1], abs=1e-4) == 0.35 * 1.10

    # Stage 3: pillar_scale = 1.00
    apply_stage(model, stage=3)
    assert pytest.approx(model.geom_pos[gid_l1_left, 0], abs=1e-4) == 0.45
    assert pytest.approx(model.geom_pos[gid_l1_left, 1], abs=1e-4) == 0.35


def test_apply_stage_stair_heights():
    """Verify that apply_stage sets correct step elevation heights across stages."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")

    def get_top_z(name: str) -> float:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid != -1, f"Missing geom {name}"
        return float(model.geom_pos[gid, 2] + model.geom_size[gid, 2])

    # Stage 1: stair_step_h = 0.005 -> steps at 5mm, 10mm, 15mm, platform at 15mm
    apply_stage(model, stage=1)
    assert pytest.approx(get_top_z("stair_step_1"), abs=1e-4) == 0.005
    assert pytest.approx(get_top_z("stair_step_2"), abs=1e-4) == 0.010
    assert pytest.approx(get_top_z("stair_step_3"), abs=1e-4) == 0.015
    assert pytest.approx(get_top_z("stair_platform"), abs=1e-4) == 0.015

    # Stage 2: stair_step_h = 0.010 -> steps at 10mm, 20mm, 30mm, platform at 30mm
    apply_stage(model, stage=2)
    assert pytest.approx(get_top_z("stair_step_1"), abs=1e-4) == 0.010
    assert pytest.approx(get_top_z("stair_step_2"), abs=1e-4) == 0.020
    assert pytest.approx(get_top_z("stair_step_3"), abs=1e-4) == 0.030
    assert pytest.approx(get_top_z("stair_platform"), abs=1e-4) == 0.030

    # Stage 3: stair_step_h = 0.015 -> steps at 15mm, 30mm, 45mm, platform at 45mm
    apply_stage(model, stage=3)
    assert pytest.approx(get_top_z("stair_step_1"), abs=1e-4) == 0.015
    assert pytest.approx(get_top_z("stair_step_2"), abs=1e-4) == 0.030
    assert pytest.approx(get_top_z("stair_step_3"), abs=1e-4) == 0.045
    assert pytest.approx(get_top_z("stair_platform"), abs=1e-4) == 0.045


def test_apply_stage_idempotent():
    """Verify that calling apply_stage repeatedly does not multiply positions cumulatively."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")
    gid_pillar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
    gid_stair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stair_step_1")

    apply_stage(model, stage=1)
    x1 = float(model.geom_pos[gid_pillar, 0])
    z1 = float(model.geom_pos[gid_stair, 2])

    apply_stage(model, stage=1)
    x2 = float(model.geom_pos[gid_pillar, 0])
    z2 = float(model.geom_pos[gid_stair, 2])

    assert pytest.approx(x1) == x2
    assert pytest.approx(z1) == z2


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
