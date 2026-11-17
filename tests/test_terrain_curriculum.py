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
    next_stair_stage,
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

    # Stage 1: stair_step_h = -0.020 -> pure slalom, all stairs submerged top_z <= 0.0 & pillars active (1.35x)
    apply_stage(model, stage=1)
    for s_name in STAIR_GEOMS:
        assert get_top_z(s_name) <= 0.0, f"{s_name} top_z must be <= 0.0 in Stage 1"
    gid_pillar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
    assert pytest.approx(float(model.geom_pos[gid_pillar, 0]), abs=1e-4) == 0.45 * 1.35

    # Stage 2: stair_step_h = 0.008 -> steps at 8mm, 16mm, 24mm, platform at 24mm
    apply_stage(model, stage=2)
    assert pytest.approx(get_top_z("stair_step_1"), abs=1e-4) == 0.008
    assert pytest.approx(get_top_z("stair_step_2"), abs=1e-4) == 0.016
    assert pytest.approx(get_top_z("stair_step_3"), abs=1e-4) == 0.024
    assert pytest.approx(get_top_z("stair_platform"), abs=1e-4) == 0.024

    # Stage 3: stair_step_h = 0.012 -> steps at 12mm, 24mm, 36mm, platform at 36mm
    apply_stage(model, stage=3)
    assert pytest.approx(get_top_z("stair_step_1"), abs=1e-4) == 0.012
    assert pytest.approx(get_top_z("stair_step_2"), abs=1e-4) == 0.024
    assert pytest.approx(get_top_z("stair_step_3"), abs=1e-4) == 0.036
    assert pytest.approx(get_top_z("stair_platform"), abs=1e-4) == 0.036


def test_stage1_slalom_submerges_stairs():
    """Verify that Stage 1 submerges stairs below floor (top_z <= 0.0) while keeping pillars active at 1.35x."""
    model = mujoco.MjModel.from_xml_path("models/mjcf/microduck_8dof.xml")
    apply_stage(model, stage=1)
    for s_name in STAIR_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, s_name)
        assert gid != -1
        top_z = float(model.geom_pos[gid, 2] + model.geom_size[gid, 2])
        assert top_z <= 0.0, f"{s_name} top surface {top_z:.4f} is not submerged below floor in Stage 1"

    gid_pillar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
    assert pytest.approx(float(model.geom_pos[gid_pillar, 0]), abs=1e-4) == 0.45 * 1.35


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
        stage, hold = next_stage(stage, ep_len_mean=450.0, rew_lin_vel_ratio=0.85, hold_count=hold, hold_iters=10, mean_distance_x=1.8)
        assert stage == 0

    assert hold == 9

    # 10th iteration satisfies hold -> advances to stage 1
    stage, hold = next_stage(stage, ep_len_mean=450.0, rew_lin_vel_ratio=0.85, hold_count=hold, hold_iters=10, mean_distance_x=1.8)
    assert stage == 1
    assert hold == 0


def test_next_stage_never_regresses():
    """Verify that next_stage never regresses to a lower stage when performance drops."""
    stage = 2
    hold = 5

    # Significant performance drop
    stage, hold = next_stage(stage, ep_len_mean=150.0, rew_lin_vel_ratio=0.2, hold_count=hold, hold_iters=10, mean_distance_x=1.8)
    assert stage == 2
    assert hold == 0  # Hold count resets, but stage never decreases


def test_stage0_unlock_threshold_0_8m():
    """Verify that Stage 0 unlocks when mean_distance_x >= 0.8m and ep_len >= 400.

    Ensures that when robot has high ep_len (456.9) but distance is only 0.70m (< 0.8m),
    it does not advance. Once distance crosses 0.8m, it accumulates hold and advances to Stage 1.
    """
    stage = 0
    hold = 0

    # Under-distance: ep_len is 456.9 but dist is 0.70m -> fails to advance
    for _ in range(5):
        stage, hold = next_stage(
            stage=stage,
            ep_len_mean=456.9,
            rew_lin_vel_ratio=0.85,
            hold_count=hold,
            hold_iters=5,
            mean_distance_x=0.70,
            success_rate=0.80,
        )
        assert stage == 0
        assert hold == 0

    # Over-distance: ep_len is 456.9 and dist reaches 0.82m (>= 0.8m) -> accumulates hold and advances
    for _ in range(4):
        stage, hold = next_stage(
            stage=stage,
            ep_len_mean=456.9,
            rew_lin_vel_ratio=0.85,
            hold_count=hold,
            hold_iters=5,
            mean_distance_x=0.82,
            success_rate=0.80,
        )
        assert stage == 0
    assert hold == 4

    stage, hold = next_stage(
        stage=stage,
        ep_len_mean=456.9,
        rew_lin_vel_ratio=0.85,
        hold_count=hold,
        hold_iters=5,
        mean_distance_x=0.82,
        success_rate=0.80,
    )
    assert stage == 1
    assert hold == 0


def test_default_hold_iters_is_five():
    """Verify default hold_iters is 5 in next_stage and next_stair_stage."""
    import inspect

    sig_stage = inspect.signature(next_stage)
    assert sig_stage.parameters["hold_iters"].default == 5

    sig_stair = inspect.signature(next_stair_stage)
    assert sig_stair.parameters["hold_iters"].default == 5

    stage, hold = 0, 0
    for _ in range(4):
        stage, hold = next_stage(
            stage=stage,
            ep_len_mean=450.0,
            rew_lin_vel_ratio=0.85,
            hold_count=hold,
            mean_distance_x=1.8,
        )
        assert stage == 0
    assert hold == 4

    # 5th iteration satisfies default hold_iters (5) -> advances to stage 1
    stage, hold = next_stage(
        stage=stage,
        ep_len_mean=450.0,
        rew_lin_vel_ratio=0.85,
        hold_count=hold,
        mean_distance_x=1.8,
    )
    assert stage == 1
    assert hold == 0


def test_stage3_min_distance_threshold_and_fallback():
    """Verify Stage 3 defines >= 1.8m clearance and out-of-range stages do not regress to 0.8m."""
    from training.terrain_curriculum import STAGE_MIN_DISTANCE_X

    assert STAGE_MIN_DISTANCE_X[3] == pytest.approx(1.8)
    assert STAGE_MIN_DISTANCE_X.get(3) >= STAGE_MIN_DISTANCE_X[2]


def test_set_stage_clears_completed_buffers_only_on_change():
    """Verify ParallelSimRunner.set_stage clears completed windows on stage transition only."""
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(num_envs=2, stage=1, num_threads=1)
    try:
        runner.completed_ep_lens.extend([480, 490])
        runner.completed_distances_x.extend([1.3, 1.4])
        runner.completed_distances_y.extend([0.05, -0.02])

        # Idempotent call with same stage must preserve buffers
        runner.set_stage(1)
        assert list(runner.completed_ep_lens) == [480, 490]
        assert list(runner.completed_distances_x) == pytest.approx([1.3, 1.4])

        # Advancing to stage 2 must clear stale stage-1 history
        runner.set_stage(2)
        assert len(runner.completed_ep_lens) == 0
        assert len(runner.completed_distances_x) == 0
        assert len(runner.completed_distances_y) == 0
    finally:
        runner.close()
