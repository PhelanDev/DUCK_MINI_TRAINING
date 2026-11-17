"""Tests for bridge stair climbing curriculum, edge-fall detection, and training pipeline."""

from pathlib import Path
import sys
import numpy as np
import pytest
import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.terrain_curriculum import apply_stair_stage, STAIR_STAGES, next_stair_stage
from training.run_parallel_sim import ParallelSimRunner
from scripts.train_ppo import train_ppo


def test_stair_geoms_exist_in_mjcf():
    """Verify that stair geoms are present in the MuJoCo model."""
    model_path = REPO_ROOT / "models" / "mjcf" / "microduck_8dof.xml"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    expected_geoms = [
        "stair_step_1",
        "stair_step_2",
        "stair_step_3",
        "stair_platform",
    ]
    for name in expected_geoms:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid != -1, f"Missing geom {name} in {model_path}"


def test_apply_stair_stage_scaling():
    """Verify apply_stair_stage sets stair coordinates properly."""
    m = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "models" / "mjcf" / "microduck_8dof.xml"))
    apply_stair_stage(m, stage=0)
    top_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "stair_platform")
    assert top_gid != -1
    # In stage 0 stairs are submerged
    assert m.geom_pos[top_gid, 2] <= 0.0


def test_next_stair_stage_progression():
    """Verify next_stair_stage advances only after holding consecutive passing criteria."""
    # Stage 0 failure: ep_len too low
    s, hold = next_stair_stage(stage=0, ep_len_mean=350.0, rew_vx_ratio=0.85, hold_count=0, hold_iters=5, mean_distance_x=1.8)
    assert s == 0 and hold == 0

    # Stage 0 failure: ep_len high but distance too short (standing still)
    s, hold = next_stair_stage(stage=0, ep_len_mean=450.0, rew_vx_ratio=0.85, hold_count=0, hold_iters=5, mean_distance_x=0.2)
    assert s == 0 and hold == 0

    # Stage 0 success count increment (ep_len_mean >= 400.0, rew_vx_ratio >= 0.70, mean_distance_x >= 1.5)
    s, hold = next_stair_stage(stage=0, ep_len_mean=405.0, rew_vx_ratio=0.75, hold_count=3, hold_iters=5, mean_distance_x=1.8)
    assert s == 0 and hold == 4

    # Stage 0 -> 1 threshold crossed
    s, hold = next_stair_stage(stage=0, ep_len_mean=405.0, rew_vx_ratio=0.75, hold_count=4, hold_iters=5, mean_distance_x=1.8)
    assert s == 1 and hold == 0


def test_edge_fall_detection_and_penalty():
    """Verify that drifting laterally off stair deck triggers termination and fall penalty."""
    runner = ParallelSimRunner(
        model_path=REPO_ROOT / "models" / "mjcf" / "microduck_8dof.xml",
        num_envs=2,
        terrain_mode="stairs",
        stage=0,
    )
    try:
        runner.reset()
        # Manually shift env 0 robot to Y = 0.50m (beyond half-width 0.40m + 0.04m)
        runner.datas[0].qpos[0] = 1.50  # inside stair region X
        runner.datas[0].qpos[1] = 0.50  # fallen off edge
        actions = np.zeros((2, 8), dtype=np.float32)
        obs, rewards, dones, infos = runner.step(actions)
        assert dones[0] == True, "Fallen robot must be terminated"
        assert rewards[0] <= -3.0, f"Expected fall penalty <= -3.0, got {rewards[0]}"
    finally:
        runner.close()


def test_train_ppo_stairs_dry_run(tmp_path):
    """Verify train_ppo runs with terrain_mode='stairs'."""
    res = train_ppo(
        model_path=str(REPO_ROOT / "models" / "mjcf" / "microduck_8dof.xml"),
        save_path=str(tmp_path / "stairs_dry_run.pt"),
        num_envs=2,
        rollout_steps=4,
        max_iterations=1,
        dry_run=True,
        terrain_mode="stairs",
        start_stage=0,
        verbose=False,
    )
    assert res["success"] is True
