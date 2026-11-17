"""Tests for pillar randomization guaranteeing spawn clearance and passable gaps."""

from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.terrain_curriculum import apply_stage, randomize_pillars

XML_V2 = Path(__file__).resolve().parent.parent / "duck_v2" / "cyberduck_v2.xml"


def test_randomize_pillars_respects_spawn_and_gap_safety():
    model = mujoco.MjModel.from_xml_path(str(XML_V2))
    apply_stage(model, stage=1)
    rng = np.random.default_rng(123)

    active = randomize_pillars(model, stage=1, rng=rng, density_rho=0.7, jitter_xy=0.08)
    assert len(active) >= 1
    for name in active:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        px, py, pz = model.geom_pos[gid]
        assert pz > 0.0
        assert np.hypot(px, py) >= 0.30

    # Paired L1/L3 pillars when both active must preserve passable gap >= 0.22m
    l1_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
    l1_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_right")
    if model.geom_pos[l1_l, 2] > 0 and model.geom_pos[l1_r, 2] > 0:
        surface_gap = abs(model.geom_pos[l1_l, 1] - model.geom_pos[l1_r, 1]) - (
            model.geom_size[l1_l, 0] + model.geom_size[l1_r, 0]
        )
        assert surface_gap >= 0.22


def test_randomize_pillars_multiple_seeds_and_corridor_bounds():
    """Verify bounds, clearance, and gap across multiple stages and seeds."""
    model = mujoco.MjModel.from_xml_path(str(XML_V2))

    for seed in range(50):
        rng = np.random.default_rng(seed)
        stage = (seed % 3) + 1  # stages 1, 2, 3
        active = randomize_pillars(
            model,
            stage=stage,
            rng=rng,
            density_rho=0.6,
            jitter_xy=0.08,
            min_spawn_dist=0.30,
            min_gap_width=0.22,
        )

        assert 1 <= len(active) <= 6

        # Check all pillars in model
        for name in [
            "pillar_l1_left",
            "pillar_l1_right",
            "pillar_l2_center",
            "pillar_l3_left",
            "pillar_l3_right",
            "pillar_back",
        ]:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            px, py, pz = model.geom_pos[gid]
            if name in active:
                assert pz > 0.0
                assert abs(py) <= 0.45 + 1e-6
                r = model.geom_size[gid, 0]
                assert np.hypot(px, py) >= 0.30 - 1e-6
            else:
                assert pz == -0.50

        # Check both pairs L1 and L3
        for l_name, r_name in [
            ("pillar_l1_left", "pillar_l1_right"),
            ("pillar_l3_left", "pillar_l3_right"),
        ]:
            gid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, l_name)
            gid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, r_name)
            if l_name in active and r_name in active:
                surface_gap = abs(model.geom_pos[gid_l, 1] - model.geom_pos[gid_r, 1]) - (
                    model.geom_size[gid_l, 0] + model.geom_size[gid_r, 0]
                )
                assert surface_gap >= 0.22 - 1e-6


def test_randomize_pillars_stage0_or_density0_noop():
    """Verify stage <= 0 or density_rho <= 0 returns all pillars unchanged without hiding/jitter."""
    model = mujoco.MjModel.from_xml_path(str(XML_V2))
    rng = np.random.default_rng(42)

    # stage=0
    active_s0 = randomize_pillars(model, stage=0, rng=rng, density_rho=0.8)
    assert len(active_s0) == 6
    for name in active_s0:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_pos[gid, 2] == 0.15

    # density_rho=0.0
    active_d0 = randomize_pillars(model, stage=1, rng=rng, density_rho=0.0)
    assert len(active_d0) == 6
    for name in active_d0:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_pos[gid, 2] == 0.15


def test_runner_randomize_pillars_changes_positions_when_enabled():
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path=str(XML_V2),
        num_envs=2,
        num_threads=1,
        stage=1,
        terrain_mode="hurdles",
        randomize_pillars=True,
        pillar_density=0.85,
        pillar_jitter_xy=0.06,
    )
    try:
        gid = runner.obstacle_geom_ids[0]
        pos_a = runner.model.geom_pos[gid].copy()
        runner.reset()
        pos_b = runner.model.geom_pos[gid].copy()
        assert not np.allclose(pos_a, pos_b)
    finally:
        runner.close()


def test_runner_randomize_pillars_false_preserves_exact_positions():
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path=str(XML_V2),
        num_envs=2,
        num_threads=1,
        stage=1,
        terrain_mode="hurdles",
        randomize_pillars=False,
    )
    try:
        orig_positions = runner.model.geom_pos.copy()
        runner.reset()
        assert np.array_equal(runner.model.geom_pos, orig_positions)
    finally:
        runner.close()


def test_runner_compute_d_min_filters_submerged_pillars():
    from training.run_parallel_sim import ParallelSimRunner

    # Use 37-dim non-LiDAR model so step() calls _compute_d_min() directly
    runner = ParallelSimRunner(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
        num_threads=1,
        stage=1,
        terrain_mode="hurdles",
        randomize_pillars=False,
    )
    try:
        assert runner.obs_dim == 37
        # Step once with active obstacles so d_min, prev_d_min are populated with positive distances
        actions = np.zeros((2, runner.action_dim), dtype=np.float32)
        obs, rewards, dones, infos = runner.step(actions)
        assert np.all(runner.d_min > 0.0)
        assert np.all(runner.prev_d_min > 0.0)

        # Before submerging, d_min is a valid float/array
        d_val = runner._compute_d_min(0)
        assert d_val is not None

        # Submerge all obstacles below floor (z <= 0)
        for gid in runner.obstacle_geom_ids:
            runner.model.geom_pos[gid, 2] = -0.50

        # Now _compute_d_min must return None since all obstacles are submerged
        assert runner._compute_d_min(0) is None
        assert runner._compute_d_min() is None

        # Step again: curr_d_min is None.
        # d_min, prev_d_min, d_min_dot must be reset to 0.0 without NaN/TypeError
        obs2, rewards2, dones2, infos2 = runner.step(actions)
        assert np.all(runner.d_min == 0.0)
        assert np.all(runner.prev_d_min == 0.0)
        assert np.all(runner.d_min_dot == 0.0)
        assert not np.isnan(rewards2).any()

        # Artificially corrupt tracking buffers to non-zero values
        runner.d_min[:] = 99.0
        runner.prev_d_min[:] = 99.0
        runner.d_min_dot[:] = 99.0

        # reset_env(0) with submerged obstacles must reset tracking buffers to 0.0
        runner.reset_env(0)
        assert runner.d_min[0] == 0.0
        assert runner.prev_d_min[0] == 0.0
        assert runner.d_min_dot[0] == 0.0
    finally:
        runner.close()


def test_train_ppo_smoke_with_randomize_pillars(tmp_path):
    from scripts.train_ppo import train_ppo

    res = train_ppo(
        model_path=str(XML_V2),
        num_envs=2,
        rollout_steps=4,
        max_iterations=1,
        dry_run=True,
        save_path=tmp_path / "smoke_ppo.pt",
        start_stage=1,
        terrain_mode="hurdles",
        randomize_pillars=True,
        pillar_density=0.85,
        pillar_jitter=0.06,
        verbose=False,
    )
    assert res["success"] is True
