from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_models_sim import (
    compute_path_unstability,
    has_obstacle_contact,
    run_benchmark_scenario,
)
from training.env_config import quat_to_yaw
from training.run_parallel_sim import ParallelSimRunner


def test_compute_path_unstability_handles_pi_wraparound():
    # Smooth wrap across +pi -> -pi (tiny real change ~0.04 rad)
    yaws = np.array([3.12, -3.14, -3.12], dtype=np.float32)
    raw_tv, norm_tv = compute_path_unstability(yaws, dist_x=1.0)
    assert raw_tv < 0.1
    assert norm_tv == pytest.approx(raw_tv, abs=1e-5)

    # Zig-zag wobble must yield higher unstability than straight heading
    zigzag = np.array([0.0, 0.4, -0.4, 0.4, -0.4], dtype=np.float32)
    raw_zz, _ = compute_path_unstability(zigzag, dist_x=1.0)
    assert raw_zz > 2.0


def test_compute_path_unstability_empty_or_single_point():
    # Empty array
    assert compute_path_unstability(np.array([], dtype=np.float32), dist_x=1.0) == (0.0, 0.0)
    # Single point
    assert compute_path_unstability(np.array([1.23], dtype=np.float32), dist_x=1.0) == (0.0, 0.0)
    # Distance near zero (< 0.1) clamped to 0.1
    yaws = np.array([0.1, 0.3], dtype=np.float32)
    raw, norm = compute_path_unstability(yaws, dist_x=0.0)
    assert raw == pytest.approx(0.2, abs=1e-5)
    assert norm == pytest.approx(0.2 / 0.1, abs=1e-5)


def test_has_obstacle_contact_detects_collision_and_ignores_non_obstacles():
    class DummyContact:
        def __init__(self, g1: int, g2: int):
            self.geom1 = g1
            self.geom2 = g2

    class DummyData:
        def __init__(self, contacts):
            self.contact = contacts
            self.ncon = len(contacts)

    # Empty contacts or empty set
    assert not has_obstacle_contact(DummyData([]), {10, 11})
    assert not has_obstacle_contact(DummyData([DummyContact(1, 2)]), set())

    # Non-obstacle contacts
    data = DummyData([DummyContact(1, 2), DummyContact(2, 3)])
    assert not has_obstacle_contact(data, {10, 11})

    # Obstacle contact on geom1
    data_hit1 = DummyData([DummyContact(1, 2), DummyContact(10, 3)])
    assert has_obstacle_contact(data_hit1, {10, 11})

    # Obstacle contact on geom2
    data_hit2 = DummyData([DummyContact(1, 11)])
    assert has_obstacle_contact(data_hit2, {10, 11})

    # Test pre-reset infos preservation on termination step in ParallelSimRunner
    runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_episode_steps=5,
        stage=1,
        enable_auto_recovery=False,
    )
    try:
        runner.reset()
        # Place env 0 touching pillar_l1_left with non-zero yaw and strong tilt to trigger done
        d0 = runner.datas[0]
        pillar_gid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
        pillar_pos = runner.model.geom_pos[pillar_gid]
        d0.qpos[0] = pillar_pos[0]
        d0.qpos[1] = pillar_pos[1]
        d0.qpos[2] = 0.15
        q_rot = np.array([0.6, 0.0, 0.7, 0.4], dtype=np.float64)
        d0.qpos[3:7] = q_rot / np.linalg.norm(q_rot)
        mujoco.mj_forward(runner.model, d0)

        actions = np.zeros((runner.num_envs, runner.action_dim), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)

        # Environment 0 must trigger termination (dones[0] is True)
        assert bool(dones[0]) is True
        # Pre-reset yaw preserved in infos[0] despite reset_env(0) execution
        assert abs(infos[0]["yaw"]) > 0.2
        # Pre-reset obstacle contact preserved in infos[0]
        assert infos[0]["obstacle_contact"] is True
        # Confirm reset_env(0) did run and reset runner.datas[0] base yaw back near 0.0
        post_yaw = quat_to_yaw(runner.datas[0].qpos[3:7])
        assert abs(post_yaw) < 0.05

        # Check all infos dictionaries contain yaw, d_min, and obstacle_contact keys
        for i in range(runner.num_envs):
            assert "yaw" in infos[i]
            assert isinstance(infos[i]["yaw"], float)
            assert "obstacle_contact" in infos[i]
            assert isinstance(infos[i]["obstacle_contact"], bool)
            assert "d_min" in infos[i]
            if infos[i]["d_min"] is not None:
                assert isinstance(infos[i]["d_min"], float)
    finally:
        runner.close()


def test_run_benchmark_scenario_returns_obstacle_metrics():
    # 1. CyberDuck V2 Stage 1
    runner_v2 = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_episode_steps=10,
        stage=1,
    )
    try:
        zero_actor = nn.Linear(runner_v2.obs_dim, runner_v2.action_dim)
        res_v2 = run_benchmark_scenario(
            runner_v2, zero_actor, cmd_vx=0.2, max_steps=10, scenario_name="Test V2 Stage 1"
        )
        assert "yaw_unstability" in res_v2 and res_v2["yaw_unstability"] >= 0.0
        assert "collision_rate_per_100" in res_v2 and 0.0 <= res_v2["collision_rate_per_100"] <= 100.0
        assert "min_clearance_m" in res_v2 and res_v2["min_clearance_m"] >= 0.0
    finally:
        runner_v2.close()

    # 2. MicroDuck V1 Stage 0
    v1_path = Path("pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml")
    if not v1_path.exists():
        v1_path = Path("models/mjcf/microduck_8dof.xml")
    runner_v1 = ParallelSimRunner(
        model_path=v1_path,
        num_envs=2,
        num_threads=1,
        max_episode_steps=10,
        stage=0,
    )
    try:
        zero_actor = nn.Linear(runner_v1.obs_dim, runner_v1.action_dim)
        res_v1 = run_benchmark_scenario(
            runner_v1, zero_actor, cmd_vx=0.2, max_steps=10, scenario_name="Test V1 Stage 0"
        )
        assert "yaw_unstability" in res_v1 and res_v1["yaw_unstability"] >= 0.0
        assert "collision_rate_per_100" in res_v1 and 0.0 <= res_v1["collision_rate_per_100"] <= 100.0
        assert "min_clearance_m" in res_v1 and res_v1["min_clearance_m"] >= 0.0
    finally:
        runner_v1.close()

    # 3. Obstacle-free environment (no LiDAR, no obstacles) verifies min_clearance_m defaults to 2.0
    free_path = Path("models/mjcf/microduck_8dof.xml")
    if free_path.exists():
        runner_free = ParallelSimRunner(
            model_path=free_path,
            num_envs=2,
            num_threads=1,
            max_episode_steps=5,
            stage=0,
        )
        try:
            runner_free.obstacle_geom_ids = []  # Simulate env without obstacles
            zero_actor = nn.Linear(runner_free.obs_dim, runner_free.action_dim)
            res_free = run_benchmark_scenario(
                runner_free, zero_actor, cmd_vx=0.0, max_steps=5, scenario_name="Test Obstacle Free"
            )
            assert res_free["min_clearance_m"] == pytest.approx(2.0, abs=1e-3)
        finally:
            runner_free.close()


def test_run_benchmark_scenario_with_nav_planner():
    runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_episode_steps=10,
        stage=1,
    )
    try:
        zero_actor = nn.Linear(runner.obs_dim, runner.action_dim)
        res = run_benchmark_scenario(
            runner,
            zero_actor,
            cmd_vx=0.3,
            max_steps=10,
            scenario_name="Test Stage 1 with L2 NavPlanner",
            use_nav_planner=True,
            goal_x=2.0,
        )
        assert res["scenario"] == "Test Stage 1 with L2 NavPlanner"
        assert "yaw_unstability" in res and res["yaw_unstability"] >= 0.0
        assert "collision_rate_per_100" in res and 0.0 <= res["collision_rate_per_100"] <= 100.0
        assert "min_clearance_m" in res and res["min_clearance_m"] >= 0.0
        assert "success_rate" in res
        assert "mean_dist_x" in res
        assert res["num_envs"] == 2
    finally:
        runner.close()


def test_run_benchmark_scenario_restores_runner_state_and_syncs_cmd():
    runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_episode_steps=10,
        stage=1,
        use_nav_planner=False,
    )
    try:
        assert runner.use_nav_planner is False
        zero_actor = nn.Linear(runner.obs_dim, runner.action_dim)

        # 1. Run scenario with use_nav_planner=True
        run_benchmark_scenario(
            runner,
            zero_actor,
            cmd_vx=0.3,
            max_steps=2,
            scenario_name="Nav True",
            use_nav_planner=True,
            goal_x=2.0,
        )

        # Runner must restore use_nav_planner back to False after run
        assert runner.use_nav_planner is False, "runner.use_nav_planner leaked True state after use_nav_planner=True run"

        # 2. Run scenario on same runner with use_nav_planner=False and cmd_vx=0.88
        run_benchmark_scenario(
            runner,
            zero_actor,
            cmd_vx=0.88,
            max_steps=2,
            scenario_name="Nav False",
            use_nav_planner=False,
        )

        assert runner.use_nav_planner is False
        assert np.allclose(runner.commands[:, 0], 0.88)
    finally:
        runner.close()
