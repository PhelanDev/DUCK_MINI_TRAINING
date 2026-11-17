import numpy as np
import pytest

from training.nav_planner import LidarWaypointPlanner, NavPlannerConfig, RAY_ANGLES


def test_nav_planner_steers_toward_goal_and_avoids_blocked_front():
    planner = LidarWaypointPlanner(NavPlannerConfig())
    # Clear front -> move forward toward goal at (2.0, 0.0)
    cmd_clear = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    assert cmd_clear[0] > 0.15 and abs(cmd_clear[2]) < 0.05

    # Blocked center & right, wide left (+15/+30 deg = indices 3,4) -> yaw left (wz > 0) and slow vx
    cmd_left = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([0.25, 0.25, 0.25, 1.80, 1.90], dtype=np.float32),
    )
    assert cmd_left[2] > 0.15
    assert cmd_left[0] < cmd_clear[0]


def test_nav_planner_steers_right_when_left_blocked():
    planner = LidarWaypointPlanner(NavPlannerConfig())
    cmd_clear = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    # Blocked center & left (indices 2, 3, 4), wide right (indices 0, 1: -30, -15 deg)
    cmd_right = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([1.80, 1.90, 0.25, 0.25, 0.25], dtype=np.float32),
    )
    assert cmd_right[2] < -0.15  # wz < 0 steers right
    assert cmd_right[0] < cmd_clear[0]


def test_nav_planner_stops_at_goal():
    planner = LidarWaypointPlanner(NavPlannerConfig(goal_tol=0.15))
    pos = np.array([1.95, 0.02])
    goal = np.array([2.0, 0.0])
    assert planner.is_goal_reached(pos, goal)
    cmd = planner.compute_command(pos, yaw=0.0, goal_xy=goal)
    assert np.allclose(cmd, 0.0)

    # Away from goal
    assert not planner.is_goal_reached(np.array([0.0, 0.0]), goal)


def test_nav_planner_handles_none_short_lidar_and_quat_wraparound():
    planner = LidarWaypointPlanner()
    pos = np.array([0.0, 0.0])
    goal = np.array([0.0, 2.0])  # +90 deg left

    # 1. lidar_ranges=None steers toward bearing (+pi/2 -> wz > 0)
    cmd_none = planner.compute_command(pos, yaw=0.0, goal_xy=goal, lidar_ranges=None)
    assert cmd_none.shape == (3,)
    assert cmd_none.dtype == np.float32
    assert cmd_none[2] > 0.2

    # 2. Fewer than 5 rays does not crash
    cmd_short = planner.compute_command(pos, yaw=0.0, goal_xy=goal, lidar_ranges=np.array([1.0, 1.0]))
    assert cmd_short.shape == (3,)
    assert cmd_short.dtype == np.float32
    assert cmd_short[2] > 0.2

    # 3. Negative / zero distance handling
    cmd_neg = planner.compute_command(
        pos,
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([-1.0, 0.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    assert not np.isnan(cmd_neg).any()
    assert not np.isinf(cmd_neg).any()

    # 4. compute_command_from_quat with identity quat [1, 0, 0, 0]
    cmd_quat = planner.compute_command_from_quat(
        pos_xy=pos,
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        goal_xy=np.array([2.0, 0.0]),
    )
    assert cmd_quat[0] > 0.15
    assert abs(cmd_quat[2]) < 0.05

    # 5. compute_command_from_quat with 90-deg yaw quat (cos(pi/4), 0, 0, sin(pi/4))
    # Facing +Y, goal at (2, 0) is to the robot's right (-90 deg -> wz < 0)
    q_90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=np.float32)
    cmd_turn_right = planner.compute_command_from_quat(
        pos_xy=pos,
        quat_wxyz=q_90,
        goal_xy=np.array([2.0, 0.0]),
    )
    assert cmd_turn_right[2] < -0.15


def test_parallel_sim_runner_nav_planner_integration():
    import mujoco
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_episode_steps=20,
        stage=1,
        use_nav_planner=True,
        nav_goal_xy=(2.0, 0.0),
    )
    try:
        assert runner.use_nav_planner is True
        assert isinstance(runner.nav_planner, LidarWaypointPlanner)
        assert np.allclose(runner.nav_goal_xy, [2.0, 0.0])

        obs = runner.reset()
        # Env 0 starts at (0, 0) facing +X, goal at (2, 0) -> positive forward velocity cmd
        assert runner.commands[0, 0] > 0.1
        # Observation channel 6:9 matches runner.commands
        assert np.allclose(obs[0, 6:9], runner.commands[0, :3], atol=1e-5)

        # Place robot close to pillar_l1_left (located around x=0.55, y=0.10)
        pillar_gid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
        p_pos = runner.model.geom_pos[pillar_gid]
        runner.datas[0].qpos[0] = p_pos[0] - 0.25  # 25cm in front of pillar
        runner.datas[0].qpos[1] = p_pos[1]
        runner.datas[0].qpos[2] = 0.128
        runner.datas[0].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        runner.datas[0].qvel[:] = 0.0
        mujoco.mj_forward(runner.model, runner.datas[0])

        runner.update_nav_commands(env_indices=[0])
        # Front ray is blocked by pillar -> robot steers away (wz != 0)
        assert abs(runner.commands[0, 2]) > 0.05

        # Place robot at goal position -> commands zeroed
        runner.datas[0].qpos[0] = 2.0
        runner.datas[0].qpos[1] = 0.0
        runner.datas[0].qpos[2] = 0.128
        runner.datas[0].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        runner.datas[0].qvel[:] = 0.0
        mujoco.mj_forward(runner.model, runner.datas[0])

        runner.update_nav_commands(env_indices=[0])
        assert np.allclose(runner.commands[0, :3], 0.0, atol=1e-5)

        # Take one step and verify next_obs contains updated nav commands
        actions = np.zeros((runner.num_envs, runner.action_dim), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)
        assert np.allclose(next_obs[0, 6:9], runner.commands[0, :3], atol=1e-5)
    finally:
        runner.close()
