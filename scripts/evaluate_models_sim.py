"""Comprehensive Simulation Benchmark and Model Evaluation for MicroDuck V1 and CyberDuck V2.

Evaluates:
- microduck_best.pt on MicroDuck V1 MJCF
- cyberduck_v2_best.pt on CyberDuck V2 MJCF
Across multiple locomotion scenarios:
1. Nominal forward walking (vx = 0.5 m/s)
2. Chibi forward walking (vx = 0.25 m/s)
3. Stationary standing balance (vx = 0.0 m/s)
4. Hurdles / obstacle terrain (Stage 1)
"""

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.env_config import _DEFAULT_CFG, quat_to_yaw
from training.run_parallel_sim import ParallelSimRunner, has_obstacle_contact
from training.nav_planner import LidarWaypointPlanner
from scripts.export_policy import apply_obs_norm_to_actor

__all__ = [
    "compute_path_unstability",
    "has_obstacle_contact",
    "run_benchmark_scenario",
    "get_xml_supported_stage",
    "load_actor",
]


def compute_path_unstability(yaw_history: np.ndarray, dist_x: float) -> Tuple[float, float]:
    """Compute Total Variation of yaw angles (raw rad and distance-normalized rad/m).

    Handles wraparound across [-pi, pi) smoothly.
    Returns:
        (raw_yaw_tv_rad, norm_yaw_unstability_rad_per_m)
    """
    yaw_arr = np.asarray(yaw_history, dtype=np.float32)
    if yaw_arr.size < 2:
        return 0.0, 0.0
    diffs = (np.diff(yaw_arr) + np.pi) % (2.0 * np.pi) - np.pi
    raw = float(np.sum(np.abs(diffs)))
    norm = float(raw / max(abs(float(dist_x)), 0.1))
    return raw, norm


def get_xml_supported_stage(xml_path: Path, requested_stage: int) -> int:
    """Return requested_stage if XML contains obstacle/stair geometry, else fallback to 0."""
    if requested_stage <= 0:
        return 0
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        has_stairs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "stair_step_1") != -1
        has_obstacles = (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left") != -1
            or mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_front") != -1
        )
        return requested_stage if (has_stairs or has_obstacles) else 0
    except Exception:
        return 0


def load_actor(checkpoint_path: Path) -> nn.Module:
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = cp.get("actor_state_dict", cp.get("model_state_dict", cp))

    obs_dim = 42
    hidden_dim = 256
    action_dim = 8

    if isinstance(cp, dict):
        obs_dim = cp.get("obs_dim", obs_dim)
        hidden_dim = cp.get("hidden_dim", hidden_dim)
        action_dim = cp.get("action_dim", action_dim)

    # Infer from state_dict keys/shapes if available
    if "0.weight" in state_dict:
        obs_dim = int(state_dict["0.weight"].shape[1])
        hidden_dim = int(state_dict["0.weight"].shape[0])
    elif "actor.0.weight" in state_dict:
        obs_dim = int(state_dict["actor.0.weight"].shape[1])
        hidden_dim = int(state_dict["actor.0.weight"].shape[0])

    for key in ("4.weight", "actor.4.weight", "2.weight", "actor.2.weight"):
        if key in state_dict:
            action_dim = int(state_dict[key].shape[0])
            break

    critic_obs_dim = None
    if isinstance(cp, dict):
        if "critic_obs_dim" in cp:
            critic_obs_dim = int(cp["critic_obs_dim"])
        elif "model_state_dict" in cp and "critic.0.weight" in cp["model_state_dict"]:
            critic_obs_dim = int(cp["model_state_dict"]["critic.0.weight"].shape[1])
        elif "critic.0.weight" in cp:
            critic_obs_dim = int(cp["critic.0.weight"].shape[1])

    actor = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, action_dim),
    )

    if "actor_state_dict" in cp:
        actor.load_state_dict(cp["actor_state_dict"])
    elif "model_state_dict" in cp:
        sub = {k.replace("actor.", ""): v for k, v in cp["model_state_dict"].items() if k.startswith("actor.")}
        actor.load_state_dict(sub if sub else cp["model_state_dict"])
    else:
        actor.load_state_dict(state_dict)

    apply_obs_norm_to_actor(actor, cp)

    actor.eval()
    actor.obs_dim = obs_dim
    actor.hidden_dim = hidden_dim
    actor.action_dim = action_dim
    actor.critic_obs_dim = critic_obs_dim
    actor.stage = int(cp.get("stage", 0)) if isinstance(cp, dict) else 0
    return actor


def run_benchmark_scenario(
    runner: ParallelSimRunner,
    actor: nn.Module,
    cmd_vx: float = 0.5,
    max_steps: int = 500,
    scenario_name: str = "Forward Walk",
    use_nav_planner: bool = False,
    goal_x: float = 2.0,
) -> Dict[str, Any]:
    effective_nav = bool(use_nav_planner or getattr(runner, "use_nav_planner", False))
    if effective_nav:
        runner.use_nav_planner = True
        runner.nav_goal_xy = (float(goal_x), 0.0)
        if getattr(runner, "nav_planner", None) is None:
            runner.nav_planner = LidarWaypointPlanner()
    obs = runner.reset()
    if effective_nav:
        runner.update_nav_commands()
        obs = runner.get_observations()
    else:
        runner.commands[:, 0] = cmd_vx
        runner.commands[:, 1] = 0.0
        runner.commands[:, 2] = 0.0

    num_envs = runner.num_envs
    alive_steps = np.zeros(num_envs, dtype=np.int32)
    accum_reward = np.zeros(num_envs, dtype=np.float32)
    max_dist_x = np.zeros(num_envs, dtype=np.float32)
    final_dist_x = np.zeros(num_envs, dtype=np.float32)
    max_tilt_deg = np.zeros(num_envs, dtype=np.float32)
    accum_torque_sq = np.zeros(num_envs, dtype=np.float32)
    accum_work = np.zeros(num_envs, dtype=np.float32)
    survived = np.ones(num_envs, dtype=bool)

    # Initial positions
    start_x = np.array([d.qpos[0] for d in runner.datas], dtype=np.float32)

    # Track obstacle avoidance & yaw trajectory metrics
    yaw_histories: List[List[float]] = [
        [float(quat_to_yaw(runner.datas[i].qpos[3:7]))] for i in range(num_envs)
    ]
    env_collided = np.zeros(num_envs, dtype=bool)
    min_clearance = np.full(num_envs, 2.0, dtype=np.float32)

    actor_in_dim = getattr(actor, "obs_dim", None)
    if actor_in_dim is None and hasattr(actor, "__getitem__") and hasattr(actor[0], "in_features"):
        actor_in_dim = actor[0].in_features

    for step in range(max_steps):
        with torch.no_grad():
            inp = torch.from_numpy(obs).float()
            if actor_in_dim is not None and inp.shape[1] != actor_in_dim:
                if inp.shape[1] > actor_in_dim:
                    inp = inp[:, :actor_in_dim]
                else:
                    inp = torch.nn.functional.pad(inp, (0, actor_in_dim - inp.shape[1]))
            actions = actor(inp).numpy()

        next_obs, rewards, dones, infos = runner.step(actions)

        # Track metrics for currently alive envs
        for i in range(num_envs):
            if survived[i]:
                alive_steps[i] += 1
                accum_reward[i] += float(rewards[i])
                curr_x = float(infos[i].get("distance_x", 0.0))
                final_dist_x[i] = curr_x
                if curr_x > max_dist_x[i]:
                    max_dist_x[i] = curr_x

                # Tilt angle from projected gravity (grav_z)
                grav_z = float(next_obs[i, 5])
                tilt_rad = math.acos(max(-1.0, min(1.0, -grav_z)))
                tilt_deg = math.degrees(tilt_rad)
                if tilt_deg > max_tilt_deg[i]:
                    max_tilt_deg[i] = tilt_deg

                # Effort / Torque
                ctrl = runner.datas[i].ctrl[:]
                qvel = runner.datas[i].qvel[6:14]
                accum_torque_sq[i] += float(np.sum(ctrl ** 2))
                accum_work[i] += float(np.sum(np.abs(ctrl * qvel))) * runner.control_dt

                # Yaw trajectory & obstacle tracking
                pre_yaw = float(infos[i].get("yaw", quat_to_yaw(runner.datas[i].qpos[3:7])))
                yaw_histories[i].append(pre_yaw)

                env_collided[i] |= bool(infos[i].get("obstacle_contact", False))

                d_val = infos[i].get("d_min")
                if d_val is None and next_obs.shape[1] >= 42:
                    lidar_slice = (
                        infos[i].get("terminal_obs", next_obs[i])[37:42]
                        if dones[i]
                        else next_obs[i, 37:42]
                    )
                    d_val = float(np.min(lidar_slice))
                if d_val is not None:
                    min_clearance[i] = min(min_clearance[i], float(d_val))

                if dones[i]:
                    survived[i] = False

        obs = next_obs
        if not np.any(survived):
            break

    success_rate = float(np.mean(alive_steps >= max_steps))
    mean_eplen = float(np.mean(alive_steps))
    mean_reward = float(np.mean(accum_reward))
    mean_dist_x = float(np.mean(final_dist_x))
    mean_max_dist_x = float(np.mean(max_dist_x))
    mean_speed_x = float(np.mean(final_dist_x / np.maximum(alive_steps * runner.control_dt, 1e-3)))
    mean_tilt = float(np.mean(max_tilt_deg))
    mean_torque_sq = float(np.mean(accum_torque_sq / np.maximum(alive_steps, 1)))
    mean_power_watt = float(np.mean(accum_work / np.maximum(alive_steps * runner.control_dt, 1e-3)))

    unstab_list = [
        compute_path_unstability(np.array(yaw_histories[i], dtype=np.float32), final_dist_x[i])[1]
        for i in range(num_envs)
    ]
    mean_yaw_unstability = float(np.mean(unstab_list)) if unstab_list else 0.0
    collision_rate_per_100 = float(np.mean(env_collided) * 100.0)
    mean_min_clearance = float(np.mean(min_clearance))

    return {
        "scenario": scenario_name,
        "cmd_vx": cmd_vx,
        "max_steps": max_steps,
        "num_envs": num_envs,
        "success_rate": success_rate,
        "mean_eplen": mean_eplen,
        "mean_reward": mean_reward,
        "mean_dist_x": mean_dist_x,
        "mean_max_dist_x": mean_max_dist_x,
        "mean_speed_x": mean_speed_x,
        "mean_tilt_deg": mean_tilt,
        "mean_torque_sq": mean_torque_sq,
        "mean_power_watt": mean_power_watt,
        "yaw_unstability": mean_yaw_unstability,
        "collision_rate_per_100": collision_rate_per_100,
        "min_clearance_m": mean_min_clearance,
    }


def main():
    parser = argparse.ArgumentParser(description="🦆 MicroDuck V1 vs CyberDuck V2 Simulation Benchmark")
    parser.add_argument("--use-nav-planner", action="store_true", help="Include Stage 1 Hurdles + L2 NavPlanner scenario")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per scenario (default: 500)")
    args = parser.parse_args()

    v1_model_path = REPO_ROOT / "pipelines" / "rpi5_stack" / "models" / "mjcf" / "microduck_rpi5_8dof.xml"
    v2_model_path = REPO_ROOT / "duck_v2" / "cyberduck_v2.xml"
    v1_checkpoint = REPO_ROOT / "models" / "checkpoints" / "microduck_best.pt"
    v2_checkpoint = REPO_ROOT / "models" / "checkpoints" / "cyberduck_v2_best.pt"

    print("================================================================================")
    print("🦆 EVALUATION REPORT: MICRODUCK V1 vs CYBERDUCK V2 IN MUJOCO SIMULATION")
    print("================================================================================")

    v1_actor = load_actor(v1_checkpoint)
    v2_actor = load_actor(v2_checkpoint)

    scenarios = [
        {"name": "Standard Forward Walk (0.50 m/s)", "cmd_vx": 0.50, "steps": args.steps, "stage": 0, "use_nav": False},
        {"name": "Chibi Walking Pace (0.25 m/s)", "cmd_vx": 0.25, "steps": args.steps, "stage": 0, "use_nav": False},
        {"name": "Stationary Balance Stand (0.0 m/s)", "cmd_vx": 0.00, "steps": min(300, args.steps), "stage": 0, "use_nav": False},
        {"name": "Stage 1 Hurdles Navigation", "cmd_vx": 0.40, "steps": args.steps, "stage": 1, "use_nav": False},
    ]
    if args.use_nav_planner:
        scenarios.append(
            {"name": "Stage 1 Hurdles + L2 NavPlanner", "cmd_vx": 0.40, "steps": args.steps, "stage": 1, "use_nav": True}
        )

    results = []

    for sc in scenarios:
        print(f"\n>>> Running Scenario: {sc['name']} ...")
        # Run V1
        stage_v1 = get_xml_supported_stage(v1_model_path, sc.get("stage", 0))
        r_v1 = ParallelSimRunner(
            model_path=v1_model_path,
            num_envs=args.num_envs,
            max_episode_steps=sc["steps"],
            stage=stage_v1,
            use_nav_planner=sc.get("use_nav", False),
        )
        try:
            res_v1 = run_benchmark_scenario(
                r_v1,
                v1_actor,
                cmd_vx=sc["cmd_vx"],
                max_steps=sc["steps"],
                scenario_name=sc["name"],
                use_nav_planner=sc.get("use_nav", False),
            )
            res_v1["model"] = "MicroDuck V1 (microduck_best.pt)"
            results.append(res_v1)
        finally:
            r_v1.close()

        # Run V2
        stage_v2 = get_xml_supported_stage(v2_model_path, sc.get("stage", 0))
        r_v2 = ParallelSimRunner(
            model_path=v2_model_path,
            num_envs=args.num_envs,
            max_episode_steps=sc["steps"],
            stage=stage_v2,
            use_nav_planner=sc.get("use_nav", False),
        )
        try:
            res_v2 = run_benchmark_scenario(
                r_v2,
                v2_actor,
                cmd_vx=sc["cmd_vx"],
                max_steps=sc["steps"],
                scenario_name=sc["name"],
                use_nav_planner=sc.get("use_nav", False),
            )
            res_v2["model"] = "CyberDuck V2 (cyberduck_v2_best.pt)"
            results.append(res_v2)
        finally:
            r_v2.close()

        print(
            f"  [V1] EpLen: {res_v1['mean_eplen']:.1f}/{sc['steps']} | DistX: {res_v1['mean_dist_x']:+.3f}m | "
            f"Reward: {res_v1['mean_reward']:+.2f} | Tilt: {res_v1['mean_tilt_deg']:.1f}° | "
            f"Unstab: {res_v1['yaw_unstability']:.2f} | Coll/100: {res_v1['collision_rate_per_100']:.1f} | "
            f"MinClr: {res_v1['min_clearance_m']:.2f}m"
        )
        print(
            f"  [V2] EpLen: {res_v2['mean_eplen']:.1f}/{sc['steps']} | DistX: {res_v2['mean_dist_x']:+.3f}m | "
            f"Reward: {res_v2['mean_reward']:+.2f} | Tilt: {res_v2['mean_tilt_deg']:.1f}° | "
            f"Unstab: {res_v2['yaw_unstability']:.2f} | Coll/100: {res_v2['collision_rate_per_100']:.1f} | "
            f"MinClr: {res_v2['min_clearance_m']:.2f}m"
        )

    header = f"{'Model':<16} | {'Scenario':<30} | {'Surv%':<6} | {'EpLen':<8} | {'DistX (m)':<10} | {'Reward':<9} | {'Tilt°':<6} | {'Unstab':<8} | {'Coll/100':<9} | {'MinClr':<8} | {'Pwr (W)':<7}"
    print("\n" + "=" * len(header))
    print("DETAILED BENCHMARK SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        m_short = "MicroDuck V1" if "V1" in r["model"] else "CyberDuck V2"
        print(
            f"{m_short:<16} | {r['scenario'][:30]:<30} | {r['success_rate']*100:5.1f}% | {r['mean_eplen']:8.1f} | "
            f"{r['mean_dist_x']:+10.3f} | {r['mean_reward']:+9.2f} | {r['mean_tilt_deg']:5.1f}° | "
            f"{r['yaw_unstability']:8.2f} | {r['collision_rate_per_100']:8.1f}% | {r['min_clearance_m']:7.2f}m | {r['mean_power_watt']:6.2f}W"
        )
    print("=" * len(header))


if __name__ == "__main__":
    main()
