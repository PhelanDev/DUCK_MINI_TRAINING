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

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.env_config import _DEFAULT_CFG
from training.run_parallel_sim import ParallelSimRunner


def get_xml_supported_stage(xml_path: Path, requested_stage: int) -> int:
    """Return requested_stage if XML contains obstacle/stair geometry, else fallback to 0."""
    if requested_stage <= 0:
        return 0
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        has_stairs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "stair_step_1") != -1
        return requested_stage if has_stairs else 0
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

    actor.eval()
    actor.obs_dim = obs_dim
    actor.hidden_dim = hidden_dim
    actor.action_dim = action_dim
    actor.stage = int(cp.get("stage", 0)) if isinstance(cp, dict) else 0
    return actor


def run_benchmark_scenario(
    runner: ParallelSimRunner,
    actor: nn.Module,
    cmd_vx: float = 0.5,
    max_steps: int = 500,
    scenario_name: str = "Forward Walk",
) -> Dict[str, Any]:
    obs = runner.reset()
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
    }


def main():
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
        {"name": "Standard Forward Walk (0.50 m/s)", "cmd_vx": 0.50, "steps": 500, "stage": 0},
        {"name": "Chibi Walking Pace (0.25 m/s)", "cmd_vx": 0.25, "steps": 500, "stage": 0},
        {"name": "Stationary Balance Stand (0.0 m/s)", "cmd_vx": 0.00, "steps": 300, "stage": 0},
        {"name": "Stage 1 Hurdles Navigation", "cmd_vx": 0.40, "steps": 500, "stage": 1},
    ]

    results = []

    for sc in scenarios:
        print(f"\n>>> Running Scenario: {sc['name']} ...")
        # Run V1
        stage_v1 = get_xml_supported_stage(v1_model_path, sc.get("stage", 0))
        r_v1 = ParallelSimRunner(
            model_path=v1_model_path,
            num_envs=16,
            max_episode_steps=sc["steps"],
            stage=stage_v1,
        )
        try:
            res_v1 = run_benchmark_scenario(
                r_v1, v1_actor, cmd_vx=sc["cmd_vx"], max_steps=sc["steps"], scenario_name=sc["name"]
            )
            res_v1["model"] = "MicroDuck V1 (microduck_best.pt)"
            results.append(res_v1)
        finally:
            r_v1.close()

        # Run V2
        stage_v2 = get_xml_supported_stage(v2_model_path, sc.get("stage", 0))
        r_v2 = ParallelSimRunner(
            model_path=v2_model_path,
            num_envs=16,
            max_episode_steps=sc["steps"],
            stage=stage_v2,
        )
        try:
            res_v2 = run_benchmark_scenario(
                r_v2, v2_actor, cmd_vx=sc["cmd_vx"], max_steps=sc["steps"], scenario_name=sc["name"]
            )
            res_v2["model"] = "CyberDuck V2 (cyberduck_v2_best.pt)"
            results.append(res_v2)
        finally:
            r_v2.close()

        print(f"  [V1] EpLen: {res_v1['mean_eplen']:.1f}/{sc['steps']} | DistX: {res_v1['mean_dist_x']:+.3f}m | Reward: {res_v1['mean_reward']:+.2f} | Tilt: {res_v1['mean_tilt_deg']:.1f}°")
        print(f"  [V2] EpLen: {res_v2['mean_eplen']:.1f}/{sc['steps']} | DistX: {res_v2['mean_dist_x']:+.3f}m | Reward: {res_v2['mean_reward']:+.2f} | Tilt: {res_v2['mean_tilt_deg']:.1f}°")

    print("\n================================================================================")
    print("DETAILED BENCHMARK SUMMARY TABLE")
    print("================================================================================")
    header = f"{'Model':<16} | {'Scenario':<30} | {'Surv%':<6} | {'EpLen':<8} | {'DistX (m)':<10} | {'Reward':<9} | {'Tilt°':<6} | {'Pwr (W)':<7}"
    print(header)
    print("-" * len(header))
    for r in results:
        m_short = "MicroDuck V1" if "V1" in r["model"] else "CyberDuck V2"
        print(
            f"{m_short:<16} | {r['scenario'][:30]:<30} | {r['success_rate']*100:5.1f}% | {r['mean_eplen']:6.1f}   | {r['mean_dist_x']:+9.3f}  | {r['mean_reward']:+8.2f}  | {r['mean_tilt_deg']:5.1f}° | {r['mean_power_watt']:6.2f}W"
        )
    print("================================================================================")


if __name__ == "__main__":
    main()
