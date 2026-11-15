#!/usr/bin/env python3
"""Closed-loop Behavioral Cloning warmup for Actor and Critic initialization."""

import argparse
from pathlib import Path
import sys
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ppo import ActorCritic
from training.advanced_gaits import BipedalGaitGenerator
from training.env_config import unscale_actions
from training.run_parallel_sim import ParallelSimRunner


def _get_actor_in_dim(actor: nn.Module) -> int:
    for attr in ("obs_dim", "in_features"):
        if hasattr(actor, attr):
            return int(getattr(actor, attr))
    if hasattr(actor, "actor"):
        return _get_actor_in_dim(actor.actor)
    if hasattr(actor, "__getitem__"):
        return _get_actor_in_dim(actor[0])
    return 42


def _collect_rollouts(
    runner: ParallelSimRunner,
    policy_fn: Callable[[np.ndarray, int], np.ndarray],
    num_steps: int,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = runner.reset()
    traj_obs, traj_act, traj_rew, traj_done = [], [], [], []
    for s in range(num_steps):
        act = policy_fn(obs, s)
        next_obs, rews, dones, _ = runner.step(act, fast=True)
        traj_obs.append(obs.copy())
        traj_act.append(act.copy())
        traj_rew.append(rews.copy())
        traj_done.append(dones.copy())
        obs = next_obs

    obs_seq, act_seq = np.array(traj_obs, dtype=np.float32), np.array(traj_act, dtype=np.float32)
    rew_seq, done_seq = np.array(traj_rew, dtype=np.float32), np.array(traj_done, dtype=bool)

    t_steps, n_envs = rew_seq.shape
    returns, running_ret = np.zeros_like(rew_seq), np.zeros(n_envs, dtype=np.float32)
    for e in range(n_envs):
        if not done_seq[-1, e]:
            tail = rew_seq[-min(20, t_steps) :, e]
            running_ret[e] = float(np.mean(tail)) / max(1.0 - gamma, 1e-4)
    for t in reversed(range(t_steps)):
        running_ret = rew_seq[t] + gamma * running_ret * (1.0 - done_seq[t].astype(np.float32))
        returns[t] = running_ret

    valid = np.ones((t_steps, n_envs), dtype=bool)
    for e in range(n_envs):
        fallen = np.where(done_seq[:, e])[0]
        if len(fallen) > 0:
            valid[fallen[0]:, e] = False
    if not np.any(valid):
        # shortcut: fallback to non-done steps if all envs fell early, upgrade when robust recovery online
        valid = ~done_seq if np.any(~done_seq) else np.ones((t_steps, n_envs), dtype=bool)

    return obs_seq[valid], act_seq[valid], returns[valid]


def collect_gait_demonstrations(
    runner: ParallelSimRunner, num_steps: int = 200, freq: float = 1.25, gamma: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect closed-loop demonstrations using BipedalGaitGenerator."""
    gait, dt = BipedalGaitGenerator(default_freq=freq), float(runner.control_dt)

    def policy_fn(_obs: np.ndarray, step: int) -> np.ndarray:
        q = gait.compute_walk(step * dt, freq=freq, roll_amp=0.14, pitch_amp=0.32, knee_amp=0.40)
        raw_act = unscale_actions(q, runner.cfg, nominal_centered=True)
        return np.tile(raw_act, (runner.num_envs, 1))

    return _collect_rollouts(runner, policy_fn, num_steps, gamma)


def collect_teacher_demonstrations(
    runner: ParallelSimRunner, teacher_actor: nn.Module, num_steps: int = 200, gamma: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect closed-loop demonstrations from a pretrained teacher policy."""
    teacher_actor.eval()
    in_dim = _get_actor_in_dim(teacher_actor)

    def policy_fn(obs: np.ndarray, _step: int) -> np.ndarray:
        feed = obs[:, :in_dim] if obs.shape[1] >= in_dim else np.pad(obs, ((0, 0), (0, in_dim - obs.shape[1])))
        with torch.no_grad():
            t_obs = torch.as_tensor(feed, dtype=torch.float32)
            out = teacher_actor.forward_actor(t_obs) if hasattr(teacher_actor, "forward_actor") else teacher_actor(t_obs)
            if isinstance(out, tuple):
                out = out[0]
            act = out.detach().cpu().numpy()
        return np.clip(act, -1.0, 1.0).astype(np.float32)

    return _collect_rollouts(runner, policy_fn, num_steps, gamma)


def train_behavior_cloning(
    obs: np.ndarray, actions: np.ndarray, returns: Optional[np.ndarray] = None,
    obs_dim: int = 42, action_dim: int = 8, hidden_dim: int = 256,
    epochs: int = 30, batch_size: int = 256, lr: float = 1e-3, device: str = "cpu",
) -> Tuple[ActorCritic, List[float]]:
    """Train Actor (MSE) and Critic (SmoothL1) to avoid cold critic shock at PPO iteration 1."""
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device) if returns is not None else None

    dataset_size, epoch_losses = len(obs), []
    for ep in range(1, epochs + 1):
        perm, total_loss, batches = torch.randperm(dataset_size, device=device), 0.0, 0
        for i in range(0, dataset_size, batch_size):
            idx = perm[i : i + batch_size]
            b_obs, b_act = obs_t[idx], act_t[idx]
            loss = F.mse_loss(model.forward_actor(b_obs), b_act)
            if ret_t is not None:
                loss = loss + 0.5 * F.smooth_l1_loss(model.forward_critic(b_obs), ret_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        epoch_losses.append(total_loss / max(1, batches))
    with torch.no_grad():
        model.log_std.data.fill_(-1.90)
    return model, epoch_losses


def save_bc_checkpoint(
    path: Union[str, Path], model: ActorCritic, obs_dim: int, action_dim: int, hidden_dim: int = 256
) -> Path:
    """Save ActorCritic weights into a PPO-compatible warm start checkpoint."""
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "actor_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "hidden_dim": int(hidden_dim),
        "stage": 0,
        "iterations": 0,
    }
    torch.save(payload, str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Closed-Loop BC Pretraining & Critic Warmup")
    parser.add_argument("--mode", choices=["gait", "teacher"], default="gait")
    parser.add_argument("--model-path", default="duck_v2/cyberduck_v2.xml")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/checkpoints/cyberduck_v2_bc_init.pt")
    args = parser.parse_args()

    runner = ParallelSimRunner(model_path=args.model_path, num_envs=args.num_envs, num_threads=1, stage=0)
    try:
        if args.mode == "gait":
            obs, acts, rets = collect_gait_demonstrations(runner, num_steps=args.steps)
        else:
            if not args.teacher_checkpoint:
                raise ValueError("--teacher-checkpoint is required when --mode=teacher")
            ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
            teacher = ActorCritic(obs_dim=ckpt.get("obs_dim", 37), action_dim=ckpt.get("action_dim", 8))
            if "actor_state_dict" in ckpt:
                teacher.actor.load_state_dict(ckpt["actor_state_dict"])
            elif "model_state_dict" in ckpt:
                teacher.load_state_dict(ckpt["model_state_dict"])
            obs, acts, rets = collect_teacher_demonstrations(runner, teacher, num_steps=args.steps)
    finally:
        runner.close()

    model, losses = train_behavior_cloning(
        obs, acts, returns=rets, obs_dim=runner.obs_dim, action_dim=runner.action_dim,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    out_path = save_bc_checkpoint(args.output, model, runner.obs_dim, runner.action_dim)
    print(f"Pretrained BC model saved to {out_path} (final loss: {losses[-1]:.4f})")


if __name__ == "__main__":
    main()
