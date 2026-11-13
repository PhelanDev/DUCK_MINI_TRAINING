"""Proximal Policy Optimization (PPO) Training Pipeline for MicroDuck 8-DOF.

Trains an Actor-Critic policy using batched MuJoCo simulations via ParallelSimRunner.
Integrates 42-dimensional perception-action observation space (IMU, commands, kinematics,
foot contacts, and 5-ray LiDAR rangefinders) and 8-DOF joint position actuation.
"""

from pathlib import Path
import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.env_config import MicroDuck8DoFConfig, _DEFAULT_CFG
from training.run_parallel_sim import ParallelSimRunner
from training.terrain_curriculum import (
    STAGES,
    STAIR_STAGES,
    STAGE_MIN_DISTANCE_X,
    apply_stage,
    apply_stair_stage,
    next_stage,
    next_stair_stage,
)

DEFAULT_MODEL_PATH = (
    REPO_ROOT / "pipelines" / "rpi5_stack" / "models" / "mjcf" / "microduck_rpi5_8dof.xml"
)
DEFAULT_SAVE_PATH = REPO_ROOT / "models" / "checkpoints" / "microduck_ppo.pt"


class ActorCritic(nn.Module):
    """Actor-Critic neural network for MicroDuck 8-DOF bipedal robot.

    Architecture:
    - Input: 42-dimensional observation vector
    - Actor: 42 -> 256 (ELU) -> 256 (ELU) -> 8 (joint actions)
    - Critic: 42 -> 256 (ELU) -> 256 (ELU) -> 1 (state value V(s))
    - Action noise: Learnable log_std parameter for continuous action distribution
    """

    def __init__(
        self,
        obs_dim: int = 42,
        action_dim: int = 8,
        hidden_dim: int = 256,
        init_noise_std: float = 0.20,
        min_log_std: float = -3.0,
        max_log_std: float = 0.0,
    ):
        super().__init__()
        if min_log_std > max_log_std:
            raise ValueError(
                f"min_log_std ({min_log_std}) must be <= max_log_std ({max_log_std})"
            )
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        # Actor Network (Mean Action)
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        # Learnable log standard deviation for Gaussian policy
        init_log_std = math.log(max(1e-4, float(init_noise_std)))
        init_log_std = max(self.min_log_std, min(self.max_log_std, init_log_std))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), init_log_std, dtype=torch.float32))

        # Critic Network (State Value V(s))
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialization for stable PPO training."""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Final actor layer with smaller gain for initial near-zero actions
        if isinstance(self.actor[-1], nn.Linear):
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Final critic layer with gain 1.0
        if isinstance(self.critic[-1], nn.Linear):
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute deterministic actor action and critic value."""
        return self.forward_actor(obs), self.forward_critic(obs)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic policy forward pass (mean action)."""
        return self.actor(obs)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimation forward pass."""
        val = self.critic(obs)
        if val.ndim > 1 and val.shape[-1] == 1:
            val = val.squeeze(-1)
        return val

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from policy distribution or evaluate given actions.

        Returns:
            (actions, log_probabilities, entropy, values)
        """
        mean = self.actor(obs)
        clamped_log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(clamped_log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.forward_critic(obs)

        return action, log_prob, entropy, value


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and returns.

    Args:
        rewards: Tensor of shape (num_steps, num_envs)
        dones: Tensor of shape (num_steps, num_envs)
        values: Tensor of shape (num_steps, num_envs)
        next_value: Tensor of shape (num_envs,) bootstrap value
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter

    Returns:
        (advantages, returns) both of shape (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=rewards.device)
    last_gae = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def apply_timeout_bootstrap(
    rewards: torch.Tensor,
    time_outs: torch.Tensor,
    terminal_values: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Bootstrap value into reward for time-limit truncations.

    When an environment reaches its time limit (time_outs=True), it is truncated
    rather than reaching an absorbing terminal failure state. Adding gamma * V(s_terminal)
    to the immediate reward ensures continuation value is accounted for in GAE returns.

    Args:
        rewards: Tensor of shape (num_envs,)
        time_outs: Boolean tensor of shape (num_envs,)
        terminal_values: Critic estimates for timed-out envs (shape (num_timeouts,) or (num_envs,))
        gamma: Discount factor

    Returns:
        Adjusted rewards tensor with bootstrapped values added at timed-out indices.
    """
    out = rewards.clone()
    if not time_outs.any():
        return out

    vals = terminal_values.squeeze(-1) if terminal_values.dim() > 1 else terminal_values
    num_timeouts = int(time_outs.sum().item())
    if vals.numel() == num_timeouts:
        out[time_outs] += gamma * vals.to(rewards.device, dtype=rewards.dtype)
    elif vals.numel() == rewards.numel():
        out[time_outs] += gamma * vals[time_outs].to(rewards.device, dtype=rewards.dtype)
    else:
        raise ValueError(f"Shape mismatch: {vals.shape} vs num timeouts {num_timeouts}")
    return out


def train_ppo(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    num_envs: int = 64,
    rollout_steps: int = 128,
    max_iterations: Optional[int] = None,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    num_epochs: int = 4,
    mini_batch_size: Optional[int] = None,
    save_path: Union[str, Path] = DEFAULT_SAVE_PATH,
    best_save_path: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    device: Optional[str] = None,
    verbose: bool = True,
    start_stage: int = 0,
    hold_iters: int = 10,
    resume: bool = False,
    load_checkpoint: Optional[Union[str, Path]] = None,
    terrain_mode: str = "hurdles",
    log_interval: int = 50,
    save_interval: int = 250,
    stand_first: bool = False,
    friction_min: float = 1.1,
    friction_max: float = 1.6,
    target_kl: float = 0.015,
    min_lr: float = 1e-5,
    max_lr: float = 1e-3,
    entropy_coef_end: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute PPO training pipeline using ParallelSimRunner."""
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if device.startswith("cuda") and not torch.cuda.is_available():
            if verbose:
                print("[WARNING] PyTorch is not compiled with CUDA on this environment. Falling back to CPU.")
            device_str = "cpu"
        else:
            device_str = device
    dev = torch.device(device_str)

    current_stage = int(start_stage)
    hold_count = 0

    if max_iterations is None:
        max_iterations = 1 if dry_run else 100

    # In dry-run mode, configure fast lightweight execution
    if dry_run:
        num_envs = min(num_envs, 4)
        rollout_steps = min(rollout_steps, 4)
        max_iterations = max(1, max_iterations)
        num_epochs = 1
        if verbose:
            print(f"[PPO] Dry-run mode enabled: {max_iterations} iteration(s), {num_envs} envs, {rollout_steps} rollout steps.")

    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_prefix = save_path.stem[:-4] if save_path.stem.endswith("_ppo") else save_path.stem
    if best_save_path is None:
        best_save_path = save_path.parent / f"{ckpt_prefix}_best.pt"
    else:
        best_save_path = Path(best_save_path).resolve()
        best_save_path.parent.mkdir(parents=True, exist_ok=True)

    latest_save_path = save_path.parent / f"{ckpt_prefix}_latest.pt"

    # Historical best tracking
    best_reward = -float("inf")
    best_dist_x = -float("inf")
    best_stage = current_stage
    has_prev_best = False

    if best_save_path.exists():
        try:
            prev_best = torch.load(str(best_save_path), map_location="cpu", weights_only=False)
            best_reward = float(
                prev_best.get(
                    "best_reward",
                    prev_best.get("metrics", {}).get("mean_reward", -float("inf")),
                )
            )
            best_dist_x = float(
                prev_best.get(
                    "best_distance_x",
                    prev_best.get("metrics", {}).get("mean_distance_x", -float("inf")),
                )
            )
            best_stage = int(prev_best.get("best_stage", prev_best.get("stage", current_stage)))
            has_prev_best = True
        except Exception:
            pass

    # Initialize environment
    sim_model_path = Path(model_path).resolve()
    if not sim_model_path.exists():
        # Fallback search if path is relative
        fallback = REPO_ROOT / model_path
        if fallback.exists():
            sim_model_path = fallback
        else:
            raise FileNotFoundError(f"MuJoCo model not found at {sim_model_path}")

    runner = ParallelSimRunner(
        model_path=sim_model_path,
        num_envs=num_envs,
        decimation=10,
        stage=current_stage,
        terrain_mode=terrain_mode,
        stand_still=stand_first,
        friction_range=(friction_min, friction_max),
    )

    obs_dim = runner.obs_dim
    action_dim = runner.action_dim
    hidden_dim = 256

    # Instantiate ActorCritic network & Adam optimizer
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(dev)
    current_lr = float(learning_rate)
    eff_entropy_end = (
        float(entropy_coef_end)
        if entropy_coef_end is not None
        else min(float(entropy_coef), 0.002)
    )
    optimizer = optim.Adam(model.parameters(), lr=current_lr, eps=1e-5)

    # Resume from checkpoint if requested or exists
    ckpt_to_load = None
    if load_checkpoint is not None:
        ckpt_to_load = Path(load_checkpoint).resolve()
    elif resume and save_path.exists():
        ckpt_to_load = save_path

    loaded_iters = 0
    total_timesteps = 0
    if ckpt_to_load is not None:
        if ckpt_to_load.exists():
            ckpt = torch.load(str(ckpt_to_load), map_location=dev, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif "actor_state_dict" in ckpt:
                model.actor.load_state_dict(ckpt["actor_state_dict"])
                if "critic_state_dict" in ckpt:
                    model.critic.load_state_dict(ckpt["critic_state_dict"])
            if "optimizer_state_dict" in ckpt and resume:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            loaded_stage = ckpt.get("stage", current_stage)
            if loaded_stage > current_stage:
                current_stage = loaded_stage
                runner.set_stage(current_stage)
            if resume:
                loaded_iters = int(ckpt.get("iterations", 0))
                total_timesteps = int(ckpt.get("total_timesteps", 0))
                if "best_reward" in ckpt:
                    best_reward = max(best_reward, float(ckpt["best_reward"]))
                if "best_distance_x" in ckpt:
                    best_dist_x = max(best_dist_x, float(ckpt["best_distance_x"]))
                if not has_prev_best and "best_stage" in ckpt:
                    best_stage = int(ckpt["best_stage"])
                if "learning_rate" in ckpt:
                    current_lr = float(ckpt["learning_rate"])
                    if target_kl > 0:
                        current_lr = min(max_lr, max(min_lr, current_lr))
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
            if verbose:
                mode_label = "Resumed training" if resume else "Transferred policy"
                print(f"[PPO] {mode_label} from checkpoint: {ckpt_to_load} (Stage {current_stage}, {total_timesteps} previous timesteps, {loaded_iters} iters)")
        else:
            if verbose:
                print(f"[PPO] Checkpoint {ckpt_to_load} not found. Starting training from scratch.")

    batch_size = num_envs * rollout_steps
    if mini_batch_size is None or mini_batch_size > batch_size:
        mini_batch_size = max(1, batch_size // 4)

    # Storage buffers
    obs_buffer = torch.zeros((rollout_steps, num_envs, obs_dim), dtype=torch.float32, device=dev)
    actions_buffer = torch.zeros((rollout_steps, num_envs, action_dim), dtype=torch.float32, device=dev)
    log_probs_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    raw_rewards_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    rewards_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    dones_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    values_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)

    # Initial state
    current_obs_np = runner.reset()
    current_obs = torch.as_tensor(current_obs_np, dtype=torch.float32, device=dev)

    start_time = time.perf_counter()
    with torch.no_grad():
        initial_action_std = float(
            torch.exp(torch.clamp(model.log_std, model.min_log_std, model.max_log_std)).mean().item()
        )

    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "mean_reward": 0.0,
        "approx_kl": 0.0,
        "learning_rate": current_lr,
        "entropy_coef": float(entropy_coef),
        "action_std_mean": initial_action_std,
    }

    start_iteration = loaded_iters + 1
    total_target_iters = loaded_iters + max_iterations

    try:
        for iteration in range(start_iteration, total_target_iters + 1):
            iter_start = time.perf_counter()

            progress = (
                0.0
                if total_target_iters <= 1
                else min(1.0, max(0.0, (iteration - 1) / (total_target_iters - 1)))
            )
            current_entropy_coef = float(entropy_coef) + (eff_entropy_end - float(entropy_coef)) * progress

            # Smooth velocity warmup over first 15 iterations: stabilize early stepping balance
            if not stand_first and iteration <= 15:
                max_cmd_vx = 0.15 + 0.35 * (iteration / 15.0)
                runner.commands[:, 0] = np.clip(runner.commands[:, 0], 0.08, max_cmd_vx)

            # ------------------------------------------------------------------
            # Phase 1: Rollout Collection
            # ------------------------------------------------------------------
            model.eval()
            step_lin_vels: List[float] = []
            for step in range(rollout_steps):
                with torch.no_grad():
                    actions, log_probs, _, values = model.get_action_and_value(current_obs)

                # Send clipped actions to physics simulator
                actions_np = actions.cpu().numpy()
                actions_clamped = np.clip(actions_np, -1.0, 1.0)

                if hasattr(runner, "step_arrays"):
                    next_obs_np, rewards_np, dones_np, step_info = runner.step_arrays(actions_clamped)
                    step_lin_vels.append(float(step_info["tracking_lin_vel"]))
                    raw_step_rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=dev)
                    raw_rewards_buffer[step] = raw_step_rewards

                    time_outs_np = step_info["time_outs"]
                    if np.any(time_outs_np):
                        timeout_mask = torch.as_tensor(time_outs_np, dtype=torch.bool, device=dev)
                        terminal_obs_tensor = torch.as_tensor(step_info["terminal_obs"], dtype=torch.float32, device=dev)
                        with torch.no_grad():
                            terminal_values = model.forward_critic(terminal_obs_tensor)
                        step_rewards = apply_timeout_bootstrap(
                            raw_step_rewards, timeout_mask, terminal_values, gamma=gamma
                        )
                    else:
                        step_rewards = raw_step_rewards
                else:
                    next_obs_np, rewards_np, dones_np, infos = runner.step(actions_clamped)
                    step_lin_vels.append(float(np.mean([info.get("tracking_lin_vel", 0.0) for info in infos])))
                    raw_step_rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=dev)
                    raw_rewards_buffer[step] = raw_step_rewards

                    step_rewards = raw_step_rewards.clone()
                    timeout_mask = torch.tensor(
                        [info.get("time_outs", False) for info in infos],
                        dtype=torch.bool,
                        device=dev,
                    )
                    if timeout_mask.any():
                        timeout_indices = torch.where(timeout_mask)[0]
                        terminal_obs_list = [infos[idx.item()]["terminal_obs"] for idx in timeout_indices]
                        terminal_obs_tensor = torch.as_tensor(
                            np.stack(terminal_obs_list), dtype=torch.float32, device=dev
                        )
                        with torch.no_grad():
                            terminal_values = model.forward_critic(terminal_obs_tensor)
                        step_rewards = apply_timeout_bootstrap(
                            step_rewards, timeout_mask, terminal_values, gamma=gamma
                        )

                # Store into buffers
                obs_buffer[step] = current_obs
                actions_buffer[step] = actions
                log_probs_buffer[step] = log_probs
                values_buffer[step] = values
                rewards_buffer[step] = step_rewards
                dones_buffer[step] = torch.as_tensor(dones_np, dtype=torch.float32, device=dev)

                current_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=dev)
                total_timesteps += num_envs

            # Compute bootstrap value for the final step
            with torch.no_grad():
                next_value = model.forward_critic(current_obs)

            # ------------------------------------------------------------------
            # Phase 2: Advantage Estimation (GAE)
            # ------------------------------------------------------------------
            advantages, returns = compute_gae(
                rewards=rewards_buffer,
                dones=dones_buffer,
                values=values_buffer,
                next_value=next_value,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            # Flatten transitions
            b_obs = obs_buffer.reshape(-1, obs_dim)
            b_actions = actions_buffer.reshape(-1, action_dim)
            b_log_probs = log_probs_buffer.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # ------------------------------------------------------------------
            # Phase 3: PPO Optimization Epochs
            # ------------------------------------------------------------------
            model.train()
            num_samples = b_obs.shape[0]
            indices = np.arange(num_samples)

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            total_kl_sum = 0.0
            update_count = 0

            for _ in range(num_epochs):
                np.random.shuffle(indices)
                epoch_kl_sum = 0.0
                epoch_batches = 0
                for start_idx in range(0, num_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, num_samples)
                    mb_inds = indices[start_idx:end_idx]

                    _, new_log_probs, entropy, new_values = model.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )

                    log_ratio = new_log_probs - b_log_probs[mb_inds]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    kl_val = float(approx_kl.item())
                    epoch_kl_sum += kl_val
                    epoch_batches += 1
                    total_kl_sum += kl_val

                    # PPO Clipped Surrogate Loss
                    mb_adv = b_advantages[mb_inds]
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Function Loss with Huber / Smooth L1 (beta=2.0)
                    mb_ret = b_returns[mb_inds]
                    value_loss = 0.5 * F.smooth_l1_loss(new_values, mb_ret, beta=2.0)

                    # Total Loss with Entropy Regularization
                    entropy_mean = entropy.mean()
                    total_loss = (
                        policy_loss
                        + value_loss_coef * value_loss
                        - current_entropy_coef * entropy_mean
                    )

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(model.min_log_std, model.max_log_std)

                    epoch_policy_loss += float(policy_loss.item())
                    epoch_value_loss += float(value_loss.item())
                    epoch_entropy += float(entropy_mean.item())
                    update_count += 1

                epoch_mean_kl = epoch_kl_sum / max(1, epoch_batches)
                if target_kl > 0 and epoch_mean_kl > 2.0 * target_kl:
                    break

            if target_kl > 0 and update_count > 0:
                iter_mean_kl = total_kl_sum / update_count
                if iter_mean_kl > target_kl * 2.0:
                    current_lr = max(min_lr, current_lr / 1.5)
                elif 0.0 < iter_mean_kl < target_kl * 0.5:
                    current_lr = min(max_lr, current_lr * 1.5)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            if update_count > 0:
                metrics["policy_loss"] = epoch_policy_loss / update_count
                metrics["value_loss"] = epoch_value_loss / update_count
                metrics["entropy"] = epoch_entropy / update_count
                metrics["total_loss"] = (
                    metrics["policy_loss"]
                    + value_loss_coef * metrics["value_loss"]
                    - current_entropy_coef * metrics["entropy"]
                )
                metrics["approx_kl"] = total_kl_sum / update_count
            else:
                metrics["approx_kl"] = 0.0

            metrics["learning_rate"] = current_lr
            metrics["entropy_coef"] = current_entropy_coef
            with torch.no_grad():
                metrics["action_std_mean"] = float(
                    torch.exp(torch.clamp(model.log_std, model.min_log_std, model.max_log_std)).mean().item()
                )
            # Evaluate terrain & obstacle curriculum progression
            rollout_stage = current_stage
            mean_ep_len = runner.get_mean_episode_length()
            mean_dist_x = runner.get_mean_distance_x()
            mean_dist_y = runner.get_mean_distance_y() if hasattr(runner, "get_mean_distance_y") else 0.0
            stage_req_dist = STAGE_MIN_DISTANCE_X.get(current_stage, 0.8)
            cohort_success_rate = runner.get_success_rate(stage_req_dist)
            rew_lin_vel_ratio = float(np.mean(step_lin_vels)) if step_lin_vels else 0.0

            if terrain_mode == "stairs":
                new_stage, hold_count = next_stair_stage(
                    stage=current_stage,
                    ep_len_mean=mean_ep_len,
                    rew_vx_ratio=rew_lin_vel_ratio,
                    hold_count=hold_count,
                    hold_iters=hold_iters,
                    mean_distance_x=mean_dist_x,
                    min_distance_x=stage_req_dist,
                    success_rate=cohort_success_rate,
                    mean_distance_y=mean_dist_y,
                )
                if new_stage != current_stage:
                    if new_stage > rollout_stage:
                        current_lr = min(current_lr, float(learning_rate))
                        for pg in optimizer.param_groups:
                            pg["lr"] = current_lr
                        metrics["learning_rate"] = current_lr
                    current_stage = new_stage
                    runner.set_stage(current_stage)
                    if verbose:
                        print(f"[Curriculum] Advanced to Stair Stage {current_stage}: {STAIR_STAGES[current_stage].description}")
            else:
                new_stage, hold_count = next_stage(
                    stage=current_stage,
                    ep_len_mean=mean_ep_len,
                    rew_lin_vel_ratio=rew_lin_vel_ratio,
                    hold_count=hold_count,
                    hold_iters=hold_iters,
                    mean_distance_x=mean_dist_x,
                    min_distance_x=stage_req_dist,
                    success_rate=cohort_success_rate,
                    mean_distance_y=mean_dist_y,
                )
                if new_stage != current_stage:
                    if new_stage > rollout_stage:
                        current_lr = min(current_lr, float(learning_rate))
                        for pg in optimizer.param_groups:
                            pg["lr"] = current_lr
                        metrics["learning_rate"] = current_lr
                    current_stage = new_stage
                    runner.set_stage(current_stage)
                    if verbose:
                        print(f"[Curriculum] Advanced to Stage {current_stage}: {STAGES[current_stage].description}")

            # Evaluate stationary standing curriculum progression
            if stand_first and getattr(runner, "stand_still", False):
                if mean_ep_len >= 350.0 or iteration >= 25:
                    runner.set_stand_still(False)
                    if verbose:
                        print(
                            f"[Curriculum] Bipedal standing stability achieved (mean_ep_len={mean_ep_len:.1f}). "
                            "Unlocking walking velocity commands!"
                        )

            metrics["mean_reward"] = float(raw_rewards_buffer.mean().item())
            metrics["bootstrapped_mean_reward"] = float(rewards_buffer.mean().item())
            metrics["mean_distance_x"] = float(mean_dist_x)
            metrics["mean_distance_y"] = float(mean_dist_y)
            metrics["success_rate"] = float(cohort_success_rate)
            metrics["stage"] = current_stage
            metrics["hold_count"] = hold_count

            iter_fps = batch_size / max(1e-6, (time.perf_counter() - iter_start))
            print_interval = max(1, min(log_interval, max_iterations))
            if verbose and (iteration % print_interval == 0 or iteration == start_iteration or iteration == total_target_iters):
                print(
                    f"[Iteration {iteration:04d}/{total_target_iters}] "
                    f"Stage: {current_stage} | "
                    f"EpLen: {mean_ep_len:5.1f}/500 | "
                    f"DistX: {mean_dist_x:4.2f}m | "
                    f"Succ: {cohort_success_rate*100:4.1f}% | "
                    f"Reward: {metrics['mean_reward']:+.3f} | "
                    f"PiLoss: {metrics['policy_loss']:.4f} | "
                    f"ValLoss: {metrics['value_loss']:.4f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"LR: {metrics['learning_rate']:.1e} | "
                    f"Std: {metrics['action_std_mean']:.3f} | "
                    f"FPS: {iter_fps:.0f}",
                    flush=True,
                )

            # Check for historical best model (Reward and DistX peak without policy collapse)
            cur_reward = metrics["mean_reward"]
            cur_dist_x = metrics["mean_distance_x"]
            is_best = False

            if not dry_run and iteration > 0:
                if rollout_stage > best_stage and cur_dist_x >= 0.0 and (cur_dist_x > 0.0 or cur_reward > 0.0):
                    is_best = True
                    best_stage = rollout_stage
                    best_reward = cur_reward
                    best_dist_x = cur_dist_x
                elif rollout_stage == best_stage:
                    if (
                        (cur_reward >= best_reward and cur_dist_x >= best_dist_x)
                        and (cur_reward > best_reward or cur_dist_x > best_dist_x)
                    ) or (cur_dist_x > best_dist_x and cur_reward >= best_reward - 0.05):
                        is_best = True
                        best_reward = max(best_reward, cur_reward)
                        best_dist_x = max(best_dist_x, cur_dist_x)

            if is_best:
                elapsed_cur = time.perf_counter() - start_time
                best_ckpt = {
                    "model_state_dict": model.state_dict(),
                    "actor_state_dict": model.actor.state_dict(),
                    "critic_state_dict": model.critic.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "hidden_dim": hidden_dim,
                    "iterations": iteration,
                    "learning_rate": current_lr,
                    "stage": rollout_stage,
                    "best_stage": best_stage,
                    "terrain_mode": terrain_mode,
                    "total_timesteps": total_timesteps,
                    "elapsed_seconds": elapsed_cur,
                    "metrics": metrics,
                    "best_reward": best_reward,
                    "best_distance_x": best_dist_x,
                }
                torch.save(best_ckpt, str(best_save_path))
                if verbose:
                    print(
                        f"[PPO] [BEST] New best model saved: {best_save_path.name} "
                        f"(Reward: {cur_reward:+.3f}, DistX: {cur_dist_x:4.2f}m, Iter {iteration})"
                    )

            # Periodic checkpoint saving (latest model)
            if not dry_run and save_interval > 0 and (iteration % save_interval == 0):
                elapsed_cur = time.perf_counter() - start_time
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "actor_state_dict": model.actor.state_dict(),
                    "critic_state_dict": model.critic.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "hidden_dim": hidden_dim,
                    "iterations": iteration,
                    "learning_rate": current_lr,
                    "stage": current_stage,
                    "best_stage": best_stage,
                    "terrain_mode": terrain_mode,
                    "total_timesteps": total_timesteps,
                    "elapsed_seconds": elapsed_cur,
                    "metrics": metrics,
                    "best_reward": best_reward,
                    "best_distance_x": best_dist_x,
                }
                torch.save(ckpt, str(save_path))
                torch.save(ckpt, str(latest_save_path))
                if verbose:
                    print(f"[PPO] Periodic checkpoint saved: {save_path.name} & {latest_save_path.name} (iter {iteration})")

        # ------------------------------------------------------------------
        # Phase 4: Save Model Checkpoint
        # ------------------------------------------------------------------
        elapsed_total = time.perf_counter() - start_time
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "actor_state_dict": model.actor.state_dict(),
            "critic_state_dict": model.critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "iterations": total_target_iters,
            "learning_rate": current_lr,
            "stage": current_stage,
            "best_stage": best_stage,
            "terrain_mode": terrain_mode,
            "total_timesteps": total_timesteps,
            "elapsed_seconds": elapsed_total,
            "metrics": metrics,
            "best_reward": best_reward,
            "best_distance_x": best_dist_x,
        }
        torch.save(checkpoint, str(save_path))
        torch.save(checkpoint, str(latest_save_path))
        if not best_save_path.exists():
            torch.save(checkpoint, str(best_save_path))
        if verbose:
            print(f"[PPO] Checkpoint saved successfully: {save_path} ({total_timesteps} steps in {elapsed_total:.2f}s)")

    finally:
        runner.close()

    return {
        "success": True,
        "iterations": total_target_iters,
        "total_timesteps": total_timesteps,
        "checkpoint_path": str(save_path),
        "latest_checkpoint_path": str(latest_save_path),
        "best_checkpoint_path": str(best_save_path),
        "metrics": metrics,
        "stage": current_stage,
        "best_stage": best_stage,
    }


def main():
    parser = argparse.ArgumentParser(description="MicroDuck 8-DOF PPO Training Pipeline")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to MuJoCo MJCF model")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_SAVE_PATH), help="Path to save PyTorch checkpoint")
    parser.add_argument(
        "--best-save-path",
        type=str,
        default=None,
        help="Path to save best PyTorch checkpoint (default: microduck_best.pt in same directory as save-path)",
    )
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel MuJoCo environments (default: 64)")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Rollout steps per iteration (default: 128)")
    parser.add_argument("--max-iterations", "--iters", type=int, default=None, help="Maximum PPO training iterations (default: 100, or 1 in dry-run)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam optimizer learning rate (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter (default: 0.95)")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipping epsilon (default: 0.2)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy loss weight (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss weight (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Run quick lightweight iterations for CI/CD testing")
    parser.add_argument("--device", type=str, default=None, help="Device to run on ('cpu', 'cuda')")
    parser.add_argument("--start-stage", type=int, default=0, help="Initial terrain curriculum stage (0 to 3)")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing save-path checkpoint")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--terrain", type=str, choices=["hurdles", "stairs"], default="hurdles", help="Terrain mode: 'hurdles' or 'stairs'")
    parser.add_argument("--log-interval", type=int, default=50, help="Print training logs every N iterations (default: 50)")
    parser.add_argument("--save-interval", type=int, default=250, help="Save periodic checkpoint every N iterations (default: 250)")
    parser.add_argument("--stand-first", action="store_true", help="Start training with zero velocity commands (stand still curriculum) until stable")
    parser.add_argument("--friction-min", type=float, default=1.1, help="Minimum floor friction for domain randomization (default: 1.1)")
    parser.add_argument("--friction-max", type=float, default=1.6, help="Maximum floor friction for domain randomization (default: 1.6)")
    parser.add_argument("--target-kl", type=float, default=0.015, help="Target KL divergence for adaptive LR and epoch early-stop (<=0 to disable, default: 0.015)")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate for adaptive KL schedule (default: 1e-5)")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="Maximum learning rate for adaptive KL schedule (default: 1e-3)")
    parser.add_argument("--entropy-coef-end", type=float, default=None, help="Target entropy coefficient at final iteration (default: None, falls back to min(entropy_coef, 0.002))")

    args = parser.parse_args()

    max_iters = args.max_iterations
    if max_iters is None:
        max_iters = 1 if args.dry_run else 100

    train_ppo(
        model_path=args.model_path,
        save_path=args.save_path,
        best_save_path=args.best_save_path,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        max_iterations=max_iters,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        dry_run=args.dry_run,
        device=args.device,
        verbose=True,
        start_stage=args.start_stage,
        resume=args.resume,
        load_checkpoint=args.load_checkpoint,
        terrain_mode=args.terrain,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        stand_first=args.stand_first,
        friction_min=args.friction_min,
        friction_max=args.friction_max,
        target_kl=args.target_kl,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        entropy_coef_end=args.entropy_coef_end,
    )


if __name__ == "__main__":
    main()
