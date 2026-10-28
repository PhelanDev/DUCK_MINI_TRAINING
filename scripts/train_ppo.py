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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.env_config import MicroDuck8DoFConfig, _DEFAULT_CFG
from training.run_parallel_sim import ParallelSimRunner

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
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

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
        std = torch.exp(self.log_std).expand_as(mean)
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


def train_ppo(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    num_envs: int = 64,
    rollout_steps: int = 32,
    max_iterations: int = 100,
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
    dry_run: bool = False,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute PPO training pipeline using ParallelSimRunner."""
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    dev = torch.device(device_str)

    # In dry-run mode, configure fast 1-iteration lightweight execution
    if dry_run:
        num_envs = min(num_envs, 4)
        rollout_steps = min(rollout_steps, 4)
        max_iterations = 1
        num_epochs = 1
        if verbose:
            print("[PPO] Dry-run mode enabled: 1 iteration, 4 envs, 4 rollout steps.")

    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
    )

    obs_dim = runner.obs_dim
    action_dim = runner.action_dim
    hidden_dim = 256

    # Instantiate ActorCritic network & Adam optimizer
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    batch_size = num_envs * rollout_steps
    if mini_batch_size is None or mini_batch_size > batch_size:
        mini_batch_size = max(1, batch_size // 4)

    # Storage buffers
    obs_buffer = torch.zeros((rollout_steps, num_envs, obs_dim), dtype=torch.float32, device=dev)
    actions_buffer = torch.zeros((rollout_steps, num_envs, action_dim), dtype=torch.float32, device=dev)
    log_probs_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    rewards_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    dones_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    values_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)

    # Initial state
    current_obs_np = runner.reset()
    current_obs = torch.as_tensor(current_obs_np, dtype=torch.float32, device=dev)

    total_timesteps = 0
    start_time = time.perf_counter()
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "mean_reward": 0.0,
    }

    try:
        for iteration in range(1, max_iterations + 1):
            iter_start = time.perf_counter()

            # ------------------------------------------------------------------
            # Phase 1: Rollout Collection
            # ------------------------------------------------------------------
            model.eval()
            for step in range(rollout_steps):
                with torch.no_grad():
                    actions, log_probs, _, values = model.get_action_and_value(current_obs)

                # Send clipped actions to physics simulator
                actions_np = actions.cpu().numpy()
                actions_clamped = np.clip(actions_np, -1.0, 1.0)

                next_obs_np, rewards_np, dones_np, _ = runner.step(actions_clamped)

                # Store into buffers
                obs_buffer[step] = current_obs
                actions_buffer[step] = actions
                log_probs_buffer[step] = log_probs
                values_buffer[step] = values
                rewards_buffer[step] = torch.as_tensor(rewards_np, dtype=torch.float32, device=dev)
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
            update_count = 0

            for _ in range(num_epochs):
                np.random.shuffle(indices)
                for start_idx in range(0, num_samples, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, num_samples)
                    mb_inds = indices[start_idx:end_idx]

                    _, new_log_probs, entropy, new_values = model.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )

                    log_ratio = new_log_probs - b_log_probs[mb_inds]
                    ratio = torch.exp(log_ratio)

                    # PPO Clipped Surrogate Loss
                    mb_adv = b_advantages[mb_inds]
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Function Loss
                    mb_ret = b_returns[mb_inds]
                    value_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()

                    # Total Loss with Entropy Regularization
                    entropy_mean = entropy.mean()
                    total_loss = (
                        policy_loss
                        + value_loss_coef * value_loss
                        - entropy_coef * entropy_mean
                    )

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                    epoch_policy_loss += float(policy_loss.item())
                    epoch_value_loss += float(value_loss.item())
                    epoch_entropy += float(entropy_mean.item())
                    update_count += 1

            if update_count > 0:
                metrics["policy_loss"] = epoch_policy_loss / update_count
                metrics["value_loss"] = epoch_value_loss / update_count
                metrics["entropy"] = epoch_entropy / update_count
                metrics["total_loss"] = (
                    metrics["policy_loss"]
                    + value_loss_coef * metrics["value_loss"]
                    - entropy_coef * metrics["entropy"]
                )
            metrics["mean_reward"] = float(rewards_buffer.mean().item())

            iter_fps = batch_size / max(1e-6, (time.perf_counter() - iter_start))
            if verbose and (iteration % max(1, max_iterations // 10) == 0 or iteration == 1):
                print(
                    f"[Iteration {iteration:03d}/{max_iterations}] "
                    f"Reward: {metrics['mean_reward']:+.3f} | "
                    f"PiLoss: {metrics['policy_loss']:.4f} | "
                    f"ValLoss: {metrics['value_loss']:.4f} | "
                    f"FPS: {iter_fps:.0f}"
                )

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
            "iterations": max_iterations,
            "total_timesteps": total_timesteps,
            "elapsed_seconds": elapsed_total,
            "metrics": metrics,
        }
        torch.save(checkpoint, str(save_path))
        if verbose:
            print(f"[PPO] Checkpoint saved successfully: {save_path} ({total_timesteps} steps in {elapsed_total:.2f}s)")

    finally:
        runner.close()

    return {
        "success": True,
        "iterations": max_iterations,
        "total_timesteps": total_timesteps,
        "checkpoint_path": str(save_path),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="MicroDuck 8-DOF PPO Training Pipeline")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to MuJoCo MJCF model")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_SAVE_PATH), help="Path to save PyTorch checkpoint")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel MuJoCo environments (default: 64)")
    parser.add_argument("--rollout-steps", type=int, default=32, help="Rollout steps per iteration (default: 32)")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum PPO training iterations (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam optimizer learning rate (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter (default: 0.95)")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipping epsilon (default: 0.2)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy loss weight (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss weight (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick iteration for CI/CD testing")
    parser.add_argument("--device", type=str, default=None, help="Device to run on ('cpu', 'cuda')")

    args = parser.parse_args()

    train_ppo(
        model_path=args.model_path,
        save_path=args.save_path,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        dry_run=args.dry_run,
        device=args.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()
