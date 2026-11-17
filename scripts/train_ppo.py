"""Proximal Policy Optimization (PPO) Training Pipeline for MicroDuck 8-DOF.

Trains an Actor-Critic policy using batched MuJoCo simulations via ParallelSimRunner.
Integrates 42-dimensional perception-action observation space (IMU, commands, kinematics,
foot contacts, and 5-ray LiDAR rangefinders) and 8-DOF joint position actuation.
"""

from pathlib import Path
import argparse
import math
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

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


class RunningMeanStd:
    """Tracks running mean and variance using parallel Welford algorithm."""

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...]] = (),
        epsilon: float = 1e-4,
        clip: float = 10.0,
        min_std: float = 0.10,
    ):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.min_std = float(min_std)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: Union[np.ndarray, torch.Tensor]) -> None:
        """Update running statistics with a batch of observations."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        batch_size = x.size // np.prod(self.shape) if len(self.shape) > 0 else x.size
        if batch_size == 0:
            return
        x_flat = x.reshape(batch_size, *self.shape)

        batch_mean = np.mean(x_flat, axis=0)
        batch_var = np.var(x_flat, axis=0)
        batch_count = float(batch_size)

        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / new_count)
        self.var = (
            self.count * self.var
            + batch_count * batch_var
            + (delta ** 2) * (self.count * batch_count / new_count)
        ) / new_count
        self.count = new_count

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize observation x: (x - mean) / max(sqrt(var + eps), min_std), clipped to [-clip, clip]."""
        if isinstance(x, torch.Tensor):
            mean_t = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            var_t = torch.as_tensor(self.var, dtype=x.dtype, device=x.device)
            std_t = torch.clamp(torch.sqrt(var_t + self.epsilon), min=self.min_std)
            norm = (x - mean_t) / std_t
            if self.clip > 0:
                norm = torch.clamp(norm, -self.clip, self.clip)
            return norm
        else:
            x_arr = np.asarray(x, dtype=np.float32)
            std = np.maximum(np.sqrt(self.var + self.epsilon).astype(np.float32), self.min_std)
            norm = (x_arr - self.mean.astype(np.float32)) / std
            if self.clip > 0:
                norm = np.clip(norm, -self.clip, self.clip)
            return norm

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "shape": self.shape,
            "clip": self.clip,
            "epsilon": self.epsilon,
            "min_std": self.min_std,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.mean = np.asarray(state_dict["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state_dict["var"], dtype=np.float64).copy()
        self.count = float(state_dict["count"])
        if "clip" in state_dict:
            self.clip = float(state_dict["clip"])
        if "epsilon" in state_dict:
            self.epsilon = float(state_dict["epsilon"])
        if "min_std" in state_dict:
            self.min_std = float(state_dict["min_std"])


class ActorCritic(nn.Module):
    """Actor-Critic neural network for MicroDuck 8-DOF bipedal robot.

    Architecture:
    - Input: Actor receives obs_dim (42), Critic receives critic_obs_dim (e.g. 53 or 42)
    - Actor: obs_dim -> 256 (ELU) -> 256 (ELU) -> 8 (joint actions)
    - Critic: critic_obs_dim -> 256 (ELU) -> 256 (ELU) -> 1 (state value V(s))
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
        critic_obs_dim: Optional[int] = None,
        squash_actions: bool = False,
    ):
        super().__init__()
        if min_log_std > max_log_std:
            raise ValueError(
                f"min_log_std ({min_log_std}) must be <= max_log_std ({max_log_std})"
            )
        self.obs_dim = int(obs_dim)
        self.critic_obs_dim = int(critic_obs_dim) if critic_obs_dim is not None else self.obs_dim
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.squash_actions = bool(squash_actions)

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
            nn.Linear(self.critic_obs_dim, self.hidden_dim),
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

    def forward(self, obs: torch.Tensor, critic_obs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute deterministic actor action and critic value."""
        return self.forward_actor(obs), self.forward_critic(critic_obs if critic_obs is not None else obs)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic policy forward pass (mean action)."""
        raw = self.actor(obs)
        return torch.tanh(raw) if self.squash_actions else raw

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimation forward pass."""
        if obs.shape[-1] < self.critic_obs_dim:
            pad = torch.zeros(*obs.shape[:-1], self.critic_obs_dim - obs.shape[-1], device=obs.device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)
        val = self.critic(obs)
        if val.ndim > 1 and val.shape[-1] == 1:
            val = val.squeeze(-1)
        return val

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        critic_obs: Optional[torch.Tensor] = None,
        pre_tanh_action: Optional[torch.Tensor] = None,
        return_pre_tanh: bool = False,
    ) -> tuple:
        """Sample actions from policy distribution or evaluate given actions.

        Returns:
            (actions, log_probabilities, entropy, values)
        """
        mean = self.actor(obs)
        clamped_log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(clamped_log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        if self.squash_actions:
            if pre_tanh_action is not None:
                latent = pre_tanh_action.detach()
                expected = torch.tanh(latent)
                if action is not None and not torch.allclose(action, expected, atol=1e-6, rtol=1e-6):
                    raise ValueError("action does not match pre_tanh_action")
                action = expected
            elif action is None:
                latent = dist.sample()
                action = torch.tanh(latent)
            else:
                if not torch.isfinite(action).all() or torch.any(action.abs() >= 1):
                    raise ValueError("Saturated actions require pre_tanh_action")
                latent = torch.atanh(action.detach())
            log_jacobian = 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
            log_prob = (dist.log_prob(latent) - log_jacobian).sum(dim=-1)
        else:
            action = dist.sample() if action is None else action
            latent = action.detach()
            log_prob = dist.log_prob(action).sum(dim=-1)
        # Regularize the base Gaussian; this is not transformed-policy entropy.
        entropy = dist.entropy().sum(dim=-1)
        c_obs = critic_obs if critic_obs is not None else obs
        value = self.forward_critic(c_obs)

        result = (action, log_prob, entropy, value)
        return (*result, latent.detach()) if return_pre_tanh else result

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Adaptive load_state_dict to smoothly support checkpoints with different critic_obs_dim."""
        if "critic.0.weight" in state_dict:
            c_in = state_dict["critic.0.weight"].shape[1]
            if c_in != self.critic[0].in_features:
                dev = self.critic[0].weight.device
                dtype = self.critic[0].weight.dtype
                self.critic[0] = nn.Linear(c_in, self.hidden_dim, device=dev, dtype=dtype)
                self.critic_obs_dim = c_in
        return super().load_state_dict(state_dict, strict=strict)


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


def deterministic_eval_protocol(model_path, terrain_mode, obs_dim, squash_actions):
    """Identity of the fixed benchmark used for walking checkpoint selection."""
    path = Path(model_path).resolve()
    return {
        "version": 1, "commands": [0.18, 0.22, 0.25], "seeds": [0, 1, 2],
        "friction": 1.2, "max_steps": 250, "control_dt": 0.02,
        "model_path": path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path),
        "terrain_mode": terrain_mode, "obs_dim": obs_dim,
        "squash_actions": bool(squash_actions), "mode": "walk",
    }


def select_deterministic_candidate(candidate, previous):
    """Return eligibility and reason without mutating either selection record."""
    if not all(math.isfinite(candidate[k]) for k in ("det_distance_x", "det_reward")):
        return False, "non_finite_evaluation"
    if candidate["det_distance_x"] < 0.05:
        return False, "below_walking_threshold"
    if previous is None:
        return True, "first_qualified_candidate"
    if previous["protocol"] != candidate["protocol"]:
        return False, "incompatible_evaluation_protocol"
    if candidate["stage"] < previous["stage"]:
        return False, "lower_stage"
    if candidate["stage"] > previous["stage"]:
        return True, "qualified_higher_stage"
    improved = (candidate["det_distance_x"], candidate["det_reward"]) > (
        previous["det_distance_x"], previous["det_reward"])
    return improved, "deterministic_improvement" if improved else "no_deterministic_improvement"


def evaluate_deterministic_rollout(
    model: ActorCritic,
    obs_rms: RunningMeanStd,
    model_path: Union[str, Path],
    stage: int,
    terrain_mode: str = "hurdles",
    stand_still: bool = False,
    max_steps: int = 250,
    device: Union[str, torch.device] = "cpu",
    cmd_vx: Optional[float] = None,
    eval_runner: Optional[ParallelSimRunner] = None,
    num_episodes: int = 3,
    commands: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
) -> Tuple[float, float]:
    """Mean final displacement and mean episode step reward on fixed resets."""
    if num_episodes < 1 or max_steps < 1:
        raise ValueError("Evaluation requires positive episode and step counts")
    commands = list(commands) if commands is not None else (
        [float(cmd_vx)] * num_episodes if cmd_vx is not None else
        [0.18, 0.22, 0.25] if num_episodes == 3 else [0.22] * num_episodes)
    seeds = list(seeds) if seeds is not None else list(range(num_episodes))
    if len(commands) != num_episodes or len(seeds) != num_episodes:
        raise ValueError("Evaluation command/seed schedule must match num_episodes")
    owns_runner = eval_runner is None
    runner = eval_runner if eval_runner is not None else ParallelSimRunner(
        model_path=str(model_path), num_envs=1, decimation=10, stage=stage,
        terrain_mode=terrain_mode, stand_still=stand_still,
        friction_range=(1.2, 1.2), randomize_pillars=False,
    )
    if runner.num_envs != 1:
        raise ValueError("Deterministic evaluation requires a dedicated single-env runner")
    prior_rng = runner.rng
    prior_friction = runner.friction_range
    was_training = model.training
    model.eval()
    try:
        if runner.curriculum_stage != stage:
            runner.set_stage(stage)
        runner.stand_still = bool(stand_still)
        runner.friction_range = (1.2, 1.2)
        distances, episode_rewards = [], []
        for command, seed in zip(commands, seeds):
            runner.rng = np.random.default_rng(seed)
            obs_raw = runner.reset()
            runner.commands[0] = [0.0 if stand_still else command, 0.0, 0.0]
            obs_raw[0] = runner._extract_obs(0)
            start_x = float(runner.datas[0].qpos[0])
            end_x, total_reward, steps_run = start_x, 0.0, 0
            for _ in range(max_steps):
                obs_t = torch.as_tensor(obs_rms.normalize(obs_raw), dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = model.forward_actor(obs_t).cpu().numpy()
                if not model.squash_actions:
                    action = np.clip(action, -1.0, 1.0)
                obs_raw, rewards, dones, info = runner.step_arrays(action)
                total_reward += float(rewards[0])
                steps_run += 1
                if bool(dones[0]):
                    # Terminal coordinates must be captured before the runner auto-reset.
                    end_x = float(info["terminal_distances_x"][0])
                    break
                end_x = float(runner.datas[0].qpos[0])
                runner.commands[0] = [0.0 if stand_still else command, 0.0, 0.0]
                obs_raw[0] = runner._extract_obs(0)
            distances.append(end_x - start_x)
            episode_rewards.append(total_reward / steps_run)
        return float(np.mean(distances)), float(np.mean(episode_rewards))
    finally:
        runner.rng = prior_rng
        runner.friction_range = prior_friction
        model.train(was_training)
        if owns_runner:
            runner.close()


def _atomic_torch_save(payload: Dict[str, Any], target_path: Union[str, Path]) -> None:
    """Save PyTorch checkpoint payload atomically using a temporary file and os.replace.

    Avoids partial/corrupt checkpoint reads when external processes (such as
    interactive_viewer.py) monitor and hot-reload checkpoint files.
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def train_ppo(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    num_envs: int = 64,
    rollout_steps: int = 128,
    max_iterations: Optional[int] = None,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.002,
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
    hold_iters: int = 5,
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
    max_log_std: float = -1.05,
    randomize_pillars: bool = False,
    pillar_density: float = 1.0,
    pillar_jitter: float = 0.06,
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
    if best_save_path in (save_path, latest_save_path):
        raise ValueError("best_save_path must be separate from progress checkpoint paths")
    if load_checkpoint is not None and not resume:
        source_path = Path(load_checkpoint).resolve()
        if source_path in (save_path, latest_save_path, best_save_path):
            raise ValueError("Transfer requires new output paths; do not overwrite the source checkpoint")

    # The actual best file is the only authority for historical selection.
    best_record = None
    best_stage = current_stage
    best_reward = best_dist_x = -float("inf")
    if resume and best_save_path.exists():
        try:
            previous = torch.load(str(best_save_path), map_location="cpu", weights_only=False)
            record = previous.get("best_selection")
            if (isinstance(record, dict) and isinstance(record.get("protocol"), dict)
                    and record["protocol"].get("mode") == "walk"
                    and all(math.isfinite(float(record[k])) for k in ("det_distance_x", "det_reward"))
                    and float(record["det_distance_x"]) >= 0.05):
                best_record = dict(record)
                best_stage = int(record["stage"])
                best_reward = float(record["mean_reward"])
                best_dist_x = float(record["mean_distance_x"])
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError):
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
        randomize_pillars=randomize_pillars,
        pillar_density=pillar_density,
        pillar_jitter_xy=pillar_jitter,
    )

    obs_dim = runner.obs_dim
    critic_obs_dim = getattr(runner, "privileged_obs_dim", obs_dim)
    action_dim = runner.action_dim
    hidden_dim = 256

    # Instantiate ActorCritic network & Adam optimizer
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        max_log_std=max_log_std,
        critic_obs_dim=critic_obs_dim,
        squash_actions=True,
    ).to(dev)

    obs_rms = RunningMeanStd(shape=obs_dim, clip=10.0)
    critic_obs_rms = RunningMeanStd(shape=critic_obs_dim, clip=10.0)

    def get_runner_critic_obs(raw_obs=None, env_indices=None):
        if hasattr(runner, "get_privileged_obs"):
            return runner.get_privileged_obs(raw_obs=raw_obs, env_indices=env_indices)
        if raw_obs is not None:
            return raw_obs if env_indices is None else raw_obs[env_indices]
        obs_list = [runner._extract_obs(i) for i in (range(runner.num_envs) if env_indices is None else env_indices)]
        return np.array(obs_list, dtype=np.float32)

    current_lr = float(learning_rate)
    eff_entropy_end = (
        float(entropy_coef_end)
        if entropy_coef_end is not None
        else min(float(entropy_coef), 0.0005)
    )
    optimizer = optim.Adam(model.parameters(), lr=current_lr, eps=1e-5)

    # Resume from checkpoint if requested or exists
    ckpt_to_load = None
    if load_checkpoint is not None:
        ckpt_to_load = Path(load_checkpoint).resolve()
    elif resume and save_path.exists():
        ckpt_to_load = save_path

    transfer_loaded = False
    optimizer_loaded = False
    loaded_iters = 0
    total_timesteps = 0
    if ckpt_to_load is not None:
        if ckpt_to_load.exists():
            ckpt = torch.load(str(ckpt_to_load), map_location=dev, weights_only=False)
            if resume:
                model.squash_actions = bool(ckpt.get("squash_actions", False))
            if "obs_rms" in ckpt:
                try:
                    obs_rms.load_state_dict(ckpt["obs_rms"])
                except Exception:
                    pass
            if "critic_obs_rms" in ckpt and ckpt.get("critic_obs_dim") == critic_obs_dim:
                try:
                    critic_obs_rms.load_state_dict(ckpt["critic_obs_rms"])
                except Exception:
                    pass

            critic_weight = None
            if "model_state_dict" in ckpt and "critic.0.weight" in ckpt["model_state_dict"]:
                critic_weight = ckpt["model_state_dict"]["critic.0.weight"]
            elif "critic_state_dict" in ckpt and "0.weight" in ckpt["critic_state_dict"]:
                critic_weight = ckpt["critic_state_dict"]["0.weight"]

            mismatched_critic = (critic_weight is not None and critic_weight.shape[1] != model.critic_obs_dim)

            model_loaded = False
            if mismatched_critic:
                if "actor_state_dict" in ckpt:
                    model.actor.load_state_dict(ckpt["actor_state_dict"])
                elif "model_state_dict" in ckpt:
                    actor_state = {k.replace("actor.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
                    model.actor.load_state_dict(actor_state)
                if "model_state_dict" in ckpt and "log_std" in ckpt["model_state_dict"]:
                    with torch.no_grad():
                        model.log_std.data.copy_(ckpt["model_state_dict"]["log_std"])

                shared_dim = min(critic_weight.shape[1], model.critic_obs_dim)
                c_dict = ckpt.get("critic_state_dict")
                m_dict = ckpt.get("model_state_dict")
                with torch.no_grad():
                    if c_dict is not None:
                        model.critic[0].weight[:, :shared_dim].copy_(c_dict["0.weight"][:, :shared_dim])
                        if "0.bias" in c_dict:
                            model.critic[0].bias.copy_(c_dict["0.bias"])
                        if "2.weight" in c_dict:
                            model.critic[2].weight.copy_(c_dict["2.weight"])
                        if "2.bias" in c_dict:
                            model.critic[2].bias.copy_(c_dict["2.bias"])
                        if "4.weight" in c_dict:
                            model.critic[4].weight.copy_(c_dict["4.weight"])
                        if "4.bias" in c_dict:
                            model.critic[4].bias.copy_(c_dict["4.bias"])
                    elif m_dict is not None:
                        model.critic[0].weight[:, :shared_dim].copy_(m_dict["critic.0.weight"][:, :shared_dim])
                        if "critic.0.bias" in m_dict:
                            model.critic[0].bias.copy_(m_dict["critic.0.bias"])
                        if "critic.2.weight" in m_dict:
                            model.critic[2].weight.copy_(m_dict["critic.2.weight"])
                        if "critic.2.bias" in m_dict:
                            model.critic[2].bias.copy_(m_dict["critic.2.bias"])
                        if "critic.4.weight" in m_dict:
                            model.critic[4].weight.copy_(m_dict["critic.4.weight"])
                        if "critic.4.bias" in m_dict:
                            model.critic[4].bias.copy_(m_dict["critic.4.bias"])
                model_loaded = True
                optimizer_loaded = False
            else:
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    model_loaded = True
                elif "actor_state_dict" in ckpt:
                    model.actor.load_state_dict(ckpt["actor_state_dict"])
                    if "critic_state_dict" in ckpt:
                        model.critic.load_state_dict(ckpt["critic_state_dict"])
                    model_loaded = True
                if "optimizer_state_dict" in ckpt and resume:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        optimizer_loaded = True
                    except Exception:
                        optimizer_loaded = False
            loaded_stage = ckpt.get("stage", current_stage)
            if loaded_stage > current_stage:
                current_stage = loaded_stage
                runner.set_stage(current_stage)
            if resume:
                loaded_iters = int(ckpt.get("iterations", 0))
                total_timesteps = int(ckpt.get("total_timesteps", 0))
                if "learning_rate" in ckpt:
                    current_lr = float(ckpt["learning_rate"])
                    if target_kl > 0:
                        current_lr = min(max_lr, max(min_lr, current_lr))
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
            elif load_checkpoint is not None and model_loaded:
                transfer_loaded = True
                raw_lr = ckpt.get("learning_rate")
                if raw_lr is not None:
                    try:
                        flt_lr = float(raw_lr)
                        if math.isfinite(flt_lr) and flt_lr > 0:
                            current_lr = min(current_lr, flt_lr)
                            if target_kl > 0:
                                current_lr = min(max_lr, max(min_lr, current_lr))
                            for pg in optimizer.param_groups:
                                pg["lr"] = current_lr
                    except (TypeError, ValueError):
                        pass
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
    critic_obs_buffer = torch.zeros((rollout_steps, num_envs, critic_obs_dim), dtype=torch.float32, device=dev)
    actions_buffer = torch.zeros((rollout_steps, num_envs, action_dim), dtype=torch.float32, device=dev)
    latent_buffer = torch.zeros_like(actions_buffer)
    log_probs_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    raw_rewards_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    rewards_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    dones_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)
    values_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=dev)

    # Initial state
    current_obs_raw = runner.reset()
    current_critic_raw = get_runner_critic_obs(raw_obs=current_obs_raw)
    obs_rms.update(current_obs_raw)
    critic_obs_rms.update(current_critic_raw)
    current_obs = torch.as_tensor(obs_rms.normalize(current_obs_raw), dtype=torch.float32, device=dev)
    current_critic_obs = torch.as_tensor(critic_obs_rms.normalize(current_critic_raw), dtype=torch.float32, device=dev)

    start_time = time.perf_counter()
    eval_runner: Optional[ParallelSimRunner] = None
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

    eval_protocol = deterministic_eval_protocol(sim_model_path, terrain_mode, obs_dim, model.squash_actions)

    def checkpoint_payload(iteration, stage, selection):
        return {
            "model_state_dict": model.state_dict(), "actor_state_dict": model.actor.state_dict(),
            "critic_state_dict": model.critic.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "obs_dim": obs_dim, "critic_obs_dim": critic_obs_dim, "action_dim": action_dim,
            "hidden_dim": hidden_dim, "obs_rms": obs_rms.state_dict(),
            "critic_obs_rms": critic_obs_rms.state_dict(), "squash_actions": model.squash_actions,
            "model_path": eval_protocol["model_path"], "model_name": sim_model_path.name,
            "iterations": iteration, "learning_rate": current_lr, "stage": stage,
            "terrain_mode": terrain_mode, "total_timesteps": total_timesteps,
            "elapsed_seconds": time.perf_counter() - start_time, "metrics": dict(metrics),
            "evaluation_protocol": dict(eval_protocol), "best_selection": selection,
            "best_stage": selection["stage"] if selection else stage,
            "best_reward": selection["mean_reward"] if selection else -float("inf"),
            "best_distance_x": selection["mean_distance_x"] if selection else -float("inf"),
            "best_det_distance_x": selection["det_distance_x"] if selection else -float("inf"),
        }

    start_iteration = loaded_iters + 1
    total_target_iters = loaded_iters + max_iterations

    try:
        for iteration in range(start_iteration, total_target_iters + 1):
            iter_start = time.perf_counter()

            # Randomize slalom pillars per iteration outside worker stepping (thread-safe)
            runner.randomize_stage_pillars()

            progress = (
                0.0
                if total_target_iters <= 1
                else min(1.0, max(0.0, (iteration - 1) / (total_target_iters - 1)))
            )
            current_entropy_coef = float(entropy_coef) + (eff_entropy_end - float(entropy_coef)) * progress

            # Smooth velocity warmup over first 15 iterations: stabilize early stepping balance
            if not stand_first and not transfer_loaded and iteration <= 15:
                max_cmd_vx = 0.15 + 0.35 * (iteration / 15.0)
                runner.commands[:, 0] = np.clip(runner.commands[:, 0], 0.08, max_cmd_vx)

            # ------------------------------------------------------------------
            # Phase 1: Rollout Collection
            # ------------------------------------------------------------------
            rollout_stage = current_stage
            rollout_standing = bool(getattr(runner, "stand_still", False))
            model.eval()
            action_stats = np.zeros(3, dtype=np.float64)
            component_totals = {}
            step_lin_vels: List[float] = []
            for step in range(rollout_steps):
                with torch.no_grad():
                    actions, log_probs, _, values, latent = model.get_action_and_value(
                        current_obs, critic_obs=current_critic_obs, return_pre_tanh=True
                    )
                    deterministic_actions = model.forward_actor(current_obs)
                    action_stats += [float(deterministic_actions.abs().mean()),
                                     float((actions - deterministic_actions).abs().mean()),
                                     float((actions.abs() > 0.99).float().mean())]

                # Squashed samples reach physics unchanged; legacy policies keep their old clipping.
                actions_np = actions.cpu().numpy()
                if not np.all(np.isfinite(actions_np)):
                    raise ValueError("Non-finite policy action")
                if model.squash_actions and np.any(np.abs(actions_np) > 1):
                    raise ValueError("Squashed policy action outside [-1, 1]")
                actions_clamped = actions_np if model.squash_actions else np.clip(actions_np, -1.0, 1.0)

                if hasattr(runner, "step_arrays"):
                    next_obs_raw, rewards_np, dones_np, step_info = runner.step_arrays(actions_clamped)
                    step_lin_vels.append(float(step_info["tracking_lin_vel"]))
                    for key, val in step_info.get("reward_components", {}).items():
                        component_totals[key] = component_totals.get(key, 0.0) + float(val)
                    raw_step_rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=dev)
                    raw_rewards_buffer[step] = raw_step_rewards

                    time_outs_np = step_info["time_outs"]
                    if np.any(time_outs_np):
                        timeout_mask = torch.as_tensor(time_outs_np, dtype=torch.bool, device=dev)
                        term_crit = step_info.get("terminal_critic_obs", step_info.get("terminal_obs"))
                        norm_term = critic_obs_rms.normalize(term_crit)
                        terminal_obs_tensor = torch.as_tensor(norm_term, dtype=torch.float32, device=dev)
                        with torch.no_grad():
                            terminal_values = model.forward_critic(terminal_obs_tensor)
                        step_rewards = apply_timeout_bootstrap(
                            raw_step_rewards, timeout_mask, terminal_values, gamma=gamma
                        )
                    else:
                        step_rewards = raw_step_rewards
                else:
                    next_obs_raw, rewards_np, dones_np, infos = runner.step(actions_clamped)
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
                        terminal_obs_list = [
                            infos[idx.item()].get("terminal_critic_obs", infos[idx.item()]["terminal_obs"])
                            for idx in timeout_indices
                        ]
                        norm_term = critic_obs_rms.normalize(np.stack(terminal_obs_list))
                        terminal_obs_tensor = torch.as_tensor(norm_term, dtype=torch.float32, device=dev)
                        with torch.no_grad():
                            terminal_values = model.forward_critic(terminal_obs_tensor)
                        step_rewards = apply_timeout_bootstrap(
                            step_rewards, timeout_mask, terminal_values, gamma=gamma
                        )

                # Store into buffers
                obs_buffer[step] = current_obs
                critic_obs_buffer[step] = current_critic_obs
                latent_buffer[step] = latent
                actions_buffer[step] = actions
                log_probs_buffer[step] = log_probs
                values_buffer[step] = values
                rewards_buffer[step] = step_rewards
                dones_buffer[step] = torch.as_tensor(dones_np, dtype=torch.float32, device=dev)

                next_critic_raw = get_runner_critic_obs(raw_obs=next_obs_raw)
                obs_rms.update(next_obs_raw)
                critic_obs_rms.update(next_critic_raw)
                current_obs = torch.as_tensor(obs_rms.normalize(next_obs_raw), dtype=torch.float32, device=dev)
                current_critic_obs = torch.as_tensor(critic_obs_rms.normalize(next_critic_raw), dtype=torch.float32, device=dev)
                total_timesteps += num_envs

            # Compute bootstrap value for the final step
            with torch.no_grad():
                next_value = model.forward_critic(current_critic_obs)

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
            b_critic_obs = critic_obs_buffer.reshape(-1, critic_obs_dim)
            b_actions = actions_buffer.reshape(-1, action_dim)
            b_latents = latent_buffer.reshape(-1, action_dim)
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

            active_lr = current_lr
            if (
                transfer_loaded
                and not optimizer_loaded
                and iteration == start_iteration
                and target_kl > 0
            ):
                active_lr = min(current_lr, 1e-4)
                for pg in optimizer.param_groups:
                    pg["lr"] = active_lr

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
                        b_obs[mb_inds], b_actions[mb_inds], critic_obs=b_critic_obs[mb_inds],
                        pre_tanh_action=b_latents[mb_inds] if model.squash_actions else None,
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
                        model.log_std.data.clamp_(model.min_log_std, model.max_log_std)

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
                    current_lr = max(min_lr, active_lr / 1.5)
                elif 0.0 < iter_mean_kl < target_kl * 0.5:
                    current_lr = min(max_lr, active_lr * 1.5)
                else:
                    current_lr = active_lr
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
            elif (
                transfer_loaded
                and not optimizer_loaded
                and iteration == start_iteration
                and target_kl > 0
            ):
                current_lr = active_lr
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
            stage_req_dist = STAGE_MIN_DISTANCE_X.get(current_stage, STAGE_MIN_DISTANCE_X[max(STAGE_MIN_DISTANCE_X)])
            cohort_success_rate = runner.get_success_rate(stage_req_dist)
            rew_lin_vel_ratio = float(np.mean(step_lin_vels)) if step_lin_vels else 0.0

            metrics["mean_reward"] = float(raw_rewards_buffer.mean().item())
            metrics["bootstrapped_mean_reward"] = float(rewards_buffer.mean().item())
            metrics["mean_distance_x"] = float(mean_dist_x)
            metrics["mean_distance_y"] = float(mean_dist_y)
            metrics["success_rate"] = float(cohort_success_rate)
            metrics["stage"] = current_stage
            metrics["hold_count"] = hold_count

            metrics["base_gaussian_entropy"] = metrics["entropy"]
            metrics["action_mean_abs"], metrics["action_noise_delta"], metrics["action_saturation_frac"] = (
                action_stats / rollout_steps).tolist()
            metrics["reward_components"] = {k: v / rollout_steps for k, v in component_totals.items()}
            metrics["evaluation_stage"] = rollout_stage
            metrics["evaluation_iteration"] = iteration
            metrics["evaluation_mode"] = "stand" if rollout_standing else "walk"
            metrics["evaluation_protocol"] = dict(eval_protocol)
            metrics["det_distance_x"] = metrics["det_reward"] = None
            metrics["best_rejection_reason"] = "dry_run" if dry_run else "standing_rollout"
            allow_best_eval = not (transfer_loaded and iteration == start_iteration
                                   and total_target_iters > start_iteration)
            if not dry_run and not rollout_standing and not allow_best_eval:
                metrics["best_rejection_reason"] = "transfer_warmup"
            if not dry_run and not rollout_standing and allow_best_eval:
                if eval_runner is None:
                    eval_runner = ParallelSimRunner(
                        model_path=str(sim_model_path), num_envs=1, decimation=10,
                        stage=rollout_stage, terrain_mode=terrain_mode, stand_still=False,
                        friction_range=(1.2, 1.2), randomize_pillars=False,
                    )
                det_x, det_reward = evaluate_deterministic_rollout(
                    model=model, obs_rms=obs_rms, model_path=sim_model_path,
                    stage=rollout_stage, terrain_mode=terrain_mode, max_steps=250,
                    device=dev, eval_runner=eval_runner,
                )
                metrics["det_distance_x"], metrics["det_reward"] = det_x, det_reward
                candidate = {
                    "stage": rollout_stage, "iteration": iteration, "protocol": dict(eval_protocol),
                    "det_distance_x": float(det_x) if det_x is not None else float("nan"),
                    "det_reward": float(det_reward) if det_reward is not None else float("nan"),
                    "mean_reward": metrics["mean_reward"], "mean_distance_x": metrics["mean_distance_x"],
                }
                accepted, reason = select_deterministic_candidate(candidate, best_record)
                metrics["best_rejection_reason"] = None if accepted else reason
                if accepted:
                    _atomic_torch_save(checkpoint_payload(iteration, rollout_stage, candidate), best_save_path)
                    best_record = candidate
                    best_stage = rollout_stage
                    best_reward, best_dist_x = candidate["mean_reward"], candidate["mean_distance_x"]
                    if verbose:
                        print(f"[PPO] [BEST] {best_save_path.name}: DetDistX={det_x:.3f}m")

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

            metrics["stage"] = current_stage
            metrics["hold_count"] = hold_count
            print_interval = max(1, min(log_interval, max_iterations))
            if verbose and (iteration % print_interval == 0 or iteration == start_iteration or iteration == total_target_iters):
                det_suffix = (f" | DetDistX: {metrics['det_distance_x']:.2f}m | DetRew: {metrics['det_reward']:+.3f}"
                              if metrics["det_distance_x"] is not None and metrics["det_reward"] is not None else "")
                print(f"[Iteration {iteration:04d}/{total_target_iters}] Stage: {current_stage} | "
                      f"EpLen: {mean_ep_len:.1f} | DistX: {mean_dist_x:.2f}m | "
                      f"Reward: {metrics['mean_reward']:+.3f} | KL: {metrics['approx_kl']:.4f}"
                      f"{det_suffix}", flush=True)

            # Periodic checkpoint saving (latest model)
            if not dry_run and save_interval > 0 and (iteration % save_interval == 0):
                ckpt = checkpoint_payload(iteration, current_stage, best_record)
                _atomic_torch_save(ckpt, save_path)
                _atomic_torch_save(ckpt, latest_save_path)
                if verbose:
                    print(f"[PPO] Periodic checkpoint saved: {save_path.name} & {latest_save_path.name} (iter {iteration})")

        # ------------------------------------------------------------------
        # Phase 4: Save Model Checkpoint
        # ------------------------------------------------------------------
        elapsed_total = time.perf_counter() - start_time
        checkpoint = checkpoint_payload(total_target_iters, current_stage, best_record)
        _atomic_torch_save(checkpoint, save_path)
        _atomic_torch_save(checkpoint, latest_save_path)
        if verbose:
            print(f"[PPO] Checkpoint saved successfully: {save_path} ({total_timesteps} steps in {elapsed_total:.2f}s)")

    finally:
        runner.close()
        if eval_runner is not None:
            eval_runner.close()

    return {
        "success": True,
        "iterations": total_target_iters,
        "total_timesteps": total_timesteps,
        "checkpoint_path": str(save_path),
        "latest_checkpoint_path": str(latest_save_path),
        "best_checkpoint_path": str(best_save_path) if best_record is not None else None,
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
    parser.add_argument("--entropy-coef", type=float, default=0.002, help="Entropy loss weight (default: 0.002)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss weight (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Run quick lightweight iterations for CI/CD testing")
    parser.add_argument("--device", type=str, default=None, help="Device to run on ('cpu', 'cuda')")
    parser.add_argument("--start-stage", type=int, default=0, help="Initial terrain curriculum stage (0 to 3)")
    parser.add_argument("--hold-iters", type=int, default=5, help="Number of consecutive iterations required to advance stage (default: 5)")
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
    parser.add_argument("--entropy-coef-end", type=float, default=None, help="Target entropy coefficient at final iteration (default: None, falls back to min(entropy_coef, 0.0005))")
    parser.add_argument("--max-log-std", type=float, default=-1.05, help="Maximum log standard deviation for action policy (default: -1.05)")
    parser.add_argument("--randomize-pillars", action="store_true", help="Enable randomizing slalom pillar selection and jitter per episode/iteration")
    parser.add_argument("--pillar-density", type=float, default=1.0, help="Fraction of slalom pillars kept active (default: 1.0)")
    parser.add_argument("--pillar-jitter", type=float, default=0.06, help="Maximum XY jitter applied to active slalom pillars in meters (default: 0.06)")

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
        hold_iters=args.hold_iters,
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
        max_log_std=args.max_log_std,
        randomize_pillars=args.randomize_pillars,
        pillar_density=args.pillar_density,
        pillar_jitter=args.pillar_jitter,
    )


if __name__ == "__main__":
    main()
