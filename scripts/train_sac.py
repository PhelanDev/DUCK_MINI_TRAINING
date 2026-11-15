import argparse, math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from training.run_parallel_sim import ParallelSimRunner


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean_head(self.net(obs)))

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(obs)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), -20.0, 2.0)
        normal = Normal(mean, log_std.exp())
        u = normal.rsample()
        action = torch.tanh(u)
        log_prob = (normal.log_prob(u) - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1, keepdim=True)
        return action, log_prob


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        mlp = lambda: nn.Sequential(nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q1, self.q2 = mlp(), mlp()

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100_000):
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add_batch(self, obs, act, rew, next_obs, done):
        n = len(obs)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx], self.act[idx] = obs, act
        self.rew[idx] = np.asarray(rew, dtype=np.float32).reshape(-1, 1)
        self.next_obs[idx] = next_obs
        self.done[idx] = np.asarray(done, dtype=np.float32).reshape(-1, 1)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda arr: torch.as_tensor(arr[idx], dtype=torch.float32, device=device)
        return to_t(self.obs), to_t(self.act), to_t(self.rew).view(-1, 1), to_t(self.next_obs), to_t(self.done).view(-1, 1)


def sac_update_step(
    actor: SquashedGaussianActor, critic: TwinQCritic, target_critic: TwinQCritic,
    log_alpha: torch.Tensor, batch: Tuple[torch.Tensor, ...],
    optimizers: Dict[str, torch.optim.Optimizer], target_entropy: float,
    gamma: float = 0.99, tau: float = 0.005,
) -> Dict[str, float]:
    obs, act, rew, next_obs, done = batch
    rew, done = rew.view(-1, 1), done.view(-1, 1)
    with torch.no_grad():
        next_act, next_log_prob = actor.sample(next_obs)
        t_q1, t_q2 = target_critic(next_obs, next_act)
        alpha = log_alpha.exp()
        target_q = rew + (1.0 - done) * gamma * (torch.min(t_q1, t_q2) - alpha * next_log_prob)

    q1, q2 = critic(obs, act)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    optimizers["critic"].zero_grad()
    critic_loss.backward()
    optimizers["critic"].step()

    critic.requires_grad_(False)
    try:
        pi_act, log_prob = actor.sample(obs)
        q1_pi, q2_pi = critic(obs, pi_act)
        actor_loss = (alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()
        optimizers["actor"].zero_grad()
        actor_loss.backward()
        optimizers["actor"].step()
    finally:
        critic.requires_grad_(True)

    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    optimizers["alpha"].zero_grad()
    alpha_loss.backward()
    optimizers["alpha"].step()
    log_alpha.data.clamp_(-10.0, 2.0)

    with torch.no_grad():
        for p, pt in zip(critic.parameters(), target_critic.parameters()):
            pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    return {
        "critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()),
        "alpha_loss": float(alpha_loss.item()), "alpha": float(alpha.item()),
    }


def save_sac_checkpoint(
    path: Union[str, Path], actor: SquashedGaussianActor, log_alpha: torch.Tensor,
    obs_dim: int, act_dim: int, hidden_dim: int = 256,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor_state_dict": actor.state_dict(), "sac_alpha": float(log_alpha.exp().item()),
        "obs_dim": obs_dim, "action_dim": act_dim, "hidden_dim": hidden_dim,
    }, p)
    return p


def train_sac(
    model_path: Union[str, Path] = "models/mjcf/microduck_8dof.xml",
    num_envs: int = 16, total_steps: int = 100_000, warmup_steps: int = 1_000,
    batch_size: int = 256, hidden_dim: int = 256,
    save_path: Union[str, Path] = "checkpoints/sac_checkpoint.pt",
    lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
    buffer_capacity: int = 100_000, device: str = "cpu",
    target_entropy: Optional[float] = None,
) -> Path:
    runner = ParallelSimRunner(model_path=model_path, num_envs=num_envs)
    try:
        obs_dim, act_dim = runner.obs_dim, runner.action_dim
        t_entropy = -float(act_dim) if target_entropy is None else float(target_entropy)
        actor, critic = SquashedGaussianActor(obs_dim, act_dim, hidden_dim).to(device), TwinQCritic(obs_dim, act_dim, hidden_dim).to(device)
        target_critic = TwinQCritic(obs_dim, act_dim, hidden_dim).to(device)
        target_critic.load_state_dict(critic.state_dict())
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        optimizers = {
            "actor": torch.optim.Adam(actor.parameters(), lr=lr),
            "critic": torch.optim.Adam(critic.parameters(), lr=lr),
            "alpha": torch.optim.Adam([log_alpha], lr=lr),
        }
        buf = ReplayBuffer(obs_dim, act_dim, capacity=buffer_capacity)
        obs, collected = runner.reset(), 0
        while collected < total_steps:
            if collected < warmup_steps:
                acts = np.random.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    acts = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device))[0].cpu().numpy()
            next_obs, rews, dones, fast_info = runner.step_arrays(acts)
            time_outs = np.asarray(fast_info.get("time_outs", np.zeros_like(dones, dtype=bool)), dtype=bool)
            true_dones = np.logical_and(dones, ~time_outs)
            buf_next_obs = next_obs.copy()
            if np.any(time_outs) and len(fast_info.get("terminal_obs", [])) == int(np.sum(time_outs)):
                buf_next_obs[time_outs] = fast_info["terminal_obs"]
            # shortcut: on true falls next_obs is post-reset with done=1 (multiplying bootstrap by 0)
            buf.add_batch(obs, acts, rews, buf_next_obs, true_dones)
            obs, collected = next_obs, collected + num_envs
            if collected >= warmup_steps and buf.size >= batch_size:
                sac_update_step(actor, critic, target_critic, log_alpha, buf.sample(batch_size, device=device), optimizers, t_entropy, gamma=gamma, tau=tau)
        return save_sac_checkpoint(save_path, actor, log_alpha, obs_dim, act_dim, hidden_dim)
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser(description="SAC training for MicroDuck")
    parser.add_argument("--model-path", default="models/mjcf/microduck_8dof.xml")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--save-path", default="checkpoints/sac_checkpoint.pt")
    parser.add_argument("--target-entropy", type=float, default=None)
    args = parser.parse_args()
    train_sac(
        model_path=args.model_path, num_envs=args.num_envs,
        total_steps=args.total_steps, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
        save_path=args.save_path, target_entropy=args.target_entropy,
    )


if __name__ == "__main__":
    main()
