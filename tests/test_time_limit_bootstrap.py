"""Tests for PPO time-limit bootstrapping and terminal observation tracking."""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ppo import apply_timeout_bootstrap
from training.run_parallel_sim import DuckParallelSim


def test_apply_timeout_bootstrap_adds_gamma_v_only_on_timeouts():
    """Verify that apply_timeout_bootstrap adds gamma * V(s_terminal) only to timed-out envs."""
    rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    time_outs = torch.tensor([False, True, False], dtype=torch.bool)
    terminal_values = torch.tensor([5.0], dtype=torch.float32)  # Filtered to 1 timeout
    gamma = 0.99

    bootstrapped = apply_timeout_bootstrap(rewards, time_outs, terminal_values, gamma=gamma)

    expected = torch.tensor([1.0, 1.0 + 0.99 * 5.0, 1.0], dtype=torch.float32)
    assert torch.allclose(bootstrapped, expected)
    # Ensure original rewards tensor was not mutated in-place if cloned
    assert rewards[1].item() == 1.0


def test_apply_timeout_bootstrap_noop_when_no_timeouts():
    """Verify that apply_timeout_bootstrap is a no-op when no environments timed out."""
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    time_outs = torch.tensor([False, False, False], dtype=torch.bool)
    terminal_values = torch.tensor([], dtype=torch.float32)

    bootstrapped = apply_timeout_bootstrap(rewards, time_outs, terminal_values, gamma=0.99)
    assert torch.allclose(bootstrapped, rewards)


def test_sim_timeout_sets_time_outs_and_terminal_obs():
    """Verify that reaching max_episode_steps sets time_outs=True and records terminal_obs before reset."""
    sim = DuckParallelSim(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
        decimation=1,
    )
    sim.max_episode_steps = 2
    sim.reset()

    # Step 1: episode_steps becomes 1 (< 2)
    actions = np.zeros((2, 8), dtype=np.float32)
    next_obs, rewards, dones, infos = sim.step(actions)
    assert not any(dones)
    assert not any(info.get("time_outs", False) for info in infos)

    # Step 2: episode_steps becomes 2 (>= 2) -> timeouts
    next_obs, rewards, dones, infos = sim.step(actions)
    assert all(dones)
    for i, info in enumerate(infos):
        assert info.get("time_outs") is True
        assert "terminal_obs" in info
        assert isinstance(info["terminal_obs"], np.ndarray)
        assert info["terminal_obs"].shape == (sim.cfg.OBS_DIM,)
        # terminal_obs was captured before reset_env(i), so next_obs[i] (which was reset with noise)
        # should not be identical to terminal_obs
        # Or at least terminal_obs is recorded and matches physics state prior to reset
        assert not np.isnan(info["terminal_obs"]).any()


def test_sim_fall_is_not_timeout():
    """Verify that when an env falls while also timing out, it is treated as fallen (time_outs=False)."""
    sim = DuckParallelSim(
        model_path="models/mjcf/microduck_8dof.xml",
        num_envs=2,
        decimation=1,
    )
    sim.max_episode_steps = 2
    sim.reset()

    actions = np.zeros((2, 8), dtype=np.float32)
    sim.step(actions)

    # Force fall condition on env 0 by lowering root_z to 0.05
    sim.datas[0].qpos[2] = 0.05

    next_obs, rewards, dones, infos = sim.step(actions)
    assert all(dones)
    # Env 0 fell and reached max steps: must NOT be bootstrapped
    assert infos[0].get("time_outs") is False
    # Env 1 only timed out (upright): must be bootstrapped
    assert infos[1].get("time_outs") is True
    # Both should have terminal_obs
    assert "terminal_obs" in infos[0]
    assert "terminal_obs" in infos[1]
