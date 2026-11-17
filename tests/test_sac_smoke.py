import math
from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.train_sac import (
    SquashedGaussianActor,
    TwinQCritic,
    ReplayBuffer,
    sac_update_step,
    save_sac_checkpoint,
    train_sac,
)
from training.run_parallel_sim import ParallelSimRunner


def test_sac_update_step_and_checkpoint(tmp_path: Path):
    obs_dim, act_dim = 42, 8
    actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim=64)
    critic = TwinQCritic(obs_dim, act_dim, hidden_dim=64)
    target_critic = TwinQCritic(obs_dim, act_dim, hidden_dim=64)
    target_critic.load_state_dict(critic.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True)

    sample_obs = torch.randn(16, obs_dim)
    action, log_prob = actor.sample(sample_obs)
    assert action.shape == (16, act_dim) and torch.all((action >= -1.0) & (action <= 1.0))
    assert log_prob.shape == (16, 1)

    det_act = actor(sample_obs)
    assert det_act.shape == (16, act_dim) and torch.all((det_act >= -1.0) & (det_act <= 1.0))

    buf = ReplayBuffer(obs_dim, act_dim, capacity=64)
    buf.add_batch(
        torch.randn(16, obs_dim).numpy(),
        action.detach().numpy(),
        torch.ones(16).numpy(),
        torch.randn(16, obs_dim).numpy(),
        torch.zeros(16).numpy(),
    )
    opts = {
        "actor": torch.optim.Adam(actor.parameters(), lr=3e-4),
        "critic": torch.optim.Adam(critic.parameters(), lr=3e-4),
        "alpha": torch.optim.Adam([log_alpha], lr=3e-4),
    }
    old_log_alpha = float(log_alpha.item())
    stats = sac_update_step(
        actor,
        critic,
        target_critic,
        log_alpha,
        buf.sample(8),
        opts,
        target_entropy=-float(act_dim),
    )
    assert all(math.isfinite(v) for v in stats.values())
    assert float(log_alpha.item()) != old_log_alpha

    ckpt_path = tmp_path / "sac_smoke.pt"
    saved_path = save_sac_checkpoint(ckpt_path, actor, log_alpha, obs_dim, act_dim, hidden_dim=64)
    assert Path(saved_path).exists()
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "actor_state_dict" in loaded
    assert "sac_alpha" in loaded
    assert math.isclose(loaded["sac_alpha"], float(log_alpha.exp().item()), rel_tol=1e-5)


def test_replay_buffer_wraparound_and_train_sac_smoke(tmp_path: Path):
    # Test buffer wraparound with unaligned capacity
    buf = ReplayBuffer(obs_dim=4, act_dim=2, capacity=10)
    batch1 = np.ones((6, 4), dtype=np.float32)
    buf.add_batch(batch1, np.zeros((6, 2)), np.ones(6), batch1, np.zeros(6))
    assert buf.size == 6 and buf.ptr == 6

    # 6 + 7 = 13 > 10, should wrap around to index 3 and cap size at 10
    batch2 = np.full((7, 4), 2.0, dtype=np.float32)
    buf.add_batch(batch2, np.zeros((7, 2)), np.ones(7), batch2, np.zeros(7))
    assert buf.size == 10 and buf.ptr == 3

    sample_batch = buf.sample(5)
    assert len(sample_batch) == 5
    s_obs, s_act, s_rew, s_next, s_done = sample_batch
    assert s_obs.shape == (5, 4)
    assert s_act.shape == (5, 2)
    assert s_rew.shape == (5, 1)
    assert s_next.shape == (5, 4)
    assert s_done.shape == (5, 1)

    # Smoke train_sac run
    ckpt_out = tmp_path / "train_smoke.pt"
    res = train_sac(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        total_steps=16,
        warmup_steps=8,
        batch_size=4,
        hidden_dim=64,
        save_path=str(ckpt_out),
    )
    assert Path(res).exists()
    loaded = torch.load(res, map_location="cpu", weights_only=False)
    assert "actor_state_dict" in loaded
    assert "sac_alpha" in loaded


def test_train_sac_handles_timeouts_and_runner_cleanup(tmp_path: Path, monkeypatch):
    # Verify configurable target_entropy
    ckpt_out = tmp_path / "custom_entropy.pt"
    res = train_sac(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        total_steps=16,
        warmup_steps=8,
        batch_size=4,
        hidden_dim=64,
        save_path=str(ckpt_out),
        target_entropy=-4.0,
    )
    assert Path(res).exists()

    # Verify timeout separation: dones=True, timeouts=True -> true_done=False (done=0.0)
    buf = ReplayBuffer(obs_dim=2, act_dim=1, capacity=10)
    obs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    acts = np.array([[0.1], [0.2]], dtype=np.float32)
    rews = np.array([1.0, 1.0], dtype=np.float32)
    next_obs = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    dones = np.array([True, True], dtype=bool)
    fast_info = {
        "time_outs": np.array([True, False], dtype=bool),
        "terminal_obs": np.array([[99.0, 99.0]], dtype=np.float32),
    }

    time_outs = np.asarray(fast_info.get("time_outs", np.zeros_like(dones, dtype=bool)), dtype=bool)
    true_dones = np.logical_and(dones, ~time_outs)
    buf_next_obs = next_obs.copy()
    if np.any(time_outs) and len(fast_info.get("terminal_obs", [])) == int(np.sum(time_outs)):
        buf_next_obs[time_outs] = fast_info["terminal_obs"]

    buf.add_batch(obs, acts, rews, buf_next_obs, true_dones)
    assert buf.done[0, 0] == 0.0
    assert np.allclose(buf.next_obs[0], [99.0, 99.0])
    assert buf.done[1, 0] == 1.0

    # Verify runner.close() is called even on exceptions during training
    close_called = []
    real_init = ParallelSimRunner.__init__

    def mock_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        real_close = self.close

        def tracking_close():
            close_called.append(True)
            real_close()

        self.close = tracking_close

    monkeypatch.setattr(ParallelSimRunner, "__init__", mock_init)

    def buggy_step_arrays(*args, **kwargs):
        raise RuntimeError("Sim crash simulation")

    monkeypatch.setattr(ParallelSimRunner, "step_arrays", buggy_step_arrays)

    with pytest.raises(RuntimeError, match="Sim crash simulation"):
        train_sac(
            model_path="duck_v2/cyberduck_v2.xml",
            num_envs=2,
            total_steps=16,
            warmup_steps=8,
            save_path=str(tmp_path / "fail.pt"),
        )
    assert len(close_called) > 0, "runner.close() was not called in finally block!"


def test_train_sac_cli_runs_directly_from_repo_root():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/train_sac.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"train_sac CLI failed: {proc.stderr}"
    assert "--total-steps" in proc.stdout
