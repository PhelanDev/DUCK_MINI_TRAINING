import inspect
import math
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ppo import ActorCritic, train_ppo

def test_train_ppo_signature_and_caps_action_std(tmp_path: Path):
    sig = inspect.signature(train_ppo)
    assert sig.parameters["max_log_std"].default == pytest.approx(-1.05)
    assert sig.parameters["entropy_coef"].default == pytest.approx(0.002)
    assert sig.parameters["hold_iters"].default == 5

    # ActorCritic class default remains 0.0 for existing unit test compatibility
    assert ActorCritic(obs_dim=42, action_dim=8).max_log_std == pytest.approx(0.0)

    res = train_ppo(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        rollout_steps=8,
        mini_batch_size=8,
        num_epochs=2,
        max_iterations=2,
        entropy_coef=0.05,  # high entropy pressure must still respect max_log_std cap
        device="cpu",
        save_path=str(tmp_path / "capped_std_ppo.pt"),
        verbose=False,
    )
    assert res["metrics"]["action_std_mean"] <= math.exp(-1.05) + 1e-4


import numpy as np
from training.run_parallel_sim import ParallelSimRunner


def test_hurdles_command_sampling_aligns_with_corridor_and_stages():
    runner0 = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=32,
        num_threads=1,
        stage=0,
        terrain_mode="hurdles",
    )
    try:
        runner0.reset()
        assert np.all((runner0.commands[:, 0] >= 0.18) & (runner0.commands[:, 0] <= 0.25))
        assert np.all(np.abs(runner0.commands[:, 1]) <= 0.03 + 1e-6)
        assert np.all(np.abs(runner0.commands[:, 2]) < 0.05)  # aligns with Stage 0 straight_cmd yaw guard
    finally:
        runner0.close()

    runner1 = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=32,
        num_threads=1,
        stage=1,
        terrain_mode="hurdles",
    )
    try:
        runner1.reset()
        assert np.all((runner1.commands[:, 0] >= 0.18) & (runner1.commands[:, 0] <= 0.25))
        assert np.all(np.abs(runner1.commands[:, 2]) <= 0.15 + 1e-6)
    finally:
        runner1.close()
