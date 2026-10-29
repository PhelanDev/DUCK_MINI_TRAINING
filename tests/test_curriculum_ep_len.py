"""Tests for completed episode length tracking in ParallelSimRunner and curriculum advancement."""

from collections import deque
from pathlib import Path
import sys
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.run_parallel_sim import ParallelSimRunner
from training.terrain_curriculum import next_stage


def test_completed_ep_lens_records_on_done():
    """Verify that when an environment terminates (timeout or fall),
    completed_ep_lens records the step count and infos[i]['episode_length'] is set."""
    runner = ParallelSimRunner(num_envs=2, max_episode_steps=5, num_threads=1)
    assert hasattr(runner, "completed_ep_lens")
    assert isinstance(runner.completed_ep_lens, deque)
    assert len(runner.completed_ep_lens) == 0

    # Step for up to 10 steps with zero actions until timeout/fall occurs
    zero_actions = np.zeros((2, runner.action_dim), dtype=np.float32)
    done_occurred = False
    for step_idx in range(10):
        _, _, dones, infos = runner.step(zero_actions)
        if np.any(dones):
            done_occurred = True
            for i in np.where(dones)[0]:
                assert "episode_length" in infos[i]
                assert infos[i]["episode_length"] >= 1
            break

    assert done_occurred
    assert len(runner.completed_ep_lens) > 0
    assert all(isinstance(l, int) for l in runner.completed_ep_lens)


def test_get_mean_episode_length_fallback_and_history():
    """Verify that get_mean_episode_length falls back to episode_steps.mean()
    when no episodes have completed, and uses completed_ep_lens average once populated."""
    runner = ParallelSimRunner(num_envs=4, max_episode_steps=100, num_threads=1)

    # Initially no completed episodes, fallback to episode_steps.mean()
    assert len(runner.completed_ep_lens) == 0
    runner.episode_steps[:] = np.array([10, 20, 30, 40], dtype=np.int32)
    assert pytest.approx(runner.get_mean_episode_length()) == 25.0

    # Manually append to completed_ep_lens
    runner.completed_ep_lens.extend([100, 200, 300])
    # Now runner.get_mean_episode_length() should return 200.0 regardless of runner.episode_steps
    assert pytest.approx(runner.get_mean_episode_length()) == 200.0


def test_curriculum_advancement_with_completed_ep_lens():
    """Simulate rollout iterations where envs complete 500 steps and reset to 0.
    Verify that get_mean_episode_length maintains >= 400 across iterations,
    allowing next_stage to accumulate hold_iters and advance stage."""
    runner = ParallelSimRunner(num_envs=4, max_episode_steps=500, num_threads=1)

    # Simulate envs completing 500 steps
    for _ in range(4):
        runner.completed_ep_lens.append(500)

    # After reset, episode_steps are reset to 0
    runner.episode_steps[:] = 0

    # episode_steps.mean() would be 0, but get_mean_episode_length() is 500.0
    assert float(runner.episode_steps.mean()) == 0.0
    assert runner.get_mean_episode_length() == 500.0

    stage = 0
    hold = 0
    hold_iters = 5
    # Over 5 iterations, runner steps from 0 to 10, mean_ep_len from completed window remains >= 400
    for it in range(hold_iters):
        mean_ep_len = runner.get_mean_episode_length()
        assert mean_ep_len >= 400.0
        rew_lin_vel_ratio = 0.85
        stage, hold = next_stage(
            stage=stage,
            ep_len_mean=mean_ep_len,
            rew_lin_vel_ratio=rew_lin_vel_ratio,
            hold_count=hold,
            hold_iters=hold_iters,
        )

    # After hold_iters consecutive iterations meeting the condition, stage must advance to 1
    assert stage == 1
    assert hold == 0

