import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import MicroDuck8DoFConfig, scale_actions, unscale_actions


def test_unscale_actions_round_trip_nominal_centered():
    cfg = MicroDuck8DoFConfig()
    rng = np.random.default_rng(7)

    safe_low = np.maximum(-0.95, (cfg.low_limits - cfg.nominal_qpos) / cfg.action_scales + 1e-3)
    safe_high = np.minimum(0.95, (cfg.high_limits - cfg.nominal_qpos) / cfg.action_scales - 1e-3)

    # 2D batch (16, 8)
    raw_2d = rng.uniform(safe_low, safe_high, size=(16, 8)).astype(np.float32)
    targets_2d = scale_actions(raw_2d, cfg, nominal_centered=True)
    recovered_2d = unscale_actions(targets_2d, cfg, nominal_centered=True)
    assert np.allclose(recovered_2d, raw_2d, atol=1e-5)
    assert recovered_2d.shape == (16, 8)
    assert recovered_2d.dtype == np.float32

    # 1D single action (8,)
    raw_1d = rng.uniform(safe_low, safe_high, size=(8,)).astype(np.float32)
    targets_1d = scale_actions(raw_1d, cfg, nominal_centered=True)
    recovered_1d = unscale_actions(targets_1d, cfg, nominal_centered=True)
    assert np.allclose(recovered_1d, raw_1d, atol=1e-5)
    assert recovered_1d.shape == (8,)
    assert recovered_1d.dtype == np.float32

    # nominal_qpos -> 0.0 verification
    nom = cfg.nominal_qpos.copy()
    recovered_nom = unscale_actions(nom, cfg, nominal_centered=True)
    assert np.allclose(recovered_nom, 0.0, atol=1e-6)
    assert recovered_nom.dtype == np.float32

    # Default cfg=None fallback test
    recovered_nom_default = unscale_actions(nom, nominal_centered=True)
    assert np.allclose(recovered_nom_default, 0.0, atol=1e-6)


def test_unscale_actions_round_trip_linear():
    cfg = MicroDuck8DoFConfig()
    rng = np.random.default_rng(42)

    # 2D batch (10, 8)
    raw_2d = rng.uniform(-0.95, 0.95, size=(10, 8)).astype(np.float32)
    targets_2d = scale_actions(raw_2d, cfg, nominal_centered=False)
    recovered_2d = unscale_actions(targets_2d, cfg, nominal_centered=False)
    assert np.allclose(recovered_2d, raw_2d, atol=1e-5)
    assert recovered_2d.shape == (10, 8)
    assert recovered_2d.dtype == np.float32

    # 1D single action (8,)
    raw_1d = rng.uniform(-0.95, 0.95, size=(8,)).astype(np.float32)
    targets_1d = scale_actions(raw_1d, cfg, nominal_centered=False)
    recovered_1d = unscale_actions(targets_1d, cfg, nominal_centered=False)
    assert np.allclose(recovered_1d, raw_1d, atol=1e-5)
    assert recovered_1d.shape == (8,)

    # Exact boundary mapping
    low = cfg.low_limits
    high = cfg.high_limits
    mid = 0.5 * (low + high)
    assert np.allclose(unscale_actions(low, cfg, nominal_centered=False), -1.0, atol=1e-5)
    assert np.allclose(unscale_actions(high, cfg, nominal_centered=False), 1.0, atol=1e-5)
    assert np.allclose(unscale_actions(mid, cfg, nominal_centered=False), 0.0, atol=1e-5)


def test_unscale_actions_out_of_bounds_clamping():
    cfg = MicroDuck8DoFConfig()
    far_low = cfg.low_limits - 10.0
    far_high = cfg.high_limits + 10.0

    # Linear mode clamping
    res_low = unscale_actions(far_low, cfg, nominal_centered=False)
    res_high = unscale_actions(far_high, cfg, nominal_centered=False)
    assert np.all(res_low == -1.0)
    assert np.all(res_high == 1.0)
    assert res_low.dtype == np.float32
    assert res_high.dtype == np.float32

    # Nominal-centered mode clamping
    res_low_nom = unscale_actions(far_low, cfg, nominal_centered=True)
    res_high_nom = unscale_actions(far_high, cfg, nominal_centered=True)
    assert np.all(res_low_nom == -1.0)
    assert np.all(res_high_nom == 1.0)
    assert res_low_nom.dtype == np.float32
    assert res_high_nom.dtype == np.float32


def test_bc_pretrain_decreases_loss_and_loads_into_ppo(tmp_path: Path):
    import torch
    from training.run_parallel_sim import ParallelSimRunner
    from scripts.pretrain_bc import collect_gait_demonstrations, train_behavior_cloning, save_bc_checkpoint
    from scripts.train_ppo import train_ppo

    runner = ParallelSimRunner(model_path="duck_v2/cyberduck_v2.xml", num_envs=2, num_threads=1, stage=0)
    try:
        obs_arr, act_arr, ret_arr = collect_gait_demonstrations(runner, num_steps=30)
    finally:
        runner.close()

    assert obs_arr.shape[1] == 42 and act_arr.shape == (len(obs_arr), 8) and ret_arr.shape == (len(obs_arr),)
    assert np.all((act_arr >= -1.0) & (act_arr <= 1.0))

    model, losses = train_behavior_cloning(obs_arr, act_arr, returns=ret_arr, obs_dim=42, action_dim=8, epochs=10, batch_size=32)
    assert losses[-1] < losses[0]

    bc_ckpt = tmp_path / "bc_init.pt"
    save_bc_checkpoint(bc_ckpt, model, obs_dim=42, action_dim=8, hidden_dim=256)
    ppo_ckpt = tmp_path / "ppo_after_bc.pt"

    result = train_ppo(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        rollout_steps=8,
        max_iterations=1,
        load_checkpoint=bc_ckpt,
        save_path=ppo_ckpt,
        verbose=False,
    )
    assert ppo_ckpt.exists()
    assert result["success"] is True
    assert np.isfinite(result["metrics"]["policy_loss"])


def test_collect_teacher_demonstrations_supports_v1_dim_slicing():
    import torch
    import torch.nn as nn
    from training.run_parallel_sim import ParallelSimRunner
    from scripts.pretrain_bc import collect_teacher_demonstrations

    runner = ParallelSimRunner(model_path="duck_v2/cyberduck_v2.xml", num_envs=2, num_threads=1, stage=0)
    try:
        teacher = nn.Linear(37, 8)
        obs_arr, act_arr, ret_arr = collect_teacher_demonstrations(runner, teacher, num_steps=15)
    finally:
        runner.close()

    assert obs_arr.shape[1] == 42
    assert act_arr.shape == (len(obs_arr), 8)
    assert ret_arr.shape == (len(obs_arr),)
    assert np.all((act_arr >= -1.0) & (act_arr <= 1.0))


def test_pretrain_bc_cli_runs_directly_from_repo_root():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/pretrain_bc.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"pretrain_bc CLI failed: {proc.stderr}"
    assert "--mode" in proc.stdout


def test_save_bc_checkpoint_includes_obs_rms(tmp_path: Path):
    """train_behavior_cloning attaches _bc_obs_rms and save_bc_checkpoint embeds obs_rms."""
    import torch
    from scripts.pretrain_bc import train_behavior_cloning, save_bc_checkpoint
    from scripts.export_policy import load_actor_from_checkpoint
    from scripts.train_ppo import train_ppo

    np.random.seed(42)
    obs = np.random.randn(80, 42).astype(np.float32) + 2.0
    acts = np.random.uniform(-0.5, 0.5, size=(80, 8)).astype(np.float32)

    model, losses = train_behavior_cloning(
        obs, acts, obs_dim=42, action_dim=8, epochs=3, batch_size=32
    )
    assert hasattr(model, "_bc_obs_rms"), "train_behavior_cloning must attach _bc_obs_rms to model"
    assert model._bc_obs_rms is not None
    assert model._bc_obs_rms.count > 0

    ckpt_path = tmp_path / "bc_with_rms.pt"
    save_bc_checkpoint(ckpt_path, model, obs_dim=42, action_dim=8)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert "obs_rms" in ckpt, "Checkpoint must contain 'obs_rms'"
    assert "mean" in ckpt["obs_rms"]
    assert "var" in ckpt["obs_rms"]
    assert np.all(ckpt["obs_rms"]["var"] > 0)

    # Verify load_actor_from_checkpoint folds obs_rms without error
    actor = load_actor_from_checkpoint(ckpt_path)
    assert actor is not None
    test_obs = torch.randn(1, 42)
    out_act = actor(test_obs)
    assert out_act.shape == (1, 8)

    # Verify train_ppo loads checkpoint with obs_rms cleanly
    ppo_save = tmp_path / "ppo_out.pt"
    res = train_ppo(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=1,
        rollout_steps=8,
        max_iterations=1,
        load_checkpoint=ckpt_path,
        save_path=ppo_save,
        dry_run=True,
        verbose=False,
    )
    assert res["success"] is True
