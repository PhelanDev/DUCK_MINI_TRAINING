"""Comprehensive Unit Tests for MicroDuck PPO Training Pipeline and Policy Export.

Covers:
1. scripts/train_ppo.py:
   - ActorCritic neural network architecture (42 -> 256 ELU -> 256 ELU -> 8 actions / 1 value)
   - Learnable log_std parameter and forward pass distributions
   - Generalized Advantage Estimation (GAE) calculation
   - PPO training loop with --dry-run option and checkpoint saving
   - CLI interface and argument parsing
2. scripts/export_policy.py:
   - Float32 ONNX model export for Raspberry Pi 5 (input 'obs' [1, 42], output 'actions' [1, 8], opset 13)
   - onnxruntime validation and numerical parity against PyTorch
   - INT8 quantized weights and INT32 biases C/C++ header generation for ESP32-S3
   - C/C++ syntax validation and compilation test via g++
3. End-to-end CLI integration from PPO dry-run checkpoint to deployment artifacts.
"""

from pathlib import Path
import math
import os
import subprocess
import sys
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_audit_cross_target_resume_and_overwrite_rejected(tmp_path):
    from scripts.train_ppo import train_ppo
    path = tmp_path / "policy_ppo.pt"
    kwargs = dict(num_envs=1, rollout_steps=2, max_iterations=1, num_epochs=1,
                  save_path=path, device="cpu", verbose=False, terrain_mode="flat")
    train_ppo(model_path=REPO_ROOT / "pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml", **kwargs)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.pt")}
    for resume in (True, False):
        with pytest.raises(ValueError, match="identity|model"):
            train_ppo(model_path=REPO_ROOT / "pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml", resume=resume, **kwargs)
        assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.pt")}


@pytest.mark.parametrize("bad", [None, {"mean": [float("nan"), 0], "var": [1, 1], "count": 1},
    {"mean": [0, 0], "var": [-1, 1], "count": 1}, {"mean": [0], "var": [1], "count": 1},
    {"mean": [0, 0], "var": [1, 1], "count": float("inf")}])
def test_audit_rms_load_is_strict_and_atomic(bad):
    from scripts.train_ppo import RunningMeanStd
    rms = RunningMeanStd(shape=2)
    before = rms.state_dict()
    with pytest.raises(ValueError, match="RMS"):
        rms.load_state_dict(bad)
    for key in before:
        np.testing.assert_equal(rms.state_dict()[key], before[key])


@pytest.mark.parametrize("num_envs,rollout_steps", [(1, 1), (1, 2), (2, 1)])
def test_audit_small_batches_are_finite(tmp_path, num_envs, rollout_steps):
    from scripts.train_ppo import train_ppo
    path = tmp_path / "tiny_ppo.pt"
    result = train_ppo(num_envs=num_envs, rollout_steps=rollout_steps, max_iterations=1,
        num_epochs=1, save_path=path, device="cpu", verbose=False, terrain_mode="flat")
    assert result["success"]
    state = torch.load(path, weights_only=False)["model_state_dict"]
    assert all(torch.isfinite(t).all() for t in state.values())


def test_audit_nonfinite_checkpoint_preserves_file(tmp_path):
    from scripts.train_ppo import _atomic_torch_save
    path = tmp_path / "valid.pt"
    _atomic_torch_save({"model_state_dict": {"w": torch.ones(1)}}, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="finite"):
        _atomic_torch_save({"model_state_dict": {"w": torch.tensor([float("nan")])}}, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("key,bad", [("obs_rms", None), ("critic_obs_rms", None),
    ("obs_rms", "nan"), ("critic_obs_rms", "shape")])
def test_audit_resume_rejects_malformed_rms_without_writes(tmp_path, key, bad):
    from scripts.train_ppo import train_ppo
    path = tmp_path / "resume_ppo.pt"
    options = dict(num_envs=1, rollout_steps=2, max_iterations=1, num_epochs=1,
        save_path=path, device="cpu", verbose=False, terrain_mode="flat")
    train_ppo(**options)
    ckpt = torch.load(path, weights_only=False)
    if bad is None:
        ckpt[key] = None
    elif bad == "nan":
        ckpt[key]["mean"][0] = float("nan")
    else:
        ckpt[key]["var"] = np.ones(1)
    torch.save(ckpt, path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.pt")}
    with pytest.raises(ValueError, match="RMS|finite"):
        train_ppo(resume=True, **options)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.pt")}


def test_audit_nonfinite_gradient_rejected_before_update(tmp_path, monkeypatch):
    from scripts.train_ppo import ActorCritic, train_ppo
    original = ActorCritic.__init__
    def bad_gradient(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.actor[0].weight.register_hook(lambda grad: grad * float("nan"))
    monkeypatch.setattr(ActorCritic, "__init__", bad_gradient)
    with pytest.raises(ValueError, match="gradients.*finite"):
        train_ppo(num_envs=1, rollout_steps=2, max_iterations=1, num_epochs=1,
            save_path=tmp_path / "bad.pt", device="cpu", verbose=False, terrain_mode="flat")
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("field,bad", [("observation_schema", None), ("terrain_mode", "stairs"),
    ("obs_dim", 37), ("squash_actions", "yes")])
def test_audit_resume_validates_contract_before_writes(tmp_path, field, bad):
    from scripts.train_ppo import train_ppo
    path = tmp_path / "contract_ppo.pt"
    options = dict(num_envs=1, rollout_steps=2, max_iterations=1, num_epochs=1,
        save_path=path, device="cpu", verbose=False, terrain_mode="flat")
    train_ppo(**options)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint[field] = bad
    torch.save(checkpoint, path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.pt")}
    with pytest.raises(ValueError):
        train_ppo(resume=True, **options)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.pt")}


def test_actor_critic_architecture_and_forward():
    """Verify ActorCritic neural architecture, dimensions, activations, and parameter gradients."""
    from scripts.train_ppo import ActorCritic

    obs_dim = 42
    action_dim = 8
    hidden_dim = 256

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    # 1. Check module attributes
    assert model.obs_dim == 42
    assert model.action_dim == 8
    assert isinstance(model.log_std, torch.nn.Parameter)
    assert model.log_std.shape == (8,)
    assert model.log_std.requires_grad is True

    # 2. Check layer dimensions and ELU activations
    # Actor: Linear(42, 256) -> ELU -> Linear(256, 256) -> ELU -> Linear(256, 8)
    assert len(model.actor) == 5
    assert isinstance(model.actor[0], torch.nn.Linear)
    assert model.actor[0].in_features == 42
    assert model.actor[0].out_features == 256
    assert isinstance(model.actor[1], torch.nn.ELU)
    assert isinstance(model.actor[2], torch.nn.Linear)
    assert model.actor[2].in_features == 256
    assert model.actor[2].out_features == 256
    assert isinstance(model.actor[3], torch.nn.ELU)
    assert isinstance(model.actor[4], torch.nn.Linear)
    assert model.actor[4].in_features == 256
    assert model.actor[4].out_features == 8

    # Critic: Linear(42, 256) -> ELU -> Linear(256, 256) -> ELU -> Linear(256, 1)
    assert len(model.critic) == 5
    assert isinstance(model.critic[0], torch.nn.Linear)
    assert model.critic[0].in_features == 42
    assert model.critic[0].out_features == 256
    assert isinstance(model.critic[1], torch.nn.ELU)
    assert isinstance(model.critic[2], torch.nn.Linear)
    assert model.critic[2].in_features == 256
    assert model.critic[2].out_features == 256
    assert isinstance(model.critic[3], torch.nn.ELU)
    assert isinstance(model.critic[4], torch.nn.Linear)
    assert model.critic[4].in_features == 256
    assert model.critic[4].out_features == 1

    # 3. Single observation forward pass
    single_obs = torch.randn(1, 42)
    action_mean = model.forward_actor(single_obs)
    val = model.forward_critic(single_obs)
    assert action_mean.shape == (1, 8)
    assert val.shape == (1,)

    # 4. Batched forward pass and get_action_and_value
    batch_size = 16
    batch_obs = torch.randn(batch_size, 42)
    actions, log_probs, entropy, values = model.get_action_and_value(batch_obs)

    assert actions.shape == (batch_size, 8)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert values.shape == (batch_size,)

    # 5. Evaluate existing actions under distribution
    fixed_actions = torch.clamp(actions, -1.0, 1.0)
    _, eval_log_probs, eval_entropy, eval_values = model.get_action_and_value(
        batch_obs, action=fixed_actions
    )
    assert eval_log_probs.shape == (batch_size,)
    assert eval_entropy.shape == (batch_size,)
    assert eval_values.shape == (batch_size,)

    # 6. Backward pass verifies gradients flow to all components (actor, critic, log_std)
    loss = eval_log_probs.sum() + eval_values.sum() - eval_entropy.sum()
    loss.backward()

    for p in model.actor.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    for p in model.critic.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    assert model.log_std.grad is not None


def test_gae_advantages_calculation():
    """Verify Generalized Advantage Estimation (GAE) with discount gamma and lambda."""
    from scripts.train_ppo import compute_gae

    num_steps = 4
    num_envs = 2
    gamma = 0.99
    gae_lambda = 0.95

    rewards = torch.tensor([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5]], dtype=torch.float32)
    dones = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    values = torch.tensor([[0.8, 0.4], [0.8, 0.4], [0.8, 0.4], [0.8, 0.4]], dtype=torch.float32)
    next_value = torch.tensor([0.8, 0.4], dtype=torch.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    assert advantages.shape == (num_steps, num_envs)
    assert returns.shape == (num_steps, num_envs)
    assert torch.all(returns == advantages + values)
    assert not torch.isnan(advantages).any()
    assert not torch.isinf(advantages).any()


def test_ppo_training_dry_run_api(tmp_path):
    """Test train_ppo function executes 1 dry-run iteration and saves a valid checkpoint."""
    from scripts.train_ppo import train_ppo, ActorCritic

    checkpoint_path = tmp_path / "microduck_ppo_dryrun.pt"

    result = train_ppo(
        num_envs=4,
        rollout_steps=4,
        max_iterations=1,
        save_path=str(checkpoint_path),
        dry_run=True,
    )

    assert result["success"] is True
    assert result["iterations"] == 1
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0
    best_path = tmp_path / "microduck_ppo_dryrun_best.pt"
    assert not best_path.exists()
    assert result["best_checkpoint_path"] is None

    # Load checkpoint and verify content
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt["obs_dim"] == 42
    assert ckpt["action_dim"] == 8
    assert ckpt["hidden_dim"] == 256

    # Test restoring model from checkpoint
    restored_model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=256, critic_obs_dim=ckpt.get("critic_obs_dim", 42))
    restored_model.load_state_dict(ckpt["model_state_dict"])
    dummy_obs = torch.zeros(1, 42)
    act = restored_model.forward_actor(dummy_obs)
    assert act.shape == (1, 8)


def test_ppo_training_cli_dry_run(tmp_path):
    """Test invoking scripts/train_ppo.py via CLI with --dry-run flag."""
    script_path = REPO_ROOT / "scripts" / "train_ppo.py"
    assert script_path.exists(), f"Missing script: {script_path}"

    ckpt_path = tmp_path / "cli_checkpoint.pt"
    cmd = [
        sys.executable,
        str(script_path),
        "--dry-run",
        "--save-path",
        str(ckpt_path),
        "--best-save-path",
        str(tmp_path / "cli_checkpoint_best.pt"),
        "--num-envs",
        "4",
        "--rollout-steps",
        "4",
        "--max-iterations",
        "1",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"train_ppo.py CLI failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "Dry-run mode enabled" in res.stdout or "dry-run" in res.stdout.lower()
    assert ckpt_path.exists()
    assert ckpt_path.stat().st_size > 1000


def test_ppo_training_cli_dry_run_iters_alias(tmp_path):
    """Test train_ppo.py --iters alias and 2-iteration dry-run execution."""
    script_path = REPO_ROOT / "scripts" / "train_ppo.py"
    ckpt_path = tmp_path / "cli_checkpoint_2iters.pt"
    cmd = [
        sys.executable,
        str(script_path),
        "--dry-run",
        "--save-path",
        str(ckpt_path),
        "--best-save-path",
        str(tmp_path / "cli_checkpoint_2iters_best.pt"),
        "--num-envs",
        "4",
        "--rollout-steps",
        "4",
        "--iters",
        "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"train_ppo.py --iters failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "2 iteration(s)" in res.stdout
    assert "[Iteration 0002/2]" in res.stdout
    assert ckpt_path.exists()


def test_duck_cli_train_dry_run(isolated_cli_env):
    """Test duck.py train CLI with --dry-run and --iters flags."""
    duck_script = REPO_ROOT / "duck.py"
    cmd1 = [sys.executable, str(duck_script), "train", "--dry-run"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, env=isolated_cli_env)
    assert res1.returncode == 0, f"duck.py train --dry-run failed:\nSTDOUT: {res1.stdout}\nSTDERR: {res1.stderr}"
    assert "1 iteration(s)" in res1.stdout
    assert "DRY-RUN" in res1.stdout

    cmd2 = [sys.executable, str(duck_script), "train", "--iters", "2", "--dry-run"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, env=isolated_cli_env)
    assert res2.returncode == 0, f"duck.py train --iters 2 --dry-run failed:\nSTDOUT: {res2.stdout}\nSTDERR: {res2.stderr}"
    assert "2 iteration(s)" in res2.stdout
    assert "[Iteration 0002/2]" in res2.stdout


def test_export_policy_onnx_and_onnxruntime(tmp_path):
    """Verify ONNX model export for Raspberry Pi 5 with input 'obs' [1, 42], output 'actions' [1, 8]."""
    import onnx
    import onnxruntime as ort
    from scripts.train_ppo import ActorCritic
    from scripts.export_policy import export_onnx

    # Create model
    torch.manual_seed(42)
    model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=256)
    model.eval()

    onnx_path = tmp_path / "microduck_rpi5_policy.onnx"
    export_onnx(model, onnx_path, opset=13)

    assert onnx_path.exists()

    # 1. Validate ONNX structure with onnx.checker
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. Inspect graph input / output names and shapes
    inputs = onnx_model.graph.input
    outputs = onnx_model.graph.output
    assert len(inputs) == 1
    assert inputs[0].name == "obs"
    assert len(outputs) == 1
    assert outputs[0].name == "actions"

    # 3. Execute with onnxruntime
    session = ort.InferenceSession(str(onnx_path))
    test_obs = np.random.randn(1, 42).astype(np.float32)
    ort_outputs = session.run(["actions"], {"obs": test_obs})
    ort_actions = ort_outputs[0]

    assert ort_actions.shape == (1, 8)

    # 4. Compare with PyTorch output
    with torch.no_grad():
        pt_actions = model.forward_actor(torch.from_numpy(test_obs)).numpy()

    np.testing.assert_allclose(ort_actions, pt_actions, atol=1e-5, rtol=1e-5)

    # 5. Batched evaluation test
    batch_obs = np.random.randn(5, 42).astype(np.float32)
    batch_ort_actions = session.run(["actions"], {"obs": batch_obs})[0]
    assert batch_ort_actions.shape == (5, 8)


def test_export_esp32_c_headers_syntax_and_compilation(tmp_path):
    """Verify INT8 quantized weights and INT32 biases header generation and C++ compilation."""
    from scripts.train_ppo import ActorCritic
    from scripts.export_policy import export_esp32_headers

    torch.manual_seed(42)
    model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=256)

    weights_header_path = tmp_path / "esp32_policy_weights.h"
    config_header_path = tmp_path / "esp32_quant_config.h"

    export_esp32_headers(
        model=model,
        weights_output_path=weights_header_path,
        config_output_path=config_header_path,
        scale_obs=127.0,
        scale_act=127.0,
    )

    assert weights_header_path.exists()
    assert config_header_path.exists()

    weights_content = weights_header_path.read_text(encoding="utf-8")
    config_content = config_header_path.read_text(encoding="utf-8")

    # Check config content
    assert "#ifndef ESP32_QUANT_CONFIG_H" in config_content
    assert "#define OBS_DIM 42" in config_content
    assert "#define ACTION_DIM 8" in config_content
    assert "OBS_MAX_BOUNDS" in config_content

    # Check weights content
    assert "#ifndef ESP32_POLICY_WEIGHTS_H" in weights_content
    assert "POLICY_W1" in weights_content
    assert "POLICY_B1" in weights_content
    assert "POLICY_W2" in weights_content
    assert "POLICY_B2" in weights_content
    assert "POLICY_W3" in weights_content
    assert "POLICY_B3" in weights_content
    assert "int8_t" in weights_content
    assert "int32_t" in weights_content
    assert "evaluate_policy_int8" in weights_content

    # Validate C++ compilation with g++
    cpp_source = f"""#include <iostream>
#include <cassert>
#include <cmath>
#include "{config_header_path.as_posix()}"
#include "{weights_header_path.as_posix()}"

int main() {{
    assert(OBS_DIM == 42);
    assert(ACTION_DIM == 8);
    assert(POLICY_LAYER1_IN == 42);
    assert(POLICY_LAYER1_OUT == 256);
    assert(POLICY_LAYER2_IN == 256);
    assert(POLICY_LAYER2_OUT == 256);
    assert(POLICY_LAYER3_IN == 256);
    assert(POLICY_LAYER3_OUT == 8);

    // Test zero observation inference
    int8_t zero_obs[42] = {{0}};
    int8_t actions[8] = {{0}};
    evaluate_policy_int8(zero_obs, actions);

    for (int i = 0; i < 8; ++i) {{
        assert(actions[i] >= -128 && actions[i] <= 127);
    }}

    // Test non-zero observation
    int8_t test_obs[42];
    for (int i = 0; i < 42; ++i) {{
        test_obs[i] = (int8_t)((i % 20) - 10);
    }}
    evaluate_policy_int8(test_obs, actions);

    std::cout << "ESP32 policy weights compilation and evaluation verified." << std::endl;
    return 0;
}}
"""
    test_cpp_file = tmp_path / "test_esp32_weights.cpp"
    test_cpp_file.write_text(cpp_source, encoding="utf-8")

    out_exe = tmp_path / "test_esp32_weights.exe"
    compile_cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        str(test_cpp_file),
        "-o",
        str(out_exe),
    ]
    compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert compile_res.returncode == 0, f"g++ failed: {compile_res.stderr}\nSTDOUT: {compile_res.stdout}"

    run_res = subprocess.run([str(out_exe)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Running compiled test failed: {run_res.stderr}\nSTDOUT: {run_res.stdout}"
    assert "ESP32 policy weights compilation and evaluation verified." in run_res.stdout


def test_cli_train_and_export_end_to_end(tmp_path):
    """Verify full CLI flow from PPO dry-run checkpoint generation to RPi5 ONNX and ESP32 headers export."""
    train_script = REPO_ROOT / "scripts" / "train_ppo.py"
    export_script = REPO_ROOT / "scripts" / "export_policy.py"

    ckpt_path = tmp_path / "pipeline_test.pt"
    best_path = tmp_path / "pipeline_test_best.pt"
    onnx_path = tmp_path / "exported_pipeline.onnx"
    pizero_onnx_path = tmp_path / "exported_pizero.onnx"
    weights_path = tmp_path / "esp32_weights.h"
    config_path = tmp_path / "esp32_config.h"

    # 1. Run train_ppo with --dry-run
    train_cmd = [
        sys.executable,
        str(train_script),
        "--dry-run",
        "--save-path",
        str(ckpt_path),
        "--best-save-path",
        str(best_path),
        "--num-envs",
        "4",
        "--rollout-steps",
        "4",
    ]
    res_train = subprocess.run(train_cmd, capture_output=True, text=True)
    assert res_train.returncode == 0, f"train_ppo failed:\n{res_train.stderr}"
    assert ckpt_path.exists()

    # Each target uses its own model checkpoint; transfer is explicit with new outputs.
    for target, model_path, output_flag, output in (
        ("rpi5", "pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml", "--onnx-output", onnx_path),
        ("pizero2w", "pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml", "--pizero-onnx-output", pizero_onnx_path),
        ("esp32", "pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml", "--esp32-weights-output", weights_path),
    ):
        selected_checkpoint = ckpt_path
        if target != "rpi5":
            selected_checkpoint = tmp_path / f"{target}_ppo.pt"
            res_transfer = subprocess.run([sys.executable, str(train_script), "--dry-run",
                "--model-path", str(REPO_ROOT / model_path), "--load-checkpoint", str(ckpt_path),
                "--save-path", str(selected_checkpoint), "--num-envs", "1", "--rollout-steps", "2"],
                capture_output=True, text=True)
            assert res_transfer.returncode == 0, res_transfer.stderr
        export_cmd = [sys.executable, str(export_script), "--target", target,
            "--checkpoint", str(selected_checkpoint), output_flag, str(output),
            "--esp32-config-output", str(config_path)]
        res_export = subprocess.run(export_cmd, capture_output=True, text=True)
        assert res_export.returncode == 0, f"export_policy failed:\n{res_export.stderr}"
    assert all(p.exists() for p in (onnx_path, pizero_onnx_path, weights_path, config_path))


def test_best_model_saved_separately_from_latest(tmp_path, monkeypatch):
    """Verify that training maintains separate latest and best model checkpoints."""
    from scripts.train_ppo import train_ppo
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))

    latest_path = tmp_path / "microduck_ppo.pt"
    best_path = tmp_path / "microduck_best.pt"

    res = train_ppo(
        num_envs=4,
        rollout_steps=4,
        max_iterations=1,
        save_path=str(latest_path),
        best_save_path=str(best_path),
        dry_run=False,
    )

    assert res["success"] is True

    assert latest_path.exists()
    assert best_path.exists()
    assert (tmp_path / "microduck_latest.pt").exists()

    ckpt_latest = torch.load(str(latest_path), map_location="cpu", weights_only=False)
    ckpt_best = torch.load(str(best_path), map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt_latest
    assert "model_state_dict" in ckpt_best


def test_train_ppo_resume_preserves_timesteps_iterations_and_target_latest(tmp_path):
    from scripts.train_ppo import train_ppo
    save_p = tmp_path / "cyberduck_v2_ppo.pt"
    best_p = tmp_path / "cyberduck_v2_best.pt"
    res1 = train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, max_iterations=1, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        save_path=str(save_p), best_save_path=str(best_p), device="cpu", log_interval=1, save_interval=1,
    )
    assert res1["total_timesteps"] == 8
    assert (tmp_path / "cyberduck_v2_latest.pt").exists()
    assert not (tmp_path / "microduck_latest.pt").exists()

    res2 = train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, max_iterations=1, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        save_path=str(save_p), best_save_path=str(best_p), resume=True, device="cpu", log_interval=1, save_interval=1,
    )
    assert res2["total_timesteps"] == 16
    assert res2["iterations"] == 2


def test_duck_train_v2_resume_prefers_ppo_over_best(monkeypatch):
    """Verify duck.py train --target v2 --resume prefers cyberduck_v2_ppo.pt over _best.pt."""
    import duck
    captured_cmd = []

    def mock_run(cmd):
        captured_cmd.extend(cmd)
        class Dummy:
            returncode = 0
        return Dummy()

    monkeypatch.setattr(duck.subprocess, "run", mock_run)

    # Case 1: cyberduck_v2_ppo.pt exists -> pass --resume
    monkeypatch.setattr(duck.Path, "exists", lambda self: "cyberduck_v2_ppo.pt" in str(self))
    args = duck.argparse.Namespace(
        command="train", target="v2", resume=True, envs=16, iters=10, dry_run=False,
        stand_first=False, device="cpu", rollout_steps=32, save_interval=10,
        log_interval=10, terrain="hurdles",
    )
    duck.cmd_train(args)
    assert "--resume" in captured_cmd
    assert "--load-checkpoint" not in captured_cmd

    # Case 2: only cyberduck_v2_best.pt exists -> pass --load-checkpoint ...cyberduck_v2_best.pt
    captured_cmd.clear()
    monkeypatch.setattr(duck.Path, "exists", lambda self: "cyberduck_v2_best.pt" in str(self))
    duck.cmd_train(args)
    assert "--load-checkpoint" in captured_cmd
    assert any("cyberduck_v2_best.pt" in arg for arg in captured_cmd)


def test_runner_mean_distance_y_and_fast_rollout_parity():
    """Verify runner.get_mean_distance_y() and fast rollout step parity against standard step."""
    from training.run_parallel_sim import ParallelSimRunner
    runner = ParallelSimRunner(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=4,
        decimation=2,
    )
    try:
        assert hasattr(runner, "get_mean_distance_y")
        mean_y = runner.get_mean_distance_y()
        assert isinstance(mean_y, float)

        actions = np.zeros((4, runner.action_dim), dtype=np.float32)
        # Fast path
        assert hasattr(runner, "step_arrays")
        obs_fast, rew_fast, dones_fast, fast_info = runner.step_arrays(actions)
        assert isinstance(fast_info, dict)
        assert "time_outs" in fast_info
        assert "tracking_lin_vel" in fast_info
        assert isinstance(fast_info["time_outs"], np.ndarray)
        assert fast_info["time_outs"].dtype == bool
        assert obs_fast.shape == (4, runner.obs_dim)
        assert rew_fast.shape == (4,)
        assert dones_fast.shape == (4,)
    finally:
        runner.close()


def test_best_save_path_strips_ppo_suffix_consistently(tmp_path, monkeypatch):
    """Default best_save_path strips trailing _ppo just like _latest.pt."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    from scripts.train_ppo import train_ppo

    save_p = tmp_path / "cyberduck_v2_ppo.pt"
    res = train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2,
        rollout_steps=4,
        max_iterations=1,
        save_path=str(save_p),
        dry_run=False,
        device="cpu",
    )
    assert Path(res["best_checkpoint_path"]) == (tmp_path / "cyberduck_v2_best.pt").resolve()
    assert Path(res["latest_checkpoint_path"]) == (tmp_path / "cyberduck_v2_latest.pt").resolve()
    assert (tmp_path / "cyberduck_v2_best.pt").exists()


def test_actor_critic_clamps_extreme_loaded_log_std_in_forward():
    """Extreme log_std values must clamp std to [exp(min_log_std), exp(max_log_std)] without breaking normal grads."""
    from scripts.train_ppo import ActorCritic

    model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=256)
    assert model.min_log_std == pytest.approx(-3.0)
    assert model.max_log_std == pytest.approx(0.0)

    obs = torch.zeros(4, 42)
    with torch.no_grad():
        model.log_std.fill_(5.0)
    _, _, entropy_hi, _ = model.get_action_and_value(obs)
    expected_max_entropy = 8 * 0.5 * math.log(2.0 * math.pi * math.e * (math.exp(0.0) ** 2))
    assert float(entropy_hi.mean().item()) == pytest.approx(expected_max_entropy, rel=1e-4)

    with torch.no_grad():
        model.log_std.fill_(-20.0)
    _, _, entropy_lo, _ = model.get_action_and_value(obs)
    expected_min_entropy = 8 * 0.5 * math.log(2.0 * math.pi * math.e * (math.exp(-3.0) ** 2))
    assert float(entropy_lo.mean().item()) == pytest.approx(expected_min_entropy, rel=1e-4)


def test_critic_value_loss_updates_across_large_return_gap(monkeypatch, tmp_path):
    """Critic must update values even when |mb_ret - mb_val| >> clip_epsilon (no v_clipped zero-grad lock)."""
    from scripts.train_ppo import ActorCritic, train_ppo
    import training.run_parallel_sim as sim_mod

    orig_step_arrays = sim_mod.ParallelSimRunner.step_arrays

    def high_reward_step(self, actions):
        obs, _, dones, _ = orig_step_arrays(self, actions)
        return obs, np.full(self.num_envs, 5.0, dtype=np.float32), np.zeros(self.num_envs, dtype=bool), {
            "time_outs": np.zeros(self.num_envs, dtype=bool),
            "tracking_lin_vel": 1.0,
        }

    monkeypatch.setattr(sim_mod.ParallelSimRunner, "step_arrays", high_reward_step)
    save_p = tmp_path / "critic_ppo.pt"
    train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=4,
        rollout_steps=8,
        mini_batch_size=8,
        num_epochs=4,
        max_iterations=1,
        learning_rate=1e-2,
        dry_run=False,
        device="cpu",
        save_path=str(save_p),
        best_save_path=str(tmp_path / "critic_best.pt"),
    )
    ckpt = torch.load(str(save_p), map_location="cpu", weights_only=False)
    restored = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=256, critic_obs_dim=ckpt.get("critic_obs_dim", 42))
    restored.load_state_dict(ckpt["model_state_dict"])
    dummy_crit_obs = torch.zeros(1, ckpt.get("critic_obs_dim", 42))
    assert float(restored.forward_critic(dummy_crit_obs).item()) > 0.5


def test_actor_critic_rejects_inverted_log_std_bounds():
    """ActorCritic must raise ValueError when min_log_std > max_log_std."""
    from scripts.train_ppo import ActorCritic

    with pytest.raises(ValueError, match="min_log_std .* must be <= max_log_std"):
        ActorCritic(min_log_std=1.0, max_log_std=-1.0)


def test_ppo_adaptive_kl_and_entropy_decay(tmp_path):
    """Strict target_kl lowers learning_rate; entropy_coef linearly reaches effective end."""
    from scripts.train_ppo import train_ppo

    res = train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=4,
        rollout_steps=8,
        mini_batch_size=8,
        num_epochs=4,
        max_iterations=2,
        learning_rate=3e-4,
        target_kl=1e-7,  # force KL > 2 * target_kl unless KL == 0
        min_lr=1e-5,
        max_lr=1e-3,
        entropy_coef=0.01,
        entropy_coef_end=0.002,
        dry_run=False,
        device="cpu",
        save_path=str(tmp_path / "kl_ppo.pt"),
    )
    assert res["metrics"]["approx_kl"] >= 0.0
    assert res["metrics"]["learning_rate"] < 3e-4
    assert res["metrics"]["learning_rate"] >= 1e-5
    assert res["metrics"]["entropy_coef"] == pytest.approx(0.002)


def test_ppo_target_kl_zero_disables_adaptive_lr_and_zero_entropy_stays_zero(tmp_path):
    """target_kl=0 disables LR adaptation/early-stop; entropy_coef=0 stays 0 even with entropy_coef_end=None."""
    from scripts.train_ppo import train_ppo

    res = train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=4,
        rollout_steps=4,
        mini_batch_size=4,
        num_epochs=2,
        max_iterations=3,
        learning_rate=3e-4,
        target_kl=0.0,
        entropy_coef=0.0,
        entropy_coef_end=None,
        dry_run=False,
        device="cpu",
        save_path=str(tmp_path / "fixed_ppo.pt"),
    )
    assert res["metrics"]["learning_rate"] == pytest.approx(3e-4)
    assert res["metrics"]["entropy_coef"] == pytest.approx(0.0)


def test_resume_restores_lr_and_handles_legacy_checkpoint_without_lr(tmp_path):
    """Resuming restores saved learning_rate and handles older checkpoints missing that key."""
    from scripts.train_ppo import train_ppo

    save_p = tmp_path / "resume_lr_ppo.pt"
    train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2,
        rollout_steps=4,
        mini_batch_size=4,
        num_epochs=1,
        max_iterations=1,
        learning_rate=1.2e-4,
        target_kl=0.0,
        device="cpu",
        save_path=str(save_p),
    )
    # Verify legacy checkpoint without 'learning_rate' key works cleanly
    ckpt = torch.load(str(save_p), map_location="cpu", weights_only=False)
    assert ckpt["learning_rate"] == pytest.approx(1.2e-4)
    ckpt.pop("learning_rate")
    torch.save(ckpt, str(save_p))

    res_legacy = train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2,
        rollout_steps=4,
        mini_batch_size=4,
        num_epochs=1,
        max_iterations=1,
        learning_rate=2.5e-4,
        target_kl=0.0,
        resume=True,
        device="cpu",
        save_path=str(save_p),
    )
    assert res_legacy["metrics"]["learning_rate"] == pytest.approx(2.5e-4)


def test_stage_promotion_best_ckpt_uses_rollout_stage_and_resets_baseline(monkeypatch, tmp_path):
    """Promoting Stage 0->1 after iter 1 must label best_ckpt Stage 0; iter 2 Stage 1 rollout resets baseline."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    import scripts.train_ppo as ppo_mod
    import training.run_parallel_sim as sim_mod

    orig_step = sim_mod.ParallelSimRunner.step_arrays
    reward_per_iter = {"val": 2.5}
    dist_per_iter = {"val": 0.9}

    def scripted_step(self, actions):
        obs, _, dones, _ = orig_step(self, actions)
        return obs, np.full(self.num_envs, reward_per_iter["val"], dtype=np.float32), dones, {
            "time_outs": np.zeros(self.num_envs, dtype=bool),
            "tracking_lin_vel": 1.0,
        }

    monkeypatch.setattr(sim_mod.ParallelSimRunner, "step_arrays", scripted_step)
    monkeypatch.setattr(sim_mod.ParallelSimRunner, "get_mean_distance_x", lambda self: dist_per_iter["val"])
    monkeypatch.setattr(ppo_mod, "next_stage", lambda **kw: (1, 0))

    save_p = tmp_path / "stage_ppo.pt"
    best_p = tmp_path / "stage_best.pt"
    res1 = ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, save_path=str(save_p), best_save_path=str(best_p), device="cpu", save_interval=1,
    )
    best1 = torch.load(str(best_p), map_location="cpu", weights_only=False)
    ckpt1 = torch.load(str(save_p), map_location="cpu", weights_only=False)
    assert res1["stage"] == 1
    assert ckpt1["stage"] == 1
    assert ckpt1["best_stage"] == 0
    assert best1["stage"] == 0

    reward_per_iter["val"] = 1.1
    dist_per_iter["val"] = 0.7
    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, resume=True, save_path=str(save_p), best_save_path=str(best_p), device="cpu", save_interval=1,
    )
    best2 = torch.load(str(best_p), map_location="cpu", weights_only=False)
    assert best2["stage"] == 1
    assert best2["best_reward"] == pytest.approx(1.1, abs=1e-4)


def test_collapsed_promoted_stage_rollout_does_not_overwrite_best_ckpt(monkeypatch, tmp_path):
    """Negative-distance collapsed Stage 1 rollout must NOT overwrite a valid Stage 0 best checkpoint."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    import scripts.train_ppo as ppo_mod
    import training.run_parallel_sim as sim_mod

    save_p = tmp_path / "collapse_ppo.pt"
    best_p = tmp_path / "collapse_best.pt"
    orig_step = sim_mod.ParallelSimRunner.step_arrays

    def bad_stage1_step(self, actions):
        obs, _, dones, _ = orig_step(self, actions)
        return obs, np.full(self.num_envs, -1.0, dtype=np.float32), dones, {
            "time_outs": np.zeros(self.num_envs, dtype=bool),
            "tracking_lin_vel": 0.0,
        }

    # Seed valid Stage 0 best and Stage 1 resume ckpt
    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, save_path=str(save_p), best_save_path=str(best_p), device="cpu", save_interval=1,
    )
    best_seed = torch.load(str(best_p), map_location="cpu", weights_only=False)
    best_seed["stage"] = 0
    best_seed["best_reward"] = 2.0
    torch.save(best_seed, str(best_p))
    ckpt_seed = torch.load(str(save_p), map_location="cpu", weights_only=False)
    ckpt_seed["stage"] = 1
    ckpt_seed["best_stage"] = 0
    torch.save(ckpt_seed, str(save_p))

    monkeypatch.setattr(ppo_mod, "evaluate_deterministic_rollout", lambda **kw: (-0.2, -1.0))
    monkeypatch.setattr(sim_mod.ParallelSimRunner, "step_arrays", bad_stage1_step)
    monkeypatch.setattr(sim_mod.ParallelSimRunner, "get_mean_distance_x", lambda self: -0.2)
    monkeypatch.setattr(ppo_mod, "next_stage", lambda **kw: (1, 0))
    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, resume=True, save_path=str(save_p), best_save_path=str(best_p), device="cpu", save_interval=1,
    )
    best_after = torch.load(str(best_p), map_location="cpu", weights_only=False)
    assert best_after["stage"] == 0
    assert best_after["best_reward"] == pytest.approx(2.0)


def test_resume_stage1_with_legacy_stage0_best_ckpt_unlocks_stage1_best(monkeypatch, tmp_path):
    """Resuming legacy Stage 1 ppo.pt (missing 'best_stage') + Stage 0 best.pt allows Stage 1 rollout to update best.pt."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    import scripts.train_ppo as ppo_mod
    import training.run_parallel_sim as sim_mod

    save_p = tmp_path / "legacy_stage_ppo.pt"
    best_p = tmp_path / "legacy_stage_best.pt"
    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, save_path=str(save_p), best_save_path=str(best_p), device="cpu", save_interval=1,
    )
    best_seed = torch.load(str(best_p), map_location="cpu", weights_only=False)
    best_seed["stage"] = 0
    best_seed["best_reward"] = 2.5
    best_seed.pop("best_stage", None)
    best_seed.pop("best_selection", None)
    best_seed.pop("best_det_distance_x", None)
    torch.save(best_seed, str(best_p))
    ckpt_seed = torch.load(str(save_p), map_location="cpu", weights_only=False)
    ckpt_seed["stage"] = 1
    ckpt_seed.pop("best_stage", None)
    ckpt_seed.pop("learning_rate", None)
    torch.save(ckpt_seed, str(save_p))

    orig_step = sim_mod.ParallelSimRunner.step_arrays

    def good_stage1_step(self, actions):
        obs, _, dones, _ = orig_step(self, actions)
        return obs, np.full(self.num_envs, 1.3, dtype=np.float32), dones, {
            "time_outs": np.zeros(self.num_envs, dtype=bool),
            "tracking_lin_vel": 1.0,
        }

    monkeypatch.setattr(sim_mod.ParallelSimRunner, "step_arrays", good_stage1_step)
    monkeypatch.setattr(sim_mod.ParallelSimRunner, "get_mean_distance_x", lambda self: 0.6)
    monkeypatch.setattr(ppo_mod, "next_stage", lambda **kw: (1, 0))
    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, resume=True, save_path=str(save_p), best_save_path=str(best_p), device="cpu", save_interval=1,
    )
    best_after = torch.load(str(best_p), map_location="cpu", weights_only=False)
    assert best_after["stage"] == 1
    assert best_after["best_reward"] == pytest.approx(1.3, abs=1e-4)


def test_stage_promotion_caps_learning_rate_to_base_lr(monkeypatch, tmp_path):
    """When adaptive KL boosts LR, advancing stage caps current_lr to base learning_rate."""
    import scripts.train_ppo as ppo_mod

    monkeypatch.setattr(ppo_mod, "next_stage", lambda **kw: (1, 0))

    save_p = tmp_path / "lr_cap_ppo.pt"
    res = ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, learning_rate=2e-4, target_kl=0.015,
        save_path=str(save_p), device="cpu",
    )
    assert res["metrics"]["learning_rate"] <= 2e-4
    ckpt = torch.load(str(save_p), map_location="cpu", weights_only=False)
    assert ckpt["learning_rate"] <= 2e-4


def test_stair_stage_promotion_caps_learning_rate_and_labels_best_ckpt(monkeypatch, tmp_path):
    """When terrain_mode='stairs', advancing stair stage caps current_lr and labels best_ckpt correctly."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    import scripts.train_ppo as ppo_mod

    monkeypatch.setattr(ppo_mod, "next_stair_stage", lambda **kw: (1, 0))

    save_p = tmp_path / "stairs_ppo.pt"
    best_p = tmp_path / "stairs_best.pt"
    res = ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, learning_rate=2e-4, terrain_mode="stairs",
        save_path=str(save_p), best_save_path=str(best_p), device="cpu", save_interval=1,
    )
    assert res["stage"] == 1
    best_ckpt = torch.load(str(best_p), map_location="cpu", weights_only=False)
    assert best_ckpt["stage"] == 0
    assert best_ckpt["best_stage"] == 0
    # Best describes the evaluated Stage 0 candidate before curriculum advances.
    assert best_ckpt["learning_rate"] > 0
    latest_ckpt = torch.load(str(save_p), map_location="cpu", weights_only=False)
    assert latest_ckpt["learning_rate"] <= 2e-4
    assert res["metrics"]["learning_rate"] <= 2e-4


def test_corrupted_best_checkpoint_does_not_crash(tmp_path):
    """Corrupted _best.pt is gracefully handled without crash and training completes."""
    import scripts.train_ppo as ppo_mod

    save_p = tmp_path / "corrupt_ppo.pt"
    best_p = tmp_path / "corrupt_best.pt"
    best_p.write_text("corrupted non-pickle data")

    res = ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2, rollout_steps=4, mini_batch_size=4, num_epochs=1,
        max_iterations=1, save_path=str(save_p), best_save_path=str(best_p), device="cpu",
    )
    assert res["success"] is True


def test_transfer_load_checkpoint_skips_velocity_warmup_and_iter1_best_lockout(monkeypatch, tmp_path):
    """Transfer loading skips early velocity clamping and prevents Iter-1 rollout from locking best_checkpoint."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    import scripts.train_ppo as ppo_mod
    import training.run_parallel_sim as sim_mod

    seed_save_p = tmp_path / "seed_transfer_ppo.pt"
    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2,
        rollout_steps=4,
        mini_batch_size=4,
        num_epochs=1,
        max_iterations=1,
        device="cpu",
        save_path=str(seed_save_p),
    )

    orig_step = sim_mod.ParallelSimRunner.step_arrays
    orig_reset = sim_mod.ParallelSimRunner.reset
    observed_cmd_vx = []
    step_calls = {"count": 0}

    def scripted_reset(self, *args, **kwargs):
        obs = orig_reset(self, *args, **kwargs)
        self.commands[:, 0] = 0.24
        return obs

    def scripted_step(self, actions):
        observed_cmd_vx.append(float(self.commands[0, 0]))
        obs, _, dones, _ = orig_step(self, actions)
        step_calls["count"] += 1
        # Iteration 1 has 4 steps, Iteration 2 has 4 steps
        reward_val = 3.0 if step_calls["count"] <= 4 else 2.0
        return obs, np.full(self.num_envs, reward_val, dtype=np.float32), dones, {
            "time_outs": np.zeros(self.num_envs, dtype=bool),
            "tracking_lin_vel": 1.0,
        }

    def scripted_dist(self):
        return 0.10 if step_calls["count"] <= 4 else 0.80

    monkeypatch.setattr(sim_mod.ParallelSimRunner, "reset", scripted_reset)
    monkeypatch.setattr(sim_mod.ParallelSimRunner, "step_arrays", scripted_step)
    monkeypatch.setattr(sim_mod.ParallelSimRunner, "get_mean_distance_x", scripted_dist)

    transfer_save_p = tmp_path / "transfer_ppo.pt"
    transfer_best_p = tmp_path / "transfer_best.pt"

    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        load_checkpoint=str(seed_save_p),
        num_envs=2,
        rollout_steps=4,
        mini_batch_size=4,
        num_epochs=1,
        max_iterations=2,
        save_interval=1,
        save_path=str(transfer_save_p),
        best_save_path=str(transfer_best_p),
        device="cpu",
    )

    # 1. Early velocity warmup clamp was skipped (commands[:, 0] stayed 0.24 > 0.174)
    assert len(observed_cmd_vx) > 0
    assert any(cmd > 0.174 for cmd in observed_cmd_vx)

    # 2. Iteration 1 (dist=0.10, reward=3.0) did not lock best.pt; Iteration 2 (dist=0.80, reward=2.0) took it
    assert transfer_best_p.exists()
    best_ckpt = torch.load(str(transfer_best_p), map_location="cpu", weights_only=False)
    assert best_ckpt["best_distance_x"] == pytest.approx(0.80, abs=1e-3)
    assert best_ckpt["best_reward"] == pytest.approx(2.0, abs=1e-3)


def test_transfer_load_checkpoint_warms_up_initial_lr_without_adam_shock(monkeypatch, tmp_path):
    """Transfer loading caps initial Adam lr to <= 1e-4 on Iter 1 update, bounds from ckpt lr, and an eligible deterministic candidate creates best.pt."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    import scripts.train_ppo as ppo_mod

    seed_p = tmp_path / "seed_no_lr_ppo.pt"
    ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=2,
        rollout_steps=4,
        mini_batch_size=4,
        num_epochs=1,
        max_iterations=1,
        device="cpu",
        save_path=str(seed_p),
    )
    # Strip learning_rate and optimizer_state_dict to simulate pure policy weight transfer without prior lr
    ckpt = torch.load(str(seed_p), map_location="cpu", weights_only=False)
    ckpt.pop("learning_rate", None)
    ckpt.pop("optimizer_state_dict", None)
    torch.save(ckpt, str(seed_p))

    captured_step_lrs = []
    orig_adam_step = torch.optim.Adam.step

    def capture_adam_step(self, *args, **kwargs):
        captured_step_lrs.append(float(self.param_groups[0]["lr"]))
        return orig_adam_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture_adam_step)

    save_p = tmp_path / "warmup_ppo.pt"
    best_p = tmp_path / "warmup_best.pt"

    res = ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        load_checkpoint=str(seed_p),
        learning_rate=3e-4,
        target_kl=0.015,
        max_iterations=1,
        rollout_steps=8,
        num_envs=4,
        mini_batch_size=8,
        num_epochs=2,
        device="cpu",
        save_path=str(save_p),
        best_save_path=str(best_p),
    )

    assert len(captured_step_lrs) > 0
    # Adam step should have been capped to <= 1e-4
    assert all(lr <= 1e-4 + 1e-12 for lr in captured_step_lrs)
    # Qualified deterministic evaluation creates best_save_path even for one iteration
    assert best_p.exists()


def test_missing_load_checkpoint_falls_back_to_scratch_warmup(monkeypatch, tmp_path):
    """When load_checkpoint does not exist, training falls back to scratch with velocity warmup applied."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    import scripts.train_ppo as ppo_mod
    import training.run_parallel_sim as sim_mod

    missing_ckpt = tmp_path / "does_not_exist.pt"
    save_p = tmp_path / "scratch_missing_ppo.pt"
    best_p = tmp_path / "scratch_missing_best.pt"

    observed_cmd_vx = []
    orig_step = sim_mod.ParallelSimRunner.step_arrays
    orig_reset = sim_mod.ParallelSimRunner.reset

    def scripted_reset(self, *args, **kwargs):
        obs = orig_reset(self, *args, **kwargs)
        self.commands[:, 0] = 0.25
        return obs

    def scripted_step(self, actions):
        observed_cmd_vx.append(float(self.commands[0, 0]))
        return orig_step(self, actions)

    monkeypatch.setattr(sim_mod.ParallelSimRunner, "reset", scripted_reset)
    monkeypatch.setattr(sim_mod.ParallelSimRunner, "step_arrays", scripted_step)

    res = ppo_mod.train_ppo(
        model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        load_checkpoint=str(missing_ckpt),
        learning_rate=3e-4,
        target_kl=0.015,
        max_iterations=1,
        rollout_steps=4,
        num_envs=2,
        mini_batch_size=4,
        num_epochs=1,
        device="cpu",
        save_path=str(save_p),
        best_save_path=str(best_p),
    )

    assert len(observed_cmd_vx) > 0
    # Velocity warmup clamp applies: max_cmd_vx = 0.15 + 0.35*(1/15) = 0.17333 <= 0.175
    assert all(cmd <= 0.175 for cmd in observed_cmd_vx)
    assert best_p.exists()


def test_running_mean_std_statistics_no_nan_and_clipping():
    """RunningMeanStd online Welford updates, variance convergence, no NaN, and clipping bounds."""
    from scripts.train_ppo import RunningMeanStd

    rms = RunningMeanStd(shape=42, clip=5.0)
    rng = np.random.default_rng(123)

    # 1. Update with distinct non-standard distribution
    true_mean = np.linspace(-3.0, 5.0, 42)
    true_std = np.linspace(0.5, 3.0, 42)
    for _ in range(50):
        batch = rng.normal(loc=true_mean, scale=true_std, size=(64, 42))
        rms.update(batch)

    assert np.allclose(rms.mean, true_mean, atol=0.25)
    assert np.allclose(rms.var, true_std ** 2, atol=0.5)

    # 2. Check normalization shifts to ~0 mean and ~1 variance
    test_batch = rng.normal(loc=true_mean, scale=true_std, size=(256, 42))
    norm_np = rms.normalize(test_batch)
    assert np.all(np.isfinite(norm_np))
    assert np.all(norm_np >= -5.0) and np.all(norm_np <= 5.0)
    assert np.allclose(np.mean(norm_np, axis=0), 0.0, atol=0.25)
    assert np.allclose(np.var(norm_np, axis=0), 1.0, atol=0.35)

    # 3. Check PyTorch tensor normalization and extreme value clipping
    test_tensor = torch.full((10, 42), 1000.0, dtype=torch.float32)
    norm_tensor = rms.normalize(test_tensor)
    assert isinstance(norm_tensor, torch.Tensor)
    assert torch.all(norm_tensor == 5.0)

    # 4. State dict roundtrip
    sd = rms.state_dict()
    rms_loaded = RunningMeanStd(shape=42, clip=5.0)
    rms_loaded.load_state_dict(sd)
    assert np.allclose(rms_loaded.mean, rms.mean)
    assert np.allclose(rms_loaded.var, rms.var)
    assert rms_loaded.count == rms.count


def test_parallel_sim_runner_privileged_obs_and_terminal_critic_obs():
    """ParallelSimRunner privileged_obs_dim is obs_dim + 11, get_privileged_obs() is finite,
    and step() includes terminal_critic_obs before resetting envs."""
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=4,
        decimation=10,
        stage=1,
    )
    try:
        assert hasattr(runner, "privileged_obs_dim")
        assert runner.privileged_obs_dim == runner.obs_dim + 11

        priv_obs = runner.get_privileged_obs()
        assert isinstance(priv_obs, np.ndarray)
        assert priv_obs.shape == (4, runner.privileged_obs_dim)
        assert np.all(np.isfinite(priv_obs))

        # Check subset extraction
        subset_priv = runner.get_privileged_obs(env_indices=[0, 2])
        assert subset_priv.shape == (2, runner.privileged_obs_dim)
        assert np.allclose(subset_priv[0], priv_obs[0])
        assert np.allclose(subset_priv[1], priv_obs[2])

        # Step with fast=True and check terminal_critic_obs key exists in step_info
        actions = np.zeros((4, runner.action_dim), dtype=np.float32)
        _, _, _, fast_info = runner.step(actions, fast=True)
        assert "terminal_critic_obs" in fast_info
        assert isinstance(fast_info["terminal_critic_obs"], np.ndarray)
        assert fast_info["terminal_critic_obs"].shape[-1] == runner.privileged_obs_dim

        # Step with fast=False and check infos
        _, _, _, infos = runner.step(actions, fast=False)
        assert isinstance(infos, list)
        assert len(infos) == 4
    finally:
        runner.close()


def test_asymmetric_critic_actor_critic_forward_and_checkpoint():
    """ActorCritic supports asymmetric critic_obs_dim != obs_dim with clean forward and state_dict."""
    from scripts.train_ppo import ActorCritic

    obs_dim = 42
    critic_obs_dim = 53
    ac = ActorCritic(obs_dim=obs_dim, action_dim=8, hidden_dim=256, critic_obs_dim=critic_obs_dim)
    assert ac.obs_dim == obs_dim
    assert ac.critic_obs_dim == critic_obs_dim
    assert ac.critic[0].in_features == critic_obs_dim
    assert ac.actor[0].in_features == obs_dim

    # Forward with explicit critic_obs
    dummy_actor_obs = torch.randn(8, obs_dim)
    dummy_critic_obs = torch.randn(8, critic_obs_dim)

    act, logp, ent, val = ac.get_action_and_value(dummy_actor_obs, critic_obs=dummy_critic_obs)
    assert act.shape == (8, 8)
    assert logp.shape == (8,)
    assert ent.shape == (8,)
    assert val.shape == (8,)

    crit_val = ac.forward_critic(dummy_critic_obs)
    assert crit_val.shape == (8,)
    assert np.all(np.isfinite(crit_val.detach().numpy()))

    # Checkpoint state_dict roundtrip
    sd = ac.state_dict()
    restored = ActorCritic(obs_dim=obs_dim, action_dim=8, hidden_dim=256, critic_obs_dim=critic_obs_dim)
    restored.load_state_dict(sd)
    assert restored.critic[0].in_features == critic_obs_dim
    assert torch.allclose(restored.forward_critic(dummy_critic_obs), crit_val)


def test_legacy_42dim_checkpoint_transfer_to_asymmetric_critic(tmp_path):
    """Legacy 42-dim critic checkpoint transfers cleanly into 53-dim asymmetric critic model."""
    from scripts.train_ppo import ActorCritic, train_ppo

    # 1. Create a legacy checkpoint where both actor and critic have obs_dim=42
    legacy_model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=256, critic_obs_dim=42)
    legacy_ckpt_path = tmp_path / "legacy_42dim_ckpt.pt"
    legacy_payload = {
        "model_state_dict": legacy_model.state_dict(),
        "actor_state_dict": legacy_model.actor.state_dict(),
        "critic_state_dict": legacy_model.critic.state_dict(),
        "obs_dim": 42,
        "action_dim": 8,
        "hidden_dim": 256,
        "stage": 0,
        "iterations": 5,
        "total_timesteps": 200,
    }
    torch.save(legacy_payload, str(legacy_ckpt_path))

    # 2. Load legacy checkpoint into train_ppo with cyberduck_v2 runner (privileged_obs_dim=53)
    save_path = tmp_path / "asym_trained.pt"
    best_save_path = tmp_path / "asym_best.pt"
    res = train_ppo(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        rollout_steps=4,
        max_iterations=1,
        dry_run=True,
        load_checkpoint=str(legacy_ckpt_path),
        save_path=str(save_path),
        best_save_path=str(best_save_path),
        verbose=False,
    )

    assert res["success"] is True
    assert save_path.exists()
    saved_ckpt = torch.load(str(save_path), map_location="cpu", weights_only=False)
    assert saved_ckpt["obs_dim"] == 42
    assert saved_ckpt["critic_obs_dim"] == 53
    assert "obs_rms" in saved_ckpt
    assert "critic_obs_rms" in saved_ckpt


def test_scratch_run_ignores_stale_higher_stage_best_checkpoint(tmp_path, monkeypatch):
    """A fresh scratch run (resume=False) must overwrite a stale higher-stage _best.pt rather than being locked out."""
    monkeypatch.setattr("scripts.train_ppo.evaluate_deterministic_rollout", lambda **kw: (0.3, 1.0))
    from scripts.train_ppo import train_ppo

    best_path = tmp_path / "scratch_best.pt"
    save_path = tmp_path / "scratch_ppo.pt"
    stale_best = {
        "stage": 1,
        "best_stage": 1,
        "best_reward": 9.9,
        "best_distance_x": 2.0,
        "metrics": {"mean_reward": 9.9, "mean_distance_x": 2.0},
    }
    torch.save(stale_best, str(best_path))

    res = train_ppo(
        model_path="duck_v2/cyberduck_v2.xml",
        start_stage=0,
        num_envs=2,
        rollout_steps=8,
        max_iterations=1,
        dry_run=False,
        resume=False,
        save_path=str(save_path),
        best_save_path=str(best_path),
        verbose=False,
    )

    assert res["success"] is True
    assert best_path.exists()
    overwritten = torch.load(str(best_path), map_location="cpu", weights_only=False)
    assert overwritten["stage"] == 0
    assert overwritten["best_stage"] == 0
    assert overwritten["iterations"] == 1


def test_evaluate_deterministic_rollout_returns_metrics_and_logs(tmp_path, capsys):
    """evaluate_deterministic_rollout computes finite metrics and train_ppo logs DetDistX/DetRew for real runs."""
    from scripts.train_ppo import (
        ActorCritic,
        RunningMeanStd,
        evaluate_deterministic_rollout,
        train_ppo,
    )

    model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=64, critic_obs_dim=53)
    obs_rms = RunningMeanStd(shape=42, clip=10.0)
    det_dist_x, det_rew = evaluate_deterministic_rollout(
        model=model,
        obs_rms=obs_rms,
        model_path=str(REPO_ROOT / "duck_v2" / "cyberduck_v2.xml"),
        stage=0,
        terrain_mode="hurdles",
        max_steps=40,
        device="cpu",
    )
    assert isinstance(det_dist_x, float) and math.isfinite(det_dist_x)
    assert isinstance(det_rew, float) and math.isfinite(det_rew)

    capsys.readouterr()
    res = train_ppo(
        model_path="duck_v2/cyberduck_v2.xml",
        start_stage=0,
        num_envs=1,
        rollout_steps=64,
        max_iterations=1,
        dry_run=False,
        resume=False,
        save_path=str(tmp_path / "det_log_ppo.pt"),
        best_save_path=str(tmp_path / "det_log_best.pt"),
        log_interval=1,
        verbose=True,
    )
    assert res["success"] is True
    out = capsys.readouterr().out
    assert "DetDistX:" in out
    assert "DetRew:" in out


def test_atomic_torch_save_replaces_cleanly(tmp_path):
    """Verify that _atomic_torch_save creates an atomic checkpoint, replaces existing cleanly,
    and cleans up temporary files on failure.
    """
    from scripts.train_ppo import _atomic_torch_save

    ckpt_path = tmp_path / "model_ckpt.pt"
    data1 = {"step": 1, "weights": torch.tensor([1.0, 2.0, 3.0])}

    # 1. Initial write
    _atomic_torch_save(data1, ckpt_path)
    assert ckpt_path.exists()
    loaded1 = torch.load(ckpt_path, weights_only=False)
    assert loaded1["step"] == 1
    assert torch.equal(loaded1["weights"], data1["weights"])

    # Verify no dangling temporary files remain
    tmp_files = list(tmp_path.glob(".*.tmp"))
    assert len(tmp_files) == 0, f"Found unexpected tmp files: {tmp_files}"

    # 2. Overwrite existing checkpoint
    data2 = {"step": 2, "weights": torch.tensor([4.0, 5.0, 6.0])}
    _atomic_torch_save(data2, ckpt_path)
    loaded2 = torch.load(ckpt_path, weights_only=False)
    assert loaded2["step"] == 2
    assert torch.equal(loaded2["weights"], data2["weights"])
    assert len(list(tmp_path.glob(".*.tmp"))) == 0

    # 3. Handle string path argument cleanly
    str_ckpt_path = str(tmp_path / "str_model_ckpt.pt")
    _atomic_torch_save(data1, str_ckpt_path)
    assert Path(str_ckpt_path).exists()
    assert len(list(tmp_path.glob(".*.tmp"))) == 0

    # 4. Cleanup on failure
    class UnpicklableObject:
        def __reduce__(self):
            raise RuntimeError("Cannot pickle this test object")

    bad_data = {"bad": UnpicklableObject()}
    fail_path = tmp_path / "fail_ckpt.pt"
    with pytest.raises(Exception):
        _atomic_torch_save(bad_data, fail_path)

    assert not fail_path.exists()
    assert len(list(tmp_path.glob(".*.tmp"))) == 0


def test_get_privileged_obs_vectorized_subsets_and_eval_runner_reuse(tmp_path):
    """ParallelSimRunner.get_privileged_obs handles empty/arbitrary subsets and eval_runner reuse."""
    from training.run_parallel_sim import ParallelSimRunner
    from scripts.train_ppo import ActorCritic, RunningMeanStd, evaluate_deterministic_rollout

    # 1. Privileged obs with hurdles
    runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=4,
        decimation=10,
        stage=1,
        terrain_mode="hurdles",
    )
    try:
        # Empty subset
        empty_priv = runner.get_privileged_obs(env_indices=[])
        assert isinstance(empty_priv, np.ndarray)
        assert empty_priv.shape == (0, runner.privileged_obs_dim)
        assert empty_priv.dtype == np.float32

        # Arbitrary subset and ordering
        all_priv = runner.get_privileged_obs()
        sub_indices = [3, 0, 2]
        subset_priv = runner.get_privileged_obs(env_indices=sub_indices)
        assert subset_priv.shape == (3, runner.privileged_obs_dim)
        for out_idx, src_idx in enumerate(sub_indices):
            np.testing.assert_allclose(subset_priv[out_idx], all_priv[src_idx], atol=1e-5)
    finally:
        runner.close()

    # 2. Privileged obs on stairs terrain
    stairs_runner = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=2,
        decimation=10,
        stage=1,
        terrain_mode="stairs",
    )
    try:
        stairs_priv = stairs_runner.get_privileged_obs()
        assert stairs_priv.shape == (2, stairs_runner.privileged_obs_dim)
        # Verify terrain height at feet matches _get_terrain_height_at_xy bit-for-bit
        for env_i in range(2):
            d = stairs_runner.datas[env_i]
            expected_lh = stairs_runner._get_terrain_height_at_xy(
                float(d.xpos[stairs_runner.left_foot_bid][0]),
                float(d.xpos[stairs_runner.left_foot_bid][1]),
            )
            expected_rh = stairs_runner._get_terrain_height_at_xy(
                float(d.xpos[stairs_runner.right_foot_bid][0]),
                float(d.xpos[stairs_runner.right_foot_bid][1]),
            )
            np.testing.assert_allclose(stairs_priv[env_i, stairs_runner.obs_dim + 5], expected_lh, atol=1e-5)
            np.testing.assert_allclose(stairs_priv[env_i, stairs_runner.obs_dim + 6], expected_rh, atol=1e-5)
    finally:
        stairs_runner.close()

    # 3. evaluate_deterministic_rollout with reused eval_runner
    eval_r = ParallelSimRunner(
        model_path="duck_v2/cyberduck_v2.xml",
        num_envs=1,
        decimation=10,
        stage=0,
        terrain_mode="hurdles",
    )
    model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=64, critic_obs_dim=53)
    obs_rms = RunningMeanStd(shape=42, clip=10.0)
    try:
        det_dist_x, det_rew = evaluate_deterministic_rollout(
            model=model,
            obs_rms=obs_rms,
            model_path="duck_v2/cyberduck_v2.xml",
            stage=1,
            terrain_mode="hurdles",
            stand_still=True,
            max_steps=10,
            device="cpu",
            eval_runner=eval_r,
        )
        assert isinstance(det_dist_x, float) and math.isfinite(det_dist_x)
        assert isinstance(det_rew, float) and math.isfinite(det_rew)
        assert eval_r.curriculum_stage == 1
        assert eval_r.stand_still is True
        # Verify eval_r was NOT closed by evaluate_deterministic_rollout
        reset_obs = eval_r.reset()
        assert reset_obs is not None
        assert reset_obs.shape[0] == 1
    finally:
        eval_r.close()
