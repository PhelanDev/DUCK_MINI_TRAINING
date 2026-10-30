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

    # Load checkpoint and verify content
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt["obs_dim"] == 42
    assert ckpt["action_dim"] == 8
    assert ckpt["hidden_dim"] == 256

    # Test restoring model from checkpoint
    restored_model = ActorCritic(obs_dim=42, action_dim=8, hidden_dim=256)
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
    onnx_path = tmp_path / "exported_pipeline.onnx"
    weights_path = tmp_path / "esp32_weights.h"
    config_path = tmp_path / "esp32_config.h"

    # 1. Run train_ppo with --dry-run
    train_cmd = [
        sys.executable,
        str(train_script),
        "--dry-run",
        "--save-path",
        str(ckpt_path),
        "--num-envs",
        "4",
        "--rollout-steps",
        "4",
    ]
    res_train = subprocess.run(train_cmd, capture_output=True, text=True)
    assert res_train.returncode == 0, f"train_ppo failed:\n{res_train.stderr}"
    assert ckpt_path.exists()

    # 2. Run export_policy
    export_cmd = [
        sys.executable,
        str(export_script),
        "--checkpoint",
        str(ckpt_path),
        "--onnx-output",
        str(onnx_path),
        "--esp32-weights-output",
        str(weights_path),
        "--esp32-config-output",
        str(config_path),
    ]
    res_export = subprocess.run(export_cmd, capture_output=True, text=True)
    assert res_export.returncode == 0, f"export_policy failed:\n{res_export.stderr}"
    assert onnx_path.exists()
    assert weights_path.exists()
    assert config_path.exists()
