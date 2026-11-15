"""Policy Export Tool for MicroDuck 8-DOF Bipedal Robot.

Exports trained Actor policy checkpoints to deployment targets:
1. Float32 ONNX model for Raspberry Pi 5 (Cortex-A76 ONNX Runtime)
2. INT8 quantized weights & INT32 biases C/C++ header for ESP32-S3 (Xtensa LX7 PIE vector accelerator)
3. Quantization configuration header (esp32_quant_config.h)
"""

from pathlib import Path
import argparse
import sys
from typing import Any, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.dual_esp32s3_stack.training.env_config import (
    DEFAULT_OBS_BOUNDS,
    generate_esp32_quant_header,
)
from scripts.train_ppo import ActorCritic

DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "microduck_ppo.pt"
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "microduck_rpi5_policy.onnx"
DEFAULT_PIZERO2W_ONNX_PATH = REPO_ROOT / "models" / "microduck_pizero2w_policy.onnx"
DEFAULT_ESP32_WEIGHTS_PATH = (
    REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "deployment" / "esp32_policy_weights.h"
)
DEFAULT_ESP32_CONFIG_PATH = (
    REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "deployment" / "esp32_quant_config.h"
)


class PolicyOnnxWrapper(nn.Module):
    """Wrapper exposing only deterministic actor output for ONNX runtime."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass taking observation [batch, 42] -> action [batch, 8]."""
        return self.actor(obs)


def apply_obs_norm_to_actor(actor: nn.Module, ckpt: Any) -> nn.Module:
    """Fold observation normalization (mean, var) from ckpt['obs_rms'] into actor's first linear layer.

    Folds:
        inv_std = 1.0 / sqrt(var + eps)
        w_fused = w * inv_std
        b_fused = b - w_fused @ mean
    Guarantees idempotency (does not double-fuse) and finite checks.
    """
    if getattr(actor, "_obs_norm_applied", False):
        return actor

    if not isinstance(ckpt, dict) or "obs_rms" not in ckpt or ckpt["obs_rms"] is None:
        return actor

    obs_rms = ckpt["obs_rms"]
    if isinstance(obs_rms, dict):
        mean = obs_rms.get("mean")
        var = obs_rms.get("var")
        eps = float(obs_rms.get("epsilon", 1e-4))
    else:
        mean = getattr(obs_rms, "mean", None)
        var = getattr(obs_rms, "var", None)
        eps = float(getattr(obs_rms, "epsilon", 1e-4))

    if mean is None or var is None:
        return actor

    mean = np.asarray(mean, dtype=np.float32).ravel()
    var = np.asarray(var, dtype=np.float32).ravel()

    # Find the first nn.Linear layer in actor
    linear_layer = None
    if isinstance(actor, (nn.Sequential, nn.ModuleList)) and len(actor) > 0 and isinstance(actor[0], nn.Linear):
        linear_layer = actor[0]
    else:
        for m in actor.modules():
            if isinstance(m, nn.Linear):
                linear_layer = m
                break

    if linear_layer is None:
        return actor

    in_features = linear_layer.in_features
    if mean.shape[0] < in_features or var.shape[0] < in_features:
        return actor
    mean = mean[:in_features]
    var = var[:in_features]

    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var)) and np.all(var >= 0)):
        return actor

    inv_std = 1.0 / np.sqrt(var + eps)
    w = linear_layer.weight.detach().cpu().numpy().astype(np.float32)
    b = (
        linear_layer.bias.detach().cpu().numpy().astype(np.float32)
        if linear_layer.bias is not None
        else np.zeros(linear_layer.out_features, dtype=np.float32)
    )

    w_fused = w * inv_std  # shape (out_features, in_features)
    b_fused = b - (w_fused @ mean)  # shape (out_features,)

    linear_layer.weight.data = torch.from_numpy(w_fused).to(
        device=linear_layer.weight.device, dtype=linear_layer.weight.dtype
    )
    if linear_layer.bias is not None:
        linear_layer.bias.data = torch.from_numpy(b_fused).to(
            device=linear_layer.bias.device, dtype=linear_layer.bias.dtype
        )
    else:
        linear_layer.bias = nn.Parameter(
            torch.from_numpy(b_fused).to(
                device=linear_layer.weight.device, dtype=linear_layer.weight.dtype
            )
        )

    actor._obs_norm_applied = True
    return actor


def load_actor_from_checkpoint(
    checkpoint_or_model: Union[str, Path, nn.Module, dict],
    obs_dim: int = 42,
    action_dim: int = 8,
    hidden_dim: int = 256,
) -> nn.Module:
    """Load Actor network from checkpoint file or dict, folding obs_rms if present."""
    if isinstance(checkpoint_or_model, nn.Module):
        if hasattr(checkpoint_or_model, "actor"):
            return checkpoint_or_model.actor
        return checkpoint_or_model

    if isinstance(checkpoint_or_model, dict):
        ckpt = checkpoint_or_model
    else:
        ckpt_path = Path(checkpoint_or_model).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    obs_d = ckpt.get("obs_dim", obs_dim) if isinstance(ckpt, dict) else obs_dim
    # Infer observation dimension directly from state_dict weights to be fully resilient
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and "actor.0.weight" in ckpt["model_state_dict"]:
            obs_d = int(ckpt["model_state_dict"]["actor.0.weight"].shape[1])
        elif "actor_state_dict" in ckpt and "0.weight" in ckpt["actor_state_dict"]:
            obs_d = int(ckpt["actor_state_dict"]["0.weight"].shape[1])
        elif "0.weight" in ckpt:
            obs_d = int(ckpt["0.weight"].shape[1])

    critic_obs_d = None
    if isinstance(ckpt, dict):
        if "critic_obs_dim" in ckpt:
            critic_obs_d = int(ckpt["critic_obs_dim"])
        elif "model_state_dict" in ckpt and "critic.0.weight" in ckpt["model_state_dict"]:
            critic_obs_d = int(ckpt["model_state_dict"]["critic.0.weight"].shape[1])
        elif "critic.0.weight" in ckpt:
            critic_obs_d = int(ckpt["critic.0.weight"].shape[1])

    act_d = ckpt.get("action_dim", action_dim) if isinstance(ckpt, dict) else action_dim
    hid_d = ckpt.get("hidden_dim", hidden_dim) if isinstance(ckpt, dict) else hidden_dim

    model = ActorCritic(obs_dim=obs_d, action_dim=act_d, hidden_dim=hid_d, critic_obs_dim=critic_obs_d)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "actor_state_dict" in ckpt:
            model.actor.load_state_dict(ckpt["actor_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    actor = model.actor
    apply_obs_norm_to_actor(actor, ckpt)
    return actor


def export_onnx(
    model: Union[nn.Module, str, Path],
    output_path: Union[str, Path] = DEFAULT_ONNX_PATH,
    opset: int = 13,
) -> Path:
    """Export Actor policy to Float32 ONNX format for Raspberry Pi 5.

    Input tensor name: 'obs' of shape [1, 42] (dynamic batch supported)
    Output tensor name: 'actions' of shape [1, 8]
    """
    actor = load_actor_from_checkpoint(model)
    actor.eval()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = PolicyOnnxWrapper(actor)
    wrapper.eval()

    # Determine input observation dimension dynamically from actor's first linear layer
    obs_dim = 42
    for m in actor.modules():
        if hasattr(m, "in_features"):
            obs_dim = int(m.in_features)
            break

    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)

    export_kwargs = {
        "export_params": True,
        "opset_version": int(opset),
        "do_constant_folding": True,
        "input_names": ["obs"],
        "output_names": ["actions"],
        "dynamic_axes": {
            "obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    }

    try:
        # PyTorch 2.9+ requires dynamo=False if onnxscript is not installed
        torch.onnx.export(wrapper, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older PyTorch versions don't take the dynamo argument
        torch.onnx.export(wrapper, dummy_input, str(output_path), **export_kwargs)

    # Validate exported ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    return output_path


def _format_array_c(arr: np.ndarray, per_line: int = 12) -> str:
    """Format 1D numpy array into a formatted C string."""
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i : i + per_line]
        lines.append("    " + ", ".join(str(int(x)) for x in chunk))
    return ",\n".join(lines)


def export_esp32_headers(
    model: Union[nn.Module, str, Path],
    weights_output_path: Union[str, Path] = DEFAULT_ESP32_WEIGHTS_PATH,
    config_output_path: Union[str, Path] = DEFAULT_ESP32_CONFIG_PATH,
    scale_obs: float = 127.0,
    scale_act: float = 127.0,
) -> Tuple[Path, Path]:
    """Export INT8 quantized weights & INT32 biases header and config header for ESP32-S3."""
    actor = load_actor_from_checkpoint(model)
    actor.eval()

    weights_output_path = Path(weights_output_path).resolve()
    weights_output_path.parent.mkdir(parents=True, exist_ok=True)

    config_output_path = Path(config_output_path).resolve()
    config_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract weights and biases
    # actor[0]: Linear(obs_dim, 256)
    # actor[2]: Linear(256, 256)
    # actor[4]: Linear(256, 8)
    w1 = actor[0].weight.detach().cpu().numpy().astype(np.float32)  # [256, obs_dim]
    obs_dim = int(w1.shape[1])
    b1 = actor[0].bias.detach().cpu().numpy().astype(np.float32)    # [256]

    w2 = actor[2].weight.detach().cpu().numpy().astype(np.float32)  # [256, 256]
    b2 = actor[2].bias.detach().cpu().numpy().astype(np.float32)    # [256]

    w3 = actor[4].weight.detach().cpu().numpy().astype(np.float32)  # [8, 256]
    b3 = actor[4].bias.detach().cpu().numpy().astype(np.float32)    # [8]

    # 2. Generate esp32_quant_config.h with dynamically adapted obs_dim
    generate_esp32_quant_header(
        scale_obs=scale_obs,
        scale_act=scale_act,
        output_path=config_output_path,
        obs_dim=obs_dim,
    )

    # Quantize Layer 1 with per-channel observation scales matching mcu2_main.ino OBS_CHANNEL_SCALES
    obs_channel_scales = (float(scale_obs) / DEFAULT_OBS_BOUNDS[:obs_dim]).astype(np.float32)
    w1_effective = w1 / obs_channel_scales  # broadcast shape (256, obs_dim) / (obs_dim,)
    max_w1 = float(np.max(np.abs(w1_effective)))
    scale_w1 = 127.0 / max(max_w1, 1e-7)
    w1_int8 = np.clip(np.round(w1_effective * scale_w1), -128, 127).astype(np.int8)
    scale_b1 = scale_w1
    b1_int32 = np.round(b1 * scale_b1).astype(np.int32)

    # Quantize Layer 2
    max_w2 = float(np.max(np.abs(w2)))
    scale_w2 = 127.0 / max(max_w2, 1e-7)
    w2_int8 = np.clip(np.round(w2 * scale_w2), -128, 127).astype(np.int8)
    scale_b2 = scale_w2 * 127.0
    b2_int32 = np.round(b2 * scale_b2).astype(np.int32)

    # Quantize Layer 3
    max_w3 = float(np.max(np.abs(w3)))
    scale_w3 = 127.0 / max(max_w3, 1e-7)
    w3_int8 = np.clip(np.round(w3 * scale_w3), -128, 127).astype(np.int8)
    scale_b3 = scale_w3 * 127.0
    b3_int32 = np.round(b3 * scale_b3).astype(np.int32)

    weights_header_code = f"""#ifndef ESP32_POLICY_WEIGHTS_H
#define ESP32_POLICY_WEIGHTS_H

/**
 * MicroDuck 8-DOF (Dual ESP32-S3 Stack) - Quantized Policy Weights & Biases
 * Model: Actor MLP [{obs_dim} -> 256 (ELU) -> 256 (ELU) -> 8]
 * Quantization: Symmetric INT8 weights, INT32 biases
 * Platform: Xtensa LX7 Dual-Core with PIE Vector Accelerator
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define POLICY_LAYER1_IN  {obs_dim}
#define POLICY_LAYER1_OUT 256
#define POLICY_LAYER2_IN  256
#define POLICY_LAYER2_OUT 256
#define POLICY_LAYER3_IN  256
#define POLICY_LAYER3_OUT 8

// Quantization Scales
static const float POLICY_W1_SCALE = {scale_w1:.6f}f;
static const float POLICY_W1_INV_SCALE = (1.0f / {scale_w1:.6f}f);
static const float POLICY_B1_SCALE = {scale_b1:.6f}f;
static const float POLICY_B1_INV_SCALE = (1.0f / {scale_b1:.6f}f);

static const float POLICY_W2_SCALE = {scale_w2:.6f}f;
static const float POLICY_W2_INV_SCALE = (1.0f / {scale_w2:.6f}f);
static const float POLICY_B2_SCALE = {scale_b2:.6f}f;
static const float POLICY_B2_INV_SCALE = (1.0f / {scale_b2:.6f}f);

static const float POLICY_W3_SCALE = {scale_w3:.6f}f;
static const float POLICY_W3_INV_SCALE = (1.0f / {scale_w3:.6f}f);
static const float POLICY_B3_SCALE = {scale_b3:.6f}f;
static const float POLICY_B3_INV_SCALE = (1.0f / {scale_b3:.6f}f);

// Layer 1 Weights [256 * {obs_dim}]
static const int8_t POLICY_W1[POLICY_LAYER1_OUT * POLICY_LAYER1_IN] = {{
{_format_array_c(w1_int8)}
}};

// Layer 1 Biases [256]
static const int32_t POLICY_B1[POLICY_LAYER1_OUT] = {{
{_format_array_c(b1_int32)}
}};

// Layer 2 Weights [256 * 256]
static const int8_t POLICY_W2[POLICY_LAYER2_OUT * POLICY_LAYER2_IN] = {{
{_format_array_c(w2_int8)}
}};

// Layer 2 Biases [256]
static const int32_t POLICY_B2[POLICY_LAYER2_OUT] = {{
{_format_array_c(b2_int32)}
}};

// Layer 3 Weights [8 * 256]
static const int8_t POLICY_W3[POLICY_LAYER3_OUT * POLICY_LAYER3_IN] = {{
{_format_array_c(w3_int8)}
}};

// Layer 3 Biases [8]
static const int32_t POLICY_B3[POLICY_LAYER3_OUT] = {{
{_format_array_c(b3_int32)}
}};

#ifdef __cplusplus
extern "C" {{
#endif

/**
 * Fast forward inference evaluating the 8-DOF policy directly in INT8 arithmetic.
 *
 * @param obs_int8    Input observation vector [{obs_dim}], quantized with OBS_CHANNEL_SCALES.
 * @param actions_out Output action vector [8], quantized with scale ACT_SCALE (127.0f).
 */
static inline void evaluate_policy_int8(const int8_t* obs_int8, int8_t* actions_out) {{
    // Layer 1: [{obs_dim}] -> [256] ELU
    float l1_out[POLICY_LAYER1_OUT];
    for (int i = 0; i < POLICY_LAYER1_OUT; ++i) {{
        int32_t acc = POLICY_B1[i];
        const int8_t* w_row = &POLICY_W1[i * POLICY_LAYER1_IN];
        for (int j = 0; j < POLICY_LAYER1_IN; ++j) {{
            acc += ((int32_t)w_row[j]) * ((int32_t)obs_int8[j]);
        }}
        float val = (float)acc * POLICY_B1_INV_SCALE;
        l1_out[i] = (val >= 0.0f) ? val : (expf(val) - 1.0f);
    }}

    // Layer 2: [256] -> [256] ELU
    float l2_out[POLICY_LAYER2_OUT];
    for (int i = 0; i < POLICY_LAYER2_OUT; ++i) {{
        float acc = (float)POLICY_B2[i] * POLICY_B2_INV_SCALE;
        const int8_t* w_row = &POLICY_W2[i * POLICY_LAYER2_IN];
        for (int j = 0; j < POLICY_LAYER2_IN; ++j) {{
            acc += ((float)w_row[j] * POLICY_W2_INV_SCALE) * l1_out[j];
        }}
        l2_out[i] = (acc >= 0.0f) ? acc : (expf(acc) - 1.0f);
    }}

    // Layer 3: [256] -> [8] Linear Output
    for (int i = 0; i < POLICY_LAYER3_OUT; ++i) {{
        float acc = (float)POLICY_B3[i] * POLICY_B3_INV_SCALE;
        const int8_t* w_row = &POLICY_W3[i * POLICY_LAYER3_IN];
        for (int j = 0; j < POLICY_LAYER3_IN; ++j) {{
            acc += ((float)w_row[j] * POLICY_W3_INV_SCALE) * l2_out[j];
        }}
        float scaled = roundf(acc * 127.0f);
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;
        actions_out[i] = (int8_t)scaled;
    }}
}}

#ifdef __cplusplus
}}
#endif

#endif // ESP32_POLICY_WEIGHTS_H
"""
    weights_output_path.write_text(weights_header_code, encoding="utf-8")
    return weights_output_path, config_output_path


def main():
    parser = argparse.ArgumentParser(description="Export MicroDuck Policy Checkpoint to ONNX and ESP32 Headers")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to .pt checkpoint")
    parser.add_argument("--onnx-output", type=str, default=str(DEFAULT_ONNX_PATH), help="Path for ONNX export (RPi 5)")
    parser.add_argument("--pizero-onnx-output", type=str, default=str(DEFAULT_PIZERO2W_ONNX_PATH), help="Path for ONNX export (Pi Zero 2 W)")
    parser.add_argument("--esp32-weights-output", type=str, default=str(DEFAULT_ESP32_WEIGHTS_PATH), help="Path for esp32_policy_weights.h")
    parser.add_argument("--esp32-config-output", type=str, default=str(DEFAULT_ESP32_CONFIG_PATH), help="Path for esp32_quant_config.h")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    parser.add_argument("--scale-obs", type=float, default=127.0, help="Observation quantization scale (default: 127.0)")
    parser.add_argument("--scale-act", type=float, default=127.0, help="Action quantization scale (default: 127.0)")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    # If using default checkpoint, automatically prefer microduck_best.pt if it exists and has better iterations
    if ckpt_path == DEFAULT_CHECKPOINT_PATH.resolve():
        best_ckpt = REPO_ROOT / "models" / "checkpoints" / "microduck_best.pt"
        if best_ckpt.exists():
            try:
                d_best = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
                d_def = torch.load(str(ckpt_path), map_location="cpu", weights_only=False) if ckpt_path.exists() else {}
                best_iter = d_best.get("iterations", 0) if isinstance(d_best, dict) else 0
                def_iter = d_def.get("iterations", 0) if isinstance(d_def, dict) else 0
                if best_iter >= def_iter:
                    ckpt_path = best_ckpt
            except Exception:
                if not ckpt_path.exists():
                    ckpt_path = best_ckpt

    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}. Train a model first via scripts/train_ppo.py.")
        sys.exit(1)

    print(f"[Export] Loading checkpoint from {ckpt_path}...")
    onnx_file = export_onnx(ckpt_path, output_path=args.onnx_output, opset=args.opset)
    print(f"[Export] Exported Float32 ONNX model (RPi 5) (opset {args.opset}): {onnx_file}")

    pizero_onnx = export_onnx(ckpt_path, output_path=args.pizero_onnx_output, opset=args.opset)
    print(f"[Export] Exported Float32 ONNX model (Pi Zero 2 W) (opset {args.opset}): {pizero_onnx}")

    weights_file, config_file = export_esp32_headers(
        ckpt_path,
        weights_output_path=args.esp32_weights_output,
        config_output_path=args.esp32_config_output,
        scale_obs=args.scale_obs,
        scale_act=args.scale_act,
    )
    print(f"[Export] Exported ESP32-S3 INT8 weights header: {weights_file}")
    print(f"[Export] Exported ESP32-S3 quant config header: {config_file}")
    print("[Export] All deployment artifacts successfully exported.")


if __name__ == "__main__":
    main()
