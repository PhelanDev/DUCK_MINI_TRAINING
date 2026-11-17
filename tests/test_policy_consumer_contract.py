"""Behavioral coverage for checkpoint consumer validation and generated inference."""

import subprocess

import numpy as np
import pytest
import torch

from scripts.train_ppo import ActorCritic, RunningMeanStd


@pytest.mark.parametrize("moving", [False, True])
def test_audit_schema_roundtrip_and_rpi_clock_parity(tmp_path, moving):
    from training.policy_contract import observation_schema
    from scripts.export_policy import export_onnx
    from pipelines.rpi_infer_core import BaseRPiInferenceEngine, pack_rpi_observation
    from training.run_parallel_sim import ParallelSimRunner
    model, _ = make_policy()
    model.observation_schema = observation_schema(42, True, 0.006)
    output = export_onnx(model, tmp_path / "clock.onnx")
    engine = BaseRPiInferenceEngine(str(output), mock=True, freq=1 / 0.006)
    runner = ParallelSimRunner(model_path="pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml",
        num_envs=1, decimation=3, terrain_mode="flat")
    try:
        runner.reset()
        runner.commands[:] = [0.2 if moving else 0, 0, 0]
        for _ in range(4):
            expected = runner.get_observations()[0]
            sensor = runner.datas[0].sensordata
            w, x, y, z = sensor[6:10]
            gravity = [2 * (w*y-x*z), -2 * (w*x+y*z), 2 * (x*x+y*y)-1]
            contact = [float(sensor[10] > 0.1), float(sensor[11] > 0.1), 0, 0]
            actual = pack_rpi_observation(sensor[3:6], gravity, runner.commands[0],
                sensor[12:20], sensor[20:28], engine.last_raw_actions, contact,
                runner._get_lidar_ranges(0), phase_clock=engine.phase_clock())
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            import torch
            with torch.no_grad():
                action = model.forward_actor(torch.from_numpy(actual).unsqueeze(0)).numpy()[0]
            np.testing.assert_allclose(engine.predict_action(actual), action, atol=2e-6)
            engine.step(actual)
            runner.step(engine.last_raw_actions.reshape(1, 8))
        engine.reset_policy_state()
        runner.reset()
        np.testing.assert_allclose(engine.phase_clock(), runner.get_observations()[0, 35:37], atol=1e-7)
        np.testing.assert_array_equal(engine.last_raw_actions, 0)
    finally:
        runner.close()


def test_audit_rpi_legacy_requires_explicit_schema(tmp_path):
    from scripts.export_policy import export_onnx
    from pipelines.rpi_infer_core import BaseRPiInferenceEngine
    from training.policy_contract import observation_schema
    model, _ = make_policy()
    output = export_onnx(model, tmp_path / "legacy.onnx")
    with pytest.raises(ValueError, match="schema|legacy"):
        BaseRPiInferenceEngine(str(output), mock=True)
    engine = BaseRPiInferenceEngine(str(output), mock=True,
        legacy_observation_schema=observation_schema(42, False, 0.02))
    assert engine.phase_clock() is None


@pytest.mark.parametrize("change", [{"period": 0}, {"control_dt": float("nan")},
    {"embed_phase_clock": "true"}, {"reset": "stop_when_still"}, {"version": 2}, {"obs_dim": 37}])
def test_audit_malformed_schema_rejected_before_export(tmp_path, change):
    from training.policy_contract import observation_schema
    from scripts.export_policy import export_onnx
    model, _ = make_policy()
    model.observation_schema = dict(observation_schema(42, True, 0.02), **change)
    with pytest.raises(ValueError, match="schema"):
        export_onnx(model, tmp_path / "invalid.onnx")
    assert not (tmp_path / "invalid.onnx").exists()


def test_audit_cli_export_rejects_cross_target_identity(tmp_path):
    from scripts.export_policy import prepare_target_exports
    model, _ = make_policy()
    path = tmp_path / "wrong-target.pt"
    torch.save(dict(model_state_dict=model.state_dict(), model_name="microduck_rpi5_8dof.xml"), path)
    with pytest.raises(ValueError, match="identity"):
        prepare_target_exports({"pizero2w": path})


def test_audit_both_pi_wrappers_run_metadata_policy(tmp_path):
    from scripts.export_policy import export_onnx
    from training.policy_contract import observation_schema
    from pipelines.rpi5_stack.deployment.infer_rpi5 import RPi5InferenceEngine
    from pipelines.pizero2w_stack.deployment.infer_pizero2w import PiZero2WInferenceEngine
    model, _ = make_policy()
    model.observation_schema = observation_schema(42, True, 0.02)
    output = export_onnx(model, tmp_path / "policy.onnx")
    for wrapper in (RPi5InferenceEngine, PiZero2WInferenceEngine):
        engine = wrapper(str(output), mock=True)
        engine.run_loop(steps=2)
        assert engine.episode_steps == 2


def make_policy():
    model = ActorCritic(hidden_dim=16, squash_actions=True)
    rms = RunningMeanStd(42, clip=0.25)
    rms.mean[:] = 0.1
    rms.var[:] = 1.0
    return model, rms


def run_generated_policy(tmp_path, weights, config, observation):
    source = tmp_path / "consumer.cpp"
    source.write_text(
        '#include <cstdio>\n#include "' + config.name + '"\n#include "' + weights.name + '"\n'
        'int main(){ int8_t obs[42]={' + ','.join(map(str, observation)) + '};'
        'int8_t action[8]; evaluate_policy_int8(obs,action);'
        'for(int i=0;i<8;i++) printf("%d ", (int)action[i]);}\n',
        encoding="utf-8",
    )
    executable = tmp_path / "consumer.exe"
    subprocess.run(["g++", "-std=c++17", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    return np.fromstring(result.stdout, sep=" ")


def test_generated_policy_preserves_clipped_rms_and_actual_hidden_width(tmp_path):
    from scripts.export_policy import export_esp32_headers
    from training.env_config import DEFAULT_OBS_BOUNDS

    model, rms = make_policy()
    with torch.no_grad():
        for parameter in model.actor.parameters():
            parameter.zero_()
        model.actor[0].weight[0, 0] = 0.5
        model.actor[2].weight[0, 0] = 1.0
        model.actor[4].weight[:, 0] = 2.0
    checkpoint = dict(model_state_dict=model.state_dict(), squash_actions=True, obs_rms=rms.state_dict())
    weights, config = export_esp32_headers(checkpoint, tmp_path / "weights.h", tmp_path / "config.h")
    observation = np.zeros(42, dtype=np.int8)
    observation[0] = 120
    actual = run_generated_policy(tmp_path, weights, config, observation)
    # Compare at the dequantized input to isolate normalization/output transforms
    # from the deliberately unchanged general INT8 input/weight quantization error.
    raw = observation.astype(np.float32) * DEFAULT_OBS_BOUNDS / 127.0
    with torch.no_grad():
        expected = model.forward_actor(torch.from_numpy(rms.normalize(raw))).numpy()
    np.testing.assert_allclose(actual, np.round(expected * 127), atol=1)


def test_direct_module_export_carries_advertised_rms_and_model_metadata(tmp_path):
    import onnxruntime as ort
    from scripts.export_policy import export_onnx
    from scripts.interactive_viewer import resolve_viewer_model_path

    model, rms = make_policy()
    model.obs_rms = rms
    model.model_path = "duck_v2/cyberduck_v2.xml"
    model.model_name = "cyberduck_v2.xml"
    output = export_onnx(model, tmp_path / "unrelated.onnx")
    raw = np.full((1, 42), 20.0, dtype=np.float32)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model.forward_actor(torch.from_numpy(rms.normalize(raw))).numpy()
    np.testing.assert_allclose(session.run(None, {"obs": raw})[0], expected, atol=2e-6)
    assert resolve_viewer_model_path(None, output).endswith("cyberduck_v2.xml")


@pytest.mark.parametrize("change", [
    {"mean": ["0"] * 42}, {"var": [True] * 42}, {"epsilon": [0.001]},
    {"min_std": "0.1"}, {"clip": True}, {"count": "10"},
])
def test_rms_rejects_non_numeric_or_non_scalar_advertised_fields(change):
    from scripts.export_policy import apply_obs_norm_to_actor

    model, rms = make_policy()
    values = dict(rms.state_dict(), **change)
    before = {key: value.clone() for key, value in model.actor.state_dict().items()}
    with pytest.raises(ValueError, match="obs_rms"):
        apply_obs_norm_to_actor(model.actor, {"obs_rms": values})
    for key, value in model.actor.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_viewer_rejects_nonfinite_checkpoint_weights(tmp_path):
    from scripts.interactive_viewer import load_viewer_policy

    model, _ = make_policy()
    with torch.no_grad():
        model.actor[0].weight[0, 0] = float("nan")
    checkpoint = tmp_path / "corrupt.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="finite"):
        load_viewer_policy(checkpoint)


def test_viewer_rejects_conflicting_checkpoint_model_identity(tmp_path):
    from scripts.interactive_viewer import resolve_viewer_model_path

    checkpoint = tmp_path / "policy.pt"
    torch.save(dict(model_path="duck_v2/cyberduck_v2.xml", model_name="microduck_8dof.xml"), checkpoint)
    with pytest.raises(ValueError, match="model"):
        resolve_viewer_model_path(None, checkpoint)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0, -1])
def test_esp32_rejects_invalid_quantization_before_writing(tmp_path, scale):
    from scripts.export_policy import export_esp32_headers

    model, _ = make_policy()
    with pytest.raises(ValueError, match="scale"):
        export_esp32_headers(model, tmp_path / "weights.h", tmp_path / "config.h", scale_obs=scale)
    assert not (tmp_path / "weights.h").exists()
    assert not (tmp_path / "config.h").exists()
