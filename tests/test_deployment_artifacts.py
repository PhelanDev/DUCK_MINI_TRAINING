"""Comprehensive Tests for Hardware Deployment Artifacts and Firmware.

Covers:
1. pipelines/rpi5_stack/deployment/infer_rpi5.py
2. pipelines/rpi5_stack/docs/rpi5_hardware_spec.md
3. pipelines/dual_esp32s3_stack/deployment/uart_packet_protocol.h
4. pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino
5. pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino
6. pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_rpi5_infer_script_import_and_mock():
    """Verify RPi5 inference runtime import, angle-to-PWM mapping, and mock mode."""
    from pipelines.rpi5_stack.deployment.infer_rpi5 import (
        RPi5InferenceEngine,
        angle_to_pwm,
        pack_rpi5_observation,
    )

    # 1. Test angle_to_pwm conversion
    # 0 rad (neutral) MUST map to 1500us across all joints, preventing knee jamming on [0.0, 1.48]
    assert angle_to_pwm(0.0, 0.0, 1.48) == 1500, "Knee 0.0 rad must map to 1500us neutral"
    assert angle_to_pwm(0.0, -0.52, 0.78) == 1500, "Head pitch 0.0 rad must map to 1500us neutral"
    assert angle_to_pwm(0.0, -0.87, 0.87) == 1500, "Hip pitch 0.0 rad must map to 1500us neutral"
    assert angle_to_pwm(0.0, -1.57, 1.57) == 1500, "Symmetric joint 0.0 rad must map to 1500us neutral"

    # Knee max extension (1.48 rad) maps to ~2442us
    knee_max_pwm = angle_to_pwm(1.48, 0.0, 1.48)
    assert 2440 <= knee_max_pwm <= 2445, f"Expected knee max ~2442us, got {knee_max_pwm}"

    min_pwm = angle_to_pwm(-np.pi / 2, -1.5708, 1.5708, min_pulse_us=500, max_pulse_us=2500)
    assert min_pwm == 500, f"Expected min PWM 500us, got {min_pwm}"

    max_pwm = angle_to_pwm(np.pi / 2, -1.5708, 1.5708, min_pulse_us=500, max_pulse_us=2500)
    assert max_pwm == 2500, f"Expected max PWM 2500us, got {max_pwm}"

    # Clamping test
    clamped_pwm = angle_to_pwm(5.0, -np.pi / 2, np.pi / 2, min_pulse_us=500, max_pulse_us=2500)
    assert clamped_pwm == 2500

    # 2. Test pack_rpi5_observation
    obs = pack_rpi5_observation(
        base_ang_vel=[0.0, 0.0, 0.0],
        proj_grav=[0.0, 0.0, -1.0],
        commands=[0.2, 0.0, 0.0],
        joint_pos=[0.0] * 8,
        joint_vel=[0.0] * 8,
        last_actions=[0.0] * 8,
        foot_contact=[1.0, 1.0, 1.0, 1.0],
        lidar_ranges=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    assert obs.shape == (42,), f"Expected shape (42,), got {obs.shape}"
    assert obs.dtype == np.float32

    # 3. Test RPi5InferenceEngine in mock mode
    engine = RPi5InferenceEngine(mock=True, freq=50.0)
    raw_obs = np.zeros(42, dtype=np.float32)
    targets, pwms = engine.step(raw_obs)

    assert targets.shape == (8,)
    assert pwms.shape == (8,)
    # Verify rate limiting between consecutive steps
    large_delta_obs = np.ones(42, dtype=np.float32)
    next_targets, _ = engine.step(large_delta_obs)
    max_step_delta = 6.0 * (1.0 / 50.0) + 1e-4  # max_vel * dt
    actual_max_delta = np.max(np.abs(next_targets - targets))
    assert actual_max_delta <= max_step_delta, (
        f"Rate limit exceeded: {actual_max_delta} > {max_step_delta}"
    )


def test_rpi5_infer_onnx_model_execution(tmp_path):
    """Test RPi5 inference runtime loading and evaluating a real ONNX model."""
    import onnx
    from onnx import helper, TensorProto
    from pipelines.rpi5_stack.deployment.infer_rpi5 import RPi5InferenceEngine

    # Build a tiny dummy ONNX model: output = MatMul(obs, W) where W is 42x8
    input_tensor = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 42])
    output_tensor = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 8])

    weights = np.zeros((42, 8), dtype=np.float32)
    weights[0, 0] = 0.5  # Simple coupling
    weight_init = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[42, 8],
        vals=weights.flatten().tolist(),
    )

    node = helper.make_node("MatMul", ["obs", "W"], ["actions"])
    graph = helper.make_graph([node], "DummyPolicy", [input_tensor], [output_tensor], [weight_init])
    opset = helper.make_opsetid("", 18)
    model = helper.make_model(graph, producer_name="duckV1_test", ir_version=10, opset_imports=[opset])

    model_file = tmp_path / "dummy_policy.onnx"
    onnx.save(model, str(model_file))

    # Initialize engine with the ONNX model
    engine = RPi5InferenceEngine(model_path=str(model_file), mock=True, freq=50.0)
    obs = np.zeros(42, dtype=np.float32)
    obs[0] = 1.0  # Should trigger action[0] = 0.5

    targets, pwms = engine.step(obs)
    assert targets.shape == (8,)
    assert pwms.shape == (8,)
    assert np.all(pwms >= 500) and np.all(pwms <= 2500)


def test_rpi5_infer_cli():
    """Verify CLI flags --mock, --mock-pca, --freq, --steps work properly."""
    script_path = REPO_ROOT / "pipelines" / "rpi5_stack" / "deployment" / "infer_rpi5.py"
    assert script_path.exists(), f"Missing script: {script_path}"

    for mock_flag in ["--mock", "--mock-pca"]:
        cmd = [
            sys.executable,
            str(script_path),
            mock_flag,
            "--freq",
            "50",
            "--steps",
            "5",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI with {mock_flag} failed with error: {res.stderr}\nOutput: {res.stdout}"
        assert "Mock PCA9685" in res.stdout or "mock" in res.stdout.lower()
        assert "Step 5" in res.stdout or "completed" in res.stdout.lower()


def test_rpi5_hardware_spec_doc():
    """Verify rpi5_hardware_spec.md exists and contains thorough technical documentation."""
    spec_path = REPO_ROOT / "pipelines" / "rpi5_stack" / "docs" / "rpi5_hardware_spec.md"
    assert spec_path.exists(), f"Missing spec: {spec_path}"

    content = spec_path.read_text(encoding="utf-8")
    assert len(content) > 500, "Documentation is too brief"

    # Key hardware specifications
    assert "5V/5A" in content or "5A" in content
    assert "PCA9685" in content
    assert "MG996R" in content
    assert "I2C" in content
    assert "I2S" in content or "INMP441" in content
    assert "LiDAR" in content or "ToF" in content
    assert "Camera" in content or "CSI" in content
    assert "2S" in content or "7.4V" in content


def test_uart_packet_protocol_cpp_compilation(tmp_path):
    """Test C/C++ header compilation and packet encode/decode/CRC validation."""
    header_path = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "uart_packet_protocol.h"
    )
    assert header_path.exists(), f"Missing protocol header: {header_path}"

    # Create a small C++ test runner to verify protocol encoding, decoding, and CRC
    cpp_source = f"""#include <iostream>
#include <cassert>
#include <cmath>
#include "{header_path.as_posix()}"

int main() {{
    // 1. Test CRC16 computation
    uint8_t sample_data[] = {{0xAA, 0x55, 0x01, 0x00, 0x04, 0xDE, 0xAD, 0xBE, 0xEF}};
    uint16_t crc = calc_crc16(sample_data, sizeof(sample_data));
    assert(crc != 0);

    // 2. Test VelocityCmdPacket encoding and decoding
    VelocityCmdPacket cmd_in;
    cmd_in.vx = 0.25f;
    cmd_in.vy = -0.10f;
    cmd_in.wz = 0.50f;

    uint8_t buffer[64];
    uint16_t encoded_len = encode_velocity_packet(cmd_in, 42, buffer, sizeof(buffer));
    assert(encoded_len > 0);
    assert(buffer[0] == 0xAA);
    assert(buffer[1] == 0x55);

    VelocityCmdPacket cmd_out;
    uint8_t seq_out = 0;
    bool ok = decode_velocity_packet(buffer, encoded_len, &cmd_out, &seq_out);
    assert(ok == true);
    assert(seq_out == 42);
    assert(std::fabs(cmd_out.vx - 0.25f) < 1e-4);
    assert(std::fabs(cmd_out.vy - (-0.10f)) < 1e-4);
    assert(std::fabs(cmd_out.wz - 0.50f) < 1e-4);

    // 3. Test LidarPerceptionPacket encoding and decoding
    LidarPerceptionPacket lidar_in;
    for (int i = 0; i < 5; ++i) {{
        lidar_in.ranges[i] = 0.5f + i * 0.2f;
    }}
    lidar_in.obstacle_detected = 1;

    encoded_len = encode_lidar_packet(lidar_in, 100, buffer, sizeof(buffer));
    assert(encoded_len > 0);

    LidarPerceptionPacket lidar_out;
    ok = decode_lidar_packet(buffer, encoded_len, &lidar_out, &seq_out);
    assert(ok == true);
    assert(seq_out == 100);
    assert(lidar_out.obstacle_detected == 1);
    for (int i = 0; i < 5; ++i) {{
        assert(std::fabs(lidar_out.ranges[i] - (0.5f + i * 0.2f)) < 1e-4);
    }}

    // 4. Test corrupted packet detection via CRC
    buffer[encoded_len - 1] ^= 0xFF; // Corrupt CRC
    ok = decode_lidar_packet(buffer, encoded_len, &lidar_out, &seq_out);
    assert(ok == false);

    // 5. Test buffer overflow protection in decode_packet
    buffer[encoded_len - 1] ^= 0xFF; // Restore valid CRC
    uint8_t p_type = 0;
    uint16_t out_len = 0;
    uint8_t tiny_dest[2];
    // Attempting to decode into a 2-byte destination when payload is larger (21 bytes)
    bool safe_decode = decode_packet(buffer, encoded_len, &p_type, &seq_out, tiny_dest, sizeof(tiny_dest), &out_len);
    assert(safe_decode == false);

    // 6. Test HeartbeatPacket encoding and C-pointer overload
    HeartbeatPacket hb_in;
    hb_in.uptime_ms = 12345;
    hb_in.status = 0;
    encoded_len = encode_heartbeat_packet(&hb_in, 7, buffer, sizeof(buffer));
    assert(encoded_len > 0);

    HeartbeatPacket hb_out;
    ok = decode_heartbeat_packet(buffer, encoded_len, &hb_out, &seq_out);
    assert(ok == true);
    assert(seq_out == 7);
    assert(hb_out.uptime_ms == 12345);
    assert(hb_out.status == 0);

    std::cout << "All protocol tests passed successfully." << std::endl;
    return 0;
}}
"""
    test_cpp_file = tmp_path / "test_protocol.cpp"
    test_cpp_file.write_text(cpp_source, encoding="utf-8")

    out_exe = tmp_path / "test_protocol.exe"
    compile_cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        str(test_cpp_file),
        "-o",
        str(out_exe),
    ]
    compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert compile_res.returncode == 0, (
        f"G++ compilation failed: {compile_res.stderr}\n{compile_res.stdout}"
    )

    run_res = subprocess.run([str(out_exe)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Protocol test execution failed: {run_res.stderr}"
    assert "All protocol tests passed successfully." in run_res.stdout


def test_mcu1_vision_audio_sketch():
    """Verify MCU 1 Arduino sketch contains necessary peripherals and high-speed UART."""
    sketch_path = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "mcu1_vision_audio_s3"
        / "mcu1_main.ino"
    )
    assert sketch_path.exists(), f"Missing sketch: {sketch_path}"

    content = sketch_path.read_text(encoding="utf-8")
    assert "uart_packet_protocol.h" in content
    assert "921600" in content, "MCU 1 UART must run at 921600 baud"
    assert "setup()" in content and "loop()" in content
    # Peripherals
    assert "camera" in content.lower() or "cam" in content.lower()
    assert "i2s" in content.lower() or "mic" in content.lower()
    assert "lidar" in content.lower() or "vl53" in content.lower() or "tof" in content.lower()


def test_mcu2_locomotion_sketch():
    """Verify MCU 2 Arduino sketch handles UART, MPU6050, 42-dim obs, and 50Hz servo loop."""
    sketch_path = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "mcu2_locomotion_s3"
        / "mcu2_main.ino"
    )
    assert sketch_path.exists(), f"Missing sketch: {sketch_path}"

    content = sketch_path.read_text(encoding="utf-8")
    assert "uart_packet_protocol.h" in content
    assert "921600" in content, "MCU 2 UART must run at 921600 baud"
    assert "setup()" in content and "loop()" in content
    # IMU, observation, control frequency
    assert "mpu" in content.lower() or "imu" in content.lower()
    assert "42" in content, "MCU 2 must assemble 42-dim observation"
    assert "50" in content or "20" in content, "MCU 2 must run 50Hz (20ms) loop"
    assert "mg996r" in content.lower() or "servo" in content.lower() or "pwm" in content.lower()


def test_dual_esp32_hardware_spec_doc():
    """Verify dual_esp32s3_hardware_spec.md exists and contains thorough technical documentation."""
    spec_path = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "docs"
        / "dual_esp32s3_hardware_spec.md"
    )
    assert spec_path.exists(), f"Missing spec: {spec_path}"

    content = spec_path.read_text(encoding="utf-8")
    assert len(content) > 500, "Documentation is too brief"

    # Key hardware specifications
    assert "ESP32-S3" in content
    assert "UART" in content and "921600" in content
    assert "MG996R" in content
    assert "INMP441" in content or "I2S" in content
    assert "MPU6050" in content or "IMU" in content
    assert "LiDAR" in content or "ToF" in content
    assert "BMS" in content or "2S" in content or "7.4V" in content
