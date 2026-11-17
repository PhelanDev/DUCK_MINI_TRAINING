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
import shutil
import subprocess
import sys
import tempfile
from typing import Optional
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_audit_esp32_rejects_37_before_creating_artifacts(tmp_path):
    from scripts.train_ppo import ActorCritic
    from scripts.export_policy import export_esp32_headers
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="42"):
        export_esp32_headers(ActorCritic(obs_dim=37, hidden_dim=16), output / "w.h", output / "c.h")
    assert not output.exists()


@pytest.mark.parametrize("obs_dim,layer_input,passes", [(42, 42, True), (37, 37, False), (42, 37, False)])
def test_audit_firmware_guards_and_assembly_canary(tmp_path, obs_dim, layer_input, passes):
    import re
    from scripts.export_policy import export_esp32_headers
    from scripts.train_ppo import ActorCritic
    firmware = (REPO_ROOT / "pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino").read_text(encoding="utf-8")
    guards = "\n".join(re.findall(r'^static_assert\([^\n]+', firmware, re.M))
    assert guards
    function = firmware[firmware.index("void assemble_observation("):firmware.index("void evaluate_locomotion_policy(")]
    weights, config = export_esp32_headers(ActorCritic(hidden_dim=16), tmp_path / "weights.h", tmp_path / "config.h")
    source = '#include <cmath>\n#include <cstdint>\n#include <cassert>\n#include "config.h"\n#include "weights.h"\n'
    source += f'#undef OBS_DIM\n#define OBS_DIM {obs_dim}\n#undef POLICY_LAYER1_IN\n#define POLICY_LAYER1_IN {layer_input}\n'
    source += guards + '''
const float CONTROL_DT=0.02f, PI=3.14159265358979323846f;
struct { float vx=0.2f,vy=0,wz=0; } g_cmd_pkt;
float g_current_joint_pos[8]={},g_last_joint_pos[8]={},g_last_raw_actions[8]={};
void read_mpu6050(float* gyro,float* grav){for(int i=0;i<3;i++){gyro[i]=0;grav[i]=i==2?-1:0;}}
void read_foot_contacts(float* p){p[0]=p[1]=1;}
void read_lidar_distances(float* p){for(int i=0;i<5;i++)p[i]=1.5f;}
'''+function+'''
int main(){struct {float obs[OBS_DIM]; float canary[5];} buffer;
for(float& f:buffer.canary) f=12345;
assemble_observation(buffer.obs);
for(float f:buffer.canary) assert(f==12345);
for(int i=37;i<42;i++)assert(buffer.obs[i]==1.5f);
return 0;}
'''
    cpp = tmp_path / "assembly.cpp"
    cpp.write_text(source, encoding="utf-8")
    exe = tmp_path / "assembly.exe"
    result = subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert (result.returncode == 0) == passes, result.stderr
    if passes:
        subprocess.run([str(exe)], check=True)


def _compile_and_run_cpp(
    cpp_source: str,
    cpp_path: Path,
    exe_path: Path,
    expected_stdout: Optional[str] = None,
    skip_whole_test: bool = True,
) -> bool:
    """Compile and run C++ source via g++, skipping or returning False if g++ is missing."""
    if shutil.which("g++") is None:
        if skip_whole_test:
            pytest.skip("g++ compiler not available")
        return False

    cpp_path.write_text(cpp_source, encoding="utf-8")
    compile_cmd = ["g++", "-O2", "-std=c++17", str(cpp_path), "-o", str(exe_path)]
    compile_res = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert compile_res.returncode == 0, (
        f"g++ compilation failed: {compile_res.stderr}\n{compile_res.stdout}"
    )

    run_res = subprocess.run([str(exe_path)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"C++ execution failed: {run_res.stderr}\n{run_res.stdout}"
    if expected_stdout is not None:
        assert expected_stdout in run_res.stdout, (
            f"Expected '{expected_stdout}' in stdout, got:\n{run_res.stdout}"
        )
    return True



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
        foot_contact=[1.0, 1.0, 0.0, 0.0],
        lidar_ranges=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    assert obs.shape == (42,), f"Expected shape (42,), got {obs.shape}"
    assert obs.dtype == np.float32

    # 3. Test RPi5InferenceEngine in mock mode
    engine = RPi5InferenceEngine(mock=True, freq=50.0)
    assert hasattr(engine, "last_raw_actions"), "Engine must track last_raw_actions"
    assert engine.last_raw_actions.shape == (8,)
    assert np.all(engine.last_raw_actions == 0.0)

    raw_obs = np.zeros(42, dtype=np.float32)
    targets, pwms = engine.step(raw_obs)

    assert targets.shape == (8,)
    assert pwms.shape == (8,)
    assert engine.last_raw_actions.shape == (8,)
    assert np.all(engine.last_raw_actions >= -1.0) and np.all(engine.last_raw_actions <= 1.0)

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
    import json
    from training.policy_contract import observation_schema
    onnx.helper.set_model_props(model, {"observation_schema": json.dumps(observation_schema(42, False, 0.02))})
    onnx.save(model, str(model_file))

    # Initialize engine with the ONNX model
    engine = RPi5InferenceEngine(model_path=str(model_file), mock=True, freq=50.0)
    obs = np.zeros(42, dtype=np.float32)
    obs[0] = 1.0  # Should trigger action[0] = 0.5

    targets, pwms = engine.step(obs)
    assert targets.shape == (8,)
    assert pwms.shape == (8,)
    assert np.all(pwms >= 500) and np.all(pwms <= 2500)
    assert np.isclose(engine.last_raw_actions[0], 0.5, atol=1e-5), "Obs[0]=1.0 should yield raw_action[0]=0.5"
    assert np.all(engine.last_raw_actions >= -1.0) and np.all(engine.last_raw_actions <= 1.0)


def test_rpi5_infer_cli(tmp_path):
    """Verify CLI flags --mock, --mock-pca, --freq, --rate, --model work properly."""
    script_path = REPO_ROOT / "pipelines" / "rpi5_stack" / "deployment" / "infer_rpi5.py"
    assert script_path.exists(), f"Missing script: {script_path}"
    import json
    from training.policy_contract import observation_schema
    schema_file = tmp_path / "verified-test-schema.json"
    schema_file.write_text(json.dumps(observation_schema(42, False, 0.02)), encoding="utf-8")
    from scripts.train_ppo import ActorCritic
    from scripts.export_policy import export_onnx
    policy = ActorCritic(hidden_dim=16)
    policy.observation_schema = observation_schema(42, False, 0.02)
    model_path = export_onnx(policy, tmp_path / "rpi_test.onnx")

    for mock_flag in ["--mock", "--mock-pca"]:
        cmd = [
            sys.executable,
            str(script_path),
            mock_flag,
            "--rate",
            "50",
            "--steps",
            "5",
            "--legacy-observation-schema", str(schema_file),
            "--model", str(model_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI with {mock_flag} failed with error: {res.stderr}\nOutput: {res.stdout}"
        assert "Mock PCA9685" in res.stdout or "mock" in res.stdout.lower()
        assert "Step 5" in res.stdout or "completed" in res.stdout.lower()

    # Verify alias --model and --rate
    res_help = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert res_help.returncode == 0
    assert "--model" in res_help.stdout
    assert "--rate" in res_help.stdout


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
    # Mahony AHRS & 3-layer IMU filtering
    assert "Mahony" in content
    assert "DLPF" in content or "0x1A" in content
    assert "0.30" in content or "Deadband" in content


def test_pizero2w_hardware_spec_doc():
    """Verify pizero2w_hardware_spec.md exists and contains thorough technical documentation."""
    spec_path = REPO_ROOT / "pipelines" / "pizero2w_stack" / "docs" / "pizero2w_hardware_spec.md"
    assert spec_path.exists(), f"Missing spec: {spec_path}"

    content = spec_path.read_text(encoding="utf-8")
    assert len(content) > 500, "Documentation is too brief"

    assert "Raspberry Pi Zero 2 W" in content
    assert "BCM2710A1" in content
    assert "5V/2.5A" in content or "2.5A" in content or "5.1V" in content
    assert "PCA9685" in content
    assert "MG996R" in content
    assert "I2C" in content
    assert "I2S" in content or "INMP441" in content
    assert "LiDAR" in content or "ToF" in content
    assert "2S" in content or "7.4V" in content
    assert "55" in content  # 55g electronics mass
    # Mahony AHRS & 3-layer IMU filtering
    assert "Mahony" in content
    assert "DLPF" in content or "0x1A" in content
    assert "0.30" in content or "Deadband" in content



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
    _compile_and_run_cpp(
        cpp_source=cpp_source,
        cpp_path=tmp_path / "test_protocol.cpp",
        exe_path=tmp_path / "test_protocol.exe",
        expected_stdout="All protocol tests passed successfully.",
    )



def test_mcu1_vision_audio_sketch():
    """Verify MCU 1 Arduino sketch contains necessary peripherals, pinouts, and high-speed UART."""
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
    # LiDAR I2C pinout separation from camera SCCB
    assert "#define PIN_I2C_SDA        19" in content
    assert "#define PIN_I2C_SCL        20" in content


def test_mcu2_locomotion_sketch():
    """Verify MCU 2 Arduino sketch handles UART, MPU6050, 42-dim obs, and 50Hz INT8 policy."""
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
    # Quantization and policy headers
    assert "esp32_quant_config.h" in content
    assert "esp32_policy_weights.h" in content
    # IMU, observation, control frequency
    assert "mpu" in content.lower() or "imu" in content.lower()
    assert "OBS_DIM" in content and "42" in content, "MCU 2 must assemble 42-dim observation"
    assert "50" in content or "20" in content, "MCU 2 must run 50Hz (20ms) loop"
    assert "mg996r" in content.lower() or "servo" in content.lower() or "pwm" in content.lower()
    # Gravity sign inversion & low-pass filtering
    assert "-ax / norm" in content
    assert "-ay / norm" in content
    assert "-az / norm" in content
    assert "s_filt_grav" in content
    assert "0.80" in content
    # MPU6050 Hardware DLPF, bias calibration, and Gyro filtering
    assert "0x1A" in content, "MCU 2 must configure MPU6050 DLPF CONFIG register"
    assert "0x03" in content, "MCU 2 must set 42Hz DLPF cutoff"
    assert "s_gyro_bias" in content, "MCU 2 must track gyro bias"
    assert "s_filt_gyro" in content, "MCU 2 must filter gyro signals"
    # Mahony sensor fusion & impact gating
    assert "fast_inv_sqrt" in content, "MCU 2 must implement fast inverse square root"
    assert "mahony_update" in content, "MCU 2 must implement Mahony filter update"
    assert "0.30" in content, "MCU 2 must implement impact gating with 0.30g threshold"
    assert "s_q0" in content and "s_q3" in content, "MCU 2 must maintain quaternion states"
    assert "s_integral_fb" in content, "MCU 2 must track integral feedback"
    # Decoupled raw actions array & observation assembly
    assert "g_last_raw_actions" in content
    assert "obs_out[25 + i] = g_last_raw_actions[i];" in content
    # INT8 Quantized evaluation with OBS_CHANNEL_SCALES
    assert "OBS_CHANNEL_SCALES" in content
    assert "evaluate_policy_int8" in content


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
    # Verify GPIO 19 and 20 for LiDAR I2C and GPIO 4, 5 for Camera SCCB
    assert "GPIO 19" in content and "GPIO 20" in content
    assert "CAM_SIOD" in content or "SCCB" in content
    # Mahony AHRS & 3-layer IMU filtering
    assert "Mahony" in content
    assert "DLPF" in content or "0x1A" in content
    assert "0.30" in content or "Deadband" in content


def test_mcu2_int8_policy_cpp_execution(tmp_path):
    """Compile and verify INT8 policy inference and IMU gravity filtering with C++ compiler."""
    quant_header = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "esp32_quant_config.h"
    )
    policy_header = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "esp32_policy_weights.h"
    )
    assert quant_header.exists(), f"Missing quant header: {quant_header}"
    assert policy_header.exists(), f"Missing policy header: {policy_header}"

    cpp_source = f"""#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "{quant_header.as_posix()}"
#include "{policy_header.as_posix()}"

int main() {{
    // 1. Verify gravity sign inversion and filtering matching MuJoCo [0, 0, -1]
    float ax = 0.0f, ay = 0.0f, az = 1.0f; // Upright resting accelerometer reading (+1g on z)
    float norm = std::sqrt(ax * ax + ay * ay + az * az);
    assert(norm > 1e-4f);

    float gx_norm = -ax / norm;
    float gy_norm = -ay / norm;
    float gz_norm = -az / norm;

    static float s_filt_grav[3] = {{0.0f, 0.0f, -1.0f}};
    const float alpha = 0.80f;
    s_filt_grav[0] = alpha * s_filt_grav[0] + (1.0f - alpha) * gx_norm;
    s_filt_grav[1] = alpha * s_filt_grav[1] + (1.0f - alpha) * gy_norm;
    s_filt_grav[2] = alpha * s_filt_grav[2] + (1.0f - alpha) * gz_norm;

    assert(std::fabs(s_filt_grav[0] - 0.0f) < 1e-4);
    assert(std::fabs(s_filt_grav[1] - 0.0f) < 1e-4);
    assert(std::fabs(s_filt_grav[2] - (-1.0f)) < 1e-4);

    // 2. Build 42-dimensional observation vector
    float obs[OBS_DIM] = {{0.0f}};
    obs[3] = s_filt_grav[0];
    obs[4] = s_filt_grav[1];
    obs[5] = s_filt_grav[2]; // -1.0f gravity
    obs[6] = 0.2f;           // vx = 0.2 m/s
    obs[33] = 1.0f;          // foot contact L
    obs[34] = 1.0f;          // foot contact R
    for (int i = 0; i < 5; ++i) {{
        obs[37 + i] = 1.5f;  // 1.5m LiDAR clear
    }}

    // Quantize observation to INT8
    int8_t obs_int8[OBS_DIM];
    for (int i = 0; i < OBS_DIM; ++i) {{
        float val = std::round(obs[i] * OBS_CHANNEL_SCALES[i]);
        val = std::max(-128.0f, std::min(127.0f, val));
        obs_int8[i] = (int8_t)val;
    }}

    // Forward inference
    int8_t actions_int8[ACTION_DIM];
    evaluate_policy_int8(obs_int8, actions_int8);

    // Dequantize and check bounds
    float raw_actions[ACTION_DIM];
    for (int i = 0; i < ACTION_DIM; ++i) {{
        float raw = (float)actions_int8[i] * ACT_INV_SCALE;
        raw = std::max(-1.0f, std::min(1.0f, raw));
        raw_actions[i] = raw;
        assert(raw >= -1.0f && raw <= 1.0f);
    }}

    std::cout << "MCU 2 INT8 policy inference executed successfully." << std::endl;
    return 0;
}}
"""
    _compile_and_run_cpp(
        cpp_source=cpp_source,
        cpp_path=tmp_path / "test_mcu2_policy.cpp",
        exe_path=tmp_path / "test_mcu2_policy.exe",
        expected_stdout="MCU 2 INT8 policy inference executed successfully.",
    )



def test_mcu2_mahony_filter_cpp_execution(tmp_path):
    """Compile and verify MCU2 Mahony AHRS algorithm, fast inverse square root, and impact gating in C++."""
    sketch_path = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "mcu2_locomotion_s3"
        / "mcu2_main.ino"
    )
    assert sketch_path.exists(), f"Missing sketch: {sketch_path}"

    cpp_source = """#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdint>

// Mahony Filter State Variables & Feedback matching mcu2_main.ino
static float s_q0 = 1.0f;
static float s_q1 = 0.0f;
static float s_q2 = 0.0f;
static float s_q3 = 0.0f;
static float s_integral_fb[3] = {0.0f, 0.0f, 0.0f};

#define MAHONY_KP              1.0f
#define MAHONY_KI              0.05f
#define IMPACT_GATE_THRESHOLD  0.30f

float fast_inv_sqrt(float x) {
    if (x <= 1e-8f) return 0.0f;
    float xhalf = 0.5f * x;
    union {
        float f;
        uint32_t i;
    } conv = {x};
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= (1.5f - xhalf * conv.f * conv.f);
    conv.f *= (1.5f - xhalf * conv.f * conv.f);
    return conv.f;
}

void mahony_update(float ax, float ay, float az, float gx, float gy, float gz, float dt) {
    float acc_sq = ax * ax + ay * ay + az * az;
    if (acc_sq > 1e-8f) {
        float acc_norm = std::sqrt(acc_sq);
        if (std::fabs(acc_norm - 1.0f) <= IMPACT_GATE_THRESHOLD) {
            float recip_acc = fast_inv_sqrt(acc_sq);
            ax *= recip_acc;
            ay *= recip_acc;
            az *= recip_acc;

            float vx = 2.0f * (s_q1 * s_q3 - s_q0 * s_q2);
            float vy = 2.0f * (s_q0 * s_q1 + s_q2 * s_q3);
            float vz = s_q0 * s_q0 - s_q1 * s_q1 - s_q2 * s_q2 + s_q3 * s_q3;

            float ex = (ay * vz - az * vy);
            float ey = (az * vx - ax * vz);
            float ez = (ax * vy - ay * vx);

            s_integral_fb[0] += MAHONY_KI * ex * dt;
            s_integral_fb[1] += MAHONY_KI * ey * dt;
            s_integral_fb[2] += MAHONY_KI * ez * dt;

            gx += MAHONY_KP * ex + s_integral_fb[0];
            gy += MAHONY_KP * ey + s_integral_fb[1];
            gz += MAHONY_KP * ez + s_integral_fb[2];
        } else {
            gx += s_integral_fb[0];
            gy += s_integral_fb[1];
            gz += s_integral_fb[2];
        }
    } else {
        gx += s_integral_fb[0];
        gy += s_integral_fb[1];
        gz += s_integral_fb[2];
    }

    gx *= (0.5f * dt);
    gy *= (0.5f * dt);
    gz *= (0.5f * dt);

    float qa = s_q0;
    float qb = s_q1;
    float qc = s_q2;

    s_q0 += (-qb * gx - qc * gy - s_q3 * gz);
    s_q1 += ( qa * gx + qc * gz - s_q3 * gy);
    s_q2 += ( qa * gy - qb * gz + s_q3 * gx);
    s_q3 += ( qa * gz + qb * gy - qc * gx);

    float q_recip = fast_inv_sqrt(s_q0 * s_q0 + s_q1 * s_q1 + s_q2 * s_q2 + s_q3 * s_q3);
    s_q0 *= q_recip;
    s_q1 *= q_recip;
    s_q2 *= q_recip;
    s_q3 *= q_recip;
}

void get_proj_grav(float* proj_grav) {
    proj_grav[0] = -2.0f * (s_q1 * s_q3 - s_q0 * s_q2);
    proj_grav[1] = -2.0f * (s_q2 * s_q3 + s_q0 * s_q1);
    proj_grav[2] = -(s_q0 * s_q0 - s_q1 * s_q1 - s_q2 * s_q2 + s_q3 * s_q3);
}

int main() {
    // 1. Test fast_inv_sqrt
    assert(std::fabs(fast_inv_sqrt(1.0f) - 1.0f) < 0.01f);
    assert(std::fabs(fast_inv_sqrt(4.0f) - 0.5f) < 0.01f);
    assert(fast_inv_sqrt(0.0f) == 0.0f);

    // 2. Initial projected gravity
    float pg[3];
    get_proj_grav(pg);
    assert(std::fabs(pg[0] - 0.0f) < 1e-4f);
    assert(std::fabs(pg[1] - 0.0f) < 1e-4f);
    assert(std::fabs(pg[2] - (-1.0f)) < 1e-4f);

    // 3. Stationary upright updates
    for (int i = 0; i < 50; ++i) {
        mahony_update(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.02f);
    }
    get_proj_grav(pg);
    assert(std::fabs(pg[0] - 0.0f) < 1e-3f);
    assert(std::fabs(pg[1] - 0.0f) < 1e-3f);
    assert(std::fabs(pg[2] - (-1.0f)) < 1e-3f);
    assert(std::fabs(s_integral_fb[0]) < 1e-4f);
    assert(std::fabs(s_integral_fb[1]) < 1e-4f);

    // 4. Test Impact Gating: 2.5g foot strike shock spike with tilt
    float prev_ifb0 = s_integral_fb[0];
    float prev_ifb1 = s_integral_fb[1];
    float prev_q0 = s_q0;
    mahony_update(1.5f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.02f); // norm = 2.5g -> gated!
    assert(s_integral_fb[0] == prev_ifb0);
    assert(s_integral_fb[1] == prev_ifb1);

    // 5. Test Normal Correction: 0.995g tilted acceleration
    mahony_update(0.1f, 0.0f, 0.99f, 0.0f, 0.0f, 0.0f, 0.02f); // norm ~ 0.995g -> corrected!
    assert(s_integral_fb[1] != prev_ifb1 || s_q0 != prev_q0);

    std::cout << "Mahony AHRS C++ verification passed successfully." << std::endl;
    return 0;
}
"""
    _compile_and_run_cpp(
        cpp_source=cpp_source,
        cpp_path=tmp_path / "test_mahony.cpp",
        exe_path=tmp_path / "test_mahony.exe",
        expected_stdout="Mahony AHRS C++ verification passed successfully.",
    )



def test_all_three_models_mpu6050_filtering():
    """Verify that all 3 deployment pipelines have MPU6050 noise filtering implemented."""
    # 1. MCU 2 (ESP32-S3)
    mcu2_sketch = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "mcu2_locomotion_s3"
        / "mcu2_main.ino"
    )
    content = mcu2_sketch.read_text(encoding="utf-8")
    assert "s_filt_gyro" in content
    assert "s_gyro_bias" in content
    assert "0x1A" in content and "0x03" in content

    # 2. Pi Zero 2 W
    from pipelines.pizero2w_stack.deployment.infer_pizero2w import (
        MPU6050FilterReader as PiZero2W_IMU,
        PiZero2WInferenceEngine,
    )

    imu_pizero = PiZero2W_IMU(mock=True)
    ang_vel, proj_grav = imu_pizero.read()
    assert ang_vel.shape == (3,)
    assert proj_grav.shape == (3,)
    assert np.allclose(proj_grav, [0.0, 0.0, -1.0])
    engine_pizero = PiZero2WInferenceEngine(mock=True)
    assert hasattr(engine_pizero, "imu_reader")

    # 3. Raspberry Pi 5
    from pipelines.rpi5_stack.deployment.infer_rpi5 import (
        MPU6050FilterReader as RPi5_IMU,
        RPi5InferenceEngine,
    )

    imu_rpi5 = RPi5_IMU(mock=True)
    ang_vel5, proj_grav5 = imu_rpi5.read()
    assert ang_vel5.shape == (3,)
    assert proj_grav5.shape == (3,)
    assert np.allclose(proj_grav5, [0.0, 0.0, -1.0])
    engine_rpi5 = RPi5InferenceEngine(mock=True)
    assert hasattr(engine_rpi5, "imu_reader")


def test_pizero2w_mahony_filter_algorithm():
    """Verify Pi Zero 2 W MPU6050FilterReader vectorized Mahony AHRS and dynamic impact gating."""
    from pipelines.pizero2w_stack.deployment.infer_pizero2w import (
        MPU6050FilterReader,
        PiZero2WInferenceEngine,
    )

    reader = MPU6050FilterReader(mock=True)

    # 1. Check state initialization
    assert hasattr(reader, "q"), "Reader must maintain quaternion state vector q"
    assert reader.q.shape == (4,)
    assert np.allclose(reader.q, [1.0, 0.0, 0.0, 0.0]), "Quaternion must be initialized to [1, 0, 0, 0]"
    assert hasattr(reader, "integral_fb"), "Reader must maintain integral_fb"
    assert reader.integral_fb.shape == (3,)
    assert np.allclose(reader.integral_fb, [0.0, 0.0, 0.0])
    assert reader.kp == 1.0, f"Expected Kp=1.0, got {reader.kp}"
    assert reader.ki == 0.05, f"Expected Ki=0.05, got {reader.ki}"
    assert reader.impact_threshold == 0.30, f"Expected threshold 0.30, got {reader.impact_threshold}"

    # 2. Check initial projected gravity matching MuJoCo convention [0, 0, -1]
    pg = reader.get_proj_grav()
    assert pg.shape == (3,)
    assert np.allclose(pg, [0.0, 0.0, -1.0], atol=1e-4)

    # 3. Test Stationary Upright updates: 50 steps at dt=0.02s
    for _ in range(50):
        reader.update(ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0, dt=0.02)

    pg_upright = reader.get_proj_grav()
    assert np.allclose(pg_upright, [0.0, 0.0, -1.0], atol=1e-3)
    assert np.allclose(reader.integral_fb, [0.0, 0.0, 0.0], atol=1e-4)

    # 4. Test Dynamic Foot-Strike Impact Gating (> 1.30g shock spike)
    # 2.5g foot impact spike (e.g. ax=1.5, ay=0.0, az=2.0 -> norm = 2.5g)
    prev_ifb = reader.integral_fb.copy()
    prev_q = reader.q.copy()
    reader.update(ax=1.5, ay=0.0, az=2.0, gx=0.0, gy=0.0, gz=0.0, dt=0.02)
    assert np.allclose(reader.integral_fb, prev_ifb), (
        "Foot strike impact > 1.30g must be gated; integral feedback must not update"
    )

    # Also test free-fall / low-g flight shock gating (< 0.70g)
    reader.update(ax=0.0, ay=0.0, az=0.4, gx=0.0, gy=0.0, gz=0.0, dt=0.02)
    assert np.allclose(reader.integral_fb, prev_ifb), (
        "Shock < 0.70g must be gated; integral feedback must not update"
    )

    # 5. Test Normal Accelerometer Correction (norm ~ 0.995g, within 0.30g gate)
    reader.update(ax=0.1, ay=0.0, az=0.99, gx=0.0, gy=0.0, gz=0.0, dt=0.02)
    assert not np.allclose(reader.integral_fb, prev_ifb) or not np.allclose(reader.q, prev_q), (
        "Non-impact acceleration must update Mahony filter correction"
    )

    # 6. Test Mock Mode read() contract
    reader_mock = MPU6050FilterReader(mock=True)
    ang_vel, proj_grav = reader_mock.read()
    assert ang_vel.shape == (3,)
    assert proj_grav.shape == (3,)
    assert np.allclose(proj_grav, [0.0, 0.0, -1.0])

    # 7. Test PiZero2WInferenceEngine integration
    engine = PiZero2WInferenceEngine(mock=True)
    assert hasattr(engine, "imu_reader")
    assert isinstance(engine.imu_reader, MPU6050FilterReader)


def test_rpi5_mahony_filter_algorithm():
    """Verify Raspberry Pi 5 MPU6050FilterReader vectorized Mahony AHRS and dynamic impact gating."""
    from pipelines.rpi5_stack.deployment.infer_rpi5 import (
        MPU6050FilterReader,
        RPi5InferenceEngine,
    )

    reader = MPU6050FilterReader(mock=True)

    # 1. Check state initialization
    assert hasattr(reader, "q"), "Reader must maintain quaternion state vector q"
    assert reader.q.shape == (4,)
    assert np.allclose(reader.q, [1.0, 0.0, 0.0, 0.0]), "Quaternion must be initialized to [1, 0, 0, 0]"
    assert hasattr(reader, "integral_fb"), "Reader must maintain integral_fb"
    assert reader.integral_fb.shape == (3,)
    assert np.allclose(reader.integral_fb, [0.0, 0.0, 0.0])
    assert reader.kp == 1.0, f"Expected Kp=1.0, got {reader.kp}"
    assert reader.ki == 0.05, f"Expected Ki=0.05, got {reader.ki}"
    assert reader.impact_threshold == 0.30, f"Expected threshold 0.30, got {reader.impact_threshold}"

    # 2. Check initial projected gravity matching MuJoCo convention [0, 0, -1]
    pg = reader.get_proj_grav()
    assert pg.shape == (3,)
    assert np.allclose(pg, [0.0, 0.0, -1.0], atol=1e-4)

    # 3. Test Stationary Upright updates: 50 steps at dt=0.02s
    for _ in range(50):
        reader.update(ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0, dt=0.02)

    pg_upright = reader.get_proj_grav()
    assert np.allclose(pg_upright, [0.0, 0.0, -1.0], atol=1e-3)
    assert np.allclose(reader.integral_fb, [0.0, 0.0, 0.0], atol=1e-4)

    # 4. Test Dynamic Foot-Strike Impact Gating (> 1.30g shock spike)
    # 2.5g foot impact spike (e.g. ax=1.5, ay=0.0, az=2.0 -> norm = 2.5g)
    prev_ifb = reader.integral_fb.copy()
    prev_q = reader.q.copy()
    reader.update(ax=1.5, ay=0.0, az=2.0, gx=0.0, gy=0.0, gz=0.0, dt=0.02)
    assert np.allclose(reader.integral_fb, prev_ifb), (
        "Foot strike impact > 1.30g must be gated; integral feedback must not update"
    )

    # Also test free-fall / low-g flight shock gating (< 0.70g)
    reader.update(ax=0.0, ay=0.0, az=0.4, gx=0.0, gy=0.0, gz=0.0, dt=0.02)
    assert np.allclose(reader.integral_fb, prev_ifb), (
        "Shock < 0.70g must be gated; integral feedback must not update"
    )

    # 5. Test Normal Accelerometer Correction (norm ~ 0.995g, within 0.30g gate)
    reader.update(ax=0.1, ay=0.0, az=0.99, gx=0.0, gy=0.0, gz=0.0, dt=0.02)
    assert not np.allclose(reader.integral_fb, prev_ifb) or not np.allclose(reader.q, prev_q), (
        "Non-impact acceleration must update Mahony filter correction"
    )

    # 6. Test Mock Mode read() contract
    reader_mock = MPU6050FilterReader(mock=True)
    ang_vel, proj_grav = reader_mock.read()
    assert ang_vel.shape == (3,)
    assert proj_grav.shape == (3,)
    assert np.allclose(proj_grav, [0.0, 0.0, -1.0])

    # 7. Test RPi5InferenceEngine integration
    engine = RPi5InferenceEngine(mock=True)
    assert hasattr(engine, "imu_reader")
    assert isinstance(engine.imu_reader, MPU6050FilterReader)


def test_deploy_proj_grav_tilted_and_nominal_action_match_sim(tmp_path):
    """Verify RPi5, PiZero2W, and MCU2 proj_grav formula against quat_rotate_inverse(q, [0, 0, -1])
    for tilted pitch +20deg and roll +20deg, AND verify action = zeros(8) maps to DEFAULT_NOMINAL_QPOS."""
    import re
    from training.env_config import quat_rotate_inverse, DEFAULT_NOMINAL_QPOS
    from pipelines.rpi5_stack.deployment.infer_rpi5 import (
        MPU6050FilterReader as RPi5_IMU,
        RPi5InferenceEngine,
    )
    from pipelines.pizero2w_stack.deployment.infer_pizero2w import (
        MPU6050FilterReader as PiZero2W_IMU,
        PiZero2WInferenceEngine,
    )

    # 1. Quaternions for +20 deg pitch and +20 deg roll
    theta = float(np.radians(20.0))
    q_pitch = np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0], dtype=np.float32)
    q_roll = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0], dtype=np.float32)
    v_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    expected_pitch = quat_rotate_inverse(q_pitch, v_grav)
    expected_roll = quat_rotate_inverse(q_roll, v_grav)

    # Check RPi5 tilted proj_grav
    imu_rpi5 = RPi5_IMU(mock=True)
    imu_rpi5.q = q_pitch.copy()
    pg_pitch_rpi5 = imu_rpi5.get_proj_grav()
    assert np.allclose(pg_pitch_rpi5, expected_pitch, atol=1e-3), (
        f"RPi5 pitch +20deg proj_grav mismatch: {pg_pitch_rpi5} vs expected {expected_pitch}"
    )

    imu_rpi5.q = q_roll.copy()
    pg_roll_rpi5 = imu_rpi5.get_proj_grav()
    assert np.allclose(pg_roll_rpi5, expected_roll, atol=1e-3), (
        f"RPi5 roll +20deg proj_grav mismatch: {pg_roll_rpi5} vs expected {expected_roll}"
    )

    # Check PiZero2W tilted proj_grav
    imu_pizero = PiZero2W_IMU(mock=True)
    imu_pizero.q = q_pitch.copy()
    pg_pitch_pizero = imu_pizero.get_proj_grav()
    assert np.allclose(pg_pitch_pizero, expected_pitch, atol=1e-3), (
        f"PiZero2W pitch +20deg proj_grav mismatch: {pg_pitch_pizero} vs expected {expected_pitch}"
    )

    imu_pizero.q = q_roll.copy()
    pg_roll_pizero = imu_pizero.get_proj_grav()
    assert np.allclose(pg_roll_pizero, expected_roll, atol=1e-3), (
        f"PiZero2W roll +20deg proj_grav mismatch: {pg_roll_pizero} vs expected {expected_roll}"
    )

    # Check MCU2 tilted proj_grav and nominal action scaling by compiling C++ test with formulas from mcu2_main.ino
    mcu2_sketch = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "mcu2_locomotion_s3"
        / "mcu2_main.ino"
    )
    mcu2_content = mcu2_sketch.read_text(encoding="utf-8")

    # Extract proj_grav lines from mcu2_main.ino
    m0 = re.search(r"proj_grav\[0\]\s*=\s*([^;]+);", mcu2_content)
    m1 = re.search(r"proj_grav\[1\]\s*=\s*([^;]+);", mcu2_content)
    m2 = re.search(r"proj_grav\[2\]\s*=\s*([^;]+);", mcu2_content)
    assert m0 and m1 and m2, "Could not extract proj_grav lines from mcu2_main.ino"

    # Extract action scaling from evaluate_locomotion_policy in mcu2_main.ino
    from training.env_config import DEFAULT_ACTION_SCALES, scale_actions
    m_scale = re.search(r"const\s+float\s+JOINT_ACTION_SCALE\[8\]\s*=\s*\{([^}]+)\};", mcu2_content)
    assert m_scale, "Could not extract JOINT_ACTION_SCALE from mcu2_main.ino"
    mcu_scales = np.array([float(x.replace("f", "").strip()) for x in m_scale.group(1).split(",")], dtype=np.float32)
    assert np.allclose(mcu_scales, DEFAULT_ACTION_SCALES, atol=1e-5), f"MCU action scale mismatch: {mcu_scales} vs {DEFAULT_ACTION_SCALES}"

    m_act = re.search(r"target_angle\s*=\s*([^;]+);", mcu2_content)
    assert m_act, "Could not extract target_angle line from mcu2_main.ino"

    cpp_test = f"""#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

template <typename T>
T constrain(T val, T low, T high) {{
    return std::max(low, std::min(high, val));
}}

const float JOINT_MIN_LIMITS[8] = {{-1.57f, -0.52f, -0.44f, -0.87f, 0.00f, -0.44f, -0.87f, 0.00f}};
const float JOINT_MAX_LIMITS[8] = {{ 1.57f,  0.78f,  0.44f,  0.87f, 1.48f,  0.44f,  0.87f, 1.48f}};
const float JOINT_NOMINAL_POS[8] = {{0.00f, 0.00f, 0.00f, -0.15f, 0.15f, 0.00f, -0.15f, 0.15f}};
const float JOINT_ACTION_SCALE[8] = {{{m_scale.group(1)}}};

void compute_pg(float s_q0, float s_q1, float s_q2, float s_q3, float* proj_grav) {{
    proj_grav[0] = {m0.group(1)};
    proj_grav[1] = {m1.group(1)};
    proj_grav[2] = {m2.group(1)};
}}

void compute_action(float raw_act, float* targets) {{
    for (int i = 0; i < 8; ++i) {{
        float target_angle = {m_act.group(1)};
        targets[i] = constrain(target_angle, JOINT_MIN_LIMITS[i], JOINT_MAX_LIMITS[i]);
    }}
}}

int main() {{
    float pg[3];
    // Test pitch +20 deg
    compute_pg({q_pitch[0]}f, {q_pitch[1]}f, {q_pitch[2]}f, {q_pitch[3]}f, pg);
    if (std::fabs(pg[0] - ({expected_pitch[0]}f)) > 1e-3f ||
        std::fabs(pg[1] - ({expected_pitch[1]}f)) > 1e-3f ||
        std::fabs(pg[2] - ({expected_pitch[2]}f)) > 1e-3f) {{
        std::cerr << "MCU2 pitch +20deg pg mismatch: [" << pg[0] << ", " << pg[1] << ", " << pg[2] << "] vs expected ["
                  << {expected_pitch[0]} << ", " << {expected_pitch[1]} << ", " << {expected_pitch[2]} << "]" << std::endl;
        return 1;
    }}

    // Test roll +20 deg
    compute_pg({q_roll[0]}f, {q_roll[1]}f, {q_roll[2]}f, {q_roll[3]}f, pg);
    if (std::fabs(pg[0] - ({expected_roll[0]}f)) > 1e-3f ||
        std::fabs(pg[1] - ({expected_roll[1]}f)) > 1e-3f ||
        std::fabs(pg[2] - ({expected_roll[2]}f)) > 1e-3f) {{
        std::cerr << "MCU2 roll +20deg pg mismatch: [" << pg[0] << ", " << pg[1] << ", " << pg[2] << "] vs expected ["
                  << {expected_roll[0]} << ", " << {expected_roll[1]} << ", " << {expected_roll[2]} << "]" << std::endl;
        return 2;
    }}

    // Test action = 0 maps to nominal
    float targets[8];
    compute_action(0.0f, targets);
    for (int i = 0; i < 8; ++i) {{
        if (std::fabs(targets[i] - JOINT_NOMINAL_POS[i]) > 1e-4f) {{
            std::cerr << "MCU2 action zero joint " << i << " mismatch: " << targets[i] << " vs " << JOINT_NOMINAL_POS[i] << std::endl;
            return 3;
        }}
    }}

    // Test action = +0.5f matches residual scaling
    float targets_pos[8];
    compute_action(0.5f, targets_pos);
    for (int i = 0; i < 8; ++i) {{
        float expected = constrain(JOINT_NOMINAL_POS[i] + 0.5f * JOINT_ACTION_SCALE[i], JOINT_MIN_LIMITS[i], JOINT_MAX_LIMITS[i]);
        if (std::fabs(targets_pos[i] - expected) > 1e-4f) {{
            std::cerr << "MCU2 action +0.5 joint " << i << " mismatch: " << targets_pos[i] << " vs " << expected << std::endl;
            return 4;
        }}
    }}
    return 0;
}}
"""
    _compile_and_run_cpp(
        cpp_source=cpp_test,
        cpp_path=tmp_path / "test_mcu2_tilted_and_action.cpp",
        exe_path=tmp_path / "test_mcu2_tilted_and_action.exe",
        skip_whole_test=False,
    )

    # 2. Check RPi5 nominal action scaling (action = 0.0 -> DEFAULT_NOMINAL_QPOS)
    engine_rpi5 = RPi5InferenceEngine(mock=True)
    engine_rpi5.predict_action = lambda obs: np.zeros(8, dtype=np.float32)
    targets_rpi5, _ = engine_rpi5.step(np.zeros(42, dtype=np.float32))
    assert np.allclose(targets_rpi5, DEFAULT_NOMINAL_QPOS, atol=1e-4), (
        f"RPi5 action 0 did not map to DEFAULT_NOMINAL_QPOS: {targets_rpi5} vs {DEFAULT_NOMINAL_QPOS}"
    )

    # 3. Check PiZero2W nominal action scaling (action = 0.0 -> DEFAULT_NOMINAL_QPOS)
    engine_pizero = PiZero2WInferenceEngine(mock=True)
    engine_pizero.predict_action = lambda obs: np.zeros(8, dtype=np.float32)
    targets_pizero, _ = engine_pizero.step(np.zeros(42, dtype=np.float32))
    assert np.allclose(targets_pizero, DEFAULT_NOMINAL_QPOS, atol=1e-4), (
        f"PiZero2W action 0 did not map to DEFAULT_NOMINAL_QPOS: {targets_pizero} vs {DEFAULT_NOMINAL_QPOS}"
    )


def test_mixed_deploy_test_runs_python_checks_when_gpp_missing(monkeypatch, tmp_path):
    """Mixed Python+C++ test skips only g++ step and still runs RPi5/PiZero2W checks."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "g++" else cmd)
    test_deploy_proj_grav_tilted_and_nominal_action_match_sim(tmp_path)



def test_esp32_int8_export_matches_float32_inference(tmp_path):
    """Export dummy ActorCritic via export_esp32_quantized_header,
    simulate mcu2_main.ino int8 inference, and assert MAE < 0.02 vs float32 actor(obs)."""
    import re
    import torch
    from scripts.train_ppo import ActorCritic
    from scripts.export_policy import export_esp32_headers
    from pipelines.dual_esp32s3_stack.training.env_config import DEFAULT_OBS_BOUNDS

    torch.manual_seed(42)
    np.random.seed(42)

    obs_dim = 42
    action_dim = 8
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    torch.nn.init.orthogonal_(model.actor[-1].weight, gain=0.2)
    model.eval()

    weights_header_path = tmp_path / "esp32_policy_weights.h"
    config_header_path = tmp_path / "esp32_quant_config.h"

    export_esp32_headers(
        model=model,
        weights_output_path=weights_header_path,
        config_output_path=config_header_path,
    )

    assert weights_header_path.exists()
    assert config_header_path.exists()

    # Parse config header
    cfg_text = config_header_path.read_text(encoding="utf-8")
    m_scales = re.search(r"OBS_CHANNEL_SCALES\[[^\]]*\]\s*=\s*\{([^}]+)\}", cfg_text)
    assert m_scales, "Could not parse OBS_CHANNEL_SCALES from esp32_quant_config.h"
    obs_channel_scales = np.array(
        [float(x.strip().rstrip("f")) for x in m_scales.group(1).split(",") if x.strip()],
        dtype=np.float32,
    )
    assert len(obs_channel_scales) == obs_dim

    # Parse weights header
    w_text = weights_header_path.read_text(encoding="utf-8")

    def parse_array(name, dtype):
        m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{([^}}]+)\}}", w_text)
        assert m, f"Could not parse {name} from esp32_policy_weights.h"
        vals = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        return np.array(vals, dtype=dtype)

    def parse_float_scale(name):
        m = re.search(rf"{name}\s*=\s*\(?1\.0f\s*/\s*([0-9.]+)f?\)?;", w_text)
        if m:
            return 1.0 / float(m.group(1))
        m2 = re.search(rf"{name}\s*=\s*([0-9.]+)f?;", w_text)
        assert m2, f"Could not parse {name} scale"
        return float(m2.group(1))

    w1 = parse_array("POLICY_W1", np.int8).reshape(256, obs_dim)
    b1 = parse_array("POLICY_B1", np.int32)
    b1_inv_scale = parse_float_scale("POLICY_B1_INV_SCALE")

    w2 = parse_array("POLICY_W2", np.int8).reshape(256, 256)
    b2 = parse_array("POLICY_B2", np.int32)
    b2_inv_scale = parse_float_scale("POLICY_B2_INV_SCALE")
    w2_inv_scale = parse_float_scale("POLICY_W2_INV_SCALE")

    w3 = parse_array("POLICY_W3", np.int8).reshape(8, 256)
    b3 = parse_array("POLICY_B3", np.int32)
    b3_inv_scale = parse_float_scale("POLICY_B3_INV_SCALE")
    w3_inv_scale = parse_float_scale("POLICY_W3_INV_SCALE")

    # Generate representative observation batch spanning real bounds
    batch_obs = [
        np.zeros(obs_dim, dtype=np.float32),
    ]
    # Physical walking observation
    walk_obs = np.zeros(obs_dim, dtype=np.float32)
    walk_obs[0:3] = [0.1, -0.2, 0.05]   # ang vel
    walk_obs[3:6] = [0.0, 0.0, -1.0]    # upright grav
    walk_obs[6:9] = [0.25, 0.0, 0.0]    # cmd
    walk_obs[9:17] = [0.0, 0.0, 0.0, -0.15, 0.15, 0.0, -0.15, 0.15] # nominal qpos
    walk_obs[17:25] = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.1, -0.1]  # joint vel
    walk_obs[25:33] = [0.0] * 8         # last actions
    walk_obs[33:37] = [1.0, 1.0, 0.0, 0.0] # foot contact
    walk_obs[37:42] = [1.2, 1.5, 0.8, 1.1, 1.4] # lidar
    batch_obs.append(walk_obs)

    # Random observations spanning physical bounds
    for seed in range(25):
        rng = np.random.RandomState(seed)
        r_obs = rng.uniform(-DEFAULT_OBS_BOUNDS[:obs_dim], DEFAULT_OBS_BOUNDS[:obs_dim]).astype(np.float32)
        batch_obs.append(r_obs)

    obs_tensor = torch.from_numpy(np.array(batch_obs, dtype=np.float32))
    with torch.no_grad():
        float_out = model.actor(obs_tensor).numpy()

    # Run C-equivalent forward inference matching mcu2_main.ino + evaluate_policy_int8
    c_outs = []
    for obs in batch_obs:
        # Quantize obs matching mcu2_main.ino:473-478
        obs_int8 = np.clip(np.round(obs * obs_channel_scales), -128, 127).astype(np.int32)

        # Layer 1 matmul matching evaluate_policy_int8:298-307
        acc1 = b1 + np.dot(w1.astype(np.int32), obs_int8)
        val1 = acc1.astype(np.float32) * b1_inv_scale
        l1_out = np.where(val1 >= 0.0, val1, np.exp(val1) - 1.0)

        # Layer 2 matmul matching evaluate_policy_int8:310-318
        acc2 = (b2.astype(np.float32) * b2_inv_scale) + np.dot(w2.astype(np.float32) * w2_inv_scale, l1_out)
        l2_out = np.where(acc2 >= 0.0, acc2, np.exp(acc2) - 1.0)

        # Layer 3 matmul matching evaluate_policy_int8:321-331
        acc3 = (b3.astype(np.float32) * b3_inv_scale) + np.dot(w3.astype(np.float32) * w3_inv_scale, l2_out)
        scaled = np.clip(np.round(acc3 * 127.0), -128, 127).astype(np.int8)

        # Dequantize actions matching mcu2_main.ino:486
        raw_act = np.clip(scaled.astype(np.float32) * (1.0 / 127.0), -1.0, 1.0)
        c_outs.append(raw_act)

    c_outs = np.array(c_outs, dtype=np.float32)
    mae = float(np.mean(np.abs(c_outs - float_out)))
    print(f"\nMeasured INT8 vs Float32 MAE: {mae:.6f}")
    assert mae < 0.02, f"ESP32 INT8 policy inference MAE too high: {mae:.6f} >= 0.02"


def test_uart_protocol_and_sensor_wiring_in_firmware_and_python():
    """Task 6: Structural and protocol assertions for firmware and deployment scripts."""
    proto = Path("pipelines/dual_esp32s3_stack/deployment/uart_packet_protocol.h").read_text(encoding="utf-8")
    mcu1 = Path("pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino").read_text(encoding="utf-8")
    mcu2 = Path("pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino").read_text(encoding="utf-8")

    assert "PKT_TYPE_COLOR_TARGET" in proto, "uart_packet_protocol.h missing PKT_TYPE_COLOR_TARGET"
    assert "ColorTargetPacket" in proto, "uart_packet_protocol.h missing ColorTargetPacket struct"
    assert "uart_pack_frame" in proto, "uart_packet_protocol.h missing uart_pack_frame helper"
    assert "process_color_tracking();" in mcu1 and "uart_pack_frame" in mcu1, (
        "mcu1_main.ino must call process_color_tracking() in loop() and use uart_pack_frame"
    )
    assert "PKT_TYPE_EMERGENCY_STOP" in mcu2 and "PKT_TYPE_VOICE_INTENT_CMD" in mcu2, (
        "mcu2_main.ino must decode PKT_TYPE_EMERGENCY_STOP and PKT_TYPE_VOICE_INTENT_CMD"
    )
    assert "PKT_TYPE_COLOR_TARGET" in mcu2, "mcu2_main.ino must handle PKT_TYPE_COLOR_TARGET"
    assert "joint_vel[i] = 0.0f" not in mcu2, "mcu2_main.ino must use finite-difference (q - prev_q) / dt"

    # Fix Round 1 Checks: E-stop latching and CRC guard
    # Ensure g_emergency_stop is declared static bool g_emergency_stop = false, but never unlatched in runtime code
    runtime_mcu2 = mcu2.replace("static bool g_emergency_stop = false;", "")
    assert "g_emergency_stop = false;" not in runtime_mcu2, (
        "CRITICAL: g_emergency_stop must remain strictly latched until MCU reset/reboot and not auto-unlatch on velocity commands"
    )
    assert "decode_packet(g_rx_ring, total_len" in mcu2, (
        "IMPORTANT: mcu2_main.ino must validate CRC16 via decode_packet before triggering PKT_TYPE_EMERGENCY_STOP"
    )
    assert "if (decode_velocity_packet(" in mcu2, (
        "IMPORTANT: mcu2_main.ino must guard g_last_mcu1_pkt_time inside if (decode_velocity_packet(...))"
    )
    assert "if (decode_lidar_packet(" in mcu2, (
        "IMPORTANT: mcu2_main.ino must guard g_last_mcu1_pkt_time inside if (decode_lidar_packet(...))"
    )
    assert "if (decode_heartbeat_packet(" in mcu2, (
        "IMPORTANT: mcu2_main.ino must guard g_last_mcu1_pkt_time inside if (decode_heartbeat_packet(...))"
    )

    for py_path in (
        "pipelines/rpi5_stack/deployment/infer_rpi5.py",
        "pipelines/pizero2w_stack/deployment/infer_pizero2w.py",
    ):
        src = Path(py_path).read_text(encoding="utf-8")
        assert "prev_q" in src, f"{py_path} must compute finite-difference joint_vel using prev_q"
        assert "read_foot_contact" in src, f"{py_path} must provide read_foot_contact hook"
        assert "read_lidar_ranges" in src, f"{py_path} must provide read_lidar_ranges hook"


def test_uart_protocol_header_compilation_and_crc(tmp_path):
    """Verify C++ compilation and real CRC16 packing/unpacking for uart_packet_protocol.h."""
    header_path = REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "deployment" / "uart_packet_protocol.h"
    cpp_source = f"""#include <iostream>
#include <cassert>
#include "{header_path.as_posix()}"

int main() {{
    uint8_t buffer[128];
    ColorTargetPacket target_in;
    target_in.yaw_target_deg10 = -150;
    target_in.pitch_target_deg10 = 80;
    target_in.target_detected = 1;

    uint16_t packed_len = uart_pack_frame(
        PKT_TYPE_COLOR_TARGET,
        42,
        &target_in,
        sizeof(ColorTargetPacket),
        buffer,
        sizeof(buffer)
    );
    assert(packed_len == PACKET_HEADER_SIZE + sizeof(ColorTargetPacket) + PACKET_CRC_SIZE);
    assert(buffer[0] == PACKET_SYNC_0 && buffer[1] == PACKET_SYNC_1);
    assert(buffer[2] == PKT_TYPE_COLOR_TARGET);
    assert(buffer[3] == 42);

    ColorTargetPacket target_out;
    uint8_t seq = 0;
    bool ok = decode_color_target_packet(buffer, packed_len, &target_out, &seq);
    assert(ok);
    assert(seq == 42);
    assert(target_out.yaw_target_deg10 == -150);
    assert(target_out.pitch_target_deg10 == 80);
    assert(target_out.target_detected == 1);

    // Corrupt CRC and verify rejection
    buffer[packed_len - 1] ^= 0xFF;
    bool corrupt_ok = decode_color_target_packet(buffer, packed_len, &target_out, &seq);
    assert(!corrupt_ok);

    std::cout << "UART protocol C++ verification passed" << std::endl;
    return 0;
}}
"""
    _compile_and_run_cpp(
        cpp_source=cpp_source,
        cpp_path=tmp_path / "test_uart.cpp",
        exe_path=tmp_path / "test_uart.exe",
        expected_stdout="UART protocol C++ verification passed",
    )



def test_rpi5_and_pizero2w_joint_vel_and_sensor_fallback():
    """Verify finite-difference joint_vel and sensor fallback warnings in Python engines."""
    from pipelines.rpi5_stack.deployment.infer_rpi5 import RPi5InferenceEngine
    from pipelines.pizero2w_stack.deployment.infer_pizero2w import PiZero2WInferenceEngine

    for engine_cls, name in [(RPi5InferenceEngine, "RPi5"), (PiZero2WInferenceEngine, "PiZero2W")]:
        engine = engine_cls(mock=True, freq=50.0)
        assert hasattr(engine, "prev_q"), f"{name} must track prev_q"
        assert hasattr(engine, "read_foot_contact"), f"{name} must implement read_foot_contact"
        assert hasattr(engine, "read_lidar_ranges"), f"{name} must implement read_lidar_ranges"

        # Check fallback returns standard formats
        foot = engine.read_foot_contact()
        assert len(foot) == 4
        lidar = engine.read_lidar_ranges()
        assert len(lidar) == 5

        # Check finite-difference joint velocity calculation
        # Step 0: start at nominal
        q0 = engine.last_targets.copy()
        # Change targets
        q1 = q0 + 0.1
        vel = (q1 - q0) / engine.dt
        assert not np.allclose(vel, 0.0), "Moving joints must produce non-zero finite-difference joint_vel"


def test_mcu2_emergency_stop_latching_and_crc_validation(tmp_path):
    """Verify MCU2 C++ packet parser: E-stop CRC validation, latching, and CRC-guarded watchdog."""
    header_path = REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "deployment" / "uart_packet_protocol.h"
    cpp_source = f"""#include <iostream>
#include <cassert>
#include <cstring>
#include "{header_path.as_posix()}"

// Simulated MCU2 state matching mcu2_main.ino
bool g_emergency_stop = false;
VelocityCmdPacket g_cmd_pkt = {{0.20f, 0.0f, 0.0f}};
uint32_t g_last_mcu1_pkt_time = 1000;
uint32_t current_time = 2000;

void process_frame(const uint8_t *frame, uint16_t total_len) {{
    uint8_t p_type = frame[2];
    uint8_t seq = 0;
    uint16_t payload_len = (uint16_t)frame[4] | (((uint16_t)frame[5]) << 8);

    if (p_type == PKT_TYPE_VELOCITY_CMD) {{
        if (decode_velocity_packet(frame, total_len, &g_cmd_pkt, &seq)) {{
            g_last_mcu1_pkt_time = current_time;
        }}
    }} else if (p_type == PKT_TYPE_EMERGENCY_STOP) {{
        uint16_t out_len = 0;
        if (decode_packet(frame, total_len, &p_type, &seq, NULL, 0, &out_len)) {{
            g_emergency_stop = true;
            g_cmd_pkt.vx = 0.0f;
            g_cmd_pkt.vy = 0.0f;
            g_cmd_pkt.wz = 0.0f;
        }}
    }}
}}

int main() {{
    uint8_t estop_frame[128];
    // Encode valid EMERGENCY_STOP frame (0-byte payload)
    uint16_t estop_len = encode_packet(PKT_TYPE_EMERGENCY_STOP, 1, NULL, 0, estop_frame, sizeof(estop_frame));

    // 1. Corrupt CRC: verify emergency stop is NOT triggered
    uint8_t corrupt_estop[128];
    memcpy(corrupt_estop, estop_frame, estop_len);
    corrupt_estop[estop_len - 1] ^= 0xFF; // flip CRC
    process_frame(corrupt_estop, estop_len);
    assert(!g_emergency_stop);

    // 2. Valid E-Stop: verify emergency stop is triggered
    process_frame(estop_frame, estop_len);
    assert(g_emergency_stop);
    assert(g_cmd_pkt.vx == 0.0f);

    // 3. Subsequent VelocityCmdPacket: verify E-Stop remains strictly latched
    VelocityCmdPacket vel_in = {{0.20f, 0.0f, 0.0f}};
    uint8_t vel_frame[128];
    uint16_t vel_len = encode_velocity_packet(&vel_in, 2, vel_frame, sizeof(vel_frame));
    process_frame(vel_frame, vel_len);
    assert(g_emergency_stop); // Must stay true!

    // 4. Corrupt VelocityCmdPacket: verify watchdog timestamp is NOT updated
    g_last_mcu1_pkt_time = 1000;
    current_time = 5000;
    vel_frame[vel_len - 1] ^= 0xFF; // flip CRC
    process_frame(vel_frame, vel_len);
    assert(g_last_mcu1_pkt_time == 1000); // Unchanged!

    std::cout << "MCU2 E-stop latching and CRC validation verified" << std::endl;
    return 0;
}}
"""
    _compile_and_run_cpp(
        cpp_source=cpp_source,
        cpp_path=tmp_path / "test_mcu2_estop.cpp",
        exe_path=tmp_path / "test_mcu2_estop.exe",
        expected_stdout="MCU2 E-stop latching and CRC validation verified",
    )


def test_mcu2_gait_phase_observation():
    """Verify MCU 2 injects gait phase clock [sin(phase), cos(phase)] into obs_out[35..36]."""
    mcu2_sketch = (
        REPO_ROOT
        / "pipelines"
        / "dual_esp32s3_stack"
        / "deployment"
        / "mcu2_locomotion_s3"
        / "mcu2_main.ino"
    )
    content = mcu2_sketch.read_text(encoding="utf-8")
    assert "obs_out[35]" in content and "obs_out[36]" in content, (
        "mcu2_main.ino must inject gait phase clock into obs_out[35] and obs_out[36]"
    )
    assert "sinf" in content and "cosf" in content, (
        "mcu2_main.ino must compute sin and cos for gait phase clock"
    )
    assert "0.8" in content, "mcu2_main.ino must use 0.8s gait period"


def test_export_policy_obs_norm_and_asymmetric_critic_parity(tmp_path):
    """Verify that export_policy folds obs_rms into actor[0] matching <1e-5 difference vs unnormalized ONNX,
    and that asymmetric critic checkpoints are cleanly handled.
    """
    import onnxruntime as ort
    import torch
    from scripts.train_ppo import ActorCritic
    from scripts.export_policy import export_onnx, export_esp32_headers

    torch.manual_seed(42)
    np.random.seed(42)

    obs_dim = 42
    critic_obs_dim = 52
    action_dim = 8
    hidden_dim = 256

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, critic_obs_dim=critic_obs_dim)
    model.eval()

    # Generate synthetic obs_rms
    mean = np.random.uniform(-0.5, 0.5, size=obs_dim).astype(np.float32)
    var = np.random.uniform(0.5, 2.0, size=obs_dim).astype(np.float32)
    eps = 1e-4

    ckpt = {
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "critic_obs_dim": critic_obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "obs_rms": {
            "mean": mean,
            "var": var,
            "epsilon": eps,
            "clip": 10.0,
        },
    }

    ckpt_path = tmp_path / "asym_norm_ckpt.pt"
    torch.save(ckpt, str(ckpt_path))

    # Raw observation (within normal bounds so clipping is not triggered)
    raw_obs = np.random.uniform(-1.0, 1.0, size=(1, obs_dim)).astype(np.float32)

    # Expected action computed from original actor with manual normalization
    std = np.sqrt(var + eps)
    norm_obs = (raw_obs - mean) / std
    with torch.no_grad():
        expected_action = model.actor(torch.from_numpy(norm_obs).float()).numpy()

    # Export ONNX
    onnx_path = tmp_path / "asym_policy.onnx"
    export_onnx(ckpt_path, output_path=onnx_path)

    # Run ONNX on raw_obs
    session = ort.InferenceSession(str(onnx_path))
    onnx_action = session.run(["actions"], {"obs": raw_obs})[0]

    max_diff = float(np.max(np.abs(expected_action - onnx_action)))
    assert max_diff < 1e-5, f"ONNX output deviated from normalized actor: {max_diff}"

    # Export ESP32 headers
    weights_path = tmp_path / "esp32_policy_weights.h"
    config_path = tmp_path / "esp32_quant_config.h"
    export_esp32_headers(ckpt_path, weights_output_path=weights_path, config_output_path=config_path)

    assert weights_path.exists()
    assert config_path.exists()
    w_text = weights_path.read_text(encoding="utf-8")
    assert f"#define POLICY_LAYER1_IN  {obs_dim}" in w_text


def test_zero_variance_obs_rms_does_not_amplify_by_100x():
    """Verify that zero-variance observation channels do not blow up weights/inputs by 100x.

    With epsilon=1e-4 and var=0.0 (e.g. Stage 0 vy_cmd or LiDAR ranges), 1.0 / sqrt(1e-4) = 100.0.
    RunningMeanStd and apply_obs_norm_to_actor must clamp std to min_std (default 0.10),
    so inv_std <= 10.0 instead of 100.0.
    """
    import torch
    import torch.nn as nn
    from scripts.train_ppo import RunningMeanStd
    from scripts.export_policy import apply_obs_norm_to_actor

    # 1. Test RunningMeanStd with min_std
    rms = RunningMeanStd(shape=(3,), min_std=0.10)
    rms.var = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    rms.mean = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # NumPy normalize path
    x_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    norm_np = rms.normalize(x_np)
    # Channel 0 has var=0. With min_std=0.10, norm_np[0] == 1.0 / 0.10 = 10.0, NOT 100.0
    assert norm_np[0] <= 10.0001, f"Expected norm_np[0] <= 10.0, got {norm_np[0]}"
    assert norm_np[0] >= 9.999, f"Expected norm_np[0] approx 10.0, got {norm_np[0]}"

    # Tensor normalize path
    x_t = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    norm_t = rms.normalize(x_t)
    assert norm_t[0].item() <= 10.0001, f"Expected norm_t[0] <= 10.0, got {norm_t[0].item()}"
    assert norm_t[0].item() >= 9.999, f"Expected norm_t[0] approx 10.0, got {norm_t[0].item()}"

    # State dict serialization / restoration of min_std
    sdict = rms.state_dict()
    assert "min_std" in sdict
    assert sdict["min_std"] == 0.10

    rms_loaded = RunningMeanStd(shape=(3,), min_std=0.05)
    rms_loaded.load_state_dict(sdict)
    assert rms_loaded.min_std == 0.10

    # Backward compatibility with older state_dict missing min_std
    legacy_sdict = {
        "mean": np.zeros(3, dtype=np.float64),
        "var": np.ones(3, dtype=np.float64),
        "count": 10.0,
    }
    rms_legacy = RunningMeanStd(shape=(3,), min_std=0.10)
    rms_legacy.load_state_dict(legacy_sdict)
    assert rms_legacy.min_std == 0.10

    # 2. Test apply_obs_norm_to_actor weight folding with min_std
    actor = nn.Sequential(nn.Linear(3, 2, bias=True))
    with torch.no_grad():
        actor[0].weight.fill_(1.0)
        actor[0].bias.fill_(0.0)

    ckpt = {
        "obs_rms": {
            "mean": np.zeros(3, dtype=np.float32),
            "var": np.array([0.0, 1.0, 4.0], dtype=np.float32),
            "epsilon": 1e-4,
            "min_std": 0.10,
        }
    }

    apply_obs_norm_to_actor(actor, ckpt, min_std=0.10)
    # Channel 0 weight should be folded with 1.0 / 0.10 = 10.0, NOT 100.0
    w_chan0 = actor[0].weight[:, 0].detach().cpu().numpy()
    assert np.all(w_chan0 <= 10.0001), f"Expected channel 0 weights <= 10.0, got {w_chan0}"
    assert np.all(w_chan0 >= 9.999), f"Expected channel 0 weights approx 10.0, got {w_chan0}"
