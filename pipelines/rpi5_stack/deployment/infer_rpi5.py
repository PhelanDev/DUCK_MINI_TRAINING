"""Raspberry Pi 5 Real-Time Inference and Actuation Runtime for MicroDuck 8-DOF.

Provides:
- Float32 ONNX Runtime inference engine for 42-dimensional observation vector
- 8-DOF MG996R servo angle-to-PWM mapping for PCA9685 I2C servo controller
- Rate-limiting, safety joint angle clamping, and mock simulation mode
- 50 Hz real-time control loop with performance metrics
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure duckV1 root is on path for relative imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.rpi5_stack.training.env_config import (
    MicroDuck8DoFConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
)

# Optional hardware libraries
try:
    import smbus2 as smbus  # type: ignore
except ImportError:
    try:
        import smbus  # type: ignore
    except ImportError:
        smbus = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def angle_to_pwm(
    angle_rad: float,
    min_rad: float = -1.57,
    max_rad: float = 1.57,
    min_pulse_us: int = 500,
    max_pulse_us: int = 2500,
    neutral_pulse_us: int = 1500,
) -> int:
    """Map joint angle in radians to servo PWM pulse width in microseconds (us).

    Neutral angle 0.0 rad maps to neutral_pulse_us (1500us).
    Servo sensitivity: 2000.0 us span over pi radians (-pi/2 to +pi/2, ~636.62 us/rad).
    Clamps angle safely within [min_rad, max_rad] and output pulse to [min_pulse_us, max_pulse_us].
    Prevents knee jamming on asymmetric joint limits (e.g. [0.0, 1.48] rad).
    """
    clamped_angle = float(np.clip(angle_rad, min_rad, max_rad))
    pulse = neutral_pulse_us + clamped_angle * (2000.0 / np.pi)
    return int(np.clip(np.round(pulse), min_pulse_us, max_pulse_us))


def pack_rpi5_observation(
    base_ang_vel: Union[List[float], np.ndarray],
    proj_grav: Union[List[float], np.ndarray],
    commands: Union[List[float], np.ndarray],
    joint_pos: Union[List[float], np.ndarray],
    joint_vel: Union[List[float], np.ndarray],
    last_actions: Union[List[float], np.ndarray],
    foot_contact: Union[List[float], np.ndarray],
    lidar_ranges: Union[List[float], np.ndarray],
) -> np.ndarray:
    """Pack raw sensory signals into standard 42-dimensional observation vector."""
    return compute_observations(
        base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
        proj_grav=np.asarray(proj_grav, dtype=np.float32),
        commands=np.asarray(commands, dtype=np.float32),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32),
        last_actions=np.asarray(last_actions, dtype=np.float32),
        foot_contact=np.asarray(foot_contact, dtype=np.float32),
        lidar_ranges=np.asarray(lidar_ranges, dtype=np.float32),
    )


class MockPCA9685:
    """Mock PCA9685 driver for offline testing and desktop simulation."""

    def __init__(self, bus: int = 1, address: int = 0x40):
        self.bus = bus
        self.address = address
        self.channel_pwms = [1500] * 16
        print(f"[MockPCA9685] Initialized mock servo controller on I2C bus {bus}, address {hex(address)}")

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        """Record commanded pulse width for given channel."""
        if 0 <= channel < 16:
            self.channel_pwms[channel] = pulse_us


class HardwarePCA9685:
    """Physical I2C PCA9685 driver via smbus2."""

    PCA9685_MODE1 = 0x00
    PCA9685_PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus_num: int = 1, address: int = 0x40, freq: int = 50):
        self.address = address
        self.freq = freq
        if smbus is None:
            raise RuntimeError("smbus2/smbus module not found. Run with --mock on desktop systems.")
        self.bus = smbus.SMBus(bus_num)

        # Reset PCA9685
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, 0x00)
        time.sleep(0.01)

        # Set PWM frequency (50 Hz for analog/digital hobby servos)
        prescale_val = int(25000000.0 / (4096.0 * freq) - 1.0 + 0.5)
        old_mode = self.bus.read_byte_data(self.address, self.PCA9685_MODE1)
        new_mode = (old_mode & 0x7F) | 0x10  # Sleep mode to set prescale
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, new_mode)
        self.bus.write_byte_data(self.address, self.PCA9685_PRESCALE, prescale_val)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode | 0xA1)  # Auto-increment

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        """Convert microsecond pulse to 12-bit PCA9685 tick value (0-4095)."""
        # Period = 1000000us / freq = 20000us at 50Hz
        period_us = 1000000.0 / self.freq
        ticks = int((pulse_us / period_us) * 4096.0)
        ticks = max(0, min(4095, ticks))

        reg_base = self.LED0_ON_L + 4 * channel
        # ON tick = 0, OFF tick = ticks
        data = [0, 0, ticks & 0xFF, (ticks >> 8) & 0xFF]
        self.bus.write_i2c_block_data(self.address, reg_base, data)


class RPi5InferenceEngine:
    """Raspberry Pi 5 Real-Time Locomotion Engine for MicroDuck 8-DOF."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        i2c_bus: int = 1,
        mock: bool = False,
        freq: float = 50.0,
    ):
        self.cfg = MicroDuck8DoFConfig()
        self.freq = float(freq)
        self.dt = 1.0 / self.freq
        self.mock = mock or (smbus is None)

        # Servo controller setup
        if self.mock:
            self.servo_controller = MockPCA9685(bus=i2c_bus)
        else:
            try:
                self.servo_controller = HardwarePCA9685(bus_num=i2c_bus, freq=int(self.freq))
            except Exception as e:
                print(f"[Warning] Failed to initialize hardware I2C ({e}). Falling back to Mock PCA9685.")
                self.mock = True
                self.servo_controller = MockPCA9685(bus=i2c_bus)

        # ONNX Model setup
        self.model_path = model_path
        self.ort_session = None
        if self.model_path and Path(self.model_path).exists():
            if ort is None:
                raise RuntimeError("onnxruntime is required to load ONNX model.")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4  # Utilize all 4 Cortex-A76 cores on Pi 5
            self.ort_session = ort.InferenceSession(self.model_path, opts)
            self.input_name = self.ort_session.get_inputs()[0].name
            print(f"[RPi5InferenceEngine] Loaded ONNX policy model from {self.model_path}")
        else:
            print("[RPi5InferenceEngine] No ONNX model provided or file not found. Running in heuristic/zero-action mode.")

        # State tracking for rate limiting
        self.last_targets = np.zeros(self.cfg.ACTION_DIM, dtype=np.float32)
        self.joint_limits_list = [
            self.cfg.JOINT_LIMITS[name] for name in self.cfg.JOINT_NAMES
        ]

    def predict_action(self, obs: np.ndarray) -> np.ndarray:
        """Run policy model inference to produce normalized actions [-1, 1]."""
        if self.ort_session is not None:
            input_tensor = obs.reshape(1, self.cfg.OBS_DIM).astype(np.float32)
            outputs = self.ort_session.run(None, {self.input_name: input_tensor})
            return np.asarray(outputs[0], dtype=np.float32).ravel()
        else:
            # Standstill / neutral policy output
            return np.zeros(self.cfg.ACTION_DIM, dtype=np.float32)

    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one control cycle:

        1. Infer action from observation
        2. Scale actions to joint physical limits
        3. Rate-limit target angles
        4. Map angles to servo PWMs and dispatch to PCA9685
        """
        assert obs.shape == (self.cfg.OBS_DIM,), (
            f"Expected obs shape ({self.cfg.OBS_DIM},), got {obs.shape}"
        )

        # 1. Infer raw actions
        raw_actions = self.predict_action(obs)

        # 2. Scale actions to joint angles
        scaled_targets = scale_actions(raw_actions, self.cfg)

        # 3. Rate-limit targets to protect MG996R gears
        clipped_targets = clip_action_rates(
            current_targets=scaled_targets,
            last_targets=self.last_targets,
            dt=self.dt,
            max_velocity=self.cfg.MAX_VELOCITY,
        )
        self.last_targets = clipped_targets.copy()

        # 4. Map to PWM signals
        pwms = np.zeros(self.cfg.ACTION_DIM, dtype=np.int32)
        for i, (angle, limits) in enumerate(zip(clipped_targets, self.joint_limits_list)):
            pwm = angle_to_pwm(
                angle_rad=float(angle),
                min_rad=limits[0],
                max_rad=limits[1],
                min_pulse_us=500,
                max_pulse_us=2500,
            )
            pwms[i] = pwm
            self.servo_controller.set_pwm_us(channel=i, pulse_us=pwm)

        return clipped_targets, pwms

    def run_loop(self, steps: Optional[int] = None) -> None:
        """Run real-time control loop at specified frequency."""
        print(f"[RPi5InferenceEngine] Starting control loop at {self.freq:.1f} Hz (dt={self.dt*1000:.1f}ms)...")
        step_count = 0
        target_period = self.dt

        # Dummy perception & state for standalone testing
        base_ang_vel = [0.0, 0.0, 0.0]
        proj_grav = [0.0, 0.0, -1.0]
        commands = [0.2, 0.0, 0.0]  # Walk forward 0.2 m/s
        foot_contact = [1.0, 1.0, 1.0, 1.0]
        lidar_ranges = [1.5, 1.5, 1.5, 1.5, 1.5]

        try:
            while steps is None or step_count < steps:
                t0 = time.perf_counter()

                # Generate observation (using previous targets as simulated joint positions)
                obs = pack_rpi5_observation(
                    base_ang_vel=base_ang_vel,
                    proj_grav=proj_grav,
                    commands=commands,
                    joint_pos=self.last_targets,
                    joint_vel=np.zeros(8, dtype=np.float32),
                    last_actions=self.last_targets,
                    foot_contact=foot_contact,
                    lidar_ranges=lidar_ranges,
                )

                targets, pwms = self.step(obs)
                step_count += 1

                elapsed = time.perf_counter() - t0
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step_count % max(1, int(self.freq)) == 0 or (steps and step_count == steps):
                    actual_hz = 1.0 / max(1e-6, (time.perf_counter() - t0))
                    print(
                        f"[Step {step_count}] Hz={actual_hz:.1f} | "
                        f"PWMs: {pwms.tolist()} | Targets: {np.round(targets, 2).tolist()}"
                    )

        except KeyboardInterrupt:
            print("\n[RPi5InferenceEngine] Control loop interrupted by user.")

        print(f"[RPi5InferenceEngine] Completed {step_count} control steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 Locomotion Inference Engine for MicroDuck 8-DOF"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to Float32 ONNX policy model")
    parser.add_argument("--i2c_bus", type=int, default=1, help="I2C bus number for PCA9685 (default: 1)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without I2C hardware")
    parser.add_argument("--freq", type=float, default=50.0, help="Control loop frequency in Hz (default: 50.0)")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to execute (default: continuous)")
    args = parser.parse_args()

    engine = RPi5InferenceEngine(
        model_path=args.model_path,
        i2c_bus=args.i2c_bus,
        mock=args.mock,
        freq=args.freq,
    )
    engine.run_loop(steps=args.steps)


if __name__ == "__main__":
    main()

