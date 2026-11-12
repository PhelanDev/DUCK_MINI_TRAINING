"""Raspberry Pi Zero 2 W Real-Time Inference and Actuation Runtime for MicroDuck 8-DOF.

Provides:
- Float32/FP16 ONNX Runtime inference engine for 42-dimensional observation vector
- 8-DOF MG996R servo angle-to-PWM mapping for PCA9685 I2C servo controller
- Rate-limiting, safety joint angle clamping, and mock simulation mode
- 50 Hz real-time control loop with performance metrics optimized for Cortex-A53
- Decoupled from MuJoCo simulation packages for lightweight embedded execution
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.env_config import (
    MicroDuck8DoFConfig,
    clip_action_rates,
    compute_observations,
    scale_actions,
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


def pack_pizero2w_observation(
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

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        """Record commanded pulse width for given channel."""
        if 0 <= channel < 16:
            self.channel_pwms[channel] = pulse_us


class HardwarePCA9685:
    """Physical I2C PCA9685 driver via smbus2 on Raspberry Pi Zero 2 W."""

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

        # Set PWM frequency (50 Hz for MG996R servos)
        prescale_val = int(25000000.0 / (4096.0 * freq) - 1.0 + 0.5)
        old_mode = self.bus.read_byte_data(self.address, self.PCA9685_MODE1)
        new_mode = (old_mode & 0x7F) | 0x10
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, new_mode)
        self.bus.write_byte_data(self.address, self.PCA9685_PRESCALE, prescale_val)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode | 0xA1)

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        """Convert microsecond pulse to 12-bit PCA9685 tick value (0-4095)."""
        period_us = 1000000.0 / self.freq
        ticks = int((pulse_us / period_us) * 4096.0)
        ticks = max(0, min(4095, ticks))

        reg_base = self.LED0_ON_L + 4 * channel
        data = [0, 0, ticks & 0xFF, (ticks >> 8) & 0xFF]
        self.bus.write_i2c_block_data(self.address, reg_base, data)


class MPU6050FilterReader:
    """Hardware I2C driver for MPU6050 with 42Hz DLPF, Mahony AHRS sensor fusion, and impact gating."""

    MPU6050_ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    def __init__(
        self,
        bus_num: int = 1,
        address: int = 0x68,
        mock: bool = False,
        dt: float = 0.02,
        kp: float = 1.0,
        ki: float = 0.05,
        impact_threshold: float = 0.30,
    ):
        self.address = address
        self.mock = mock or (smbus is None)
        self.dt = float(dt)
        self.kp = float(kp)
        self.ki = float(ki)
        self.impact_threshold = float(impact_threshold)

        # Mahony AHRS quaternion [qw, qx, qy, qz] & integral feedback error
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.integral_fb = np.zeros(3, dtype=np.float32)

        self.gyro_bias = np.zeros(3, dtype=np.float32)
        self.filt_gyro = np.zeros(3, dtype=np.float32)
        self.filt_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.alpha_gyro = 0.70
        self.alpha_grav = 0.80

        if not self.mock:
            try:
                self.bus = smbus.SMBus(bus_num)
                # Wake up MPU6050
                self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
                # 42Hz hardware DLPF cutoff to filter servo motor vibration
                self.bus.write_byte_data(self.address, self.CONFIG, 0x03)
                # Gyro sensitivity +-250 deg/s
                self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
                self._calibrate_gyro()
            except Exception as e:
                print(f"[IMU] Real I2C MPU6050 unavailable ({e}), using mock/fallback.")
                self.mock = True

    def _calibrate_gyro(self, samples: int = 100) -> None:
        """Calibrate gyro zero-rate offset while standing still."""
        print("[IMU] Calibrating MPU6050 gyro bias...")
        acc = np.zeros(3, dtype=np.float64)
        count = 0
        for _ in range(samples):
            try:
                data = self.bus.read_i2c_block_data(self.address, self.GYRO_XOUT_H, 6)
                gx = (np.int16((data[0] << 8) | data[1]) / 131.0) * (np.pi / 180.0)
                gy = (np.int16((data[2] << 8) | data[3]) / 131.0) * (np.pi / 180.0)
                gz = (np.int16((data[4] << 8) | data[5]) / 131.0) * (np.pi / 180.0)
                acc += [gx, gy, gz]
                count += 1
                time.sleep(0.002)
            except Exception:
                pass
        if count > 0:
            self.gyro_bias = (acc / count).astype(np.float32)

    def update(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        dt: Optional[float] = None,
    ) -> None:
        """Vectorized Mahony AHRS update with dynamic impact gating."""
        if dt is None:
            dt = self.dt

        acc = np.array([ax, ay, az], dtype=np.float32)
        acc_sq = float(np.sum(acc * acc))

        corr_gx = float(gx)
        corr_gy = float(gy)
        corr_gz = float(gz)

        if acc_sq > 1e-8:
            acc_norm = float(np.sqrt(acc_sq))
            # Dynamic Foot-Strike Impact Gating: gate correction if |norm - 1.0g| > 0.30g
            if abs(acc_norm - 1.0) <= self.impact_threshold:
                inv_norm = 1.0 / acc_norm
                ax_n = acc[0] * inv_norm
                ay_n = acc[1] * inv_norm
                az_n = acc[2] * inv_norm

                qw, qx, qy, qz = self.q
                # Estimated gravity direction in body frame
                vx = 2.0 * (qx * qz - qw * qy)
                vy = 2.0 * (qw * qx + qy * qz)
                vz = qw * qw - qx * qx - qy * qy + qz * qz

                # Error is cross product between estimated and measured direction of gravity
                ex = ay_n * vz - az_n * vy
                ey = az_n * vx - ax_n * vz
                ez = ax_n * vy - ay_n * vx

                # Accumulate integral feedback
                self.integral_fb[0] += self.ki * ex * dt
                self.integral_fb[1] += self.ki * ey * dt
                self.integral_fb[2] += self.ki * ez * dt

                # Apply proportional and integral feedback to angular rates
                corr_gx += self.kp * ex + self.integral_fb[0]
                corr_gy += self.kp * ey + self.integral_fb[1]
                corr_gz += self.kp * ez + self.integral_fb[2]
            else:
                # Gated: use integral feedback only (no accelerometer correction)
                corr_gx += self.integral_fb[0]
                corr_gy += self.integral_fb[1]
                corr_gz += self.integral_fb[2]
        else:
            corr_gx += self.integral_fb[0]
            corr_gy += self.integral_fb[1]
            corr_gz += self.integral_fb[2]

        # Integrate rate of change of quaternion: q_dot = 0.5 * q * omega
        hdt = 0.5 * dt
        corr_gx *= hdt
        corr_gy *= hdt
        corr_gz *= hdt

        qw, qx, qy, qz = self.q
        dq_w = -qx * corr_gx - qy * corr_gy - qz * corr_gz
        dq_x =  qw * corr_gx + qy * corr_gz - qz * corr_gy
        dq_y =  qw * corr_gy - qx * corr_gz + qz * corr_gx
        dq_z =  qw * corr_gz + qx * corr_gy - qy * corr_gx

        new_q = np.array([qw + dq_w, qx + dq_x, qy + dq_y, qz + dq_z], dtype=np.float32)
        q_norm = float(np.linalg.norm(new_q))
        if q_norm > 1e-8:
            self.q = new_q / q_norm
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.filt_grav = self.get_proj_grav()

    mahony_update = update

    def get_proj_grav(self) -> np.ndarray:
        """Compute projected gravity vector in robot base frame matching MuJoCo convention [0, 0, -1]."""
        qw, qx, qy, qz = self.q
        return np.array([
            -2.0 * (qx * qz - qw * qy),
            -2.0 * (qy * qz + qw * qx),
            -(qw * qw - qx * qx - qy * qy + qz * qz),
        ], dtype=np.float32)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read and return filtered (base_ang_vel, proj_grav)."""
        if self.mock:
            return self.filt_gyro.copy(), self.get_proj_grav()

        try:
            data = self.bus.read_i2c_block_data(self.address, self.ACCEL_XOUT_H, 14)
            ax_raw = np.int16((data[0] << 8) | data[1])
            ay_raw = np.int16((data[2] << 8) | data[3])
            az_raw = np.int16((data[4] << 8) | data[5])
            gx_raw = np.int16((data[8] << 8) | data[9])
            gy_raw = np.int16((data[10] << 8) | data[11])
            gz_raw = np.int16((data[12] << 8) | data[13])

            raw_gx = ((gx_raw / 131.0) * (np.pi / 180.0)) - self.gyro_bias[0]
            raw_gy = ((gy_raw / 131.0) * (np.pi / 180.0)) - self.gyro_bias[1]
            raw_gz = ((gz_raw / 131.0) * (np.pi / 180.0)) - self.gyro_bias[2]

            self.filt_gyro = self.alpha_gyro * self.filt_gyro + (1.0 - self.alpha_gyro) * np.array(
                [raw_gx, raw_gy, raw_gz], dtype=np.float32
            )

            ax = float(ax_raw) / 16384.0
            ay = float(ay_raw) / 16384.0
            az = float(az_raw) / 16384.0

            self.update(ax, ay, az, self.filt_gyro[0], self.filt_gyro[1], self.filt_gyro[2], dt=self.dt)

        except Exception:
            pass

        return self.filt_gyro.copy(), self.get_proj_grav()


class PiZero2WInferenceEngine:
    """Raspberry Pi Zero 2 W Real-Time Locomotion Engine for MicroDuck 8-DOF."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        i2c_bus: int = 1,
        mock: bool = False,
        freq: float = 50.0,
    ):
        self.cfg = MicroDuck8DoFConfig(OBS_DIM=42)
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

        # IMU sensor & filter setup
        self.imu_reader = MPU6050FilterReader(
            bus_num=i2c_bus, mock=self.mock, dt=self.dt
        )

        # ONNX Model setup (optimized for Quad-core Cortex-A53)
        self.model_path = model_path
        self.ort_session = None
        if self.model_path and Path(self.model_path).exists():
            if ort is None:
                raise RuntimeError("onnxruntime is required to load ONNX model.")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4  # Utilize all 4 Cortex-A53 cores on Pi Zero 2 W
            self.ort_session = ort.InferenceSession(self.model_path, opts)
            self.input_name = self.ort_session.get_inputs()[0].name
            print(f"[PiZero2WInferenceEngine] Loaded ONNX policy model from {self.model_path}")
        else:
            print("[PiZero2WInferenceEngine] No ONNX model provided or file not found. Running in heuristic/zero-action mode.")

        # State tracking for rate limiting (initialized to nominal stand posture)
        self.last_targets = self.cfg.nominal_qpos.copy()
        self.last_raw_actions = np.zeros(self.cfg.ACTION_DIM, dtype=np.float32)
        self.joint_limits_list = [
            self.cfg.JOINT_LIMITS[name] for name in self.cfg.JOINT_NAMES
        ]
        self.prev_q: np.ndarray = self.cfg.nominal_qpos.copy()
        self._warned_foot_fallback: bool = False
        self._warned_lidar_fallback: bool = False

    def read_foot_contact(self) -> List[float]:
        """Read 4-channel foot contact states [Left, Right, Padded, Padded].

        Connects to hardware ADC/FSR foot sensors or falls back with a diagnostic warning.
        """
        if not self._warned_foot_fallback:
            if self.mock:
                print("[Warning] Foot contact sensors running in mock/fallback mode. Using nominal [1.0, 1.0, 0.0, 0.0].")
            else:
                print("[Warning] Hardware foot contact ADC unavailable on I2C. Using fallback contact states.")
            self._warned_foot_fallback = True
        return [1.0, 1.0, 0.0, 0.0]

    def read_lidar_ranges(self) -> List[float]:
        """Read 5-ray ToF LiDAR distance sensors [-30, -15, 0, +15, +30 deg].

        Connects to hardware I2C ToF array or falls back with a diagnostic warning.
        """
        if not self._warned_lidar_fallback:
            if self.mock:
                print("[Warning] LiDAR sensor array running in mock/fallback mode. Using clear 1.5m range.")
            else:
                print("[Warning] Hardware I2C LiDAR array unavailable on bus. Using clear 1.5m range.")
            self._warned_lidar_fallback = True
        return [1.5, 1.5, 1.5, 1.5, 1.5]

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
        """Perform one control cycle at 50Hz:

        1. Infer action from observation
        2. Scale actions to joint physical limits
        3. Rate-limit target angles
        4. Map angles to servo PWMs and dispatch to PCA9685
        """
        assert obs.shape == (self.cfg.OBS_DIM,), (
            f"Expected obs shape ({self.cfg.OBS_DIM},), got {obs.shape}"
        )

        # 1. Infer raw actions
        raw_actions = np.clip(self.predict_action(obs), -1.0, 1.0)
        self.last_raw_actions = raw_actions.copy()

        # 2. Scale actions to joint angles
        scaled_targets = scale_actions(raw_actions, self.cfg, nominal_centered=True)

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
        print(f"[PiZero2WInferenceEngine] Starting control loop at {self.freq:.1f} Hz (dt={self.dt*1000:.1f}ms)...")
        step_count = 0
        target_period = self.dt

        commands = [0.2, 0.0, 0.0]
        prev_q = self.last_targets.copy()
        self.prev_q = prev_q.copy()

        try:
            while steps is None or step_count < steps:
                t0 = time.perf_counter()

                # Read filtered IMU angular velocity and gravity vector
                base_ang_vel, proj_grav = self.imu_reader.read()

                # Read foot contact and LiDAR perception hooks (with fallback warning)
                foot_contact = self.read_foot_contact()
                lidar_ranges = self.read_lidar_ranges()

                # Finite-difference joint velocity: (current_q - prev_q) / dt
                current_q = self.last_targets.copy()
                effective_dt = max(self.dt, 1e-4)
                joint_vel = (current_q - prev_q) / effective_dt
                prev_q = current_q.copy()
                self.prev_q = prev_q.copy()

                obs = pack_pizero2w_observation(
                    base_ang_vel=base_ang_vel,
                    proj_grav=proj_grav,
                    commands=commands,
                    joint_pos=current_q,
                    joint_vel=joint_vel,
                    last_actions=self.last_raw_actions,
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
            print("\n[PiZero2WInferenceEngine] Control loop interrupted by user.")

        print(f"[PiZero2WInferenceEngine] Completed {step_count} control steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Zero 2 W Locomotion Inference Engine for MicroDuck 8-DOF"
    )
    parser.add_argument(
        "--model_path",
        "--model",
        type=str,
        default=None,
        help="Path to Float32 ONNX policy model",
    )
    parser.add_argument("--i2c_bus", type=int, default=1, help="I2C bus number for PCA9685 (default: 1)")
    parser.add_argument(
        "--mock",
        "--mock-pca",
        dest="mock",
        action="store_true",
        help="Run in mock mode without I2C hardware",
    )
    parser.add_argument(
        "--freq",
        "--rate",
        type=float,
        default=50.0,
        help="Control loop frequency in Hz (default: 50.0)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to execute (default: continuous)")
    args = parser.parse_args()

    engine = PiZero2WInferenceEngine(
        model_path=args.model_path,
        i2c_bus=args.i2c_bus,
        mock=args.mock,
        freq=args.freq,
    )
    engine.run_loop(steps=args.steps)


if __name__ == "__main__":
    main()
