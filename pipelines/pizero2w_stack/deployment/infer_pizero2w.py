"""Raspberry Pi Zero 2 W Real-Time Inference and Actuation Runtime for MicroDuck 8-DOF.

Thin parameterized wrapper delegating shared inference and hardware control to pipelines.rpi_infer_core.
"""

import sys
from pathlib import Path

# Ensure duckV1 root is on path for relative imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.rpi_infer_core import (
    angle_to_pwm,
    pack_rpi_observation as pack_pizero2w_observation,
    MockPCA9685,
    HardwarePCA9685,
    MPU6050FilterReader,
    BaseRPiInferenceEngine,
    run_rpi_cli,
)

DEFAULT_MODEL_PATH = (
    REPO_ROOT / "pipelines" / "pizero2w_stack" / "deployment" / "microduck_pizero2w_policy.onnx"
)


class PiZero2WInferenceEngine(BaseRPiInferenceEngine):
    """Raspberry Pi Zero 2 W Real-Time Locomotion Engine for MicroDuck 8-DOF."""

    def __init__(
        self,
        model_path: str = None,
        i2c_bus: int = 1,
        mock: bool = False,
        freq: float = 50.0,
    ):
        super().__init__(
            model_path=model_path,
            i2c_bus=i2c_bus,
            mock=mock,
            freq=freq,
            engine_name="PiZero2WInferenceEngine",
        )
        # Explicit contract hooks preserving: prev_q, read_foot_contact, read_lidar_ranges
        self.prev_q = self.last_targets.copy()

    def read_foot_contact(self):
        """Read 4-channel foot contact states [Left, Right, Padded, Padded]."""
        return super().read_foot_contact()

    def read_lidar_ranges(self):
        """Read 5-ray ToF LiDAR distance sensors [-30, -15, 0, +15, +30 deg]."""
        return super().read_lidar_ranges()


def main():
    run_rpi_cli(
        PiZero2WInferenceEngine,
        "Raspberry Pi Zero 2 W Locomotion Inference Engine for MicroDuck 8-DOF",
        default_model=str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None,
    )


if __name__ == "__main__":
    main()
