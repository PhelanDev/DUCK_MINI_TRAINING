"""Lightweight AI Vision & Visual Servoing Module for MicroDuck 8-DOF and CyberDuck V2 Pro.

Provides:
- TargetColor: Color presets for target detection (YELLOW_DUCK, RED_BALL, GREEN_CUP, BLUE_CUBE).
- ColorBlobTracker: Real-time OpenCV/NumPy color segmentation and centroid tracking.
- VisualServoingHeadController: Closed-loop proportional visual servoing controller for 2-DOF neck
  (head_yaw, head_pitch) with anti-jitter deadband, rate limiting, and joint boundary safety.
- Simulation and Decoupled Pi Zero 2 W / ESP32-S3 integration helpers.
"""

from enum import Enum
import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# Optional OpenCV import with pure-NumPy fallback
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class TargetColor(str, Enum):
    """Supported color target presets."""

    YELLOW = "yellow"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


# HSV Range definitions: (lower_hsv, upper_hsv) in OpenCV conventions (H: 0-179, S: 0-255, V: 0-255)
HSV_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    TargetColor.YELLOW.value: ((20, 100, 100), (35, 255, 255)),
    TargetColor.GREEN.value: ((36, 80, 80), (85, 255, 255)),
    TargetColor.BLUE.value: ((100, 100, 100), (130, 255, 255)),
    # Red wraps around 0/180; primary lower band:
    TargetColor.RED.value: ((0, 120, 100), (10, 255, 255)),
}
RED_SECONDARY_BAND: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
    (170, 120, 100),
    (179, 255, 255),
)


class ColorBlobTracker:
    """Fast, low-latency color segmentation and centroid detection for embedded vision."""

    def __init__(self, target_color: Union[str, TargetColor] = TargetColor.YELLOW, min_area: float = 30.0):
        self.target_color = TargetColor(target_color).value if isinstance(target_color, (str, TargetColor)) else "yellow"
        self.min_area = float(min_area)

    def track(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Detect color blob and return normalized image offsets.

        Args:
            image_rgb: RGB image array of shape (H, W, 3), uint8.

        Returns:
            Dictionary containing:
              - detected: bool
              - norm_x: float in [-1.0, 1.0] (0.0 = centered, >0 = right, <0 = left)
              - norm_y: float in [-1.0, 1.0] (0.0 = centered, >0 = down, <0 = up)
              - center_px: Tuple[int, int] (cx, cy)
              - radius_px: float
              - area_px: float
        """
        result: Dict[str, Any] = {
            "detected": False,
            "norm_x": 0.0,
            "norm_y": 0.0,
            "center_px": (0, 0),
            "radius_px": 0.0,
            "area_px": 0.0,
        }

        if image_rgb is None or image_rgb.size == 0:
            return result

        h, w = image_rgb.shape[:2]
        if h == 0 or w == 0:
            return result

        if _HAS_CV2:
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            lower, upper = HSV_RANGES.get(self.target_color, HSV_RANGES[TargetColor.YELLOW.value])
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

            if self.target_color == TargetColor.RED.value:
                # Include upper red band
                mask2 = cv2.inRange(
                    hsv,
                    np.array(RED_SECONDARY_BAND[0], dtype=np.uint8),
                    np.array(RED_SECONDARY_BAND[1], dtype=np.uint8),
                )
                mask = cv2.bitwise_or(mask, mask2)

            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            # Find largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area >= self.min_area:
                    ((cx, cy), radius) = cv2.minEnclosingCircle(largest)
                    # Convert to normalized coordinates in [-1.0, 1.0]
                    norm_x = (cx - (w / 2.0)) / (w / 2.0)
                    norm_y = (cy - (h / 2.0)) / (h / 2.0)

                    result["detected"] = True
                    result["norm_x"] = float(np.clip(norm_x, -1.0, 1.0))
                    result["norm_y"] = float(np.clip(norm_y, -1.0, 1.0))
                    result["center_px"] = (int(cx), int(cy))
                    result["radius_px"] = float(radius)
                    result["area_px"] = float(area)
        else:
            # Pure NumPy fallback for environments without OpenCV
            r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]
            if self.target_color == TargetColor.YELLOW.value:
                mask = (r > 150) & (g > 150) & (b < 100)
            elif self.target_color == TargetColor.RED.value:
                mask = (r > 160) & (g < 90) & (b < 90)
            elif self.target_color == TargetColor.GREEN.value:
                mask = (g > 150) & (r < 100) & (b < 100)
            else:
                mask = (b > 150) & (r < 100) & (g < 100)

            ys, xs = np.where(mask)
            if len(xs) >= self.min_area:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                norm_x = (cx - (w / 2.0)) / (w / 2.0)
                norm_y = (cy - (h / 2.0)) / (h / 2.0)
                result["detected"] = True
                result["norm_x"] = float(np.clip(norm_x, -1.0, 1.0))
                result["norm_y"] = float(np.clip(norm_y, -1.0, 1.0))
                result["center_px"] = (int(cx), int(cy))
                result["area_px"] = float(len(xs))
                result["radius_px"] = float(np.sqrt(len(xs) / np.pi))

        return result


class VisualServoingHeadController:
    """Closed-loop visual servoing controller for 2-DOF neck gimbal (head_yaw, head_pitch).

    Enforces:
    - Deadband: suppresses small jitter when the target is already near the center of the FOV.
    - Rate limiter: limits maximum angular step (|delta q| <= max_rate) to safeguard MG996R servos.
    - Physical angle clamping: head_yaw in [-1.57, 1.57], head_pitch in [-0.52, 0.78].
    """

    def __init__(
        self,
        kp_yaw: float = 0.60,
        kp_pitch: float = 0.50,
        deadband: float = 0.06,
        max_rate: float = 0.08,
        yaw_limits: Tuple[float, float] = (-1.57, 1.57),
        pitch_limits: Tuple[float, float] = (-0.52, 0.78),
    ):
        self.kp_yaw = float(kp_yaw)
        self.kp_pitch = float(kp_pitch)
        self.deadband = float(deadband)
        self.max_rate = float(max_rate)
        self.yaw_limits = yaw_limits
        self.pitch_limits = pitch_limits

        self.current_yaw: float = 0.0
        self.current_pitch: float = 0.0

    def reset(self, yaw: float = 0.0, pitch: float = 0.0) -> None:
        """Reset internal neck gimbal angles."""
        self.current_yaw = float(yaw)
        self.current_pitch = float(pitch)

    def update(
        self,
        norm_x: float,
        norm_y: float,
        detected: bool,
        dt: float = 0.033,
    ) -> Tuple[float, float]:
        """Update closed-loop visual servoing angles based on target offset.

        Args:
            norm_x: Horizontal target offset in [-1.0, 1.0] (>0 means target is to the right).
            norm_y: Vertical target offset in [-1.0, 1.0] (>0 means target is below center).
            detected: Whether target is currently in view.
            dt: Control update period in seconds.

        Returns:
            Tuple[float, float]: (target_head_yaw, target_head_pitch) in radians.
        """
        if not detected:
            # Smoothly return to center / neutral position when target is lost
            delta_yaw = np.clip(0.0 - self.current_yaw, -self.max_rate, self.max_rate)
            delta_pitch = np.clip(0.0 - self.current_pitch, -self.max_rate, self.max_rate)
            self.current_yaw += delta_yaw * 0.5
            self.current_pitch += delta_pitch * 0.5
            return self.current_yaw, self.current_pitch

        # Apply deadband filter to eliminate jitter
        eff_x = 0.0 if abs(norm_x) < self.deadband else (norm_x - np.sign(norm_x) * self.deadband)
        eff_y = 0.0 if abs(norm_y) < self.deadband else (norm_y - np.sign(norm_y) * self.deadband)

        # Proportional correction:
        # If target is to the right (eff_x > 0), head needs to rotate right (yaw decreases / increases depending on frame)
        # Standard convention: yaw increases counter-clockwise (left). So if target is to the right (x > 0), yaw decreases.
        step_yaw = -self.kp_yaw * eff_x * (dt / 0.033)
        # If target is below center (eff_y > 0), head needs to pitch downwards (pitch increases).
        step_pitch = self.kp_pitch * eff_y * (dt / 0.033)

        # Slew rate limit to protect MG996R gears
        step_yaw = float(np.clip(step_yaw, -self.max_rate, self.max_rate))
        step_pitch = float(np.clip(step_pitch, -self.max_rate, self.max_rate))

        self.current_yaw = float(
            np.clip(self.current_yaw + step_yaw, self.yaw_limits[0], self.yaw_limits[1])
        )
        self.current_pitch = float(
            np.clip(self.current_pitch + step_pitch, self.pitch_limits[0], self.pitch_limits[1])
        )

        return self.current_yaw, self.current_pitch


def render_head_camera(
    model: Any,
    data: Any,
    camera_name: str = "head_camera",
    width: int = 320,
    height: int = 240,
    renderer: Optional[Any] = None,
) -> np.ndarray:
    """Render an RGB frame from the robot's onboard head camera in MuJoCo."""
    import mujoco

    if renderer is None:
        renderer = mujoco.Renderer(model, height=height, width=width)

    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    return rgb


def build_mcu1_vision_packet(head_yaw: float, head_pitch: float) -> bytes:
    """Construct binary command packet for UART transmission to ESP32-S3 MCU1 or Pi Zero 2 W."""
    import struct

    # Packet: [SYNC_BYTE_1 (0xAA), SYNC_BYTE_2 (0x55), CMD_HEAD (0x10), yaw_deg_int16, pitch_deg_int16, checksum]
    yaw_deg = int(round(math.degrees(head_yaw) * 10))
    pitch_deg = int(round(math.degrees(head_pitch) * 10))
    chk = (yaw_deg + pitch_deg) & 0xFF
    return struct.pack("<BBBhhB", 0xAA, 0x55, 0x10, yaw_deg, pitch_deg, chk)
