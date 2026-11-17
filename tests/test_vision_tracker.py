"""Unit tests for AI Vision Tracker & Visual Servoing (training/vision_tracker.py)."""

import math
import os
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repo root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.vision_tracker import (
    TargetColor,
    ColorBlobTracker,
    VisualServoingHeadController,
    render_head_camera,
    build_mcu1_vision_packet,
)


def test_target_color_enum():
    """Verify supported color target names."""
    assert TargetColor.YELLOW.value == "yellow"
    assert TargetColor.RED.value == "red"
    assert TargetColor.GREEN.value == "green"
    assert TargetColor.BLUE.value == "blue"


def test_color_blob_tracker_empty():
    """Verify tracker gracefully handles empty or black images."""
    tracker = ColorBlobTracker(TargetColor.YELLOW)
    empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = tracker.track(empty_img)
    assert not res["detected"]
    assert res["norm_x"] == 0.0
    assert res["norm_y"] == 0.0


def test_color_blob_tracker_synthetic_yellow_target():
    """Verify tracking of a yellow target placed at top-right of the frame."""
    tracker = ColorBlobTracker(TargetColor.YELLOW)
    img = np.zeros((240, 320, 3), dtype=np.uint8)  # Black background

    # Draw yellow circle (RGB: 255, 230, 0) centered at (240, 60), radius = 25
    y, x = np.ogrid[:240, :320]
    mask = ((x - 240) ** 2 + (y - 60) ** 2) <= 25 ** 2
    img[mask] = [255, 230, 0]

    res = tracker.track(img)
    assert res["detected"], "Yellow target should be detected"
    assert res["norm_x"] > 0.35, f"Target should be to the right of center: {res['norm_x']}"
    assert res["norm_y"] < -0.35, f"Target should be above center: {res['norm_y']}"
    assert res["area_px"] > 1000


def test_color_blob_tracker_synthetic_red_target():
    """Verify tracking of a red ball."""
    tracker = ColorBlobTracker(TargetColor.RED)
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Draw red circle (RGB: 240, 20, 20) centered at (50, 150) (bottom-left)
    y, x = np.ogrid[:200, :200]
    mask = ((x - 50) ** 2 + (y - 150) ** 2) <= 20 ** 2
    img[mask] = [240, 20, 20]

    res = tracker.track(img)
    assert res["detected"], "Red target should be detected"
    assert res["norm_x"] < -0.30, "Target should be to the left"
    assert res["norm_y"] > 0.30, "Target should be below center"


def test_visual_servoing_deadband_and_limits():
    """Verify anti-jitter deadband and servo rate/angle limits."""
    ctrl = VisualServoingHeadController(deadband=0.08, max_rate=0.05)
    ctrl.reset(0.0, 0.0)

    # 1. Target within deadband (|norm| <= 0.08): head should NOT move
    yaw, pitch = ctrl.update(norm_x=0.04, norm_y=-0.03, detected=True)
    assert yaw == 0.0 and pitch == 0.0, "Deadband must suppress small jitter"

    # 2. Large horizontal offset to the right: yaw steps towards target, bounded by max_rate
    yaw, pitch = ctrl.update(norm_x=0.80, norm_y=0.0, detected=True)
    assert yaw < 0.0, "Yaw should turn right (negative)"
    assert abs(yaw) <= 0.05 + 1e-5, f"Rate limit exceeded: {yaw}"

    # 3. Step multiple times: should clamp safely at physical joint limits
    for _ in range(100):
        yaw, pitch = ctrl.update(norm_x=1.0, norm_y=1.0, detected=True)

    assert yaw >= ctrl.yaw_limits[0] and yaw <= ctrl.yaw_limits[1]
    assert pitch >= ctrl.pitch_limits[0] and pitch <= ctrl.pitch_limits[1]


def test_visual_servoing_lost_target_return_to_center():
    """Verify neck smoothly returns towards neutral (0, 0) when target is lost."""
    ctrl = VisualServoingHeadController(max_rate=0.05)
    ctrl.reset(yaw=0.80, pitch=-0.30)

    yaw, pitch = ctrl.update(norm_x=0.0, norm_y=0.0, detected=False)
    assert abs(yaw) < 0.80, "Yaw should decay toward 0.0"
    assert abs(pitch) < 0.30, "Pitch should decay toward 0.0"


def test_build_mcu1_vision_packet():
    """Verify binary UART packet format for MCU1 ESP32-S3 / Pi Zero 2 W."""
    pkt = build_mcu1_vision_packet(head_yaw=0.15, head_pitch=-0.10)
    assert len(pkt) == 8, f"Expected 8 bytes, got {len(pkt)}"
    assert pkt[0] == 0xAA and pkt[1] == 0x55, "Missing sync bytes"
    assert pkt[2] == 0x10, "Wrong command byte"


def test_render_head_camera_in_mujoco():
    """Verify head camera renders a valid RGB image from MuJoCo models."""
    import mujoco

    models = [
        "models/mjcf/microduck_8dof.xml",
        "duck_v2/cyberduck_v2.xml",
    ]
    for mpath in models:
        model = mujoco.MjModel.from_xml_path(mpath)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        rgb = render_head_camera(model, data, camera_name="head_camera", width=160, height=120)
        assert rgb.shape == (120, 160, 3)
        assert rgb.dtype == np.uint8
