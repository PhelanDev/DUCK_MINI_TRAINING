"""Advanced Gait & Behavior Engine for MicroDuck 8-DOF and CyberDuck V2 Pro.

Implements:
- GaitPattern Enum:
  - WALK: Nominal stable bipedal walking gait.
  - WADDLE: Iconic duck waddling gait with exaggerated pelvic sway, alternating leg clearance,
    and synchronized bionic duck head nodding.
  - SPOT_TURN_LEFT / SPOT_TURN_RIGHT: Turning-in-place bipedal gait without forward translation.
  - HEAD_DANCE: Rhythmic Lissajous head bobbing and inquisitive gaze gesture while standing.
- BipedalGaitGenerator: Periodic trajectory generator enforcing MG996R physical angle limits.
- AdvancedBehaviorController: Fused controller coordinating gait generator with
  SelfRightingController for automatic fall detection and reflex recovery.
"""

from enum import Enum
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np

from training.env_config import (
    DEFAULT_NOMINAL_QPOS,
    JOINT_LIMITS,
)
from training.self_righting import (
    FallClassifier,
    FallType,
    RecoveryState,
    SelfRightingController,
)

JOINT_NAMES = [
    "head_yaw",
    "head_pitch",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
]


class GaitPattern(str, Enum):
    """Supported locomotion and gesture behavioral patterns."""

    WALK = "walk"
    WADDLE = "waddle"
    SPOT_TURN_LEFT = "spot_turn_left"
    SPOT_TURN_RIGHT = "spot_turn_right"
    HEAD_DANCE = "head_dance"


class BipedalGaitGenerator:
    """Kinematic trajectory generator for 8-DOF bipedal robot gaits and gestures.

    Joint Order:
      0: head_yaw
      1: head_pitch
      2: left_hip_roll
      3: left_hip_pitch
      4: left_knee_pitch
      5: right_hip_roll
      6: right_hip_pitch
      7: right_knee_pitch
    """

    def __init__(
        self,
        default_freq: float = 1.0,
        joint_limits: Optional[np.ndarray] = None,
    ):
        self.default_freq = float(default_freq)
        self.joint_limits = joint_limits if joint_limits is not None else JOINT_LIMITS

    def clamp_targets(self, targets: np.ndarray) -> np.ndarray:
        """Clamp joint angles to safe MG996R servo limits."""
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return np.clip(targets, low, high).astype(np.float32)

    def compute_walk(
        self,
        t: float,
        freq: Optional[float] = None,
        roll_amp: float = 0.04,
        pitch_amp: float = 0.06,
        knee_amp: float = 0.10,
    ) -> np.ndarray:
        """Nominal forward stepping gait with subtle head counter-motion."""
        f = freq if freq is not None else self.default_freq
        phase = 2.0 * math.pi * f * t
        s = math.sin(phase)

        # Lateral shift toward support foot
        l_roll = roll_amp * s
        r_roll = roll_amp * s

        # Leg swing centered at nominal -0.15 rad
        l_pitch = -0.15 - pitch_amp * s
        r_pitch = -0.15 + pitch_amp * s

        # Knee flexion for clearance
        l_knee = 0.15 + knee_amp * max(0.0, s)
        r_knee = 0.15 + knee_amp * max(0.0, -s)

        # Subtle head counter-yaw and pitch
        head_yaw = -0.05 * s
        head_pitch = 0.02

        raw = np.array(
            [head_yaw, head_pitch, l_roll, l_pitch, l_knee, r_roll, r_pitch, r_knee],
            dtype=np.float32,
        )
        return self.clamp_targets(raw)

    def compute_waddle(
        self,
        t: float,
        freq: Optional[float] = None,
        sway_amp: float = 0.09,
        pitch_amp: float = 0.08,
        knee_amp: float = 0.14,
    ) -> np.ndarray:
        """Iconic duck waddling gait: exaggerated hip roll sway and bionic duck head nodding.

        The pelvic sway shifts center of mass fully over the stance leg, allowing the swing leg
        to clear ground with a wide, charming duck waddle. Head nods rhythmically twice per gait
        cycle (pecking/nodding frequency = 2 * gait_freq).
        """
        f = freq if freq is not None else (self.default_freq * 0.9)
        phase = 2.0 * math.pi * f * t
        s = math.sin(phase)
        c = math.cos(phase)

        # Exaggerated hip roll sway (synchronized left-right pelvis tilt)
        l_roll = sway_amp * s
        r_roll = sway_amp * s

        # Sagittal swing shifted 90 deg (cos phase) so maximum swing occurs when CoM is loaded
        l_pitch = -0.15 - pitch_amp * c
        r_pitch = -0.15 + pitch_amp * c

        # Higher knee lift for adorable waddle clearance
        l_knee = 0.15 + knee_amp * max(0.0, c)
        r_knee = 0.15 + knee_amp * max(0.0, -c)

        # Bionic duck head nodding: double-frequency pitch nod, gentle counter-yaw
        head_nod = 0.05 + 0.07 * abs(s)
        head_yaw = -0.08 * s

        raw = np.array(
            [head_yaw, head_nod, l_roll, l_pitch, l_knee, r_roll, r_pitch, r_knee],
            dtype=np.float32,
        )
        return self.clamp_targets(raw)

    def compute_spot_turn(
        self,
        t: float,
        direction: int = 1,
        freq: Optional[float] = None,
        turn_amp: float = 0.08,
        knee_amp: float = 0.12,
    ) -> np.ndarray:
        """Turning-in-place bipedal gait without forward displacement.

        Args:
            t: Current time in seconds.
            direction: +1 for turning Left (counter-clockwise), -1 for turning Right (clockwise).
            freq: Stepping frequency in Hz.
            turn_amp: Hip roll and differential pitch amplitude.
            knee_amp: Knee clearance height.
        """
        f = freq if freq is not None else (self.default_freq * 1.1)
        phase = 2.0 * math.pi * f * t
        s = math.sin(phase)
        c = math.cos(phase)
        dir_sign = 1.0 if direction >= 0 else -1.0

        # Lateral weight shifting
        roll_shift = 0.06 * s
        l_roll = roll_shift
        r_roll = roll_shift

        # Differential hip pitch: opposing signs cause body to spin in place without translating forward
        l_pitch = -0.15 + dir_sign * turn_amp * c
        r_pitch = -0.15 - dir_sign * turn_amp * c

        # Alternating knee lifts
        l_knee = 0.15 + knee_amp * max(0.0, c)
        r_knee = 0.15 + knee_amp * max(0.0, -c)

        # Head anticipates the turn by looking into the direction of rotation
        head_yaw = 0.22 * dir_sign + 0.05 * s
        head_pitch = 0.02

        raw = np.array(
            [head_yaw, head_pitch, l_roll, l_pitch, l_knee, r_roll, r_pitch, r_knee],
            dtype=np.float32,
        )
        return self.clamp_targets(raw)

    def compute_head_dance(
        self,
        t: float,
        freq: Optional[float] = None,
        yaw_amp: float = 0.35,
        pitch_amp: float = 0.18,
    ) -> np.ndarray:
        """Expressive Lissajous head-bobbing and inquisitive scanning while maintaining solid stance."""
        f = freq if freq is not None else 1.2
        phase_yaw = 2.0 * math.pi * f * t
        phase_pitch = 4.0 * math.pi * f * t

        head_yaw = yaw_amp * math.sin(phase_yaw)
        head_pitch = 0.04 + pitch_amp * math.cos(phase_pitch)

        # Stance remains stable at nominal qpos
        raw = np.array(
            [
                head_yaw,
                head_pitch,
                DEFAULT_NOMINAL_QPOS[2],
                DEFAULT_NOMINAL_QPOS[3],
                DEFAULT_NOMINAL_QPOS[4],
                DEFAULT_NOMINAL_QPOS[5],
                DEFAULT_NOMINAL_QPOS[6],
                DEFAULT_NOMINAL_QPOS[7],
            ],
            dtype=np.float32,
        )
        return self.clamp_targets(raw)

    def compute_gait(
        self,
        t: float,
        pattern: Union[str, GaitPattern] = GaitPattern.WADDLE,
        **kwargs,
    ) -> np.ndarray:
        """Dispatcher computing target joint angles for the specified gait pattern."""
        pat = GaitPattern(pattern) if isinstance(pattern, str) else pattern

        if pat == GaitPattern.WADDLE:
            return self.compute_waddle(t, **kwargs)
        elif pat == GaitPattern.WALK:
            return self.compute_walk(t, **kwargs)
        elif pat == GaitPattern.SPOT_TURN_LEFT:
            return self.compute_spot_turn(t, direction=1, **kwargs)
        elif pat == GaitPattern.SPOT_TURN_RIGHT:
            return self.compute_spot_turn(t, direction=-1, **kwargs)
        elif pat == GaitPattern.HEAD_DANCE:
            return self.compute_head_dance(t, **kwargs)
        else:
            return self.compute_walk(t, **kwargs)


class AdvancedBehaviorController:
    """Unified behavior controller combining bipedal gaits with autonomous fall recovery reflex.

    If a fall is detected via FallClassifier, the controller automatically takes over joint
    commands, triggers the 3-phase SelfRightingController, and smoothly resumes gait generation
    once standing stability is restored.
    """

    def __init__(
        self,
        default_pattern: GaitPattern = GaitPattern.WADDLE,
        gait_freq: float = 1.0,
        enable_fall_recovery: bool = True,
    ):
        self.pattern = default_pattern
        self.gait_gen = BipedalGaitGenerator(default_freq=gait_freq)
        self.recovery_ctrl = SelfRightingController()
        self.enable_fall_recovery = enable_fall_recovery
        self.time: float = 0.0

    @property
    def is_recovering(self) -> bool:
        """True if the controller is actively executing fall recovery reflex."""
        return self.recovery_ctrl.is_recovering

    def set_pattern(self, pattern: Union[str, GaitPattern]) -> None:
        """Change the active gait or behavior pattern."""
        self.pattern = GaitPattern(pattern) if isinstance(pattern, str) else pattern

    def trigger_fall_recovery(
        self,
        curr_qpos: np.ndarray,
        fall_type: FallType = FallType.PRONE,
    ) -> None:
        """Manually trigger self-righting reflex."""
        self.recovery_ctrl.trigger(fall_type, curr_qpos)

    def step(
        self,
        dt: float,
        curr_qpos: np.ndarray,
        root_z: float,
        proj_gravity: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Compute the next joint target set.

        Args:
            dt: Timestep advance in seconds.
            curr_qpos: Current 8 joint positions.
            root_z: Center of mass height (m).
            proj_gravity: Projected gravity vector in body frame [gx, gy, gz], shape (3,).

        Returns:
            targets: Clamped 8-DOF joint angle targets.
            recovering: True if in recovery reflex, False if nominal gait active.
        """
        self.time += dt

        # 1. Fall detection & reflex takeover
        if self.enable_fall_recovery:
            if not self.recovery_ctrl.is_recovering:
                detected_fall = FallClassifier.classify(root_z, proj_gravity)
                if detected_fall is not None:
                    self.recovery_ctrl.trigger(detected_fall, curr_qpos)

            if self.recovery_ctrl.is_recovering:
                recovery_targets, is_done = self.recovery_ctrl.step(
                    curr_qpos, root_z, proj_gravity
                )
                if is_done:
                    # Recovery finished: reset recovery state to IDLE and resume nominal gait
                    self.recovery_ctrl.state = RecoveryState.IDLE
                    self.recovery_ctrl.last_target = None
                    return self.gait_gen.compute_gait(self.time, self.pattern), False
                return self.gait_gen.clamp_targets(recovery_targets), True

        # 2. Nominal gait execution
        targets = self.gait_gen.compute_gait(self.time, self.pattern)
        return targets, False
