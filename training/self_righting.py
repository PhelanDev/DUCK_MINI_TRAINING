"""Self-Righting Reflex Recovery Controller for MicroDuck 8-DOF (MG996R).

Implements:
- FallType Enum: 4 distinct fall directions (PRONE, SUPINE, LATERAL_LEFT, LATERAL_RIGHT).
- FallClassifier: Distinguishes true falls from legitimate postures (SIT, LIE).
- SelfRightingController: Sequential 3-phase reflex recovery finite state machine
  (ORIENT_PRONE -> CROUCH_TO_SIT -> PUSH_TO_STAND -> DONE) with rate limiting
  (|delta q| <= 0.10 rad/step) to protect MG996R gears, and safety lock (FAILED_LOCKED)
  after 3 failed attempts.
"""

from enum import Enum, IntEnum
from typing import Optional, Tuple, Union
import numpy as np

from training.env_config import (
    LocomotionMode,
    DEFAULT_NOMINAL_QPOS,
    SIT_NOMINAL_QPOS,
    LIE_NOMINAL_QPOS,
)

# Nominal recovery postures for 8-DOF joints:
# (head_yaw, head_pitch, l_hip_roll, l_hip_pitch, l_knee_pitch, r_hip_roll, r_hip_pitch, r_knee_pitch)
PRONE_NOMINAL_QPOS: np.ndarray = LIE_NOMINAL_QPOS.copy()
STAND_NOMINAL_QPOS: np.ndarray = DEFAULT_NOMINAL_QPOS.copy()


class FallType(str, Enum):
    """4-way fall direction categorization."""

    PRONE = "prone"
    SUPINE = "supine"
    LATERAL_LEFT = "lateral_left"
    LATERAL_RIGHT = "lateral_right"


class RecoveryState(IntEnum):
    """Finite states of the self-righting recovery controller."""

    IDLE = 0
    PHASE1_ORIENT_PRONE = 1
    PHASE2_CROUCH_TO_SIT = 2
    PHASE3_PUSH_TO_STAND = 3
    DONE = 4
    FAILED_LOCKED = 5


class FallClassifier:
    """Classifies robot fall states or validates intentional poses."""

    @staticmethod
    def classify(
        root_z: float,
        proj_gravity: np.ndarray,
        cmd_mode: Union[LocomotionMode, int] = LocomotionMode.WALK,
    ) -> Optional[FallType]:
        """Classify robot posture into FallType or return None if upright/valid.

        Args:
            root_z: Center of mass height (m).
            proj_gravity: Projected gravity vector in body frame [gx, gy, gz], shape (3,).
            cmd_mode: Commanded locomotion mode (WALK=0, SIT=1, LIE=2).

        Returns:
            Optional[FallType]: Direction of fall, or None if legitimate posture.
        """
        grav = np.asarray(proj_gravity, dtype=np.float32)
        mode = int(cmd_mode)
        rz = float(root_z)

        # 1. Check if robot is in a legitimate intentional pose based on mode
        if mode == int(LocomotionMode.LIE):
            # In LIE mode: z >= 0.025m, not upside down (grav_z <= 0.0), no lateral rollover (|grav_y| <= 0.6)
            if rz >= 0.025 and grav[2] <= 0.0 and abs(grav[1]) <= 0.6:
                return None
        elif mode == int(LocomotionMode.SIT):
            # In SIT mode: z >= 0.050m, grav_z <= -0.707, no lateral rollover (|grav_y| <= 0.6)
            if rz >= 0.050 and grav[2] <= -0.707 and abs(grav[1]) <= 0.6:
                return None
        else:
            # In WALK mode: z >= 0.080m, grav_z <= -0.707, no lateral rollover (|grav_y| <= 0.6)
            if rz >= 0.080 and grav[2] <= -0.707 and abs(grav[1]) <= 0.6:
                return None

        # 2. Posture is fallen: classify into 4 distinct fall directions
        if grav[1] > 0.6:
            return FallType.LATERAL_LEFT
        if grav[1] < -0.6:
            return FallType.LATERAL_RIGHT
        if grav[2] > 0.0:
            return FallType.SUPINE
        if rz < 0.080 and grav[2] <= 0.0:
            return FallType.PRONE

        # Fallback for forward/backward pitch
        return FallType.PRONE if grav[2] <= 0.0 else FallType.SUPINE


class SelfRightingController:
    """Sequential 3-phase reflex self-righting controller for MicroDuck 8-DOF."""

    IDLE = RecoveryState.IDLE
    PHASE1_ORIENT_PRONE = RecoveryState.PHASE1_ORIENT_PRONE
    PHASE2_CROUCH_TO_SIT = RecoveryState.PHASE2_CROUCH_TO_SIT
    PHASE3_PUSH_TO_STAND = RecoveryState.PHASE3_PUSH_TO_STAND
    DONE = RecoveryState.DONE
    FAILED_LOCKED = RecoveryState.FAILED_LOCKED

    def __init__(
        self,
        max_rate: float = 0.10,
        max_retries: int = 3,
        phase_timeout: int = 40,
    ):
        self.max_rate = float(max_rate)
        self.max_retries = int(max_retries)
        self.phase_timeout = int(phase_timeout)
        self.reset()

    def reset(self) -> None:
        """Reset controller to IDLE initial state."""
        self.state = self.IDLE
        self.attempts = 0
        self.phase_steps = 0
        self.last_target: Optional[np.ndarray] = None
        self.fall_type: Optional[FallType] = None

    def trigger(
        self,
        fall_type: Optional[FallType] = None,
        qpos: Optional[np.ndarray] = None,
    ) -> None:
        """Trigger reflex recovery initiation."""
        self.fall_type = fall_type
        self.state = self.PHASE1_ORIENT_PRONE
        self.phase_steps = 0
        if self.attempts == 0:
            self.attempts = 1
        if qpos is not None:
            self.last_target = np.asarray(qpos, dtype=np.float32).copy()

    @property
    def is_recovering(self) -> bool:
        """Check if controller is currently in active recovery phases."""
        return self.state in (
            self.PHASE1_ORIENT_PRONE,
            self.PHASE2_CROUCH_TO_SIT,
            self.PHASE3_PUSH_TO_STAND,
        )

    @property
    def is_done(self) -> bool:
        """Check if recovery has successfully finished."""
        return self.state == self.DONE

    @property
    def is_failed(self) -> bool:
        """Check if recovery has failed and entered safety lock."""
        return self.state == self.FAILED_LOCKED

    @property
    def is_push_completed(self) -> bool:
        """Check if PHASE3_PUSH_TO_STAND has reached standing target pose."""
        if self.state != self.PHASE3_PUSH_TO_STAND or self.last_target is None:
            return False
        return bool(np.allclose(self.last_target, STAND_NOMINAL_QPOS, atol=1e-2))

    def step(
        self,
        qpos_joints: np.ndarray,
        root_z: float,
        proj_gravity: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Compute next target joint positions rate-limited at <= 0.10 rad/step.

        Args:
            qpos_joints: Current joint positions (8,).
            root_z: Base CoM height (m).
            proj_gravity: Projected gravity vector (3,).

        Returns:
            Tuple[np.ndarray, bool]: (target_joint_positions (8,), is_complete)
        """
        qpos = np.asarray(qpos_joints, dtype=np.float32)
        grav = np.asarray(proj_gravity, dtype=np.float32)
        rz = float(root_z)

        # Initialize target on first step if not set
        if self.last_target is None:
            self.last_target = qpos.copy()

        # 1. If IDLE: trigger recovery if not standing
        if self.state == self.IDLE:
            if rz >= 0.18 and grav[2] < -0.85:
                return self.last_target.copy(), True
            self.trigger(FallClassifier.classify(rz, grav), qpos=qpos)

        # 2. If DONE: return target and completion
        if self.state == self.DONE:
            return self.last_target.copy(), True

        # 3. If FAILED_LOCKED: return target and failure
        if self.state == self.FAILED_LOCKED:
            return self.last_target.copy(), False

        # 4. Phase trajectory targets
        self.phase_steps += 1
        target_pose = self.last_target

        if self.state == self.PHASE1_ORIENT_PRONE:
            target_pose = PRONE_NOMINAL_QPOS
            delta = np.clip(target_pose - self.last_target, -self.max_rate, self.max_rate)
            self.last_target = (self.last_target + delta).astype(np.float32)
            is_at_target = bool(np.allclose(self.last_target, target_pose, atol=1e-2))
            if (is_at_target and self.phase_steps >= 10) or self.phase_steps >= self.phase_timeout:
                self.state = self.PHASE2_CROUCH_TO_SIT
                self.phase_steps = 0
            return self.last_target.copy(), False

        elif self.state == self.PHASE2_CROUCH_TO_SIT:
            target_pose = SIT_NOMINAL_QPOS
            delta = np.clip(target_pose - self.last_target, -self.max_rate, self.max_rate)
            self.last_target = (self.last_target + delta).astype(np.float32)
            is_at_target = bool(np.allclose(self.last_target, target_pose, atol=1e-2))
            if (is_at_target and self.phase_steps >= 10) or self.phase_steps >= self.phase_timeout:
                self.state = self.PHASE3_PUSH_TO_STAND
                self.phase_steps = 0
            return self.last_target.copy(), False

        elif self.state == self.PHASE3_PUSH_TO_STAND:
            target_pose = STAND_NOMINAL_QPOS

            # Check if standing height and orientation achieved
            if rz >= 0.18 and grav[2] < -0.85:
                self.state = self.DONE
                return self.last_target.copy(), True

            delta = np.clip(target_pose - self.last_target, -self.max_rate, self.max_rate)
            self.last_target = (self.last_target + delta).astype(np.float32)

            if rz >= 0.18 and grav[2] < -0.85:
                self.state = self.DONE
                return self.last_target.copy(), True

            # Check timeout for retry logic
            if self.phase_steps >= self.phase_timeout:
                self.attempts += 1
                if self.attempts > self.max_retries:
                    self.state = self.FAILED_LOCKED
                    return self.last_target.copy(), False
                else:
                    self.state = self.PHASE1_ORIENT_PRONE
                    self.phase_steps = 0
                    return self.last_target.copy(), False

            return self.last_target.copy(), False
