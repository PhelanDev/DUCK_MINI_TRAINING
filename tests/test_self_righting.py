"""Tests for Self-Righting Recovery Controller and ParallelSimRunner Integration (Task 3).

Covers:
- FallType Enum (PRONE, SUPINE, LATERAL_LEFT, LATERAL_RIGHT)
- RecoveryState IntEnum (IDLE, PHASE1_ORIENT_PRONE, PHASE2_CROUCH_TO_SIT, PHASE3_PUSH_TO_STAND, DONE, FAILED_LOCKED)
- FallClassifier:
  - 4-way fall direction classification (PRONE, SUPINE, LATERAL_LEFT, LATERAL_RIGHT)
  - Posture legitimacy check across WALK, SIT, and LIE modes
- SelfRightingController:
  - 3-phase reflex state transitions (ORIENT_PRONE -> CROUCH_TO_SIT -> PUSH_TO_STAND -> DONE)
  - MG996R servo protection via joint rate limiting (|delta q| <= 0.10 rad/step)
  - Safety lock into FAILED_LOCKED after 3 failed recovery attempts
- ParallelSimRunner integration:
  - enable_auto_recovery parameter and per-env recovery controllers
  - Episode continuation during recovery without premature termination
  - Successful righting to nominal standing posture (z >= 0.18m, grav_z < -0.85)
"""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.env_config import (
    MicroDuck8DoFConfig,
    LocomotionMode,
    DEFAULT_NOMINAL_QPOS,
    SIT_NOMINAL_QPOS,
    LIE_NOMINAL_QPOS,
)
from training.self_righting import (
    FallType,
    RecoveryState,
    FallClassifier,
    SelfRightingController,
)
from training.run_parallel_sim import ParallelSimRunner


def test_fall_type_and_recovery_state_enums():
    """Verify FallType and RecoveryState enum values and class constants."""
    assert hasattr(FallType, "PRONE")
    assert hasattr(FallType, "SUPINE")
    assert hasattr(FallType, "LATERAL_LEFT")
    assert hasattr(FallType, "LATERAL_RIGHT")

    assert FallType.PRONE.value == "prone"
    assert FallType.SUPINE.value == "supine"
    assert FallType.LATERAL_LEFT.value == "lateral_left"
    assert FallType.LATERAL_RIGHT.value == "lateral_right"

    assert hasattr(RecoveryState, "IDLE")
    assert hasattr(RecoveryState, "PHASE1_ORIENT_PRONE")
    assert hasattr(RecoveryState, "PHASE2_CROUCH_TO_SIT")
    assert hasattr(RecoveryState, "PHASE3_PUSH_TO_STAND")
    assert hasattr(RecoveryState, "DONE")
    assert hasattr(RecoveryState, "FAILED_LOCKED")

    assert RecoveryState.IDLE == 0
    assert RecoveryState.PHASE1_ORIENT_PRONE == 1
    assert RecoveryState.PHASE2_CROUCH_TO_SIT == 2
    assert RecoveryState.PHASE3_PUSH_TO_STAND == 3
    assert RecoveryState.DONE == 4
    assert RecoveryState.FAILED_LOCKED == 5

    # Check SelfRightingController class aliases
    assert SelfRightingController.IDLE == RecoveryState.IDLE
    assert SelfRightingController.PHASE1_ORIENT_PRONE == RecoveryState.PHASE1_ORIENT_PRONE
    assert SelfRightingController.PHASE2_CROUCH_TO_SIT == RecoveryState.PHASE2_CROUCH_TO_SIT
    assert SelfRightingController.PHASE3_PUSH_TO_STAND == RecoveryState.PHASE3_PUSH_TO_STAND
    assert SelfRightingController.DONE == RecoveryState.DONE
    assert SelfRightingController.FAILED_LOCKED == RecoveryState.FAILED_LOCKED


def test_fall_classifier_directions():
    """Verify 4-way fall direction classification under WALK mode."""
    # 1. Nominal upright standing: no fall
    upright_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    assert FallClassifier.classify(0.194, upright_grav, LocomotionMode.WALK) is None

    # 2. Prone fall (face-down, low height, grav_z < 0)
    prone_grav = np.array([0.0, 0.0, -0.90], dtype=np.float32)
    assert FallClassifier.classify(0.050, prone_grav, LocomotionMode.WALK) == FallType.PRONE

    # 3. Supine fall (on back, chest up, grav_z > 0)
    supine_grav = np.array([0.0, 0.0, 0.85], dtype=np.float32)
    assert FallClassifier.classify(0.050, supine_grav, LocomotionMode.WALK) == FallType.SUPINE

    # 4. Lateral left fall (grav_y > 0.6)
    left_grav = np.array([0.0, 0.75, -0.66], dtype=np.float32)
    assert FallClassifier.classify(0.080, left_grav, LocomotionMode.WALK) == FallType.LATERAL_LEFT

    # 5. Lateral right fall (grav_y < -0.6)
    right_grav = np.array([-0.10, -0.75, -0.65], dtype=np.float32)
    assert FallClassifier.classify(0.080, right_grav, LocomotionMode.WALK) == FallType.LATERAL_RIGHT


def test_fall_classifier_valid_sit_and_lie_modes():
    """Verify FallClassifier discriminates intentional sitting/lying from actual falls."""
    # 1. Valid SIT posture: z = 0.075m, grav_z = -0.98 -> None (not fallen)
    sit_grav = np.array([0.0, 0.0, -0.98], dtype=np.float32)
    assert FallClassifier.classify(0.075, sit_grav, LocomotionMode.SIT) is None

    # 2. Fallen SIT posture: z = 0.035m < 0.050m -> fallen (PRONE)
    assert FallClassifier.classify(0.035, sit_grav, LocomotionMode.SIT) == FallType.PRONE

    # 3. Fallen SIT posture: lateral roll grav_y > 0.6 -> fallen (LATERAL_LEFT)
    sit_roll_grav = np.array([0.0, 0.80, -0.50], dtype=np.float32)
    assert FallClassifier.classify(0.075, sit_roll_grav, LocomotionMode.SIT) == FallType.LATERAL_LEFT

    # 4. Valid LIE posture: z = 0.040m, grav_z = -0.50 -> None (not fallen)
    lie_grav = np.array([0.0, 0.0, -0.50], dtype=np.float32)
    assert FallClassifier.classify(0.040, lie_grav, LocomotionMode.LIE) is None

    # 5. Fallen LIE posture: upside down grav_z > 0.0 -> fallen (SUPINE)
    lie_upside_grav = np.array([0.0, 0.0, 0.80], dtype=np.float32)
    assert FallClassifier.classify(0.040, lie_upside_grav, LocomotionMode.LIE) == FallType.SUPINE


def test_self_righting_controller_phase_transitions():
    """Verify sequential 3-phase state transitions: ORIENT_PRONE -> CROUCH_TO_SIT -> PUSH_TO_STAND -> DONE."""
    ctrl = SelfRightingController()
    assert ctrl.state == SelfRightingController.IDLE

    # Robot starts fallen on belly at z = 0.05m
    qpos = np.array([0.0, 0.0, 0.0, -0.65, 0.05, 0.0, -0.65, 0.05], dtype=np.float32)
    grav = np.array([0.0, 0.0, -0.95], dtype=np.float32)
    root_z = 0.050

    # Step 1: IDLE transitions to PHASE1_ORIENT_PRONE
    target, is_done = ctrl.step(qpos, root_z, grav)
    assert not is_done
    assert ctrl.state == SelfRightingController.PHASE1_ORIENT_PRONE
    assert ctrl.is_recovering

    # Step through Phase 1 until it transitions to Phase 2
    steps = 0
    while ctrl.state == SelfRightingController.PHASE1_ORIENT_PRONE and steps < 50:
        target, is_done = ctrl.step(target, root_z, grav)
        steps += 1
    assert ctrl.state == SelfRightingController.PHASE2_CROUCH_TO_SIT

    # Step through Phase 2 until it transitions to Phase 3
    steps = 0
    while ctrl.state == SelfRightingController.PHASE2_CROUCH_TO_SIT and steps < 50:
        target, is_done = ctrl.step(target, root_z, grav)
        steps += 1
    assert ctrl.state == SelfRightingController.PHASE3_PUSH_TO_STAND

    # Step through Phase 3: once robot reaches upright standing (z >= 0.18, grav_z < -0.85), transitions to DONE
    standing_grav = np.array([0.0, 0.0, -0.98], dtype=np.float32)
    standing_z = 0.194
    target, is_done = ctrl.step(target, standing_z, standing_grav)
    assert is_done
    assert ctrl.state == SelfRightingController.DONE
    assert ctrl.is_done


def test_self_righting_rate_limiting():
    """Verify joint rate limit |delta q| <= 0.10 rad/step is strictly enforced on all 8 joints."""
    ctrl = SelfRightingController(max_rate=0.10)
    # Start with arbitrary joint configuration
    curr_qpos = np.zeros(8, dtype=np.float32)
    root_z = 0.050
    grav = np.array([0.0, 0.0, -0.95], dtype=np.float32)

    last_target = curr_qpos.copy()
    ctrl.trigger(FallType.PRONE, curr_qpos)

    # Step 60 times covering transitions across phases
    for _ in range(60):
        target, _ = ctrl.step(last_target, root_z, grav)
        delta = np.abs(target - last_target)
        assert np.all(delta <= 0.10 + 1e-5), f"Max joint velocity exceeded: {np.max(delta)} > 0.10"
        last_target = target.copy()


def test_self_righting_locked_after_three_failed_attempts():
    """Verify safety lock transitions to FAILED_LOCKED after 3 consecutive failed recovery attempts."""
    # Use shorter phase_timeout to speed up test
    ctrl = SelfRightingController(max_retries=3, phase_timeout=25)
    curr_qpos = np.zeros(8, dtype=np.float32)
    # Robot is physically stuck on floor: z never reaches 0.18m
    stuck_z = 0.050
    stuck_grav = np.array([0.0, 0.0, -0.50], dtype=np.float32)

    ctrl.trigger(FallType.PRONE, curr_qpos)
    target = curr_qpos.copy()

    # Step until locked or timeout
    for _ in range(300):
        target, is_done = ctrl.step(target, stuck_z, stuck_grav)
        if ctrl.state == SelfRightingController.FAILED_LOCKED:
            break

    assert ctrl.state == SelfRightingController.FAILED_LOCKED
    assert ctrl.is_failed
    assert not is_done


def test_parallel_sim_runner_auto_recovery_integration():
    """Verify ParallelSimRunner with enable_auto_recovery=True recovers fallen robot to standing."""
    sim = ParallelSimRunner(num_envs=2, decimation=1, enable_auto_recovery=True)
    assert hasattr(sim, "enable_auto_recovery")
    assert sim.enable_auto_recovery is True
    assert hasattr(sim, "recovery_controllers")
    assert len(sim.recovery_controllers) == 2

    # Put robot in env 0 into fallen prone position on floor (z = 0.04m < 0.08m)
    d0 = sim.datas[0]
    d0.qpos[0:3] = [0.0, 0.0, 0.040]
    d0.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d0.qpos[7:15] = LIE_NOMINAL_QPOS
    d0.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(sim.model, d0)

    actions = np.zeros((2, 8), dtype=np.float32)

    # First step: env 0 must NOT terminate immediately because auto_recovery intercepts it
    next_obs, rewards, dones, infos = sim.step(actions)
    assert not dones[0], "Fallen episode should not be immediately terminated when auto_recovery is enabled"
    assert sim.recovery_controllers[0].is_recovering

    # Step simulation until recovery completes
    recovered = False
    for step_idx in range(120):
        next_obs, rewards, dones, infos = sim.step(actions)
        # Check if env 0 has recovered to nominal standing height and upright gravity
        root_z = float(sim.datas[0].qpos[2])
        w, x, y, z = sim.datas[0].sensordata[6:10]
        grav_z = float(2.0 * (x * x + y * y) - 1.0)
        if root_z >= 0.18 and grav_z < -0.85:
            recovered = True
            break

    assert recovered, f"Robot failed to auto-recover: root_z={sim.datas[0].qpos[2]:.3f}"
