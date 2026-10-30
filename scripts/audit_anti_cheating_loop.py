#!/usr/bin/env python3
"""5-Pass Zero-Tolerance Anti-Cheating & Vulnerability Audit Loop.

Executes a strict 5-pass continuous verification loop covering 12 critical
reward hacking, kinodynamic exploit, and physical integrity rules.

Zero-Tolerance Rule:
If any failure, reward leak, or exploit occurs at any pass k in {1, 2, 3, 4, 5},
the consecutive pass counter is IMMEDIATELY RESET TO ZERO, requiring 5 clean
passes from scratch before certification is granted.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from typing import Callable, List, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.test_5pass_anti_cheating_audit import (
    test_audit_1_strict_tilt_45deg,
    test_audit_2_trampling_in_place_guard,
    test_audit_3_speed_diving_clip,
    test_audit_4_alive_bonus_and_standstill_penalty,
    test_audit_5_curriculum_cohort_success_rate,
    test_audit_6_stair_edge_fall_detection,
    test_audit_7_slalom_corridor_bounds,
    test_audit_8_joint_limit_dwelling_penalty,
    test_audit_9_rate_limiting_delta_q,
    test_audit_10_multimode_unauthorized_pose,
    test_audit_11_self_righting_recovery_lock,
    test_audit_12_voice_intent_crc16_integrity,
    test_audit_13_four_golden_hoops_anti_vibration,
    test_audit_14_anti_backward_and_forward_gait_lock,
)

AUDIT_CHECKS: List[Tuple[str, Callable[[], None]]] = [
    (
        "Audit 1: Strict Tilt 45-deg Fall Termination (grav_z > -0.707)",
        test_audit_1_strict_tilt_45deg,
    ),
    (
        "Audit 2: Trampling In-Place Air-Time Guard (v_x < 0.05 -> r_air=0)",
        test_audit_2_trampling_in_place_guard,
    ),
    (
        "Audit 3: Speed Diving Linear Velocity Tracking Clip (v_x <= cmd_x + 0.05)",
        test_audit_3_speed_diving_clip,
    ),
    (
        "Audit 4: Alive Bonus <= 0.05 & Anti-Milking Standstill Penalty",
        test_audit_4_alive_bonus_and_standstill_penalty,
    ),
    (
        "Audit 5: Mandatory Cohort Success Rate >= 70% in Curriculum Progression",
        test_audit_5_curriculum_cohort_success_rate,
    ),
    (
        "Audit 6: Stair Boundary Edge-Fall Detection (|y| > 0.44m in stair zone)",
        test_audit_6_stair_edge_fall_detection,
    ),
    (
        "Audit 7: Slalom Corridor Bypass Detection (|y| > 0.55m in x in [0.2, 1.2]m)",
        test_audit_7_slalom_corridor_bounds,
    ),
    (
        "Audit 8: Joint Limit Dwelling Penalty (< 5% Physical Range Margin)",
        test_audit_8_joint_limit_dwelling_penalty,
    ),
    (
        "Audit 9: MG996R Actuator Rate Limiting (|delta_q| <= 0.10 rad/step)",
        test_audit_9_rate_limiting_delta_q,
    ),
    (
        "Audit 10: Multi-Mode Unauthorized Pose Collapse & Rest Reward Exclusion",
        test_audit_10_multimode_unauthorized_pose,
    ),
    (
        "Audit 11: Self-Righting Finite Loop & 3-Attempt Lockout (FAILED_LOCKED)",
        test_audit_11_self_righting_recovery_lock,
    ),
    (
        "Audit 12: Voice Intent Packet CRC16 Validation & Desync Immunity",
        test_audit_12_voice_intent_crc16_integrity,
    ),
    (
        "Audit 13: 4 Golden Hoops Anti-Vibration & Locomotion Integrity Certification",
        test_audit_13_four_golden_hoops_anti_vibration,
    ),
    (
        "Audit 14: Anti-Backward Velocity Penalty (-4.0*vx) & Forward Gait Lock (vx > 0.02)",
        test_audit_14_anti_backward_and_forward_gait_lock,
    ),
]


def run_5pass_audit_loop(
    required_passes: int = 5,
    max_attempts: int = 50,
    verbose: bool = True,
) -> bool:
    """Run the 5-pass zero-tolerance audit loop."""
    print("=" * 80)
    print(" 5-PASS ZERO-TOLERANCE ANTI-CHEATING & VULNERABILITY AUDIT LOOP")
    print(f" Target: {required_passes} consecutive flawless passes without single error or reset")
    print(f" Audited Modules: {len(AUDIT_CHECKS)} physical integrity & reward hacking checkpoints")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    consecutive_passes = 0
    attempt_count = 0
    pass_timings: List[float] = []

    loop_start_time = time.perf_counter()

    while consecutive_passes < required_passes and attempt_count < max_attempts:
        attempt_count += 1
        pass_idx = consecutive_passes + 1
        if verbose:
            print(f"\n>>> Starting Pass {pass_idx}/{required_passes} (Attempt #{attempt_count})...")

        pass_start_time = time.perf_counter()
        pass_failed = False
        failed_audit_name = ""
        error_detail = ""

        for check_idx, (name, check_fn) in enumerate(AUDIT_CHECKS, 1):
            check_t0 = time.perf_counter()
            try:
                check_fn()
                check_dur = time.perf_counter() - check_t0
                if verbose:
                    print(f"  [OK] [{check_idx:02d}/{len(AUDIT_CHECKS):02d}] {name} ({check_dur * 1000.0:.1f}ms)")
            except Exception as e:
                pass_failed = True
                failed_audit_name = name
                error_detail = f"{type(e).__name__}: {str(e)}"
                if verbose:
                    print(f"  [FAIL] [{check_idx:02d}/{len(AUDIT_CHECKS):02d}] {name}")
                    print(f"         Error: {error_detail}")
                    traceback.print_exc()
                break

        pass_duration = time.perf_counter() - pass_start_time

        if pass_failed:
            print("\n" + "!" * 80)
            print(f" [ZERO-TOLERANCE AUDIT BREACH DETECTED]")
            print(f" Failed Check : {failed_audit_name}")
            print(f" Details      : {error_detail}")
            print(f" [RESET] Consecutive passes count reset from {consecutive_passes} back to 0!")
            print("!" * 80)
            consecutive_passes = 0
            pass_timings.clear()
        else:
            consecutive_passes += 1
            pass_timings.append(pass_duration)
            print(
                f" [PASS {consecutive_passes}/{required_passes} SUCCESS] "
                f"All {len(AUDIT_CHECKS)}/{len(AUDIT_CHECKS)} audits passed flawlessly in {pass_duration:.3f}s"
            )

    total_time = time.perf_counter() - loop_start_time
    print("\n" + "=" * 80)
    if consecutive_passes == required_passes:
        print(" 5-PASS ZERO-TOLERANCE AUDIT CERTIFICATION GRANTED: 100% CLEAN")
        print("=" * 80)
        for i, t in enumerate(pass_timings, 1):
            print(f"   - Pass {i}/{required_passes}: PASSED ({t:.3f}s, 14/14 checks clean)")
        print(f" Total Continuous Audit Time : {total_time:.3f}s")
        print(f" Total Attempts Required     : {attempt_count}")
        print(f" Residual Vulnerabilities    : ZERO (0)")
        print(f" Status                      : PASSED & VERIFIED")
        print("=" * 80)
        return True
    else:
        print(" 5-PASS AUDIT FAILED TO ACHIEVE REQUIRED CONSECUTIVE PASSES")
        print(f" Max attempts ({max_attempts}) reached without {required_passes} consecutive clean passes.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Pass Zero-Tolerance Anti-Cheating Audit Loop")
    parser.add_argument("--passes", type=int, default=5, help="Number of consecutive clean passes required")
    parser.add_argument("--max-retries", type=int, default=50, help="Max total loop attempts before aborting")
    args = parser.parse_args()

    success = run_5pass_audit_loop(required_passes=args.passes, max_attempts=args.max_retries)
    sys.exit(0 if success else 1)
