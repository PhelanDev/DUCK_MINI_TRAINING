"""Terrain and Obstacle Curriculum Manager for MicroDuck 8-DOF.

Controls progression of difficulty by:
1. Scaling slalom pillar distance (XY scaling factor: 1.80x -> 1.35x -> 1.10x -> 1.00x).
2. Gradually raising staircase steps & landing platform (stair_step_h: submerged -> submerged -> 0.008m -> 0.012m).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import mujoco
import numpy as np


@dataclass(frozen=True)
class Stage:
    stage_id: int
    pillar_scale: float
    stair_step_h: float
    description: str


STAGES: List[Stage] = [
    Stage(
        stage_id=0,
        pillar_scale=1.80,
        stair_step_h=-0.020,
        description="Stage 0: Wide slalom pillars (1.8x), submerged stairs",
    ),
    Stage(
        stage_id=1,
        pillar_scale=1.35,
        stair_step_h=-0.020,
        description="Stage 1: Pure slalom pillars (1.35x), submerged stairs",
    ),
    Stage(
        stage_id=2,
        pillar_scale=1.10,
        stair_step_h=0.008,
        description="Stage 2: Narrow slalom (1.1x), low stairs (+8mm/step, +24mm landing)",
    ),
    Stage(
        stage_id=3,
        pillar_scale=1.00,
        stair_step_h=0.012,
        description="Stage 3: Full challenge (1.0x slalom, +12mm/step stairs, +36mm landing)",
    ),
]

# Baseline XML positions of the 6 slalom pillars
_BASE_PILLAR_POS: Dict[str, Tuple[float, float, float]] = {
    "pillar_l1_left": (0.45, 0.35, 0.15),
    "pillar_l1_right": (0.45, -0.35, 0.15),
    "pillar_l2_center": (0.80, 0.08, 0.15),
    "pillar_l3_left": (1.15, 0.25, 0.15),
    "pillar_l3_right": (1.15, -0.25, 0.15),
    "pillar_back": (-0.75, 0.0, 0.15),
}


def apply_stage(model: mujoco.MjModel, stage: int) -> None:
    """Apply curriculum geometry parameters to MuJoCo model geom_pos.

    Modifies the shared model in-place; all parallel environments immediately observe
    the updated slalom pillar and stair coordinates. Idempotent across repeated calls.
    """
    stage_idx = max(0, min(stage, len(STAGES) - 1))
    cfg = STAGES[stage_idx]

    # 1. Update Slalom Pillars (3-layer course)
    for name, base_pos in _BASE_PILLAR_POS.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            model.geom_pos[gid, 0] = base_pos[0] * cfg.pillar_scale
            model.geom_pos[gid, 1] = base_pos[1] * cfg.pillar_scale
            model.geom_pos[gid, 2] = base_pos[2]

    # 2. Update Staircase & Platform (pos_z)
    stair_multipliers = {
        "stair_step_1": 1,
        "stair_step_2": 2,
        "stair_step_3": 3,
        "stair_platform": 3,
        "stair_dn_1": 2,
        "stair_dn_2": 1,
    }
    for s_name, mult in stair_multipliers.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, s_name)
        if gid != -1:
            half_z = model.geom_size[gid, 2]
            if cfg.stair_step_h <= 0.0:
                # Submerge all 4 blocks below floor: top surface = pos_z + half_z <= 0.0
                model.geom_pos[gid, 2] = -half_z - 0.005
            else:
                top_z = mult * cfg.stair_step_h
                model.geom_pos[gid, 2] = top_z - half_z


def randomize_pillars(
    model: mujoco.MjModel,
    stage: int,
    rng: np.random.Generator,
    density_rho: float = 1.0,
    jitter_xy: float = 0.06,
    min_spawn_dist: float = 0.30,
    min_gap_width: float = 0.22,
) -> List[str]:
    """Randomize active slalom pillars, positions, and maintain safety clearances.

    Prevents policy from memorizing static coordinates while ensuring:
    1. Inactive pillars are submerged to z = -0.50m.
    2. Active pillars stay within corridor bounds (|y| <= 0.45m).
    3. Clear spawn distance away from robot origin (hypot(x, y) >= min_spawn_dist + r).
    4. Passable corridor gaps between paired pillars (gap >= min_gap_width).
    """
    apply_stage(model, stage)
    available = [
        name for name in _BASE_PILLAR_POS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1
    ]
    if stage <= 0 or density_rho <= 0.0 or not available:
        return available

    n_keep = max(1, int(np.ceil(np.clip(density_rho, 0.0, 1.0) * len(available))))
    selected_indices = rng.choice(len(available), size=n_keep, replace=False)
    active_set = {available[i] for i in selected_indices}
    active_pillars = [name for name in available if name in active_set]

    # Submerge inactive pillars
    for name in available:
        if name not in active_set:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            model.geom_pos[gid, 2] = -0.50

    # Jitter and enforce spawn clearance for active pillars
    for name in active_pillars:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        r = float(model.geom_size[gid, 0])
        jitter = rng.uniform(-jitter_xy, jitter_xy, size=2)
        px = float(model.geom_pos[gid, 0] + jitter[0])
        py = float(model.geom_pos[gid, 1] + jitter[1])
        py = float(np.clip(py, -0.45, 0.45))

        req_dist = min_spawn_dist + r
        dist = float(np.hypot(px, py))
        if dist < req_dist:
            if dist < 1e-6:
                px = req_dist
                py = 0.0
            else:
                scale = req_dist / dist
                px *= scale
                py *= scale
                if abs(py) > 0.45:
                    py = float(np.clip(py, -0.45, 0.45))
                    rem_sq = req_dist**2 - py**2
                    if rem_sq > 0.0:
                        sign_x = 1.0 if px >= 0 else -1.0
                        px = sign_x * float(np.sqrt(rem_sq))

        model.geom_pos[gid, 0] = px
        model.geom_pos[gid, 1] = py

    # Guarantee passable gap between paired pillars (L1 left/right, L3 left/right)
    pairs = [
        ("pillar_l1_left", "pillar_l1_right"),
        ("pillar_l3_left", "pillar_l3_right"),
    ]
    for left_name, right_name in pairs:
        if left_name in active_set and right_name in active_set:
            gid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, left_name)
            gid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, right_name)
            r_l = float(model.geom_size[gid_l, 0])
            r_r = float(model.geom_size[gid_r, 0])
            target_dist = min(0.90, min_gap_width + r_l + r_r)
            py_l = float(model.geom_pos[gid_l, 1])
            py_r = float(model.geom_pos[gid_r, 1])

            if py_l - py_r < target_dist:
                half_t = target_dist / 2.0
                center_y = float(np.clip((py_l + py_r) / 2.0, -0.45 + half_t, 0.45 - half_t))
                py_l = center_y + half_t
                py_r = center_y - half_t
                model.geom_pos[gid_l, 1] = py_l
                model.geom_pos[gid_r, 1] = py_r

                # Re-verify spawn distance if py adjustment shifted pillar inward
                for gid, py_val in [(gid_l, py_l), (gid_r, py_r)]:
                    r_p = float(model.geom_size[gid, 0])
                    req = min_spawn_dist + r_p
                    px_val = float(model.geom_pos[gid, 0])
                    if np.hypot(px_val, py_val) < req:
                        rem_sq = req**2 - py_val**2
                        if rem_sq > 0.0:
                            sign_x = 1.0 if px_val >= 0 else -1.0
                            model.geom_pos[gid, 0] = sign_x * float(np.sqrt(rem_sq))

    return active_pillars


STAGE_MIN_DISTANCE_X: Dict[int, float] = {
    0: 0.8,   # Stage 0 -> 1: Flat floor clearance >= 0.8m (prevent overfitting to flat floor)
    1: 1.2,   # Stage 1 -> 2: 5mm stair clearance >= 1.2m
    2: 1.5,   # Stage 2 -> 3: 10mm stair clearance >= 1.5m
    3: 1.8,   # Stage 3: 15mm stair clearance onto top platform >= 1.8m
}


def next_stage(
    stage: int,
    ep_len_mean: float,
    rew_lin_vel_ratio: float = 0.0,
    hold_count: int = 0,
    hold_iters: int = 5,
    rew_vx_ratio: Optional[float] = None,
    mean_distance_x: float = 0.0,
    min_distance_x: float = 0.8,
    success_rate: float = 1.0,
    min_success_rate: float = 0.70,
    mean_distance_y: float = 0.0,
    max_lateral_drift_y: float = 0.25,
) -> Tuple[int, int]:
    """Evaluate whether curriculum should advance to the next stage.

    Requires maintaining success criteria for `hold_iters` consecutive iterations.
    Curriculum strictly moves forward and never regresses.
    Mandatory physical condition: robot must actually traverse forward along X
    (mean_distance_x >= stage_target_dist), cohort success rate must be >= min_success_rate
    (>= 70% of robot episodes reach stage_target_dist), and lateral drift must be small
    (|mean_distance_y| < max_lateral_drift_y). Standing still or spinning sideways is strictly
    forbidden from advancing stages.

    Returns:
        (new_stage, new_hold_count)
    """
    if rew_vx_ratio is not None:
        rew_lin_vel_ratio = rew_vx_ratio
    if stage >= len(STAGES) - 1:
        return stage, hold_count

    # Stage-appropriate minimum forward distance:
    # If caller leaves min_distance_x at default (0.8), use STAGE_MIN_DISTANCE_X mapping.
    # Otherwise, honor explicit caller-provided threshold.
    if min_distance_x == 0.8:
        stage_target_dist = STAGE_MIN_DISTANCE_X.get(stage, STAGE_MIN_DISTANCE_X[max(STAGE_MIN_DISTANCE_X)])
    else:
        stage_target_dist = min_distance_x

    dist_met = (mean_distance_x >= stage_target_dist)
    success_met = (success_rate >= min_success_rate)
    lateral_met = (abs(float(mean_distance_y)) < float(max_lateral_drift_y))
    condition_met = False
    if stage == 0:
        # Stage 0 -> 1: ep_len_mean >= 400.0, forward distance >= 0.8m, cohort success_rate >= 0.70, lateral drift < 0.25m
        condition_met = (ep_len_mean >= 400.0) and dist_met and success_met and lateral_met
    elif stage == 1:
        # Stage 1 -> 2: sustained walking and slalom navigation past stage_target_dist, cohort success_rate >= 0.70
        condition_met = (ep_len_mean >= 420.0) and dist_met and success_met and lateral_met
    elif stage == 2:
        # Stage 2 -> 3: sustained stepping and stair negotiation past stage_target_dist, cohort success_rate >= 0.70
        condition_met = (ep_len_mean >= 420.0) and dist_met and success_met and lateral_met

    if condition_met:
        new_hold = hold_count + 1
        if new_hold >= hold_iters:
            return stage + 1, 0
        return stage, new_hold
    else:
        return stage, 0


# Backward compatibility aliases for training/interactive scripts
apply_stair_stage = apply_stage


def next_stair_stage(
    stage: int,
    ep_len_mean: float,
    rew_vx_ratio: Optional[float] = None,
    hold_count: int = 0,
    hold_iters: int = 5,
    rew_lin_vel_ratio: Optional[float] = None,
    mean_distance_x: float = 0.0,
    min_distance_x: float = 0.8,
    success_rate: float = 1.0,
    min_success_rate: float = 0.70,
    mean_distance_y: float = 0.0,
    max_lateral_drift_y: float = 0.25,
) -> Tuple[int, int]:
    """Progression evaluator for stair curriculum supporting rew_vx_ratio keyword."""
    vel_ratio = (
        rew_vx_ratio
        if rew_vx_ratio is not None
        else (rew_lin_vel_ratio if rew_lin_vel_ratio is not None else 0.0)
    )
    return next_stage(
        stage=stage,
        ep_len_mean=ep_len_mean,
        rew_lin_vel_ratio=vel_ratio,
        hold_count=hold_count,
        hold_iters=hold_iters,
        mean_distance_x=mean_distance_x,
        min_distance_x=min_distance_x,
        success_rate=success_rate,
        min_success_rate=min_success_rate,
        mean_distance_y=mean_distance_y,
        max_lateral_drift_y=max_lateral_drift_y,
    )


@dataclass(frozen=True)
class StairStage:
    stage_id: int
    step_height: float
    stair_width: float
    description: str


STAIR_STAGES: List[StairStage] = [
    StairStage(
        stage_id=i,
        step_height=max(0.0, s.stair_step_h),
        stair_width=0.800,
        description=s.description,
    )
    for i, s in enumerate(STAGES)
]
