"""Terrain and Obstacle Curriculum Manager for MicroDuck 8-DOF.

Controls progression of difficulty by:
1. Scaling slalom pillar distance (XY scaling factor: 1.80x -> 1.35x -> 1.10x -> 1.00x).
2. Gradually raising staircase steps & landing platform (stair_step_h: submerged -> 0.005m -> 0.010m -> 0.015m).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import mujoco


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
        stair_step_h=0.005,
        description="Stage 1: Medium slalom (1.35x), low stairs (+5mm/step)",
    ),
    Stage(
        stage_id=2,
        pillar_scale=1.10,
        stair_step_h=0.010,
        description="Stage 2: Narrow slalom (1.1x), medium stairs (+10mm/step)",
    ),
    Stage(
        stage_id=3,
        pillar_scale=1.00,
        stair_step_h=0.015,
        description="Stage 3: Full challenge (1.0x slalom, +15mm/step stairs, +45mm landing)",
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


def next_stage(
    stage: int,
    ep_len_mean: float,
    rew_lin_vel_ratio: float,
    hold_count: int,
    hold_iters: int = 10,
) -> Tuple[int, int]:
    """Evaluate whether curriculum should advance to the next stage.

    Requires maintaining success criteria for `hold_iters` consecutive iterations.
    Curriculum strictly moves forward and never regresses.

    Returns:
        (new_stage, new_hold_count)
    """
    if stage >= len(STAGES) - 1:
        return stage, hold_count

    condition_met = False
    if stage == 0:
        # Stage 0 -> 1: ep_len_mean >= 400 and linear velocity reward >= 70% of max
        condition_met = (ep_len_mean >= 400.0) and (rew_lin_vel_ratio >= 0.70)
    elif stage == 1:
        # Stage 1 -> 2: sustained walking and slalom navigation
        condition_met = (ep_len_mean >= 420.0) and (rew_lin_vel_ratio >= 0.70)
    elif stage == 2:
        # Stage 2 -> 3: sustained stepping and stair negotiation
        condition_met = (ep_len_mean >= 420.0) and (rew_lin_vel_ratio >= 0.70)

    if condition_met:
        new_hold = hold_count + 1
        if new_hold >= hold_iters:
            return stage + 1, 0
        return stage, new_hold
    else:
        return stage, 0


# Backward compatibility aliases for training/interactive scripts
apply_stair_stage = apply_stage
next_stair_stage = next_stage


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
