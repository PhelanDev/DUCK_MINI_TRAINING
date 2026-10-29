"""Terrain and Obstacle Curriculum Manager for MicroDuck 8-DOF.

Controls progression of difficulty by:
1. Scaling obstacle pillar distance (XY scaling factor: 2.0x -> 1.3x -> 1.0x).
2. Gradually raising submerged ground hurdles (pos_z: submerged -> 0.005m -> 0.015m -> 0.025m).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import mujoco


@dataclass(frozen=True)
class Stage:
    stage_id: int
    pillar_scale: float
    hurdle_z: float
    description: str


STAGES: List[Stage] = [
    Stage(
        stage_id=0,
        pillar_scale=2.00,
        hurdle_z=-0.020,
        description="Stage 0: Wide pillars (2.0x), completely submerged hurdles",
    ),
    Stage(
        stage_id=1,
        pillar_scale=1.30,
        hurdle_z=0.005,
        description="Stage 1: Medium pillars (1.3x), low hurdle elevation (+5mm)",
    ),
    Stage(
        stage_id=2,
        pillar_scale=1.00,
        hurdle_z=0.015,
        description="Stage 2: Default pillars (1.0x), medium hurdle elevation (+15mm)",
    ),
    Stage(
        stage_id=3,
        pillar_scale=1.00,
        hurdle_z=0.025,
        description="Stage 3: Full challenge (1.0x pillars, +25mm high hurdles)",
    ),
]

# Baseline XML positions of the 4 obstacle cylinders
_BASE_PILLAR_POS: Dict[str, Tuple[float, float, float]] = {
    "obstacle_front": (0.85, 0.0, 0.25),
    "obstacle_left": (0.65, 0.35, 0.25),
    "obstacle_right": (0.65, -0.35, 0.25),
    "obstacle_back": (-0.75, 0.0, 0.25),
}


def apply_stage(model: mujoco.MjModel, stage: int) -> None:
    """Apply curriculum geometry parameters to MuJoCo model geom_pos.

    Modifies the shared model in-place; all parallel environments immediately observe
    the updated obstacle and hurdle coordinates. Idempotent across repeated calls.
    """
    stage_idx = max(0, min(stage, len(STAGES) - 1))
    cfg = STAGES[stage_idx]

    # 1. Update Obstacle Pillar Positions
    for name, base_pos in _BASE_PILLAR_POS.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            model.geom_pos[gid, 0] = base_pos[0] * cfg.pillar_scale
            model.geom_pos[gid, 1] = base_pos[1] * cfg.pillar_scale
            model.geom_pos[gid, 2] = base_pos[2]

    # 2. Update Hurdle Heights (pos_z)
    for h_name in ["hurdle_1", "hurdle_2", "hurdle_back"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, h_name)
        if gid != -1:
            if stage_idx == 0:
                # Submerged below floor (top surface <= 0.0)
                # hurdle_2 has half_z = 0.020, so pos_z must be -0.025 to keep top at -0.005m
                model.geom_pos[gid, 2] = -0.025 if h_name == "hurdle_2" else -0.020
            else:
                model.geom_pos[gid, 2] = cfg.hurdle_z


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
        # Stage 1 -> 2: sustained walking proficiency
        condition_met = ep_len_mean >= 420.0
    elif stage == 2:
        # Stage 2 -> 3: sustained stepping over hurdles
        condition_met = ep_len_mean >= 420.0

    if condition_met:
        new_hold = hold_count + 1
        if new_hold >= hold_iters:
            return stage + 1, 0
        return stage, new_hold
    else:
        return stage, 0

