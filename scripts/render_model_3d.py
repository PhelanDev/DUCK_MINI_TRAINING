#!/usr/bin/env python3
"""Render high-resolution 3D views of the official Pollen Robotics MicroDuck 8-DOF CAD model."""

import os
import shutil
import mujoco
import numpy as np
from PIL import Image

def render_sculpted_duck():
    # Prefer official CAD model if available, fallback to 8-DOF physical model
    cad_xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "microduck_cad.xml")
    if not os.path.exists(cad_xml_path):
        cad_xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "microduck_8dof.xml")

    model = mujoco.MjModel.from_xml_path(cad_xml_path)
    data = mujoco.MjData(model)

    # Initial forward pass to compute kinematic transforms
    mujoco.mj_forward(model, data)

    # Align upright standing posture so both feet touch floor plane exactly at Z=0
    foot_left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")
    if foot_left_body >= 0:
        z_foot = data.xpos[foot_left_body][2]
        data.qpos[2] += (0.015 - z_foot)
    else:
        foot_left_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
        if foot_left_geom >= 0:
            z_bottom = data.geom_xpos[foot_left_geom][2] - 0.004
            data.qpos[2] -= z_bottom
        else:
            data.qpos[2] = 0.127

    # Keep all 8 joints at nominal zero stand pose
    data.qpos[3:7] = [1, 0, 0, 0] # upright orientation
    if len(data.qpos) > 7:
        data.qpos[7:] = 0.0
    mujoco.mj_forward(model, data)

    media_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "media")
    os.makedirs(media_dir, exist_ok=True)
    brain_dir = r"C:\Users\LEGION\.gemini\antigravity\brain\eac089d5-7756-4b70-896e-d09247ae73c7"

    renderer = mujoco.Renderer(model, width=640, height=480)

    # 1. Isometric 3D View
    cam_iso = mujoco.MjvCamera()
    cam_iso.lookat = [0.0, 0.0, 0.12]
    cam_iso.distance = 0.48
    cam_iso.azimuth = 135
    cam_iso.elevation = -18
    renderer.update_scene(data, camera=cam_iso)
    img_iso = renderer.render()
    iso_path = os.path.join(media_dir, "microduck_iso_3d.png")
    Image.fromarray(img_iso).save(iso_path)
    if os.path.exists(brain_dir):
        shutil.copy2(iso_path, os.path.join(brain_dir, "microduck_iso_3d.png"))

    # 2. Front 3D View
    cam_front = mujoco.MjvCamera()
    cam_front.lookat = [0.0, 0.0, 0.12]
    cam_front.distance = 0.45
    cam_front.azimuth = 180
    cam_front.elevation = -10
    renderer.update_scene(data, camera=cam_front)
    img_front = renderer.render()
    front_path = os.path.join(media_dir, "microduck_front.png")
    Image.fromarray(img_front).save(front_path)
    if os.path.exists(brain_dir):
        shutil.copy2(front_path, os.path.join(brain_dir, "microduck_front.png"))

    # 3. Side 3D View
    cam_side = mujoco.MjvCamera()
    cam_side.lookat = [0.0, 0.0, 0.12]
    cam_side.distance = 0.45
    cam_side.azimuth = 90
    cam_side.elevation = -10
    renderer.update_scene(data, camera=cam_side)
    img_side = renderer.render()
    side_path = os.path.join(media_dir, "microduck_side.png")
    Image.fromarray(img_side).save(side_path)
    if os.path.exists(brain_dir):
        shutil.copy2(side_path, os.path.join(brain_dir, "microduck_side.png"))

    # 4. Foot Close-up View (showing precision flat sole on floor)
    cam_foot = mujoco.MjvCamera()
    cam_foot.lookat = [-0.03, 0.06, 0.015]
    cam_foot.distance = 0.16
    cam_foot.azimuth = 135
    cam_foot.elevation = -14
    renderer.update_scene(data, camera=cam_foot)
    img_foot = renderer.render()
    foot_path = os.path.join(media_dir, "microduck_foot_closeup.png")
    Image.fromarray(img_foot).save(foot_path)
    if os.path.exists(brain_dir):
        shutil.copy2(foot_path, os.path.join(brain_dir, "microduck_foot_closeup.png"))

    print(f"Official Microduck CAD Renders successfully generated in {media_dir} and synced to {brain_dir}!")

if __name__ == "__main__":
    render_sculpted_duck()
