"""
Render high-resolution studio 3D views of CyberDuck V2 with pristine studio PBR lighting.
"""
import os
import shutil
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image

def render_v2():
    repo_root = Path(__file__).resolve().parent.parent.parent
    v2_dir = repo_root / "duck_v2"
    media_dir = v2_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    xml_path = v2_dir / "cyberduck_v2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Compute transforms
    mujoco.mj_forward(model, data)

    # Align standing posture so soles touch ground at z=0.000m
    sole_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_sole_geom")
    z_bottom = data.geom_xpos[sole_geom][2] - 0.004
    data.qpos[2] -= z_bottom
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # Configure studio visual options: hide collision hulls (grp 3) and sensory sites (grp 4)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    opt.geomgroup[2] = 1
    opt.geomgroup[3] = 0
    opt.sitegroup[4] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0

    renderer = mujoco.Renderer(model, width=1024, height=768)

    # 1. Isometric 3D
    cam_iso = mujoco.MjvCamera()
    cam_iso.lookat = [0.0, 0.0, 0.11]
    cam_iso.distance = 0.46
    cam_iso.azimuth = 135
    cam_iso.elevation = -18
    renderer.update_scene(data, camera=cam_iso, scene_option=opt)
    img_iso = renderer.render()
    p_iso = media_dir / "cyberduck_v2_iso.png"
    Image.fromarray(img_iso).save(str(p_iso))
    print(f"Rendered: {p_iso}")

    # 2. Front 3D
    cam_front = mujoco.MjvCamera()
    cam_front.lookat = [0.0, 0.0, 0.10]
    cam_front.distance = 0.44
    cam_front.azimuth = 180
    cam_front.elevation = -10
    renderer.update_scene(data, camera=cam_front, scene_option=opt)
    img_front = renderer.render()
    p_front = media_dir / "cyberduck_v2_front.png"
    Image.fromarray(img_front).save(str(p_front))
    print(f"Rendered: {p_front}")

    # 3. Side Profile
    cam_side = mujoco.MjvCamera()
    cam_side.lookat = [0.0, 0.0, 0.10]
    cam_side.distance = 0.44
    cam_side.azimuth = 90
    cam_side.elevation = -10
    renderer.update_scene(data, camera=cam_side, scene_option=opt)
    img_side = renderer.render()
    p_side = media_dir / "cyberduck_v2_side.png"
    Image.fromarray(img_side).save(str(p_side))
    print(f"Rendered: {p_side}")

    # 4. Chest & Reactor Closeup
    cam_wing = mujoco.MjvCamera()
    cam_wing.lookat = [0.010, 0.000, 0.11]
    cam_wing.distance = 0.22
    cam_wing.azimuth = 145
    cam_wing.elevation = -12
    renderer.update_scene(data, camera=cam_wing, scene_option=opt)
    img_wing = renderer.render()
    p_wing = media_dir / "cyberduck_v2_wing_closeup.png"
    Image.fromarray(img_wing).save(str(p_wing))
    print(f"Rendered: {p_wing}")

    # 5. Foot Closeup
    cam_foot = mujoco.MjvCamera()
    cam_foot.lookat = [0.005, 0.040, 0.025]
    cam_foot.distance = 0.20
    cam_foot.azimuth = 145
    cam_foot.elevation = -22
    renderer.update_scene(data, camera=cam_foot, scene_option=opt)
    img_foot = renderer.render()
    p_foot = media_dir / "cyberduck_v2_foot_closeup.png"
    Image.fromarray(img_foot).save(str(p_foot))
    print(f"Rendered: {p_foot}")

    # 5b. Neck Joint Closeup
    cam_neck = mujoco.MjvCamera()
    cam_neck.lookat = [0.010, 0.000, 0.150]
    cam_neck.distance = 0.16
    cam_neck.azimuth = 145
    cam_neck.elevation = -5
    renderer.update_scene(data, camera=cam_neck, scene_option=opt)
    img_neck = renderer.render()
    p_neck = media_dir / "cyberduck_v2_neck_closeup.png"
    Image.fromarray(img_neck).save(str(p_neck))
    print(f"Rendered: {p_neck}")

    # 6. Exploded View 3D
    com = data.subtree_com[model.body("trunk_base").id].copy()
    for i in range(model.ngeom):
        if model.geom_group[i] == 2:
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            pos = data.geom_xpos[i]
            disp = pos - com
            dist = np.linalg.norm(disp)
            dir_vec = disp / (dist + 1e-5)
            if "head" in body_name or "neck" in body_name:
                dir_vec = np.array([0.0, 0.0, 1.2]) + dir_vec * 0.4
            elif "foot" in body_name or "sole" in body_name:
                dir_vec = np.array([0.0, 0.0, -1.0]) + np.array([dir_vec[0], dir_vec[1], 0.0]) * 0.6
            elif "left" in body_name:
                dir_vec = np.array([0.0, 1.2, 0.0]) + dir_vec * 0.3
            elif "right" in body_name:
                dir_vec = np.array([0.0, -1.2, 0.0]) + dir_vec * 0.3
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            model.geom_pos[i] += dir_vec * 0.045
            
    mujoco.mj_forward(model, data)
    cam_exp = mujoco.MjvCamera()
    cam_exp.lookat = [0.0, 0.0, 0.12]
    cam_exp.distance = 0.52
    cam_exp.azimuth = 135
    cam_exp.elevation = -18
    renderer.update_scene(data, camera=cam_exp, scene_option=opt)
    img_exp = renderer.render()
    p_exp = media_dir / "cyberduck_v2_exploded.png"
    Image.fromarray(img_exp).save(str(p_exp))
    print(f"Rendered: {p_exp}")

if __name__ == "__main__":
    render_v2()
