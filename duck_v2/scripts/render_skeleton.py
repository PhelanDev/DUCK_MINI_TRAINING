"""
Render HD Views of CyberDuck V2 Pro Mechanical Skeleton & Servo Chassis.
"""
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image

def render_skeleton_views():
    v2_dir = Path(__file__).resolve().parent.parent
    cad_dir = v2_dir / "cad"
    media_dir = v2_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # 1. XML specifically for high-definition skeleton rendering
    xml_skeleton = """
<mujoco model="cyberduck_v2_skeleton">
  <compiler angle="radian" meshdir="cad" autolimits="true" />
  <visual>
    <global offwidth="1024" offheight="768" />
    <quality shadowsize="4096" />
    <map shadowclip="0.5" shadowscale="0.6" />
    <headlight ambient="0.45 0.48 0.52" diffuse="0.65 0.65 0.70" specular="0.25 0.25 0.25" />
  </visual>

  <asset>
    <mesh name="skeleton_mesh" file="cyberduck_v2_skeleton.obj" />
    <material name="skeleton_titanium" rgba="0.45 0.52 0.60 1.0" specular="0.7" shininess="0.6" />
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.30 0.40" rgb2="0.10 0.12 0.16" width="512" height="512" />
    <texture name="grid_floor" type="2d" builtin="checker" width="512" height="512" rgb1="0.18 0.22 0.28" rgb2="0.12 0.15 0.20" />
    <material name="floor_mat" texture="grid_floor" texrepeat="20 20" reflectance="0.10" />
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.90 0.90 0.95" specular="0.3 0.3 0.3" pos="1.5 1.2 2.2" dir="-1.5 -1.2 -1.8" castshadow="true" />
    <light directional="true" diffuse="0.60 0.62 0.70" specular="0.2 0.2 0.2" pos="-1.5 -1.2 1.8" dir="1.5 1.2 -1.4" castshadow="false" />
    <light directional="false" diffuse="0.55 0.58 0.65" specular="0.1 0.1 0.1" pos="0 0 2.2" dir="0 0 -1" castshadow="false" />
    <geom name="floor" type="plane" size="5 5 0.1" material="floor_mat" friction="1.0" />

    <body name="skeleton_body" pos="0 0 0.125">
      <geom type="mesh" mesh="skeleton_mesh" material="skeleton_titanium" />
    </body>
  </worldbody>
</mujoco>
"""
    temp_xml_path = v2_dir / "temp_skeleton_render.xml"
    with open(temp_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_skeleton)

    model = mujoco.MjModel.from_xml_path(str(temp_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "skeleton_body")
    center = data.xpos[body_id].copy()

    renderer = mujoco.Renderer(model, 768, 1024)

    views = [
        ("cyberduck_v2_skeleton_iso.png", 140.0, -14.0, 0.46),
        ("cyberduck_v2_skeleton_front.png", 180.0, -8.0, 0.44),
        ("cyberduck_v2_skeleton_side.png", 90.0, -8.0, 0.44),
    ]

    for filename, azimuth, elevation, distance in views:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = center.copy()
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth

        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        out_path = media_dir / filename
        Image.fromarray(img).save(str(out_path))
        print(f"[OK] Rendered {out_path}")

    # 2. Cutaway / X-Ray composite: Skeleton inside semi-transparent outer armor
    xml_composite = """
<mujoco model="cyberduck_v2_composite">
  <compiler angle="radian" meshdir="cad" autolimits="true" />
  <visual>
    <global offwidth="1024" offheight="768" />
    <quality shadowsize="4096" />
    <headlight ambient="0.45 0.48 0.52" diffuse="0.65 0.65 0.70" specular="0.25 0.25 0.25" />
  </visual>

  <asset>
    <mesh name="skeleton_mesh" file="cyberduck_v2_skeleton.obj" />
    <mesh name="outer_armor" file="cyberduck_v2_assembly.obj" />
    <material name="skeleton_gold" rgba="0.85 0.58 0.22 1.0" specular="0.8" shininess="0.7" />
    <material name="glass_armor" rgba="0.35 0.38 0.42 0.35" specular="0.4" shininess="0.5" />
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.30 0.40" rgb2="0.10 0.12 0.16" width="512" height="512" />
    <texture name="grid_floor" type="2d" builtin="checker" width="512" height="512" rgb1="0.18 0.22 0.28" rgb2="0.12 0.15 0.20" />
    <material name="floor_mat" texture="grid_floor" texrepeat="20 20" reflectance="0.10" />
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.90 0.90 0.95" specular="0.3 0.3 0.3" pos="1.5 1.2 2.2" dir="-1.5 -1.2 -1.8" castshadow="true" />
    <light directional="true" diffuse="0.60 0.62 0.70" specular="0.2 0.2 0.2" pos="-1.5 -1.2 1.8" dir="1.5 1.2 -1.4" castshadow="false" />
    <geom name="floor" type="plane" size="5 5 0.1" material="floor_mat" friction="1.0" />

    <body name="composite_body" pos="0 0 0.125">
      <geom type="mesh" mesh="skeleton_mesh" material="skeleton_gold" />
      <geom type="mesh" mesh="outer_armor" material="glass_armor" />
    </body>
  </worldbody>
</mujoco>
"""
    with open(temp_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_composite)

    model_comp = mujoco.MjModel.from_xml_path(str(temp_xml_path))
    data_comp = mujoco.MjData(model_comp)
    mujoco.mj_forward(model_comp, data_comp)

    renderer_comp = mujoco.Renderer(model_comp, 768, 1024)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = center.copy()
    cam.distance = 0.48
    cam.elevation = -14.0
    cam.azimuth = 140.0

    renderer_comp.update_scene(data_comp, camera=cam)
    img_comp = renderer_comp.render()
    cutaway_path = media_dir / "cyberduck_v2_chassis_with_armor.png"
    Image.fromarray(img_comp).save(str(cutaway_path))
    print(f"[OK] Rendered Cutaway Composite: {cutaway_path}")

    if temp_xml_path.exists():
        temp_xml_path.unlink()

if __name__ == "__main__":
    render_skeleton_views()
