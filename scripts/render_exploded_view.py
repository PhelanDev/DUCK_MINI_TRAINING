"""
Render an exploded 3D view of the MicroDuck CAD model using MuJoCo.
"""
import os
import shutil
import mujoco
import numpy as np
from PIL import Image

def generate_exploded_mjcf():
    base_xml = "models/mjcf/microduck_cad.xml"
    with open(base_xml, "r", encoding="utf-8") as f:
        content = f.read()

    # Load model to get geom positions and body hierarchy
    model = mujoco.MjModel.from_xml_path(base_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com = data.subtree_com[model.body("trunk_base").id].copy()

    # We can create a dedicated exploded visual XML where trunk_base has freejoint
    # and all bodies have expanded offsets
    # Simpler: modify render_exploded_view by reading microduck_cad.xml,
    # adding visual offsets to geoms or bodies!
    pass

def render_exploded():
    xml_path = "models/mjcf/microduck_cad.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Let's displace bodies/geoms in data or model
    com = data.subtree_com[model.body("trunk_base").id].copy()
    
    # In MuJoCo, data.geom_xpos is computed from forward kinematics,
    # but model.geom_pos can be modified directly before forward!
    for i in range(model.ngeom):
        if model.geom_group[i] == 2 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            mesh_id = model.geom_dataid[i]
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)

            pos = data.geom_xpos[i]
            disp = pos - com
            dist = np.linalg.norm(disp)
            dir_vec = disp / (dist + 1e-5)

            # Explosion directions
            if any(k in body_name for k in ["head", "neck", "jaw"]):
                dir_vec = np.array([0.0, 0.0, 1.2]) + dir_vec * 0.3
            elif any(k in body_name for k in ["foot", "sole", "ankle"]):
                dir_vec = np.array([0.0, 0.0, -1.0]) + np.array([dir_vec[0], dir_vec[1], 0.0]) * 0.6
            elif "_l" in body_name or "left" in body_name:
                dir_vec = np.array([0.0, 1.0, 0.0]) + dir_vec * 0.3
            elif "_r" in body_name or "right" in body_name:
                dir_vec = np.array([0.0, -1.0, 0.0]) + dir_vec * 0.3
            elif "np_f970" in mesh_name:
                dir_vec = np.array([-1.2, 0.0, 0.0])
                
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            # Apply displacement in geom_pos (local frame)
            # Since geom is attached to body, let's displace body pos or geom pos
            model.geom_pos[i] += dir_vec * 0.045  # 45mm expansion

    # Forward to recompute transforms
    mujoco.mj_forward(model, data)

    media_dir = "assets/media"
    brain_dir = os.environ.get("BRAIN_DIR", os.environ.get("GEMINI_BRAIN_DIR", ""))

    renderer = mujoco.Renderer(model, width=640, height=480)
    
    cam = mujoco.MjvCamera()
    cam.lookat = [0.0, 0.0, 0.13]
    cam.distance = 0.55
    cam.azimuth = 135
    cam.elevation = -20
    renderer.update_scene(data, camera=cam)
    img = renderer.render()

    out_path = os.path.join(media_dir, "microduck_exploded_3d.png")
    Image.fromarray(img).save(out_path)
    if brain_dir and os.path.exists(brain_dir):
        shutil.copy2(out_path, os.path.join(brain_dir, "microduck_exploded_3d.png"))

    print(f"Rendered exploded view: {out_path}")

if __name__ == "__main__":
    render_exploded()
