"""
Export CyberDuck V2 CAD Assembly to standard OBJ, MTL, Exploded OBJ, and STL ZIP.
"""
import os
import shutil
import zipfile
import mujoco
import numpy as np

def export_v2_cad():
    cad_dir = "duck_v2/cad"
    os.makedirs(cad_dir, exist_ok=True)
    brain_dir = os.environ.get("BRAIN_DIR", os.environ.get("GEMINI_BRAIN_DIR", ""))
    
    xml_path = "duck_v2/cyberduck_v2.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    materials = {}
    for i in range(model.nmat):
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        rgba = model.mat_rgba[i]
        materials[mat_name] = rgba
    materials['default_mat'] = np.array([0.5, 0.5, 0.5, 1.0])

    com = data.subtree_com[model.body("trunk_base").id].copy()
    geoms = []

    for i in range(model.ngeom):
        if model.geom_group[i] == 2:
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            mat_id = model.geom_matid[i]
            mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) if mat_id >= 0 else 'default_mat'
            pos = data.geom_xpos[i].copy()
            rot = data.geom_xmat[i].reshape(3, 3).copy()
            
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = model.geom_dataid[i]
                mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                vert_adr = model.mesh_vertadr[mesh_id]
                vert_num = model.mesh_vertnum[mesh_id]
                face_adr = model.mesh_faceadr[mesh_id]
                face_num = model.mesh_facenum[mesh_id]
                v_local = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
                faces = model.mesh_face[face_adr : face_adr + face_num].copy()
            elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                mesh_name = f"box_{body_name}"
                s = model.geom_size[i]
                dx, dy, dz = s[0], s[1], s[2]
                v_local = np.array([
                    [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
                    [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
                ], dtype=np.float32)
                faces = np.array([
                    [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                    [0, 5, 1], [0, 4, 5], [2, 7, 3], [2, 6, 7],
                    [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
                ], dtype=np.int32)
            elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                mesh_name = f"cyl_{body_name}"
                r, half_l = model.geom_size[i][0], model.geom_size[i][1]
                theta = np.linspace(0, 2*np.pi, 24)
                v_bot = np.stack([r*np.cos(theta), r*np.sin(theta), np.full_like(theta, -half_l)], axis=-1)
                v_top = np.stack([r*np.cos(theta), r*np.sin(theta), np.full_like(theta, half_l)], axis=-1)
                v_local = np.vstack([v_bot, v_top])
                faces = []
                for j in range(23):
                    faces.append([j, j+1, j+24])
                    faces.append([j+1, j+25, j+24])
                faces = np.array(faces, dtype=np.int32)
            else:
                continue

            v_world = v_local @ rot.T + pos
            disp = pos - com
            dist = np.linalg.norm(disp)
            dir_vec = disp / (dist + 1e-5)
            if "head" in body_name or "neck" in body_name:
                dir_vec = np.array([0.0, 0.0, 1.2]) + dir_vec * 0.4
            elif "foot" in body_name or "sole" in body_name:
                dir_vec = np.array([0.0, 0.0, -1.0]) + np.array([dir_vec[0], dir_vec[1], 0.0]) * 0.6
            elif "wing" in mesh_name or "left" in body_name:
                dir_vec = np.array([0.0, 1.2, 0.0]) + dir_vec * 0.3
            elif "right" in body_name:
                dir_vec = np.array([0.0, -1.2, 0.0]) + dir_vec * 0.3
            elif "thruster" in mesh_name:
                dir_vec = np.array([-1.5, 0.0, 0.0])
            dir_vec = dir_vec / np.linalg.norm(dir_vec)

            geoms.append({
                'name': mesh_name,
                'mat_name': mat_name,
                'verts': v_world,
                'faces': faces,
                'dir_vec': dir_vec
            })

    # 1. Write MTL
    mtl_p = os.path.join(cad_dir, "cyberduck_v2_assembly.mtl")
    with open(mtl_p, "w", encoding="utf-8") as f:
        f.write("# Materials for CyberDuck V2\n\n")
        for mat_name, rgba in materials.items():
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Ka {rgba[0]*0.2:.4f} {rgba[1]*0.2:.4f} {rgba[2]*0.2:.4f}\n")
            f.write(f"Kd {rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f}\n")
            f.write(f"Ks 0.4000 0.4000 0.4000\n")
            f.write(f"Ns 40.0\n")
            f.write(f"d {rgba[3]:.4f}\n")
            f.write(f"illum 2\n\n")

    # 2. Write Assembled OBJ
    obj_p = os.path.join(cad_dir, "cyberduck_v2_assembly.obj")
    with open(obj_p, "w", encoding="utf-8") as f:
        f.write("# CyberDuck V2 Complete CAD Assembly\n")
        f.write("mtllib cyberduck_v2_assembly.mtl\n\n")
        vert_offset = 0
        for idx, g in enumerate(geoms):
            f.write(f"o {g['name']}_{idx+1:02d}\n")
            f.write(f"usemtl {g['mat_name']}\n")
            for v in g['verts']:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in g['faces']:
                f.write(f"f {face[0]+vert_offset+1} {face[1]+vert_offset+1} {face[2]+vert_offset+1}\n")
            vert_offset += len(g['verts'])
            f.write("\n")
    print(f"Exported: {obj_p} ({os.path.getsize(obj_p)/1024/1024:.2f} MB)")

    # 3. Write Exploded OBJ
    exp_p = os.path.join(cad_dir, "cyberduck_v2_exploded.obj")
    with open(exp_p, "w", encoding="utf-8") as f:
        f.write("# CyberDuck V2 Exploded View\n")
        f.write("mtllib cyberduck_v2_assembly.mtl\n\n")
        vert_offset = 0
        exp_dist = 0.065
        for idx, g in enumerate(geoms):
            f.write(f"o {g['name']}_{idx+1:02d}\n")
            f.write(f"usemtl {g['mat_name']}\n")
            offset = g['dir_vec'] * exp_dist
            for v in g['verts']:
                ve = v + offset
                f.write(f"v {ve[0]:.6f} {ve[1]:.6f} {ve[2]:.6f}\n")
            for face in g['faces']:
                f.write(f"f {face[0]+vert_offset+1} {face[1]+vert_offset+1} {face[2]+vert_offset+1}\n")
            vert_offset += len(g['verts'])
            f.write("\n")
    print(f"Exported: {exp_p} ({os.path.getsize(exp_p)/1024/1024:.2f} MB)")

    # 4. Package STL files
    zip_p = os.path.join(cad_dir, "cyberduck_v2_stl_parts.zip")
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("duck_v2/meshes"):
            for file in files:
                if file.endswith(".stl"):
                    full_p = os.path.join(root, file)
                    zf.write(full_p, arcname=file)
    print(f"Exported: {zip_p} ({os.path.getsize(zip_p)/1024/1024:.2f} MB)")

    # Copy to brain dir
    if brain_dir and os.path.exists(brain_dir):
        shutil.copy2(obj_p, os.path.join(brain_dir, "cyberduck_v2_assembly.obj"))
        shutil.copy2(mtl_p, os.path.join(brain_dir, "cyberduck_v2_assembly.mtl"))
        shutil.copy2(exp_p, os.path.join(brain_dir, "cyberduck_v2_exploded.obj"))
        shutil.copy2(zip_p, os.path.join(brain_dir, "cyberduck_v2_stl_parts.zip"))

if __name__ == "__main__":
    export_v2_cad()
