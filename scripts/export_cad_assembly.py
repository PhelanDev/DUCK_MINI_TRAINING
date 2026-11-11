"""
Export MicroDuck 8-DOF CAD Assembly to standard OBJ + MTL and Exploded OBJ format.
Extracts all 70 visual mesh instances from models/mjcf/microduck_cad.xml.
"""
import os
import zipfile
import mujoco
import numpy as np

def export_assembly():
    os.makedirs("assets/cad", exist_ok=True)
    xml_path = "models/mjcf/microduck_cad.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Material definitions
    materials = {}
    for i in range(model.nmat):
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        rgba = model.mat_rgba[i]
        materials[mat_name] = rgba

    # Add default material
    materials['default_material'] = np.array([0.7, 0.7, 0.7, 1.0])

    # Collect visual mesh geoms
    geoms = []
    # Center of mass of the whole robot for exploded view calculation
    com = data.subtree_com[model.body("trunk_base").id].copy()

    for i in range(model.ngeom):
        if model.geom_group[i] == 2 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            mesh_id = model.geom_dataid[i]
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            mat_id = model.geom_matid[i]
            mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) if mat_id >= 0 else 'default_material'
            
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]
            
            v_local = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
            faces = model.mesh_face[face_adr : face_adr + face_num].copy()
            
            pos = data.geom_xpos[i].copy()
            rot = data.geom_xmat[i].reshape(3, 3).copy()
            
            v_world = v_local @ rot.T + pos
            
            # Exploded displacement vector based on body and position relative to COM
            disp = pos - com
            # Normalize and scale
            dist = np.linalg.norm(disp)
            if dist > 1e-4:
                dir_vec = disp / dist
            else:
                dir_vec = np.array([0.0, 0.0, 1.0])
                
            # Customize explosion direction for aesthetics
            if "head" in body_name or "neck" in body_name or "jaw" in body_name:
                dir_vec = np.array([0.0, 0.0, 1.5]) + dir_vec * 0.5
            elif "foot" in body_name or "sole" in body_name or "ankle" in body_name:
                dir_vec = np.array([0.0, 0.0, -1.2]) + np.array([dir_vec[0], dir_vec[1], 0.0]) * 0.8
            elif "left" in body_name or "_l" in body_name:
                dir_vec = np.array([0.0, 1.2, 0.0]) + dir_vec * 0.4
            elif "right" in body_name or "_r" in body_name:
                dir_vec = np.array([0.0, -1.2, 0.0]) + dir_vec * 0.4
            elif "np_f970" in mesh_name:
                dir_vec = np.array([-1.5, 0.0, 0.0])
                
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            geoms.append({
                'geom_id': i,
                'body_name': body_name,
                'mesh_name': mesh_name,
                'mat_name': mat_name,
                'verts': v_world,
                'faces': faces,
                'pos': pos,
                'dir_vec': dir_vec,
                'dim_mm': (v_local.max(axis=0) - v_local.min(axis=0)) * 1000.0
            })

    print(f"Collected {len(geoms)} visual geoms.")

    # 1. Write MTL file
    mtl_path = "assets/cad/microduck_assembly.mtl"
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("# Material definitions for MicroDuck 8-DOF CAD Assembly\n")
        f.write("# Generated from models/mjcf/microduck_cad.xml\n\n")
        for mat_name, rgba in materials.items():
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Ka {rgba[0]*0.2:.4f} {rgba[1]*0.2:.4f} {rgba[2]*0.2:.4f}\n")
            f.write(f"Kd {rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f}\n")
            f.write(f"Ks 0.2500 0.2500 0.2500\n")
            f.write(f"Ns 32.0\n")
            f.write(f"d {rgba[3]:.4f}\n")
            f.write(f"illum 2\n\n")

    # 2. Write Assembled OBJ file
    obj_path = "assets/cad/microduck_assembly.obj"
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# MicroDuck 8-DOF CAD Complete Assembly\n")
        f.write(f"# Total parts: {len(geoms)}\n")
        f.write(f"mtllib microduck_assembly.mtl\n\n")
        
        vert_offset = 0
        for idx, g in enumerate(geoms):
            f.write(f"# Part {idx+1}: {g['mesh_name']} (body: {g['body_name']})\n")
            f.write(f"o {g['mesh_name']}_{idx+1:02d}\n")
            f.write(f"usemtl {g['mat_name']}\n")
            
            # Write vertices
            for v in g['verts']:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
            # Write faces (1-based index)
            for face in g['faces']:
                f.write(f"f {face[0]+vert_offset+1} {face[1]+vert_offset+1} {face[2]+vert_offset+1}\n")
                
            vert_offset += len(g['verts'])
            f.write("\n")

    print(f"Exported assembled OBJ: {obj_path} ({os.path.getsize(obj_path)/1024/1024:.2f} MB)")

    # 3. Write Exploded OBJ file (explosion factor = 0.08m = 80mm displacement)
    exploded_obj_path = "assets/cad/microduck_exploded_assembly.obj"
    with open(exploded_obj_path, "w", encoding="utf-8") as f:
        f.write("# MicroDuck 8-DOF CAD Exploded View Assembly\n")
        f.write(f"# Total parts: {len(geoms)}\n")
        f.write(f"mtllib microduck_assembly.mtl\n\n")
        
        vert_offset = 0
        explode_dist = 0.075  # 75mm explosion
        for idx, g in enumerate(geoms):
            f.write(f"# Part {idx+1}: {g['mesh_name']} (body: {g['body_name']})\n")
            f.write(f"o {g['mesh_name']}_{idx+1:02d}\n")
            f.write(f"usemtl {g['mat_name']}\n")
            
            offset = g['dir_vec'] * explode_dist
            for v in g['verts']:
                v_exp = v + offset
                f.write(f"v {v_exp[0]:.6f} {v_exp[1]:.6f} {v_exp[2]:.6f}\n")
                
            for face in g['faces']:
                f.write(f"f {face[0]+vert_offset+1} {face[1]+vert_offset+1} {face[2]+vert_offset+1}\n")
                
            vert_offset += len(g['verts'])
            f.write("\n")

    print(f"Exported exploded OBJ: {exploded_obj_path} ({os.path.getsize(exploded_obj_path)/1024/1024:.2f} MB)")

    # 4. Create ZIP package of all 43 STL parts
    zip_path = "assets/cad/microduck_all_stl_parts.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("assets/meshes"):
            for file in files:
                if file.endswith(".stl"):
                    full_p = os.path.join(root, file)
                    zf.write(full_p, arcname=file)
    print(f"Exported all STL parts ZIP: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.2f} MB)")

    return geoms, materials

if __name__ == "__main__":
    export_assembly()
