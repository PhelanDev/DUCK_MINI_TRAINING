from pathlib import Path
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
xml_path = str(REPO_ROOT / 'models' / 'mjcf' / 'microduck_cad.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

parts = []
for i in range(model.ngeom):
    if model.geom_group[i] == 2 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mesh_id = model.geom_dataid[i]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        mat_id = model.geom_matid[i]
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) if mat_id >= 0 else 'default'
        rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[i]
        
        vert_adr = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[vert_adr : vert_adr + vert_num]
        
        bbox_min = verts.min(axis=0) * 1000.0
        bbox_max = verts.max(axis=0) * 1000.0
        dims = bbox_max - bbox_min
        
        pos = data.geom_xpos[i]
        rot = data.geom_xmat[i].reshape(3, 3)
        
        parts.append({
            'geom_id': i,
            'body': body_name,
            'mesh': mesh_name,
            'material': mat_name,
            'rgba': rgba.tolist(),
            'pos': pos.tolist(),
            'dims': dims.tolist(),
            'vert_count': vert_num
        })

print(f'Total parts: {len(parts)}')
unique_meshes = sorted(list(set(p['mesh'] for p in parts)))
print(f'Unique mesh types: {len(unique_meshes)}')
for m in unique_meshes:
    count = sum(1 for p in parts if p['mesh'] == m)
    sample = next(p for p in parts if p['mesh'] == m)
    d = sample['dims']
    print(f'- {m:<38}: x{count:2d} | Dims: {d[0]:5.1f} x {d[1]:5.1f} x {d[2]:5.1f} mm')
