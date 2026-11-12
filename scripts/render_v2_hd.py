import os
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image

def render_views():
    repo_root = Path(__file__).resolve().parent.parent
    v2_dir = repo_root / "duck_v2"
    media_dir = v2_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    xml_path = v2_dir / "cyberduck_v2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert body_id != -1

    # Standard studio camera option: visual geoms on, collision hulls & sensory sites off
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    opt.geomgroup[2] = 1
    opt.geomgroup[3] = 0
    opt.sitegroup[4] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0

    width, height = 1024, 768
    renderer = mujoco.Renderer(model, height, width)

    # 1. Front View (Directly facing chest and helmet)
    cam_front = mujoco.MjvCamera()
    cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_front.lookat = data.xpos[body_id].copy()
    cam_front.distance = 0.50
    cam_front.elevation = -10.0
    cam_front.azimuth = 180.0  # Front angle

    renderer.update_scene(data, camera=cam_front, scene_option=opt)
    img_front = renderer.render()
    front_path = media_dir / "cyberduck_v2_front_hd.png"
    Image.fromarray(img_front).save(str(front_path))
    print(f"[OK] Rendered Front HD: {front_path}")

    # 2. Isometric 3/4 View (Hero perspective)
    cam_iso = mujoco.MjvCamera()
    cam_iso.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_iso.lookat = data.xpos[body_id].copy()
    cam_iso.distance = 0.52
    cam_iso.elevation = -14.0
    cam_iso.azimuth = 140.0  # 3/4 Isometric

    renderer.update_scene(data, camera=cam_iso, scene_option=opt)
    img_iso = renderer.render()
    iso_path = media_dir / "cyberduck_v2_iso_hd.png"
    Image.fromarray(img_iso).save(str(iso_path))
    print(f"[OK] Rendered Iso HD: {iso_path}")

    # Visual Quality Self-Check
    # Verify no pure red sensory sites are visible
    # In earlier buggy renders, sensory sites were red (R > 200, G < 60, B < 60)
    red_mask_front = (img_front[:, :, 0] > 200) & (img_front[:, :, 1] < 60) & (img_front[:, :, 2] < 60)
    red_count_front = np.sum(red_mask_front)
    print(f"[Check] Front view pure red sensory artifact pixels: {red_count_front}")
    assert red_count_front == 0, f"Found {red_count_front} red sensory artifact pixels in front view!"

    red_mask_iso = (img_iso[:, :, 0] > 200) & (img_iso[:, :, 1] < 60) & (img_iso[:, :, 2] < 60)
    red_count_iso = np.sum(red_mask_iso)
    print(f"[Check] Iso view pure red sensory artifact pixels: {red_count_iso}")
    assert red_count_iso == 0, f"Found {red_count_iso} red sensory artifact pixels in iso view!"

    print("[SUCCESS] All visual artifact checks passed! Images are clean and radiant.")

if __name__ == "__main__":
    render_views()
