import os
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def render_leg_joints():
    repo_root = Path(__file__).resolve().parent.parent
    xml_path = repo_root / "duck_v2" / "cyberduck_v2.xml"
    media_dir = repo_root / "duck_v2" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Enable joint visualization flag (draws joint axes and rotation cylinders!)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    opt.geomgroup[2] = 1
    opt.geomgroup[3] = 0
    opt.sitegroup[2] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1  # Highlights joints in 3D!

    width, height = 1024, 768
    renderer = mujoco.Renderer(model, height, width)

    # Joint positions in world frame
    hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll")
    hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")
    knee_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_pitch")

    p_roll = data.xanchor[hip_roll_id]
    p_pitch = data.xanchor[hip_pitch_id]
    p_knee = data.xanchor[knee_pitch_id]

    print("Joint Anchors:")
    print("  left_hip_roll  :", p_roll)
    print("  left_hip_pitch :", p_pitch)
    print("  left_knee_pitch:", p_knee)

    # Focus point: center of left leg
    leg_center = (p_roll + p_knee) / 2.0

    views = [
        {
            "name": "leg_joints_iso.png",
            "title": "GÓC PHỐI CẢNH 3/4 (PERSPECTIVE)",
            "azimuth": 135.0,
            "elevation": -12.0,
            "distance": 0.30,
        },
        {
            "name": "leg_joints_side.png",
            "title": "GÓC NHÌN BÊN HÔNG (SIDE VIEW - XEM 2 TRỤC PITCH)",
            "azimuth": 90.0,
            "elevation": -8.0,
            "distance": 0.30,
        },
        {
            "name": "leg_joints_front.png",
            "title": "GÓC NHÌN CHÍNH DIỆN (FRONT VIEW - XEM TRỤC ROLL)",
            "azimuth": 180.0,
            "elevation": -10.0,
            "distance": 0.30,
        },
    ]

    for v in views:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = leg_center.copy()
        cam.distance = v["distance"]
        cam.elevation = v["elevation"]
        cam.azimuth = v["azimuth"]

        renderer.update_scene(data, camera=cam, scene_option=opt)
        rgb = renderer.render()
        img = Image.fromarray(rgb)
        
        # Add informative label banner on image
        draw = ImageDraw.Draw(img)
        # Top banner
        draw.rectangle([0, 0, width, 44], fill=(20, 24, 32, 230))
        draw.text((20, 12), f"CYBERDUCK V2 PRO - {v['title']}", fill=(0, 235, 255))
        
        # Bottom info card
        draw.rectangle([0, height - 85, width, height], fill=(15, 18, 24, 240))
        draw.text((20, height - 75), "[1] Khớp Háng Lăn (Hip Roll)   : Trục X [1,0,0] - Lắc nghiêng chân sang 2 bên sườn", fill=(255, 200, 60))
        draw.text((20, height - 50), "[2] Khớp Háng Gập (Hip Pitch) : Trục Y [0,1,0] - Nâng đùi bước tới trước / lùi sau", fill=(100, 220, 255))
        draw.text((20, height - 25), "[3] Khớp Gối Gập (Knee Pitch) : Trục Y [0,1,0] - Co gập cẳng chân lên / duỗi ra", fill=(120, 255, 140))

        out_path = media_dir / v["name"]
        img.save(str(out_path))
        print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    render_leg_joints()
