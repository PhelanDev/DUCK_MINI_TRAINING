"""
Organize MicroDuck STL files into dedicated folders for 3D printing categorized by material.
"""
import os
import shutil
import zipfile

SOURCE_DIR = "assets/meshes"
TARGET_DIR = "assets/3d_print_stl"

CATEGORIES = {
    "1_TPU_Elastic_ChongVaDap": [
        "soft_mouth_top.stl",
        "jaw_soft.stl",
        "sole_left.stl",
        "sole_right.stl",
        "tire.stl",
    ],
    "2_PACF_PETG_ChiuLuc_KhungXuong": [
        "trunk_base.stl",
        "hip_l.stl",
        "upper_leg_left.stl",
        "upper_leg_right.stl",
        "upper_leg_rigidity_plate.stl",
        "leg.stl",
        "neck.stl",
        "neck_pitch.stl",
        "yaw2roll.stl",
        "yaw_roll_motion.stl",
        "motor_support.stl",
        "ankle_left.stl",
        "ankle_right.stl",
        "foot_left.stl",
        "foot_right.stl",
    ],
    "3_PLA_PETG_VoNgoai_ThamMy": [
        "top_head_shell.stl",
        "bottom_head_shell.stl",
        "face_part.stl",
        "jaw.stl",
        "left_shell.stl",
        "right_shell.stl",
        "noenoeil.stl",
        "m12_lens_holder.stl",
        "banana_pcb_locker.stl",
        "power_support.stl",
        "rim.stl",
    ],
    "4_LinhKienThuongMai_KiemTraDungSai": [
        "xl330.stl",
        "np_f970.stl",
        "pcb__raspberry_pi_zero_2_w.stl",
        "elec_rpi_robot_hat_pcb.stl",
        "seeed_bearing__configuration__22x16x4.stl",
        "seeed_bearing__configuration_default.stl",
        "bearing_roll.stl",
        "lens.stl",
        "speaker.stl",
    ]
}

def organize():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    for cat, files in CATEGORIES.items():
        cat_dir = os.path.join(TARGET_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for fname in files:
            src = os.path.join(SOURCE_DIR, fname)
            if os.path.exists(src):
                dst = os.path.join(cat_dir, fname)
                shutil.copy2(src, dst)
            else:
                print(f"Warning: {src} not found")
                
    # Create print-ready ZIP archive
    zip_path = os.path.join(TARGET_DIR, "MicroDuck_3D_Print_STLs_Full_Set.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(TARGET_DIR):
            for file in files:
                if file.endswith(".stl"):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, TARGET_DIR)
                    zf.write(full_p, arcname=rel_p)
                    
    print(f"Organized 3D print files in: {TARGET_DIR}")
    print(f"Created ZIP: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    organize()
