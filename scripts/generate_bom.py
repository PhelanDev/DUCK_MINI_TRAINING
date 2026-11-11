import os
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('models/mjcf/microduck_cad.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

bom = {}

# Classify parts into modules
MODULE_MAP = {
    "head": "1. Cụm Đầu & Thị Giác (Head & Vision)",
    "jaw": "1. Cụm Đầu & Thị Giác (Head & Vision)",
    "mouth": "1. Cụm Đầu & Thị Giác (Head & Vision)",
    "face": "1. Cụm Đầu & Thị Giác (Head & Vision)",
    "lens": "1. Cụm Đầu & Thị Giác (Head & Vision)",
    "noenoeil": "1. Cụm Đầu & Thị Giác (Head & Vision)",
    "neck": "2. Cụm Cổ & Bậc Xoay (Neck & Pan/Tilt)",
    "yaw": "2. Cụm Cổ & Bậc Xoay (Neck & Pan/Tilt)",
    "trunk": "3. Cụm Thân Chính & Khung (Torso & Chassis)",
    "shell": "3. Cụm Thân Chính & Khung (Torso & Chassis)",
    "support": "3. Cụm Thân Chính & Khung (Torso & Chassis)",
    "locker": "3. Cụm Thân Chính & Khung (Torso & Chassis)",
    "pcb": "4. Mạch Điện Tử & Nguồn (Electronics & Battery)",
    "np_f970": "4. Mạch Điện Tử & Nguồn (Electronics & Battery)",
    "speaker": "4. Mạch Điện Tử & Nguồn (Electronics & Battery)",
    "elec": "4. Mạch Điện Tử & Nguồn (Electronics & Battery)",
    "xl330": "5. Động Cơ Servo (Actuators)",
    "hip": "6. Cụm Hông & Khớp Đùi (Hips & Upper Leg)",
    "upper_leg": "6. Cụm Hông & Khớp Đùi (Hips & Upper Leg)",
    "bearing": "7. Vòng Bi Bạc Đạn & Khớp Xoay (Bearings & Joints)",
    "leg": "8. Cụm Cẳng Chân & Gối (Shin & Knees)",
    "ankle": "9. Cụm Mắt Cá & Bàn Chân (Ankles & Feet)",
    "foot": "9. Cụm Mắt Cá & Bàn Chân (Ankles & Feet)",
    "sole": "9. Cụm Mắt Cá & Bàn Chân (Ankles & Feet)",
}

MATERIAL_SPECS = {
    "top_head_shell": ("Nhựa in 3D PLA/PETG", "Vỏ nón bảo hộ đầu trước & đỉnh", "0.2mm, infill 20%"),
    "bottom_head_shell": ("Nhựa in 3D PLA/PETG", "Vỏ đầu sau & hốc tản nhiệt", "0.2mm, infill 20%"),
    "face_part": ("Nhựa in 3D PLA/PETG", "Mặt nạ gắn camera", "0.2mm, infill 25%"),
    "soft_mouth_top": ("Nhựa dẻo TPU (Shore 95A)", "Mỏ vịt trên chống va đập", "0.2mm, infill 100%"),
    "jaw": ("Nhựa in 3D PLA/PETG", "Khung hàm dưới cơ khí", "0.2mm, infill 30%"),
    "jaw_soft": ("Nhựa dẻo TPU", "Đệm môi hàm dưới mềm", "0.2mm, infill 100%"),
    "lens": ("Nhựa quang học / Kính", "Thấu kính giả lập camera M12", "Mua sẵn COOTS"),
    "m12_lens_holder": ("Nhựa in 3D PLA", "Ngàm ren giữ camera M12", "0.12mm, infill 50%"),
    "noenoeil": ("Nhựa in 3D PLA Đen", "Con ngươi cảm xúc / Mắt vịt", "0.15mm, infill 100%"),
    "neck": ("Nhựa in 3D PETG/PA-CF", "Trục cổ chịu tải uốn", "0.2mm, infill 50%"),
    "neck_pitch": ("Nhựa in 3D PETG/PA-CF", "Khớp gật gù cổ", "0.2mm, infill 50%"),
    "yaw2roll": ("Nhựa in 3D PETG", "Cùm chuyển hướng trục cổ", "0.2mm, infill 50%"),
    "yaw_roll_motion": ("Nhựa in 3D PETG", "Cần truyền động xoay cổ", "0.2mm, infill 40%"),
    "trunk_base": ("Nhựa in 3D PETG/PA-CF", "Xương sống khung thân chính", "0.2mm, infill 40%, 4 walls"),
    "left_shell": ("Nhựa in 3D PLA", "Vỏ ốp lườn sườn trái vịt", "0.2mm, infill 15%"),
    "right_shell": ("Nhựa in 3D PLA", "Vỏ ốp lườn sườn phải vịt", "0.2mm, infill 15%"),
    "banana_pcb_locker": ("Nhựa in 3D PLA", "Ngàm khóa khay bo mạch điện tử", "0.2mm, infill 30%"),
    "power_support": ("Nhựa in 3D PLA", "Giá đỡ nguồn & mạch phân phối điện", "0.2mm, infill 35%"),
    "motor_support": ("Nhựa in 3D PETG", "Giá bắt động cơ servo thân", "0.2mm, infill 50%"),
    "np_f970": ("Pin Li-ion Sony NP-F970", "Bộ nguồn 7.4V 6600mAh tiêu chuẩn", "Linh kiện thương mại"),
    "pcb__raspberry_pi_zero_2_w": ("FR4 / PCB", "Bo mạch máy tính trung tâm Pi Zero 2W", "Linh kiện thương mại"),
    "elec_rpi_robot_hat_pcb": ("FR4 / PCB", "Shield điều khiển servo Robot HAT", "Linh kiện thương mại"),
    "speaker": ("Loa mini 8Ω 1W", "Loa phát âm thanh tiếng vịt kêu quác quác", "Linh kiện thương mại"),
    "xl330": ("Robotis Dynamixel XL330", "Servo thông minh bus TTL điều khiển vị trí", "Linh kiện thương mại"),
    "hip_l": ("Nhựa in 3D PETG/PA-CF", "Giá đỡ cùm xoay hông chịu lực lớn", "0.2mm, infill 60%, 5 walls"),
    "upper_leg_left": ("Nhựa in 3D PETG/PA-CF", "Khung xương đùi trên bên trái", "0.2mm, infill 60%"),
    "upper_leg_right": ("Nhựa in 3D PETG/PA-CF", "Khung xương đùi trên bên phải", "0.2mm, infill 60%"),
    "upper_leg_rigidity_plate": ("Nhôm CNC / PETG đúc", "Thanh giằng gia cường chống vặn đùi", "0.15mm, infill 100%"),
    "leg": ("Nhựa in 3D PETG/PA-CF", "Thanh đòn bắp chân", "0.2mm, infill 50%"),
    "bearing_roll": ("Thép mạ Chrome", "Bạc đạn xoay trục servo", "Linh kiện thương mại"),
    "seeed_bearing__configuration__22x16x4": ("Thép vòng bi 22x16x4", "Bạc đạn rãnh sâu trợ lực khớp", "Linh kiện thương mại"),
    "seeed_bearing__configuration_default": ("Thép vòng bi", "Bạc đạn tiêu chuẩn", "Linh kiện thương mại"),
    "ankle_left": ("Nhựa in 3D PETG", "Khớp mắt cá chân trái", "0.2mm, infill 60%"),
    "ankle_right": ("Nhựa in 3D PETG", "Khớp mắt cá chân phải", "0.2mm, infill 60%"),
    "foot_left": ("Nhựa in 3D PETG Vàng", "Khung bàn chân trái 3 ngón chịu tải", "0.2mm, infill 50%"),
    "foot_right": ("Nhựa in 3D PETG Vàng", "Khung bàn chân phải 3 ngón chịu tải", "0.2mm, infill 50%"),
    "sole_left": ("Cao su non TPU (Shore 85A)", "Đế bàn chân trái chống trượt giảm xóc", "0.2mm, infill 100%"),
    "sole_right": ("Cao su non TPU (Shore 85A)", "Đế bàn chân phải chống trượt giảm xóc", "0.2mm, infill 100%"),
}

for i in range(model.nmesh):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
    adr = model.mesh_vertadr[i]
    num = model.mesh_vertnum[i]
    v = model.mesh_vert[adr:adr+num]
    dims = (v.max(axis=0) - v.min(axis=0)) * 1000.0
    
    # Find module
    mod = "10. Chi Tiết Cơ Khí Khác (Hardware)"
    for k, v_mod in MODULE_MAP.items():
        if k in name.lower():
            mod = v_mod
            break
            
    spec = MATERIAL_SPECS.get(name, ("Nhựa in 3D PETG", "Linh kiện cơ khí MicroDuck", "0.2mm, infill 40%"))
    
    # Count occurrences in assembly
    count = 0
    for g in range(model.ngeom):
        if model.geom_group[g] == 2 and model.geom_dataid[g] == i:
            count += 1
            
    bom[name] = {
        'name': name,
        'module': mod,
        'dims': dims,
        'verts': num,
        'count': count,
        'material': spec[0],
        'desc': spec[1],
        'print_spec': spec[2]
    }

# Print sorted by module
modules = sorted(list(set(b['module'] for b in bom.values())))
for m in modules:
    print(f"\n### {m}")
    print("| Tên Chi Tiết (STL) | SL | Kích Thước (DxRxC mm) | Vật Liệu Đề Xuất | Thông Số In / Xuất Xứ | Công Dụng |")
    print("| :--- | :---: | :---: | :--- | :--- | :--- |")
    for name, item in sorted(bom.items()):
        if item['module'] == m:
            d = item['dims']
            print(f"| `{name}.stl` | {item['count']} | {d[0]:.1f} x {d[1]:.1f} x {d[2]:.1f} | {item['material']} | {item['print_spec']} | {item['desc']} |")
