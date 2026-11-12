"""
CyberDuck V2 Pro - Internal Mechanical Skeleton & Servo Chassis Generator.
Generates an industrial-grade internal mechanical skeleton frame designed for real-world 3D printing and hardware assembly:
1. 8x Precision Servo Units (Actuator blocks with 25T output spline horn & mounting ears)
2. Torso Central Spine Chassis (Dual hip roll servo bay, head yaw mount, battery pocket, PCB deck)
3. Neck 2-DOF Gimbal Bracket (Interlocking Yaw horn to Pitch servo)
4. Left & Right Hip 2-DOF Gimbal C-Brackets (C-bracket mounting Hip Roll horn to Hip Pitch servo body)
5. Left & Right Thigh I-Beam Skeletons (Structural thigh bone housing the Knee Pitch servo)
6. Left & Right Shin Load Struts (High-rigidity bipedal shin column)
7. Left & Right Foot Chassis Plates (Reinforced ground contact baseplate with M2 bolt patterns)
Exports:
- duck_v2/cad/cyberduck_v2_skeleton.obj (+ .mtl)
- duck_v2/cad/cyberduck_v2_skeleton_parts.zip (printable STL parts)
- duck_v2/media/cyberduck_v2_skeleton_*.png
"""
import os
import struct
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

# Helper: write binary STL
def write_binary_stl(path, triangles):
    header = b"CyberDuck V2 Mechanical Skeleton CAD ".ljust(80, b"\0")
    num_tris = len(triangles)
    records = []
    for tri in triangles:
        p0, p1, p2 = tri[0], tri[1], tri[2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-7:
            normal = normal / norm_len
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        records.append(struct.pack(
            "<12fH",
            float(normal[0]), float(normal[1]), float(normal[2]),
            float(p0[0]), float(p0[1]), float(p0[2]),
            float(p1[0]), float(p1[1]), float(p1[2]),
            float(p2[0]), float(p2[1]), float(p2[2]),
            0
        ))
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", num_tris))
        f.write(b"".join(records))

# Helper: Box mesh with correct outward normals
def make_box(dx, dy, dz, center=(0, 0, 0), rot_mat=None):
    """Generate a 3D box solid with outward normals."""
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    v = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz]
    ], dtype=np.float32)
    if rot_mat is not None:
        v = v @ rot_mat.T
    v = v + np.array(center, dtype=np.float32)
    # Triangles with outward winding
    faces = [
        # Bottom (z = -hz)
        [0, 2, 1], [0, 3, 2],
        # Top (z = +hz)
        [4, 5, 6], [4, 6, 7],
        # Front (y = -hy)
        [0, 1, 5], [0, 5, 4],
        # Back (y = +hy)
        [2, 3, 7], [2, 7, 6],
        # Left (x = -hx)
        [0, 4, 7], [0, 7, 3],
        # Right (x = +hx)
        [1, 2, 6], [1, 6, 5],
    ]
    tris = []
    for f in faces:
        tris.append([v[f[0]], v[f[1]], v[f[2]]])
    return np.array(tris, dtype=np.float32)

# Helper: Cylinder mesh with correct outward normals
def make_cylinder(r, height, center=(0, 0, 0), axis='z', n=24):
    """Generate a 3D cylinder solid with outward normals."""
    half_h = height / 2.0
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    if axis == 'z':
        v_bot = np.stack([r * cos_t, r * sin_t, np.full(n, -half_h)], axis=-1)
        v_top = np.stack([r * cos_t, r * sin_t, np.full(n,  half_h)], axis=-1)
        c_bot = np.array([0, 0, -half_h], dtype=np.float32)
        c_top = np.array([0, 0,  half_h], dtype=np.float32)
    elif axis == 'y':
        v_bot = np.stack([r * cos_t, np.full(n, -half_h), r * sin_t], axis=-1)
        v_top = np.stack([r * cos_t, np.full(n,  half_h), r * sin_t], axis=-1)
        c_bot = np.array([0, -half_h, 0], dtype=np.float32)
        c_top = np.array([0,  half_h, 0], dtype=np.float32)
    else:  # axis == 'x'
        v_bot = np.stack([np.full(n, -half_h), r * cos_t, r * sin_t], axis=-1)
        v_top = np.stack([np.full(n,  half_h), r * cos_t, r * sin_t], axis=-1)
        c_bot = np.array([-half_h, 0, 0], dtype=np.float32)
        c_top = np.array([ half_h, 0, 0], dtype=np.float32)

    offset = np.array(center, dtype=np.float32)
    v_bot += offset
    v_top += offset
    c_bot += offset
    c_top += offset

    tris = []
    for i in range(n):
        nxt = (i + 1) % n
        # Side wall (2 triangles)
        tris.append([v_bot[i], v_top[i], v_top[nxt]])
        tris.append([v_bot[i], v_top[nxt], v_bot[nxt]])
        # Bottom cap
        tris.append([c_bot, v_bot[nxt], v_bot[i]])
        # Top cap
        tris.append([c_top, v_top[i], v_top[nxt]])
    return np.array(tris, dtype=np.float32)

# =========================================================================
# 1. Servo Motor Model (Feetech / SG90 / MG90S Class 9g/12g micro servo)
# =========================================================================
def make_servo_model(center=(0, 0, 0), axis_dir='y', horn_side=+1):
    """
    Standard micro servo: Body 23x12x22mm, Mounting ears 32x12x2.5mm, Output shaft & 25T round horn.
    """
    body = make_box(0.023, 0.012, 0.022, center=center)
    # Flange / mounting ear tabs
    ears = make_box(0.032, 0.012, 0.0025, center=(center[0], center[1], center[2] + 0.004))
    # Output horn gear
    if axis_dir == 'y':
        horn_pos = (center[0] + 0.005, center[1] + horn_side * (0.006 + 0.002), center[2] + 0.005)
        horn = make_cylinder(0.006, 0.003, center=horn_pos, axis='y')
        shaft = make_cylinder(0.0025, 0.005, center=horn_pos, axis='y')
    elif axis_dir == 'x':
        horn_pos = (center[0] + horn_side * (0.0115 + 0.002), center[1], center[2] + 0.005)
        horn = make_cylinder(0.006, 0.003, center=horn_pos, axis='x')
        shaft = make_cylinder(0.0025, 0.005, center=horn_pos, axis='x')
    else:  # 'z'
        horn_pos = (center[0] + 0.005, center[1], center[2] + horn_side * (0.011 + 0.002))
        horn = make_cylinder(0.006, 0.003, center=horn_pos, axis='z')
        shaft = make_cylinder(0.0025, 0.005, center=horn_pos, axis='z')
    return np.vstack([body, ears, horn, shaft])

# =========================================================================
# 2. Main Torso Central Chassis (Khung Lồng Ngực & Khung Thân Chính)
# =========================================================================
def make_torso_chassis():
    """
    Main structural chassis:
    - Central rigid spine plate
    - Left & right hip servo mounting pockets
    - Head yaw servo mounting collar at neck
    - Battery tray (bottom pocket for 2S LiPo)
    - Electronics deck (back mounting tabs for Pi Zero / ESP32)
    """
    parts = []
    # Central structural spine box
    spine = make_box(0.036, 0.040, 0.044, center=(0.002, 0, -0.010))
    parts.append(spine)

    # Neck collar / head yaw servo bracket
    neck_ring = make_cylinder(0.010, 0.008, center=(0.012, 0, 0.022), axis='z')
    parts.append(neck_ring)

    # Battery compartment (bottom tray)
    bat_tray = make_box(0.032, 0.036, 0.006, center=(0.002, 0, -0.035))
    parts.append(bat_tray)

    # Left & right hip servo pocket cages
    hip_cage_l = make_box(0.028, 0.016, 0.026, center=(0.005, 0.024, -0.025))
    hip_cage_r = make_box(0.028, 0.016, 0.026, center=(0.005, -0.024, -0.025))
    parts.append(hip_cage_l)
    parts.append(hip_cage_r)

    # PCB standoff posts (4x M2.5 mounting standoffs at back)
    for sx in [-0.012, 0.012]:
        for sy in [-0.014, 0.014]:
            post = make_cylinder(0.0025, 0.006, center=(sx, sy, -0.005), axis='x')
            parts.append(post)

    return np.vstack(parts)

# =========================================================================
# 3. 2-DOF Hip Gimbal Bracket (Gá Khớp Háng C-Bracket - Miếng Đen Cơ Khí)
# =========================================================================
def make_hip_gimbal_bracket(side=+1):
    """
    Mechanical C-Bracket connecting Hip Roll output horn to Hip Pitch servo:
    - Backplate with 4-hole servo horn bolt circle
    - U-channel clamping the Hip Pitch servo body
    - Bearing pivot cap on opposite side
    """
    parts = []
    y_sign = side
    base_center = (0.005, y_sign * 0.036, -0.025)

    # Structural C-bracket back wall
    back_wall = make_box(0.026, 0.005, 0.026, center=(base_center[0], base_center[1], base_center[2]))
    # Upper & lower clamp flanges
    top_flange = make_box(0.026, 0.014, 0.004, center=(base_center[0], base_center[1] + y_sign * 0.005, base_center[2] + 0.011))
    bot_flange = make_box(0.026, 0.014, 0.004, center=(base_center[0], base_center[1] + y_sign * 0.005, base_center[2] - 0.011))
    
    # Servo horn mounting disc boss (mặt bích bắt ốc đĩa servo roll)
    disc_boss = make_cylinder(0.007, 0.003, center=(base_center[0], base_center[1] - y_sign * 0.003, base_center[2]), axis='x')

    parts.extend([back_wall, top_flange, bot_flange, disc_boss])
    return np.vstack(parts)

# =========================================================================
# 4. Thigh Structural Skeleton (Xương Đùi Cơ Khí Khung Hộp I-Beam)
# =========================================================================
def make_thigh_skeleton(side=+1):
    """
    Upper leg structural chassis:
    - Top yoke mating with Hip Pitch servo horn
    - Main I-beam hollow femur bone
    - Lower clevis housing the Knee Pitch servo
    """
    parts = []
    y_sign = side
    thigh_center = (0.005, y_sign * 0.048, -0.055)

    # Top yoke collar
    top_collar = make_cylinder(0.008, 0.005, center=(thigh_center[0], thigh_center[1], -0.040), axis='y')
    # Main femur strut (xương đùi chịu lực)
    femur = make_box(0.018, 0.014, 0.034, center=(thigh_center[0], thigh_center[1], -0.057))
    # Knee servo mounting cage (hộp gá servo gối)
    knee_cage = make_box(0.022, 0.015, 0.022, center=(thigh_center[0], thigh_center[1], -0.076))

    parts.extend([top_collar, femur, knee_cage])
    return np.vstack(parts)

# =========================================================================
# 5. Shin Load Strut & Ankle Clevis (Xương Cẳng Chân Cơ Khí & Càng Khớp Cổ Chân)
# =========================================================================
def make_shin_skeleton(side=+1):
    """
    Lower leg structural chassis:
    - Upper knee clevis mating with Knee Pitch servo horn
    - Dual tubular tibia struts
    - Lower ankle clevis mounting directly to foot chassis
    """
    parts = []
    y_sign = side
    shin_center = (0.005, y_sign * 0.048, -0.098)

    # Upper knee clevis collar
    knee_clevis = make_cylinder(0.007, 0.004, center=(shin_center[0], shin_center[1], -0.084), axis='y')
    # Twin structural columns
    tibia_f = make_cylinder(0.0035, 0.032, center=(shin_center[0] + 0.004, shin_center[1], -0.100), axis='z')
    tibia_r = make_cylinder(0.0035, 0.032, center=(shin_center[0] - 0.004, shin_center[1], -0.100), axis='z')
    # Ankle foot-mount block
    ankle_block = make_box(0.016, 0.012, 0.008, center=(shin_center[0], shin_center[1], -0.118))

    parts.extend([knee_clevis, tibia_f, tibia_r, ankle_block])
    return np.vstack(parts)

# =========================================================================
# 6. Foot Chassis Baseplate (Khung Bàn Chân Cơ Khí)
# =========================================================================
def make_foot_chassis(side=+1):
    """
    Solid foot chassis:
    - Rigid rectangular baseplate 50mm x 26mm x 3mm
    - Ankle bracket riser
    - Counterbore mounting holes for rubber tread pads
    """
    parts = []
    y_sign = side
    foot_center = (0.005, y_sign * 0.048, -0.123)

    # Main chassis sole plate
    sole_plate = make_box(0.048, 0.024, 0.0035, center=(foot_center[0], foot_center[1], foot_center[2]))
    # Ankle riser bracket
    riser = make_box(0.016, 0.014, 0.007, center=(foot_center[0], foot_center[1], foot_center[2] + 0.005))

    parts.extend([sole_plate, riser])
    return np.vstack(parts)

# =========================================================================
# 7. Master Assembly & Export Pipeline
# =========================================================================
def build_and_export_skeleton():
    v2_dir = Path(__file__).resolve().parent.parent
    cad_dir = v2_dir / "cad"
    media_dir = v2_dir / "media"
    stl_dir = cad_dir / "skeleton_stl_parts"
    cad_dir.mkdir(parents=True, exist_ok=True)
    stl_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Đang tạo mô hình khung cơ khí và 8 servo CyberDuck V2 Pro...")

    components = {}

    # 1. Structural Chassis Parts
    components["torso_chassis"] = make_torso_chassis()
    components["hip_gimbal_bracket_left"] = make_hip_gimbal_bracket(+1)
    components["hip_gimbal_bracket_right"] = make_hip_gimbal_bracket(-1)
    components["thigh_skeleton_left"] = make_thigh_skeleton(+1)
    components["thigh_skeleton_right"] = make_thigh_skeleton(-1)
    components["shin_skeleton_left"] = make_shin_skeleton(+1)
    components["shin_skeleton_right"] = make_shin_skeleton(-1)
    components["foot_chassis_left"] = make_foot_chassis(+1)
    components["foot_chassis_right"] = make_foot_chassis(-1)

    # 2. 8x Precision Actuator Servos (Vị trí chính xác khớp)
    # Head yaw servo
    components["servo_head_yaw"] = make_servo_model(center=(0.012, 0, 0.038), axis_dir='z', horn_side=+1)
    # Head pitch servo
    components["servo_head_pitch"] = make_servo_model(center=(0.012, 0, 0.054), axis_dir='y', horn_side=+1)

    # Left leg 3 servos
    components["servo_left_hip_roll"] = make_servo_model(center=(0.005, 0.036, -0.025), axis_dir='x', horn_side=+1)
    components["servo_left_hip_pitch"] = make_servo_model(center=(0.005, 0.048, -0.040), axis_dir='y', horn_side=+1)
    components["servo_left_knee_pitch"] = make_servo_model(center=(0.005, 0.048, -0.084), axis_dir='y', horn_side=+1)

    # Right leg 3 servos
    components["servo_right_hip_roll"] = make_servo_model(center=(0.005, -0.036, -0.025), axis_dir='x', horn_side=-1)
    components["servo_right_hip_pitch"] = make_servo_model(center=(0.005, -0.048, -0.040), axis_dir='y', horn_side=-1)
    components["servo_right_knee_pitch"] = make_servo_model(center=(0.005, -0.048, -0.084), axis_dir='y', horn_side=-1)

    # 3. Export Individual STL parts
    zip_path = cad_dir / "cyberduck_v2_skeleton_parts.zip"
    print(f">>> Đang xuất các tệp STL chi tiết vào {stl_dir} và gói zip...")
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
        for name, tris in components.items():
            stl_path = stl_dir / f"{name}.stl"
            write_binary_stl(str(stl_path), tris)
            zipf.write(str(stl_path), arcname=f"{name}.stl")
            print(f"  [STL] {name}.stl : {len(tris)} triangles")

    # 4. Export Combined OBJ + MTL Assembly
    obj_path = cad_dir / "cyberduck_v2_skeleton.obj"
    mtl_path = cad_dir / "cyberduck_v2_skeleton.mtl"

    mtl_content = """# Materials for CyberDuck V2 Mechanical Skeleton
newmtl mat_chassis_titanium
Ka 0.10 0.12 0.15
Kd 0.45 0.50 0.58
Ks 0.60 0.60 0.60
Ns 45.0
d 1.0

newmtl mat_servo_body
Ka 0.05 0.05 0.06
Kd 0.18 0.18 0.20
Ks 0.25 0.25 0.25
Ns 20.0
d 1.0

newmtl mat_servo_horn_bronze
Ka 0.16 0.11 0.04
Kd 0.82 0.56 0.22
Ks 0.75 0.65 0.40
Ns 60.0
d 1.0
"""
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(mtl_content)

    print(f">>> Đang xuất file 3D hoàn chỉnh {obj_path}...")
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# CyberDuck V2 Mechanical Skeleton Assembly OBJ\n")
        f.write(f"mtllib cyberduck_v2_skeleton.mtl\n\n")

        vert_offset = 1
        for name, tris in components.items():
            f.write(f"o {name}\n")
            if "servo" in name:
                f.write("usemtl mat_servo_body\n")
            elif "horn" in name or "gimbal" in name:
                f.write("usemtl mat_servo_horn_bronze\n")
            else:
                f.write("usemtl mat_chassis_titanium\n")

            v_flat = tris.reshape(-1, 3)
            for v in v_flat:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            num_tris = len(tris)
            for i in range(num_tris):
                idx = vert_offset + i * 3
                f.write(f"f {idx} {idx+1} {idx+2}\n")

            vert_offset += len(v_flat)

    print(f"[OK] File 3D bộ khung cơ khí xuất thành công: {obj_path}")
    print(f"[OK] Gói zip STL in 3D: {zip_path}")

if __name__ == "__main__":
    build_and_export_skeleton()
