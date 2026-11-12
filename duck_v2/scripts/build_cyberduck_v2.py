"""
Cohesive CyberDuck V2 Parametric 3D Generator (Clean Studio Mecha Aesthetic - Slim & Athletic Edition).
Generates 16 high-density, beautifully sculpted STL components:
- Compact athletic cranium with slender neck cowl and ear rosettes (proportional to torso)
- Continuous solid 3D curved panoramic visor (Master Chief / Daft Punk style)
- Sleek sculpted aerodynamic metallic duckbill (compact spatulate bionic bill)
- Pectoral chest armor with slim collar, chiseled flank plates, and flared pelvic hip sockets
- Multi-tier illuminated reactor turbine core
- Dorsal twin-nacelle thruster backpack (no wings - clean, athletic silhouette)
- Bionic universal hip gimbals connecting pelvis to thighs without air gaps
- Bionic sculpted thighs with upper trochanter cap and lateral hip shield
- Bionic shins with integrated knee strike plate and calf casing
- 3-toed tactical webbed feet with ankle clevis
- Solid rubber soles with parallel grip tread ribs (exact Z=0.000m ground contact)
"""
import os
import struct
import numpy as np

def write_binary_stl(path, triangles):
    header = b"CyberDuck V2 Cohesive Studio Solid CAD ".ljust(80, b"\0")
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

def superellipsoid(rx, ry, rz, r=0.4, s=0.4, center=(0, 0, 0), rot_mat=None, n_u=32, n_v=32):
    """Generate high-density smooth superellipsoid mesh."""
    def c(w, m):
        cos_w = np.cos(w)
        return np.sign(cos_w) * (np.abs(cos_w) ** m)
    def s_fn(w, m):
        sin_w = np.sin(w)
        return np.sign(sin_w) * (np.abs(sin_w) ** m)

    u = np.linspace(-np.pi / 2, np.pi / 2, n_u)
    v = np.linspace(-np.pi, np.pi, n_v)
    U, V = np.meshgrid(u, v)

    X = rx * c(U, r) * c(V, s)
    Y = ry * c(U, r) * s_fn(V, s)
    Z = rz * s_fn(U, r)

    pts = np.stack([X, Y, Z], axis=-1)
    if rot_mat is not None:
        pts = pts @ rot_mat.T
    pts = pts + np.array(center, dtype=np.float32)

    verts = pts.reshape(-1, 3)
    tris = []
    for i in range(n_v - 1):
        for j in range(n_u - 1):
            p0 = i * n_u + j
            p1 = i * n_u + (j + 1)
            p2 = (i + 1) * n_u + j
            p3 = (i + 1) * n_u + (j + 1)
            tris.append([verts[p0], verts[p2], verts[p1]])
            tris.append([verts[p1], verts[p2], verts[p3]])
    return np.array(tris, dtype=np.float32)

def filleted_cylinder(r, half_length, fillet=0.002, center=(0, 0, 0), axis='z', n_theta=32, n_z=16):
    """Generate smooth capped cylinder with filleted bevel edges."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z_rel = np.linspace(-half_length, half_length, n_z)
    Theta, Z_rel = np.meshgrid(theta, z_rel)

    edge_dist = np.maximum(0.0, np.abs(Z_rel) - (half_length - fillet))
    r_mod = r - fillet + np.sqrt(np.maximum(0.0, fillet**2 - edge_dist**2))

    if axis == 'z':
        X = r_mod * np.cos(Theta)
        Y = r_mod * np.sin(Theta)
        Z = Z_rel
    elif axis == 'y':
        X = r_mod * np.cos(Theta)
        Y = Z_rel
        Z = -r_mod * np.sin(Theta)
    else:  # 'x'
        X = Z_rel
        Y = r_mod * np.cos(Theta)
        Z = r_mod * np.sin(Theta)

    pts = np.stack([X, Y, Z], axis=-1) + np.array(center, dtype=np.float32)
    verts = pts.reshape(-1, 3)
    tris = []
    for i in range(n_z - 1):
        for j in range(n_theta - 1):
            p0 = i * n_theta + j
            p1 = i * n_theta + (j + 1)
            p2 = (i + 1) * n_theta + j
            p3 = (i + 1) * n_theta + (j + 1)
            tris.append([verts[p0], verts[p1], verts[p2]])
            tris.append([verts[p1], verts[p3], verts[p2]])

    # Caps
    c_bot = np.array(center) + (np.array([-half_length, 0, 0]) if axis=='x' else (np.array([0, -half_length, 0]) if axis=='y' else np.array([0, 0, -half_length])))
    c_top = np.array(center) + (np.array([half_length, 0, 0]) if axis=='x' else (np.array([0, half_length, 0]) if axis=='y' else np.array([0, 0, half_length])))
    for j in range(n_theta - 1):
        tris.append([c_bot, verts[j+1], verts[j]])
        top_idx = (n_z - 1) * n_theta + j
        tris.append([c_top, verts[top_idx], verts[top_idx+1]])
    return np.array(tris, dtype=np.float32)

# =========================================================================
# 1. Compact Athletic Head (Proportionate Helmet, Slender Cowl, Panoramic Visor, Beak)
# =========================================================================

def generate_sleek_helmet():
    """
    Compact athletic helmet with smooth aerodynamic pebble curvature.
    Proportionate to torso (scaled down by ~25%), slender neck cowl, cheek armor, and ear rosettes.
    """
    # Compact Athletic Cranium (60mm length x 52mm width x 44mm height)
    cranium = superellipsoid(0.030, 0.026, 0.022, r=0.55, s=0.55, center=(0.002, 0, 0.016))
    
    # Sleek Top Aero-Spine
    crest = superellipsoid(0.026, 0.004, 0.004, r=0.35, s=0.35, center=(0.001, 0, 0.036))
    
    # Cheek Armor flanking face
    cheek_l = superellipsoid(0.018, 0.004, 0.012, r=0.35, s=0.35, center=(0.008, 0.024, 0.008))
    cheek_r = superellipsoid(0.018, 0.004, 0.012, r=0.35, s=0.35, center=(0.008, -0.024, 0.008))
    
    # Ear Sensor Rosettes
    ear_l = filleted_cylinder(0.008, 0.003, fillet=0.001, center=(-0.002, 0.026, 0.016), axis='y')
    ear_r = filleted_cylinder(0.008, 0.003, fillet=0.001, center=(-0.002, -0.026, 0.016), axis='y')
    
    # Sculpted Helmet Base Trim (Vòm đáy nón bo tròn ôm lấy khớp gập cổ)
    base_cup = filleted_cylinder(0.016, 0.004, fillet=0.001, center=(0.000, 0, 0.000), axis='z')
    chin_rim = superellipsoid(0.012, 0.013, 0.004, r=0.30, s=0.30, center=(0.012, 0, -0.001))
    
    return np.vstack([cranium, crest, cheek_l, cheek_r, ear_l, ear_r, base_cup, chin_rim])

def generate_bionic_neck():
    """
    Bionic 2-DOF Universal Neck Gimbal (Đồng bộ tuyệt đối ngôn ngữ thiết kế với khớp háng bionic).
    Cầu nối vật lý giữa trục xoay Yaw (Z=0.000m) tại cổ áo ngực và trục xoay Pitch (Z=0.016m) tại đáy mũ bảo hiểm.
    Gồm:
    - Yaw Turret Knuckle: Khớp xoay mâm cổ ăn khớp khít với cổ áo ngực.
    - Bionic Dogbone Link: Thân liên kết cơ bắp bo tròn nguyên khối vững chãi.
    - Pitch Bearing Knuckle: Khớp gập đầu trục Y tròn mịn.
    - Titanium Servo Bearing Caps: Nắp che ổ bi xoay 2 bên trục Y đồng điệu với rosettes tai nón.
    - Front Throat Accent & Rear Spine Channel: Gờ vát ức khí động học và rãnh cáp sống lưng.
    """
    # 1. Yaw Turret Knuckle (Khớp xoay Yaw trục Z tiếp xúc cổ ngực)
    yaw_knuckle = filleted_cylinder(0.013, 0.003, fillet=0.0008, center=(0, 0, 0.003), axis='z')
    
    # 2. Bionic Neck Dogbone Core (Thân liên kết cơ bắp bo tròn)
    dogbone = superellipsoid(0.010, 0.009, 0.006, r=0.32, s=0.32, center=(0.001, 0, 0.008))
    
    # 3. Pitch Bearing Knuckle (Khớp gập Pitch trục Y tại Z=0.016m)
    pitch_knuckle = filleted_cylinder(0.0085, 0.010, fillet=0.0012, center=(0, 0, 0.016), axis='y')
    
    # 4. Titanium Servo Bearing Caps (Nắp che ổ bi xoay 2 bên trục Y)
    cap_l = filleted_cylinder(0.0060, 0.002, fillet=0.0005, center=(0, 0.011, 0.016), axis='y')
    cap_r = filleted_cylinder(0.0060, 0.002, fillet=0.0005, center=(0, -0.011, 0.016), axis='y')
    
    # 5. Front Throat Bionic Accent & Rear Spine Channel
    throat_accent = superellipsoid(0.005, 0.005, 0.005, r=0.25, s=0.25, center=(0.008, 0, 0.007))
    spine_rib = superellipsoid(0.004, 0.004, 0.005, r=0.25, s=0.25, center=(-0.007, 0, 0.007))
    
    return np.vstack([yaw_knuckle, dogbone, pitch_knuckle, cap_l, cap_r, throat_accent, spine_rib])

def generate_panoramic_visor():
    """
    Continuous solid 3D curved panoramic visor (Master Chief / Daft Punk style).
    Proportionate to compact athletic helmet, 3mm shell thickness.
    """
    n_theta = 36
    n_z = 18
    theta = np.linspace(-np.pi / 2.6, np.pi / 2.6, n_theta)
    z = np.linspace(0.008, 0.022, n_z)
    Theta, Z = np.meshgrid(theta, z)

    # Outer surface
    r_out = 0.032
    x_out = r_out * np.cos(Theta) + 0.001
    y_out = r_out * np.sin(Theta) * 0.90
    pts_out = np.stack([x_out, y_out, Z], axis=-1)

    # Inner surface (3mm depth)
    r_in = 0.029
    x_in = r_in * np.cos(Theta) + 0.001
    y_in = r_in * np.sin(Theta) * 0.90
    pts_in = np.stack([x_in, y_in, Z], axis=-1)

    v_out = pts_out.reshape(-1, 3)
    v_in = pts_in.reshape(-1, 3)
    n_pts = len(v_out)
    all_verts = np.vstack([v_out, v_in])

    tris = []
    # Front outer face
    for i in range(n_z - 1):
        for j in range(n_theta - 1):
            p0 = i * n_theta + j
            p1 = i * n_theta + (j + 1)
            p2 = (i + 1) * n_theta + j
            p3 = (i + 1) * n_theta + (j + 1)
            tris.append([all_verts[p0], all_verts[p2], all_verts[p1]])
            tris.append([all_verts[p1], all_verts[p2], all_verts[p3]])

    # Back inner face
    for i in range(n_z - 1):
        for j in range(n_theta - 1):
            p0 = n_pts + i * n_theta + j
            p1 = n_pts + i * n_theta + (j + 1)
            p2 = n_pts + (i + 1) * n_theta + j
            p3 = n_pts + (i + 1) * n_theta + (j + 1)
            tris.append([all_verts[p0], all_verts[p1], all_verts[p2]])
            tris.append([all_verts[p1], all_verts[p3], all_verts[p2]])

    # Top & bottom caps
    for j in range(n_theta - 1):
        o0 = (n_z - 1) * n_theta + j
        o1 = (n_z - 1) * n_theta + (j + 1)
        i0 = n_pts + o0
        i1 = n_pts + o1
        tris.append([all_verts[o0], all_verts[o1], all_verts[i0]])
        tris.append([all_verts[o1], all_verts[i1], all_verts[i0]])
        b0 = j
        b1 = j + 1
        ib0 = n_pts + b0
        ib1 = n_pts + b1
        tris.append([all_verts[b0], all_verts[ib0], all_verts[b1]])
        tris.append([all_verts[b1], all_verts[ib0], all_verts[ib1]])

    # Left & right side edges
    for i in range(n_z - 1):
        l_out0 = i * n_theta
        l_out1 = (i + 1) * n_theta
        l_in0 = n_pts + l_out0
        l_in1 = n_pts + l_out1
        tris.append([all_verts[l_out0], all_verts[l_in0], all_verts[l_out1]])
        tris.append([all_verts[l_out1], all_verts[l_in0], all_verts[l_in1]])
        r_out0 = i * n_theta + (n_theta - 1)
        r_out1 = (i + 1) * n_theta + (n_theta - 1)
        r_in0 = n_pts + r_out0
        r_in1 = n_pts + r_out1
        tris.append([all_verts[r_out0], all_verts[r_out1], all_verts[r_in0]])
        tris.append([all_verts[r_out1], all_verts[r_in1], all_verts[r_in0]])

    return np.array(tris, dtype=np.float32)

def generate_sculpted_beak():
    """
    Compact spatulate duckbill: wide, flat, aerodynamic, and proportional to compact head.
    """
    # Upper spatulate duckbill foil
    upper_bill = superellipsoid(0.019, 0.016, 0.0035, r=0.35, s=0.35, center=(0.036, 0, 0.004))
    
    # Spatulate rounded duckbill front tip
    bill_tip = superellipsoid(0.009, 0.013, 0.0025, r=0.3, s=0.3, center=(0.050, 0, 0.002))
    
    # Lower mandible foil directly underneath
    lower_mandible = superellipsoid(0.015, 0.013, 0.0025, r=0.35, s=0.35, center=(0.034, 0, 0.000))
    
    # Solid aerodynamic root connecting to cranium
    root = superellipsoid(0.009, 0.017, 0.006, r=0.3, s=0.3, center=(0.025, 0, 0.003))
    
    return np.vstack([upper_bill, bill_tip, lower_mandible, root])

# =========================================================================
# 2. Torso, Slim Collar, Reactor & Dorsal Thrusters (Clean Silhouette, No Wings)
# =========================================================================

def generate_sculpted_chest():
    """
    Sculpted aerodynamic torso with slender neck collar, chiseled flank plates,
    and flared pelvic hip sockets. Clean silhouette with no distracting wings.
    """
    # Main Pectoral Torso Block
    chest_block = superellipsoid(0.028, 0.036, 0.032, r=0.4, s=0.4, center=(0.010, 0, 0.002))
    
    # Integrated Neck Collar Socket (Vòm cổ áo ngực bo tròn fillet đón khớp cổ gimbal)
    collar = filleted_cylinder(0.016, 0.006, fillet=0.0015, center=(0.012, 0, 0.034), axis='z')
    collar_lip = filleted_cylinder(0.0145, 0.002, fillet=0.0005, center=(0.012, 0, 0.038), axis='z')
    
    # Shoulder Mount Armors (left & right)
    sh_l = superellipsoid(0.020, 0.010, 0.016, r=0.3, s=0.3, center=(0.006, 0.036, 0.014))
    sh_r = superellipsoid(0.020, 0.010, 0.016, r=0.3, s=0.3, center=(0.006, -0.036, 0.014))
    
    # Chiseled Lat / Flank Armor Plates (sleek side profile now that wings are removed)
    flank_l = superellipsoid(0.026, 0.006, 0.022, r=0.35, s=0.35, center=(-0.002, 0.036, -0.004))
    flank_r = superellipsoid(0.026, 0.006, 0.022, r=0.35, s=0.35, center=(-0.002, -0.036, -0.004))
    
    # Central Pelvic Keel
    pelvis = superellipsoid(0.022, 0.024, 0.014, r=0.35, s=0.35, center=(0.006, 0, -0.024))
    
    # Flared Pelvic Hip Sockets (Hốc khớp hông bao bọc kín ổ bi háng)
    hip_socket_l = superellipsoid(0.016, 0.012, 0.014, r=0.30, s=0.30, center=(0.005, 0.036, -0.024))
    hip_socket_r = superellipsoid(0.016, 0.012, 0.014, r=0.30, s=0.30, center=(0.005, -0.036, -0.024))
    
    return np.vstack([chest_block, collar, collar_lip, sh_l, sh_r, flank_l, flank_r, pelvis, hip_socket_l, hip_socket_r])

def generate_multi_tier_reactor():
    """
    Multi-tier circular arc reactor core recessed into the chest.
    Outer metallic bevel ring with glowing inner cyan energy emitter.
    """
    core_emitter = filleted_cylinder(0.011, 0.003, fillet=0.0008, center=(0.038, 0, 0.004), axis='x')
    outer_bezel = filleted_cylinder(0.014, 0.0015, fillet=0.0004, center=(0.037, 0, 0.004), axis='x')
    return np.vstack([core_emitter, outer_bezel])

def generate_dorsal_thrusters():
    """
    Aerodynamic twin-nacelle flight pack integrated on the dorsal spine.
    Dual turbine housings with conical exhaust vectoring nozzles.
    """
    housing = superellipsoid(0.018, 0.028, 0.030, r=0.35, s=0.35, center=(-0.032, 0, 0.004))
    turbine_l = filleted_cylinder(0.009, 0.036, fillet=0.0015, center=(-0.038, 0.016, 0.008), axis='z')
    turbine_r = filleted_cylinder(0.009, 0.036, fillet=0.0015, center=(-0.038, -0.016, 0.008), axis='z')
    nozzle_l = filleted_cylinder(0.007, 0.008, fillet=0.001, center=(-0.038, 0.016, -0.016), axis='z')
    nozzle_r = filleted_cylinder(0.007, 0.008, fillet=0.001, center=(-0.038, -0.016, -0.016), axis='z')
    return np.vstack([housing, turbine_l, turbine_r, nozzle_l, nozzle_r])

# =========================================================================
# 3. Bionic Limbs & Ground Contact Soles
# =========================================================================

def generate_bionic_hip(side=+1):
    """
    Bionic universal hip gimbal in dark carbon.
    Physically bridges the pelvic socket at (0,0,0) [roll axis X] to the thigh head
    at (0, side*0.012, -0.015) [pitch axis Y].
    """
    # 1. Roll Pivot Core (centered at 0,0,0, axis X, fitting into pelvic hip socket)
    roll_knuckle = filleted_cylinder(0.0085, 0.006, fillet=0.0015, center=(0, 0, 0), axis='x')
    
    # 2. Bionic Dogbone Link Arm (bridging 0,0,0 to 0, side*0.012, -0.015)
    dogbone = superellipsoid(0.010, 0.007, 0.009, r=0.3, s=0.3, center=(0.001, side * 0.006, -0.0075))
    
    # 3. Pitch Bearing Knuckle (at 0, side*0.012, -0.015, axis Y, where thigh attaches)
    pitch_knuckle = filleted_cylinder(0.0095, 0.0055, fillet=0.0015, center=(0, side * 0.012, -0.015), axis='y')
    
    # 4. Mechanical Accent Gusset
    gusset = superellipsoid(0.008, 0.0045, 0.007, r=0.25, s=0.25, center=(-0.002, side * 0.006, -0.0075))
    
    return np.vstack([roll_knuckle, dogbone, pitch_knuckle, gusset])

def generate_bionic_thigh(side=+1):
    """
    Bionic thigh armor in ceramic white.
    Sculpted muscle plate with upper trochanter cap cupping the hip knuckle,
    lateral protective hip shield, and sleek downward taper to the knee.
    """
    # Upper Femoral Trochanter Cap (cups the hip pitch knuckle at 0,0,0)
    trochanter = filleted_cylinder(0.012, 0.007, fillet=0.002, center=(0, 0, 0), axis='y')
    
    # Lateral Hip Shield (rises up to overlap the hip joint from the outside)
    hip_shield = superellipsoid(0.016, 0.004, 0.014, r=0.25, s=0.25, center=(0.002, side * 0.008, 0.004))
    
    # Main Sculpted Thigh Muscle Casing
    thigh_body = superellipsoid(0.014, 0.009, 0.019, r=0.35, s=0.35, center=(0.002, 0, -0.020))
    
    # Lateral Aerodynamic Rib
    rib = superellipsoid(0.016, 0.003, 0.015, r=0.25, s=0.25, center=(0.004, side * 0.009, -0.020))
    
    return np.vstack([trochanter, hip_shield, thigh_body, rib])

def generate_bionic_shin():
    """
    Bionic shin segment in dark carbon.
    Includes precision knee cap at (0,0,0), forward knee strike shield, and calf casing.
    """
    # Precision Knee Pivot Cap
    knee_pivot = filleted_cylinder(0.011, 0.010, fillet=0.0015, center=(0, 0, 0.0), axis='y')
    
    # Forward Angled Knee Strike Plate
    knee_shield = superellipsoid(0.013, 0.009, 0.012, r=0.25, s=0.25, center=(0.012, 0, -0.002))
    
    # Tapered Bionic Shin / Calf Casing
    shin_body = superellipsoid(0.011, 0.008, 0.015, r=0.35, s=0.35, center=(-0.001, 0, -0.018))
    
    # Ankle Pivot Bearing (centered at foot mount 0.004, 0, -0.036)
    ankle_bearing = filleted_cylinder(0.008, 0.010, fillet=0.001, center=(0.004, 0, -0.036), axis='y')
    
    return np.vstack([knee_pivot, knee_shield, shin_body, ankle_bearing])

def generate_bionic_foot(side=+1):
    """
    Athletic 3-claw bionic foot in metallic gold with ankle clevis at (0,0,0),
    compact arched heel counter, and 3 sleek tapered forward claws.
    Proportional to athletic biped body (length ~50mm, width ~26mm).
    """
    # Ankle Clevis Mount (centered at 0,0,0)
    clevis = filleted_cylinder(0.007, 0.009, fillet=0.001, center=(0.0, 0, 0.0), axis='y')
    
    # Compact Arched Heel Counter (reaches back to -0.019)
    heel = superellipsoid(0.011, 0.012, 0.0055, r=0.3, s=0.3, center=(-0.008, 0, -0.001))
    
    # Central Forward Claw (reaches forward to +0.031)
    toe_mid = superellipsoid(0.016, 0.0045, 0.0045, r=0.3, s=0.3, center=(0.015, 0, -0.002))
    
    # Left & Right Outrigger Claws (flaring out to y = ±0.011)
    toe_l = superellipsoid(0.014, 0.004, 0.0045, r=0.3, s=0.3, center=(0.013, 0.010, -0.002))
    toe_r = superellipsoid(0.014, 0.004, 0.0045, r=0.3, s=0.3, center=(0.013, -0.010, -0.002))
    
    return np.vstack([clevis, heel, toe_mid, toe_l, toe_r])

def generate_bionic_sole():
    """
    High-traction rubber sole with parallel grip tread ribs.
    Proportionate to compact athletic foot (length ~48mm, width ~24mm).
    Bottom of tread ribs at local z = -0.0050m guarantees exact Z=0.000m ground contact!
    """
    # Rubber sole base plate (length ~48mm from x = -0.019 to +0.029)
    base = superellipsoid(0.024, 0.013, 0.0012, r=0.2, s=0.3, center=(0.005, 0, -0.0025))
    
    # 5 Parallel Rubber Grip Treads (width 22mm, half_length = 0.011)
    treads = []
    for x in [-0.012, -0.003, 0.006, 0.015, 0.024]:
        tr = filleted_cylinder(0.0010, 0.011, fillet=0.0003, center=(x, 0, -0.0040), axis='y')
        treads.append(tr)
    return np.vstack([base] + treads)

# =========================================================================
# Main Builder Pipeline
# =========================================================================

def build_all_cohesive():
    out_dir = "duck_v2/meshes"
    os.makedirs(out_dir, exist_ok=True)
    
    parts = {
        "cyber_helm.stl": generate_sleek_helmet(),
        "cyber_visor.stl": generate_panoramic_visor(),
        "cyber_beak.stl": generate_sculpted_beak(),
        "cyber_neck.stl": generate_bionic_neck(),
        "cyber_chest.stl": generate_sculpted_chest(),
        "cyber_reactor.stl": generate_multi_tier_reactor(),
        "cyber_thruster_pack.stl": generate_dorsal_thrusters(),
        "cyber_hip_left.stl": generate_bionic_hip(+1),
        "cyber_hip_right.stl": generate_bionic_hip(-1),
        "cyber_thigh_left.stl": generate_bionic_thigh(+1),
        "cyber_thigh_right.stl": generate_bionic_thigh(-1),
        "cyber_shin_left.stl": generate_bionic_shin(),
        "cyber_shin_right.stl": generate_bionic_shin(),
        "cyber_foot_left.stl": generate_bionic_foot(+1),
        "cyber_foot_right.stl": generate_bionic_foot(-1),
        "cyber_sole_left.stl": generate_bionic_sole(),
        "cyber_sole_right.stl": generate_bionic_sole(),
    }
    
    # Clean up old wing STL files if present
    for old_file in ["cyber_wing_left.stl", "cyber_wing_right.stl"]:
        old_p = os.path.join(out_dir, old_file)
        if os.path.exists(old_p):
            os.remove(old_p)
    
    total_tris = 0
    for name, tris in parts.items():
        p = os.path.join(out_dir, name)
        write_binary_stl(p, tris)
        total_tris += len(tris)
        print(f"Generated {p:<36}: {len(tris):5d} triangles")
    print(f"\nAthletic CyberDuck V2 built successfully! Total smooth triangles: {total_tris:,}")

if __name__ == "__main__":
    build_all_cohesive()
