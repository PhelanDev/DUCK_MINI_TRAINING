# 🦆 CYBERDUCK V2 PRO (PHIÊN BẢN ROBOT MECHA-AVIAN CAO CẤP)

> **Dự án:** Robot hai chân bipedal CyberDuck V2 Pro (High-Definition Parametric CAD & Khung Xương Cơ Khí)  
> **Ngôn ngữ thiết kế:** Bo cong siêu mượt mà (Continuous Filleted Superellipsoids & Right-Handed Outward Normals), Phối màu Giáp xám nhám (Matte Titanium Slate) kết hợp Chi tiết vàng đồng ánh kim (Burnished Metallic Bronze), Khớp háng đa hướng bionic gimbal liền mạch với hốc xương chậu (không hở khe, không gai nhọn xuyên giáp), Cụm cảm biến 5 tia ToF LiDAR micro-lens chìm phẳng trên viền kính ngắm Cyan Neon, Ba lô phản lực đẩy phản lực kép sau lưng, Bàn chân 3 móng vàng đồng bọc đế cao su rãnh gai tiếp xúc sàn phẳng tại $Z = 0.000\text{ m}$.  
> **Độ chi tiết hình học:** Hơn **125.780 tam giác cong mượt mà** trên 17 linh kiện giáp ngoài và bộ khung cơ khí bên trong độc lập.

---

## 📁 Cấu Trúc Thư Mục `duck_v2/`

```text
duck_v2/
├── cyberduck_v2.xml            # Mô hình vật lý MuJoCo 8-DOF hoàn chỉnh (Giáp xám nhám & vàng đồng)
├── meshes/                     # 17 file STL lưới 3D độ nét cao (>1000 tris/file)
│   ├── cyber_helm.stl          # Nón bảo hộ bo cong + gờ crest + tai nghe cảm biến (12.586 tris)
│   ├── cyber_visor.stl         # Kính ngắm toàn cảnh Cyan Neon (2.588 tris)
│   ├── cyber_beak.stl          # Mỏ vàng đồng khí động học + hàm dưới + khe gió (7.688 tris)
│   ├── cyber_neck.stl          # Khớp cổ bionic gimbal 2-DOF (9.734 tris)
│   ├── cyber_chest.stl         # Giáp ngực vát cơ bắp + hốc khớp hông bao bọc kín (17.360 tris)
│   ├── cyber_reactor.stl       # Lò phản ứng Arc Reactor ngực phát sáng (1.984 tris)
│   ├── cyber_thruster_pack.stl # Ba lô phản lực đẩy phản lực kép sau lưng (5.890 tris)
│   ├── cyber_hip_left.stl      # Khung khớp háng bionic gimbal bên trái (5.828 tris)
│   ├── cyber_hip_right.stl     # Khung khớp háng bionic gimbal bên phải (5.828 tris)
│   ├── cyber_thigh_left.stl    # Giáp đùi bionic + nắp bảo vệ mấu chuyển hông trái (6.758 tris)
│   ├── cyber_thigh_right.stl   # Giáp đùi bionic + nắp bảo vệ mấu chuyển hông phải (6.758 tris)
│   ├── cyber_shin_left.stl     # Khung cẳng chân carbon bionic trái (5.828 tris)
│   ├── cyber_shin_right.stl    # Khung cẳng chân carbon bionic phải (5.828 tris)
│   ├── cyber_foot_left.stl     # Khung chân 3 móng vàng đồng trái (8.680 tris)
│   ├── cyber_foot_right.stl    # Khung chân 3 móng vàng đồng phải (8.680 tris)
│   ├── cyber_sole_left.stl     # Đế cao su rãnh gai chống trượt trái (6.882 tris)
│   └── cyber_sole_right.stl    # Đế cao su rãnh gai chống trượt phải (6.882 tris)
├── cad/                        # File 3D CAD tiêu chuẩn (Vỏ giáp & Bộ khung xương cơ khí)
│   ├── cyberduck_v2_assembly.obj  # File 3D lắp ráp vỏ giáp hoàn chỉnh (4.40 MB)
│   ├── cyberduck_v2_assembly.mtl  # Thư viện màu sắc PBR titanium xám & vàng đồng
│   ├── cyberduck_v2_exploded.obj  # File 3D phân rã vỏ giáp (4.41 MB)
│   ├── cyberduck_v2_stl_parts.zip # Gói nén trọn bộ 17 file STL vỏ ngoài (1.55 MB)
│   ├── cyberduck_v2_skeleton.obj  # File 3D bộ khung cơ khí và 8 servo (383 KB)
│   ├── cyberduck_v2_skeleton.mtl  # Thư viện màu khung nhôm & vỏ servo MG996R
│   ├── cyberduck_v2_skeleton_parts.zip # Gói nén STL từng linh kiện khung để in 3D (31 KB)
│   └── skeleton_stl_parts/        # Thư mục chứa 17 file STL khung cơ khí & servo
├── media/                      # Bộ ảnh render studio HD đa góc nhìn
│   ├── cyberduck_v2_front_hd.png       # Render trực diện HD sạch bóng lỗi thị giác
│   ├── cyberduck_v2_iso_hd.png         # Render góc 3/4 Isometric HD giáp xám nhám & vàng đồng
│   ├── cyberduck_v2_skeleton_iso.png   # Render 3D bộ khung cơ khí & 8 servo góc Isometric
│   ├── cyberduck_v2_skeleton_front.png # Render trực diện bộ khung cơ khí
│   ├── cyberduck_v2_skeleton_side.png  # Render nhìn nghiêng bộ khung cơ khí
│   ├── cyberduck_v2_chassis_with_armor.png # Render X-Ray khung xương nằm trong vỏ giáp
│   ├── leg_joints_iso.png              # Sơ đồ 3 khớp chân 3D Isometric
│   ├── leg_joints_side.png             # Sơ đồ 3 khớp chân 3D nhìn ngang
│   └── leg_joints_front.png            # Sơ đồ 3 khớp chân 3D nhìn trực diện
├── scripts/
│   ├── build_cyberduck_v2.py          # Tái tạo 17 file STL vỏ giáp từ toán học tham số
│   ├── build_cyberduck_v2_skeleton.py # Tái tạo khung xương cơ khí và 8 servo
│   ├── render_cyberduck_v2.py         # Render bộ ảnh trực quan HD
│   ├── render_v2_hd.py                # Render & tự động kiểm tra pixel artifact
│   ├── render_skeleton.py             # Render bộ ảnh khung xương cơ khí
│   └── export_cyberduck_v2_cad.py     # Xuất file CAD OBJ + MTL + ZIP
```

---

## 💎 Chuẩn Thị Giác & Cơ Khí Nâng Cấp (Visual & Mechanical Upgrades)

1. **Khắc phục lỗi thị giác quả cầu đỏ & chấm đỏ cảm biến**:
   - 5 tia ToF LiDAR rangefinder chìm phẳng trên gờ nón nắp trên kính ngắm (`pos="0.026 0 0.022"`), góc quét nan quạt $[-30^\circ, +30^\circ]$, với điểm chỉ thị cyan micro-lens phát sáng neon, triệt tiêu hoàn toàn hiện tượng quả cầu đỏ trôi lơ lửng ngoài không khí.
   - Cảm biến IMU telemetry được tích hợp chìm trong thân, triệt tiêu chấm đỏ lồi bất thường trên ngực.
2. **Khắc phục lỗi gai nhọn & xuyên giáp ở khớp háng (Hip Joint)**:
   - Sửa thuật toán `filleted_cylinder`: đảo chiều vector pháp tuyến tam giác theo đúng quy tắc bàn tay phải (100% outward normals), triệt tiêu hiện tượng bóng đổ răng cưa và mặt tam giác bị xuyên thấu.
   - Định tâm lại hốc xương chậu `hip_socket_l/r` tại $Y = \pm 0.036\text{ m}$ khớp chính xác với trục xoay háng `left_hip` / `right_hip`.
   - Giảm bán kính ổ bi `roll_knuckle` từ $11\text{ mm}$ xuống $8.5\text{ mm}$, giúp khớp háng nằm gọn hoàn toàn bên trong hốc xương chậu, triệt tiêu 100% hai gai nhọn đâm xuyên giáp.
3. **Phối màu Hi-End Cyberpunk**:
   - Giáp chính: Xám nhám Titanium Slate (`rgba="0.34 0.36 0.40 1.0"`).
   - Chi tiết điểm nhấn: Vàng đồng ánh kim Burnished Metallic Bronze (`rgba="0.80 0.54 0.22 1.0"`).
   - Khung xương & Khớp nối: Đen Carbon nhám (`rgba="0.15 0.16 0.18 1.0"`).
   - Kính ngắm & Lò phản ứng: Xanh Cyan Neon phát quang rực rỡ (`rgba="0.0 0.92 1.0 1.0"`).
4. **Bộ Khung Cơ Khí Riêng Biệt (Mechanical Chassis & Servos)**:
   - File CAD 3D `cyberduck_v2_skeleton.obj` thể hiện đầy đủ cấu trúc khung chịu lực bên trong và vị trí lắp ráp 8 động cơ servo MG996R.
   - Gói file STL `cyberduck_v2_skeleton_parts.zip` tách riêng từng part cơ khí để sẵn sàng đưa vào phần mềm cắt lát (Slicer: Cura/PrusaSlicer) in 3D thực tế.

---

## 🚀 Lệnh Mở Xem Nhanh (PowerShell)

```powershell
# 1. Mở mô hình 3D vỏ giáp ngoài hoàn chỉnh bằng Windows 3D Viewer / Blender:
Start-Process "duck_v2/cad/cyberduck_v2_assembly.obj"

# 2. Mở mô hình 3D bộ khung xương cơ khí & 8 servo:
Start-Process "duck_v2/cad/cyberduck_v2_skeleton.obj"

# 3. Chạy trình giả lập tương tác 3D MuJoCo Viewer:
python duck.py view --v2

# 2. Mở mô hình 3D phân rã bung linh kiện (Exploded View):
Start-Process "duck_v2/cad/cyberduck_v2_exploded.obj"

# 3. Mở thư mục chứa file STL để đưa vào máy in 3D:
explorer.exe "duck_v2\cad"
```
