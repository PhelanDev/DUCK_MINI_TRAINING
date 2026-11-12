# 🦆 CYBERDUCK V2 PRO (PHIÊN BẢN ROBOT MECHA-AVIAN CAO CẤP)

> **Dự án:** Robot hai chân bipedal CyberDuck V2 Pro (High-Definition Parametric CAD)  
> **Ngôn ngữ thiết kế:** Bo cong siêu mượt mà (Continuous Filleted Superellipsoids), Mũ nón khí động học có gờ tản nhiệt & tai nghe cảm biến, Kính ngắm quang học mắt kép Cyan Neon phát sáng, Mỏ vịt vàng Gold kim loại lướt gió, Lò phản ứng Arc Reactor ngực, Cặp cánh cơ khí Swept Aero-Wings vát 3 tầng lông, Ba lô phản lực đẩy phản lực kép sau lưng, Giáp gối nhọn kèm xi lanh thủy lực, Bàn chân móng cơ khí 3 ngón với đế cao su rãnh gai tiếp xúc sàn phẳng tại $Z = 0.000\text{ m}$.  
> **Độ chi tiết hình học:** Hơn **77.800 tam giác cong mượt mà** (không còn hình hộp phẳng thô ráp).

---

## 📁 Cấu Trúc Thư Mục `duck_v2/`

```text
duck_v2/
├── cyberduck_v2.xml            # Mô hình vật lý MuJoCo 8-DOF hoàn chỉnh
├── meshes/                     # 13 file STL lưới 3D độ nét cao (>1000 tris/file)
│   ├── cyber_helm.stl          # Nón bảo hộ bo cong + gờ crest + tai nghe cảm biến (7.812 tris)
│   ├── cyber_visor.stl         # Kính ngắm thấu kính mắt kép Cyan Neon (2.914 tris)
│   ├── cyber_beak.stl          # Mỏ vàng khí động học + hàm dưới + khe gió (4.964 tris)
│   ├── cyber_chest.stl         # Giáp ngực cơ bắp bo tròn + gờ cổ (6.758 tris)
│   ├── cyber_reactor.stl       # Lò phản ứng Arc Reactor ngực phát sáng (1.984 tris)
│   ├── cyber_wing_left.stl     # Cánh cơ khí vát gió 3 tầng bên trái (6.758 tris)
│   ├── cyber_wing_right.stl    # Cánh cơ khí vát gió 3 tầng bên phải (6.758 tris)
│   ├── cyber_thruster_pack.stl # Ba lô phản lực đẩy phản lực kép sau lưng (5.890 tris)
│   ├── cyber_knee_guard.stl    # Giáp gối mũi nhọn + xi lanh thủy lực (2.914 tris)
│   ├── cyber_foot_left.stl     # Khung chân 3 móng cơ khí trái (8.680 tris)
│   ├── cyber_foot_right.stl    # Khung chân 3 móng cơ khí phải (8.680 tris)
│   ├── cyber_sole_left.stl     # Đế cao su rãnh gai chống trượt trái (6.882 tris)
│   └── cyber_sole_right.stl    # Đế cao su rãnh gai chống trượt phải (6.882 tris)
├── cad/                        # File 3D CAD tiêu chuẩn
│   ├── cyberduck_v2_assembly.obj  # File 3D lắp ráp hoàn chỉnh (2.81 MB)
│   ├── cyberduck_v2_assembly.mtl  # Thư viện màu sắc PBR ánh kim & phát quang
│   ├── cyberduck_v2_exploded.obj  # File 3D phân rã linh kiện (2.82 MB)
│   └── cyberduck_v2_stl_parts.zip # Gói nén trọn bộ 13 file STL (0.86 MB)
├── media/                      # Bộ ảnh render studio HD đa góc nhìn
│   ├── cyberduck_v2_front_hd.png   # Render trực diện HD sạch bóng lỗi thị giác
│   ├── cyberduck_v2_iso_hd.png     # Render góc 3/4 Isometric HD giáp trắng ngà
│   ├── cyberduck_v2_iso.png
│   ├── cyberduck_v2_front.png
│   ├── cyberduck_v2_side.png
│   ├── cyberduck_v2_wing_closeup.png
│   ├── cyberduck_v2_foot_closeup.png
│   └── cyberduck_v2_exploded.png
├── scripts/
│   ├── build_cyberduck_v2.py       # Tái tạo 13 file STL từ toán học tham số
│   ├── render_cyberduck_v2.py      # Render bộ ảnh trực quan HD
│   ├── render_v2_hd.py             # Render & tự động kiểm tra pixel artifact
│   └── export_cyberduck_v2_cad.py  # Xuất file CAD OBJ + MTL + ZIP
```

---

## 💎 Chuẩn Thị Giác & Ánh Sáng Studio Mới (Visual & Shading Upgrades)

1. **Triệt tiêu hoàn toàn quả cầu đỏ & chấm đỏ cảm biến**:
   - Toàn bộ sensory sites (`imu`, `imu_site`, `site_lidar_m30..p30`, foot touch sites) được cấu hình `rgba="0 0 0 0"` và đưa vào `group="4"`.
   - 5 tia LiDAR rangefinder được dời flush phẳng với kính ngắm (`pos="0.025 0 0.022"`), triệt tiêu hoàn toàn hiện tượng điểm đỏ trôi lơ lửng trước mặt robot.
2. **Nâng cấp ánh sáng Studio PBR & Triệt tiêu Shadow Acne**:
   - Thêm camera headlight fill (`ambient="0.45 0.47 0.50"`), loại bỏ góc chết tối đen trên giáp.
   - Nâng độ phân giải bản đồ đổ bóng lên `shadowsize="4096"` với `shadowclip="0.5"`.
   - Khóa `castshadow="false"` trên các đèn phụ, chỉ giữ 1 key light chính đổ bóng để triệt tiêu hiện tượng bóng răng cưa (shadow acne) chồng chéo.
   - Nâng cấp vật liệu `mecha_white` đạt chuẩn ceramic trắng tinh khiết (`rgba="0.98 0.99 1.0 1.0"`, `emission="0.08"`).
3. **Triệt tiêu viền vàng hộp va chạm ở bàn chân**:
   - Lớp `class="collision"` được gán `rgba="0.2 0.5 0.8 0"` và viewer group 3/4 được khóa mặc định, đảm bảo bàn chân hiển thị đúng chất liệu vàng kim và đế cao su.

---

## 🚀 Lệnh Mở Xem Nhanh (PowerShell)

```powershell
# 1. Mở mô hình 3D lắp ráp hoàn chỉnh bằng Windows 3D Viewer / Blender:
Start-Process "duck_v2/cad/cyberduck_v2_assembly.obj"

# 2. Mở mô hình 3D phân rã bung linh kiện (Exploded View):
Start-Process "duck_v2/cad/cyberduck_v2_exploded.obj"

# 3. Mở thư mục chứa file STL để đưa vào máy in 3D:
explorer.exe "duck_v2\cad"
```
