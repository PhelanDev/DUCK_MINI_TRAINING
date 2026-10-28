# Thiết Kế Kiến Trúc & Mô Hình Mô Phỏng Robot MicroDuck 8-DOF (MG996R)
**Ngày lập:** 2026-10-28  
**Trạng thái:** Chờ duyệt kế hoạch triển khai (Pending Implementation Plan)  
**Mục tiêu:** Xây dựng mô hình vật lý 8 servo (MG996R) chuẩn hóa cho MuJoCo & URDF, tích hợp đường ống chạy mô phỏng song song (Parallel Simulation) phục vụ huấn luyện học tăng cường (Reinforcement Learning - Sim-to-Real).

---

## 1. Bối Cảnh & Động Lực (Background & Motivation)
Robot gốc MicroDuck (Pollen Robotics & Hugging Face) sử dụng 15 servo Dynamixel XL330 đắt đỏ và phức tạp. Để đồ án đạt tính thực tiễn cao, chi phí hợp lý và dễ chế tạo tại Việt Nam, hệ thống được tinh giản về **8 động cơ servo MG996R**:
- **Cụm chân (6 DOF - 3 DOF/chân):** 
  - Hip Roll (trục X): Lắc nghiêng hông để chuyển trọng tâm sang chân trụ khi bước đi.
  - Hip Pitch (trục Y): Đưa chân tiến/lùi.
  - Knee Pitch (trục Y): Co/gập gối nâng chân tránh vấp sàn.
  - Bàn chân bản rộng (Wide footplate) có đế cao su ma sát cao, cố định với cẳng chân để loại bỏ khớp cổ chân.
- **Cụm đầu (2 DOF):** 
  - Head Yaw (trục Z): Quay trái/phải.
  - Head Pitch (trục Y): Gật lên/xuống hỗ trợ camera quan sát.

---

## 2. Thông Số Động Học & Vật Lý (Kinematics & Physics Spec)

### 2.1. Phân bổ khối lượng & Kích thước
- **Tổng chiều cao:** $\sim 230\text{ mm}$.
- **Tổng khối lượng:** $\approx 780\text{ g}$ (8 servo MG996R $\times 55\text{ g} = 440\text{ g}$, thân và khung in 3D $\sim 160\text{ g}$, pin 2S $\sim 120\text{ g}$, mạch ESP32 + IMU $\sim 60\text{ g}$).
- **Trọng tâm (CoM):** Đặt thấp tại nửa dưới `base_link` để tăng độ ổn định động học.

| Link Name | Khối lượng (g) | Kích thước (mm) | Mô tả |
| :--- | :--- | :--- | :--- |
| `base_link` | 320 | $110 \times 80 \times 70$ | Thân chính, chứa pin 2S, mạch điều khiển, IMU ở tâm |
| `head_yaw_link` / `neck` | 30 | $35 \times 35 \times 30$ | Khớp quay cổ |
| `head_link` | 65 | $60 \times 50 \times 45$ | Đầu vịt, gắn camera và mỏ |
| `left_hip_roll_link` | 60 | $45 \times 40 \times 40$ | Khối xoay hông ngang |
| `left_thigh_link` | 70 | $35 \times 30 \times 65$ | Đùi (chiều dài đùi $L_1 = 65\text{ mm}$) kèm servo gối |
| `left_shin_link` | 50 | $30 \times 25 \times 65$ | Cẳng chân (chiều dài $L_2 = 65\text{ mm}$) |
| `left_foot_link` | 35 | $75 \times 55 \times 10$ | Bàn chân bản to, mặt đế tiếp xúc ma sát lớn |
| `right_*` | Đối xứng | Đối xứng | Các link chân phải đối xứng qua trục X |

### 2.2. Đặc tính Actuator MG996R trong MuJoCo
- **Giới hạn mô-men xoắn (Stall Torque):** $\tau_{\max} = 1.0\text{ N}\cdot\text{m}$ (tương đương $\sim 10.2\text{ kg}\cdot\text{cm}$ ở 5V).
- **Vận tốc góc tối đa:** $\omega_{\max} = 6.0\text{ rad/s}$ ($\approx 344^\circ/\text{s}$).
- **Giảm chấn khớp (Damping):** $d = 0.15\text{ N}\cdot\text{s/rad}$ (đặc tính nhông kim loại).
- **Armature (Quán tính rotor):** $0.005\text{ kg}\cdot\text{m}^2$.
- **Giới hạn góc quay:**
  - `head_yaw`: $[-90^\circ, +90^\circ]$ (rad: $[-1.57, +1.57]$)
  - `head_pitch`: $[-30^\circ, +45^\circ]$ (rad: $[-0.52, +0.78]$)
  - `hip_roll`: $[-25^\circ, +25^\circ]$ (rad: $[-0.44, +0.44]$)
  - `hip_pitch`: $[-50^\circ, +50^\circ]$ (rad: $[-0.87, +0.87]$)
  - `knee_pitch`: $[0^\circ, +85^\circ]$ (rad: $[0.0, +1.48]$)

### 2.3. Mô hình tiếp xúc bàn chân (Contact Model)
- Ma sát trượt/lăn: `friction="1.2 0.005 0.0001"`.
- Tham số ổn định tiếp xúc không nảy: `solref="0.005 1"` và `solimp="0.9 0.95 0.001 0.5 2"`.

---

## 3. Kiến Trúc Mô Phỏng Song Song Để Huấn Luyện (Parallel RL Training Pipeline)

Để huấn luyện hàng nghìn robot ảo cùng lúc (từ $2048$ đến $8192$ môi trường) trên GPU:
1. **MuJoCo MJCF XML (`microduck_8dof.xml`):**
   - Định nghĩa hình học va chạm chuẩn `capsule` và `box` để tối đa hóa tốc độ dò va chạm GPU (Warp collision pipeline).
   - Thiết lập các site cảm biến: IMU (gia tốc kế 3 trục, con quay hồi chuyển 3 trục) tại tâm thân, cảm biến tiếp xúc (contact sensors) tại 4 góc mỗi bàn chân.
2. **Không gian quan sát (Observation Space - 37 dims):**
   - Vận tốc góc gốc thân (Base angular velocity): 3 dims.
   - Véc-tơ trọng trường quy chiếu thân (Projected gravity): 3 dims.
   - Lệnh vận tốc mục tiêu (Velocity command $[v_x, v_y, \omega_z]$): 3 dims.
   - Góc hiện tại của 8 khớp (Joint positions): 8 dims.
   - Vận tốc hiện tại của 8 khớp (Joint velocities): 8 dims.
   - Hành động ở bước trước (Previous actions): 8 dims.
   - Lực tiếp xúc bàn chân (Feet contact status): 4 dims (tùy chọn).
3. **Không gian hành động (Action Space - 8 dims):**
   - Độ lệch góc mục tiêu cho 8 servo: $\Delta q \in [-1, 1]$ ánh xạ sang dải làm việc vật lý.
4. **Hàm thưởng (Reward Functions):**
   - $R_{\text{track\_v\_xy}}$: Bám sát vận tốc dài mong muốn.
   - $R_{\text{track\_omega\_z}}$: Bám sát tốc độ quay yaw mong muốn.
   - $R_{\text{upright}}$: Thưởng cho tư thế thân thẳng đứng (trục Z hướng lên).
   - $R_{\text{feet\_air\_time}}$: Thưởng cho nhịp bước chân lăng lên cao tránh lê chân.
   - $P_{\text{torque}}, P_{\text{action\_rate}}$: Phạt tiêu tốn năng lượng và phạt giật góc đột ngột để bảo vệ nhông MG996R khi nạp xuống robot thật (Sim-to-Real safety).

---

## 4. Cấu Trúc Dự Án (Deliverables)

```text
d:\duckV1\
├── docs\
│   └── superpowers\
│       └── specs\
│           └── 2026-10-28-microduck-8dof-design.md     # Bản đặc tả này
├── models\
│   ├── mjcf\
│   │   └── microduck_8dof.xml                          # Mô hình MJCF chính thức
│   └── urdf\
│       └── microduck_8dof.urdf                         # Mô hình URDF đồng bộ
├── training\
│   ├── __init__.py
│   ├── env_config.py                                   # Cấu hình observation, action, reward
│   └── run_parallel_sim.py                             # Script khởi chạy N môi trường song song kiểm tra FPS
└── scripts\
    ├── verify_model.py                                 # Kiểm tra tính toàn vẹn vật lý và load thử vào MuJoCo
    └── interactive_viewer.py                           # Cửa sổ 3D kéo thanh trượt test góc
```

---

## 5. Kế Hoạch Xác Thực (Verification Plan)
1. **Kiểm tra cú pháp & cấu trúc:** Chạy `python scripts/verify_model.py` để biên dịch MJCF và URDF với MuJoCo engine, kiểm tra không có lỗi va chạm tự thân (self-collision) bất thường, kiểm tra khối lượng tổng $\sim 780\text{ g}$.
2. **Kiểm tra độ vững tĩnh:** Đặt robot đứng trên sàn phẳng với trọng lực $-9.81\text{ m/s}^2$, kiểm tra robot không bị sụp khớp gối dưới mô-men giữ của MG996R.
3. **Kiểm tra khả năng chạy song song:** Chạy `python training/run_parallel_sim.py --num_envs 64` (hoặc test vector envs) để xác nhận tải song song nhiều con vịt mô phỏng mượt mà không bị nghẽn RAM/CPU.
4. **Kiểm tra giao diện tương tác:** Chạy `scripts/interactive_viewer.py` để trực tiếp điều khiển cả 8 khớp.
