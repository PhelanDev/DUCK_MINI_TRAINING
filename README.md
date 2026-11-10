# MicroDuck 8-DOF: Bipedal Robot Platform & Tri-Stack Sim-to-Real Architecture

> **Nền tảng Robot Hai Chân MicroDuck 8-DOF** — Chuẩn hóa động học **Chibi Squat Kinematics**, tích hợp cụm cảm biến nhận thức đầu vịt (**5-ray ToF LiDAR + Camera + Mic**) và kiến trúc 3 nhánh phần cứng độc lập (**Raspberry Pi 5**, **Dual ESP32-S3**, **Raspberry Pi Zero 2 W**).

---

## 🦆 1. Tổng Quan Dự Án & Triết Lý Thiết Kế

Dự án phát triển và chế tạo robot hai chân (Bipedal Duck Robot) **MicroDuck 8-DOF**, tối ưu hóa toàn diện cho việc chế tạo giá rẻ, độ bền cao và tính sẵn có của linh kiện tại Việt Nam:
- **Hệ thống truyền động (8 DOF):** Sử dụng 8 động cơ servo kim loại phổ thông **MG996R** (mô-men xoắn định mức $1.0\text{ N}\cdot\text{m}$, vận tốc góc $6.0\text{ rad/s}$), thay thế hoàn toàn cấu hình 15 servo Dynamixel XL330 đắt đỏ và mỏng manh của các phiên bản ban đầu.
- **Đột phá hình học Chibi Squat Kinematics:** 
  * Thu ngắn chiều dài đùi và cẳng chân xuống **$32\text{ mm}$** (giảm hơn $50\%$ chiều dài chân).
  * Thu gọn kích thước bàn chân về **$52 \times 42 \times 8\text{ mm}$**, tăng khoảng hở giữa 2 chân lên **$28 - 58\text{ mm}$** (loại trừ hoàn toàn rủi ro 2 chân va đập khi lắc hông sải bước).
  * Hạ độ cao thân chính (`base_link`) xuống **$Z = 95\text{ mm}$**, đưa tổng chiều cao toàn thân về **$\approx 19.5\text{ cm}$**. Tỉ lệ cơ thể đạt chuẩn $1 : 0.9 : 1$ (đầu : thân : chân) mang vóc dáng Chibi múp míp, cực kỳ dễ thương.
  * Trọng tâm CoM dồn sâu xuống vùng đáy thân, giảm cánh tay đòn trọng lực $>50\%$. Mô-men lật nghiêng cực đại hạ xuống $\le 0.35\text{ N}\cdot\text{m}$ (bỏ xa giới hạn $1.0\text{ N}\cdot\text{m}$ của MG996R).
- **Mô phỏng vật lý đa nền tảng:** Đồng bộ hóa song song giữa mô hình **MuJoCo (MJCF XML)** dùng cho huấn luyện học tăng cường (RL - Reinforcement Learning) và **ROS/PyBullet (URDF)** dùng cho trực quan hóa, kiểm định động học và điều khiển Sim-to-Real.

---

## 🏗️ 2. Ba Nhánh Phần Cứng Độc Lập (Tri-Stack Architecture)

Hệ thống được thiết kế theo kiến trúc phân tách rõ ràng thành **3 stacks phần cứng**, đáp ứng từ phân khúc siêu tiết kiệm điện tới máy trạm tính toán biên cao cấp:

```text
                                 [MicroDuck 8-DOF Core]
                     (Chibi Squat Kinematics: L1=32mm, L2=32mm, 8x MG996R)
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
   [Raspberry Pi 5 Stack]      [Dual ESP32-S3 Stack]      [Pi Zero 2 W Stack]
    (pipelines/rpi5_stack)    (pipelines/dual_esp32s3)    (pipelines/pizero2w)
    • Khối lượng: 764g         • Khối lượng: 684g          • Khối lượng: 649g (Nhẹ nhất)
    • Linux OS 64-bit          • Bare-metal FreeRTOS       • Linux OS Headless
    • Quad Cortex-A76 2.4GHz   • Dual Xtensa LX7 240MHz    • Quad Cortex-A53 1.0GHz
    • Float32 ONNX Policy      • INT8 ESP-NN MCU2 Policy   • Float32/INT8 ONNX Policy
    • Giao tiếp I2C PCA9685    • UART Binary 921.6kbps     • Giao tiếp I2C PCA9685
    • Hỗ trợ Hailo-8 NPU       • 8x LEDC Direct PWM        • Standalone Single-Board
```

### Bảng So Sánh Chi Tiết 3 Hardware Stacks

| Thông số / Tiêu chí | Raspberry Pi 5 Stack | Dual ESP32-S3 Stack | Raspberry Pi Zero 2 W Stack |
| :--- | :--- | :--- | :--- |
| **Thư mục mã nguồn** | `pipelines/rpi5_stack/` | `pipelines/dual_esp32s3_stack/` | `pipelines/pizero2w_stack/` |
| **Tổng khối lượng** | **$764\text{ g}$** ($0.764\text{ kg}$) | **$684\text{ g}$** ($0.684\text{ kg}$) | **$649\text{ g}$** ($0.649\text{ kg}$) |
| **Vi xử lý trung tâm** | BCM2712 Quad A76 @ 2.4GHz | 2x ESP32-S3 Dual LX7 @ 240MHz | BCM2710A1 Quad A53 @ 1.0GHz |
| **Bộ nhớ RAM** | 4GB / 8GB LPDDR4X | 2x 512KB SRAM + 8MB PSRAM | 512MB LPDDR2 |
| **Hệ điều hành / Nền tảng**| Raspberry Pi OS 64-bit | FreeRTOS (Dual-MCU Bare Metal)| Raspberry Pi OS Lite (Headless)|
| **Điều khiển Servo** | I2C PCA9685 (PWM 16 kênh) | 8 kênh LEDC PWM trực tiếp | I2C PCA9685 (PWM 16 kênh) |
| **Định dạng Policy RL** | Float32 ONNX Runtime ($50\text{ Hz}$)| INT8 ESP-NN C++ Array ($50\text{ Hz}$)| Float32/INT8 ONNX ($50\text{ Hz}$) |
| **Cảm biến nhận thức** | 5-ray LiDAR, Camera CSI, Mic | 5-ray LiDAR, OV2640, I2S Mic | 5-ray LiDAR ToF, Camera, Mic |
| **Giao tiếp liên chip** | Bus nội bộ SoC | UART nhị phân $921,600\text{ bps}$ | Bus nội bộ SoC |
| **Mục đích sử dụng** | R&D chuyên sâu, xử lý VLM/SLAM | Sản phẩm đồ chơi giá rẻ, pin lâu | Robot cá nhân nhỏ gọn, full Linux |

---

## 📐 3. Cây Động Học & Thông Số Khâu Chibi Squat (Kinematic Tree)

```text
                  [base_link] (Thân chính, Pin LiPo 2S, Mạch nguồn, IMU 6-DOF)
                 /           \                                  \
        [neck_link]       [left_hip_roll_link]        [right_hip_roll_link]
             |                     |                            |
        [head_link]        [left_thigh_link]            [right_thigh_link]
                                   |                            |
                           [left_shin_link]             [right_shin_link]
                                   |                            |
                           [left_foot_link]             [right_foot_link]
```

### 3.1. Bảng Kích Thước & Khối Lượng Từng Khâu (Chibi Squat Kinematics)

| Khâu (Link Name) | Khối Lượng | Kích Thước Khối ($X \times Y \times Z$) | Đặc Tính Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `base_link` | $275\text{ g} - 390\text{ g}$ | $110 \times 80 \times 70\text{ mm}$ | Thân chính, vị trí $Z_{\text{base}} = 0.095\text{ m}$ |
| `neck_link` | $30\text{ g}$ | $35 \times 35 \times 30\text{ mm}$ | Cổ quay theo trục thẳng đứng ($Z$) |
| `head_link` | $60\text{ g}$ | $60 \times 50 \times 45\text{ mm}$ | Đầu vịt mang camera, mic và 5 tia LiDAR |
| `left_hip_roll_link` | $50\text{ g}$ | $45 \times 40 \times 40\text{ mm}$ | Cụm servo hông lắc nghiêng trọng tâm ($X$) |
| `left_thigh_link` | **$40\text{ g}$** | **$35 \times 30 \times 32\text{ mm}$** | Đùi Chibi ngắn ($L_1 = \mathbf{32\text{ mm}}$) |
| `left_shin_link` | **$30\text{ g}$** | **$30 \times 25 \times 32\text{ mm}$** | Cẳng chân Chibi ngắn ($L_2 = \mathbf{32\text{ mm}}$) |
| `left_foot_link` | **$22\text{ g}$** | **$52 \times 42 \times 8\text{ mm}$** | Bàn chân Chibi gọn gàng, đế cao su ma sát cao |
| `right_*_link` | Đối xứng | Đối xứng qua mặt phẳng $X$ | Nhánh chân phải đối xứng |

### 3.2. Bảng Giới Hạn Góc & Mô-Men Khớp MG996R

| Tên Khớp | Trục Quay | Giới Hạn Góc (Radian) | Giới Hạn Góc (Độ) | Mô-Men Cực Đại | Vận Tốc Cực Đại |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `head_yaw` | $Z$ | $[-1.57, +1.57]$ | $\pm 90^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `head_pitch` | $Y$ | $[-0.52, +0.78]$ | $-30^\circ \to +45^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `left_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `left_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `left_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `right_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `right_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `right_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |

---

## 👁️ 4. Cụm Cảm Biến Nhận Thức Đầu Vịt (Perception Suite)

Toàn bộ các mô hình MJCF và URDF của 3 stack đều trang bị đồng bộ cụm nhận thức:
- **LiDAR 5 tia ToF (`rangefinder`):** Bố trí hình quạt trên mỏ vịt với các góc lệch `[-30°, -15°, 0°, +15°, +30°]`, tầm quét tối đa $2.0\text{ m}$, tần số cập nhật $50\text{ Hz}$.
- **Camera góc rộng:** FOV $60^\circ$ gắn tại trán vịt, hướng về phía trước theo dõi vật cản và nhận diện người/khuôn mặt.
- **Microphone MEMS:** Đặt tại đỉnh đầu thu âm tương tác giọng nói hai chiều.
- **IMU 6-DOF:** Đo gia tốc 3 trục ($a_x, a_y, a_z$) và vận tốc góc 3 trục ($\omega_x, \omega_y, \omega_z$) đặt tại trọng tâm thân.
- **Cảm biến tiếp đất:** Đặt tại đế 2 bàn chân phát hiện pha chạm đất (stance phase) và pha lăng (swing phase).

---

## 🛠️ 5. Hướng Dẫn Sử Dụng & Vận Hành CLI (`duck.py`)

Dự án cung cấp công cụ dòng lệnh hợp nhất `duck.py` để điều hành mọi tác vụ:

### 5.1. Xem Trạng Thái Toàn Hệ Thống
```bash
python duck.py status
```
*Hiển thị bảng trạng thái mô hình vật lý, checkpoints, và tình trạng sẵn sàng của cả 3 hardware stacks (RPi5, Dual ESP32, PiZero2W).*

### 5.2. Kiểm Định Mô Hình Vật Lý (Cross-Verification)
```bash
# Kiểm tra mô hình mặc định
python duck.py verify

# Kiểm tra chuyên sâu từng mô hình bằng script
python scripts/verify_model.py --mjcf pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml --urdf pipelines/pizero2w_stack/models/urdf/microduck_pizero2w_8dof.urdf
```
*Kiểm tra tính nhất quán 8 khớp, ma trận quán tính dương, bất đẳng thức tam giác vật rắn, vị trí CoM và mô phỏng 100 bước không nổ số.*

### 5.3. Huấn Luyện Học Tăng Cường (PPO Training) Theo Từng Nhánh Phần Cứng
```bash
# Huấn luyện cho Raspberry Pi 5
python duck.py train --target rpi5 --steps 2000000

# Huấn luyện cho Raspberry Pi Zero 2 W
python duck.py train --target pizero2w --steps 2000000

# Huấn luyện cho Dual ESP32-S3
python duck.py train --target esp32 --steps 2000000
```

### 5.4. Xuất Policy Triển Khai (Export Deployment Artifacts)
```bash
# Xuất ONNX model (cho RPi 5 và Pi Zero 2 W)
python duck.py export --target rpi5
python duck.py export --target pizero2w

# Xuất C++ INT8 Header (cho Dual ESP32-S3)
python duck.py export --target esp32
```

### 5.5. Chạy Trực Quan Hóa 3D (Interactive Viewer)
```bash
python duck.py view
```
*Mở giao diện mô phỏng vật lý 3D với các phím nóng điều khiển dáng đi waddling, ngồi xuống, nằm nghỉ, tự đứng dậy và vượt chướng ngại vật.*

---

## 📂 6. Cấu Trúc Thư Mục Dự Án (Repository Structure)

```text
E:\duckV1/
├── duck.py                                   # CLI hợp nhất điều hành toàn bộ dự án
├── README.md                                 # Tài liệu kiến trúc và hướng dẫn chính thức
├── .gitignore                                # Cấu hình loại trừ file rác, file tạm
├── models/                                   # Mô hình vật lý cơ sở & chính sách xuất
│   ├── mjcf/microduck_8dof.xml               # MuJoCo XML Base Chibi Squat Model
│   ├── urdf/microduck_8dof.urdf              # ROS/PyBullet URDF Base Model
│   ├── microduck_pizero2w_policy.onnx        # ONNX Policy tối ưu cho Pi Zero 2 W
│   └── microduck_ppo.pt                      # Checkpoint PyTorch huấn luyện
├── pipelines/                                # 3 Nhánh phần cứng Sim-to-Real chuyên biệt
│   ├── rpi5_stack/                           # [Stack 1] Raspberry Pi 5
│   │   ├── models/ (mjcf & urdf)             # Mô hình vật lý tải trọng 764g
│   │   ├── training/ (env_config, runner)    # Cấu hình PPO MuJoCo Float32
│   │   ├── deployment/ (infer_rpi5.py)       # Runtime suy luận 50Hz qua PCA9685
│   │   └── docs/rpi5_hardware_spec.md        # Đặc tả kỹ thuật phần cứng RPi 5
│   ├── dual_esp32s3_stack/                   # [Stack 2] Dual ESP32-S3
│   │   ├── models/ (mjcf & urdf)             # Mô hình vật lý tải trọng 684g
│   │   ├── training/ (env_config, runner)    # Môi trường PPO tối ưu hóa INT8
│   │   ├── firmware/                         # Mã nguồn C++/Arduino cho MCU 1 & MCU 2
│   │   └── docs/dual_esp32_hardware_spec.md  # Đặc tả kỹ thuật phần cứng Dual ESP32
│   └── pizero2w_stack/                       # [Stack 3] Raspberry Pi Zero 2 W
│       ├── models/ (mjcf & urdf)             # Mô hình vật lý tải trọng 649g
│       ├── training/ (env_config, runner)    # Cấu hình PPO MuJoCo song song
│       ├── deployment/ (infer_pizero2w.py)   # Runtime suy luận độc lập qua PCA9685
│       └── docs/pizero2w_hardware_spec.md    # Đặc tả kỹ thuật phần cứng Pi Zero 2 W
├── scripts/                                  # Công cụ kiểm chứng, xuất mô hình, huấn luyện
│   ├── verify_model.py                       # Cross-validation vật lý toàn diện
│   ├── export_policy.py                      # Xuất mô hình sang ONNX / C++ INT8
│   └── train_ppo.py                          # Thuật toán PPO đa môi trường
├── training/                                 # Lõi mô phỏng huấn luyện song song
│   ├── env_config.py                         # Không gian quan sát (42 dims) & phần thưởng
│   └── run_parallel_sim.py                   # Bộ chạy mô phỏng song song MuJoCo
├── docs/                                     # Tài liệu thiết kế và đặc tả kỹ thuật
│   └── specs/                                # Các tài liệu spec kỹ thuật chính thức
└── viewer_3d.html                            # Trình xem 3D trực quan trên nền WebGL
```

---

## 🔒 7. Bản Quyền & Giấy Phép
Dự án được phát triển và duy trì bởi **PhelanDev**. Mọi quyền được bảo lưu. Mọi đóng góp hoặc tích hợp phần cứng vui lòng tuân thủ quy chuẩn kỹ thuật và các quy định an toàn Sim-to-Real.
