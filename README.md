# MicroDuck 8-DOF

Robot hai chân 8 bậc tự do sử dụng servo kim loại MG996R, phục vụ nghiên cứu và huấn luyện học tăng cường Sim-to-Real trên ba cấu hình phần cứng độc lập: Raspberry Pi 5, Dual ESP32-S3 và Raspberry Pi Zero 2 W.

Mô hình vật lý được đồng bộ song song giữa MuJoCo (MJCF) để huấn luyện mô phỏng và URDF để kiểm định động học, trực quan hóa trên ROS hoặc PyBullet.

---

## Mục lục

1. [Cấu hình phần cứng và động học](#1-cấu-hình-phần-cứng-và-động-học)
2. [Ba cấu hình phần cứng](#2-ba-cấu-hình-phần-cứng)
3. [Cụm nhận thức đầu vịt](#3-cụm-nhận-thức-đầu-vịt)
4. [Cài đặt và chạy nhanh](#4-cài-đặt-và-chạy-nhanh)
5. [Sử dụng CLI duck.py](#5-sử-dụng-cli-duckpy)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Giấy phép](#7-giấy-phép)

---

## 1. Cấu hình phần cứng và động học

Robot sử dụng 8 servo MG996R nhông kim loại (mô-men xoắn danh định 1.0 N·m, vận tốc góc 6.0 rad/s). Cấu hình này thay thế cho bản gốc dùng 15 servo Dynamixel nhằm giảm chi phí linh kiện và tăng khả năng chịu va đập khi té ngã.

### Thông số hình học Chibi Squat

Phiên bản hiện tại sử dụng cấu hình chân ngắn (Chibi Squat) với tỷ lệ đầu, thân và chân xấp xỉ 1 : 0.9 : 1. Chiều cao toàn thân đạt 195 mm, trọng tâm dồn thấp về đáy thân chính:

- Chiều dài đùi ($L_1$): 32 mm (khối lượng 40 g).
- Chiều dài cẳng chân ($L_2$): 32 mm (khối lượng 30 g).
- Kích thước bàn chân: 52 × 42 × 8 mm (khối lượng 22 g).
- Khoảng hở giữa mép trong hai bàn chân: 28 mm đến 58 mm, tránh va chạm cơ học khi bước đi hoặc lắc hông.
- Vị trí gốc thân chính (`base_link`): $Z = 95\text{ mm}$ ($0.095\text{ m}$).
- Cánh tay đòn trọng lực ngắn giúp mô-men lật nghiêng khi nghiêng người bước đi giảm xuống dưới 0.35 N·m, nằm an toàn trong giới hạn 1.0 N·m của động cơ MG996R.

### Bảng kích thước và khối lượng các khâu

| Khâu | Khối lượng | Kích thước ($X \times Y \times Z$) | Ghi chú |
| :--- | :---: | :---: | :--- |
| `base_link` | 275 g - 390 g | 110 × 80 × 70 mm | Thân chính, chứa mạch điều khiển, pin và cảm biến IMU |
| `neck_link` | 30 g | 35 × 35 × 30 mm | Khớp quay cổ quanh trục thẳng đứng $Z$ |
| `head_link` | 60 g | 60 × 50 × 45 mm | Đầu vịt, mang camera, microphone và cụm 5 tia LiDAR |
| `left_hip_roll_link` | 50 g | 45 × 40 × 40 mm | Khớp lắc hông quanh trục $X$ để chuyển trọng tâm |
| `left_thigh_link` | 40 g | 35 × 30 × 32 mm | Đùi $L_1 = 32\text{ mm}$, gắn servo khớp gối |
| `left_shin_link` | 30 g | 30 × 25 × 32 mm | Cẳng chân $L_2 = 32\text{ mm}$ |
| `left_foot_link` | 22 g | 52 × 42 × 8 mm | Bàn chân đế phẳng, mặt dưới gắn cao su tăng ma sát |
| `right_*_link` | Đối xứng | Đối xứng | Đối xứng với nhánh chân trái qua trục $X$ |

### Bảng góc quay và giới hạn các khớp

| Tên khớp | Trục quay | Giới hạn góc (Radian) | Giới hạn góc (Độ) | Mô-men cực đại | Vận tốc cực đại |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `head_yaw` | $Z$ | $[-1.57, +1.57]$ | $\pm 90^\circ$ | 1.0 N·m | 6.0 rad/s |
| `head_pitch` | $Y$ | $[-0.52, +0.78]$ | $-30^\circ \to +45^\circ$ | 1.0 N·m | 6.0 rad/s |
| `left_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | 1.0 N·m | 6.0 rad/s |
| `left_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | 1.0 N·m | 6.0 rad/s |
| `left_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | 1.0 N·m | 6.0 rad/s |
| `right_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | 1.0 N·m | 6.0 rad/s |
| `right_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | 1.0 N·m | 6.0 rad/s |
| `right_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | 1.0 N·m | 6.0 rad/s |

---

## 2. Ba cấu hình phần cứng

Dự án tách biệt thành 3 nhánh phần cứng độc lập trong thư mục `pipelines/`. Mỗi nhánh có mô hình vật lý, môi trường huấn luyện và runtime suy luận riêng.

| Tiêu chí | Raspberry Pi 5 Stack | Dual ESP32-S3 Stack | Raspberry Pi Zero 2 W Stack |
| :--- | :--- | :--- | :--- |
| **Thư mục** | `pipelines/rpi5_stack/` | `pipelines/dual_esp32s3_stack/` | `pipelines/pizero2w_stack/` |
| **Tổng khối lượng** | 764 g (0.764 kg) | 684 g (0.684 kg) | 649 g (0.649 kg) |
| **Vi xử lý** | Broadcom BCM2712 Quad A76 @ 2.4GHz | 2x ESP32-S3 Dual LX7 @ 240MHz | Broadcom BCM2710A1 Quad A53 @ 1.0GHz |
| **Bộ nhớ** | 4GB / 8GB LPDDR4X | 2x 512KB SRAM + 8MB PSRAM | 512MB LPDDR2 |
| **Hệ điều hành** | Raspberry Pi OS 64-bit | FreeRTOS (bare metal) | Raspberry Pi OS Lite 64-bit |
| **Điều khiển servo** | Mạch mở rộng PCA9685 qua bus I2C | 8 kênh LEDC PWM trực tiếp | Mạch mở rộng PCA9685 qua bus I2C |
| **Định dạng policy** | Float32 ONNX Runtime (50 Hz) | Mảng C++ lượng tử hóa INT8 ESP-NN (50 Hz) | Float32 ONNX Runtime (50 Hz) |
| **Giao tiếp liên chip** | Bus nội bộ | UART nhị phân 921,600 bps có mã kiểm tra CRC16 | Bus nội bộ |
| **Mục đích sử dụng** | Nghiên cứu AI biên, tích hợp xử lý ảnh hoặc NPU | Sản phẩm nhúng chi phí thấp, tiết kiệm pin | Bản Linux độc lập, kích thước gọn nhẹ |

---

## 3. Cụm nhận thức đầu vịt

Cả ba mô hình đều tích hợp cụm cảm biến trên đầu (`head_link`) và thân chính (`base_link`):

- **LiDAR 5 tia ToF:** Cảm biến đo khoảng cách đặt theo hình quạt ở góc lệch -30°, -15°, 0°, +15°, +30°, tầm đo tối đa 2.0 m, tần số cập nhật 50 Hz.
- **Camera góc rộng:** FOV 60°, đặt ở mặt trước đầu vịt (`cam_lens`).
- **Microphone:** Đặt tại đỉnh đầu để thu nhận âm thanh.
- **IMU 6-DOF:** Đo gia tốc 3 trục và vận tốc góc 3 trục đặt tại tâm thân chính `base_link`.
- **Cảm biến tiếp đất:** Đặt dưới đế hai bàn chân (`left_foot_touch_site`, `right_foot_touch_site`) để nhận biết pha chạm sàn.

---

## 4. Cài đặt và chạy nhanh

### Yêu cầu môi trường

- Python 3.10 trở lên
- Hệ điều hành: Linux, macOS hoặc Windows

### Cài đặt các gói phụ thuộc

```bash
git clone https://github.com/PhelanDev/DUCK_MINI_TRAINING.git
cd DUCK_MINI_TRAINING
pip install numpy pytest onnxruntime
```

Nếu môi trường có cài đặt sẵn MuJoCo:
```bash
pip install mujoco
```

### Kiểm định mô hình vật lý

Chạy script kiểm chứng sự đồng bộ giữa mô hình MuJoCo và URDF:

```bash
python scripts/verify_model.py
```

Kết quả in ra sẽ đối chiếu thông số khớp, ma trận quán tính, bất đẳng thức tam giác và vị trí trọng tâm CoM trên cả hai mô hình.

---

## 5. Sử dụng CLI duck.py

Công cụ dòng lệnh `duck.py` cung cấp các lệnh quản trị và vận hành toàn dự án:

### Xem trạng thái hệ thống
```bash
python duck.py status
```
Hiển thị thông tin kiểm định mô hình, các checkpoint hiện có và tình trạng của 3 cấu hình phần cứng.

### Kiểm định toàn bộ mô hình
```bash
python duck.py verify
```

### Mở trình xem 3D trực quan
```bash
python duck.py view
```
Mở giao diện mô phỏng 3D tương tác trên nền MuJoCo với các phím tắt điều khiển dáng đi, ngồi xuống, nằm nghỉ và đứng dậy.

### Huấn luyện học tăng cường (PPO) theo target
```bash
# Huấn luyện cho Raspberry Pi 5
python duck.py train --target rpi5 --steps 2000000

# Huấn luyện cho Raspberry Pi Zero 2 W
python duck.py train --target pizero2w --steps 2000000

# Huấn luyện cho Dual ESP32-S3
python duck.py train --target esp32 --steps 2000000
```

### Xuất mô hình phục vụ triển khai
```bash
# Xuất mô hình ONNX cho Pi 5 hoặc Pi Zero 2 W
python duck.py export --target rpi5
python duck.py export --target pizero2w

# Xuất mảng trọng số C++ INT8 cho Dual ESP32-S3
python duck.py export --target esp32
```

---

## 6. Cấu trúc thư mục

```text
duckV1/
├── duck.py                                   # CLI điều hành dự án
├── README.md                                 # Tài liệu kỹ thuật
├── LICENSE                                   # Giấy phép MIT
├── .gitignore                                # Cấu hình loại trừ file tạm
├── models/                                   # Mô hình vật lý gốc
│   ├── mjcf/microduck_8dof.xml               # Mô hình MuJoCo XML gốc
│   ├── urdf/microduck_8dof.urdf              # Mô hình URDF gốc
│   ├── microduck_pizero2w_policy.onnx        # Policy ONNX xuất cho Pi Zero 2 W
│   └── microduck_ppo.pt                      # Checkpoint huấn luyện PyTorch
├── pipelines/                                # Ba cấu hình phần cứng độc lập
│   ├── rpi5_stack/                           # Nhánh Raspberry Pi 5
│   │   ├── models/ (mjcf & urdf)             # Mô hình vật lý 764 g
│   │   ├── training/                         # Môi trường huấn luyện Float32
│   │   ├── deployment/ (infer_rpi5.py)       # Runtime suy luận 50Hz qua PCA9685
│   │   └── docs/rpi5_hardware_spec.md        # Đặc tả kỹ thuật phần cứng
│   ├── dual_esp32s3_stack/                   # Nhánh Dual ESP32-S3
│   │   ├── models/ (mjcf & urdf)             # Mô hình vật lý 684 g
│   │   ├── training/                         # Môi trường huấn luyện tối ưu hóa INT8
│   │   ├── firmware/                         # Mã nguồn C++/Arduino cho MCU 1 và MCU 2
│   │   └── docs/dual_esp32_hardware_spec.md  # Đặc tả kỹ thuật phần cứng
│   └── pizero2w_stack/                       # Nhánh Raspberry Pi Zero 2 W
│       ├── models/ (mjcf & urdf)             # Mô hình vật lý 649 g
│       ├── training/                         # Môi trường huấn luyện Pi Zero 2 W
│       ├── deployment/ (infer_pizero2w.py)   # Runtime suy luận 50Hz qua PCA9685
│       └── docs/pizero2w_hardware_spec.md    # Đặc tả kỹ thuật phần cứng
├── scripts/                                  # Scripts tiện ích
│   ├── verify_model.py                       # Kiểm định tính hợp lệ của mô hình vật lý
│   ├── export_policy.py                      # Xuất mô hình sang ONNX hoặc C++ INT8
│   └── train_ppo.py                          # Script huấn luyện thuật toán PPO
├── training/                                 # Lõi mô phỏng MuJoCo
│   ├── env_config.py                         # Cấu hình không gian trạng thái và hàm thưởng
│   └── run_parallel_sim.py                   # Mô phỏng song song nhiều môi trường
├── docs/                                     # Tài liệu thiết kế
│   └── specs/                                # Các văn bản đặc tả kỹ thuật
└── viewer_3d.html                            # Giao diện trực quan hóa 3D WebGL
```

---

## 7. Giấy phép

Dự án được phân phối theo giấy phép mã nguồn mở [MIT License](LICENSE). Bản quyền thuộc về PhelanDev (2026).
