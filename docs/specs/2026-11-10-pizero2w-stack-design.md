# Thiết Kế Kiến Trúc Nhánh Phần Cứng Thứ 3: Raspberry Pi Zero 2 W (MicroDuck 8-DOF)

**Ngày lập:** 2026-11-10  
**Trạng thái:** Đã phê duyệt (Approved by User)  
**Mục tiêu:** Bổ sung nhánh phần cứng thứ ba (`pipelines/pizero2w_stack/`) vào hệ sinh thái robot hai chân MicroDuck 8-DOF, hoạt động song song và đồng bộ tuyệt đối với 2 nhánh hiện có (`rpi5_stack` và `dual_esp32s3_stack`).

---

## 1. Tổng Quan Nhánh Phần Cứng `pizero2w_stack`

Raspberry Pi Zero 2 W mang đến điểm cân bằng lý tưởng giữa khả năng tính toán hệ điều hành Linux đầy đủ và trọng lượng siêu nhẹ:
- **Vi xử lý trung tâm:** Raspberry Pi Zero 2 W (Broadcom BCM2710A1 Quad-core 64-bit ARM Cortex-A53 @ 1.0GHz, 512MB LPDDR2 RAM).
- **Hệ điều hành:** Raspberry Pi OS 64-bit Lite (Bookworm).
- **Tổng khối lượng danh định:** $\approx 745\text{ g} - 755\text{ g}$ (nhẹ hơn Pi 5 $\approx 110\text{ g}$, nhẹ hơn Dual ESP32-S3 $\approx 25\text{ g}$).
- **Cụm linh kiện điện tử (`pizero2w_electronics`):** $\approx 55\text{ g}$ (Bo Pi Zero 2 W $16\text{ g}$, tản nhiệt nhôm phẳng $8\text{ g}$, module Buck Mini-360 5V/2.5A $5\text{ g}$, bo điều khiển servo PCA9685 $10\text{ g}$, hệ thống cáp nối $16\text{ g}$).
- **Trọng tâm (CoM):** Hạ thấp hơn so với Pi 5, bố trí cân đối tại trục trung tâm thân robot ($Z \approx +12\text{ mm}$ từ base link).

---

## 2. So Sánh Thông Số Kỹ Thuật 3 Nhánh Phần Cứng

| Tiêu Chí | Nhánh 1: `rpi5_stack` | Nhánh 2: `dual_esp32s3_stack` | Nhánh 3: `pizero2w_stack` (Mới) |
| :--- | :--- | :--- | :--- |
| **SoC / Bộ não** | Raspberry Pi 5 (Quad A76 @ 2.4GHz) | Dual ESP32-S3 (Dual LX7 @ 240MHz) | Raspberry Pi Zero 2 W (Quad A53 @ 1.0GHz) |
| **Bộ nhớ RAM** | 4GB LPDDR4X | 512KB SRAM + 8MB PSRAM (mỗi MCU) | 512MB LPDDR2 |
| **Hệ điều hành** | Linux (Pi OS 64-bit) | Bare-metal / FreeRTOS (Dual Core) | Linux (Pi OS 64-bit Lite) |
| **Tổng khối lượng** | $\approx 860\text{ g}$ | $\approx 780\text{ g}$ | $\approx 750\text{ g}$ |
| **Khối lượng điện tử** | $140\text{ g}$ (Pi 5 + Active Cooler) | $90\text{ g}$ (2 MCU + Carrier board) | $55\text{ g}$ (Pi Zero 2W + tản nhiệt mỏng) |
| **Nguồn cấp máy tính**| $5.1\text{V} / 5.0\text{A}$ ($25\text{W}$, PD) | $5.0\text{V} / 2.0\text{A}$ ($10\text{W}$) | $5.1\text{V} / 2.5\text{A}$ ($12.5\text{W}$, Mini Buck) |
| **Nguồn cấp Servo** | $6.0\text{V} / 15.0\text{A}$ (Buck riêng) | $6.0\text{V} / 15.0\text{A}$ (Buck riêng) | $6.0\text{V} / 15.0\text{A}$ (Buck riêng) |
| **Suy luận Policy** | Float32 ONNX Runtime | INT8 C-Header Vector PIE | Float32 / FP16 ONNX Runtime ARM64 |
| **Độ trễ suy luận** | $< 0.4\text{ ms}$ | $\approx 0.8\text{ ms} - 1.2\text{ ms}$ | $\approx 1.2\text{ ms} - 1.6\text{ ms}$ (chu kỳ 50Hz = 20ms) |
| **Camera** | 22-pin MIPI CSI-2 | Cổng song song DVP OV2640 | Mini CSI-2 (qua cáp 22p-15p Zero) |
| **Microphone** | I2S INMP441 (40-pin GPIO) | I2S INMP441 (GPIO trực tiếp) | I2S INMP441 (40-pin GPIO) |
| **LiDAR 5 tia** | I2C TCA9548A / 5x ToF | I2C TCA9548A / 5x ToF | I2C TCA9548A / 5x ToF |

---

## 3. Cấu Trúc File & Thư Mục Đồng Bộ (`pipelines/pizero2w_stack`)

```text
d:\duckV1\
├── pipelines/
│   ├── rpi5_stack/
│   ├── dual_esp32s3_stack/
│   └── pizero2w_stack/
│       ├── models/
│       │   ├── mjcf/microduck_pizero2w_8dof.xml       # Mô hình MuJoCo MJCF khối lượng 750g
│       │   └── urdf/microduck_pizero2w_8dof.urdf      # Mô hình URDF đồng bộ hình học và động học
│       ├── training/
│       │   ├── __init__.py
│       │   ├── env_config.py                          # Cấu hình Obs 42D, bounds & normalization
│       │   └── run_parallel_sim.py                    # Huấn luyện mô phỏng song song MuJoCo
│       ├── deployment/
│       │   ├── __init__.py
│       │   └── infer_pizero2w.py                      # Kịch bản suy luận thời gian thực trên Pi Zero 2 W
│       └── docs/
│           └── pizero2w_hardware_spec.md              # Đặc tả nguồn 5V/2.5A, sơ đồ chân GPIO 40-pin
├── tests/
│   ├── test_rpi5_model.py
│   ├── test_esp32_model.py
│   └── test_pizero2w_model.py                         # Bộ kiểm thử tự động toàn diện cho Pi Zero 2 W
├── scripts/
│   ├── export_policy.py                               # Hỗ trợ xuất microduck_pizero2w_policy.onnx
│   └── verify_model.py                                # Hỗ trợ kiểm chứng mô hình vật lý 3D
└── duck.py                                            # Cập nhật hiển thị status & export cho 3 nhánh
```

---

## 4. Tiêu Chuẩn Kỹ Thuật Cụ Thể

### 4.1. Động học và Vật lý 3D (MJCF & URDF)
- Khớp: 8 khớp quay tiêu chuẩn (`head_yaw`, `head_pitch`, `left_hip_roll`, `left_hip_pitch`, `left_knee_pitch`, `right_hip_roll`, `right_hip_pitch`, `right_knee_pitch`).
- Cảm biến đầu vịt: Cụm 5 tia ToF LiDAR `[-30°, -15°, 0°, +15°, +30°]` với cự ly quét $2.0\text{ m}$, camera góc nhìn $60^\circ$ và microphone số.
- Động lực học: Mô-men xoắn servo MG996R giới hạn $1.0\text{ N}\cdot\text{m}$, vận tốc góc tối đa $6.0\text{ rad/s}$.

### 4.2. Kịch bản Suy luận Nhúng (`infer_pizero2w.py`)
- Thiết kế độc lập (Decoupled Architecture), không phụ thuộc vào gói mô phỏng huấn luyện `mujoco`, đảm bảo chạy mượt trên Raspberry Pi OS Lite và trong các môi trường CI/CD không có MuJoCo.
- Hỗ trợ đầy đủ cờ `--mock` (mô phỏng quan sát) và `--mock-pca` (mô phỏng driver PWM servo I2C).
- Chu kỳ điều khiển vòng lặp $50\text{ Hz}$ ($20\text{ ms}$), ghi nhận jitter và thời gian đáp ứng từng bước.

### 4.3. Kiểm thử Tự động (TDD & Verification)
- File kiểm thử [`tests/test_pizero2w_model.py`](file:///d:/duckV1/tests/test_pizero2w_model.py) xác thực:
  1. Cấu trúc MJCF: 8 khớp, giới hạn góc, actuator motor, IMU, 5 tia LiDAR, touch sensor bàn chân.
  2. Cấu trúc URDF: Các link, joint, visual và collision.
  3. Tính nhất quán động học & phân bổ khối lượng giữa MJCF và URDF.
  4. Script suy luận `infer_pizero2w.py`: Khả năng import, chuyển đổi góc sang PWM, và chạy CLI với cờ `--mock`.
