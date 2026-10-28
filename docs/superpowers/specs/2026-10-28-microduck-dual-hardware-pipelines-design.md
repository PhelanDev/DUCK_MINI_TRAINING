# Thiết Kế Kiến Trúc Hai Nhánh Phần Cứng: Raspberry Pi 5 vs Dual ESP32-S3 (MicroDuck 8-DOF)

**Ngày lập:** 2026-10-28  
**Trạng thái:** Đã duyệt bởi người dùng (Approved by User)  
**Mục tiêu:** Tinh chỉnh và chia tách dự án thành 2 nhánh pipeline phần cứng và huấn luyện độc lập:
1. **Nhánh 1 (`pipelines/rpi5_stack/`):** Sử dụng Raspberry Pi 5 4GB làm vi xử lý trung tâm, chạy mô hình mạng nơ-ron Float32 ONNX Runtime trên Linux, tải trọng robot $\sim 860\text{ g}$.
2. **Nhánh 2 (`pipelines/dual_esp32s3_stack/`):** Sử dụng 2 vi điều khiển ESP32-S3 phân tầng (MCU 1: Thị giác, Âm thanh & LiDAR $\rightarrow$ giao tiếp UART $\rightarrow$ MCU 2: Cân bằng đi bộ 50Hz chạy mô hình INT8), tải trọng robot $\sim 780\text{ g}$.
Loại bỏ hoàn toàn kiến trúc K230. Cả 2 nhánh đều được trang bị cụm cảm biến đầu vịt: **LiDAR 5 tia quét vật cản + Camera quang học + Microphone I2S MEMS**.

---

## 1. Cụm Cảm Biến Trên Đầu Vịt (Perception Suite)

Cả hai mô hình robot đều được nâng cấp cụm đầu vịt với 3 cảm biến nhận thức:
1. **LiDAR 5 tia (ToF Ray Scanner):**
   - 5 cảm biến `<rangefinder>` trong MuJoCo bố trí góc quạt ngang $[-30^\circ, -15^\circ, 0^\circ, +15^\circ, +30^\circ]$.
   - Tầm đo: $0.05\text{ m} - 2.0\text{ m}$.
   - Tác dụng: Cung cấp trực tiếp 5 giá trị khoảng cách tới vật cản vào vector quan sát (Observation Space) để chính sách học tăng cường (RL Policy) tự động né chướng ngại vật khi bước đi.
2. **Camera quang học (RGB Camera):**
   - Camera góc nhìn $60^\circ$ FOV gắn trên trán vịt kèm geom ống kính (`cam_lens`), chuẩn hóa trục quang học nhìn về trước ($+X$).
3. **Microphone I2S MEMS:**
   - Cổng thu âm thanh số và site cảm biến gắn trên đầu vịt (mô phỏng module INMP441) phục vụ nhận diện từ khóa giọng nói ("Hey Duck").

---

## 2. So Sánh Đặc Tính Kỹ Thuật Giữa 2 Nhánh

| Tiêu Chí | Nhánh 1: `rpi5_stack` | Nhánh 2: `dual_esp32s3_stack` |
| :--- | :--- | :--- |
| **Phần cứng chính** | Raspberry Pi 5 (4GB RAM, Quad Cortex-A76 @ 2.4GHz) | 2x ESP32-S3 DevKit N16R8 (Dual Xtensa LX7 @ 240MHz) |
| **Tổng khối lượng** | $\approx 860\text{ g}$ (Pi 5 + tản nhiệt quạt + nguồn Buck 5V/5A + pin) | $\approx 780\text{ g}$ (2 board mạch nhẹ, pin 2S 18650 đặt thấp) |
| **Trọng tâm (CoM)** | CoM cao hơn $\sim 10\text{ mm}$ (Pi 5 đặt ở lưng/ngực robot) | CoM thấp sát đáy thân |
| **Độ dài chân & Servo** | Đùi $65\text{ mm}$, cẳng chân $65\text{ mm}$, 8 servo MG996R | Đùi $65\text{ mm}$, cẳng chân $65\text{ mm}$, 8 servo MG996R |
| **Định dạng Policy** | Float32 ONNX Model (chạy trực tiếp ONNX Runtime Python) | INT8 Quantized Model (tăng tốc bằng tập lệnh vector PIE) |
| **Không gian quan sát** | 42 chiều (37 chiều động học + 5 khoảng cách LiDAR) | 42 chiều (37 chiều động học + 5 khoảng cách LiDAR) |
| **Không gian hành động**| 8 góc đặt mục tiêu servo MG996R | 8 góc đặt mục tiêu servo MG996R |
| **Giao tiếp phần cứng** | I2C điều khiển module PCA9685 hoặc UART servo driver | Giao thức gói nhị phân UART Packet ($921,600\text{ bps}$) giữa 2 MCU |

---

## 3. Cấu Trúc Cây Thư Mục Dự Án (Project Tree)

```text
d:\duckV1\
├── pipelines/
│   ├── rpi5_stack/
│   │   ├── models/
│   │   │   ├── mjcf/microduck_rpi5_8dof.xml            # Mô hình MJCF trọng lượng Pi 5 (860g)
│   │   │   └── urdf/microduck_rpi5_8dof.urdf           # Mô hình URDF đồng bộ Pi 5
│   │   ├── training/
│   │   │   ├── env_config.py                           # Cấu hình Obs 42 chiều, Float32 weights
│   │   │   └── run_parallel_sim.py                     # Script chạy mô phỏng song song Pi 5
│   │   ├── deployment/
│   │   │   └── infer_rpi5.py                           # Runtime điều khiển trên Raspberry Pi 5
│   │   └── docs/
│   │       └── rpi5_hardware_spec.md                   # Sơ đồ chân, nguồn 5V/5A, Camera CSI & I2S Mic
│   │
│   └── dual_esp32s3_stack/
│       ├── models/
│       │   ├── mjcf/microduck_esp32_8dof.xml           # Mô hình MJCF trọng lượng Dual ESP32 (780g)
│       │   └── urdf/microduck_esp32_8dof.urdf          # Mô hình URDF đồng bộ Dual ESP32
│       ├── training/
│       │   ├── env_config.py                           # Cấu hình Obs 42 chiều, INT8 scaling
│       │   └── run_parallel_sim.py                     # Script chạy mô phỏng song song ESP32
│       ├── deployment/
│       │   ├── mcu1_vision_audio_s3/                   # Mã nguồn MCU 1 (Vision, Audio, ToF LiDAR)
│       │   │   └── mcu1_main.ino
│       │   ├── mcu2_locomotion_s3/                     # Mã nguồn MCU 2 (Locomotion RL Policy 50Hz)
│       │   │   └── mcu2_main.ino
│       │   └── uart_packet_protocol.h                  # Giao thức truyền tin nhị phân tốc độ cao
│       └── docs/
│           └── dual_esp32s3_hardware_spec.md           # Sơ đồ mạch phân tầng 2 MCU + Pin 2S + MG996R
│
├── tests/                                              # Bộ kiểm thử tự động cho cả 2 nhánh
└── README.md                                           # Hướng dẫn lựa chọn và huấn luyện 2 nhánh
```

---

## 4. Kế Hoạch Xác Thực (Verification Criteria)
1. **Mô hình vật lý:** Cả 2 mô hình `microduck_rpi5_8dof.xml` và `microduck_esp32_8dof.xml` đều biên dịch hợp lệ trên MuJoCo, hiển thị đầy đủ 5 tia rangefinder, camera và mic. Khối lượng khớp đúng thông số ($\sim 860\text{ g}$ cho Pi 5 và $\sim 780\text{ g}$ cho Dual ESP32).
2. **Huấn luyện song song:** Cả 2 nhánh `rpi5_stack` và `dual_esp32s3_stack` đều có script `run_parallel_sim.py` chạy mượt mà trên môi trường vector hóa đạt $>2,000\text{ FPS}$.
3. **Triển khai nhúng:**
   - Script `infer_rpi5.py` nạp mô hình ONNX và điều khiển servo thành công.
   - Giao thức `uart_packet_protocol.h` và firmware mẫu cho 2 MCU ESP32-S3 đảm bảo toàn vẹn dữ liệu qua CRC16/CRC32.

