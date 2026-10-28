# Dual Hardware Pipelines (RPi 5 vs Dual ESP32-S3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai cấu trúc dự án phân tách 2 nhánh phần cứng độc lập (`pipelines/rpi5_stack/` và `pipelines/dual_esp32s3_stack/`) với cụm cảm biến nhận thức hoàn chỉnh (LiDAR 5 tia, Camera, Microphone) và bộ công cụ huấn luyện song song, suy luận nhúng tương ứng.

**Architecture:** Tách biệt hoàn toàn mô hình vật lý, cấu hình huấn luyện quan sát 42 chiều (37 trạng thái động học + 5 khoảng cách LiDAR), và mã nguồn triển khai cho 2 bài toán phần cứng đồ án: Raspberry Pi 5 4GB (tải trọng ~860g, Float32 ONNX) và Dual ESP32-S3 (tải trọng ~780g, INT8, giao thức UART nhị phân giữa 2 MCU).

**Tech Stack:** Python 3.10+, MuJoCo (MJCF XML), URDF, NumPy, C/C++ (Arduino/ESP-IDF header & sketch), PyTest.

**Spec:** docs/superpowers/specs/2026-10-28-microduck-dual-hardware-pipelines-design.md

## Global Constraints

- Cụm cảm biến đầu vịt (Perception Suite): 5 tia `<rangefinder>` quét ngang $[-30^\circ, -15^\circ, 0^\circ, +15^\circ, +30^\circ]$, tầm đo $0.05\text{ m} - 2.0\text{ m}$, camera quang học $60^\circ$ FOV với geom ống kính, microphone I2S MEMS site.
- Nhánh RPi 5 (`pipelines/rpi5_stack/`): Tổng khối lượng $\approx 860\text{ g}$, CoM cao hơn $10\text{ mm}$, môi trường huấn luyện tối ưu cho mô hình Float32 ONNX Runtime trên Linux.
- Nhánh Dual ESP32-S3 (`pipelines/dual_esp32s3_stack/`): Tổng khối lượng $\approx 780\text{ g}$, CoM thấp sát đáy thân, cấu hình huấn luyện sẵn sàng cho lượng tử hóa INT8 (tập lệnh PIE), mã nguồn phân tầng 2 MCU giao tiếp UART Packet có CRC.
- Vector quan sát mở rộng từ 37 chiều lên 42 chiều (`OBS_DIM = 42`).
- 8 servo MG996R giữ nguyên giới hạn: torque $\tau_{\max} = 1.0\text{ N}\cdot\text{m}$, damping $0.15\text{ N}\cdot\text{s/rad}$, rate limiting $6.0\text{ rad/s} \times \Delta t$.

## Review Focus

1. Kiểm tra 5 tia LiDAR trong MJCF: Đảm bảo các site hướng đúng góc tỏa quạt ngang $[-30^\circ, -15^\circ, 0^\circ, +15^\circ, +30^\circ]$ chiếu về phía trước ($+X$).
2. Kiểm tra khối lượng và quán tính: Nhánh RPi 5 phải đạt $860\text{ g} \pm 5\%$ và nhánh Dual ESP32-S3 phải đạt $780\text{ g} \pm 5\%$, tất cả tensor quán tính thỏa mãn bất đẳng thức tam giác.
3. Kiểm tra tính toàn vẹn của vector quan sát 42 chiều: 37 chiều động học gốc + 5 chiều khoảng cách tia LiDAR, đảm bảo không bị lệch thứ tự index khi đọc cảm biến trong MuJoCo.
4. Kiểm tra giao thức UART Packet giữa 2 chip ESP32-S3: Struct nhị phân chứa header `0xAA 0x55`, ID gói, payload vận tốc + khoảng cách, và checksum CRC16.
5. Kiểm tra script chạy song song `run_parallel_sim.py` của cả 2 nhánh hoạt động độc lập và đạt tốc độ $>2,000\text{ FPS}$.

---

### Task 1: Khởi Tạo Thư Mục & Mô Hình Nhánh Raspberry Pi 5 (`pipelines/rpi5_stack/models/`)

**Files:**
- Create: `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`
- Create: `pipelines/rpi5_stack/models/urdf/microduck_rpi5_8dof.urdf`
- Test: `tests/test_rpi5_model.py`

**Interfaces:**
- Produces: Mô hình MJCF và URDF với tải trọng 860g, 5 tia `<rangefinder>` LiDAR, camera lens geom, mic site và 8 servo MG996R.

- [ ] **Step 1: Write failing test in `tests/test_rpi5_model.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `pipelines/rpi5_stack/models/`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 2: Mô Hình Nhánh Dual ESP32-S3 (`pipelines/dual_esp32s3_stack/models/`)

**Files:**
- Create: `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`
- Create: `pipelines/dual_esp32s3_stack/models/urdf/microduck_esp32_8dof.urdf`
- Test: `tests/test_esp32_model.py`

**Interfaces:**
- Produces: Mô hình MJCF và URDF với tải trọng 780g, 5 tia LiDAR, camera lens, mic site, trọng tâm thấp và 8 servo MG996R.

- [ ] **Step 1: Write failing test in `tests/test_esp32_model.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `pipelines/dual_esp32s3_stack/models/`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 3: Cấu Hình Môi Trường Huấn Luyện 42 Chiều (Perception RL) Cho Cả 2 Nhánh

**Files:**
- Create: `pipelines/rpi5_stack/training/env_config.py`
- Create: `pipelines/rpi5_stack/training/run_parallel_sim.py`
- Create: `pipelines/dual_esp32s3_stack/training/env_config.py`
- Create: `pipelines/dual_esp32s3_stack/training/run_parallel_sim.py`
- Test: `tests/test_pipeline_training.py`

**Interfaces:**
- Produces: `env_config.py` xử lý quan sát 42 chiều (kèm trích xuất 5 tia LiDAR và hàm thưởng né vật cản `obstacle_avoidance`), `run_parallel_sim.py` chạy benchmark song song đa luồng cho từng nhánh.

- [ ] **Step 1: Write failing test in `tests/test_pipeline_training.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement training modules in both stacks**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 4: Triển Khai Phần Mềm & Firmware Nhúng Cho 2 Nhánh

**Files:**
- Create: `pipelines/rpi5_stack/deployment/infer_rpi5.py`
- Create: `pipelines/rpi5_stack/docs/rpi5_hardware_spec.md`
- Create: `pipelines/dual_esp32s3_stack/deployment/uart_packet_protocol.h`
- Create: `pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino`
- Create: `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`
- Create: `pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md`
- Test: `tests/test_deployment_artifacts.py`

**Interfaces:**
- Produces: Runtime Python trên Pi 5 nạp ONNX điều khiển PCA9685/MG996R; Giao thức nhị phân UART Packet kèm sketch Arduino C++ cho 2 MCU ESP32-S3; Sơ đồ nối dây phần cứng chi tiết.

- [ ] **Step 1: Write failing test in `tests/test_deployment_artifacts.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement deployment code, firmware, and hardware specs**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 5: Cập Nhật README.md & Toàn Bộ Hệ Thống Kiểm Thử

**Files:**
- Modify: `README.md`
- Test: Full project test suite (`pytest -v`)

**Interfaces:**
- Produces: Tài liệu tổng hợp so sánh 2 nhánh, hướng dẫn huấn luyện và nạp xuống phần cứng thật cho đồ án tốt nghiệp; toàn bộ test suites đạt 100% PASS.

- [ ] **Step 1: Update README.md with dual-pipeline guidance and sensor suite**
- [ ] **Step 2: Run full pytest suite across all tests**
- [ ] **Step 3: Commit and verify clean git status**
