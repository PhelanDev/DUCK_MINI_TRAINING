# Đặc Tả Kỹ Thuật (Design Specification): Khắc Phục Toàn Diện 10 Điểm Nghẽn Kỹ Thuật MicroDuck 8-DOF

**Ngày tạo:** 2026-10-28T22:55:00+07:00  
**Tác giả:** Antigravity Autonomous Meta-Orchestrator & Pair Programmer  
**Mục tiêu:** Giải quyết triệt để 10 điểm nghẽn kỹ thuật từ lõi mô phỏng MuJoCo, thuật toán học tăng cường PPO, cho đến mã nguồn firmware nhúng và script triển khai phần cứng thực tế.

---

## 1. Bối Cảnh & Mục Tiêu Kỹ Thuật
Dự án MicroDuck 8-DOF đã xây dựng thành công 2 nhánh phần cứng độc lập (Raspberry Pi 5 và Dual ESP32-S3) với cụm cảm biến nhận thức (5-ray LiDAR, Camera, I2S Mic). Tuy nhiên, để huấn luyện thành công mô hình Sim-to-Real có khả năng bước đi tự chủ, tránh vật cản và nạp xuống vi điều khiển hoạt động chính xác, 10 điểm nghẽn sau cần được giải quyết dứt điểm theo 5 giai đoạn:

1. **Vật cản thực tế trong MuJoCo:** Mô hình XML thiếu hình học va chạm xung quanh khiến 5 tia LiDAR luôn trả về `-1.0`.
2. **Hàm thưởng tiếp xúc chân (`foot_contact`):** Logic cũ cộng thưởng tĩnh cho 2 chân chạm sàn khiến robot bị "chôn chân", không dám nhấc chân bước đi.
3. **Giới hạn độ dài tập mô phỏng (Episode Truncation):** Thiếu `max_episode_steps = 500` ($10\text{ s}$ ở $50\text{ Hz}$) khiến môi trường không tự ngắt khi robot đứng yên hoặc hết thời gian.
4. **Ngẫu nhiên hóa miền (Domain Randomization):** Thiếu nhiễu góc khớp khởi tạo ($\pm 0.03\text{ rad}$) và ngẫu nhiên hóa lệnh vận tốc ($v_x, v_y, \omega_z$).
5. **Lệch pha Auto-Reset trong mô phỏng song song:** Khi môi trường ngã hoặc timeout, `next_obs` không được cập nhật ngay bằng trạng thái sau khi reset, làm mạng RL học nhầm trạng thái ngã làm khởi đầu.
6. **Thiếu thuật toán PPO huấn luyện chuẩn:** Cần xây dựng `scripts/train_ppo.py` sử dụng mạng Actor-Critic (PyTorch) với đầu vào 42 chiều, 2 lớp ẩn 256 nơ-ron ELU, đầu ra 8 chiều.
7. **Thiếu script xuất chính sách (`scripts/export_policy.py`):** Xuất song song file ONNX Float32 cho Pi 5 và header C/C++ lượng tử hóa INT8 (`esp32_quant_config.h`, `esp32_policy_weights.h`) cho ESP32-S3.
8. **Lệch đơn vị `last_actions` trên Pi 5:** Truyền nhầm góc radian `last_targets` vào chiều $25..32$ của vector quan sát thay vì hành động chuẩn hóa $[-1.0, 1.0]$.
9. **Xung đột chân I2C & sai dấu trục IMU trên Dual ESP32-S3:**
   - MCU 1: Cụm LiDAR dùng chung bus SCCB của Camera (GPIO 4, 5) $\rightarrow$ Đổi sang GPIO 19, 20.
   - MCU 2: Dấu trục gia tốc IMU bị ngược hệ trọng trường $[0, 0, -1]$ của MuJoCo; thiếu bộ lọc thông thấp trọng trường chiếu và chưa tích hợp suy luận INT8 với `OBS_CHANNEL_SCALES`.
10. **Trình trực quan hóa 3D (`scripts/interactive_viewer.py`) và bộ test suite toàn dự án:** Hoàn thiện vòng lặp passive viewer, chế độ CLI tương tác và đảm bảo 100% test pass.

---

## 2. Thiết Kế Chi Tiết Từng Giai Đoạn

### Giai Đoạn 1: Chuẩn Hóa Mô Hình Vật Lý XML & Hàm Thưởng
- **1.1. Bổ sung vật cản LiDAR vào XML:**
  Bổ sung 4 cột hình trụ vào `<worldbody>` trong cả 3 tệp MJCF:
  - `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`
  - `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`
  - `models/mjcf/microduck_8dof.xml`
  ```xml
  <geom name="obstacle_front" type="cylinder" size="0.08 0.25" pos="0.85 0.0 0.25" rgba="0.8 0.2 0.2 1.0" contype="1" conaffinity="1"/>
  <geom name="obstacle_left" type="cylinder" size="0.08 0.25" pos="0.65 0.35 0.25" rgba="0.8 0.2 0.2 1.0" contype="1" conaffinity="1"/>
  <geom name="obstacle_right" type="cylinder" size="0.08 0.25" pos="0.65 -0.35 0.25" rgba="0.8 0.2 0.2 1.0" contype="1" conaffinity="1"/>
  <geom name="obstacle_back" type="cylinder" size="0.08 0.25" pos="-0.75 0.0 0.25" rgba="0.8 0.2 0.2 1.0" contype="1" conaffinity="1"/>
  ```
- **1.2. Sửa hàm thưởng `foot_contact`:**
  Áp dụng trong cả 3 tệp `env_config.py`:
  - Khi có lệnh di chuyển ($\|v_{xy}^{\text{cmd}}\| + |\omega_z^{\text{cmd}}| > 0.05$): Thưởng khi chỉ có đúng 1 chân tiếp đất (`num_contacts == 1`) để khuyến khích dáng đi nhấc chân luân phiên.
  - Khi lệnh đứng yên: Thưởng khi cả 2 chân tiếp đất (`num_contacts == 2 * 0.5`).

---

### Giai Đoạn 2: Nâng Cấp Bộ Mô Phỏng Song Song (`run_parallel_sim.py`)
Áp dụng đồng bộ cho cả 3 tệp:
- `pipelines/rpi5_stack/training/run_parallel_sim.py`
- `pipelines/dual_esp32s3_stack/training/run_parallel_sim.py`
- `training/run_parallel_sim.py`
- **Khởi tạo:**
  `self.max_episode_steps = 500` ($10\text{ s}$ ở $dt = 0.02\text{ s}$), `self.episode_steps = np.zeros(num_envs, dtype=np.int32)`, `self.rng = np.random.default_rng(seed=42)`.
- **Hàm `reset_env(env_idx)`:**
  - Nhiễu góc khớp xuất phát: `d.qpos[7:15] = self.rng.uniform(-0.03, 0.03, size=8)`, kẹp gối $\ge 0$.
  - Lệnh vận tốc ngẫu nhiên: $v_x \in [0.0, 0.25]\text{ m/s}$, $v_y \in [-0.08, 0.08]\text{ m/s}$, $\omega_z \in [-0.50, 0.50]\text{ rad/s}$.
  - Đặt lại `self.episode_steps[env_idx] = 0`.
- **Hàm `step(actions)`:**
  - `self.episode_steps += 1`
  - `timeouts = self.episode_steps >= self.max_episode_steps`
  - `dones = (root_z < 0.08) | (grav_z > -0.3) | diverged | timeouts`
  - Khi `np.any(dones)`: Cập nhật ngay `next_obs[i] = self.reset_env(i)` để đảm bảo tính nhất quán của trạng thái bước tiếp theo.

---

### Giai Đoạn 3: Xây Dựng Thuật Toán Huấn Luyện PPO & Xuất Mô Hình
- **Tệp mới 1: `scripts/train_ppo.py`**
  - Mạng `ActorCritic`:
    * Shared / Separate MLP: Input 42 dims $\rightarrow$ Linear(42, 256) $\rightarrow$ ELU $\rightarrow$ Linear(256, 256) $\rightarrow$ ELU.
    * Actor Head: Linear(256, 8) với Gaussian log_std có thể học.
    * Critic Head: Linear(256, 1) ước lượng Value $V(s)$.
  - Thuật toán PPO (Clipping objective $\epsilon = 0.2$, GAE $\lambda = 0.95$, $\gamma = 0.99$).
  - Thu thập Rollout đa luồng từ `ParallelSimRunner` (mặc định 64 envs, horizon 24 bước/iter).
  - Hỗ trợ lưu checkpoint mô hình `.pt`.
- **Tệp mới 2: `scripts/export_policy.py`**
  - Nạp checkpoint `.pt` từ ActorCritic.
  - Xuất mô hình ONNX Float32: `models/microduck_rpi5_policy.onnx` với input name `obs` (shape `[1, 42]`), output name `actions` (shape `[1, 8]`), opset 13.
  - Xuất header cấu hình và trọng số lượng tử hóa INT8:
    * `esp32_quant_config.h` (sinh qua `generate_esp32_quant_header`).
    * `esp32_policy_weights.h` (ma trận trọng số int8_t kèm bias int32_t phục vụ suy luận nhanh trên ESP32-S3).

---

### Giai Đoạn 4: Đồng Bộ Hóa Mã Nguồn Triển Khai Phần Cứng
- **4.1. Raspberry Pi 5 (`pipelines/rpi5_stack/deployment/infer_rpi5.py`):**
  - Khởi tạo `self.last_raw_actions = np.zeros(8, dtype=np.float32)`.
  - Trong `step()`: Lưu lại `self.last_raw_actions = np.clip(raw_actions, -1.0, 1.0)`.
  - Trong `run_loop()`: Truyền `last_actions=self.last_raw_actions` vào `pack_rpi5_observation` thay vì truyền góc radian `self.last_targets`.
- **4.2. MCU 1 Vision/LiDAR (`mcu1_main.ino` & spec):**
  - Tách chân I2C cho cụm 5 LiDAR ToF: Đổi `PIN_I2C_SDA` sang GPIO 19 và `PIN_I2C_SCL` sang GPIO 20 để giải phóng GPIO 4, 5 cho SCCB Camera OV2640.
  - Cập nhật tài liệu phần cứng [`dual_esp32s3_hardware_spec.md`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md).
- **4.3. MCU 2 Locomotion Reflex (`mcu2_main.ino`):**
  - Đảo dấu vector gia tốc IMU MPU6050: $g_x = -a_x/\|a\|$, $g_y = -a_y/\|a\|$, $g_z = -a_z/\|a\|$ kết hợp bộ lọc thông thấp IIR ($\alpha = 0.80$) để khi đứng yên thẳng đứng, `proj_grav` trả về chính xác $[0.0, 0.0, -1.0]$.
  - Tách biến toàn cục `float g_last_raw_actions[8]` lưu hành động $[-1.0, 1.0]$ cho chiều $25..32$ của vector quan sát.
  - Lượng tử hóa vector quan sát 42 chiều sang `int8_t obs_int8[42]` sử dụng `OBS_CHANNEL_SCALES[i]` từ header lượng tử hóa.

---

### Giai Đoạn 5: Khôi Phục Công Cụ & Kiểm Thử Tự Động
- **5.1. Khôi phục `scripts/interactive_viewer.py`:**
  - Hoàn thiện đầy đủ các chế độ: Passive GUI viewer với `mujoco.viewer.launch_passive`, tự động fallback sang Headless Mode khi không có màn hình hiển thị, chế độ CLI tương tác đặt góc và tư thế mẫu, cùng hàm `main()`.
- **5.2. Kiểm thử tự động:**
  - Bổ sung bài test cho `foot_contact` mới, bài test cho `ParallelSimRunner` Auto-Reset và Episode Truncation.
  - Bổ sung bài test cho PPO network & export policy trong `tests/`.
  - Đảm bảo toàn bộ test suite đạt 100% PASS trên `pytest -v`.
- **5.3. Ghi chép nhật ký:**
  - Cập nhật đầy đủ vào [`his.md`](file:///d:/duckV1/his.md).

