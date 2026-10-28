# Kế Hoạch Triển Khai (Implementation Plan): Khắc Phục 10 Điểm Nghẽn Kỹ Thuật MicroDuck 8-DOF

**Ngày tạo:** 2026-10-28T22:56:00+07:00  
**Tài liệu đặc tả liên quan:** `docs/superpowers/specs/2026-10-28-microduck-10-technical-bottlenecks-spec.md`  
**Phương pháp thực thi:** Subagent-Driven Development (SDD) & Ponytail Protocol (Lazy Senior Dev)

---

## Danh Sách Các Task Thực Thi (5 Giai Đoạn)

### Task 1: Chuẩn Hóa Vật Lý & Hàm Thưởng (Giai đoạn 1)
- **Tệp cần sửa:**
  - `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`
  - `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`
  - `models/mjcf/microduck_8dof.xml`
  - `pipelines/rpi5_stack/training/env_config.py`
  - `pipelines/dual_esp32s3_stack/training/env_config.py`
  - `training/env_config.py`
- **Mục tiêu:**
  - Thêm 4 cột hình trụ chướng ngại vật (`obstacle_front`, `obstacle_left`, `obstacle_right`, `obstacle_back`) vào `<worldbody>` của cả 3 file XML để kích hoạt 5 tia LiDAR từ $0.05\text{ m} \to 2.0\text{ m}$.
  - Sửa hàm thưởng `foot_contact` trong cả 3 file `env_config.py`: khi $\|v_{xy}^{\text{cmd}}\| + |\omega_z^{\text{cmd}}| > 0.05$ thì thưởng khi $1$ chân chạm đất (`num_contacts == 1`); khi lệnh đứng yên thì thưởng khi $2$ chân chạm đất.
- **Kiểm thử:**
  - Chạy `pytest tests/test_env_config.py tests/test_mjcf_model.py tests/test_rpi5_model.py tests/test_esp32_model.py -v`.
- **Commit:** `feat(sim): add lidar obstacles to mjcf models and update foot contact gait reward`

---

### Task 2: Nâng Cấp Bộ Mô Phỏng Song Song (Giai đoạn 2)
- **Tệp cần sửa:**
  - `pipelines/rpi5_stack/training/run_parallel_sim.py`
  - `pipelines/dual_esp32s3_stack/training/run_parallel_sim.py`
  - `training/run_parallel_sim.py`
- **Mục tiêu:**
  - Bổ sung `self.max_episode_steps = 500`, `self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)`, `self.rng = np.random.default_rng(seed=42)`.
  - Nâng cấp `reset_env`: Thêm nhiễu góc khớp ban đầu $[-0.03, 0.03]$ (kẹp gối $\ge 0$), ngẫu nhiên hóa lệnh vận tốc ($v_x \in [0.0, 0.25]$, $v_y \in [-0.08, 0.08]$, $\omega_z \in [-0.50, 0.50]$).
  - Nâng cấp `step`: Đếm số bước, xác định timeout khi `episode_steps >= max_episode_steps`, gộp `timeouts` vào `dones`, và cập nhật ngay `next_obs[i] = self.reset_env(i)` cho các môi trường vừa reset.
- **Kiểm thử:**
  - Chạy `pytest tests/test_parallel_runner.py tests/test_pipeline_training.py -v`.
  - Đo tốc độ FPS trên 64 môi trường: `python pipelines/rpi5_stack/training/run_parallel_sim.py --num_envs 64 --steps 50`.
- **Commit:** `feat(sim): implement episode truncation, domain randomization and aligned auto-reset`

---

### Task 3: Xây Dựng Pipeline Huấn Luyện PPO & Xuất Mô Hình (Giai đoạn 3)
- **Tệp cần tạo mới:**
  - `scripts/train_ppo.py`
  - `scripts/export_policy.py`
- **Mục tiêu:**
  - `train_ppo.py`: Kiến trúc PyTorch `ActorCritic` (MLP 42 dims $\rightarrow$ 256 ELU $\rightarrow$ 256 ELU $\rightarrow$ Actor 8 dims / Critic 1 dim). Thu thập rollout từ `ParallelSimRunner` (mặc định 64 envs) và tối ưu hóa PPO loss với GAE. Hỗ trợ cờ `--dry-run` để chạy nhanh 1 iteration trong unit tests và xuất checkpoint `.pt`.
  - `export_policy.py`: Nạp checkpoint PyTorch `.pt`, xuất file `models/microduck_rpi5_policy.onnx` (Float32, opset 13) và xuất header C/C++ `pipelines/dual_esp32s3_stack/deployment/esp32_quant_config.h` cùng `esp32_policy_weights.h` (INT8 weights cho ESP32-S3).
- **Kiểm thử:**
  - Tạo mới `tests/test_ppo_training.py` kiểm tra kiến trúc ActorCritic, PPO step, xuất ONNX và sinh header INT8.
- **Commit:** `feat(rl): implement ppo training pipeline and policy export for rpi5 onnx and esp32 int8`

---

### Task 4: Đồng Bộ Hóa Mã Nguồn Triển Khai Phần Cứng (Giai đoạn 4)
- **Tệp cần sửa:**
  - `pipelines/rpi5_stack/deployment/infer_rpi5.py`
  - `pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino`
  - `pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md`
  - `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`
- **Mục tiêu:**
  - 4.1 Pi 5: Tách `self.last_raw_actions` (chuẩn hóa $[-1.0, 1.0]$) khỏi `self.last_targets` (radian) trong `infer_rpi5.py` và truyền `last_actions=self.last_raw_actions` vào `pack_rpi5_observation`.
  - 4.2 MCU 1: Đổi `PIN_I2C_SDA` sang GPIO 19 và `PIN_I2C_SCL` sang GPIO 20 để tránh xung đột với bus SCCB camera OV2640. Cập nhật tài liệu phần cứng tương ứng.
  - 4.3 MCU 2: Đảo dấu gia tốc IMU $g_x = -a_x/\|a\|$, $g_y = -a_y/\|a\|$, $g_z = -a_z/\|a\|$ kết hợp bộ lọc thông thấp IIR ($\alpha = 0.80$); tách `g_last_raw_actions` $[-1.0, 1.0]$; tích hợp bước lượng tử hóa INT8 với `OBS_CHANNEL_SCALES`.
- **Kiểm thử:**
  - Chạy `pytest tests/test_deployment_artifacts.py -v`.
- **Commit:** `fix(hardware): sync last_raw_actions, separate lidar i2c pins and correct imu gravity sign`

---

### Task 5: Khôi Phục Công Cụ Trực Quan & Kiểm Thử Toàn Diện (Giai đoạn 5)
- **Tệp cần sửa:**
  - `scripts/interactive_viewer.py`
  - `tests/test_interactive_viewer.py`
  - `tests/test_pipeline_training.py`
  - `README.md`
  - `his.md`
- **Mục tiêu:**
  - Rà soát và hoàn thiện `scripts/interactive_viewer.py` (vòng lặp 3D passive viewer, headless fallback, CLI parser).
  - Cập nhật các bài test kiểm thử tự động, chạy `pytest -v` đảm bảo 100% test pass.
  - Cập nhật nhật ký phát hiện và xử lý vào `his.md`.
- **Commit:** `docs: update interactive viewer, full test verification and history log`

