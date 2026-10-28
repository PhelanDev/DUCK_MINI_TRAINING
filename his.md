# Nhật Ký Lịch Sử Khắc Phục Lỗ Hổng & Tối Ưu Hệ Thống (History & Audit Log)

**Văn bản này ghi nhận toàn bộ các phát hiện kỹ thuật, nguyên nhân gốc rễ (Root Cause), các điểm mã nguồn đã sửa và kết quả kiểm thử đối soát cho dự án MicroDuck 8-DOF.**

---

## 📌 Phiên Làm Việc: 2026-10-28 22:35:00+07:00

### 1. Khắc Phục Lỗ Hổng "Reward Hacking" Ở Hàm Thưởng `obstacle_avoidance`
- **Tệp ảnh hưởng:**
  - `pipelines/rpi5_stack/training/env_config.py` (dòng 275–280)
  - `pipelines/dual_esp32s3_stack/training/env_config.py` (dòng 275–280)
  - `training/env_config.py` (dòng 273–279)
  - `README.md` (mục 5.3)
  - `tests/test_pipeline_training.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Nhánh `else` trong biểu thức `np.where(min_dist < OBSTACLE_THRESHOLD, min_dist - OBSTACLE_THRESHOLD, min_dist)` trả về giá trị khoảng cách nguyên bản `min_dist` khi robot ở xa vật cản ($d \ge 0.35\text{ m}$).
  - Với khoảng cách quét tối đa của LiDAR là $2.0\text{ m}$ và trọng số thưởng dương $0.2$, robot nhận được tới $0.2 \times 2.0 = +0.4$ điểm thưởng ở mỗi bước mô phỏng chỉ bằng việc đứng yên hoặc quay mặt nhìn vào khoảng trống (Reward Hacking). Robot sợ di chuyển về phía trước vì tiếp cận bất kỳ mục tiêu nào cũng làm giảm `min_dist` và bị mất điểm.
- **Biện pháp xử lý:**
  - Sửa nhánh `else` trả về đúng $0.0$:
    ```python
    r_obstacle = np.where(
        min_dist < _DEFAULT_CFG.OBSTACLE_THRESHOLD,
        min_dist - _DEFAULT_CFG.OBSTACLE_THRESHOLD,
        0.0,
    ).astype(np.float32)
    ```
  - Cập nhật công thức và giải thích trong `README.md` mục 5.3: Khi $d_{\min} \ge 0.35\text{ m}$, hàm thưởng triệt tiêu về $0.0$, buộc robot chỉ tập trung bám vận tốc lệnh $v_{xy}^{\text{cmd}}$. Chỉ khi $d_{\min} < 0.35\text{ m}$ robot mới chịu phạt âm tuyến tính để né vật cản.
  - Sửa lại assertion trong `tests/test_pipeline_training.py` xác nhận `r_clear["obstacle_avoidance"] == 0.0`.

---

### 2. Chuẩn Hóa Lượng Tử Hóa INT8 Trên ESP32-S3 Chống Méo Tín Hiệu (Clipping Distortion)
- **Tệp ảnh hưởng:**
  - `pipelines/dual_esp32s3_stack/training/env_config.py` (dòng 325–460)
  - `tests/test_pipeline_training.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Vector quan sát 42 chiều trước đó bị nhân đồng loạt với `scale = 127.0` rồi kẹp (clip) vào `[-128, 127]`.
  - Nhiều đại lượng vật lý thực tế vượt quá $1.0$ rất xa:
    * Vận tốc góc thân `base_ang_vel` và vận tốc góc khớp `joint_vel` lên tới $6.0\text{ rad/s}$ ($6.0 \times 127 = 762 \rightarrow$ bị bão hòa thành $127$).
    * 5 tia LiDAR từ $0.05\text{ m} \to 2.0\text{ m}$ ($2.0 \times 127 = 254 \rightarrow$ mọi cự ly từ $1.0\text{ m} \to 2.0\text{ m}$ bị cắt ngọn).
    * Giới hạn quay khớp `head_yaw` tới $1.57\text{ rad}$ và `knee_pitch` tới $1.48\text{ rad}$ cũng bị cắt ngọn.
- **Biện pháp xử lý:**
  - Thiết lập vector giới hạn vật lý `DEFAULT_OBS_BOUNDS` (42 chiều) tương ứng với từng thành phần.
  - Hàm `quantize_obs_int8` chuẩn hóa từng thành phần về `[-1.0, 1.0]` trước khi nhân tỷ lệ:
    $$\text{obs}_{\text{norm}}[i] = \text{clip}\left(\frac{\text{obs}[i]}{\text{bounds}[i]}, -1.0, 1.0\right)$$
    $$\text{obs}_{\text{int8}}[i] = \text{round}(\text{obs}_{\text{norm}}[i] \times 127.0)$$
  - Bổ sung hàm giải lượng tử hóa `dequantize_obs_int8`.
  - Cập nhật hàm `generate_esp32_quant_header` xuất mảng tỷ lệ kênh `OBS_CHANNEL_SCALES[42]` và `OBS_MAX_BOUNDS[42]` để nạp trực tiếp vào firmware C/C++ của MCU 2.

---

### 3. Đồng Bộ Hóa Cảm Biến Chạm Sàn (Foot Contact) Giữa Mô Phỏng & Phần Cứng Thật
- **Tệp ảnh hưởng:**
  - `pipelines/rpi5_stack/training/run_parallel_sim.py`
  - `pipelines/dual_esp32s3_stack/training/run_parallel_sim.py`
  - `training/run_parallel_sim.py`
  - `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`
  - `pipelines/rpi5_stack/deployment/infer_rpi5.py`
  - `tests/test_parallel_runner.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Lúc train trong MuJoCo, cảm biến chạm sàn `sensordata[10:12]` trả về lực phản lực liên tục (Newton, $\sim 3.8\text{ N}$) và 2 chiều cuối luôn gán cứng $0.0$: `[F_left, F_right, 0.0, 0.0]`.
  - Trong khi mã nguồn triển khai thực tế trên MCU 2 và Pi 5 lại gán cả 4 phần tử bằng $1.0$: `[1.0, 1.0, 1.0, 1.0]`. Khi nạp xuống robot thật, các trọng số neural network kết nối với chiều 35:37 (vốn chỉ học số 0) sẽ bị kích hoạt bất thường, làm méo dáng đi của robot.
- **Biện pháp xử lý:**
  - Nhị phân hóa lực tiếp xúc trong mô phỏng: Chuyển đổi lực $> 0.1\text{ N}$ thành giá trị logic `1.0` (ngược lại `0.0`):
    ```python
    left_touch = 1.0 if float(d.sensordata[10]) > 0.1 else 0.0
    right_touch = 1.0 if float(d.sensordata[11]) > 0.1 else 0.0
    foot_contact = np.array([left_touch, right_touch, 0.0, 0.0], dtype=np.float32)
    ```
  - Trong `mcu2_main.ino` và `infer_rpi5.py`, gán chuẩn xác định:
    `foot_contact = [1.0, 1.0, 0.0, 0.0]` (2 chân tiếp đất ở trạng thái cơ sở, 2 chiều đệm đặt 0).

---

### 4. Bổ Sung Tên Cờ Dòng Lệnh (CLI Flag Aliases) Cho Script Triển Khai
- **Tệp ảnh hưởng:**
  - `pipelines/rpi5_stack/deployment/infer_rpi5.py`
  - `tests/test_deployment_artifacts.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - `README.md` hướng dẫn chạy lệnh triển khai với `--model` và `--rate`, trong khi mã nguồn gốc chỉ chấp nhận `--model_path` và `--freq`, gây lỗi `unrecognized arguments`.
- **Biện pháp xử lý:**
  - Bổ sung alias tương thích kép vào `argparse`:
    * `--model` là alias cho `--model_path`
    * `--rate` là alias cho `--freq`
    * `--mock-pca` là alias cho `--mock`
  - Thêm bài kiểm thử tự động xác thực tất cả các biến thể cờ CLI.

---

### 5. Đồng Bộ Toàn Diện File Gốc `training/run_parallel_sim.py` & `training/env_config.py`
- **Tệp ảnh hưởng:**
  - `training/run_parallel_sim.py`
  - `training/env_config.py`
- **Nội dung đồng bộ:**
  - Tích hợp phép chiếu vận tốc thân qua phép quay quaternion nghịch đảo `quat_rotate_inverse`.
  - Vector hóa toàn bộ quá trình tính toán tensor động học và hàm thưởng trên CPU.
  - Đồng bộ logic tiếp xúc bàn chân và dọn dẹp các lệnh gán đè `last_actions` dư thừa.

---

### 6. Kết Quả Kiểm Thử Toàn Diện (Full Test Suite Verification)
- **Lệnh chạy:** `pytest -v`
- **Kết quả:** **64 / 64 tests PASSED (100%)**
- **Thời gian chạy:** 8.36 giây trên môi trường Python 3.14.
- **Trạng thái Git:** Toàn bộ thay đổi đã được kiểm tra tính tương thích, mã sạch, không còn lỗi tiềm ẩn.

---

## 📌 Phiên Làm Việc: 2026-10-28 23:55:00+07:00 - Chiến Dịch Khắc Phục Toàn Diện 10 Điểm Nghẽn Kỹ Thuật (10 Technical Bottlenecks Campaign)

Chiến dịch khắc phục toàn diện 10 điểm nghẽn kỹ thuật Sim-to-Real trên robot MicroDuck 8-DOF, chia thành 5 giai đoạn triển khai tuần tự theo kế hoạch chuẩn hóa SDD:

### Giai Đoạn 1: Chuẩn Hóa Vật Lý & Hàm Thưởng (Task 1)
- **Commit:** `683c681` (`feat(sim): add lidar obstacles to mjcf models and update foot contact gait reward`)
- **Điểm nghẽn 1: Thiếu vật cản trong mô hình MuJoCo MJCF**
  - *Tệp ảnh hưởng:* `models/mjcf/microduck_8dof.xml`, `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`, `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`.
  - *Nguyên nhân:* Cụm cảm biến ToF LiDAR 5 tia quét vào không gian vô tận ($2.0\text{ m}$), mô hình không có mục tiêu phản xạ quang học và không có vật cản vật lý để học né tránh.
  - *Khắc phục:* Bổ sung 4 cột chướng ngại vật dạng hình trụ `obstacle_front` ($[+0.8, 0, 0.25]$), `obstacle_left` ($[+0.5, +0.6, 0.25]$), `obstacle_right` ($[+0.5, -0.6, 0.25]$), `obstacle_back` ($[-0.6, 0, 0.25]$) với `size="0.08 0.25"` và `contype="1" conaffinity="1"` vào `<worldbody>`.
- **Điểm nghẽn 2: Hàm thưởng tiếp xúc chân sơ sài**
  - *Tệp ảnh hưởng:* `training/env_config.py`, `pipelines/*/training/env_config.py`, `tests/test_env_config.py`.
  - *Nguyên nhân:* Tính tổng đơn thuần lực tiếp xúc khiến robot có xu hướng kéo lê 2 chân hoặc nhảy bổng hai chân để ăn điểm.
  - *Khắc phục:* Thiết kế lại hàm thưởng dáng đi bipedal: khi có lệnh di chuyển ($v_{\text{cmd}} > 0.05\text{ m/s}$), chỉ thưởng $+1.0$ khi chính xác 1 chân tiếp đất (`num_contacts == 1`); khi robot dừng ($v_{\text{cmd}} \le 0.05\text{ m/s}$), thưởng $+0.5$ khi cả 2 chân tiếp đất (`num_contacts == 2`).

### Giai Đoạn 2: Nâng Cấp Bộ Mô Phỏng Song Song (Task 2)
- **Commit:** `b67b735` (`feat(sim): implement episode truncation, domain randomization and aligned auto-reset`)
- **Điểm nghẽn 3: Episode kéo dài vô hạn không có giới hạn ngắt (Truncation)**
  - *Tệp ảnh hưởng:* `training/run_parallel_sim.py`, `pipelines/*/training/run_parallel_sim.py`.
  - *Nguyên nhân:* Episode không bị giới hạn thời gian làm phân phối dữ liệu huấn luyện bị lệch trọng số và robot có thể đứng yên tích lũy điểm thưởng vô hạn.
  - *Khắc phục:* Thêm `max_episode_steps = 500` ($10\text{ s}$ ở $50\text{ Hz}$), đếm `episode_steps += 1`, kích hoạt cờ `timeouts` kết hợp vào `dones` và trả về `infos["timeout"] = True`.
- **Điểm nghẽn 4: Thiếu ngẫu nhiên hóa miền không gian trạng thái (Domain Randomization)**
  - *Tệp ảnh hưởng:* `training/run_parallel_sim.py`, `pipelines/*/training/run_parallel_sim.py`.
  - *Nguyên nhân:* Robot luôn khởi tạo ở tư thế tĩnh lý tưởng và vận tốc mặc định, dẫn đến policy thiếu tính thích nghi khi gặp nhiễu vật lý thực tế.
  - *Khắc phục:* Khi `reset_env`, áp dụng nhiễu đồng đều cho góc khớp $\Delta q \in [-0.03, +0.03]\text{ rad}$ (đảm bảo góc gập gối $\ge 0.0$) và lấy mẫu ngẫu nhiên lệnh vận tốc: $v_x \in [0.0, 0.25]\text{ m/s}$, $v_y \in [-0.08, 0.08]\text{ m/s}$, $\omega_z \in [-0.5, 0.5]\text{ rad/s}$.
- **Điểm nghẽn 5: Quan sát lệch pha khi môi trường kết thúc (Out-of-Sync Done Observation)**
  - *Tệp ảnh hưởng:* `training/run_parallel_sim.py`, `pipelines/*/training/run_parallel_sim.py`.
  - *Nguyên nhân:* Khi môi trường bị `done`, observation trả về trong mảng `next_obs` là trạng thái robot đã ngã xuống sàn thay vì trạng thái reset của episode mới, làm sai lệch hàm cập nhật Bellman trong bộ đệm RL.
  - *Khắc phục:* Cập nhật tức thời `next_obs[i] = self.reset_env(i)` ngay trong hàm `step()` khi phát hiện `dones[i] == True`, đồng thời đặt lại `last_actions[i] = 0.0`.

### Giai Đoạn 3: Pipeline Huấn Luyện PPO & Xuất Mô Hình (Task 3)
- **Commit:** `cd08bf9` (`feat(rl): implement ppo training pipeline and policy export for rpi5 onnx and esp32 int8`)
- **Điểm nghẽn 6: Thiếu pipeline huấn luyện học tăng cường chuẩn mực**
  - *Tệp ảnh hưởng:* `scripts/train_ppo.py`, `tests/test_ppo_training.py`.
  - *Nguyên nhân:* Dự án chỉ có bộ mô phỏng mà chưa có module huấn luyện PPO chính thức tương thích với không gian quan sát 42 chiều và 8 hành động.
  - *Khắc phục:* Xây dựng `scripts/train_ppo.py` với kiến trúc mạng nơ-ron Actor-Critic ($42 \to 256 \to 256 \to 8$), kích hoạt ELU, Orthogonal Initialization, Generalized Advantage Estimation (GAE $\gamma=0.99, \lambda=0.95$), Clipped surrogate objective ($\epsilon=0.2$), lưu checkpoint `.pt` và cờ `--dry-run` cho CI/CD.
- **Điểm nghẽn 7: Thiếu công cụ xuất mô hình đa nền tảng**
  - *Tệp ảnh hưởng:* `scripts/export_policy.py`, `models/microduck_rpi5_policy.onnx`, `pipelines/dual_esp32s3_stack/deployment/*.h`.
  - *Nguyên nhân:* Khó khăn trong việc chuyển giao mạng Actor từ PyTorch sang định dạng nhúng trên Linux SBC và vi điều khiển.
  - *Khắc phục:* Xây dựng `scripts/export_policy.py`:
    * Xuất Float32 ONNX Opset 13 cho Raspberry Pi 5 (`models/microduck_rpi5_policy.onnx`), kiểm chứng sai số tuyệt đối $< 2 \times 10^{-7}$ so với PyTorch.
    * Lượng tử hóa đối xứng INT8, sinh mảng trọng số `const int8_t` trong `esp32_policy_weights.h` và cấu hình `esp32_quant_config.h` (kèm hàm nhúng `evaluate_policy_int8`) cho Dual ESP32-S3, biên dịch C++ độc lập.

### Giai Đoạn 4: Đồng Bộ Hóa Mã Nguồn Triển Khai Phần Cứng (Task 4)
- **Commit:** `02a2ce9` (`fix(hardware): sync last_raw_actions, separate lidar i2c pins and correct imu gravity sign`)
- **Điểm nghẽn 8: Lệch định dạng hành động bước trước (`last_raw_actions`)**
  - *Tệp ảnh hưởng:* `pipelines/rpi5_stack/deployment/infer_rpi5.py`, `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`.
  - *Nguyên nhân:* Trong mô phỏng, chiều $25..32$ của vector quan sát là hành động chuẩn hóa $[-1.0, 1.0]$. Tuy nhiên, trên mã nguồn chạy thật lại truyền góc radian mục tiêu $q_{\text{target}}$, làm sai lệch toàn bộ phân phối đầu vào của mạng nơ-ron.
  - *Khắc phục:* Lưu trữ và cập nhật `last_raw_actions` chuẩn hóa $[-1.0, 1.0]$ ở cả Pi 5 (`infer_rpi5.py`) và MCU 2 (`mcu2_main.ino`).
- **Điểm nghẽn 9: Xung đột bus I2C giữa LiDAR và Camera trên MCU 1**
  - *Tệp ảnh hưởng:* `pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino`, `dual_esp32s3_hardware_spec.md`.
  - *Nguyên nhân:* Cụm 5 ToF LiDAR và giao tiếp SCCB của Camera OV2640 cùng dùng chung chân GPIO 4 và 5 gây nghẽn bus và treo hệ thống.
  - *Khắc phục:* Tách cụm LiDAR 5 tia sang bus I2C chuyên biệt trên GPIO 19 (SDA) và GPIO 20 (SCL), giữ nguyên GPIO 4/5 cho camera.
- **Điểm nghẽn 10: Sai lệch dấu vector trọng trường IMU & Suy luận INT8 trên MCU 2**
  - *Tệp ảnh hưởng:* `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`, `tests/test_deployment_artifacts.py`.
  - *Nguyên nhân:* Gia tốc trọng trường đọc từ MPU6050 hướng lên là $+1.0$, trong khi hệ quy chiếu MuJoCo là $-1.0$. Đồng thời MCU 2 thiếu mã suy luận mạng nơ-ron cục bộ.
  - *Khắc phục:* Đảo dấu các trục gia tốc (`-ax`, `-ay`, `-az`) kết hợp bộ lọc số IIR thông thấp ($\alpha = 0.80$), nhúng trực tiếp hàm `evaluate_policy_int8()` để thực hiện suy luận cục bộ $< 2.8\text{ ms}$ tại $50\text{ Hz}$.

### Giai Đoạn 5: Khôi Phục Công Cụ Trực Quan & Kiểm Thử Toàn Diện (Task 5)
- **Tệp ảnh hưởng:**
  - `scripts/interactive_viewer.py`
  - `tests/test_interactive_viewer.py`
  - `README.md`
  - `his.md`
- **Nội dung thực hiện:**
  - Hoàn thiện `scripts/interactive_viewer.py`: Chuẩn hóa CLI parser, hỗ trợ trực quan hóa 3D passive viewer, tự động chuyển đổi chế độ headless mode khi không có màn hình hiển thị kèm báo cáo chỉ số thăng bằng, hỗ trợ dòng lệnh tương tác `--interactive` điều khiển góc từng khớp, hàm `main()` trả về mã thoát $0$.
  - Mở rộng bộ kiểm thử `tests/test_interactive_viewer.py`: Bổ sung bài test kiểm tra hàm `main()` ở chế độ headless trả về $0$ và kiểm thử tự động toàn bộ chuỗi lệnh trong CLI viewer.
  - Cập nhật toàn diện tài liệu `README.md` bằng tiếng Việt phản ánh đầy đủ 10 điểm nghẽn kỹ thuật, hướng dẫn huấn luyện PPO và triển khai mạch thật.
  - Ghi nhận lịch sử chi tiết 10 điểm nghẽn kỹ thuật vào `his.md`.

---

### Kết Quả Kiểm Thử Toàn Diện Cuối Cùng (Final Test Suite Verification)
- **Lệnh chạy:** `pytest -v`
- **Kết quả:** **83 / 83 tests PASSED (100%)**
- **Độ ổn định:** Toàn bộ các bài kiểm tra mô hình vật lý, môi trường mô phỏng song song, huấn luyện PPO, xuất mô hình ONNX/INT8, firmware Arduino C++, và công cụ trực quan 3D đều vượt qua tuyệt đối.
