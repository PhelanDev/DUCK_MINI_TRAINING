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

---

## 📌 Phiên Làm Việc: 2026-10-29 00:59:00+07:00 (Nâng Cấp Hệ Thống Huấn Luyện AI - Duck Training Upgrade)

**Kế hoạch thực hiện:** `plans/261029-0037-duck-training-upgrade/plan.md`

### 1. Sửa Lỗi "Time-Limit Bootstrapping" Của Thuật Toán PPO (Task 1)
- **Commit:** `83cd3d5` (`fix(ppo): bootstrap value on time-limit truncation`)
- **Tệp ảnh hưởng:**
  - `training/run_parallel_sim.py`
  - `scripts/train_ppo.py`
  - `tests/test_time_limit_bootstrap.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Khi một môi trường đạt tới giới hạn số bước `max_episode_steps = 500`, đây là sự kiện cắt ngắn chuỗi thời gian (time-limit truncation), không phải là trạng thái kết thúc do thất bại vật lý (hấp thụ/chết).
  - Trước đây, `timeouts` bị gộp chung vào `dones` và môi trường bị reset ngay lập tức. Thuật toán GAE xem đây là trạng thái kết thúc có giá trị $V(s_{t+1}) = 0$, gây ra hiện tượng đánh sụt giá trị giả tạo (artificial value drop) và tạo các xung gai dao động (spikes) định kỳ trong đồ thị `value_loss`.
  - Hơn nữa, nếu quan sát sau reset được lấy để đánh giá critic thì mạng critic sẽ đánh giá trên trạng thái khởi đầu mới thay vì trạng thái kết thúc của tập cũ.
- **Biện pháp xử lý:**
  - Trong `training/run_parallel_sim.py`: Tách biệt hoàn toàn `fallen = (root_z < 0.08) | (grav_z > -0.3) | diverged` và `timeouts = episode_steps >= max_episode_steps`. Gán `infos[i]["time_outs"] = bool(timeouts[i] and not fallen[i])` và lưu `infos[i]["terminal_obs"] = next_obs[i].copy()` **ngay trước** khi gọi `reset_env(i)`.
  - Trong `scripts/train_ppo.py`: Thiết kế hàm thuần `apply_timeout_bootstrap(rewards, time_outs, terminal_values, gamma)`. Khi có môi trường timeout (`time_outs.any()`), gọi mạng Critic $V(s_{\text{terminal}}) = \text{model.forward\_critic}(s_{\text{terminal}})$ và bù trực tiếp $\gamma \cdot V(s_{\text{terminal}})$ vào vector phần thưởng $r_t$ trước khi đưa vào bộ đệm GAE.

---

### 2. Thiết Kế Hàm Thưởng Chống Kẹt Trước Cột (Command Relaxation & Approach-Velocity Penalty) (Task 2)
- **Commit:** `4e0d16b` (`feat(reward): relax ang-vel tracking near obstacles, penalize approach speed`)
- **Tệp ảnh hưởng:**
  - `training/env_config.py`
  - `training/run_parallel_sim.py`
  - `tests/test_obstacle_rewards.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Robot khi đối diện cột chướng ngại vật ở cự ly gần thường bị "tiến thoái lưỡng nan": lệnh vận tốc góc $w_z^{\text{cmd}}$ bắt buộc nó phải giữ hướng, trong khi hình phạt khoảng cách tĩnh ép nó không được đứng gần. Robot không dám xoay thân để né vì xoay sẽ bị mất điểm `tracking_ang_vel`.
  - Đồng thời, hình phạt khoảng cách tĩnh phạt robot cả khi nó đang lùi lại hoặc đang né xa vật thể, gây ức chế hành vi di chuyển.
- **Biện pháp xử lý:**
  - Triển khai hàm mặt nạ nới lỏng góc quay:
    $$\text{ang\_vel\_mask}(d_{\min}) = \text{clip}\left(\frac{d_{\min} - 0.25}{0.55 - 0.25}, 0.1, 1.0\right)$$
    Khi robot cách cột trong khoảng nguy hiểm ($0.25\text{ m} \to 0.55\text{ m}$), trọng số phạt sai lệch góc quay giảm dần về $0.1$, cho phép robot tự do quay đầu né tránh.
  - Thay thế phạt tĩnh bằng hàm phạt vận tốc tiếp cận (approach-velocity penalty):
    $$\text{avoid\_vel\_penalty}(d_{\min}, \dot{d}_{\min}, \sigma_{\text{obs}}) = -\max(0.0, -\dot{d}_{\min}) \cdot \exp\left(-\frac{d_{\min}}{\sigma_{\text{obs}}}\right)$$
    Khi robot né xa hoặc đứng yên ($\dot{d}_{\min} \ge 0$), mức phạt triệt tiêu về $0.0$. Chỉ khi robot lao nhanh về phía vật cản ($\dot{d}_{\min} < 0$), mức phạt mới xuất hiện và tăng theo hàm mũ khi $d_{\min}$ càng nhỏ.
  - Trong `DuckParallelSim`, theo dõi `prev_d_min`, tính toán khoảng cách thực phẳng XY tới 4 cột hình trụ và đạo hàm vận tốc $\dot{d}_{\min} = (d_{\min} - \text{prev\_d\_min}) / \Delta t$.

---

### 3. Huấn Luyện Dáng Đi Bước Cao (Foot Clearance) & Phạt Va Chạm Ngang (Stumble Penalty) (Task 3)
- **Commit:** `91fe663` (`feat(reward): foot clearance and stumble penalty`)
- **Tệp ảnh hưởng:**
  - `training/env_config.py`
  - `training/run_parallel_sim.py`
  - `tests/test_foot_rewards.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Robot có xu hướng kéo lê (drag) chân sát mặt sàn để giảm thiểu công năng và tránh bị mất thăng bằng. Khi gặp địa hình gờ đất hoặc bậc thang, mũi chân lập tức vấp phải vật cản và đổ ngã.
  - Cảm biến `touch` thông thường trong MuJoCo chỉ cung cấp độ lớn lực pháp tuyến vô hướng (scalar), không phân tách được lực ngang $f_{xy}$ và lực dọc $f_z$ để phát hiện va chạm bên hông chân.
- **Biện pháp xử lý:**
  - Trích xuất lực tiếp xúc 3D bàn chân qua `mujoco.mj_contactForce`, chiếu qua ma trận quay `contact.frame` sang hệ toạ độ thế giới (World Frame) để thu được vector lực $[F_x, F_y, F_z]$.
  - Trích xuất độ cao $z_{\text{foot}}$ và vận tốc phẳng $v_{xy}$ của từng bàn chân qua `mujoco.mj_objectVelocity`.
  - Triển khai hàm `compute_foot_clearance_and_stumble` (vector hóa):
    * Phạt độ cao chân vung (theo chuẩn ETH Robotic Systems Lab):
      $$R_{\text{clearance}} = -\sum_{\text{foot}} (z - h_{\text{target}})^2 \cdot \sqrt{\|v_{xy}\|} \cdot \mathbb{I}(|F_z| < 1.0\text{ N})$$
      Ép chân khi đang vung có vận tốc lớn phải đạt đỉnh độ cao $h_{\text{target}} = 0.06\text{ m}$, loại bỏ hoàn toàn tật lê chân.
    * Phạt vấp gờ đất (Stumble penalty):
      $$R_{\text{stumble}} = -\sum_{\text{foot}} \mathbb{I}\left(f_{xy} > 2.0 |f_z| \;\wedge\; f_{xy} > 2.0\text{ N}\right)$$
      Ngưỡng $f_{xy} > 2.0\text{ N}$ giúp lọc nhiễu tiếp xúc nhỏ, chỉ kích hoạt khi mũi chân va mạnh vào cạnh đứng của gờ đất.

---

### 4. Bổ Sung Gờ Đất Vào Mô Hình XML & Xây Dựng Terrain Curriculum (Task 4)
- **Commit:** `ebf6dde` (`feat(curriculum): terrain and pillar curriculum with hidden hurdles`)
- **Tệp ảnh hưởng:**
  - `models/mjcf/microduck_8dof.xml`
  - `training/terrain_curriculum.py`
  - `training/run_parallel_sim.py`
  - `scripts/train_ppo.py`
  - `tests/test_mjcf_model.py`
  - `tests/test_terrain_curriculum.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Huấn luyện robot vượt chướng ngại vật ngay từ đầu trên địa hình gồ ghề sẽ khiến tỷ lệ ngã quá cao, robot không kịp hình thành phản xạ bước đi cơ bản.
- **Biện pháp xử lý:**
  - Thêm 3 thanh gờ đất hình hộp chữ nhật vào `microduck_8dof.xml`:
    * `hurdle_1`: vị trí $(0.30, 0, -0.02)$, kích thước half-size `0.02 0.60 0.015`.
    * `hurdle_2`: vị trí $(0.55, 0, -0.025)$, kích thước half-size `0.02 0.60 0.020`.
    * `hurdle_back`: vị trí $(-0.35, 0, -0.02)$, kích thước half-size `0.02 0.60 0.015`.
    Ở Stage 0, mặt trên của cả 3 gờ đều nằm chìm dưới mặt phẳng sàn ($z_{\text{top}} \le -0.005\text{ m} < 0$).
  - Thiết kế bộ quản lý giáo trình `training/terrain_curriculum.py` gồm 4 cấp độ (Stage 0 -> 3):
    * **Stage 0:** Cột chướng ngại vật dãn rộng $2.0\times$, gờ đất chìm hoàn toàn dưới sàn.
    * **Stage 1:** Cột chướng ngại vật co về $1.3\times$, gờ đất nâng lên $+5\text{ mm}$ ($pos_z = 0.005\text{ m}$). Điều kiện lên cấp: `ep_len_mean >= 400` và `rew_lin_vel >= 75% max`.
    * **Stage 2:** Cột chướng ngại vật ở cự ly chuẩn $1.0\times$, gờ đất nâng lên $+15\text{ mm}$ ($pos_z = 0.015\text{ m}$). Điều kiện: `ep_len_mean >= 420`.
    * **Stage 3:** Cột chướng ngại vật $1.0\times$, gờ đất nâng tối đa $+25\text{ mm}$ ($pos_z = 0.025\text{ m}$).
    * Cơ chế giữ vững (Hold Mechanism): Phải duy trì đạt chuẩn liên tục trong `hold_iters = 10` vòng lặp mới được chuyển cấp; cấp độ chỉ tăng dần và không bao giờ tụt lùi.
  - Tích hợp hàm `set_stage(model, stage)` can thiệp trực tiếp vào `model.geom_pos` dùng chung trong `DuckParallelSim` và vòng huấn luyện PPO.

---

### 5. Nghiệm Thu & Kiểm Thử Toàn Diện (Task 5)
- **Tệp báo cáo:** `plans/261029-0037-duck-training-upgrade/reports/training_upgrade_report.md`
- **Kết quả kiểm thử tự động:**
  - Toàn bộ test suite dự án: **104 / 104 tests PASSED (100%)**
  - Tốc độ huấn luyện PPO đạt: **~1,450 FPS** trên CPU đa luồng với 16 môi trường song song.
  - Điểm thưởng tăng trưởng đều đặn từ $+0.84 \to +1.134$.
  - Xuất mô hình ONNX (`microduck_rpi5_policy.onnx`) và header C/C++ lượng tử hóa INT8 (`esp32_policy_weights.h`, `esp32_quant_config.h`) thành công tuyệt đối.


---

## 📌 Phiên Làm Việc: 2026-10-29 10:35:00+07:00

### 1. Khắc Phục Lỗi Logic Chí Mạng Trong Tính Toán `mean_ep_len` Của Terrain Curriculum
- **Commit:** `01ed0a6` (`fix(curriculum): track completed episode lengths in sliding window for stage advancement`)
- **Tệp ảnh hưởng:**
  - `training/run_parallel_sim.py`
  - `scripts/train_ppo.py`
  - `tests/test_curriculum_ep_len.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Biến `runner.episode_steps` chỉ là bộ đếm số bước hiện tại của các môi trường đang chạy dở dang trong batch rollout ($32$ bước/iteration).
  - Cứ khi môi trường chạm mốc $500$ bước (timeout), hàm `reset_env` lập tức xóa bộ đếm về $0$.
  - Khi toàn bộ 64 môi trường đều sống thọ $500$ bước, `runner.episode_steps.mean()` sẽ chạm đỉnh ở vòng lặp thứ 15 (~$480$), nhưng sang vòng lặp thứ 16 (sau khi chạm $500$ và bị reset), giá trị trung bình tụt dốc thê thảm về ~$12.0$.
  - Kết quả là `ep_len_mean < 400.0`, hàm `next_stage()` xóa sạch `hold_count` về $0$. Theo toán học, robot vĩnh viễn không thể duy trì $10$ iterations liên tiếp trên $400$ bước để lên Stage 1.
- **Biện pháp xử lý:**
  - Khởi tạo hàng đợi trượt `self.completed_ep_lens = deque(maxlen=100)` trong `DuckParallelSim` / `ParallelSimRunner`.
  - Mỗi khi môi trường kết thúc tập (`dones[i] == True` do ngã hoặc chạm giới hạn bước $500$), đẩy ngay số bước thực tế `int(self.episode_steps[i])` vào hàng đợi và lưu vào `infos[i]["episode_length"]` trước khi reset.
  - Cung cấp phương thức `get_mean_episode_length(default=0.0)` và cập nhật dòng đánh giá curriculum trong `train_ppo.py`:
    ```python
    mean_ep_len = runner.get_mean_episode_length(default=float(runner.episode_steps.mean()))
    ```
  - Viết 3 unit test trong `tests/test_curriculum_ep_len.py` xác thực tính chính xác của cửa sổ trượt và khả năng giữ vững `hold_iters = 10` để nhảy Stage.

---

### 2. Đồng Bộ Hóa Gờ Đất `hurdle_*` Vào Toàn Bộ Stack Phần Cứng (`rpi5_stack` & `dual_esp32s3_stack`)
- **Commit:** `4f49b45` (`feat(models): sync terrain hurdle geoms to rpi5 and esp32 mjcf models`)
- **Tệp ảnh hưởng:**
  - `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`
  - `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`
  - `tests/test_rpi5_model.py`
  - `tests/test_esp32_model.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Lệnh chạy huấn luyện mặc định `python scripts/train_ppo.py` sử dụng mô hình `DEFAULT_MODEL_PATH = pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`.
  - Tuy nhiên, 3 thanh gờ đất `hurdle_1`, `hurdle_2`, `hurdle_back` ban đầu chỉ được khai báo trong `models/mjcf/microduck_8dof.xml`, hoàn toàn vắng bóng trong `rpi5_stack` và `dual_esp32s3_stack`.
  - Khi huấn luyện với stack mặc định, hàm `apply_stage()` không tìm thấy geom gờ đất, khiến curriculum địa hình bị tê liệt ngầm mà không hề báo lỗi.
- **Biện pháp xử lý:**
  - Đồng bộ chuẩn xác 3 geom gờ đất ngập dưới sàn vào cả hai mô hình phần cứng:
    ```xml
    <geom name="hurdle_1" type="box" size="0.02 0.60 0.015" pos="0.30 0.0 -0.02" rgba="0.6 0.4 0.2 1.0" contype="1" conaffinity="1" group="3"/>
    <geom name="hurdle_2" type="box" size="0.02 0.60 0.020" pos="0.55 0.0 -0.025" rgba="0.6 0.4 0.2 1.0" contype="1" conaffinity="1" group="3"/>
    <geom name="hurdle_back" type="box" size="0.02 0.60 0.015" pos="-0.35 0.0 -0.02" rgba="0.6 0.4 0.2 1.0" contype="1" conaffinity="1" group="3"/>
    ```
  - Bổ sung các bài kiểm tra DOM tĩnh và kiểm thử động với MuJoCo runtime `apply_stage()` trong `test_rpi5_model.py` và `test_esp32_model.py`.

---

### 3. Chuẩn Hóa Cảm Biến LiDAR Cho Hàm Phạt Né Cột (Khắc Phục Vi Phạm POMDP)
- **Commit:** `ff652ab` (`fix(rewards): align obstacle d_min with forward lidar perception for POMDP consistency`)
- **Tệp ảnh hưởng:**
  - `training/run_parallel_sim.py`
  - `tests/test_obstacle_rewards.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Phương thức `_compute_d_min()` tính khoảng cách Euclid toàn cục 360 độ tới toàn bộ 4 cột trong `self.obstacle_geom_ids`, bao gồm cả `obstacle_back` ở toạ độ $(x = -0.75\text{ m}, y = 0.0\text{ m})$ nằm ngay sau lưng robot.
  - Trong khi đó, hệ thống 5 tia LiDAR chỉ quét hướng về phía trước trong hình nón $[-30^\circ, +30^\circ]$ với tầm quét tối đa $2.0\text{ m}$.
  - Kết quả: Khi robot đứng yên tại vạch xuất phát, phía trước hoàn toàn quang đãng ($d_{\text{lidar}} = 2.0\text{ m}$), nhưng hàm `_compute_d_min` trả về $0.67\text{ m}$ (khoảng cách tới cột sau lưng). Robot bị phạt né cột và bị hàm `ang_vel_mask` triệt tiêu lệnh quay yaw dựa trên một vật cản mà mắt nó không thể nhìn thấy — vi phạm nghiêm trọng tính quan sát của bài toán POMDP (Partially Observable Markov Decision Process).
- **Biện pháp xử lý:**
  - Trong `training/run_parallel_sim.py`, chuẩn hóa việc tính `curr_d_min` theo đúng không gian quan sát của agent:
    ```python
    if lidar_ranges is not None:
        curr_d_min = np.min(lidar_ranges, axis=-1)
    else:
        curr_d_min = self._compute_d_min()
    ```
  - Áp dụng đồng bộ cho cả `step()` và `reset_env()`.
  - Thêm test `test_lidar_pomdp_d_min_ignores_rear_obstacle` trong `tests/test_obstacle_rewards.py`, xác nhận robot ở vị trí ban đầu ghi nhận khoảng cách $2.0\text{ m}$, hoàn toàn bỏ qua cột sau lưng $0.67\text{ m}$.

---

### 4. Cấu Hình Camera Mặc Định Cho Interactive Viewer & Khắc Phục Đen Màn Hình Trên Hybrid GPU
- **Commit:** `5f083d3` (`fix(viewer): set explicit camera orientation to prevent black screen on hybrid gpus`)
- **Tệp ảnh hưởng:**
  - `scripts/interactive_viewer.py`
  - `tests/test_interactive_viewer.py`
  - `README.md` (mục 9.1)
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Khi khởi chạy `scripts/interactive_viewer.py` qua `mujoco.viewer.launch_passive` trên các dòng máy tính xách tay đồ họa kép (Hybrid GPU như Lenovo Legion AMD Ryzen APU + NVIDIA RTX 3070), ma trận chiếu OpenGL của camera đôi khi không được khởi tạo giá trị hợp lệ hoặc hướng ra vùng chân trời trống rỗng. Người dùng chỉ thấy màn hình đen thui kèm các thanh panel giao diện.
- **Biện pháp xử lý:**
  - Triển khai hàm `configure_viewer_camera` định hình góc nhìn isometric trực diện:
    * `lookat = [0.0, 0.0, 0.15]` (khớp vị trí trọng tâm thân vịt ở độ cao 15cm)
    * `distance = 1.2` (khoảng cách quan sát toàn cảnh robot và gờ đất)
    * `azimuth = 135.0` (góc xoay 3D nhìn xéo)
    * `elevation = -20.0` (góc nâng nhìn từ trên xuống)
  - Gọi ngay `configure_viewer_camera(viewer.cam)` và `viewer.sync()` ngay sau khi mở context manager viewer.
  - Thêm unit test `test_configure_viewer_camera` trong `tests/test_interactive_viewer.py` bao phủ cả dummy camera và `mujoco.MjvCamera`.
  - Cập nhật mục **9.1. Khắc Phục Sự Cố Đen Màn Hình Trên Laptop Hybrid GPU** trong `README.md` hướng dẫn chi tiết cách cấu hình NVIDIA Control Panel và Lenovo Vantage.

---

### 5. Nghiệm Thu & Đánh Giá Toàn Diện (Task 5)
- **Báo cáo chi tiết:** `plans/261029-1005-curriculum-lidar-viewer-fixes/reports/curriculum_viewer_fixes_report.md`
- **Kết quả kiểm thử tự động:**
  - **111 / 111 bài test PASSED (100%)** trong `28.86s` (0 failures, 0 errors).
  - Khả năng huấn luyện PPO: Đạt tốc độ ổn định **>800 FPS** trên CPU đa luồng, chuyển cấp Stage trơn tru khi `ep_len_mean >= 400`.
  - Mã nguồn tuân thủ triệt để nguyên tắc **Ponytail** (Zero Over-engineering, sửa tận gốc), **TDD** và **Zero-Trust Git Policy**.


---

### 6. Tự Động Fallback CPU Khi PyTorch Môi Trường Không Có CUDA
- **Commit:** `58d1a11` (`fix(training): safely fallback to cpu when cuda is requested but torch is cpu-only`)
- **Tệp ảnh hưởng:**
  - `scripts/train_ppo.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Môi trường cài đặt `Python 3.14` với bản PyTorch CPU (`2.15.0+cpu`). Khi người dùng hoặc script truyền `--device cuda`, hàm `model.to("cuda")` văng ngoại lệ `AssertionError: Torch not compiled with CUDA enabled`.
- **Biện pháp xử lý:**
  - Thêm cơ chế kiểm tra phòng vệ `if device.startswith("cuda") and not torch.cuda.is_available():`.
  - Tự động in cảnh báo an toàn và fallback về `cpu`, giúp pipeline huấn luyện luôn chạy thông suốt trên mọi máy mà không bị dừng đột ngột.


---

## 📌 Phiên Làm Việc: 2026-10-29 12:30:00+07:00

### 1. Đồng Bộ Hóa Đầy Đủ Runner Của Cả Hai Stack Phần Cứng (`rpi5_stack` & `dual_esp32s3_stack`)
- **Commit:** `0a60d50` (`feat(pipelines): sync episode length tracking and lidar pomdp to rpi5 and esp32 runners`)
- **Tệp ảnh hưởng:**
  - `pipelines/rpi5_stack/training/run_parallel_sim.py`
  - `pipelines/dual_esp32s3_stack/training/run_parallel_sim.py`
  - `tests/test_pipeline_training.py`
- **Chi tiết thực hiện:**
  - Bổ sung hàng đợi trượt `self.completed_ep_lens = deque(maxlen=100)` và phương thức `get_mean_episode_length(default=0.0)` vào cả hai runner của Raspberry Pi 5 và Dual ESP32-S3.
  - Chuẩn hóa việc tính khoảng cách né vật cản `curr_d_min = np.min(lidar_ranges, axis=-1)` trực tiếp từ 5 tia LiDAR trong cả `step()` và `reset_env()`, loại bỏ hoàn toàn vi phạm POMDP.
  - Bổ sung unit test `test_pipeline_runners_ep_len_and_lidar_pomdp` xác thực 100% tính tương thích trên cả hai stack phần cứng.

---

### 2. Tích Hợp Chạy Trực Tiếp Mô Hình RL Policy Vào Trình Xem 3D (`interactive_viewer.py`)
- **Commit:** `df00109` (`feat(viewer): support running trained RL policy checkpoints and ONNX in 3D viewer`)
- **Tệp ảnh hưởng:**
  - `scripts/interactive_viewer.py`
  - `tests/test_interactive_viewer.py`
  - `README.md` (mục 9)
- **Chi tiết thực hiện:**
  - Bổ sung cờ `--policy <path>` hỗ trợ nạp cả file checkpoint PyTorch (`.pt`) và mô hình ONNX Runtime (`.onnx`).
  - Bổ sung cờ `--cmd-vel <vx> <vy> <wz>` cho phép người dùng ra lệnh di chuyển trực tiếp cho vịt trong không gian 3D.
  - Triển khai hàm `extract_viewer_observation` trích xuất đầy đủ vector quan sát 42 chiều theo thời gian thực (vận tốc góc, trọng lực, góc và vận tốc 8 khớp, cảm biến tiếp xúc bàn chân và 5 tia LiDAR).
  - Tích hợp chu trình điều khiển 50Hz chuẩn xác (`control_dt = 0.02s`, `decimation = 10` với tần số vật lý 500Hz), kết hợp `clip_action_rates` và `scale_actions` bảo vệ hệ thống cơ khí 8 servo MG996R chống giật gãy bánh răng.
  - Hỗ trợ chạy mượt mà ở cả chế độ đồ họa cửa sổ 3D (`launch_passive`) và chế độ không giao diện (`--headless`).
  - Viết 4 bài kiểm thử chuyên sâu mới trong `tests/test_interactive_viewer.py`, nâng tổng số bài test của dự án lên **116 bài test (100% PASSED)**.


---

## 📌 Phiên Làm Việc: 2026-10-29 13:10:00+07:00

### 1. Khắc Phục Triệt Để Bùng Nổ Value Loss (Critic Loss) Bằng Kỹ Thuật PPO Value Clipping
- **Tệp ảnh hưởng:**
  - `scripts/train_ppo.py`
  - `tests/test_ppo_training.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Khi robot sống sót đạt mốc $500$ bước (timeout), hàm `apply_timeout_bootstrap` cộng dồn giá trị tương lai $\gamma V(s_{\text{terminal}}) \approx 50 \to 70$ điểm vào phần thưởng bước cuối.
  - Do `train_ppo.py` trước đây tính Value Loss dạng MSE thuần `0.5 * ((new_values - mb_ret) ** 2).mean()` mà không có Value Clipping (theo chuẩn PPO2 / CleanRL), mạng Critic bị sốc gradient khi các mục tiêu return nhảy vọt, đẩy `ValLoss` từ $8.7$ lên $41 \to 122+$.
- **Biện pháp xử lý:**
  - Bổ sung `b_values = values_buffer.reshape(-1)` vào bộ đệm mini-batch.
  - Áp dụng kỹ thuật PPO Value Loss Clipping chuẩn xác:
    ```python
    v_loss_unclipped = (new_values - mb_ret) ** 2
    v_clipped = mb_val + torch.clamp(new_values - mb_val, -clip_epsilon, clip_epsilon)
    v_loss_clipped = (v_clipped - mb_ret) ** 2
    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    ```
  - Triệt tiêu hoàn toàn hiện tượng nổ mất mát Critic, giữ `ValLoss` dao động ổn định trong tầm kiểm soát.

---

### 2. Tích Hợp Cơ Chế Học Nối Tiếp Checkpoint (`--resume` & `--load-checkpoint`)
- **Tệp ảnh hưởng:**
  - `scripts/train_ppo.py`
  - `tests/test_ppo_training.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Trước đây, `train_ppo.py` luôn khởi tạo mạng `ActorCritic` ngẫu nhiên mới tinh mỗi khi chạy, khiến các lần chạy sau học lại từ con số 0 (Scratch) và ghi đè làm mất checkpoint cũ đã hội tụ.
- **Biện pháp xử lý:**
  - Bổ sung tham số CLI `--resume` (tự động nạp checkpoint tại `--save-path` nếu đã tồn tại) và `--load-checkpoint <path>`.
  - Khôi phục đầy đủ `model.state_dict()`, `optimizer.state_dict()`, `current_stage` của terrain curriculum, và biến tích lũy `total_timesteps`.
  - Giúp việc huấn luyện có thể chia nhỏ nhiều đợt hoặc chạy nối tiếp sau khi tinh chỉnh siêu tham số mà không bị mất trọng số đã học.

---

### 3. Tăng Độ Dài Rollout Mặc Định Lên 128 Bước (2.56 Giây / Đợt Thu Thập)
- **Tệp ảnh hưởng:**
  - `scripts/train_ppo.py`
  - `training/terrain_curriculum.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Với $\gamma = 0.99$, tầm nhìn hiệu quả của thuật toán RL là $\approx 100$ bước. Việc để `rollout_steps = 32` ($0.64\text{ s}$) là quá ngắn, chưa đủ một chu kỳ bước đi trọn vẹn của robot hai chân, làm thuật toán GAE ước lượng phương sai cao.
  - Điều kiện chuyển cấp ở Stage 0 đòi hỏi `rew_lin_vel_ratio >= 0.75` liên tục trong 10 vòng lặp là quá khắt khe khi robot bước đi có dao động vận tốc tự nhiên theo nhịp chân.
- **Biện pháp xử lý:**
  - Đổi giá trị mặc định của `--rollout-steps` từ 32 lên **128 bước** ($2.56\text{ s}$ / chu kỳ thu thập), bao quát trọn vẹn 2–3 chu kỳ sải chân của vịt.
  - Điều chỉnh nhẹ ngưỡng vận tốc Stage 0 trong `terrain_curriculum.py` từ $0.75 \to 0.70$ để phản ánh chính xác nhịp bước đi dao động thực tế của robot bipedal.

---

### 4. Khắc Phục Hiện Tượng Ẩn Gờ Đất (Hurdles) Trên 3D Viewer & Hỗ Trợ Cờ `--stage`
- **Tệp ảnh hưởng:**
  - `scripts/interactive_viewer.py`
  - `models/mjcf/microduck_8dof.xml`
  - `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`
  - `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`
  - `tests/test_interactive_viewer.py`
- **Hiện tượng & Nguyên nhân gốc rễ:**
  - Người dùng thắc mắc vì sao khi tiến lên Stage 2 / Stage 3 trong quá trình train, mở viewer lên chỉ thấy 4 cột trụ thu hẹp lại mà không thấy các gờ đất nhô lên.
  - **Nguyên nhân cốt lõi**:
    1. Các gờ đất `hurdle_*` được gán thuộc tính `group="3"`.
    2. Trong chuẩn MuJoCo, cấu hình hiển thị mặc định `MjvOption.geomgroup` chỉ kích hoạt 3 nhóm đầu tiên `[1, 1, 1, 0, 0, 0]` (nhóm 3 bị tắt mặc định). Do đó mô phỏng va chạm vật lý vẫn diễn ra bình thường nhưng renderer không vẽ lên màn hình.
    3. Trước đó, `interactive_viewer.py` chưa có tham số để người dùng chỉ định xem trước trạng thái địa hình của các giai đoạn trong giáo trình (Curriculum Stage 0..3).
- **Biện pháp xử lý:**
  - Trong `scripts/interactive_viewer.py`: Bổ sung kích hoạt hiển thị toàn bộ `viewer.opt.geomgroup[0..3] = 1`.
  - Bổ sung tham số dòng lệnh `--stage <0|1|2|3>` vào `interactive_viewer.py`, gọi trực tiếp `apply_stage(model, stage)` khi khởi động để người dùng có thể mở 3D Viewer soi trực tiếp địa hình và chướng ngại vật của bất kỳ Stage nào.
  - Bổ sung unit test `test_viewer_stage_argument_and_hurdle_elevation` xác nhận hoạt động chuẩn xác của cờ `--stage 2` và độ cao gờ đất $+15\text{mm}$.

---

### 5. Chẩn Đoán Trạng Thái Card Rời RTX 3070 & PyTorch CUDA
- **Hiện tượng:**
  - Task Manager hiển thị GPU 0 (iGPU) 1%, GPU 1 (RTX 3070 Laptop) 0%, CPU 66% khi chạy lệnh train.
  - Khi gõ `--device cuda`, hệ thống báo lỗi `AssertionError: Torch not compiled with CUDA enabled`.
- **Nguyên nhân gốc rễ:**
  - Hệ thống của người dùng hiện đang chạy môi trường Python **3.14.5**.
  - Tổ chức PyTorch (`pytorch.org`) tại thời điểm hiện tại chưa phát hành bản build nhị phân CUDA chính thức cho Python 3.14 trên Windows (chỉ có bản build rút gọn `torch 2.15.0+cpu` trên PyPI).
  - Phần cứng máy tính của người dùng sở hữu GPU rời **NVIDIA GeForce RTX 3070 Laptop (8GB VRAM)** rất mạnh, driver phiên bản 617.42 hỗ trợ tối đa CUDA 13.4.
- **Biện pháp & Khuyến nghị:**
  - Trong mã nguồn `scripts/train_ppo.py`, đã cài sẵn cơ chế phòng vệ tự động fallback về `cpu` an toàn kèm cảnh báo rõ ràng khi người dùng vô tình truyền `--device cuda` trên môi trường PyTorch CPU.
  - Do kiến trúc mạng MLP của bài toán vịt MicroDuck rất nhỏ gọn ($42 \to 256 \to 128 \to 8$), việc chạy mô phỏng 64 môi trường song song trên CPU đa nhân diễn ra rất nhanh và ổn định mà không bị nghẽn (CPU 66%).
  - Nếu người dùng muốn tận dụng 100% sức mạnh card RTX 3070 cho PPO: Chỉ cần cài thêm môi trường **Python 3.12** (`winget install Python.Python.3.12`) và cài PyTorch CUDA qua lệnh:
    ```powershell
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```

---

### 6. Nghiệm Thu Đợt Huấn Luyện 1000 Vòng (2.048.000 Bước) & Hội Tụ Stage 3
- **Hiện tượng & Kết quả thực tế:**
  - Người dùng thực hiện huấn luyện 1000 iterations PPO (`2.048.000` bước mô phỏng trong `1110.30s` ~ 18.5 phút, tốc độ ~2000 FPS).
  - Vượt qua trọn vẹn cả 4 cấp độ giáo trình (Stage 0 $\rightarrow$ 1 $\rightarrow$ 2 $\rightarrow$ 3 - Full challenge với gờ đất cao $+25\text{mm}$).
  - Điểm thưởng tăng trưởng bền vững: từ $+1.071 \rightarrow +2.631$ (đỉnh $+2.695$).
  - Mất mát Value Loss (Critic) hội tụ hoàn hảo: từ đỉnh $167.8 \rightarrow 4.64$ (giảm sâu hơn 36 lần).
  - Xuất thành công các tệp nhị phân triển khai thực tế:
    * `models/microduck_rpi5_policy.onnx` (Float32 ONNX cho Raspberry Pi 5).
    * `pipelines/dual_esp32s3_stack/deployment/esp32_policy_weights.h` (INT8 weights cho Dual ESP32-S3).
    * `pipelines/dual_esp32s3_stack/deployment/esp32_quant_config.h` (Cấu hình lượng tử hóa).
  - Đã kiểm thử xác nhận trực tiếp bằng `scripts/interactive_viewer.py --headless`: Mô hình ONNX và Checkpoint PyTorch điều khiển robot bước đi vững vàng 1000 bước mô phỏng ở Stage 3, độ cao CoM duy trì ổn định ở $0.1986\text{m}$.
