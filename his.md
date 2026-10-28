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
