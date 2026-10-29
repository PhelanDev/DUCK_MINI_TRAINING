# Kế Hoạch Khắc Phục 4 Lỗi Chí Mạng (Curriculum, Hurdle Sync, LiDAR POMDP, Viewer Camera)

- **Ngày lập:** 2026-10-29
- **Kế hoạch:** `plans/261029-1005-curriculum-lidar-viewer-fixes/plan.md`
- **Mục tiêu:** Khắc phục triệt để 4 điểm nghẽn kỹ thuật:
  1. Lỗi logic tính `mean_ep_len` khiến vịt vĩnh viễn kẹt ở Stage 0.
  2. Bất đồng bộ 3 gờ đất `hurdle_*` giữa mô hình gốc và 2 stack phần cứng (`rpi5_stack`, `dual_esp32s3_stack`).
  3. Lỗi POMDP hàm phạt khoảng cách né cột (dùng trực tiếp 5 tia LiDAR thay vì tọa độ ảo toàn cục).
  4. Lỗi màn hình đen của MuJoCo Passive Viewer trên laptop đồ họa kép (Hybrid iGPU/dGPU).

---

## 1. Kiến Trúc & Quyết Định Kỹ Thuật (Architecture & Decisions)

### 1.1. Lịch sử tập chạy (Sliding Window of Completed Episode Lengths)
- **Vấn đề:** `runner.episode_steps` là bộ đếm tức thời của các env đang chạy dở dang. Cứ sau 500 bước (timeout), `episode_steps` bị reset về 0, làm `mean_ep_len` rơi từ ~480 xuống ~12. Với `hold_iters = 10`, điều kiện $\ge 400$ không bao giờ duy trì được quá 3 iterations liên tiếp.
- **Giải pháp:**
  - Trong `ParallelSimRunner` (`training/run_parallel_sim.py`), thêm `self.completed_ep_lens = deque(maxlen=100)`.
  - Trong `step()`: Khi env $i$ kết thúc (`dones[i] == True`), trước khi gọi `reset_env(i)`, đẩy độ dài tập vừa hoàn thành vào hàng đợi:
    `self.completed_ep_lens.append(int(self.episode_steps[i]))`.
    Đồng thời ghi vào `infos[i]["episode_length"] = int(self.episode_steps[i])`.
  - Thêm phương thức `runner.get_mean_episode_length() -> float`: Trả về `float(np.mean(self.completed_ep_lens))` nếu hàng đợi có dữ liệu; nếu chưa có tập nào kết thúc thì fallback về `float(self.episode_steps.mean())`.
  - Trong `scripts/train_ppo.py`: Thay `runner.episode_steps.mean()` bằng `runner.get_mean_episode_length()`.

### 1.2. Đồng bộ gờ đất sang `microduck_rpi5_8dof.xml` và `microduck_esp32_8dof.xml`
- **Vấn đề:** `DEFAULT_MODEL_PATH` trong `train_ppo.py` trỏ tới `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`, nhưng 3 geom `hurdle_*` mới chỉ có trong `models/mjcf/microduck_8dof.xml`. Hàm `apply_stage()` bỏ qua do không tìm thấy geom.
- **Giải pháp:**
  - Bổ sung 3 geom `hurdle_1`, `hurdle_2`, `hurdle_back` vào cả 2 tệp XML của `rpi5_stack` và `dual_esp32s3_stack` với cùng tọa độ, kích thước, contype/conaffinity và `group="3"`.

### 1.3. Chuẩn hóa `curr_d_min` theo 5 tia LiDAR (POMDP Alignment)
- **Vấn đề:** `_compute_d_min()` tính khoảng cách Euclid từ tọa độ thân robot tới 4 cột trong World Frame. Khi robot có LiDAR (góc quét $[-30^\circ, +30^\circ]$ phía trước), nếu robot lùi về phía cột sau lưng (nằm ngoài góc nhìn LiDAR), robot không nhìn thấy cột nhưng vẫn bị phạt nặng, vi phạm tính khả quan sát cục bộ (POMDP). Với model 37 chiều không có LiDAR, robot hoàn toàn mù trước cột nhưng vẫn bị phạt.
- **Giải pháp:**
  - Trong `training/run_parallel_sim.py`:
    ```python
    if lidar_ranges is not None:
        curr_d_min = np.min(lidar_ranges, axis=-1)
    else:
        curr_d_min = self._compute_d_min()
    ```
  - Khi robot có LiDAR, cự ly né vật cản được đo trực tiếp từ 5 tia mắt thần. Vừa loại bỏ hoàn toàn tính toán khoảng cách tọa độ toàn cục dư thừa (nhanh hơn gấp nhiều lần), vừa bảo đảm tính nhất quán triệt để của không gian quan sát POMDP.

### 1.4. Thiết lập Camera mặc định cho MuJoCo Passive Viewer
- **Vấn đề:** Trên máy chạy GPU kép (AMD/Intel iGPU + NVIDIA dGPU), MuJoCo passive viewer mở lên bị đen khung hình do viewport camera chưa được định vị đúng điểm nhìn robot.
- **Giải pháp:**
  - Trong `scripts/interactive_viewer.py`, ngay sau `with mujoco.viewer.launch_passive(model, data) as viewer:`:
    ```python
    viewer.cam.lookat[:] = [0.0, 0.0, 0.15]
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0
    ```
  - Cập nhật hướng dẫn thiết lập Windows Graphics Settings trong tài liệu.

---

## 2. Kế Hoạch Thực Hiện Chi Tiết (5 Tasks)

### Task 1: Sửa Logic Điều Kiện Curriculum `mean_ep_len`
- **Tệp sửa:**
  - `training/run_parallel_sim.py`
  - `scripts/train_ppo.py`
- **Tệp test mới:** `tests/test_curriculum_ep_len.py`
- **Các bước:**
  1. *RED:* Viết test kiểm tra `completed_ep_lens` lưu giữ độ dài các tập đã kết thúc, `get_mean_episode_length()` phản ánh đúng độ dài tập hoàn thành, và `next_stage()` duy trì được qua các tập 500 bước.
  2. *GREEN:* Cài đặt `deque(maxlen=100)` trong `ParallelSimRunner`, đẩy độ dài tập trước khi reset env, cập nhật `train_ppo.py`.
  3. *VERIFY:* Chạy `pytest tests/test_curriculum_ep_len.py -v`.
  4. *COMMIT:* `git commit -m "fix(curriculum): track completed episode lengths in sliding window for stage advancement"`

### Task 2: Đồng Bộ Gờ Đất `hurdle_*` Vào Toàn Bộ Stack Phần Cứng
- **Tệp sửa:**
  - `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`
  - `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`
- **Tệp test sửa:**
  - `tests/test_rpi5_model.py`
  - `tests/test_esp32_model.py`
- **Các bước:**
  1. *RED:* Bổ sung test assertion trong `test_rpi5_model.py` và `test_esp32_model.py` yêu cầu tồn tại 3 geom `hurdle_1`, `hurdle_2`, `hurdle_back` và `apply_stage()` thay đổi được độ cao gờ.
  2. *GREEN:* Chèn 3 thẻ `<geom name="hurdle_*" .../>` vào cả 2 file XML.
  3. *VERIFY:* Chạy `pytest tests/test_rpi5_model.py tests/test_esp32_model.py tests/test_terrain_curriculum.py -v`.
  4. *COMMIT:* `git commit -m "feat(models): sync terrain hurdle geoms to rpi5 and esp32 mjcf models"`

### Task 3: Chuẩn Hóa Cảm Biến LiDAR Cho Hàm Phạt Né Cột (POMDP)
- **Tệp sửa:**
  - `training/run_parallel_sim.py`
- **Tệp test sửa:**
  - `tests/test_obstacle_rewards.py`
- **Các bước:**
  1. *RED:* Viết test xác nhận khi có `lidar_ranges`, `d_min` được tính từ `np.min(lidar_ranges, axis=-1)` thay vì khoảng cách Euclid toàn cục.
  2. *GREEN:* Cập nhật logic tính `curr_d_min` trong `step()` và `reset_env()`.
  3. *VERIFY:* Chạy `pytest tests/test_obstacle_rewards.py -v`.
  4. *COMMIT:* `git commit -m "fix(rewards): align obstacle d_min with forward lidar perception for POMDP consistency"`

### Task 4: Cấu Hình Camera Góc Nhìn Mặc Định Cho Interactive Viewer
- **Tệp sửa:**
  - `scripts/interactive_viewer.py`
- **Tệp test sửa:**
  - `tests/test_interactive_viewer.py`
- **Các bước:**
  1. Thêm camera view orientation `viewer.cam.lookat`, `distance`, `azimuth`, `elevation` ngay sau `launch_passive`.
  2. Cập nhật test xác nhận cấu hình camera.
  3. *VERIFY:* Chạy `pytest tests/test_interactive_viewer.py -v`.
  4. *COMMIT:* `git commit -m "fix(viewer): set explicit camera orientation to prevent black screen on hybrid gpus"`

### Task 5: Nghiệm Thu & Cập Nhật Lịch Sử `his.md`
- **Các bước:**
  1. Chạy huấn luyện PPO thử nghiệm 30 vòng lặp, kiểm chứng bước nhảy lên Stage 1 khi `mean_ep_len >= 400`.
  2. Chạy toàn bộ test suite `pytest -v` đảm bảo 100% PASS.
  3. Cập nhật chi tiết các phát hiện, nguyên nhân gốc rễ và mã nguồn sửa đổi vào `d:\duckV1\his.md`.
  4. Tạo báo cáo tại `plans/261029-1005-curriculum-lidar-viewer-fixes/reports/curriculum_viewer_fixes_report.md`.
  5. Hỏi ý kiến người dùng trước khi thực hiện `git push`.

