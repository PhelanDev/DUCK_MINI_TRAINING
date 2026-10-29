# Đặc Tả Thiết Kế Kỹ Thuật: Hệ Thống Cọc 3 Lớp Zic-Zac Kết Hợp Dải Cầu Thang & Curriculum Động Cho MicroDuck 8-DOF

**Ngày tạo**: 2026-10-29  
**Dự án**: MicroDuck 8-DOF Bipedal Robot  
**Trạng thái**: Đã phê duyệt ý tưởng (Design Approved)  
**Tác giả**: Antigravity Pair Programmer & Meta-Orchestrator  

---

## 1. Mục Tiêu & Bối Cảnh (Context & Goal)

### 1.1 Mục tiêu tổng quát
Xây dựng một hệ thống địa hình phức hợp liên hoàn dành cho robot hai chân MicroDuck 8-DOF gồm 2 phân khu nối tiếp:
1. **Phân khu Rừng cọc né tránh 3 lớp (3-Layer Slalom Pillars)**: Thử thách khả năng đánh lái, uốn lượn thắt nút chai và giữ thăng bằng động khi robot di chuyển qua các khe hẹp.
2. **Phân khu Cầu thang tăng dần và bục đích (Staircase & Landing Platform)**: Thử thách khả năng co gối, nhấc mũi chân vượt chướng ngại vật theo phương thẳng đứng và tiếp đất ổn định trên bục cao.

### 1.2 Nguyên tắc thiết kế (Ponytail & Engineering Constraints)
* **Zero Over-engineering & Native MuJoCo**: Tái sử dụng các hình học cơ bản (`cylinder`, `box`), không dùng heightfield để tránh làm suy giảm FPS khi chạy song song 32–64 luồng.
* **In-place Geometry Mutation**: Thay đổi trực tiếp `model.geom_pos` trong bộ nhớ qua hàm `apply_stage()`, bảo đảm tính `idempotent` và không phải tải lại file XML.
* **Bảo toàn Giao diện Cảm biến**: Giữ nguyên kích thước vector quan sát (37 chiều cơ bản / 42 chiều kèm LiDAR), không làm thay đổi trọng số mạng nơ-ron hoặc firmware C++ trên ESP32-S3 và Raspberry Pi 5.

---

## 2. Kiến Trúc Hình Học Địa Hình (Terrain Geometry & Coordinates)

Đường đua được bố trí dọc theo trục $+X$ với tâm đối xứng quanh trục $Y = 0$:

```
        Y+ (Trái)
         ^
         |   [Pillar L1-Left]        [Pillar L2-Center]        [Pillar L3-Left]       [Stair 1] [Stair 2] [Stair 3] [Platform]
         |      (0.45, 0.35)             (0.80, 0.08)             (1.15, 0.25)        (1.40)    (1.52)    (1.64)     (1.86)
X- <-----+-------------------------------------------------------------------------------------------------------------> X+
  (Lùi)  |                                   (Robot Start: 0, 0)
         |   [Pillar L1-Right]                                 [Pillar L3-Right]
         |      (0.45, -0.35)                                     (1.15, -0.25)
         v
        Y- (Phải)
```

### 2.1 Thông số Hệ thống Cọc 3 Lớp (Slalom Pillars)
Tất cả các cọc đều có vật liệu `rgba="0.8 0.2 0.2 1.0"`, `contype="1"`, `conaffinity="1"`:
* **Lớp 1 (Cổng vào mở rộng)**:
  * `pillar_l1_left`: `type="cylinder"`, `size="0.06 0.15"` (bán kính $0.06\text{ m}$, cao $0.30\text{ m}$), baseline pos: `(0.45, 0.35, 0.15)`.
  * `pillar_l1_right`: `type="cylinder"`, `size="0.06 0.15"`, baseline pos: `(0.45, -0.35, 0.15)`.
* **Lớp 2 (Cọc trung tâm so le / Slalom Pivot)**:
  * `pillar_l2_center`: `type="cylinder"`, `size="0.07 0.15"`, baseline pos: `(0.80, 0.08, 0.15)` (lệch sang trái $8\text{ cm}$ buộc robot phải chuyển hướng lượn lách).
* **Lớp 3 (Cổng hẹp định hướng chân cầu thang / Bottleneck Gate)**:
  * `pillar_l3_left`: `type="cylinder"`, `size="0.06 0.15"`, baseline pos: `(1.15, 0.25, 0.15)`.
  * `pillar_l3_right`: `type="cylinder"`, `size="0.06 0.15"`, baseline pos: `(1.15, -0.25, 0.15)`.
* **Cọc chắn sau (An toàn)**:
  * `pillar_back`: `type="cylinder"`, `size="0.08 0.15"`, baseline pos: `(-0.75, 0.0, 0.15)`.

### 2.2 Thông số Dải Cầu Thang & Bục Đích (Stairs & Platform)
Tất cả các bậc thang có bề rộng theo trục $Y$ là $0.80\text{ m}$ (half-width $0.40\text{ m}$), vật liệu `rgba="0.4 0.6 0.7 1.0"`, `contype="1"`, `conaffinity="1"`, `group="3"`:
* **Bậc 1 (`stair_step_1`)**:
  * Kích thước: `size="0.06 0.40 0.0075"` (sâu $120\text{ mm}$, cao $15\text{ mm}$).
  * Baseline pos: `(1.40, 0.0, 0.0075)` $\rightarrow$ Mặt trên $z_{\text{top}} = +0.015\text{ m}$.
* **Bậc 2 (`stair_step_2`)**:
  * Kích thước: `size="0.06 0.40 0.015"` (sâu $120\text{ mm}$, cao $30\text{ mm}$).
  * Baseline pos: `(1.52, 0.0, 0.015)` $\rightarrow$ Mặt trên $z_{\text{top}} = +0.030\text{ m}$.
* **Bậc 3 (`stair_step_3`)**:
  * Kích thước: `size="0.06 0.40 0.0225"` (sâu $120\text{ mm}$, cao $45\text{ mm}$).
  * Baseline pos: `(1.64, 0.0, 0.0225)` $\rightarrow$ Mặt trên $z_{\text{top}} = +0.045\text{ m}$.
* **Bục đích (`stair_platform`)**:
  * Kích thước: `size="0.16 0.40 0.0225"` (sâu $320\text{ mm}$, cao $45\text{ mm}$).
  * Baseline pos: `(1.86, 0.0, 0.0225)` $\rightarrow$ Mặt trên $z_{\text{top}} = +0.045\text{ m}$.

---

## 3. Lộ Trình Tăng Tiến Độ Khó (Curriculum Architecture)

Cấu trúc `Stage` trong `training/terrain_curriculum.py`:

```python
@dataclass(frozen=True)
class Stage:
    stage_id: int
    pillar_scale: float      # Hệ số co dãn khoảng cách các cọc slalom
    stair_step_h: float      # Độ cao mỗi bậc thang (m)
    description: str
```

### 3.1 Bảng 4 Cấp Độ Huấn Luyện
1. **Stage 0 (Khởi động & Bước cơ bản)**:
   * `pillar_scale = 1.80`: Khe hở cọc rất rộng ($> 0.9\text{ m}$).
   * `stair_step_h = -0.020`: Toàn bộ các bậc thang và bục hạ ngầm dưới sàn ($z_{\text{top}} \le 0.0$).
   * *Mục tiêu*: Robot học đứng vững, sải bước thẳng và làm quen né cọc cự ly lớn.
2. **Stage 1 (Nhập môn Bậc thấp)**:
   * `pillar_scale = 1.35`: Cọc thu hẹp nhẹ.
   * `stair_step_h = 0.005`: Mỗi bậc nhô cao $5\text{ mm}$ (Bậc 1: $5\text{ mm}$, Bậc 2: $10\text{ mm}$, Bậc 3 & Bục: $15\text{ mm}$).
   * *Mục tiêu*: Học nhấc mũi chân nhẹ, không bị vấp mép bậc.
3. **Stage 2 (Thử thách Nâng cao)**:
   * `pillar_scale = 1.10`: Cọc hẹp, đòi hỏi đánh lái zic-zac rõ rệt.
   * `stair_step_h = 0.010`: Mỗi bậc cao $10\text{ mm}$ (Bậc 1: $10\text{ mm}$, Bậc 2: $20\text{ mm}$, Bậc 3 & Bục: $30\text{ mm}$).
   * *Mục tiêu*: Gập gối nhấc chân rõ rệt, duy trì vận tốc khi đổi hướng.
4. **Stage 3 (Chinh phục Toàn diện)**:
   * `pillar_scale = 1.00`: Đúng kích thước chuẩn thiết kế (hành lang bottleneck hẹp).
   * `stair_step_h = 0.015`: Bậc cao chuẩn $15\text{ mm}$ (Bậc 1: $15\text{ mm}$, Bậc 2: $30\text{ mm}$, Bậc 3 & Bục: $45\text{ mm}$).
   * *Mục tiêu*: Vượt chuỗi cọc slalom $\rightarrow$ Leo 3 bậc $\rightarrow$ Đứng ổn định trên bục đích.

### 3.2 Quy Tắc Đánh Giá Thăng Cấp (`next_stage`)
* Điều kiện kích hoạt:
  * `ep_len_mean >= 400.0` (Robot sống sót và đi dài).
  * `rew_lin_vel_ratio >= 0.80` (Bám tốt vận tốc lệnh).
  * Duy trì liên tục trong `hold_iters = 10` lượt kiểm tra liên tiếp.
* Nguyên tắc một chiều: **Chỉ tiến, không lùi** (`stage = max(current_stage, next_stage)`).

---

## 4. Tích Hợp Hàm Thưởng & Vector Quan Sát

### 4.1 Cập Nhật `_compute_d_min()` Trong `run_parallel_sim.py`
Tập hợp danh sách ID hình học chướng ngại vật:
```python
_SLALOM_PILLAR_NAMES = [
    "pillar_l1_left", "pillar_l1_right",
    "pillar_l2_center",
    "pillar_l3_left", "pillar_l3_right",
    "pillar_back"
]
```
Hàm `_compute_d_min()` tính khoảng cách Euclid từ tâm robot đến từng cọc, trừ đi bán kính cọc ($0.06\text{ m} - 0.08\text{ m}$) và kẹp dưới tại $0.0$.

### 4.2 Hàm Thưởng Leo Bậc (`stair_climb_reward`)
* Khi robot tiến vào khu vực cầu thang ($x \ge 1.35\text{ m}$):
  * Thưởng độ cao: $r_{\text{height}} = \text{clip}(z_{\text{base}} - z_{\text{floor}}, 0, 0.05) \times w_{\text{stair}}$.
  * Thưởng tiếp đất thành công trên bậc/bục thay vì rơi xuống sàn.
* Phạt va chạm cạnh bậc (`stumble`): Khi lực cản tác động vào chân theo phương ngang vượt ngưỡng trong lúc chân đang ở pha vung (swing phase).

---

## 5. Danh Sách Tệp Cần Chỉnh Sửa & Kế Hoạch Kiểm Thử

### 5.1 Các tệp chỉnh sửa trực tiếp:
1. `models/mjcf/microduck_8dof.xml`: Thay thế chướng ngại vật cũ bằng cụm 6 cọc slalom + 3 bậc thang + 1 bục đích.
2. `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`: Đồng bộ cụm hình học mới.
3. `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`: Đồng bộ cụm hình học mới.
4. `training/terrain_curriculum.py`: Cập nhật `Stage`, danh sách cọc slalom, logic `apply_stage()` cho cả cọc và bậc thang.
5. `training/run_parallel_sim.py` & các bản copy trong `pipelines/`: Cập nhật danh sách ID geom cọc và tính toán `d_min`.
6. `training/env_config.py` & các bản copy: Bổ sung trọng số và tính toán thưởng leo bậc.

### 5.2 Kế hoạch Kiểm thử TDD (Verification Plan):
* Cập nhật `tests/test_terrain_curriculum.py`:
  * Kiểm tra Stage 0: Tất cả các bậc thang chìm dưới sàn ($z_{\text{top}} \le 0.0$).
  * Kiểm tra Stage 1, 2, 3: Bậc thang tăng dần tương ứng ($5\text{ mm}$, $10\text{ mm}$, $15\text{ mm}$/bậc).
  * Kiểm tra scaling cọc slalom theo đúng tỷ lệ $1.80 \to 1.35 \to 1.10 \to 1.00$.
  * Kiểm tra tính `idempotent` của `apply_stage()`.
* Cập nhật `tests/test_mjcf_model.py`: Xác thực tất cả các geom mới tồn tại, không bị xung đột va chạm nội tại (`exclude`).
* Chạy toàn bộ test suite (`pytest`) bảo đảm 118+ bài test xanh 100%.
