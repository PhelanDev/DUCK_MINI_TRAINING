# MicroDuck 8-DOF: Robot Hai Chân & Kiến Trúc Hai Nhánh Phần Cứng (Sim-to-Real)

Dự án phát triển và chuẩn hóa nền tảng robot hai chân (Bipedal Robot) **MicroDuck 8-DOF**, sử dụng 8 động cơ servo phổ thông **MG996R nhông kim loại**. Hệ thống tích hợp toàn diện mô hình vật lý **MuJoCo MJCF** và **ROS/PyBullet URDF**, cụm cảm biến nhận thức hoàn chỉnh trên đầu (**LiDAR 5 tia ToF + Camera quang học + Microphone I2S MEMS**), đi kèm hai pipeline phần cứng và huấn luyện học tăng cường (Reinforcement Learning - Sim-to-Real) độc lập:
1. **Nhánh Raspberry Pi 5 (`pipelines/rpi5_stack/`):** Tải trọng danh định $\approx 860\text{ g}$, suy luận mô hình Float32 ONNX Runtime trực tiếp trên Linux OS, giao tiếp điều khiển servo qua bus I2C (PCA9685).
2. **Nhánh Dual ESP32-S3 (`pipelines/dual_esp32s3_stack/`):** Tải trọng danh định $\approx 780\text{ g}$, kiến trúc phân tầng 2 vi điều khiển (MCU 1 xử lý Thị giác/Âm thanh/LiDAR $\leftrightarrow$ giao thức UART nhị phân $921,600\text{ bps}$ $\leftrightarrow$ MCU 2 chạy RL Locomotion Policy $50\text{ Hz}$ tối ưu lượng tử hóa INT8).

---

## 1. Tổng Quan Kiến Trúc Robot & Triết Lý Thiết Kế

Phiên bản gốc của MicroDuck sử dụng 15 servo thông minh Dynamixel XL330 đắt đỏ và cấu trúc điều khiển phức tạp. Nhằm tối ưu hóa chi phí gia công chế tạo, độ bền cơ khí chống va đập khi té ngã và tính sẵn có của linh kiện tại Việt Nam, robot được thiết kế lại toàn diện về cấu hình **8 servo MG996R nhông kim loại**:

- **Cấu hình bậc tự do (8 DOF):**
  - **Cụm đầu (2 DOF):** Khớp quay cổ `head_yaw` ($\pm 90^\circ$) và gật gù `head_pitch` ($-30^\circ \to +45^\circ$) hỗ trợ quét cảm biến không gian.
  - **Cụm chân (6 DOF, 3 DOF mỗi chân):** Khớp lắc hông `hip_roll` ($\pm 25^\circ$), khớp nâng đùi `hip_pitch` ($\pm 50^\circ$), khớp gập gối `knee_pitch` ($0^\circ \to +85^\circ$).
- **Chiều cao tổng thể:** $\sim 230\text{ mm}$ (Độ cao tâm trọng trường CoM khi đứng thẳng: $\approx 0.170\text{ m}$).
- **Trọng tâm (Center of Mass - CoM):** Bố trí dồn về nửa dưới của thân chính (`base_link`), hạ thấp trọng tâm và tăng tối đa độ ổn định chống lật khi sải bước.

---

## 2. Cây Động Học & Kích Thước Các Link (Kinematic Tree)

Cây động học chuẩn hóa của robot được mô tả dưới dạng cây phân cấp:

```text
                  [base_link] (Thân chính, Pin, Mạch điều khiển, IMU)
                 /           \                              \
        [neck_link]       [left_hip_roll_link]    [right_hip_roll_link]
             |                     |                        |
        [head_link]        [left_thigh_link]        [right_thigh_link]
                                   |                        |
                           [left_shin_link]         [right_shin_link]
                                   |                        |
                           [left_foot_link]         [right_foot_link]
```

### 2.1. Bảng Thông Số Kích Thước & Khối Lượng Từng Link

| Link Name | Khối lượng (Dual ESP32 / Pi 5) | Kích thước ($X \times Y \times Z$) | Chức năng cơ học |
| :--- | :---: | :---: | :--- |
| `base_link` | $310\text{ g}~/~390\text{ g}$ | $110 \times 80 \times 70\text{ mm}$ | Thân chính, chứa pin, bo mạch điều khiển và IMU |
| `neck_link` | $30\text{ g}~/~30\text{ g}$ | $35 \times 35 \times 30\text{ mm}$ | Khối quay cổ theo trục thẳng đứng ($Z$) |
| `head_link` | $60\text{ g}~/~60\text{ g}$ | $60 \times 50 \times 45\text{ mm}$ | Đầu vịt mang camera, mic và cụm 5 tia ToF LiDAR |
| `left_hip_roll_link` | $50\text{ g}~/~50\text{ g}$ | $45 \times 40 \times 40\text{ mm}$ | Cụm xoay hông lắc nghiêng chuyển trọng tâm ($X$) |
| `left_thigh_link` | $65\text{ g}~/~65\text{ g}$ | $35 \times 30 \times 65\text{ mm}$ | Đùi ($L_1 = 65\text{ mm}$), chứa servo gối |
| `left_shin_link` | $45\text{ g}~/~45\text{ g}$ | $30 \times 25 \times 65\text{ mm}$ | Cẳng chân ($L_2 = 65\text{ mm}$) |
| `left_foot_link` | $30\text{ g}~/~30\text{ g}$ | $75 \times 55 \times 10\text{ mm}$ | Bàn chân bản to, mặt đế cao su ma sát cao |
| `right_*_link` | Đối xứng | Đối xứng qua mặt phẳng $X$ | Đối xứng chân phải |
| **Tổng khối lượng** | $\approx \mathbf{780\text{ g}}~/~\mathbf{860\text{ g}}$ | Chiều cao: $\sim 230\text{ mm}$ | Đạt chuẩn thiết kế cho từng nhánh phần cứng |

### 2.2. Bảng Giới Hạn Khớp & Đặc Tính Động Cơ MG996R

Tất cả các khớp đều sử dụng trục quay chuẩn revolute với giới hạn vật lý an toàn:

| Tên khớp | Trục quay | Giới hạn góc (Radian) | Giới hạn góc (Độ) | Mô-men $\tau_{\max}$ | Vận tốc $\omega_{\max}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `head_yaw` | $Z$ | $[-1.57, +1.57]$ | $\pm 90^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `head_pitch` | $Y$ | $[-0.52, +0.78]$ | $-30^\circ \to +45^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `left_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `left_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `left_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `right_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `right_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |
| `right_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | $1.0\text{ N}\cdot\text{m}$ | $6.0\text{ rad/s}$ |

---

## 3. Cụm Cảm Biến Nhận Thức Trên Đầu Vịt (Perception Suite)

Cả hai mô hình robot (`microduck_rpi5_8dof` và `microduck_esp32_8dof`) đều được tích hợp cụm cảm biến nhận thức hoàn chỉnh trên đầu (`head_link`):

```text
               ┌────────────────────────────────────────────────────────┐
               │         ĐẦU VỊT NHẬN THỨC (PERCEPTION HEAD)             │
               │                                                        │
               │             [Camera 60° FOV] (cam_lens)                │
               │                                                        │
               │   (-30°)      (-15°)     (0°)     (+15°)     (+30°)    │
               │     \           \         |         /          /       │
               │      \           \        |        /          /        │
               │       ▼           ▼       ▼       ▼          ▼         │
               │     [L1]        [L2]     [L3]    [L4]       [L5]       │
               │     LiDAR 5 Tia Quét ToF (Tầm đo 0.05m - 2.0m)         │
               │                                                        │
               │             [Mic I2S MEMS] (INMP441 Site)              │
               └────────────────────────────────────────────────────────┘
```

1. **Cụm LiDAR 5 tia ToF (5-Ray Rangefinder Scanner):**
   - 5 cảm biến quang học `<rangefinder>` trong MuJoCo bố trí góc quạt ngang $\theta \in [-30^\circ, -15^\circ, 0^\circ, +15^\circ, +30^\circ]$.
   - Cự ly đo khả dụng: $0.05\text{ m} \to 2.0\text{ m}$.
   - Cung cấp trực tiếp 5 giá trị khoảng cách tới chướng ngại vật vào vector quan sát (Observation Space) để chính sách RL tự động phát hiện và né vật cản thời gian thực.
2. **Camera quang học góc nhìn rộng (RGB Optical Camera):**
   - Camera góc mở $60^\circ$ FOV gắn trên trán vịt với hình học ống kính kim loại (`cam_lens`), chuẩn hóa trục quang học nhìn thẳng phía trước ($+X$).
   - Nhánh Pi 5 kết nối qua cáp MIPI-CSI (Raspberry Pi Camera Module v2/v3). Nhánh ESP32 kết nối camera DVP OV2640 qua giao diện phần cứng của ESP32-S3.
3. **Microphone số I2S MEMS (Digital Audio Sensor):**
   - Cổng thu âm thanh số (mô phỏng module INMP441) kết nối chuẩn I2S 24-bit/48kHz phục vụ tác vụ nhận diện từ khóa giọng nói ("Hey Duck") và định vị nguồn âm thanh.

---

## 4. Hai Nhánh Phần Cứng Độc Lập (Two Independent Pipelines)

Để đáp ứng các mục tiêu nghiên cứu và ứng dụng thực tế khác nhau trong đồ án tốt nghiệp, hệ thống được cấu trúc thành 2 pipeline độc lập:

### 4.1. Bảng So Sánh Kỹ Thuật Tổng Hợp

| Tiêu Chí | Nhánh 1: `pipelines/rpi5_stack/` | Nhánh 2: `pipelines/dual_esp32s3_stack/` |
| :--- | :--- | :--- |
| **Vi xử lý trung tâm** | Raspberry Pi 5 4GB (Quad Cortex-A76 @ 2.4GHz) | 2x ESP32-S3 DevKit N16R8 (Dual Xtensa LX7 @ 240MHz) |
| **Hệ điều hành / Firmware** | Linux (Raspberry Pi OS 64-bit Lite, Python 3.10+) | Bare-metal C++ (Arduino Core / ESP-IDF Framework) |
| **Tổng khối lượng robot** | $\approx \mathbf{860\text{ g}}$ (Trọng lượng Pi 5 + Quạt tản nhiệt + Buck 5A) | $\approx \mathbf{780\text{ g}}$ (2 bo mạch nhẹ, bố trí thấp, pin 2S đặt đáy) |
| **Độ cao trọng tâm (CoM)** | CoM cao hơn $\approx 10\text{ mm}$ do khối Pi 5 ở thân trên | CoM thấp sát đáy thân $\rightarrow$ Tăng độ thăng bằng bipedal |
| **Định dạng mạng nơ-ron** | Float32 ONNX Model (chạy ONNX Runtime CPU) | INT8 Quantized Model (tăng tốc bằng tập lệnh vector PIE) |
| **Chu kỳ điều khiển RL** | $50\text{ Hz}$ ($20\text{ ms}$/bước, suy luận $< 4\text{ ms}$) | $50\text{ Hz}$ ($20\text{ ms}$/bước, suy luận $< 3\text{ ms}$) |
| **Không gian quan sát** | **42 chiều** (37 động học + 5 khoảng cách LiDAR) | **42 chiều** (37 động học + 5 khoảng cách LiDAR) |
| **Giao thức điều khiển** | I2C Bus 400kHz tới PCA9685 điều khiển 8 MG996R | Phân tầng: UART Packet $921,600\text{ bps}$ + 8 kênh PWM |
| **Nhu cầu nguồn điện** | Pin 2S $\rightarrow$ Buck 5V/5A (Pi 5) + Buck 6V/15A (Servo) | Pin 2S $\rightarrow$ Buck 5V/3A (Logic) + Buck 6V/15A (Servo) |
| **Tài liệu đặc tả chi tiết** | [`pipelines/rpi5_stack/docs/rpi5_hardware_spec.md`](file:///d:/duckV1/pipelines/rpi5_stack/docs/rpi5_hardware_spec.md) | [`pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md) |

---

### 4.2. Chi Tiết Nhánh 1: Raspberry Pi 5 Stack (`pipelines/rpi5_stack/`)

- **Kiến trúc:** Máy tính đơn bo (SBC) hiệu năng cao chạy trực tiếp hệ điều hành Linux.
- **Mô hình vật lý:** `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml` (tổng khối lượng $0.860\text{ kg}$, vị trí pin và SBC nâng nhẹ CoM).
- **Môi trường huấn luyện:** `pipelines/rpi5_stack/training/env_config.py` và `run_parallel_sim.py` (huấn luyện weights Float32 với độ chính xác cao).
- **Runtime triển khai:** `pipelines/rpi5_stack/deployment/infer_rpi5.py` nạp file ONNX, đọc dữ liệu IMU/LiDAR, tính toán mạng nơ-ron và gửi xung qua PCA9685.
- **Phân phối nguồn điện:**
  - Khối nguồn chính: Pin LiPo/Li-ion 2S ($7.4\text{V} - 8.4\text{V}$, dung lượng $2200\text{mAh}$, dòng xả $25\text{C}$).
  - Module Buck số 1: $5.1\text{V} / 5.0\text{A}$ ($25\text{W}$) cấp riêng cho bo mạch Raspberry Pi 5.
  - Module Buck số 2: $6.0\text{V} / 15.0\text{A}$ đỉnh cấp riêng cho 8 động cơ MG996R qua cọc nguồn PCA9685 (kèm tụ đệm $4700\mu\text{F} / 16\text{V}$).
  - Star Ground: Nối chung Mass tại một điểm duy nhất để triệt tiêu dòng lặp (ground loop).

---

### 4.3. Chi Tiết Nhánh 2: Dual ESP32-S3 Stack (`pipelines/dual_esp32s3_stack/`)

- **Kiến trúc:** Phân tầng 2 vi điều khiển ESP32-S3 DevKit N16R8 (Xtensa Dual-core LX7 @ 240MHz):
  - **MCU 1 (Perception & Command Brain):** Chịu trách nhiệm chụp ảnh từ Camera OV2640, nhận diện giọng nói qua microphone I2S MEMS, quét 5 cảm biến khoảng cách ToF VL53L1X qua I2C. Đóng gói dữ liệu nhận thức và gửi sang MCU 2.
  - **MCU 2 (Locomotion RL Brain):** Đọc cảm biến con quay hồi chuyển IMU (MPU6050/BNO085), nhận dữ liệu nhận thức từ MCU 1 qua UART, ghép vector quan sát 42 chiều, chạy mô hình nơ-ron định dạng INT8 (thời gian tính toán $\approx 2.8\text{ ms}$) và xuất 8 kênh xung PWM trực tiếp tới servo MG996R theo chu kỳ nghiêm ngặt $50\text{ Hz}$ ($20\text{ ms}$).
- **Giao thức UART nhị phân tốc độ cao (`uart_packet_protocol.h`):**
  - Baud rate: **$921,600\text{ bps}$** (thời gian truyền gói chỉ $\sim 300\mu\text{s}$, chiếm $< 1.7\%$ chu kỳ $20\text{ ms}$).
  - Cấu trúc khung truyền: `[0xAA] [0x55] [Type] [SeqID] [Len_L] [Len_H] [Payload...] [CRC16_L] [CRC16_H]`.
  - Cơ chế kiểm lỗi: Mã kiểm tra dư thừa tuần hoàn CRC16-CCITT đảm bảo không có xung đột hay nhiễu byte trên đường truyền.
- **Mã nguồn firmware:**
  - MCU 1 Sketch: `pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino`.
  - MCU 2 Sketch: `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`.

---

## 5. Không Gian Quan Sát 42 Chiều & Huấn Luyện Học Tăng Cường (Perception RL)

Môi trường điều khiển được thiết kế theo chuẩn học tăng cường Sim-to-Real ở chu kỳ điều khiển $50\text{ Hz}$ ($dt = 0.02\text{ s}$).

### 5.1. Cấu Trúc Vector Quan Sát 42 Chiều

$$\mathbf{o}_t \in \mathbb{R}^{42}$$

| Chỉ số Index | Tên thành phần | Số chiều | Nguồn dữ liệu & Ý nghĩa vật lý |
| :---: | :--- | :---: | :--- |
| `0 : 3` | Vận tốc góc thân ($\boldsymbol{\omega}_{\text{base}}$) | 3 | Con quay hồi chuyển (Gyroscope) $\omega_x, \omega_y, \omega_z$ (rad/s) |
| `3 : 6` | Véc-tơ trọng trường quy chiếu ($\mathbf{g}_{\text{proj}}$) | 3 | Hình chiếu trọng trường lên hệ thân: $\mathbf{R}^T [0, 0, -1]^T$ |
| `6 : 9` | Lệnh vận tốc mong muốn ($\mathbf{v}_{\text{cmd}}$) | 3 | Lệnh điều khiển người dùng: $[v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}}]$ |
| `9 : 17` | Góc hiện tại của 8 khớp ($\mathbf{q}_{\text{joint}}$) | 8 | Đọc từ phản hồi vị trí khớp (rad) |
| `17 : 25` | Vận tốc hiện tại của 8 khớp ($\dot{\mathbf{q}}_{\text{joint}}$) | 8 | Đạo hàm vận tốc góc của 8 khớp (rad/s) |
| `25 : 33` | Hành động ở bước liền trước ($\mathbf{a}_{t-1}$) | 8 | Giá trị lệnh hành động $a_{t-1}$ (đảm bảo độ mượt mà quỹ đạo) |
| `33 : 37` | Cảm biến tiếp xúc bàn chân ($\mathbf{c}_{\text{feet}}$) | 4 | Tín hiệu chạm sàn 4 điểm (mũi/gót bàn chân trái và phải) |
| `37 : 42` | Khoảng cách 5 tia LiDAR ($\mathbf{d}_{\text{lidar}}$) | **5** | Khoảng cách quét ngang 5 góc $[-30^\circ, -15^\circ, 0^\circ, +15^\circ, +30^\circ]$ |

### 5.2. Không Gian Hành Động (Action Space - 8 Chiều)

Mạng Actor dự đoán hành động chuẩn hóa $\mathbf{a}_t \in [-1, +1]^8$. Module `env_config.py` thực hiện:
1. **Ánh xạ dải góc (Action Scaling):** Ánh xạ từ $[-1, +1]$ sang góc radian vật lý của từng khớp dựa trên `JOINT_LIMITS`.
2. **Kẹp biến thiên góc (Rate Limiting):** Giới hạn độ dịch chuyển góc tối đa $\Delta q \le \omega_{\max} \cdot \Delta t = 6.0 \times 0.02 = 0.12\text{ rad/bước}$ để bảo vệ nhông hộp số MG996R, loại bỏ hiện tượng giật rung cơ khí.

### 5.3. Hàm Thưởng Huấn Luyện Đa Thành Phần (Reward Shaping)

$$R_{\text{total}} = w_v R_{\text{lin\_vel}} + w_\omega R_{\text{ang\_vel}} + w_u R_{\text{upright}} + w_c R_{\text{contact}} + w_{\text{obs}} R_{\text{obstacle\_avoidance}} - w_\tau P_{\text{torque}} - w_{\Delta a} P_{\text{action\_rate}}$$

- **Bám vận tốc dài trong Body Frame ($R_{\text{lin\_vel}}$):** Vận tốc thế giới $\mathbf{v}_{\text{world}}$ bắt buộc phải được chuyển đổi về hệ thân qua phép quay nghịch đảo quaternion $q^{-1}$ (`quat_rotate_inverse`) trước khi so khớp với lệnh $v_{xy}^{\text{cmd}}$:
  $$R_{\text{lin\_vel}} = \exp\left(-\frac{\|v_{xy}^{\text{body}} - v_{xy}^{\text{cmd}}\|^2}{0.25}\right)$$
- **Né chướng ngại vật ($R_{\text{obstacle\_avoidance}}$):** Khi khoảng cách đo được từ bất kỳ tia LiDAR nào nhỏ hơn khoảng cách an toàn ($d_{\min} < 0.35\text{ m}$), thuật toán phạt nặng hành vi tiếp tục tiến tới và thưởng khi robot quay đầu hoặc lùi xa vật cản:
  $$R_{\text{obstacle\_avoidance}} = \sum_{i=1}^{5} \min(0.0, \ln(d_i / 0.35))$$
- **Giữ thân thẳng đứng ($R_{\text{upright}}$):** Giữ thành phần trọng trường chiếu $g_z \to -1$.

### 5.4. Cảnh Báo Phần Cứng Thực Tế: Sim-to-Real Với Servo RC PWM MG996R

> [!WARNING]
> **Khoảng cách Sim-to-Real giữa mô phỏng và đời thực đối với Servo RC PWM:**
> Trong không gian quan sát 42 chiều, có tới **16 chiều** đại diện cho trạng thái nội tại (`joint_pos` 8 chiều và `joint_vel` 8 chiều).
>
> 1. **Vấn đề "Mù" Trạng Thái Khớp trên Servo RC PWM (MG996R + PCA9685):**
>    - Servo RC PWM truyền thống chỉ nhận tín hiệu xung 1 chiều từ PCA9685 ($500\mu\text{s} - 2500\mu\text{s}$) và **hoàn toàn không có đường truyền trả lời về góc quay thực tế**.
>    - Nếu đưa policy huấn luyện từ mô phỏng ra robot thật sử dụng MG996R mà không có encoder phản hồi, việc nhét góc đặt cũ $a_{t-1}$ vào làm `joint_pos` và tính sai phân làm `joint_vel` sẽ làm mạng RL bị **mù trạng thái tải thực tế**. Khi một chân bị cản trở hoặc sụp tải trọng, robot không nhận biết được độ lệch góc và sẽ ngã sấp mặt ngay lập tức.
> 2. **Hai Phương Án Xử Lý Cho Đồ Án & Thực Tế:**
>    - **Phương án A (Độ chiết áp MG996R - Tiết kiệm chi phí):** Tháo nắp đáy servo MG996R, hàn câu 1 sợi dây từ chân giữa của chiết áp (Wiper) đưa về kênh ADC của vi điều khiển (ESP32-S3 hoặc chip ADC 16-bit ADS1115 qua I2C) để đo trực tiếp điện áp tỷ lệ với góc khớp.
>    - **Phương án B (Khuyến nghị chuẩn công nghiệp - Smart Serial Bus Servo):** Thay thế cụm MG996R + PCA9685 bằng dòng Smart Bus Servo có encoder từ tính 12-bit (như **Feetech SCS0009** $2.3\text{ kg}\cdot\text{cm}$ cho bản nhẹ, hoặc **Feetech STS3215 / Waveshare ST3215** $19\text{ kg}\cdot\text{cm}$ cho bản chuẩn) giao tiếp UART Half-Duplex $1\text{ Mbps}$. Giao thức này cho phép đọc đồng thời cả vị trí góc thực tế, tốc độ, dòng tải và nhiệt độ của 8 khớp trong thời gian $< 2\text{ ms}$, đảm bảo khớp hoàn toàn với 16 chiều proprioceptive state trong mạng RL.

---

## 6. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
d:\duckV1\
├── models/                                         # Mô hình cơ sở (Base Models)
│   ├── mjcf/microduck_8dof.xml                     # Mô hình MuJoCo MJCF
│   └── urdf/microduck_8dof.urdf                    # Mô hình URDF đồng bộ
├── pipelines/                                      # Hai nhánh phần cứng độc lập
│   ├── rpi5_stack/                                 # Nhánh Raspberry Pi 5 (860g)
│   │   ├── models/
│   │   │   ├── mjcf/microduck_rpi5_8dof.xml        # MJCF 5-ray LiDAR, Camera, Mic
│   │   │   └── urdf/microduck_rpi5_8dof.urdf       # URDF đồng bộ Pi 5
│   │   ├── training/
│   │   │   ├── env_config.py                       # Cấu hình Obs 42 chiều, Float32 weights
│   │   │   └── run_parallel_sim.py                 # Runner mô phỏng song song Pi 5
│   │   ├── deployment/
│   │   │   └── infer_rpi5.py                       # Runtime điều khiển Pi 5 qua PCA9685
│   │   └── docs/
│   │       └── rpi5_hardware_spec.md               # Sơ đồ đấu nối, nguồn 5V/5A, CSI/I2S
│   │
│   └── dual_esp32s3_stack/                         # Nhánh Dual ESP32-S3 (780g)
│       ├── models/
│       │   ├── mjcf/microduck_esp32_8dof.xml       # MJCF 5-ray LiDAR, Camera, Mic, CoM thấp
│       │   └── urdf/microduck_esp32_8dof.urdf      # URDF đồng bộ Dual ESP32
│       ├── training/
│       │   ├── env_config.py                       # Cấu hình Obs 42 chiều, INT8 scaling
│       │   └── run_parallel_sim.py                 # Runner mô phỏng song song ESP32
│       ├── deployment/
│       │   ├── uart_packet_protocol.h              # Giao thức nhị phân liên MCU (CRC16)
│       │   ├── mcu1_vision_audio_s3/               # Firmware MCU 1 (Vision/Audio/LiDAR)
│       │   │   └── mcu1_main.ino
│       │   └── mcu2_locomotion_s3/                 # Firmware MCU 2 (Locomotion RL 50Hz)
│       │       └── mcu2_main.ino
│       └── docs/
│           └── dual_esp32s3_hardware_spec.md       # Sơ đồ mạch, phân tách nguồn, UART
├── training/                                       # Cấu hình huấn luyện cơ bản (Base 37-dim)
│   ├── env_config.py
│   └── run_parallel_sim.py
├── scripts/                                        # Tiện ích trực quan và kiểm tra
│   ├── verify_model.py                             # Script kiểm tra chéo vật lý tự động
│   └── interactive_viewer.py                       # Trực quan hóa 3D tương tác MuJoCo
├── tests/                                          # Toàn bộ test suite tự động (PyTest)
└── README.md                                       # Sách hướng dẫn kỹ thuật toàn diện
```

---

## 7. Hướng Dẫn Cài Đặt Môi Trường (Installation)

Yêu cầu môi trường **Python 3.10+**. Khuyến nghị sử dụng môi trường ảo:

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Trên Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Trên Linux / macOS:
source venv/bin/activate

# Cài đặt các gói phụ thuộc
pip install mujoco numpy pytest onnx onnxruntime
```

---

## 8. Kiểm Tra Tính Toàn Vẹn Vật Lý (Model Verification)

Trước khi tiến hành trực quan hoặc huấn luyện, hãy chạy script kiểm tra chéo tự động:

```bash
# 1. Kiểm tra mô hình cơ sở
python scripts/verify_model.py

# 2. Kiểm tra mô hình nhánh Raspberry Pi 5 (~860g)
python scripts/verify_model.py --mjcf pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml --urdf pipelines/rpi5_stack/models/urdf/microduck_rpi5_8dof.urdf

# 3. Kiểm tra mô hình nhánh Dual ESP32-S3 (~780g)
python scripts/verify_model.py --mjcf pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml --urdf pipelines/dual_esp32s3_stack/models/urdf/microduck_esp32_8dof.urdf
```

### Các tiêu chuẩn thẩm định của `verify_model.py`:
1. **Động học khớp:** Khớp nối 8 revolute joints (`head_yaw`, `head_pitch`, `left_hip_roll`, `left_hip_pitch`, `left_knee_pitch`, `right_hip_roll`, `right_hip_pitch`, `right_knee_pitch`) khớp 100% giữa MJCF và URDF về trục quay, dải góc và mô-men xoắn.
2. **Khối lượng & Quán tính:** Tổng khối lượng đạt chuẩn danh định trong biên độ $\pm 10\%$, tất cả tensor quán tính đường chéo $I_{xx}, I_{yy}, I_{zz} > 0$ và thỏa mãn bất đẳng thức tam giác vật lý.
3. **Vị trí trọng tâm CoM:** Nằm ở nửa dưới thân chính, đối xứng hai bên.
4. **Mô phỏng động học 100 bước:** Ổn định dưới gia tốc trọng trường $-9.81\text{ m/s}^2$ không có hiện tượng sụp đổ hay nổ số thực (NaN explosion).

---

## 9. Trực Quan Hóa 3D & Kiểm Thử Tư Thế (Interactive 3D Viewer)

Sử dụng `scripts/interactive_viewer.py` để trực quan hóa mô hình trong không gian 3D, kiểm tra độ vững và mô phỏng bước đi:

```bash
# Mở cửa sổ trực quan hóa 3D tư thế đứng mặc định
python scripts/interactive_viewer.py

# Xem chu kỳ đi bộ mô phỏng (Dynamic Gait Demo)
python scripts/interactive_viewer.py --gait-demo

# Khởi tạo với các tư thế mẫu: stand, squat, left_step, right_step, bow
python scripts/interactive_viewer.py --pose squat
python scripts/interactive_viewer.py --pose left_step

# Chế độ tương tác dòng lệnh tinh chỉnh góc từng khớp bằng bàn phím
python scripts/interactive_viewer.py --interactive

# Chế độ không màn hình (Headless Mode dành cho CI/CD và Server từ xa)
python scripts/interactive_viewer.py --headless --duration 3.0 --gait-demo
```

---

## 10. Mô Phỏng Song Song & Đo Hiệu Năng FPS (Parallel Sim Benchmark)

Hệ thống cung cấp module mô phỏng song song đa luồng theo lô (Batched Vectorized Simulator) tối ưu hóa tối đa CPU:

```bash
# 1. Chạy benchmark nhánh Raspberry Pi 5 (64 môi trường)
python pipelines/rpi5_stack/training/run_parallel_sim.py --num_envs 64 --steps 200

# 2. Chạy benchmark nhánh Dual ESP32-S3 (64 môi trường)
python pipelines/dual_esp32s3_stack/training/run_parallel_sim.py --num_envs 64 --steps 200

# 3. Chạy benchmark nhánh cơ sở (128 môi trường, 4 luồng CPU)
python training/run_parallel_sim.py --num_envs 128 --steps 100 --threads 4
```

### Kết quả đo kiểm thực tế trên CPU:
- **Tốc độ mô phỏng (FPS):** Đạt từ **15,000 đến 35,000+ Steps/Giây** trên CPU phổ thông.
- **Tỷ lệ tăng tốc thời gian thực (Real-time Factor):** Nhanh hơn thời gian thực từ **300x đến 700x**, cho phép hoàn thành 10 triệu bước tương tác RL chỉ trong vài phút.

---

## 11. Hướng Dẫn Nạp & Triển Khai Trên Phần Cứng Thật (Deployment Guide)

### 11.1. Triển Khai Trên Nhánh Raspberry Pi 5

1. **Chuẩn bị môi trường trên Raspberry Pi OS (64-bit Lite):**
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-smbus i2c-tools
   pip3 install numpy onnxruntime
   ```
2. **Kích hoạt I2C trên Pi 5:**
   ```bash
   sudo raspi-config
   # Chọn Interface Options -> I2C -> Enable -> Reboot
   sudo i2cdetect -y 1  # Kiểm tra thấy địa chỉ 0x40 của PCA9685
   ```
3. **Khởi chạy chương trình điều khiển thời gian thực:**
   ```bash
   # Chạy chế độ mô phỏng kiểm thử an toàn (Mock PCA9685)
   python pipelines/rpi5_stack/deployment/infer_rpi5.py --mock-pca --rate 50

   # Chạy trên phần cứng thật với file model ONNX Float32
   python pipelines/rpi5_stack/deployment/infer_rpi5.py --model models/microduck_rpi5_policy.onnx --rate 50
   ```
   *Chi tiết sơ đồ chân GPIO và phân phối nguồn 5V/5A + 6V/15A xem tại:* [`pipelines/rpi5_stack/docs/rpi5_hardware_spec.md`](file:///d:/duckV1/pipelines/rpi5_stack/docs/rpi5_hardware_spec.md).

---

### 11.2. Triển Khai Trên Nhánh Dual ESP32-S3

1. **Cài đặt công cụ lập trình:**
   - Cài đặt **Arduino IDE 2.x** hoặc **ESP-IDF v5.x**.
   - Cài đặt gói bo mạch `esp32` by Espressif Systems (phiên bản $\ge 2.0.14$).
   - Chọn Board: `ESP32S3 Dev Module`, Flash Size: `16MB`, PSRAM: `OPI PSRAM`.
2. **Nối dây liên kết giữa 2 bo ESP32-S3:**
   - Nối chéo UART2:
     - `MCU 1 GPIO 17 (TX)` $\rightarrow$ `MCU 2 GPIO 18 (RX)`
     - `MCU 1 GPIO 18 (RX)` $\leftarrow$ `MCU 2 GPIO 17 (TX)`
     - Nối chung đường mass `GND` giữa 2 bo mạch.
3. **Nạp Firmware:**
   - **Nạp MCU 1:** Mở sketch [`pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino) và nhấn **Upload**.
   - **Nạp MCU 2:** Mở sketch [`pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino) và nhấn **Upload**.
   - Header giao thức [`pipelines/dual_esp32s3_stack/deployment/uart_packet_protocol.h`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/deployment/uart_packet_protocol.h) tự động đồng bộ cấu trúc gói tin có CRC16 giữa 2 vi điều khiển.
4. **Lượng tử hóa mô hình INT8:**
   - Dùng script chuyển đổi mạng nơ-ron Actor sang dạng bảng mảng hằng số `const int8_t policy_weights[]` đặt trực tiếp trong bộ nhớ Flash của MCU 2, kích hoạt tập lệnh SIMD/PIE của Xtensa LX7 để đạt tốc độ suy luận $< 3\text{ ms}$.
   *Chi tiết sơ đồ chân, mạch chia kênh và nguồn phân tầng xem tại:* [`pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md).

---

## 12. Chạy Toàn Bộ Test Suite Tự Động (Project Test Suite)

Dự án sở hữu bộ kiểm thử tự động toàn diện thẩm định toàn bộ mô hình vật lý, cấu hình huấn luyện và mã nguồn triển khai:

```bash
pytest -v
```

### Các hạng mục kiểm thử tự động gồm:
- **`test_rpi5_model.py`:** Thẩm định mô hình MJCF và URDF nhánh Pi 5 (ngân sách khối lượng $860\text{ g}$, 5 tia LiDAR, thấu kính camera, mic site).
- **`test_esp32_model.py`:** Thẩm định mô hình MJCF và URDF nhánh ESP32 (ngân sách khối lượng $780\text{ g}$, CoM thấp, 5 tia LiDAR, camera, mic).
- **`test_pipeline_training.py`:** Thẩm định vector quan sát 42 chiều, hàm thưởng né vật cản `obstacle_avoidance`, các hàm lượng tử hóa INT8 và runner song song của cả 2 nhánh.
- **`test_deployment_artifacts.py`:** Thẩm định script Python `infer_rpi5.py`, biên dịch cú pháp C/C++ giao thức nhị phân `uart_packet_protocol.h`, kiểm tra sketch Arduino MCU 1 & MCU 2 và 2 bộ tài liệu đặc tả phần cứng.
- **`test_mjcf_model.py` & `test_urdf_model.py`:** Thẩm định mô hình cơ sở.
- **`test_parallel_runner.py` & `test_env_config.py`:** Thẩm định cấu hình môi trường gốc.
- **`test_verify_script.py` & `test_interactive_viewer.py`:** Thẩm định script kiểm tra chéo và trình trực quan 3D.
