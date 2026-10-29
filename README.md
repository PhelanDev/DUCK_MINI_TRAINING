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

### 3.1. Bốn Cột Chướng Ngại Vật Trong Môi Trường Mô Phỏng (`<worldbody>`)

Cả 3 mô hình MuJoCo MJCF (`microduck_8dof.xml`, `microduck_rpi5_8dof.xml`, `microduck_esp32_8dof.xml`) được bổ sung 4 cột chướng ngại vật hình trụ (`obstacle_front`, `obstacle_left`, `obstacle_right`, `obstacle_back`) với kích thước bán kính $0.08\text{ m}$, chiều cao $0.50\text{ m}$ (`size="0.08 0.25"`), bật cờ tiếp xúc `contype="1" conaffinity="1"` đặt trực tiếp trong `<worldbody>`:
- `obstacle_front`: Tọa độ $[+0.80, 0.00, 0.25]\text{ m}$ (ngay phía trước hướng di chuyển chính).
- `obstacle_left`: Tọa độ $[+0.50, +0.60, 0.25]\text{ m}$ (chếch phía trước bên trái).
- `obstacle_right`: Tọa độ $[+0.50, -0.60, 0.25]\text{ m}$ (chếch phía trước bên phải).
- `obstacle_back`: Tọa độ $[-0.60, 0.00, 0.25]\text{ m}$ (phía sau robot).

Các cột vật cản này cung cấp mục tiêu phản xạ chùm tia ToF LiDAR thực tế, tạo môi trường đa dạng cho bài toán né tránh chướng ngại vật trong quá trình huấn luyện RL.

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
- **Né chướng ngại vật ($R_{\text{obstacle\_avoidance}}$):** Khi khoảng cách đo được từ bất kỳ tia LiDAR nào nhỏ hơn cự ly an toàn ($d_{\min} < d_{\text{safe}} = 0.35\text{ m}$), thuật toán phạt tuyến tính theo mức độ vi phạm để robot tránh va chạm. Khi đường đi thông thoáng ($d_{\min} \ge 0.35\text{ m}$), giá trị thưởng bằng $0.0$ (loại bỏ hoàn toàn lỗ hổng Reward Hacking khiến robot đứng yên nhìn vào khoảng trống):
  $$R_{\text{obstacle\_avoidance}} = \min(0.0, d_{\min} - 0.35)$$
- **Thưởng dáng đi nhấc chân luân phiên ($R_{\text{contact}}$):** Khuyến khích chu kỳ bước đi hai chân tự nhiên (Gait Cycle), phân biệt rõ ràng giữa trạng thái di chuyển và đứng yên:
  - Khi có lệnh di chuyển ($\|\mathbf{v}_{xy}^{\text{cmd}}\| > 0.05\text{ m/s}$): Robot hai chân phải luân phiên nhấc một chân (Swing Phase) trong khi chân còn lại bám trụ (Stance Phase). Thưởng tối đa $+1.0$ khi **chính xác 1 chân tiếp xúc sàn** (`num_contacts == 1`). Nhận thưởng $0.0$ nếu kéo lê cả 2 chân trên sàn (`num_contacts == 2`) hoặc nhảy bổng cả hai chân lên không (`num_contacts == 0`).
  - Khi đứng yên ($\|\mathbf{v}_{xy}^{\text{cmd}}\| \le 0.05\text{ m/s}$): Thưởng $+0.5$ khi **cả 2 chân cùng tiếp đất** (`num_contacts == 2`) để duy trì thế đứng thăng bằng ổn định; nhận $0.0$ nếu đứng co một chân.
- **Giữ thân thẳng đứng ($R_{\text{upright}}$):** Giữ thành phần trọng trường chiếu $g_z \to -1$.
- **Phạt rung giật hành động ($P_{\text{action\_rate}}$):** Phạt bình phương sai phân hành động liên tiếp $\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ để triệt tiêu hiện tượng giật cơ khí và bảo vệ bánh răng hộp số MG996R.

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

# Quan sát robot bước đi trực tiếp bằng mạng nơ-ron AI đã huấn luyện (RL Policy Mode)
# 1. Nạp checkpoint PyTorch (.pt) vừa huấn luyện:
python scripts/interactive_viewer.py --policy models/checkpoints/microduck_ppo.pt

# 2. Ra lệnh vận tốc di chuyển (Tiến 0.2 m/s, quay trái 0.1 rad/s):
python scripts/interactive_viewer.py --policy models/checkpoints/microduck_ppo.pt --cmd-vel 0.2 0.0 0.1

# 3. Nạp mô hình ONNX tối ưu (dành cho Raspberry Pi 5):
python scripts/interactive_viewer.py --policy models/microduck_rpi5_policy.onnx --cmd-vel 0.25 0.0 0.0
```

### 9.1. Khắc Phục Sự Cố Đen Màn Hình Trên Laptop Hybrid GPU (Lenovo Legion AMD Ryzen + NVIDIA RTX 3070)

Trên các dòng laptop sử dụng kiến trúc đồ họa kép (Hybrid / Optimus GPU như **Lenovo Legion AMD Ryzen + NVIDIA GeForce RTX 3070**), cửa sổ MuJoCo Passive Viewer đôi khi khởi tạo trước khi ma trận chiếu của OpenGL đồng bộ với driver đồ họa rời, dẫn đến hiện tượng **màn hình render 3D bị đen** hoặc camera đặt ngoài vùng hiển thị của robot.

**Cơ chế giải quyết tự động:**
- Script `scripts/interactive_viewer.py` tự động kích hoạt hàm `configure_viewer_camera(viewer.cam)` và `viewer.sync()` ngay sau khi mở context viewer:
  - Tọa độ tâm quan sát: `lookat = [0.0, 0.0, 0.15]` (tập trung vào thân robot).
  - Khoảng cách camera: `distance = 1.2` m.
  - Góc phương vị & nâng: `azimuth = 135.0°`, `elevation = -20.0°`.
  - Đồng bộ khung hình khởi tạo: `viewer.sync()`.

**Khắc phục thủ công trên Windows (nếu vẫn gặp lỗi hiển thị):**
1. **Thiết lập NVIDIA Control Panel:**
   - Mở *NVIDIA Control Panel* $\rightarrow$ *Manage 3D Settings* $\rightarrow$ *Program Settings*.
   - Thêm đường dẫn `python.exe` của môi trường đang chạy.
   - Chọn bộ xử lý đồ họa ưu tiên: *High-performance NVIDIA processor*.
2. **Kích hoạt dGPU Mode (MUX Switch) trên Lenovo Legion:**
   - Trong ứng dụng *Lenovo Vantage*, chuyển từ *Hybrid Mode* sang *dGPU Mode* (chỉ sử dụng card rời RTX 3070) và khởi động lại máy.
3. **Chế độ Headless:**
   - Nếu chạy trong môi trường không có màn hình hoặc lỗi driver GUI OpenGL, luôn có thể dùng cờ `--headless` để kiểm thử vật lý mà không cần hiển thị cửa sổ.


---

## 10. Bộ Mô Phỏng Song Song & Đo Hiệu Năng FPS (ParallelSimRunner & Benchmark)

Hệ thống cung cấp module mô phỏng song song đa luồng theo lô (`ParallelSimRunner`) tối ưu hóa tối đa CPU với các cơ chế Sim-to-Real trọng yếu:

1. **Giới Hạn Bước Episode (Episode Truncation - 500 Steps):**
   - Mỗi tập mô phỏng được giới hạn tối đa `max_episode_steps = 500` bước ($10\text{ s}$ ở chu kỳ $50\text{ Hz}$, $dt = 0.02\text{ s}$).
   - Khi `episode_steps >= 500`, cờ `timeouts` được kích hoạt và gán vào cờ `dones`, đồng thời trả về `infos["timeout"] = True`. Điều này ngăn chặn mạng RL khai thác các tập vô tận để tích lũy điểm thưởng tĩnh và chuẩn hóa phân phối dữ liệu huấn luyện.
2. **Ngẫu Nhiên Hóa Miền Không Gian Trạng Thái (Domain Randomization):**
   - **Góc khớp ban đầu khi Reset:** Bổ sung nhiễu đồng đều $\Delta q_{\text{init}} \in [-0.03, +0.03]\text{ rad}$ cho tất cả các khớp (đảm bảo góc gập gối $\ge 0.0$).
   - **Lệnh vận tốc ngẫu nhiên:** Vận tốc mục tiêu được lấy mẫu ngẫu nhiên:
     - Tiến/lùi: $v_x \in [0.0, 0.25]\text{ m/s}$
     - Trượt ngang: $v_y \in [-0.08, +0.08]\text{ m/s}$
     - Quay thân: $\omega_z \in [-0.50, +0.50]\text{ rad/s}$
3. **Đồng Bộ Hóa Quan Sát Tự Động Reset (Aligned Auto-Reset Observation):**
   - Khi một môi trường bị ngã hoặc hết thời gian (`dones[i] == True`), hàm `step()` ngay lập tức cập nhật `next_obs[i] = self.reset_env(i)`.
   - Cơ chế này đảm bảo observation trả về cho thuật toán RL luôn là trạng thái bắt đầu sạch của episode mới, giải quyết triệt để lỗi lệch pha trạng thái (out-of-sync transition) trong bộ đệm RolloutBuffer. Đồng thời, `last_actions` của môi trường reset được đưa về $0.0$.

### Lệnh Chạy Kiểm Thử Hiệu Năng Benchmark:

```bash
# 1. Chạy benchmark nhánh Raspberry Pi 5 (64 môi trường, 200 bước)
python pipelines/rpi5_stack/training/run_parallel_sim.py --num_envs 64 --steps 200

# 2. Chạy benchmark nhánh Dual ESP32-S3 (64 môi trường, 200 bước)
python pipelines/dual_esp32s3_stack/training/run_parallel_sim.py --num_envs 64 --steps 200

# 3. Chạy benchmark nhánh cơ sở (128 môi trường, 4 luồng CPU)
python training/run_parallel_sim.py --num_envs 128 --steps 100 --threads 4
```

### Kết quả đo kiểm thực tế trên CPU:
- **Tốc độ mô phỏng (FPS):** Đạt từ **15,000 đến 35,000+ Steps/Giây** trên CPU phổ thông.
- **Tỷ lệ tăng tốc thời gian thực (Real-time Factor):** Nhanh hơn thời gian thực từ **300x đến 700x**, cho phép hoàn thành 10 triệu bước tương tác RL chỉ trong vài phút.

---

## 11. Pipeline Huấn Luyện PPO & Xuất Mô Hình Triển Khai (PPO Training & Policy Export)

### 11.1. Pipeline Huấn Luyện PPO (`scripts/train_ppo.py`)

Hệ thống tích hợp thuật toán học tăng cường **Proximal Policy Optimization (PPO)** tối ưu hóa cho bài toán điều khiển robot hai chân:

- **Kiến trúc mạng Actor-Critic:**
  - Không gian đầu vào: Vector quan sát 42 chiều.
  - Tầng ẩn: 2 tầng fully-connected gồm 256 nơ-ron kích hoạt `nn.ELU`.
  - Actor Head: Dự đoán trung bình hành động 8 chiều $\mathbf{a}_t \in [-1, 1]^8$, đi kèm tham số học độ lệch chuẩn `log_std`. Khởi tạo trực giao (Orthogonal Initialization) với gain 0.01 ở tầng ra cuối để tránh biến thiên đột ngột ở các bước đầu.
  - Critic Head: Ước lượng hàm giá trị trạng thái $V(s) \in \mathbb{R}^1$.
- **Ước lượng lợi thế tổng quát (Generalized Advantage Estimation - GAE):**
  - Hệ số chiết khấu $\gamma = 0.99$, hệ số làm mịn GAE $\lambda = 0.95$.
- **Hàm mất mát PPO:**
  - PPO Clipped Surrogate Loss với ngưỡng cắt $\epsilon = 0.2$.
  - Trọng số hàm mất mát Critic $c_1 = 0.5$, hệ số thưởng Entropy $c_2 = 0.01$ để duy trì tính khám phá.

```bash
# Huấn luyện PPO với 64 môi trường song song (100,000 timesteps):
python scripts/train_ppo.py --num_envs 64 --total_timesteps 100000

# Chạy kiểm thử tích hợp nhanh (Dry-run 1 epoch, 4 envs, 4 steps) phục vụ CI/CD:
python scripts/train_ppo.py --dry-run
```

Checkpoints được tự động lưu tại `models/checkpoints/microduck_ppo.pt` chứa đầy đủ trọng số Actor, Critic, trạng thái Adam optimizer và cấu hình siêu tham số.

---

### 11.2. Xuất Chính Sách Triển Khai Đa Nền Tảng (`scripts/export_policy.py`)

Công cụ `scripts/export_policy.py` tự động chuyển đổi checkpoint PyTorch thành các định dạng tối ưu trên phần cứng đích:

```bash
# Xuất mô hình từ checkpoint huấn luyện PPO:
python scripts/export_policy.py --checkpoint models/checkpoints/microduck_ppo.pt
```

1. **Dành cho Raspberry Pi 5 (Float32 ONNX):**
   - Xuất file ONNX: `models/microduck_rpi5_policy.onnx` (Input: `[1, 42]`, Output: `[1, 8]`, Opset 13).
   - Tương thích hoàn toàn với ONNX Runtime trên Linux ARM64, độ sai lệch số học so với PyTorch $< 2 \times 10^{-7}$.
2. **Dành cho Dual ESP32-S3 (INT8 C/C++ Headers):**
   - Lượng tử hóa đối xứng trọng số thành `const int8_t` và bias thành `const int32_t`.
   - Sinh file header trọng số: `pipelines/dual_esp32s3_stack/deployment/esp32_policy_weights.h` tích hợp hàm nội suy nhúng siêu tốc `evaluate_policy_int8()`.
   - Sinh file cấu hình thang đo: `pipelines/dual_esp32s3_stack/deployment/esp32_quant_config.h` chứa mảng hằng số `OBS_CHANNEL_SCALES[42]` và `ACT_INV_SCALE`.
   - Tương thích hoàn toàn với trình biên dịch `g++ / xtensa-esp32s3-elf-g++` không yêu cầu bất kỳ thư viện ngoài nào.

---

## 12. Hướng Dẫn Nạp & Triển Khai Trên Phần Cứng Thật (Deployment Guide)

### 12.1. Triển Khai Trên Nhánh Raspberry Pi 5

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
3. **Đồng bộ hóa trạng thái hành động (`last_raw_actions`):**
   - Trong `infer_rpi5.py`, biến `self.last_raw_actions` lưu giữ hành động chuẩn hóa $[-1.0, 1.0]$ của mạng Actor và đưa vào các chiều $25..32$ của vector quan sát (thay vì đưa góc radian), đảm bảo đồng bộ 100% với môi trường mô phỏng.
4. **Khởi chạy chương trình điều khiển thời gian thực:**
   ```bash
   # Chạy chế độ mô phỏng kiểm thử an toàn (Mock PCA9685)
   python pipelines/rpi5_stack/deployment/infer_rpi5.py --mock-pca --rate 50

   # Chạy trên phần cứng thật với file model ONNX Float32
   python pipelines/rpi5_stack/deployment/infer_rpi5.py --model models/microduck_rpi5_policy.onnx --rate 50
   ```
   *Chi tiết sơ đồ chân GPIO và phân phối nguồn 5V/5A + 6V/15A xem tại:* [`pipelines/rpi5_stack/docs/rpi5_hardware_spec.md`](file:///d:/duckV1/pipelines/rpi5_stack/docs/rpi5_hardware_spec.md).

---

### 12.2. Triển Khai Trên Nhánh Dual ESP32-S3

1. **Cài đặt công cụ lập trình:**
   - Cài đặt **Arduino IDE 2.x** hoặc **ESP-IDF v5.x** kèm gói bo mạch `esp32` ($\ge 2.0.14$).
   - Chọn Board: `ESP32S3 Dev Module`, Flash Size: `16MB`, PSRAM: `OPI PSRAM`.
2. **Phân tách bus I2C cho cụm 5-ray LiDAR trên MCU 1:**
   - Để tránh xung đột phần cứng với bus SCCB của Camera OV2640 (GPIO 4: SIOD, GPIO 5: SIOC), cụm 5 cảm biến khoảng cách ToF VL53L1X được đấu nối vào **bus I2C chuyên biệt**:
     - `PIN_I2C_SDA`: **GPIO 19**
     - `PIN_I2C_SCL`: **GPIO 20**
3. **Nối dây liên kết UART giữa 2 bo ESP32-S3:**
   - Nối chéo UART2 ($921,600\text{ bps}$):
     - `MCU 1 GPIO 17 (TX)` $\rightarrow$ `MCU 2 GPIO 18 (RX)`
     - `MCU 1 GPIO 18 (RX)` $\leftarrow$ `MCU 2 GPIO 17 (TX)`
     - Nối chung đường mass `GND` giữa 2 bo mạch.
4. **Hiệu chỉnh IMU & Suy luận INT8 trên MCU 2:**
   - **Đảo dấu vector trọng trường IMU:** Các trục gia tốc MPU6050 được đảo dấu (`-ax`, `-ay`, `-az`) kết hợp bộ lọc thông thấp số IIR ($\alpha = 0.80$), đảm bảo khi robot đứng thẳng, vector trọng trường đo được là $[0, 0, -1]$ khớp tuyệt đối với hệ quy chiếu MuJoCo.
   - **Đồng bộ hóa `g_last_raw_actions`:** Mảng `g_last_raw_actions` $[-1.0, 1.0]$ được đưa vào các chiều $25..32$ của vector quan sát.
   - **Suy luận trực tiếp INT8:** Hàm `evaluate_policy_int8()` thực thi nội suy mạng nơ-ron Actor trực tiếp trên MCU 2 trong thời gian $< 2.8\text{ ms}$, xuất 8 kênh xung PWM trực tiếp tới servo MG996R theo chu kỳ nghiêm ngặt $50\text{ Hz}$ ($20\text{ ms}$).
5. **Nạp Firmware:**
   - **Nạp MCU 1:** Mở sketch [`pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino) và nhấn **Upload**.
   - **Nạp MCU 2:** Mở sketch [`pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino) và nhấn **Upload**.
   *Chi tiết sơ đồ chân, mạch chia kênh và nguồn phân tầng xem tại:* [`pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md`](file:///d:/duckV1/pipelines/dual_esp32s3_stack/docs/dual_esp32s3_hardware_spec.md).

---

## 13. Chạy Toàn Bộ Test Suite Tự Động (Project Test Suite)

Dự án sở hữu bộ kiểm thử tự động toàn diện thẩm định toàn bộ mô hình vật lý, cấu hình huấn luyện và mã nguồn triển khai:

```bash
pytest -v
```

### Các hạng mục kiểm thử tự động gồm:
- **`test_rpi5_model.py`:** Thẩm định mô hình MJCF và URDF nhánh Pi 5 (ngân sách khối lượng $860\text{ g}$, 5 tia LiDAR, thấu kính camera, mic site, 4 cột chướng ngại vật).
- **`test_esp32_model.py`:** Thẩm định mô hình MJCF và URDF nhánh ESP32 (ngân sách khối lượng $780\text{ g}$, CoM thấp, 5 tia LiDAR, camera, mic, 4 cột chướng ngại vật).
- **`test_mjcf_model.py` & `test_urdf_model.py`:** Thẩm định cấu trúc hình học, quán tính và 4 cột chướng ngại vật của mô hình cơ sở.
- **`test_env_config.py` & `test_pipeline_training.py`:** Thẩm định không gian quan sát 42 chiều, hàm thưởng né vật cản, hàm thưởng nhấc chân luân phiên `foot_contact`, lượng tử hóa INT8 và runner song song.
- **`test_parallel_runner.py`:** Thẩm định `ParallelSimRunner` (Episode truncation 500 bước, domain randomization, aligned auto-reset, nhị phân hóa tiếp xúc chân).
- **`test_ppo_training.py`:** Thẩm định mạng ActorCritic, GAE advantages, PPO dry-run CLI, xuất Float32 ONNX parity, và biên dịch C++ INT8 evaluation.
- **`test_deployment_artifacts.py`:** Thẩm định runtime `infer_rpi5.py` (`last_raw_actions`), giao thức `uart_packet_protocol.h`, chân I2C GPIO 19/20 trong `mcu1_main.ino`, đảo dấu IMU và suy luận INT8 trong `mcu2_main.ino`, cùng 2 bộ tài liệu đặc tả phần cứng.
- **`test_verify_script.py` & `test_interactive_viewer.py`:** Thẩm định script kiểm tra chéo và trình trực quan 3D tương tác.

---

## 14. Bảng Tổng Hợp 10 Điểm Nghẽn Kỹ Thuật Đã Được Khắc Phục (10 Technical Bottlenecks Summary)

| STT | Điểm Nghẽn Kỹ Thuật (Bottleneck) | Nguyên Nhân Gốc Rễ (Root Cause) | Giải Pháp Kỹ Thuật Đã Triển Khai | Tệp Nguồn Ảnh Hưởng |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Thiếu vật cản trong mô hình MJCF** | Cảm biến LiDAR 5 tia quét vào hư không vô tận ($2.0\text{ m}$), robot không có đối tượng phản xạ để học né tránh. | Thêm 4 cột hình trụ `obstacle_*` (`size="0.08 0.25"`) vào `<worldbody>` cả 3 mô hình MJCF. | `models/mjcf/microduck_8dof.xml`, `pipelines/*/models/mjcf/*.xml` |
| **2** | **Hàm thưởng tiếp xúc chân sơ sài** | Tính tổng đơn thuần tiếp xúc chân dẫn tới việc kéo lê 2 chân hoặc nhảy tâng tâng. | Thiết kế hàm thưởng dáng đi bipedal: thưởng $+1.0$ khi đúng 1 chân chạm sàn (di chuyển), thưởng $+0.5$ khi cả 2 chân chạm sàn (đứng yên). | `training/env_config.py`, `pipelines/*/training/env_config.py` |
| **3** | **Không giới hạn độ dài tập mô phỏng** | Episode kéo dài vô tận khiến phân phối dữ liệu RL bị lệch, robot đứng yên tích điểm vô hạn. | Tích hợp Episode Truncation tại 500 bước ($10\text{ s}$ ở $50\text{ Hz}$), đánh dấu `infos["timeout"] = True` và kết thúc tập. | `training/run_parallel_sim.py`, `pipelines/*/training/run_parallel_sim.py` |
| **4** | **Thiếu ngẫu nhiên hóa khi reset** | Mọi episode khởi tạo ở đúng một tư thế tĩnh và lệnh vận tốc cố định, policy kém tổng quát. | Ngẫu nhiên hóa góc khớp ban đầu $[-0.03, +0.03]\text{ rad}$ và lấy mẫu ngẫu nhiên lệnh vận tốc $[v_x, v_y, \omega_z]$ khi reset. | `training/run_parallel_sim.py`, `pipelines/*/training/run_parallel_sim.py` |
| **5** | **Observation trả về lệch pha khi Done** | Khi done, runner trả về observation của trạng thái đã ngã thay vì trạng thái reset ban đầu, làm méo buffer RL. | Cập nhật tức thời `next_obs[i] = self.reset_env(i)` ngay trong hàm `step()` khi phát hiện done. | `training/run_parallel_sim.py`, `pipelines/*/training/run_parallel_sim.py` |
| **6** | **Thiếu pipeline huấn luyện RL hoàn chỉnh** | Chưa có mã nguồn huấn luyện end-to-end cho mô hình 42 chiều quan sát và 8 hành động. | Xây dựng pipeline `scripts/train_ppo.py` hoàn chỉnh: Actor-Critic ($42 \to 256 \to 256 \to 8$), GAE, Clipped PPO và cờ `--dry-run`. | `scripts/train_ppo.py`, `tests/test_ppo_training.py` |
| **7** | **Thiếu công cụ xuất mô hình đa nền tảng** | Khó khăn khi chuyển đổi mạng PyTorch sang phần cứng thật trên Linux Pi 5 và vi điều khiển ESP32-S3. | Xây dựng `scripts/export_policy.py`: xuất Float32 ONNX cho Pi 5 và INT8 C/C++ header nhúng (`evaluate_policy_int8`) cho ESP32-S3. | `scripts/export_policy.py`, `models/microduck_rpi5_policy.onnx`, `pipelines/dual_esp32s3_stack/deployment/*.h` |
| **8** | **Lệch định dạng hành động bước trước** | Trên mạch thật đưa góc radian vào observation chiều $25..32$, trong khi mô phỏng học trên hành động $[-1.0, 1.0]$. | Chuẩn hóa mảng `last_raw_actions` lưu trữ giá trị chuẩn hóa $[-1.0, 1.0]$ trên cả `infer_rpi5.py` và `mcu2_main.ino`. | `pipelines/rpi5_stack/deployment/infer_rpi5.py`, `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino` |
| **9** | **Xung đột chân I2C LiDAR trên MCU 1** | Cụm 5 ToF LiDAR và SCCB của Camera OV2640 dùng chung GPIO 4/5 gây nghẽn và treo bus I2C. | Tách biệt hoàn toàn cụm LiDAR sang bus I2C chuyên dụng trên GPIO 19 (SDA) và GPIO 20 (SCL). | `pipelines/dual_esp32s3_stack/deployment/mcu1_vision_audio_s3/mcu1_main.ino`, `dual_esp32s3_hardware_spec.md` |
| **10** | **Sai lệch dấu vector trọng trường IMU** | Chiều gia tốc đứng thẳng đo từ MPU6050 là $+1$, trong khi MuJoCo quy chiếu trọng trường là $-1$. | Đảo dấu các trục gia tốc (`-ax`, `-ay`, `-az`) kết hợp bộ lọc IIR $\alpha=0.80$, nhúng suy luận INT8 trực tiếp trên MCU 2. | `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino` |
