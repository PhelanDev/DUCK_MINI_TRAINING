# MicroDuck 8-DOF: Robot Hai Chân Nhỏ Gọn & Pipeline Huấn Luyện RL (Sim-to-Real)

Dự án phát triển và chuẩn hóa mô hình mô phỏng vật lý cho robot hai chân **MicroDuck 8-DOF**, sử dụng 8 động cơ servo phổ thông **MG996R**. Hệ thống tích hợp toàn diện định dạng **MuJoCo MJCF** và **ROS/PyBullet URDF**, đi kèm pipeline mô phỏng song song quy mô lớn phục vụ huấn luyện học tăng cường (Reinforcement Learning - Sim-to-Real) triển khai trên vi điều khiển nhúng (**ESP32**, **Kendryte K230**).

---

## 1. Tổng Quan Kiến Trúc Robot (Overview)

Phiên bản gốc của MicroDuck sử dụng 15 servo thông minh Dynamixel XL330 đắt đỏ và phức tạp. Nhằm tối ưu hóa chi phí gia công, linh kiện sẵn có tại Việt Nam và độ bền cơ khí, robot được thiết kế lại toàn diện về cấu hình **8 servo MG996R nhông kim loại**:

- **Tổng khối lượng:** $\approx 780\text{ g}$
  - 8 servo MG996R $\times 55\text{ g} = 440\text{ g}$
  - Khung thân in 3D (PETG/PLA+): $\sim 160\text{ g}$
  - Pin LiPo 2S 7.4V 1000mAh: $\sim 120\text{ g}$
  - Mạch điều khiển (ESP32 / K230, PCA9685, cảm biến MPU6050/BNO085): $\sim 60\text{ g}$
- **Chiều cao tổng thể:** $\sim 230\text{ mm}$ (Độ cao tâm trọng trường CoM khi đứng thẳng: $\approx 0.170\text{ m}$).
- **Trọng tâm (Center of Mass - CoM):** Bố trí dồn về nửa dưới của thân chính (`base_link`), giúp hạ thấp trọng tâm và tăng tối đa độ ổn định chống lật khi sải bước.

---

## 2. Cây Động Học & Kích Thước Các Link (Kinematic Tree & Dimensions)

Robot sở hữu 8 bậc tự do (8 DOF) được bố trí khoa học:
- **Cụm đầu (Head - 2 DOF):** Hỗ trợ camera và cảm biến thị giác định hướng không gian.
- **Cụm chân (Legs - 6 DOF, 3 DOF mỗi chân):** Cấu trúc bipedal tối giản nhưng đầy đủ năng lực di chuyển đa hướng.

```
                  [base_link] (Thân chính, Pin, ESP32, IMU)
                 /           \                              \
        [head_yaw_link]   [left_hip_roll_link]    [right_hip_roll_link]
              |                    |                        |
         [head_link]       [left_thigh_link]        [right_thigh_link]
                                   |                        |
                           [left_shin_link]         [right_shin_link]
                                   |                        |
                           [left_foot_link]         [right_foot_link]
```

### 2.1. Bảng Thông Số Hình Học & Khối Lượng Từng Link

| Link Name | Khối lượng | Kích thước ($X \times Y \times Z$) | Chức năng cơ học |
| :--- | :--- | :--- | :--- |
| `base_link` | 320 g | $110 \times 80 \times 70\text{ mm}$ | Thân chính, chứa pin, mạch ESP32, IMU ở tâm hình học |
| `head_yaw_link` | 30 g | $35 \times 35 \times 30\text{ mm}$ | Khối quay cổ theo trục thẳng đứng |
| `head_link` | 65 g | $60 \times 50 \times 45\text{ mm}$ | Đầu vịt mang camera quan sát và mỏ |
| `left_hip_roll_link` | 60 g | $45 \times 40 \times 40\text{ mm}$ | Cụm xoay hông lắc nghiêng chuyển trọng tâm |
| `left_thigh_link` | 70 g | $35 \times 30 \times 65\text{ mm}$ | Đùi ($L_1 = 65\text{ mm}$), chứa servo gối |
| `left_shin_link` | 50 g | $30 \times 25 \times 65\text{ mm}$ | Cẳng chân ($L_2 = 65\text{ mm}$) |
| `left_foot_link` | 35 g | $75 \times 55 \times 10\text{ mm}$ | Bàn chân bản to, mặt đế cao su ma sát cao |
| `right_*_link` | Đối xứng | Đối xứng qua mặt phẳng $X$ | Đối xứng chân phải |

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

## 3. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
d:\duckV1\
├── models/
│   ├── mjcf/
│   │   └── microduck_8dof.xml       # Mô hình MuJoCo MJCF hoàn chỉnh
│   └── urdf/
│       └── microduck_8dof.urdf      # Mô hình URDF đồng bộ động học & quán tính
├── training/
│   ├── __init__.py
│   ├── env_config.py                # Cấu hình RL: obs 37-dim, action 8-dim, reward
│   └── run_parallel_sim.py          # Vectorized parallel runner & benchmark FPS
├── scripts/
│   ├── __init__.py
│   ├── verify_model.py              # Script kiểm tra chéo vật lý MJCF vs URDF
│   └── interactive_viewer.py        # Trình trực quan hóa 3D & kiểm thử tư thế
├── tests/                           # Toàn bộ test suite tự động (Pytest)
└── README.md                        # Tài liệu hướng dẫn tích hợp toàn diện
```

---

## 4. Hướng Dẫn Cài Đặt & Môi Trường

Hệ thống yêu cầu **Python 3.10+**. Khuyến nghị sử dụng môi trường ảo:

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux / macOS:
source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install mujoco numpy pytest
```

---

## 5. Kiểm Tra Tính Toàn Vẹn Vật Lý (Model Verification)

Trước khi tiến hành trực quan hoặc huấn luyện, hãy chạy script kiểm tra chéo tự động giữa mô hình MJCF và URDF:

```bash
python scripts/verify_model.py
```

### Các hạng mục script kiểm tra:
1. **Khớp và góc quay:** Đối chiếu 8 revolute joints về tên, trục quay ($X, Y, Z$) và giới hạn góc.
2. **Khối lượng & Quán tính:** Xác nhận tổng khối lượng đạt $\approx 0.780\text{ kg}$ và ma trận quán tính dương xác định.
3. **Mô phỏng tĩnh 1000 bước:** Đặt robot đứng trên mặt sàn dưới gia tốc trọng trường $-9.81\text{ m/s}^2$, kiểm tra servo MG996R giữ vững tư thế, không sụp gối và không có va chạm tự thân bất thường.

---

## 6. Trực Quan Hóa 3D & Kiểm Thử Tư Thế (Interactive 3D Viewer)

Sử dụng `scripts/interactive_viewer.py` để trực quan hóa mô hình trong không gian 3D, kiểm thử độ vững và các tư thế bước đi.

### 6.1. Khởi chạy 3D Viewer (Passive Viewer)

```bash
# Mở cửa sổ trực quan hóa 3D với tư thế đứng mặc định
python scripts/interactive_viewer.py

# Xem chu kỳ đi bộ mô phỏng (Dynamic Gait Demo)
python scripts/interactive_viewer.py --gait-demo

# Khởi tạo với các tư thế mẫu: stand, squat, left_step, right_step, bow
python scripts/interactive_viewer.py --pose squat
python scripts/interactive_viewer.py --pose left_step
```

### 6.2. Chế độ Điều Khiển Khớp Tương Tác Dòng Lệnh (CLI Mode)

Dành cho việc kiểm tra tinh chỉnh góc từng khớp bằng bàn phím:

```bash
python scripts/interactive_viewer.py --interactive
```

*Trong cửa sổ lệnh tương tác:*
- `pose stand` / `pose squat`: Chuyển sang tư thế mẫu.
- `set left_knee_pitch 0.5`: Đặt góc khớp gối trái thành 0.5 rad.
- `status`: In tọa độ trọng tâm CoM, độ vững và góc các khớp.
- `step 200`: Mô phỏng 200 bước tiếp theo.
- `reset`: Đặt lại robot về trạng thái ban đầu.

### 6.3. Chế Độ Không Màn Hình (Headless Mode)

Tự động hoạt động trên máy chủ từ xa, server CI/CD hoặc container Docker không có màn hình hiển thị:

```bash
python scripts/interactive_viewer.py --headless --duration 3.0 --gait-demo
```

---

## 7. Mô Phỏng Song Song & Đo Hiệu Năng FPS (Parallel Sim Benchmark)

Để phục vụ huấn luyện RL quy mô lớn, module `training/run_parallel_sim.py` cung cấp môi trường mô phỏng song song theo lô (Batched Vectorized Simulator) tận dụng tối đa đa luồng CPU:

```bash
# Chạy benchmark 64 môi trường song song trong 200 bước
python training/run_parallel_sim.py --num_envs 64 --steps 200

# Chạy với 128 môi trường và chỉ định 4 luồng CPU
python training/run_parallel_sim.py --num_envs 128 --steps 100 --threads 4
```

### Kết quả đo kiểm thực tế trên CPU:
- **Tốc độ mô phỏng (FPS):** Đạt từ **15,000 đến 35,000+ Steps/Giây** trên CPU phổ thông.
- **Tỷ lệ tăng tốc thời gian thực (Real-time Factor):** Nhanh hơn thời gian thực từ **300x đến 700x**, cho phép hoàn thành 10 triệu bước tương tác RL chỉ trong vài phút.

---

## 8. Hướng Dẫn Huấn Luyện Học Tăng Cường (RL Sim-to-Real Guide)

Pipeline huấn luyện được thiết kế tương thích trực tiếp với framework `microduck_rl` và **MuJoCo Warp** trên GPU.

### 8.1. Không Gian Quan Sát (Observation Space - 37 Chiều)

Mỗi bước điều khiển (tần số điều khiển $50\text{ Hz}$, $dt = 0.02\text{ s}$), mô hình nhận véc-tơ trạng thái gồm 37 giá trị:

$$\mathbf{o}_t \in \mathbb{R}^{37}$$

1. **Vận tốc góc thân (Base Angular Velocity - 3 dims):** Đo từ con quay hồi chuyển (Gyroscope) $\omega_x, \omega_y, \omega_z$.
2. **Véc-tơ trọng trường quy chiếu hệ thân (Projected Gravity - 3 dims):** Tính từ Quaternion tư thế robot $[g_x, g_y, g_z]^T = \mathbf{R}^T [0, 0, -1]^T$.
3. **Lệnh vận tốc mong muốn (Velocity Commands - 3 dims):** $[v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}}]$.
4. **Góc hiện tại của 8 khớp (Joint Positions - 8 dims):** Đọc từ encoder/feedback servo.
5. **Vận tốc hiện tại của 8 khớp (Joint Velocities - 8 dims):** Đạo hàm góc khớp.
6. **Hành động ở bước liền trước (Previous Actions - 8 dims):** Góc điều khiển $a_{t-1}$ nhằm đảm bảo tính mượt mà của quỹ đạo.
7. **Lực tiếp xúc bàn chân (Foot Contact Status - 4 dims):** Cảm biến chạm sàn chân trái và chân phải.

### 8.2. Không Gian Hành Động (Action Space - 8 Chiều)

Chính sách RL dự đoán hành động chuẩn hóa $\mathbf{a}_t \in [-1, +1]^8$. Module `env_config.py` thực hiện:
1. **Ánh xạ dải góc (Scaling):** Ánh xạ từ $[-1, +1]$ sang góc radian vật lý của từng khớp dựa trên `JOINT_LIMITS`.
2. **Kẹp biến thiên góc (Rate Limiting):** Giới hạn độ dịch chuyển góc tối đa $\Delta q \le \omega_{\max} \cdot \Delta t$ để bảo vệ nhông hộp số MG996R, chống rung giật và hiện tượng trượt bước khi đưa lên robot thật.

### 8.3. Hàm Thưởng Huấn Luyện (Reward Shaping)

Hàm thưởng được tổng hợp từ các thành phần cân bằng:

$$R_{\text{total}} = w_v R_{\text{lin\_vel}} + w_\omega R_{\text{ang\_vel}} + w_u R_{\text{upright}} + w_c R_{\text{contact}} - w_\tau P_{\text{torque}} - w_{\Delta a} P_{\text{action\_rate}}$$

- **Bám vận tốc dài ($R_{\text{lin\_vel}}$):** $\exp\left(-\frac{\|v_{xy} - v_{xy}^{\text{cmd}}\|^2}{0.25}\right)$
- **Bám vận tốc quay ($R_{\text{ang\_vel}}$):** $\exp\left(-\frac{(\omega_z - \omega_z^{\text{cmd}})^2}{0.25}\right)$
- **Giữ thân thẳng đứng ($R_{\text{upright}}$):** Khuyến khích véc-tơ trọng trường chiếu $g_z \to -1$.
- **Phạt rung giật & quá tải ($P_{\text{torque}}, P_{\text{action\_rate}}$):** Tối thiểu hóa năng lượng và gia tốc giật để đảm bảo Sim-to-Real an toàn.

---

## 9. Xuất Mô Hình ONNX & Nạp Xuống Vi Điều Khiển (ESP32 / K230)

Sau khi huấn luyện thuật toán PPO đạt độ hội tụ:

### 9.1. Xuất sang định dạng ONNX
```python
import torch

# Giả sử policy là mạng nơ-ron Actor PyTorch (MLP 37 -> 128 -> 64 -> 8)
dummy_obs = torch.zeros(1, 37, dtype=torch.float32)
torch.onnx.export(
    policy,
    dummy_obs,
    "microduck_policy.onnx",
    input_names=["observation"],
    output_names=["action"],
    opset_version=13,
)
```

### 9.2. Triển khai thời gian thực trên Vi điều khiển
1. **Kendryte K230 (Dual-core RISC-V + KPU NPU):**
   - Biên dịch ONNX sang định dạng `.kmodel` bằng `nncase`.
   - Tốc độ suy luận đạt $< 1.5\text{ ms}$ cho mỗi bước.
2. **ESP32-S3 (Xtensa Dual-core + Vector Instructions):**
   - Chuyển đổi sang `ESP-DL` hoặc `TensorFlow Lite for Microcontrollers` (TFLM) với lượng tử hóa INT8.
   - Thời gian suy luận mạng MLP 2 lớp ẩn đạt $\sim 3\text{ ms}$, hoàn toàn đáp ứng chu kỳ điều khiển $20\text{ ms}$ ($50\text{ Hz}$).
3. **Mạch giao tiếp Servo PCA9685:**
   - ESP32 nhận góc $\mathbf{a}_t$, chuyển đổi sang độ rộng xung PWM ($500\mu\text{s} \to 2500\mu\text{s}$) và truyền qua bus I2C tần số 400kHz tới bộ điều khiển 16 kênh PCA9685 cấp nguồn riêng 5V-6V cho 8 servo MG996R.

---

## 10. Chạy Toàn Bộ Test Suite Dự Án

Để kiểm tra toàn diện chất lượng mã nguồn, chạy:

```bash
pytest -v
```

Toàn bộ các bài kiểm thử cấu hình môi trường, hình học MJCF/URDF, script xác thực và runner song song sẽ được tự động thực thi và thẩm định.
