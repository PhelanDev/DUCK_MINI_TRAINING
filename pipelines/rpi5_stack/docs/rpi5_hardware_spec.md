# Tài Liệu Đặc Tả Phần Cứng: Nhánh Raspberry Pi 5 (MicroDuck 8-DOF)

**Dự án:** Robot Bipedal 8 bậc tự do MicroDuck 8-DOF  
**Nhánh phần cứng:** `pipelines/rpi5_stack`  
**Bộ xử lý trung tâm:** Raspberry Pi 5 (4GB RAM, Broadcom BCM2712 Quad-core Cortex-A76 @ 2.4GHz)  
**Tổng khối lượng danh định:** $\approx 860\text{ g}$ (Bao gồm pin 2S 2200mAh, tản nhiệt Active Cooler, module Buck 5V/5A, PCA9685 và 8 servo MG996R)  

---

## 1. Kiến Trúc Hệ Thống & Phân Phối Nguồn Điện (Power Distribution)

Do robot sử dụng 8 servo công suất lớn TowerPro MG996R (dòng đỉnh stall lên đến $2.5\text{A}$ mỗi servo) và bo mạch Raspberry Pi 5 đòi hỏi nguồn cấp ổn định $5\text{V}/5\text{A}$ ($25\text{W}$ chuẩn PD), việc tách biệt hoàn toàn 2 đường nguồn là nguyên tắc an toàn tuyệt đối nhằm chống sập nguồn (brownout) và nhiễu sóng xung ngược (inductive back-EMF).

### 1.1. Sơ đồ phân phối điện áp

```
               ┌────────────────────────────────────────────────────────┐
               │    Khối Nguồn Chính: Pin LiPo/Li-ion 2S (7.4V - 8.4V)   │
               │         Dung lượng: 2200mAh - 25C (Xả liên tục 55A)     │
               └───────────┬────────────────────────────────┬───────────┘
                           │                                │
                 (Đường nguồn số MCU)             (Đường động lực Servo)
                           │                                │
            ┌──────────────▼─────────────┐   ┌──────────────▼─────────────┐
            │   Module Buck DC-DC 5V/5A  │   │  Module Buck DC-DC 6V/15A  │
            │   (XL4015 hoặc TPS54560)   │   │  (Hoặc UBEC máy bay mô hình│
            └──────────────┬─────────────┘   └──────────────┬─────────────┘
                           │ 5.0V / 5.0A                    │ 6.0V / 15.0A
            ┌──────────────▼─────────────┐                  │
            │      Raspberry Pi 5        │                  │
            │  (Cấp qua USB-C hoặc GPIO) │                  │
            └───────┬─────────────┬──────┘                  │
              3.3V  │  I2C        │ I2S                     │
             Logic  │  Bus        │ Bus                     │
                    │             │                         │
     ┌──────────────▼────────┐    │          ┌──────────────▼─────────────┐
     │  Bo Mở Rộng PCA9685   │    │          │ Cổng V+ Khối Servo PCA9685 │
     │  (Logic điều khiển)   │    │          │  (Kèm tụ đệm 4700uF / 16V) │
     └──────────────┬────────┘    │          └──────────────┬─────────────┘
                    │             │                         │
                    └─────────────┼─────────────────────────┘
                                  ▼
                     [8x Servo MG996R: Chân, Cổ]
```

### 1.2. Thông số nguồn chi tiết

1. **Nguồn cấp cho Raspberry Pi 5:**
   - Điện áp: $5.1\text{V} \pm 0.1\text{V}$.
   - Dòng điện danh định: Tối thiểu $5.0\text{A}$ để Pi 5 kích hoạt chế độ đầy tải (Full Performance Profile mà không hiện cảnh báo sụt nguồn trên Linux kernel).
   - Điểm nối: Cổng USB-C hoặc chân GPIO 5V (Pin 2, Pin 4) và GND (Pin 6, Pin 9). Khuyến nghị hàn dây tiết diện $\ge 18\text{ AWG}$ để giảm sụt áp trên dây.

2. **Nguồn cấp cho dàn động lực 8x MG996R:**
   - Điện áp: $6.0\text{V}$ (Giúp servo MG996R đạt tốc độ $0.16\text{ s}/60^\circ$ và mô-men xoắn $11.0\text{ kg}\cdot\text{cm}$).
   - Dòng điện danh định: $12\text{A} - 15\text{A}$ tải đỉnh (khi robot đổi hướng đột ngột hoặc chịu lực tiếp đất).
   - Tụ điện chống nhiễu: Bắt buộc hàn song song tụ hóa Low-ESR $4700\mu\text{F} / 16\text{V}$ và tụ gốm $100\text{nF}$ ngay tại ngõ vào cọc nguồn xanh của PCA9685 để hấp thụ sốc điện cảm ứng từ cuộn dây động cơ.
   - Nguyên tắc nối mass: Nối chung Mass (GND) giữa mạch Logic của Pi 5 và mạch Động lực tại một điểm duy nhất (Star Ground Topology) để tránh tạo vòng lặp dòng (ground loop).

---

## 2. Sơ Đồ Đấu Nối Chân GPIO (Raspberry Pi 5 Pinout Mapping)

Raspberry Pi 5 trang bị đầu cắm 40-pin GPIO tiêu chuẩn. Các ngoại vi phục vụ nhận thức và điều khiển vận động được phân bổ tối ưu như sau:

| Tên Ngoại Vi | Tín Hiệu | Chân Pi 5 (Physical Pin) | Tên GPIO (BCM) | Ghi Chú & Đặc Tính |
| :--- | :--- | :--- | :--- | :--- |
| **PCA9685 (I2C Servo)** | VCC (Logic) | Pin 1 | 3.3V Power | Nguồn nuôi chip PCA9685 |
| | SDA | Pin 3 | GPIO 2 (SDA1) | Đường truyền dữ liệu I2C (Bus 1, 400kHz) |
| | SCL | Pin 5 | GPIO 3 (SCL1) | Đường xung nhịp I2C (Bus 1) |
| | GND | Pin 6 | Ground | Mass chung |
| | OE (Output Enable) | Pin 11 | GPIO 17 | Tích cực mức thấp (kéo GND để mở xuất xung) |
| **MPU6050 (6-DOF IMU)** | VDD | Pin 1 | 3.3V Power | Cấp nguồn nuôi cảm biến 3.3V |
| | SDA | Pin 3 | GPIO 2 (SDA1) | Đường dữ liệu I2C (Địa chỉ 0x68) |
| | SCL | Pin 5 | GPIO 3 (SCL1) | Đường xung nhịp I2C (Bus 1, 400kHz) |
| | GND | Pin 6 | Ground | Mass chung |
| **INMP441 (I2S Mic)** | VDD | Pin 17 | 3.3V Power | Nguồn nuôi microphone |
| | GND | Pin 20 | Ground | Mass tín hiệu âm thanh |
| | SD (Serial Data) | Pin 38 | GPIO 20 (PCM_DIN) | Dữ liệu âm thanh 24-bit số gửi về Pi 5 |
| | WS / LRCL | Pin 35 | GPIO 19 (PCM_FS) | Tín hiệu Word Select (Trái / Phải) |
| | SCK / BCLK | Pin 12 | GPIO 18 (PCM_CLK) | Tín hiệu xung nhịp lấy mẫu bit (Continuous Clock) |
| | L/R Select | Pin 14 | Ground | Nối đất để chọn kênh âm thanh Mono trái |
| **Cụm 5 Tia ToF LiDAR** | VCC | Pin 2 | 5V Power / 3.3V | Nguồn cấp cho cụm cảm biến đo xa |
| | SDA | Pin 27 | GPIO 0 (ID_SD) / I2C1 | Giao tiếp I2C qua bộ chia kênh TCA9548A |
| | SCL | Pin 28 | GPIO 1 (ID_SC) / I2C1 | Xung nhịp chia kênh quét 5 góc $[-30^\circ \dots +30^\circ]$ |
| | XSHUT_1..5 | Pin 13, 15, 16, 22, 29 | GPIO 27, 22, 23, 25, 5 | Chân reset địa chỉ I2C động cho 5 sensor |
| **Quạt Active Cooler** | Fan Header | Cổng JST 4-pin Pi 5 | PWM / Tacho | Điều tốc quạt theo nhiệt độ SoC BCM2712 |

---

## 3. Cụm Cảm Biến Đầu Vịt (Perception Suite)

### 3.1. Camera quang học (RGB Camera)
- **Chuẩn kết nối:** Cổng ribbon 22-pin MIPI CSI-2 tích hợp sẵn trên bo Pi 5 (CAM0 hoặc CAM1, 4-lane tốc độ cao).
- **Mô hình đề xuất:** Raspberry Pi Camera Module 3 (cảm biến Sony IMX708 12MP, có autofocus) hoặc module tương thích có ống kính góc rộng $60^\circ - 75^\circ$ FOV.
- **Tùy chọn phụ:** Camera USB UVC chuẩn công nghiệp độ trễ thấp gắn tại cổng USB 3.0 (xanh dương).

### 3.2. Microphone số I2S INMP441
- **Vị trí lắp đặt:** Bố trí bên trong mỏ vịt hoặc đỉnh sọ, hướng ra trước để thu nhận từ khóa "Hey Duck".
- **Cấu hình trên Linux (`/boot/firmware/config.txt`):**
  ```ini
  # Kích hoạt driver I2S Full-duplex trên Pi 5
  dtoverlay=googlevoicehat-soundcard
  # hoặc driver tương đương ALSA I2S MEMS
  dtoverlay=i2s-mmap
  ```
- **Tần số lấy mẫu:** $16,000\text{ Hz}$, $16\text{-bit}$ hoặc $24\text{-bit}$ định dạng PCM nguyên bản.

### 3.3. Cảm biến khoảng cách 5 tia LiDAR (Rangefinder Scanner)
- **Nguyên lý:** 5 module ToF quang học bước sóng $940\text{nm}$ (VL53L0X hoặc VL53L1X) được bố trí theo góc quạt cung tròn ngang:
  - Cảm biến 1: $-30^\circ$ (Trái ngoài)
  - Cảm biến 2: $-15^\circ$ (Trái giữa)
  - Cảm biến 3: $0^\circ$ (Chính diện)
  - Cảm biến 4: $+15^\circ$ (Phải giữa)
  - Cảm biến 5: $+30^\circ$ (Phải ngoài)
- **Tầm đo:** $0.05\text{ m} \rightarrow 2.0\text{ m}$, tần số lấy mẫu $\ge 50\text{ Hz}$.
- **Đầu ra:** 5 giá trị khoảng cách chuẩn hóa trực tiếp nạp vào vector quan sát 42 chiều của mạng nơ-ron để kích hoạt hàm thưởng né chướng ngại vật (`obstacle_avoidance`).

---

## 4. Bảng Ánh Xạ Kênh PCA9685 Tới Khớp Robot (MG996R Channel Mapping)

Bo mạch mở rộng I2C PCA9685 được gán 8 kênh PWM đầu tiên tương ứng với 8 bậc tự do của robot:

| Kênh PCA9685 | Tên Khớp Động Học | Giới Hạn Góc (rad) | Giới Hạn Xung PWM ($\mu\text{s}$) | Góc Trung Tính |
| :--- | :--- | :--- | :--- | :--- |
| **Kênh 0** | `head_yaw` | $[-1.57, +1.57]$ | $500\mu\text{s} \leftrightarrow 2500\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |
| **Kênh 1** | `head_pitch` | $[-0.52, +0.78]$ | $1168\mu\text{s} \leftrightarrow 2000\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |
| **Kênh 2** | `left_hip_roll` | $[-0.44, +0.44]$ | $1220\mu\text{s} \leftrightarrow 1780\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |
| **Kênh 3** | `left_hip_pitch` | $[-0.87, +0.87]$ | $945\mu\text{s} \leftrightarrow 2055\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |
| **Kênh 4** | `left_knee_pitch` | $[0.0, +1.48]$ | $1500\mu\text{s} \leftrightarrow 2442\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |
| **Kênh 5** | `right_hip_roll` | $[-0.44, +0.44]$ | $1220\mu\text{s} \leftrightarrow 1780\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |
| **Kênh 6** | `right_hip_pitch` | $[-0.87, +0.87]$ | $945\mu\text{s} \leftrightarrow 2055\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |
| **Kênh 7** | `right_knee_pitch`| $[0.0, +1.48]$ | $1500\mu\text{s} \leftrightarrow 2442\mu\text{s}$ | $1500\mu\text{s}$ ($0^\circ$) |

---

## 5. Kiến Trúc Bộ Lọc Quán Tính IMU 3 Tầng & Mahony AHRS (Locomotion Stability)

Trên nền tảng Raspberry Pi 5, tác vụ tiếp nhận và lọc cảm biến quán tính được thực thi qua lớp `MPU6050FilterReader` trong `pipelines/rpi5_stack/deployment/infer_rpi5.py`. Với 4 nhân Cortex-A76 @ 2.4GHz, hệ thống xử lý bộ lọc quán tính 3 tầng kết hợp Mahony AHRS với độ trễ cực thấp:

```
    ┌────────────────────────────────────────────────────────┐
    │       Cảm biến MPU6050 trên I2C-1 (Bus 1, Addr 0x68)   │
    └───────────────────────────┬────────────────────────────┘
                                │ I2C Fast-Mode 400kHz
    ┌───────────────────────────▼────────────────────────────┐
    │  TẦNG 1: Hardware DLPF (Digital Low-Pass Filter)       │
    │  - Thanh ghi CONFIG (0x1A) = 0x03                      │
    │  - Tần số cắt: 42Hz (Gyro) / 44Hz (Accel), trễ ~4.8ms  │
    │  - Khử triệt để rung chấn servo MG996R & quạt làm mát  │
    └───────────────────────────┬────────────────────────────┘
                                │
    ┌───────────────────────────▼────────────────────────────┐
    │  TẦNG 2: Calibration Độ Lệch & Bộ Lọc Số IIR Gyro      │
    │  - Thu thập 100 mẫu tĩnh lúc khởi động để trừ bias     │
    │  - Bộ lọc IIR: w_filt = 0.70*w_filt + 0.30*w_unbiased  │
    │  - Cung cấp 3 chiều base_ang_vel mượt mà vào obs[0..2] │
    └───────────────────────────┬────────────────────────────┘
                                │
    ┌───────────────────────────▼────────────────────────────┐
    │  TẦNG 3: Vectorized Mahony AHRS & Impact Gating        │
    │  - Tính toán ma trận/quaternion vector hóa NumPy       │
    │  - Deadband gating: | ||a|| - 1.0g | <= 0.30g          │
    │    * Khi va chạm (>1.30g / <0.70g): ngắt gia tốc kế     │
    │    * Trạng thái chuẩn: Kp=1.0, Ki=0.05 hiệu chỉnh trôi │
    │  - Chiếu trọng lực MuJoCo: proj_grav = R^T * [0,0,-1]  │
    │  - Xuất 3 chiều proj_grav vào obs[3..5]                │
    └────────────────────────────────────────────────────────┘
```

### 5.1. Tầng 1: Bộ lọc thông thấp phần cứng MPU6050 DLPF
- **Thanh ghi cấu hình:** Thiết lập thanh ghi `CONFIG (0x1A) = 0x03` qua giao tiếp `smbus2`.
- **Tần số cắt:** $42\text{ Hz}$ cho Gyroscope và $44\text{ Hz}$ cho Accelerometer (thời gian trễ nhóm $4.8\text{ ms}$).
- **Hiệu quả:** Triệt tiêu trực tiếp nhiễu rung cơ học cao tần sinh ra từ 8 servo MG996R và quạt tản nhiệt Active Cooler của Pi 5 ngay tại bộ chuyển đổi phần cứng của cảm biến.

### 5.2. Tầng 2: Hiệu chuẩn độ lệch tĩnh & Bộ lọc phần mềm IIR Gyro
1. **Startup Zero-Rate Bias Calibration:**
   - Khi dịch vụ `infer_rpi5.py` khởi chạy, hàm `_calibrate_gyro(samples=100)` đọc 100 mẫu khi robot đứng yên để xác định độ lệch không đổi $\mathbf{b}_{\text{gyro}}$.
   - Giá trị này được trừ trực tiếp trong thời gian thực:
     $$\mathbf{\omega}_{\text{unbiased}} = \left(\frac{\text{raw\_gyro}}{131.0} \cdot \frac{\pi}{180.0}\right) - \mathbf{b}_{\text{gyro}}$$
2. **Bộ lọc IIR cho Vận tốc góc thân:**
   - Vận tốc góc được làm mượt qua bộ lọc số IIR bậc 1 với hệ số $\alpha_{\text{gyro}} = 0.70$:
     $$\mathbf{\omega}_{\text{filt}}[t] = 0.70 \cdot \mathbf{\omega}_{\text{filt}}[t-1] + 0.30 \cdot \mathbf{\omega}_{\text{unbiased}}[t]$$
   - Cung cấp 3 thành phần `base_ang_vel` tin cậy vào vector quan sát `obs[0..2]`.

### 5.3. Tầng 3: Vectorized Mahony AHRS & Ngắt va chạm (Dynamic Foot-Strike Impact Gating)
1. **Thuật toán Mahony AHRS Vector Hóa:**
   - Trạng thái tư thế được lưu trữ dưới dạng quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]$ chuẩn hóa, lặp ở tần số $50\text{ Hz}$ ($\Delta t = 20\text{ ms}$).
   - Tận dụng tập lệnh ARMv8 NEON SIMD qua NumPy, thời gian thực thi thuật toán Mahony trên Pi 5 chỉ mất $< 0.08\text{ ms}$.
2. **Cơ Chế Dynamic Foot-Strike Impact Gating (Deadband 0.30g):**
   - **Ngưỡng kiểm tra:** $\text{impact\_threshold} = 0.30\text{g}$ quanh $1.0\text{g}$ ($0.70\text{g} \le \|\mathbf{a}\| \le 1.30\text{g}$).
   - **Khi va chạm tiếp đất (Foot Strike Shock):** Gia tốc kế đo được các đỉnh xung kích từ $1.5\text{g} - 2.5\text{g}$ ($|\|\mathbf{a}\| - 1.0\text{g}| > 0.30\text{g}$). Hệ thống ngắt hoàn toàn hiệu chỉnh gia tốc kế ($K_p = 0, K_i = 0$), giữ nguyên sai số tích phân và chỉ dùng vận tốc góc con quay hồi chuyển để tích phân quaternion tư thế, ngăn ngừa hoàn toàn hiện tượng méo góc khi tiếp đất.
   - **Khi di chuyển ổn định:** Cổng mở, hệ thống áp dụng hệ số tỉ lệ $K_p = 1.0$ và hệ số tích phân $K_i = 0.05$ để triệt tiêu trôi dạt góc dài hạn.
3. **Tính toán Trọng Lực Chiếu MuJoCo (Projected Gravity):**
   - Phương thức `get_proj_grav()` trích xuất vector trọng lực chiếu từ quaternion hiện tại:
     $$\mathbf{g}_{\text{proj}} = \begin{bmatrix} 2(q_x q_z - q_w q_y) \\ 2(q_y q_z + q_w q_x) \\ -(q_w^2 - q_x^2 - q_y^2 + q_z^2) \end{bmatrix}$$
   - Ở tư thế đứng thẳng cân bằng, $\mathbf{g}_{\text{proj}} = [0.0, 0.0, -1.0]$, đồng nhất tuyệt đối với mô hình huấn luyện mô phỏng MuJoCo.
4. **Hỗ trợ Chế Độ Mô Phỏng & CI/CD Mock Mode:**
   - Khi chạy với cờ `--mock` hoặc trên máy trạm không có phần cứng I2C vật lý, `MPU6050FilterReader` tự động trả về vector tĩnh $[0, 0, 0]$ cho vận tốc góc và $[0, 0, -1]$ cho trọng lực chiếu, đảm bảo kiểm thử toàn diện vượt qua 100%.

---

## 6. Quy Trình Vận Hành & An Toàn Thực Địa

1. **Khởi động an toàn (Power-up Sequence):**
   - Bật công tắc mạch logic vi xử lý (Buck 5V) $\rightarrow$ Chờ Pi 5 boot hoàn tất hệ điều hành Raspberry Pi OS 64-bit và nạp dịch vụ `infer_rpi5.service`.
   - Bật công tắc mạch động lực Servo (Buck 6V). PCA9685 bắt đầu xuất xung giữ tư thế đứng thẳng (Neutral Stand).
2. **Cơ chế khẩn cấp (Emergency Stop / Watchdog):**
   - Khi góc nghiêng Pitch/Roll của robot vượt quá $\pm 45^\circ$ (ngã đổ), phần mềm lập tức ngắt xung PWM (kéo chân OE lên mức cao hoặc xuất xung 0) để bảo vệ bánh răng hộp số MG996R khỏi bị kẹt gãy do quá tải.
3. **Giới hạn tốc độ góc (Velocity Rate Limiter):**
   - Đảm bảo bước nhảy góc đặt không vượt quá $\Delta \theta_{\max} = 6.0\text{ rad/s} \times 0.02\text{ s} = 0.12\text{ rad} \approx 6.88^\circ$ trong mỗi chu kỳ $20\text{ms}$.

