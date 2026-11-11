# Tài Liệu Đặc Tả Phần Cứng: Nhánh Raspberry Pi Zero 2 W (MicroDuck 8-DOF)

**Dự án:** Robot Bipedal 8 bậc tự do MicroDuck 8-DOF  
**Nhánh phần cứng:** `pipelines/pizero2w_stack`  
**Bộ xử lý trung tâm:** Raspberry Pi Zero 2 W (Broadcom BCM2710A1 Quad-core 64-bit Cortex-A53 @ 1.0GHz, 512MB LPDDR2 RAM)  
**Tổng khối lượng danh định:** $\approx 745\text{ g} - 755\text{ g}$ (Bao gồm pin 2S 2200mAh/18650, module Buck 5V/2.5A, tản nhiệt nhôm phẳng, bo mở rộng PCA9685 và 8 servo MG996R)  

---

## 1. Kiến Trúc Hệ Thống & Phân Phối Nguồn Điện (Power Distribution)

Do sử dụng 8 servo công suất lớn TowerPro MG996R (dòng đỉnh stall lên đến $2.5\text{A}$ mỗi servo) và bo máy tính Raspberry Pi Zero 2 W, việc tách biệt hoàn toàn 2 đường nguồn cấp (Đường Logic số & Đường Động lực servo) là nguyên tắc an toàn bắt buộc nhằm triệt tiêu hiện tượng sụt áp (brownout) và triệt tiêu xung điện cảm ứng ngược (inductive back-EMF).

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
            │ Module Buck Mini-360 5V/2.5A│   │  Module Buck DC-DC 6V/15A  │
            │   (hoặc MP1584EN nhỏ gọn)  │   │  (Hoặc UBEC máy bay mô hình)│
            └──────────────┬─────────────┘   └──────────────┬─────────────┘
                           │ 5.1V / 2.5A                    │ 6.0V / 15.0A
            ┌──────────────▼─────────────┐                  │
            │   Raspberry Pi Zero 2 W    │                  │
            │  (Cấp qua USB/GPIO Pin 2,4)│                  │
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

1. **Nguồn cấp cho Raspberry Pi Zero 2 W:**
   - Điện áp: $5.1\text{V} \pm 0.1\text{V}$.
   - Dòng điện danh định: $1.5\text{A} - 2.5\text{A}$ (Pi Zero 2 W tiêu thụ trung bình chỉ $0.4\text{A} - 0.7\text{A}$ khi chạy suy luận 4 nhân A53; công suất đỉnh khi stream WiFi/Camera đạt tối đa $\approx 1.2\text{A}$).
   - Module hạ áp: Sử dụng module Buck Mini-360 hoặc MP1584EN siêu nhẹ ($5\text{ g}$) cho hiệu suất $\ge 92\%$, giảm tối đa diện tích và khối lượng khoang ngực robot.
   - Điểm nối: Cổng Micro-USB PWR IN hoặc chân GPIO 5V (Pin 2, Pin 4) và GND (Pin 6).

2. **Nguồn cấp cho dàn động lực 8x MG996R:**
   - Điện áp: $6.0\text{V}$ (tối ưu hóa tốc độ $0.16\text{ s}/60^\circ$ và mô-men xoắn $11.0\text{ kg}\cdot\text{cm}$ của servo MG996R).
   - Dòng điện danh định: $12\text{A} - 15\text{A}$ tải đỉnh khi tiếp đất và đảo chiều bước đi.
   - Tụ điện lọc nhiễu: Hàn song song tụ hóa Low-ESR $4700\mu\text{F} / 16\text{V}$ và tụ gốm $100\text{nF}$ ngay tại cọc nguồn cấp servo của bo PCA9685.
   - Nguyên tắc nối mass: Mass Logic của Pi Zero 2 W và Mass Động lực nối chung duy nhất tại 1 điểm (Star Ground Topology) để tránh ground loop.

---

## 2. Sơ Đồ Đấu Nối Chân GPIO (Raspberry Pi Zero 2 W Pinout Mapping)

Raspberry Pi Zero 2 W sử dụng chuẩn chân 40-pin GPIO tiêu chuẩn của hệ sinh thái Raspberry Pi:

| Tên Ngoại Vi | Tín Hiệu | Chân Pi Zero 2 W (Physical) | Tên GPIO (BCM) | Ghi Chú & Đặc Tính |
| :--- | :--- | :--- | :--- | :--- |
| **PCA9685 (I2C Servo)** | VCC (Logic) | Pin 1 | 3.3V Power | Nguồn nuôi chip PCA9685 (3.3V logic) |
| | SDA | Pin 3 | GPIO 2 (SDA1) | Đường truyền dữ liệu I2C (Bus 1, 400kHz Fast Mode) |
| | SCL | Pin 5 | GPIO 3 (SCL1) | Đường xung nhịp I2C (Bus 1) |
| | GND | Pin 6 | Ground | Mass chung |
| | OE (Output Enable) | Pin 11 | GPIO 17 | Tích cực mức thấp (kéo GND để mở PWM) |
| **MPU6050 (6-DOF IMU)** | VDD | Pin 1 | 3.3V Power | Cấp nguồn nuôi cảm biến 3.3V |
| | SDA | Pin 3 | GPIO 2 (SDA1) | Đường dữ liệu I2C (Địa chỉ 0x68) |
| | SCL | Pin 5 | GPIO 3 (SCL1) | Đường xung nhịp I2C (Bus 1, 400kHz) |
| | GND | Pin 6 | Ground | Mass chung |
| **INMP441 (I2S Mic)** | VDD | Pin 17 | 3.3V Power | Nguồn nuôi microphone số MEMS |
| | GND | Pin 20 | Ground | Mass tín hiệu âm thanh |
| | SD (Serial Data) | Pin 38 | GPIO 20 (PCM_DIN) | Dữ liệu âm thanh số 24-bit PCM |
| | WS / LRCL | Pin 35 | GPIO 19 (PCM_FS) | Tín hiệu chọn kênh Word Select |
| | SCK / BCLK | Pin 12 | GPIO 18 (PCM_CLK) | Xung nhịp đồng hồ lấy mẫu bit |
| | L/R Select | Pin 14 | Ground | Chọn kênh Mono trái |
| **Cụm 5 Tia ToF LiDAR** | VCC | Pin 2 | 5V Power | Nguồn cấp qua IC ổn áp 3.3V nội bộ cảm biến |
| | SDA | Pin 27 | GPIO 0 (ID_SD) / I2C1 | Giao tiếp I2C qua IC chia kênh TCA9548A |
| | SCL | Pin 28 | GPIO 1 (ID_SC) / I2C1 | Xung nhịp quét ToF LiDAR |
| | XSHUT_1..5 | Pin 13, 15, 16, 22, 29 | GPIO 27, 22, 23, 25, 5 | Chân reset địa chỉ I2C tuần tự cho 5 sensor |

---

## 3. Cụm Nhận Thức Đầu Vịt (Perception Suite)

### 3.1. Camera quang học (MIPI CSI-2 Mini)
- **Chuẩn kết nối:** Cổng ribbon FPC 15-pin pitch 0.5mm tích hợp sẵn trên bo Pi Zero 2 W (sử dụng cáp chuyển đổi 22-pin sang 15-pin hoặc module camera thiết kế riêng cho dòng Pi Zero).
- **Cảm biến khuyến nghị:** Raspberry Pi Camera Module V2 (Sony IMX219 8MP) hoặc Camera Module 3 góc nhìn $60^\circ - 75^\circ$ FOV.
- **Tốc độ khung hình:** Thiết lập $640\times 480 @ 30\text{ FPS}$ cho độ trễ truyền dữ liệu $< 20\text{ ms}$.

### 3.2. Microphone số I2S INMP441
- **Vị trí lắp đặt:** Bố trí bên trong khoang mỏ vịt hướng thẳng về trước.
- **Cấu hình trên Linux (`/boot/firmware/config.txt`):**
  ```ini
  # Kích hoạt driver I2S audio trên Pi Zero 2 W
  dtoverlay=googlevoicehat-soundcard
  dtoverlay=i2s-mmap
  ```

---

## 4. Kiến Trúc Bộ Lọc Quán Tính IMU 3 Tầng & Mahony AHRS (Locomotion Stability)

Trên nền tảng Raspberry Pi Zero 2 W, tác vụ đọc và lọc dữ liệu quán tính được hiện thực hóa qua lớp `MPU6050FilterReader` trong `pipelines/pizero2w_stack/deployment/infer_pizero2w.py`. Để khắc phục rung chấn động cơ servo MG996R và lực sốc tiếp đất của robot hai chân, hệ thống triển khai kiến trúc 3 tầng:

```
    ┌────────────────────────────────────────────────────────┐
    │       Cảm biến MPU6050 trên I2C-1 (Bus 1, Addr 0x68)   │
    └───────────────────────────┬────────────────────────────┘
                                │ I2C Fast-Mode 400kHz
    ┌───────────────────────────▼────────────────────────────┐
    │  TẦNG 1: Hardware DLPF (Digital Low-Pass Filter)       │
    │  - Thanh ghi CONFIG (0x1A) = 0x03                      │
    │  - Tần số cắt: 42Hz (Gyro) / 44Hz (Accel), trễ ~4.8ms  │
    │  - Triệt tiêu sóng hài cơ học của 8 động cơ MG996R     │
    └───────────────────────────┬────────────────────────────┘
                                │
    ┌───────────────────────────▼────────────────────────────┐
    │  TẦNG 2: Calibration Độ Lệch & Bộ Lọc Số IIR Gyro      │
    │  - Lấy mẫu tĩnh 100 mẫu khi khởi động để trừ bias      │
    │  - Bộ lọc IIR: w_filt = 0.70*w_filt + 0.30*w_unbiased  │
    │  - Xuất 3 chiều base_ang_vel mượt mà vào obs[0..2]     │
    └───────────────────────────┬────────────────────────────┘
                                │
    ┌───────────────────────────▼────────────────────────────┐
    │  TẦNG 3: Vectorized Mahony AHRS & Impact Gating        │
    │  - Tính toán ma trận/quaternion tối ưu hóa qua NumPy   │
    │  - Deadband gating: | ||a|| - 1.0g | <= 0.30g          │
    │    * Tiếp đất (>1.30g hoặc <0.70g): ngắt gia tốc kế     │
    │    * Bình thường: Kp=1.0, Ki=0.05 hiệu chỉnh trôi gyro  │
    │  - Chiếu trọng lực MuJoCo: proj_grav = R^T * [0,0,-1]  │
    │  - Xuất 3 chiều proj_grav vào obs[3..5]                │
    └────────────────────────────────────────────────────────┘
```

### 4.1. Tầng 1: Bộ lọc phần cứng thông thấp MPU6050 DLPF
- **Thanh ghi cấu hình:** Thiết lập thanh ghi `CONFIG (0x1A) = 0x03` qua thư viện `smbus2`.
- **Tần số cắt:** $42\text{ Hz}$ cho con quay hồi chuyển và $44\text{ Hz}$ cho gia tốc kế (độ trễ nhóm $4.8\text{ ms}$).
- **Ưu điểm:** Loại bỏ ngay từ tầng vật lý các nhiễu tần số cao phát sinh do rung động bánh răng servo và xung kích cơ khí khi bước đi trước khi dữ liệu được CPU Cortex-A53 đọc vào.

### 4.2. Tầng 2: Hiệu chuẩn độ lệch tĩnh & Bộ lọc số IIR Gyro
1. **Startup Zero-Rate Bias Calibration:**
   - Khi khởi tạo `MPU6050FilterReader`, hàm `_calibrate_gyro(samples=100)` thu thập 100 mẫu liên tiếp trong trạng thái robot đứng yên danh định.
   - Vector độ lệch tĩnh $\mathbf{b}_{\text{gyro}}$ được lưu lại và tự động trừ khử khỏi dữ liệu con quay hồi chuyển thô trong mọi bước điều khiển:
     $$\mathbf{\omega}_{\text{unbiased}} = \left(\frac{\text{raw\_gyro}}{131.0} \cdot \frac{\pi}{180.0}\right) - \mathbf{b}_{\text{gyro}}$$
2. **Bộ lọc số IIR cho Vận tốc góc:**
   - Vận tốc góc sau khi trừ bias được lọc tiếp qua bộ lọc số IIR bậc 1 với hệ số suy giảm $\alpha_{\text{gyro}} = 0.70$:
     $$\mathbf{\omega}_{\text{filt}}[t] = 0.70 \cdot \mathbf{\omega}_{\text{filt}}[t-1] + 0.30 \cdot \mathbf{\omega}_{\text{unbiased}}[t]$$
   - Đảm bảo 3 thành phần `base_ang_vel` trong vector quan sát 42 chiều (`obs[0..2]`) luôn ổn định và không chứa gai xung nhiễu.

### 4.3. Tầng 3: Vectorized Mahony AHRS & Ngắt va chạm (Dynamic Foot-Strike Impact Gating)
1. **Thuật toán Mahony AHRS Vector Hóa:**
   - Biểu diễn tư thế bằng quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]$ chuẩn hóa, cập nhật ở chu kỳ $\Delta t = 0.02\text{ s}$ ($50\text{ Hz}$).
   - Sử dụng các phép toán mảng vector hóa của NumPy trên 4 nhân ARM Cortex-A53, cho thời gian tính toán $< 0.15\text{ ms}$ mỗi bước lặp.
2. **Cơ Chế Dynamic Foot-Strike Impact Gating (Deadband 0.30g):**
   - **Ngưỡng kiểm tra:** $\text{impact\_threshold} = 0.30\text{g}$ quanh $1.0\text{g}$.
   - **Khi chân tiếp đất (Foot-Strike Shock Spike):** Chuẩn gia tốc $\|\mathbf{a}\|$ vọt lên $1.5\text{g} - 2.5\text{g}$ ($|\|\mathbf{a}\| - 1.0\text{g}| > 0.30\text{g}$). Lúc này hiệu chỉnh gia tốc kế được ngắt hoàn toàn ($K_p = 0, K_i = 0$), giữ nguyên sai số tích phân và chỉ dùng vận tốc góc gyro đã khử trôi để cập nhật tư thế. Điều này ngăn chặn hiện tượng lật góc giả làm méo vector quan sát.
   - **Khi di chuyển ổn định:** $|\|\mathbf{a}\| - 1.0\text{g}| \le 0.30\text{g}$, hệ thống kích hoạt hiệu chỉnh tỉ lệ ($K_p = 1.0$) và tích phân ($K_i = 0.05$) với sai số có hướng $\mathbf{e} = \mathbf{a}_{\text{norm}} \times \mathbf{v}$, triệt tiêu trôi góc con quay hồi chuyển theo thời gian dài.
3. **Tính toán Trọng Lực Chiếu MuJoCo (Projected Gravity):**
   - Phương thức `get_proj_grav()` trích xuất vector trọng lực chiếu từ quaternion hiện tại:
     $$\mathbf{g}_{\text{proj}} = \begin{bmatrix} 2(q_x q_z - q_w q_y) \\ 2(q_y q_z + q_w q_x) \\ -(q_w^2 - q_x^2 - q_y^2 + q_z^2) \end{bmatrix}$$
   - Trả về chính xác $[0.0, 0.0, -1.0]$ khi robot đứng thẳng, khớp 100% với không gian quan sát trong mô phỏng huấn luyện.
4. **Hỗ trợ Chế Độ Mô Phỏng & Kiểm Thử Ngoại Tuyến (Mock Mode):**
   - Khi chạy với cờ `--mock` hoặc khi không phát hiện phần cứng bus I2C vật lý, `MPU6050FilterReader` tự động chuyển sang chế độ giả lập an toàn, trả về vector tĩnh $[0, 0, 0]$ cho vận tốc góc và $[0, 0, -1]$ cho trọng lực chiếu, đảm bảo pipeline kiểm thử CI/CD chạy 100% không phụ thuộc phần cứng.

---

## 5. Quản Lý Nhiệt Độ & Khối Lượng (Thermal & Weight Optimization)

1. **Giải pháp tản nhiệt thụ động (Passive Thermal Solution):**
   - SoC BCM2710A1 trang bị tản nhiệt nhôm phẳng anodized ($14\text{ mm} \times 14\text{ mm} \times 6\text{ mm}$) dán bằng keo dẫn nhiệt 3M.
   - Do mức tiêu thụ công suất toàn tải chỉ $\approx 2.5\text{W}$, tản nhiệt thụ động duy trì nhiệt độ SoC dưới $62^\circ\text{C}$ trong điều kiện phòng $28^\circ\text{C}$ mà không cần quạt tản nhiệt, loại bỏ hoàn toàn rung chấn cơ học lên cảm biến IMU.

2. **Phân bổ khối lượng chi tiết (Weight Breakdown):**
   - Bo mạch Pi Zero 2 W: $16\text{ g}$
   - Tản nhiệt nhôm phẳng: $8\text{ g}$
   - Module Buck Mini-360 (5V/2.5A): $5\text{ g}$
   - Bo mở rộng servo PCA9685: $10\text{ g}$
   - Hệ thống dây cáp nối và giắc cắm: $16\text{ g}$
   - **Tổng cụm điện tử thân trên:** **$55\text{ g}$** (nhẹ hơn $85\text{ g}$ so với cụm $140\text{ g}$ của Pi 5).
   - Thân cơ sở (khung in 3D + khớp nối): $220\text{ g}$
   - Bộ 8 servo MG996R: $440\text{ g}$
   - Khối đầu, mắt, cổ: $30\text{ g}$
   - **Tổng khối lượng toàn thân robot:** **$\approx 745\text{ g} - 755\text{ g}$**.
