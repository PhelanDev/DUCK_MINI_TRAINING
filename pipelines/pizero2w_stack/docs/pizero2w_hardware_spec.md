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

## 4. Quản Lý Nhiệt Độ & Khối Lượng (Thermal & Weight Optimization)

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
