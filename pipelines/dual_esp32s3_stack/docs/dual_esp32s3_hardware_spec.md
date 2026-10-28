# Tài Liệu Đặc Tả Phần Cứng: Nhánh Dual ESP32-S3 (MicroDuck 8-DOF)

**Dự án:** Robot Bipedal 8 bậc tự do MicroDuck 8-DOF  
**Nhánh phần cứng:** `pipelines/dual_esp32s3_stack`  
**Bộ vi xử lý:** Phân tầng 2 vi điều khiển ESP32-S3 DevKit N16R8 (Xtensa Dual-Core LX7 @ 240MHz, 16MB Flash, 8MB Octal PSRAM)  
- **MCU 1 (Perception Brain):** Thị giác (Camera), Âm thanh (I2S Mic), Quét 5 tia ToF LiDAR & Định vị chướng ngại vật.  
- **MCU 2 (Locomotion Brain):** Đọc cảm biến IMU MPU6050, lắp ghép vector quan sát 42 chiều, chạy mạng nơ-ron điều khiển dáng đi 50Hz, xuất xung PWM điều khiển 8 servo MG996R.  
**Tổng khối lượng danh định:** $\approx 780\text{ g}$ (Nhẹ hơn nhánh Pi 5 $\sim 80\text{ g}$, trọng tâm thấp hơn $\sim 10\text{ mm}$ giúp bipedal thăng bằng vượt trội).  

---

## 1. Kiến Trúc Mạch Cơ Sở (Carrier Board Topology) & Phân Tách Đường Nguồn

Hệ thống sử dụng bo mạch mẹ phân tầng (Custom Carrier Board) gắn 2 module ESP32-S3 song song. Để triệt tiêu hoàn toàn hiện tượng sụt áp nguồn số do dòng khởi động của 8 động cơ MG996R, hệ thống phân chia thành 3 tầng nguồn độc lập:

```
               ┌────────────────────────────────────────────────────────┐
               │    Khối Nguồn Chính: Pin LiPo/Li-ion 2S (7.4V - 8.4V)   │
               │         Dung lượng: 2200mAh - 25C (Xả liên tục 55A)     │
               └───────────┬────────────────────────────────┬───────────┘
                           │                                │
                (Đường Nguồn Số - Logic)          (Đường Động Lực - Servos)
                           │                                │
            ┌──────────────▼─────────────┐   ┌──────────────▼─────────────┐
            │   Module Buck DC-DC 5V/3A  │   │  Module Buck DC-DC 6V/15A  │
            │   (TPS54331 / MP2307)      │   │  (UBEC chuyên dụng 15A max)│
            └──────────────┬─────────────┘   └──────────────┬─────────────┘
                           │ 5.0V Logic Rail                │ 6.0V High Current
            ┌──────────────┴─────────────┐                  │
            │                            │                  │
    ┌───────▼────────┐           ┌───────▼────────┐         │
    │  LDO 3.3V #1   │           │  LDO 3.3V #2   │         │
    │ (AMS1117-3.3)  │           │ (AMS1117-3.3)  │         │
    └───────┬────────┘           └───────┬────────┘         │
            │ 3.3V (1A max)              │ 3.3V (1A max)    │
    ┌───────▼────────┐           ┌───────▼────────┐         │
    │     MCU 1      │  UART2    │     MCU 2      │         │
    │  (Perception)  ├───────────┤  (Locomotion)  │         │
    │  Camera + Mic  │ @921600   │  IMU + 8x PWM  │         │
    └───────┬────────┘           └───────┬────────┘         │
            │                            │                  │
    [Camera + I2S Mic]                   │                  │
    [5-Ray ToF LiDAR]                    └──────────┬───────┘
                                                    │ 8x Kênh Xung PWM (LEDC)
                                                    ▼
                                      ┌───────────────────────────┐
                                      │ Cụm 8x Servo TowerPro     │
                                      │ MG996R (Tụ 4700uF/16V)    │
                                      └───────────────────────────┘
```

### 1.1. Chi tiết phân tầng điện áp

1. **Đường nguồn động lực 6.0V (Servo Power Rail):**
   - Cung cấp bởi mạch Buck công suất cao $6\text{V} / 15\text{A}$ đỉnh trực tiếp từ pin 2S.
   - Trang bị tụ hóa Low-ESR dung lượng cao $4700\mu\text{F} / 16\text{V}$ kết hợp tụ gốm $100\text{nF}$ đặt sát chân cấp nguồn của dàn servo.
   - Dây dẫn nguồn servo sử dụng đồng bọc silicon chịu nhiệt chuẩn 16-18 AWG.

2. **Đường nguồn số 3.3V (MCU & Sensor Rails):**
   - Pin 2S qua module Buck hạ xuống $5.0\text{V}$, sau đó chia làm 2 nhánh qua 2 IC LDO $3.3\text{V}$ riêng biệt cho MCU 1 và MCU 2. Cách ly này ngăn nhiễu sóng RF của Wi-Fi/Bluetooth hoặc camera ảnh hưởng sang đường clock của cảm biến góc nghiêng MPU6050.

3. **Chống vòng lặp mass (Star Grounding Layout):**
   - Tất cả các đường GND của Servo, Buck, MCU 1, MCU 2 và cảm biến hội tụ về một điểm duy nhất (Star Ground Node) ngay tại cực âm của pack pin.

---

## 2. Giao Tiếp UART Liên MCU (Inter-MCU High-Speed Link)

Hai vi điều khiển ESP32-S3 giao tiếp thông qua phần cứng UART2 chuyên dụng với cấu hình tốc độ cao:

- **Baud rate:** $921,600\text{ bps}$ (Truyền 1 byte mất $\approx 10.85\mu\text{s}$).
- **Cấu hình khung:** 8 Data bits, No Parity, 1 Stop bit (8N1).
- **Giao thức:** Nhị phân có kiểm tra tính toàn vẹn CRC16-CCITT (định nghĩa trong `uart_packet_protocol.h`).
- **Thời gian truyền gói:**
  - `LidarPerceptionPacket` (30 bytes wire): $\sim 325\mu\text{s}$ ($\ll 20\text{ms}$ chu kỳ điều khiển, chiếm dưới $1.7\%$ băng thông).
  - `VelocityCmdPacket` (20 bytes wire): $\sim 217\mu\text{s}$.

| Tín Hiệu | Chân MCU 1 (GPIO) | Chân MCU 2 (GPIO) | Hướng Dữ Liệu |
| :--- | :--- | :--- | :--- |
| **UART_TX2** | GPIO 17 | GPIO 18 (RX2) | MCU 1 $\rightarrow$ MCU 2 (Perception & Command) |
| **UART_RX2** | GPIO 18 | GPIO 17 (TX2) | MCU 2 $\rightarrow$ MCU 1 (Telemetry & Heartbeat) |
| **GND** | GND | GND | Điểm chung tham chiếu logic |

---

## 3. Sơ Đồ Đấu Nối Ngoại Vi MCU 1 (Perception Brain)

MCU 1 tập trung toàn bộ năng lực phần cứng cho nhận thức môi trường (Camera, Mic, LiDAR):

| Chức Năng | Tên Tín Hiệu | Chân ESP32-S3 (GPIO) | Ghi Chú |
| :--- | :--- | :--- | :--- |
| **Camera (OV2640 DVP)** | CAM_Y2 - CAM_Y9 | GPIO 11, 9, 8, 10, 12, 14, 13, 21 | 8-bit DVP Data Bus |
| | CAM_XCLK | GPIO 15 | Xung master clock 20MHz |
| | CAM_PCLK | GPIO 16 | Xung pixel clock |
| | CAM_VSYNC | GPIO 6 | Tín hiệu đồng bộ khung |
| | CAM_HREF | GPIO 7 | Tín hiệu đồng bộ dòng |
| | CAM_SIOD / SIOC | GPIO 4, GPIO 5 | I2C SCCB điều khiển cảm biến (độc quyền) |
| **Micro I2S (INMP441)** | I2S_BCLK | GPIO 1 | Xung nhịp bit lấy mẫu âm thanh |
| | I2S_WS (LRCL) | GPIO 2 | Chọn kênh Word Select (Trái/Phải) |
| | I2S_DIN (SD) | GPIO 42 | Dữ liệu âm thanh số 24-bit trả về |
| | VDD / GND | 3.3V / GND | Nguồn nuôi micro âm thanh mỏ vịt |
| **LiDAR 5 Tia (ToF)** | I2C_SDA | GPIO 19 | Giao tiếp I2C đọc khoảng cách 5 hướng (tách rời SCCB Camera) |
| | I2C_SCL | GPIO 20 | Xung nhịp I2C 400kHz |
| | XSHUT_1..5 | GPIO 38, 39, 40, 41, 47 | Reset và định địa chỉ động 5 cảm biến |

---

## 4. Sơ Đồ Đấu Nối Ngoại Vi MCU 2 (Locomotion & Servo Brain)

MCU 2 chịu trách nhiệm đọc tư thế bipedal và điều khiển 8 servo MG996R qua phần cứng LEDC PWM:

| Ngoại Vi | Chân ESP32-S3 (GPIO) | Kênh LEDC | Khớp Robot Tương Ứng | Giới Hạn Góc |
| :--- | :--- | :--- | :--- | :--- |
| **MPU6050 IMU** | GPIO 4 (SDA), GPIO 5 (SCL) | - | Trục góc nghiêng, gia tốc CoM | I2C @ 400kHz |
| **MG996R Servo 0** | GPIO 6 | Kênh 0 | `head_yaw` | $[-1.57, +1.57]\text{ rad}$ |
| **MG996R Servo 1** | GPIO 7 | Kênh 1 | `head_pitch` | $[-0.52, +0.78]\text{ rad}$ |
| **MG996R Servo 2** | GPIO 8 | Kênh 2 | `left_hip_roll` | $[-0.44, +0.44]\text{ rad}$ |
| **MG996R Servo 3** | GPIO 9 | Kênh 3 | `left_hip_pitch` | $[-0.87, +0.87]\text{ rad}$ |
| **MG996R Servo 4** | GPIO 10 | Kênh 4 | `left_knee_pitch` | $[0.0, +1.48]\text{ rad}$ |
| **MG996R Servo 5** | GPIO 11 | Kênh 5 | `right_hip_roll` | $[-0.44, +0.44]\text{ rad}$ |
| **MG996R Servo 6** | GPIO 12 | Kênh 6 | `right_hip_pitch` | $[-0.87, +0.87]\text{ rad}$ |
| **MG996R Servo 7** | GPIO 13 | Kênh 7 | `right_knee_pitch` | $[0.0, +1.48]\text{ rad}$ |

---

## 5. Quản Lý Pin & Mạch Bảo Vệ (Battery Management & Safety)

1. **Thông số Pack Pin:**
   - Cấu hình: $2\text{S}1\text{P}$ LiPo hoặc 2 cell $18650$ dung lượng $2200\text{mAh} - 2500\text{mAh}$.
   - Điện áp danh định: $7.4\text{V}$ (Đầy tải $8.4\text{V}$, ngắt bảo vệ $6.4\text{V}$).
2. **Mạch bảo vệ BMS (Battery Management System):**
   - Tích hợp bo bảo vệ $2\text{S} - 20\text{A}$ chống ngắn mạch, chống sạc quá dòng ($>20\text{A}$) và chống xả kiệt ($<3.0\text{V}$ mỗi cell).
3. **Cơ chế ngắt an toàn bằng phần mềm (Software Watchdog):**
   - MCU 2 theo dõi điện áp pin qua cầu phân áp vào chân ADC (GPIO 1). Nếu điện áp xuống dưới $6.6\text{V}$, robot tự động ngồi gập gối về trạng thái nghỉ và ngắt xung PWM để bảo vệ pin không bị phồng/cháy nổ.
   - Nếu mất gói tín hiệu từ MCU 1 quá $500\text{ms}$, MCU 2 tự động dừng bước và giữ thăng bằng tại chỗ.

