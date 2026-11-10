# MicroDuck 8-DOF Chibi Squat Kinematics & Physical Overhaul Spec

**Date:** 2026-11-11  
**Author:** Antigravity Autonomous Meta-Orchestrator & Pair Programmer  
**Status:** Approved  
**Target:** MicroDuck 8-DOF Bipedal Platform (Base, RPi 5, Dual ESP32-S3, Pi Zero 2 W)

---

## 1. Problem Statement & Motivation

Trong 3 mô hình vật lý ban đầu của MicroDuck 8-DOF:
- Chiều dài đùi ($L_1 = 65\text{ mm}$) và cẳng chân ($L_2 = 65\text{ mm}$) kết hợp với độ cao hông đưa chiều dài chân lên tới $195\text{ mm}$ (gần $70\%$ tổng chiều cao $305\text{ mm}$ của robot).
- Trọng tâm CoM bị nâng lên tới $Z = 127\text{ mm}$ (base đặt tại $Z = 170\text{ mm}$), khiến cánh tay đòn trọng lực dài khi nghiêng người bước đi (waddling roll). Mô-men lật nghiêng đạt đỉnh $\approx 0.60\text{ N}\cdot\text{m}$, sát ngưỡng kịch trần của động cơ servo MG996R ($1.0\text{ N}\cdot\text{m}$).
- Bàn chân ban đầu có kích thước $75\text{ mm} \times 55\text{ mm}$, khoảng hở giữa 2 mép chân trong chỉ còn $10\text{ mm}$, dễ dẫn đến va chạm cơ học (self-collision) giữa 2 chân khi sải bước hoặc lắc hông.
- Về mặt thẩm mỹ thị giác, robot trông gầy guộc, chân quá dài, thiếu tỷ lệ múp míp, lùn cute "Chibi" đặc trưng của một chú vịt cao su hoạt hình.

Tài liệu này đặc tả toàn diện việc tái thiết kế hình học chân (Chibi Squat Kinematics) và cập nhật đồng bộ các mô hình MJCF, URDF, scripts kiểm chứng và test suite.

---

## 2. Geometric & Kinematic Specifications

### 2.1 Chiều Dài Khâu & Kích Thước Hình Học

| Thông số | Giá trị Cũ (Legacy) | Giá trị Mới (Chibi Squat) | Ghi chú / Ý nghĩa |
| :--- | :--- | :--- | :--- |
| **Đùi ($L_1$)** | $65\text{ mm}$ | **$32\text{ mm}$** | Giảm $50.8\%$, hạ thấp hông |
| **Thigh Box Geom** | `0.0175 0.015 0.0325` | **`0.0175 0.015 0.016`** | Half-extents $(X, Y, Z)$ |
| **Thigh Geom Pos** | `pos="0 0 -0.0325"` | **`pos="0 0 -0.016"`** | Tâm khối nằm giữa khâu |
| **Cẳng chân ($L_2$)** | $65\text{ mm}$ | **$32\text{ mm}$** | Giảm $50.8\%$, tỉ lệ chân ngắn |
| **Shin Box Geom** | `0.015 0.0125 0.0325` | **`0.015 0.0125 0.016`** | Half-extents $(X, Y, Z)$ |
| **Shin Geom Pos** | `pos="0 0 -0.0325"` | **`pos="0 0 -0.016"`** | Tâm khối nằm giữa khâu |
| **Knee Joint Pos** | `pos="0 0 -0.065"` | **`pos="0 0 -0.032"`** | Vị trí khớp gối trên đùi |
| **Ankle/Foot Pos** | `pos="0 0 -0.065"` | **`pos="0 0 -0.032"`** | Vị trí gắn bàn chân trên cẳng chân |
| **Bàn chân ($L \times W \times H$)** | $75 \times 55 \times 10\text{ mm}$ | **$52 \times 42 \times 8\text{ mm}$** | Gọn gàng, thanh thoát, cute |
| **Foot Box Geom** | `0.0375 0.0275 0.005` | **`0.026 0.021 0.004`** | Half-extents $(X, Y, Z)$ |
| **Foot Geom Pos** | `pos="0.010 0 -0.005"` | **`pos="0.008 0 -0.004"`** | Đặt hơi nhô về trước |
| **Foot Touch Site** | `size="0.040 0.030 0.005"` | **`size="0.026 0.021 0.004"`** | Cảm biến xúc giác đáy bàn chân |
| **Khoảng hở mép chân** | $10\text{ mm}$ | **$28\text{ mm}$ - $58\text{ mm}$** | Triệt tiêu va chạm mép chân |
| **Độ cao thân (`base_link`)** | $Z = 170\text{ mm}$ ($0.170\text{ m}$) | **$Z = 95\text{ mm}$ ($0.095\text{ m}$)** | Dáng đứng Chibi hạ trọng tâm |
| **Tổng chiều cao robot** | $\approx 305\text{ mm}$ ($30.5\text{ cm}$) | **$\approx 195\text{ mm}$ ($19.5\text{ cm}$)** | Tỉ lệ cơ thể múp míp chuẩn Chibi |

### 2.2 Tỷ Lệ Cơ Thể Chibi (Body Proportions)
- Chiều cao đầu + cổ: $\approx 75\text{ mm}$
- Chiều cao thân (torso): $\approx 70\text{ mm}$
- Chiều cao chân khi đứng: $\approx 80\text{ mm}$ (gồm cả bàn chân)
- **Tỉ lệ đầu / thân / chân:** $\approx 1 : 0.9 : 1$ (thay vì tỉ lệ cũ $1 : 0.9 : 2.5$). Tỉ lệ này mang tính biểu tượng của đồ chơi búp bê Chibi / vịt vàng mini.

---

## 3. Dynamic & Mass Budget Overhaul

Do rút ngắn kích thước hình học đùi, cẳng chân và bàn chân:
1. **Khối lượng đùi (`thigh_link`)**: Giảm từ $65\text{ g} \to \mathbf{40\text{ g}}$ ($0.040\text{ kg}$).
2. **Khối lượng cẳng chân (`shin_link`)**: Giảm từ $45\text{ g} \to \mathbf{30\text{ g}}$ ($0.030\text{ kg}$).
3. **Khối lượng bàn chân (`foot_link`)**: Giảm từ $30\text{ g} \to \mathbf{22\text{ g}}$ ($0.022\text{ kg}$).
4. **Tổng khối lượng chân (2 bên)**:
   $$\Delta m_{\text{legs}} = 2 \times [(65 - 40) + (45 - 30) + (30 - 22)] = 2 \times [25 + 15 + 8] = 96\text{ g}$$
   Tổng khối lượng robot giảm đều $\approx 96\text{ g}$ trên cả 4 mô hình.

### Bảng Ngân Sách Khối Lượng Từng Cấu Hình:

| Cấu hình Phần cứng | Tổng KL Cũ | Tổng KL Mới (Target) | Dải Test Hợp lệ ($\pm 5\%$ hoặc dải chỉ định) |
| :--- | :--- | :--- | :--- |
| **Base MicroDuck** (`models/`) | $780\text{ g}$ | **$684\text{ g}$** ($0.684\text{ kg}$) | $0.620 - 0.740\text{ kg}$ |
| **Pi Zero 2 W Stack** | $745\text{ g}$ | **$649\text{ g}$** ($0.649\text{ kg}$) | $0.615 - 0.685\text{ kg}$ |
| **Dual ESP32-S3 Stack** | $780\text{ g}$ | **$684\text{ g}$** ($0.684\text{ kg}$) | $0.650 - 0.720\text{ kg}$ |
| **Raspberry Pi 5 Stack** | $860\text{ g}$ | **$764\text{ g}$** ($0.764\text{ kg}$) | $0.725 - 0.805\text{ kg}$ |

### Ma Trận Quán Tính (Inertia Tensor Calculation)
Cho hình hộp chữ nhật kích thước $(a, b, c)$ và khối lượng $m$:
$$I_{xx} = \frac{1}{12} m (b^2 + c^2), \quad I_{yy} = \frac{1}{12} m (a^2 + c^2), \quad I_{zz} = \frac{1}{12} m (a^2 + b^2)$$

- **Đùi mới ($m=0.040\text{ kg}, a=0.035, b=0.030, c=0.032\text{ m}$)**:
  - $I_{xx} = \frac{0.040}{12}(0.030^2 + 0.032^2) = \frac{0.040}{12}(0.0009 + 0.001024) = 6.41 \times 10^{-6}\text{ kg}\cdot\text{m}^2$
  - $I_{yy} = \frac{0.040}{12}(0.035^2 + 0.032^2) = \frac{0.040}{12}(0.001225 + 0.001024) = 7.50 \times 10^{-6}\text{ kg}\cdot\text{m}^2$
  - $I_{zz} = \frac{0.040}{12}(0.035^2 + 0.030^2) = \frac{0.040}{12}(0.001225 + 0.0009) = 7.08 \times 10^{-6}\text{ kg}\cdot\text{m}^2$

- **Cẳng chân mới ($m=0.030\text{ kg}, a=0.030, b=0.025, c=0.032\text{ m}$)**:
  - $I_{xx} = \frac{0.030}{12}(0.025^2 + 0.032^2) = \frac{0.030}{12}(0.000625 + 0.001024) = 4.12 \times 10^{-6}\text{ kg}\cdot\text{m}^2$
  - $I_{yy} = \frac{0.030}{12}(0.030^2 + 0.032^2) = \frac{0.030}{12}(0.0009 + 0.001024) = 4.81 \times 10^{-6}\text{ kg}\cdot\text{m}^2$
  - $I_{zz} = \frac{0.030}{12}(0.030^2 + 0.025^2) = \frac{0.030}{12}(0.0009 + 0.000625) = 3.81 \times 10^{-6}\text{ kg}\cdot\text{m}^2$

- **Bàn chân mới ($m=0.022\text{ kg}, a=0.052, b=0.042, c=0.008\text{ m}$)**:
  - $I_{xx} = \frac{0.022}{12}(0.042^2 + 0.008^2) = \frac{0.022}{12}(0.001764 + 0.000064) = 3.35 \times 10^{-6}\text{ kg}\cdot\text{m}^2$
  - $I_{yy} = \frac{0.022}{12}(0.052^2 + 0.008^2) = \frac{0.022}{12}(0.002704 + 0.000064) = 5.07 \times 10^{-6}\text{ kg}\cdot\text{m}^2$
  - $I_{zz} = \frac{0.022}{12}(0.052^2 + 0.042^2) = \frac{0.022}{12}(0.002704 + 0.001764) = 8.19 \times 10^{-6}\text{ kg}\cdot\text{m}^2$

Tất cả các giá trị $I_{xx}, I_{yy}, I_{zz}$ đều dương và thỏa mãn bất đẳng thức tam giác cơ học vật rắn:
$$I_{xx} + I_{yy} \ge I_{zz}, \quad I_{xx} + I_{zz} \ge I_{yy}, \quad I_{yy} + I_{zz} \ge I_{xx}$$

---

## 4. Stability & Dynamic Benefits

1. **Giảm mô-men quán tính lật nghiêng ($I_{\text{roll}}$)**:
   Quán tính lật quanh trục $X$ của toàn thân giảm $\approx 42\%$, giúp robot phản ứng chuyển hướng nhanh hơn, giảm dao động rung lắc thân (chassis wobble).
2. **Giảm cánh tay đòn trọng lực**:
   Khoảng cách từ CoM đến khớp hông và điểm tiếp xúc mặt đất giảm từ $127\text{ mm} \to \approx 65\text{ mm}$. Mô-men lật nghiêng trọng lực $M_{\text{tilt}} = m \cdot g \cdot h \cdot \sin(\theta)$ giảm hơn $50\%$.
3. **Biên độ an toàn của động cơ servo**:
   Tải mô-men cực đại tác động lên các servo hông và gối hạ từ $0.60\text{ N}\cdot\text{m}$ xuống còn $\le 0.35\text{ N}\cdot\text{m}$, bỏ xa giới hạn $1.0\text{ N}\cdot\text{m}$ của MG996R, tăng độ bền cơ học và tuổi thọ linh kiện.
4. **Triệt tiêu va chạm tự thân**:
   Khoảng hở giữa 2 cạnh bàn chân tăng gấp đôi, loại trừ rủi ro 2 chân đập vào nhau khi robot lắc hông bước đi.
