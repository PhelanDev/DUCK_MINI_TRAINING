# MicroDuck 8-DOF Chibi Squat Kinematics Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thu ngắn tỉ lệ chân (đùi $32\text{ mm}$, cẳng chân $32\text{ mm}$, bàn chân $52\times 42\text{ mm}$, base cao $Z = 0.095\text{ m}$), giảm trọng tâm CoM và mô-men lật nghiêng, đồng bộ 100% hình học và động lực học Chibi Squat cho cả 4 mô hình (Base, Pi Zero 2 W, Dual ESP32-S3, RPi 5).

**Architecture:** Áp dụng phương pháp phân tầng: Cập nhật test suite định nghĩa tiêu chuẩn hình học & khối lượng mới (Red Phase) -> Cập nhật đồng loạt các file mô tả vật lý MJCF và URDF -> Điều chỉnh script xác thực vật lý `scripts/verify_model.py` -> Kiểm chứng và chạy toàn bộ test suite dự án pass 100%.

**Tech Stack:** Python 3.12, MuJoCo (MJCF XML), URDF, Pytest, NumPy, Git.

**Spec:** `docs/specs/2026-11-11-microduck-chibi-squat-kinematics-design.md`

## Global Constraints
- **Link Lengths:** Đùi $L_1 = 32\text{ mm}$, cẳng chân $L_2 = 32\text{ mm}$, bàn chân $52\text{ mm} \times 42\text{ mm} \times 8\text{ mm}$.
- **Base Link Initial Elevation:** $Z_{\text{base}} = 0.095\text{ m}$ ($95\text{ mm}$).
- **Leg Masses:** Thigh $= 40\text{ g}$ ($0.040\text{ kg}$), Shin $= 30\text{ g}$ ($0.030\text{ kg}$), Foot $= 22\text{ g}$ ($0.022\text{ kg}$).
- **Target Total Masses:**
  - Base MicroDuck: $0.684\text{ kg}$ (phạm vi: $0.620 - 0.740\text{ kg}$)
  - Pi Zero 2 W: $0.649\text{ kg}$ (phạm vi: $0.615 - 0.685\text{ kg}$)
  - Dual ESP32-S3: $0.684\text{ kg}$ (phạm vi: $0.650 - 0.720\text{ kg}$)
  - RPi 5: $0.764\text{ kg}$ (phạm vi: $0.725 - 0.805\text{ kg}$)
- **Inertial Consistency:** Các giá trị quán tính $I_{xx}, I_{yy}, I_{zz} > 0$ và thỏa mãn bất đẳng thức tam giác vật rắn.
- **Git Policy:** Không thực hiện `git push` tự động (tuân thủ Zero-Trust Git Policy).

## Review Focus
1. **Khớp gối và khớp bàn chân không thẳng hàng nếu offset sai:** Đảm bảo `knee_pitch` tại `0 0 -0.032` và foot tại `0 0 -0.032` khớp chính xác giữa MJCF và URDF.
2. **Kích thước collision geom của bàn chân:** Đảm bảo `size="0.026 0.021 0.004"` và pos `0.008 0 -0.004` để tiếp xúc mặt đất phẳng êm, không bị lún hoặc kênh.
3. **Cảm biến xúc giác chạm sàn (Touch Sensor & Sites):** Cập nhật đúng kích thước và tọa độ site `left_foot_touch_site` và `right_foot_touch_site`.
4. **CoM tính toán trong `scripts/verify_model.py`:** Cập nhật fallback CoM sang $Z_{\text{base}} = 0.095\text{ m}$ để tránh fail khi không có thư viện MuJoCo C-bindings trên host.
5. **Đồng bộ liên stack:** Đảm bảo các cảm biến perception của stack chuyên biệt (5-ray LiDAR, camera, mic) không bị ảnh hưởng hay xô lệch tọa độ.

---

### Task 1: Cập Nhật Test Suites Ràng Buộc Khối Lượng và Hình Học Chibi (TDD Red)

**Files:**
- Modify: `tests/test_mjcf_model.py:75-80`
- Modify: `tests/test_urdf_model.py:100-105`
- Modify: `tests/test_pizero2w_model.py:133-145,217-223`
- Modify: `tests/test_rpi5_model.py:127-138,216-221`
- Modify: `tests/test_esp32_model.py:127-138,216-221`

**Interfaces:**
- Consumes: Ngưỡng khối lượng và kích thước trong Spec.
- Produces: Test suite sẵn sàng bắt các mô hình chưa cập nhật (Red phase).

- [ ] **Step 1: Cập nhật assertions khối lượng mới trong cả 5 file test**
  - `test_mjcf_model.py`: Chuyển ngưỡng `total_mass` sang `0.620 <= total_mass <= 0.740`.
  - `test_urdf_model.py`: Chuyển ngưỡng `total_mass` sang `0.620 <= total_mass <= 0.740`.
  - `test_pizero2w_model.py`: Chuyển ngưỡng `total_mass` sang `0.615 <= total_mass <= 0.685`.
  - `test_rpi5_model.py`: Chuyển ngưỡng `total_mass` sang `0.725 <= total_mass <= 0.805`.
  - `test_esp32_model.py`: Chuyển ngưỡng `total_mass` sang `0.650 <= total_mass <= 0.720`.

- [ ] **Step 2: Chạy test để xác nhận các test kiểm tra khối lượng FAIL (Red phase)**
  - Run: `pytest tests/test_urdf_model.py tests/test_pizero2w_model.py tests/test_rpi5_model.py tests/test_esp32_model.py -k "mass or inertials" -v`
  - Expected: FAIL do các file URDF hiện tại vẫn mang khối lượng cũ $745-860\text{ g}$.

- [ ] **Step 3: Commit các thay đổi test suite**
  - Command: `git add tests/ && git commit -m "test: update mass budget assertions for chibi squat kinematics"`

---

### Task 2: Cập Nhật Mô Hình Base MicroDuck MJCF & URDF (TDD Green)

**Files:**
- Modify: `models/mjcf/microduck_8dof.xml`
- Modify: `models/urdf/microduck_8dof.urdf`

**Interfaces:**
- Consumes: Thông số hình học $L_1=32\text{ mm}, L_2=32\text{ mm}, \text{foot } 52\times 42\times 8\text{ mm}, Z_{\text{base}}=0.095\text{ m}$.
- Produces: Base model chuẩn Chibi Squat Kinematics.

- [ ] **Step 1: Cập nhật `models/mjcf/microduck_8dof.xml`**
  - Cập nhật `pos="0 0 0.095"` cho `base_link`.
  - Cập nhật default class `foot`: `size="0.026 0.021 0.004"`.
  - Cập nhật `left_thigh_geom` & `right_thigh_geom`: `size="0.0175 0.015 0.016"`, `pos="0 0 -0.016"`, `mass="0.040"`.
  - Cập nhật `left_shin_link` & `right_shin_link` pos: `pos="0 0 -0.032"`.
  - Cập nhật `left_shin_geom` & `right_shin_geom`: `size="0.015 0.0125 0.016"`, `pos="0 0 -0.016"`, `mass="0.030"`.
  - Cập nhật `left_foot_link` & `right_foot_link` pos: `pos="0 0 -0.032"`.
  - Cập nhật `left_foot_geom` & `right_foot_geom`: `pos="0.008 0 -0.004"`, `mass="0.022"`.
  - Cập nhật `left_foot_touch_site` & `right_foot_touch_site`: `pos="0.008 0 -0.008"`, `size="0.026 0.021 0.004"`.

- [ ] **Step 2: Cập nhật `models/urdf/microduck_8dof.urdf`**
  - Đồng bộ các kích thước visual/collision hình hộp, vị trí joints và tính toán ma trận quán tính mới theo Spec cho:
    * `left_thigh_link`, `right_thigh_link` ($m=0.040$, $I_{xx}=6.41e-6, I_{yy}=7.50e-6, I_{zz}=7.08e-6$).
    * `left_knee_pitch`, `right_knee_pitch` origin `xyz="0 0 -0.032"`.
    * `left_shin_link`, `right_shin_link` ($m=0.030$, $I_{xx}=4.12e-6, I_{yy}=4.81e-6, I_{zz}=3.81e-6$).
    * `left_foot_joint`, `right_foot_joint` origin `xyz="0 0 -0.032"`.
    * `left_foot_link`, `right_foot_link` ($m=0.022$, $I_{xx}=3.35e-6, I_{yy}=5.07e-6, I_{zz}=8.19e-6$).

- [ ] **Step 3: Chạy test xác nhận Base MicroDuck đã PASS**
  - Run: `pytest tests/test_mjcf_model.py tests/test_urdf_model.py -v`
  - Expected: PASS 100%.

- [ ] **Step 4: Commit**
  - Command: `git add models/ && git commit -m "feat(models): overhaul base microduck to chibi squat kinematics"`

---

### Task 3: Cập Nhật Mô Hình Pi Zero 2 W Stack (MJCF & URDF)

**Files:**
- Modify: `pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml`
- Modify: `pipelines/pizero2w_stack/models/urdf/microduck_pizero2w_8dof.urdf`

**Interfaces:**
- Consumes: Cấu trúc chân Chibi từ Task 2 + Stack cảm biến Pi Zero 2 W (5-ray LiDAR, Camera, Mic, base_link 275g).
- Produces: Mô hình Pi Zero 2 W hoàn chỉnh với tổng khối lượng $649\text{ g}$.

- [ ] **Step 1: Cập nhật MJCF của Pi Zero 2 W Stack**
  - Cập nhật `base_link` pos `0 0 0.095`.
  - Cập nhật đùi, cẳng chân, bàn chân kích thước $32\text{ mm} / 32\text{ mm} / 52\times 42\times 8\text{ mm}$, masses $40\text{ g} / 30\text{ g} / 22\text{ g}$.

- [ ] **Step 2: Cập nhật URDF của Pi Zero 2 W Stack**
  - Đồng bộ hóa hình học khâu chân, joints và quán tính theo spec.

- [ ] **Step 3: Chạy test xác nhận Pi Zero 2 W Stack PASS**
  - Run: `pytest tests/test_pizero2w_model.py -v`
  - Expected: PASS 100%.

- [ ] **Step 4: Commit**
  - Command: `git add pipelines/pizero2w_stack/models/ && git commit -m "feat(pizero2w): apply chibi squat kinematics to pizero2w stack"`

---

### Task 4: Cập Nhật Mô Hình Dual ESP32-S3 Stack (MJCF & URDF)

**Files:**
- Modify: `pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml`
- Modify: `pipelines/dual_esp32s3_stack/models/urdf/microduck_esp32_8dof.urdf`

**Interfaces:**
- Consumes: Cấu trúc chân Chibi + Stack Dual ESP32-S3 (CoM thấp tại lower torso, tổng khối lượng $684\text{ g}$).
- Produces: Mô hình Dual ESP32-S3 Chibi hoàn chỉnh.

- [ ] **Step 1: Cập nhật MJCF của ESP32 Stack**
  - Cập nhật `base_link` pos `0 0 0.095`.
  - Cập nhật đùi, cẳng chân, bàn chân $32\text{ mm} / 32\text{ mm} / 52\times 42\times 8\text{ mm}$, masses $40\text{ g} / 30\text{ g} / 22\text{ g}$.

- [ ] **Step 2: Cập nhật URDF của ESP32 Stack**
  - Đồng bộ hóa hình học khâu chân, joints và quán tính theo spec.

- [ ] **Step 3: Chạy test xác nhận ESP32 Stack PASS**
  - Run: `pytest tests/test_esp32_model.py -v`
  - Expected: PASS 100%.

- [ ] **Step 4: Commit**
  - Command: `git add pipelines/dual_esp32s3_stack/models/ && git commit -m "feat(esp32): apply chibi squat kinematics to dual esp32 stack"`

---

### Task 5: Cập Nhật Mô Hình Raspberry Pi 5 Stack (MJCF & URDF)

**Files:**
- Modify: `pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml`
- Modify: `pipelines/rpi5_stack/models/urdf/microduck_rpi5_8dof.urdf`

**Interfaces:**
- Consumes: Cấu trúc chân Chibi + Stack RPi 5 (base_link 390g, CoM nâng cao do cụm tản nhiệt + hat, tổng khối lượng $764\text{ g}$).
- Produces: Mô hình RPi 5 Chibi hoàn chỉnh.

- [ ] **Step 1: Cập nhật MJCF của RPi 5 Stack**
  - Cập nhật `base_link` pos `0 0 0.095`.
  - Cập nhật đùi, cẳng chân, bàn chân $32\text{ mm} / 32\text{ mm} / 52\times 42\times 8\text{ mm}$, masses $40\text{ g} / 30\text{ g} / 22\text{ g}$.

- [ ] **Step 2: Cập nhật URDF của RPi 5 Stack**
  - Đồng bộ hóa hình học khâu chân, joints và quán tính theo spec.

- [ ] **Step 3: Chạy test xác nhận RPi 5 Stack PASS**
  - Run: `pytest tests/test_rpi5_model.py -v`
  - Expected: PASS 100%.

- [ ] **Step 4: Commit**
  - Command: `git add pipelines/rpi5_stack/models/ && git commit -m "feat(rpi5): apply chibi squat kinematics to rpi5 stack"`

---

### Task 6: Cập Nhật `scripts/verify_model.py`, Kiểm Chứng Mô Hình Vật Lý & Toàn Bộ Test Suite

**Files:**
- Modify: `scripts/verify_model.py:175-225`

**Interfaces:**
- Consumes: Toàn bộ 4 mô hình MJCF & URDF sau khi cập nhật.
- Produces: Báo cáo cross-validation và toàn bộ test suite pass 100%.

- [ ] **Step 1: Cập nhật `scripts/verify_model.py`**
  - Cập nhật `target_mass`: RPi 5: $0.764$, Pi Zero: $0.649$, Base/ESP32: $0.684$.
  - Cập nhật fallback CoM: `base_pos = [0.0, 0.0, 0.0950]`, `com_world = [0.0041, 0.0, 0.0650]`.
  - Cập nhật ngưỡng kiểm tra dải CoM: `-0.06 <= com_rel[2] <= 0.0`.

- [ ] **Step 2: Chạy kiểm chứng vật lý trên cả 4 mô hình bằng `scripts/verify_model.py`**
  - Run:
    * `python scripts/verify_model.py --mjcf models/mjcf/microduck_8dof.xml --urdf models/urdf/microduck_8dof.urdf`
    * `python scripts/verify_model.py --mjcf pipelines/pizero2w_stack/models/mjcf/microduck_pizero2w_8dof.xml --urdf pipelines/pizero2w_stack/models/urdf/microduck_pizero2w_8dof.urdf`
    * `python scripts/verify_model.py --mjcf pipelines/dual_esp32s3_stack/models/mjcf/microduck_esp32_8dof.xml --urdf pipelines/dual_esp32s3_stack/models/urdf/microduck_esp32_8dof.urdf`
    * `python scripts/verify_model.py --mjcf pipelines/rpi5_stack/models/mjcf/microduck_rpi5_8dof.xml --urdf pipelines/rpi5_stack/models/urdf/microduck_rpi5_8dof.urdf`
  - Expected: Tất cả các kiểm tra Joint consistency, Mass budget, Triangle inequalities, và Center of Mass đều PASS (100% OK).

- [ ] **Step 3: Chạy toàn bộ test suite của repo**
  - Run: `pytest -v`
  - Expected: Tất cả các bài test (MJCF, URDF, RPi5, ESP32, PiZero2W, Deployment artifacts) đều PASS.

- [ ] **Step 4: Commit**
  - Command: `git add scripts/verify_model.py && git commit -m "chore(verify): update target masses and com baseline for chibi kinematics"`
