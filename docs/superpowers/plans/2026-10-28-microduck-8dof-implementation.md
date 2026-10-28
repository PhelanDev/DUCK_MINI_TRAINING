# MicroDuck 8-DOF Robot Model & Parallel Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng mô hình vật lý 8 servo (MG996R) chuẩn hóa cho MuJoCo & URDF, kèm môi trường cấu hình và script chạy mô phỏng song song phục vụ huấn luyện học tăng cường (RL Sim-to-Real).

**Architecture:** Thiết kế mô hình robot bipedal tinh giản 8 bậc tự do (2 DOF đầu, 3 DOF mỗi chân) bằng MuJoCo MJCF XML với collision primitives (`box`, `capsule`) tối ưu hóa tính toán va chạm GPU. Xây dựng URDF đồng bộ, module cấu hình không gian quan sát (Observation 37-dim) và hành động (Action 8-dim), kèm script kiểm tra tính toàn vẹn vật lý và benchmark tốc độ mô phỏng song song (Parallel Simulation FPS).

**Tech Stack:** Python 3.10+, MuJoCo (MJCF XML), URDF, NumPy, XML ElementTree, PyTest.

**Spec:** docs/superpowers/specs/2026-10-28-microduck-8dof-design.md

## Global Constraints

- Cấu hình 8 servo MG996R: 2 DOF đầu (`head_yaw`, `head_pitch`), 6 DOF chân (`left_hip_roll`, `left_hip_pitch`, `left_knee_pitch`, `right_hip_roll`, `right_hip_pitch`, `right_knee_pitch`).
- Khối lượng tổng $\approx 780\text{ g}$, trong đó `base_link` $\approx 320\text{ g}$, đùi dài $L_1 = 65\text{ mm}$, cẳng chân dài $L_2 = 65\text{ mm}$, bàn chân $75 \times 55 \times 10\text{ mm}$.
- Giới hạn mô-men xoắn MG996R: $\tau_{\max} = 1.0\text{ N}\cdot\text{m}$, vận tốc góc $\omega_{\max} = 6.0\text{ rad/s}$, damping $0.15\text{ N}\cdot\text{s/rad}$.
- Bàn chân dùng ma sát cao và hệ số giảm chấn tiếp xúc: `friction="1.2 0.005 0.0001"`, `solref="0.005 1"`, `solimp="0.9 0.95 0.001 0.5 2"`.
- Không gian quan sát (Observation Space) chuẩn 37 chiều; không gian hành động (Action Space) 8 chiều.

## Review Focus

1. Kiểm tra trục quay động học theo chuẩn ROS/MuJoCo: Hip roll quay quanh trục X, Hip pitch quay quanh trục Y, Knee pitch quay quanh trục Y, Head yaw quay trục Z, Head pitch quay trục Y.
2. Kiểm tra hiện tượng tự va chạm (Self-collision): Khe hở giữa đùi và cẳng chân đảm bảo không bị kẹt khi đứng thẳng ($0^\circ$).
3. Kiểm tra tính hợp lệ của ma trận quán tính (Inertia tensor): Mọi link phải có $I_{xx}, I_{yy}, I_{zz} > 0$ và thỏa mãn bất đẳng thức tam giác quán tính.
4. Kiểm tra giới hạn hành động (Action scaling): Đảm bảo hàm chuyển đổi hành động $[-1, 1]$ ánh xạ chính xác vào dải góc vật lý của MG996R.
5. Kiểm tra khả năng mở rộng của script mô phỏng song song khi tăng số lượng môi trường (`num_envs = 16, 64, 256, 1024, 4096`).

---

### Task 1: Mô Hình MuJoCo MJCF XML Chuẩn Hóa (`models/mjcf/microduck_8dof.xml`)

**Files:**
- Create: `models/mjcf/microduck_8dof.xml`
- Test: `tests/test_mjcf_model.py`

**Interfaces:**
- Produces: File `microduck_8dof.xml` chứa định nghĩa đầy đủ thẻ `<mujoco>`, `<compiler>`, `<default>`, `<asset>`, `<worldbody>`, `<actuator>`, `<sensor>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mjcf_model.py
import xml.etree.ElementTree as ET
import os

def test_mjcf_xml_structure_and_joints():
    xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "microduck_8dof.xml")
    assert os.path.exists(xml_path), f"File {xml_path} does not exist"
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.tag == "mujoco"
    assert root.attrib.get("model") == "microduck_8dof"
    
    # Check 8 joints
    expected_joints = {
        "head_yaw": [-1.57, 1.57],
        "head_pitch": [-0.52, 0.78],
        "left_hip_roll": [-0.44, 0.44],
        "left_hip_pitch": [-0.87, 0.87],
        "left_knee_pitch": [0.0, 1.48],
        "right_hip_roll": [-0.44, 0.44],
        "right_hip_pitch": [-0.87, 0.87],
        "right_knee_pitch": [0.0, 1.48],
    }
    joints = {j.attrib["name"]: j for j in root.iter("joint") if j.attrib.get("name") in expected_joints}
    assert len(joints) == 8, f"Expected 8 joints, found {len(joints)}"
    
    # Check actuators
    actuators = list(root.iter("motor"))
    assert len(actuators) == 8, f"Expected 8 motor actuators, found {len(actuators)}"
    for act in actuators:
        assert float(act.attrib["ctrlrange"].split()[1]) == 1.0
        assert float(act.attrib["forcerange"].split()[1]) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mjcf_model.py -v`  
Expected: FAIL with "File ... does not exist"

- [ ] **Step 3: Implement `models/mjcf/microduck_8dof.xml`**

Tạo file `models/mjcf/microduck_8dof.xml` bao gồm:
- Định nghĩa `<default>` với damping 0.15 cho khớp và tham số ma sát sàn 1.2 cho geom bàn chân.
- Thân `base_link` mang cảm biến IMU (`accelerometer`, `gyro`) và site trọng tâm.
- Nhánh đầu 2 DOF và 2 nhánh chân mỗi bên 3 DOF với các kích thước và khối lượng chuẩn xác từ spec.
- 8 actuators kiểu `motor` tương ứng với 8 servo MG996R.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mjcf_model.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/mjcf/microduck_8dof.xml tests/test_mjcf_model.py
git commit -m "feat(model): implement standardized microduck 8-dof mjcf model"
```

---

### Task 2: Mô Hình URDF Đồng Bộ (`models/urdf/microduck_8dof.urdf`)

**Files:**
- Create: `models/urdf/microduck_8dof.urdf`
- Test: `tests/test_urdf_model.py`

**Interfaces:**
- Produces: File `microduck_8dof.urdf` có đầy đủ 8 khớp revolute, các link với inertial (mass & tensor), visual và collision.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_urdf_model.py
import xml.etree.ElementTree as ET
import os

def test_urdf_structure_and_actuators():
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "models", "urdf", "microduck_8dof.urdf")
    assert os.path.exists(urdf_path), f"File {urdf_path} does not exist"
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib.get("name") == "microduck_8dof"
    
    revolute_joints = [j for j in root.iter("joint") if j.attrib.get("type") == "revolute"]
    assert len(revolute_joints) == 8
    
    # Check effort limit is 1.0 (MG996R) and velocity is 6.0
    for j in revolute_joints:
        limit = j.find("limit")
        assert limit is not None
        assert float(limit.attrib["effort"]) == 1.0
        assert float(limit.attrib["velocity"]) == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_urdf_model.py -v`  
Expected: FAIL with "File ... does not exist"

- [ ] **Step 3: Implement `models/urdf/microduck_8dof.urdf`**

Tạo file `models/urdf/microduck_8dof.urdf` đồng bộ hoàn toàn với MJCF:
- Gốc `base_link` và các link `neck_link`, `head_link`, `left_hip_roll_link`, `left_thigh_link`, `left_shin_link`, `left_foot_link` (và các link phải đối xứng).
- Các thẻ `<inertial>` có khối lượng chuẩn và tensor quán tính hợp lệ.
- Giới hạn `<limit effort="1.0" velocity="6.0" .../>` và `<dynamics damping="0.15"/>`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_urdf_model.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/urdf/microduck_8dof.urdf tests/test_urdf_model.py
git commit -m "feat(model): implement synchronized microduck 8-dof urdf model"
```

---

### Task 3: Cấu Hình Môi Trường Huấn Luyện RL (`training/env_config.py`)

**Files:**
- Create: `training/__init__.py`, `training/env_config.py`
- Test: `tests/test_env_config.py`

**Interfaces:**
- Produces:
  - Class `MicroDuck8DoFConfig`: chứa hằng số chiều quan sát (OBS_DIM = 37), chiều hành động (ACTION_DIM = 8), dải góc, trọng số reward.
  - Hàm `compute_observations(base_ang_vel, proj_grav, commands, joint_pos, joint_vel, last_actions, foot_contact) -> np.ndarray` (kích thước 37).
  - Hàm `scale_actions(raw_actions: np.ndarray) -> np.ndarray` (ánh xạ $[-1, 1]$ sang góc mục tiêu vật lý).
  - Hàm `compute_rewards(...) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_config.py
import numpy as np
import pytest
from training.env_config import MicroDuck8DoFConfig, compute_observations, scale_actions

def test_env_config_dimensions():
    cfg = MicroDuck8DoFConfig()
    assert cfg.OBS_DIM == 37
    assert cfg.ACTION_DIM == 8
    assert len(cfg.JOINT_NAMES) == 8

def test_observation_packing():
    obs = compute_observations(
        base_ang_vel=np.zeros(3),
        proj_grav=np.array([0.0, 0.0, -1.0]),
        commands=np.array([0.5, 0.0, 0.0]),
        joint_pos=np.zeros(8),
        joint_vel=np.zeros(8),
        last_actions=np.zeros(8),
        foot_contact=np.zeros(4)
    )
    assert obs.shape == (37,)
    assert obs.dtype == np.float32

def test_action_scaling():
    raw_action = np.array([-1.0, 1.0, 0.0, -1.0, 1.0, -1.0, 1.0, 0.0])
    scaled = scale_actions(raw_action)
    assert scaled.shape == (8,)
    # Verify values remain within physical limits
    cfg = MicroDuck8DoFConfig()
    for i, name in enumerate(cfg.JOINT_NAMES):
        low, high = cfg.JOINT_LIMITS[name]
        assert low <= scaled[i] <= high
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_config.py -v`  
Expected: FAIL with "No module named 'training'"

- [ ] **Step 3: Implement `training/env_config.py`**

Triển khai cấu hình chứa:
- `MicroDuck8DoFConfig` với đầy đủ danh sách khớp, giới hạn góc của MG996R.
- Các hàm vector hóa tính observation 37 chiều và ánh xạ hành động an toàn chống vượt ngưỡng vật lý.
- Bộ trọng số hàm thưởng cân bằng (velocity tracking, upright bonus, torque penalty).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/env_config.py tests/test_env_config.py
git commit -m "feat(training): add 8-dof rl environment configuration and observation pipeline"
```

---

### Task 4: Script Xác Thực Mô Hình Vật Lý (`scripts/verify_model.py`)

**Files:**
- Create: `scripts/verify_model.py`
- Test: `tests/test_verify_script.py`

**Interfaces:**
- Produces: CLI script `scripts/verify_model.py` kiểm tra toàn diện cấu trúc XML/URDF, tính toán tổng khối lượng, vị trí trọng tâm (CoM) và tính hợp lệ của tensor quán tính.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify_script.py
import subprocess
import sys

def test_verify_script_execution():
    res = subprocess.run([sys.executable, "scripts/verify_model.py"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Verification SUCCESS" in res.stdout
    assert "Total Mass:" in res.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verify_script.py -v`  
Expected: FAIL with "cannot find script"

- [ ] **Step 3: Implement `scripts/verify_model.py`**

Triển khai script:
- Đọc cả `microduck_8dof.xml` và `microduck_8dof.urdf`.
- Kiểm tra tính nhất quán giữa 2 file (khớp, dải góc, khối lượng).
- Phân tích trọng tâm (CoM) đảm bảo CoM nằm thấp tại thân dưới.
- In ra bảng tóm tắt cấu trúc và xuất thông báo `Verification SUCCESS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_verify_script.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_model.py tests/test_verify_script.py
git commit -m "feat(scripts): add comprehensive physical model verification script"
```

---

### Task 5: Đường Ống Mô Phỏng Song Song & Benchmark FPS (`training/run_parallel_sim.py`)

**Files:**
- Create: `training/run_parallel_sim.py`
- Test: `tests/test_parallel_runner.py`

**Interfaces:**
- Produces: CLI tool `training/run_parallel_sim.py --num_envs <N> --steps <S>` chạy song song N môi trường robot để đo FPS tính toán và kiểm tra độ ổn định.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parallel_runner.py
import subprocess
import sys

def test_parallel_sim_benchmark():
    res = subprocess.run(
        [sys.executable, "training/run_parallel_sim.py", "--num_envs", "16", "--steps", "50"],
        capture_output=True, text=True
    )
    assert res.returncode == 0
    assert "Parallel Simulation Benchmark" in res.stdout
    assert "Steps Per Second (FPS):" in res.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parallel_runner.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `training/run_parallel_sim.py`**

Triển khai mô phỏng song song theo lô (batched vectorized simulator):
- Tải mô hình `microduck_8dof.xml`.
- Chạy song song nhiều instance trạng thái robot (nếu máy có sẵn `mujoco` thì dùng C binding, nếu môi trường standalone thuần Python thì chạy vector dynamics step).
- Đo lường thời gian thực thi, tính toán và in ra `Steps Per Second (FPS)` cùng tỷ lệ tăng tốc so với thời gian thực (Real-time Factor).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parallel_runner.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/run_parallel_sim.py tests/test_parallel_runner.py
git commit -m "feat(training): implement batched parallel simulation runner and fps benchmark"
```

---

### Task 6: Script Trực Quan 3D & Tài Liệu Tích Hợp Huấn Luyện (`scripts/interactive_viewer.py` & `README.md`)

**Files:**
- Create: `scripts/interactive_viewer.py`
- Create: `README.md`
- Test: `tests/test_interactive_viewer.py`

**Interfaces:**
- Produces: 
  - `scripts/interactive_viewer.py`: Script khởi chạy MuJoCo passive viewer để điều khiển 8 khớp bằng bàn phím hoặc thanh trượt khi có thư viện `mujoco`.
  - `README.md`: Hướng dẫn chi tiết cách chạy kiểm tra mô hình, cách khởi động mô phỏng song song để huấn luyện RL với `microduck_rl`.

- [ ] **Step 1: Write the test**

```python
# tests/test_interactive_viewer.py
import os

def test_viewer_and_docs_exist():
    assert os.path.exists("scripts/interactive_viewer.py")
    assert os.path.exists("README.md")
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "MicroDuck 8-DOF" in content
        assert "run_parallel_sim.py" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interactive_viewer.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `scripts/interactive_viewer.py` and `README.md`**

Triển khai:
- `scripts/interactive_viewer.py`: Hỗ trợ mở viewer hiển thị robot đứng thăng bằng, in ra tư thế các khớp và hướng dẫn tương tác.
- `README.md`: Hướng dẫn toàn diện từ A-Z cách sử dụng mô hình 8 servo này trong dự án.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interactive_viewer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/interactive_viewer.py README.md tests/test_interactive_viewer.py
git commit -m "docs: add interactive 3d viewer and complete project usage guide"
```
