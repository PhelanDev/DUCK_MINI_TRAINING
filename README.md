# MicroDuck 8-DOF & CyberDuck V2 Pro

[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/PhelanDev/badge.svg)](https://hub.xerj.org/contributors/PhelanDev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo 3.x](https://img.shields.io/badge/Physics-MuJoCo%203.x-emerald.svg)](https://mujoco.org/)

> 🏆 **XERJ Corpus Hub Featured Contribution**:
> MicroDuck's decoupled real-time inference runtime on Raspberry Pi Zero 2 W was selected by [XERJ Corpus Hub](https://hub.xerj.org/contributors/PhelanDev) as an open reference memory solution for AI coding agents.
> 
> *"We came across your 'feat(pizero2w): add decoupled real-time inference runtime for Pi Zero 2 W with tests' in PhelanDev/DUCK_MINI_TRAINING and think it could help others. It is a substantial piece of work, with 6 files and about 460 lines across pipelines and tests, in Python, with the tests landing in the same commit."*
> — **Ivan**, Founder of XERJ ([github.com/xerj-org/xerj](https://github.com/xerj-org/xerj))

MicroDuck is an 8-DOF bipedal robot built with metal-gear MG996R servos for Sim-to-Real reinforcement learning research. It supports three standalone hardware targets: Raspberry Pi 5, Dual ESP32-S3, and Raspberry Pi Zero 2 W, alongside the next-generation **CyberDuck V2 Pro (Athletic Bionic Droid Edition)**.

Physical models stay in sync between MuJoCo (MJCF) for simulation training and URDF for kinematic validation in ROS or PyBullet.

---

## Table of contents

1. [CyberDuck V2 Pro (Athletic Bionic Droid Edition)](#1-cyberduck-v2-pro-athletic-bionic-droid-edition)
2. [Advanced Gaits & AI Vision Tracking](#2-advanced-gaits--ai-vision-tracking)
3. [Hardware configuration and kinematics](#3-hardware-configuration-and-kinematics)
4. [Hardware stacks](#4-hardware-stacks)
5. [Perception suite](#5-perception-suite)
6. [Installation and quick start](#6-installation-and-quick-start)
7. [CLI usage (duck.py)](#7-cli-usage-duckpy)
8. [Repository structure](#8-repository-structure)
9. [License](#9-license)

---

## 1. CyberDuck V2 Pro (Athletic Bionic Droid Edition)

<p align="center">
  <img src="duck_v2/media/cyberduck_v2_iso.png" width="48%" alt="CyberDuck V2 Pro Isometric View" />
  <img src="duck_v2/media/cyberduck_v2_front.png" width="48%" alt="CyberDuck V2 Pro Front View" />
</p>

CyberDuck V2 Pro represents a high-definition mechanical and aerodynamic redesign of the 8-DOF bipedal platform:

- **17 Precision CAD Parts**: High-resolution STLs (>125,000 smooth polygon triangles) featuring an aerodynamic helmet, glowing cyan visor, metallic gold beak, filleted collar, cyber chest armor, and rear thruster pack.
- **2-DOF Universal Neck Gimbal**: Matches the mechanical language of the hip universal joints, lowering the head by 8 mm into an aggressive athletic stance and eliminating gap artifacts with a filleted collar.
- **Proportional Athletic Feet**: Downscaled from bulky prototypes to $50\text{ mm} \times 26\text{ mm} \times 10\text{ mm}$ with a perfectly planar $Z = 0.000\text{ m}$ sole contact and high-traction rubber treads.
- **Production-Ready CAD Exports**: Complete 3D assembly in `duck_v2/cad/cyberduck_v2_assembly.obj` with MTL PBR materials and pre-packed 3D-printable parts in `duck_v2/cad/cyberduck_v2_stl_parts.zip`.
- **Full Sensor Suite Integration**: Equipped with base 6-DOF IMU, dual foot touch sensors, 8-joint telemetry, and an onboard head camera, fully compatible with the 37-dim and 42-dim PPO training pipelines.

---

## 2. Advanced Gaits & AI Vision Tracking

### 🦆 Advanced Behavioral Gaits (`training/advanced_gaits.py`)
MicroDuck features specialized bipedal behaviors beyond standard forward walking:
- **Waddling Gait (Dáng đi lạch bạch)**: Iconic bionic duck locomotion with synchronized pelvic roll sway ($0.09\text{ rad}$) and double-frequency rhythmic head nodding.
- **Spot Turn (Xoay tròn tại chỗ)**: High-agility turning-in-place without forward drift, utilizing opposing hip pitches and stance leg pivoting.
- **Head Dance & Gaze**: Expressive Lissajous figure head-bobbing for life-like social interaction while standing.
- **Fall Recovery (Tự đứng dậy khi ngã)**: Finite State Machine reflex recovery controller (`ORIENT_PRONE` $\to$ `CROUCH_TO_SIT` $\to$ `PUSH_TO_STAND`) with $|\Delta q| \le 0.10\text{ rad/step}$ rate limiting to protect MG996R gears.

### 👁️ AI Vision: Target Tracking & Visual Servoing (`training/vision_tracker.py`)
- **Lightweight Color Blob Tracking**: Real-time HSV/RGB segmentation tracking yellow balls, ducks, or red targets at 30+ FPS.
- **Closed-Loop Visual Servoing**: Drives 2-DOF neck gimbal (`head_yaw`, `head_pitch`) with anti-jitter deadband ($0.06$) and slew rate limiting.
- **Embedded & MCU1 Firmware Support**: Optimized sub-sampled C++ vision loop in `mcu1_main.ino` transmitting 8-byte sync command packets over UART to servo controllers.

---

## 3. Hardware configuration and kinematics

The robot uses 8 metal-gear MG996R servos (rated torque 1.0 N·m, angular velocity 6.0 rad/s). This setup replaces the original 15-servo Dynamixel build to lower build costs and withstand hard impacts during falls.

### Chibi Squat geometric parameters

MicroDuck uses a short-leg geometry (Chibi Squat) with a head-to-torso-to-leg ratio of roughly 1 : 0.9 : 1. Standing height is 195 mm, with the center of mass concentrated near the base of the torso:

- Thigh length ($L_1$): 32 mm (mass 40 g).
- Shin length ($L_2$): 32 mm (mass 30 g).
- Foot dimensions: 52 × 42 × 8 mm (mass 22 g).
- Inner foot clearance: 28 mm to 58 mm, preventing foot collision during lateral sway and gait transitions.
- Torso base frame (`base_link`): $Z = 95\text{ mm}$ ($0.095\text{ m}$).
- The short gravity lever arm keeps roll overturning torque during single-support gait phases below 0.35 N·m, well inside the 1.0 N·m limit of MG996R servos.

### Link dimensions and mass breakdown

| Link | Mass | Dimensions ($X \times Y \times Z$) | Description |
| :--- | :---: | :---: | :--- |
| `base_link` | 275 g - 390 g | 110 × 80 × 70 mm | Main torso, housing compute boards, battery pack, and 6-DOF IMU |
| `neck_link` | 30 g | 35 × 35 × 30 mm | Neck yaw joint rotating around the vertical $Z$-axis |
| `head_link` | 60 g | 60 × 50 × 45 mm | Duck head carrying wide-angle camera, microphone, and 5-beam ToF LiDAR |
| `left_hip_roll_link` | 50 g | 45 × 40 × 40 mm | Hip roll joint rotating around $X$-axis for lateral weight transfer |
| `left_thigh_link` | 40 g | 35 × 30 × 32 mm | Thigh $L_1 = 32\text{ mm}$, housing knee pitch servo |
| `left_shin_link` | 30 g | 30 × 25 × 32 mm | Shin $L_2 = 32\text{ mm}$ |
| `left_foot_link` | 22 g | 52 × 42 × 8 mm | Flat foot sole with high-friction rubber pads |
| `right_*_link` | Symmetrical | Symmetrical | Symmetrical to the left leg across the $X$-axis |

### Joint range and limits

| Joint | Axis | Limits (rad) | Limits (deg) | Max torque | Max velocity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `head_yaw` | $Z$ | $[-1.57, +1.57]$ | $\pm 90^\circ$ | 1.0 N·m | 6.0 rad/s |
| `head_pitch` | $Y$ | $[-0.52, +0.78]$ | $-30^\circ \to +45^\circ$ | 1.0 N·m | 6.0 rad/s |
| `left_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | 1.0 N·m | 6.0 rad/s |
| `left_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | 1.0 N·m | 6.0 rad/s |
| `left_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | 1.0 N·m | 6.0 rad/s |
| `right_hip_roll` | $X$ | $[-0.44, +0.44]$ | $\pm 25^\circ$ | 1.0 N·m | 6.0 rad/s |
| `right_hip_pitch` | $Y$ | $[-0.87, +0.87]$ | $\pm 50^\circ$ | 1.0 N·m | 6.0 rad/s |
| `right_knee_pitch` | $Y$ | $[0.00, +1.48]$ | $0^\circ \to +85^\circ$ | 1.0 N·m | 6.0 rad/s |

---

## 4. Hardware stacks

The project maintains three standalone hardware targets under `pipelines/`. Each stack has its own physics model, training environment, and inference runtime:

| Specification | Raspberry Pi 5 stack | Dual ESP32-S3 stack | Raspberry Pi Zero 2 W stack |
| :--- | :--- | :--- | :--- |
| **Directory** | `pipelines/rpi5_stack/` | `pipelines/dual_esp32s3_stack/` | `pipelines/pizero2w_stack/` |
| **Total mass** | 764 g (0.764 kg) | 684 g (0.684 kg) | 649 g (0.649 kg) |
| **Processor** | Broadcom BCM2712 Quad A76 @ 2.4 GHz | 2× ESP32-S3 Dual LX7 @ 240 MHz | Broadcom BCM2710A1 Quad A53 @ 1.0 GHz |
| **Memory** | 4 GB / 8 GB LPDDR4X | 2× 512 KB SRAM + 8 MB PSRAM | 512 MB LPDDR2 |
| **Operating system** | Raspberry Pi OS 64-bit | FreeRTOS (bare metal) | Raspberry Pi OS Lite 64-bit |
| **Servo control** | PCA9685 I2C PWM driver | 8-channel hardware LEDC PWM | PCA9685 I2C PWM driver |
| **Policy format** | Float32 ONNX Runtime (50 Hz) | Quantized INT8 ESP-NN C++ Array (50 Hz) | Float32 ONNX Runtime (50 Hz) |
| **Inter-chip bus** | Internal bus | Binary UART @ 921,600 bps with CRC16 | Internal bus |
| **Primary use case** | Edge AI research, vision pipelines, or NPU workloads | Low-cost embedded deployment, power efficiency | Compact standalone Linux robot |

---

## 5. Perception suite

Sensors are mounted on the head (`head_link`) and torso (`base_link`):

- **5-Beam ToF LiDAR:** Fan-beam distance sensors oriented at -30°, -15°, 0°, +15°, and +30°, maximum range 2.0 m, 50 Hz update rate.
- **Wide-Angle Camera:** 60° FOV lens on the front of the head (`cam_lens`).
- **Microphone:** Mounted at the top of the head for audio input.
- **6-DOF IMU:** 3-axis accelerometer and 3-axis gyroscope centered in the main body (`base_link`).

---

## 6. Installation and quick start

### Prerequisites

- Python 3.10 or higher
- Linux, macOS, or Windows

### Dependencies

```bash
git clone https://github.com/PhelanDev/DUCK_MINI_TRAINING.git
cd DUCK_MINI_TRAINING
pip install numpy pytest onnxruntime
```

For local MuJoCo simulation:
```bash
pip install mujoco
```

### Physical model verification

Check consistency between the MuJoCo (MJCF) and URDF models:

```bash
python scripts/verify_model.py
```

The script checks kinematic dimensions, inertia matrices, triangle inequalities, and center-of-mass alignment across both models.

---

## 7. CLI usage (duck.py)

The unified `duck.py` CLI handles system status, verification, interactive 3D simulation, behavioral gaits, visual servoing, training, and policy export:

### System status
```bash
python duck.py status
```
Shows model verification results, policy checkpoints, CyberDuck V2 readiness, and hardware stack status.

### Model verification
```bash
python duck.py verify
```

### Interactive 3D simulation & Behavioral Gaits
```bash
# View baseline MicroDuck 8-DOF
python duck.py view

# View CyberDuck V2 Pro (Athletic Bionic Droid)
python duck.py view --v2

# Demonstrate iconic duck waddling gait (lắc hông nhịp nhàng 2 bên & gật đầu)
python duck.py view --gait waddle

# Demonstrate spot turn (xoay tròn 360 độ tại chỗ không trôi)
python duck.py view --gait spot_turn

# Demonstrate standing head dance gesture
python duck.py view --gait head_dance

# Activate AI Vision Visual Servoing (theo dõi mục tiêu qua camera đầu)
python duck.py view --track --target-color yellow

# Combine CyberDuck V2 with waddling gait and visual tracking
python duck.py view --v2 --gait waddle --track
```

### Reinforcement learning training (PPO)
```bash
# Train for Raspberry Pi 5
python duck.py train --target rpi5 --envs 256 --iterations 500

# Train for Raspberry Pi Zero 2 W
python duck.py train --target pizero2w --envs 256 --iterations 500

# Train for Dual ESP32-S3
python duck.py train --target esp32 --envs 256 --iterations 500

# Train directly on CyberDuck V2 Pro
python duck.py train --target v2 --envs 256 --iterations 500
```

### Policy export
```bash
# Export ONNX model for Pi 5 or Pi Zero 2 W
python duck.py export --target rpi5
python duck.py export --target pizero2w

# Export quantized INT8 C++ header for Dual ESP32-S3
python duck.py export --target esp32
```

---

## 8. Repository structure

```text
duckV1/
├── duck.py                                   # Unified operations CLI
├── README.md                                 # Technical documentation
├── LICENSE                                   # MIT License
├── .gitignore                                # Git ignore configuration
├── duck_v2/                                  # CyberDuck V2 Pro (Athletic Droid)
│   ├── cyberduck_v2.xml                      # MuJoCo MJCF model with full sensor suite
│   ├── cad/                                  # 3D CAD Assembly OBJ, MTL & STL Zip
│   ├── meshes/                               # 13 custom high-definition STL meshes
│   └── media/                                # High-res renders (ISO, front, exploded)
├── models/                                   # Base physical models
│   ├── mjcf/microduck_8dof.xml               # Base MuJoCo XML model
│   ├── urdf/microduck_8dof.urdf              # Base URDF model
│   ├── microduck_pizero2w_policy.onnx        # Exported ONNX policy for Pi Zero 2 W
│   └── microduck_ppo.pt                      # PyTorch training checkpoint
├── pipelines/                                # Hardware targets
│   ├── rpi5_stack/                           # Raspberry Pi 5 target
│   ├── dual_esp32s3_stack/                   # Dual ESP32-S3 target
│   └── pizero2w_stack/                       # Raspberry Pi Zero 2 W target (XERJ corpus)
├── scripts/                                  # Utility scripts
│   ├── verify_model.py                       # Physical model consistency verification
│   ├── interactive_viewer.py                 # Interactive 3D visualizer and simulator
│   ├── export_policy.py                      # Policy export (ONNX or INT8 C++ Header)
│   └── train_ppo.py                          # Multi-environment PPO training script
├── training/                                 # Simulation & Perception core
│   ├── advanced_gaits.py                     # Waddle, spot turn, head dance & fall recovery
│   ├── vision_tracker.py                     # AI vision color blob tracker & visual servoing
│   ├── self_righting.py                      # 3-Phase reflex recovery finite state machine
│   ├── env_config.py                         # Observation space and reward function
│   └── run_parallel_sim.py                   # Parallel multi-environment simulation
└── docs/                                     # Architecture documentation
    └── specs/                                # Engineering specifications
```

---

## 9. License

Distributed under the [MIT License](LICENSE). Copyright (c) 2026 PhelanDev.
