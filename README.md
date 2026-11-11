# MicroDuck 8-DOF

[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/PhelanDev/badge.svg)](https://hub.xerj.org/contributors/PhelanDev)

MicroDuck is an 8-DOF bipedal robot built with metal-gear MG996R servos for Sim-to-Real reinforcement learning research. It supports three standalone hardware targets: Raspberry Pi 5, Dual ESP32-S3, and Raspberry Pi Zero 2 W.

Physical models stay in sync between MuJoCo (MJCF) for simulation training and URDF for kinematic validation in ROS or PyBullet.

---

## Table of contents

1. [Hardware configuration and kinematics](#1-hardware-configuration-and-kinematics)
2. [Hardware stacks](#2-hardware-stacks)
3. [Perception suite](#3-perception-suite)
4. [Installation and quick start](#4-installation-and-quick-start)
5. [CLI usage (duck.py)](#5-cli-usage-duckpy)
6. [Repository structure](#6-repository-structure)
7. [License](#7-license)

---

## 1. Hardware configuration and kinematics

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

## 2. Hardware stacks

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

## 3. Perception suite

Sensors are mounted on the head (`head_link`) and torso (`base_link`):

- **5-Beam ToF LiDAR:** Fan-beam distance sensors oriented at -30°, -15°, 0°, +15°, and +30°, maximum range 2.0 m, 50 Hz update rate.
- **Wide-Angle Camera:** 60° FOV lens on the front of the head (`cam_lens`).
- **Microphone:** Mounted at the top of the head for audio input.
- **6-DOF IMU:** 3-axis accelerometer and 3-axis gyroscope centered in the main body (`base_link`).

---

## 4. Installation and quick start

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

## 5. CLI usage (duck.py)

The `duck.py` CLI handles system status, verification, simulation, training, and policy export:

### System status
```bash
python duck.py status
```
Shows model verification results, policy checkpoints, and stack readiness.

### Model verification
```bash
python duck.py verify
```

### Interactive 3D simulation
```bash
python duck.py view
```
Opens an interactive MuJoCo visualizer with keyboard controls for walking, crouching, resting, and standing.

### Reinforcement learning training (PPO)
```bash
# Train for Raspberry Pi 5
python duck.py train --target rpi5 --envs 256 --iterations 500

# Train for Raspberry Pi Zero 2 W
python duck.py train --target pizero2w --envs 256 --iterations 500

# Train for Dual ESP32-S3
python duck.py train --target esp32 --envs 256 --iterations 500
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

## 6. Repository structure

```text
duckV1/
├── duck.py                                   # Unified operations CLI
├── README.md                                 # Technical documentation
├── LICENSE                                   # MIT License
├── .gitignore                                # Git ignore configuration
├── models/                                   # Base physical models
│   ├── mjcf/microduck_8dof.xml               # Base MuJoCo XML model
│   ├── urdf/microduck_8dof.urdf              # Base URDF model
│   ├── microduck_pizero2w_policy.onnx        # Exported ONNX policy for Pi Zero 2 W
│   └── microduck_ppo.pt                      # PyTorch training checkpoint
├── pipelines/                                # Hardware targets
│   ├── rpi5_stack/                           # Raspberry Pi 5 target
│   │   ├── models/ (mjcf & urdf)             # 764 g physical model
│   │   ├── training/                         # Float32 training environment
│   │   ├── deployment/ (infer_rpi5.py)       # 50 Hz inference runtime via PCA9685
│   │   └── docs/rpi5_hardware_spec.md        # Hardware specifications
│   ├── dual_esp32s3_stack/                   # Dual ESP32-S3 target
│   │   ├── models/ (mjcf & urdf)             # 684 g physical model
│   │   ├── training/                         # INT8 training environment
│   │   ├── firmware/                         # C++/Arduino firmware for MCU 1 and MCU 2
│   │   └── docs/dual_esp32_hardware_spec.md  # Hardware specifications
│   └── pizero2w_stack/                       # Raspberry Pi Zero 2 W target
│       ├── models/ (mjcf & urdf)             # 649 g physical model
│       ├── training/                         # Pi Zero 2 W training environment
│       ├── deployment/ (infer_pizero2w.py)   # 50 Hz inference runtime via PCA9685
│       └── docs/pizero2w_hardware_spec.md    # Hardware specifications
├── scripts/                                  # Utility scripts
│   ├── verify_model.py                       # Physical model consistency verification
│   ├── interactive_viewer.py                 # Interactive 3D visualizer and simulator
│   ├── export_policy.py                      # Policy export (ONNX or INT8 C++ Header)
│   └── train_ppo.py                          # Multi-environment PPO training script
├── training/                                 # MuJoCo simulation core
│   ├── env_config.py                         # Observation space and reward function
│   └── run_parallel_sim.py                   # Parallel multi-environment simulation
└── docs/                                     # Architecture documentation
    └── specs/                                # Engineering specifications
```

---

## 7. License

Distributed under the [MIT License](LICENSE). Copyright (c) 2026 PhelanDev.
