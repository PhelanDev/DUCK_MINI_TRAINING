#!/usr/bin/env python3
"""MicroDuck 8-DOF Bipedal Robot - Unified Operations CLI (duck.py).

Provides a single standardized entry point for:
- `status`: Inspect model checkpoint metrics, stage, and hardware readiness
- `verify`: Run kinematics cross-validation and automated test suites
- `train`: Launch PPO reinforcement learning with MuJoCo parallel simulation
- `view`: Open interactive 3D viewer or browser dashboard
- `export`: Export trained policy to ONNX (Raspberry Pi 5) & INT8 C++ (ESP32-S3)
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe UTF-8 console output for Windows CLI
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



def cmd_status(args: argparse.Namespace) -> int:
    """Display comprehensive status of MicroDuck 8-DOF robot and training metrics."""
    print("=" * 72)
    print("🦆 MicroDuck 8-DOF Bipedal Robot - System & Training Status")
    print("=" * 72)

    # 1. Model Architecture
    print("\n[1] BỘ NÃO AI (NEURAL NETWORK ARCHITECTURE)")
    print("  • Mô hình triển khai: Actor MLP [42 -> 256 (ELU) -> 256 (ELU) -> 8]")
    print("  • Số nơ-ron Actor   : 520 nơ-ron (256 + 256 + 8 output servos)")
    print("  • Số tham số Actor  : 78.856 weights & biases (78.864 gồm log_std)")
    print("  • Mạng Critic (PPO) : 513 nơ-ron | 77.057 tham số")
    print("  • Không gian quan sát: 42 chiều (IMU, 8 góc, 8 vận tốc, last action, Phase Clock, 5-ray LiDAR)")

    # 2. Checkpoints Status
    ckpt_dir = REPO_ROOT / "models" / "checkpoints"
    best_ckpt = ckpt_dir / "microduck_best.pt"
    latest_ckpt = ckpt_dir / "microduck_latest.pt"

    print("\n[2] TIẾN TRÌNH HUẤN LUYỆN (CHECKPOINTS & CURRICULUM)")
    if best_ckpt.exists():
        try:
            import torch
            data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            stage = data.get("stage", 0)
            iters = data.get("iterations", 0)
            steps = data.get("total_timesteps", 0)
            best_r = data.get("best_reward", 0.0)
            dist_x = data.get("best_distance_x", 0.0)
            print(f"  • Checkpoint tốt nhất: {best_ckpt.name}")
            print(f"    - Giai đoạn địa hình: Stage {stage} (Mục tiêu Stage 1: distance_x >= 0.80m, success >= 70%)")
            print(f"    - Số vòng lặp (Iters): {iters} iterations ({steps:,} timesteps)")
            print(f"    - Điểm thưởng kỷ lục : {best_r:+.4f}")
            print(f"    - Quãng đường tiến X : {dist_x * 100:.2f} cm")
        except Exception as e:
            print(f"  • Không thể đọc {best_ckpt.name}: {e}")
    else:
        print("  • Chưa có checkpoint microduck_best.pt")

    # 3. Git Status
    print("\n[3] TRẠNG THÁI MÃ NGUỒN & HỆ THỐNG")
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        print(f"  • Git Branch / Commit: {git_branch} ({git_hash})")
    except Exception:
        print("  • Git: Không thể lấy thông tin commit")

    # 4. Hardware Deployment Targets
    onnx_file = REPO_ROOT / "models" / "microduck_rpi5_policy.onnx"
    esp32_h = REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "deployment" / "esp32_policy_weights.h"
    print("\n[4] TỆP TRIỂN KHAI PHẦN CỨNG (DEPLOYMENT ARTIFACTS)")
    print(f"  • RPi 5 Float32 ONNX : {'[CÓ SẴN]' if onnx_file.exists() else '[CHƯA XUẤT]'} ({onnx_file.name})")
    print(f"  • ESP32-S3 INT8 C-Header: {'[CÓ SẴN]' if esp32_h.exists() else '[CHƯA XUẤT]'} ({esp32_h.name})")

    print("\n" + "=" * 72)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify physical kinematics consistency and run automated test suite."""
    print(">>> [1/2] Đang xác thực mô hình vật lý 3D (MJCF vs URDF)...")
    res_physics = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "verify_model.py")])
    if res_physics.returncode != 0:
        print("LỖI: Xác thực mô hình vật lý thất bại!")
        return res_physics.returncode

    if not args.skip_tests:
        print("\n>>> [2/2] Đang chạy bộ kiểm thử tự động (pytest)...")
        pytest_args = [sys.executable, "-m", "pytest"]
        if args.fast:
            pytest_args.extend(["tests/test_eplen_suicide_fix.py", "tests/test_interactive_viewer.py", "-q"])
        else:
            pytest_args.extend(["tests/", "-q"])
        res_tests = subprocess.run(pytest_args)
        return res_tests.returncode

    return 0


def cmd_view(args: argparse.Namespace) -> int:
    """Launch interactive 3D visualizer or open web dashboard."""
    if args.web:
        web_file = REPO_ROOT / "viewer_3d.html"
        print(f">>> Đang mở giao diện 3D Web Dashboard: {web_file}")
        webbrowser.open(f"file://{web_file.resolve()}")
        return 0

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "interactive_viewer.py"),
        "--stage",
        str(args.stage),
    ]
    if args.policy:
        cmd.extend(["--policy", str(args.policy)])
    if args.headless:
        cmd.append("--headless")

    print(f">>> Khởi chạy Interactive 3D Viewer (Stage {args.stage})...")
    return subprocess.run(cmd).returncode


def cmd_train(args: argparse.Namespace) -> int:
    """Launch PPO reinforcement learning training pipeline."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_ppo.py"),
        "--num-envs",
        str(args.envs),
        "--max-iterations",
        str(args.iterations),
    ]
    if args.resume:
        cmd.extend(["--resume", str(args.resume)])
    if args.stand_first:
        cmd.append("--stand-first")

    print(f">>> Khởi động huấn luyện PPO ({args.envs} envs, tối đa {args.iterations} iters)...")
    return subprocess.run(cmd).returncode


def cmd_export(args: argparse.Namespace) -> int:
    """Export trained policy to ONNX and INT8 C header."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "export_policy.py")]
    if args.checkpoint:
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    print(">>> Đang xuất mô hình ra ONNX (RPi 5) và INT8 Header (ESP32-S3)...")
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="🦆 MicroDuck 8-DOF Bipedal Robot - Unified Operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available operations")

    # status
    p_status = subparsers.add_parser("status", help="Xem báo cáo tổng quan về mô hình và tiến trình học")

    # verify
    p_verify = subparsers.add_parser("verify", help="Kiểm tra tính nhất quán vật lý và chạy test suite")
    p_verify.add_argument("--skip-tests", action="store_true", help="Chỉ kiểm tra vật lý, bỏ qua pytest")
    p_verify.add_argument("--fast", action="store_true", help="Chỉ chạy các test cốt lõi nhanh")

    # view
    p_view = subparsers.add_parser("view", help="Mở trình xem 3D trực quan tương tác hoặc web dashboard")
    p_view.add_argument("--web", action="store_true", help="Mở giao diện Web Dashboard (viewer_3d.html)")
    p_view.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3], help="Giai đoạn địa hình (0-3)")
    p_view.add_argument("--policy", type=str, default=None, help="Đường dẫn file checkpoint policy (.pt)")
    p_view.add_argument("--headless", action="store_true", help="Chạy chế độ giả lập không cửa sổ")

    # train
    p_train = subparsers.add_parser("train", help="Huấn luyện mô hình bước đi bằng PPO đa môi trường")
    p_train.add_argument("--envs", type=int, default=256, help="Số lượng môi trường song song (mặc định: 256)")
    p_train.add_argument("--iterations", type=int, default=500, help="Số vòng lặp huấn luyện (mặc định: 500)")
    p_train.add_argument("--resume", type=str, default=None, help="Tiếp tục học từ checkpoint")
    p_train.add_argument("--stand-first", action="store_true", help="Ưu tiên học đứng thăng bằng trước")

    # export
    p_export = subparsers.add_parser("export", help="Xuất policy sang ONNX và INT8 C Header")
    p_export.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint cần xuất")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "status": cmd_status,
        "verify": cmd_verify,
        "view": cmd_view,
        "train": cmd_train,
        "export": cmd_export,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
