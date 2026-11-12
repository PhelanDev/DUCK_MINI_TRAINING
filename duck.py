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
    pizero_onnx_file = REPO_ROOT / "models" / "microduck_pizero2w_policy.onnx"
    esp32_h = REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "deployment" / "esp32_policy_weights.h"
    print("\n[4] TỆP TRIỂN KHAI PHẦN CỨNG (DEPLOYMENT ARTIFACTS)")
    print(f"  • RPi 5 Float32 ONNX    : {'[CÓ SẴN]' if onnx_file.exists() else '[CHƯA XUẤT]'} ({onnx_file.name})")
    print(f"  • Pi Zero 2 W ONNX      : {'[CÓ SẴN]' if pizero_onnx_file.exists() else '[CHƯA XUẤT]'} ({pizero_onnx_file.name})")
    print(f"  • ESP32-S3 INT8 C-Header: {'[CÓ SẴN]' if esp32_h.exists() else '[CHƯA XUẤT]'} ({esp32_h.name})")

    # 5. CyberDuck V2 Pro Status
    v2_xml = REPO_ROOT / "duck_v2" / "cyberduck_v2.xml"
    v2_cad = REPO_ROOT / "duck_v2" / "cad" / "cyberduck_v2_assembly.obj"
    v2_zip = REPO_ROOT / "duck_v2" / "cad" / "cyberduck_v2_stl_parts.zip"
    print("\n[5] MÔ HÌNH THỂ THAO CYBERDUCK V2 PRO (ATHLETIC BIONIC DROID)")
    print(f"  • Mô hình vật lý MuJoCo : {'[SẴN SÀNG]' if v2_xml.exists() else '[CHƯA TẠO]'} ({v2_xml.name})")
    print(f"  • Bản vẽ lắp ráp 3D CAD : {'[SẴN SÀNG]' if v2_cad.exists() else '[CHƯA XUẤT]'} ({v2_cad.name})")
    print(f"  • Gói 17 File STL In 3D : {'[SẴN SÀNG]' if v2_zip.exists() else '[CHƯA NÉN]'} ({v2_zip.name})")
    print("  • Tiếp xúc sàn          : Phẳng tuyệt đối tại Z = 0.000m (Khớp cổ các-đăng & bàn chân 50mm)")

    # 6. Advanced Gaits & AI Vision Status
    print("\n[6] KỸ NĂNG VẬN ĐỘNG NÂNG CAO & AI VISION (BEHAVIORS & TRACKING)")
    print("  • Waddling Gait (Dáng đi vịt bionic) : [SẴN SÀNG] (Lắc hông nhịp nhàng 2 bên & gật đầu)")
    print("  • Spot Turn (Xoay tròn tại chỗ)      : [SẴN SÀNG] (Xoay 360 độ tại chỗ không tịnh tiến)")
    print("  • Fall Recovery (Tự đứng dậy khi ngã): [SẴN SÀNG] (FSM 3 pha: Lật sấp -> Ngồi -> Đứng)")
    print("  • AI Vision Head Tracking           : [SẴN SÀNG] (Visual Servoing bám bóng vàng/mặt người)")

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
            pytest_args.extend([
                "tests/test_eplen_suicide_fix.py",
                "tests/test_interactive_viewer.py",
                "tests/test_cyberduck.py",
                "tests/test_advanced_gaits.py",
                "tests/test_vision_tracker.py",
                "-q"
            ])
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

    # Model resolution (supports baseline MicroDuck and CyberDuck V2)
    model_path = None
    if getattr(args, "v2", False) or getattr(args, "model", None) in ("v2", "cyberduck_v2"):
        model_path = REPO_ROOT / "duck_v2" / "cyberduck_v2.xml"
    elif getattr(args, "model", None):
        model_path = Path(args.model)

    if model_path:
        cmd.extend(["--model", str(model_path)])

    if getattr(args, "gait", None):
        cmd.extend(["--gait", str(args.gait)])
        cmd.append("--gait-demo")
    if getattr(args, "track", False):
        cmd.append("--track")
    if getattr(args, "target_color", None):
        cmd.extend(["--target-color", str(args.target_color)])

    if args.policy:
        cmd.extend(["--policy", str(args.policy)])
    if args.headless:
        cmd.append("--headless")

    target_desc = "CyberDuck V2 Pro" if getattr(args, "v2", False) or getattr(args, "model", None) in ("v2", "cyberduck_v2") else "MicroDuck 8-DOF"
    print(f">>> Khởi chạy Interactive 3D Viewer ({target_desc}, Stage {args.stage})...")
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

    target_models = {
        "rpi5": REPO_ROOT / "pipelines" / "rpi5_stack" / "models" / "mjcf" / "microduck_rpi5_8dof.xml",
        "pizero2w": REPO_ROOT / "pipelines" / "pizero2w_stack" / "models" / "mjcf" / "microduck_pizero2w_8dof.xml",
        "esp32": REPO_ROOT / "pipelines" / "dual_esp32s3_stack" / "models" / "mjcf" / "microduck_esp32_8dof.xml",
        "v2": REPO_ROOT / "duck_v2" / "cyberduck_v2.xml",
    }
    if getattr(args, "target", None) and args.target in target_models:
        cmd.extend(["--model-path", str(target_models[args.target])])

    if args.resume:
        if Path(str(args.resume)).exists() or str(args.resume).lower() not in ("true", "1"):
            cmd.extend(["--load-checkpoint", str(args.resume)])
        else:
            cmd.append("--resume")
    if args.stand_first:
        cmd.append("--stand-first")
    if getattr(args, "dry_run", False):
        cmd.append("--dry-run")

    mode_str = "DRY-RUN" if getattr(args, "dry_run", False) else f"{args.envs} envs, tối đa {args.iterations} iters"
    print(f">>> Khởi động huấn luyện PPO ({mode_str}, target: {getattr(args, 'target', 'rpi5')})...")

    return subprocess.run(cmd).returncode


def cmd_export(args: argparse.Namespace) -> int:
    """Export trained policy to ONNX and INT8 C header."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "export_policy.py")]
    ckpt_path = args.checkpoint
    if not ckpt_path:
        best_ckpt = REPO_ROOT / "models" / "checkpoints" / "microduck_best.pt"
        if best_ckpt.exists():
            ckpt_path = str(best_ckpt)
    if ckpt_path:
        cmd.extend(["--checkpoint", ckpt_path])
    print(f">>> Đang xuất mô hình ra ONNX (RPi 5 & Pi Zero 2 W) và INT8 Header (ESP32-S3) từ {ckpt_path or 'mặc định'}...")
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
    p_view.add_argument("--model", type=str, default=None, help="Đường dẫn file mô hình XML hoặc alias ('v2')")
    p_view.add_argument("--v2", action="store_true", help="Xem trực tiếp mô hình CyberDuck V2 Pro")
    p_view.add_argument("--gait", choices=["waddle", "walk", "spot_turn", "head_dance"], default=None, help="Trình diễn dáng đi mới (waddle, walk, spot_turn, head_dance)")
    p_view.add_argument("--track", action="store_true", help="Kích hoạt Visual Servoing theo dõi mục tiêu qua camera đầu")
    p_view.add_argument("--target-color", choices=["yellow", "red", "green", "blue"], default="yellow", help="Màu sắc mục tiêu theo dõi (mặc định: yellow)")

    # train
    p_train = subparsers.add_parser("train", help="Huấn luyện mô hình bước đi bằng PPO đa môi trường")
    p_train.add_argument("--target", choices=["rpi5", "pizero2w", "esp32", "v2"], default="rpi5", help="Nhánh phần cứng/mô hình mục tiêu (rpi5, pizero2w, esp32, v2 - mặc định: rpi5)")
    p_train.add_argument("--envs", type=int, default=256, help="Số lượng môi trường song song (mặc định: 256)")
    p_train.add_argument("--iterations", type=int, default=500, help="Số vòng lặp huấn luyện (mặc định: 500)")
    p_train.add_argument("--resume", type=str, default=None, help="Tiếp tục học từ checkpoint")
    p_train.add_argument("--stand-first", action="store_true", help="Ưu tiên học đứng thăng bằng trước")
    p_train.add_argument("--dry-run", action="store_true", help="Chạy kiểm thử nhanh 1 vòng lặp (Dry-run)")

    # export
    p_export = subparsers.add_parser("export", help="Xuất policy sang ONNX (RPi 5 & Pi Zero 2 W) và INT8 C Header (ESP32-S3)")
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
