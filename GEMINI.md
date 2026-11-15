# DUCKV1 PROJECT CONSTITUTION (Cross-Session & Cross-Model Synchronized Protocol)

> **Language Protocol**: VIETNAMESE (Giao tiếp, báo cáo, giải thích 100% bằng Tiếng Việt; tên biến/hàm/commit message bằng Tiếng Anh).
> **Áp dụng bắt buộc cho mọi phiên làm việc (Sessions) và mọi mô hình AI (Gemini, Claude, Subagents, Reviewers).**

---

## 🔒 QUY TẮC 3 BƯỚC BẢO TOÀN & BÀN GIAO PHIÊN (MANDATORY 3-STEP SESSION PROTOCOL)

Trong **mọi phiên làm việc** của dự án `duckV1`, trước khi thử nghiệm tính năng mới hoặc trước khi kết thúc phiên, **BẮT BUỘC** thực hiện đầy đủ 3 bước sau:

### 1. Đánh dấu Điểm Neo Khôi Phục (Anchor Tagging — Git + Checkpoint `.pt`)
- Khi có một bản code và mô hình chạy tốt/ổn định, phải **gắn Git tag rõ ràng** (ví dụ: `v1-flat-walk-stable`) vào đúng commit đó.
- Đồng thời **lưu bản sao/snapshot cố định của file checkpoint `.pt`** tương ứng với tên tag (ví dụ: `models/checkpoints/v1-flat-walk-stable.pt`).
- **Mục tiêu**: Sau này thử nghiệm bất kỳ thuật toán hay hàm thưởng nào bị hỏng, luôn có điểm neo (cả code lẫn trọng số `.pt`) để quay về ngay lập tức.

### 2. Tách biệt Code Đang Thử Nghiệm sang Nhánh WIP Riêng (WIP Branch Isolation)
- Mọi thay đổi đang sửa dở hoặc đang thử nghiệm (như SAC, BC, Navigation, tinh chỉnh Reward mới) **KHÔNG ĐƯỢC để bừa trên nhánh `main`**.
- Phải cất vào một nhánh riêng có tiền tố `wip/...` (ví dụ: `wip/nav-sac`). Không vứt bỏ code đã viết, chỉ tách khỏi nhánh chính (`main`) để giữ nhánh chính luôn sạch và chạy ổn định.

### 3. Ghi Nhật Ký Bàn Giao 3–5 Dòng vào `README.md` và `his.md` (3-5 Line Handover Log)
- Cuối mỗi đợt làm việc, bắt buộc ghi ngắn gọn **3–5 dòng** vào cả `README.md` (mục Trạng Thái Hiện Tại) và `his.md`:
  1. **Đang dừng ở đâu**: Nhánh nào, tag nào, checkpoint `.pt` nào đang là bản chạy tốt nhất.
  2. **Đã thử những gì**: Các nhánh `wip/...`, thuật toán, hoặc các file vừa thử nghiệm và kết quả.
  3. **Giả thuyết tiếp theo là gì**: Bước tiếp theo cần làm khi quay lại (dù là phiên sau hay 3 tháng sau).

---

## 📌 TRẠNG THÁI HIỆN TẠI (Cập nhật: 2026-11-15)
- **Đang dừng ở đâu:** Nhánh `main` gắn tag `v1-flat-walk-stable` kèm checkpoint neo `models/checkpoints/v1-flat-walk-stable.pt`; bài 1 đi bộ mặt phẳng Stage 0 (`cyberduck_v2.xml`) đang hội tụ ổn định (`Reward: -7.24 -> +2.19`, `DistX: +0.36m`, `Succ: 8%`, `Std: ~0.20`, hết lỗi nổ `log_std` và đi lùi).
- **Đã thử những gì:** Đã thử nghiệm 5 mô-đun nâng cấp (Randomize Pillars, Metrics `Unstab/Coll/MinClr`, BC Warmup Actor+Critic, 5-ray LiDAR Corridor Reward, 2-Layer Nav + SAC) và chạy audit $\ge 50$ vòng cho 6 bài; phát hiện & sửa 6 lỗi ở 7 file (`run_parallel_sim.py`, `env_config.py`, `train_ppo.py`, `pretrain_bc.py`, `train_sac.py`, `nav_planner.py`, `evaluate_models_sim.py`) — toàn bộ 7 file này đã được cất riêng sang nhánh `wip/nav-sac` (commit `650269db`).
- **Giả thuyết tiếp theo:** Chờ đợt train PPO Stage 0 dài hạn (`2300` vòng, `num_envs >= 64`) hội tụ vững dáng đi phẳng trên `main`, sau đó mới cherry-pick từng phần từ `wip/nav-sac` (ưu tiên lọc tia LiDAR chạm sàn + BC warmup) để lên Stage 1–3 né cọc mà không làm hỏng dáng đi cơ sở.
