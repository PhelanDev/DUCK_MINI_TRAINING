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
- **Đang dừng ở đâu:** Nhánh `main` đã hợp nhất 4 bản vá chống nhún từ `wip/fix-gait-touch-reward`; điểm neo `v1-flat-walk-stable` (`dc948dbd` + `models/checkpoints/v1-flat-walk-stable.pt`) và nhánh `wip/nav-sac` (`650269db`) vẫn được bảo toàn nguyên vẹn.
- **Đã thử những gì:** Kiểm chứng thực nghiệm trên WSL GPU (`cyberduck_v2.xml`, `64x256`, `--max-lr 5e-4`) sau 4 bản vá (`touch` site `0.027 0.015 0.005`, `r_foot_contact [:2]`, `r_gait_sync` + `dwell_gate/contact_gate/vx>0.02`, lọc LiDAR `z_hit<=0.012`) đã hết kẹt nhún tại chỗ: Stage 0 đạt `DistX: 0.96m`, `Succ: 80.0%`, `Reward: +2.317` ở `Iter 150` và tự thăng lên Stage 1 (`DistX: 1.01m`, `Reward: +2.521` tại `Iter 165`, `Std: 0.171`).
- **Giả thuyết tiếp theo:** Hoàn tất train `1500` iters qua Stage 1–3, lưu snapshot `.pt` + gắn tag neo mới cho bản đa tầng trước khi cherry-pick các thành phần còn lại từ `wip/nav-sac`.
