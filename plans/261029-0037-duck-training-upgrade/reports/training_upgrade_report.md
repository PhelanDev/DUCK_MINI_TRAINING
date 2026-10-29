# Báo Cáo Nghiệm Thu Nâng Cấp Huấn Luyện AI (Duck Training Upgrade Report)

- **Ngày thực hiện:** 2026-10-29
- **Kế hoạch tham chiếu:** `plans/261029-0037-duck-training-upgrade/plan.md`
- **Trạng thái:** HOÀN THÀNH TOÀN DIỆN (104/104 Unit Tests PASS)

---

## 1. Mục Tiêu Đạt Được

1. **Khắc phục triệt để Time-Limit Bootstrapping trong PPO:**
   - Phân định rạch ròi giữa ngã vật lý (`fallen: root_z < 0.08 | grav_z > -0.3 | diverged`) và hết giờ mô phỏng (`timeouts: episode_steps >= max_episode_steps`).
   - Ghi nhận `terminal_obs` nguyên bản ngay trước khi `reset_env(i)`.
   - Hàm thuần `apply_timeout_bootstrap(rewards, time_outs, terminal_values, gamma)` bù thêm $\gamma \cdot V(s_{\text{terminal}})$ cho phần thưởng tức thời, triệt tiêu hiện tượng gai dao động (spikes) giả của hàm giá trị Critic.

2. **Cơ chế Chống Kẹt Trước Cột (Command Relaxation & Approach-Velocity Penalty):**
   - Mặt nạ suy giảm `ang_vel_mask(d_min)` cho phép robot tự do quay đầu khi tiếp cận vật cản ($0.25\text{ m} \to 0.55\text{ m}$).
   - Thay thế phạt khoảng cách tĩnh bằng hàm phạt vận tốc tiếp cận `avoid_vel_penalty(d_min, d_min_dot, sigma_obs)`: Khi di chuyển lùi hoặc né xa vật thể ($d_{\min\_dot} > 0$), hàm phạt triệt tiêu về $0.0$. Càng lao nhanh về phía vật thể ở cự ly gần thì mức phạt càng tăng theo hàm mũ âm.

3. **Huấn Luyện Bước Chân Cao (Foot Clearance) & Chống Vấp (Stumble Penalty):**
   - Trích xuất động học bàn chân ($z_{\text{foot}}, v_{xy}$) và lực tiếp xúc 3D ($F_x, F_y, F_z$) bằng `mujoco.mj_contactForce` qua ma trận quay `contact.frame` sang hệ quy chiếu thế giới (World Frame).
   - Hàm `compute_foot_clearance_and_stumble` (vector hóa):
     * Phạt chân kéo lê sàn trong pha vung chân: $-\sum (z - h_{\text{target}})^2 \cdot \sqrt{\|v_{xy}\|} \cdot \text{is\_swing}$.
     * Phạt va chạm ngang khi mũi chân va phải gờ đất: $f_{xy} > 2|f_z|$ và $f_{xy} > 2.0\text{ N}$.

4. **Gờ Đất Chìm Trong XML & Lộ Trình Huấn Luyện Địa Hình (Terrain Curriculum):**
   - Bổ sung 3 geom `hurdle_1`, `hurdle_2`, `hurdle_back` vào `microduck_8dof.xml` chìm dưới sàn ($z_{\text{top}} \le 0.0\text{ m}$) ở Stage 0.
   - Module `training/terrain_curriculum.py` quản lý 4 cấp độ (Stage 0 đến 3), tự động nâng độ cao gờ ($0.005\text{ m} \to 0.015\text{ m} \to 0.025\text{ m}$) và co hẹp khoảng cách các cột chướng ngại vật khi robot đạt đủ tiêu chí bước đi ổn định liên tục trong 10 vòng lặp.

5. **Kiểm Thử & Xuất Mô Hình:**
   - Toàn bộ 104 bài test tự động chạy qua PyTest đều đạt 100% PASS.
   - Script `scripts/export_policy.py` xuất thành công cả file ONNX Float32 (`models/microduck_rpi5_policy.onnx`) và các header C/C++ lượng tử hóa INT8 (`esp32_policy_weights.h`, `esp32_quant_config.h`).

