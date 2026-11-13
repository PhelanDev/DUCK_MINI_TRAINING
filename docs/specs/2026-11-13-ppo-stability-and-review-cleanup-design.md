# PPO Training Stability & Review Cleanup Design Spec

**Date:** 2026-11-13
**Author:** Antigravity Autonomous Meta-Orchestrator & Pair Programmer
**Status:** Approved
**Target:** PPO Training Pipeline (`scripts/train_ppo.py`), Parallel Runner (`training/run_parallel_sim.py`), Test Suite (`tests/test_ppo_training.py`, `tests/test_deployment_artifacts.py`)

---

## 1. Bối cảnh & Mục tiêu (Problem Statement & Motivation)

Qua đối chiếu kết quả huấn luyện dài hạn (`--target v2 --iters 10000 --resume`) và đợt Final Whole-Branch Review trên [plan.md](file:///E:/duckV1/plans/261113-0122-fix-v2-training-and-sim2real-audit/plan.md), hệ thống vẫn còn hai nhóm vấn đề chưa được xử lý trong `scripts/train_ppo.py`, `training/run_parallel_sim.py` và `tests/test_deployment_artifacts.py`:

1. **Bất ổn thuật toán PPO khi train dài & chuyển Stage curriculum (`scripts/train_ppo.py`)**:
   - **Critic bị khóa gradient ở `v_clipped` (`L551-557`)**: Giá trị `v_clipped = mb_val + torch.clamp(new_values - mb_val, -clip_epsilon, clip_epsilon)` (`clip_epsilon = 0.2`) kết hợp `torch.max(v_loss_unclipped, v_loss_clipped)` triệt tiêu gradient của Critic bằng `0` mỗi khi return thô (`mb_ret`) tăng lệch vượt `±0.2` so với `mb_val`, khiến `ValLoss` leo thang (`3.9091 -> 14.4634`) và làm lệch GAE advantages.
   - **Thiếu điều chỉnh LR theo KL & early-stop (`L229, L346, L532-570`)**: Learning rate cố định `3e-4` suốt `10000` iterations qua `num_epochs=4` dễ phá policy khi advantage dao động mạnh lúc chuyển từ Stage 0 (địa hình phẳng) sang Stage 1 (vật cản slalom).
   - **Exploration `self.log_std` chưa chặn biên & `entropy_coef` cố định (`L77, L134, L233`)**: `log_std` tự do trôi dẫn tới nguy cơ sụp noise sớm hoặc bùng nhiễu hành động trên terrain khó.
   - **Đánh dấu nhầm `_best.pt` khi thăng hạng Stage (`L606-705`)**: `current_stage` được nâng trước khi chấm `is_best`, khiến checkpoint mang nhãn Stage mới bằng metric của Stage cũ; đồng thời `best_reward = max(best_reward, cur_reward)` khóa ngưỡng thưởng Stage cao theo kỷ lục địa hình phẳng Stage 0.
2. **3 điểm Minor từ Final Whole-Branch Review**:
   - Gán trùng `self.cfg` tại `training/run_parallel_sim.py:87,92`.
   - Tên fallback `_best.pt` chưa thống nhất với `_latest.pt` khi `save_path` kết thúc bằng `*_ppo.pt` tại `scripts/train_ppo.py:284-295`.
   - Các test gọi trình biên dịch `g++` trong `tests/test_deployment_artifacts.py` thiếu guard `shutil.which("g++")`.

Tài liệu này đặc tả phương án **Approach A**: ổn định hóa PPO theo chuẩn RSL-RL/CleanRL và xử lý 3 điểm Minor mà **vẫn giữ tương thích 100%** với cấu trúc `ActorCritic` hiện tại, các checkpoint cũ (`cyberduck_v2_best.pt`, `microduck_best.pt`), `apply_timeout_bootstrap`, ONNX export và INT8 C++ export.

---

## 2. Ràng buộc Toàn cục (Global Constraints)

- **Giữ nguyên kiến trúc mạng & tương thích export**: `ActorCritic` giữ `obs_dim=42`, `action_dim=8`, `hidden_dim=256`, `ELU`, `self.actor`, `self.critic`, và `self.log_std` là `torch.nn.Parameter` shape `(action_dim,)`. Không thêm lớp `RunningMeanStd` vào observation hay return scaling làm sai lệch `export_policy.py` và `apply_timeout_bootstrap`.
- **Tương thích checkpoint cũ & `duck.py`**: Checkpoint cũ không có khóa `"learning_rate"` vẫn nạp bình thường khi `--resume` hoặc `--load-checkpoint`. Tham số mặc định của `train_ppo(...)` và `main()` không phá vỡ bất kỳ lệnh gọi nào từ `duck.py`.
- **An toàn môi trường test**: Các test hỗn hợp Python + C++ (ví dụ `test_deploy_proj_grav_tilted_and_nominal_action_match_sim`) không được skip phần Python assertions nếu máy thiếu `g++`.

---

## 3. Thiết kế Chi tiết (Architecture & Components)

### 3.1 Kẹp biên `log_std` & Giảm tuyến tính `entropy_coef` (`scripts/train_ppo.py`)

- **`ActorCritic.__init__` & `get_action_and_value`**:
  - Bổ sung tham số mặc định `min_log_std: float = -3.0` (`std >= ~0.0498`) và `max_log_std: float = 0.0` (`std <= 1.0`), giữ nguyên `init_noise_std: float = 0.20` (`log_std ≈ -1.609` nằm gọn bên trong `[-3.0, 0.0]`).
  - Trong `get_action_and_value`, tính `std = torch.exp(torch.clamp(self.log_std, self.min_log_std, self.max_log_std)).expand_as(mean)` để bảo vệ cả bước forward khi nạp từ checkpoint cũ có `log_std` cực đoan.
  - Ngay sau mỗi `optimizer.step()`, kẹp tại chỗ trong `with torch.no_grad():`: `model.log_std.clamp_(model.min_log_std, model.max_log_std)` nhằm tránh trôi tham số khỏi vùng hợp lệ.
- **Linear Entropy Decay**:
  - Thêm tham số `entropy_coef_end: Optional[float] = None` vào `train_ppo(...)` và flag `--entropy-coef-end` (type `float`, default `None`) vào CLI `main()`.
  - Giá trị đích hiệu dụng `eff_entropy_end = min(entropy_coef, 0.002)` khi `entropy_coef_end is None`, ngược lại dùng `float(entropy_coef_end)`.
  - Đầu mỗi iteration trong `range(start_iteration, total_target_iters + 1)`:
    - Nếu `total_target_iters <= 1`: `progress = 0.0`.
    - Ngược lại: `progress = min(1.0, max(0.0, (iteration - 1) / (total_target_iters - 1)))`.
    - `current_entropy_coef = entropy_coef + (eff_entropy_end - entropy_coef) * progress`.
  - Dùng `current_entropy_coef` trong công thức `total_loss` và ghi lại vào `metrics["entropy_coef"]`.

### 3.2 Adaptive KL Learning Rate & Epoch Early-Stop (`scripts/train_ppo.py`)

- Thêm tham số vào `train_ppo(...)` và `main()`:
  - `target_kl: float = 0.015` (`--target-kl`, đặt `<= 0` để tắt adaptive KL & early-stop).
  - `min_lr: float = 1e-5` (`--min-lr`), `max_lr: float = 1e-3` (`--max-lr`).
- **Tính Schulman `approx_kl` (luôn không âm trong kỳ vọng)**:
  - Trong vòng minibatch, dưới `with torch.no_grad():`:
    `approx_kl = ((ratio - 1.0) - log_ratio).mean()`
  - Cộng dồn `epoch_kl` theo từng epoch và `total_kl` theo tổng minibatch đã chạy.
- **Epoch Early-Stop**:
  - Cuối mỗi epoch, nếu `target_kl > 0` và `epoch_mean_kl > 2.0 * target_kl`: dừng vòng lặp `num_epochs` sớm (`break`) để chặn các epoch dư thừa làm hỏng policy.
- **Adaptive LR cuối phase update**:
  - Với `iter_mean_kl = total_kl / update_count`: nếu `target_kl > 0` và `update_count > 0`:
    - Nếu `iter_mean_kl > target_kl * 2.0`: `current_lr = max(min_lr, current_lr / 1.5)`
    - Else nếu `0.0 < iter_mean_kl < target_kl * 0.5`: `current_lr = min(max_lr, current_lr * 1.5)`
  - Đồng bộ ngay vào `for pg in optimizer.param_groups: pg["lr"] = current_lr`.
- **Lưu & khôi phục LR khi `--resume`**:
  - Ghi `"learning_rate": current_lr` vào `best_ckpt`, periodic `ckpt`, và final `checkpoint`.
  - Khi `resume=True` và `"learning_rate" in ckpt`: đặt `current_lr = min(max_lr, max(min_lr, float(ckpt["learning_rate"])))` (vẫn tương thích `min_lr`/`max_lr` mới) và cập nhật `optimizer.param_groups`. Khi nạp checkpoint cũ thiếu khóa `"learning_rate"` hoặc `resume=False`, bắt đầu từ tham số `learning_rate`.
- **Chẩn đoán log & metrics**:
  - Bổ sung vào `metrics`: `"approx_kl"`, `"learning_rate"`, `"entropy_coef"`, `"action_std_mean"` (`float(torch.exp(torch.clamp(model.log_std, model.min_log_std, model.max_log_std)).mean().item())`).
  - Thêm vào dòng log iteration: `KL: {metrics['approx_kl']:.4f} | LR: {metrics['learning_rate']:.1e} | Std: {metrics['action_std_mean']:.3f}`.

### 3.3 Sửa lỗi Critic Gradient Lock (`scripts/train_ppo.py`)

- Thay thế khối `v_loss_unclipped` / `v_clipped` / `torch.max` ở `scripts/train_ppo.py:551-557` bằng loss Huber không kẹp biên:
  ```python
  mb_ret = b_returns[mb_inds]
  value_loss = 0.5 * F.smooth_l1_loss(new_values, mb_ret, beta=2.0)
  ```
- Cơ sở: giữ nguyên ngưỡng kháng outlier bootstrap (đạo hàm bị giới hạn bởi `±0.5` khi sai số `> 2.0`, bậc hai mịn khi sai số `<= 2.0`), đồng thời loại bỏ hoàn toàn vùng gradient `0` do `torch.clamp(..., -clip_epsilon, clip_epsilon)` gây ra khi Critic cần đuổi theo mức return dài hạn (`10..50+`).

### 3.4 Chấm `_best.pt` theo `rollout_stage` & Phanh LR khi lên Stage (`scripts/train_ppo.py`)

- Trước khi gọi `next_stair_stage` / `next_stage`, lưu lại `rollout_stage = current_stage`.
- Khi curriculum nâng hạng (`new_stage > rollout_stage`):
  - Cập nhật `current_stage = new_stage`, `runner.set_stage(current_stage)`.
  - Phanh an toàn LR về tối đa base LR: `current_lr = min(current_lr, learning_rate)` và đồng bộ `optimizer.param_groups`, giúp iteration đầu tiên trên terrain khó không khởi chạy với LR đã bị tăng lên `1e-3` ở Stage cũ.
- Khi đánh giá `is_best` (với `not dry_run and iteration > 0`):
  - Nếu `rollout_stage > best_stage and cur_dist_x >= 0.0 and (cur_dist_x > 0.0 or cur_reward > 0.0)`:
    - Đánh dấu `is_best = True`, cập nhật `best_stage = rollout_stage`, **đặt lại baseline mới** `best_reward = cur_reward` và `best_dist_x = cur_dist_x`.
  - Else nếu `rollout_stage == best_stage` và thỏa tiêu chí vượt trội:
    - `(cur_reward >= best_reward and cur_dist_x >= best_dist_x) and (cur_reward > best_reward or cur_dist_x > best_dist_x)` hoặc `(cur_dist_x > best_dist_x and cur_reward >= best_reward - 0.05)` -> `is_best = True`, cập nhật `best_reward = max(best_reward, cur_reward)`, `best_dist_x = max(best_dist_x, cur_dist_x)`.
  - Trong `best_ckpt`, ghi `"stage": rollout_stage` (đúng stage đã thực sự đo rollout metric) và `"best_stage": best_stage`. Periodic `ckpt` và final `checkpoint` tiếp tục ghi `"stage": current_stage` để `--resume` khôi phục đúng curriculum, đồng thời ghi `"best_stage": best_stage` (cập nhật khi `resume`: `best_stage = max(best_stage, int(ckpt.get("best_stage", prev_best_stage)))` — **không** nâng `best_stage = max(best_stage, current_stage)` khi `best_save_path` chưa thực sự có kỷ lục ở stage mới).

### 3.5 Xử lý 3 điểm Minor từ Final Review

1. **`training/run_parallel_sim.py:87-92`**: Xóa dòng gán trùng `self.cfg = cfg if cfg is not None else _DEFAULT_CFG` ở dòng `92`.
2. **`scripts/train_ppo.py:284-295`**: Rút chung một biến tiền tố `ckpt_prefix = save_path.stem[:-4] if save_path.stem.endswith("_ppo") else save_path.stem` cho cả:
   - `best_save_path = save_path.parent / f"{ckpt_prefix}_best.pt"` (khi `best_save_path is None`)
   - `latest_save_path = save_path.parent / f"{ckpt_prefix}_latest.pt"`
   - Kết quả: `microduck_ppo.pt -> microduck_best.pt`, `cyberduck_v2_ppo.pt -> cyberduck_v2_best.pt`, và `microduck_ppo_dryrun.pt -> microduck_ppo_dryrun_best.pt`.
3. **`tests/test_deployment_artifacts.py`**: Tách helper dùng chung `_compile_and_run_cpp(cpp_source, cpp_path, exe_path)` (hoặc check `shutil.which("g++")`):
   - Với các test chỉ kiểm tra C++ (`L209`, `L431`, `L536`, `L1212`, `L1299`), gọi `pytest.skip("g++ compiler not available")` khi `shutil.which("g++") is None`.
   - Với test hỗn hợp `test_deploy_proj_grav_tilted_and_nominal_action_match_sim` (`L880-1037`), chỉ bỏ qua khối compile/chạy C++ nếu `shutil.which("g++") is None` để các assertion Python của RPi5 và PiZero2W (`L1022-1037`) vẫn chạy đầy đủ.

---

## 4. Chiến lược Kiểm thử & Tiêu chí Nghiệm thu (Testing & Acceptance Criteria)

- **Bổ sung unit tests trong `tests/test_ppo_training.py`**:
  1. `test_actor_critic_log_std_clamping_preserves_gradients`: xác nhận `std` bị giới hạn trong `[exp(min_log_std), exp(max_log_std)]` kể cả khi set cực đoan, vẫn là `nn.Parameter` có gradient hợp lệ ở vùng trung tâm.
  2. `test_critic_value_loss_updates_across_large_return_gap`: tái hiện tình huống `|mb_ret - mb_val| > clip_epsilon` (`0.2`) và chứng minh Critic nhận gradient khác 0 và giảm sai số sau update (regression test cho lỗi `v_clipped`).
  3. `test_ppo_adaptive_kl_and_entropy_decay`: xác minh `approx_kl >= 0`, `learning_rate` tự giảm khi `target_kl` rất nhỏ, và `entropy_coef` giảm tuyến tính đúng biên qua nhiều iteration.
  4. `test_best_save_path_prefix_and_resume_lr_compat`: kiểm tra fallback `cyberduck_v2_ppo.pt -> cyberduck_v2_best.pt` khi không truyền `best_save_path`, `--resume` khôi phục `learning_rate`, và nạp checkpoint cũ thiếu khóa `"learning_rate"` không lỗi.
  5. `test_stage_promotion_best_ckpt_uses_rollout_stage_and_resets_baseline`: mock curriculum chuyển `0 -> 1` sau iter 1, kiểm tra `best_ckpt` iter 1 giữ `"stage": 0` trong khi `latest` có `"stage": 1`, và iter 2 (chạy thật Stage 1 với reward thấp hơn kỷ lục Stage 0) vẫn cập nhật được `best_ckpt` theo baseline mới của Stage 1.
- **Kiểm thử guard `g++` ở `tests/test_deployment_artifacts.py`**: kiểm tra đường dẫn skip khi mock `shutil.which` trả về `None` không làm lỗi test hỗn hợp.
- **Chạy nghiệm thu toàn repo**: `python duck.py verify` và `pytest -q` phải pass `100%` (0 lỗi, 0 test bị sửa yếu assertion).
