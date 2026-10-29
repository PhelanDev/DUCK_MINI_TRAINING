# Duck Training Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: dùng `subagent-driven-development` (khuyến nghị) hoặc `executing-plans`. Tiến độ theo checkbox `- [ ]`.

**Goal:** Sửa time-limit bootstrapping của PPO, thêm reward chống kẹt trước cột, thêm bước cao qua gờ đất, và curriculum cho cột/gờ.

**Architecture:** Mọi thay đổi nằm trong luồng sẵn có: `DuckParallelSim.step()` (chia sẻ **một** `self.model` cho mọi env, nên sửa `model.geom_pos` là áp dụng toàn cục), `compute_rewards()` + `REWARD_WEIGHTS` trong `env_config.py`, và vòng rollout trong `train_ppo.py`. Không thêm dependency.

**Tech Stack:** Python, NumPy, MuJoCo, PyTorch, pytest.

**Spec:** yêu cầu của user trong hội thoại (3 hạng mục + lộ trình 3 bước).

## Global Constraints

- Obs dim giữ nguyên (42 khi có lidar; foot_contact = `obs[33:37]`, lidar = `obs[37:42]`). Blind locomotion với gờ đất.
- `max_episode_steps = 500`, `gamma = 0.99`.
- Cột hiện tại trong [microduck_8dof.xml](file:///d:/duckV1/models/mjcf/microduck_8dof.xml#L34-L37): front `(0.85, 0, 0.25)`, left `(0.65, 0.35, 0.25)`, right `(0.65, -0.35, 0.25)`, back `(-0.75, 0, 0.25)`. Đây là vị trí gốc (Stage 3), **không** phải 0.80 như bản nháp.
- Gờ (half-size, từ bản nháp): `hurdle_1 (0.30,0,·)` size `0.02 0.60 0.015`; `hurdle_2 (0.55,0,·)` size `0.02 0.60 0.020`; `hurdle_back (-0.35,0,·)` size `0.02 0.60 0.015`. Chìm: `pos_z = -0.02`. Nâng: `0.005 → 0.015 → 0.025`.
- Reward: `w_clearance ∈ [1.5, 2.5]`, `w_stumble = 0.5`, `h_target = 0.06`, `d_safe = 0.55`, `d_danger = 0.25`, `w_ang ∈ [0.1, 1.0]`.
- Chỉ áp dụng cho `models/` + `training/` + `scripts/` gốc. Hai pipeline `rpi5_stack`, `dual_esp32s3_stack` sync **sau** khi gốc ổn định (ngoài phạm vi plan này).
- Không `git push` khi chưa được user duyệt.

## Review Focus

1. Env vừa **ngã vừa hết giờ** ở cùng step: phải coi là ngã (không bootstrap).
2. Không env nào timeout trong step: không được gọi critic với tensor rỗng.
3. `terminal_obs` phải là obs **trước** `reset_env` (không phải obs sau reset).
4. Chân chạm gờ cạnh hông (lực ngang lớn, `F_z ≈ 0`): `f_xy > 2.0` chặn false-positive khi lực nhiễu nhỏ.
5. Lidar (`obs[37:42]`) có thể quét thấy gờ sau khi nâng → kiểm tra `geomgroup`/`contype` để lidar không trả về khoảng cách gờ như cột.
6. Curriculum chỉ được **tăng** stage (không lùi) và phải log stage hiện tại.

## File Structure

| File | Trách nhiệm |
|---|---|
| `training/run_parallel_sim.py` (sửa) | tách `time_outs`/`terminal_obs`; foot kinematics + contact force; áp dụng curriculum lên `model.geom_pos` |
| `training/env_config.py` (sửa) | `REWARD_WEIGHTS` mới; `compute_rewards` thêm 3 thành phần |
| `training/terrain_curriculum.py` (tạo) | logic thuần: bảng stage, `next_stage()`, `apply_stage(model, stage)` |
| `scripts/train_ppo.py` (sửa) | bù `γ·V(s_terminal)` cho timeout; gọi curriculum mỗi iteration |
| `models/mjcf/microduck_8dof.xml` (sửa) | thêm 3 geom `hurdle_*` chìm |
| `tests/test_time_limit_bootstrap.py`, `tests/test_foot_rewards.py`, `tests/test_terrain_curriculum.py` (tạo) | test cho từng task |

---

### Task 1: Time-limit bootstrapping (làm trước tiên)

**Files:**
- Modify: [run_parallel_sim.py:303-320](file:///d:/duckV1/training/run_parallel_sim.py#L303-L320) — hiện đã có `timeouts` nhưng gộp vào `dones` và reset ngay; `infos[i]["timeout"]` chưa đủ.
- Modify: [train_ppo.py:282-292](file:///d:/duckV1/scripts/train_ppo.py#L282-L292) — chưa dùng `infos` (`_`).
- Test: `tests/test_time_limit_bootstrap.py`

**Interfaces:**
- Produces: `infos[i]["time_outs"]: bool` (= timeout **và không** ngã), `infos[i]["terminal_obs"]: np.ndarray(obs_dim,)` có mặt khi `dones[i]`.
- Produces: `apply_timeout_bootstrap(rewards: Tensor, time_outs: Tensor[bool], terminal_values: Tensor, gamma: float) -> Tensor` trong `train_ppo.py` (hàm thuần để test).
- Lưu ý: critic gọi qua `model.forward_critic(obs)` (không có `agent.get_value`); kiểm tra shape trả về để `.squeeze(-1)` đúng.

- [ ] **Step 1: Viết test fail** — trong `test_time_limit_bootstrap.py`:
  - `test_apply_timeout_bootstrap_adds_gamma_v_only_on_timeouts`: rewards `[1,1,1]`, mask `[F,T,F]`, values `[_,5.0,_]` (đã lọc), γ=0.99 → `[1, 1+4.95, 1]`.
  - `test_apply_timeout_bootstrap_noop_when_no_timeouts`: mask toàn False → trả về rewards không đổi, **không** gọi critic.
  - `test_sim_timeout_sets_time_outs_and_terminal_obs`: env `max_episode_steps=2`, step 2 lần → `infos[i]["time_outs"] is True`, `terminal_obs.shape == (obs_dim,)`, và `terminal_obs != returned next_obs[i]` (obs đã reset).
  - `test_sim_fall_is_not_timeout`: ép `root_z < 0.08` đồng thời `episode_steps >= max` → `time_outs is False`, `dones is True`.
- [ ] **Step 2: Chạy, xác nhận FAIL** — `pytest tests/test_time_limit_bootstrap.py -v` → FAIL (thiếu key/hàm).
- [ ] **Step 3: Sửa `step()`** — tính `fallen = (root_z<0.08)|(grav_z>-0.3)|diverged`; `dones = fallen | timeouts`; `infos[i]["time_outs"] = bool(timeouts[i] and not fallen[i])`; với mỗi env done: `infos[i]["terminal_obs"] = next_obs[i].copy()` **trước** `reset_env`. Giữ key `"timeout"` cũ (đang có caller khác — grep trước khi xóa).
- [ ] **Step 4: Sửa `train_ppo.py`** — nhận `infos`, dựng mask, nếu `.any()` thì lấy `terminal_obs` các env đó, `with torch.no_grad(): v = model.forward_critic(...)`, cộng bù qua `apply_timeout_bootstrap` **trước** khi ghi `rewards_buffer[step]`. Không sửa `compute_gae`.
- [ ] **Step 5: Chạy test PASS** — `pytest tests/test_time_limit_bootstrap.py -v`.
- [ ] **Step 6: Kiểm chứng thực nghiệm** — `python scripts/train_ppo.py` 50 iterations (flags theo `--help`), so sánh `value_loss` quanh khi `ep_len_mean → 500` với baseline (log baseline trước khi sửa, lưu vào `plans/261029-0037-duck-training-upgrade/reports/`). Kỳ vọng: value_loss không còn spike định kỳ.
- [ ] **Step 7: Commit** — `git commit -m "fix(ppo): bootstrap value on time-limit truncation"`.

---

### Task 2: Reward chống kẹt trước cột (Command Relaxation + approach-velocity penalty)

**Files:**
- Modify: [env_config.py:61 (REWARD_WEIGHTS), :189 (compute_rewards)](file:///d:/duckV1/training/env_config.py#L189)
- Modify: `run_parallel_sim.py` — tính `d_min` và `d_min_dot` từ vị trí base tới 4 cột (đọc `model.geom_xpos` của `obstacle_*`, tính trong mặt phẳng XY trừ bán kính cột `0.08`).
- Test: `tests/test_obstacle_rewards.py`

**Interfaces:**
- Produces: `compute_rewards(..., d_min: np.ndarray | None = None, d_min_dot: np.ndarray | None = None, ...)` — mặc định `None` → hành vi cũ (tương thích test hiện có).
- Produces: `ang_vel_mask(d_min) -> np.ndarray` = `clip((d_min - 0.25)/(0.55 - 0.25), 0.1, 1.0)`; nhân vào `tracking_ang_vel`.
- Produces: `avoid_vel_penalty(d_min, d_min_dot, sigma_obs) -> np.ndarray` = `-max(0, -d_min_dot) * exp(-d_min/sigma_obs)`.
- Thay thế phạt khoảng cách tĩnh của `obstacle_avoidance` hiện có bằng `avoid_vel_penalty` (đọc `compute_rewards` để xác nhận thành phần nào đang phạt tĩnh; `sigma_obs` đặt vào `REWARD_WEIGHTS`/config, khởi đầu 0.3).

- [ ] **Step 1: Test fail** — `test_ang_mask_values`: `d=0.60→1.0`, `0.55→1.0`, `0.40→0.5`, `0.25→0.1`, `0.10→0.1`. `test_avoid_penalty_zero_when_receding` (`d_min_dot>0` → 0). `test_avoid_penalty_stronger_when_closer` (cùng `d_dot=-0.3`, `d=0.2` phạt lớn hơn `d=0.8`). `test_compute_rewards_backward_compatible` (không truyền `d_min` → kết quả bằng cũ).
- [ ] **Step 2: Chạy, xác nhận FAIL.**
- [ ] **Step 3: Cài đặt** hai hàm thuần trong `env_config.py`, nối vào `compute_rewards`; `d_min_dot = (d_min - prev_d_min)/control_dt` lưu `self.prev_d_min` (reset trong `reset_env`).
- [ ] **Step 4: PASS** — `pytest tests/test_obstacle_rewards.py -v`.
- [ ] **Step 5: Commit** — `feat(reward): relax ang-vel tracking near obstacles, penalize approach speed`.

---

### Task 3: Foot clearance + stumble reward

**Files:**
- Modify: `run_parallel_sim.py` — trích `foot_z (N,2)`, `foot_vel_xy (N,2,2)`, `foot_contact_forces (N,2,3)`.
- Modify: `env_config.py` — thêm `rew_foot_clearance`, `pen_stumble` vào `compute_rewards` + weights (`foot_clearance: 2.0`, `stumble: 0.5`).
- Test: `tests/test_foot_rewards.py`

**Interfaces:**
- Produces: `compute_foot_clearance_and_stumble(foot_z, foot_vel_xy, foot_contact_forces, target_height=0.06) -> (rew_clearance, pen_stumble)` trong `env_config.py` — **vector hóa theo env** (shape `(N,2)`, `(N,2,2)`, `(N,2,3)` → hai mảng `(N,)`), thay vì bản 1-env trong nháp.
- Nguồn dữ liệu (quyết định, vì `touch` sensor chỉ cho **scalar** lực pháp tuyến, không đủ cho Fx/Fy/Fz): lực 3D lấy bằng `mujoco.mj_contactForce` trên các contact có `geom1/geom2` là `left_foot_geom`/`right_foot_geom`. Gờ chắn ngang trục X nên lực mũi chân nằm ở pháp tuyến contact (nghiêng), vì vậy phải đổi sang frame **world**: nhân `contact.frame` với lực contact để ra `(Fx,Fy,Fz)`, rồi cộng dồn theo từng foot. Vận tốc chân: `mj_objectVelocity` của body `*_foot_link` (flg_local=0), lấy `[3:5]`. Độ cao: `data.xpos[foot_body][2]` (hoặc đáy geom nếu cần so với mặt đất chuẩn).
- Hai lưu ý sửa so với nháp: `is_swing` dùng ngưỡng lực `< 1.0` giữ nguyên; `sqrt(vel_xy_norm)` thêm `+1e-8`-an toàn không cần (norm ≥ 0), nhưng clip `vel_xy_norm` ở ~1.0 m/s để tránh reward bùng nổ khi vung chân.

- [ ] **Step 1: Test fail** — `test_stance_foot_zero_clearance_penalty` (lực `[0,0,3]`, vel 0 → 0); `test_swing_dragging_penalized_more_than_lifted` (z=0.0 phạt > z=0.06; z=0.06 → 0); `test_stumble_triggers_on_lateral_force` (`F=[5,0,1]` → `-1`); `test_no_stumble_on_small_noise` (`F=[1.5,0,0.1]` → 0 do `f_xy ≤ 2.0`); `test_vectorized_shapes` (N=4 → `(4,)`).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Cài đặt** hàm theo công thức ETH: `-Σ (z - h)² · sqrt(‖v_xy‖) · is_swing`; stumble = `-Σ[(f_xy > 2·|f_z|) & (f_xy > 2.0)]`.
- [ ] **Step 4: Nối vào `step()`/`compute_rewards`**, thêm vào `infos` các key `rew_foot_clearance`, `pen_stumble` để log.
- [ ] **Step 5: PASS** + chạy lại `pytest tests -q` (không hồi quy).
- [ ] **Step 6: Commit** — `feat(reward): foot clearance and stumble penalty`.

---

### Task 4: Gờ đất trong XML (chìm) + terrain curriculum

**Files:**
- Modify: [microduck_8dof.xml:34-37](file:///d:/duckV1/models/mjcf/microduck_8dof.xml#L34-L37) — thêm 3 geom `hurdle_*`, `pos_z=-0.02`, `contype=1 conaffinity=1`. Cân nhắc `group` riêng để lidar bỏ qua (xem Review Focus 5).
- Create: `training/terrain_curriculum.py`
- Modify: `run_parallel_sim.py` (gọi `apply_stage` khi khởi tạo / khi `set_stage`), `scripts/train_ppo.py` (đánh giá stage mỗi iteration)
- Test: `tests/test_terrain_curriculum.py`, bổ sung `tests/test_mjcf_model.py`

**Interfaces:**
- Produces: `STAGES: list[Stage]` với `Stage(pillar_scale: float, hurdle_z: float)`:
  | Stage | pillar_scale | hurdle `pos_z` | Điều kiện lên stage |
  |---|---|---|---|
  | 0 | 2.00 | −0.02 (chìm) | khởi đầu |
  | 1 | 1.30 | 0.005 | `ep_len_mean ≥ 400` và `rew_lin_vel ≥ 0.75·max` |
  | 2 | 1.00 (= vị trí XML gốc) | 0.015 | `ep_len_mean ≥ 420` duy trì ở stage 1 |
  | 3 | 1.00 | 0.025 | `ep_len_mean ≥ 420` ở stage 2 |
- `pillar_scale` nhân vào **XY gốc** của XML (front `0.85→×s`…); `1.0` = gốc, nên bảng "bán kính 2.0/1.30/0.80" của bản nháp quy đổi thành hệ số tỉ lệ. "Duy trì" = điều kiện đúng trong ≥ `hold_iters=10` iteration liên tiếp (mặc định, chỉnh được).
- Produces: `next_stage(stage: int, ep_len_mean: float, rew_lin_vel_ratio: float, hold_count: int) -> tuple[int, int]` (stage mới, hold_count mới); `apply_stage(model, stage: int) -> None` ghi `model.geom_pos[gid]` (id qua `mj_name2id`, cache base pos lần đầu). Hurdle `pos_z` ghi trực tiếp; không cần `mj_forward` ngoài `reset_env` hiện có.

- [ ] **Step 1: Test fail** — `test_stage0_hides_hurdles_below_floor` (đỉnh gờ = `pos_z + half_z < 0`: `-0.02+0.015 = -0.005`; hurdle_2: `-0.02+0.02 = 0` → **đặt hurdle_2 `pos_z=-0.025`** nếu đỉnh phải `< 0`; test khẳng định đỉnh ≤ 0 cho mọi gờ); `test_apply_stage_scales_pillars` (stage1 → front x = `0.85*1.3`); `test_apply_stage_idempotent` (gọi 2 lần không nhân dồn); `test_next_stage_requires_hold` (đủ điều kiện 9 iter → ở yên, iter 10 → lên); `test_next_stage_never_regresses`; `test_mjcf_has_hurdles` (3 geom tồn tại, model compile).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Cài đặt** XML + `terrain_curriculum.py`; nối vào sim và train loop; log `stage` cùng metrics, cho phép `--start-stage` để resume.
- [ ] **Step 4: PASS** — `pytest tests/test_terrain_curriculum.py tests/test_mjcf_model.py -v`; chạy `python scripts/verify_model.py` để chắc URDF/MJCF không lệch (nếu script so sánh geom).
- [ ] **Step 5: Commit** — `feat(curriculum): terrain and pillar curriculum with hidden hurdles`.

---

### Task 5: Chạy huấn luyện theo lộ trình & nghiệm thu

- [ ] **Giai đoạn A (Stage 0):** train đến khi đi vững, `foot_clearance` tăng, `p_z` swing ≈ 0.05–0.07 (log trung bình). Lưu checkpoint.
- [ ] **Giai đoạn B (Stage 1→3):** để curriculum tự nâng; theo dõi tỉ lệ ngã và `pen_stumble` mỗi khi lên stage; nếu `ep_len_mean` sụt >30% sau lên stage, giữ `hold_iters` lớn hơn hoặc hạ `w_stumble`.
- [ ] **Nghiệm thu:** vịt qua `hurdle_1/2` (stage 3) không vấp ngã, lách `obstacle_front` quanh `x≈0.45`, không đứng khựng (vận tốc trung bình `v_x > 0.5·cmd`) — ghi video bằng [interactive_viewer.py](file:///d:/duckV1/scripts/interactive_viewer.py) và báo cáo vào `plans/261029-0037-duck-training-upgrade/reports/`.
- [ ] Chạy toàn bộ `pytest tests -q` xanh trước khi báo hoàn tất; **hỏi user** trước khi push.

## Self-Review

- **Coverage:** 1.1/1.2 → Task 1; 2.1 → Task 2; 2.2 + 3.1 + 3.3 (curriculum gờ) → Task 4; 3.2 → Task 3; lộ trình → Task 5. 
- **Lệch so với bản nháp (đã sửa trong plan):** vị trí cột gốc; `agent.get_value` → `model.forward_critic`; touch sensor không cho lực 3D → dùng `mj_contactForce`; hàm foot vector hóa theo env; infos hiện đã có `"timeout"` cần giữ tương thích.
- **Rủi ro đã biết:** `geom_pos` dùng chung nên mọi env cùng stage (chấp nhận, YAGNI cho per-env); đồng bộ 2 pipeline phụ là việc sau.
