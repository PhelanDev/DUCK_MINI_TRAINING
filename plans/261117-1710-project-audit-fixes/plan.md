# Project Audit & Correctness Fixes — Implementation Plan

> **Trạng thái: REVIEWED / IMPLEMENTATION NOT STARTED — 2026-11-17.** Mọi checkbox bên dưới là việc của phiên sửa tiếp theo.
> Phạm vi: toàn working tree tại `bd25e347310640fc6edcc8306d0db4503ed7486e`, nhánh `wip/fix-ppo-det-locomotion`; gồm cả hai chỉnh sửa đường dẫn inference Pi có sẵn. Không giới hạn vào diff của commit cuối.
> Bằng chứng: source + probe CPU/C++/ONNX; GitNexus cùng HEAD, static callers là lower bound. Không chạy robot thật hoặc train dài.
> Vị trí plan: `plans/261117-1710-project-audit-fixes/plan.md`, theo cấu trúc `plans/<YYMMDD-HHMM-slug>/plan.md` của dự án.
> Evidence provenance schema 2 trong §11 là snapshot lịch sử trước khi chuyển vị trí: dirty digest `cbf1acdeb97fcceec82660476e627ca482c5b9eb0906c8997bf310b29cc2de93`, manifest 16 paths. `generated_plan_path` trong snapshot giữ nguyên vị trí cũ để không sửa receipt đã ghi nhận; `plan_path` là vị trí hiện tại. Re-anchor trước implementation.

## 1. Mục tiêu

Đưa các lỗi đã tái hiện thành những task sửa độc lập, có test hồi quy và cổng nghiệm thu. Phiên audit không thay mã nguồn, test, trọng số, branch hay tag. Theo quy ước dự án, implementation cần bảo toàn điểm neo, làm trên nhánh WIP và bàn giao 3–5 dòng ở README/his.

## 2. Hành vi hiện tại

- [verified] `duck.py:199` chọn XML theo target, nhưng chỉ V2 được đặt checkpoint riêng ở `duck.py:208`.
- [verified] `scripts/train_ppo.py:602` tạo runner với config gốc; clock bật theo `training/run_parallel_sim.py:252`, khác packing trên Pi.
- [verified] Runner dựng next observation ở `training/run_parallel_sim.py:1047` trước cập nhật last action ở `:1296`.
- [verified] `scripts/export_policy.py:310` hỗ trợ 37/42 chiều; firmware MCU2 luôn viết đến index 41.
- [verified] Baseline thực chạy: **479 passed, 14 warnings, 146.95s**; kiểm tra vật lý qua `python duck.py verify --skip-tests` thành công. Warning chủ yếu deprecation ONNX exporter, chưa được tính là lỗi correctness.

## 3. Kiến trúc và các ràng buộc liên quan

- [verified] `training/env_config.py` định nghĩa observation/reward; `ParallelSimRunner` phục vụ PPO, BC, SAC và benchmark. Sửa history ở runner chung, giữ reward action-rate dùng action cũ.
- [verified] `pipelines/rpi_infer_core.py` là runtime dùng chung; hai file `infer_rpi5.py`/`infer_pizero2w.py` là wrapper. Giữ nguyên hai edit đường dẫn model đang có.
- [verified] MLP actor/export hiện có các lớp Linear ở 0/2/4, cùng normalization/action transform. Không thêm tanh lần hai hoặc thay layout state dict.
- [verified] `GEMINI.md`/`CLAUDE.md` yêu cầu tiếng Việt, anchor `v1-flat-walk-stable` + checkpoint cùng tên, WIP isolation và handover. Tag và file `.pt` hiện có; audit không đánh giá lại chất lượng vận động của anchor.

## 4. Các vấn đề đã xác minh

P1 = nên sửa trước train/deploy tiếp; P2 = lỗi đầu vào biên hoặc khả năng tái lập.

| ID | Mức | Vị trí | Trigger, bằng chứng và ảnh hưởng |
| --- | --- | --- | --- |
| F1 | P1 | `training/run_parallel_sim.py:1047`, `:1296` | Chạy action 0.25, env không done: returned obs[25:33] toàn 0; gọi get_observations ngay sau cho toàn 0.25. PPO/SAC/BC dùng history trễ một bước, terminal observation cũng chịu cùng ordering. |
| F2 | P1 | `pipelines/rpi_infer_core.py:65`, `:496` | PPO root có clock [sin,cos] ở [35:37], pack Pi không truyền clock. Cùng trạng thái reset: train [0,1], runtime [0,0]. Probe ONNX đang ship của RPi5 trên hai input chỉ khác clock cho max action delta 0.047432456; đây là kết quả của artifact/input đã thử, không phải mọi policy. |
| F3 | P1 | `scripts/export_policy.py:310`; `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino:546` | Export actor 37 chiều thành công → OBS_DIM=37. assemble_observation vẫn ghi LiDAR [37..41]: vượt biên 5 float. Probe C++ trích nguyên hàm firmware làm cả 5 canary 12345 sau buffer đổi thành 1.50; chưa kết luận board đã crash. |
| F4 | P1 | `duck.py:208`; `scripts/train_ppo.py:662` | RPi5/PiZero/ESP32 dùng chung microduck_ppo/latest/best; resume không đối chiếu model identity. Probe RPi5 → resume PiZero cùng path trả success=True, iterations=2, metadata bị đổi sang PiZero. Train mới cũng có thể ghi đè output target khác. |
| F5 | P2 | `scripts/train_ppo.py:665`, `:670`; `:119` | Checkpoint thật bị đặt obs_rms=None vẫn resume success=True vì except: pass, actor cũ đi với RMS mặc định. Loader RMS cũng chưa validate đầy đủ trước mutate; strict exporter không bảo vệ đường resume. |
| F6 | P2 | `scripts/train_ppo.py:975` | 1 env × 1 rollout step: std mặc định correction=1 tạo NaN; train trả success=True, policy_loss=NaN và ghi cả 13 tensor model không finite. CLI cho phép cấu hình này. |
| F7 | P2/phụ | `.gitignore:6`; `duck.py:122` | git ls-files tests/ rỗng, 47 test_*.py chỉ có local. Clean checkout không có bộ test mà duck verify mặc định gọi. Đây là mismatch giữa lệnh verify công bố và nội dung Git; việc ignore test tự nó có thể là chủ ý. |

[verified] F1–F6 dựa source và probe trực tiếp. F7 dựa Git inventory/config. Không đưa các nghi vấn scalar/batch reward hoặc pillar randomization đã bị probe/caller bác bỏ vào danh sách lỗi.

[graph] `context(name=step,file_path=training/run_parallel_sim.py,kind=Method)` tìm caller PPO/BC/benchmark/step_arrays; `impact(target_uid=Method:training/run_parallel_sim.py:ParallelSimRunner.step#2,direction=upstream,maxDepth=2)` trả 5 direct symbols, 14 total; báo lower-bound do 3 receiver call sites unresolved. Không dùng số risk CRITICAL của graph làm severity của lỗi.

## 5. PDG và thứ tự cập nhật trạng thái

- [graph] `impact(mode=pdg,line=1049,target_uid=...step#2,direction=upstream,maxDepth=1)` trả slice cục bộ 3 statements (973,1047,1048), truncated bởi depth; interprocedural reach là callgraph bridge.
- [graph] `pdg_query(mode=flows,target=training/run_parallel_sim.py,variable=last_actions)` trả 0. Không suy luận attribute flow/ordering từ kết quả rỗng.
- [verified] Ordering F1 được chứng minh bằng source và probe: reward ở `:1170` cần history cũ; observation cho bước kế tiếp cần action vừa thực thi. Snapshot terminal trước auto-reset; reset history zero cho đúng env.
- [graph] `explain(target=scripts/train_ppo.py)` không có persisted finding; điều đó không chứng minh checkpoint/load an toàn vì field/implicit flows không được mô hình hóa đầy đủ.

## 6. Thay đổi đề xuất

1. **F1 — runner:** giữ previous-actions riêng cho reward, cập nhật history quan sát trước khi đóng gói next/terminal observations. Không thay công thức action-rate.
2. **F2 — schema quan sát:** ghi rõ embed_phase_clock, period, control_dt và quy tắc reset trong checkpoint → ONNX metadata. `load_actor_from_checkpoint`/`export_onnx` truyền metadata; `BaseRPiInferenceEngine` đọc và pack đúng clock cho cả Pi. Clock phải khớp runner ở cả zero-command và moving; không tự thêm quy tắc dừng clock mà runner không có. Legacy thiếu metadata phải có chế độ explicit/nguồn gốc đã xác minh, không đoán từ obs_dim.
3. **F3 — ràng buộc ESP32:** với firmware hiện tại, `export_esp32_headers` chỉ nhận 42 chiều/8 action; reject 37 trước tạo cả config và weights. Giữ ONNX legacy 37 nếu còn được hỗ trợ riêng. Thêm compile-time guard nhất quán OBS_DIM/layer input trong firmware để artifact cũ không âm thầm chạy sai.
4. **F4 — checkpoint identity:** namespace RPi5 `microduck_rpi5_*`, PiZero `microduck_pizero2w_*`, ESP32 `microduck_esp32_*`; giữ V2 `cyberduck_v2_*`. Chuỗi * gồm ppo/latest/best. Cập nhật default export/status theo namespace liên quan. Không tự đổi tên/xóa checkpoint cũ; resume kiểm tra model identity/schema/terrain/shape/transform trước load. Transfer khác model chỉ đi qua đường explicit có reset optimizer/selection thích hợp.
5. **F5 — RMS:** validate đầy đủ actor/critic statistics, shape và finite trước gán state. Key vắng là legacy; key có nhưng malformed phải lỗi rõ, không silently reset hoặc partially load.
6. **F6 — finite training:** dùng population std hoặc bỏ chuẩn hóa advantage nếu một mẫu; kiểm tra finite loss/grad trước optimizer và tensor trước publish checkpoint. Không làm best/latest/anchor bị thay bởi checkpoint lỗi.
7. **F7 — tái lập:** đề xuất đưa tests cần thiết vào Git và bỏ riêng ignore tests/. Nếu tests chủ ý private, cần đổi contract public của verify và ghi hướng dẫn nhận bộ test thay vì công bố verify đầy đủ trên clean clone. Quyết định này cần xác nhận khi implementation, không tự đổi ignore trong audit.

## 7. Trình tự implementation — Task / Step / Exit gate

### Task 0: Chuẩn bị và khóa bằng chứng

- [ ] Step 1: kiểm tra HEAD/status, re-anchor nếu source đổi; xác minh tag + checkpoint stable, không di chuyển/ghi đè anchor.
- [ ] Step 2: dùng current WIP phù hợp hoặc tạo nhánh `wip/fix-project-audit`/worktree cô lập. Worktree mới không tự có tests/GEMINI/CLAUDE đang ignored: phải mang theo đúng bản local đã pin trước baseline. Giữ nguyên hai edit Pi trong working tree gốc, không reset/stash tùy tiện.
- [ ] Step 3: chạy baseline một lần, xác minh Python/MuJoCo/PyTorch/ONNX/g++; mọi artifact probe nằm trong TemporaryDirectory.
- [ ] Exit gate: baseline thực và snapshot ghi nhận, không có checkpoint triển khai bị ghi đè.

### Task 1: Sửa F1 — history/terminal observations

- [ ] RED: action A rồi B, fast/normal/nonterminal/timeout/reset-partial; assert history đúng giá trị, không chỉ shape/finite.
- [ ] GREEN: sửa ordering tại step, giữ previous action phục vụ reward; terminal actor/critic observations nhất quán.
- [ ] Exit gate: test runner qua và full suite kiểm các consumer BC/SAC/PPO/benchmark; reward action-rate vẫn dùng delta B-A với công thức và weight cũ.

### Task 2: Sửa F4 — cô lập checkpoint theo target

- [ ] RED: CLI routing 4 target, train tuần tự không overwrite, cross-target resume từ chối trước ghi.
- [ ] GREEN: namespace và identity validation, cập nhật default consumers/export/status. Khi không có --checkpoint, `export --target all` phải chọn checkpoint riêng cho từng target và báo rõ file còn thiếu; --checkpoint explicit giữ lựa chọn của người dùng kèm kiểm tra compatibility. Không auto migrate checkpoint cũ.
- [ ] Exit gate: resume cùng target giữ optimizer/iteration; transfer khác target là explicit và không giữ selection baseline không tương thích.

### Task 3: Sửa F5/F6 — chặn trạng thái huấn luyện hỏng

- [ ] RED: RMS None/NaN/negative variance/wrong shape, singleton 1x1; khẳng định lỗi không thay bytes checkpoint hợp lệ.
- [ ] GREEN: validate RMS trước mutate; advantage population std/skip singleton; finite guards trước update/save.
- [ ] Exit gate: cấu hình biên finite hoặc reject rõ; 1x2/2x1 và resume hợp lệ không regression.

### Task 4: Sửa F2 — schema train/export/RPi

- [ ] RED: observation/action parity qua reset và nhiều bước, zero-command và moving, control timestep khác mặc định, metadata thiếu/sai.
- [ ] GREEN: serialize schema, truyền qua exporter và consumer core; phase phải cùng thứ tự bước/reset với runner.
- [ ] Exit gate: obs clock khớp và ONNX action parity đạt tolerance; cả RPi5/PiZero mock loop chạy bằng artifact temp.

### Task 5: Sửa F3 — dimension safety ESP32

- [ ] RED: export 37 thất bại trước tạo file, 42 assembly không ghi qua canary; compile-time reject artifact incompatible.
- [ ] GREEN: validate export trước side effect và guard firmware; không thay UART/điều khiển servo ngoài phạm vi.
- [ ] Exit gate: compiled host C++ canary + generated policy/header checks qua; board smoke là bước riêng khi có phần cứng.

### Task 6: Sửa F7 và bàn giao

- [ ] Chốt chính sách tests, đưa bộ test mà verify yêu cầu vào Git hoặc sửa public verify contract và tài liệu.
- [ ] Chạy focused suite rồi full suite/verify đúng một lượt cuối; tăng coverage khi có failure/thay đổi mới.
- [ ] Kiểm tra git diff/status, giữ edit Pi; ghi handover 3–5 dòng ở README.md/his.md. Chỉ tạo anchor code+weights mới khi vận động thực tế được kiểm chứng.
- [ ] Exit gate: clone/export HEAD mới chạy được verification đã tài liệu hóa, không dựa file ignored tại máy tác giả.

## 8. Chiến lược test

Test files đã tồn tại, các scenario dưới đây là bổ sung đề xuất, chưa được viết:

| File | Scenario cần khóa |
| --- | --- |
| `tests/test_parallel_runner.py` | action A/B, fast/normal, terminal timeout, auto-reset, partial batch reset, reward action-rate không đổi |
| `tests/test_duck_cli.py` | routing 4 target; resume/export/status và export --target all dùng đúng namespace |
| `tests/test_ppo_training.py` | model identity mismatch; strict actor/critic RMS; singleton advantage; không save tensor NaN |
| `tests/test_deployment_artifacts.py` | train/RPi observation parity nhiều bước; ONNX action parity; ESP32 37 reject/42 C++ canary |
| `tests/test_policy_consumer_contract.py` | checkpoint → exporter → runtime schema; legacy/malformed metadata |

Trước sửa phải thấy RED do lỗi hành vi, sau sửa GREEN. Không làm yếu assertion hiện có. Dùng tham số không mặc định (hidden_dim nhỏ, control_dt khác) để tránh hardcode. Không chạy train dài chỉ để chứng minh lỗi số học.

Lệnh thực có trong repo/toolchain:

```powershell
python -m pytest tests/test_parallel_runner.py tests/test_ppo_training.py tests/test_duck_cli.py tests/test_deployment_artifacts.py tests/test_policy_consumer_contract.py -q
python -m pytest tests/ -q -p no:cacheprovider
python duck.py verify --skip-tests
python duck.py verify
```

Trong audit đã chạy full pytest và verify --skip-tests; lệnh focused và full verify ở trên là cổng dành cho implementation, không được ghi nhận là đã chạy trong audit.

## 9. Rủi ro và phạm vi ảnh hưởng

- [graph/source-derived] Direct dependent của step: step_arrays, train_ppo, _collect_rollouts, run_benchmark_scenario, run_benchmark. Đã xác minh các callsite bằng source; ảnh hưởng SAC đi qua step_arrays. Năm nhóm này đều cần regression/smoke sau F1.
- [inferred] F1/F2 thay input distribution của policy cũ; suite xanh chỉ chứng minh contract, không chứng minh robot đi tốt. Giữ code+weights anchor và benchmark deterministic trước train lại.
- [inferred] Đổi namespace/checkpoint schema không được âm thầm migrate hoặc reuse best baseline sai target. Legacy identity/clock cần quyết định rõ và test riêng.
- [verified] Test C++ chạy trên host, Pi inference probe chạy mock; timing/I2C/servo/board memory chưa được nghiệm thu phần cứng.
- [inferred] Tránh recompute observation/lidar không cần thiết trong step: sửa snapshot/ordering rõ ràng, không thêm full sensor pass trong vòng 50Hz mà không đo.

## 10. Files dự kiến thay đổi

| File | Trách nhiệm |
| --- | --- |
| `training/run_parallel_sim.py` | F1 history + terminal snapshots |
| `scripts/train_ppo.py` | F2 schema, F4 identity, F5 RMS, F6 finite |
| `duck.py` | F4 routing train/export/status |
| `scripts/export_policy.py` | F2 metadata, F3 reject incompatible dimension |
| `pipelines/rpi_infer_core.py` | F2 runtime clock/schema |
| `pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino` | F3 compile-time dimension guard |
| Năm test files tại §8 | Regression cụ thể từng finding |
| `.gitignore`, `README.md`, `his.md` | F7 theo quyết định tests; docs và handover |

Không đưa generated weights/header/ONNX/checkpoint hiện hành vào diff sửa; validation chỉ dùng artifacts temp. Publishing policy sau train là việc riêng.

## 11. Implementation context pack

```json
{
  "implementation_context": {
    "plan_path": "plans/261117-1710-project-audit-fixes/plan.md",
    "evidence_provenance_note": "Historical snapshot captured before plan relocation and handover link updates; generated_plan_path records the original snapshot exclusion. Re-anchor evidence before implementation; current plan path is plan_path.",
    "task_summary": "Sửa 6 lỗi correctness đã tái hiện và 1 khoảng trống tái lập test; chưa triển khai trong phiên audit.",
    "acceptance_criteria": [
      "F1-F6 có regression test RED trước/GREEN sau",
      "Checkpoint các target không ghi đè lẫn nhau",
      "Clean checkout chạy được verify cùng các test được công bố"
    ],
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "bd25e347310640fc6edcc8306d0db4503ed7486e",
      "generated_plan_path": "docs/plans/2026-11-17-gitnexus-plan-project-audit-fixes.md",
      "global_dirty_digest": {
        "algorithm": "sha256",
        "canonicalization": "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records",
        "value": "cbf1acdeb97fcceec82660476e627ca482c5b9eb0906c8997bf310b29cc2de93"
      },
      "cited_path_manifest": [
        {
          "path": ".gitignore",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:41367675fff4a0144aa254199dc9910550bd288d61948ddb2828deeb70dcebcd",
          "index_digest": "sha256:41367675fff4a0144aa254199dc9910550bd288d61948ddb2828deeb70dcebcd",
          "worktree_digest": "sha256:41367675fff4a0144aa254199dc9910550bd288d61948ddb2828deeb70dcebcd",
          "untracked_digest": "absent"
        },
        {
          "path": "CLAUDE.md",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:492d8d8bc92777049d02c0a96adf7bb1ce869fe366e89c9b5e9f6e09147e741c"
        },
        {
          "path": "GEMINI.md",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:7391e2c39f1313dc9ff95eadf5607f3df69fb59fbf1fbc24e8dc63e64ae3bed3"
        },
        {
          "path": "README.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "unstaged",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:048a45dcb52610470872b8f577173ab8781b7053556831852f6fbbc451537af6",
          "index_digest": "sha256:048a45dcb52610470872b8f577173ab8781b7053556831852f6fbbc451537af6",
          "worktree_digest": "sha256:7c9ce53096086190f56ebd643de398b7dfe90c61a0e8f549f3d4274ee6be45bb",
          "untracked_digest": "absent"
        },
        {
          "path": "duck.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:67287ec3c3bd23d98d1ffc8d20c95085a3a72e0af73086f4eb4b035ce88989e0",
          "index_digest": "sha256:67287ec3c3bd23d98d1ffc8d20c95085a3a72e0af73086f4eb4b035ce88989e0",
          "worktree_digest": "sha256:67287ec3c3bd23d98d1ffc8d20c95085a3a72e0af73086f4eb4b035ce88989e0",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:d4b9cd5176c534bb8051b0f7b0a6a6e0d4981761165c949658da75d12cb05bd8",
          "index_digest": "sha256:d4b9cd5176c534bb8051b0f7b0a6a6e0d4981761165c949658da75d12cb05bd8",
          "worktree_digest": "sha256:d4b9cd5176c534bb8051b0f7b0a6a6e0d4981761165c949658da75d12cb05bd8",
          "untracked_digest": "absent"
        },
        {
          "path": "pipelines/rpi_infer_core.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:6d984bb22a42e4dee4d82bac4abb3fcbe1dc08ef7a959c44669d325c0f9f658f",
          "index_digest": "sha256:6d984bb22a42e4dee4d82bac4abb3fcbe1dc08ef7a959c44669d325c0f9f658f",
          "worktree_digest": "sha256:6d984bb22a42e4dee4d82bac4abb3fcbe1dc08ef7a959c44669d325c0f9f658f",
          "untracked_digest": "absent"
        },
        {
          "path": "scripts/export_policy.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:91e75a33c1bfd38ecec03ce7e210bd7567305a288cb06ba1912fc89e59b75727",
          "index_digest": "sha256:91e75a33c1bfd38ecec03ce7e210bd7567305a288cb06ba1912fc89e59b75727",
          "worktree_digest": "sha256:41ea85418a1221e10442872072a7d60abe92da120ec781e9b4c17d92de9803cf",
          "untracked_digest": "absent"
        },
        {
          "path": "scripts/train_ppo.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:4a536b4a3d2bebfe96d0fec807f26c3901dff2d91395b17ba6d190713cd7fd24",
          "index_digest": "sha256:4a536b4a3d2bebfe96d0fec807f26c3901dff2d91395b17ba6d190713cd7fd24",
          "worktree_digest": "sha256:32a69687a5764142e994c328dbc11048a39df5465197ab94ccb7f0eb98bb8e99",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/test_deployment_artifacts.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:1a95bf083617a15e417868e70e703d7857c60efa41e24b2c2034a5bb25bed181"
        },
        {
          "path": "tests/test_duck_cli.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:3fd8a9a4285ba56541e9ad9b36f49279d1f662cd1012a0d7241fc91fecf929b1"
        },
        {
          "path": "tests/test_parallel_runner.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:eb30c605ace9ce67b758e82ec4d78311f615763e838c2c5aa8ba5f419b08ed39"
        },
        {
          "path": "tests/test_policy_consumer_contract.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:7723d320e7790f1804cde945d43a4c734cd1f8f7bf096562f5d3f11248fa58a0"
        },
        {
          "path": "tests/test_ppo_training.py",
          "object_kind": {
            "head": "absent",
            "index": "absent",
            "worktree": "absent",
            "untracked": "regular"
          },
          "state": "untracked",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "absent",
          "index_digest": "absent",
          "worktree_digest": "absent",
          "untracked_digest": "sha256:268985532d765454fea584654ceeaba98c6a34c082c11d39537b7019c2a47eed"
        },
        {
          "path": "training/env_config.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:bddc930245b45dbec02c86219519cc62985a943ddbba19a5568304ae8572cc5d",
          "index_digest": "sha256:bddc930245b45dbec02c86219519cc62985a943ddbba19a5568304ae8572cc5d",
          "worktree_digest": "sha256:bddc930245b45dbec02c86219519cc62985a943ddbba19a5568304ae8572cc5d",
          "untracked_digest": "absent"
        },
        {
          "path": "training/run_parallel_sim.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:9d102d595f1c399b9ef5dedee48f53b167064e9f67aa005f436ec901b770666a",
          "index_digest": "sha256:9d102d595f1c399b9ef5dedee48f53b167064e9f67aa005f436ec901b770666a",
          "worktree_digest": "sha256:70282f1813cf758053d797072719fd496cda4491921ae53ad1529c1b4b9923b3",
          "untracked_digest": "absent"
        }
      ]
    },
    "primary_symbols": [
      {
        "symbol": "ParallelSimRunner.step",
        "file": "training/run_parallel_sim.py",
        "lines": "973-1374",
        "role": "observation/history/terminal state"
      },
      {
        "symbol": "pack_rpi_observation",
        "file": "pipelines/rpi_infer_core.py",
        "lines": "65-85",
        "role": "runtime observation packing"
      },
      {
        "symbol": "export_esp32_headers",
        "file": "scripts/export_policy.py",
        "lines": "278-319",
        "role": "dimension contract firmware/export"
      },
      {
        "symbol": "cmd_train",
        "file": "duck.py",
        "lines": "185-251",
        "role": "hardware target routing"
      },
      {
        "symbol": "train_ppo",
        "file": "scripts/train_ppo.py",
        "lines": "490-1258",
        "role": "checkpoint/load/update contracts; re-anchor endpoint before editing"
      }
    ],
    "related_symbols": [
      {
        "symbol": "ParallelSimRunner.step_arrays",
        "relationship": "CALLS step",
        "relevance": "PPO/SAC fast path"
      },
      {
        "symbol": "run_benchmark_scenario",
        "relationship": "CALLS step",
        "relevance": "benchmark behavior after history fix"
      },
      {
        "symbol": "_collect_rollouts",
        "relationship": "CALLS step",
        "relevance": "BC demonstrations"
      },
      {
        "symbol": "run_benchmark",
        "relationship": "CALLS step",
        "relevance": "runner performance smoke"
      },
      {
        "symbol": "RunningMeanStd.load_state_dict",
        "relationship": "CALLS from train_ppo",
        "relevance": "strict atomic RMS load"
      }
    ],
    "execution_path": [
      "duck CLI resolves target",
      "PPO collects runner observations/actions",
      "checkpoint stores model/RMS/observation schema",
      "ONNX/C export retains contracts",
      "runtime packs matching raw observations"
    ],
    "pdg_constraints": [
      {
        "description": "Source-derived: reward sử dụng previous actions; next_obs phải sử dụng current action.",
        "affected_statements": [
          "training/run_parallel_sim.py:1049",
          "training/run_parallel_sim.py:1170",
          "training/run_parallel_sim.py:1296"
        ],
        "implementation_consequence": "Không chỉ chuyển assignment lên trước reward; giữ snapshot history cũ."
      }
    ],
    "architectural_patterns": [
      {
        "pattern": "Shared runtime + thin hardware wrappers",
        "example_location": "pipelines/rpi_infer_core.py:338",
        "usage_guidance": "Sửa packing/clock tại core cho cả hai Pi."
      },
      {
        "pattern": "Task / Step / test-first / exit gate",
        "example_location": "plans/261116-2350-ppo-locomotion-overhaul/plan.md",
        "usage_guidance": "Dùng checklist ở mục 7; mọi checkpoint validation dùng thư mục tạm."
      }
    ],
    "files_to_modify": [
      {
        "file": "training/run_parallel_sim.py",
        "symbols": [
          "ParallelSimRunner.step"
        ],
        "intended_change": "F1: giữ action cũ cho reward, next_obs/terminal_obs chứa action vừa thực thi."
      },
      {
        "file": "pipelines/rpi_infer_core.py",
        "symbols": [
          "pack_rpi_observation",
          "BaseRPiInferenceEngine"
        ],
        "intended_change": "F2: tái tạo clock theo schema/metadata của policy và đồng bộ reset/timestep."
      },
      {
        "file": "scripts/export_policy.py",
        "symbols": [
          "export_onnx",
          "load_actor_from_checkpoint",
          "export_esp32_headers"
        ],
        "intended_change": "F2: truyền schema quan sát; F3: từ chối obs_dim=37 cho firmware hiện tại trước khi ghi artifact."
      },
      {
        "file": "duck.py",
        "symbols": [
          "cmd_train",
          "cmd_export",
          "cmd_status"
        ],
        "intended_change": "F4: namespace checkpoint riêng cho cả bốn target."
      },
      {
        "file": "scripts/train_ppo.py",
        "symbols": [
          "train_ppo",
          "RunningMeanStd.load_state_dict"
        ],
        "intended_change": "F2/F4: lưu và validate identity/schema; F5: strict RMS; F6: singleton advantage hữu hạn và finite guards."
      },
      {
        "file": ".gitignore",
        "symbols": [],
        "intended_change": "F7: đưa các test được duck verify yêu cầu vào Git; không tự động đổi các ignore khác."
      },
      {
        "file": "pipelines/dual_esp32s3_stack/deployment/mcu2_locomotion_s3/mcu2_main.ino",
        "symbols": [
          "assemble_observation"
        ],
        "intended_change": "F3: compile-time dimension guard so incompatible generated headers are rejected."
      }
    ],
    "tests": [
      {
        "file": "tests/test_parallel_runner.py",
        "scenarios": [
          "A rồi B: returned obs[25:33] bằng A rồi B ở fast và normal",
          "timeout terminal chứa action cuối; reset obs chứa zero",
          "action_rate vẫn dùng delta B-A với công thức/weight cũ; reset một env không ảnh hưởng env khác"
        ]
      },
      {
        "file": "tests/test_deployment_artifacts.py",
        "scenarios": [
          "RPi observation clock khớp runner qua nhiều control steps và reset",
          "ONNX action parity trên cùng sensor/clock input",
          "ESP32 export 37 bị reject trước ghi; export 42 chạy C++ assembly có canary nguyên vẹn"
        ]
      },
      {
        "file": "tests/test_duck_cli.py",
        "scenarios": [
          "bốn target có save/best/latest riêng",
          "resume đúng namespace; không chọn checkpoint target khác",
          "export --target all without explicit checkpoint routes each target separately; missing target checkpoint is a clear error"
        ]
      },
      {
        "file": "tests/test_ppo_training.py",
        "scenarios": [
          "resume cross-target bị reject trước thay file; transfer vẫn explicit",
          "RMS None/NaN/negative variance/wrong shape bị reject trước mutate",
          "1 env x 1 step hữu hạn hoặc reject rõ trước ghi; 1x2 và 2x1 vẫn hữu hạn"
        ]
      },
      {
        "file": "tests/test_policy_consumer_contract.py",
        "scenarios": [
          "schema observation serialize/load/export round-trip",
          "legacy metadata thiếu được xử lý rõ, không suy diễn clock chỉ từ obs_dim"
        ]
      }
    ],
    "verification_commands": [
      "python -m pytest tests/test_parallel_runner.py tests/test_ppo_training.py tests/test_duck_cli.py tests/test_deployment_artifacts.py tests/test_policy_consumer_contract.py -q",
      "python -m pytest tests/ -q -p no:cacheprovider",
      "python duck.py verify --skip-tests",
      "python duck.py verify"
    ],
    "risks": [
      "Observation-history fix đổi phân phối đầu vào policy cũ",
      "Legacy checkpoint chưa có clock schema",
      "Đổi namespace không được tự động di chuyển/ghi đè checkpoint cũ",
      "Chưa chạy firmware trên board thật"
    ],
    "assumptions": [
      "Checkpoint và firmware hiện hành cần 42 chiều khi deploy ESP32; xác minh lại trước implementation",
      "Các test ignored hiện tại là test local; xác minh chính sách công bố trước sửa .gitignore"
    ],
    "open_questions": [
      "Policy legacy nào đã train với/không với clock? cần provenance hoặc cấu hình explicit",
      "Có chủ ý giữ tests private không? nếu có cần thay contract public của duck verify"
    ],
    "avoid": [
      "Không sửa source khi chỉ review/plan",
      "Không reset/stash/ghi đè hai dirty file inference Pi",
      "Không di chuyển tag hoặc thay checkpoint v1-flat-walk-stable",
      "Không ghi đè artifacts đang ship khi test",
      "Không coi suite xanh là bằng chứng locomotion/deployment đã đạt",
      "Không suy luận clock từ kích thước 42 vì hai schema cùng chiều"
    ]
  }
}
```

## 12. Giả định, câu hỏi mở và việc hoãn

- [assumed] Hướng tối thiểu của F3 là chỉ export 42 cho firmware hiện tại; cần xác minh có thiết bị legacy 37 cần giữ không. Nếu có, task phải triển khai assembly theo schema thay vì chỉ reject.
- [assumed] Chưa biết policy legacy nào train có clock; phải kiểm metadata/provenance hoặc yêu cầu cấu hình explicit trước deploy. Không suy luận từ 42 chiều hoặc tên file.
- [verified] Tests và plans hiện ignored; không tự force-add/unignore trong phiên review. F7 cần chốt chính sách công bố trước implementation.
- [verified] Index đúng HEAD nhưng hai constants dirty có sẵn; source là bằng chứng cuối cùng. Không refresh index chỉ để làm đẹp báo cáo vì findings không phụ thuộc graph hai constants này.
- [verified] Người dùng đã cho phép ghi Markdown thông thường và yêu cầu lưu trong `plans/` của dự án. Snapshot nguồn do helper tạo được giữ nguyên như bằng chứng lịch sử; đường dẫn plan hiện tại theo quy ước repo, không theo path validator của helper GitNexus. Khi implementation phải re-anchor nguồn và không xem snapshot trước di chuyển là snapshot working tree hiện tại.
- Hoãn: tuning reward mới, train GPU dài, thiết kế CAD/HTML viewer, tối ưu latency phần cứng, publish artifacts và merge. Các warning ONNX deprecation có thể xử lý riêng sau lỗi correctness.

## 13. Definition of Done

- [ ] F1–F6 có regression tests fail-before/pass-after; F7 đã có public verification contract tái lập.
- [ ] Clock/observation-history parity cho RPi, checkpoint RMS/identity/finite và dimension safety cho ESP32 đạt các scenario đã định; không ngụ ý toàn bộ firmware đã đạt sim-to-real.
- [ ] Không checkpoint NaN hoặc cross-target overwrite; path/file bytes của anchor được bảo toàn.
- [ ] Focused suite, full suite và verification documented xanh, số pass/skips/warnings thực được ghi lại.
- [ ] Cập nhật handover README/his, status sạch ngoài những thay đổi đã phân loại; không tuyên bố đã đạt locomotion thật chỉ nhờ tests.
