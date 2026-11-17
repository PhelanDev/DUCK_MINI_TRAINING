# Kết quả thực hiện — 2026-11-17

Hoàn tất Task 0–6 trên nhánh `wip/fix-project-audit`, xuất phát từ
`bd25e347310640fc6edcc8306d0db4503ed7486e`. Không gọi subagent; chỉ đọc source,
tests và consumer cần cho plan. Giữ nguyên nội dung plan gốc.

## Re-anchor và quyết định

- Plan SHA-256: `d350baa12a146c0d66684ae634e2e380b9c80b4d51523964407b73abb42ce1e6`.
- HEAD trùng plan; SHA-256 của toàn bộ source/tests dẫn chiếu khớp snapshot.
  README đổi phần bàn giao, hai wrapper Pi có edit đường dẫn từ trước.
- Snapshot schema 2 ở path cũ là bằng chứng lịch sử. Plan đã cho phép đường dẫn
  `plans/`; helper safe-read không hỗ trợ Windows/path này. Re-anchor bằng Git
  diff, HEAD và hash từng file dẫn chiếu, không giả lập receipt schema 2.
- Người dùng xác nhận đưa tests vào Git. Bổ sung 47 file `test_*.py`, conftest và
  helper hiện có; thêm regression. Dashboard được đưa vào Git nguyên hành vi
  vì một test hiện có yêu cầu file đang ignored này; không sửa HTML/JS.
- Thêm `training/policy_contract.py` làm seam chung cho namespace, identity và
  observation schema giữa producer/exporter/runtime, tránh lặp contract.
- Chỉ thêm tham số legacy schema vào hai wrapper Pi; hai dòng đường dẫn model
  có sẵn giữ nguyên working tree và không đưa vào commit.
- Gom implementation thành một commit đã kiểm chứng vì các bước cùng sửa
  `train_ppo.py`/export schema và bộ tests chỉ được đưa vào Git tại F7. README
  cùng `his.md` local được cập nhật; giữ nguyên ignore `his.md` theo phạm vi plan.

## Kết quả theo finding

| Finding | Thay đổi và kiểm chứng |
| --- | --- |
| F1 | Next/terminal observation chứa action vừa chạy, reward giữ previous action; fast/normal, timeout, critic terminal, partial reset đều qua. Không thêm sensor pass. |
| F4 | Namespace riêng 4 target; resume kiểm identity/schema/terrain/transform/model tensor shapes trước load; overwrite khác target bị chặn, transfer explicit giữ output mới/reset optimizer và selection. Export all preflight từng checkpoint riêng và identity trước ghi. |
| F5 | RMS actor/critic validate finite, shape, variance, scalar metadata trước gán; malformed field không còn bị nuốt; regression xác nhận bytes output không đổi khi reject. Key RMS vắng vẫn là legacy. |
| F6 | Population std cho singleton; loss/gradient finite trước optimizer, model finite sau update; recursive tensor/array guard trước atomic save. Các batch 1×1, 1×2, 2×1 hữu hạn. |
| F2 | Checkpoint → exporter → ONNX giữ schema clock/period/control_dt/reset; Pi tăng clock sau action, cả zero command. Test sensor observation toàn vector + ONNX action parity qua reset/nhiều bước ở control_dt=0.006, hidden_dim=16; hai wrapper mock loop qua. Legacy thiếu schema cần cấu hình explicit, không đoán từ obs_dim. |
| F3 | ESP32 chỉ export 42/8, reject 37 trước tạo directory/header; firmware static_assert OBS_DIM/layer input/output. Host C++ trích nguyên assemble_observation: canary nguyên vẹn ở 42; compile reject 37 hoặc layer lệch. ONNX API riêng vẫn hỗ trợ legacy 37. |
| F7 | Bỏ riêng ignore tests/, stage tests và các file cần thiết; clean Git export chạy verify đầy đủ. README có dependency, namespace, legacy schema và lệnh verification. |

RED thực chạy trước sửa: F1 history `2 failed`; F4 namespace/cross-target
`4 failed`; F5/F6 RMS/singleton/checkpoint `7 failed`; F2/F3 missing clock/schema
và export 37 `4 failed`. Sau sửa mọi regression đều GREEN. Test cũ được đổi sang
artifact tạm có identity/schema và transfer explicit, không làm yếu assertions.

## Verification

| Lệnh / môi trường | Kết quả |
| --- | --- |
| Baseline `python -m pytest tests/ -q -p no:cacheprovider` | 479 passed, 14 warnings, 141.97s |
| Focused suite 5 file tại §8 | 146 passed, 16 warnings, 155.18s |
| Full suite cuối trên working tree | 521 passed, 26 warnings, 240.25s |
| `python duck.py verify --skip-tests` | Exit 0, joint/mass/inertia/CoM và 100 bước simulation thành công |
| `python duck.py verify` trên bản xuất sạch Git index | Exit 0; physics thành công; 521 passed, 26 warnings, 230.95s |
| C++ assembly canary và guard | 3 passed, g++ C++17; generated header checks trong full suite đều qua |

Bản xuất sạch được tạo bằng `git write-tree` + `git archive`, không copy file
ignored từ working tree, tree `29a2b9494fc89048f994e012c93496c4553829c4`.
Sau kiểm chứng chỉ cập nhật Markdown bàn giao. Shell wrapper in log từng bị
`UnicodeEncodeError` khi xuất tiếng Việt; tiến trình verify thật trả exit 0,
log UTF-8 xác nhận kết quả ở trên.

Môi trường: Python 3.12, NumPy 2.5.4, PyTorch 2.15.0+cpu, MuJoCo 4.0.0,
ONNX 1.23.2; g++ từ MSYS2 UCRT64. Warnings là ONNX exporter deprecation.

Graph kiểm tra các consumer PPO/SAC/BC/benchmark/viewer/exporter; không suy luận
attribute flow từ PDG rỗng. Index cuối trước commit có runner identity schema 4,
CLI 1.6.12, runnerIdentityStatus=current, incompleteReasons=[], status=up-to-date.
Gate staged trước commit quan sát đủ 618 symbols/57 affected processes, không
partial/truncated; đa số symbols mới là tests vừa được công bố, source nằm trong
phạm vi các task nêu trên. Static callers vẫn chỉ là lower bound.

## Bảo toàn và giới hạn

Tag `v1-flat-walk-stable` vẫn trỏ `dc948dbd0f4ab778f29dd360aeaf51770500b756`.
Checkpoint neo SHA-256:
`4c066fa10610a06d0a4d288ecb64d7675cb4fa4720610bde992fbaf5921bb1b6`.
Không sửa weights/header/ONNX/checkpoint đang ship; toàn bộ test artifact nằm tạm.

DoD correctness và clean checkout đã đạt; không train dài, không nghiệm thu
locomotion/latency trên robot, không deploy/publish/merge hoặc tạo neo mới.
Checkpoint legacy thiếu provenance không được resume trực tiếp. Phần cứng Pi và
ESP32 cần smoke riêng trước triển khai thực tế.
