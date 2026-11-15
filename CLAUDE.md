# CLAUDE.md — DuckV1 Cross-Model & Cross-Session Rules

Read and strictly follow `GEMINI.md` in the repository root. Every session and every model (Claude, Gemini, Subagents) MUST enforce the **3-Step Session Protocol**:

1. **Anchor Tagging (`git tag` + `.pt` snapshot)**: Tag working milestones (currently `v1-flat-walk-stable` on `main` + `models/checkpoints/v1-flat-walk-stable.pt`) so we always have a known-good code + weights rollback point.
2. **WIP Branch Isolation (`wip/...`)**: Keep in-progress or experimental changes (currently 7 files for BC, SAC, LiDAR floor filter, and 2-layer nav on `wip/nav-sac`) on dedicated `wip/*` branches so `main` stays clean.
3. **3–5 Line Handover Log (`README.md` & `his.md`)**: Always update the 3–5 line status summary in `README.md` and `his.md` answering:
   - **Đang dừng ở đâu** (Where we stopped / current anchor)
   - **Đã thử những gì** (What was tried & where it is stored)
   - **Giả thuyết tiếp theo là gì** (Next hypothesis to test)
