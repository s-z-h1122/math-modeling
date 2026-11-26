# 2026 APMCM Problem B — 问题一（v2.1 WLR-WE-FBS）独立空分支

本分支从空提交历史新建（`git checkout --orphan model/q1-wlr-we-fbs-v21`），归档问题一截至 **Phase 2** 的完整建模文档、可移植代码与验收指标表。

## 目录结构
- `src/q1/`：问题一 v2.1 核心重构引擎（`q1_core_v21.py`）、Phase 1 端到端盲测与 5 方法消融对比（`q1_benchmark_v21.py`）、Phase 2 全 10 炉批量重构与自动化验收（`q1_reconstruct_all_v21.py`）、光谱解剖与数据画像核验脚本（已按团队规范替换为相对路径与环境变量，无本机绝对路径）。
- `docs/`：从赛题初步分析、原始数据画像核验、无上下文独立同行评审、开源社区调研到 v2.1 定稿计划与 Phase 1–2 验收总结的完整演进文档（`01_`–`08_`）。
- `results/`：Phase 1 5 方法盲测表、Phase 2 全 10 炉自动化验收总表（10/10 PASS）、光谱解剖明细表与 4 联诊断图。

> 注：原始 `.xlsb`（~120 MB）及重构 `.npz` 缓存（~234 MB）遵循仓库 `<50 MB` 规范未入库，可通过环境变量 `RAW_XLSB_DIR` / `Q1_CACHE_DIR` 指定本地路径一键复现。
