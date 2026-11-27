# 2026 APMCM Problem B — 问题一（v2.1 WLR-WE-FBS）独立空分支

本分支从空提交历史新建（`git checkout --orphan model/q1-wlr-we-fbs-v21`），归档问题一截至 **Phase 2** 的完整建模文档、可移植代码与验收指标表。

## 目录结构
- `src/q1/`：问题一 v2.1 核心重构引擎（`q1_core_v21.py`）、Phase 1 端到端盲测与 5 方法消融对比（`q1_benchmark_v21.py`）、Phase 2 全 10 炉批量重构与自动化验收（`q1_reconstruct_all_v21.py`）、光谱解剖与数据画像核验脚本（已按团队规范替换为相对路径与环境变量，无本机绝对路径）。
- `docs/`：从赛题初步分析、原始数据画像核验、无上下文独立同行评审、开源社区调研到 v2.1 定稿计划与 Phase 1–2 验收总结的完整演进文档（`01_`–`08_`）。
- `results/`：Phase 1 5 方法盲测表、Phase 2 全 10 炉自动化验收总表（10/10 PASS）、光谱解剖明细表与 4 联诊断图。

> 注：原始 `.xlsb`（~120 MB）及重构 `.npz` 缓存（~234 MB）遵循仓库 `<50 MB` 规范未入库，可通过环境变量 `RAW_XLSB_DIR` / `Q1_CACHE_DIR` 指定本地路径一键复现。

## v2.2 与问题二、问题三完整流水线

本分支新增了可独立复现的完整实现：

- `src/common/preprocess_core.py`：主吹/补吹分段、附件 5/6 耗氧守恒物理时间、NIST 三线波长校准；
- `src/q1/q1_core_v22.py` 与 `q1_reconstruct_all_v22.py`：严格黑帧、分段 PCHIP、Q1 平滑展示与 Q3 脉冲保真双输出；
- `src/q2/q2_reconstruct_delay.py`：单调累计耗氧、TV 流量与原始/物理约束时滞；
- `src/q3/q3_loho_compare.py`：相同因果特征与折内归一化下的 Ridge、SVR、ELM、正则 ELM、DNN、SG-MoE 十折留一炉公平对比；
- `src/validate_full_pipeline.py`：对十炉、物理守恒、时滞单调性、120 组 LOHO 指标做独立验收。

安装 `requirements.txt` 后，可运行：

```powershell
.\scripts\run_full_pipeline.ps1 -CacheDir <10个npz缓存目录> -Python <python.exe>
```

大型 `.npz` 中间产物写入 `run_outputs/` 且不入库；可审计的 CSV、图和实验结论保存在 `results/` 与 `docs/`。
