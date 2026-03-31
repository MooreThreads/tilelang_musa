# Tilelang_MUSA 升级到 v0.1.8 迁移计划

## Goal Description
基于官方 `tilelang` 的 `heads/v0.1.8`（当前工作副本提交 `0765d1d6`），将 `tilelang_musa_6` 中 `47039f0..HEAD`（当前 `cd3e4084`，共 226 个提交）的 MUSA 相关能力按“功能模块”迁移，而不是按提交逐个搬运。迁移顺序为“基础功能 -> async_copy/TMA/SQMMA/barrier -> 扩展特性”，并在远端 MUSA 环境中逐步回归测试，最终达到 `musa_tests` 通过率 >= 90%。

量化指标约定：本计划将“`musa_tests` 通过率 >= 90%”视为硬性验收门槛。

## Acceptance Criteria

以下验收均遵循 TDD：每个 AC 同时定义正向（应通过）和反向（应失败）测试。

- AC-1: 完成迁移基线与功能分桶，形成可执行迁移清单
  - Positive Tests (expected to PASS):
    - 可以从 `tilelang_musa_6` 的 `47039f0..HEAD` 导出变更清单，并按功能分桶（至少包含：基础、async_copy、TMA、SQMMA、barrier、测试与工具链）。
    - 每个分桶都映射到目标目录（例如 `src/target`、`src/transform`、`src/tl_templates`、`tilelang/*`、`musa_tests/*`）及候选迁移顺序。
  - Negative Tests (expected to FAIL):
    - 若仅给出“按 commit 迁移”的清单而无功能分桶，则判定 AC-1 不通过。
    - 若无法说明某个分桶对应到哪些目标路径，则判定 AC-1 不通过。

- AC-2: 基础 MUSA 后端可编译、可运行、可做最小正确性验证
  - Positive Tests (expected to PASS):
    - 在 v0.1.8 分支上完成最小 MUSA 代码生成与运行链路（Tilelang -> TIR -> MUSA C -> mtcc）。
    - 远端执行基础用例 `musa_tests/basic/add.py` 通过。
    - 再至少通过 2 个基础回归用例（例如 `musa_tests/basic/mm_fma.py`、`musa_tests/basic/test_copy_to_local.py`）。
  - Negative Tests (expected to FAIL):
    - 若移除/禁用 MUSA target 注册后仍“通过”基础用例，则说明测试无效，AC-2 不通过。
    - 若生成源码无法编译为可执行内核（mtcc 失败），AC-2 不通过。

- AC-3: TMA 功能完成迁移并支持开关行为一致
  - Positive Tests (expected to PASS):
    - `musa_tests/basic/test_tma_1d.py` 与 `musa_tests/basic/test_tma_fma.py` 通过。
    - 至少 1 个依赖 TMA 路径的功能用例通过（如 `musa_tests/basic/addk_tme.py`）。
    - 在启用 TMA 时，生成代码可观测到 TMA 相关路径被采用（通过日志、IR 或源码片段验证）。
  - Negative Tests (expected to FAIL):
    - 显式设置 `TL_DISABLE_TMA_LOWER=True` 后，依赖 TMA 的“必须启用 TMA 才能通过”的断言应失败或走非 TMA 退化路径。
    - 若 TMA 开关前后行为无差异且缺乏合理解释，AC-3 不通过。

- AC-4: SQMMA/ldsm 特性迁移完成并通过代表性回归
  - Positive Tests (expected to PASS):
    - 基础 SQMMA 用例通过（如 `musa_tests/basic/sqmma_trans_b.py`）。
    - `musa_tests/ldsm_sqmma/` 下至少 2 个代表性测试通过（含 splitK/splitM 场景至少各 1 个）。
    - 对涉及 `TL_DISABLE_SQMMA` 的路径，开关行为与预期一致。
  - Negative Tests (expected to FAIL):
    - 人为构造不满足 SQMMA 形状约束的输入时，测试应报错或显式拒绝，而非静默给出错误结果。
    - 若关闭 SQMMA 后仍声称命中 SQMMA 专用路径，AC-4 不通过。

- AC-5: 回归稳定性与问题修复能力达到可维护状态
  - Positive Tests (expected to PASS):
    - `musa_tests/python/issue/` 下至少 5 个历史问题回归测试通过（覆盖编译链路、算子正确性、布局/向量化中的至少 3 类问题）。
    - 新增或迁移的关键 pass/codegen 变更有对应测试用例映射关系。
  - Negative Tests (expected to FAIL):
    - 若关键修复仅靠手工验证、没有可重复测试脚本，则 AC-5 不通过。
    - 若出现“测试通过但功能不可复现”的不稳定现象，AC-5 不通过。

- AC-6: 最终 `musa_tests` 通过率达到目标
  - Positive Tests (expected to PASS):
    - 在远端环境执行 `musa_tests`（允许按可执行能力分批），最终统计通过率 >= 90%。
    - 输出测试汇总（总数、通过、失败、跳过、通过率）并可复查。
  - Negative Tests (expected to FAIL):
    - 通过率 < 90% 则 AC-6 不通过。
    - 若无统一统计口径（不同批次不可合并对比），AC-6 不通过。

## Path Boundaries

### Upper Bound (Maximum Scope)
在 v0.1.8 上完整落地 MUSA 后端核心能力（基础/TMA/SQMMA/关键 issue 回归），并建立可重复的远端测试流程；对主要变更点提供“功能分桶 -> 代码路径 -> 测试用例”三向映射，最终 `musa_tests` 通过率稳定 >= 90%。

### Lower Bound (Minimum Scope)
在 v0.1.8 上完成最小可用 MUSA 迁移：基础链路可运行，TMA 与 SQMMA 各有至少一组代表功能可用，且有可复现测试与统计，最终仍需满足通过率 >= 90% 的硬门槛。

### Allowed Choices
- Can use: `git diff/log/range-diff/cherry-pick -n` 等手段抽取补丁；允许在 v0.1.8 上做必要适配重写；允许将无法在当前硬件执行的用例标记为跳过（需有明确原因）。
- Cannot use: 按 commit 机械搬运替代功能分桶；通过放宽正确性阈值“刷通过率”；删除关键测试来规避失败；引入与 MUSA 目标无关的大范围重构。

## Feasibility Hints and Suggestions

### Conceptual Approach
1. 先做“变更盘点与分桶”：从 `47039f0..HEAD` 提取变更，按基础/TMA/SQMMA/测试框架归类。
2. 在 v0.1.8 建立最小 MUSA 可运行闭环：优先打通 target、runtime module、模板与基础 pass。
3. 逐桶迁移并小步验证：每合入一桶即运行对应最小测试集，不等待“大一统”后再验证。
4. 先基础后高级：基础算子和通路稳定后再启用 TMA，再推进 SQMMA/ldsm 复杂路径。
5. 最后统一回归和补漏：聚焦失败分布做针对性修复，冲刺 90% 通过率门槛。

### Relevant References
- `tilelang/`（目标仓库，v0.1.8 基线）
- `tilelang_musa_6/`（MUSA 现有实现来源，主分支）
- `tilelang/src/target/` 与 `tilelang_musa_6/src/target/` - codegen/runtime target 对照迁移
- `tilelang/src/transform/` 与 `tilelang_musa_6/src/transform/` - pass 差异与功能回填
- `tilelang/src/tl_templates/` 与 `tilelang_musa_6/src/tl_templates/musa/` - 模板与内建函数支持
- `tilelang_musa_6/musa_tests/` - 分阶段回归测试来源

## Dependencies and Sequence

### Milestones
1. Milestone 1: 迁移准备与分桶基线
   - Phase A: 固定源/目标基线（`tilelang_musa_6@47039f0..cd3e4084` -> `tilelang@v0.1.8`）。
   - Phase B: 生成功能分桶清单、映射目标路径、定义每桶最小测试入口。
2. Milestone 2: 基础功能迁移
   - Phase A: 迁移基础 codegen/runtime/模板与必要语言层改动。
   - Phase B: 通过 `musa_tests/basic/add.py` + 2 个基础用例，形成第一版可运行链路。
3. Milestone 3: TMA 能力迁移
   - Phase A: 迁移 TMA 相关 pass 与模板支持。
   - Phase B: 通过 TMA 代表用例并验证开关行为一致。
4. Milestone 4: SQMMA/ldsm 能力迁移
   - Phase A: 迁移 SQMMA 指令形状、布局推导及相关 codegen 逻辑。
   - Phase B: 通过 `ldsm_sqmma` 代表回归并修复高频失败模式。
5. Milestone 5: 全量回归与达标收敛
   - Phase A: 执行分批 `musa_tests` 回归并生成统一统计。
   - Phase B: 针对失败集迭代修复，达到并稳定在 >= 90% 通过率。

依赖关系：Milestone 2 是 Milestone 3/4 的前置；Milestone 3 和 4 可局部交叉迭代，但最终都必须在 Milestone 5 前收敛。

## Implementation Notes

### Code Style Requirements
- 业务实现代码与注释不应包含 `AC-`、`Milestone`、`Phase` 等计划术语。
- 计划术语仅用于文档与跟踪，不应污染生产代码命名。
- 测试脚本建议统一输出结构化结果（便于汇总通过率），但不改变原始语义断言。

### Assumptions and Clarifications
- Draft 中提到的目标目录 `tilelang_musa`，在当前工作区对应为 `tilelang/`（v0.1.8 分支工作副本）。
- Tilelang/MUSA 程序与核心测试在远端环境执行，本地仅做代码组织、静态检查与轻量脚本处理。
