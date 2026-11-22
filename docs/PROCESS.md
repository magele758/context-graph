# Context Graph · 生产过程回放

> [!IMPORTANT]
> 这是**事后回放**：本仓库初版由 Antigravity 主会话派出的子 agent 在其自身会话中完成，**没有**经过 OPC OS 工作流（无计划审批、无 Herdr 隔离工作树、无独立评审）。本文件由该子 agent 的原始会话日志自动生成，用于让过程可追溯；评测数据由实现者自建，尚未独立验证。

| 项 | 值 |
|---|---|
| 执行者 | Antigravity 子 agent（模型 `gemini-4-carbon-medium`，会话 `4bd145f3-5811-4bc8-a529-a33016466ab5`） |
| 起止 | 2026-11-22T12:39:17Z → 2026-11-22T13:07:25Z（28 分钟） |
| 工具调用 | 50 次：写/改文件 23，命令 16（失败 4），测试运行 1 |
| Token | 输入 230,846，输出 113,350，缓存命中 3,690,951 |
| 产出文件 | 18 个 |

## 原始任务书（主会话下发，未删改）

<details><summary>展开</summary>

> 你负责完整落地第 1 个独立课题项目与服务：**Context Graph**，目录为 `/Users/penglei/developer/ai-github/context-graph`（目录已创建，当前为空）。
> 
> ## 背景与研究目标
> 当前主流 Coding Agent / General Agent 的 Context 仍然是线性 `messages[]` 追加序列，存在三大核心痛点：
> 1. **Context Rot（注意力稀释与噪声污染）**：同一文件多次读取修改后的旧版本（v1/v2/v3）、排障死胡同分支、冗长工具输出堆积在一维序列里，导致相互矛盾的旧状态干扰推理，且每轮累计 Token 呈 `O(T^2)` 膨胀。
> 2. **滑动窗口截断丢失早期关键约束**：简单截断头部会丢掉第 1~3 轮确立的架构红线与全局不变量。
> 3. **朴素图检索 / 中段删改破坏 LLM KV-Cache 前缀命中率**：如果在 Prompt 前半段频繁重排或删改节点，会导致底层推理引擎的 Prefix KV-Cache 失效，重算 Prefill 成本暴涨 5~10 倍。
> 
> 因此，你需要先深入研究并撰写方案文档，再实现一个**零外部臃肿依赖（基于本机 Node v26 原生 TypeScript、`node:sqlite`、`node:http`、`node:crypto`、`node:test`）**的独立开源级项目与服务 `@opc/context-graph`，并用真实可复现实验量化验证其相比线性 Context 与朴素 RAG 的核心价值！
> 
> ## 交付要求（全部在 `/Users/penglei/developer/ai-github/context-graph` 内完成）
> 
> 1. **深度研究方案 `docs/RESEARCH.md`**：
>    - 系统剖析线性 Context、Sliding Window、Naive Vector RAG 与 Graph-Structured Context 的理论边界与工程权衡。
>    - 提出 **Two-Zone KV-Cache-Aware Property DAG（双区前缀缓存友好型属性有向无环图）** 核心架构：
>      - **节点模型（NodeKind）**：`s
> <truncated 4410 bytes>
> 比 4 种策略：
>      1. `linear_full`（全量线性追加，无截断）
>      2. `linear_sliding_window`（线性滑窗截断至预算内）
>      3. `naive_vector_rag`（按语义相似度 Top-K 切片拼接，重排全文）
>      4. `context_graph_two_zone`（本项目双区图结构编译器）
>      输出 5 项硬核量化指标：
>      - `criticalConstraintRecall`（早期硬约束召回率 %）
>      - `latestStateAccuracy`（文件最新状态召回且无旧版冲突率 %）
>      - `staleTokenRatio`（过期/死胡同噪声 Token 占比 %）
>      - `kvCacheHitRate`（32 轮累计 Prefix KV-Cache 命中率 %）
>      - `effectiveCostIndex`（按 Cached Token 0.1x 计费折算的真实推理成本指数，以 `linear_full` 无缓存或滑窗为参照）
>    - `src/cli.ts` & `src/index.ts`：支持 `serve`、`mcp`、`benchmark`、`demo` 子命令。
>    - `src/context-graph.test.ts`：使用 `node:test` + `node:assert/strict` 编写覆盖图存储、环检测、`supersedes` 遮蔽、Multi-LOD 裁剪、KV-Cache 前缀稳定性、HTTP 服务全接口端到端测试、MCP JSON-RPC 协议握手与调用测试、Benchmark 断言测试（至少 15+ 测试用例）。
> 
> 3. **实测价值评估报告 `docs/EVALUATION.md` 与 `README.md`**：
>    - 运行 `node src/cli.ts benchmark` 获取真实实验数据，把真实输出表格、核心发现、适用边界（什么场景收益最大、什么场景简单线性就够用）以及接入 Agent Loop 的落地方案写进 `docs/EVALUATION.md`。
>    - 写好清晰的 `README.md`、`LICENSE`、`.gitignore`。
> 
> 4. **Git 仓库初始化与干净提交**：
>    - 在 `/Users/penglei/developer/ai-github/context-graph` 执行 `git init -b main`，确保 `npm test` 100% 通过且无多余临时文件，提交干净的初始 commit（不要启动常驻后台进程阻塞退出，主控稍后会统一启动常驻服务）。
> 
> 完成后，向我汇报：仓库结构、测试通过情况、Benchmark 实测对比核心数据表、关键文件路径及 Git Commit SHA。

</details>

任务书中预先写定了架构、指标和“彻底防御 / 量化验证价值”等期望结论，评测场景也由同一执行者设计——这是本次评测不能作为价值证明的主要原因。

## 测试运行轨迹

| 时间 | 结果 |
|---|---|
| 21:04:01 | 23 通过 / 0 失败 |

## 失败与返工

- 20:40:45 Check assembly module in @ppeng/agent-loop：`ls -la /Users/penglei/self-test-grounding/ppeng-agent-core/packages/agent-loop/src/assembly` → **失败 (exit 1)**
- 20:57:30 Run typecheck, demo, and 32-turn benchmark：`cd /Users/penglei/developer/ai-github/context-graph && chmod +x src/cli.ts && npm run typecheck && node src/cli.ts demo && node src/cli.ts benchmark` → **失败 (exit 2)**
- 20:59:08 Run strict typecheck and re-run 32-turn benchmark：`cd /Users/penglei/developer/ai-github/context-graph && npm run typecheck && node src/cli.ts benchmark` → **失败 (exit 2)**
- 21:00:02 Verify strict typecheck and run 32-turn benchmark：`cd /Users/penglei/developer/ai-github/context-graph && npm run typecheck && node src/cli.ts benchmark` → **失败 (exit 2)**

## 完整时间线

| 时间 | 阶段 | 动作 |
|---|---|---|
| 20:39:17 | 环境检查 | Check Node version and sibling repos：`node -v && npm -v && which tsc \|\| true; ls -la /Users/penglei/developer/ai-github/context-graph; ls -la /Users/penglei/developer/ai-github/opc-commerce-kit 2>/d` → 成功 |
| 20:39:36 | 环境检查 | 阅读 `/Users/penglei/developer/ai-github/opc-commerce-kit/package.json` |
| 20:39:36 | 环境检查 | 阅读 `/Users/penglei/developer/ai-github/opc-commerce-kit/tsconfig.json` |
| 20:39:36 | 环境检查 | 阅读 `/Users/penglei/developer/ai-github/opc-commerce-kit/.gitignore` |
| 20:39:36 | 环境检查 | 阅读 `/Users/penglei/developer/ai-github/opc-commerce-kit/LICENSE` |
| 20:39:59 | 环境检查 | Check agent-loop repo and store.ts style：`ls -d /Users/penglei/developer/ai-github/*agent* 2>/dev/null \|\| true; head -n 40 /Users/penglei/developer/ai-github/opc-commerce-kit/src/store.ts` → 成功 |
| 20:40:09 | 环境检查 | Check tsc behavior without node_modules and check loop repos：`tsc -p /Users/penglei/developer/ai-github/opc-commerce-kit/tsconfig.json 2>&1 \| head -n 20; ls -d /Users/penglei/developer/ai-github/*loop* 2>/dev/null \|\| true` → 成功 |
| 20:40:18 | 环境检查 | Run tsc on opc-commerce-kit and list its src files：`tsc -p /Users/penglei/developer/ai-github/opc-commerce-kit/tsconfig.json; echo "EXIT=$?"; ls -la /Users/penglei/developer/ai-github/opc-commerce-kit/src` → 成功 |
| 20:40:27 | 环境检查 | 阅读 `/Users/penglei/developer/ai-github/opc-commerce-kit/src/server.ts` |
| 20:40:27 | 环境检查 | 阅读 `/Users/penglei/developer/ai-github/opc-commerce-kit/src/cli.ts` |
| 20:40:27 | 环境检查 | 阅读 `/Users/penglei/developer/ai-github/agent-chatroom/package.json` |
| 20:40:36 | 环境检查 | Inspect @ppeng/agent-loop source files：`ls -la /Users/penglei/self-test-grounding/ppeng-agent-core/packages/agent-loop/src 2>/dev/null \|\| ls -la /Users/penglei/developer/self-test-grounding/ppeng-agen` → 成功 |
| 20:40:45 | 环境检查 | 阅读 `/Users/penglei/self-test-grounding/ppeng-agent-core/packages/agent-loop/src/types.ts` |
| 20:40:45 | 环境检查 | Check assembly module in @ppeng/agent-loop：`ls -la /Users/penglei/self-test-grounding/ppeng-agent-core/packages/agent-loop/src/assembly` → **失败 (exit 1)** |
| 20:41:06 | 环境检查 | 阅读 `/Users/penglei/developer/self-test-grounding/ppeng-agent-core/packages/agent-loop/src/types.ts` |
| 20:41:06 | 环境检查 | Check assembly module in @ppeng/agent-loop：`ls -la /Users/penglei/developer/self-test-grounding/ppeng-agent-core/packages/agent-loop/src/assembly` → 成功 |
| 20:41:12 | 实现 | 写入 `package.json`（覆盖） |
| 20:41:12 | 实现 | 写入 `tsconfig.json`（覆盖） |
| 20:41:12 | 实现 | 写入 `.gitignore`（覆盖） |
| 20:41:12 | 实现 | 写入 `LICENSE`（覆盖） |
| 20:45:06 | 方案研究 | 写入 `docs/RESEARCH.md`（覆盖） |
| 20:46:23 | 实现 | 写入 `src/types.ts`（覆盖） |
| 20:47:02 | 实现 | 写入 `src/vector.ts`（覆盖） |
| 20:47:52 | 实现 | 写入 `src/graph-store.ts`（覆盖） |
| 20:49:57 | 实现 | 写入 `src/compiler.ts`（覆盖） |
| 20:51:44 | 基准评测 | 写入 `src/benchmark.ts`（覆盖） |
| 20:55:09 | 实现 | 写入 `src/server.ts`（覆盖） |
| 20:55:09 | 实现 | 写入 `src/mcp.ts`（覆盖） |
| 20:56:32 | 实现 | 写入 `src/cli.ts`（覆盖） |
| 20:56:32 | 实现 | 写入 `src/index.ts`（覆盖） |
| 20:57:30 | 基准评测 | Run typecheck, demo, and 32-turn benchmark：`cd /Users/penglei/developer/ai-github/context-graph && chmod +x src/cli.ts && npm run typecheck && node src/cli.ts demo && node src/cli.ts benchmark` → **失败 (exit 2)** |
| 20:57:43 | 基准评测 | Run demo and benchmark directly on Node v26：`ls -la /Users/penglei/developer/ai-github/opc-commerce-kit/node_modules 2>/dev/null \|\| find /Users/penglei/developer/ai-github/opc-commerce-kit -maxdepth 2 -nam` → 成功 |
| 20:57:57 | 实现 | 写入 `src/node-builtins.d.ts`（覆盖） |
| 20:58:50 | 实现 | 修改 `src/compiler.ts` |
| 20:58:50 | 基准评测 | 修改 `src/benchmark.ts` |
| 20:59:08 | 基准评测 | Run strict typecheck and re-run 32-turn benchmark：`cd /Users/penglei/developer/ai-github/context-graph && npm run typecheck && node src/cli.ts benchmark` → **失败 (exit 2)** |
| 20:59:24 | 实现 | 写入 `src/node-builtins.d.ts`（覆盖） |
| 21:00:02 | 基准评测 | Verify strict typecheck and run 32-turn benchmark：`cd /Users/penglei/developer/ai-github/context-graph && npm run typecheck && node src/cli.ts benchmark` → **失败 (exit 2)** |
| 21:00:10 | 实现 | 修改 `src/node-builtins.d.ts` |
| 21:00:35 | 基准评测 | Run strict typecheck and updated benchmark：`cd /Users/penglei/developer/ai-github/context-graph && npm run typecheck && node src/cli.ts benchmark` → 成功 |
| 21:00:43 | 基准评测 | 修改 `src/benchmark.ts` |
| 21:01:10 | 基准评测 | Run symmetric 32-turn benchmark：`cd /Users/penglei/developer/ai-github/context-graph && node src/cli.ts benchmark` → 成功 |
| 21:01:37 | 测试与修复 | 写入 `src/context-graph.test.ts`（覆盖） |
| 21:04:01 | 测试与修复 | Run strict typecheck and full test suite：`cd /Users/penglei/developer/ai-github/context-graph && npm run typecheck && npm test` → 成功（测试 23 通过 / 0 失败） |
| 21:04:27 | 基准评测 | Capture benchmark sensitivity across 30%, 40%, and 50% budgets：`cd /Users/penglei/developer/ai-github/context-graph && echo "=== 30% ===" && node src/cli.ts benchmark --budget-ratio 0.3 && echo "=== 40% ===" && node src/cli.` → 成功 |
| 21:04:36 | 报告与文档 | 写入 `docs/EVALUATION.md`（覆盖） |
| 21:05:35 | 报告与文档 | 写入 `README.md`（覆盖） |
| 21:06:16 | 测试与修复 | Verify typecheck, tests, and create initial git commit：`cd /Users/penglei/developer/ai-github/context-graph && npm run typecheck && npm test && git init -b main && git status --short && git add . && git commit -m "fe` → 成功 |
| 21:06:27 | 环境检查 | manage_task |
| 21:06:41 | 环境检查 | send_message |

## 产出文件

- `package.json`
- `tsconfig.json`
- `.gitignore`
- `LICENSE`
- `docs/RESEARCH.md`
- `src/types.ts`
- `src/vector.ts`
- `src/graph-store.ts`
- `src/compiler.ts`
- `src/benchmark.ts`
- `src/server.ts`
- `src/mcp.ts`
- `src/cli.ts`
- `src/index.ts`
- `src/node-builtins.d.ts`
- `src/context-graph.test.ts`
- `docs/EVALUATION.md`
- `README.md`
