# Empirical Evaluation & Production Integration Report (`@opc/context-graph`)

> Reproducible command: `npm run benchmark` (`node src/cli.ts benchmark --budget-ratio 0.4`)

---

## 1. Benchmark Methodology

To evaluate whether graph-structured context genuinely outperforms flat `messages[]` arrays and vector retrieval under realistic engineering constraints, `src/benchmark.ts` executes a deterministic **32-turn multi-file software engineering trajectory**:

1. **Turns 1–2 (Global Security & Architecture Charter)**: Establishes system instructions, delivery goal, module boundary `task_frame`, zero-dependency stack decision, and **3 non-negotiable invariants**:
   - `C1`: Every SQLite query MUST bind `tenant_id` via parameterized `?` statements (zero string interpolation).
   - `C2`: Webhook HMAC-SHA256 verification MUST use `crypto.timingSafeEqual` with a `300s` clock-skew guard prior to DB access.
   - `C3`: Monetary values MUST use integer `amount_cents` guarded by `UNIQUE(tenant_id, idempotency_key)`.
2. **Turns 3–27 (Iterative Multi-File Evolution + 2 Dead-End Diagnostic Branches)**:
   - Three core source modules (`src/auth.ts`, `src/ledger.ts`, `src/webhook.ts`) each evolve through **4 versions (`v1` $\to$ `v2` $\to$ `v3` $\to$ `v4`)** (`12` total file snapshots, `9` of which become obsolete).
   - **Dead-End Branch #1 (Turns 5–7)**: False OIDC JWKS socket-pool memory leak hypothesis generating verbose stack & heap dumps before being disproved and retracted.
   - **Dead-End Branch #2 (Turns 13–15)**: Failed userspace spinlock experiment for `SQLITE_BUSY` generating lock-contention traces before switching to native `WAL` + `BEGIN IMMEDIATE`.
3. **Turns 28–31 (Post-Integration Soak Telemetry)**: Routine verification & latency soak logs pushing earlier turns upstream.
4. **Turn 32 (Cross-Module Synthesis Audit)**: Requests a full compliance and signature audit across `src/auth.ts`, `src/ledger.ts`, and `src/webhook.ts`.

All four strategies are evaluated using **exact integer token-ID Longest Common Prefix (LCP)** (`longestCommonPrefixLength(prevTokenIds, currTokenIds)`) and industry-standard cached prefill pricing (`0.1x` for Prefix KV-Cache hits, `1.0x` for uncached tokens).

---

## 2. Primary Results (40% Token Budget Ceiling = 3,235 Tokens)

Full 32-turn linear context reaches **8,088 tokens** on Turn 32. Under a strict **40% budget (`3,235 tokens`)**:

| Strategy | Budget Compliant | Final Tokens (Budget) | Critical Constraint Recall | Latest State Accuracy (No Conflict) | Stale Token Ratio | Prefix KV-Cache Hit Rate | Effective Cost Index (Sliding=100) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `linear_full` | NO (Over) | 8088 / 3235 | 100.0% | 0.0% | 46.9% | 94.1% | **36.4** (20957 tok) |
| `linear_sliding_window` | YES | 2970 / 3235 | 0.0% | 66.7% | 19.0% | 31.6% | **100.0** (57532 tok) |
| `naive_vector_rag` | YES | 3225 / 3235 | 0.0% | 0.0% | 70.0% | 1.9% | **142.4** (81919 tok) |
| `context_graph_two_zone` | YES | 3062 / 3235 | 100.0% | 100.0% | 0.0% | 66.6% | **47.2** (27132 tok) |

### Cumulative 32-Turn Token & Billing Breakdown

| Strategy | Cumulative Prompt Tokens | Cached Prefix Tokens (`0.1x`) | Uncached Prefill Tokens (`1.0x`) | Warm-Turn Hit Rate (T2–T32) | Effective Billable Tokens | Cost vs. Uncached Linear |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_full` | 136,779 | 128,691 | 8,088 | 94.2% | 20,957.1 | 15.3% |
| `linear_sliding_window` | 80,411 | 25,421 | 54,990 | 31.7% | 57,532.1 | 42.1% |
| `naive_vector_rag` | 83,338 | 1,577 | 81,761 | 1.9% | 81,918.7 | 59.9% |
| **`context_graph_two_zone`** | **67,671** | **45,043** | **22,628** | **66.9%** | **27,132.3** | **19.8%** |

---

## 3. Budget Sensitivity Sweep (30% / 40% / 50% Budgets)

| Budget Ratio (Cap) | Strategy | Constraint Recall | Latest State Accuracy | Stale Token Ratio | KV-Cache Hit Rate | Effective Billable Tokens |
| :---: | :--- | :---: | :---: | :---: | :---: | ---: |
| **30%** (2,426 tok) | `linear_sliding_window` | 0.0% | 66.7% | 10.4% | 28.0% | 47,355 |
| **30%** (2,426 tok) | `naive_vector_rag` | 0.0% | 0.0% | 61.9% | 2.4% | 64,372 |
| **30%** (2,426 tok) | **`context_graph_two_zone`** | **100.0%** | **100.0%** | **0.0%** | **60.5%** | **26,353** |
| **40%** (3,235 tok) | `linear_sliding_window` | 0.0% | 66.7% | 19.0% | 31.6% | 57,532 |
| **40%** (3,235 tok) | `naive_vector_rag` | 0.0% | 0.0% | 70.0% | 1.9% | 81,919 |
| **40%** (3,235 tok) | **`context_graph_two_zone`** | **100.0%** | **100.0%** | **0.0%** | **66.6%** | **27,132** |
| **50%** (4,044 tok) | `linear_sliding_window` | 0.0% | **33.3%** | 18.6% | 47.5% | 54,963 |
| **50%** (4,044 tok) | `naive_vector_rag` | 0.0% | 0.0% | 68.1% | 1.6% | 97,253 |
| **50%** (4,044 tok) | **`context_graph_two_zone`** | **100.0%** | **100.0%** | **0.0%** | **67.4%** | **29,052** |

---

## 4. Key Empirical Findings

### Finding 1: Expanding a Linear Sliding Window Can *Decrease* Accuracy (Non-Monotonic Context Rot)
Observe `linear_sliding_window` between the **40% budget (`66.7%` latest state accuracy)** and **50% budget (`33.3%` latest state accuracy)**: widening the linear window from `3,235` to `4,044` tokens actually cut conflict-free accuracy in half! A wider linear window reaches back to Turn 25 (`src/webhook.ts` v3) and re-admits it alongside Turn 27 (`src/webhook.ts` v4), re-introducing deprecated signature drift while still failing to reach Turn 10 (`src/auth.ts` v4) or Turns 1–2 (`C1..C3`).

### Finding 2: Naive Vector RAG Is Simultaneously the Most Expensive and Most Conflicted Policy
`naive_vector_rag` incurs **81,919 effective billable tokens** at 40% budget (**3.02x the cost** of `context_graph_two_zone` and **3.91x the cost** of `linear_full`) because similarity re-ranking reduces Prefix KV-Cache hits to **1.9%**—only the tiny static system header is cached. Worse, because `v1`, `v2`, `v3`, and `v4` of each source file all have high cosine similarity to file-level audit queries, **70.0% of its retrieved budget is consumed by superseded file versions and dead-end traces**, crowding out Turn 1–2 invariants (`0.0%` constraint recall, `0.0%` conflict-free state accuracy).

### Finding 3: Volatility-Stratified Two-Zone Compilation Captures Both Accuracy and KV-Cache Economics
By anchoring `pinned_prefix` and `stable_frame` nodes in **Zone 1** (`seq ASC`) and stratifying **Zone 2** into durable structural state (`decision` / `artifact` / `summary` in `seq ASC`) ahead of volatile `ephemeral_leaf` observations (`seq ASC`), `@opc/context-graph`:
- Sustains **66.6% cumulative Prefix KV-Cache hit rate** (**52.8% cheaper** than `linear_sliding_window` and **66.9% cheaper** than `naive_vector_rag`).
- Achieves **100.0% `criticalConstraintRecall`**, **100.0% `latestStateAccuracy`**, and **0.0% `staleTokenRatio`** across all tested budget ratios (`30%`, `40%`, `50%`).
- Uses timestamp-free canonical XML node blocks (`renderCanonicalNodeBlock`) so floating-point PPR score updates never mutate rendered token prefixes.

---

## 5. Applicability Boundaries: When to Use Context Graph vs. Linear Context

| Workload Profile | Recommended Strategy | Engineering Rationale |
| :--- | :--- | :--- |
| **Short single-shot or $\le 8$-turn Q&A** (`< 4k` tokens, zero file edits) | `linear_full` | Append-only prefix cache hits $>90\%$ with negligible stale noise; DAG indexing overhead brings no material benefit. |
| **Pure append-only log summarization / creative writing** (monolithic narrative, no state mutation) | `linear_full` + periodic hierarchical summary | No `resourceKey` version collisions (`supersedes`) or dead-end branches to occlude. |
| **Multi-file Coding Agents ($T \ge 12$ turns, repeated read/edit/test cycles)** | **`context_graph_two_zone`** | Eliminates `v1..v(k-1)` code collisions via automatic `resourceKey` `supersedes` chains while keeping architectural constraints pinned in Zone 1. |
| **Autonomous Debugging / Sweeps / Deep Research** (hypothesis branching & dead ends) | **`context_graph_two_zone`** | `retractScopes` / `retractNodeIds` excises 1,000+ token stack-trace dead ends immediately upon root-cause discovery, avoiding the recurring `0.1x` quadratic cache tax. |
| **Multi-Subtask Orchestration (`opc-os`, `@ppeng/agent-loop` subagents)** | **`context_graph_two_zone`** | `scoped_to` + `depends_on` isolates subtask noise while preserving cross-module contracts. |

---

## 6. Landing Guide: Integrating into `@ppeng/agent-loop`

`@ppeng/agent-loop` (`L0/L1` `SessionMessage` WAL surface) already carries `seq`, `key`, `replaces`, `hidden`, and `parts: MessagePart[]`. Integrating `@opc/context-graph` requires a lightweight two-hook adapter (~45 LOC) with zero breaking changes to loop consumers:

### 6.1 Post-Step Ingestion Hook (`onTurnCommitted`)
Map committed `SessionMessage` items into `GraphStore.ingestTurn`:
1. **System & Charter (`seq <= 2` or `key: "system:*"` / `"constraint:*"`)** $\to$ `kind: "system" | "constraint"`, `tier: "pinned_prefix"`.
2. **File Tool Results (`read_file`, `write_file`, `edit_file`)** $\to$ `artifacts: [{ path, content }]` (`resourceKey: "file:" + path`), automatically superseding prior snapshots of the same file.
3. **CLI / Test Tool Outputs (`run_command`)** $\to$ `toolCalls: [{ toolName, output, isError }]` (`tier: "ephemeral_leaf"`).
4. **Rollback / Dead-End Pruning (`msg.hidden === true` or `msg.replaces`)** $\to$ `retractNodeIds` / `retractScopes`.

### 6.2 Pre-Model Context Assembly Hook (`assembleModelInput`)
Replace linear `fold(messages)` slicing before calling the LLM provider:

```ts
import { GraphStore, compileContext } from "@opc/context-graph";

const store = new GraphStore(process.env.DATA_PATH ?? "data/context-graph.db");

export function buildTurnModelMessages(params: {
  sessionId: string;
  latestUserPrompt: string;
  activeScope?: string;
  tokenBudget: number;
}) {
  const compiled = compileContext(store, {
    graphId: params.sessionId,
    focusQuery: params.latestUserPrompt,
    activeScope: params.activeScope ?? "global",
    maxTokens: params.tokenBudget,
    format: "messages",
  });

  // Directly compatible with OpenAI / Anthropic / vLLM automatic prefix caching
  return {
    messages: compiled.messages,
    kvCacheMetrics: compiled.kvCache,
    conflicts: compiled.conflicts,
  };
}
```
