import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { after, before, describe, it } from "node:test";
import {
  classifyToolEvent,
  ingestToolArtifacts,
  normalizeCommandKey,
  pruneTrajectoryEvents,
} from "./artifact-engine.ts";
import { runBenchmark } from "./benchmark.ts";
import { runCli } from "./cli.ts";
import { compileContext, DEFAULT_COMPACT_MAX_CHARS } from "./compiler.ts";
import {
  compactCodeExcerpt,
  compactCommandExcerpt,
  extractAppFileReferences,
  sanitizeNeutralText,
  truncateAtWordBoundary,
} from "./excerpt.ts";
import { GraphStore } from "./graph-store.ts";
import { ContextGraphMcpSession, handleMcpJsonRpc, MCP_TOOLS, startMcpStdioServer } from "./mcp.ts";
import {
  createContextGraphServer,
  isLoopbackHost,
  type ContextGraphServerInstance,
} from "./server.ts";
import {
  ContextGraphError,
  CycleDetectedError,
  NotFoundError,
  type ToolTrajectoryEvent,
} from "./types.ts";
import {
  cosineSimilarity,
  countTokens,
  embedText,
  longestCommonPrefixLength,
  tokenizeToIds,
} from "./vector.ts";

// Extend node:fs declaration locally for mkdtempSync in test
declare module "node:fs" {
  export function mkdtempSync(prefix: string): string;
}

describe("Vector, Tokenizer & LCP Primitives", () => {
  it("1. tokenizes deterministically and computes exact Longest Common Prefix", () => {
    const t1 = tokenizeToIds("system: enforce tenant_id isolation\nuser: inspect auth.ts v1");
    const t2 = tokenizeToIds("system: enforce tenant_id isolation\nuser: inspect auth.ts v2");
    assert.ok(t1.length > 8);
    assert.equal(t1.length, countTokens("system: enforce tenant_id isolation\nuser: inspect auth.ts v1"));

    const lcp = longestCommonPrefixLength(t1, t2);
    // Everything up to 'v1' vs 'v2' matches identically
    assert.equal(lcp, t1.length - 1);
    assert.equal(longestCommonPrefixLength([], t1), 0);
  });

  it("2. produces unit-normalized semantic vectors distinguishing related vs unrelated code", () => {
    const vAuth1 = embedText("verify HMAC-SHA256 signature with timingSafeEqual in src/auth.ts", "file:src/auth.ts");
    const vAuth2 = embedText("audit src/auth.ts token signature timingSafeEqual check");
    const vCss = embedText("adjust flexbox grid border-radius padding in styles/button.css", "file:styles/button.css");

    const simRelated = cosineSimilarity(vAuth1, vAuth2);
    const simUnrelated = cosineSimilarity(vAuth1, vCss);
    assert.ok(simRelated > 0.35, `Expected high similarity, got ${simRelated}`);
    assert.ok(simRelated > simUnrelated + 0.25, `Expected ${simRelated} >> ${simUnrelated}`);
  });
});

describe("GraphStore Persistence, Versioning & DAG Invariants", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "cg-store-test-"));
  });

  after(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("3. persists graphs, nodes, and edges across SQLite file re-opens", () => {
    const dbFile = join(tmpDir, "persist.db");
    const store1 = new GraphStore(dbFile);
    const g = store1.createGraph({ id: "g_persist", name: "Persistent Graph", defaultBudget: 2048 });
    const n1 = store1.addNode(g.id, {
      id: "c1",
      kind: "constraint",
      content: "All SQL queries must bind tenant_id.",
    });
    const n2 = store1.addNode(g.id, {
      id: "d1",
      kind: "decision",
      content: "Use SQLite prepared statements everywhere.",
      dependsOn: [n1.id],
    });
    assert.equal(n2.seq, 2);
    store1.close();

    const store2 = new GraphStore(dbFile);
    const snap = store2.getGraphSnapshot("g_persist");
    assert.equal(snap.graph.defaultBudget, 2048);
    assert.equal(snap.stats.totalNodes, 2);
    assert.equal(snap.stats.totalEdges, 1);
    assert.equal(snap.edges[0]?.kind, "depends_on");
    store2.close();
  });

  it("4. automatically chains `supersedes` edges and increments version on matching resourceKey", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Versioning Test" });

    const v1 = store.addNode(g.id, {
      id: "auth_v1",
      kind: "artifact",
      resourceKey: "file:src/auth.ts",
      content: "export const v = 1;",
    });
    const v2 = store.addNode(g.id, {
      id: "auth_v2",
      kind: "artifact",
      resourceKey: "file:src/auth.ts",
      content: "export const v = 2;",
    });
    const v3 = store.addNode(g.id, {
      id: "auth_v3",
      kind: "artifact",
      resourceKey: "file:src/auth.ts",
      content: "export const v = 3;",
    });

    assert.equal(v1.version, 1);
    assert.equal(v2.version, 2);
    assert.equal(v3.version, 3);

    const refreshedV1 = store.getNode(g.id, "auth_v1");
    const refreshedV2 = store.getNode(g.id, "auth_v2");
    const refreshedV3 = store.getNode(g.id, "auth_v3");

    assert.equal(refreshedV1.status, "superseded");
    assert.equal(refreshedV1.supersededBy, "auth_v2");
    assert.equal(refreshedV2.status, "superseded");
    assert.equal(refreshedV2.supersededBy, "auth_v3");
    assert.equal(refreshedV3.status, "active");

    const supEdges = store.listEdges(g.id, { kind: "supersedes" });
    assert.equal(supEdges.length, 2);
    store.close();
  });

  it("5. rejects self-loops and transitive cycles on `depends_on` and `supersedes` edges", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Cycle Detection" });

    store.addNode(g.id, { id: "A", kind: "decision", content: "Decision A" });
    store.addNode(g.id, { id: "B", kind: "decision", content: "Decision B", dependsOn: ["A"] });
    store.addNode(g.id, { id: "C", kind: "decision", content: "Decision C", dependsOn: ["B"] });

    // Self-loop must throw CycleDetectedError
    assert.throws(
      () => store.addEdge(g.id, { sourceId: "A", targetId: "A", kind: "depends_on" }),
      CycleDetectedError,
    );

    // Transitive cycle A -> C (since C -> B -> A already exists) must throw CycleDetectedError
    assert.throws(
      () => store.addEdge(g.id, { sourceId: "A", targetId: "C", kind: "depends_on" }),
      CycleDetectedError,
    );

    // Failed cycle inside addNode transaction must roll back node insertion cleanly
    assert.throws(() => {
      store.addNode(g.id, { id: "A", kind: "decision", content: "Duplicate ID" });
    });
    assert.equal(store.listNodes(g.id).length, 3);
    store.close();
  });

  it("6. supports atomic turn ingestion (`ingestTurn`), scope retraction, and Mermaid export", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Ingest & Mermaid" });

    store.ingestTurn(g.id, {
      turn: 1,
      scope: "global",
      system: "System Charter",
      goal: "Ship zero-bug gateway",
      constraints: [{ id: "c_root", content: "Never use eval()." }],
    });

    store.ingestTurn(g.id, {
      turn: 2,
      scope: "dead-end-branch",
      userMessage: "Try experimental regex parser",
      toolCalls: [
        { id: "bad_trace", toolName: "run_test", output: "Error: catastrophic backtracking", isError: true },
      ],
    });

    const turn3 = store.ingestTurn(g.id, {
      turn: 3,
      scope: "parser",
      retractScopes: ["dead-end-branch"],
      artifacts: [{ id: "parser_v1", path: "src/parser.ts", content: "export function parseSafe() {}", dependsOn: ["c_root"] }],
    });

    assert.equal(turn3.retractedNodeIds.length, 2);
    assert.equal(store.getNode(g.id, "bad_trace").status, "retracted");

    const mermaidAll = store.toMermaid(g.id);
    assert.match(mermaidAll, /flowchart TD/);
    assert.match(mermaidAll, /Zone 1: Append-Mostly Stable Prefix Spine/);
    assert.match(mermaidAll, /Zone 2: Dynamic Working Subgraph/);
    assert.match(mermaidAll, /depends_on/);

    const mermaidActiveOnly = store.toMermaid(g.id, { includeInactive: false });
    assert.ok(!mermaidActiveOnly.includes("bad_trace"));
    store.close();
  });
});

describe("Two-Zone KV-Cache-Aware Compiler (`compileContext`)", () => {
  it("7. preserves Zone 1 Prefix KV-Cache stability when Zone 2 mutates across turns", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "KV Prefix Stability", defaultBudget: 1600 });

    store.ingestTurn(g.id, {
      turn: 1,
      system: "Core Agent Runtime System Instructions",
      goal: "Implement deterministic billing engine",
      constraints: [
        "CONSTRAINT-1: Integer cents only.",
        "CONSTRAINT-2: Idempotent ledger writes.",
      ],
      taskFrame: { id: "tf_core", content: "Module contract: ledger + webhook." },
      artifacts: [{ path: "src/billing.ts", content: "export const BILLING_V = 1;" }],
    });

    const c1 = compileContext(store, {
      graphId: g.id,
      focusQuery: "Review billing v1",
      activeScope: "billing",
    });
    assert.equal(c1.kvCache.prefixCachedTokens, 0); // cold start

    // Turn 2: Replace billing.ts with v2 and add observation in Zone 2
    store.ingestTurn(g.id, {
      turn: 2,
      scope: "billing",
      userMessage: "Upgrade billing.ts to v2 with integer cents validation.",
      artifacts: [{ path: "src/billing.ts", content: "export const BILLING_V = 2;\nexport function chargeCents(c: number) { return c; }" }],
    });

    const c2 = compileContext(store, {
      graphId: g.id,
      focusQuery: "Verify billing v2",
      activeScope: "billing",
    });

    // Entire Zone 1 prefix MUST be cached token-for-token!
    assert.ok(
      c2.kvCache.prefixCachedTokens >= c1.kvCache.zone1Tokens - 2,
      `Expected prefixCachedTokens (${c2.kvCache.prefixCachedTokens}) >= zone1 (${c1.kvCache.zone1Tokens})`,
    );
    assert.ok(c2.kvCache.prefixHitRate > 0.45);
    assert.ok(!c2.promptText.includes("BILLING_V = 1"), "Superseded v1 must be occluded");
    assert.ok(c2.promptText.includes("BILLING_V = 2"), "Active v2 must be present");
    store.close();
  });

  it("8. pulls transitive `depends_on` ancestors and forward-resolves superseded dependencies", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Closure & Forward Resolution", defaultBudget: 2000 });

    // Ancestor v1 -> later superseded by Ancestor v2
    store.addNode(g.id, {
      id: "schema_v1",
      kind: "artifact",
      resourceKey: "file:src/schema.ts",
      content: "export interface TenantRowV1 { id: string }",
    });
    store.addNode(g.id, {
      id: "repo_mid",
      kind: "decision",
      content: "Repository layer maps TenantRow to domain entity.",
      dependsOn: ["schema_v1"],
    });
    store.addNode(g.id, {
      id: "handler_leaf",
      kind: "artifact",
      resourceKey: "file:src/handler.ts",
      content: "export function getTenantEndpoint() { return 'ok'; }",
      dependsOn: ["repo_mid"],
    });

    // Now schema_v2 supersedes schema_v1!
    store.addNode(g.id, {
      id: "schema_v2",
      kind: "artifact",
      resourceKey: "file:src/schema.ts",
      content: "export interface TenantRowV2 { id: string; isolated: true }",
    });

    const compiled = compileContext(store, {
      graphId: g.id,
      focusNodeIds: ["handler_leaf"],
      focusQuery: "getTenantEndpoint",
      maxTokens: 1500,
    });

    const z2Ids = new Set(compiled.zone2Nodes.map((n) => n.id));
    assert.ok(z2Ids.has("handler_leaf"));
    assert.ok(z2Ids.has("repo_mid"), "Direct depends_on ancestor must be pulled");
    assert.ok(z2Ids.has("schema_v2"), "Superseded transitive dependency must forward-resolve to schema_v2");
    assert.ok(!z2Ids.has("schema_v1"), "Superseded schema_v1 must remain occluded");
    store.close();
  });

  it("9. gracefully degrades Zone 2 LODs (`full` -> `summary` -> `stub`) under tightening budgets", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Multi-LOD Knapsack" });

    store.addNode(g.id, {
      id: "sys",
      kind: "system",
      content: "Compact system header.",
    });

    for (let i = 1; i <= 6; i++) {
      const longBody = [
        `export function serviceModule${i}(input: string): string {`,
        ...Array.from({ length: 12 }, (_, k) => `  const step_${i}_${k} = input.trim().toLowerCase();`),
        `  return "module_${i}_done";`,
        `}`,
      ].join("\n");
      store.addNode(g.id, {
        id: `mod_${i}`,
        kind: "artifact",
        resourceKey: `file:src/mod_${i}.ts`,
        content: longBody,
        summary: `serviceModule${i}(input: string): string`,
      });
    }

    const generous = compileContext(store, {
      graphId: g.id,
      focusQuery: "serviceModule1 serviceModule2",
      maxTokens: 1800,
      recordCacheState: false,
    });
    const tight = compileContext(store, {
      graphId: g.id,
      focusQuery: "serviceModule1",
      maxTokens: 340,
      recordCacheState: false,
    });

    assert.ok(generous.kvCache.totalTokens <= 1800);
    assert.ok(tight.kvCache.totalTokens <= 340, `Expected <= 340, got ${tight.kvCache.totalTokens}`);
    assert.ok(generous.diagnostics.lodCounts.full > tight.diagnostics.lodCounts.full);
    assert.ok(tight.diagnostics.lodCounts.summary + tight.diagnostics.lodCounts.stub >= 1);
    store.close();
  });

  it("10. detects unresolved `conflicts_with` edges and clears alert when resolved", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Conflict Detection" });

    store.addNode(g.id, {
      id: "dec_rest",
      kind: "decision",
      content: "Expose synchronous REST JSON webhook acknowledgment.",
    });
    store.addNode(g.id, {
      id: "dec_grpc_only",
      kind: "decision",
      content: "Disable all HTTP JSON endpoints; allow gRPC binary streams only.",
      conflictsWith: ["dec_rest"],
    });

    const cWithConflict = compileContext(store, {
      graphId: g.id,
      focusQuery: "webhook endpoint protocol",
    });
    assert.equal(cWithConflict.conflicts.length, 1);
    assert.match(cWithConflict.promptText, /\[CONFLICT ALERT\]/);

    // Retract conflicting decision -> conflict alert disappears
    store.updateNode(g.id, "dec_grpc_only", { status: "retracted" });
    const cResolved = compileContext(store, {
      graphId: g.id,
      focusQuery: "webhook endpoint protocol",
    });
    assert.equal(cResolved.conflicts.length, 0);
    assert.ok(!cResolved.promptText.includes("[CONFLICT ALERT]"));
    store.close();
  });

  it("11. consolidates leaf observations when covered by a `summarizes` node", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Summarizes Consolidation" });

    store.addNode(g.id, {
      id: "obs_1",
      kind: "observation",
      content: "Test shard 1 passed 42 assertions in 120ms with zero warnings.",
    });
    store.addNode(g.id, {
      id: "obs_2",
      kind: "observation",
      content: "Test shard 2 passed 38 assertions in 115ms with zero warnings.",
    });
    store.addNode(g.id, {
      id: "sum_shards",
      kind: "summary",
      content: "All 80 unit assertions across shards 1-2 passed cleanly.",
      summarizes: ["obs_1", "obs_2"],
    });

    const compiled = compileContext(store, {
      graphId: g.id,
      focusQuery: "unit test shard results",
      maxTokens: 160,
    });
    assert.ok(compiled.kvCache.totalTokens <= 160);
    assert.ok(compiled.zone2Nodes.some((n) => n.id === "sum_shards"));
    store.close();
  });

  it("12. emits OpenAI `messages[]`, XML `promptText`, and `@ppeng/agent-loop` wire messages", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ name: "Dual Format Output" });
    store.ingestTurn(g.id, {
      turn: 1,
      system: "System rules.",
      userMessage: "Audit src/index.ts",
      artifacts: [{ path: "src/index.ts", content: "export const READY = true;" }],
    });

    const both = compileContext(store, { graphId: g.id, focusQuery: "index.ts", format: "both" });
    assert.ok(both.messages.length >= 2);
    assert.equal(both.messages[0]?.role, "system");
    assert.match(both.promptText, /<zone_1_stable_prefix>/);
    assert.match(both.promptText, /<zone_2_working_subgraph>/);
    assert.ok(both.agentLoopMessages.length >= 3);
    assert.equal(both.agentLoopMessages[0]?.zone, 1);

    const onlyMessages = compileContext(store, {
      graphId: g.id,
      focusQuery: "index.ts",
      format: "messages",
      recordCacheState: false,
    });
    assert.equal(onlyMessages.promptText, "");
    assert.ok(onlyMessages.messages.length >= 2);
    store.close();
  });
});

describe("HTTP REST Microservice End-to-End", () => {
  let app: ContextGraphServerInstance;
  let baseUrl: string;

  before(async () => {
    app = createContextGraphServer({ dataPath: ":memory:", port: 0 });
    const bound = await app.start(0, "127.0.0.1");
    baseUrl = bound.url;
  });

  after(async () => {
    await app.close();
  });

  it("13. serves GET /health with service metadata and live counts", async () => {
    const res = await fetch(`${baseUrl}/health`);
    assert.equal(res.status, 200);
    const body = (await res.json()) as { ok: boolean; service: string; version: string; graphsCount: number };
    assert.equal(body.ok, true);
    assert.equal(body.service, "context-graph");
    assert.equal(body.version, "0.1.0");
    assert.equal(body.graphsCount, 0);
  });

  it("14. creates, lists, and fetches graphs via /v1/graphs", async () => {
    const createRes = await fetch(`${baseUrl}/v1/graphs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: "http_g1", name: "HTTP E2E Graph", defaultBudget: 1500 }),
    });
    assert.equal(createRes.status, 201);

    const listRes = await fetch(`${baseUrl}/v1/graphs`);
    const listBody = (await listRes.json()) as { graphs: Array<{ id: string }> };
    assert.equal(listBody.graphs.length, 1);
    assert.equal(listBody.graphs[0]?.id, "http_g1");

    const getRes = await fetch(`${baseUrl}/v1/graphs/http_g1`);
    assert.equal(getRes.status, 200);

    const missingRes = await fetch(`${baseUrl}/v1/graphs/non_existent`);
    assert.equal(missingRes.status, 404);
  });

  it("15. handles node creation, edge linking, and returns HTTP 400 on DAG cycles", async () => {
    const n1Res = await fetch(`${baseUrl}/v1/graphs/http_g1/nodes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: "n_a", kind: "constraint", content: "Strict TLS 1.3 only." }),
    });
    assert.equal(n1Res.status, 201);

    const n2Res = await fetch(`${baseUrl}/v1/graphs/http_g1/nodes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: "n_b", kind: "decision", content: "Configure TLS 1.3 cipher suite.", dependsOn: ["n_a"] }),
    });
    assert.equal(n2Res.status, 201);

    // Attempt cycle n_a -> depends_on -> n_b
    const cycleRes = await fetch(`${baseUrl}/v1/graphs/http_g1/edges`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sourceId: "n_a", targetId: "n_b", kind: "depends_on" }),
    });
    assert.equal(cycleRes.status, 400);
    const errBody = (await cycleRes.json()) as { ok: boolean; error: { code: string } };
    assert.equal(errBody.ok, false);
    assert.equal(errBody.error.code, "CYCLE_DETECTED");
  });

  it("16. ingests turns, compiles context stateful KV-Cache, and exports Mermaid over HTTP", async () => {
    const ing1 = await fetch(`${baseUrl}/v1/graphs/http_g1/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        turn: 2,
        scope: "tls",
        artifacts: [{ path: "src/tls.ts", content: "export const MIN_TLS = '1.2'; // draft v1" }],
      }),
    });
    assert.equal(ing1.status, 201);

    const comp1 = await (
      await fetch(`${baseUrl}/v1/graphs/http_g1/compile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ focusQuery: "Check src/tls.ts", activeScope: "tls" }),
      })
    ).json() as { kvCache: { prefixHitRate: number } };
    assert.equal(comp1.kvCache.prefixHitRate, 0);

    // Second turn supersedes src/tls.ts -> compile again -> stateful KV cache hit > 0!
    await fetch(`${baseUrl}/v1/graphs/http_g1/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        turn: 3,
        scope: "tls",
        artifacts: [{ path: "src/tls.ts", content: "export const MIN_TLS = '1.3'; // hardened v2", dependsOn: ["n_a"] }],
      }),
    });

    const comp2 = await (
      await fetch(`${baseUrl}/v1/graphs/http_g1/compile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ focusQuery: "Verify src/tls.ts TLS 1.3", activeScope: "tls" }),
      })
    ).json() as { kvCache: { prefixHitRate: number }; promptText: string };
    assert.ok(comp2.kvCache.prefixHitRate > 0.35);
    assert.ok(comp2.promptText.includes("1.3"));
    assert.ok(!comp2.promptText.includes("1.2"));

    const mmdRes = await fetch(`${baseUrl}/v1/graphs/http_g1/mermaid?format=text`);
    assert.equal(mmdRes.status, 200);
    const mmdText = await mmdRes.text();
    assert.match(mmdText, /flowchart TD/);
  });

  it("17. runs online benchmark via POST /v1/benchmark", async () => {
    const res = await fetch(`${baseUrl}/v1/benchmark`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ budgetRatio: 0.4 }),
    });
    assert.equal(res.status, 200);
    const body = (await res.json()) as {
      ok: boolean;
      report: { strategies: { context_graph_two_zone: { latestStateAccuracy: number } } };
    };
    assert.equal(body.ok, true);
    assert.equal(body.report.strategies.context_graph_two_zone.latestStateAccuracy, 100);
  });
});

describe("MCP JSON-RPC 2.0 Server & Stdio Transport", () => {
  it("18. completes MCP handshake and lists all 7 Context Graph tools", () => {
    const store = new GraphStore(":memory:");
    const session = new ContextGraphMcpSession(store);

    const initResp = session.handleMessage({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2024-11-05", capabilities: {} },
    });
    assert.ok(initResp && "result" in initResp);

    // Notification returns null
    const notifResp = session.handleMessage({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    });
    assert.equal(notifResp, null);

    const listResp = session.handleMessage({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
    }) as { result: { tools: typeof MCP_TOOLS } };
    assert.equal(listResp.result.tools.length, 7);
    const names = listResp.result.tools.map((t) => t.name);
    assert.deepEqual(names, [
      "cg_create_graph",
      "cg_add_node",
      "cg_link_nodes",
      "cg_ingest_turn",
      "cg_ingest_artifacts",
      "cg_compile_context",
      "cg_export_mermaid",
    ]);
    store.close();
  });

  it("19. executes all 6 MCP tools end-to-end and marks tool errors cleanly", () => {
    const store = new GraphStore(":memory:");
    const session = new ContextGraphMcpSession(store);

    const callTool = (id: number, name: string, args: Record<string, unknown>) =>
      session.handleMessage({
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: { name, arguments: args },
      }) as { result: { isError: boolean; structuredContent?: Record<string, unknown> } };

    const r1 = callTool(1, "cg_create_graph", { id: "mcp_g", name: "MCP Graph" });
    assert.equal(r1.result.isError, false);

    const r2 = callTool(2, "cg_add_node", {
      graphId: "mcp_g",
      id: "mcp_c1",
      kind: "constraint",
      content: "Enforce Ed25519 signature check.",
    });
    assert.equal(r2.result.isError, false);

    const r3 = callTool(3, "cg_add_node", {
      graphId: "mcp_g",
      id: "mcp_d1",
      kind: "decision",
      content: "Verify license header with Ed25519.",
    });
    assert.equal(r3.result.isError, false);

    const r4 = callTool(4, "cg_link_nodes", {
      graphId: "mcp_g",
      sourceId: "mcp_d1",
      targetId: "mcp_c1",
      kind: "depends_on",
    });
    assert.equal(r4.result.isError, false);

    // Cycle error surfaces as MCP tool `isError: true`
    const rCycle = callTool(5, "cg_link_nodes", {
      graphId: "mcp_g",
      sourceId: "mcp_c1",
      targetId: "mcp_d1",
      kind: "depends_on",
    });
    assert.equal(rCycle.result.isError, true);

    const rIngest = callTool(6, "cg_ingest_turn", {
      graphId: "mcp_g",
      turn: 2,
      artifacts: [{ path: "src/lic.ts", content: "export function checkEd25519() { return true; }" }],
    });
    assert.equal(rIngest.result.isError, false);

    const rCompile = callTool(7, "cg_compile_context", {
      graphId: "mcp_g",
      focusQuery: "Ed25519 license check",
    });
    assert.equal(rCompile.result.isError, false);

    const rMermaid = callTool(8, "cg_export_mermaid", { graphId: "mcp_g" });
    assert.equal(rMermaid.result.isError, false);
    assert.match(String(rMermaid.result.structuredContent?.mermaid), /flowchart TD/);
    store.close();
  });

  it("20. streams newline-delimited and Content-Length framed JSON-RPC over stdio", async () => {
    const input = new PassThrough();
    const output = new PassThrough();
    const chunks: string[] = [];
    output.on("data", (c) => chunks.push(typeof c === "string" ? c : c.toString("utf8")));

    const handle = startMcpStdioServer({ dataPath: ":memory:", input, output });
    input.write(`${JSON.stringify({ jsonrpc: "2.0", id: "ping-1", method: "ping" })}\n`);
    await new Promise((r) => setTimeout(r, 20));
    assert.ok(chunks.join("").includes('"id":"ping-1"'));

    // Also verify Content-Length framing
    const body = JSON.stringify({ jsonrpc: "2.0", id: "init-cl", method: "initialize" });
    input.write(`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
    await new Promise((r) => setTimeout(r, 20));
    assert.ok(chunks.join("").includes('"id":"init-cl"'));

    handle.stop();
  });
});

describe("CLI & 32-Turn Comparative Benchmark Suite", () => {
  it("21. executes CLI `demo --json`, `--help`, and rejects unknown subcommands", async () => {
    let out = "";
    let err = "";
    const helpCode = await runCli(["--help"], { stdout: (s) => { out += s; } });
    assert.equal(helpCode, 0);
    assert.match(out, /context-graph <command>/);

    out = "";
    const demoCode = await runCli(["demo", "--json"], { stdout: (s) => { out += s; } });
    assert.equal(demoCode, 0);
    const demoParsed = JSON.parse(out) as { turn2Cache: { prefixHitRate: number } };
    assert.ok(demoParsed.turn2Cache.prefixHitRate > 0.4);

    const badCode = await runCli(["nonexistent"], { stderr: (s) => { err += s; } });
    assert.equal(badCode, 1);
    assert.match(err, /Unknown command/);
  });

  it("22. validates NotFoundError on missing graph operations", () => {
    const store = new GraphStore(":memory:");
    assert.throws(() => store.getGraph("missing_id"), NotFoundError);
    store.close();
  });

  it("23. proves `context_graph_two_zone` superiority across all 5 core metrics on the 32-turn benchmark", () => {
    const report = runBenchmark({ budgetRatio: 0.4 });
    const full = report.strategies.linear_full;
    const sliding = report.strategies.linear_sliding_window;
    const rag = report.strategies.naive_vector_rag;
    const cg = report.strategies.context_graph_two_zone;

    assert.equal(report.totalTurns, 32);

    // 1. Budget Compliance: linear_full exceeds budget; Context Graph stays strictly within 40% budget
    assert.equal(full.budgetCompliant, false);
    assert.equal(cg.budgetCompliant, true);
    assert.ok(cg.finalTurnTokens <= report.tokenBudget);

    // 2. Critical Constraint Recall: Context Graph preserves 100%, Sliding Window loses early constraints
    assert.equal(cg.criticalConstraintRecall, 100);
    assert.ok(cg.criticalConstraintRecall > sliding.criticalConstraintRecall);

    // 3. Conflict-Free Latest State Accuracy: Context Graph hits 100% with zero v1..v3 collisions
    assert.equal(cg.latestStateAccuracy, 100);
    assert.ok(cg.latestStateAccuracy > full.latestStateAccuracy);
    assert.ok(cg.latestStateAccuracy > sliding.latestStateAccuracy);
    assert.ok(cg.latestStateAccuracy > rag.latestStateAccuracy);

    // 4. Stale Token Ratio: Context Graph eliminates superseded & dead-end noise (0%)
    assert.equal(cg.staleTokenRatio, 0);
    assert.ok(cg.staleTokenRatio < full.staleTokenRatio);
    assert.ok(cg.staleTokenRatio < sliding.staleTokenRatio);
    assert.ok(cg.staleTokenRatio < rag.staleTokenRatio);

    // 5. Prefix KV-Cache Hit Rate & Effective Cost Index: beats both budget-compliant baselines decisively
    assert.ok(
      cg.kvCacheHitRate > sliding.kvCacheHitRate + 25,
      `Expected CG hit rate (${cg.kvCacheHitRate}%) >> Sliding (${sliding.kvCacheHitRate}%)`,
    );
    assert.ok(
      cg.kvCacheHitRate > rag.kvCacheHitRate + 50,
      `Expected CG hit rate (${cg.kvCacheHitRate}%) >> RAG (${rag.kvCacheHitRate}%)`,
    );
    assert.ok(
      cg.effectiveCostIndex < 60,
      `Expected CG effectiveCostIndex (${cg.effectiveCostIndex}) < 60 vs Sliding=100`,
    );
    assert.ok(cg.effectiveCostIndex < rag.effectiveCostIndex);
  });
});

describe("Bearer Auth, Fail-Closed Protection & Streamable HTTP /mcp", () => {
  it("24. refuses to start on non-loopback hosts without >=16-char authToken (fail-closed)", async () => {
    // Verify loopback host classification
    assert.equal(isLoopbackHost("127.0.0.1"), true);
    assert.equal(isLoopbackHost("::1"), true);
    assert.equal(isLoopbackHost("localhost"), true);
    assert.equal(isLoopbackHost("LOCALHOST"), true);
    assert.equal(isLoopbackHost("100.108.223.41"), false);
    assert.equal(isLoopbackHost("0.0.0.0"), false);
    assert.equal(isLoopbackHost("::"), false);
    assert.equal(isLoopbackHost("127.0.0.2"), false);
    assert.equal(isLoopbackHost(""), false);

    // Non-loopback host without authToken must reject with ContextGraphError
    const noTokenApp = createContextGraphServer({ dataPath: ":memory:", authToken: "" });
    try {
      await assert.rejects(
        () => noTokenApp.start(0, "100.108.223.41"),
        (err: unknown) =>
          err instanceof ContextGraphError && err.code === "INSECURE_BIND_REFUSED",
      );
      await assert.rejects(
        () => noTokenApp.start(0, "0.0.0.0"),
        (err: unknown) =>
          err instanceof ContextGraphError && err.code === "INSECURE_BIND_REFUSED",
      );
    } finally {
      await noTokenApp.close();
    }

    // Non-loopback host with <16-char authToken must reject without leaking token
    const shortSecret = "short-token-15c";
    assert.equal(shortSecret.length, 15);
    const shortTokenApp = createContextGraphServer({
      dataPath: ":memory:",
      authToken: shortSecret,
    });
    try {
      await assert.rejects(
        () => shortTokenApp.start(0, "100.108.223.41"),
        (err: unknown) =>
          err instanceof ContextGraphError &&
          err.code === "INSECURE_BIND_REFUSED" &&
          !err.message.includes(shortSecret),
      );
    } finally {
      await shortTokenApp.close();
    }

    // Whitespace-only >=16-char token trims to empty and must also be rejected
    const blankTokenApp = createContextGraphServer({
      dataPath: ":memory:",
      authToken: "                    ",
    });
    try {
      await assert.rejects(
        () => blankTokenApp.start(0, "100.108.223.41"),
        ContextGraphError,
      );
    } finally {
      await blankTokenApp.close();
    }

    // Verify CG_AUTH_TOKEN env var pickup & CLI fail-closed non-leakage
    const prevToken = process.env.CG_AUTH_TOKEN;
    try {
      process.env.CG_AUTH_TOKEN = "env_secret_token_9876543210";
      const envApp = createContextGraphServer({ dataPath: ":memory:", port: 0 });
      try {
        const bound = await envApp.start(0, "127.0.0.1");
        const unauthRes = await fetch(`${bound.url}/v1/graphs`);
        assert.equal(unauthRes.status, 401);
        const okRes = await fetch(`${bound.url}/v1/graphs`, {
          headers: { Authorization: "Bearer env_secret_token_9876543210" },
        });
        assert.equal(okRes.status, 200);
      } finally {
        await envApp.close();
      }

      const leakedSecret = "tiny_secret_123";
      process.env.CG_AUTH_TOKEN = leakedSecret;
      let cliOut = "";
      let cliErr = "";
      await assert.rejects(
        () =>
          runCli(["serve", "--host", "100.108.223.41", "--port", "0", "--db", ":memory:"], {
            stdout: (s) => { cliOut += s; },
            stderr: (s) => { cliErr += s; },
          }),
        (err: unknown) =>
          err instanceof ContextGraphError && !err.message.includes(leakedSecret),
      );
      assert.ok(!cliOut.includes(leakedSecret));
      assert.ok(!cliErr.includes(leakedSecret));
    } finally {
      if (prevToken === undefined) {
        delete process.env.CG_AUTH_TOKEN;
      } else {
        process.env.CG_AUTH_TOKEN = prevToken;
      }
    }
  });

  it("25. enforces Bearer token auth on /v1/* and /mcp and splits minimal vs full /health", async () => {
    const token = "cg_bearer_secret_token_2026_tailscale";
    const app = createContextGraphServer({
      dataPath: ":memory:",
      port: 0,
      authToken: token,
    });
    const bound = await app.start(0, "127.0.0.1");
    const authHeaders = { Authorization: `Bearer ${token}` };

    try {
      // Unauthenticated GET /health returns ONLY minimal fields
      const healthNoAuth = await fetch(`${bound.url}/health`);
      assert.equal(healthNoAuth.status, 200);
      const healthNoAuthBody = (await healthNoAuth.json()) as Record<string, unknown>;
      assert.deepEqual(healthNoAuthBody, {
        ok: true,
        service: "context-graph",
        version: "0.1.0",
      });
      assert.equal("dbPath" in healthNoAuthBody, false);
      assert.equal("graphsCount" in healthNoAuthBody, false);

      // Wrong token on GET /health also returns ONLY minimal fields
      const healthWrongAuth = await fetch(`${bound.url}/health`, {
        headers: { Authorization: "Bearer wrong_token_value_1234567890" },
      });
      assert.equal(healthWrongAuth.status, 200);
      const healthWrongBody = (await healthWrongAuth.json()) as Record<string, unknown>;
      assert.deepEqual(healthWrongBody, {
        ok: true,
        service: "context-graph",
        version: "0.1.0",
      });

      // Valid Bearer token on GET /health returns full metadata & stats
      const healthAuthed = await fetch(`${bound.url}/health`, { headers: authHeaders });
      assert.equal(healthAuthed.status, 200);
      const healthAuthedBody = (await healthAuthed.json()) as Record<string, unknown>;
      assert.equal(healthAuthedBody.ok, true);
      assert.equal(healthAuthedBody.service, "context-graph");
      assert.equal(healthAuthedBody.version, "0.1.0");
      assert.equal(healthAuthedBody.dbPath, ":memory:");
      assert.equal(typeof healthAuthedBody.uptimeSec, "number");
      assert.equal(healthAuthedBody.graphsCount, 0);
      assert.equal(healthAuthedBody.nodesCount, 0);
      assert.equal(healthAuthedBody.edgesCount, 0);

      // /v1/graphs: missing token & wrong token (different/same length) -> 401
      const expected401 = {
        ok: false,
        error: {
          code: "UNAUTHORIZED",
          message: "Valid Bearer token required",
        },
      };

      const graphsNoAuth = await fetch(`${bound.url}/v1/graphs`);
      assert.equal(graphsNoAuth.status, 401);
      assert.deepEqual(await graphsNoAuth.json(), expected401);

      const graphsShortWrong = await fetch(`${bound.url}/v1/graphs`, {
        headers: { Authorization: "Bearer short" },
      });
      assert.equal(graphsShortWrong.status, 401);
      assert.deepEqual(await graphsShortWrong.json(), expected401);

      const graphsSameLenWrong = await fetch(`${bound.url}/v1/graphs`, {
        headers: { Authorization: "Bearer cg_bearer_secret_token_2026_tailscalX" },
      });
      assert.equal(graphsSameLenWrong.status, 401);
      assert.deepEqual(await graphsSameLenWrong.json(), expected401);

      // /v1/graphs with valid Bearer token -> 200
      const graphsAuthed = await fetch(`${bound.url}/v1/graphs`, { headers: authHeaders });
      assert.equal(graphsAuthed.status, 200);
      assert.deepEqual(await graphsAuthed.json(), { ok: true, graphs: [] });

      // /mcp: missing or wrong token -> 401 across POST/GET/DELETE
      const mcpPostNoAuth = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "ping" }),
      });
      assert.equal(mcpPostNoAuth.status, 401);
      assert.deepEqual(await mcpPostNoAuth.json(), expected401);

      const mcpPostWrongAuth = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer invalid_mcp_token",
        },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "ping" }),
      });
      assert.equal(mcpPostWrongAuth.status, 401);
      assert.deepEqual(await mcpPostWrongAuth.json(), expected401);

      const mcpGetNoAuth = await fetch(`${bound.url}/mcp`, { method: "GET" });
      assert.equal(mcpGetNoAuth.status, 401);

      const mcpDeleteNoAuth = await fetch(`${bound.url}/mcp`, { method: "DELETE" });
      assert.equal(mcpDeleteNoAuth.status, 401);

      // /mcp with valid Bearer token succeeds
      const mcpPingAuthed = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: { ...authHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: "p1", method: "ping" }),
      });
      assert.equal(mcpPingAuthed.status, 200);
      assert.deepEqual(await mcpPingAuthed.json(), { jsonrpc: "2.0", id: "p1", result: {} });
    } finally {
      await app.close();
    }
  });

  it("26. completes Streamable HTTP /mcp handshake, tools/call workflow, batch, GET 405, and DELETE 200", async () => {
    const token = "cg_streamable_mcp_token_2026_secure";
    const app = createContextGraphServer({
      dataPath: ":memory:",
      port: 0,
      authToken: token,
    });
    const bound = await app.start(0, "127.0.0.1");
    const baseHeaders = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
    };

    try {
      // 1. POST /mcp initialize -> 200 + Mcp-Session-Id header
      const initRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: baseHeaders,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "initialize",
          params: {
            protocolVersion: "2025-03-26",
            capabilities: {},
            clientInfo: { name: "streamable-http-test", version: "1.0.0" },
          },
        }),
      });
      assert.equal(initRes.status, 200);
      assert.match(initRes.headers.get("content-type") ?? "", /^application\/json\b/i);
      const sessionId = initRes.headers.get("mcp-session-id");
      assert.ok(sessionId && sessionId.length > 0, "Expected Mcp-Session-Id header on initialize");

      const initBody = (await initRes.json()) as {
        jsonrpc: string;
        id: number;
        result: { protocolVersion: string; serverInfo: { name: string } };
      };
      assert.equal(initBody.jsonrpc, "2.0");
      assert.equal(initBody.id, 1);
      assert.equal(initBody.result.protocolVersion, "2025-03-26");
      assert.equal(initBody.result.serverInfo.name, "@opc/context-graph");

      const sessionHeaders = {
        ...baseHeaders,
        "Mcp-Session-Id": sessionId,
      };

      // 2. POST /mcp notifications/initialized -> 202 Accepted with empty body
      const notifRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: JSON.stringify({
          jsonrpc: "2.0",
          method: "notifications/initialized",
        }),
      });
      assert.equal(notifRes.status, 202);
      assert.equal(await notifRes.text(), "");

      // 3. POST /mcp tools/list -> 200 with all 7 tools
      const listRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 2,
          method: "tools/list",
        }),
      });
      assert.equal(listRes.status, 200);
      const listBody = (await listRes.json()) as {
        result: { tools: Array<{ name: string }> };
      };
      assert.equal(listBody.result.tools.length, 7);
      assert.ok(listBody.result.tools.some((t) => t.name === "cg_create_graph"));
      assert.ok(listBody.result.tools.some((t) => t.name === "cg_ingest_turn"));
      assert.ok(listBody.result.tools.some((t) => t.name === "cg_ingest_artifacts"));
      assert.ok(listBody.result.tools.some((t) => t.name === "cg_compile_context"));

      // 4. POST /mcp tools/call -> cg_create_graph
      const createRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 3,
          method: "tools/call",
          params: {
            name: "cg_create_graph",
            arguments: {
              id: "http_mcp_g1",
              name: "Streamable HTTP MCP Graph",
              defaultBudget: 1600,
            },
          },
        }),
      });
      assert.equal(createRes.status, 200);
      const createBody = (await createRes.json()) as {
        result: { isError: boolean; structuredContent: { ok: boolean; graph: { id: string } } };
      };
      assert.equal(createBody.result.isError, false);
      assert.equal(createBody.result.structuredContent.ok, true);
      assert.equal(createBody.result.structuredContent.graph.id, "http_mcp_g1");

      // 5. POST /mcp tools/call -> cg_ingest_turn
      const ingestRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 4,
          method: "tools/call",
          params: {
            name: "cg_ingest_turn",
            arguments: {
              graphId: "http_mcp_g1",
              turn: 1,
              scope: "deploy",
              system: "You are a Tailscale deployment assistant.",
              goal: "Expose Context Graph over Tailscale on Ubuntu1.",
              constraints: ["Must require CG_AUTH_TOKEN on non-loopback hosts."],
              artifacts: [
                {
                  path: "deploy/context-graph.env",
                  content: "HOST=100.108.223.41\nPORT=52341\n",
                },
              ],
            },
          },
        }),
      });
      assert.equal(ingestRes.status, 200);
      const ingestBody = (await ingestRes.json()) as {
        result: { isError: boolean; structuredContent: { addedNodes: unknown[] } };
      };
      assert.equal(ingestBody.result.isError, false);
      assert.ok(ingestBody.result.structuredContent.addedNodes.length >= 4);

      // 6. POST /mcp tools/call -> cg_compile_context (verify non-empty compiled output)
      const compileRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 5,
          method: "tools/call",
          params: {
            name: "cg_compile_context",
            arguments: {
              graphId: "http_mcp_g1",
              focusQuery: "Ubuntu1 Tailscale HOST and CG_AUTH_TOKEN",
              activeScope: "deploy",
            },
          },
        }),
      });
      assert.equal(compileRes.status, 200);
      const compileBody = (await compileRes.json()) as {
        result: {
          isError: boolean;
          structuredContent: {
            ok: boolean;
            promptText: string;
            messages: Array<{ role: string; content: string }>;
            kvCache: { totalTokens: number };
          };
        };
      };
      assert.equal(compileBody.result.isError, false);
      assert.equal(compileBody.result.structuredContent.ok, true);
      assert.ok(compileBody.result.structuredContent.promptText.length > 0);
      assert.ok(compileBody.result.structuredContent.promptText.includes("100.108.223.41"));
      assert.ok(compileBody.result.structuredContent.messages.length >= 2);
      assert.ok(compileBody.result.structuredContent.kvCache.totalTokens > 0);

      // 7. JSON-RPC batch (mixed request + notification -> 200 array; notifications-only -> 202)
      const batchMixedRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: JSON.stringify([
          { jsonrpc: "2.0", id: "b-1", method: "ping" },
          { jsonrpc: "2.0", method: "notifications/cancelled" },
        ]),
      });
      assert.equal(batchMixedRes.status, 200);
      assert.deepEqual(await batchMixedRes.json(), [{ jsonrpc: "2.0", id: "b-1", result: {} }]);

      const batchNotifRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: JSON.stringify([{ jsonrpc: "2.0", method: "notifications/initialized" }]),
      });
      assert.equal(batchNotifRes.status, 202);
      assert.equal(await batchNotifRes.text(), "");

      // Invalid JSON on POST /mcp -> 400 JSON-RPC Parse error
      const badJsonRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers: sessionHeaders,
        body: "{bad-json",
      });
      assert.equal(badJsonRes.status, 400);
      const badJsonBody = (await badJsonRes.json()) as { error: { code: number } };
      assert.equal(badJsonBody.error.code, -32700);

      // Direct unit check on handleMcpJsonRpc export (single & batch)
      const directBatch = handleMcpJsonRpc(app.store, [
        { jsonrpc: "2.0", id: "d1", method: "ping" },
        { jsonrpc: "2.0", method: "notifications/custom" },
      ]);
      assert.deepEqual(directBatch, [{ jsonrpc: "2.0", id: "d1", result: {} }]);

      // 8. GET /mcp -> 405 Method Not Allowed with Allow: POST, DELETE
      const getMcpRes = await fetch(`${bound.url}/mcp`, {
        method: "GET",
        headers: { Authorization: `Bearer ${token}` },
      });
      assert.equal(getMcpRes.status, 405);
      assert.equal(getMcpRes.headers.get("allow"), "POST, DELETE");

      // 9. DELETE /mcp -> 200 OK { ok: true }
      const delMcpRes = await fetch(`${bound.url}/mcp`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
          "Mcp-Session-Id": sessionId,
        },
      });
      assert.equal(delMcpRes.status, 200);
      assert.deepEqual(await delMcpRes.json(), { ok: true });
    } finally {
      await app.close();
    }
  });
});

describe("v0.2 Declaration-Aware Excerpt & App-Frame Command Engine (`src/excerpt.ts`)", () => {
  it("27. sanitizes upstream truncation and pruned markers cleanly while preserving line structure", () => {
    const dirty = [
      "export const RETRY_LIMIT = 5;",
      "[pruned 1234 chars; head/tail shown]",
      "export const SYNC_WINDOW_MS = 2500; …[pruned 88 chars]",
      "const tail = true; [pruned 42 chars] // done",
    ].join("\n");

    const clean = sanitizeNeutralText(dirty);
    assert.ok(!clean.includes("[pruned"));
    assert.ok(!clean.includes("head/tail shown"));
    assert.ok(clean.includes("export const RETRY_LIMIT = 5;"));
    assert.ok(clean.includes("export const SYNC_WINDOW_MS = 2500;"));
    assert.ok(clean.includes("const tail = true; // done"));
  });

  it("28. truncates strictly at word/line/punctuation boundaries without slicing numbers or identifiers", () => {
    // Must NEVER slice `@line 41` into `@line 4`
    const lineMsg = "Failure reported at @line 41 in module dispatcher";
    const cutNum = truncateAtWordBoundary(lineMsg, 31); // "... @line 4" would be 27 + 4 = 31
    assert.ok(cutNum.length <= 31);
    assert.ok(!cutNum.includes("@line 4 "), `Must not slice number 41 -> 4: ${cutNum}`);
    assert.ok(!cutNum.endsWith("@line 4 ..."), `Must not slice number 41 -> 4: ${cutNum}`);

    // Must NEVER slice `SYNC_WINDOW_MS` into `SYNC_WIN`
    const declMsg = "export const SYNC_WINDOW_MS = 3000; export const NEXT = 1;";
    const cutId = truncateAtWordBoundary(declMsg, 26);
    assert.ok(cutId.length <= 26);
    assert.ok(!cutId.includes("SYNC_WIN"), `Must not slice identifier mid-token: ${cutId}`);

    // Must NEVER slice decimal `0.0825` into `0.08` or `0.`
    const decMsg = "ratio = 0.0825 applied to subtotal";
    const cutDec = truncateAtWordBoundary(decMsg, 16);
    assert.ok(cutDec.length <= 16);
    assert.ok(!cutDec.includes("0.08"), `Must not slice decimal literal: ${cutDec}`);
    assert.ok(!cutDec.includes("0. ..."), `Must not break on decimal point: ${cutDec}`);

    // Must NEVER slice path:line `src/router.ts:41:9` mid-line-number
    const frameMsg = "at dispatch (src/router.ts:41:9) during request";
    const cutFrame = truncateAtWordBoundary(frameMsg, 34);
    assert.ok(cutFrame.length <= 34);
    assert.ok(!cutFrame.includes("src/router.ts:4 "), `Must not slice line number: ${cutFrame}`);
  });

  it("29. prioritizes top-level declarations/constants over imports and trailing braces in compactCodeExcerpt", () => {
    const source = [
      "// Router dispatch policy",
      "import { EventEmitter } from 'node:events';",
      "import { readFileSync } from 'node:fs';",
      "import { createHash } from 'node:crypto';",
      "import { join, resolve } from 'node:path';",
      "",
      "export const BACKOFF_BASE_MS = 250;",
      "export const MAX_QUEUE_DEPTH = 128;",
      "export interface RoutePolicy {",
      "  timeoutMs: number;",
      "  strictMode: boolean;",
      "}",
      "export function dispatchRoute(path: string, policy: RoutePolicy): boolean {",
      "  const normalized = path.trim();",
      "  for (let i = 0; i < 10; i++) {",
      "    if (normalized.length > i) {",
      "      // filler body line",
      "    }",
      "  }",
      "  return normalized.length > 0;",
      "}",
      "function trailingHelper() {",
      "  return [",
      "    1,",
      "    2,",
      "  ];",
      "}",
    ].join("\n");

    const excerpt = compactCodeExcerpt(source, 260);
    assert.ok(excerpt.length <= 260, `Expected <= 260 chars, got ${excerpt.length}`);
    assert.ok(excerpt.includes("export const BACKOFF_BASE_MS = 250;"));
    assert.ok(excerpt.includes("export const MAX_QUEUE_DEPTH = 128;"));
    assert.ok(excerpt.includes("export interface RoutePolicy"));
    assert.ok(excerpt.includes("export function dispatchRoute"));
    assert.ok(!excerpt.includes("trailingHelper"), "Should not waste budget on trailing boilerplate");

    // diffText mode strips boilerplate ack lines
    const diffExcerpt = compactCodeExcerpt("Edited src/router.ts", 200, {
      diffText: "Edited src/router.ts\nWrote src/router.ts (512 bytes)\n- export const MAX_QUEUE_DEPTH = 128;\n+ export const MAX_QUEUE_DEPTH = 256;",
    });
    assert.ok(!diffExcerpt.includes("Edited src/router.ts"));
    assert.ok(!diffExcerpt.includes("Wrote src/router.ts"));
    assert.ok(diffExcerpt.includes("- export const MAX_QUEUE_DEPTH = 128;"));
    assert.ok(diffExcerpt.includes("+ export const MAX_QUEUE_DEPTH = 256;"));
  });

  it("30. preserves FAIL header, primary diagnostics, and app source frame while stripping node_modules in compactCommandExcerpt", () => {
    const noisyFailure = [
      "> svc@1.0.0 test",
      "> vitest run",
      "FAIL src/router.test.ts > rejects stale nonce header",
      "  Error [ERR_NONCE_EXPIRED]: request nonce exceeded 300s replay window",
      "    at validateNonce (src/router.ts:41:11)",
      "    at processTicksAndRejections (node:internal/process/task_queues:95:5)",
      "    at runTest (node_modules/vitest/dist/chunk-runner.js:1284:12)",
      "    at async node_modules/tinypool/dist/worker.js:88:19",
      "Test Files  1 failed (1)",
      "Tests  1 failed | 17 passed (18)",
      "Command exited with code 1",
    ].join("\n");

    const compactFail = compactCommandExcerpt(noisyFailure, false, 240);
    assert.ok(compactFail.length <= 240, `Expected <= 240 chars, got ${compactFail.length}`);
    assert.ok(compactFail.includes("FAIL src/router.test.ts"), `Missing FAIL header: ${compactFail}`);
    assert.ok(compactFail.includes("ERR_NONCE_EXPIRED"), `Missing error code: ${compactFail}`);
    assert.ok(compactFail.includes("src/router.ts:41:11"), `Missing app stack frame: ${compactFail}`);
    assert.ok(!compactFail.includes("node_modules/"), "Must strip node_modules stack frames");
    assert.ok(!compactFail.includes("node:internal/"), "Must strip node:internal stack frames");
    assert.ok(!compactFail.includes("Command exited with code 1"), "Should drop redundant exit banner");

    // Tight budget still preserves diagnostic + app source frame
    const tightFail = compactCommandExcerpt(noisyFailure, false, 130);
    assert.ok(tightFail.length <= 130);
    assert.ok(tightFail.includes("ERR_NONCE_EXPIRED"));
    assert.ok(tightFail.includes("src/router.ts:41:11"));

    // Passing command returns concise 1-line summary <= 120 chars
    const passOutput = [
      "> svc@1.0.0 test",
      "> vitest run",
      " ✓ src/router.test.ts (18 tests) 42ms",
      "Tests  18 passed (18)",
    ].join("\n");
    const compactPass = compactCommandExcerpt(passOutput, true, 300);
    assert.ok(compactPass.length <= 120);
    assert.ok(!compactPass.includes("\n"));
    assert.ok(compactPass.includes("18 passed"));
  });
});

describe("v0.2 Tool-Trajectory Classifier, Pruner & Artifact DAG Engine (`src/artifact-engine.ts`)", () => {
  it("31. classifies known tool families strictly, ignores unknown tools with path, and preserves monorepo cd scoping", () => {
    const unknownWithPath = classifyToolEvent({
      callId: "u1",
      tool: "http_fetch",
      args: { path: "/v1/orders", method: "GET" },
      ok: true,
      output: '{"ok":true}',
    });
    assert.equal(unknownWithPath.family, "other");
    assert.equal(unknownWithPath.resourceKey, null);

    // Monorepo cd scoping preserved while no-op cd . && / cd $PWD && stripped
    assert.equal(normalizeCommandKey("cd . && npm test"), "npm test");
    assert.equal(normalizeCommandKey('cd "$PWD" &&  npm   test'), "npm test");
    assert.equal(normalizeCommandKey("cd packages/alpha/ && npm test"), "cd packages/alpha && npm test");
    assert.notEqual(
      normalizeCommandKey("cd packages/alpha && npm test"),
      normalizeCommandKey("cd packages/beta && npm test"),
    );

    const cmdA = classifyToolEvent({
      callId: "cA",
      tool: "bash",
      args: { command: "cd packages/alpha && npm test" },
      ok: true,
      output: "ok",
    });
    const cmdB = classifyToolEvent({
      callId: "cB",
      tool: "bash",
      args: { command: "cd packages/beta && npm test" },
      ok: true,
      output: "ok",
    });
    assert.equal(cmdA.family, "command");
    assert.notEqual(cmdA.resourceKey, cmdB.resourceKey);

    // machineId isolates same path across hosts
    const m1 = classifyToolEvent({
      callId: "m1",
      tool: "read_file",
      args: { path: "src/config.ts" },
      machineId: "host-a",
      ok: true,
      output: "export const A = 1;",
    });
    const m2 = classifyToolEvent({
      callId: "m2",
      tool: "read_file",
      args: { path: "src/config.ts" },
      machineId: "host-b",
      ok: true,
      output: "export const A = 2;",
    });
    assert.equal(m1.resourceKey, "file:host-a:src/config.ts");
    assert.equal(m2.resourceKey, "file:host-b:src/config.ts");
  });

  it("32. enforces all trajectory pruning invariants purely without mutating inputs", () => {
    const events: ToolTrajectoryEvent[] = [
      // Base read -> partial edit -> MUST keep both unpruned until a full_read supersedes them
      { callId: "r1", tool: "read_file", args: { path: "src/worker.ts" }, ok: true, output: "export const W = 1;\n".repeat(15) },
      { callId: "e1", tool: "edit_file", args: { path: "src/worker.ts", old_text: "W = 1", new_text: "W = 2" }, ok: true, output: "Edited src/worker.ts" },
      // Disjoint ranged reads on src/big.ts -> NEVER prune each other
      { callId: "rg1", tool: "read_file", args: { path: "src/big.ts", start_line: 1, end_line: 50 }, ok: true, output: "chunk 1 content ".repeat(10) },
      { callId: "rg2", tool: "read_file", args: { path: "src/big.ts", start_line: 51, end_line: 100 }, ok: true, output: "chunk 2 content ".repeat(10) },
      // Identical range re-read -> prunes ONLY rg1, keeps rg2!
      { callId: "rg1b", tool: "read_file", args: { path: "src/big.ts", start_line: 1, end_line: 50 }, ok: true, output: "chunk 1 refreshed" },
      // Failed edit on src/guard.ts MUST NEVER prune prior valid read_file
      { callId: "g_read", tool: "read_file", args: { path: "src/guard.ts" }, ok: true, output: "export const GUARD_MODE = 'strict';" },
      { callId: "g_err", tool: "edit_file", args: { path: "src/guard.ts", old_text: "missing", new_text: "x" }, ok: false, output: "Error: target not found" },
      // Monorepo commands: alpha re-run prunes alpha#1, does NOT prune beta#1
      { callId: "t_a1", tool: "bash", args: { command: "cd packages/alpha && npm test" }, ok: false, output: "FAIL alpha\n".repeat(12) },
      { callId: "t_b1", tool: "bash", args: { command: "cd packages/beta && npm test" }, ok: true, output: "PASS beta" },
      { callId: "t_a2", tool: "bash", args: { command: "cd packages/alpha && npm test" }, ok: true, output: "PASS alpha" },
    ];

    const snapshotBefore = JSON.stringify(events);
    const res = pruneTrajectoryEvents(events);
    assert.equal(JSON.stringify(events), snapshotBefore, "Input array must remain unmutated");

    const prunedSet = new Set(res.prunedCallIds);
    assert.ok(!prunedSet.has("r1"), "partial_edit must NEVER prune preceding read_file");
    assert.ok(!prunedSet.has("e1"));
    assert.ok(prunedSet.has("rg1"), "identical ranged_read must prune older identical range");
    assert.ok(!prunedSet.has("rg2"), "disjoint ranged_read must NEVER be pruned");
    assert.ok(!prunedSet.has("g_read"), "failed file op must NEVER prune valid read_file");
    assert.ok(!prunedSet.has("g_err"));
    assert.ok(prunedSet.has("t_a1"), "newer identical command must prune older run");
    assert.ok(!prunedSet.has("t_b1"), "different monorepo cd command must stay isolated");
    assert.ok(res.charsSaved > 0);
    assert.ok(res.tokensSaved > 0);

    const prunedRg1 = res.events.find((e) => e.callId === "rg1")!;
    assert.equal(prunedRg1.pruned, true);
    assert.match(prunedRg1.output, /^\[superseded: newer ranged_read on file:src\/big\.ts#range=1-50\]$/);

    // Adding a full_read on src/worker.ts supersedes BOTH r1 and e1
    const withFull = pruneTrajectoryEvents([
      ...events,
      { callId: "r2", tool: "read_file", args: { path: "src/worker.ts" }, ok: true, output: "export const W = 2;" },
    ]);
    assert.ok(withFull.prunedCallIds.includes("r1"));
    assert.ok(withFull.prunedCallIds.includes("e1"));
  });

  it("33. ingests tool artifacts idempotently across 2-round multi-compaction with cross-compaction cleanup", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ id: "g_art", name: "Artifact Engine Test" });

    // Round 1: read -> partial_edit + disjoint ranged reads + failed edit on another file + failing test
    const round1: ToolTrajectoryEvent[] = [
      {
        callId: "c1",
        tool: "read_file",
        args: { path: "src/worker.ts" },
        ok: true,
        output: "export const POLL_MS = 1000;\nexport function tick() { return POLL_MS; }",
      },
      {
        callId: "c2",
        tool: "edit_file",
        args: {
          path: "src/worker.ts",
          old_text: "export const POLL_MS = 1000;",
          new_text: "export const POLL_MS = 2500;",
        },
        ok: true,
        output: "Edited src/worker.ts",
      },
      {
        callId: "c3",
        tool: "read_file",
        args: { path: "src/chunks.ts", start_line: 1, end_line: 40 },
        ok: true,
        output: "export const CHUNK_A = 1;",
      },
      {
        callId: "c4",
        tool: "read_file",
        args: { path: "src/chunks.ts", start_line: 41, end_line: 80 },
        ok: true,
        output: "export const CHUNK_B = 2;",
      },
      {
        callId: "c5",
        tool: "read_file",
        args: { path: "src/guard.ts" },
        ok: true,
        output: "export const STRICT_GUARD = true;",
      },
      {
        callId: "c6",
        tool: "edit_file",
        args: { path: "src/guard.ts", old_text: "nope", new_text: "yes" },
        ok: false,
        output: "Error: old_text not found in src/guard.ts",
      },
      {
        callId: "c7",
        tool: "bash",
        args: { command: "npm test" },
        ok: false,
        output: "FAIL src/worker.test.ts\nError [ERR_POLL_DRIFT]: expected 2500\n  at tick (src/worker.ts:2:10)",
      },
    ];

    const r1 = ingestToolArtifacts(store, g.id, round1, { turn: 1 });
    assert.equal(r1.ingestedCount, 7);
    assert.equal(r1.unresolvedFailureNodeIds.length, 1);

    // Base read_file + #edit delta are BOTH active for src/worker.ts
    const activeWorkerR1 = store
      .listNodes(g.id, { status: "active" })
      .filter((n) => n.metadata.fileScopeKey === "src/worker.ts");
    assert.equal(activeWorkerR1.length, 2);
    assert.ok(activeWorkerR1.some((n) => n.resourceKey === "file:src/worker.ts"));
    assert.ok(activeWorkerR1.some((n) => n.resourceKey === "file:src/worker.ts#edit"));

    // Failed edit stored under file_err:src/guard.ts and does NOT supersede file:src/guard.ts
    const activeGuardR1 = store
      .listNodes(g.id, { status: "active" })
      .filter((n) => n.metadata.fileScopeKey === "src/guard.ts");
    assert.equal(activeGuardR1.length, 2);
    assert.ok(activeGuardR1.some((n) => n.resourceKey === "file:src/guard.ts"));
    assert.ok(activeGuardR1.some((n) => n.resourceKey === "file_err:src/guard.ts"));

    // Shifted overlapping retry window MUST be 100% idempotent (c2..c7 skipped, zero errors)
    const retryOverlap = ingestToolArtifacts(store, g.id, round1.slice(2), { turn: 1 });
    assert.equal(retryOverlap.ingestedCount, 0);
    assert.equal(retryOverlap.skippedCount, 5);

    // Round 2 (multi-compaction): verification full_read of src/worker.ts + full_write of src/chunks.ts + passing npm test
    const round2: ToolTrajectoryEvent[] = [
      {
        callId: "c8",
        tool: "read_file",
        args: { path: "src/worker.ts" },
        ok: true,
        output: "export const POLL_MS = 2500;\nexport function tick() { return POLL_MS; }",
      },
      {
        callId: "c9",
        tool: "write_file",
        args: {
          path: "src/chunks.ts",
          content: "export const CHUNK_A = 10;\nexport const CHUNK_B = 20;",
        },
        ok: true,
        output: "Wrote src/chunks.ts",
      },
      {
        callId: "c10",
        tool: "bash",
        args: { command: "npm test" },
        ok: true,
        output: "Tests  12 passed (12)",
      },
    ];

    const r2 = ingestToolArtifacts(store, g.id, round2, { turn: 2 });
    assert.equal(r2.ingestedCount, 3);
    assert.equal(r2.unresolvedFailureNodeIds.length, 0, "Passing test must supersede failing test");

    // Verification read of src/worker.ts MUST supersede BOTH old base AND Round 1 #edit in SQLite,
    // while retaining salience >= 0.90 and editedScope: true!
    const workerNodes = store
      .listNodes(g.id)
      .filter((n) => n.metadata.fileScopeKey === "src/worker.ts");
    const activeWorkerR2 = workerNodes.filter((n) => n.status === "active");
    assert.equal(activeWorkerR2.length, 1);
    assert.equal(activeWorkerR2[0]?.resourceKey, "file:src/worker.ts");
    assert.ok((activeWorkerR2[0]?.salience ?? 0) >= 0.9, `Expected salience >= 0.90, got ${activeWorkerR2[0]?.salience}`);
    assert.equal(activeWorkerR2[0]?.metadata.editedScope, true);

    const supersededWorkerEdit = workerNodes.find((n) => n.resourceKey === "file:src/worker.ts#edit")!;
    assert.equal(supersededWorkerEdit.status, "superseded");
    assert.equal(supersededWorkerEdit.supersededBy, activeWorkerR2[0]?.id);

    // Full write of src/chunks.ts MUST supersede BOTH disjoint #range=... nodes
    const activeChunks = store
      .listNodes(g.id, { status: "active" })
      .filter((n) => n.metadata.fileScopeKey === "src/chunks.ts");
    assert.equal(activeChunks.length, 1);
    assert.equal(activeChunks[0]?.resourceKey, "file:src/chunks.ts");
    assert.ok(activeChunks[0]?.content.includes("CHUNK_A = 10"));

    store.close();
  });
});

describe("v0.2 Compact Markdown Compiler & Priority Budget Engine", () => {
  it("34. compiles compact_markdown strictly within maxOutputChars with 3-tier priority and zero XML/pruned noise", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ id: "g_compact", name: "Compact Markdown Test" });

    ingestToolArtifacts(store, g.id, [
      // 3 unmodified read-only background noise files
      {
        callId: "bg1",
        tool: "read_file",
        args: { path: "src/docs-index.ts" },
        ok: true,
        output: "export const DOC_SECTION_1 = 'overview';\n".repeat(12),
      },
      {
        callId: "bg2",
        tool: "read_file",
        args: { path: "src/formatting.ts" },
        ok: true,
        output: "export function padLeft(s: string) { return s.padStart(4); }\n".repeat(12),
      },
      {
        callId: "bg3",
        tool: "read_file",
        args: { path: "src/cli-banner.ts" },
        ok: true,
        output: "export const BANNER_TITLE = 'console';\n".repeat(12),
      },
      // Active working edited file (base + partial edit)
      {
        callId: "wk_r",
        tool: "read_file",
        args: { path: "src/dispatcher.ts" },
        ok: true,
        output: [
          "import { EventEmitter } from 'node:events';",
          "export const RETRY_DELAY_MS = 450;",
          "export const MAX_INFLIGHT = 32;",
          "export function dispatchBatch(items: string[]): number {",
          "  return items.length;",
          "}",
          "[pruned 999 chars; head/tail shown]",
        ].join("\n"),
      },
      {
        callId: "wk_e",
        tool: "edit_file",
        args: {
          path: "src/dispatcher.ts",
          old_text: "export const MAX_INFLIGHT = 32;",
          new_text: "export const MAX_INFLIGHT = 64;",
        },
        ok: true,
        output: "Edited src/dispatcher.ts",
      },
      // Active failing check with noisy node_modules frames
      {
        callId: "chk_f",
        tool: "bash",
        args: { command: "npm test" },
        ok: false,
        output: [
          "FAIL src/dispatcher.test.ts > caps inflight batch",
          "  Error [ERR_BATCH_OVERFLOW]: received 64 expected <= 32",
          "    at dispatchBatch (src/dispatcher.ts:41:7)",
          "    at runTask (node_modules/vitest/dist/runner.js:901:14)",
          "    at processTicksAndRejections (node:internal/process/task_queues:95:5)",
          "Tests  1 failed | 17 passed (18)",
          "Command exited with code 1",
        ].join("\n"),
      },
    ]);

    const compiled = compileContext(store, {
      graphId: g.id,
      format: "compact_markdown",
      maxOutputChars: 950,
    });

    const out = compiled.promptText;
    assert.ok(out.length > 0 && out.length <= 950, `Expected 1..950 chars, got ${out.length}`);
    assert.ok(!out.includes("<node "), "compact_markdown must emit zero XML tags");
    assert.ok(!out.includes("<zone_1_"), "compact_markdown must emit zero XML tags");
    assert.ok(!out.includes("[pruned"), "Must contain zero [pruned markers");
    assert.ok(!out.includes("head/tail shown"), "Must contain zero head/tail markers");
    assert.ok(!/\(v\d+,\s*read_file,\s*ok\)/i.test(out), "Must contain zero (vN, read_file, ok) phrasing");

    // Tier 1 failing check preserves error code & app source frame without node_modules
    assert.ok(out.includes("[check failed]"), `Expected [check failed] tier: ${out}`);
    assert.ok(out.includes("ERR_BATCH_OVERFLOW"));
    assert.ok(out.includes("src/dispatcher.ts:41:7"));
    assert.ok(!out.includes("node_modules/"));

    // Tier 2 preserves base snapshot BEFORE #edit delta, with full exported constants intact
    const baseIdx = out.indexOf("`src/dispatcher.ts`:");
    const editIdx = out.indexOf("[file edit] `src/dispatcher.ts`:");
    assert.ok(baseIdx >= 0 && editIdx > baseIdx, `Base snapshot must precede #edit delta:\n${out}`);
    assert.ok(out.includes("RETRY_DELAY_MS = 450"));
    assert.ok(out.includes("MAX_INFLIGHT = 64"));

    // Tier 3 collapses unmodified background files into a single compact line
    assert.ok(out.includes("- [other read files]:"), `Expected collapsed other read files:\n${out}`);
    assert.ok(out.includes("`src/docs-index.ts`"));
    assert.ok(!out.includes("DOC_SECTION_1"), "Unmodified background file body must not bloat output");

    // XML mode also benefits from unresolvedFailure priority utility boost
    const xmlCompiled = compileContext(store, {
      graphId: g.id,
      format: "xml",
      maxTokens: 320,
      recordCacheState: false,
    });
    const failEntry = xmlCompiled.zone2Nodes.find((n) => n.resourceKey === "cmd:npm test");
    assert.ok(failEntry, "Failing check must be selected in XML mode too");
    assert.equal(failEntry?.lod, "full", "Failing check must reach full LOD instead of stub starvation");

    store.close();
  });

  it("35. elevates failed-edit target files to Tier 2, supports read-only fallback, and preserves Zone 1 KV prefix", () => {
    // Part A: Failed edit elevates target file scope to Tier 2 while collapsing unrelated reads
    const storeA = new GraphStore(":memory:");
    const gA = storeA.createGraph({ id: "g_failed_edit", name: "Failed Edit Target Elevation" });
    ingestToolArtifacts(storeA, gA.id, [
      {
        callId: "r_target",
        tool: "read_file",
        args: { path: "src/throttle.ts" },
        ok: true,
        output: "export const WINDOW_SECONDS = 60;\nexport const MAX_REQUESTS = 120;\nexport function allowRequest() { return true; }",
      },
      {
        callId: "e_fail",
        tool: "edit_file",
        args: { path: "src/throttle.ts", old_text: "nonexistent", new_text: "x" },
        ok: false,
        output: "Error: old_text did not match any lines in src/throttle.ts",
      },
      {
        callId: "r_bg1",
        tool: "read_file",
        args: { path: "src/readme-note.ts" },
        ok: true,
        output: "export const NOTE_ALPHA = 'background noise';\n".repeat(10),
      },
      {
        callId: "r_bg2",
        tool: "read_file",
        args: { path: "src/metrics-stub.ts" },
        ok: true,
        output: "export const METRIC_BETA = 'background noise';\n".repeat(10),
      },
    ]);

    const resA = compileContext(storeA, {
      graphId: gA.id,
      format: "compact_markdown",
      maxOutputChars: 950,
    });
    assert.ok(resA.promptText.length <= 950);
    assert.ok(resA.promptText.includes("WINDOW_SECONDS = 60"), "Failed-edit target must get full declaration excerpt");
    assert.ok(resA.promptText.includes("MAX_REQUESTS = 120"));
    assert.ok(resA.promptText.includes("- [other read files]:"));
    assert.ok(!resA.promptText.includes("NOTE_ALPHA"), "Background noise must stay collapsed");
    storeA.close();

    // Part B: Pure read-only trajectory (no Tier 1/Tier 2) gives full excerpts to read files + Zone 1 prefix caching
    const storeB = new GraphStore(":memory:");
    const gB = storeB.createGraph({ id: "g_readonly", name: "Read-Only Fallback" });
    storeB.addNode(gB.id, {
      id: "c_pin",
      kind: "constraint",
      tier: "pinned_prefix",
      content: "Never disable TLS verification.",
    });
    ingestToolArtifacts(storeB, gB.id, [
      {
        callId: "ro1",
        tool: "read_file",
        args: { path: "src/settings.ts" },
        ok: true,
        output: "export const LISTEN_PORT = 8443;\nexport const TLS_REQUIRED = true;",
      },
    ]);

    const c1 = compileContext(storeB, {
      graphId: gB.id,
      format: "compact_markdown",
      maxOutputChars: 950,
    });
    assert.ok(c1.promptText.includes("Never disable TLS verification."));
    assert.ok(c1.promptText.includes("LISTEN_PORT = 8443"));
    assert.ok(c1.promptText.includes("TLS_REQUIRED = true"));
    assert.ok(!c1.promptText.includes("[other read files]"), "Read-only trajectory should show full excerpts");

    // Mutate Zone 2 -> Zone 1 prefix still hits KV cache!
    ingestToolArtifacts(storeB, gB.id, [
      {
        callId: "ro2",
        tool: "bash",
        args: { command: "npm run lint" },
        ok: true,
        output: "0 errors",
      },
    ]);
    const c2 = compileContext(storeB, {
      graphId: gB.id,
      format: "compact_markdown",
      maxOutputChars: 950,
    });
    assert.ok(c2.kvCache.prefixCachedTokens > 0, "Zone 1 prefix should be cached in compact_markdown mode");
    storeB.close();
  });
});

describe("v0.2 HTTP /ingest-artifacts, Compile Formats & MCP Tool Annotations E2E", () => {
  it("36. serves Bearer-guarded POST /v1/graphs/:id/ingest-artifacts and compact_markdown POST /compile", async () => {
    const token = "cg_v2_artifact_engine_bearer_secret";
    const app = createContextGraphServer({ dataPath: ":memory:", port: 0, authToken: token });
    const bound = await app.start(0, "127.0.0.1");
    const authHeaders = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    };

    try {
      // Create graph
      const createRes = await fetch(`${bound.url}/v1/graphs`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({ id: "http_v2_g1", name: "v0.2 HTTP Artifact Graph" }),
      });
      assert.equal(createRes.status, 201);

      const sampleEvents: ToolTrajectoryEvent[] = [
        {
          callId: "ev1",
          tool: "read_file",
          args: { path: "src/queue.ts" },
          ok: true,
          output: "export const QUEUE_CAP = 50;\nexport function enqueue(x: string) { return x; }",
        },
        {
          callId: "ev2",
          tool: "edit_file",
          args: {
            path: "src/queue.ts",
            old_text: "export const QUEUE_CAP = 50;",
            new_text: "export const QUEUE_CAP = 100;",
          },
          ok: true,
          output: "Edited src/queue.ts",
        },
        {
          callId: "ev3",
          tool: "bash",
          args: { command: "npm test" },
          ok: false,
          output: "FAIL src/queue.test.ts\nError [ERR_QUEUE_FULL]: limit 100\n  at enqueue (src/queue.ts:2:12)",
        },
      ];

      // 401 without Bearer auth
      const unauthRes = await fetch(`${bound.url}/v1/graphs/http_v2_g1/ingest-artifacts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events: sampleEvents }),
      });
      assert.equal(unauthRes.status, 401);

      // 400 when events is missing
      const invalidRes = await fetch(`${bound.url}/v1/graphs/http_v2_g1/ingest-artifacts`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({ prune: true }),
      });
      assert.equal(invalidRes.status, 400);

      // 200 with valid Bearer auth + prune + compile
      const okRes = await fetch(`${bound.url}/v1/graphs/http_v2_g1/ingest-artifacts`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({
          events: sampleEvents,
          prune: true,
          compile: true,
          compileOptions: { format: "compact_markdown", maxOutputChars: 800 },
        }),
      });
      assert.equal(okRes.status, 200);
      const body = (await okRes.json()) as {
        ok: boolean;
        graphId: string;
        ingested: { ingestedCount: number; unresolvedFailureNodeIds: string[] };
        pruned: { prunedCount: number; events: unknown[] };
        compiled: { promptText: string };
      };
      assert.equal(body.ok, true);
      assert.equal(body.graphId, "http_v2_g1");
      assert.equal(body.ingested.ingestedCount, 3);
      assert.equal(body.ingested.unresolvedFailureNodeIds.length, 1);
      assert.equal(body.pruned.events.length, 3);
      assert.ok(body.compiled.promptText.length <= 800);
      assert.ok(body.compiled.promptText.includes("ERR_QUEUE_FULL"));
      assert.ok(body.compiled.promptText.includes("src/queue.ts:2:12"));
      assert.ok(body.compiled.promptText.includes("QUEUE_CAP = 100"));

      // Direct POST /v1/graphs/:id/compile supports format & maxOutputChars
      const compileRes = await fetch(`${bound.url}/v1/graphs/http_v2_g1/compile`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({ format: "compact_markdown", maxOutputChars: 500 }),
      });
      assert.equal(compileRes.status, 200);
      const compBody = (await compileRes.json()) as { ok: boolean; promptText: string };
      assert.equal(compBody.ok, true);
      assert.ok(compBody.promptText.length <= 500);
      assert.ok(!compBody.promptText.includes("<node "));
    } finally {
      await app.close();
    }
  });

  it("37. exposes cg_ingest_artifacts and annotations.readOnlyHint on every tool over /mcp", async () => {
    const token = "cg_v2_mcp_annotations_secret_token";
    const app = createContextGraphServer({ dataPath: ":memory:", port: 0, authToken: token });
    const bound = await app.start(0, "127.0.0.1");
    const headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    };

    try {
      const listRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers,
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list" }),
      });
      assert.equal(listRes.status, 200);
      const listBody = (await listRes.json()) as {
        result: {
          tools: Array<{ name: string; annotations?: { readOnlyHint?: boolean } }>;
        };
      };

      const tools = listBody.result.tools;
      assert.equal(tools.length, 7);
      const byName = new Map(tools.map((t) => [t.name, t]));
      assert.ok(byName.has("cg_ingest_artifacts"));

      // Every tool must declare annotations.readOnlyHint accurately
      assert.equal(byName.get("cg_create_graph")?.annotations?.readOnlyHint, false);
      assert.equal(byName.get("cg_add_node")?.annotations?.readOnlyHint, false);
      assert.equal(byName.get("cg_link_nodes")?.annotations?.readOnlyHint, false);
      assert.equal(byName.get("cg_ingest_turn")?.annotations?.readOnlyHint, false);
      assert.equal(byName.get("cg_ingest_artifacts")?.annotations?.readOnlyHint, false);
      assert.equal(byName.get("cg_compile_context")?.annotations?.readOnlyHint, true);
      assert.equal(byName.get("cg_export_mermaid")?.annotations?.readOnlyHint, true);

      // End-to-end tools/call workflow with cg_ingest_artifacts + compact_markdown compile
      const createRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 2,
          method: "tools/call",
          params: { name: "cg_create_graph", arguments: { id: "mcp_v2_g1", name: "MCP v0.2" } },
        }),
      });
      assert.equal(createRes.status, 200);

      const ingestRes = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 3,
          method: "tools/call",
          params: {
            name: "cg_ingest_artifacts",
            arguments: {
              graphId: "mcp_v2_g1",
              prune: true,
              compile: true,
              compileOptions: { format: "compact_markdown", maxOutputChars: 700 },
              events: [
                {
                  callId: "m_r1",
                  tool: "read_file",
                  args: { path: "src/codec.ts" },
                  ok: true,
                  output: "export const CODEC_VERSION = 2;\nexport function encodeFrame(b: Uint8Array) { return b; }",
                },
                {
                  callId: "m_t1",
                  tool: "bash",
                  args: { command: "npm test" },
                  ok: false,
                  output: "FAIL src/codec.test.ts\nError [ERR_FRAME_CRC]: checksum mismatch\n  at encodeFrame (src/codec.ts:2:15)",
                },
              ],
            },
          },
        }),
      });
      assert.equal(ingestRes.status, 200);
      const ingestBody = (await ingestRes.json()) as {
        result: {
          isError: boolean;
          structuredContent: {
            ok: boolean;
            ingested: { ingestedCount: number };
            compiled: { promptText: string };
          };
        };
      };
      assert.equal(ingestBody.result.isError, false);
      assert.equal(ingestBody.result.structuredContent.ok, true);
      assert.equal(ingestBody.result.structuredContent.ingested.ingestedCount, 2);
      assert.ok(ingestBody.result.structuredContent.compiled.promptText.length <= 700);
      assert.ok(ingestBody.result.structuredContent.compiled.promptText.includes("ERR_FRAME_CRC"));
      assert.ok(ingestBody.result.structuredContent.compiled.promptText.includes("src/codec.ts:2:15"));
    } finally {
      await app.close();
    }
  });
});

describe("v0.2 Review Hardening Regression Suite (B1-B4 & NB1-NB3)", () => {
  it("38. [B1] executes ReDoS-adversarial inputs in strictly linear O(N) time (< 100ms)", () => {
    const spaces = " ".repeat(20_000) + "x";
    const brackets = "[".repeat(20_000);
    const parens = "(".repeat(20_000);
    const repeatedPaths = "src/".repeat(5_000);
    const inputs = [spaces, brackets, parens, repeatedPaths];

    const store = new GraphStore(":memory:");
    const g = store.createGraph({ id: "g_redos", name: "ReDoS Regression" });
    ingestToolArtifacts(store, g.id, [
      {
        callId: "rd1",
        tool: "read_file",
        args: { path: "src/spaces.ts" },
        ok: true,
        output: spaces,
      },
      {
        callId: "rd2",
        tool: "read_file",
        args: { path: "src/brackets.ts" },
        ok: true,
        output: brackets,
      },
      {
        callId: "rd3",
        tool: "bash",
        args: { command: "npm test" },
        ok: false,
        output: `${parens}\n${repeatedPaths}`,
      },
    ]);

    const t0 = Date.now();
    for (const s of inputs) {
      sanitizeNeutralText(s);
      compactCommandExcerpt(s, false, 300);
    }
    const compiled = compileContext(store, {
      graphId: g.id,
      format: "compact_markdown",
      maxOutputChars: 950,
    });
    const elapsedMs = Date.now() - t0;

    assert.ok(compiled.promptText.length <= 950);
    assert.ok(
      elapsedMs < 100,
      `Expected strictly linear O(N) execution < 100ms, got ${elapsedMs}ms`,
    );
    store.close();
  });

  it("39. [B2 & NB3] rejects malformed HTTP / MCP payloads with 400 VALIDATION_ERROR and forwards MCP excerpt limits", async () => {
    const token = "cg_v2_validation_hardening_secret_key";
    const app = createContextGraphServer({ dataPath: ":memory:", port: 0, authToken: token });
    const bound = await app.start(0, "127.0.0.1");
    const headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    };

    try {
      await fetch(`${bound.url}/v1/graphs`, {
        method: "POST",
        headers,
        body: JSON.stringify({ id: "g_val", name: "Validation Graph" }),
      });

      const assert400 = async (path: string, payload: unknown, label: string) => {
        const res = await fetch(`${bound.url}${path}`, {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
        });
        assert.equal(res.status, 400, `${label}: expected HTTP 400, got ${res.status}`);
        const body = (await res.json()) as { ok: boolean; error: { code: string } };
        assert.equal(body.ok, false);
        assert.equal(body.error.code, "VALIDATION_ERROR", `${label}: expected VALIDATION_ERROR`);
      };

      // Malformed events payloads -> 400 VALIDATION_ERROR (never 500 or 200)
      await assert400("/v1/graphs/g_val/ingest-artifacts", { events: [null] }, "events:[null]");
      await assert400(
        "/v1/graphs/g_val/ingest-artifacts",
        { events: [{ callId: "1", tool: 123, ok: true, output: "x" }] },
        "non-string tool",
      );
      await assert400(
        "/v1/graphs/g_val/ingest-artifacts",
        { events: [{ callId: "1", tool: "read_file", ok: true, output: { bad: 1 } }] },
        "non-string output",
      );
      await assert400(
        "/v1/graphs/g_val/ingest-artifacts",
        { events: [{ callId: "", tool: "read_file", ok: true, output: "x" }] },
        "empty callId",
      );
      await assert400(
        "/v1/graphs/g_val/ingest-artifacts",
        { events: [{ callId: "1", tool: "read_file", ok: true, output: "x", machineId: 42 }] },
        "non-string machineId",
      );
      await assert400(
        "/v1/graphs/g_val/ingest-artifacts",
        { events: [{ callId: "1", tool: "read_file", ok: "yes", output: "x" }] },
        "non-boolean ok",
      );
      const tooMany = Array.from({ length: 2001 }, (_, i) => ({
        callId: `c${i}`,
        tool: "read_file",
        args: { path: "src/a.ts" },
        ok: true,
        output: "x",
      }));
      await assert400("/v1/graphs/g_val/ingest-artifacts", { events: tooMany }, ">2000 events");

      // Invalid compileOptions on /ingest-artifacts fails BEFORE mutating graph
      await assert400(
        "/v1/graphs/g_val/ingest-artifacts",
        {
          events: [{ callId: "atomic_1", tool: "read_file", args: { path: "src/a.ts" }, ok: true, output: "const A = 1;" }],
          compile: true,
          compileOptions: { format: "bogus_format" },
        },
        "bogus compileOptions.format",
      );
      assert.equal(app.store.listNodes("g_val").length, 0, "Invalid compileOptions must reject atomically before ingestion");

      // Malformed /compile payloads -> 400 VALIDATION_ERROR
      await assert400("/v1/graphs/g_val/compile", { format: "yaml" }, "bogus format");
      await assert400("/v1/graphs/g_val/compile", { maxOutputChars: -5 }, "negative maxOutputChars");
      await assert400("/v1/graphs/g_val/compile", { maxOutputChars: 0 }, "zero maxOutputChars");
      await assert400("/v1/graphs/g_val/compile", { maxOutputChars: 4001 }, "oversized maxOutputChars");
      await assert400("/v1/graphs/g_val/compile", { maxOutputChars: "500" }, "string maxOutputChars");
      await assert400("/v1/graphs/g_val/compile", { focusQuery: 999 }, "non-string focusQuery");
      await assert400("/v1/graphs/g_val/compile", { pprIterations: 0 }, "pprIterations=0");
      await assert400("/v1/graphs/g_val/compile", { pprIterations: 101 }, "pprIterations=101");
      await assert400("/v1/graphs/g_val/compile", { pprAlpha: 1.5 }, "pprAlpha>1");
      await assert400("/v1/graphs/g_val/compile", { focusNodeIds: [123] }, "non-string focusNodeIds");

      // MCP validation errors surface cleanly as isError: true with VALIDATION_ERROR,
      // and MCP cg_ingest_artifacts forwards maxCodeExcerptChars / maxCommandExcerptChars
      const mcpBad = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 10,
          method: "tools/call",
          params: {
            name: "cg_ingest_artifacts",
            arguments: { graphId: "g_val", events: [null] },
          },
        }),
      });
      const mcpBadBody = (await mcpBad.json()) as {
        result: { isError: boolean; content: Array<{ text: string }> };
      };
      assert.equal(mcpBadBody.result.isError, true);
      assert.ok(mcpBadBody.result.content[0]?.text.includes("VALIDATION_ERROR"));

      const mcpCaps = await fetch(`${bound.url}/mcp`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 11,
          method: "tools/call",
          params: {
            name: "cg_ingest_artifacts",
            arguments: {
              graphId: "g_val",
              maxCodeExcerptChars: 70,
              maxCommandExcerptChars: 80,
              events: [
                {
                  callId: "mc1",
                  tool: "read_file",
                  args: { path: "src/long.ts" },
                  ok: true,
                  output: "export const ALPHA = 1;\nexport const BETA = 2;\nexport const GAMMA = 3;\nexport const DELTA = 4;",
                },
                {
                  callId: "mc2",
                  tool: "bash",
                  args: { command: "npm test" },
                  ok: false,
                  output: "FAIL src/long.test.ts\nError [ERR_DEMO_FAIL]: something failed in long test suite\n  at run (src/long.ts:1:1)",
                },
              ],
            },
          },
        }),
      });
      const mcpCapsBody = (await mcpCaps.json()) as { result: { isError: boolean } };
      assert.equal(mcpCapsBody.result.isError, false);
      const storedNodes = app.store.listNodes("g_val");
      const codeNode = storedNodes.find((n) => n.resourceKey === "file:src/long.ts")!;
      const cmdNode = storedNodes.find((n) => n.resourceKey === "cmd:npm test")!;
      assert.ok(codeNode.content.length <= 70, `Expected <= 70, got ${codeNode.content.length}`);
      assert.ok(cmdNode.content.length <= 80, `Expected <= 80, got ${cmdNode.content.length}`);
    } finally {
      await app.close();
    }
  });

  it("40. [B3, NB2 & NB3] strictly enforces small maxOutputChars (20, 50, 79, 80), guards non-finite caps, and syncs messages", () => {
    // NB2: non-finite & float guards
    assert.equal(truncateAtWordBoundary("hello world", Number.NaN), "");
    assert.equal(truncateAtWordBoundary("hello world", Number.POSITIVE_INFINITY), "");
    assert.equal(truncateAtWordBoundary("hello world", -10), "");
    const floatCut = truncateAtWordBoundary("alpha beta gamma", 9.9);
    assert.ok(floatCut.length <= 9);

    const store = new GraphStore(":memory:");
    const g = store.createGraph({ id: "g_tiny_cap", name: "Tiny maxOutputChars" });
    store.addNode(g.id, {
      id: "c_pin",
      kind: "constraint",
      tier: "pinned_prefix",
      content: "Keep output strictly bounded at every character cap.",
    });
    ingestToolArtifacts(store, g.id, [
      {
        callId: "t1",
        tool: "read_file",
        args: { path: "src/service.ts" },
        ok: true,
        output: "export const SERVICE_PORT = 9090;\nexport function startService() { return SERVICE_PORT; }",
      },
      {
        callId: "t2",
        tool: "bash",
        args: { command: "npm test" },
        ok: false,
        output: "FAIL src/service.test.ts\nError [ERR_BIND_FAIL]: port busy\n  at startService (src/service.ts:2:10)",
      },
    ]);

    for (const cap of [20, 50, 79, 80]) {
      const res = compileContext(store, {
        graphId: g.id,
        format: "compact_markdown",
        maxOutputChars: cap,
        recordCacheState: false,
      });
      assert.ok(
        res.promptText.length <= cap,
        `maxOutputChars=${cap}: expected promptText.length <= ${cap}, got ${res.promptText.length} (${res.promptText})`,
      );
      const msgJoined = res.messages.map((m) => m.content).join("\n");
      assert.ok(
        msgJoined.length <= cap,
        `maxOutputChars=${cap}: messages must stay <= ${cap}, got ${msgJoined.length}`,
      );
      assert.equal(msgJoined, res.promptText, `maxOutputChars=${cap}: messages must stay synced with promptText`);
    }
    store.close();
  });

  it("41. [B4] eliminates Tier 1 & Tier 2 double-squeezing so full FAIL header and indented file bodies stay intact", () => {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ id: "g_nosqueeze", name: "No Double Squeeze" });

    const singleFailure =
      "FAIL src/auth.test.ts\nError [ERR_TOKEN_INVALID]: bad signature\n  at verify (src/auth.ts:42:9)";
    assert.equal(singleFailure.length, 93);

    const indentedFile = [
      "export function buildPolicy() {",
      "    const retries = 3;",
      "    const timeoutMs = 1500;",
      "    const strict = true;",
      "    return { retries, timeoutMs, strict };",
      "}",
    ].join("\n");
    assert.ok(indentedFile.length >= 140 && indentedFile.length <= 170);

    ingestToolArtifacts(store, g.id, [
      {
        callId: "ns_r",
        tool: "read_file",
        args: { path: "src/policy.ts" },
        ok: true,
        output: indentedFile,
      },
      {
        callId: "ns_w",
        tool: "write_file",
        args: { path: "src/policy.ts", content: indentedFile },
        ok: true,
        output: "Wrote src/policy.ts",
      },
      {
        callId: "ns_f",
        tool: "bash",
        args: { command: "npm test" },
        ok: false,
        output: singleFailure,
      },
    ]);

    const compiled = compileContext(store, {
      graphId: g.id,
      format: "compact_markdown",
      maxOutputChars: 950,
    });

    // Tier 1: must preserve FAIL header + error diagnostic + app stack frame completely
    assert.ok(
      compiled.promptText.includes("FAIL src/auth.test.ts"),
      `Tier 1 must preserve FAIL header without double-squeezing:\n${compiled.promptText}`,
    );
    assert.ok(compiled.promptText.includes("Error [ERR_TOKEN_INVALID]: bad signature"));
    assert.ok(compiled.promptText.includes("at verify (src/auth.ts:42:9)"));

    // Tier 2: indented 150-char file must preserve entire body including trailing return statement
    assert.ok(compiled.promptText.includes("const retries = 3;"));
    assert.ok(compiled.promptText.includes("const timeoutMs = 1500;"));
    assert.ok(
      compiled.promptText.includes("return { retries, timeoutMs, strict };"),
      `Tier 2 must preserve trailing lines of indented file:\n${compiled.promptText}`,
    );
    store.close();
  });

  it("42. [NB1] keeps test filenames containing 'error' in headers so real Error diagnostics are never displaced", () => {
    const output = [
      "FAIL src/error-handler.test.ts > retries transient gateway failure",
      "  Error [ERR_UPSTREAM_TIMEOUT]: upstream server timed out after 5000ms",
      "    at callUpstream (src/client.ts:88:11)",
      "    at runTask (node_modules/vitest/dist/runner.js:120:4)",
    ].join("\n");

    const tight = compactCommandExcerpt(output, false, 130);
    assert.ok(tight.length <= 130, `Expected <= 130, got ${tight.length}`);
    assert.ok(
      tight.includes("ERR_UPSTREAM_TIMEOUT"),
      `Real diagnostic must never be displaced by error-handler.test.ts header:\n${tight}`,
    );
    assert.ok(
      tight.includes("src/client.ts:88:11"),
      `App stack frame must be preserved:\n${tight}`,
    );
  });
});

describe("v0.2.1 Refinement Suite (V2.1-1, V2.1-2, V2.1-3)", () => {
  it("43. [V2.1-1] prunes large args payload fields on superseded full_write events without mutating inputs", () => {
    const staleBody = Array.from(
      { length: 80 },
      (_, i) => `export const STALE_CONST_${i} = ${i * 100};`,
    ).join("\n");
    const freshBody = "export const FRESH_CONST = 999;";

    const events: ToolTrajectoryEvent[] = [
      {
        callId: "w1",
        tool: "write_file",
        args: { path: "src/module.ts", content: staleBody, encoding: "utf8" },
        ok: true,
        output: "Wrote src/module.ts",
      },
      {
        callId: "w2",
        tool: "create_file",
        args: { path: "src/alt.ts", file_content: staleBody },
        ok: true,
        output: "Created src/alt.ts",
      },
      {
        callId: "w1_new",
        tool: "write_file",
        args: { path: "src/module.ts", content: freshBody, encoding: "utf8" },
        ok: true,
        output: "Wrote src/module.ts",
      },
      {
        callId: "r2_new",
        tool: "read_file",
        args: { path: "src/alt.ts" },
        ok: true,
        output: "export const ALT_FINAL = 1;",
      },
    ];

    const origSnapshot = JSON.stringify(events);
    const res = pruneTrajectoryEvents(events);
    assert.equal(JSON.stringify(events), origSnapshot, "Caller events and args must never be mutated");
    assert.equal(res.prunedCount, 2);

    const prunedW1 = res.events.find((e) => e.callId === "w1")!;
    assert.equal(prunedW1.pruned, true);
    assert.equal(prunedW1.args?.path, "src/module.ts", "path metadata must remain untouched");
    assert.equal(prunedW1.args?.encoding, "utf8", "other args metadata must remain untouched");
    assert.equal(
      prunedW1.args?.content,
      "[superseded: newer full_write on file:src/module.ts]",
    );

    const prunedW2 = res.events.find((e) => e.callId === "w2")!;
    assert.equal(prunedW2.args?.path, "src/alt.ts");
    assert.equal(
      prunedW2.args?.file_content,
      "[superseded: newer full_read on file:src/alt.ts]",
    );

    // Active w1_new must remain completely unpruned
    const keptW1New = res.events.find((e) => e.callId === "w1_new")!;
    assert.equal(keptW1New.pruned, false);
    assert.equal(keptW1New.args?.content, freshBody);

    // Savings must credit both pruned 4KB+ payloads (not just the tiny 19-byte output ack!)
    assert.ok(
      res.charsSaved > staleBody.length * 2 - 200,
      `Expected charsSaved > ${staleBody.length * 2 - 200}, got ${res.charsSaved}`,
    );
    assert.ok(res.tokensSaved > 400, `Expected substantial tokensSaved, got ${res.tokensSaved}`);
  });

  it("44. [V2.1-2] extracts app file refs linearly, links depends_on from failures, and promotes cited read-only files to Tier 2", () => {
    const stackOutput = [
      "FAIL packages/core/src/session.test.ts > verifies key rotation",
      "  Error [ERR_KEY_EXPIRED]: active key rotation failed",
      "    at runTest (node_modules/vitest/dist/runner.js:440:12)",
      "    at processTicks (node:internal/process/task_queues:95:5)",
      "    at verifyKey (/workspace/repo/src/security/keyring.ts:142:9)",
      "    at packages/core/src/session.test.ts:50:11",
    ].join("\n");

    const refs = extractAppFileReferences(stackOutput);
    assert.ok(refs.includes("src/security/keyring.ts"), `Missing implementation ref: ${refs}`);
    assert.ok(refs.includes("packages/core/src/session.test.ts"), `Missing monorepo path ref: ${refs}`);
    assert.equal(refs[0], "src/security/keyring.ts", "Non-test implementation file should come first");
    assert.ok(!refs.some((r) => r.includes("node_modules")), "Must exclude node_modules");
    assert.ok(!refs.some((r) => r.includes("task_queues")), "Must exclude node:internal");

    // ReDoS linearity check on extractAppFileReferences
    const t0 = Date.now();
    extractAppFileReferences(" ".repeat(20_000) + "x");
    extractAppFileReferences("src/".repeat(5_000));
    extractAppFileReferences("(".repeat(20_000));
    assert.ok(Date.now() - t0 < 50, "extractAppFileReferences must execute in linear O(N) time");

    // Graph promotion test: read-only file cited by failing stack trace is promoted to Tier 2
    // while unrelated read-only files stay collapsed in [other read files]
    const store = new GraphStore(":memory:");
    const g = store.createGraph({ id: "g_cited_promo", name: "Failure-Cited Promotion" });

    ingestToolArtifacts(store, g.id, [
      {
        callId: "r_key",
        tool: "read_file",
        args: { path: "src/security/keyring.ts" },
        ok: true,
        output: "export const KEY_TTL_SECONDS = 3600;\nexport const ROTATION_GRACE_SEC = 120;\nexport function verifyKey(id: string) { return id.length > 0; }",
      },
      {
        callId: "r_noise1",
        tool: "read_file",
        args: { path: "src/ui/theme.ts" },
        ok: true,
        output: "export const THEME_ACCENT = '#123456';\n".repeat(8),
      },
      {
        callId: "r_noise2",
        tool: "read_file",
        args: { path: "src/docs/changelog.ts" },
        ok: true,
        output: "export const CHANGELOG_HEADING = 'v1';\n".repeat(8),
      },
      {
        callId: "cmd_fail",
        tool: "bash",
        args: { command: "npm test" },
        ok: false,
        output: stackOutput,
      },
    ]);

    // Verify genuine depends_on edge from failing command to read-only src/security/keyring.ts
    const nodes = store.listNodes(g.id, { status: "active" });
    const failNode = nodes.find((n) => n.resourceKey === "cmd:npm test")!;
    const keyNode = nodes.find((n) => n.resourceKey === "file:src/security/keyring.ts")!;
    const depEdges = store.listEdges(g.id, { kind: "depends_on" });
    assert.ok(
      depEdges.some((e) => e.sourceId === failNode.id && e.targetId === keyNode.id),
      "Failing command must add depends_on edge to active file cited in stack trace",
    );

    const compiled = compileContext(store, {
      graphId: g.id,
      format: "compact_markdown",
    });
    assert.ok(compiled.promptText.length <= DEFAULT_COMPACT_MAX_CHARS);
    assert.ok(
      compiled.promptText.includes("KEY_TTL_SECONDS = 3600"),
      `Failure-cited read-only file must be promoted to Tier 2:\n${compiled.promptText}`,
    );
    assert.ok(compiled.promptText.includes("ROTATION_GRACE_SEC = 120"));
    assert.ok(compiled.promptText.includes("- [other read files]:"));
    assert.ok(!compiled.promptText.includes("THEME_ACCENT"), "Unrelated read-only files must stay collapsed");
    store.close();
  });

  it("45. [V2.1-2] strictly prefers non-test implementation frames over test harness frames in compactCommandExcerpt", () => {
    // Go failure: *_test.go appears before implementation .go file
    const goFail = [
      "--- FAIL: TestQueueDrain (0.02s)",
      "    Error: ERR_QUEUE_STALLED: worker stopped before drain completed",
      "    pkg/queue/worker_test.go:84 +0x1c",
      "    pkg/queue/worker.go:118 +0x2a",
    ].join("\n");
    const goEx = compactCommandExcerpt(goFail, false, 180);
    assert.ok(goEx.length <= 180);
    assert.ok(goEx.includes("pkg/queue/worker.go:118"), `Expected non-test Go frame, got:\n${goEx}`);

    // TypeScript failure: *.test.ts appears before implementation .ts file
    const tsFail = [
      "FAIL src/auth/jwt.test.ts > rejects invalid audience",
      "  Error [ERR_AUD_MISMATCH]: audience check failed",
      "    at src/auth/jwt.test.ts:50:12",
      "    at verifyJwt (src/auth/jwt.ts:142:9)",
    ].join("\n");
    const tsEx = compactCommandExcerpt(tsFail, false, 180);
    assert.ok(tsEx.length <= 180);
    assert.ok(tsEx.includes("src/auth/jwt.ts:142:9"), `Expected non-test TS frame, got:\n${tsEx}`);

    // Python failure: test_*.py appears before implementation .py file
    const pyFail = [
      "FAILED tests/test_retry.py::test_backoff",
      '  File "tests/test_retry.py", line 24, in test_backoff',
      '  File "src/worker/retry.py", line 67, in compute_backoff',
      "  ValueError: backoff exceeded cap",
    ].join("\n");
    const pyEx = compactCommandExcerpt(pyFail, false, 180);
    assert.ok(pyEx.length <= 180);
    assert.ok(pyEx.includes("src/worker/retry.py"), `Expected non-test Python frame, got:\n${pyEx}`);
  });

  it("46. [V2.1-3] defaults compact_markdown to <= 720 chars with dense P1-focused excerpts while supporting larger explicit caps", () => {
    assert.equal(DEFAULT_COMPACT_MAX_CHARS, 720);

    // P1 density stop: once declarations fill >= 65% of cap, trailing P2 boilerplate is not packed
    const denseSource = [
      "export const SYNC_BATCH_SIZE = 64;",
      "export const HEARTBEAT_SEC = 15;",
      "export const RETRY_JITTER_MS = 200;",
      "export interface SchedulerConfig {",
      "  concurrency: number;",
      "  drainTimeoutMs: number;",
      "}",
      "export function createScheduler(cfg: SchedulerConfig) {",
      "  const internalTempA = cfg.concurrency + 1;",
      "  const internalTempB = cfg.drainTimeoutMs * 2;",
      "  const verboseBoilerplateLine = internalTempA + internalTempB;",
      "  return verboseBoilerplateLine;",
      "}",
    ].join("\n");
    const denseEx = compactCodeExcerpt(denseSource, 220);
    assert.ok(denseEx.length <= 220);
    assert.ok(denseEx.includes("SYNC_BATCH_SIZE = 64"));
    assert.ok(denseEx.includes("HEARTBEAT_SEC = 15"));
    assert.ok(
      !denseEx.includes("verboseBoilerplateLine"),
      `P1-dense excerpt should skip trailing P2 boilerplate:\n${denseEx}`,
    );

    const store = new GraphStore(":memory:");
    const g = store.createGraph({ id: "g_default_720", name: "Default 720 Density" });
    ingestToolArtifacts(store, g.id, [
      {
        callId: "d_r1",
        tool: "read_file",
        args: { path: "src/scheduler.ts" },
        ok: true,
        output: denseSource,
      },
      {
        callId: "d_e1",
        tool: "edit_file",
        args: {
          path: "src/scheduler.ts",
          old_text: "export const SYNC_BATCH_SIZE = 64;",
          new_text: "export const SYNC_BATCH_SIZE = 128;",
        },
        ok: true,
        output: "Edited src/scheduler.ts",
      },
      {
        callId: "d_bg1",
        tool: "read_file",
        args: { path: "src/banner.ts" },
        ok: true,
        output: "export const BANNER = 'hello';\n".repeat(15),
      },
      {
        callId: "d_bg2",
        tool: "read_file",
        args: { path: "src/colors.ts" },
        ok: true,
        output: "export const PRIMARY = 'blue';\n".repeat(15),
      },
      {
        callId: "d_fail",
        tool: "bash",
        args: { command: "npm test" },
        ok: false,
        output: [
          "FAIL src/scheduler.test.ts > bounds batch size",
          "  Error [ERR_BATCH_LIMIT]: batch 128 exceeded max 64",
          "    at src/scheduler.test.ts:19:5",
          "    at createScheduler (src/scheduler.ts:8:12)",
        ].join("\n"),
      },
    ]);

    const defaultRes = compileContext(store, {
      graphId: g.id,
      format: "compact_markdown",
    });
    assert.ok(
      defaultRes.promptText.length > 0 && defaultRes.promptText.length <= 720,
      `Default compact_markdown must be <= 720 chars, got ${defaultRes.promptText.length}`,
    );
    assert.ok(defaultRes.promptText.includes("ERR_BATCH_LIMIT"));
    assert.ok(defaultRes.promptText.includes("src/scheduler.ts:8:12"));
    assert.ok(defaultRes.promptText.includes("SYNC_BATCH_SIZE = 128"));
    assert.ok(defaultRes.promptText.includes("- [other read files]:"));

    // Explicit maxOutputChars up to 4000 still works
    const expandedRes = compileContext(store, {
      graphId: g.id,
      format: "compact_markdown",
      maxOutputChars: 2000,
      recordCacheState: false,
    });
    assert.ok(expandedRes.promptText.length <= 2000);
    assert.ok(expandedRes.promptText.includes("ERR_BATCH_LIMIT"));
    store.close();
  });
});

describe("v0.2.2 Excerpt Ordering & Diff Preservation Suite (V2.2-1 & V2.2-2)", () => {
  it("47. [V2.2-1] omits Vitest runner file banners and leads with Error diagnostic + non-test source frame", () => {
    const vitestOut = [
      "❯ src/foo.test.ts (18 tests | 1 failed) 42ms",
      "  × rejects invalid digest header",
      "    → Error [ERR_X]: digest verification failed for request payload",
      "      at verifyPayload (src/foo.ts:94:15)",
      "      at runTask (node_modules/vitest/dist/runner.js:412:9)",
      "Test Files  1 failed (1)",
      "Tests  1 failed | 17 passed (18)",
    ].join("\n");

    const excerpt = compactCommandExcerpt(vitestOut, false, 240);
    assert.ok(excerpt.length <= 240, `Expected <= 240, got ${excerpt.length}`);
    assert.ok(
      !excerpt.includes("src/foo.test.ts (18 tests"),
      `Must omit test-file summary count banner:\n${excerpt}`,
    );
    assert.ok(
      excerpt.includes("Error [ERR_X]: digest verification failed"),
      `Must preserve Error diagnostic:\n${excerpt}`,
    );
    assert.ok(
      excerpt.includes("src/foo.ts:94:15"),
      `Must preserve non-test implementation frame:\n${excerpt}`,
    );
    // Error diagnostic MUST appear ahead of any test-case path
    const errIdx = excerpt.indexOf("Error [ERR_X]");
    const frameIdx = excerpt.indexOf("src/foo.ts:94:15");
    assert.ok(errIdx >= 0 && frameIdx > errIdx, `Expected Error -> impl source order:\n${excerpt}`);

    // Pure banner + error + frame (no intermediate `×` line): literally starts with Error [ERR_X]
    const vitestDirect = [
      "❯ src/foo.test.ts (18 tests | 1 failed)",
      "  → Error [ERR_X]: signature mismatch",
      "    at fn (src/foo.ts:94:15)",
    ].join("\n");
    const directEx = compactCommandExcerpt(vitestDirect, false, 200);
    assert.ok(directEx.startsWith("Error [ERR_X]:"), `Expected excerpt to lead with Error:\n${directEx}`);
    assert.ok(directEx.includes("src/foo.ts:94:15"));
    assert.ok(!directEx.includes("src/foo.test.ts"));
  });

  it("48. [V2.2-1] strips test-file prefix from Go Error diagnostics and pairs TestFoo + Error + non-test source", () => {
    const goOut = [
      "--- FAIL: TestFoo (0.19s)",
      "    foo_test.go:84: Error [ERR_Y]: queue worker stalled during flush",
      "    pkg/mod/foo.go:118 +0x1c4",
      "FAIL\tgithub.com/example/mod/pkg/mod\t0.214s",
    ].join("\n");

    const excerpt = compactCommandExcerpt(goOut, false, 220);
    assert.ok(excerpt.length <= 220, `Expected <= 220, got ${excerpt.length}`);
    assert.ok(excerpt.includes("--- FAIL: TestFoo"), `Must keep named Go test:\n${excerpt}`);
    assert.ok(!excerpt.includes("(0.19s)"), `Should strip timing noise:\n${excerpt}`);
    assert.ok(
      excerpt.includes("Error [ERR_Y]: queue worker stalled during flush"),
      `Must keep clean Error diagnostic:\n${excerpt}`,
    );
    assert.ok(
      !excerpt.includes("foo_test.go:84:"),
      `Must strip leading test-file prefix when non-test source frame exists:\n${excerpt}`,
    );
    assert.ok(
      excerpt.includes("pkg/mod/foo.go:118"),
      `Must preserve non-test implementation source frame:\n${excerpt}`,
    );
    assert.ok(!excerpt.includes("+0x1c4"), `Should strip noisy hex PC offset:\n${excerpt}`);
  });

  it("49. [V2.2-2] always preserves + (new/replacement) lines in tight partial_edit diff excerpts", () => {
    const longDiff = [
      '- export const LEGACY_ENDPOINT_URL = "https://deprecated-v1.internal.service.local/api/dispatch/v1/legacy";',
      '+ export const ACTIVE_ENDPOINT_URL = "https://v2.service.local/api/dispatch";',
    ].join("\n");

    // Direct diffText option under tight maxChars = 110
    const viaOpt = compactCodeExcerpt("Edited src/client.ts", 110, { diffText: longDiff });
    assert.ok(viaOpt.length <= 110, `Expected <= 110, got ${viaOpt.length}`);
    assert.ok(
      viaOpt.includes("+ export const ACTIVE_ENDPOINT_URL"),
      `Must ALWAYS preserve + new replacement line under tight budget:\n${viaOpt}`,
    );
    assert.ok(
      viaOpt.includes("https://v2.service.local/api/dispatch"),
      `Full + replacement value should stay intact:\n${viaOpt}`,
    );

    // Re-excerpting a stored pure diff block (without opts.diffText) also preserves + lines
    const viaStored = compactCodeExcerpt(longDiff, 110);
    assert.ok(viaStored.length <= 110);
    assert.ok(
      viaStored.includes("+ export const ACTIVE_ENDPOINT_URL"),
      `Stored diff re-excerpt must also preserve + replacement line:\n${viaStored}`,
    );

    // Both - and + very long (>140 chars each) under maxChars = 110: + still gets >= 65% of budget
    const bothLongDiff = [
      '- export const OLD_POLICY = { mode: "legacy_fallback", retryBackoffMs: 1200, maxAttempts: 4, strictHeaders: false, auditLog: "off" };',
      '+ export const NEW_POLICY = { mode: "strict_primary", retryBackoffMs: 300, maxAttempts: 8, strictHeaders: true, auditLog: "on" };',
    ].join("\n");
    const bothCut = compactCodeExcerpt("", 110, { diffText: bothLongDiff });
    assert.ok(bothCut.length <= 110);
    assert.ok(
      bothCut.includes("+ export const NEW_POLICY"),
      `Both-long diff must allocate majority budget to + new line:\n${bothCut}`,
    );
  });

  it("50. [V2.2 ReDoS] maintains strictly linear O(N) performance (< 10ms) on 20KB adversarial inputs", () => {
    const inputs = [
      " ".repeat(20_000) + "x",
      "[".repeat(20_000),
      "(".repeat(20_000),
      "a".repeat(20_000),
      "❯ " + "a".repeat(20_000),
    ];
    // Warm up JIT once before measuring
    compactCommandExcerpt("x", false, 120);
    compactCodeExcerpt("- a\n+ b", 120, { diffText: "- a\n+ b" });

    const t0 = Date.now();
    for (const s of inputs) {
      sanitizeNeutralText(s);
      compactCommandExcerpt(s, false, 200);
      compactCodeExcerpt(s, 200, { diffText: s });
    }
    const elapsedMs = Date.now() - t0;
    assert.ok(elapsedMs < 10, `Expected linear O(N) runtime < 10ms, got ${elapsedMs}ms`);
  });
});
