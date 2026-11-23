import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { after, before, describe, it } from "node:test";
import { runBenchmark } from "./benchmark.ts";
import { runCli } from "./cli.ts";
import { compileContext } from "./compiler.ts";
import { GraphStore } from "./graph-store.ts";
import { ContextGraphMcpSession, handleMcpJsonRpc, MCP_TOOLS, startMcpStdioServer } from "./mcp.ts";
import {
  createContextGraphServer,
  isLoopbackHost,
  type ContextGraphServerInstance,
} from "./server.ts";
import { ContextGraphError, CycleDetectedError, NotFoundError } from "./types.ts";
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
  it("18. completes MCP handshake and lists all 6 Context Graph tools", () => {
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
    assert.equal(listResp.result.tools.length, 6);
    const names = listResp.result.tools.map((t) => t.name);
    assert.deepEqual(names, [
      "cg_create_graph",
      "cg_add_node",
      "cg_link_nodes",
      "cg_ingest_turn",
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

      // 3. POST /mcp tools/list -> 200 with all 6 tools
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
      assert.equal(listBody.result.tools.length, 6);
      assert.ok(listBody.result.tools.some((t) => t.name === "cg_create_graph"));
      assert.ok(listBody.result.tools.some((t) => t.name === "cg_ingest_turn"));
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
