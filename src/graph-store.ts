import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import {
  CONTEXT_TIERS,
  ContextGraphError,
  CycleDetectedError,
  EDGE_KINDS,
  MESSAGE_ROLES,
  NODE_KINDS,
  NODE_STATUSES,
  NotFoundError,
  type AddEdgeInput,
  type AddNodeInput,
  type ContextTier,
  type CreateGraphInput,
  type EdgeKind,
  type GraphEdge,
  type GraphNode,
  type GraphRecord,
  type GraphSnapshot,
  type GraphStats,
  type IngestTurnInput,
  type IngestTurnResult,
  type MessageRole,
  type NodeKind,
  type NodeStatus,
  type UpdateNodePatch,
} from "./types.ts";
import { countTokens, deriveStub, deriveSummary, embedText } from "./vector.ts";

export const DEFAULT_BUDGET_TOKENS = 4096 as const;

const DEFAULT_TIER_BY_KIND: Record<NodeKind, ContextTier> = {
  system: "pinned_prefix",
  goal: "pinned_prefix",
  constraint: "pinned_prefix",
  task_frame: "stable_frame",
  decision: "working_subgraph",
  artifact: "working_subgraph",
  fact: "working_subgraph",
  summary: "working_subgraph",
  observation: "ephemeral_leaf",
};

const DEFAULT_ROLE_BY_KIND: Record<NodeKind, MessageRole> = {
  system: "system",
  goal: "user",
  constraint: "system",
  task_frame: "system",
  decision: "assistant",
  artifact: "tool",
  fact: "user",
  summary: "assistant",
  observation: "tool",
};

const DEFAULT_SALIENCE_BY_KIND: Record<NodeKind, number> = {
  system: 1.0,
  goal: 0.98,
  constraint: 0.96,
  task_frame: 0.88,
  decision: 0.85,
  artifact: 0.82,
  summary: 0.78,
  fact: 0.74,
  observation: 0.48,
};

const DEFAULT_EDGE_WEIGHT: Record<EdgeKind, number> = {
  depends_on: 0.9,
  supersedes: 1.0,
  summarizes: 0.75,
  elaborates: 0.65,
  scoped_to: 0.55,
  conflicts_with: 0.8,
};

const DAG_EDGE_KINDS = new Set<EdgeKind>([
  "depends_on",
  "supersedes",
  "summarizes",
  "elaborates",
]);

interface RawGraphRow {
  id: string;
  name: string;
  description: string;
  default_budget: number;
  compile_count: number;
  last_compiled_tokens_json: string;
  metadata_json: string;
  created_at: string;
  updated_at: string;
}

interface RawNodeRow {
  id: string;
  graph_id: string;
  seq: number;
  turn: number;
  kind: NodeKind;
  role: MessageRole;
  scope: string;
  tier: ContextTier;
  content: string;
  summary: string;
  stub: string;
  tokens: number;
  summary_tokens: number;
  stub_tokens: number;
  salience: number;
  status: NodeStatus;
  resource_key: string | null;
  version: number;
  superseded_by: string | null;
  embedding_json: string;
  metadata_json: string;
  created_at: string;
  updated_at: string;
}

interface RawEdgeRow {
  id: string;
  graph_id: string;
  source_id: string;
  target_id: string;
  kind: EdgeKind;
  weight: number;
  metadata_json: string;
  created_at: string;
}

function clampSalience(v: number): number {
  if (!Number.isFinite(v)) return 0.5;
  return Math.max(0, Math.min(1, v));
}

function sanitizeMermaidLabel(raw: string): string {
  return raw
    .replace(/[\r\n]+/g, " ")
    .replace(/[<>"]/g, "'")
    .replace(/\[/g, "(")
    .replace(/\]/g, ")")
    .replace(/\s+/g, " ")
    .trim();
}

export class GraphStore {
  readonly dbPath: string;
  private readonly db: DatabaseSync;
  private closed = false;
  private txDepth = 0;

  constructor(dbPath = ":memory:") {
    this.dbPath = dbPath === ":memory:" ? ":memory:" : resolve(dbPath);
    if (this.dbPath !== ":memory:") {
      mkdirSync(dirname(this.dbPath), { recursive: true });
    }
    this.db = new DatabaseSync(this.dbPath);
    this.initSchema();
  }

  private initSchema(): void {
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA foreign_keys = ON;
      PRAGMA synchronous = NORMAL;

      CREATE TABLE IF NOT EXISTS graphs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        default_budget INTEGER NOT NULL DEFAULT 4096,
        compile_count INTEGER NOT NULL DEFAULT 0,
        last_compiled_tokens_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS nodes (
        id TEXT NOT NULL,
        graph_id TEXT NOT NULL REFERENCES graphs(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        turn INTEGER NOT NULL DEFAULT 1,
        kind TEXT NOT NULL,
        role TEXT NOT NULL,
        scope TEXT NOT NULL DEFAULT 'global',
        tier TEXT NOT NULL,
        content TEXT NOT NULL,
        summary TEXT NOT NULL,
        stub TEXT NOT NULL,
        tokens INTEGER NOT NULL,
        summary_tokens INTEGER NOT NULL,
        stub_tokens INTEGER NOT NULL,
        salience REAL NOT NULL DEFAULT 0.7,
        status TEXT NOT NULL DEFAULT 'active',
        resource_key TEXT,
        version INTEGER NOT NULL DEFAULT 1,
        superseded_by TEXT,
        embedding_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (graph_id, id)
      );

      CREATE INDEX IF NOT EXISTS idx_nodes_graph_seq ON nodes(graph_id, seq);
      CREATE INDEX IF NOT EXISTS idx_nodes_graph_status ON nodes(graph_id, status);
      CREATE INDEX IF NOT EXISTS idx_nodes_graph_resource ON nodes(graph_id, resource_key);

      CREATE TABLE IF NOT EXISTS edges (
        id TEXT NOT NULL,
        graph_id TEXT NOT NULL REFERENCES graphs(id) ON DELETE CASCADE,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        PRIMARY KEY (graph_id, id),
        UNIQUE (graph_id, source_id, target_id, kind),
        FOREIGN KEY (graph_id, source_id) REFERENCES nodes(graph_id, id) ON DELETE CASCADE,
        FOREIGN KEY (graph_id, target_id) REFERENCES nodes(graph_id, id) ON DELETE CASCADE
      );

      CREATE INDEX IF NOT EXISTS idx_edges_graph_kind ON edges(graph_id, kind);
      CREATE INDEX IF NOT EXISTS idx_edges_graph_source ON edges(graph_id, source_id);
      CREATE INDEX IF NOT EXISTS idx_edges_graph_target ON edges(graph_id, target_id);
    `);
  }

  close(): void {
    if (!this.closed) {
      this.db.close();
      this.closed = true;
    }
  }

  private hydrateGraph(row: RawGraphRow): GraphRecord {
    return {
      id: row.id,
      name: row.name,
      description: row.description,
      defaultBudget: row.default_budget,
      compileCount: row.compile_count,
      lastCompiledTokenIds: JSON.parse(row.last_compiled_tokens_json) as number[],
      metadata: JSON.parse(row.metadata_json) as Record<string, unknown>,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }

  private hydrateNode(row: RawNodeRow): GraphNode {
    return {
      id: row.id,
      graphId: row.graph_id,
      seq: row.seq,
      turn: row.turn,
      kind: row.kind,
      role: row.role,
      scope: row.scope,
      tier: row.tier,
      content: row.content,
      summary: row.summary,
      stub: row.stub,
      tokens: row.tokens,
      summaryTokens: row.summary_tokens,
      stubTokens: row.stub_tokens,
      salience: row.salience,
      status: row.status,
      resourceKey: row.resource_key,
      version: row.version,
      supersededBy: row.superseded_by,
      embedding: JSON.parse(row.embedding_json) as number[],
      metadata: JSON.parse(row.metadata_json) as Record<string, unknown>,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }

  private hydrateEdge(row: RawEdgeRow): GraphEdge {
    return {
      id: row.id,
      graphId: row.graph_id,
      sourceId: row.source_id,
      targetId: row.target_id,
      kind: row.kind,
      weight: row.weight,
      metadata: JSON.parse(row.metadata_json) as Record<string, unknown>,
      createdAt: row.created_at,
    };
  }

  /**
   * Executes `fn` inside a single atomic SQLite transaction (`BEGIN IMMEDIATE` /
   * `COMMIT` / `ROLLBACK`). Supports safe re-entrant nested calls via SQLite
   * `SAVEPOINT`s so callers can batch multi-write workflows without issuing
   * nested `BEGIN IMMEDIATE` errors or per-statement WAL fsyncs.
   */
  runInTransaction<T>(fn: () => T): T {
    if (this.txDepth === 0) {
      this.db.exec("BEGIN IMMEDIATE");
      this.txDepth = 1;
      try {
        const result = fn();
        this.db.exec("COMMIT");
        return result;
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch {
          // Ignore rollback errors if SQLite already aborted the transaction
        }
        throw err;
      } finally {
        this.txDepth = 0;
      }
    }

    const spName = `cg_sp_${this.txDepth}`;
    this.db.exec(`SAVEPOINT ${spName}`);
    this.txDepth++;
    try {
      const result = fn();
      this.db.exec(`RELEASE SAVEPOINT ${spName}`);
      return result;
    } catch (err) {
      try {
        this.db.exec(`ROLLBACK TO SAVEPOINT ${spName}`);
        this.db.exec(`RELEASE SAVEPOINT ${spName}`);
      } catch {
        // Ignore savepoint cleanup errors if outer transaction was aborted
      }
      throw err;
    } finally {
      this.txDepth--;
    }
  }

  hasGraph(graphId: string): boolean {
    const row = this.db
      .prepare("SELECT 1 FROM graphs WHERE id = ?")
      .get(graphId);
    return Boolean(row);
  }

  private ensureGraphExists(graphId: string): GraphRecord {
    const row = this.db
      .prepare("SELECT * FROM graphs WHERE id = ?")
      .get(graphId) as RawGraphRow | undefined;
    if (!row) {
      throw new NotFoundError(`Graph '${graphId}' not found`);
    }
    return this.hydrateGraph(row);
  }

  createGraph(input: CreateGraphInput): GraphRecord {
    if (!input.name || typeof input.name !== "string") {
      throw new ContextGraphError("Graph 'name' is required", "VALIDATION_ERROR", 400);
    }
    const id = input.id?.trim() || `cg_${randomUUID().slice(0, 12)}`;
    const now = new Date().toISOString();
    const defaultBudget =
      typeof input.defaultBudget === "number" && input.defaultBudget >= 128
        ? Math.floor(input.defaultBudget)
        : DEFAULT_BUDGET_TOKENS;

    const existing = this.db.prepare("SELECT id FROM graphs WHERE id = ?").get(id);
    if (existing) {
      throw new ContextGraphError(`Graph '${id}' already exists`, "ALREADY_EXISTS", 409);
    }

    this.db
      .prepare(
        `INSERT INTO graphs (
          id, name, description, default_budget, compile_count,
          last_compiled_tokens_json, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 0, '[]', ?, ?, ?)`,
      )
      .run(
        id,
        input.name.trim(),
        input.description?.trim() ?? "",
        defaultBudget,
        JSON.stringify(input.metadata ?? {}),
        now,
        now,
      );

    return this.getGraph(id);
  }

  listGraphs(): GraphRecord[] {
    const rows = this.db
      .prepare("SELECT * FROM graphs ORDER BY created_at DESC")
      .all() as unknown as RawGraphRow[];
    return rows.map((r) => this.hydrateGraph(r));
  }

  getGraph(graphId: string): GraphRecord {
    return this.ensureGraphExists(graphId);
  }

  deleteGraph(graphId: string): boolean {
    const res = this.db.prepare("DELETE FROM graphs WHERE id = ?").run(graphId);
    return Number(res.changes) > 0;
  }

  countsSummary(): { graphsCount: number; nodesCount: number; edgesCount: number } {
    const g = this.db.prepare("SELECT COUNT(*) as c FROM graphs").get() as { c: number };
    const n = this.db.prepare("SELECT COUNT(*) as c FROM nodes").get() as { c: number };
    const e = this.db.prepare("SELECT COUNT(*) as c FROM edges").get() as { c: number };
    return { graphsCount: Number(g.c), nodesCount: Number(n.c), edgesCount: Number(e.c) };
  }

  saveCompiledTokens(graphId: string, tokenIds: readonly number[]): void {
    const now = new Date().toISOString();
    this.db
      .prepare(
        `UPDATE graphs
         SET last_compiled_tokens_json = ?, compile_count = compile_count + 1, updated_at = ?
         WHERE id = ?`,
      )
      .run(JSON.stringify(tokenIds), now, graphId);
  }

  getNode(graphId: string, nodeId: string): GraphNode {
    this.ensureGraphExists(graphId);
    const row = this.db
      .prepare("SELECT * FROM nodes WHERE graph_id = ? AND id = ?")
      .get(graphId, nodeId) as RawNodeRow | undefined;
    if (!row) {
      throw new NotFoundError(`Node '${nodeId}' not found in graph '${graphId}'`);
    }
    return this.hydrateNode(row);
  }

  listNodes(
    graphId: string,
    filter?: { status?: NodeStatus; kind?: NodeKind; scope?: string; resourceKey?: string },
  ): GraphNode[] {
    this.ensureGraphExists(graphId);
    const clauses = ["graph_id = ?"];
    const params: Array<string | number> = [graphId];
    if (filter?.status) {
      clauses.push("status = ?");
      params.push(filter.status);
    }
    if (filter?.kind) {
      clauses.push("kind = ?");
      params.push(filter.kind);
    }
    if (filter?.scope) {
      clauses.push("scope = ?");
      params.push(filter.scope);
    }
    if (filter?.resourceKey) {
      clauses.push("resource_key = ?");
      params.push(filter.resourceKey);
    }
    const sql = `SELECT * FROM nodes WHERE ${clauses.join(" AND ")} ORDER BY seq ASC`;
    const rows = this.db.prepare(sql).all(...params) as unknown as RawNodeRow[];
    return rows.map((r) => this.hydrateNode(r));
  }

  listEdges(graphId: string, filter?: { kind?: EdgeKind }): GraphEdge[] {
    this.ensureGraphExists(graphId);
    if (filter?.kind) {
      const rows = this.db
        .prepare("SELECT * FROM edges WHERE graph_id = ? AND kind = ? ORDER BY created_at ASC, id ASC")
        .all(graphId, filter.kind) as unknown as RawEdgeRow[];
      return rows.map((r) => this.hydrateEdge(r));
    }
    const rows = this.db
      .prepare("SELECT * FROM edges WHERE graph_id = ? ORDER BY created_at ASC, id ASC")
      .all(graphId) as unknown as RawEdgeRow[];
    return rows.map((r) => this.hydrateEdge(r));
  }

  private addNodeInternal(
    graphId: string,
    input: AddNodeInput,
  ): { node: GraphNode; autoEdges: GraphEdge[]; supersededIds: string[] } {
    if (!NODE_KINDS.includes(input.kind)) {
      throw new ContextGraphError(`Invalid node kind '${String(input.kind)}'`, "VALIDATION_ERROR", 400);
    }
    if (typeof input.content !== "string" || input.content.trim().length === 0) {
      throw new ContextGraphError("Node 'content' must be a non-empty string", "VALIDATION_ERROR", 400);
    }
    const role = input.role ?? DEFAULT_ROLE_BY_KIND[input.kind];
    if (!MESSAGE_ROLES.includes(role)) {
      throw new ContextGraphError(`Invalid role '${String(role)}'`, "VALIDATION_ERROR", 400);
    }
    const tier = input.tier ?? DEFAULT_TIER_BY_KIND[input.kind];
    if (!CONTEXT_TIERS.includes(tier)) {
      throw new ContextGraphError(`Invalid tier '${String(tier)}'`, "VALIDATION_ERROR", 400);
    }
    const status = input.status ?? "active";
    if (!NODE_STATUSES.includes(status)) {
      throw new ContextGraphError(`Invalid status '${String(status)}'`, "VALIDATION_ERROR", 400);
    }

    const seqRow = this.db
      .prepare("SELECT COALESCE(MAX(seq), 0) as max_seq, COALESCE(MAX(turn), 1) as max_turn FROM nodes WHERE graph_id = ?")
      .get(graphId) as { max_seq: number; max_turn: number };
    const seq = Number(seqRow.max_seq) + 1;
    const turn = typeof input.turn === "number" && input.turn >= 1 ? Math.floor(input.turn) : Number(seqRow.max_turn);

    const id = input.id?.trim() || `n_${seq}_${randomUUID().slice(0, 8)}`;
    const dup = this.db
      .prepare("SELECT id FROM nodes WHERE graph_id = ? AND id = ?")
      .get(graphId, id);
    if (dup) {
      throw new ContextGraphError(`Node '${id}' already exists in graph '${graphId}'`, "ALREADY_EXISTS", 409);
    }

    const resourceKey = input.resourceKey?.trim() ? input.resourceKey.trim() : null;
    let version = 1;
    let priorActiveForResource: RawNodeRow[] = [];

    if (resourceKey) {
      const maxVerRow = this.db
        .prepare("SELECT COALESCE(MAX(version), 0) as max_ver FROM nodes WHERE graph_id = ? AND resource_key = ?")
        .get(graphId, resourceKey) as { max_ver: number };
      version = Number(maxVerRow.max_ver) + 1;
      if (status === "active") {
        priorActiveForResource = this.db
          .prepare("SELECT * FROM nodes WHERE graph_id = ? AND resource_key = ? AND status = 'active' ORDER BY seq ASC")
          .all(graphId, resourceKey) as unknown as RawNodeRow[];
      }
    }

    const content = input.content;
    const summary = input.summary?.trim() || deriveSummary(content, input.kind, resourceKey);
    const stub = input.stub?.trim() || deriveStub(id, input.kind, resourceKey, version, summary);

    const tokens = Math.max(1, countTokens(content));
    const summaryTokens = Math.max(1, countTokens(summary));
    const stubTokens = Math.max(1, countTokens(stub));
    const salience = clampSalience(input.salience ?? DEFAULT_SALIENCE_BY_KIND[input.kind]);
    const scope = input.scope?.trim() || "global";
    const embedding = embedText(`${summary}\n${content}`, resourceKey);
    const now = new Date().toISOString();

    this.db
      .prepare(
        `INSERT INTO nodes (
          id, graph_id, seq, turn, kind, role, scope, tier,
          content, summary, stub, tokens, summary_tokens, stub_tokens,
          salience, status, resource_key, version, superseded_by,
          embedding_json, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)`,
      )
      .run(
        id,
        graphId,
        seq,
        turn,
        input.kind,
        role,
        scope,
        tier,
        content,
        summary,
        stub,
        tokens,
        summaryTokens,
        stubTokens,
        salience,
        status,
        resourceKey,
        version,
        JSON.stringify(embedding),
        JSON.stringify(input.metadata ?? {}),
        now,
        now,
      );

    const autoEdges: GraphEdge[] = [];
    const supersededIds: string[] = [];

    for (const prev of priorActiveForResource) {
      const edge = this.addEdgeInternal(graphId, {
        sourceId: id,
        targetId: prev.id,
        kind: "supersedes",
        metadata: { autoResourceKey: resourceKey, fromVersion: prev.version, toVersion: version },
      });
      autoEdges.push(edge);
      supersededIds.push(prev.id);
    }

    for (const targetId of input.dependsOn ?? []) {
      autoEdges.push(this.addEdgeInternal(graphId, { sourceId: id, targetId, kind: "depends_on" }));
    }
    for (const targetId of input.scopedTo ?? []) {
      autoEdges.push(this.addEdgeInternal(graphId, { sourceId: id, targetId, kind: "scoped_to" }));
    }
    for (const targetId of input.summarizes ?? []) {
      autoEdges.push(this.addEdgeInternal(graphId, { sourceId: id, targetId, kind: "summarizes" }));
    }
    for (const targetId of input.elaborates ?? []) {
      autoEdges.push(this.addEdgeInternal(graphId, { sourceId: id, targetId, kind: "elaborates" }));
    }
    for (const targetId of input.conflictsWith ?? []) {
      autoEdges.push(this.addEdgeInternal(graphId, { sourceId: id, targetId, kind: "conflicts_with" }));
    }

    this.db.prepare("UPDATE graphs SET updated_at = ? WHERE id = ?").run(now, graphId);
    return { node: this.getNode(graphId, id), autoEdges, supersededIds };
  }

  addNode(graphId: string, input: AddNodeInput): GraphNode {
    this.ensureGraphExists(graphId);
    return this.runInTransaction(() => this.addNodeInternal(graphId, input).node);
  }

  updateNode(graphId: string, nodeId: string, patch: UpdateNodePatch): GraphNode {
    const existing = this.getNode(graphId, nodeId);
    const content = patch.content ?? existing.content;
    const summary =
      patch.summary ??
      (patch.content !== undefined
        ? deriveSummary(content, existing.kind, existing.resourceKey)
        : existing.summary);
    const stub =
      patch.stub ??
      (patch.content !== undefined || patch.summary !== undefined
        ? deriveStub(existing.id, existing.kind, existing.resourceKey, existing.version, summary)
        : existing.stub);
    const tier = patch.tier ?? existing.tier;
    if (!CONTEXT_TIERS.includes(tier)) {
      throw new ContextGraphError(`Invalid tier '${String(tier)}'`, "VALIDATION_ERROR", 400);
    }
    const status = patch.status ?? existing.status;
    if (!NODE_STATUSES.includes(status)) {
      throw new ContextGraphError(`Invalid status '${String(status)}'`, "VALIDATION_ERROR", 400);
    }
    const scope = patch.scope?.trim() || existing.scope;
    const salience = patch.salience !== undefined ? clampSalience(patch.salience) : existing.salience;
    const metadata = patch.metadata !== undefined ? { ...existing.metadata, ...patch.metadata } : existing.metadata;
    const embedding =
      patch.content !== undefined || patch.summary !== undefined
        ? embedText(`${summary}\n${content}`, existing.resourceKey)
        : existing.embedding;

    const now = new Date().toISOString();
    this.db
      .prepare(
        `UPDATE nodes
         SET content = ?, summary = ?, stub = ?,
             tokens = ?, summary_tokens = ?, stub_tokens = ?,
             tier = ?, scope = ?, salience = ?, status = ?,
             embedding_json = ?, metadata_json = ?, updated_at = ?
         WHERE graph_id = ? AND id = ?`,
      )
      .run(
        content,
        summary,
        stub,
        Math.max(1, countTokens(content)),
        Math.max(1, countTokens(summary)),
        Math.max(1, countTokens(stub)),
        tier,
        scope,
        salience,
        status,
        JSON.stringify(embedding),
        JSON.stringify(metadata),
        now,
        graphId,
        nodeId,
      );

    this.db.prepare("UPDATE graphs SET updated_at = ? WHERE id = ?").run(now, graphId);
    return this.getNode(graphId, nodeId);
  }

  deleteNode(graphId: string, nodeId: string): boolean {
    this.ensureGraphExists(graphId);
    const res = this.db
      .prepare("DELETE FROM nodes WHERE graph_id = ? AND id = ?")
      .run(graphId, nodeId);
    return Number(res.changes) > 0;
  }

  /**
   * Checks whether `goalId` is reachable from `startId` along directed edges of `kind`
   * (or across DAG causal edges) in `graphId`.
   */
  private isReachable(graphId: string, startId: string, goalId: string, kind: EdgeKind): boolean {
    if (startId === goalId) return true;
    const stmt = this.db.prepare(
      "SELECT target_id FROM edges WHERE graph_id = ? AND source_id = ? AND kind = ?",
    );
    const visited = new Set<string>([startId]);
    const stack: string[] = [startId];
    while (stack.length > 0) {
      const current = stack.pop()!;
      const rows = stmt.all(graphId, current, kind) as Array<{ target_id: string }>;
      for (const row of rows) {
        if (row.target_id === goalId) {
          return true;
        }
        if (!visited.has(row.target_id)) {
          visited.add(row.target_id);
          stack.push(row.target_id);
        }
      }
    }
    return false;
  }

  private addEdgeInternal(graphId: string, input: AddEdgeInput): GraphEdge {
    if (!EDGE_KINDS.includes(input.kind)) {
      throw new ContextGraphError(`Invalid edge kind '${String(input.kind)}'`, "VALIDATION_ERROR", 400);
    }

    const srcRow = this.db
      .prepare("SELECT id, version FROM nodes WHERE graph_id = ? AND id = ?")
      .get(graphId, input.sourceId) as { id: string; version: number } | undefined;
    if (!srcRow) {
      throw new NotFoundError(`Source node '${input.sourceId}' not found in graph '${graphId}'`);
    }

    const tgtRow = this.db
      .prepare("SELECT id, version FROM nodes WHERE graph_id = ? AND id = ?")
      .get(graphId, input.targetId) as { id: string; version: number } | undefined;
    if (!tgtRow) {
      throw new NotFoundError(`Target node '${input.targetId}' not found in graph '${graphId}'`);
    }

    if (input.sourceId === input.targetId) {
      if (DAG_EDGE_KINDS.has(input.kind)) {
        throw new CycleDetectedError(
          `Self-loop rejected on '${input.kind}' edge for node '${input.sourceId}'`,
        );
      }
      throw new ContextGraphError("Self-loop edges are not allowed", "VALIDATION_ERROR", 400);
    }

    if (DAG_EDGE_KINDS.has(input.kind)) {
      if (this.isReachable(graphId, input.targetId, input.sourceId, input.kind)) {
        throw new CycleDetectedError(
          `Cycle detected on '${input.kind}': '${input.targetId}' already reaches '${input.sourceId}'`,
        );
      }
    }

    const existing = this.db
      .prepare(
        "SELECT * FROM edges WHERE graph_id = ? AND source_id = ? AND target_id = ? AND kind = ?",
      )
      .get(graphId, input.sourceId, input.targetId, input.kind) as RawEdgeRow | undefined;
    if (existing) {
      return this.hydrateEdge(existing);
    }

    const id = input.id?.trim() || `e_${randomUUID().slice(0, 10)}`;
    const weight =
      typeof input.weight === "number" && Number.isFinite(input.weight) && input.weight > 0
        ? input.weight
        : DEFAULT_EDGE_WEIGHT[input.kind];
    const now = new Date().toISOString();

    this.db
      .prepare(
        `INSERT INTO edges (id, graph_id, source_id, target_id, kind, weight, metadata_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        id,
        graphId,
        input.sourceId,
        input.targetId,
        input.kind,
        weight,
        JSON.stringify(input.metadata ?? {}),
        now,
      );

    if (input.kind === "supersedes") {
      const nextVersion = Math.max(srcRow.version, tgtRow.version + 1);
      this.db
        .prepare(
          "UPDATE nodes SET status = 'superseded', superseded_by = ?, updated_at = ? WHERE graph_id = ? AND id = ?",
        )
        .run(input.sourceId, now, graphId, input.targetId);
      if (nextVersion > srcRow.version) {
        this.db
          .prepare("UPDATE nodes SET version = ?, updated_at = ? WHERE graph_id = ? AND id = ?")
          .run(nextVersion, now, graphId, input.sourceId);
      }
    }

    this.db.prepare("UPDATE graphs SET updated_at = ? WHERE id = ?").run(now, graphId);
    const created = this.db
      .prepare("SELECT * FROM edges WHERE graph_id = ? AND id = ?")
      .get(graphId, id) as RawEdgeRow;
    return this.hydrateEdge(created);
  }

  addEdge(graphId: string, input: AddEdgeInput): GraphEdge {
    this.ensureGraphExists(graphId);
    return this.runInTransaction(() => this.addEdgeInternal(graphId, input));
  }

  /**
   * One-click structured or raw conversational/tool turn ingestion into the DAG.
   */
  ingestTurn(graphId: string, input: IngestTurnInput): IngestTurnResult {
    this.ensureGraphExists(graphId);
    return this.runInTransaction(() => {
      const turnRow = this.db
        .prepare("SELECT COALESCE(MAX(turn), 0) as max_turn FROM nodes WHERE graph_id = ?")
        .get(graphId) as { max_turn: number };
      const turn =
        typeof input.turn === "number" && input.turn >= 1
          ? Math.floor(input.turn)
          : Number(turnRow.max_turn) + 1;
      const scope = input.scope?.trim() || "global";

      const addedNodes: GraphNode[] = [];
      const addedEdges: GraphEdge[] = [];
      const supersededSet = new Set<string>();
      const retractedIds: string[] = [];

      const recordAdded = (res: { node: GraphNode; autoEdges: GraphEdge[]; supersededIds: string[] }) => {
        addedNodes.push(res.node);
        addedEdges.push(...res.autoEdges);
        for (const sid of res.supersededIds) supersededSet.add(sid);
        return res.node;
      };

      let activeFrameId = input.taskFrameId;

      if (input.system) {
        recordAdded(
          this.addNodeInternal(graphId, {
            turn,
            kind: "system",
            role: "system",
            scope: "global",
            tier: "pinned_prefix",
            content: input.system,
            resourceKey: "system:primary",
          }),
        );
      }

      if (input.goal) {
        recordAdded(
          this.addNodeInternal(graphId, {
            turn,
            kind: "goal",
            role: "user",
            scope: "global",
            tier: "pinned_prefix",
            content: input.goal,
          }),
        );
      }

      for (const c of input.constraints ?? []) {
        const item = typeof c === "string" ? { content: c } : c;
        recordAdded(
          this.addNodeInternal(graphId, {
            id: item.id,
            turn,
            kind: "constraint",
            role: "system",
            scope: "global",
            tier: "pinned_prefix",
            content: item.content,
            resourceKey: item.resourceKey ?? null,
          }),
        );
      }

      if (input.taskFrame) {
        const frameNode = recordAdded(
          this.addNodeInternal(graphId, {
            id: input.taskFrame.id,
            turn,
            kind: "task_frame",
            role: "system",
            scope: input.taskFrame.scope ?? scope,
            tier: "stable_frame",
            content: input.taskFrame.content,
            summary: input.taskFrame.summary,
          }),
        );
        activeFrameId = frameNode.id;
      }

      const scopedTo = activeFrameId ? [activeFrameId] : undefined;

      if (input.userMessage) {
        recordAdded(
          this.addNodeInternal(graphId, {
            turn,
            kind: "fact",
            role: "user",
            scope,
            tier: "working_subgraph",
            salience: 0.8,
            content: input.userMessage,
            scopedTo,
          }),
        );
      }

      for (const f of input.facts ?? []) {
        const item = typeof f === "string" ? { content: f } : f;
        recordAdded(
          this.addNodeInternal(graphId, {
            id: item.id,
            turn,
            kind: "fact",
            role: "user",
            scope,
            tier: "working_subgraph",
            content: item.content,
            resourceKey: item.resourceKey ?? null,
            dependsOn: item.dependsOn,
            scopedTo,
          }),
        );
      }

      for (const d of input.decisions ?? []) {
        recordAdded(
          this.addNodeInternal(graphId, {
            id: d.id,
            turn,
            kind: "decision",
            role: "assistant",
            scope,
            tier: d.tier ?? "working_subgraph",
            salience: d.salience ?? 0.88,
            content: d.content,
            summary: d.summary,
            resourceKey: d.resourceKey ?? null,
            dependsOn: d.dependsOn,
            conflictsWith: d.conflictsWith,
            scopedTo,
            metadata: d.metadata,
          }),
        );
      }

      for (const art of input.artifacts ?? []) {
        const normPath = art.path.replace(/^\.\//, "").trim();
        recordAdded(
          this.addNodeInternal(graphId, {
            id: art.id,
            turn,
            kind: "artifact",
            role: "tool",
            scope,
            tier: "working_subgraph",
            salience: art.salience ?? 0.86,
            content: art.content,
            summary: art.summary,
            stub: art.stub,
            resourceKey: `file:${normPath}`,
            dependsOn: art.dependsOn,
            scopedTo,
            metadata: { path: normPath, ...(art.metadata ?? {}) },
          }),
        );
      }

      for (const tc of input.toolCalls ?? []) {
        const isFileArtifact =
          Boolean(tc.path) &&
          !tc.isError &&
          /^(read|write|edit|view|cat|create)/i.test(tc.toolName);
        const resourceKey =
          tc.resourceKey ?? (isFileArtifact && tc.path ? `file:${tc.path.replace(/^\.\//, "")}` : null);
        recordAdded(
          this.addNodeInternal(graphId, {
            id: tc.id,
            turn,
            kind: isFileArtifact ? "artifact" : "observation",
            role: "tool",
            scope,
            tier: isFileArtifact ? "working_subgraph" : "ephemeral_leaf",
            salience: tc.salience ?? (tc.isError ? 0.35 : isFileArtifact ? 0.84 : 0.48),
            content: `[tool:${tc.toolName}] ${tc.output}`,
            summary: tc.summary,
            resourceKey,
            dependsOn: tc.dependsOn,
            scopedTo,
            metadata: {
              toolName: tc.toolName,
              args: tc.args ?? {},
              isError: Boolean(tc.isError),
              ...(tc.metadata ?? {}),
            },
          }),
        );
      }

      if (input.assistantMessage) {
        recordAdded(
          this.addNodeInternal(graphId, {
            turn,
            kind: "fact",
            role: "assistant",
            scope,
            tier: "working_subgraph",
            salience: 0.72,
            content: input.assistantMessage,
            scopedTo,
          }),
        );
      }

      for (const raw of input.rawMessages ?? []) {
        const inferredKind: NodeKind =
          raw.kind ??
          (raw.role === "system"
            ? "system"
            : raw.role === "tool"
              ? "observation"
              : "fact");
        recordAdded(
          this.addNodeInternal(graphId, {
            id: raw.id,
            turn,
            kind: inferredKind,
            role: raw.role,
            scope,
            tier: raw.tier,
            status: raw.hidden ? "retracted" : "active",
            resourceKey: raw.resourceKey ?? null,
            content: raw.content,
            scopedTo,
          }),
        );
      }

      for (const sum of input.summaries ?? []) {
        recordAdded(
          this.addNodeInternal(graphId, {
            id: sum.id,
            turn,
            kind: "summary",
            role: "assistant",
            scope: sum.scope ?? scope,
            tier: "working_subgraph",
            salience: 0.82,
            content: sum.content,
            summarizes: sum.targetNodeIds,
            scopedTo,
          }),
        );
      }

      const now = new Date().toISOString();
      for (const retractId of input.retractNodeIds ?? []) {
        const res = this.db
          .prepare(
            "UPDATE nodes SET status = 'retracted', updated_at = ? WHERE graph_id = ? AND id = ? AND status != 'retracted'",
          )
          .run(now, graphId, retractId);
        if (Number(res.changes) > 0) {
          retractedIds.push(retractId);
        }
      }

      for (const retractScope of input.retractScopes ?? []) {
        const rows = this.db
          .prepare(
            "SELECT id FROM nodes WHERE graph_id = ? AND scope = ? AND status = 'active'",
          )
          .all(graphId, retractScope) as Array<{ id: string }>;
        for (const r of rows) {
          this.db
            .prepare("UPDATE nodes SET status = 'retracted', updated_at = ? WHERE graph_id = ? AND id = ?")
            .run(now, graphId, r.id);
          retractedIds.push(r.id);
        }
      }

      return {
        graphId,
        turn,
        addedNodes,
        addedEdges,
        supersededNodeIds: Array.from(supersededSet),
        retractedNodeIds: retractedIds,
      };
    });
  }

  getStats(graphId: string): GraphStats {
    const nodes = this.listNodes(graphId);
    const edges = this.listEdges(graphId);
    let activeNodes = 0;
    let supersededNodes = 0;
    let retractedNodes = 0;
    let zone1ActiveNodes = 0;
    let zone2ActiveNodes = 0;
    let totalRawTokens = 0;
    let activeFullTokens = 0;

    for (const n of nodes) {
      totalRawTokens += n.tokens;
      if (n.status === "active") {
        activeNodes++;
        activeFullTokens += n.tokens;
        if (n.tier === "pinned_prefix" || n.tier === "stable_frame") {
          zone1ActiveNodes++;
        } else {
          zone2ActiveNodes++;
        }
      } else if (n.status === "superseded") {
        supersededNodes++;
      } else {
        retractedNodes++;
      }
    }

    return {
      totalNodes: nodes.length,
      activeNodes,
      supersededNodes,
      retractedNodes,
      totalEdges: edges.length,
      zone1ActiveNodes,
      zone2ActiveNodes,
      totalRawTokens,
      activeFullTokens,
    };
  }

  getGraphSnapshot(graphId: string): GraphSnapshot {
    const graph = this.getGraph(graphId);
    const nodes = this.listNodes(graphId);
    const edges = this.listEdges(graphId);
    const stats = this.getStats(graphId);
    return { graph, stats, nodes, edges };
  }

  /**
   * Exports the Property DAG topology as a GitHub-Flavored Mermaid flowchart.
   */
  toMermaid(graphId: string, options?: { includeInactive?: boolean }): string {
    const includeInactive = options?.includeInactive ?? true;
    const nodes = this.listNodes(graphId).filter((n) => includeInactive || n.status === "active");
    const nodeIdSet = new Set(nodes.map((n) => n.id));
    const edges = this.listEdges(graphId).filter(
      (e) => nodeIdSet.has(e.sourceId) && nodeIdSet.has(e.targetId),
    );

    const safeId = (id: string) => `N_${id.replace(/[^A-Za-z0-9_]/g, "_")}`;

    const zone1 = nodes.filter((n) => n.tier === "pinned_prefix" || n.tier === "stable_frame");
    const zone2 = nodes.filter((n) => n.tier !== "pinned_prefix" && n.tier !== "stable_frame");

    const lines: string[] = [
      "flowchart TD",
      "  classDef pinned fill:#eef6ff,stroke:#1d4ed8,stroke-width:2px,color:#0f172a;",
      "  classDef active fill:#ecfdf5,stroke:#059669,stroke-width:1.5px,color:#064e3b;",
      "  classDef superseded fill:#f8fafc,stroke:#94a3b8,stroke-dasharray: 4 4,color:#64748b;",
      "  classDef retracted fill:#fef2f2,stroke:#fca5a5,stroke-dasharray: 2 2,color:#991b1b;",
    ];

    const renderNodeLine = (n: GraphNode) => {
      const resInfo = n.resourceKey ? ` ${n.resourceKey} v${n.version}` : "";
      const snippet = sanitizeMermaidLabel(n.summary).slice(0, 48);
      const label = `${n.kind}#${n.seq}${resInfo} (${n.status}): ${snippet}`;
      const cls =
        n.status === "superseded"
          ? "superseded"
          : n.status === "retracted"
            ? "retracted"
            : n.tier === "pinned_prefix" || n.tier === "stable_frame"
              ? "pinned"
              : "active";
      return `    ${safeId(n.id)}["${label}"]:::${cls}`;
    };

    lines.push('  subgraph Zone1["Zone 1: Append-Mostly Stable Prefix Spine"]');
    if (zone1.length === 0) {
      lines.push('    Z1_EMPTY["(empty)"]');
    } else {
      for (const n of zone1) lines.push(renderNodeLine(n));
    }
    lines.push("  end");

    lines.push('  subgraph Zone2["Zone 2: Dynamic Working Subgraph"]');
    if (zone2.length === 0) {
      lines.push('    Z2_EMPTY["(empty)"]');
    } else {
      for (const n of zone2) lines.push(renderNodeLine(n));
    }
    lines.push("  end");

    for (const e of edges) {
      const s = safeId(e.sourceId);
      const t = safeId(e.targetId);
      if (e.kind === "supersedes") {
        lines.push(`  ${s} -.->|supersedes| ${t}`);
      } else if (e.kind === "summarizes") {
        lines.push(`  ${s} ==>|summarizes| ${t}`);
      } else if (e.kind === "conflicts_with") {
        lines.push(`  ${s} <-->|conflicts_with| ${t}`);
      } else {
        lines.push(`  ${s} -->|${e.kind}| ${t}`);
      }
    }

    return lines.join("\n");
  }
}
