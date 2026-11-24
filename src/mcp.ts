import type { Readable, Writable } from "node:stream";
import {
  ingestToolArtifacts,
  pruneTrajectoryEvents,
  validateIngestArtifactsOptions,
  validateToolEvents,
} from "./artifact-engine.ts";
import { compileContext, validateCompileOptions } from "./compiler.ts";
import { GraphStore } from "./graph-store.ts";
import {
  ContextGraphError,
  EDGE_KINDS,
  NODE_KINDS,
  type AddEdgeInput,
  type AddNodeInput,
  type CreateGraphInput,
  type IngestTurnInput,
} from "./types.ts";

export const MCP_PROTOCOL_VERSION = "2024-11-05" as const;
export const SUPPORTED_MCP_PROTOCOL_VERSIONS = [
  "2025-06-18",
  "2025-03-26",
  "2024-11-05",
] as const;

export interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: string | number | null;
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: string | number | null;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

export const MCP_TOOLS = [
  {
    name: "cg_create_graph",
    description:
      "Create a new Two-Zone KV-Cache-Aware Context Graph workspace for an agent session or task.",
    annotations: { readOnlyHint: false },
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Optional explicit graph identifier." },
        name: { type: "string", description: "Human-readable graph name." },
        description: { type: "string", description: "Optional description." },
        defaultBudget: { type: "integer", description: "Default compilation token budget (default: 4096)." },
      },
      required: ["name"],
    },
  },
  {
    name: "cg_add_node",
    description:
      "Add a typed node to a Context Graph. Automatically versions and links `supersedes` when `resourceKey` matches a prior active node.",
    annotations: { readOnlyHint: false },
    inputSchema: {
      type: "object",
      properties: {
        graphId: { type: "string" },
        id: { type: "string" },
        turn: { type: "integer" },
        kind: { type: "string", enum: [...NODE_KINDS] },
        scope: { type: "string" },
        tier: {
          type: "string",
          enum: ["pinned_prefix", "stable_frame", "working_subgraph", "ephemeral_leaf"],
        },
        content: { type: "string" },
        summary: { type: "string" },
        stub: { type: "string" },
        salience: { type: "number" },
        resourceKey: { type: "string", description: "e.g. file:src/auth.ts" },
        dependsOn: { type: "array", items: { type: "string" } },
      },
      required: ["graphId", "kind", "content"],
    },
  },
  {
    name: "cg_link_nodes",
    description:
      "Create a directed typed edge (`depends_on`, `supersedes`, `summarizes`, `elaborates`, `scoped_to`, `conflicts_with`). Enforces strict DAG cycle rejection.",
    annotations: { readOnlyHint: false },
    inputSchema: {
      type: "object",
      properties: {
        graphId: { type: "string" },
        sourceId: { type: "string" },
        targetId: { type: "string" },
        kind: { type: "string", enum: [...EDGE_KINDS] },
        weight: { type: "number" },
      },
      required: ["graphId", "sourceId", "targetId", "kind"],
    },
  },
  {
    name: "cg_ingest_turn",
    description:
      "Ingest a full agent/user/tool turn (constraints, decisions, file artifacts, tool outputs, or dead-end scope retractions) in one atomic transaction.",
    annotations: { readOnlyHint: false },
    inputSchema: {
      type: "object",
      properties: {
        graphId: { type: "string" },
        turn: { type: "integer" },
        scope: { type: "string" },
        system: { type: "string" },
        goal: { type: "string" },
        constraints: { type: "array", items: { type: "string" } },
        userMessage: { type: "string" },
        assistantMessage: { type: "string" },
        artifacts: {
          type: "array",
          items: {
            type: "object",
            properties: {
              path: { type: "string" },
              content: { type: "string" },
              summary: { type: "string" },
              dependsOn: { type: "array", items: { type: "string" } },
            },
            required: ["path", "content"],
          },
        },
        retractNodeIds: { type: "array", items: { type: "string" } },
        retractScopes: { type: "array", items: { type: "string" } },
      },
      required: ["graphId"],
    },
  },
  {
    name: "cg_ingest_artifacts",
    description:
      "Ingest structured tool-trajectory events (file reads, ranged reads, partial edits, full writes, commands, searches) with idempotent versioning, cross-compaction supersession cleanup, optional trajectory pruning, and optional compilation.",
    annotations: { readOnlyHint: false },
    inputSchema: {
      type: "object",
      properties: {
        graphId: { type: "string" },
        events: {
          type: "array",
          items: {
            type: "object",
            properties: {
              callId: { type: "string" },
              tool: { type: "string" },
              args: { type: "object" },
              ok: { type: "boolean" },
              output: { type: "string" },
              seq: { type: "integer" },
              turn: { type: "integer" },
              machineId: { type: "string" },
              scope: { type: "string" },
            },
            required: ["callId", "tool", "ok", "output"],
          },
        },
        scope: { type: "string" },
        turn: { type: "integer" },
        machineId: { type: "string" },
        maxCodeExcerptChars: { type: "integer" },
        maxCommandExcerptChars: { type: "integer" },
        autoCreate: { type: "boolean" },
        graphName: { type: "string" },
        prune: { type: "boolean" },
        compile: { type: "boolean" },
        compileOptions: { type: "object" },
      },
      required: ["graphId", "events"],
    },
  },
  {
    name: "cg_compile_context",
    description:
      "Compile the Context Graph into a Two-Zone KV-Cache-friendly XML or compact_markdown prompt & OpenAI messages[] within budget, reporting exact Prefix KV-Cache hit rate.",
    annotations: { readOnlyHint: true },
    inputSchema: {
      type: "object",
      properties: {
        graphId: { type: "string" },
        focusQuery: { type: "string" },
        focusNodeIds: { type: "array", items: { type: "string" } },
        activeScope: { type: "string" },
        maxTokens: { type: "integer" },
        format: {
          type: "string",
          enum: ["both", "messages", "prompt", "xml", "compact_markdown"],
        },
        maxOutputChars: {
          type: "integer",
          description: "Maximum output character budget for compact_markdown format (default: 720, max: 4000).",
        },
      },
      required: ["graphId"],
    },
  },
  {
    name: "cg_export_mermaid",
    description:
      "Export the Two-Zone Property DAG topology as a GitHub-flavored Mermaid flowchart.",
    annotations: { readOnlyHint: true },
    inputSchema: {
      type: "object",
      properties: {
        graphId: { type: "string" },
        includeInactive: { type: "boolean" },
      },
      required: ["graphId"],
    },
  },
] as const;

export class ContextGraphMcpSession {
  readonly store: GraphStore;

  constructor(store: GraphStore) {
    this.store = store;
  }

  handlePayload(payload: unknown): JsonRpcResponse | JsonRpcResponse[] | null {
    if (Array.isArray(payload)) {
      if (payload.length === 0) {
        return {
          jsonrpc: "2.0",
          id: null,
          error: { code: -32600, message: "Invalid JSON-RPC 2.0 request" },
        };
      }
      const responses = payload
        .map((item) => this.handleMessage(item as JsonRpcRequest))
        .filter((r): r is JsonRpcResponse => r !== null);
      return responses.length > 0 ? responses : null;
    }
    return this.handleMessage(payload as JsonRpcRequest);
  }

  handleMessage(req: JsonRpcRequest): JsonRpcResponse | null {
    if (!req || typeof req !== "object" || Array.isArray(req)) {
      return {
        jsonrpc: "2.0",
        id: null,
        error: { code: -32600, message: "Invalid JSON-RPC 2.0 request" },
      };
    }

    const isNotification = req.id === undefined;
    const id = req.id ?? null;

    if (
      req.jsonrpc === "2.0" &&
      req.method === undefined &&
      ("result" in req || "error" in req)
    ) {
      return null;
    }

    if (req.jsonrpc !== "2.0" || typeof req.method !== "string") {
      return {
        jsonrpc: "2.0",
        id,
        error: { code: -32600, message: "Invalid JSON-RPC 2.0 request" },
      };
    }

    try {
      if (req.method.startsWith("notifications/")) {
        return null;
      }

      if (req.method === "initialize") {
        if (isNotification) return null;
        const requestedVersion =
          typeof req.params?.protocolVersion === "string"
            ? req.params.protocolVersion
            : undefined;
        const negotiatedVersion =
          requestedVersion &&
          (SUPPORTED_MCP_PROTOCOL_VERSIONS as readonly string[]).includes(requestedVersion)
            ? requestedVersion
            : MCP_PROTOCOL_VERSION;
        return {
          jsonrpc: "2.0",
          id,
          result: {
            protocolVersion: negotiatedVersion,
            capabilities: {
              tools: { listChanged: false },
            },
            serverInfo: {
              name: "@opc/context-graph",
              version: "0.1.0",
            },
          },
        };
      }

      if (req.method === "ping") {
        if (isNotification) return null;
        return { jsonrpc: "2.0", id, result: {} };
      }

      if (req.method === "tools/list") {
        if (isNotification) return null;
        return {
          jsonrpc: "2.0",
          id,
          result: { tools: MCP_TOOLS },
        };
      }

      if (req.method === "tools/call") {
        if (isNotification) return null;
        const params = req.params ?? {};
        const toolName = String(params.name ?? "");
        const args = (params.arguments ?? {}) as Record<string, unknown>;
        const toolOutput = this.invokeTool(toolName, args);
        return {
          jsonrpc: "2.0",
          id,
          result: {
            content: [{ type: "text", text: JSON.stringify(toolOutput, null, 2) }],
            structuredContent: toolOutput,
            isError: false,
          },
        };
      }

      if (isNotification) return null;
      return {
        jsonrpc: "2.0",
        id,
        error: { code: -32601, message: `Method not found: ${req.method}` },
      };
    } catch (err) {
      if (req.method === "tools/call" && !isNotification) {
        const message = err instanceof Error ? err.message : String(err);
        const code = err instanceof ContextGraphError ? err.code : "TOOL_EXECUTION_ERROR";
        return {
          jsonrpc: "2.0",
          id,
          result: {
            content: [{ type: "text", text: JSON.stringify({ ok: false, code, message }) }],
            isError: true,
          },
        };
      }
      const message = err instanceof Error ? err.message : "Internal MCP error";
      return {
        jsonrpc: "2.0",
        id,
        error: { code: -32603, message },
      };
    }
  }

  private invokeTool(name: string, args: Record<string, unknown>): unknown {
    switch (name) {
      case "cg_create_graph": {
        const graph = this.store.createGraph(args as unknown as CreateGraphInput);
        return { ok: true, graph };
      }
      case "cg_add_node": {
        const graphId = String(args.graphId ?? "");
        const { graphId: _ignored, ...rest } = args;
        const node = this.store.addNode(graphId, rest as unknown as AddNodeInput);
        return { ok: true, node };
      }
      case "cg_link_nodes": {
        const graphId = String(args.graphId ?? "");
        const { graphId: _ignored, ...rest } = args;
        const edge = this.store.addEdge(graphId, rest as unknown as AddEdgeInput);
        return { ok: true, edge };
      }
      case "cg_ingest_turn": {
        const graphId = String(args.graphId ?? "");
        const { graphId: _ignored, ...rest } = args;
        const res = this.store.ingestTurn(graphId, rest as unknown as IngestTurnInput);
        return { ok: true, ...res };
      }
      case "cg_ingest_artifacts": {
        const graphId = String(args.graphId ?? "");
        const events = validateToolEvents(args.events);
        const ingestOpts = validateIngestArtifactsOptions(args);
        const validatedCompileOpts =
          args.compileOptions !== undefined
            ? validateCompileOptions(args.compileOptions)
            : undefined;
        const ingested = ingestToolArtifacts(this.store, graphId, events, ingestOpts);
        const pruned = args.prune === true ? pruneTrajectoryEvents(events) : undefined;
        const compiled =
          args.compile === true
            ? compileContext(this.store, {
                ...(validatedCompileOpts ?? {}),
                graphId,
              })
            : undefined;
        return {
          ok: true,
          graphId,
          ingested,
          ...(pruned ? { pruned } : {}),
          ...(compiled ? { compiled } : {}),
        };
      }
      case "cg_compile_context": {
        const graphId = String(args.graphId ?? "");
        const { graphId: _ignored, ...rest } = args;
        const validatedOpts = validateCompileOptions(rest);
        const res = compileContext(this.store, {
          ...validatedOpts,
          graphId,
        });
        return {
          ok: true,
          graphId: res.graphId,
          kvCache: res.kvCache,
          diagnostics: res.diagnostics,
          conflicts: res.conflicts,
          messages: res.messages,
          promptText: res.promptText,
        };
      }
      case "cg_export_mermaid": {
        const graphId = String(args.graphId ?? "");
        const includeInactive = args.includeInactive !== false;
        const mermaid = this.store.toMermaid(graphId, { includeInactive });
        return { ok: true, graphId, mermaid };
      }
      default:
        throw new ContextGraphError(`Unknown MCP tool '${name}'`, "UNKNOWN_TOOL", 400);
    }
  }
}

/**
 * Stateless helper to execute a single JSON-RPC 2.0 MCP request or batch array
 * against a GraphStore. Returns `null` when all requests are notifications.
 */
export function handleMcpJsonRpc(
  store: GraphStore,
  req: unknown,
): JsonRpcResponse | JsonRpcResponse[] | null {
  const session = new ContextGraphMcpSession(store);
  return session.handlePayload(req);
}

/**
 * Starts JSON-RPC 2.0 MCP Server over stdio (supports newline-delimited JSON
 * as well as Content-Length framed messages).
 */
export function startMcpStdioServer(options?: {
  store?: GraphStore;
  dataPath?: string;
  input?: Readable;
  output?: Writable;
}): { session: ContextGraphMcpSession; stop: () => void } {
  const ownsStore = !options?.store;
  const store =
    options?.store ??
    new GraphStore(options?.dataPath ?? process.env.DATA_PATH ?? "data/context-graph.db");
  const session = new ContextGraphMcpSession(store);
  const input = options?.input ?? process.stdin;
  const output = options?.output ?? process.stdout;

  let buffer = Buffer.alloc(0);
  let useContentLengthFraming = false;

  const writeResponse = (resp: JsonRpcResponse | JsonRpcResponse[]) => {
    const serialized = JSON.stringify(resp);
    if (useContentLengthFraming) {
      const byteLen = Buffer.byteLength(serialized, "utf8");
      output.write(`Content-Length: ${byteLen}\r\n\r\n${serialized}`);
    } else {
      output.write(`${serialized}\n`);
    }
  };

  const dispatchRawJson = (jsonStr: string) => {
    const trimmed = jsonStr.trim();
    if (!trimmed) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      writeResponse({
        jsonrpc: "2.0",
        id: null,
        error: { code: -32700, message: "Parse error" },
      });
      return;
    }

    const resp = session.handlePayload(parsed);
    if (resp) writeResponse(resp);
  };

  const processBuffer = () => {
    while (buffer.length > 0) {
      const headAscii = buffer.subarray(0, Math.min(buffer.length, 32)).toString("utf8");
      if (/^Content-Length:/i.test(headAscii)) {
        useContentLengthFraming = true;
        const sepIdx = buffer.indexOf("\r\n\r\n");
        if (sepIdx === -1) return;
        const headerText = buffer.subarray(0, sepIdx).toString("utf8");
        const m = headerText.match(/Content-Length:\s*(\d+)/i);
        const contentLen = m ? Number.parseInt(m[1]!, 10) : 0;
        const bodyStart = sepIdx + 4;
        if (buffer.length < bodyStart + contentLen) return;
        const bodyStr = buffer.subarray(bodyStart, bodyStart + contentLen).toString("utf8");
        buffer = buffer.subarray(bodyStart + contentLen);
        dispatchRawJson(bodyStr);
      } else {
        const nlIdx = buffer.indexOf(0x0a); // '\n'
        if (nlIdx === -1) return;
        const line = buffer.subarray(0, nlIdx).toString("utf8");
        buffer = buffer.subarray(nlIdx + 1);
        dispatchRawJson(line);
      }
    }
  };

  const onData = (chunk: Buffer | string) => {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, "utf8");
    buffer = Buffer.concat([buffer, buf]);
    processBuffer();
  };

  const onEnd = () => {
    if (buffer.length > 0) {
      const rem = buffer.toString("utf8");
      buffer = Buffer.alloc(0);
      dispatchRawJson(rem);
    }
    if (ownsStore) store.close();
  };

  input.on("data", onData);
  input.on("end", onEnd);

  return {
    session,
    stop: () => {
      input.off("data", onData);
      input.off("end", onEnd);
      if (ownsStore) store.close();
    },
  };
}
