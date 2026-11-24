import { createHash, randomUUID, timingSafeEqual } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { gunzipSync } from "node:zlib";
import {
  ingestToolArtifacts,
  pruneTrajectoryEvents,
  validateIngestArtifactsOptions,
  validateToolEvents,
} from "./artifact-engine.ts";
import { runBenchmark } from "./benchmark.ts";
import { compileContext, validateCompileOptions } from "./compiler.ts";
import { GraphStore } from "./graph-store.ts";
import { handleMcpJsonRpc, type JsonRpcResponse } from "./mcp.ts";
import {
  ContextGraphError,
  type AddEdgeInput,
  type AddNodeInput,
  type CreateGraphInput,
  type IngestTurnInput,
  type UpdateNodePatch,
} from "./types.ts";

export const SERVICE_NAME = "context-graph" as const;
export const SERVICE_VERSION = "0.1.0" as const;
export const DEFAULT_HOST = "127.0.0.1" as const;
export const DEFAULT_PORT = 52341 as const;
export const MIN_AUTH_TOKEN_LENGTH = 16 as const;

const MAX_BODY_BYTES = 2_097_152; // 2 MiB
const LOCAL_CORS_ORIGIN_RE = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i;
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);
const SAFE_SESSION_ID_RE = /^[\x21-\x7e]{1,128}$/;

export interface ContextGraphServerOptions {
  store?: GraphStore;
  dataPath?: string;
  host?: string;
  port?: number;
  authToken?: string;
}

export interface ContextGraphServerInstance {
  server: Server;
  store: GraphStore;
  start: (port?: number, host?: string) => Promise<{ host: string; port: number; url: string }>;
  close: () => Promise<void>;
}

export function isLoopbackHost(host: string): boolean {
  return LOOPBACK_HOSTS.has(host.trim().toLowerCase());
}

function sha256Digest(value: string): Buffer {
  return createHash("sha256").update(value).digest();
}

function extractBearerToken(req: IncomingMessage): string | null {
  const header = req.headers.authorization;
  if (typeof header !== "string") return null;
  const match = header.trim().match(/^Bearer\s+(.+)$/i);
  if (!match) return null;
  const token = match[1]!.trim();
  return token.length > 0 ? token : null;
}

function verifyBearerToken(req: IncomingMessage, expectedDigest: Buffer): boolean {
  const candidate = extractBearerToken(req);
  if (!candidate) return false;
  const candidateDigest = sha256Digest(candidate);
  return (
    candidateDigest.length === expectedDigest.length &&
    timingSafeEqual(candidateDigest, expectedDigest)
  );
}

function resolveSessionId(req: IncomingMessage): string {
  const raw = req.headers["mcp-session-id"];
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (SAFE_SESSION_ID_RE.test(trimmed)) {
      return trimmed;
    }
  }
  return randomUUID();
}

function hasSuccessfulInitialize(
  payload: unknown,
  rpcResult: JsonRpcResponse | JsonRpcResponse[],
): boolean {
  if (!Array.isArray(payload) && !Array.isArray(rpcResult)) {
    return (
      payload !== null &&
      typeof payload === "object" &&
      (payload as Record<string, unknown>).method === "initialize" &&
      rpcResult.error === undefined
    );
  }
  if (Array.isArray(payload) && Array.isArray(rpcResult)) {
    const initIds = new Set<string | number | null>();
    for (const item of payload) {
      if (
        item !== null &&
        typeof item === "object" &&
        !Array.isArray(item) &&
        (item as Record<string, unknown>).method === "initialize" &&
        (item as Record<string, unknown>).id !== undefined
      ) {
        initIds.add(((item as Record<string, unknown>).id as string | number | null) ?? null);
      }
    }
    if (initIds.size === 0) return false;
    return rpcResult.some((r) => initIds.has(r.id) && r.error === undefined);
  }
  return false;
}

function resolveContentEncoding(req: IncomingMessage): "identity" | "gzip" {
  const raw = req.headers["content-encoding"];
  if (raw === undefined) return "identity";
  const joined = Array.isArray(raw) ? raw.join(",") : raw;
  const trimmed = joined.trim().toLowerCase();
  if (!trimmed || trimmed === "identity") return "identity";
  if (trimmed === "gzip") return "gzip";
  throw new ContextGraphError(
    `Unsupported Content-Encoding '${joined.trim()}'`,
    "UNSUPPORTED_MEDIA_TYPE",
    415,
  );
}

async function readRawBody(req: IncomingMessage): Promise<string> {
  const encoding = resolveContentEncoding(req);
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buf.length;
    if (total > MAX_BODY_BYTES) {
      throw new ContextGraphError("Request body exceeds 2 MiB limit", "PAYLOAD_TOO_LARGE", 413);
    }
    chunks.push(buf);
  }
  if (total === 0) {
    if (encoding === "gzip") {
      throw new ContextGraphError("Invalid gzip request body", "INVALID_GZIP", 400);
    }
    return "";
  }
  const rawBuf = Buffer.concat(chunks, total);
  if (encoding === "gzip") {
    try {
      const decoded = gunzipSync(rawBuf, { maxOutputLength: MAX_BODY_BYTES });
      return decoded.toString("utf8").trim();
    } catch (err) {
      const code =
        err && typeof err === "object" && "code" in err
          ? String((err as { code?: unknown }).code)
          : "";
      if (code === "ERR_BUFFER_TOO_LARGE") {
        throw new ContextGraphError(
          "Decompressed request body exceeds maximum size",
          "PAYLOAD_TOO_LARGE",
          413,
        );
      }
      throw new ContextGraphError("Invalid gzip request body", "INVALID_GZIP", 400);
    }
  }
  return rawBuf.toString("utf8").trim();
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const raw = await readRawBody(req);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new ContextGraphError("JSON body must be an object", "INVALID_JSON", 400);
    }
    return parsed as Record<string, unknown>;
  } catch (err) {
    if (err instanceof ContextGraphError) throw err;
    throw new ContextGraphError("Invalid JSON payload", "INVALID_JSON", 400);
  }
}

function sendJson(
  res: ServerResponse,
  statusCode: number,
  body: unknown,
  origin?: string,
  extraHeaders?: Record<string, string | number>,
): void {
  const payload = JSON.stringify(body);
  const headers: Record<string, string | number> = {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(payload),
    ...extraHeaders,
  };
  if (origin && LOCAL_CORS_ORIGIN_RE.test(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
    if (extraHeaders && "Mcp-Session-Id" in extraHeaders) {
      headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id";
    }
  }
  res.writeHead(statusCode, headers);
  res.end(payload);
}

function sendNoBody(
  res: ServerResponse,
  statusCode: number,
  origin?: string,
  extraHeaders?: Record<string, string | number>,
): void {
  const headers: Record<string, string | number> = {
    "Content-Length": 0,
    ...extraHeaders,
  };
  if (origin && LOCAL_CORS_ORIGIN_RE.test(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
  }
  res.writeHead(statusCode, headers);
  res.end();
}

function sendText(res: ServerResponse, statusCode: number, text: string, origin?: string): void {
  const headers: Record<string, string | number> = {
    "Content-Type": "text/plain; charset=utf-8",
    "Content-Length": Buffer.byteLength(text),
  };
  if (origin && LOCAL_CORS_ORIGIN_RE.test(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
  }
  res.writeHead(statusCode, headers);
  res.end(text);
}

export function createContextGraphServer(
  options: ContextGraphServerOptions = {},
): ContextGraphServerInstance {
  const ownsStore = !options.store;
  const dataPath =
    options.dataPath ??
    (process.env.DATA_PATH?.trim() || "data/context-graph.db");
  const store = options.store ?? new GraphStore(dataPath);
  const startedAtMs = Date.now();
  const authToken =
    options.authToken !== undefined
      ? options.authToken.trim() || undefined
      : process.env.CG_AUTH_TOKEN?.trim() || undefined;
  const authTokenDigest = authToken ? sha256Digest(authToken) : null;

  const server = createServer(async (req, res) => {
    const origin = req.headers.origin;
    const method = req.method ?? "GET";

    if (method === "OPTIONS") {
      const headers: Record<string, string> = {
        "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers":
          "Content-Type, Authorization, Mcp-Session-Id, MCP-Protocol-Version",
      };
      if (origin && LOCAL_CORS_ORIGIN_RE.test(origin)) {
        headers["Access-Control-Allow-Origin"] = origin;
      }
      res.writeHead(204, headers);
      res.end();
      return;
    }

    const parsedUrl = new URL(req.url ?? "/", "http://127.0.0.1");
    const pathname = parsedUrl.pathname.replace(/\/+$/, "") || "/";
    const isAuthorized = authTokenDigest === null || verifyBearerToken(req, authTokenDigest);

    try {
      // GET /health
      if (method === "GET" && pathname === "/health") {
        if (!isAuthorized) {
          sendJson(
            res,
            200,
            {
              ok: true,
              service: SERVICE_NAME,
              version: SERVICE_VERSION,
            },
            origin,
          );
          return;
        }
        const counts = store.countsSummary();
        sendJson(
          res,
          200,
          {
            ok: true,
            service: SERVICE_NAME,
            version: SERVICE_VERSION,
            dbPath: store.dbPath,
            uptimeSec: Math.round((Date.now() - startedAtMs) / 100) / 10,
            ...counts,
          },
          origin,
        );
        return;
      }

      // Require valid Bearer token on all other endpoints when authToken is configured
      if (!isAuthorized) {
        sendJson(
          res,
          401,
          {
            ok: false,
            error: {
              code: "UNAUTHORIZED",
              message: "Valid Bearer token required",
            },
          },
          origin,
          { "WWW-Authenticate": "Bearer" },
        );
        return;
      }

      // Streamable HTTP MCP endpoint: /mcp
      if (pathname === "/mcp") {
        if (method === "POST") {
          const rawBody = await readRawBody(req);
          if (!rawBody) {
            sendJson(
              res,
              400,
              { jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } },
              origin,
            );
            return;
          }
          let parsed: unknown;
          try {
            parsed = JSON.parse(rawBody);
          } catch {
            sendJson(
              res,
              400,
              { jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } },
              origin,
            );
            return;
          }
          const rpcResult = handleMcpJsonRpc(store, parsed);
          if (rpcResult === null) {
            sendNoBody(res, 202, origin);
            return;
          }
          const extraHeaders: Record<string, string> = {};
          if (hasSuccessfulInitialize(parsed, rpcResult)) {
            extraHeaders["Mcp-Session-Id"] = resolveSessionId(req);
          }
          sendJson(res, 200, rpcResult, origin, extraHeaders);
          return;
        }

        if (method === "GET") {
          sendJson(
            res,
            405,
            {
              ok: false,
              error: {
                code: "METHOD_NOT_ALLOWED",
                message: "GET /mcp SSE stream is not supported; use POST /mcp",
              },
            },
            origin,
            { Allow: "POST, DELETE" },
          );
          return;
        }

        if (method === "DELETE") {
          sendJson(res, 200, { ok: true }, origin);
          return;
        }

        sendJson(
          res,
          405,
          {
            ok: false,
            error: {
              code: "METHOD_NOT_ALLOWED",
              message: `Method ${method} is not allowed on /mcp`,
            },
          },
          origin,
          { Allow: "POST, DELETE" },
        );
        return;
      }

      // POST /v1/graphs
      if (method === "POST" && pathname === "/v1/graphs") {
        const body = await readJsonBody(req);
        const graph = store.createGraph(body as unknown as CreateGraphInput);
        sendJson(res, 201, { ok: true, graph }, origin);
        return;
      }

      // GET /v1/graphs
      if (method === "GET" && pathname === "/v1/graphs") {
        const graphs = store.listGraphs();
        sendJson(res, 200, { ok: true, graphs }, origin);
        return;
      }

      // POST /v1/benchmark
      if (method === "POST" && pathname === "/v1/benchmark") {
        const body = await readJsonBody(req);
        const budgetRatio =
          typeof body.budgetRatio === "number" ? body.budgetRatio : undefined;
        const report = runBenchmark({ budgetRatio });
        sendJson(res, 200, { ok: true, report }, origin);
        return;
      }

      // Graph-scoped routes: /v1/graphs/:id[/...]
      const graphMatch = pathname.match(/^\/v1\/graphs\/([^/]+)(?:\/(.+))?$/);
      if (graphMatch) {
        const graphId = decodeURIComponent(graphMatch[1]!);
        const subPath = graphMatch[2] ?? "";

        if (!subPath && method === "GET") {
          const snapshot = store.getGraphSnapshot(graphId);
          sendJson(res, 200, { ok: true, ...snapshot }, origin);
          return;
        }

        if (!subPath && method === "DELETE") {
          const deleted = store.deleteGraph(graphId);
          sendJson(res, 200, { ok: true, deleted }, origin);
          return;
        }

        if (subPath === "nodes" && method === "POST") {
          const body = await readJsonBody(req);
          const node = store.addNode(graphId, body as unknown as AddNodeInput);
          sendJson(res, 201, { ok: true, node }, origin);
          return;
        }

        if (subPath === "nodes" && method === "GET") {
          const nodes = store.listNodes(graphId);
          sendJson(res, 200, { ok: true, nodes }, origin);
          return;
        }

        const nodePatchMatch = subPath.match(/^nodes\/([^/]+)$/);
        if (nodePatchMatch && method === "PATCH") {
          const nodeId = decodeURIComponent(nodePatchMatch[1]!);
          const body = await readJsonBody(req);
          const node = store.updateNode(graphId, nodeId, body as unknown as UpdateNodePatch);
          sendJson(res, 200, { ok: true, node }, origin);
          return;
        }

        if (subPath === "edges" && method === "POST") {
          const body = await readJsonBody(req);
          const edge = store.addEdge(graphId, body as unknown as AddEdgeInput);
          sendJson(res, 201, { ok: true, edge }, origin);
          return;
        }

        if (subPath === "edges" && method === "GET") {
          const edges = store.listEdges(graphId);
          sendJson(res, 200, { ok: true, edges }, origin);
          return;
        }

        if (subPath === "ingest" && method === "POST") {
          const body = await readJsonBody(req);
          const result = store.ingestTurn(graphId, body as unknown as IngestTurnInput);
          sendJson(res, 201, { ok: true, ...result }, origin);
          return;
        }

        if (subPath === "ingest-artifacts" && method === "POST") {
          const body = await readJsonBody(req);
          const events = validateToolEvents(body.events);
          const ingestOpts = validateIngestArtifactsOptions(body);
          const validatedCompileOpts =
            body.compileOptions !== undefined
              ? validateCompileOptions(body.compileOptions)
              : undefined;

          const ingested = ingestToolArtifacts(store, graphId, events, ingestOpts);
          const pruned = body.prune === true ? pruneTrajectoryEvents(events) : undefined;
          const compiled =
            body.compile === true
              ? compileContext(store, {
                  ...(validatedCompileOpts ?? {}),
                  graphId,
                })
              : undefined;

          if (ingestOpts.responseMode === "slim") {
            const slimPruned = pruned
              ? {
                  ...pruned,
                  events: pruned.events.filter((ev, idx) => {
                    const orig = events[idx];
                    if (!orig) return true;
                    return (
                      ev.pruned === true ||
                      ev.output !== orig.output ||
                      ev.args !== orig.args
                    );
                  }),
                }
              : undefined;
            const slimCompiled = compiled
              ? {
                  promptText: compiled.promptText,
                  format: validatedCompileOpts?.format ?? "both",
                  totalTokens: compiled.kvCache.totalTokens,
                  budgetTokens: compiled.maxTokens,
                }
              : undefined;

            sendJson(
              res,
              200,
              {
                ok: true,
                graphId,
                ingestedSummary: {
                  ingestedCount: ingested.ingestedCount,
                  skippedCount: ingested.skippedCount,
                },
                ...(slimPruned ? { pruned: slimPruned } : {}),
                ...(slimCompiled ? { compiled: slimCompiled } : {}),
              },
              origin,
            );
            return;
          }

          sendJson(
            res,
            200,
            {
              ok: true,
              graphId,
              ingested,
              ...(pruned ? { pruned } : {}),
              ...(compiled ? { compiled } : {}),
            },
            origin,
          );
          return;
        }

        if (subPath === "compile" && method === "POST") {
          const body = await readJsonBody(req);
          const validatedOpts = validateCompileOptions(body);
          const compiled = compileContext(store, {
            ...validatedOpts,
            graphId,
          });
          sendJson(res, 200, { ok: true, ...compiled }, origin);
          return;
        }

        if (subPath === "mermaid" && method === "GET") {
          const includeInactive = parsedUrl.searchParams.get("includeInactive") !== "false";
          const mermaid = store.toMermaid(graphId, { includeInactive });
          const wantsPlain =
            parsedUrl.searchParams.get("format") === "text" ||
            (req.headers.accept?.includes("text/plain") && !req.headers.accept?.includes("application/json"));
          if (wantsPlain) {
            sendText(res, 200, mermaid, origin);
          } else {
            sendJson(res, 200, { ok: true, graphId, mermaid }, origin);
          }
          return;
        }
      }

      sendJson(
        res,
        404,
        { ok: false, error: { code: "ROUTE_NOT_FOUND", message: `Route ${method} ${pathname} not found` } },
        origin,
      );
    } catch (err) {
      if (err instanceof ContextGraphError) {
        sendJson(
          res,
          err.statusCode,
          { ok: false, error: { code: err.code, message: err.message } },
          origin,
        );
        return;
      }
      const msg = err instanceof Error ? err.message : "Internal server error";
      sendJson(
        res,
        500,
        { ok: false, error: { code: "INTERNAL_ERROR", message: msg } },
        origin,
      );
    }
  });

  const defaultHost = options.host ?? (process.env.HOST?.trim() || DEFAULT_HOST);
  const rawEnvPort = process.env.PORT?.trim();
  const defaultPort =
    options.port ??
    (rawEnvPort && rawEnvPort.length > 0 ? Number.parseInt(rawEnvPort, 10) : DEFAULT_PORT);

  return {
    server,
    store,
    start: (port = defaultPort, host = defaultHost) => {
      const hostToBind = host.trim();
      if (!isLoopbackHost(hostToBind) && (!authToken || authToken.length < MIN_AUTH_TOKEN_LENGTH)) {
        return Promise.reject(
          new ContextGraphError(
            `Refusing to start on non-loopback host "${hostToBind}" without authToken (CG_AUTH_TOKEN) of at least ${MIN_AUTH_TOKEN_LENGTH} characters`,
            "INSECURE_BIND_REFUSED",
            400,
          ),
        );
      }
      if (!Number.isInteger(port) || port < 0 || port > 65535) {
        return Promise.reject(
          new ContextGraphError(`Invalid listen port: ${port}`, "INVALID_PORT", 400),
        );
      }
      return new Promise((resolveStart, rejectStart) => {
        const onError = (err: Error) => {
          server.off("error", onError);
          rejectStart(err);
        };
        server.once("error", onError);
        server.listen(port, hostToBind, () => {
          server.off("error", onError);
          const addr = server.address() as AddressInfo;
          const formattedHost = addr.family === "IPv6" ? `[${addr.address}]` : addr.address;
          resolveStart({
            host: addr.address,
            port: addr.port,
            url: `http://${formattedHost}:${addr.port}`,
          });
        });
      });
    },
    close: () =>
      new Promise((resolveClose, rejectClose) => {
        const finalize = () => {
          if (ownsStore) store.close();
          resolveClose();
        };
        if (!server.listening) {
          finalize();
          return;
        }
        server.close((err) => {
          if (err) {
            rejectClose(err);
            return;
          }
          finalize();
        });
      }),
  };
}
