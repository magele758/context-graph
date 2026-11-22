import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { runBenchmark } from "./benchmark.ts";
import { compileContext } from "./compiler.ts";
import { GraphStore } from "./graph-store.ts";
import {
  ContextGraphError,
  type AddEdgeInput,
  type AddNodeInput,
  type CompileOptions,
  type CreateGraphInput,
  type IngestTurnInput,
  type UpdateNodePatch,
} from "./types.ts";

export const SERVICE_NAME = "context-graph" as const;
export const SERVICE_VERSION = "0.1.0" as const;
export const DEFAULT_HOST = "127.0.0.1" as const;
export const DEFAULT_PORT = 52341 as const;

const MAX_BODY_BYTES = 2_097_152; // 2 MiB
const LOCAL_CORS_ORIGIN_RE = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i;

export interface ContextGraphServerOptions {
  store?: GraphStore;
  dataPath?: string;
  host?: string;
  port?: number;
}

export interface ContextGraphServerInstance {
  server: Server;
  store: GraphStore;
  start: (port?: number, host?: string) => Promise<{ host: string; port: number; url: string }>;
  close: () => Promise<void>;
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
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
  if (total === 0) return {};
  const raw = Buffer.concat(chunks).toString("utf8").trim();
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

function sendJson(res: ServerResponse, statusCode: number, body: unknown, origin?: string): void {
  const payload = JSON.stringify(body);
  const headers: Record<string, string | number> = {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(payload),
  };
  if (origin && LOCAL_CORS_ORIGIN_RE.test(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
  }
  res.writeHead(statusCode, headers);
  res.end(payload);
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
    process.env.DATA_PATH ??
    "data/context-graph.db";
  const store = options.store ?? new GraphStore(dataPath);
  const startedAtMs = Date.now();

  const server = createServer(async (req, res) => {
    const origin = req.headers.origin;
    const method = req.method ?? "GET";

    if (method === "OPTIONS") {
      const headers: Record<string, string> = {
        "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
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

    try {
      // GET /health
      if (method === "GET" && pathname === "/health") {
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

        if (subPath === "compile" && method === "POST") {
          const body = await readJsonBody(req);
          const compiled = compileContext(store, {
            ...(body as Omit<CompileOptions, "graphId">),
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

  const defaultHost = options.host ?? process.env.HOST ?? DEFAULT_HOST;
  const defaultPort =
    options.port ??
    (process.env.PORT ? Number.parseInt(process.env.PORT, 10) : DEFAULT_PORT);

  return {
    server,
    store,
    start: (port = defaultPort, host = defaultHost) =>
      new Promise((resolveStart, rejectStart) => {
        const onError = (err: Error) => {
          server.off("error", onError);
          rejectStart(err);
        };
        server.once("error", onError);
        server.listen(port, host, () => {
          server.off("error", onError);
          const addr = server.address() as AddressInfo;
          resolveStart({
            host: addr.address,
            port: addr.port,
            url: `http://${addr.address}:${addr.port}`,
          });
        });
      }),
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
