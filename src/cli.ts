#!/usr/bin/env node
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { runBenchmark } from "./benchmark.ts";
import { compileContext } from "./compiler.ts";
import { GraphStore } from "./graph-store.ts";
import { startMcpStdioServer } from "./mcp.ts";
import { createContextGraphServer, DEFAULT_HOST, DEFAULT_PORT } from "./server.ts";

const HELP_TEXT = `@opc/context-graph v0.1.0 — Two-Zone KV-Cache-Aware Property DAG Context Engine

Usage:
  context-graph <command> [options]

Commands:
  serve       Start the HTTP REST microservice (default 127.0.0.1:52341)
  mcp         Start the JSON-RPC 2.0 MCP Server over stdio
  benchmark   Run the 32-turn multi-file engineering agent benchmark
  demo        Run a live multi-turn Context Graph compilation & Mermaid demo

Options:
  --db <path>             SQLite database path (default: $DATA_PATH or data/context-graph.db)
  --port <port>           HTTP listen port (default: $PORT or 52341)
  --host <host>           HTTP listen host (default: $HOST or 127.0.0.1)
  --budget-ratio <ratio>  Token budget ratio vs full linear history for benchmark (default: 0.4)
  --json                  Emit machine-readable JSON output
  -h, --help              Show this help message
`;

export interface CliIo {
  stdout?: (chunk: string) => void;
  stderr?: (chunk: string) => void;
}

export async function runCli(
  argv: string[] = process.argv.slice(2),
  io?: CliIo,
): Promise<number> {
  const writeOut = io?.stdout ?? ((s: string) => { process.stdout.write(s); });
  const writeErr = io?.stderr ?? ((s: string) => { process.stderr.write(s); });

  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      help: { type: "boolean", short: "h" },
      json: { type: "boolean" },
      db: { type: "string" },
      port: { type: "string" },
      host: { type: "string" },
      "budget-ratio": { type: "string" },
    },
  });

  const cmd = positionals[0];
  if (values.help || !cmd) {
    writeOut(HELP_TEXT);
    return 0;
  }

  const dbPath =
    typeof values.db === "string"
      ? values.db
      : process.env.DATA_PATH ?? "data/context-graph.db";

  if (cmd === "serve") {
    const port =
      typeof values.port === "string"
        ? Number.parseInt(values.port, 10)
        : process.env.PORT
          ? Number.parseInt(process.env.PORT, 10)
          : DEFAULT_PORT;
    const host =
      typeof values.host === "string" ? values.host : process.env.HOST ?? DEFAULT_HOST;

    const app = createContextGraphServer({ dataPath: dbPath, host, port });
    const bound = await app.start(port, host);
    writeOut(
      `[context-graph] HTTP service listening on ${bound.url} (db=${app.store.dbPath})\n`,
    );

    await new Promise<void>((resolveStop) => {
      const shutdown = () => {
        app.close().finally(() => resolveStop());
      };
      process.once("SIGINT", shutdown);
      process.once("SIGTERM", shutdown);
    });
    return 0;
  }

  if (cmd === "mcp") {
    startMcpStdioServer({ dataPath: dbPath });
    await new Promise<void>((resolveStop) => {
      process.stdin.once("end", () => resolveStop());
      process.once("SIGINT", () => resolveStop());
      process.once("SIGTERM", () => resolveStop());
    });
    return 0;
  }

  if (cmd === "benchmark") {
    const budgetRatio =
      typeof values["budget-ratio"] === "string"
        ? Number.parseFloat(values["budget-ratio"])
        : 0.4;
    const report = runBenchmark({ budgetRatio });
    if (values.json) {
      writeOut(`${JSON.stringify(report, null, 2)}\n`);
      return 0;
    }

    writeOut(`# ${report.scenarioName}\n\n`);
    writeOut(
      `- Total Turns: **${report.totalTurns}**\n` +
        `- Full Linear Final Context: **${report.fullLinearFinalTokens} tokens**\n` +
        `- Target Token Budget (${Math.round(report.budgetRatio * 100)}%): **${report.tokenBudget} tokens**\n` +
        `- Cached Token Pricing Weight: **${report.cachedTokenWeight}x** (Uncached = 1.0x)\n\n`,
    );
    writeOut(`${report.markdownTable}\n\n`);
    writeOut(`## Detailed Cumulative Breakdown\n\n`);
    for (const s of Object.values(report.strategies)) {
      writeOut(
        `- **${s.strategy}**: totalPrompt=${s.totalPromptTokens}, cached=${s.totalCachedTokens}, ` +
          `uncached=${s.totalUncachedTokens}, warmHitRate=${s.warmTurnKvCacheHitRate.toFixed(1)}%, ` +
          `effectiveBillable=${s.effectiveBillableTokens}, costVsLinearUncached=${s.costIndexVsLinearUncached.toFixed(1)}%\n`,
      );
    }
    return 0;
  }

  if (cmd === "demo") {
    const store = new GraphStore(":memory:");
    const g = store.createGraph({
      id: "demo_graph",
      name: "Live Two-Zone Context Graph Demo",
      defaultBudget: 900,
    });

    store.ingestTurn(g.id, {
      turn: 1,
      scope: "auth",
      system: "You are an autonomous TypeScript security engineer.",
      goal: "Refactor authentication to constant-time token verification.",
      constraints: [
        {
          id: "c_timing",
          content: "MUST use crypto.timingSafeEqual for all signature comparisons.",
        },
      ],
      artifacts: [
        {
          id: "auth_v1",
          path: "src/auth.ts",
          content: "export function verify(a: string, b: string) { return a === b; } // insecure v1",
        },
      ],
    });

    const c1 = compileContext(store, {
      graphId: g.id,
      focusQuery: "Inspect current auth implementation",
      activeScope: "auth",
    });

    store.ingestTurn(g.id, {
      turn: 2,
      scope: "auth",
      userMessage: "Replace insecure === check with timingSafeEqual in src/auth.ts.",
      artifacts: [
        {
          id: "auth_v2",
          path: "src/auth.ts",
          content:
            "import { timingSafeEqual } from 'node:crypto';\nexport function verifyV2(a: Buffer, b: Buffer) { return a.length === b.length && timingSafeEqual(a, b); }",
          dependsOn: ["c_timing"],
        },
      ],
    });

    const c2 = compileContext(store, {
      graphId: g.id,
      focusQuery: "Verify updated src/auth.ts complies with timing constraint",
      activeScope: "auth",
    });

    if (values.json) {
      writeOut(
        `${JSON.stringify(
          {
            graphId: g.id,
            turn1Cache: c1.kvCache,
            turn2Cache: c2.kvCache,
            excludedNodes: c2.excludedNodes,
            mermaid: store.toMermaid(g.id),
          },
          null,
          2,
        )}\n`,
      );
    } else {
      writeOut(`=== @opc/context-graph Live Demo ===\n`);
      writeOut(`Turn 1 Tokens: ${c1.kvCache.totalTokens} (Cold start)\n`);
      writeOut(
        `Turn 2 Tokens: ${c2.kvCache.totalTokens} | Prefix Cached: ${c2.kvCache.prefixCachedTokens} (${(c2.kvCache.prefixHitRate * 100).toFixed(1)}% KV-Cache Hit Rate!)\n`,
      );
      writeOut(
        `Occluded Superseded Nodes: ${c2.excludedNodes.map((e) => `${e.id}(${e.reason})`).join(", ")}\n\n`,
      );
      writeOut(`--- Mermaid Topology ---\n${store.toMermaid(g.id)}\n`);
    }
    store.close();
    return 0;
  }

  writeErr(`Unknown command '${cmd}'. Run 'context-graph --help' for usage.\n`);
  return 1;
}

const isDirectEntry =
  process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url;

if (isDirectEntry) {
  runCli().then(
    (code) => {
      process.exitCode = code;
    },
    (err) => {
      process.stderr.write(`[context-graph] Fatal error: ${err instanceof Error ? err.message : String(err)}\n`);
      process.exitCode = 1;
    },
  );
}
