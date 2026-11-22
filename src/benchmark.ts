import { CACHED_TOKEN_COST_WEIGHT, compileContext } from "./compiler.ts";
import { GraphStore } from "./graph-store.ts";
import type {
  BenchmarkReport,
  BenchmarkStrategyName,
  IngestTurnInput,
  StrategyBenchmarkMetrics,
  TurnTelemetry,
} from "./types.ts";
import {
  cosineSimilarity,
  countTokens,
  embedText,
  longestCommonPrefixLength,
  tokenizeToIds,
} from "./vector.ts";

export interface BenchmarkOptions {
  budgetRatio?: number;
}

interface LinearChunk {
  id: string;
  turn: number;
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  tokens: number;
  embedding: number[];
  resourceKey?: string;
  version?: number;
  deadEndGroup?: string;
  isSystemRoot?: boolean;
}

interface TurnSpec {
  turn: number;
  scope: string;
  focusQuery: string;
  ingest: IngestTurnInput;
  linearChunks: Array<{
    id: string;
    role: "system" | "user" | "assistant" | "tool";
    text: string;
    resourceKey?: string;
    version?: number;
    deadEndGroup?: string;
    isSystemRoot?: boolean;
  }>;
  retractedDeadEndGroupsAfterTurn?: string[];
}

const CONSTRAINT_MARKERS = [
  "INVARIANT-C1: TENANT_SQL_PARAM_BINDING",
  "INVARIANT-C2: WEBHOOK_TIMING_SAFE_300S",
  "INVARIANT-C3: INTEGER_CENTS_IDEMPOTENCY",
] as const;

const FILE_STATE_GROUND_TRUTH = [
  {
    file: "src/auth.ts",
    latestMarker: "verifyTenantTokenV4",
    staleMarkers: [
      "verifyAuthV1_InsecureDecode",
      "verifyAuthV2_BasicHmac",
      "verifyAuthV3_KeyRingDraft",
    ],
  },
  {
    file: "src/ledger.ts",
    latestMarker: "recordLedgerEntryV4",
    staleMarkers: [
      "postLedgerV1_FloatAmount",
      "postLedgerV2_IntegerDraft",
      "postLedgerV3_SpinRetry",
    ],
  },
  {
    file: "src/webhook.ts",
    latestMarker: "handleWebhookIngressV4",
    staleMarkers: [
      "ingestWebhookV1_NaiveEq",
      "ingestWebhookV2_HmacOnly",
      "ingestWebhookV3_NonceDraft",
    ],
  },
] as const;

function buildAuthFileVersion(v: 1 | 2 | 3 | 4): string {
  if (v === 1) {
    return [
      "// file: src/auth.ts (v1 - initial prototype)",
      "export const AUTH_MODULE_VERSION = 1;",
      "export function verifyAuthV1_InsecureDecode(header: string): { userId: string } {",
      "  const raw = Buffer.from(header.replace('Bearer ', ''), 'base64url').toString('utf8');",
      "  const parsed = JSON.parse(raw) as { sub: string; exp: number };",
      "  // TODO: missing signature check, missing tenant_id claim, missing skew guard",
      "  if (Date.now() > parsed.exp * 1000) throw new Error('expired');",
      "  return { userId: parsed.sub };",
      "}",
      "// helper notes: temporary base64 decode for local smoke harness before HMAC key setup.",
    ].join("\n");
  }
  if (v === 2) {
    return [
      "// file: src/auth.ts (v2 - basic HMAC signature)",
      "import { createHmac } from 'node:crypto';",
      "export const AUTH_MODULE_VERSION = 2;",
      "export function verifyAuthV2_BasicHmac(token: string, secret: string): { userId: string; tenantId: string } {",
      "  const [payloadB64, sigHex] = token.split('.');",
      "  const expected = createHmac('sha256', secret).update(payloadB64 ?? '').digest('hex');",
      "  if (sigHex !== expected) throw new Error('bad signature'); // still non-constant-time",
      "  const body = JSON.parse(Buffer.from(payloadB64!, 'base64url').toString('utf8'));",
      "  return { userId: body.sub, tenantId: body.tid };",
      "}",
    ].join("\n");
  }
  if (v === 3) {
    return [
      "// file: src/auth.ts (v3 - key rotation draft)",
      "import { createHmac, timingSafeEqual } from 'node:crypto';",
      "export const AUTH_MODULE_VERSION = 3;",
      "export function verifyAuthV3_KeyRingDraft(token: string, keys: Record<string, string>) {",
      "  const [kid, payloadB64, sigHex] = token.split('.');",
      "  const secret = keys[kid ?? 'default'];",
      "  if (!secret) throw new Error('unknown kid');",
      "  const mac = createHmac('sha256', secret).update(`${kid}.${payloadB64}`).digest();",
      "  const provided = Buffer.from(sigHex ?? '', 'hex');",
      "  if (mac.length !== provided.length || !timingSafeEqual(mac, provided)) throw new Error('invalid mac');",
      "  return JSON.parse(Buffer.from(payloadB64!, 'base64url').toString('utf8'));",
      "}",
    ].join("\n");
  }
  return [
    "// file: src/auth.ts (v4 - production hardened tenant token verifier)",
    "import { createHmac, timingSafeEqual } from 'node:crypto';",
    "export const AUTH_MODULE_VERSION = 4;",
    "export interface VerifiedTenantPrincipal { tenantId: string; actorId: string; keyId: string; issuedAt: number }",
    "export function verifyTenantTokenV4(token: string, keyRing: Readonly<Record<string, string>>, nowSec = Math.floor(Date.now() / 1000)): VerifiedTenantPrincipal {",
    "  const parts = token.split('.');",
    "  if (parts.length !== 3) throw new Error('AUTH_MALFORMED_TOKEN');",
    "  const [keyId, payloadB64, sigHex] = parts as [string, string, string];",
    "  const secret = keyRing[keyId];",
    "  if (!secret) throw new Error('AUTH_UNKNOWN_KEY_ID');",
    "  const expectedMac = createHmac('sha256', secret).update(`${keyId}.${payloadB64}`).digest();",
    "  const candidateMac = Buffer.from(sigHex, 'hex');",
    "  if (expectedMac.length !== candidateMac.length || !timingSafeEqual(expectedMac, candidateMac)) {",
    "    throw new Error('AUTH_BAD_SIGNATURE');",
    "  }",
    "  const claims = JSON.parse(Buffer.from(payloadB64, 'base64url').toString('utf8')) as { tid: string; sub: string; iat: number; exp: number };",
    "  if (!claims.tid || Math.abs(nowSec - claims.iat) > 300 || nowSec >= claims.exp) {",
    "    throw new Error('AUTH_CLAIMS_REJECTED');",
    "  }",
    "  return { tenantId: claims.tid, actorId: claims.sub, keyId, issuedAt: claims.iat };",
    "}",
  ].join("\n");
}

function buildLedgerFileVersion(v: 1 | 2 | 3 | 4): string {
  if (v === 1) {
    return [
      "// file: src/ledger.ts (v1 - naive float prototype)",
      "import type { DatabaseSync } from 'node:sqlite';",
      "export const LEDGER_MODULE_VERSION = 1;",
      "export function postLedgerV1_FloatAmount(db: DatabaseSync, tenantId: string, amountUsd: string) {",
      "  const numericDollars = Number.parseFloat(amountUsd); // BUG: violates integer cents rule",
      "  db.exec(`INSERT INTO ledger_entries VALUES ('${tenantId}', ${numericDollars})`); // BUG: SQL interpolation",
      "  return numericDollars;",
      "}",
    ].join("\n");
  }
  if (v === 2) {
    return [
      "// file: src/ledger.ts (v2 - integer cents migration)",
      "import type { DatabaseSync } from 'node:sqlite';",
      "export const LEDGER_MODULE_VERSION = 2;",
      "export function postLedgerV2_IntegerDraft(db: DatabaseSync, tenantId: string, amountCents: number, idemKey: string) {",
      "  if (!Number.isSafeInteger(amountCents)) throw new Error('LEDGER_NON_INTEGER_CENTS');",
      "  const stmt = db.prepare('INSERT INTO ledger_entries (tenant_id, amount_cents, idempotency_key) VALUES (?, ?, ?)');",
      "  return stmt.run(tenantId, amountCents, idemKey);",
      "}",
    ].join("\n");
  }
  if (v === 3) {
    return [
      "// file: src/ledger.ts (v3 - busy retry experiment)",
      "import type { DatabaseSync } from 'node:sqlite';",
      "export const LEDGER_MODULE_VERSION = 3;",
      "export function postLedgerV3_SpinRetry(db: DatabaseSync, tenantId: string, amountCents: number, idemKey: string) {",
      "  if (!Number.isSafeInteger(amountCents)) throw new Error('LEDGER_NON_INTEGER_CENTS');",
      "  db.exec('BEGIN IMMEDIATE');",
      "  try {",
      "    const stmt = db.prepare('INSERT INTO ledger_entries (tenant_id, amount_cents, idempotency_key) VALUES (?, ?, ?)');",
      "    const res = stmt.run(tenantId, amountCents, idemKey);",
      "    db.exec('COMMIT');",
      "    return res;",
      "  } catch (err) { db.exec('ROLLBACK'); throw err; }",
      "}",
    ].join("\n");
  }
  return [
    "// file: src/ledger.ts (v4 - production idempotent integer-cents ledger)",
    "import type { DatabaseSync } from 'node:sqlite';",
    "export const LEDGER_MODULE_VERSION = 4;",
    "export interface LedgerWriteOutcome { inserted: boolean; entryId: string; amountCents: number; tenantId: string }",
    "export function recordLedgerEntryV4(db: DatabaseSync, params: { tenantId: string; entryId: string; amountCents: number; idempotencyKey: string; currency: string }): LedgerWriteOutcome {",
    "  if (!Number.isSafeInteger(params.amountCents) || params.amountCents <= 0) {",
    "    throw new Error('LEDGER_INVALID_INTEGER_CENTS');",
    "  }",
    "  db.exec('BEGIN IMMEDIATE');",
    "  try {",
    "    const existing = db.prepare('SELECT entry_id, amount_cents FROM ledger_entries WHERE tenant_id = ? AND idempotency_key = ?').get(params.tenantId, params.idempotencyKey) as { entry_id: string; amount_cents: number } | undefined;",
    "    if (existing) {",
    "      db.exec('COMMIT');",
    "      return { inserted: false, entryId: existing.entry_id, amountCents: existing.amount_cents, tenantId: params.tenantId };",
    "    }",
    "    db.prepare('INSERT INTO ledger_entries (entry_id, tenant_id, amount_cents, currency, idempotency_key) VALUES (?, ?, ?, ?, ?)').run(params.entryId, params.tenantId, params.amountCents, params.currency, params.idempotencyKey);",
    "    db.exec('COMMIT');",
    "    return { inserted: true, entryId: params.entryId, amountCents: params.amountCents, tenantId: params.tenantId };",
    "  } catch (err) {",
    "    db.exec('ROLLBACK');",
    "    throw err;",
    "  }",
    "}",
  ].join("\n");
}

function buildWebhookFileVersion(v: 1 | 2 | 3 | 4): string {
  if (v === 1) {
    return [
      "// file: src/webhook.ts (v1 - initial webhook handler)",
      "import { createHmac } from 'node:crypto';",
      "export const WEBHOOK_MODULE_VERSION = 1;",
      "export function ingestWebhookV1_NaiveEq(rawBody: string, sigHeader: string, secret: string) {",
      "  const digest = createHmac('sha256', secret).update(rawBody).digest('hex');",
      "  if (digest !== sigHeader) throw new Error('WEBHOOK_BAD_SIG'); // insecure string === without timestamp window",
      "  return JSON.parse(rawBody);",
      "}",
    ].join("\n");
  }
  if (v === 2) {
    return [
      "// file: src/webhook.ts (v2 - timingSafeEqual + 300s skew check)",
      "import { createHmac, timingSafeEqual } from 'node:crypto';",
      "export const WEBHOOK_MODULE_VERSION = 2;",
      "export function ingestWebhookV2_HmacOnly(rawBody: Buffer, timestampSec: number, sigHex: string, secret: string, nowSec: number) {",
      "  if (Math.abs(nowSec - timestampSec) > 300) throw new Error('WEBHOOK_TIMESTAMP_SKEW_EXCEEDED');",
      "  const mac = createHmac('sha256', secret).update(`${timestampSec}.`).update(rawBody).digest();",
      "  const sig = Buffer.from(sigHex, 'hex');",
      "  if (mac.length !== sig.length || !timingSafeEqual(mac, sig)) throw new Error('WEBHOOK_SIG_MISMATCH');",
      "  return JSON.parse(rawBody.toString('utf8'));",
      "}",
    ].join("\n");
  }
  if (v === 3) {
    return [
      "// file: src/webhook.ts (v3 - replay nonce guard)",
      "import { createHmac, timingSafeEqual } from 'node:crypto';",
      "export const WEBHOOK_MODULE_VERSION = 3;",
      "export function ingestWebhookV3_NonceDraft(rawBody: Buffer, timestampSec: number, sigHex: string, secret: string, seenNonces: Set<string>, nowSec: number) {",
      "  if (Math.abs(nowSec - timestampSec) > 300) throw new Error('WEBHOOK_SKEW');",
      "  const mac = createHmac('sha256', secret).update(`${timestampSec}.`).update(rawBody).digest();",
      "  const sig = Buffer.from(sigHex, 'hex');",
      "  if (mac.length !== sig.length || !timingSafeEqual(mac, sig)) throw new Error('WEBHOOK_SIG');",
      "  const event = JSON.parse(rawBody.toString('utf8')) as { eventId: string; amountCents: number };",
      "  if (seenNonces.has(event.eventId)) return { duplicate: true };",
      "  seenNonces.add(event.eventId);",
      "  return { duplicate: false, event };",
      "}",
    ].join("\n");
  }
  return [
    "// file: src/webhook.ts (v4 - full end-to-end verified gateway ingress)",
    "import { createHmac, timingSafeEqual } from 'node:crypto';",
    "import type { DatabaseSync } from 'node:sqlite';",
    "import { verifyTenantTokenV4 } from './auth.ts';",
    "import { recordLedgerEntryV4, type LedgerWriteOutcome } from './ledger.ts';",
    "export const WEBHOOK_MODULE_VERSION = 4;",
    "export function handleWebhookIngressV4(deps: { db: DatabaseSync; keyRing: Record<string, string>; webhookSecret: string }, req: { authToken: string; rawBody: Buffer; timestampSec: number; signatureHex: string; nowSec?: number }): LedgerWriteOutcome {",
    "  const nowSec = req.nowSec ?? Math.floor(Date.now() / 1000);",
    "  if (Math.abs(nowSec - req.timestampSec) > 300) {",
    "    throw new Error('WEBHOOK_REPLAY_WINDOW_EXCEEDED');",
    "  }",
    "  const expected = createHmac('sha256', deps.webhookSecret).update(`${req.timestampSec}.`).update(req.rawBody).digest();",
    "  const received = Buffer.from(req.signatureHex, 'hex');",
    "  if (expected.length !== received.length || !timingSafeEqual(expected, received)) {",
    "    throw new Error('WEBHOOK_SIGNATURE_INVALID');",
    "  }",
    "  const principal = verifyTenantTokenV4(req.authToken, deps.keyRing, nowSec);",
    "  const payload = JSON.parse(req.rawBody.toString('utf8')) as { entryId: string; amountCents: number; idempotencyKey: string; currency: string };",
    "  return recordLedgerEntryV4(deps.db, {",
    "    tenantId: principal.tenantId,",
    "    entryId: payload.entryId,",
    "    amountCents: payload.amountCents,",
    "    idempotencyKey: payload.idempotencyKey,",
    "    currency: payload.currency,",
    "  });",
    "}",
  ].join("\n");
}

function buildDeadEndLog(branch: "jwk" | "sqlite", step: number): string {
  if (branch === "jwk") {
    const frames = Array.from({ length: 14 }, (_, idx) =>
      `  at RemoteJwksFetcher.pollSocketFrame (node_internal/jwks_pool_${step}.ts:${110 + idx * 7}:19) [DEADEND_JWK_TRACE_${step}_${idx}]`,
    ).join("\n");
    return [
      `[DEADEND_JWK_HYPOTHESIS step=${step}] Investigating suspected OIDC JWKS TLS session cache leak in auth worker:`,
      `HeapSnapshot delta=+48.2MB rss=312MB sockets_open=64 keepAliveTimeout=5000ms retryBackoff=${step * 250}ms`,
      frames,
      `Diagnostic dump ${step}: inspecting TLSSocket._handle Epoll waiter queue and X509 cert chain cache entries...`,
    ].join("\n");
  }
  const frames = Array.from({ length: 14 }, (_, idx) =>
    `  at CustomUserspaceSpinLock.acquireSlot (scripts/spin_retry_${step}.ts:${40 + idx * 5}:13) [DEADEND_SQLITE_SPIN_${step}_${idx}]`,
  ).join("\n");
  return [
    `[DEADEND_SQLITE_SPINLOCK step=${step}] Testing custom userspace spinlock & busy-wait jitter loop for SQLITE_BUSY:`,
    `Worker contention trace: lock_wait_p99=${420 + step * 95}ms aborted_tx=${12 + step * 4} wal_frames=4096`,
    frames,
    `Spinlock trace ${step}: threads livelocked under nested write cursor before SQLite busy_timeout configured...`,
  ].join("\n");
}

function buildRoutineDiagnosticOutput(turn: number, topic: string): string {
  const lines = Array.from({ length: 8 }, (_, idx) =>
    `  [check ${turn}.${idx + 1}] ${topic} metric_bucket_${idx + 1}=ok latency_us=${180 + turn * 3 + idx * 11} alloc_kb=${64 + idx * 8}`,
  ).join("\n");
  return `[ci-telemetry turn=${turn}] Running ${topic} verification suite:\n${lines}`;
}

/**
 * Generates the deterministic 32-turn multi-file engineering trajectory.
 */
export function build32TurnTrajectory(): TurnSpec[] {
  const specs: TurnSpec[] = [];

  // Turn 1: System charter + global goal + 3 non-negotiable security & architecture constraints
  const sysText =
    "You are Principal Payment Systems Agent responsible for building and auditing the multi-tenant gateway (`src/auth.ts`, `src/ledger.ts`, `src/webhook.ts`).";
  const goalText =
    "Deliver a zero-trust, replay-safe, idempotent multi-tenant payment webhook & ledger gateway passing strict security audit.";
  const c1 =
    "INVARIANT-C1: TENANT_SQL_PARAM_BINDING — Every SQLite query MUST bind `tenant_id` via parameterized statements (`?`); raw SQL string interpolation is strictly forbidden.";
  const c2 =
    "INVARIANT-C2: WEBHOOK_TIMING_SAFE_300S — Every HMAC-SHA256 check MUST use `crypto.timingSafeEqual` with a strict `300` second timestamp skew window before any database access.";
  const c3 =
    "INVARIANT-C3: INTEGER_CENTS_IDEMPOTENCY — All monetary amounts MUST use safe integer `amount_cents` (never IEEE-754 floats) guarded by `UNIQUE(tenant_id, idempotency_key)`.";

  specs.push({
    turn: 1,
    scope: "global",
    focusQuery: "Initialize payment gateway architecture charter and SQL/HMAC security invariants",
    ingest: {
      turn: 1,
      scope: "global",
      system: sysText,
      goal: goalText,
      constraints: [
        { id: "c_sql_tenant", content: c1, resourceKey: "constraint:c1" },
        { id: "c_hmac_skew", content: c2, resourceKey: "constraint:c2" },
      ],
    },
    linearChunks: [
      { id: "t1_sys", role: "system", text: sysText, isSystemRoot: true },
      { id: "t1_goal", role: "user", text: goalText },
      { id: "t1_c1", role: "user", text: c1 },
      { id: "t1_c2", role: "user", text: c2 },
    ],
  });

  // Turn 2: Monetary precision constraint C3 + Stable Architectural Task Frame
  const frameText =
    "ARCHITECTURE_FRAME: Three decoupled modules: (1) `src/auth.ts` verifies tenant tokens, (2) `src/ledger.ts` persists idempotent integer-cent entries, (3) `src/webhook.ts` verifies ingress signatures & orchestrates auth + ledger.";
  const d1Text =
    "DECISION-D1: Use Node native `node:crypto` + `node:sqlite` (`DatabaseSync`) with zero third-party runtime packages.";
  specs.push({
    turn: 2,
    scope: "global",
    focusQuery: "Define monetary integer-cents invariant and module boundaries across auth.ts, ledger.ts, and webhook.ts",
    ingest: {
      turn: 2,
      scope: "global",
      constraints: [
        { id: "c_int_cents", content: c3, resourceKey: "constraint:c3" },
      ],
      taskFrame: { id: "frame_arch", content: frameText, scope: "global" },
      decisions: [
        {
          id: "dec_zero_dep",
          content: d1Text,
          tier: "stable_frame",
          resourceKey: "decision:runtime-stack",
          dependsOn: ["c_sql_tenant", "c_hmac_skew", "c_int_cents"],
        },
      ],
    },
    linearChunks: [
      { id: "t2_c3", role: "user", text: c3 },
      { id: "t2_frame", role: "assistant", text: frameText },
      { id: "t2_d1", role: "assistant", text: d1Text },
    ],
  });

  // Turn 3: src/auth.ts v1
  const authV1 = buildAuthFileVersion(1);
  specs.push({
    turn: 3,
    scope: "auth",
    focusQuery: "Scaffold initial src/auth.ts token decoder prototype",
    ingest: {
      turn: 3,
      scope: "auth",
      userMessage: "Create initial `src/auth.ts` prototype to unblock local routing tests.",
      artifacts: [{ id: "art_auth_v1", path: "src/auth.ts", content: authV1 }],
    },
    linearChunks: [
      { id: "t3_user", role: "user", text: "Create initial `src/auth.ts` prototype to unblock local routing tests." },
      { id: "t3_tool", role: "tool", text: authV1, resourceKey: "file:src/auth.ts", version: 1 },
    ],
  });

  // Turn 4: src/auth.ts v2
  const authV2 = buildAuthFileVersion(2);
  specs.push({
    turn: 4,
    scope: "auth",
    focusQuery: "Add HMAC-SHA256 verification and tenantId claim to src/auth.ts",
    ingest: {
      turn: 4,
      scope: "auth",
      userMessage: "Upgrade `src/auth.ts` to v2 with HMAC-SHA256 signature verification and tenantId extraction.",
      artifacts: [{ id: "art_auth_v2", path: "src/auth.ts", content: authV2, dependsOn: ["c_sql_tenant"] }],
    },
    linearChunks: [
      { id: "t4_user", role: "user", text: "Upgrade `src/auth.ts` to v2 with HMAC-SHA256 signature verification and tenantId extraction." },
      { id: "t4_tool", role: "tool", text: authV2, resourceKey: "file:src/auth.ts", version: 2 },
    ],
  });

  // Turns 5-7: Dead-End Branch #1 (JWKS TLS socket leak false hypothesis)
  for (const step of [1, 2, 3] as const) {
    const t = 4 + step;
    const log = buildDeadEndLog("jwk", step);
    const isResolutionTurn = step === 3;
    specs.push({
      turn: t,
      scope: "debug-jwk-deadend",
      focusQuery: `Debug auth worker intermittent timeout step ${step}`,
      ingest: {
        turn: t,
        scope: "debug-jwk-deadend",
        userMessage: `Investigate auth worker timeout hypothesis #${step} (JWKS socket pool).`,
        toolCalls: [
          {
            id: `obs_deadend_jwk_${step}`,
            toolName: " diagnostic_profiler",
            output: log,
            isError: true,
          },
        ],
        ...(isResolutionTurn
          ? {
              retractScopes: ["debug-jwk-deadend"],
              facts: [
                {
                  id: "fact_jwk_resolved",
                  content:
                    "RESOLVED: Auth timeout was caused by `localhost` IPv6 DNS fallback in test harness (`127.0.0.1` fixed it); remote JWKS leak hypothesis retracted.",
                },
              ],
            }
          : {}),
      },
      linearChunks: [
        { id: `t${t}_user`, role: "user", text: `Investigate auth worker timeout hypothesis #${step} (JWKS socket pool).`, deadEndGroup: "jwk" },
        { id: `t${t}_tool`, role: "tool", text: log, deadEndGroup: "jwk" },
        ...(isResolutionTurn
          ? [{
              id: `t${t}_asst`,
              role: "assistant" as const,
              text: "RESOLVED: Timeout was `localhost` IPv6 loopback resolution in test harness; JWKS hypothesis was a dead end.",
            }]
          : []),
      ],
      ...(isResolutionTurn ? { retractedDeadEndGroupsAfterTurn: ["jwk"] } : {}),
    });
  }

  // Turn 8: src/auth.ts v3
  const authV3 = buildAuthFileVersion(3);
  specs.push({
    turn: 8,
    scope: "auth",
    focusQuery: "Refactor src/auth.ts to v3 with key rotation ring and timingSafeEqual",
    ingest: {
      turn: 8,
      scope: "auth",
      userMessage: "Update `src/auth.ts` to v3 supporting key rotation (`kid`) and `timingSafeEqual`.",
      artifacts: [{ id: "art_auth_v3", path: "src/auth.ts", content: authV3, dependsOn: ["c_hmac_skew"] }],
    },
    linearChunks: [
      { id: "t8_user", role: "user", text: "Update `src/auth.ts` to v3 supporting key rotation (`kid`) and `timingSafeEqual`." },
      { id: "t8_tool", role: "tool", text: authV3, resourceKey: "file:src/auth.ts", version: 3 },
    ],
  });

  // Turn 9: Auth security review observation
  const t9Obs = buildRoutineDiagnosticOutput(9, "auth-token-fuzzing");
  specs.push({
    turn: 9,
    scope: "auth",
    focusQuery: "Run fuzzing audit on src/auth.ts v3 claims validation",
    ingest: {
      turn: 9,
      scope: "auth",
      userMessage: "Run claim boundary fuzz tests against `src/auth.ts` v3.",
      toolCalls: [{ id: "obs_auth_fuzz", toolName: "run_fuzzer", output: t9Obs }],
    },
    linearChunks: [
      { id: "t9_user", role: "user", text: "Run claim boundary fuzz tests against `src/auth.ts` v3." },
      { id: "t9_tool", role: "tool", text: t9Obs },
    ],
  });

  // Turn 10: src/auth.ts v4 (FINAL)
  const authV4 = buildAuthFileVersion(4);
  const dAuthText =
    "DECISION-D2: Finalize `verifyTenantTokenV4` in `src/auth.ts` enforcing `timingSafeEqual`, mandatory `tid` claim, and 300s `iat` clock-skew ceiling.";
  specs.push({
    turn: 10,
    scope: "auth",
    focusQuery: "Finalize src/auth.ts v4 verifyTenantTokenV4 with 300s skew and tenantId guard",
    ingest: {
      turn: 10,
      scope: "auth",
      userMessage: "Harden `src/auth.ts` to final v4 (`verifyTenantTokenV4`) with 300s `iat` skew enforcement.",
      decisions: [
        {
          id: "dec_auth_v4",
          content: dAuthText,
          resourceKey: "decision:auth-contract",
          dependsOn: ["c_sql_tenant", "c_hmac_skew"],
        },
      ],
      artifacts: [
        {
          id: "art_auth_v4",
          path: "src/auth.ts",
          content: authV4,
          dependsOn: ["c_sql_tenant", "c_hmac_skew", "dec_auth_v4"],
        },
      ],
    },
    linearChunks: [
      { id: "t10_user", role: "user", text: "Harden `src/auth.ts` to final v4 (`verifyTenantTokenV4`) with 300s `iat` skew enforcement." },
      { id: "t10_asst", role: "assistant", text: dAuthText },
      { id: "t10_tool", role: "tool", text: authV4, resourceKey: "file:src/auth.ts", version: 4 },
    ],
  });

  // Turn 11: Switch to ledger scope -> src/ledger.ts v1
  const ledgerV1 = buildLedgerFileVersion(1);
  specs.push({
    turn: 11,
    scope: "ledger",
    focusQuery: "Create initial src/ledger.ts prototype",
    ingest: {
      turn: 11,
      scope: "ledger",
      userMessage: "Start `ledger` module: draft `src/ledger.ts` v1 entry writer.",
      artifacts: [{ id: "art_ledger_v1", path: "src/ledger.ts", content: ledgerV1 }],
    },
    linearChunks: [
      { id: "t11_user", role: "user", text: "Start `ledger` module: draft `src/ledger.ts` v1 entry writer." },
      { id: "t11_tool", role: "tool", text: ledgerV1, resourceKey: "file:src/ledger.ts", version: 1 },
    ],
  });

  // Turn 12: src/ledger.ts v2 (fix float + SQL parameterization)
  const ledgerV2 = buildLedgerFileVersion(2);
  specs.push({
    turn: 12,
    scope: "ledger",
    focusQuery: "Fix float precision and SQL parameterization in src/ledger.ts v2",
    ingest: {
      turn: 12,
      scope: "ledger",
      userMessage: "Fix C1 and C3 violations in `src/ledger.ts`: switch to integer `amount_cents` and parameterized SQL.",
      artifacts: [
        {
          id: "art_ledger_v2",
          path: "src/ledger.ts",
          content: ledgerV2,
          dependsOn: ["c_sql_tenant", "c_int_cents"],
        },
      ],
    },
    linearChunks: [
      { id: "t12_user", role: "user", text: "Fix C1 and C3 violations in `src/ledger.ts`: switch to integer `amount_cents` and parameterized SQL." },
      { id: "t12_tool", role: "tool", text: ledgerV2, resourceKey: "file:src/ledger.ts", version: 2 },
    ],
  });

  // Turns 13-15: Dead-End Branch #2 (Userspace spinlock livelock on SQLITE_BUSY)
  for (const step of [1, 2, 3] as const) {
    const t = 12 + step;
    const log = buildDeadEndLog("sqlite", step);
    const isResolution = step === 3;
    specs.push({
      turn: t,
      scope: "debug-sqlite-deadend",
      focusQuery: `Diagnose SQLITE_BUSY parallel contention spinlock attempt ${step}`,
      ingest: {
        turn: t,
        scope: "debug-sqlite-deadend",
        userMessage: `Run custom userspace spinlock experiment #${step} for parallel SQLite writes.`,
        toolCalls: [
          {
            id: `obs_deadend_sqlite_${step}`,
            toolName: "concurrency_bench",
            output: log,
            isError: true,
          },
        ],
        ...(isResolution
          ? {
              retractScopes: ["debug-sqlite-deadend"],
              decisions: [
                {
                  id: "dec_sqlite_wal",
                  content:
                    "DECISION-D3: Discard custom spinlock; use native SQLite `WAL` + `PRAGMA busy_timeout = 5000` + `BEGIN IMMEDIATE` transactions.",
                  resourceKey: "decision:sqlite-concurrency",
                  dependsOn: ["c_int_cents"],
                },
              ],
            }
          : {}),
      },
      linearChunks: [
        { id: `t${t}_user`, role: "user", text: `Run custom userspace spinlock experiment #${step} for parallel SQLite writes.`, deadEndGroup: "sqlite" },
        { id: `t${t}_tool`, role: "tool", text: log, deadEndGroup: "sqlite" },
        ...(isResolution
          ? [{
              id: `t${t}_asst`,
              role: "assistant" as const,
              text: "DECISION-D3: Retract userspace spinlock dead-end; native WAL + `BEGIN IMMEDIATE` solves contention cleanly.",
            }]
          : []),
      ],
      ...(isResolution ? { retractedDeadEndGroupsAfterTurn: ["sqlite"] } : {}),
    });
  }

  // Turn 16: src/ledger.ts v3
  const ledgerV3 = buildLedgerFileVersion(3);
  specs.push({
    turn: 16,
    scope: "ledger",
    focusQuery: "Update src/ledger.ts to v3 with BEGIN IMMEDIATE transaction",
    ingest: {
      turn: 16,
      scope: "ledger",
      userMessage: "Wrap `src/ledger.ts` writes in `BEGIN IMMEDIATE` transaction (v3).",
      artifacts: [{ id: "art_ledger_v3", path: "src/ledger.ts", content: ledgerV3, dependsOn: ["dec_sqlite_wal"] }],
    },
    linearChunks: [
      { id: "t16_user", role: "user", text: "Wrap `src/ledger.ts` writes in `BEGIN IMMEDIATE` transaction (v3)." },
      { id: "t16_tool", role: "tool", text: ledgerV3, resourceKey: "file:src/ledger.ts", version: 3 },
    ],
  });

  // Turn 17-18: Ledger property tests & schema verification
  for (const t of [17, 18] as const) {
    const obs = buildRoutineDiagnosticOutput(t, `ledger-idempotency-check-${t}`);
    specs.push({
      turn: t,
      scope: "ledger",
      focusQuery: `Verify ledger duplicate key behavior batch ${t}`,
      ingest: {
        turn: t,
        scope: "ledger",
        userMessage: `Execute duplicate replay test batch ${t} on ledger.`,
        toolCalls: [{ id: `obs_ledger_test_${t}`, toolName: "node_test", output: obs }],
      },
      linearChunks: [
        { id: `t${t}_user`, role: "user", text: `Execute duplicate replay test batch ${t} on ledger.` },
        { id: `t${t}_tool`, role: "tool", text: obs },
      ],
    });
  }

  // Turn 19: src/ledger.ts v4 (FINAL)
  const ledgerV4 = buildLedgerFileVersion(4);
  specs.push({
    turn: 19,
    scope: "ledger",
    focusQuery: "Finalize src/ledger.ts v4 recordLedgerEntryV4 with tenant_id + idempotency_key lookup",
    ingest: {
      turn: 19,
      scope: "ledger",
      userMessage: "Finalize `src/ledger.ts` v4 (`recordLedgerEntryV4`) returning deterministic idempotent outcome per `(tenant_id, idempotency_key)`.",
      artifacts: [
        {
          id: "art_ledger_v4",
          path: "src/ledger.ts",
          content: ledgerV4,
          dependsOn: ["c_sql_tenant", "c_int_cents", "dec_sqlite_wal"],
        },
      ],
      summaries: [
        {
          id: "sum_ledger_tests",
          content: "Ledger verification summary: duplicate `(tenant_id, idempotency_key)` replays return `{ inserted: false }` with zero double-charging.",
          targetNodeIds: ["obs_ledger_test_17", "obs_ledger_test_18"],
          scope: "ledger",
        },
      ],
    },
    linearChunks: [
      { id: "t19_user", role: "user", text: "Finalize `src/ledger.ts` v4 (`recordLedgerEntryV4`) returning deterministic idempotent outcome per `(tenant_id, idempotency_key)`." },
      { id: "t19_tool", role: "tool", text: ledgerV4, resourceKey: "file:src/ledger.ts", version: 4 },
    ],
  });

  // Turn 20: Switch to webhook scope -> src/webhook.ts v1
  const webhookV1 = buildWebhookFileVersion(1);
  specs.push({
    turn: 20,
    scope: "webhook",
    focusQuery: "Scaffold src/webhook.ts v1 endpoint parser",
    ingest: {
      turn: 20,
      scope: "webhook",
      userMessage: "Create `src/webhook.ts` v1 to parse incoming provider webhooks.",
      artifacts: [{ id: "art_webhook_v1", path: "src/webhook.ts", content: webhookV1 }],
    },
    linearChunks: [
      { id: "t20_user", role: "user", text: "Create `src/webhook.ts` v1 to parse incoming provider webhooks." },
      { id: "t20_tool", role: "tool", text: webhookV1, resourceKey: "file:src/webhook.ts", version: 1 },
    ],
  });

  // Turn 21: Security audit decision + src/webhook.ts v2
  const webhookV2 = buildWebhookFileVersion(2);
  const dWebhook =
    "DECISION-D4: Enforce `timingSafeEqual` over raw `Buffer` bytes and reject `|now - timestamp| > 300s` prior to JSON parsing or DB access.";
  specs.push({
    turn: 21,
    scope: "webhook",
    focusQuery: "Upgrade src/webhook.ts to v2 with timingSafeEqual and 300s replay window",
    ingest: {
      turn: 21,
      scope: "webhook",
      userMessage: "Fix timing side-channel and missing 300s timestamp window in `src/webhook.ts` (v2).",
      decisions: [
        {
          id: "dec_webhook_sec",
          content: dWebhook,
          resourceKey: "decision:webhook-security",
          dependsOn: ["c_hmac_skew"],
        },
      ],
      artifacts: [
        {
          id: "art_webhook_v2",
          path: "src/webhook.ts",
          content: webhookV2,
          dependsOn: ["c_hmac_skew", "dec_webhook_sec"],
        },
      ],
    },
    linearChunks: [
      { id: "t21_user", role: "user", text: "Fix timing side-channel and missing 300s timestamp window in `src/webhook.ts` (v2)." },
      { id: "t21_asst", role: "assistant", text: dWebhook },
      { id: "t21_tool", role: "tool", text: webhookV2, resourceKey: "file:src/webhook.ts", version: 2 },
    ],
  });

  // Turns 22-24: Webhook payload stress tests
  for (const t of [22, 23, 24] as const) {
    const obs = buildRoutineDiagnosticOutput(t, `webhook-signature-vector-${t}`);
    specs.push({
      turn: t,
      scope: "webhook",
      focusQuery: `Run HMAC test vectors batch ${t} on webhook ingress`,
      ingest: {
        turn: t,
        scope: "webhook",
        userMessage: `Run malformed signature & replay timestamp test vectors #${t}.`,
        toolCalls: [{ id: `obs_wh_vec_${t}`, toolName: "sig_vectors", output: obs }],
      },
      linearChunks: [
        { id: `t${t}_user`, role: "user", text: `Run malformed signature & replay timestamp test vectors #${t}.` },
        { id: `t${t}_tool`, role: "tool", text: obs },
      ],
    });
  }

  // Turn 25: src/webhook.ts v3
  const webhookV3 = buildWebhookFileVersion(3);
  specs.push({
    turn: 25,
    scope: "webhook",
    focusQuery: "Update src/webhook.ts to v3 with replay deduplication draft",
    ingest: {
      turn: 25,
      scope: "webhook",
      userMessage: "Add event deduplication draft to `src/webhook.ts` (v3).",
      artifacts: [{ id: "art_webhook_v3", path: "src/webhook.ts", content: webhookV3, dependsOn: ["dec_webhook_sec"] }],
    },
    linearChunks: [
      { id: "t25_user", role: "user", text: "Add event deduplication draft to `src/webhook.ts` (v3)." },
      { id: "t25_tool", role: "tool", text: webhookV3, resourceKey: "file:src/webhook.ts", version: 3 },
    ],
  });

  // Turn 26: Integration wiring check
  const t26Obs = buildRoutineDiagnosticOutput(26, "cross-module-contract-check");
  specs.push({
    turn: 26,
    scope: "webhook",
    focusQuery: "Check wiring between webhook.ts, verifyTenantTokenV4, and recordLedgerEntryV4",
    ingest: {
      turn: 26,
      scope: "webhook",
      userMessage: "Verify `webhook.ts` replaces in-memory Set with `verifyTenantTokenV4` + `recordLedgerEntryV4`.",
      toolCalls: [{ id: "obs_wire_26", toolName: "typecheck", output: t26Obs }],
    },
    linearChunks: [
      { id: "t26_user", role: "user", text: "Verify `webhook.ts` replaces in-memory Set with `verifyTenantTokenV4` + `recordLedgerEntryV4`." },
      { id: "t26_tool", role: "tool", text: t26Obs },
    ],
  });

  // Turn 27: src/webhook.ts v4 (FINAL)
  const webhookV4 = buildWebhookFileVersion(4);
  specs.push({
    turn: 27,
    scope: "webhook",
    focusQuery: "Finalize src/webhook.ts v4 handleWebhookIngressV4 integrating auth.ts v4 and ledger.ts v4",
    ingest: {
      turn: 27,
      scope: "webhook",
      userMessage: "Finalize `src/webhook.ts` v4 (`handleWebhookIngressV4`) wiring `verifyTenantTokenV4` and `recordLedgerEntryV4`.",
      artifacts: [
        {
          id: "art_webhook_v4",
          path: "src/webhook.ts",
          content: webhookV4,
          dependsOn: [
            "c_sql_tenant",
            "c_hmac_skew",
            "c_int_cents",
            "dec_webhook_sec",
            "art_auth_v4",
            "art_ledger_v4",
          ],
        },
      ],
      summaries: [
        {
          id: "sum_wh_vectors",
          content: "Webhook test vector summary: 100% of tampered HMACs and >300s skewed timestamps rejected pre-DB.",
          targetNodeIds: ["obs_wh_vec_22", "obs_wh_vec_23", "obs_wh_vec_24"],
          scope: "webhook",
        },
      ],
    },
    linearChunks: [
      { id: "t27_user", role: "user", text: "Finalize `src/webhook.ts` v4 (`handleWebhookIngressV4`) wiring `verifyTenantTokenV4` and `recordLedgerEntryV4`." },
      { id: "t27_tool", role: "tool", text: webhookV4, resourceKey: "file:src/webhook.ts", version: 4 },
    ],
  });

  // Turns 28-31: Post-integration soak & telemetry turns (pushes earlier context further upstream)
  for (const t of [28, 29, 30, 31] as const) {
    const obs = buildRoutineDiagnosticOutput(t, `soak-end2end-pipeline-stage-${t}`);
    specs.push({
      turn: t,
      scope: "soak",
      focusQuery: `Run end-to-end soak stage ${t} telemetry validation`,
      ingest: {
        turn: t,
        scope: "soak",
        userMessage: `Execute soak pipeline validation stage ${t} and collect p99 latency metrics.`,
        toolCalls: [{ id: `obs_soak_${t}`, toolName: "soak_runner", output: obs }],
      },
      linearChunks: [
        { id: `t${t}_user`, role: "user", text: `Execute soak pipeline validation stage ${t} and collect p99 latency metrics.` },
        { id: `t${t}_tool`, role: "tool", text: obs },
      ],
    });
  }

  // Turn 32: Final Comprehensive Integration & Security Audit Synthesis Query
  const finalQuery =
    "Conduct the final production readiness & cross-module audit across `src/auth.ts`, `src/ledger.ts`, and `src/webhook.ts`: (1) verify strict compliance with every initial architectural & security rule established at project kickoff, (2) confirm the exact active entrypoint signatures of all three files with zero deprecated draft drift, and (3) summarize end-to-end request flow.";
  specs.push({
    turn: 32,
    scope: "integration",
    focusQuery: finalQuery,
    ingest: {
      turn: 32,
      scope: "integration",
      userMessage: finalQuery,
    },
    linearChunks: [{ id: "t32_user", role: "user", text: finalQuery }],
  });

  return specs;
}

function evaluateFinalPromptAccuracy(promptText: string): {
  criticalConstraintRecall: number;
  latestStateAccuracy: number;
  latestStateRawRecall: number;
} {
  let constraintHits = 0;
  for (const marker of CONSTRAINT_MARKERS) {
    if (promptText.includes(marker)) constraintHits++;
  }
  const criticalConstraintRecall = Math.round((constraintHits / CONSTRAINT_MARKERS.length) * 10000) / 100;

  let cleanLatestHits = 0;
  let rawLatestHits = 0;
  for (const f of FILE_STATE_GROUND_TRUTH) {
    const hasLatest = promptText.includes(f.latestMarker);
    const hasStale = f.staleMarkers.some((sm) => promptText.includes(sm));
    if (hasLatest) rawLatestHits++;
    if (hasLatest && !hasStale) cleanLatestHits++;
  }

  return {
    criticalConstraintRecall,
    latestStateAccuracy: Math.round((cleanLatestHits / FILE_STATE_GROUND_TRUTH.length) * 10000) / 100,
    latestStateRawRecall: Math.round((rawLatestHits / FILE_STATE_GROUND_TRUTH.length) * 10000) / 100,
  };
}

function computeChunkStaleStatus(
  chunk: LinearChunk,
  latestResourceVersion: ReadonlyMap<string, number>,
  retractedGroups: ReadonlySet<string>,
): boolean {
  if (chunk.deadEndGroup && retractedGroups.has(chunk.deadEndGroup)) {
    return true;
  }
  if (chunk.resourceKey && chunk.version !== undefined) {
    const latest = latestResourceVersion.get(chunk.resourceKey) ?? chunk.version;
    if (chunk.version < latest) return true;
  }
  return false;
}

function summarizeStrategyMetrics(
  strategy: BenchmarkStrategyName,
  budgetTokens: number,
  finalPromptText: string,
  turnTelemetry: TurnTelemetry[],
): Omit<StrategyBenchmarkMetrics, "effectiveCostIndex" | "costIndexVsLinearUncached"> {
  const accuracy = evaluateFinalPromptAccuracy(finalPromptText);
  const lastTurn = turnTelemetry[turnTelemetry.length - 1]!;

  let totalPromptTokens = 0;
  let totalCachedTokens = 0;
  let totalUncachedTokens = 0;
  let totalStaleTokens = 0;
  let warmPromptTokens = 0;
  let warmCachedTokens = 0;

  for (const t of turnTelemetry) {
    totalPromptTokens += t.promptTokens;
    totalCachedTokens += t.prefixCachedTokens;
    totalUncachedTokens += t.uncachedTokens;
    totalStaleTokens += t.staleTokens;
    if (t.turn > 1) {
      warmPromptTokens += t.promptTokens;
      warmCachedTokens += t.prefixCachedTokens;
    }
  }

  const kvCacheHitRate =
    totalPromptTokens > 0 ? Math.round((totalCachedTokens / totalPromptTokens) * 10000) / 100 : 0;
  const warmTurnKvCacheHitRate =
    warmPromptTokens > 0 ? Math.round((warmCachedTokens / warmPromptTokens) * 10000) / 100 : 0;
  const staleTokenRatio =
    lastTurn.promptTokens > 0
      ? Math.round((lastTurn.staleTokens / lastTurn.promptTokens) * 10000) / 100
      : 0;
  const cumulativeStaleTokenRatio =
    totalPromptTokens > 0 ? Math.round((totalStaleTokens / totalPromptTokens) * 10000) / 100 : 0;
  const effectiveBillableTokens =
    Math.round((totalCachedTokens * CACHED_TOKEN_COST_WEIGHT + totalUncachedTokens * 1.0) * 100) / 100;

  return {
    strategy,
    budgetTokens,
    finalTurnTokens: lastTurn.promptTokens,
    budgetCompliant: turnTelemetry.every((t) => !t.budgetExceeded),
    criticalConstraintRecall: accuracy.criticalConstraintRecall,
    latestStateAccuracy: accuracy.latestStateAccuracy,
    latestStateRawRecall: accuracy.latestStateRawRecall,
    staleTokenRatio,
    cumulativeStaleTokenRatio,
    kvCacheHitRate,
    warmTurnKvCacheHitRate,
    totalPromptTokens,
    totalCachedTokens,
    totalUncachedTokens,
    effectiveBillableTokens,
    turnTelemetry,
  };
}

export function formatBenchmarkMarkdownTable(report: BenchmarkReport): string {
  const order: BenchmarkStrategyName[] = [
    "linear_full",
    "linear_sliding_window",
    "naive_vector_rag",
    "context_graph_two_zone",
  ];

  const header = [
    "| Strategy | Budget Compliant | Final Tokens (Budget) | Critical Constraint Recall | Latest State Accuracy (No Conflict) | Stale Token Ratio | Prefix KV-Cache Hit Rate | Effective Cost Index (Sliding=100) |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ];

  const rows = order.map((name) => {
    const m = report.strategies[name];
    const compliant = m.budgetCompliant ? "YES" : "NO (Over)";
    return `| \`${m.strategy}\` | ${compliant} | ${m.finalTurnTokens} / ${m.budgetTokens} | ${m.criticalConstraintRecall.toFixed(1)}% | ${m.latestStateAccuracy.toFixed(1)}% | ${m.staleTokenRatio.toFixed(1)}% | ${m.kvCacheHitRate.toFixed(1)}% | **${m.effectiveCostIndex.toFixed(1)}** (${Math.round(m.effectiveBillableTokens)} tok) |`;
  });

  return [...header, ...rows].join("\n");
}

/**
 * Runs the 32-turn multi-file engineering agent benchmark across all 4 context strategies.
 */
export function runBenchmark(options?: BenchmarkOptions): BenchmarkReport {
  const budgetRatio =
    typeof options?.budgetRatio === "number" && options.budgetRatio >= 0.2 && options.budgetRatio <= 0.8
      ? options.budgetRatio
      : 0.4;

  const trajectory = build32TurnTrajectory();

  // Pre-hydrate linear chunks per turn so we know the exact full linear token count
  const hydratedByTurn: Array<{
    spec: TurnSpec;
    chunks: LinearChunk[];
  }> = trajectory.map((spec) => ({
    spec,
    chunks: spec.linearChunks.map((c) => {
      const formatted = `[${c.role}] ${c.text}`;
      return {
        id: c.id,
        turn: spec.turn,
        role: c.role,
        text: formatted,
        tokens: countTokens(formatted) + 1,
        embedding: embedText(c.text, c.resourceKey ?? null),
        resourceKey: c.resourceKey,
        version: c.version,
        deadEndGroup: c.deadEndGroup,
        isSystemRoot: c.isSystemRoot,
      };
    }),
  }));

  // Compute full linear final turn size to set exact 40% token budget
  const allLinearText = hydratedByTurn
    .flatMap((t) => t.chunks.map((c) => c.text))
    .join("\n");
  const fullLinearFinalTokens = tokenizeToIds(allLinearText).length;
  const tokenBudget = Math.max(512, Math.floor(fullLinearFinalTokens * budgetRatio));

  // ============================================================================
  // 1. Strategy: linear_full (Append-only full linear history)
  // ============================================================================
  const linearFullTelemetry: TurnTelemetry[] = [];
  const linearHistory: LinearChunk[] = [];
  const latestResVer1 = new Map<string, number>();
  const retractedGroups1 = new Set<string>();
  let prevLinearFullIds: number[] = [];
  let finalLinearFullText = "";

  for (const { spec, chunks } of hydratedByTurn) {
    for (const c of chunks) {
      linearHistory.push(c);
      if (c.resourceKey && c.version !== undefined) {
        latestResVer1.set(c.resourceKey, Math.max(latestResVer1.get(c.resourceKey) ?? 0, c.version));
      }
    }
    for (const g of spec.retractedDeadEndGroupsAfterTurn ?? []) {
      retractedGroups1.add(g);
    }

    const promptText = linearHistory.map((c) => c.text).join("\n");
    const tokenIds = tokenizeToIds(promptText);
    const cached = prevLinearFullIds.length > 0 ? longestCommonPrefixLength(prevLinearFullIds, tokenIds) : 0;
    const staleTokens = linearHistory
      .filter((c) => computeChunkStaleStatus(c, latestResVer1, retractedGroups1))
      .reduce((acc, c) => acc + c.tokens, 0);

    linearFullTelemetry.push({
      turn: spec.turn,
      promptTokens: tokenIds.length,
      prefixCachedTokens: cached,
      uncachedTokens: tokenIds.length - cached,
      hitRate: prevLinearFullIds.length > 0 ? cached / tokenIds.length : 0,
      staleTokens: Math.min(tokenIds.length, staleTokens),
      budgetExceeded: tokenIds.length > tokenBudget,
    });
    prevLinearFullIds = tokenIds;
    finalLinearFullText = promptText;
  }

  // ============================================================================
  // 2. Strategy: linear_sliding_window (Head-truncated to fit tokenBudget)
  // ============================================================================
  const slidingTelemetry: TurnTelemetry[] = [];
  const slidingPool: LinearChunk[] = [];
  const latestResVer2 = new Map<string, number>();
  const retractedGroups2 = new Set<string>();
  let prevSlidingIds: number[] = [];
  let finalSlidingText = "";

  for (const { spec, chunks } of hydratedByTurn) {
    for (const c of chunks) {
      slidingPool.push(c);
      if (c.resourceKey && c.version !== undefined) {
        latestResVer2.set(c.resourceKey, Math.max(latestResVer2.get(c.resourceKey) ?? 0, c.version));
      }
    }
    for (const g of spec.retractedDeadEndGroupsAfterTurn ?? []) {
      retractedGroups2.add(g);
    }

    const sysRoot = slidingPool[0]?.isSystemRoot ? slidingPool[0] : undefined;
    const tailCandidates = sysRoot ? slidingPool.slice(1) : slidingPool.slice(0);
    const budgetForTail = tokenBudget - (sysRoot ? sysRoot.tokens : 0) - 4;

    const keptTail: LinearChunk[] = [];
    let running = 0;
    for (let i = tailCandidates.length - 1; i >= 0; i--) {
      const item = tailCandidates[i]!;
      if (running + item.tokens <= budgetForTail || keptTail.length === 0) {
        keptTail.push(item);
        running += item.tokens;
      } else {
        break;
      }
    }
    keptTail.reverse();
    const windowChunks = sysRoot ? [sysRoot, ...keptTail] : keptTail;

    const promptText = windowChunks.map((c) => c.text).join("\n");
    const tokenIds = tokenizeToIds(promptText);
    const cached = prevSlidingIds.length > 0 ? longestCommonPrefixLength(prevSlidingIds, tokenIds) : 0;
    const staleTokens = windowChunks
      .filter((c) => computeChunkStaleStatus(c, latestResVer2, retractedGroups2))
      .reduce((acc, c) => acc + c.tokens, 0);

    slidingTelemetry.push({
      turn: spec.turn,
      promptTokens: tokenIds.length,
      prefixCachedTokens: cached,
      uncachedTokens: tokenIds.length - cached,
      hitRate: prevSlidingIds.length > 0 ? cached / tokenIds.length : 0,
      staleTokens: Math.min(tokenIds.length, staleTokens),
      budgetExceeded: tokenIds.length > tokenBudget,
    });
    prevSlidingIds = tokenIds;
    finalSlidingText = promptText;
  }

  // ============================================================================
  // 3. Strategy: naive_vector_rag (Top-K cosine similarity re-ranked chunks)
  // ============================================================================
  const ragTelemetry: TurnTelemetry[] = [];
  const ragPool: LinearChunk[] = [];
  const latestResVer3 = new Map<string, number>();
  const retractedGroups3 = new Set<string>();
  let prevRagIds: number[] = [];
  let finalRagText = "";

  for (const { spec, chunks } of hydratedByTurn) {
    for (const c of chunks) {
      ragPool.push(c);
      if (c.resourceKey && c.version !== undefined) {
        latestResVer3.set(c.resourceKey, Math.max(latestResVer3.get(c.resourceKey) ?? 0, c.version));
      }
    }
    for (const g of spec.retractedDeadEndGroupsAfterTurn ?? []) {
      retractedGroups3.add(g);
    }

    const sysRoot = ragPool[0]?.isSystemRoot ? ragPool[0] : undefined;
    const latestChunk = chunks[chunks.length - 1]!;
    const candidates = ragPool.filter(
      (c) => c.id !== sysRoot?.id && c.id !== latestChunk.id,
    );

    const qVec = embedText(spec.focusQuery);
    const scored = candidates
      .map((c) => ({ chunk: c, sim: cosineSimilarity(qVec, c.embedding) }))
      .sort((a, b) => b.sim - a.sim);

    const picked: LinearChunk[] = [];
    let used = (sysRoot ? sysRoot.tokens : 0) + latestChunk.tokens + 4;
    for (const item of scored) {
      if (used + item.chunk.tokens <= tokenBudget) {
        picked.push(item.chunk);
        used += item.chunk.tokens;
      }
    }

    // Standard Naive Vector RAG orders retrieved context by relevance score
    const ragSelected = [
      ...(sysRoot ? [sysRoot] : []),
      ...picked,
      ...(latestChunk.id !== sysRoot?.id ? [latestChunk] : []),
    ];

    const promptText = ragSelected.map((c) => c.text).join("\n");
    const tokenIds = tokenizeToIds(promptText);
    const cached = prevRagIds.length > 0 ? longestCommonPrefixLength(prevRagIds, tokenIds) : 0;
    const staleTokens = ragSelected
      .filter((c) => computeChunkStaleStatus(c, latestResVer3, retractedGroups3))
      .reduce((acc, c) => acc + c.tokens, 0);

    ragTelemetry.push({
      turn: spec.turn,
      promptTokens: tokenIds.length,
      prefixCachedTokens: cached,
      uncachedTokens: tokenIds.length - cached,
      hitRate: prevRagIds.length > 0 ? cached / tokenIds.length : 0,
      staleTokens: Math.min(tokenIds.length, staleTokens),
      budgetExceeded: tokenIds.length > tokenBudget,
    });
    prevRagIds = tokenIds;
    finalRagText = promptText;
  }

  // ============================================================================
  // 4. Strategy: context_graph_two_zone (Two-Zone KV-Cache-Aware Property DAG)
  // ============================================================================
  const store = new GraphStore(":memory:");
  const graph = store.createGraph({
    id: "bench_32_turn",
    name: "32-Turn Gateway Engineering Benchmark",
    defaultBudget: tokenBudget,
  });

  const cgTelemetry: TurnTelemetry[] = [];
  let finalCgText = "";

  for (const { spec } of hydratedByTurn) {
    store.ingestTurn(graph.id, spec.ingest);
    const compiled = compileContext(store, {
      graphId: graph.id,
      focusQuery: spec.focusQuery,
      activeScope: spec.scope,
      maxTokens: tokenBudget,
      recordCacheState: true,
    });

    // Stale token verification on compiled nodes (any superseded/retracted leakage)
    let cgStaleTokens = 0;
    for (const entry of [...compiled.zone1Nodes, ...compiled.zone2Nodes]) {
      const node = store.getNode(graph.id, entry.id);
      if (node.status !== "active" || entry.renderedText.includes("DEADEND_")) {
        cgStaleTokens += entry.tokens;
      }
    }

    cgTelemetry.push({
      turn: spec.turn,
      promptTokens: compiled.kvCache.totalTokens,
      prefixCachedTokens: compiled.kvCache.prefixCachedTokens,
      uncachedTokens: compiled.kvCache.uncachedTokens,
      hitRate: compiled.kvCache.prefixHitRate,
      staleTokens: cgStaleTokens,
      budgetExceeded: compiled.kvCache.totalTokens > tokenBudget,
    });
    finalCgText = compiled.promptText;
  }

  store.close();

  const rawLinearFull = summarizeStrategyMetrics(
    "linear_full",
    tokenBudget,
    finalLinearFullText,
    linearFullTelemetry,
  );
  const rawSliding = summarizeStrategyMetrics(
    "linear_sliding_window",
    tokenBudget,
    finalSlidingText,
    slidingTelemetry,
  );
  const rawRag = summarizeStrategyMetrics(
    "naive_vector_rag",
    tokenBudget,
    finalRagText,
    ragTelemetry,
  );
  const rawCg = summarizeStrategyMetrics(
    "context_graph_two_zone",
    tokenBudget,
    finalCgText,
    cgTelemetry,
  );

  const slidingRefCost = rawSliding.effectiveBillableTokens || 1;
  const linearUncachedRefCost = rawLinearFull.totalPromptTokens || 1;

  const enrich = (
    m: Omit<StrategyBenchmarkMetrics, "effectiveCostIndex" | "costIndexVsLinearUncached">,
  ): StrategyBenchmarkMetrics => ({
    ...m,
    effectiveCostIndex: Math.round((m.effectiveBillableTokens / slidingRefCost) * 10000) / 100,
    costIndexVsLinearUncached:
      Math.round((m.effectiveBillableTokens / linearUncachedRefCost) * 10000) / 100,
  });

  const strategies: Record<BenchmarkStrategyName, StrategyBenchmarkMetrics> = {
    linear_full: enrich(rawLinearFull),
    linear_sliding_window: enrich(rawSliding),
    naive_vector_rag: enrich(rawRag),
    context_graph_two_zone: enrich(rawCg),
  };

  const report: BenchmarkReport = {
    scenarioName: "32-Turn Multi-File Payment Gateway Engineering Agent Benchmark",
    totalTurns: trajectory.length,
    fullLinearFinalTokens,
    budgetRatio,
    tokenBudget,
    cachedTokenWeight: CACHED_TOKEN_COST_WEIGHT,
    generatedAt: new Date().toISOString(),
    strategies,
    markdownTable: "",
  };
  report.markdownTable = formatBenchmarkMarkdownTable(report);
  return report;
}
