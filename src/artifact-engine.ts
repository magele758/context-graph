import {
  compactCodeExcerpt,
  compactCommandExcerpt,
  extractAppFileReferences,
  sanitizeNeutralText,
  stripVendorStackFrames,
  truncateAtWordBoundary,
} from "./excerpt.ts";
import type { GraphStore } from "./graph-store.ts";
import {
  ContextGraphError,
  type ClassifiedToolEvent,
  type EditReplacementPair,
  type FileOpKind,
  type GraphEdge,
  type GraphNode,
  type IngestArtifactsOptions,
  type IngestArtifactsResult,
  type PrunedTrajectoryEvent,
  type PruneTrajectoryResult,
  type ToolTrajectoryEvent,
} from "./types.ts";
import { countTokens, deterministicId } from "./vector.ts";

const READ_FILE_TOOLS = new Set(["read_file", "view_file"]);
const PARTIAL_EDIT_TOOLS = new Set([
  "edit_file",
  "apply_patch",
  "replace_file_content",
  "multi_replace_file_content",
  "patch_file",
  "notebook_edit",
]);
const FULL_WRITE_TOOLS = new Set(["write_file", "write_to_file", "create_file"]);
const DELETE_FILE_TOOLS = new Set(["delete_file", "remove_file"]);

const COMMAND_TOOLS = new Set([
  "bash",
  "exec",
  "shell",
  "bg_run",
  "run_command",
  "work_evidence",
]);

const SEARCH_TOOLS = new Set([
  "grep",
  "glob",
  "find_by_name",
  "grep_search",
]);

const WRITE_PAYLOAD_ARG_KEYS = [
  "content",
  "CodeContent",
  "text",
  "data",
  "body",
  "file_content",
] as const;

const DEFAULT_CODE_EXCERPT_CHARS = 600;
const DEFAULT_CMD_EXCERPT_CHARS = 420;
export const MAX_TOOL_EVENTS_PER_BATCH = 2000 as const;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Validates raw HTTP / MCP `events` payload strictly, enforcing batch bounds
 * (`<= 2000`) and full per-item schema safety so malformed payloads always fail
 * cleanly with `ContextGraphError("VALIDATION_ERROR", 400)` instead of HTTP 500.
 */
export function validateToolEvents(raw: unknown): ToolTrajectoryEvent[] {
  if (!Array.isArray(raw)) {
    throw new ContextGraphError(
      "Request body must include an 'events' array",
      "VALIDATION_ERROR",
      400,
    );
  }
  if (raw.length > MAX_TOOL_EVENTS_PER_BATCH) {
    throw new ContextGraphError(
      `'events' array exceeds maximum batch size of ${MAX_TOOL_EVENTS_PER_BATCH}`,
      "VALIDATION_ERROR",
      400,
    );
  }

  const validated: ToolTrajectoryEvent[] = new Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i];
    if (!isPlainObject(item)) {
      throw new ContextGraphError(
        `events[${i}] must be a non-null plain object`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (typeof item.callId !== "string" || item.callId.trim().length === 0) {
      throw new ContextGraphError(
        `events[${i}].callId must be a non-empty string`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (typeof item.tool !== "string" || item.tool.trim().length === 0) {
      throw new ContextGraphError(
        `events[${i}].tool must be a non-empty string`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (typeof item.output !== "string") {
      throw new ContextGraphError(
        `events[${i}].output must be a string`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (typeof item.ok !== "boolean") {
      throw new ContextGraphError(
        `events[${i}].ok must be a boolean`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (item.args !== undefined && !isPlainObject(item.args)) {
      throw new ContextGraphError(
        `events[${i}].args must be a plain object when provided`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (item.machineId !== undefined && typeof item.machineId !== "string") {
      throw new ContextGraphError(
        `events[${i}].machineId must be a string when provided`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (item.scope !== undefined && typeof item.scope !== "string") {
      throw new ContextGraphError(
        `events[${i}].scope must be a string when provided`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (
      item.turn !== undefined &&
      (typeof item.turn !== "number" || !Number.isInteger(item.turn) || item.turn < 1)
    ) {
      throw new ContextGraphError(
        `events[${i}].turn must be a positive integer when provided`,
        "VALIDATION_ERROR",
        400,
      );
    }
    if (
      item.seq !== undefined &&
      (typeof item.seq !== "number" || !Number.isInteger(item.seq) || item.seq < 0)
    ) {
      throw new ContextGraphError(
        `events[${i}].seq must be a non-negative integer when provided`,
        "VALIDATION_ERROR",
        400,
      );
    }

    validated[i] = {
      callId: item.callId,
      tool: item.tool,
      ok: item.ok,
      output: item.output,
      ...(item.args !== undefined ? { args: item.args } : {}),
      ...(item.machineId !== undefined ? { machineId: item.machineId } : {}),
      ...(item.scope !== undefined ? { scope: item.scope } : {}),
      ...(item.turn !== undefined ? { turn: item.turn } : {}),
      ...(item.seq !== undefined ? { seq: item.seq } : {}),
    };
  }
  return validated;
}

/**
 * Validates optional ingestion fields on HTTP `/ingest-artifacts` and MCP `cg_ingest_artifacts`.
 */
export function validateIngestArtifactsOptions(
  raw: Record<string, unknown>,
): IngestArtifactsOptions {
  if (raw.options !== undefined && !isPlainObject(raw.options)) {
    throw new ContextGraphError(
      "'options' must be a plain object when provided",
      "VALIDATION_ERROR",
      400,
    );
  }
  if (raw.prune !== undefined && typeof raw.prune !== "boolean") {
    throw new ContextGraphError("'prune' must be a boolean", "VALIDATION_ERROR", 400);
  }
  if (raw.compile !== undefined && typeof raw.compile !== "boolean") {
    throw new ContextGraphError("'compile' must be a boolean", "VALIDATION_ERROR", 400);
  }

  const nested = isPlainObject(raw.options) ? raw.options : {};
  const merged: Record<string, unknown> = {
    ...nested,
    ...(raw.scope !== undefined ? { scope: raw.scope } : {}),
    ...(raw.turn !== undefined ? { turn: raw.turn } : {}),
    ...(raw.machineId !== undefined ? { machineId: raw.machineId } : {}),
    ...(raw.maxCodeExcerptChars !== undefined
      ? { maxCodeExcerptChars: raw.maxCodeExcerptChars }
      : {}),
    ...(raw.maxCommandExcerptChars !== undefined
      ? { maxCommandExcerptChars: raw.maxCommandExcerptChars }
      : {}),
    ...(raw.autoCreate !== undefined ? { autoCreate: raw.autoCreate } : {}),
    ...(raw.graphName !== undefined ? { graphName: raw.graphName } : {}),
    ...(raw.responseMode !== undefined ? { responseMode: raw.responseMode } : {}),
  };

  if (merged.scope !== undefined && typeof merged.scope !== "string") {
    throw new ContextGraphError("'scope' must be a string", "VALIDATION_ERROR", 400);
  }
  if (merged.machineId !== undefined && typeof merged.machineId !== "string") {
    throw new ContextGraphError("'machineId' must be a string", "VALIDATION_ERROR", 400);
  }
  if (
    merged.turn !== undefined &&
    (typeof merged.turn !== "number" || !Number.isInteger(merged.turn) || merged.turn < 1)
  ) {
    throw new ContextGraphError("'turn' must be a positive integer", "VALIDATION_ERROR", 400);
  }
  if (
    merged.maxCodeExcerptChars !== undefined &&
    (typeof merged.maxCodeExcerptChars !== "number" ||
      !Number.isInteger(merged.maxCodeExcerptChars) ||
      merged.maxCodeExcerptChars < 1 ||
      merged.maxCodeExcerptChars > 20000)
  ) {
    throw new ContextGraphError(
      "'maxCodeExcerptChars' must be an integer between 1 and 20000",
      "VALIDATION_ERROR",
      400,
    );
  }
  if (
    merged.maxCommandExcerptChars !== undefined &&
    (typeof merged.maxCommandExcerptChars !== "number" ||
      !Number.isInteger(merged.maxCommandExcerptChars) ||
      merged.maxCommandExcerptChars < 1 ||
      merged.maxCommandExcerptChars > 20000)
  ) {
    throw new ContextGraphError(
      "'maxCommandExcerptChars' must be an integer between 1 and 20000",
      "VALIDATION_ERROR",
      400,
    );
  }
  if (merged.autoCreate !== undefined && typeof merged.autoCreate !== "boolean") {
    throw new ContextGraphError("'autoCreate' must be a boolean", "VALIDATION_ERROR", 400);
  }
  let normalizedGraphName: string | undefined;
  if (merged.graphName !== undefined) {
    if (typeof merged.graphName !== "string") {
      throw new ContextGraphError(
        "'graphName' must be a non-empty string of at most 120 characters",
        "VALIDATION_ERROR",
        400,
      );
    }
    const trimmed = merged.graphName.trim();
    if (trimmed.length === 0 || trimmed.length > 120 || merged.graphName.length > 120) {
      throw new ContextGraphError(
        "'graphName' must be a non-empty string of at most 120 characters",
        "VALIDATION_ERROR",
        400,
      );
    }
    normalizedGraphName = trimmed;
  }

  if (
    merged.responseMode !== undefined &&
    merged.responseMode !== "full" &&
    merged.responseMode !== "slim"
  ) {
    throw new ContextGraphError(
      "'responseMode' must be 'full' or 'slim'",
      "VALIDATION_ERROR",
      400,
    );
  }

  return {
    ...(merged.scope !== undefined ? { scope: merged.scope } : {}),
    ...(merged.turn !== undefined ? { turn: merged.turn } : {}),
    ...(merged.machineId !== undefined ? { machineId: merged.machineId } : {}),
    ...(merged.maxCodeExcerptChars !== undefined
      ? { maxCodeExcerptChars: merged.maxCodeExcerptChars }
      : {}),
    ...(merged.maxCommandExcerptChars !== undefined
      ? { maxCommandExcerptChars: merged.maxCommandExcerptChars }
      : {}),
    ...(merged.autoCreate !== undefined ? { autoCreate: merged.autoCreate } : {}),
    ...(normalizedGraphName !== undefined ? { graphName: normalizedGraphName } : {}),
    ...(merged.responseMode !== undefined ? { responseMode: merged.responseMode } : {}),
  };
}

function readStringArg(
  args: Record<string, unknown> | undefined,
  keys: readonly string[],
): string | undefined {
  if (!args) return undefined;
  for (const key of keys) {
    const val = args[key];
    if (typeof val === "string" && val.trim().length > 0) {
      return val.trim();
    }
  }
  return undefined;
}

function readRawStringArg(
  args: Record<string, unknown> | undefined,
  keys: readonly string[],
): string | undefined {
  if (!args) return undefined;
  for (const key of keys) {
    const val = args[key];
    if (typeof val === "string" && val.length > 0) {
      return val;
    }
  }
  return undefined;
}

function readFiniteIntArg(
  args: Record<string, unknown> | undefined,
  keys: readonly string[],
): { key: string; value: number } | undefined {
  if (!args) return undefined;
  for (const key of keys) {
    const val = args[key];
    if (typeof val === "number" && Number.isFinite(val)) {
      return { key, value: Math.floor(val) };
    }
    if (typeof val === "string" && /^\d+$/.test(val.trim())) {
      return { key, value: Number.parseInt(val.trim(), 10) };
    }
  }
  return undefined;
}

function trimTrailingSlashes(s: string): string {
  let end = s.length;
  while (end > 0 && s.charCodeAt(end - 1) === 47) {
    end--;
  }
  return end === s.length ? s : s.slice(0, end);
}

export function normalizeArtifactPath(rawPath: string): string {
  const unified = rawPath.trim().replace(/\\/g, "/");
  const isAbs = unified.startsWith("/");
  const collapsed = unified.replace(/\/{2,}/g, "/");
  if (isAbs) {
    return collapsed.length > 1 ? trimTrailingSlashes(collapsed) || "/" : collapsed;
  }
  const stripped = trimTrailingSlashes(collapsed.replace(/^(?:\.\/)+/, ""));
  return stripped || ".";
}

/**
 * Normalizes a shell command into a canonical resource key while strictly
 * preserving directory scoping (`cd packages/a && npm test` != `cd packages/b && npm test`).
 * Strips only no-op directory prefixes (`cd . &&`, `cd ./ &&`, `cd $PWD &&`).
 */
export function normalizeCommandKey(command: string): string {
  let s = command.trim().replace(/\s+/g, " ");
  // Strip leading no-op `cd . &&` / `cd ./ &&` / `cd "$PWD" &&` prefixes
  const noopCdRe =
    /^cd\s+(?:(?:'\.\/?'|"\.\/?"|\.\/?)|(?:'\$PWD'|"\$PWD"|\$PWD|'\$\{PWD\}'|"\$\{PWD\}"|\$\{PWD\}))\s*(?:&&|;)\s*/i;
  while (noopCdRe.test(s)) {
    s = s.replace(noopCdRe, "").trim();
  }
  // Normalize trailing slash on real `cd <subdir>/ &&` while preserving `<subdir>`
  s = s.replace(/^(cd\s+[^\s;&|]+?)\/+(\s*(?:&&|;))/i, "$1$2");
  return s;
}

function extractPatchFileInfo(
  args: Record<string, unknown> | undefined,
): { path: string; opKind: FileOpKind } | undefined {
  const patch = readRawStringArg(args, ["patch", "diff", "input"]);
  if (!patch) return undefined;

  const ops: Array<{ path: string; opKind: FileOpKind }> = [];
  const headerRe = /^\*\*\*\s+(Update|Add|Delete)\s+File:\s+(.+?)\s*$/gm;
  let m: RegExpExecArray | null;
  while ((m = headerRe.exec(patch)) !== null) {
    const verb = m[1]!.toLowerCase();
    const filePath = m[2]!.trim();
    if (!filePath) continue;
    const opKind: FileOpKind =
      verb === "delete" ? "delete" : verb === "add" ? "full_write" : "partial_edit";
    ops.push({ path: filePath, opKind });
  }
  if (ops.length === 1) {
    return ops[0]!;
  }
  if (ops.length > 1) {
    return undefined;
  }

  // Unified diff fallback (`+++ b/<path>`)
  const unifiedMatches = Array.from(patch.matchAll(/^\+\+\+\s+(?:b\/)?([^\s\t]+)$/gm))
    .map((x) => x[1]!)
    .filter((p) => p !== "/dev/null");
  if (unifiedMatches.length === 1) {
    return { path: unifiedMatches[0]!, opKind: "partial_edit" };
  }
  return undefined;
}

function extractRangeInfo(
  args: Record<string, unknown> | undefined,
): { rangeKey: string; rangeLabel: string } | undefined {
  const startArg = readFiniteIntArg(args, [
    "offset_line",
    "start_line",
    "startLine",
    "StartLine",
    "from_line",
    "fromLine",
    "offset",
  ]);
  const endArg = readFiniteIntArg(args, ["end_line", "endLine", "EndLine", "to_line", "toLine"]);
  const limitArg = readFiniteIntArg(args, ["limit", "num_lines", "line_count", "max_lines"]);

  const start = startArg?.value;
  const end = endArg?.value;
  const limit = limitArg?.value;

  // No upper bound and starting from line 0/1 => full read
  if (end === undefined && limit === undefined && (start === undefined || start <= 1)) {
    return undefined;
  }

  if (start !== undefined && end !== undefined) {
    return { rangeKey: `${start}-${end}`, rangeLabel: `L${start}-${end}` };
  }
  if (start !== undefined && limit !== undefined) {
    if (startArg!.key === "offset" && start === 0) {
      return { rangeKey: `0+${limit}`, rangeLabel: `L1-${limit}` };
    }
    const computedEnd = start + Math.max(1, limit) - 1;
    return { rangeKey: `${start}-${computedEnd}`, rangeLabel: `L${start}-${computedEnd}` };
  }
  if (limit !== undefined) {
    return { rangeKey: `1-${Math.max(1, limit)}`, rangeLabel: `L1-${Math.max(1, limit)}` };
  }
  if (end !== undefined) {
    return { rangeKey: `1-${Math.max(1, end)}`, rangeLabel: `L1-${Math.max(1, end)}` };
  }
  if (start !== undefined && start > 1) {
    return { rangeKey: `${start}-eof`, rangeLabel: `L${start}-eof` };
  }
  return undefined;
}

const MAX_EDIT_PAIRS = 16 as const;

function readDefinedStringArg(
  args: Record<string, unknown> | undefined,
  keys: readonly string[],
): string | undefined {
  if (!args) return undefined;
  for (const key of keys) {
    const val = args[key];
    if (typeof val === "string") {
      return val;
    }
  }
  return undefined;
}

function trimSharedContextPair(rawOld: string, rawNew: string): EditReplacementPair | null {
  const cleanOld = sanitizeNeutralText(rawOld);
  const cleanNew = sanitizeNeutralText(rawNew);
  if (cleanOld === cleanNew) return null;

  const oldLines = cleanOld.length > 0 ? cleanOld.split("\n") : [];
  const newLines = cleanNew.length > 0 ? cleanNew.split("\n") : [];

  let prefix = 0;
  while (
    prefix < oldLines.length &&
    prefix < newLines.length &&
    oldLines[prefix] === newLines[prefix]
  ) {
    prefix++;
  }
  let oldEnd = oldLines.length - 1;
  let newEnd = newLines.length - 1;
  while (
    oldEnd >= prefix &&
    newEnd >= prefix &&
    oldLines[oldEnd] === newLines[newEnd]
  ) {
    oldEnd--;
    newEnd--;
  }

  const trimmedOld = oldLines
    .slice(prefix, oldEnd + 1)
    .map((l) => l.trimEnd())
    .join("\n")
    .trim();
  const trimmedNew = newLines
    .slice(prefix, newEnd + 1)
    .map((l) => l.trimEnd())
    .join("\n")
    .trim();

  if (!trimmedOld && !trimmedNew) return null;
  if (trimmedOld === trimmedNew) return null;
  return { oldText: trimmedOld, newText: trimmedNew };
}

/**
 * Strictly linear O(N) extraction of `-`/`+` replacement pairs from a unified
 * or compact diff string (skips diff headers and pairs consecutive `-` and `+` runs).
 */
export function extractPairsFromDiffText(diffText: string): EditReplacementPair[] {
  if (!diffText) return [];
  const lines = sanitizeNeutralText(diffText).split("\n");
  const pairs: EditReplacementPair[] = [];

  let removedRun: string[] = [];
  let addedRun: string[] = [];

  const stripDiffMarker = (line: string): string => {
    // Strip leading `-` or `+`, and strip a single separating space when formatted as `- text` / `+ text`
    const rest = line.slice(1);
    return rest.startsWith(" ") ? rest.slice(1) : rest;
  };

  const flushHunk = () => {
    if (removedRun.length === 0 && addedRun.length === 0) return;
    if (pairs.length < MAX_EDIT_PAIRS) {
      const pair = trimSharedContextPair(removedRun.join("\n"), addedRun.join("\n"));
      if (pair) pairs.push(pair);
    }
    removedRun = [];
    addedRun = [];
  };

  for (const rawLine of lines) {
    if (
      /^(?:---|\+\+\+|\*\*\*|@@|diff --git|index )/.test(rawLine)
    ) {
      flushHunk();
      continue;
    }
    if (rawLine.startsWith("-")) {
      if (addedRun.length > 0) {
        flushHunk();
      }
      removedRun.push(stripDiffMarker(rawLine));
    } else if (rawLine.startsWith("+")) {
      addedRun.push(stripDiffMarker(rawLine));
    } else {
      flushHunk();
    }
  }
  flushHunk();
  return pairs;
}

/**
 * Pure linear O(N) extractor returning structured `{ oldText, newText }` edit
 * replacement pairs from single-edit args, multi-edit arrays (`ReplacementChunks`
 * / `edits`), or fallback unified/apply_patch diff hunks (capped at 16 pairs).
 */
export function extractEditPairs(
  args: Record<string, unknown> | undefined,
): EditReplacementPair[] {
  if (!args) return [];
  const pairs: EditReplacementPair[] = [];

  const oldText = readDefinedStringArg(args, [
    "old_string",
    "oldString",
    "old_text",
    "oldText",
    "TargetContent",
    "search",
  ]);
  const newText = readDefinedStringArg(args, [
    "new_string",
    "newString",
    "new_text",
    "newText",
    "ReplacementContent",
    "replace",
  ]);
  if (oldText !== undefined || newText !== undefined) {
    const pair = trimSharedContextPair(oldText ?? "", newText ?? "");
    if (pair) pairs.push(pair);
    return pairs;
  }

  const chunks = (args.ReplacementChunks ?? args.edits) as unknown;
  if (Array.isArray(chunks) && chunks.length > 0) {
    for (const chunk of chunks) {
      if (pairs.length >= MAX_EDIT_PAIRS) break;
      if (!chunk || typeof chunk !== "object" || Array.isArray(chunk)) continue;
      const rec = chunk as Record<string, unknown>;
      const cOld = readDefinedStringArg(rec, [
        "old_string",
        "oldString",
        "old_text",
        "oldText",
        "TargetContent",
        "search",
      ]);
      const cNew = readDefinedStringArg(rec, [
        "new_string",
        "newString",
        "new_text",
        "newText",
        "ReplacementContent",
        "replace",
      ]);
      if (cOld === undefined && cNew === undefined) continue;
      const pair = trimSharedContextPair(cOld ?? "", cNew ?? "");
      if (pair) pairs.push(pair);
    }
    if (pairs.length > 0) return pairs;
  }

  const patch = readRawStringArg(args, ["patch", "diff"]);
  if (patch) {
    return extractPairsFromDiffText(patch);
  }

  return pairs;
}

export function formatEditPairsAsDiff(pairs: readonly EditReplacementPair[]): string {
  const parts: string[] = [];
  for (const p of pairs) {
    const diff = formatLineReplacementDiff(p.oldText, p.newText);
    if (diff.length > 0) parts.push(diff);
  }
  return parts.join("\n").trim();
}

function normalizeStoredEditPairs(raw: unknown): EditReplacementPair[] {
  if (!Array.isArray(raw)) return [];
  const out: EditReplacementPair[] = [];
  for (const item of raw) {
    if (out.length >= MAX_EDIT_PAIRS) break;
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const rec = item as Record<string, unknown>;
    if (typeof rec.oldText === "string" && typeof rec.newText === "string") {
      if (rec.oldText !== rec.newText && (rec.oldText.trim() || rec.newText.trim())) {
        out.push({ oldText: rec.oldText, newText: rec.newText });
      }
    }
  }
  return out;
}

/**
 * Accumulates chained partial edits on the same file:
 * - Transitive chain `A -> B` followed by `B -> C` collapses in-place to `A -> C`
 *   (and `A -> B -> A` cancels cleanly).
 * - Repeated edit with matching trimmed `oldText` updates to the newer `newText`.
 * - Distinct edits append cleanly without wiping earlier edits (capped at 16 most recent).
 */
export function mergeEditPairs(
  existing: readonly EditReplacementPair[],
  incoming: readonly EditReplacementPair[],
): EditReplacementPair[] {
  const merged: EditReplacementPair[] = existing.map((p) => ({
    oldText: p.oldText,
    newText: p.newText,
  }));

  for (const next of incoming) {
    const nextOldTrim = next.oldText.trim();
    const nextNewTrim = next.newText.trim();

    // 1. Transitive chain collapse: prior `A -> B` + incoming `B -> C` => `A -> C`
    if (nextOldTrim.length > 0) {
      const chainIdx = merged.findIndex(
        (p) => p.newText.trim().length > 0 && p.newText.trim() === nextOldTrim,
      );
      if (chainIdx >= 0) {
        const prior = merged[chainIdx]!;
        if (prior.oldText.trim() === nextNewTrim) {
          // Exact revert A -> B -> A cancels out cleanly
          merged.splice(chainIdx, 1);
        } else {
          merged[chainIdx] = { oldText: prior.oldText, newText: next.newText };
        }
        continue;
      }

      // 2. Same `oldText` replaced again -> update in-place with latest `newText`
      const sameOldIdx = merged.findIndex((p) => p.oldText.trim() === nextOldTrim);
      if (sameOldIdx >= 0) {
        merged[sameOldIdx] = { oldText: next.oldText, newText: next.newText };
        continue;
      }
    }

    // 3. Deduplicate identical pair
    if (merged.some((p) => p.oldText === next.oldText && p.newText === next.newText)) {
      continue;
    }

    merged.push({ oldText: next.oldText, newText: next.newText });
  }

  return merged.length > MAX_EDIT_PAIRS
    ? merged.slice(merged.length - MAX_EDIT_PAIRS)
    : merged;
}

function formatLineReplacementDiff(oldText: string, newText: string): string {
  const oldLines = sanitizeNeutralText(oldText).split("\n");
  const newLines = sanitizeNeutralText(newText).split("\n");

  // Trim identical leading and trailing context lines to isolate the actual edit delta
  let prefix = 0;
  while (
    prefix < oldLines.length &&
    prefix < newLines.length &&
    oldLines[prefix] === newLines[prefix]
  ) {
    prefix++;
  }
  let oldEnd = oldLines.length - 1;
  let newEnd = newLines.length - 1;
  while (
    oldEnd >= prefix &&
    newEnd >= prefix &&
    oldLines[oldEnd] === newLines[newEnd]
  ) {
    oldEnd--;
    newEnd--;
  }

  const out: string[] = [];
  for (let i = prefix; i <= oldEnd; i++) {
    const line = oldLines[i]!.trimEnd();
    if (line.length > 0) out.push(`- ${line}`);
  }
  for (let i = prefix; i <= newEnd; i++) {
    const line = newLines[i]!.trimEnd();
    if (line.length > 0) out.push(`+ ${line}`);
  }
  return out.join("\n").trim();
}

function extractEditDiffText(args: Record<string, unknown> | undefined): string | undefined {
  if (!args) return undefined;

  const oldText = readRawStringArg(args, [
    "old_text",
    "oldText",
    "old_string",
    "oldString",
    "TargetContent",
    "search",
  ]);
  const newText = readRawStringArg(args, [
    "new_text",
    "newText",
    "new_string",
    "newString",
    "ReplacementContent",
    "replace",
  ]);
  if (oldText !== undefined || newText !== undefined) {
    const formatted = formatLineReplacementDiff(oldText ?? "", newText ?? "");
    if (formatted.length > 0) return formatted;
  }

  const chunks = (args.ReplacementChunks ?? args.edits) as unknown;
  if (Array.isArray(chunks) && chunks.length > 0) {
    const parts: string[] = [];
    for (const chunk of chunks) {
      if (!chunk || typeof chunk !== "object") continue;
      const rec = chunk as Record<string, unknown>;
      const cOld = readRawStringArg(rec, [
        "old_text",
        "oldText",
        "old_string",
        "oldString",
        "TargetContent",
      ]);
      const cNew = readRawStringArg(rec, [
        "new_text",
        "newText",
        "new_string",
        "newString",
        "ReplacementContent",
      ]);
      const diff = formatLineReplacementDiff(cOld ?? "", cNew ?? "");
      if (diff.length > 0) parts.push(diff);
    }
    if (parts.length > 0) return parts.join("\n");
  }

  const patch = readRawStringArg(args, ["patch", "diff"]);
  if (patch) {
    const cleanPatch = sanitizeNeutralText(patch);
    const changed = cleanPatch
      .split("\n")
      .filter((line) => {
        if (/^(?:---|\+\+\+|\*\*\*|@@|diff --git|index )/.test(line)) return false;
        return line.startsWith("+") || line.startsWith("-");
      });
    if (changed.length > 0) {
      return changed.join("\n");
    }
    return cleanPatch;
  }

  return undefined;
}

/**
 * Classifies a tool trajectory event strictly by known tool family without
 * prose regex heuristics or auto-classifying unknown tools that happen to carry
 * a generic `path` argument.
 */
export function classifyToolEvent(event: ToolTrajectoryEvent): ClassifiedToolEvent {
  const toolName = (event.tool ?? "").trim().toLowerCase();
  const args = event.args;
  const machineId =
    event.machineId?.trim() ||
    readStringArg(args, ["machineId", "machine_id"]) ||
    undefined;

  const isReadTool = READ_FILE_TOOLS.has(toolName);
  const isPartialEditTool = PARTIAL_EDIT_TOOLS.has(toolName);
  const isFullWriteTool = FULL_WRITE_TOOLS.has(toolName);
  const isDeleteTool = DELETE_FILE_TOOLS.has(toolName);

  if (isReadTool || isPartialEditTool || isFullWriteTool || isDeleteTool) {
    let rawPath = readStringArg(args, [
      "path",
      "file_path",
      "filePath",
      "target_file",
      "TargetFile",
      "AbsolutePath",
      "filename",
      "file",
    ]);
    let inferredOpKind: FileOpKind | undefined;

    if (!rawPath && toolName === "apply_patch") {
      const patchInfo = extractPatchFileInfo(args);
      if (patchInfo) {
        rawPath = patchInfo.path;
        inferredOpKind = patchInfo.opKind;
      }
    }

    if (!rawPath) {
      return {
        event,
        family: "other",
        resourceKey: null,
        displayLabel: event.tool,
      };
    }

    const normPath = normalizeArtifactPath(rawPath);
    const fileScopeKey = machineId ? `${machineId}:${normPath}` : normPath;

    let fileOpKind: FileOpKind;
    let rangeKey: string | undefined;
    let rangeLabel: string | undefined;
    let diffText: string | undefined;
    let editPairs: EditReplacementPair[] | undefined;

    if (isReadTool) {
      const rangeInfo = extractRangeInfo(args);
      if (rangeInfo) {
        fileOpKind = "ranged_read";
        rangeKey = rangeInfo.rangeKey;
        rangeLabel = rangeInfo.rangeLabel;
      } else {
        fileOpKind = "full_read";
      }
    } else if (isDeleteTool) {
      fileOpKind = "delete";
    } else if (isFullWriteTool) {
      fileOpKind = "full_write";
    } else {
      fileOpKind = inferredOpKind ?? "partial_edit";
      if (fileOpKind === "partial_edit") {
        const extractedPairs = extractEditPairs(args);
        if (extractedPairs.length > 0) {
          editPairs = extractedPairs;
          diffText = formatEditPairsAsDiff(extractedPairs);
        } else {
          diffText = extractEditDiffText(args);
        }
      }
    }

    let resourceKey: string;
    let displayLabel: string;
    if (!event.ok) {
      resourceKey = `file_err:${fileScopeKey}`;
      displayLabel = `${normPath} (${event.tool} error)`;
    } else if (fileOpKind === "partial_edit") {
      resourceKey = `file:${fileScopeKey}#edit`;
      displayLabel = `${normPath} (edit)`;
    } else if (fileOpKind === "ranged_read" && rangeKey) {
      resourceKey = `file:${fileScopeKey}#range=${rangeKey}`;
      displayLabel = `${normPath} (${rangeLabel})`;
    } else {
      resourceKey = `file:${fileScopeKey}`;
      displayLabel = normPath;
    }

    return {
      event,
      family: "file",
      fileOpKind,
      path: normPath,
      fileScopeKey,
      rangeKey,
      rangeLabel,
      diffText,
      editPairs,
      resourceKey,
      displayLabel,
    };
  }

  // Known search tools
  if (SEARCH_TOOLS.has(toolName)) {
    const query = readStringArg(args, ["pattern", "query", "Query", "Pattern", "regex", "glob"]) ?? "";
    const rawSearchPath = readStringArg(args, [
      "path",
      "dir",
      "directory",
      "SearchPath",
      "SearchDirectory",
    ]);
    if (!query && !rawSearchPath) {
      return {
        event,
        family: "other",
        resourceKey: null,
        displayLabel: event.tool,
      };
    }
    const normSearchPath = rawSearchPath ? normalizeArtifactPath(rawSearchPath) : ".";
    const normQuery = query.replace(/\s+/g, " ").trim();
    const scopePrefix = machineId ? `${machineId}:` : "";
    const searchKey = `${scopePrefix}${toolName}:${normSearchPath}:${normQuery}`;
    return {
      event,
      family: "search",
      searchKey,
      resourceKey: `search:${searchKey}`,
      displayLabel: `${toolName} ${normQuery || normSearchPath}`,
    };
  }

  // Command / test tools (explicit set OR carrying a non-empty command/cmd string arg)
  const rawCommand = readStringArg(args, ["command", "cmd", "CommandLine"]);
  if (COMMAND_TOOLS.has(toolName) || rawCommand !== undefined) {
    if (!rawCommand) {
      return {
        event,
        family: "other",
        resourceKey: null,
        displayLabel: event.tool,
      };
    }
    const normCmd = normalizeCommandKey(rawCommand);
    const scopedCmd = machineId ? `${machineId}:${normCmd}` : normCmd;
    return {
      event,
      family: "command",
      command: normCmd,
      resourceKey: `cmd:${scopedCmd}`,
      displayLabel: normCmd,
    };
  }

  return {
    event,
    family: "other",
    resourceKey: null,
    displayLabel: event.tool,
  };
}

/**
 * Pure, non-mutating trajectory pruner enforcing exact artifact versioning invariants:
 * - Successful `full_read` / `full_write` / `delete` supersedes & prunes all earlier ops on same `fileScopeKey`.
 * - Successful `ranged_read` ONLY supersedes an earlier `ranged_read` with the identical `rangeKey`.
 * - `partial_edit` NEVER prunes a preceding `read_file` or `ranged_read`.
 * - Failed file ops (`ok === false`) NEVER prune any earlier valid file op.
 * - Command & search runs on the same normalized `resourceKey` supersede & prune older runs.
 */
export function pruneTrajectoryEvents(
  events: readonly ToolTrajectoryEvent[],
): PruneTrajectoryResult {
  const classified = events.map((ev) => classifyToolEvent(ev));

  interface PruneMark {
    supersededByCallId: string;
    stub: string;
  }

  const pruneMarks = new Map<number, PruneMark>();
  const activeFileOps = new Map<
    string,
    Array<{ idx: number; kind: FileOpKind; rangeKey?: string }>
  >();
  const activeByKey = new Map<string, number>();

  for (let i = 0; i < classified.length; i++) {
    const c = classified[i]!;
    const ev = c.event;

    if (c.family === "file" && c.fileScopeKey && c.fileOpKind) {
      // Failed file ops NEVER prune earlier valid file ops
      if (!ev.ok) {
        continue;
      }

      const scopeKey = c.fileScopeKey;
      const priorList = activeFileOps.get(scopeKey) ?? [];

      if (
        c.fileOpKind === "full_read" ||
        c.fileOpKind === "full_write" ||
        c.fileOpKind === "delete"
      ) {
        const stub = `[superseded: newer ${c.fileOpKind} on file:${scopeKey}]`;
        for (const prior of priorList) {
          pruneMarks.set(prior.idx, { supersededByCallId: ev.callId, stub });
        }
        activeFileOps.set(scopeKey, [{ idx: i, kind: c.fileOpKind }]);
      } else if (c.fileOpKind === "ranged_read") {
        const remaining: Array<{ idx: number; kind: FileOpKind; rangeKey?: string }> = [];
        const stub = `[superseded: newer ranged_read on file:${scopeKey}#range=${c.rangeKey}]`;
        for (const prior of priorList) {
          if (prior.kind === "ranged_read" && prior.rangeKey === c.rangeKey) {
            pruneMarks.set(prior.idx, { supersededByCallId: ev.callId, stub });
          } else {
            remaining.push(prior);
          }
        }
        remaining.push({ idx: i, kind: "ranged_read", rangeKey: c.rangeKey });
        activeFileOps.set(scopeKey, remaining);
      } else if (c.fileOpKind === "partial_edit") {
        // partial_edit NEVER prunes a preceding read_file or ranged_read
        priorList.push({ idx: i, kind: "partial_edit" });
        activeFileOps.set(scopeKey, priorList);
      }
      continue;
    }

    if ((c.family === "command" || c.family === "search") && c.resourceKey) {
      const prevIdx = activeByKey.get(c.resourceKey);
      if (prevIdx !== undefined) {
        const stub = `[superseded: newer ${ev.tool} on ${c.resourceKey}]`;
        pruneMarks.set(prevIdx, { supersededByCallId: ev.callId, stub });
      }
      activeByKey.set(c.resourceKey, i);
    }
  }

  const outEvents: PrunedTrajectoryEvent[] = [];
  const prunedCallIds: string[] = [];
  let charsSaved = 0;
  let tokensSaved = 0;

  for (let i = 0; i < events.length; i++) {
    const ev = events[i]!;
    const mark = pruneMarks.get(i);
    if (mark) {
      const c = classified[i]!;
      const origOutput = ev.output ?? "";
      let eventOrigChars = origOutput.length;
      prunedCallIds.push(ev.callId);
      charsSaved += Math.max(0, origOutput.length - mark.stub.length);
      tokensSaved += Math.max(0, countTokens(origOutput) - countTokens(mark.stub));

      // On superseded full_write events, also prune large inline payload fields inside a cloned `args`
      let prunedArgs: Record<string, unknown> | undefined;
      if (
        c.family === "file" &&
        c.fileOpKind === "full_write" &&
        ev.args &&
        isPlainObject(ev.args)
      ) {
        for (const key of WRITE_PAYLOAD_ARG_KEYS) {
          const val = ev.args[key];
          if (typeof val === "string" && val.length > mark.stub.length) {
            if (!prunedArgs) {
              prunedArgs = { ...ev.args };
            }
            prunedArgs[key] = mark.stub;
            eventOrigChars += val.length;
            charsSaved += val.length - mark.stub.length;
            tokensSaved += Math.max(0, countTokens(val) - countTokens(mark.stub));
          }
        }
      }

      outEvents.push({
        ...ev,
        ...(prunedArgs ? { args: prunedArgs } : {}),
        output: mark.stub,
        pruned: true,
        supersededByCallId: mark.supersededByCallId,
        originalChars: eventOrigChars,
      });
    } else {
      const c = classified[i]!;
      // Lossless vendor/runtime stack-frame compaction on non-superseded bulky command outputs
      if (c.family === "command" && typeof ev.output === "string" && ev.output.length > 0) {
        const stripped = stripVendorStackFrames(ev.output);
        if (stripped.strippedLines >= 4 && stripped.text.length < ev.output.length) {
          charsSaved += ev.output.length - stripped.text.length;
          tokensSaved += Math.max(0, countTokens(ev.output) - countTokens(stripped.text));
          outEvents.push({
            ...ev,
            output: stripped.text,
            pruned: false,
          });
          continue;
        }
      }
      outEvents.push({
        ...ev,
        pruned: false,
      });
    }
  }

  return {
    events: outEvents,
    prunedCallIds,
    prunedCount: prunedCallIds.length,
    charsSaved,
    tokensSaved,
  };
}

function buildDeterministicEventNodeId(
  graphId: string,
  c: ClassifiedToolEvent,
): string {
  const ev = c.event;
  if (ev.callId && ev.callId.trim().length > 0) {
    const seed = `${graphId}|${ev.callId.trim()}|${ev.tool.trim().toLowerCase()}|${c.resourceKey ?? ""}|${ev.ok ? "ok" : "err"}`;
    return deterministicId("art", seed);
  }
  const seed = `${graphId}|${ev.tool.trim().toLowerCase()}|${c.resourceKey ?? ""}|${ev.ok ? "ok" : "err"}|${JSON.stringify(ev.args ?? {})}|${ev.output}`;
  return deterministicId("art", seed);
}

function extractWriteContent(
  args: Record<string, unknown> | undefined,
  output: string,
): string {
  const argContent = readRawStringArg(args, WRITE_PAYLOAD_ARG_KEYS);
  if (argContent && argContent.trim().length > 0) {
    return argContent;
  }
  return output;
}

const SAFE_GRAPH_ID_RE = /^[A-Za-z0-9._:@-]{1,120}$/;

/**
 * Ingests a batch of tool-trajectory events into a Context Graph inside a
 * single atomic SQLite transaction with deterministic idempotent node IDs,
 * file/range/edit versioning, command failure linking, optional atomic
 * `autoCreate` graph provisioning, and server-native cross-compaction cleanup
 * of superseded `#edit` and `#range=...` sub-nodes.
 */
export function ingestToolArtifacts(
  store: GraphStore,
  graphId: string,
  events: readonly ToolTrajectoryEvent[],
  opts?: IngestArtifactsOptions,
): IngestArtifactsResult {
  return store.runInTransaction(() => {
    if (opts?.autoCreate === true) {
      if (!SAFE_GRAPH_ID_RE.test(graphId) || graphId.includes("..")) {
        throw new ContextGraphError(
          "'graphId' must match /^[A-Za-z0-9._:@-]{1,120}$/ without '..' when autoCreate is enabled",
          "VALIDATION_ERROR",
          400,
        );
      }
      if (!store.hasGraph(graphId)) {
        try {
          store.createGraph({
            id: graphId,
            name: opts.graphName?.trim() || graphId,
          });
        } catch (err) {
          if (!(err instanceof ContextGraphError && err.code === "ALREADY_EXISTS")) {
            throw err;
          }
        }
      }
    } else {
      // Preserve 100% existing behavior: fail with 404 NotFoundError if graph is missing
      store.getGraph(graphId);
    }

    return ingestToolArtifactsInTx(store, graphId, events, opts);
  });
}

function ingestToolArtifactsInTx(
  store: GraphStore,
  graphId: string,
  events: readonly ToolTrajectoryEvent[],
  opts?: IngestArtifactsOptions,
): IngestArtifactsResult {
  const existingNodes = store.listNodes(graphId);
  const existingById = new Map<string, GraphNode>();
  const activeByResourceKey = new Map<string, GraphNode[]>();
  const activeFileSubNodesByScope = new Map<string, Set<string>>();
  const activeFileErrorsByScope = new Map<string, Set<string>>();
  const latestBaseFileNodeByScope = new Map<string, string>();
  const editedFileScopeKeys = new Set<string>();
  const recentEditedFileNodeIds: string[] = [];

  const maxExistingTurn = existingNodes.reduce((m, n) => Math.max(m, n.turn), 0);

  const registerActiveNode = (node: GraphNode) => {
    if (node.status !== "active") return;
    if (node.resourceKey) {
      const list = activeByResourceKey.get(node.resourceKey) ?? [];
      list.push(node);
      activeByResourceKey.set(node.resourceKey, list);
    }

    const meta = node.metadata;
    const scopeKey =
      typeof meta.fileScopeKey === "string"
        ? meta.fileScopeKey
        : node.resourceKey?.startsWith("file:")
          ? node.resourceKey.slice("file:".length).replace(/#(?:edit|range=.*)$/, "")
          : node.resourceKey?.startsWith("file_err:")
            ? node.resourceKey.slice("file_err:".length)
            : undefined;

    if (scopeKey) {
      if (meta.editedScope === true || meta.fileOpKind === "partial_edit" || meta.fileOpKind === "full_write" || meta.fileOpKind === "delete") {
        editedFileScopeKeys.add(scopeKey);
        const idx = recentEditedFileNodeIds.indexOf(node.id);
        if (idx >= 0) recentEditedFileNodeIds.splice(idx, 1);
        recentEditedFileNodeIds.push(node.id);
      }

      if (node.resourceKey === `file:${scopeKey}`) {
        latestBaseFileNodeByScope.set(scopeKey, node.id);
      } else if (
        node.resourceKey === `file:${scopeKey}#edit` ||
        node.resourceKey?.startsWith(`file:${scopeKey}#range=`)
      ) {
        let subSet = activeFileSubNodesByScope.get(scopeKey);
        if (!subSet) {
          subSet = new Set();
          activeFileSubNodesByScope.set(scopeKey, subSet);
        }
        subSet.add(node.id);
      } else if (node.resourceKey === `file_err:${scopeKey}`) {
        let errSet = activeFileErrorsByScope.get(scopeKey);
        if (!errSet) {
          errSet = new Set();
          activeFileErrorsByScope.set(scopeKey, errSet);
        }
        errSet.add(node.id);
      }
    }
  };

  for (const n of existingNodes) {
    existingById.set(n.id, n);
    registerActiveNode(n);
  }

  const defaultScope = opts?.scope?.trim() || "global";
  const defaultTurn =
    typeof opts?.turn === "number" && opts.turn >= 1
      ? Math.floor(opts.turn)
      : Math.max(1, maxExistingTurn || 1);
  const maxCodeChars =
    typeof opts?.maxCodeExcerptChars === "number" && opts.maxCodeExcerptChars >= 64
      ? Math.floor(opts.maxCodeExcerptChars)
      : DEFAULT_CODE_EXCERPT_CHARS;
  const maxCmdChars =
    typeof opts?.maxCommandExcerptChars === "number" && opts.maxCommandExcerptChars >= 64
      ? Math.floor(opts.maxCommandExcerptChars)
      : DEFAULT_CMD_EXCERPT_CHARS;

  const addedNodes: GraphNode[] = [];
  const addedEdges: GraphEdge[] = [];
  const supersededSet = new Set<string>();
  let skippedCount = 0;

  const markNodeSupersededInCache = (oldId: string) => {
    supersededSet.add(oldId);
    const oldNode = existingById.get(oldId);
    if (oldNode?.resourceKey) {
      const list = activeByResourceKey.get(oldNode.resourceKey);
      if (list) {
        const filtered = list.filter((x) => x.id !== oldId);
        if (filtered.length > 0) activeByResourceKey.set(oldNode.resourceKey, filtered);
        else activeByResourceKey.delete(oldNode.resourceKey);
      }
    }
    const idx = recentEditedFileNodeIds.indexOf(oldId);
    if (idx >= 0) recentEditedFileNodeIds.splice(idx, 1);
    for (const [, set] of activeFileSubNodesByScope) set.delete(oldId);
    for (const [, set] of activeFileErrorsByScope) set.delete(oldId);
  };

  const supersedeIdsByEdge = (
    newNodeId: string,
    targetIds: Iterable<string>,
    reason: string,
    fileScopeKey: string,
  ) => {
    for (const targetId of Array.from(targetIds)) {
      if (targetId === newNodeId) continue;
      const edge = store.addEdge(graphId, {
        sourceId: newNodeId,
        targetId,
        kind: "supersedes",
        metadata: { reason, fileScopeKey },
      });
      addedEdges.push(edge);
      markNodeSupersededInCache(targetId);
    }
  };

  for (const rawEvent of events) {
    const effectiveEvent: ToolTrajectoryEvent =
      opts?.machineId && !rawEvent.machineId
        ? { ...rawEvent, machineId: opts.machineId }
        : rawEvent;
    const c = classifyToolEvent(effectiveEvent);
    if (c.family === "other" || !c.resourceKey) {
      skippedCount++;
      continue;
    }

    const nodeId = buildDeterministicEventNodeId(graphId, c);
    if (existingById.has(nodeId)) {
      skippedCount++;
      continue;
    }

    const turn =
      typeof effectiveEvent.turn === "number" && effectiveEvent.turn >= 1
        ? Math.floor(effectiveEvent.turn)
        : defaultTurn;
    const scope = effectiveEvent.scope?.trim() || defaultScope;

    // Record prior active nodes sharing the exact resourceKey (store.addNode will auto-supersede them)
    const priorSameKey = (activeByResourceKey.get(c.resourceKey) ?? []).map((n) => n.id);

    if (c.family === "file" && c.fileScopeKey && c.fileOpKind && c.path) {
      const scopeKey = c.fileScopeKey;

      // 1. Failed file op -> ephemeral_leaf observation under file_err:<scopedPath>
      if (!effectiveEvent.ok) {
        const excerpt =
          compactCommandExcerpt(effectiveEvent.output, false, maxCmdChars) ||
          truncateAtWordBoundary(sanitizeNeutralText(effectiveEvent.output), maxCmdChars) ||
          `${effectiveEvent.tool} failed on ${c.path}`;
        const summary = truncateAtWordBoundary(`${effectiveEvent.tool} failed on ${c.path}: ${excerpt}`, 240);
        const stub = `${c.path}: ${effectiveEvent.tool} failed`;
        const baseId = latestBaseFileNodeByScope.get(scopeKey);
        const dependsOn = baseId && baseId !== nodeId ? [baseId] : undefined;

        const node = store.addNode(graphId, {
          id: nodeId,
          turn,
          kind: "observation",
          role: "tool",
          scope,
          tier: "ephemeral_leaf",
          salience: 0.72,
          content: excerpt,
          summary,
          stub,
          resourceKey: c.resourceKey,
          dependsOn,
          metadata: {
            artifactFamily: "file_error",
            fileOpKind: c.fileOpKind,
            fileScopeKey: scopeKey,
            path: c.path,
            displayLabel: c.displayLabel,
            tool: effectiveEvent.tool,
            ok: false,
            callId: effectiveEvent.callId,
          },
        });

        for (const pid of priorSameKey) markNodeSupersededInCache(pid);
        existingById.set(node.id, node);
        registerActiveNode(node);
        addedNodes.push(node);
        continue;
      }

      // 2. Partial edit -> file:<scopedPath>#edit, preserves untouched base read_file
      //    and accumulates chained partial edits across rounds without wiping prior edits!
      if (c.fileOpKind === "partial_edit") {
        let priorPairs: EditReplacementPair[] = [];
        if (priorSameKey.length > 0) {
          const latestPriorId = priorSameKey[priorSameKey.length - 1]!;
          const latestPriorNode = existingById.get(latestPriorId);
          if (latestPriorNode) {
            if (Array.isArray(latestPriorNode.metadata.editPairs)) {
              priorPairs = normalizeStoredEditPairs(latestPriorNode.metadata.editPairs);
            } else {
              priorPairs = extractPairsFromDiffText(latestPriorNode.content);
            }
          }
        }
        const incomingPairs =
          c.editPairs && c.editPairs.length > 0
            ? c.editPairs
            : c.diffText
              ? extractPairsFromDiffText(c.diffText)
              : [];
        const hadPairsSource = priorPairs.length > 0 || incomingPairs.length > 0;
        const mergedPairs = hadPairsSource ? mergeEditPairs(priorPairs, incomingPairs) : [];
        const mergedDiffText =
          mergedPairs.length > 0
            ? formatEditPairsAsDiff(mergedPairs)
            : hadPairsSource
              ? undefined
              : c.diffText;

        const excerpt =
          compactCodeExcerpt(effectiveEvent.output, maxCodeChars, { diffText: mergedDiffText }) ||
          `edited ${c.path}`;
        const summary = truncateAtWordBoundary(excerpt, 240);
        const stub = c.displayLabel;
        const baseId = latestBaseFileNodeByScope.get(scopeKey);
        const dependsOn = baseId && baseId !== nodeId ? [baseId] : undefined;

        const node = store.addNode(graphId, {
          id: nodeId,
          turn,
          kind: "artifact",
          role: "tool",
          scope,
          tier: "working_subgraph",
          salience: 0.92,
          content: excerpt,
          summary,
          stub,
          resourceKey: c.resourceKey,
          dependsOn,
          metadata: {
            artifactFamily: "file",
            fileOpKind: "partial_edit",
            fileScopeKey: scopeKey,
            editedScope: true,
            ...(hadPairsSource ? { editPairs: mergedPairs } : {}),
            path: c.path,
            displayLabel: c.displayLabel,
            tool: effectiveEvent.tool,
            ok: true,
            callId: effectiveEvent.callId,
          },
        });

        for (const pid of priorSameKey) markNodeSupersededInCache(pid);
        existingById.set(node.id, node);
        registerActiveNode(node);
        addedNodes.push(node);

        // Successful edit resolves any active file_err on this scope
        const priorErrs = activeFileErrorsByScope.get(scopeKey);
        if (priorErrs && priorErrs.size > 0) {
          supersedeIdsByEdge(node.id, priorErrs, "successful_edit_resolves_file_error", scopeKey);
        }
        continue;
      }

      // 3. Disjoint ranged read -> file:<scopedPath>#range=<rangeKey>
      if (c.fileOpKind === "ranged_read") {
        const isEdited = editedFileScopeKeys.has(scopeKey);
        const excerpt =
          compactCodeExcerpt(effectiveEvent.output, maxCodeChars) || c.displayLabel;
        const summary = truncateAtWordBoundary(excerpt, 240);
        const stub = c.displayLabel;

        const node = store.addNode(graphId, {
          id: nodeId,
          turn,
          kind: "artifact",
          role: "tool",
          scope,
          tier: "working_subgraph",
          salience: isEdited ? 0.90 : 0.84,
          content: excerpt,
          summary,
          stub,
          resourceKey: c.resourceKey,
          metadata: {
            artifactFamily: "file",
            fileOpKind: "ranged_read",
            fileScopeKey: scopeKey,
            rangeKey: c.rangeKey,
            rangeLabel: c.rangeLabel,
            editedScope: isEdited,
            path: c.path,
            displayLabel: c.displayLabel,
            tool: effectiveEvent.tool,
            ok: true,
            callId: effectiveEvent.callId,
          },
        });

        for (const pid of priorSameKey) markNodeSupersededInCache(pid);
        existingById.set(node.id, node);
        registerActiveNode(node);
        addedNodes.push(node);
        continue;
      }

      // 4. Full snapshot (full_read / full_write / delete) -> file:<scopedPath>
      if (c.fileOpKind === "full_write" || c.fileOpKind === "delete") {
        editedFileScopeKeys.add(scopeKey);
      }
      const isEdited = editedFileScopeKeys.has(scopeKey);
      const salience =
        c.fileOpKind === "full_write"
          ? 0.92
          : c.fileOpKind === "delete"
            ? 0.88
            : isEdited
              ? 0.90
              : 0.72;

      const rawSource =
        c.fileOpKind === "delete"
          ? `deleted ${c.path}`
          : c.fileOpKind === "full_write"
            ? extractWriteContent(effectiveEvent.args, effectiveEvent.output)
            : effectiveEvent.output;
      const excerpt =
        c.fileOpKind === "delete"
          ? `deleted ${c.path}`
          : compactCodeExcerpt(rawSource, maxCodeChars) || c.path;
      const summary = truncateAtWordBoundary(excerpt, 240);
      const stub = c.path;

      const node = store.addNode(graphId, {
        id: nodeId,
        turn,
        kind: "artifact",
        role: "tool",
        scope,
        tier: "working_subgraph",
        salience,
        content: excerpt,
        summary,
        stub,
        resourceKey: c.resourceKey,
        metadata: {
          artifactFamily: "file",
          fileOpKind: c.fileOpKind,
          fileScopeKey: scopeKey,
          editedScope: isEdited,
          path: c.path,
          displayLabel: c.displayLabel,
          tool: effectiveEvent.tool,
          ok: true,
          callId: effectiveEvent.callId,
        },
      });

      for (const pid of priorSameKey) markNodeSupersededInCache(pid);
      existingById.set(node.id, node);

      // Server-native cross-compaction cleanup: supersede any prior active #edit & #range=... sub-nodes
      const subNodes = activeFileSubNodesByScope.get(scopeKey);
      if (subNodes && subNodes.size > 0) {
        supersedeIdsByEdge(node.id, subNodes, "full_snapshot_supersedes_subnode", scopeKey);
      }
      if (c.fileOpKind === "full_write" || c.fileOpKind === "delete") {
        const priorErrs = activeFileErrorsByScope.get(scopeKey);
        if (priorErrs && priorErrs.size > 0) {
          supersedeIdsByEdge(node.id, priorErrs, "full_write_resolves_file_error", scopeKey);
        }
      }

      registerActiveNode(node);
      addedNodes.push(node);
      continue;
    }

    // Command / test execution events
    if (c.family === "command" && c.command) {
      const ok = effectiveEvent.ok;
      const excerpt =
        compactCommandExcerpt(effectiveEvent.output, ok, ok ? Math.min(maxCmdChars, 120) : maxCmdChars) ||
        (ok ? `${c.command}: ok` : `${c.command}: failed`);
      const summary = truncateAtWordBoundary(excerpt, 240);
      const stub = `${c.command}: ${ok ? "ok" : "failed"}`;

      // Link active failing command to both cited read/active files in its stack trace
      // and recent edited files via `depends_on` edges.
      const citedFilePaths = !ok ? extractAppFileReferences(effectiveEvent.output) : [];
      const citedNodeIds: string[] = [];
      if (!ok && citedFilePaths.length > 0) {
        const evMachine = effectiveEvent.machineId?.trim();
        for (const refPath of citedFilePaths) {
          const directKey = evMachine ? `file:${evMachine}:${refPath}` : `file:${refPath}`;
          const directMatches = activeByResourceKey.get(directKey);
          if (directMatches && directMatches.length > 0) {
            for (const m of directMatches) {
              if (!citedNodeIds.includes(m.id)) citedNodeIds.push(m.id);
            }
            continue;
          }
          // Suffix/scoped fallback matching active `file:` nodes (incl. ranged reads or absolute paths)
          for (const [rk, list] of activeByResourceKey) {
            if (!rk.startsWith("file:")) continue;
            for (const m of list) {
              const nodePath =
                typeof m.metadata.path === "string"
                  ? m.metadata.path
                  : rk.slice("file:".length).replace(/#(?:edit|range=.*)$/, "");
              if (evMachine) {
                const sk = typeof m.metadata.fileScopeKey === "string" ? m.metadata.fileScopeKey : "";
                if (!sk.startsWith(`${evMachine}:`)) continue;
              }
              if (nodePath === refPath || nodePath.endsWith(`/${refPath}`)) {
                if (!citedNodeIds.includes(m.id)) citedNodeIds.push(m.id);
              }
            }
          }
          if (citedNodeIds.length >= 6) break;
        }
      }

      const dependsOn = !ok
        ? Array.from(new Set([...citedNodeIds.slice(0, 6), ...recentEditedFileNodeIds.slice(-4)])).filter(
            (id) => id !== nodeId,
          )
        : undefined;

      const node = store.addNode(graphId, {
        id: nodeId,
        turn,
        kind: "observation",
        role: "tool",
        scope,
        tier: "working_subgraph",
        salience: ok ? 0.65 : 0.98,
        content: excerpt,
        summary,
        stub,
        resourceKey: c.resourceKey,
        dependsOn: dependsOn && dependsOn.length > 0 ? dependsOn : undefined,
        metadata: {
          artifactFamily: "command",
          unresolvedFailure: !ok,
          command: c.command,
          status: ok ? "ok" : "failed",
          ...(citedFilePaths.length > 0 ? { citedFilePaths } : {}),
          tool: effectiveEvent.tool,
          ok,
          callId: effectiveEvent.callId,
        },
      });

      for (const pid of priorSameKey) markNodeSupersededInCache(pid);
      existingById.set(node.id, node);
      registerActiveNode(node);
      addedNodes.push(node);
      continue;
    }

    // Search events
    if (c.family === "search") {
      const clean = sanitizeNeutralText(effectiveEvent.output);
      const excerpt = truncateAtWordBoundary(clean, 240) || c.displayLabel;
      const summary = truncateAtWordBoundary(excerpt, 160);
      const stub = c.displayLabel;

      const node = store.addNode(graphId, {
        id: nodeId,
        turn,
        kind: "observation",
        role: "tool",
        scope,
        tier: "ephemeral_leaf",
        salience: 0.42,
        content: excerpt,
        summary,
        stub,
        resourceKey: c.resourceKey,
        metadata: {
          artifactFamily: "search",
          searchKey: c.searchKey,
          tool: effectiveEvent.tool,
          ok: effectiveEvent.ok,
          callId: effectiveEvent.callId,
        },
      });

      for (const pid of priorSameKey) markNodeSupersededInCache(pid);
      existingById.set(node.id, node);
      registerActiveNode(node);
      addedNodes.push(node);
    }
  }

  // Compute live summary from active state
  const finalActiveNodes = store.listNodes(graphId, { status: "active" });
  const activeScopeSet = new Set<string>();
  const unresolvedFailureNodeIds: string[] = [];
  for (const n of finalActiveNodes) {
    if (n.metadata.unresolvedFailure === true) {
      unresolvedFailureNodeIds.push(n.id);
    }
    if (typeof n.metadata.fileScopeKey === "string") {
      activeScopeSet.add(n.metadata.fileScopeKey);
    }
  }

  return {
    graphId,
    ingestedCount: addedNodes.length,
    skippedCount,
    addedNodes,
    addedEdges,
    supersededNodeIds: Array.from(supersededSet),
    activeFileScopes: Array.from(activeScopeSet),
    unresolvedFailureNodeIds,
  };
}
