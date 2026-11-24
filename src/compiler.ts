import {
  compactCodeExcerpt,
  compactCommandExcerpt,
  sanitizeNeutralText,
  stripRegexLinear,
  truncateAtWordBoundary,
} from "./excerpt.ts";
import type { GraphStore } from "./graph-store.ts";
import {
  ContextGraphError,
  type AgentLoopCompatibleMessage,
  type ChatMessage,
  type CompiledNodeEntry,
  type CompileFormat,
  type CompileOptions,
  type CompileResult,
  type ConflictWarning,
  type ExcludedNodeEntry,
  type GraphEdge,
  type GraphNode,
  type LodLevel,
  type NodeKind,
} from "./types.ts";
import {
  cosineSimilarity,
  countTokens,
  embedText,
  longestCommonPrefixLength,
  tokenizeToIds,
} from "./vector.ts";

export const CACHED_TOKEN_COST_WEIGHT = 0.1 as const;

const KIND_UTILITY_MULTIPLIER: Record<NodeKind, number> = {
  system: 1.5,
  goal: 1.45,
  constraint: 1.45,
  task_frame: 1.25,
  decision: 1.28,
  artifact: 1.35,
  summary: 1.12,
  fact: 1.0,
  observation: 0.68,
};

const EDGE_FORWARD_WEIGHT: Record<string, number> = {
  depends_on: 0.9,
  elaborates: 0.65,
  summarizes: 0.45,
  scoped_to: 0.45,
  conflicts_with: 0.65,
};

const EDGE_REVERSE_WEIGHT: Record<string, number> = {
  depends_on: 0.35,
  elaborates: 0.5,
  summarizes: 0.55,
  scoped_to: 0.35,
  conflicts_with: 0.65,
};

function selectLodBody(node: GraphNode, lod: LodLevel): string {
  if (lod === "full") return node.content;
  if (lod === "summary") return node.summary;
  return node.stub;
}

/**
 * Deterministic, timestamp-free node block formatter so identical (node, lod)
 * sequences produce 100% identical token IDs across compilation turns.
 */
export function renderCanonicalNodeBlock(node: GraphNode, lod: LodLevel): string {
  const resAttr = node.resourceKey ? ` resource="${node.resourceKey}" v="${node.version}"` : "";
  const body = selectLodBody(node, lod);
  return `<node id="${node.id}" kind="${node.kind}" scope="${node.scope}" lod="${lod}"${resAttr}>\n${body}\n</node>`;
}

function buildForwardResolutionMap(nodes: readonly GraphNode[]): Map<string, GraphNode | null> {
  const byId = new Map<string, GraphNode>();
  for (const n of nodes) byId.set(n.id, n);

  const resolved = new Map<string, GraphNode | null>();
  for (const n of nodes) {
    let curr: GraphNode | undefined = n;
    const seen = new Set<string>();
    while (curr && curr.status === "superseded" && curr.supersededBy && !seen.has(curr.id)) {
      seen.add(curr.id);
      curr = byId.get(curr.supersededBy);
    }
    if (curr && curr.status === "active") {
      resolved.set(n.id, curr);
    } else {
      resolved.set(n.id, null);
    }
  }
  return resolved;
}

function lexicalResourceBoost(query: string, node: GraphNode): number {
  if (!query || !node.resourceKey) return 0;
  const qLower = query.toLowerCase();
  const rawKey = node.resourceKey.replace(/^file:/, "").toLowerCase();
  const baseName = rawKey.split("/").pop() ?? rawKey;
  if (qLower.includes(rawKey)) return 0.5;
  if (baseName.length >= 3 && qLower.includes(baseName)) return 0.42;
  return 0;
}

export const COMPILE_FORMATS: readonly CompileFormat[] = [
  "both",
  "messages",
  "prompt",
  "xml",
  "compact_markdown",
] as const;

const DEFAULT_COMPACT_MAX_CHARS = 950 as const;
const MAX_COMPACT_OUTPUT_CHARS = 4000 as const;
const STALE_META_TAG_RE = /\(v\d+,[ \t]*[^,)\n]+,[ \t]*(?:ok|err|error|failed)\)/gi;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Validates HTTP / MCP / SDK compile options strictly, rejecting malformed
 * types, invalid formats, or out-of-range numeric bounds with 400 VALIDATION_ERROR.
 */
export function validateCompileOptions(raw: unknown): Omit<CompileOptions, "graphId"> {
  if (raw === undefined) return {};
  if (!isPlainObject(raw)) {
    throw new ContextGraphError("compileOptions must be a plain object", "VALIDATION_ERROR", 400);
  }

  const out: Omit<CompileOptions, "graphId"> = {};

  if (raw.format !== undefined) {
    if (typeof raw.format !== "string" || !COMPILE_FORMATS.includes(raw.format as CompileFormat)) {
      throw new ContextGraphError(
        `Invalid 'format' value; expected one of: ${COMPILE_FORMATS.join(", ")}`,
        "VALIDATION_ERROR",
        400,
      );
    }
    out.format = raw.format as CompileFormat;
  }

  if (raw.maxOutputChars !== undefined) {
    if (
      typeof raw.maxOutputChars !== "number" ||
      !Number.isInteger(raw.maxOutputChars) ||
      raw.maxOutputChars < 1 ||
      raw.maxOutputChars > MAX_COMPACT_OUTPUT_CHARS
    ) {
      throw new ContextGraphError(
        `'maxOutputChars' must be an integer between 1 and ${MAX_COMPACT_OUTPUT_CHARS}`,
        "VALIDATION_ERROR",
        400,
      );
    }
    out.maxOutputChars = raw.maxOutputChars;
  }

  if (raw.maxTokens !== undefined) {
    if (
      typeof raw.maxTokens !== "number" ||
      !Number.isInteger(raw.maxTokens) ||
      raw.maxTokens < 64
    ) {
      throw new ContextGraphError(
        "'maxTokens' must be an integer >= 64",
        "VALIDATION_ERROR",
        400,
      );
    }
    out.maxTokens = raw.maxTokens;
  }

  if (raw.zone1MaxRatio !== undefined) {
    if (
      typeof raw.zone1MaxRatio !== "number" ||
      !Number.isFinite(raw.zone1MaxRatio) ||
      raw.zone1MaxRatio <= 0 ||
      raw.zone1MaxRatio >= 0.9
    ) {
      throw new ContextGraphError(
        "'zone1MaxRatio' must be a finite number strictly between 0 and 0.9",
        "VALIDATION_ERROR",
        400,
      );
    }
    out.zone1MaxRatio = raw.zone1MaxRatio;
  }

  if (raw.focusQuery !== undefined) {
    if (typeof raw.focusQuery !== "string") {
      throw new ContextGraphError("'focusQuery' must be a string", "VALIDATION_ERROR", 400);
    }
    out.focusQuery = raw.focusQuery;
  }

  if (raw.activeScope !== undefined) {
    if (typeof raw.activeScope !== "string") {
      throw new ContextGraphError("'activeScope' must be a string", "VALIDATION_ERROR", 400);
    }
    out.activeScope = raw.activeScope;
  }

  if (raw.focusNodeIds !== undefined) {
    if (
      !Array.isArray(raw.focusNodeIds) ||
      !raw.focusNodeIds.every((id) => typeof id === "string")
    ) {
      throw new ContextGraphError(
        "'focusNodeIds' must be an array of strings",
        "VALIDATION_ERROR",
        400,
      );
    }
    out.focusNodeIds = raw.focusNodeIds;
  }

  if (raw.pprIterations !== undefined) {
    if (
      typeof raw.pprIterations !== "number" ||
      !Number.isInteger(raw.pprIterations) ||
      raw.pprIterations < 1 ||
      raw.pprIterations > 100
    ) {
      throw new ContextGraphError(
        "'pprIterations' must be an integer between 1 and 100",
        "VALIDATION_ERROR",
        400,
      );
    }
    out.pprIterations = raw.pprIterations;
  }

  if (raw.pprAlpha !== undefined) {
    if (
      typeof raw.pprAlpha !== "number" ||
      !Number.isFinite(raw.pprAlpha) ||
      raw.pprAlpha <= 0 ||
      raw.pprAlpha >= 1
    ) {
      throw new ContextGraphError(
        "'pprAlpha' must be a finite number strictly between 0 and 1",
        "VALIDATION_ERROR",
        400,
      );
    }
    out.pprAlpha = raw.pprAlpha;
  }

  if (raw.recordCacheState !== undefined) {
    if (typeof raw.recordCacheState !== "boolean") {
      throw new ContextGraphError("'recordCacheState' must be a boolean", "VALIDATION_ERROR", 400);
    }
    out.recordCacheState = raw.recordCacheState;
  }

  if (raw.previousTokenIds !== undefined) {
    if (
      !Array.isArray(raw.previousTokenIds) ||
      !raw.previousTokenIds.every((t) => typeof t === "number" && Number.isInteger(t))
    ) {
      throw new ContextGraphError(
        "'previousTokenIds' must be an array of integers",
        "VALIDATION_ERROR",
        400,
      );
    }
    out.previousTokenIds = raw.previousTokenIds;
  }

  return out;
}

function getFileScopeKey(node: GraphNode): string | null {
  if (typeof node.metadata.fileScopeKey === "string" && node.metadata.fileScopeKey.trim()) {
    return node.metadata.fileScopeKey.trim();
  }
  if (node.resourceKey?.startsWith("file:")) {
    return node.resourceKey.slice("file:".length).replace(/#(?:edit|range=.*)$/, "");
  }
  if (node.resourceKey?.startsWith("file_err:")) {
    return node.resourceKey.slice("file_err:".length);
  }
  return null;
}

function getFileDisplayPath(node: GraphNode): string {
  if (typeof node.metadata.path === "string" && node.metadata.path.trim()) {
    return node.metadata.path.trim();
  }
  const scopeKey = getFileScopeKey(node);
  if (scopeKey) return scopeKey;
  return node.resourceKey ?? node.id;
}

function isFileArtifactNode(node: GraphNode): boolean {
  if (node.metadata.artifactFamily === "file") return true;
  return node.kind === "artifact" && Boolean(node.resourceKey?.startsWith("file:"));
}

function isFileErrorNode(node: GraphNode): boolean {
  return (
    node.metadata.artifactFamily === "file_error" ||
    Boolean(node.resourceKey?.startsWith("file_err:"))
  );
}

function isCommandNode(node: GraphNode): boolean {
  return (
    node.metadata.artifactFamily === "command" ||
    Boolean(node.resourceKey?.startsWith("cmd:"))
  );
}

function cleanCompactFragment(text: string): string {
  return stripRegexLinear(sanitizeNeutralText(text), STALE_META_TAG_RE).trim();
}

function collapseNewlinesToSpace(text: string): string {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .join(" ");
}

function flattenCodeLines(text: string): string {
  const lines = cleanCompactFragment(text)
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length === 0) return "";
  let out = lines[0]!;
  for (let i = 1; i < lines.length; i++) {
    const prev = lines[i - 1]!;
    const curr = lines[i]!;
    const prevIsComment = /^(?:\/\/|#|\/\*|--)/.test(prev);
    const isDiffLine = /^[+-]\s/.test(prev) || /^[+-]\s/.test(curr);
    if (prevIsComment || isDiffLine) {
      out += ` | ${curr}`;
    } else if (/[;{,:]$/.test(prev)) {
      out += ` ${curr}`;
    } else {
      out += `; ${curr}`;
    }
  }
  return out;
}

function flattenCommandLines(text: string): string {
  return cleanCompactFragment(text)
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .join(" | ");
}

function allocateWaterFilling(desired: readonly number[], totalBudget: number): number[] {
  const n = desired.length;
  if (n === 0 || totalBudget <= 0) return desired.map(() => 0);
  const sum = desired.reduce((a, b) => a + b, 0);
  if (sum <= totalBudget) {
    return [...desired];
  }

  const alloc = new Array<number>(n).fill(0);
  const active = new Set<number>();
  for (let i = 0; i < n; i++) active.add(i);
  let rem = totalBudget;

  while (active.size > 0 && rem > 0) {
    const share = Math.floor(rem / active.size);
    if (share <= 0) break;
    let saturatedAny = false;
    for (const idx of Array.from(active)) {
      if (desired[idx]! <= alloc[idx]! + share) {
        const give = desired[idx]! - alloc[idx]!;
        alloc[idx] = desired[idx]!;
        rem -= give;
        active.delete(idx);
        saturatedAny = true;
      }
    }
    if (!saturatedAny) {
      for (const idx of active) {
        alloc[idx] = alloc[idx]! + share;
        rem -= share;
      }
      break;
    }
  }
  return alloc;
}

interface CompactMarkdownBuildArgs {
  store: GraphStore;
  graph: ReturnType<GraphStore["getGraph"]>;
  options: CompileOptions;
  focusQuery: string;
  activeScope: string;
  maxTokens: number;
  zone1MaxRatio: number;
  nodes: readonly GraphNode[];
  activeNodes: readonly GraphNode[];
  zone1Candidates: readonly GraphNode[];
  zone2Candidates: readonly GraphNode[];
  excludedNodes: ExcludedNodeEntry[];
  activeConflictEdges: readonly GraphEdge[];
  pprScoreById: ReadonlyMap<string, number>;
  utilityOf: (n: GraphNode) => number;
  isPriorityFileNode: (n: GraphNode) => boolean;
  supersededOccluded: number;
  retractedPruned: number;
  fullLinearTokens: number;
}

function compileCompactMarkdown(args: CompactMarkdownBuildArgs): CompileResult {
  const {
    store,
    graph,
    options,
    focusQuery,
    activeScope,
    maxTokens,
    zone1MaxRatio,
    nodes,
    activeNodes,
    zone1Candidates,
    zone2Candidates,
    excludedNodes,
    activeConflictEdges,
    pprScoreById,
    utilityOf,
    isPriorityFileNode,
    supersededOccluded,
    retractedPruned,
    fullLinearTokens,
  } = args;

  const maxOutputChars =
    typeof options.maxOutputChars === "number" &&
    Number.isFinite(options.maxOutputChars) &&
    options.maxOutputChars >= 1
      ? Math.min(MAX_COMPACT_OUTPUT_CHARS, Math.floor(options.maxOutputChars))
      : DEFAULT_COMPACT_MAX_CHARS;

  const lodCounts: Record<LodLevel, number> = { full: 0, summary: 0, stub: 0 };

  // 1. Zone 1: Deterministic pinned_prefix / stable_frame lines sorted by seq ASC
  const zone1Entries: CompiledNodeEntry[] = [];
  const zone1CharCap =
    zone2Candidates.length > 0
      ? Math.floor(maxOutputChars * zone1MaxRatio)
      : maxOutputChars;
  let zone1UsedChars = 0;

  for (const n of zone1Candidates) {
    const prefix = `- [${n.kind}]: `;
    const sep = zone1Entries.length > 0 ? 1 : 0;
    const rem = zone1CharCap - zone1UsedChars - sep;
    if (rem <= prefix.length + 4) break;
    const body = collapseNewlinesToSpace(cleanCompactFragment(n.content));
    const fitted = truncateAtWordBoundary(body, rem - prefix.length);
    if (!fitted) break;
    const line = `${prefix}${fitted}`;
    const lod: LodLevel = fitted.length >= body.length ? "full" : "summary";
    lodCounts[lod]++;
    zone1UsedChars += sep + line.length;
    zone1Entries.push({
      id: n.id,
      seq: n.seq,
      turn: n.turn,
      kind: n.kind,
      role: n.role,
      tier: n.tier,
      scope: n.scope,
      zone: 1,
      lod,
      version: n.version,
      resourceKey: n.resourceKey,
      tokens: countTokens(line),
      activationScore: pprScoreById.get(n.id) ?? 0,
      renderedText: line,
      pulledByClosure: false,
    });
  }

  // 2. Partition Zone 2 active nodes into strict 3-tier engineering priority
  const tier1Nodes: GraphNode[] = [];
  const tier2RawNodes: GraphNode[] = [];
  let unmodifiedReadFiles: GraphNode[] = [];
  const passingChecks: GraphNode[] = [];
  const otherTier3Nodes: GraphNode[] = [];

  for (const n of zone2Candidates) {
    if (n.metadata.unresolvedFailure === true) {
      tier1Nodes.push(n);
    } else if (isFileErrorNode(n) || isPriorityFileNode(n) || n.kind === "decision") {
      tier2RawNodes.push(n);
    } else if (isFileArtifactNode(n)) {
      unmodifiedReadFiles.push(n);
    } else if (isCommandNode(n)) {
      passingChecks.push(n);
    } else {
      otherTier3Nodes.push(n);
    }
  }

  tier1Nodes.sort((a, b) => {
    const diff = utilityOf(b) - utilityOf(a);
    return Math.abs(diff) > 1e-12 ? diff : a.seq - b.seq;
  });

  unmodifiedReadFiles.sort((a, b) => {
    const diff = utilityOf(b) - utilityOf(a);
    return Math.abs(diff) > 1e-12 ? diff : a.seq - b.seq;
  });

  passingChecks.sort((a, b) => b.seq - a.seq);
  otherTier3Nodes.sort((a, b) => {
    const diff = utilityOf(b) - utilityOf(a);
    return Math.abs(diff) > 1e-12 ? diff : a.seq - b.seq;
  });

  // Read-only trajectory fallback: if there are NO Tier 1 / Tier 2 items, promote
  // active read-only files to Tier 2 so they get full declaration-aware excerpts.
  const promotedReadOnly = tier1Nodes.length === 0 && tier2RawNodes.length === 0 && unmodifiedReadFiles.length > 0;
  if (promotedReadOnly) {
    tier2RawNodes.push(...unmodifiedReadFiles);
    unmodifiedReadFiles = [];
  }

  // Group Tier 2 file & file_error nodes by fileScopeKey so base snapshot ALWAYS precedes #edit delta / file_error
  const fileNodeSubOrder = (n: GraphNode): number => {
    if (isFileErrorNode(n)) return 3;
    const op = n.metadata.fileOpKind;
    if (op === "partial_edit" || n.resourceKey?.endsWith("#edit")) return 2;
    if (op === "ranged_read" || n.resourceKey?.includes("#range=")) return 1;
    return 0;
  };

  interface Tier2Group {
    maxUtility: number;
    minSeq: number;
    nodes: GraphNode[];
  }
  const groupMap = new Map<string, Tier2Group>();
  const tier2Groups: Tier2Group[] = [];

  for (const n of tier2RawNodes) {
    const scopeKey = isFileArtifactNode(n) || isFileErrorNode(n) ? getFileScopeKey(n) : null;
    if (scopeKey) {
      let grp = groupMap.get(scopeKey);
      if (!grp) {
        grp = { maxUtility: utilityOf(n), minSeq: n.seq, nodes: [] };
        groupMap.set(scopeKey, grp);
        tier2Groups.push(grp);
      } else {
        grp.maxUtility = Math.max(grp.maxUtility, utilityOf(n));
        grp.minSeq = Math.min(grp.minSeq, n.seq);
      }
      grp.nodes.push(n);
    } else {
      tier2Groups.push({ maxUtility: utilityOf(n), minSeq: n.seq, nodes: [n] });
    }
  }

  for (const grp of tier2Groups) {
    grp.nodes.sort((a, b) => {
      const subDiff = fileNodeSubOrder(a) - fileNodeSubOrder(b);
      if (subDiff !== 0) return subDiff;
      return a.seq - b.seq;
    });
  }
  tier2Groups.sort((a, b) => {
    const diff = b.maxUtility - a.maxUtility;
    return Math.abs(diff) > 1e-12 ? diff : a.minSeq - b.minSeq;
  });

  const tier2Nodes: GraphNode[] = [];
  for (const grp of tier2Groups) {
    tier2Nodes.push(...grp.nodes);
  }

  interface RenderedZone2Line {
    tierRank: 1 | 2 | 3;
    text: string;
    nodes: GraphNode[];
    lod: LodLevel;
  }

  const zone2Lines: RenderedZone2Line[] = [];
  const includedNodeIds = new Set<string>(zone1Entries.map((e) => e.id));

  let remainingChars =
    maxOutputChars - zone1UsedChars - (zone1Entries.length > 0 ? 1 : 0);

  const pushZone2Line = (item: RenderedZone2Line): boolean => {
    const sep = zone2Lines.length > 0 ? 1 : 0;
    if (item.text.length + sep > remainingChars) return false;
    zone2Lines.push(item);
    remainingChars -= item.text.length + sep;
    for (const n of item.nodes) includedNodeIds.add(n.id);
    return true;
  };

  // --- Tier 1: Unresolved Failing Checks ---
  if (tier1Nodes.length > 0 && remainingChars >= 1) {
    const tier1CapTotal =
      tier2Nodes.length > 0
        ? Math.min(remainingChars, Math.max(360, Math.floor(remainingChars * 0.45)))
        : remainingChars;
    const tier1Entries = tier1Nodes.map((n) => {
      const cmd =
        typeof n.metadata.command === "string" && n.metadata.command.trim()
          ? n.metadata.command.trim()
          : (n.resourceKey?.replace(/^cmd:/, "") ?? n.id);
      const prefix = `- [check failed] \`${cmd}\`: `;
      const desiredBody = truncateAtWordBoundary(
        flattenCommandLines(compactCommandExcerpt(n.content, false, 314)),
        320,
      );
      return {
        node: n,
        prefix,
        desiredBody,
        desiredLen: prefix.length + desiredBody.length,
      };
    });
    const sepOverhead = Math.max(0, tier1Entries.length - 1);
    const tier1Allocs = allocateWaterFilling(
      tier1Entries.map((e) => e.desiredLen),
      Math.max(0, tier1CapTotal - sepOverhead),
    );

    for (let i = 0; i < tier1Entries.length; i++) {
      const entry = tier1Entries[i]!;
      const sep = zone2Lines.length > 0 ? 1 : 0;
      const budget = Math.min(tier1Allocs[i]!, remainingChars - sep);
      if (budget <= 0) continue;
      if (budget <= entry.prefix.length + 4) {
        // Tiny budget safety: keep a cleanly truncated lead snippet if nothing is emitted yet
        if (zone1Entries.length === 0 && zone2Lines.length === 0) {
          const tiny = truncateAtWordBoundary(
            `${entry.prefix}${entry.desiredBody}`,
            remainingChars - sep,
          );
          if (tiny) {
            pushZone2Line({ tierRank: 1, text: tiny, nodes: [entry.node], lod: "stub" });
          }
        }
        continue;
      }
      const exBudget = budget - entry.prefix.length;
      let flat: string;
      if (exBudget >= entry.desiredBody.length) {
        // Emit cached desiredBody directly to prevent double-squeezing signal loss
        flat = entry.desiredBody;
      } else {
        const cmdBudget = exBudget > 48 ? exBudget - 6 : exBudget;
        const rawEx = compactCommandExcerpt(entry.node.content, false, cmdBudget);
        flat = truncateAtWordBoundary(flattenCommandLines(rawEx), exBudget);
      }
      if (!flat) continue;
      const line = `${entry.prefix}${flat}`;
      pushZone2Line({ tierRank: 1, text: line, nodes: [entry.node], lod: "full" });
    }
  }

  // Pre-build optional compact `[other read files]` line to estimate tiny reserve
  const buildOtherReadFilesLine = (maxLen: number): { line: string; included: GraphNode[] } | null => {
    if (unmodifiedReadFiles.length === 0 || maxLen < 32) return null;
    const prefix = "- [other read files]: ";
    if (maxLen <= prefix.length + 4) return null;
    const seenPaths = new Set<string>();
    const pickedParts: string[] = [];
    const included: GraphNode[] = [];
    let len = prefix.length;

    for (const n of unmodifiedReadFiles) {
      const p = getFileDisplayPath(n);
      if (seenPaths.has(p)) {
        included.push(n);
        continue;
      }
      const token = `\`${p}\``;
      const addLen = (pickedParts.length > 0 ? 2 : 0) + token.length;
      if (len + addLen <= maxLen) {
        seenPaths.add(p);
        pickedParts.push(token);
        included.push(n);
        len += addLen;
      } else {
        break;
      }
    }
    if (pickedParts.length === 0) return null;
    return { line: `${prefix}${pickedParts.join(", ")}`, included };
  };

  // --- Tier 2: Edited scopes, #edit deltas, ranged chunks, failed-edit target files, decisions ---
  if (tier2Nodes.length > 0 && remainingChars >= 1) {
    const previewOther = buildOtherReadFilesLine(140);
    const otherReserve =
      previewOther && remainingChars - (previewOther.line.length + 1) >= tier2Nodes.length * 160
        ? previewOther.line.length + 1
        : 0;
    const tier2BudgetPool = Math.max(0, remainingChars - otherReserve);

    const buildTier2Prefix = (n: GraphNode): string => {
      if (n.kind === "decision") return "- [decision]: ";
      const pathLabel = getFileDisplayPath(n);
      if (isFileErrorNode(n)) return `- [edit failed] \`${pathLabel}\`: `;
      const op = n.metadata.fileOpKind;
      if (op === "partial_edit" || n.resourceKey?.endsWith("#edit")) {
        return `- [file edit] \`${pathLabel}\`: `;
      }
      if (op === "ranged_read" || n.resourceKey?.includes("#range=")) {
        const rangeLabel =
          typeof n.metadata.rangeLabel === "string" && n.metadata.rangeLabel
            ? n.metadata.rangeLabel
            : (n.resourceKey?.split("#range=")[1] ?? "range");
        return `- [file] \`${pathLabel} (${rangeLabel})\`: `;
      }
      if (n.metadata.editedScope === true || op === "full_write" || op === "delete") {
        return `- [file edited] \`${pathLabel}\`: `;
      }
      return `- [file] \`${pathLabel}\`: `;
    };

    const renderTier2Body = (n: GraphNode, cap: number): string => {
      if (isFileErrorNode(n)) {
        const errCap = Math.min(cap, 180);
        return truncateAtWordBoundary(
          flattenCommandLines(compactCommandExcerpt(n.content, false, errCap > 48 ? errCap - 6 : errCap)),
          cap,
        );
      }
      if (n.kind === "decision") {
        return truncateAtWordBoundary(
          collapseNewlinesToSpace(cleanCompactFragment(n.content)),
          cap,
        );
      }
      const codeEx = compactCodeExcerpt(n.content, cap);
      return truncateAtWordBoundary(flattenCodeLines(codeEx), cap);
    };

    const tier2Entries = tier2Nodes.map((n) => {
      const prefix = buildTier2Prefix(n);
      const cap = isFileErrorNode(n) ? 160 : 340;
      const desiredBody = renderTier2Body(n, cap);
      return {
        node: n,
        prefix,
        desiredBody,
        desiredLen: prefix.length + desiredBody.length,
      };
    });

    const sepOverhead =
      (zone2Lines.length > 0 ? 1 : 0) + Math.max(0, tier2Entries.length - 1);
    const tier2Allocs = allocateWaterFilling(
      tier2Entries.map((e) => e.desiredLen),
      Math.max(0, tier2BudgetPool - sepOverhead),
    );

    for (let i = 0; i < tier2Entries.length; i++) {
      const entry = tier2Entries[i]!;
      const sep = zone2Lines.length > 0 ? 1 : 0;
      const budget = Math.min(tier2Allocs[i]!, remainingChars - sep);
      if (budget <= 0) continue;
      if (budget <= entry.prefix.length + 4) {
        if (zone1Entries.length === 0 && zone2Lines.length === 0) {
          const tiny = truncateAtWordBoundary(
            `${entry.prefix}${entry.desiredBody}`,
            remainingChars - sep,
          );
          if (tiny) {
            pushZone2Line({ tierRank: 2, text: tiny, nodes: [entry.node], lod: "stub" });
          }
        }
        continue;
      }
      const exBudget = budget - entry.prefix.length;
      // Reuse cached desiredBody directly when it fits to prevent re-excerption double-squeezing
      const body =
        exBudget >= entry.desiredBody.length
          ? entry.desiredBody
          : renderTier2Body(entry.node, exBudget);
      if (!body) continue;
      const line = `${entry.prefix}${body}`;
      const fullFlattenedLen = isFileArtifactNode(entry.node)
        ? flattenCodeLines(entry.node.content).length
        : cleanCompactFragment(entry.node.content).length;
      pushZone2Line({
        tierRank: 2,
        text: line,
        nodes: [entry.node],
        lod: body.length >= fullFlattenedLen ? "full" : "summary",
      });
    }
  }

  // --- Tier 3: Collapsed unmodified read-only files, passing checks, other low-priority items ---
  if (unmodifiedReadFiles.length > 0 && remainingChars >= 36) {
    const sep = zone2Lines.length > 0 ? 1 : 0;
    const otherBuilt = buildOtherReadFilesLine(Math.min(180, remainingChars - sep));
    if (otherBuilt) {
      pushZone2Line({
        tierRank: 3,
        text: otherBuilt.line,
        nodes: otherBuilt.included,
        lod: "stub",
      });
    }
  }

  for (const n of passingChecks) {
    const sep = zone2Lines.length > 0 ? 1 : 0;
    if (remainingChars - sep < 44) break;
    const cmd =
      typeof n.metadata.command === "string" && n.metadata.command.trim()
        ? n.metadata.command.trim()
        : (n.resourceKey?.replace(/^cmd:/, "") ?? n.id);
    const prefix = `- [check ok] \`${cmd}\`: `;
    const exCap = Math.min(90, remainingChars - sep - prefix.length);
    if (exCap < 10) break;
    const ex = truncateAtWordBoundary(
      flattenCommandLines(compactCommandExcerpt(n.content, true, exCap)),
      exCap,
    );
    if (!ex) continue;
    pushZone2Line({ tierRank: 3, text: `${prefix}${ex}`, nodes: [n], lod: "summary" });
  }

  for (const n of otherTier3Nodes) {
    // Skip low-priority search results when budget is tight
    if (n.metadata.artifactFamily === "search" && remainingChars < 220) continue;
    const sep = zone2Lines.length > 0 ? 1 : 0;
    if (remainingChars - sep < 44) break;
    const prefix = `- [${n.kind}]: `;
    const exCap = Math.min(180, remainingChars - sep - prefix.length);
    if (exCap < 12) break;
    const body = truncateAtWordBoundary(
      collapseNewlinesToSpace(cleanCompactFragment(n.content)),
      exCap,
    );
    if (!body) continue;
    pushZone2Line({ tierRank: 3, text: `${prefix}${body}`, nodes: [n], lod: "summary" });
  }

  // Conflict warnings (compact markdown line if room)
  const nodeMap = new Map(activeNodes.map((n) => [n.id, n]));
  const conflicts: ConflictWarning[] = activeConflictEdges.map((ce) => {
    const sNode = nodeMap.get(ce.sourceId)!;
    const tNode = nodeMap.get(ce.targetId)!;
    return {
      edgeId: ce.id,
      sourceNodeId: sNode.id,
      targetNodeId: tNode.id,
      sourceExcerpt: sNode.summary.slice(0, 100),
      targetExcerpt: tNode.summary.slice(0, 100),
      includedInContext: includedNodeIds.has(sNode.id) || includedNodeIds.has(tNode.id),
    };
  });

  const activeConflicts = conflicts.filter((c) => c.includedInContext);
  if (activeConflicts.length > 0 && remainingChars >= 40) {
    const sep = zone2Lines.length > 0 ? 1 : 0;
    const alertLine = truncateAtWordBoundary(
      `- [conflict]: ${activeConflicts.map((c) => `${c.sourceNodeId} <-> ${c.targetNodeId}`).join("; ")}`,
      remainingChars - sep,
    );
    if (alertLine.length + sep <= remainingChars) {
      zone2Lines.push({ tierRank: 3, text: alertLine, nodes: [], lod: "summary" });
      remainingChars -= alertLine.length + sep;
    }
  }

  // Assemble text and strictly enforce both `maxOutputChars` and `maxTokens`
  const buildPromptText = () => {
    const z1 = zone1Entries.map((e) => e.renderedText).join("\n");
    const z2 = zone2Lines.map((l) => l.text).join("\n");
    if (z1 && z2) return `${z1}\n${z2}`;
    return z1 || z2;
  };

  let finalPromptText = buildPromptText();
  let tokenIds = tokenizeToIds(finalPromptText);

  while (
    (finalPromptText.length > maxOutputChars || tokenIds.length > maxTokens) &&
    zone2Lines.length > 0
  ) {
    // Drop lowest-priority (highest tierRank) line from the tail first
    let dropIdx = -1;
    let worstRank = -1;
    for (let i = zone2Lines.length - 1; i >= 0; i--) {
      if (zone2Lines[i]!.tierRank > worstRank) {
        worstRank = zone2Lines[i]!.tierRank;
        dropIdx = i;
      }
    }
    if (dropIdx < 0) break;
    if (zone2Lines.length === 1 && finalPromptText.length > maxOutputChars) {
      const targetLen = Math.max(0, maxOutputChars - zone1UsedChars - (zone1Entries.length > 0 ? 1 : 0));
      zone2Lines[0] = {
        ...zone2Lines[0]!,
        text: truncateAtWordBoundary(zone2Lines[0]!.text, targetLen),
      };
      finalPromptText = buildPromptText();
      tokenIds = tokenizeToIds(finalPromptText);
      break;
    }
    const removed = zone2Lines.splice(dropIdx, 1)[0]!;
    for (const rn of removed.nodes) includedNodeIds.delete(rn.id);
    finalPromptText = buildPromptText();
    tokenIds = tokenizeToIds(finalPromptText);
  }

  // Final cleanliness pass guaranteeing zero pruned/head-tail markers and synced messages
  let hardTruncated = false;
  if (finalPromptText.length > maxOutputChars) {
    finalPromptText = truncateAtWordBoundary(finalPromptText, maxOutputChars);
    hardTruncated = true;
  }

  finalPromptText = cleanCompactFragment(finalPromptText);
  if (finalPromptText.length > maxOutputChars) {
    finalPromptText = truncateAtWordBoundary(finalPromptText, maxOutputChars);
    hardTruncated = true;
  }
  tokenIds = tokenizeToIds(finalPromptText);

  // Build Zone 2 CompiledNodeEntry list and record budget-pruned exclusions
  const zone2Entries: CompiledNodeEntry[] = [];
  for (const line of zone2Lines) {
    for (const n of line.nodes) {
      lodCounts[line.lod]++;
      zone2Entries.push({
        id: n.id,
        seq: n.seq,
        turn: n.turn,
        kind: n.kind,
        role: n.role,
        tier: n.tier,
        scope: n.scope,
        zone: 2,
        lod: line.lod,
        version: n.version,
        resourceKey: n.resourceKey,
        tokens: countTokens(line.text),
        activationScore: pprScoreById.get(n.id) ?? 0,
        renderedText: line.text,
        pulledByClosure: false,
      });
    }
  }

  for (const cand of zone2Candidates) {
    if (!includedNodeIds.has(cand.id)) {
      excludedNodes.push({
        id: cand.id,
        kind: cand.kind,
        resourceKey: cand.resourceKey,
        version: cand.version,
        reason: "budget_pruned",
        tokensSaved: cand.tokens,
      });
    }
  }

  let zone1Text = cleanCompactFragment(zone1Entries.map((e) => e.renderedText).join("\n"));
  let zone2Text = cleanCompactFragment(zone2Lines.map((l) => l.text).join("\n"));

  const messages: ChatMessage[] = [];
  if (hardTruncated) {
    // Keep messages strictly consistent with final truncated promptText
    if (finalPromptText) {
      const onlyZone1 = zone1Entries.length > 0 && zone2Lines.length === 0;
      messages.push({
        role: onlyZone1 ? "system" : "user",
        name: onlyZone1 ? "zone_1_stable_prefix" : "zone_2_working_subgraph",
        content: finalPromptText,
      });
      if (onlyZone1) {
        zone1Text = finalPromptText;
        zone2Text = "";
      } else {
        zone1Text = "";
        zone2Text = finalPromptText;
      }
    } else {
      zone1Text = "";
      zone2Text = "";
    }
  } else {
    if (zone1Text) {
      messages.push({ role: "system", name: "zone_1_stable_prefix", content: zone1Text });
    }
    if (zone2Text) {
      messages.push({ role: "user", name: "zone_2_working_subgraph", content: zone2Text });
    }
  }

  const agentLoopMessages: AgentLoopCompatibleMessage[] = [
    ...zone1Entries,
    ...zone2Entries,
  ].map((entry) => ({
    id: entry.id,
    sessionId: graph.id,
    role: entry.role,
    parts: [{ type: "text", text: entry.renderedText }],
    seq: entry.seq,
    ...(entry.resourceKey ? { key: entry.resourceKey } : {}),
    lod: entry.lod,
    zone: entry.zone,
  }));

  const prevTokenIds = options.previousTokenIds ?? graph.lastCompiledTokenIds;
  const prefixCachedTokens =
    prevTokenIds.length > 0 ? longestCommonPrefixLength(prevTokenIds, tokenIds) : 0;
  const totalTokens = tokenIds.length;
  const uncachedTokens = totalTokens - prefixCachedTokens;
  const prefixHitRate =
    prevTokenIds.length > 0 && totalTokens > 0 ? prefixCachedTokens / totalTokens : 0;
  const effectiveBillableTokens =
    Math.round((prefixCachedTokens * CACHED_TOKEN_COST_WEIGHT + uncachedTokens * 1.0) * 100) / 100;

  const actualZone1Tokens = countTokens(zone1Text);
  const actualZone2Tokens = countTokens(zone2Text);
  const framingTokens = Math.max(0, totalTokens - actualZone1Tokens - actualZone2Tokens);

  if (options.recordCacheState !== false && options.previousTokenIds === undefined) {
    store.saveCompiledTokens(graph.id, tokenIds);
  }

  return {
    graphId: graph.id,
    focusQuery,
    activeScope,
    maxTokens,
    messages,
    promptText: finalPromptText,
    agentLoopMessages,
    tokenIds,
    zone1Nodes: zone1Entries,
    zone2Nodes: zone2Entries,
    excludedNodes,
    conflicts,
    kvCache: {
      totalTokens,
      zone1Tokens: actualZone1Tokens,
      zone2Tokens: actualZone2Tokens,
      framingTokens,
      prefixCachedTokens,
      uncachedTokens,
      prefixHitRate: Math.round(prefixHitRate * 10000) / 10000,
      cachedCostDiscount: CACHED_TOKEN_COST_WEIGHT,
      effectiveBillableTokens,
    },
    diagnostics: {
      totalGraphNodes: nodes.length,
      activeCandidates: activeNodes.length,
      supersededOccluded,
      retractedPruned,
      closureAdditions: 0,
      lodCounts,
      compressionRatioVsFullLinear:
        fullLinearTokens > 0 ? Math.round((totalTokens / fullLinearTokens) * 10000) / 10000 : 1,
    },
  };
}

export function compileContext(store: GraphStore, options: CompileOptions): CompileResult {
  const { graphId, ...rawCompileOpts } = options;
  const validatedOpts = validateCompileOptions(rawCompileOpts);
  const effectiveOptions: CompileOptions = { ...validatedOpts, graphId };

  const snapshot = store.getGraphSnapshot(effectiveOptions.graphId);
  const { graph, nodes, edges } = snapshot;

  const focusQuery = effectiveOptions.focusQuery?.trim() ?? "";
  const activeScope = effectiveOptions.activeScope?.trim() || "global";
  const maxTokens =
    typeof effectiveOptions.maxTokens === "number" && effectiveOptions.maxTokens >= 64
      ? Math.floor(effectiveOptions.maxTokens)
      : graph.defaultBudget;
  const zone1MaxRatio =
    typeof effectiveOptions.zone1MaxRatio === "number" &&
    effectiveOptions.zone1MaxRatio > 0 &&
    effectiveOptions.zone1MaxRatio < 0.9
      ? effectiveOptions.zone1MaxRatio
      : 0.45;
  const pprAlpha = effectiveOptions.pprAlpha ?? 0.85;
  const pprIterations = effectiveOptions.pprIterations ?? 20;

  const resolutionMap = buildForwardResolutionMap(nodes);
  const excludedNodes: ExcludedNodeEntry[] = [];
  const activeNodes: GraphNode[] = [];
  let supersededOccluded = 0;
  let retractedPruned = 0;
  let fullLinearTokens = 0;

  for (const n of nodes) {
    fullLinearTokens += n.tokens;
    if (n.status === "superseded") {
      supersededOccluded++;
      excludedNodes.push({
        id: n.id,
        kind: n.kind,
        resourceKey: n.resourceKey,
        version: n.version,
        reason: "superseded",
        supersededBy: resolutionMap.get(n.id)?.id ?? n.supersededBy,
        tokensSaved: n.tokens,
      });
    } else if (n.status === "retracted") {
      retractedPruned++;
      excludedNodes.push({
        id: n.id,
        kind: n.kind,
        resourceKey: n.resourceKey,
        version: n.version,
        reason: "retracted",
        tokensSaved: n.tokens,
      });
    } else {
      activeNodes.push(n);
    }
  }

  // Partition active nodes into Zone 1 (Stable Prefix Spine) and Zone 2 (Working Subgraph)
  const zone1Candidates: GraphNode[] = [];
  const zone2Candidates: GraphNode[] = [];
  for (const n of activeNodes) {
    if (n.tier === "pinned_prefix" || n.tier === "stable_frame") {
      zone1Candidates.push(n);
    } else {
      zone2Candidates.push(n);
    }
  }

  // Zone 1 is strictly sorted by monotonic creation sequence (seq ASC)
  zone1Candidates.sort((a, b) => a.seq - b.seq);

  // Precompute exact rendered block & token counts for active nodes across LODs
  const renderedCache = new Map<string, Record<LodLevel, { text: string; tokens: number }>>();
  const getRendered = (node: GraphNode, lod: LodLevel) => {
    let entry = renderedCache.get(node.id);
    if (!entry) {
      const fullText = renderCanonicalNodeBlock(node, "full");
      const sumText = renderCanonicalNodeBlock(node, "summary");
      const stubText = renderCanonicalNodeBlock(node, "stub");
      entry = {
        full: { text: fullText, tokens: countTokens(fullText) + 1 },
        summary: { text: sumText, tokens: countTokens(sumText) + 1 },
        stub: { text: stubText, tokens: countTokens(stubText) + 1 },
      };
      renderedCache.set(node.id, entry);
    }
    return entry[lod];
  };

  // Step 2: Compile Zone 1 within zone1BudgetCap
  const zone1Cap = Math.floor(maxTokens * zone1MaxRatio);
  const zone1Lod = new Map<string, LodLevel>();
  let zone1Tokens = 0;
  for (const n of zone1Candidates) {
    zone1Lod.set(n.id, "full");
    zone1Tokens += getRendered(n, "full").tokens;
  }

  if (zone1Tokens > zone1Cap) {
    // Gracefully compress older stable_frame nodes first, preserving exact seq order
    for (const n of zone1Candidates) {
      if (zone1Tokens <= zone1Cap) break;
      if (n.tier === "stable_frame") {
        const delta = getRendered(n, "full").tokens - getRendered(n, "summary").tokens;
        if (delta > 0) {
          zone1Lod.set(n.id, "summary");
          zone1Tokens -= delta;
        }
      }
    }
  }
  if (zone1Tokens > maxTokens - 32) {
    // Extreme tiny-budget safety fallback
    for (const n of zone1Candidates) {
      if (zone1Tokens <= maxTokens - 32) break;
      const curr = zone1Lod.get(n.id)!;
      if (curr !== "stub") {
        const delta = getRendered(n, curr).tokens - getRendered(n, "stub").tokens;
        if (delta > 0) {
          zone1Lod.set(n.id, "stub");
          zone1Tokens -= delta;
        }
      }
    }
  }

  // Step 3: Personalized PageRank over all active nodes
  const activeIndex = new Map<string, number>();
  activeNodes.forEach((n, idx) => activeIndex.set(n.id, idx));
  const N = activeNodes.length;
  const maxTurn = activeNodes.reduce((m, n) => Math.max(m, n.turn), 1);
  const focusSet = new Set(options.focusNodeIds ?? []);
  const queryEmbedding = focusQuery ? embedText(focusQuery) : null;

  const seed = new Float64Array(N);
  let seedSum = 0;
  for (let i = 0; i < N; i++) {
    const node = activeNodes[i]!;
    let s = 0.16 * node.salience + 0.12 * Math.exp(-0.08 * Math.max(0, maxTurn - node.turn));
    if (queryEmbedding) {
      const sim = Math.max(0, cosineSimilarity(queryEmbedding, node.embedding));
      s += 0.58 * sim + lexicalResourceBoost(focusQuery, node);
    }
    if (activeScope === "integration") {
      if (node.kind === "artifact" || node.kind === "decision") s += 0.22;
    } else if (activeScope !== "global") {
      if (node.scope === activeScope) {
        s += 0.3;
      } else if (node.scope !== "global") {
        s *= 0.55;
      }
    }
    if (node.turn === maxTurn && node.role === "user" && node.tier !== "pinned_prefix") {
      s += 0.95;
    }
    if (node.metadata.unresolvedFailure === true) {
      s += 0.65;
    }
    if (focusSet.has(node.id)) {
      s += 1.25;
    }
    seed[i] = Math.max(1e-6, s);
    seedSum += seed[i]!;
  }
  for (let i = 0; i < N; i++) {
    seed[i] = seed[i]! / (seedSum || 1);
  }

  // Build weighted adjacency & active depends_on / summarizes / conflicts maps
  const outgoing = Array.from({ length: N }, () => [] as Array<{ to: number; w: number }>);
  const activeDependsOn = new Map<string, Set<string>>();
  const summarizedTargets = new Map<string, Set<string>>();
  const activeConflictEdges: GraphEdge[] = [];

  for (const e of edges) {
    if (e.kind === "supersedes") continue;
    const resolvedSrc = resolutionMap.get(e.sourceId);
    const resolvedTgt = resolutionMap.get(e.targetId);
    if (!resolvedSrc || !resolvedTgt || resolvedSrc.id === resolvedTgt.id) continue;

    const si = activeIndex.get(resolvedSrc.id);
    const ti = activeIndex.get(resolvedTgt.id);
    if (si === undefined || ti === undefined) continue;

    const fw = (EDGE_FORWARD_WEIGHT[e.kind] ?? 0.5) * e.weight;
    const rw = (EDGE_REVERSE_WEIGHT[e.kind] ?? 0.35) * e.weight;
    outgoing[si]!.push({ to: ti, w: fw });
    outgoing[ti]!.push({ to: si, w: rw });

    if (e.kind === "depends_on") {
      let set = activeDependsOn.get(resolvedSrc.id);
      if (!set) {
        set = new Set();
        activeDependsOn.set(resolvedSrc.id, set);
      }
      set.add(resolvedTgt.id);
    } else if (e.kind === "summarizes") {
      let set = summarizedTargets.get(resolvedSrc.id);
      if (!set) {
        set = new Set();
        summarizedTargets.set(resolvedSrc.id, set);
      }
      set.add(resolvedTgt.id);
    } else if (e.kind === "conflicts_with") {
      activeConflictEdges.push({
        ...e,
        sourceId: resolvedSrc.id,
        targetId: resolvedTgt.id,
      });
    }
  }

  const outSum = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (const edge of outgoing[i]!) sum += edge.w;
    outSum[i] = sum;
  }

  let rank = Float64Array.from(seed);
  for (let iter = 0; iter < pprIterations; iter++) {
    const next = new Float64Array(N);
    let danglingMass = 0;
    for (let i = 0; i < N; i++) {
      if (outSum[i]! <= 1e-12) {
        danglingMass += rank[i]!;
      } else {
        const share = rank[i]! / outSum[i]!;
        for (const edge of outgoing[i]!) {
          next[edge.to] += share * edge.w;
        }
      }
    }
    for (let j = 0; j < N; j++) {
      next[j] = (1 - pprAlpha) * seed[j]! + pprAlpha * (next[j]! + danglingMass * seed[j]!);
    }
    rank = next;
  }

  const pprScoreById = new Map<string, number>();
  for (let i = 0; i < N; i++) {
    pprScoreById.set(activeNodes[i]!.id, rank[i]!);
  }

  // Step 4: Causal Ancestor Closure & Multi-LOD Knapsack Planning in Zone 2
  const zone1IdSet = new Set(zone1Candidates.map((n) => n.id));
  const zone2ById = new Map(zone2Candidates.map((n) => [n.id, n]));

  const computeZone2Closure = (rootId: string): GraphNode[] => {
    const closure: GraphNode[] = [];
    const visited = new Set<string>([rootId]);
    const stack = [rootId];
    while (stack.length > 0) {
      const curr = stack.pop()!;
      const deps = activeDependsOn.get(curr);
      if (!deps) continue;
      for (const depId of deps) {
        if (!visited.has(depId)) {
          visited.add(depId);
          stack.push(depId);
          if (!zone1IdSet.has(depId)) {
            const z2Node = zone2ById.get(depId);
            if (z2Node) closure.push(z2Node);
          }
        }
      }
    }
    return closure;
  };

  const editedFileScopes = new Set<string>();
  const errorTargetFileScopes = new Set<string>();
  for (const n of activeNodes) {
    const scopeKey = getFileScopeKey(n);
    if (!scopeKey) continue;
    if (
      n.metadata.editedScope === true ||
      n.metadata.fileOpKind === "partial_edit" ||
      n.metadata.fileOpKind === "full_write" ||
      n.metadata.fileOpKind === "delete"
    ) {
      editedFileScopes.add(scopeKey);
    }
    if (isFileErrorNode(n)) {
      errorTargetFileScopes.add(scopeKey);
    }
  }
  const priorityFileScopes = new Set<string>([...editedFileScopes, ...errorTargetFileScopes]);

  const isPriorityFileNode = (n: GraphNode): boolean => {
    if (n.metadata.editedScope === true || n.metadata.fileOpKind === "ranged_read") {
      return true;
    }
    if (isFileArtifactNode(n)) {
      const sk = getFileScopeKey(n);
      if (sk && priorityFileScopes.has(sk)) return true;
    }
    return false;
  };

  const utilityOf = (n: GraphNode): number => {
    const base = pprScoreById.get(n.id) ?? 0.001;
    const isLatestUser = n.turn === maxTurn && n.role === "user";
    const isUnresolvedFailure = n.metadata.unresolvedFailure === true;
    const kindMult = isUnresolvedFailure ? 2.2 : KIND_UTILITY_MULTIPLIER[n.kind];
    const scopeBoost = !isUnresolvedFailure && isPriorityFileNode(n) ? 1.35 : 1.0;
    return base * kindMult * scopeBoost * (isLatestUser ? 2.5 : 1.0);
  };

  if (options.format === "compact_markdown") {
    return compileCompactMarkdown({
      store,
      graph,
      options,
      focusQuery,
      activeScope,
      maxTokens,
      zone1MaxRatio,
      nodes,
      activeNodes,
      zone1Candidates,
      zone2Candidates,
      excludedNodes,
      activeConflictEdges,
      pprScoreById,
      utilityOf,
      isPriorityFileNode,
      supersededOccluded,
      retractedPruned,
      fullLinearTokens,
    });
  }

  const rankedZone2 = [...zone2Candidates].sort((a, b) => utilityOf(b) - utilityOf(a));

  const wrapperSample =
    `<zone_1_stable_prefix>\n</zone_1_stable_prefix>\n<zone_2_working_subgraph>\n</zone_2_working_subgraph>`;
  const framingAllowance = countTokens(wrapperSample) + (activeConflictEdges.length > 0 ? 48 : 6);
  const availableZone2Budget = Math.max(0, maxTokens - zone1Tokens - framingAllowance);

  const selectedZone2Lod = new Map<string, LodLevel>();
  const pulledByClosureSet = new Set<string>();
  let usedZone2Tokens = 0;

  // Phase A: Admit top nodes + their depends_on ancestors at baseline compact LOD
  for (const cand of rankedZone2) {
    if (selectedZone2Lod.has(cand.id)) continue;
    const ancestors = computeZone2Closure(cand.id).filter((a) => !selectedZone2Lod.has(a.id));
    const batch = [cand, ...ancestors];

    let batchCost = 0;
    const batchLods = new Map<string, LodLevel>();
    for (const member of batch) {
      const isCurrentPrompt = member.turn === maxTurn && member.role === "user";
      const isPriorityFailure = member.metadata.unresolvedFailure === true;
      const remBudget = availableZone2Budget - usedZone2Tokens - batchCost;
      const baseLod: LodLevel =
        (isCurrentPrompt || isPriorityFailure) && getRendered(member, "full").tokens <= remBudget
          ? "full"
          : isPriorityFailure && getRendered(member, "summary").tokens <= remBudget
            ? "summary"
            : "stub";
      batchLods.set(member.id, baseLod);
      batchCost += getRendered(member, baseLod).tokens;
    }

    if (usedZone2Tokens + batchCost <= availableZone2Budget) {
      for (const member of batch) {
        selectedZone2Lod.set(member.id, batchLods.get(member.id)!);
        if (member.id !== cand.id) {
          pulledByClosureSet.add(member.id);
        }
      }
      usedZone2Tokens += batchCost;
    }
  }

  // Phase B: If an admitted summary node covers leaf nodes that are only stubs and not depended upon, consolidate
  const dependedUponIds = new Set<string>();
  for (const [, deps] of activeDependsOn) {
    for (const d of deps) dependedUponIds.add(d);
  }
  for (const [sumId, targets] of summarizedTargets) {
    if (!selectedZone2Lod.has(sumId)) continue;
    // Upgrade summary node to at least "summary" if affordable
    if (selectedZone2Lod.get(sumId) === "stub") {
      const sumNode = zone2ById.get(sumId);
      if (sumNode) {
        const delta = getRendered(sumNode, "summary").tokens - getRendered(sumNode, "stub").tokens;
        if (usedZone2Tokens + delta <= availableZone2Budget) {
          selectedZone2Lod.set(sumId, "summary");
          usedZone2Tokens += delta;
        }
      }
    }
    for (const targetId of targets) {
      if (
        selectedZone2Lod.get(targetId) === "stub" &&
        !dependedUponIds.has(targetId) &&
        !focusSet.has(targetId)
      ) {
        const tNode = zone2ById.get(targetId);
        if (tNode && tNode.kind === "observation") {
          usedZone2Tokens -= getRendered(tNode, "stub").tokens;
          selectedZone2Lod.delete(targetId);
          excludedNodes.push({
            id: tNode.id,
            kind: tNode.kind,
            resourceKey: tNode.resourceKey,
            version: tNode.version,
            reason: "summarized",
            tokensSaved: tNode.tokens,
          });
        }
      }
    }
  }

  // Phase C: Greedy marginal-gain LOD upgrades (stub -> summary -> full)
  while (true) {
    let bestId: string | null = null;
    let bestNextLod: LodLevel | null = null;
    let bestDelta = 0;
    let bestPriority = -1;

    for (const [nodeId, currLod] of selectedZone2Lod) {
      if (currLod === "full") continue;
      const node = zone2ById.get(nodeId)!;
      const nextLod: LodLevel = currLod === "stub" ? "summary" : "full";
      const delta = Math.max(0, getRendered(node, nextLod).tokens - getRendered(node, currLod).tokens);
      if (usedZone2Tokens + delta > availableZone2Budget) continue;

      const kindBoost =
        node.metadata.unresolvedFailure === true
          ? 2.4
          : node.kind === "artifact"
            ? isPriorityFileNode(node)
              ? 1.95
              : 1.75
            : node.kind === "decision"
              ? 1.45
              : node.turn === maxTurn && node.role === "user"
                ? 2.2
                : 1.0;
      const stepGain = utilityOf(node) * kindBoost * (nextLod === "summary" ? 0.44 : 0.38);
      const priority = stepGain / Math.pow(Math.max(1, delta), 0.52);
      if (priority > bestPriority) {
        bestPriority = priority;
        bestId = nodeId;
        bestNextLod = nextLod;
        bestDelta = delta;
      }
    }

    if (!bestId || !bestNextLod) break;
    selectedZone2Lod.set(bestId, bestNextLod);
    usedZone2Tokens += bestDelta;
  }

  // Record budget-pruned active Zone 2 nodes
  const summarizedExcludedIds = new Set(
    excludedNodes.filter((e) => e.reason === "summarized").map((e) => e.id),
  );
  for (const cand of zone2Candidates) {
    if (!selectedZone2Lod.has(cand.id) && !summarizedExcludedIds.has(cand.id)) {
      excludedNodes.push({
        id: cand.id,
        kind: cand.kind,
        resourceKey: cand.resourceKey,
        version: cand.version,
        reason: "budget_pruned",
        tokensSaved: cand.tokens,
      });
    }
  }

  // Step 5: Canonical seq ASC serialization in both Zone 1 & Zone 2
  const lodCounts: Record<LodLevel, number> = { full: 0, summary: 0, stub: 0 };

  const zone1Entries: CompiledNodeEntry[] = zone1Candidates.map((n) => {
    const lod = zone1Lod.get(n.id)!;
    lodCounts[lod]++;
    const rendered = getRendered(n, lod);
    return {
      id: n.id,
      seq: n.seq,
      turn: n.turn,
      kind: n.kind,
      role: n.role,
      tier: n.tier,
      scope: n.scope,
      zone: 1,
      lod,
      version: n.version,
      resourceKey: n.resourceKey,
      tokens: rendered.tokens,
      activationScore: pprScoreById.get(n.id) ?? 0,
      renderedText: rendered.text,
      pulledByClosure: false,
    };
  });

  const selectedZone2Nodes = zone2Candidates.filter((n) => selectedZone2Lod.has(n.id));
  // Volatility-stratified canonical ordering inside Zone 2:
  //   Sub-zone 2A: Durable structural state (`decision`, `artifact`, `summary`) in seq ASC
  //   Sub-zone 2B: Conversational facts & `ephemeral_leaf` observations in seq ASC
  //   Sub-zone 2C: Current-turn user prompt at the tail
  // Placing lower-churn structural nodes ahead of volatile observations extends
  // Longest Common Prefix KV-Cache hits deep into Zone 2 across consecutive turns.
  const zone2SubtierRank = (n: GraphNode): number => {
    if (n.turn === maxTurn && n.role === "user") return 3;
    if (n.tier === "ephemeral_leaf" || n.kind === "observation") return 2;
    if (n.kind === "decision" || n.kind === "artifact" || n.kind === "summary") return 0;
    return 1;
  };
  selectedZone2Nodes.sort((a, b) => {
    const rankDiff = zone2SubtierRank(a) - zone2SubtierRank(b);
    if (rankDiff !== 0) return rankDiff;
    return a.seq - b.seq;
  });

  const zone2Entries: CompiledNodeEntry[] = selectedZone2Nodes.map((n) => {
    const lod = selectedZone2Lod.get(n.id)!;
    lodCounts[lod]++;
    const rendered = getRendered(n, lod);
    return {
      id: n.id,
      seq: n.seq,
      turn: n.turn,
      kind: n.kind,
      role: n.role,
      tier: n.tier,
      scope: n.scope,
      zone: 2,
      lod,
      version: n.version,
      resourceKey: n.resourceKey,
      tokens: rendered.tokens,
      activationScore: pprScoreById.get(n.id) ?? 0,
      renderedText: rendered.text,
      pulledByClosure: pulledByClosureSet.has(n.id),
    };
  });

  // Step 6: Conflict Detection
  const includedIds = new Set<string>([
    ...zone1Entries.map((e) => e.id),
    ...zone2Entries.map((e) => e.id),
  ]);
  const nodeMap = new Map(activeNodes.map((n) => [n.id, n]));
  const conflicts: ConflictWarning[] = activeConflictEdges.map((ce) => {
    const sNode = nodeMap.get(ce.sourceId)!;
    const tNode = nodeMap.get(ce.targetId)!;
    return {
      edgeId: ce.id,
      sourceNodeId: sNode.id,
      targetNodeId: tNode.id,
      sourceExcerpt: sNode.summary.slice(0, 100),
      targetExcerpt: tNode.summary.slice(0, 100),
      includedInContext: includedIds.has(sNode.id) || includedIds.has(tNode.id),
    };
  });

  const conflictBanner =
    conflicts.some((c) => c.includedInContext)
      ? `\n<conflict_alerts>\n${conflicts
          .filter((c) => c.includedInContext)
          .map((c) => `[CONFLICT ALERT] ${c.sourceNodeId} <-> ${c.targetNodeId}`)
          .join("\n")}\n</conflict_alerts>`
      : "";

  const zone1Text = zone1Entries.map((e) => e.renderedText).join("\n");
  const zone2Text = zone2Entries.map((e) => e.renderedText).join("\n");
  const promptText =
    `<zone_1_stable_prefix>\n${zone1Text}\n</zone_1_stable_prefix>\n` +
    `<zone_2_working_subgraph>${conflictBanner}\n${zone2Text}\n</zone_2_working_subgraph>`;

  // Final tight budget trim if rare rounding pushed 1-2 tokens over maxTokens
  let finalPromptText = promptText;
  let tokenIds = tokenizeToIds(finalPromptText);
  while (tokenIds.length > maxTokens && zone2Entries.length > 1) {
    // Downgrade or drop lowest-utility non-tail entry in Zone 2
    let victimIdx = -1;
    let lowestScore = Infinity;
    for (let i = 0; i < zone2Entries.length - 1; i++) {
      const entry = zone2Entries[i]!;
      const z2Node = zone2ById.get(entry.id);
      const score = z2Node ? utilityOf(z2Node) : entry.activationScore;
      if (score < lowestScore) {
        lowestScore = score;
        victimIdx = i;
      }
    }
    if (victimIdx < 0) break;
    const victim = zone2Entries[victimIdx]!;
    if (victim.lod !== "stub") {
      const node = zone2ById.get(victim.id)!;
      lodCounts[victim.lod]--;
      lodCounts.stub++;
      const stubRendered = getRendered(node, "stub");
      zone2Entries[victimIdx] = {
        ...victim,
        lod: "stub",
        tokens: stubRendered.tokens,
        renderedText: stubRendered.text,
      };
    } else {
      lodCounts[victim.lod]--;
      zone2Entries.splice(victimIdx, 1);
    }
    const updatedZ2 = zone2Entries.map((e) => e.renderedText).join("\n");
    finalPromptText =
      `<zone_1_stable_prefix>\n${zone1Text}\n</zone_1_stable_prefix>\n` +
      `<zone_2_working_subgraph>${conflictBanner}\n${updatedZ2}\n</zone_2_working_subgraph>`;
    tokenIds = tokenizeToIds(finalPromptText);
  }

  // Construct OpenAI messages[] and @ppeng/agent-loop compatible messages
  const messages: ChatMessage[] = [];
  if (zone1Entries.length > 0) {
    messages.push({
      role: "system",
      name: "zone_1_stable_prefix",
      content: zone1Text,
    });
  }
  if (conflictBanner) {
    messages.push({
      role: "system",
      name: "conflict_alerts",
      content: conflictBanner.trim(),
    });
  }
  for (const entry of zone2Entries) {
    messages.push({
      role: entry.role,
      content: entry.renderedText,
    });
  }

  const agentLoopMessages: AgentLoopCompatibleMessage[] = [
    ...zone1Entries,
    ...zone2Entries,
  ].map((entry) => ({
    id: entry.id,
    sessionId: graph.id,
    role: entry.role,
    parts: [{ type: "text", text: entry.renderedText }],
    seq: entry.seq,
    ...(entry.resourceKey ? { key: entry.resourceKey } : {}),
    lod: entry.lod,
    zone: entry.zone,
  }));

  // Exact Token-Level Prefix KV-Cache Accounting
  const prevTokenIds = options.previousTokenIds ?? graph.lastCompiledTokenIds;
  const prefixCachedTokens =
    prevTokenIds.length > 0 ? longestCommonPrefixLength(prevTokenIds, tokenIds) : 0;
  const totalTokens = tokenIds.length;
  const uncachedTokens = totalTokens - prefixCachedTokens;
  const prefixHitRate =
    prevTokenIds.length > 0 && totalTokens > 0 ? prefixCachedTokens / totalTokens : 0;
  const effectiveBillableTokens =
    Math.round((prefixCachedTokens * CACHED_TOKEN_COST_WEIGHT + uncachedTokens * 1.0) * 100) / 100;

  const actualZone1Tokens = countTokens(`<zone_1_stable_prefix>\n${zone1Text}\n</zone_1_stable_prefix>`);
  const actualZone2Tokens = zone2Entries.reduce((acc, e) => acc + e.tokens, 0);
  const framingTokens = Math.max(0, totalTokens - actualZone1Tokens - actualZone2Tokens);

  if (options.recordCacheState !== false && options.previousTokenIds === undefined) {
    store.saveCompiledTokens(graph.id, tokenIds);
  }

  return {
    graphId: graph.id,
    focusQuery,
    activeScope,
    maxTokens,
    messages: options.format === "prompt" ? [] : messages,
    promptText: options.format === "messages" ? "" : finalPromptText,
    agentLoopMessages,
    tokenIds,
    zone1Nodes: zone1Entries,
    zone2Nodes: zone2Entries,
    excludedNodes,
    conflicts,
    kvCache: {
      totalTokens,
      zone1Tokens: actualZone1Tokens,
      zone2Tokens: actualZone2Tokens,
      framingTokens,
      prefixCachedTokens,
      uncachedTokens,
      prefixHitRate: Math.round(prefixHitRate * 10000) / 10000,
      cachedCostDiscount: CACHED_TOKEN_COST_WEIGHT,
      effectiveBillableTokens,
    },
    diagnostics: {
      totalGraphNodes: nodes.length,
      activeCandidates: activeNodes.length,
      supersededOccluded,
      retractedPruned,
      closureAdditions: pulledByClosureSet.size,
      lodCounts,
      compressionRatioVsFullLinear:
        fullLinearTokens > 0 ? Math.round((totalTokens / fullLinearTokens) * 10000) / 10000 : 1,
    },
  };
}
