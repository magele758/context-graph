/**
 * Declaration-aware code excerpting, application-frame-preserving command
 * diagnostics, and word-boundary-safe text sanitization.
 *
 * Pure zero-dependency helpers designed to avoid character-level 50/50
 * head/tail slicing that corrupts identifiers, numeric literals, line numbers,
 * and source stack frames, while maintaining strictly linear O(N) scanning.
 */

const PRUNED_LINE_RE =
  /^[ \t]*(?:(?:\u2026|\.{3})[ \t]*)?\[(?:pruned|truncated)\b[^\]\[\n]*\][ \t]*$/gim;
const PRUNED_INLINE_RE =
  /(?:(?:\u2026|\.{3})[ \t]*)?\[(?:pruned|truncated)\b[^\]\[\n]*\]/gi;
const HEAD_TAIL_TAG_RE =
  /\[[^\]\[\n]*head\/tail[ \t]+shown[^\]\[\n]*\]/gi;

const BOILERPLATE_ACK_LINE_RE =
  /^(?:successfully\s+)?(?:edited|updated|wrote|written|created|patched|modified|saved|applied\s+patch(?:\s+to)?|file\s+(?:updated|edited|written|saved))\b[^\n]*$/i;

const VENDOR_STACK_FRAME_RE =
  /(?:node_modules[\\/]|node:internal[\\/]|internal[\\/]modules[\\/]|\(node:[a-z0-9_/-]+:\d+|\bat\s+node:[a-z0-9_/-]+:\d+|\(internal\/[a-z0-9_/-]+:\d+|(?:^|[\s(/\\])(?:vendor|site-packages|dist-packages)[\\/]|\/go\/src\/(?:runtime|testing)[\\/]|^runtime(?:\.[a-z0-9_]+\(|\/[a-z0-9_]+(?:\.[a-z0-9_]+\(|\.(?:go|s):\d+)))/i;

const REDUNDANT_CMD_SUMMARY_RE =
  /^(?:tests?\s+\d+\s+(?:failed|passed)|test\s+files?\s+\d+\s+|command\s+exited\s+with\s+code\s+\d+|exit\s+code[:\s]+\d+|duration\s+\d|start\s+at\s+\d|transform\s+\d|npm\s+(?:err!|error)\b|elifecycle\b|>\s+\S+@\S+\s+|>\s+(?:vitest|jest|mocha|node|tsc|pytest|go\s+test|cargo\s+test)\b)/i;

const FAIL_HEADER_RE =
  /^(?:FAIL\b|FAILED\b|---\s*FAIL:|not\s+ok\s+\d|[\u2715\u2717\u2716\u00d7\u25cf\u276f]\s+\S|FAIL\s+\S)/;

const RUNNER_FILE_BANNER_RE =
  /^(?:(?:FAIL|FAILED)\s+|[\u276f\u2713\u2715\u2717\u2716\u00d7\u2022\u25cf]\s*)?[^\s()]+(?:\.(?:test|spec)\.[a-z0-9]+|_test\.[a-z0-9]+)\s*\(\d+\s+tests?\b[^)\n]*\)/i;

const LEADING_TEST_FILE_PREFIX_RE =
  /^[^\s():'"\n]+(?:\.(?:test|spec)\.[a-z0-9]+|_test\.[a-z0-9]+):\d+(?::\d+)?:\s+/i;

const LEADING_ARROW_RE = /^(?:\u2192|->)[ \t]+/;
const GO_FAIL_TIMING_RE = /^(---\s*FAIL:\s+\S+)[ \t]+\(\d+(?:\.\d+)?m?s\)$/;

const ERROR_DIAG_RE =
  /(?:\b(?:AssertionError|TypeError|SyntaxError|ReferenceError|RangeError|URIError|EvalError|AggregateError)\b|\b[A-Z][A-Za-z0-9]*Error\b(?:\s*[:[(]|$)|\bError\s*[:[(]|\b(?:error|fatal)\s*:|\bERR_[A-Z0-9_]+\b|\bE(?:CONNREFUSED|CONNRESET|TIMEDOUT|NOENT|ACCES|ADDRINUSE|PIPE|PERM|EXIST|NOTDIR|ISDIR)\b|^(?:Expected|Received|Actual|Diff|Message|Cause|panic|Traceback|Exception|Assertion\s+failed)\s*[:(]|\berror\s+TS\d+:|\berror\[E\d+\]:)/;

const APP_STACK_FRAME_RE =
  /(?:^\s*at\s+.*(?::\d+(?::\d+)?\)?$)|^\s*File\s+"[^"\n]+",\s+line\s+\d+|(?:^|[\s('"])(?:src|packages|apps|lib|server|client|api|cmd|pkg|internal|tests?|spec|bin)\/[^\s():'"\n]+\.\w+:\d+(?::\d+)?|\([^\s()\n]+\.\w+:\d+(?::\d+)?\))/i;

const TEST_FILE_FRAME_RE =
  /(?:\.(?:test|spec)\.[a-z0-9]+|_test\.[a-z0-9]+|(?:^|[\\/])test_[^\\/\s():'"\n]+\.[a-z0-9]+|(?:^|[\\/])__tests__[\\/])/i;

const SOURCE_ROOT_PREFIXES = [
  "packages/",
  "apps/",
  "src/",
  "pkg/",
  "lib/",
  "internal/",
  "cmd/",
  "server/",
  "client/",
  "api/",
  "bin/",
] as const;

const FILE_EXT_RE = /\.[A-Za-z][A-Za-z0-9]{0,7}$/;

/**
 * Strips trailing `:line` / `:line:col` / trailing `:` suffixes from a path token
 * using a strictly linear O(N) backward scan.
 */
function stripLineColSuffix(token: string): string {
  let end = token.length;
  while (end > 0 && token.charCodeAt(end - 1) === 58) {
    end--;
  }
  // Strip up to two `:digits` segments (`:line:col` or `:line`)
  for (let pass = 0; pass < 2; pass++) {
    let i = end;
    while (i > 0) {
      const c = token.charCodeAt(i - 1);
      if (c >= 48 && c <= 57) i--;
      else break;
    }
    if (i < end && i > 0 && token.charCodeAt(i - 1) === 58) {
      end = i - 1;
    } else {
      break;
    }
  }
  return end === token.length ? token : token.slice(0, end);
}

/**
 * Strictly O(N) extractor returning deduplicated non-vendor relative application
 * source file paths cited in failing command/test outputs (e.g. `src/...`,
 * `packages/...`, `apps/...`, `pkg/...`, `lib/...`, `internal/...`, `cmd/...`),
 * excluding `node_modules/` and `node:internal/`.
 * Non-test source files are ordered ahead of test harness files.
 */
export function extractAppFileReferences(output: string): string[] {
  if (!output) return [];
  const clean = sanitizeNeutralText(output);
  if (!clean) return [];

  const nonTestRefs: string[] = [];
  const testRefs: string[] = [];
  const seen = new Set<string>();

  const lines = clean.split("\n");
  for (const rawLine of lines) {
    if (!rawLine || VENDOR_STACK_FRAME_RE.test(rawLine)) continue;
    const tokens = rawLine.split(/[\s()'"<>,;]+/);
    for (const rawTok of tokens) {
      if (!rawTok || rawTok.length < 4) continue;
      let tok = rawTok.replace(/\\/g, "/");
      if (tok.startsWith("file://")) {
        tok = tok.slice("file://".length);
      }
      const lowerTok = tok.toLowerCase();
      if (
        lowerTok.includes("node_modules/") ||
        lowerTok.startsWith("node:") ||
        lowerTok.includes("internal/modules/")
      ) {
        continue;
      }

      const pathPart = stripLineColSuffix(tok);
      if (!pathPart || !FILE_EXT_RE.test(pathPart)) continue;

      // Find the earliest matching known source root prefix (so `packages/foo/src/a.ts`
      // extracts `packages/foo/src/a.ts` rather than truncating to `src/a.ts`).
      let bestStart = -1;
      for (const prefix of SOURCE_ROOT_PREFIXES) {
        let start = -1;
        if (pathPart.startsWith(prefix)) {
          start = 0;
        } else {
          const idx = pathPart.indexOf(`/${prefix}`);
          if (idx >= 0) {
            start = idx + 1;
          }
        }
        if (start >= 0 && (bestStart === -1 || start < bestStart)) {
          bestStart = start;
        }
      }

      if (bestStart < 0) continue;
      const rel = pathPart.slice(bestStart);
      if (!rel || seen.has(rel)) continue;
      seen.add(rel);
      if (TEST_FILE_FRAME_RE.test(rel)) {
        testRefs.push(rel);
      } else {
        nonTestRefs.push(rel);
      }
    }
  }

  return [...nonTestRefs, ...testRefs];
}

const PASS_SUMMARY_RE =
  /(?:\b\d+\s+(?:passed|passing|tests?\s+passed|specs?\s+passed|succeeded)\b|\bpass\s+\d+|\bPASS\b|\ball\s+\d+.*pass|\b0\s+errors?\b|\bbuild\s+succeeded\b|\bcompiled\s+successfully\b|^ok\s+\S+)/i;

const CLOSING_ONLY_RE = /^[}\])];?,?$/;

const IMPORT_LINE_RE =
  /^(?:import\b|from\s+\S+\s+import\b|export\s+\*\s+from\b|const\s+\S+\s*=\s*require\(|#include\b|using\s+\S+;|use\s+[A-Za-z0-9_:]+;)/;

const TOP_LEVEL_DECL_RE =
  /^(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(?:const|let|var|type|interface|class|function|enum)(?=\s|\*)/;

const EXPORTED_DECL_RE =
  /^(?:export\s+(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(?:const|let|var|type|interface|class|function|enum)(?=\s|\*)|pub(?:\([^)\n]*\))?\s+(?:async\s+)?(?:struct|enum|trait|fn|type|const|static)(?=\s))/;

const MULTILANG_DECL_RE =
  /^(?:pub(?:\([^)\n]*\))?\s+)?(?:async\s+)?(?:def|class|struct|enum|trait|impl|fn|func|type|const|static|let)(?=\s|<)/;

const ENV_OR_INI_KEY_RE =
  /^(?:[A-Z][A-Z0-9_.-]*\s*[:=]|\[[A-Za-z0-9_.-]+\]$)/;

const CODE_TOP_CONST_ASSIGN_RE = /^[A-Z][A-Z0-9_]*\s*=/;

const JSON_OR_YAML_KEY_RE =
  /^(?:"[A-Za-z0-9_.-]+"\s*:|[A-Za-z_][A-Za-z0-9_.-]*\s*:\s*\S)/;

const NON_KV_CONTROL_RE = /^(?:case\b|default\s*:|https?:)/i;

const LINE_NUMBER_GUTTER_RE = /^\s*\d+\s*(?:\||->|\u2192|\t|:\s)/;

/**
 * Strictly O(N) backward scan trimming trailing horizontal whitespace (` ` / `\t`),
 * avoiding quadratic regex backtracking on long space-prefixed lines.
 */
export function trimHorizontalEnd(line: string): string {
  let end = line.length;
  while (end > 0) {
    const ch = line.charCodeAt(end - 1);
    if (ch === 32 || ch === 9) {
      end--;
    } else {
      break;
    }
  }
  return end === line.length ? line : line.slice(0, end);
}

/**
 * Strictly linear O(N) marker removal with clean adjacent horizontal-space
 * collapse. Expects `re` to have the global flag and NO unanchored leading
 * whitespace quantifier.
 */
export function stripRegexLinear(text: string, re: RegExp): string {
  re.lastIndex = 0;
  let match = re.exec(text);
  if (!match) return text;

  let out = "";
  let cursor = 0;
  const len = text.length;

  while (match) {
    const start = match.index;
    const end = start + match[0]!.length;

    // Expand left over horizontal spaces back to cursor
    let l = start;
    while (l > cursor) {
      const ch = text.charCodeAt(l - 1);
      if (ch === 32 || ch === 9) l--;
      else break;
    }

    // Expand right over horizontal spaces
    let r = end;
    while (r < len) {
      const ch = text.charCodeAt(r);
      if (ch === 32 || ch === 9) r++;
      else break;
    }

    out += text.slice(cursor, l);

    const hasLeftNonNewline = l > 0 && text.charCodeAt(l - 1) !== 10;
    const hasRightNonNewline = r < len && text.charCodeAt(r) !== 10;
    if (hasLeftNonNewline && hasRightNonNewline) {
      out += " ";
    }

    cursor = r;
    if (re.lastIndex < r) {
      re.lastIndex = r;
    }
    if (end === start) {
      re.lastIndex = start + 1;
    }
    match = re.exec(text);
  }

  out += text.slice(cursor);
  return out;
}

/**
 * Strips upstream truncation / pruning markers (e.g. `[pruned 1234 chars; head/tail shown]`)
 * and normalizes line endings & whitespace in strictly linear O(N) time without
 * destroying code indentation.
 */
export function sanitizeNeutralText(text: string): string {
  if (!text) return "";
  let s = text.replace(/\r\n?/g, "\n");
  // Drop standalone marker lines completely
  s = s.replace(PRUNED_LINE_RE, "");
  // Drop inline markers & collapse directly adjacent horizontal spaces linearly
  s = stripRegexLinear(s, PRUNED_INLINE_RE);
  s = stripRegexLinear(s, HEAD_TAIL_TAG_RE);
  // Trim trailing horizontal spaces per line in O(N) and collapse excessive blank runs
  s = s
    .split("\n")
    .map((line) => trimHorizontalEnd(line))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return s;
}

/**
 * Truncates `text` to at most `maxChars` characters WITHOUT cutting mid-word,
 * mid-identifier, or mid-number. Always backs up to the last clean whitespace,
 * newline, or safe punctuation boundary before appending ` ...`.
 */
export function truncateAtWordBoundary(text: string, maxChars: number): string {
  if (!Number.isFinite(maxChars) || maxChars <= 0) return "";
  const cap = Math.floor(maxChars);
  const s = text.trim();
  if (s.length <= cap) return s;

  if (cap < 4) {
    return s.slice(0, cap);
  }
  const suffix = cap >= 8 ? " ..." : "...";
  const limit = cap - suffix.length;
  if (limit <= 0) {
    return s.slice(0, cap);
  }

  for (let k = limit; k >= 1; k--) {
    const prev = s[k - 1]!;
    const next = k < s.length ? s[k]! : "";

    // 1. Clean whitespace / newline boundary right after kept prefix
    if (next && /\s/.test(next)) {
      const prefix = s.slice(0, k).trimEnd();
      if (prefix.length > 0) {
        if (prefix.endsWith("...") || prefix.endsWith("\u2026")) {
          return prefix;
        }
        return `${prefix}${suffix}`;
      }
      continue;
    }

    // 2. Clean separator / closing punctuation right before cut
    //    (guard against commas inside numeric literals like 1,000)
    if (/[;|)}]/.test(prev)) {
      const prefix = s.slice(0, k).trimEnd();
      if (prefix.length > 0) {
        return `${prefix}${suffix}`;
      }
      continue;
    }
    if (prev === ",") {
      const prev2 = k >= 2 ? s[k - 2]! : "";
      const isNumericComma = /\d/.test(prev2) && /\d/.test(next);
      if (!isNumericComma) {
        const prefix = s.slice(0, k).trimEnd();
        if (prefix.length > 0) {
          return `${prefix}${suffix}`;
        }
      }
      continue;
    }

    // 3. Sentence period ONLY when not followed by digit/identifier/path char
    if (prev === "." && (!next || /[\s;,|)}\]]/.test(next))) {
      const prefix = s.slice(0, k).trimEnd();
      if (prefix.length > 0) {
        if (prefix.endsWith("...") || prefix.endsWith("\u2026")) {
          return prefix;
        }
        return `${prefix}${suffix}`;
      }
    }
  }

  // Fallback only when the very first unbroken token exceeds `limit`
  return `${s.slice(0, limit)}${suffix}`;
}

function stripOptionalLineNumber(line: string, hasGutter: boolean): string {
  if (!hasGutter) return line;
  return line.replace(LINE_NUMBER_GUTTER_RE, "");
}

function detectLineNumberGutter(lines: readonly string[]): boolean {
  let matched = 0;
  const sample = Math.min(lines.length, 8);
  for (let i = 0; i < sample; i++) {
    if (LINE_NUMBER_GUTTER_RE.test(lines[i]!)) {
      matched++;
    }
  }
  return sample >= 2 && matched >= Math.ceil(sample * 0.6);
}

function countLeadingIndent(line: string): number {
  let indent = 0;
  for (let i = 0; i < line.length; i++) {
    const ch = line.charCodeAt(i);
    if (ch === 32) indent += 1;
    else if (ch === 9) indent += 4;
    else break;
  }
  return indent;
}

function hasUnclosedTrailingBrace(line: string): boolean {
  if (!line.endsWith("{")) return false;
  let open = 0;
  let close = 0;
  for (let i = 0; i < line.length; i++) {
    const ch = line.charCodeAt(i);
    if (ch === 123) open++;
    else if (ch === 125) close++;
  }
  return open > close;
}

function stripTrailingOpenBrace(line: string): string {
  if (!hasUnclosedTrailingBrace(line)) return line;
  const withoutBrace = trimHorizontalEnd(line.slice(0, -1));
  if (withoutBrace.endsWith("=")) {
    return trimHorizontalEnd(withoutBrace.slice(0, -1));
  }
  return withoutBrace;
}

function classifyCodeLinePriority(
  rawLine: string,
  idx: number,
  hasGutter: boolean,
  hasCodeDecls: boolean,
  hasExportedDecls: boolean,
): { priority: number; indent: number } {
  const unnumbered = stripOptionalLineNumber(rawLine, hasGutter);
  const trimmed = unnumbered.trim();
  const indent = countLeadingIndent(unnumbered);
  if (!trimmed || CLOSING_ONLY_RE.test(trimmed)) {
    return { priority: 4, indent };
  }

  // Leading module header comment
  if (
    idx < 3 &&
    trimmed.length <= 120 &&
    /^(?:\/\/|\/\*|\*|--|#(?!include\b)|""")/.test(trimmed)
  ) {
    return { priority: 0, indent };
  }

  // Imports go below exported declarations & constants so long import blocks never starve symbols
  if (IMPORT_LINE_RE.test(trimmed)) {
    return { priority: 3, indent };
  }

  // Top-level declarations, constants, function/class/interface signatures.
  // When a module explicitly uses `export`/`pub`, reserve P1 for its exported API
  // surface so unexported trailing private helpers don't crowd the excerpt.
  if (indent <= 2 && (TOP_LEVEL_DECL_RE.test(trimmed) || MULTILANG_DECL_RE.test(trimmed))) {
    if (hasExportedDecls && !EXPORTED_DECL_RE.test(trimmed)) {
      return { priority: 2, indent };
    }
    return { priority: 1, indent };
  }

  // Top-level CONSTANT = value assignments in code files (e.g. Python module constants),
  // or full ENV / INI key assignments in config files
  if (hasCodeDecls) {
    if (!hasExportedDecls && indent <= 2 && CODE_TOP_CONST_ASSIGN_RE.test(trimmed)) {
      return { priority: 1, indent };
    }
    // Indented key: value lines inside code files (TS interfaces, JS objects, Go structs)
    // stay P2 rather than P1 so fields are never orphaned from their enclosing block.
    return { priority: 2, indent };
  }

  // Config / env / INI assignments
  if (indent <= 2 && ENV_OR_INI_KEY_RE.test(trimmed)) {
    return { priority: 1, indent };
  }

  // Shallow JSON / YAML config key-value lines (only for non-code config/data files)
  if (indent <= 4 && JSON_OR_YAML_KEY_RE.test(trimmed) && !NON_KV_CONTROL_RE.test(trimmed)) {
    return { priority: 1, indent };
  }

  // Other meaningful early head/body lines
  return { priority: 2, indent };
}

function stripBoilerplateAcks(text: string): string {
  const lines = text.split("\n");
  const kept: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (BOILERPLATE_ACK_LINE_RE.test(trimmed)) continue;
    kept.push(trimHorizontalEnd(line));
  }
  return kept.join("\n").trim();
}

/**
 * Produces a declaration-aware, word/line-boundary-safe excerpt of a source or
 * config file (or an edit diff when `opts.diffText` is supplied).
 *
 * Never performs 50/50 character-level head/tail splitting and never wastes
 * budget on trailing closing braces.
 */
export function compactCodeExcerpt(
  content: string,
  maxChars: number,
  opts?: { diffText?: string },
): string {
  if (!Number.isFinite(maxChars) || maxChars <= 0) return "";
  const cap = Math.floor(maxChars);

  if (opts?.diffText && opts.diffText.trim().length > 0) {
    const cleanDiff = stripBoilerplateAcks(sanitizeNeutralText(opts.diffText));
    if (cleanDiff.length > 0) {
      return packDiffLines(cleanDiff.split("\n"), cap);
    }
  }

  const clean = sanitizeNeutralText(content);
  if (!clean) return "";
  if (clean.length <= cap) {
    return clean;
  }

  const rawLines = clean
    .split("\n")
    .map((l) => trimHorizontalEnd(l))
    .filter((l) => l.trim().length > 0);
  if (rawLines.length === 0) return "";

  // If `content` itself is a stored diff block (containing both `-` and `+` lines),
  // preserve `+` replacement lines ahead of `-` removed lines under tight budgets.
  if (isPureDiffBlock(rawLines)) {
    return packDiffLines(rawLines, cap);
  }

  const hasGutter = detectLineNumberGutter(rawLines);
  let hasCodeDecls = false;
  let hasExportedDecls = false;
  for (const rawLine of rawLines) {
    const unnumbered = stripOptionalLineNumber(rawLine, hasGutter);
    const trimmed = unnumbered.trim();
    if (!trimmed) continue;
    const indent = countLeadingIndent(unnumbered);
    if (indent <= 2 && (TOP_LEVEL_DECL_RE.test(trimmed) || MULTILANG_DECL_RE.test(trimmed))) {
      hasCodeDecls = true;
      if (EXPORTED_DECL_RE.test(trimmed)) {
        hasExportedDecls = true;
      }
    }
  }

  const classified = rawLines.map((line, idx) => {
    const { priority, indent } = classifyCodeLinePriority(
      line,
      idx,
      hasGutter,
      hasCodeDecls,
      hasExportedDecls,
    );
    return {
      idx,
      text: hasGutter ? line.replace(/^\s+/, "") : line,
      priority,
      indent,
    };
  });

  // Pack in priority order: P1 declarations FIRST -> P0 leading comment only when sparse/comment-only -> P2 -> P3
  const selected = new Map<number, string>();
  let usedChars = 0;

  const tryAdd = (idx: number, text: string, allowTruncateSingleLine: boolean): boolean => {
    if (selected.has(idx)) return false;
    const sepLen = selected.size > 0 ? 1 : 0;
    const remaining = cap - usedChars - sepLen;
    if (remaining <= 0) return false;
    if (text.length <= remaining) {
      selected.set(idx, text);
      usedChars += sepLen + text.length;
      return true;
    }
    if (allowTruncateSingleLine && remaining >= 28) {
      const shortened = truncateAtWordBoundary(text, remaining);
      if (shortened.length > 0 && shortened.length <= remaining) {
        selected.set(idx, shortened);
        usedChars += sepLen + shortened.length;
        return true;
      }
    }
    return false;
  };

  // P1: pack real declarations, constants, signatures, and config keys FIRST
  const p1 = classified.filter((c) => c.priority === 1);
  for (const item of p1) {
    if (!tryAdd(item.idx, item.text, selected.size === 0)) {
      if (cap - usedChars >= 48) {
        tryAdd(item.idx, item.text, true);
      }
    }
  }
  const p1UsedChars = usedChars;

  // P0: include at most 1 leading module comment ONLY if no P1 declarations exist
  // or P1 is sparse (< 45% of cap) and the comment fits cleanly within 35% of cap.
  const p0 = classified.filter((c) => c.priority === 0);
  if (
    p0.length > 0 &&
    (p1.length === 0 ||
      (p1UsedChars < Math.floor(cap * 0.45) && p0[0]!.text.length <= Math.floor(cap * 0.35)))
  ) {
    tryAdd(p0[0]!.idx, p0[0]!.text, p1.length === 0 && selected.size === 0);
  }

  // Once top-level declarations/constants (P1) fill >= 65% of cap, stop packing
  // lower-priority P2 body boilerplate and P3 imports so excerpts stay dense.
  const p1FilledEnough = p1.length > 0 && p1UsedChars >= Math.floor(cap * 0.65);

  if (!p1FilledEnough) {
    // P2: meaningful head/body lines in source order
    const p2 = classified.filter((c) => c.priority === 2);
    for (const item of p2) {
      if (cap - usedChars < 16) break;
      if (!tryAdd(item.idx, item.text, selected.size === 0 || cap - usedChars >= 48)) {
        if (cap - usedChars < 48) break;
      }
    }

    // P3: imports only if space still remains
    const p3 = classified.filter((c) => c.priority === 3);
    for (const item of p3) {
      if (cap - usedChars < 24) break;
      tryAdd(item.idx, item.text, false);
    }
  }

  if (selected.size === 0) {
    return truncateAtWordBoundary(classified[0]!.text, cap);
  }

  // Emit in original source line order, cleaning dangling unclosed `{` on P1 code block
  // headers whose indented body lines were not included.
  const orderedEntries = Array.from(selected.entries()).sort((a, b) => a[0] - b[0]);
  const ordered = orderedEntries.map(([origIdx, text], pos) => {
    if (!hasCodeDecls) return text;
    const meta = classified[origIdx]!;
    if (meta.priority !== 1 || !hasUnclosedTrailingBrace(text)) {
      return text;
    }
    const nextEntry = orderedEntries[pos + 1];
    const hasImmediateBody =
      nextEntry !== undefined &&
      nextEntry[0] === origIdx + 1 &&
      classified[nextEntry[0]!]!.indent > meta.indent;
    return hasImmediateBody ? text : stripTrailingOpenBrace(text);
  });

  const joined = ordered.join("\n");
  return joined.length <= cap ? joined : truncateAtWordBoundary(joined, cap);
}

function isPureDiffBlock(lines: readonly string[]): boolean {
  let hasRemoved = false;
  let hasAdded = false;
  for (const line of lines) {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("-") && !trimmed.startsWith("---")) {
      hasRemoved = true;
    } else if (trimmed.startsWith("+") && !trimmed.startsWith("+++")) {
      hasAdded = true;
    } else {
      return false;
    }
  }
  return hasRemoved && hasAdded;
}

/**
 * Packs diff lines within `maxChars` while guaranteeing that `+` (added/new)
 * replacement lines are NEVER dropped to keep `-` (removed/old) lines.
 * Allocates at least 65% of `maxChars` to `+` lines (up to 100% when tight).
 */
function packDiffLines(lines: readonly string[], maxChars: number): string {
  const nonEmpty = lines.map((l) => trimHorizontalEnd(l)).filter((l) => l.trim().length > 0);
  if (nonEmpty.length === 0) return "";
  const full = nonEmpty.join("\n");
  if (full.length <= maxChars) return full;

  const removedLines: string[] = [];
  const addedLines: string[] = [];
  for (const line of nonEmpty) {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("-") && !trimmed.startsWith("---")) {
      removedLines.push(line);
    } else if (trimmed.startsWith("+") && !trimmed.startsWith("+++")) {
      addedLines.push(line);
    }
  }

  if (addedLines.length === 0 || removedLines.length === 0) {
    return packLinesSequentially(nonEmpty, maxChars);
  }

  // Always preserve `+` replacement lines: give them at least 65% of maxChars.
  const minAddedBudget = Math.ceil(maxChars * 0.65);
  const fullAdded = packLinesSequentially(addedLines, maxChars);

  // If all `+` lines fit with comfortable room for a `-` context line, include both
  if (fullAdded.length <= maxChars - 24) {
    const remForRemoved = maxChars - fullAdded.length - 1;
    const compactRemoved = packLinesSequentially(removedLines, remForRemoved);
    if (compactRemoved.length > 0 && compactRemoved.length + 1 + fullAdded.length <= maxChars) {
      return `${compactRemoved}\n${fullAdded}`;
    }
    return fullAdded;
  }

  // Tight budget: reserve at least 65% for `+` lines; include compact `-` only if budget allows
  const maxRemovedBudget = maxChars - minAddedBudget - 1;
  if (maxChars >= 48 && maxRemovedBudget >= 18) {
    const compactRemoved = packLinesSequentially(removedLines, maxRemovedBudget);
    if (compactRemoved.length > 0) {
      const addedBudget = maxChars - compactRemoved.length - 1;
      const compactAdded = packLinesSequentially(addedLines, addedBudget);
      if (compactAdded.length > 0 && compactRemoved.length + 1 + compactAdded.length <= maxChars) {
        return `${compactRemoved}\n${compactAdded}`;
      }
    }
  }

  return fullAdded;
}

function stripTrailingHexOffset(line: string): string {
  const idx = line.lastIndexOf("+0x");
  if (idx <= 1) return line;
  const beforeChar = line.charCodeAt(idx - 1);
  if (beforeChar !== 32 && beforeChar !== 9) return line;
  for (let i = idx + 3; i < line.length; i++) {
    const c = line.charCodeAt(i);
    const isHex =
      (c >= 48 && c <= 57) || (c >= 65 && c <= 70) || (c >= 97 && c <= 102);
    if (!isHex) return line;
  }
  return idx + 3 < line.length ? trimHorizontalEnd(line.slice(0, idx)) : line;
}

/**
 * Strictly O(N) extractor for Go-style stack-trace function lines
 * (e.g. `example.com/mod/pkg/queue.(*WorkerPool).AwaitAck(0xc000...)` ->
 * `queue.(*WorkerPool).AwaitAck`). Returns null for non-call lines.
 */
function extractShortGoFuncName(line: string): string | null {
  const trimmed = line.trim();
  if (trimmed.length < 4 || !trimmed.endsWith(")")) return null;

  // Match the trailing `(...)` argument list with a single backward paren-depth scan
  let depth = 0;
  let argOpenIdx = -1;
  for (let i = trimmed.length - 1; i >= 0; i--) {
    const ch = trimmed.charCodeAt(i);
    if (ch === 41) {
      depth++;
    } else if (ch === 40) {
      depth--;
      if (depth === 0) {
        argOpenIdx = i;
        break;
      }
    }
  }
  if (argOpenIdx <= 0 || depth !== 0) return null;

  const symbol = trimmed.slice(0, argOpenIdx).trim();
  if (
    !symbol ||
    !symbol.includes(".") ||
    symbol.includes(":") ||
    /\s/.test(symbol)
  ) {
    return null;
  }

  const lastSlash = symbol.lastIndexOf("/");
  const shortName = lastSlash >= 0 ? symbol.slice(lastSlash + 1) : symbol;
  if (
    shortName.length < 2 ||
    shortName.length > 80 ||
    !/^[A-Za-z0-9_.*()[\]]+$/.test(shortName)
  ) {
    return null;
  }
  return shortName;
}

function isGoFuncCallCandidate(trimmed: string): boolean {
  if (
    !trimmed ||
    FAIL_HEADER_RE.test(trimmed) ||
    ERROR_DIAG_RE.test(trimmed) ||
    APP_STACK_FRAME_RE.test(trimmed) ||
    /\.\w+:\d+/.test(trimmed)
  ) {
    return false;
  }
  return extractShortGoFuncName(trimmed) !== null;
}

/**
 * Normalizes a bare `<path>:<line>` stack frame by removing trailing hex PC
 * offsets and trimming absolute prefixes down to a known relative source root
 * (e.g. `/abs/repo/pkg/demo/srv.go:42 +0x9a` -> `pkg/demo/srv.go:42`).
 */
function isBarePathFrameLine(line: string): boolean {
  const noHex = stripTrailingHexOffset(line).trim();
  if (!noHex || noHex.includes("://") || /\s/.test(noHex) || !noHex.includes("/")) {
    return false;
  }
  const pathOnly = stripLineColSuffix(noHex);
  return pathOnly.length < noHex.length && FILE_EXT_RE.test(pathOnly);
}

function normalizeFramePathLine(frameLine: string): string {
  let clean = stripTrailingHexOffset(frameLine).trim().replace(/\\/g, "/");
  if (clean.startsWith("file://")) {
    clean = clean.slice("file://".length);
  }
  if (clean.startsWith("/") || /^[A-Za-z]:\//.test(clean)) {
    let bestStart = -1;
    for (const prefix of SOURCE_ROOT_PREFIXES) {
      const idx = clean.indexOf(`/${prefix}`);
      if (idx >= 0) {
        const start = idx + 1;
        if (bestStart === -1 || start < bestStart) {
          bestStart = start;
        }
      }
    }
    if (bestStart >= 0) {
      clean = clean.slice(bestStart);
    }
  }
  return clean;
}

function computeVendorDropMask(rawLines: readonly string[]): {
  drop: Uint8Array;
  strippedLines: number;
} {
  const n = rawLines.length;
  const drop = new Uint8Array(n);
  let strippedLines = 0;

  for (let i = 0; i < n; i++) {
    const trimmed = rawLines[i]!.trim();
    if (trimmed.length > 0 && VENDOR_STACK_FRAME_RE.test(trimmed)) {
      drop[i] = 1;
      strippedLines++;
    }
  }

  // Pair Go vendor/runtime function lines (line i) with their vendor file lines (line i+1)
  for (let i = 0; i + 1 < n; i++) {
    if (drop[i] === 0 && drop[i + 1] === 1) {
      const trimmed = rawLines[i]!.trim();
      if (isGoFuncCallCandidate(trimmed)) {
        drop[i] = 1;
        strippedLines++;
      }
    }
  }

  return { drop, strippedLines };
}

/**
 * Losslessly strips vendor/runtime stack-frame lines (`node_modules/`,
 * `node:internal/`, `vendor/`, `site-packages/`, `dist-packages/`, Go runtime
 * & testing stdlib pairs) from command outputs in strictly linear O(N) time.
 *
 * Preserves all application frames, test lines, error messages, assertions,
 * and headers untouched without inserting `[pruned` or truncation markers.
 */
export function stripVendorStackFrames(output: string): {
  text: string;
  strippedLines: number;
} {
  if (!output) return { text: output, strippedLines: 0 };
  const rawLines = output.split("\n");
  const { drop, strippedLines } = computeVendorDropMask(rawLines);
  if (strippedLines === 0) {
    return { text: output, strippedLines: 0 };
  }

  const kept: string[] = [];
  let prevWasDropped = false;
  let prevKeptWasBlank = false;

  for (let i = 0; i < rawLines.length; i++) {
    if (drop[i] === 1) {
      prevWasDropped = true;
      continue;
    }
    const line = rawLines[i]!;
    const isBlank = line.trim().length === 0;
    if (isBlank && prevWasDropped && prevKeptWasBlank) {
      continue;
    }
    kept.push(line);
    prevKeptWasBlank = isBlank;
    prevWasDropped = false;
  }

  if (prevWasDropped) {
    while (kept.length > 0 && kept[kept.length - 1]!.trim().length === 0) {
      kept.pop();
    }
  }

  return { text: kept.join("\n"), strippedLines };
}

function packLinesSequentially(lines: readonly string[], maxChars: number): string {
  const nonEmpty = lines.map((l) => trimHorizontalEnd(l)).filter((l) => l.trim().length > 0);
  if (nonEmpty.length === 0) return "";
  const full = nonEmpty.join("\n");
  if (full.length <= maxChars) return full;

  const picked: string[] = [];
  let used = 0;
  for (const line of nonEmpty) {
    const sep = picked.length > 0 ? 1 : 0;
    const rem = maxChars - used - sep;
    if (rem <= 0) break;
    if (line.length <= rem) {
      picked.push(line);
      used += sep + line.length;
    } else if (picked.length === 0 || rem >= 28) {
      const cut = truncateAtWordBoundary(line, rem);
      if (cut.length > 0) {
        picked.push(cut);
        used += sep + cut.length;
      }
      break;
    } else {
      break;
    }
  }
  const out = picked.join("\n");
  return out.length <= maxChars ? out : truncateAtWordBoundary(out, maxChars);
}

/**
 * Produces a diagnostic-preserving, vendor-noise-free excerpt of a command or
 * test runner output.
 *
 * When `ok === false`, strips `node_modules/` and `node:internal/` stack frames
 * and prioritizes:
 *   1. Failing suite/test header (`FAIL ...`, `✕ ...`, `FAILED ...`, `--- FAIL: ...`)
 *   2. Primary error diagnostics (`Error ...`, `AssertionError ...`, `ERR_...`, `Expected/Received`)
 *   3. First application-code stack frame (`at ... (src/...:line)`)
 *
 * When `ok === true`, emits a concise 1-line pass summary (`<= Math.min(maxChars, 120)`).
 */
export function compactCommandExcerpt(
  output: string,
  ok: boolean,
  maxChars: number,
): string {
  if (!Number.isFinite(maxChars) || maxChars <= 0) return "";
  const cap = Math.floor(maxChars);
  const clean = sanitizeNeutralText(output);
  if (!clean) return "";

  const allLines = clean
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (allLines.length === 0) return "";

  if (ok) {
    const passCap = Math.min(cap, 120);
    // Prefer an explicit pass/summary line (often near the bottom of test runner output)
    const passMatches = allLines.filter(
      (l) => !VENDOR_STACK_FRAME_RE.test(l) && PASS_SUMMARY_RE.test(l),
    );
    if (passMatches.length > 0) {
      return truncateAtWordBoundary(passMatches[passMatches.length - 1]!, passCap);
    }
    const meaningful = allLines.filter((l) => !/^>\s+/.test(l) && !VENDOR_STACK_FRAME_RE.test(l));
    const chosen = meaningful[0] ?? allLines[0]!;
    return truncateAtWordBoundary(chosen, passCap);
  }

  // Failure path: drop noisy vendor/runtime stack frames (including Go func+file pairs) first
  const { drop: vendorDrop } = computeVendorDropMask(allLines);
  const nonVendorLines: string[] = [];
  for (let i = 0; i < allLines.length; i++) {
    if (vendorDrop[i] === 0) nonVendorLines.push(allLines[i]!);
  }
  const workingLines = nonVendorLines.length > 0 ? nonVendorLines : allLines;

  const headers: string[] = [];
  const runnerBanners: string[] = [];
  const diagnostics: string[] = [];
  const appFrames: Array<{ text: string; idx: number }> = [];
  const fallback: string[] = [];

  for (let lineIdx = 0; lineIdx < workingLines.length; lineIdx++) {
    const line = workingLines[lineIdx]!;
    const isRedundantSummary = REDUNDANT_CMD_SUMMARY_RE.test(line);
    if (!isRedundantSummary && RUNNER_FILE_BANNER_RE.test(line)) {
      runnerBanners.push(line);
      continue;
    }

    const isExplicitStackLine = /^\s*(?:at\s+\S|File\s+"[^"\n]+",\s+line\s+\d+)/.test(line);
    const isHeader = FAIL_HEADER_RE.test(line) && !isRedundantSummary;
    const hasCitedPathWithLine =
      !isRedundantSummary && /:\d+/.test(line) && extractAppFileReferences(line).length > 0;
    const isFrame =
      !isRedundantSummary &&
      (APP_STACK_FRAME_RE.test(line) || hasCitedPathWithLine || isBarePathFrameLine(line));
    const diagProbe = line
      .replace(LEADING_ARROW_RE, "")
      .replace(LEADING_TEST_FILE_PREFIX_RE, "");
    const isDiag =
      !isRedundantSummary && (ERROR_DIAG_RE.test(line) || ERROR_DIAG_RE.test(diagProbe));

    if (isExplicitStackLine && isFrame) {
      appFrames.push({ text: line, idx: lineIdx });
    } else if (isHeader) {
      // Check test/suite headers BEFORE diagnostics so test filenames like
      // `FAIL src/error-handler.test.ts` never displace real Error diagnostics.
      const cleanHeader = line.replace(GO_FAIL_TIMING_RE, "$1");
      headers.push(cleanHeader);
      if (isFrame && /\.\w+:\d+/.test(line)) {
        appFrames.push({ text: line, idx: lineIdx });
      }
    } else if (isDiag) {
      diagnostics.push(line.replace(LEADING_ARROW_RE, ""));
    } else if (isFrame) {
      appFrames.push({ text: line, idx: lineIdx });
    } else if (!isRedundantSummary) {
      fallback.push(line);
    }
  }

  // Strictly prefer a non-test application source frame (e.g. worker.go:118, jwt.ts:142)
  // over test harness frames (*_test.go, *.test.ts, *.spec.ts, test_*.py)
  const nonTestFrameEntry = appFrames.find((f) => !TEST_FILE_FRAME_RE.test(f.text));
  const primaryFrameEntry = nonTestFrameEntry ?? appFrames[0];
  let primaryFrame: string | undefined;
  if (primaryFrameEntry) {
    const rawText = primaryFrameEntry.text;
    const isBarePathFrame = !/^\s*(?:at\s+\S|File\s+"[^"\n]+",\s+line\s+\d+)/.test(rawText);
    if (isBarePathFrame) {
      const cleanPathLine = normalizeFramePathLine(rawText);
      const prevLine = primaryFrameEntry.idx > 0 ? workingLines[primaryFrameEntry.idx - 1] : undefined;
      const shortFunc =
        prevLine &&
        !VENDOR_STACK_FRAME_RE.test(prevLine) &&
        !FAIL_HEADER_RE.test(prevLine) &&
        !ERROR_DIAG_RE.test(prevLine) &&
        !/\.\w+:\d+/.test(prevLine)
          ? extractShortGoFuncName(prevLine)
          : null;
      primaryFrame = shortFunc ? `at ${shortFunc} (${cleanPathLine})` : cleanPathLine;
    } else {
      primaryFrame = stripTrailingHexOffset(rawText);
    }
  }
  const hasNonTestFrame = Boolean(nonTestFrameEntry);

  const primaryHeader = headers[0];
  let primaryDiag = diagnostics[0];
  if (primaryDiag && hasNonTestFrame && LEADING_TEST_FILE_PREFIX_RE.test(primaryDiag)) {
    const strippedDiag = primaryDiag.replace(LEADING_TEST_FILE_PREFIX_RE, "").trim();
    if (strippedDiag.length > 0) {
      primaryDiag = strippedDiag;
    }
  }

  // If no structured failure signals were found, fall back cleanly to non-summary lines
  if (!primaryHeader && !primaryDiag && !primaryFrame) {
    const pool =
      fallback.length > 0 ? fallback : runnerBanners.length > 0 ? runnerBanners : workingLines;
    return packLinesSequentially(pool, cap);
  }

  // Order parts so Error diagnostic + non-test implementation source are NEVER obscured
  // behind a `*.test.ts` / `*_test.go` path; keep named non-file headers (e.g. `--- FAIL: TestFoo`)
  // at the front and place test-file headers after diagnostic + app frame when space allows.
  const headerMentionsTestFile = Boolean(primaryHeader && TEST_FILE_FRAME_RE.test(primaryHeader));
  const leadHeader = primaryHeader && !headerMentionsTestFile ? primaryHeader : undefined;
  const trailingHeader =
    primaryHeader && headerMentionsTestFile && primaryDiag ? primaryHeader : undefined;
  const fallbackOnlyHeader =
    primaryHeader && headerMentionsTestFile && !primaryDiag ? primaryHeader : undefined;

  const essential: string[] = [];
  const pushUnique = (list: string[], value: string | undefined) => {
    if (value && !list.includes(value)) list.push(value);
  };

  pushUnique(essential, leadHeader ?? fallbackOnlyHeader);
  pushUnique(essential, primaryDiag);
  pushUnique(essential, primaryFrame);
  pushUnique(essential, trailingHeader);

  const joinParts = (parts: readonly string[]) => parts.join("\n");

  if (joinParts(essential).length <= cap) {
    // Everything essential fits! Greedily include secondary diagnostics (e.g. Expected / Received)
    const chosen: string[] = [];
    pushUnique(chosen, leadHeader ?? fallbackOnlyHeader);
    pushUnique(chosen, primaryDiag);

    const tailItems: string[] = [];
    pushUnique(tailItems, primaryFrame);
    pushUnique(tailItems, trailingHeader);
    const tailReserve =
      tailItems.filter((t) => !chosen.includes(t)).reduce((acc, t) => acc + t.length + 1, 0);
    let currentLen = joinParts(chosen).length;

    for (let i = 1; i < diagnostics.length && i < 4; i++) {
      let extra = diagnostics[i]!;
      if (hasNonTestFrame && LEADING_TEST_FILE_PREFIX_RE.test(extra)) {
        const cleaned = extra.replace(LEADING_TEST_FILE_PREFIX_RE, "").trim();
        if (cleaned) extra = cleaned;
      }
      if (chosen.includes(extra) || tailItems.includes(extra)) continue;
      const nextLen = currentLen + (chosen.length > 0 ? 1 : 0) + extra.length;
      if (nextLen + tailReserve <= cap) {
        chosen.push(extra);
        currentLen = nextLen;
      }
    }

    for (const t of tailItems) pushUnique(chosen, t);
    return joinParts(chosen);
  }

  // Tight budget: protect primaryDiag and primaryFrame first; shrink or drop header if needed
  const coreWithoutHeader: string[] = [];
  pushUnique(coreWithoutHeader, primaryDiag);
  pushUnique(coreWithoutHeader, primaryFrame);

  if (primaryHeader && coreWithoutHeader.length > 0) {
    const coreLen = joinParts(coreWithoutHeader).length;
    const remForHeader = cap - coreLen - 1;
    if (remForHeader >= 28) {
      const shortHeader = truncateAtWordBoundary(primaryHeader, remForHeader);
      return headerMentionsTestFile
        ? joinParts([...coreWithoutHeader, shortHeader])
        : joinParts([shortHeader, ...coreWithoutHeader]);
    }
    if (coreLen <= cap) {
      return joinParts(coreWithoutHeader);
    }
  }

  // Even coreWithoutHeader exceeds cap: split cleanly between diag and frame at word boundaries
  if (primaryDiag && primaryFrame && primaryFrame !== primaryDiag) {
    const frameBudget = Math.min(primaryFrame.length, Math.max(36, Math.floor(cap * 0.4)));
    const diagBudget = cap - frameBudget - 1;
    if (diagBudget >= 24) {
      return joinParts([
        truncateAtWordBoundary(primaryDiag, diagBudget),
        truncateAtWordBoundary(primaryFrame, frameBudget),
      ]);
    }
  }

  const single = primaryDiag ?? primaryFrame ?? primaryHeader ?? workingLines[0]!;
  return truncateAtWordBoundary(single, cap);
}
