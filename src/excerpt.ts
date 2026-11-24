/**
 * Declaration-aware code excerpting, application-frame-preserving command
 * diagnostics, and word-boundary-safe text sanitization.
 *
 * Pure zero-dependency helpers designed to avoid character-level 50/50
 * head/tail slicing that corrupts identifiers, numeric literals, line numbers,
 * and source stack frames.
 */

const PRUNED_LINE_RE =
  /^[ \t]*(?:\u2026|\.{3})?[ \t]*\[(?:pruned|truncated)\b[^\]]*\][ \t]*$/gim;
const PRUNED_INLINE_RE =
  /[ \t]*(?:\u2026|\.{3})?[ \t]*\[(?:pruned|truncated)\b[^\]]*\][ \t]*/gi;
const HEAD_TAIL_TAG_RE =
  /[ \t]*\[[^\]]*head\/tail\s+shown[^\]]*\][ \t]*/gi;

const BOILERPLATE_ACK_LINE_RE =
  /^(?:successfully\s+)?(?:edited|updated|wrote|written|created|patched|modified|saved|applied\s+patch(?:\s+to)?|file\s+(?:updated|edited|written|saved))\b[^\n]*$/i;

const VENDOR_STACK_FRAME_RE =
  /(?:node_modules[\\/]|node:internal[\\/]|internal[\\/]modules[\\/]|\(node:[a-z0-9_/-]+:\d+|\bat\s+node:[a-z0-9_/-]+:\d+|\(internal\/[a-z0-9_/-]+:\d+)/i;

const REDUNDANT_CMD_SUMMARY_RE =
  /^(?:tests?\s+\d+\s+(?:failed|passed)|test\s+files?\s+\d+\s+|command\s+exited\s+with\s+code\s+\d+|exit\s+code[:\s]+\d+|duration\s+\d|start\s+at\s+\d|transform\s+\d|npm\s+(?:err!|error)\b|elifecycle\b|>\s+\S+@\S+\s+|>\s+(?:vitest|jest|mocha|node|tsc|pytest|go\s+test|cargo\s+test)\b)/i;

const FAIL_HEADER_RE =
  /^(?:FAIL\b|FAILED\b|---\s*FAIL:|not\s+ok\s+\d|[\u2715\u2717\u2716\u00d7\u25cf\u276f]\s+\S|FAIL\s+\S)/;

const ERROR_DIAG_RE =
  /(?:\b(?:AssertionError|TypeError|SyntaxError|ReferenceError|RangeError|URIError|EvalError|AggregateError|Error)\b|\bERR_[A-Z0-9_]+\b|\bE(?:CONNREFUSED|CONNRESET|TIMEDOUT|NOENT|ACCES|ADDRINUSE|PIPE|PERM|EXIST|NOTDIR|ISDIR)\b|^(?:Expected|Received|Actual|Diff|Message|Cause|panic|Traceback|Exception|Assertion\s+failed)\s*[:(]|\berror\s+TS\d+:|\berror\[E\d+\]:)/i;

const APP_STACK_FRAME_RE =
  /(?:^\s*at\s+.*(?::\d+(?::\d+)?\)?$)|^\s*File\s+"[^"]+",\s+line\s+\d+|(?:\b(?:src|packages|apps|lib|server|client|api|cmd|pkg|internal|tests?|spec|bin)\/[^\s):'"]+\.\w+:\d+(?::\d+)?)|\([^\s)]+\.\w+:\d+(?::\d+)?\))/i;

const PASS_SUMMARY_RE =
  /(?:\b\d+\s+(?:passed|passing|tests?\s+passed|specs?\s+passed|succeeded)\b|\bpass\s+\d+|\bPASS\b|\ball\s+\d+.*pass|\b0\s+errors?\b|\bbuild\s+succeeded\b|\bcompiled\s+successfully\b|^ok\s+\S+)/i;

const CLOSING_ONLY_RE = /^[}\])];?,?$/;

const IMPORT_LINE_RE =
  /^(?:import\b|from\s+\S+\s+import\b|export\s+\*\s+from\b|const\s+\S+\s*=\s*require\(|#include\b|using\s+\S+;|use\s+[A-Za-z0-9_:]+;)/;

const TOP_LEVEL_DECL_RE =
  /^(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(?:const|let|var|type|interface|class|function|enum)\b/;

const MULTILANG_DECL_RE =
  /^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:def|class|struct|enum|trait|impl|fn|func|type|const|static|let)\b/;

const ENV_OR_INI_KEY_RE =
  /^(?:[A-Z][A-Z0-9_.-]*\s*[:=]|\[[A-Za-z0-9_.-]+\]$)/;

const JSON_OR_YAML_KEY_RE =
  /^(?:"[A-Za-z0-9_.-]+"\s*:|[A-Za-z_][A-Za-z0-9_.-]*\s*:\s*\S)/;

const NON_KV_CONTROL_RE = /^(?:case\b|default\s*:|https?:)/i;

const LINE_NUMBER_GUTTER_RE = /^\s*\d+\s*(?:\||->|\u2192|\t|:\s)/;

/**
 * Strips upstream truncation / pruning markers (e.g. `[pruned 1234 chars; head/tail shown]`)
 * and normalizes line endings & whitespace without destroying code indentation.
 */
export function sanitizeNeutralText(text: string): string {
  if (!text) return "";
  let s = text.replace(/\r\n?/g, "\n");
  // Drop standalone marker lines completely
  s = s.replace(PRUNED_LINE_RE, "");
  // Drop inline markers, preserving a single space only when flanked by non-whitespace
  s = s.replace(PRUNED_INLINE_RE, (match, offset: number, full: string) => {
    const before = offset > 0 ? full[offset - 1]! : "";
    const afterIdx = offset + match.length;
    const after = afterIdx < full.length ? full[afterIdx]! : "";
    if (before && after && !/\s/.test(before) && !/\s/.test(after)) {
      return " ";
    }
    if (before && after && !/\n/.test(before) && !/\n/.test(after) && /\s/.test(match)) {
      return " ";
    }
    return "";
  });
  s = s.replace(HEAD_TAIL_TAG_RE, (match, offset: number, full: string) => {
    const before = offset > 0 ? full[offset - 1]! : "";
    const afterIdx = offset + match.length;
    const after = afterIdx < full.length ? full[afterIdx]! : "";
    if (before && after && !/\s/.test(before) && !/\s/.test(after)) {
      return " ";
    }
    return "";
  });
  // Trim trailing horizontal spaces per line and collapse excessive blank runs
  s = s
    .split("\n")
    .map((line) => line.replace(/[ \t]+$/g, ""))
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
  if (maxChars <= 0) return "";
  const s = text.trim();
  if (s.length <= maxChars) return s;

  if (maxChars < 4) {
    return s.slice(0, maxChars);
  }
  const suffix = maxChars >= 8 ? " ..." : "...";
  const limit = maxChars - suffix.length;
  if (limit <= 0) {
    return s.slice(0, maxChars);
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

function classifyCodeLinePriority(rawLine: string, idx: number, hasGutter: boolean): number {
  const unnumbered = stripOptionalLineNumber(rawLine, hasGutter);
  const trimmed = unnumbered.trim();
  if (!trimmed || CLOSING_ONLY_RE.test(trimmed)) {
    return 4;
  }

  const indentMatch = unnumbered.match(/^(\s*)/);
  const indent = indentMatch ? indentMatch[1]!.length : 0;

  // Leading module header comment
  if (
    idx < 3 &&
    trimmed.length <= 120 &&
    /^(?:\/\/|\/\*|\*|--|#(?!include\b)|""")/.test(trimmed)
  ) {
    return 0;
  }

  // Imports go below exported declarations & constants so long import blocks never starve symbols
  if (IMPORT_LINE_RE.test(trimmed)) {
    return 3;
  }

  // Top-level declarations, constants, function/class/interface signatures
  if (indent <= 2 && (TOP_LEVEL_DECL_RE.test(trimmed) || MULTILANG_DECL_RE.test(trimmed))) {
    return 1;
  }

  // Config / env / INI assignments
  if (indent <= 2 && ENV_OR_INI_KEY_RE.test(trimmed)) {
    return 1;
  }

  // Shallow JSON / YAML / object config key-value lines
  if (indent <= 4 && JSON_OR_YAML_KEY_RE.test(trimmed) && !NON_KV_CONTROL_RE.test(trimmed)) {
    return 1;
  }

  // Other meaningful early head/body lines
  return 2;
}

function stripBoilerplateAcks(text: string): string {
  const lines = text.split("\n");
  const kept: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (BOILERPLATE_ACK_LINE_RE.test(trimmed)) continue;
    kept.push(line.replace(/[ \t]+$/g, ""));
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
  if (maxChars <= 0) return "";

  if (opts?.diffText && opts.diffText.trim().length > 0) {
    const cleanDiff = stripBoilerplateAcks(sanitizeNeutralText(opts.diffText));
    if (cleanDiff.length > 0) {
      return packLinesSequentially(cleanDiff.split("\n"), maxChars);
    }
  }

  const clean = sanitizeNeutralText(content);
  if (!clean) return "";
  if (clean.length <= maxChars) {
    return clean;
  }

  const rawLines = clean
    .split("\n")
    .map((l) => l.replace(/[ \t]+$/g, ""))
    .filter((l) => l.trim().length > 0);
  if (rawLines.length === 0) return "";

  const hasGutter = detectLineNumberGutter(rawLines);
  const classified = rawLines.map((line, idx) => ({
    idx,
    text: hasGutter ? line.replace(/^\s+/, "") : line,
    priority: classifyCodeLinePriority(line, idx, hasGutter),
  }));

  // Pack in priority order (P0 header comment max 1 -> P1 declarations -> P2 early lines -> P3 imports)
  const selected = new Map<number, string>();
  let usedChars = 0;

  const tryAdd = (idx: number, text: string, allowTruncateSingleLine: boolean): boolean => {
    if (selected.has(idx)) return false;
    const sepLen = selected.size > 0 ? 1 : 0;
    const remaining = maxChars - usedChars - sepLen;
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

  // P0: at most 1 leading module header comment (only if budget leaves ample room for declarations)
  const p0 = classified.filter((c) => c.priority === 0);
  if (p0.length > 0 && p0[0]!.text.length <= Math.floor(maxChars * 0.35)) {
    tryAdd(p0[0]!.idx, p0[0]!.text, false);
  }

  // P1: all fitting top-level declarations, constants, signatures, config keys in source order
  const p1 = classified.filter((c) => c.priority === 1);
  for (const item of p1) {
    if (!tryAdd(item.idx, item.text, selected.size === 0)) {
      // If a long declaration didn't fit whole, allow clean word-boundary truncation when enough room exists
      if (maxChars - usedChars >= 48) {
        tryAdd(item.idx, item.text, true);
      }
    }
  }

  // P2: meaningful head/body lines in source order
  const p2 = classified.filter((c) => c.priority === 2);
  for (const item of p2) {
    if (maxChars - usedChars < 16) break;
    if (!tryAdd(item.idx, item.text, selected.size === 0 || maxChars - usedChars >= 48)) {
      if (maxChars - usedChars < 48) break;
    }
  }

  // P3: imports only if space still remains
  const p3 = classified.filter((c) => c.priority === 3);
  for (const item of p3) {
    if (maxChars - usedChars < 24) break;
    tryAdd(item.idx, item.text, false);
  }

  if (selected.size === 0) {
    return truncateAtWordBoundary(classified[0]!.text, maxChars);
  }

  // Emit in original source line order
  const ordered = Array.from(selected.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, text]) => text);

  const joined = ordered.join("\n");
  return joined.length <= maxChars ? joined : truncateAtWordBoundary(joined, maxChars);
}

function packLinesSequentially(lines: readonly string[], maxChars: number): string {
  const nonEmpty = lines.map((l) => l.replace(/[ \t]+$/g, "")).filter((l) => l.trim().length > 0);
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
  if (maxChars <= 0) return "";
  const clean = sanitizeNeutralText(output);
  if (!clean) return "";

  const allLines = clean
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (allLines.length === 0) return "";

  if (ok) {
    const cap = Math.min(maxChars, 120);
    // Prefer an explicit pass/summary line (often near the bottom of test runner output)
    const passMatches = allLines.filter(
      (l) => !VENDOR_STACK_FRAME_RE.test(l) && PASS_SUMMARY_RE.test(l),
    );
    if (passMatches.length > 0) {
      return truncateAtWordBoundary(passMatches[passMatches.length - 1]!, cap);
    }
    const meaningful = allLines.filter((l) => !/^>\s+/.test(l) && !VENDOR_STACK_FRAME_RE.test(l));
    const chosen = meaningful[0] ?? allLines[0]!;
    return truncateAtWordBoundary(chosen, cap);
  }

  // Failure path: drop noisy vendor/runtime stack frames first
  const nonVendorLines = allLines.filter((l) => !VENDOR_STACK_FRAME_RE.test(l));
  const workingLines = nonVendorLines.length > 0 ? nonVendorLines : allLines;

  const headers: string[] = [];
  const diagnostics: string[] = [];
  const appFrames: string[] = [];
  const fallback: string[] = [];

  for (const line of workingLines) {
    const isRedundantSummary = REDUNDANT_CMD_SUMMARY_RE.test(line);
    const isExplicitStackLine = /^\s*(?:at\s+\S|File\s+"[^"]+",\s+line\s+\d+)/.test(line);
    const isHeader = FAIL_HEADER_RE.test(line) && !isRedundantSummary;
    const isFrame = APP_STACK_FRAME_RE.test(line) && !isRedundantSummary;
    const isDiag = ERROR_DIAG_RE.test(line) && !isRedundantSummary;

    if (isExplicitStackLine && isFrame) {
      appFrames.push(line);
    } else if (isHeader && !isDiag) {
      headers.push(line);
      if (isFrame && /\.\w+:\d+/.test(line)) {
        appFrames.push(line);
      }
    } else if (isDiag) {
      diagnostics.push(line);
    } else if (isFrame) {
      appFrames.push(line);
    } else if (!isRedundantSummary) {
      fallback.push(line);
    }
  }

  // Prefer a non-test application source frame when both app and test frames appear
  const primaryHeader = headers[0];
  const primaryDiag = diagnostics[0];
  const primaryFrame =
    appFrames.find((f) => !/\.(?:test|spec)\.[a-z0-9]+:\d+/i.test(f)) ?? appFrames[0];

  // If no structured failure signals were found, fall back cleanly to non-summary lines
  if (!primaryHeader && !primaryDiag && !primaryFrame) {
    const pool = fallback.length > 0 ? fallback : workingLines;
    return packLinesSequentially(pool, maxChars);
  }

  // Build priority-ordered bundle guaranteeing diagnostic + app source frame are never starved
  const essential: Array<{ role: "header" | "diag" | "frame"; text: string }> = [];
  if (primaryHeader) essential.push({ role: "header", text: primaryHeader });
  if (primaryDiag) essential.push({ role: "diag", text: primaryDiag });
  if (primaryFrame && primaryFrame !== primaryDiag && primaryFrame !== primaryHeader) {
    essential.push({ role: "frame", text: primaryFrame });
  }

  const joinParts = (parts: readonly string[]) => parts.join("\n");
  const essentialTexts = essential.map((e) => e.text);

  if (joinParts(essentialTexts).length <= maxChars) {
    // Everything essential fits! Greedily include secondary diagnostics (e.g. Expected / Received)
    const chosen: string[] = [];
    if (primaryHeader) chosen.push(primaryHeader);
    if (primaryDiag) chosen.push(primaryDiag);

    const frameReserve = primaryFrame && primaryFrame !== primaryDiag && primaryFrame !== primaryHeader
      ? primaryFrame.length + 1
      : 0;
    let currentLen = joinParts(chosen).length;

    for (let i = 1; i < diagnostics.length && i < 4; i++) {
      const extra = diagnostics[i]!;
      if (chosen.includes(extra) || extra === primaryFrame) continue;
      const nextLen = currentLen + (chosen.length > 0 ? 1 : 0) + extra.length;
      if (nextLen + frameReserve <= maxChars) {
        chosen.push(extra);
        currentLen = nextLen;
      }
    }

    if (primaryFrame && !chosen.includes(primaryFrame)) {
      chosen.push(primaryFrame);
    }
    return joinParts(chosen);
  }

  // Tight budget: protect primaryDiag and primaryFrame first; shrink or drop header if needed
  const coreWithoutHeader: string[] = [];
  if (primaryDiag) coreWithoutHeader.push(primaryDiag);
  if (primaryFrame && primaryFrame !== primaryDiag) coreWithoutHeader.push(primaryFrame);

  if (primaryHeader && coreWithoutHeader.length > 0) {
    const coreLen = joinParts(coreWithoutHeader).length;
    const remForHeader = maxChars - coreLen - 1;
    if (remForHeader >= 28) {
      const shortHeader = truncateAtWordBoundary(primaryHeader, remForHeader);
      return joinParts([shortHeader, ...coreWithoutHeader]);
    }
    if (coreLen <= maxChars) {
      return joinParts(coreWithoutHeader);
    }
  }

  // Even coreWithoutHeader exceeds maxChars: split cleanly between diag and frame at word boundaries
  if (primaryDiag && primaryFrame && primaryFrame !== primaryDiag) {
    const frameBudget = Math.min(primaryFrame.length, Math.max(36, Math.floor(maxChars * 0.4)));
    const diagBudget = maxChars - frameBudget - 1;
    if (diagBudget >= 24) {
      return joinParts([
        truncateAtWordBoundary(primaryDiag, diagBudget),
        truncateAtWordBoundary(primaryFrame, frameBudget),
      ]);
    }
  }

  const single = primaryDiag ?? primaryFrame ?? primaryHeader ?? workingLines[0]!;
  return truncateAtWordBoundary(single, maxChars);
}
