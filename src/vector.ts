import { createHash } from "node:crypto";
import type { NodeKind } from "./types.ts";

export const EMBEDDING_DIM = 128 as const;

const STOPWORDS = new Set([
  "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
  "have", "has", "had", "not", "but", "you", "all", "can", "her", "his",
  "one", "our", "out", "day", "get", "use", "into", "than", "then", "them",
  "these", "those", "will", "would", "should", "could", "const", "let", "var",
  "return", "true", "false", "null", "undefined", "void", "import", "from",
]);

/**
 * Deterministic 32-bit FNV-1a hash with configurable seed.
 */
export function fnv1a32(input: string, seed = 0x811c9dc5): number {
  let hash = seed >>> 0;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/**
 * Splits text into BPE-like lexical + punctuation + CJK tokens and maps each
 * token deterministically to a 31-bit integer token ID for exact KV-Cache LCP.
 */
const TOKEN_REGEX = /[A-Za-z_][A-Za-z0-9_]{0,15}|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\sA-Za-z0-9_\u4e00-\u9fff]|\n/g;

export function tokenizeToIds(text: string): number[] {
  if (!text) return [];
  const matches = text.match(TOKEN_REGEX);
  if (!matches) return [];
  const ids = new Array<number>(matches.length);
  for (let i = 0; i < matches.length; i++) {
    ids[i] = (fnv1a32(matches[i]!, 0x9e3779b9) & 0x7fffffff) || 1;
  }
  return ids;
}

export function countTokens(text: string): number {
  if (!text) return 0;
  const matches = text.match(TOKEN_REGEX);
  return matches ? matches.length : 0;
}

/**
 * Computes exact Longest Common Prefix (LCP) length between two token ID arrays,
 * faithfully modeling causal Transformer Prefix KV-Cache reuse.
 */
export function longestCommonPrefixLength(prevIds: readonly number[], currIds: readonly number[]): number {
  const limit = Math.min(prevIds.length, currIds.length);
  let i = 0;
  while (i < limit && prevIds[i] === currIds[i]) {
    i++;
  }
  return i;
}

function splitIdentifierSubtokens(word: string): string[] {
  const parts = word
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_\-./:]+/g, " ")
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length >= 2);
  return parts;
}

/**
 * Deterministic 128-dim L2-normalized semantic vector using signed feature hashing
 * across word unigrams, identifier sub-tokens, word bigrams, resource keys, and char 4-grams.
 */
export function embedText(
  text: string,
  resourceKey?: string | null,
  dim: number = EMBEDDING_DIM,
): number[] {
  const vec = new Float64Array(dim);
  const tf = new Map<string, number>();

  const addFeature = (feat: string, weight: number) => {
    tf.set(feat, (tf.get(feat) ?? 0) + weight);
  };

  if (resourceKey) {
    const cleanKey = resourceKey.toLowerCase().trim();
    addFeature(`rk:${cleanKey}`, 3.5);
    for (const seg of splitIdentifierSubtokens(cleanKey)) {
      addFeature(`rks:${seg}`, 2.2);
    }
  }

  const rawWords = text.match(/[A-Za-z_][A-Za-z0-9_./:-]{1,64}|\d+|[\u4e00-\u9fff]{1,4}/g) ?? [];
  const normStream: string[] = [];

  for (const raw of rawWords) {
    const lower = raw.toLowerCase();
    const isStop = STOPWORDS.has(lower);
    addFeature(`w:${lower}`, isStop ? 0.15 : 1.0);
    if (!isStop) {
      normStream.push(lower);
    }
    const subtokens = splitIdentifierSubtokens(raw);
    for (const sub of subtokens) {
      if (!STOPWORDS.has(sub)) {
        addFeature(`id:${sub}`, 1.25);
        if (sub !== lower) normStream.push(sub);
      }
    }
    if (lower.length >= 5 && !isStop) {
      for (let i = 0; i <= Math.min(lower.length - 4, 12); i += 2) {
        addFeature(`c4:${lower.slice(i, i + 4)}`, 0.35);
      }
    }
  }

  for (let i = 0; i + 1 < normStream.length && i < 512; i++) {
    addFeature(`bg:${normStream[i]}_${normStream[i + 1]}`, 0.85);
  }

  for (const [feat, rawWeight] of tf) {
    const scaled = rawWeight > 1 ? (1 + Math.log(rawWeight)) : rawWeight;
    const h1 = fnv1a32(feat, 0x811c9dc5);
    const h2 = fnv1a32(feat, 0x27d4eb2d);
    const idx = h1 % dim;
    const sign = (h2 & 1) === 0 ? 1 : -1;
    vec[idx] += sign * scaled;
  }

  let normSq = 0;
  for (let i = 0; i < dim; i++) {
    normSq += vec[i]! * vec[i]!;
  }
  if (normSq > 1e-12) {
    const inv = 1 / Math.sqrt(normSq);
    for (let i = 0; i < dim; i++) {
      vec[i] = Math.round(vec[i]! * inv * 1e6) / 1e6;
    }
  }

  return Array.from(vec);
}

export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    const va = a[i]!;
    const vb = b[i]!;
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA * normB);
  return denom > 1e-12 ? dot / denom : 0;
}

/**
 * Deterministic multi-LOD extractive summary generator for code artifacts,
 * decisions, constraints, and verbose tool observations.
 */
export function deriveSummary(
  content: string,
  kind: NodeKind,
  resourceKey?: string | null,
): string {
  const trimmed = content.trim();
  if (!trimmed) return "";

  const lines = trimmed
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (trimmed.length <= 180 && lines.length <= 3) {
    return trimmed;
  }

  if (kind === "artifact" || (resourceKey && resourceKey.startsWith("file:"))) {
    const sigLines = lines.filter((l) =>
      /^(export\s+|function\s+|async\s+function\s+|interface\s+|type\s+|class\s+|const\s+[A-Z0-9_]+\s*=)/.test(l),
    );
    const header = resourceKey ? `[${resourceKey}]` : "[artifact]";
    if (sigLines.length > 0) {
      const picked = sigLines.slice(0, 5).map((s) => s.replace(/\{$/, "").trim());
      return `${header} Signatures: ${picked.join(" | ").slice(0, 320)}`;
    }
  }

  if (kind === "observation") {
    const important = lines.filter((l) =>
      /(error|fail|pass|ok|warn|exception|status|summary|result|timeout)/i.test(l),
    );
    const chosen = (important.length > 0 ? important : lines).slice(0, 3);
    return chosen.join(" ; ").slice(0, 260);
  }

  const head = lines.slice(0, 3).join(" ");
  return head.length > 260 ? `${head.slice(0, 257)}...` : head;
}

export function deriveStub(
  id: string,
  kind: NodeKind,
  resourceKey: string | null,
  version: number,
  summary: string,
): string {
  const headline = summary.replace(/\s+/g, " ").trim().slice(0, 88);
  const resTag = resourceKey ? ` ${resourceKey}#v${version}` : "";
  return `[stub ${id} (${kind}${resTag})]: ${headline}`;
}

export function deterministicId(prefix: string, seed: string): string {
  const digest = createHash("sha256").update(seed).digest("hex").slice(0, 12);
  return `${prefix}_${digest}`;
}
