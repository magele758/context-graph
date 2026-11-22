import type { GraphStore } from "./graph-store.ts";
import type {
  AgentLoopCompatibleMessage,
  ChatMessage,
  CompiledNodeEntry,
  CompileOptions,
  CompileResult,
  ConflictWarning,
  ExcludedNodeEntry,
  GraphEdge,
  GraphNode,
  LodLevel,
  NodeKind,
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

export function compileContext(store: GraphStore, options: CompileOptions): CompileResult {
  const snapshot = store.getGraphSnapshot(options.graphId);
  const { graph, nodes, edges } = snapshot;

  const focusQuery = options.focusQuery?.trim() ?? "";
  const activeScope = options.activeScope?.trim() || "global";
  const maxTokens =
    typeof options.maxTokens === "number" && options.maxTokens >= 64
      ? Math.floor(options.maxTokens)
      : graph.defaultBudget;
  const zone1MaxRatio =
    typeof options.zone1MaxRatio === "number" && options.zone1MaxRatio > 0 && options.zone1MaxRatio < 0.9
      ? options.zone1MaxRatio
      : 0.45;
  const pprAlpha = options.pprAlpha ?? 0.85;
  const pprIterations = options.pprIterations ?? 20;

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

  const utilityOf = (n: GraphNode): number => {
    const base = pprScoreById.get(n.id) ?? 0.001;
    const isLatestUser = n.turn === maxTurn && n.role === "user";
    return base * KIND_UTILITY_MULTIPLIER[n.kind] * (isLatestUser ? 2.5 : 1.0);
  };

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
      const baseLod: LodLevel =
        isCurrentPrompt && getRendered(member, "full").tokens <= availableZone2Budget - usedZone2Tokens
          ? "full"
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
        node.kind === "artifact"
          ? 1.75
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
      if (entry.activationScore < lowestScore) {
        lowestScore = entry.activationScore;
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
