export const NODE_KINDS = [
  "system",
  "goal",
  "constraint",
  "decision",
  "task_frame",
  "artifact",
  "fact",
  "observation",
  "summary",
] as const;

export type NodeKind = (typeof NODE_KINDS)[number];

export const CONTEXT_TIERS = [
  "pinned_prefix",
  "stable_frame",
  "working_subgraph",
  "ephemeral_leaf",
] as const;

export type ContextTier = (typeof CONTEXT_TIERS)[number];

export const NODE_STATUSES = [
  "active",
  "superseded",
  "retracted",
] as const;

export type NodeStatus = (typeof NODE_STATUSES)[number];

export const EDGE_KINDS = [
  "depends_on",
  "supersedes",
  "summarizes",
  "elaborates",
  "scoped_to",
  "conflicts_with",
] as const;

export type EdgeKind = (typeof EDGE_KINDS)[number];

export const LOD_LEVELS = [
  "full",
  "summary",
  "stub",
] as const;

export type LodLevel = (typeof LOD_LEVELS)[number];

export const MESSAGE_ROLES = [
  "system",
  "user",
  "assistant",
  "tool",
] as const;

export type MessageRole = (typeof MESSAGE_ROLES)[number];

export interface GraphRecord {
  id: string;
  name: string;
  description: string;
  defaultBudget: number;
  compileCount: number;
  lastCompiledTokenIds: number[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface CreateGraphInput {
  id?: string;
  name: string;
  description?: string;
  defaultBudget?: number;
  metadata?: Record<string, unknown>;
}

export interface GraphNode {
  id: string;
  graphId: string;
  seq: number;
  turn: number;
  kind: NodeKind;
  role: MessageRole;
  scope: string;
  tier: ContextTier;
  content: string;
  summary: string;
  stub: string;
  tokens: number;
  summaryTokens: number;
  stubTokens: number;
  salience: number;
  status: NodeStatus;
  resourceKey: string | null;
  version: number;
  supersededBy: string | null;
  embedding: number[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface AddNodeInput {
  id?: string;
  turn?: number;
  kind: NodeKind;
  role?: MessageRole;
  scope?: string;
  tier?: ContextTier;
  content: string;
  summary?: string;
  stub?: string;
  salience?: number;
  status?: NodeStatus;
  resourceKey?: string | null;
  metadata?: Record<string, unknown>;
  dependsOn?: string[];
  scopedTo?: string[];
  summarizes?: string[];
  elaborates?: string[];
  conflictsWith?: string[];
}

export interface UpdateNodePatch {
  content?: string;
  summary?: string;
  stub?: string;
  tier?: ContextTier;
  scope?: string;
  salience?: number;
  status?: NodeStatus;
  metadata?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  graphId: string;
  sourceId: string;
  targetId: string;
  kind: EdgeKind;
  weight: number;
  metadata: Record<string, unknown>;
  createdAt: string;
}

export interface AddEdgeInput {
  id?: string;
  sourceId: string;
  targetId: string;
  kind: EdgeKind;
  weight?: number;
  metadata?: Record<string, unknown>;
}

export interface IngestArtifactInput {
  id?: string;
  path: string;
  content: string;
  summary?: string;
  stub?: string;
  salience?: number;
  dependsOn?: string[];
  metadata?: Record<string, unknown>;
}

export interface IngestDecisionInput {
  id?: string;
  content: string;
  summary?: string;
  resourceKey?: string;
  tier?: ContextTier;
  salience?: number;
  dependsOn?: string[];
  conflictsWith?: string[];
  metadata?: Record<string, unknown>;
}

export interface IngestToolCallInput {
  id?: string;
  toolName: string;
  args?: Record<string, unknown>;
  output: string;
  summary?: string;
  path?: string;
  resourceKey?: string;
  isError?: boolean;
  salience?: number;
  dependsOn?: string[];
  metadata?: Record<string, unknown>;
}

export interface RawTurnMessage {
  id?: string;
  role: MessageRole;
  content: string;
  kind?: NodeKind;
  tier?: ContextTier;
  resourceKey?: string;
  hidden?: boolean;
}

export interface IngestTurnInput {
  turn?: number;
  scope?: string;
  taskFrameId?: string;
  system?: string;
  goal?: string;
  constraints?: Array<string | { id?: string; content: string; resourceKey?: string }>;
  taskFrame?: { id?: string; content: string; summary?: string; scope?: string };
  userMessage?: string;
  assistantMessage?: string;
  facts?: Array<string | { id?: string; content: string; resourceKey?: string; dependsOn?: string[] }>;
  decisions?: IngestDecisionInput[];
  artifacts?: IngestArtifactInput[];
  toolCalls?: IngestToolCallInput[];
  summaries?: Array<{ id?: string; content: string; targetNodeIds: string[]; scope?: string }>;
  rawMessages?: RawTurnMessage[];
  retractNodeIds?: string[];
  retractScopes?: string[];
}

export interface IngestTurnResult {
  graphId: string;
  turn: number;
  addedNodes: GraphNode[];
  addedEdges: GraphEdge[];
  supersededNodeIds: string[];
  retractedNodeIds: string[];
}

export interface GraphStats {
  totalNodes: number;
  activeNodes: number;
  supersededNodes: number;
  retractedNodes: number;
  totalEdges: number;
  zone1ActiveNodes: number;
  zone2ActiveNodes: number;
  totalRawTokens: number;
  activeFullTokens: number;
}

export interface GraphSnapshot {
  graph: GraphRecord;
  stats: GraphStats;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ChatMessage {
  role: MessageRole;
  content: string;
  name?: string;
}

/**
 * Wire-compatible projection for `@ppeng/agent-loop` SessionMessage consumers.
 */
export interface AgentLoopCompatibleMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant" | "system" | "tool";
  parts: Array<{ type: "text"; text: string }>;
  seq: number;
  key?: string;
  lod: LodLevel;
  zone: 1 | 2;
}

export interface CompiledNodeEntry {
  id: string;
  seq: number;
  turn: number;
  kind: NodeKind;
  role: MessageRole;
  tier: ContextTier;
  scope: string;
  zone: 1 | 2;
  lod: LodLevel;
  version: number;
  resourceKey: string | null;
  tokens: number;
  activationScore: number;
  renderedText: string;
  pulledByClosure: boolean;
}

export interface ExcludedNodeEntry {
  id: string;
  kind: NodeKind;
  resourceKey: string | null;
  version: number;
  reason: "superseded" | "retracted" | "budget_pruned" | "summarized";
  supersededBy?: string | null;
  tokensSaved: number;
}

export interface ConflictWarning {
  edgeId: string;
  sourceNodeId: string;
  targetNodeId: string;
  sourceExcerpt: string;
  targetExcerpt: string;
  includedInContext: boolean;
}

export interface KvCacheMetrics {
  totalTokens: number;
  zone1Tokens: number;
  zone2Tokens: number;
  framingTokens: number;
  prefixCachedTokens: number;
  uncachedTokens: number;
  prefixHitRate: number;
  cachedCostDiscount: number;
  effectiveBillableTokens: number;
}

export type CompileFormat =
  | "both"
  | "messages"
  | "prompt"
  | "xml"
  | "compact_markdown";

export interface CompileOptions {
  graphId: string;
  focusQuery?: string;
  focusNodeIds?: string[];
  activeScope?: string;
  maxTokens?: number;
  zone1MaxRatio?: number;
  pprAlpha?: number;
  pprIterations?: number;
  format?: CompileFormat;
  maxOutputChars?: number;
  recordCacheState?: boolean;
  previousTokenIds?: number[];
}

export type ToolFamily = "file" | "command" | "search" | "other";

export type FileOpKind =
  | "full_read"
  | "ranged_read"
  | "partial_edit"
  | "full_write"
  | "delete";

export interface ToolTrajectoryEvent {
  callId: string;
  tool: string;
  args?: Record<string, unknown>;
  ok: boolean;
  output: string;
  seq?: number;
  turn?: number;
  machineId?: string;
  scope?: string;
}

export interface EditReplacementPair {
  oldText: string;
  newText: string;
}

export interface ClassifiedToolEvent {
  event: ToolTrajectoryEvent;
  family: ToolFamily;
  fileOpKind?: FileOpKind;
  path?: string;
  fileScopeKey?: string;
  rangeKey?: string;
  rangeLabel?: string;
  diffText?: string;
  editPairs?: EditReplacementPair[];
  command?: string;
  searchKey?: string;
  resourceKey: string | null;
  displayLabel: string;
}

export interface PrunedTrajectoryEvent extends ToolTrajectoryEvent {
  pruned: boolean;
  supersededByCallId?: string;
  originalChars?: number;
}

export interface PruneTrajectoryResult {
  events: PrunedTrajectoryEvent[];
  prunedCallIds: string[];
  prunedCount: number;
  charsSaved: number;
  tokensSaved: number;
}

export type IngestResponseMode = "full" | "slim";

export interface IngestArtifactsOptions {
  scope?: string;
  turn?: number;
  machineId?: string;
  maxCodeExcerptChars?: number;
  maxCommandExcerptChars?: number;
  autoCreate?: boolean;
  graphName?: string;
  responseMode?: IngestResponseMode;
}

export interface IngestArtifactsResult {
  graphId: string;
  ingestedCount: number;
  skippedCount: number;
  addedNodes: GraphNode[];
  addedEdges: GraphEdge[];
  supersededNodeIds: string[];
  activeFileScopes: string[];
  unresolvedFailureNodeIds: string[];
}

export interface CompileResult {
  graphId: string;
  focusQuery: string;
  activeScope: string;
  maxTokens: number;
  messages: ChatMessage[];
  promptText: string;
  agentLoopMessages: AgentLoopCompatibleMessage[];
  tokenIds: number[];
  zone1Nodes: CompiledNodeEntry[];
  zone2Nodes: CompiledNodeEntry[];
  excludedNodes: ExcludedNodeEntry[];
  conflicts: ConflictWarning[];
  kvCache: KvCacheMetrics;
  diagnostics: {
    totalGraphNodes: number;
    activeCandidates: number;
    supersededOccluded: number;
    retractedPruned: number;
    closureAdditions: number;
    lodCounts: Record<LodLevel, number>;
    compressionRatioVsFullLinear: number;
  };
}

export type BenchmarkStrategyName =
  | "linear_full"
  | "linear_sliding_window"
  | "naive_vector_rag"
  | "context_graph_two_zone";

export interface TurnTelemetry {
  turn: number;
  promptTokens: number;
  prefixCachedTokens: number;
  uncachedTokens: number;
  hitRate: number;
  staleTokens: number;
  budgetExceeded: boolean;
}

export interface StrategyBenchmarkMetrics {
  strategy: BenchmarkStrategyName;
  budgetTokens: number;
  finalTurnTokens: number;
  budgetCompliant: boolean;
  criticalConstraintRecall: number;
  latestStateAccuracy: number;
  latestStateRawRecall: number;
  staleTokenRatio: number;
  cumulativeStaleTokenRatio: number;
  kvCacheHitRate: number;
  warmTurnKvCacheHitRate: number;
  totalPromptTokens: number;
  totalCachedTokens: number;
  totalUncachedTokens: number;
  effectiveBillableTokens: number;
  effectiveCostIndex: number;
  costIndexVsLinearUncached: number;
  turnTelemetry: TurnTelemetry[];
}

export interface BenchmarkReport {
  scenarioName: string;
  totalTurns: number;
  fullLinearFinalTokens: number;
  budgetRatio: number;
  tokenBudget: number;
  cachedTokenWeight: number;
  generatedAt: string;
  strategies: Record<BenchmarkStrategyName, StrategyBenchmarkMetrics>;
  markdownTable: string;
}

export class ContextGraphError extends Error {
  readonly code: string;
  readonly statusCode: number;

  constructor(message: string, code: string, statusCode = 400) {
    super(message);
    this.name = "ContextGraphError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

export class CycleDetectedError extends ContextGraphError {
  constructor(message: string) {
    super(message, "CYCLE_DETECTED", 400);
    this.name = "CycleDetectedError";
  }
}

export class NotFoundError extends ContextGraphError {
  constructor(message: string) {
    super(message, "NOT_FOUND", 404);
    this.name = "NotFoundError";
  }
}
