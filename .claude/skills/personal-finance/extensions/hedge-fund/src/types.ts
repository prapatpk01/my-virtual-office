// ── Hedge Fund Extension Types ──

export type AlpacaEnv = "paper" | "live"

export interface HedgeFundConfig {
  readonly alpacaApiKey: string
  readonly alpacaApiSecret: string
  readonly alpacaEnv: AlpacaEnv
  readonly alpacaBaseUrl: string
  readonly alpacaDataUrl: string
  readonly finnhubApiKey: string
  readonly maxPositionPct: number
  readonly maxDrawdownPct: number
  readonly defaultUniverse: readonly string[]
}

// ── Market Data ──

export interface PriceBar {
  readonly t: string
  readonly o: number
  readonly h: number
  readonly l: number
  readonly c: number
  readonly v: number
}

export interface AssetSnapshot {
  readonly symbol: string
  readonly latestPrice: number
  readonly prevClose: number
  readonly dayChange: number
  readonly dayChangePct: number
  readonly volume: number
}

// ── Research ──

export type AssetClass = "equity" | "crypto" | "etf"
export type MarketCondition = "RISK_ON" | "RISK_OFF" | "NEUTRAL"
export type SentimentLabel = "BULLISH" | "NEUTRAL" | "BEARISH"

export interface ResearchItem {
  readonly symbol: string
  readonly assetClass: AssetClass
  readonly latestPrice: number
  readonly dayChange: number
  readonly dayChangePct: number
  readonly newsHeadlines: readonly string[]
  readonly sentimentScore: number // -1 to 1
  readonly sentimentLabel: SentimentLabel
}

export interface ResearchBrief {
  readonly generatedAt: string
  readonly team: "Research"
  readonly symbols: readonly string[]
  readonly items: readonly ResearchItem[]
  readonly marketCondition: MarketCondition
}

// ── Quant Signals ──

export type SignalDirection = "BUY" | "SELL" | "HOLD"
export type SignalStrength = "STRONG" | "MODERATE" | "WEAK"
export type MACDSignal = "ABOVE_SIGNAL" | "BELOW_SIGNAL"
export type MarketRegime = "TRENDING" | "RANGING" | "VOLATILE"

export interface TradeSignal {
  readonly symbol: string
  readonly direction: SignalDirection
  readonly strength: SignalStrength
  readonly confidence: number // 0–100
  readonly reasons: readonly string[]
  readonly suggestedNotional: number
  readonly rsi14: number | null
  readonly macdSignal: MACDSignal | null
}

export interface SignalsReport {
  readonly generatedAt: string
  readonly team: "Quant"
  readonly signals: readonly TradeSignal[]
  readonly marketRegime: MarketRegime
}

// ── Risk ──

export interface RiskDecision {
  readonly symbol: string
  readonly approved: boolean
  readonly originalDirection: SignalDirection
  readonly finalDirection: SignalDirection
  readonly originalNotional: number
  readonly approvedNotional: number
  readonly reasoning: string
  readonly riskFlags: readonly string[]
}

export interface RiskReport {
  readonly generatedAt: string
  readonly team: "Risk"
  readonly portfolioValue: number
  readonly currentCash: number
  readonly decisions: readonly RiskDecision[]
  readonly portfolioRiskScore: number // 0–100
  readonly tradingHalted: boolean
  readonly haltReason: string | null
}

// ── Execution ──

export type TradeStatus = "submitted" | "rejected" | "error"

export interface ExecutedTrade {
  readonly symbol: string
  readonly orderId: string | null
  readonly side: "buy" | "sell"
  readonly notional: number
  readonly status: TradeStatus
  readonly message: string
}

export interface ExecutionReport {
  readonly generatedAt: string
  readonly team: "Execution"
  readonly trades: readonly ExecutedTrade[]
  readonly totalNotional: number
  readonly successCount: number
  readonly failCount: number
}

// ── Daily Cycle ──

export type CycleMode = "research_only" | "signals_only" | "full_dry_run" | "full"

export interface DailyCycleReport {
  readonly cycleId: string
  readonly runAt: string
  readonly mode: CycleMode
  readonly symbols: readonly string[]
  readonly research: ResearchBrief | null
  readonly signals: SignalsReport | null
  readonly risk: RiskReport | null
  readonly execution: ExecutionReport | null
  readonly summary: string
}

// ── Fund Status ──

export interface FundPosition {
  readonly symbol: string
  readonly qty: number
  readonly marketValue: number
  readonly unrealizedPL: number
  readonly unrealizedPLPct: number
  readonly currentPrice: number
  readonly side: "long" | "short"
}

export interface FundStatus {
  readonly asOf: string
  readonly portfolioValue: number
  readonly cash: number
  readonly equity: number
  readonly buyingPower: number
  readonly dayPL: number
  readonly positions: readonly FundPosition[]
  readonly positionCount: number
  readonly tradingEnvironment: AlpacaEnv
}

// ── Runtime-mutable Fund Config ──

export interface FundConfig {
  universe: string[]
  maxPositionPct: number
  maxDrawdownPct: number
  baseNotionalPerTrade: number
}

// ── Shared Tool Infra ──

export interface ToolContext {
  readonly config: HedgeFundConfig
  readonly fundConfig: FundConfig
}

export interface ToolResult<T> {
  readonly success: boolean
  readonly data?: T
  readonly error?: string
}
