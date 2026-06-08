import type {
  RiskReport,
  RiskDecision,
  TradeSignal,
  ToolContext,
  ToolResult,
} from "../types.js"
import { alpacaTradeRequest } from "../client.js"

interface AlpacaAccount {
  portfolio_value: string
  cash: string
  equity: string
  last_equity: string
}

interface AlpacaPosition {
  symbol: string
  market_value: string
}

export const hfRiskReviewTool = {
  name: "hf_risk_review",
  description:
    "【Risk Management Team】Evaluate proposed trade signals against portfolio constraints: position concentration limits, drawdown limits, and available buying power. Returns approved / rejected decisions with final notional sizing.",
  input_schema: {
    type: "object",
    required: ["signals"],
    properties: {
      signals: {
        type: "array",
        items: { type: "object" },
        description: "TradeSignal[] from hf_generate_signals",
      },
    },
    additionalProperties: false,
  },
  async handler(
    input: { signals: TradeSignal[] },
    context: ToolContext
  ): Promise<ToolResult<RiskReport>> {
    try {
      const { config, fundConfig } = context
      const { maxPositionPct, maxDrawdownPct } = fundConfig

      const [account, positions] = await Promise.all([
        alpacaTradeRequest<AlpacaAccount>(config, "/v2/account"),
        alpacaTradeRequest<AlpacaPosition[]>(config, "/v2/positions"),
      ])

      const portfolioValue = parseFloat(account.portfolio_value)
      const currentCash = parseFloat(account.cash)
      const equity = parseFloat(account.equity)
      const lastEquity = parseFloat(account.last_equity)

      const drawdown = lastEquity > 0 ? (lastEquity - equity) / lastEquity : 0
      const tradingHalted = drawdown > maxDrawdownPct
      const haltReason = tradingHalted
        ? `Drawdown ${(drawdown * 100).toFixed(2)}% exceeds limit ${(maxDrawdownPct * 100).toFixed(1)}% — trading halted`
        : null

      const positionMap = new Map<string, number>()
      for (const p of positions) {
        positionMap.set(p.symbol, Math.abs(parseFloat(p.market_value)))
      }

      const maxPositionValue = portfolioValue * maxPositionPct

      const decisions: RiskDecision[] = input.signals.map(signal => {
        if (signal.direction === "HOLD") {
          return {
            symbol: signal.symbol,
            approved: false,
            originalDirection: "HOLD",
            finalDirection: "HOLD",
            originalNotional: 0,
            approvedNotional: 0,
            reasoning: "Signal is HOLD — no trade required",
            riskFlags: [],
          }
        }

        if (tradingHalted) {
          return {
            symbol: signal.symbol,
            approved: false,
            originalDirection: signal.direction,
            finalDirection: "HOLD",
            originalNotional: signal.suggestedNotional,
            approvedNotional: 0,
            reasoning: haltReason!,
            riskFlags: ["DRAWDOWN_BREACH"],
          }
        }

        const existing = positionMap.get(signal.symbol) ?? 0
        const remaining = maxPositionValue - existing

        if (remaining <= 0) {
          return {
            symbol: signal.symbol,
            approved: false,
            originalDirection: signal.direction,
            finalDirection: "HOLD",
            originalNotional: signal.suggestedNotional,
            approvedNotional: 0,
            reasoning: `Position already at max capacity (${(maxPositionPct * 100).toFixed(0)}% of portfolio)`,
            riskFlags: ["POSITION_LIMIT"],
          }
        }

        const flags: string[] = []
        let approved = Math.min(signal.suggestedNotional, remaining, currentCash * 0.95)
        approved = Math.floor(approved * 100) / 100

        if (approved < signal.suggestedNotional) flags.push("NOTIONAL_REDUCED")

        if (approved < 1) {
          return {
            symbol: signal.symbol,
            approved: false,
            originalDirection: signal.direction,
            finalDirection: "HOLD",
            originalNotional: signal.suggestedNotional,
            approvedNotional: 0,
            reasoning: "Insufficient buying power",
            riskFlags: ["INSUFFICIENT_CASH"],
          }
        }

        return {
          symbol: signal.symbol,
          approved: true,
          originalDirection: signal.direction,
          finalDirection: signal.direction,
          originalNotional: signal.suggestedNotional,
          approvedNotional: approved,
          reasoning: flags.length
            ? "Approved at reduced notional due to position/cash limits"
            : "Approved — within all risk parameters",
          riskFlags: flags,
        }
      })

      const totalExposure = [...positionMap.values()].reduce((a, b) => a + b, 0)
      const portfolioRiskScore = Math.min(100, Math.round((totalExposure / portfolioValue) * 100))

      return {
        success: true,
        data: {
          generatedAt: new Date().toISOString(),
          team: "Risk",
          portfolioValue,
          currentCash,
          decisions,
          portfolioRiskScore,
          tradingHalted,
          haltReason,
        },
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  },
}
