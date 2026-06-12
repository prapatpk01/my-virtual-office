import type {
  ExecutionReport,
  ExecutedTrade,
  RiskDecision,
  ToolContext,
  ToolResult,
} from "../types.js"
import { alpacaTradeRequest } from "../client.js"

interface AlpacaOrderResult {
  id: string
  status: string
}

interface AlpacaPosition {
  symbol: string
  qty: string
  side: string
}

export const hfExecuteTradesTool = {
  name: "hf_execute_trades",
  description:
    "【Execution Team】Submit approved trades as market orders via Alpaca. Only decisions with approved: true are executed. Requires confirm: true as a safety gate.",
  input_schema: {
    type: "object",
    required: ["risk_decisions", "confirm"],
    properties: {
      risk_decisions: {
        type: "array",
        items: { type: "object" },
        description: "RiskDecision[] from hf_risk_review",
      },
      confirm: {
        type: "boolean",
        description: "Safety gate — must be explicitly true to submit live orders",
      },
    },
    additionalProperties: false,
  },
  async handler(
    input: { risk_decisions: RiskDecision[]; confirm: boolean },
    context: ToolContext
  ): Promise<ToolResult<ExecutionReport>> {
    try {
      if (input.confirm !== true) {
        return { success: false, error: "Execution requires confirm: true" }
      }

      const approved = input.risk_decisions.filter(d => d.approved && d.approvedNotional > 0)
      const trades: ExecutedTrade[] = []

      // Fetch existing positions to guard against unintended short sells
      const positions = await alpacaTradeRequest<AlpacaPosition[]>(context.config, "/v2/positions")
      const heldSymbols = new Set(positions.filter(p => parseFloat(p.qty) > 0).map(p => p.symbol))

      for (const decision of approved) {
        const side = decision.finalDirection === "BUY" ? "buy" : "sell"

        // Skip SELL if no long position exists — prevents accidental short selling
        if (side === "sell" && !heldSymbols.has(decision.symbol)) {
          trades.push({
            symbol: decision.symbol,
            orderId: null,
            side,
            notional: decision.approvedNotional,
            status: "rejected",
            message: `No long position in ${decision.symbol} — SELL skipped to avoid short`,
          })
          continue
        }

        try {
          const order = await alpacaTradeRequest<AlpacaOrderResult>(
            context.config,
            "/v2/orders",
            {
              method: "POST",
              body: {
                symbol: decision.symbol,
                notional: String(decision.approvedNotional),
                side,
                type: "market",
                time_in_force: "day",
                client_order_id: `hf-${decision.symbol}-${Date.now()}`,
              },
            }
          )
          trades.push({
            symbol: decision.symbol,
            orderId: order.id,
            side,
            notional: decision.approvedNotional,
            status: "submitted",
            message: `Order ${order.id} submitted — ${order.status}`,
          })
        } catch (err) {
          trades.push({
            symbol: decision.symbol,
            orderId: null,
            side,
            notional: decision.approvedNotional,
            status: "error",
            message: err instanceof Error ? err.message : String(err),
          })
        }
      }

      const successCount = trades.filter(t => t.status === "submitted").length
      const totalNotional = trades
        .filter(t => t.status === "submitted")
        .reduce((s, t) => s + t.notional, 0)

      return {
        success: true,
        data: {
          generatedAt: new Date().toISOString(),
          team: "Execution",
          trades,
          totalNotional,
          successCount,
          failCount: trades.length - successCount,
        },
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  },
}
