import type { FundStatus, FundPosition, ToolContext, ToolResult } from "../types.js"
import { alpacaTradeRequest } from "../client.js"

interface AlpacaAccount {
  portfolio_value: string
  cash: string
  equity: string
  buying_power: string
  last_equity: string
}

interface AlpacaPosition {
  symbol: string
  qty: string
  market_value: string
  unrealized_pl: string
  unrealized_plpc: string
  current_price: string
  side: "long" | "short"
}

export const hfGetFundStatusTool = {
  name: "hf_get_fund_status",
  description:
    "Get current hedge fund status: portfolio value, cash balance, day P&L, all open positions, and trading environment.",
  input_schema: {
    type: "object",
    properties: {},
    additionalProperties: false,
  },
  async handler(
    _input: Record<string, never>,
    context: ToolContext
  ): Promise<ToolResult<FundStatus>> {
    try {
      const { config } = context

      const [account, rawPositions] = await Promise.all([
        alpacaTradeRequest<AlpacaAccount>(config, "/v2/account"),
        alpacaTradeRequest<AlpacaPosition[]>(config, "/v2/positions"),
      ])

      const equity = parseFloat(account.equity)
      const lastEquity = parseFloat(account.last_equity)

      const positions: FundPosition[] = rawPositions.map(p => ({
        symbol: p.symbol,
        qty: parseFloat(p.qty),
        marketValue: parseFloat(p.market_value),
        unrealizedPL: parseFloat(p.unrealized_pl),
        unrealizedPLPct: parseFloat(p.unrealized_plpc) * 100,
        currentPrice: parseFloat(p.current_price),
        side: p.side,
      }))

      return {
        success: true,
        data: {
          asOf: new Date().toISOString(),
          portfolioValue: parseFloat(account.portfolio_value),
          cash: parseFloat(account.cash),
          equity,
          buyingPower: parseFloat(account.buying_power),
          dayPL: equity - lastEquity,
          positions,
          positionCount: positions.length,
          tradingEnvironment: config.alpacaEnv,
        },
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  },
}
