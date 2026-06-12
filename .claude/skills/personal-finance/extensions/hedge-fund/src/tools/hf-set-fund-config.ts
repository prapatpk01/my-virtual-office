import type { FundConfig, ToolContext, ToolResult } from "../types.js"

export const hfSetFundConfigTool = {
  name: "hf_set_fund_config",
  description:
    "Update hedge fund runtime parameters: trading universe, position size limits, drawdown limit, and base notional per trade. Changes take effect immediately for subsequent tool calls.",
  input_schema: {
    type: "object",
    properties: {
      universe: {
        type: "array",
        items: { type: "string" },
        description: "New trading universe (ticker symbols, e.g. [\"AAPL\", \"GLD\", \"BTCUSD\"])",
      },
      max_position_pct: {
        type: "number",
        description: "Max single-position size as fraction of portfolio value (0.01–0.5)",
        minimum: 0.01,
        maximum: 0.5,
      },
      max_drawdown_pct: {
        type: "number",
        description: "Max drawdown before trading halt (0.01–0.5)",
        minimum: 0.01,
        maximum: 0.5,
      },
      base_notional_per_trade: {
        type: "number",
        description: "Base dollar notional per trade signal",
        minimum: 1,
      },
    },
    additionalProperties: false,
  },
  async handler(
    input: {
      universe?: string[]
      max_position_pct?: number
      max_drawdown_pct?: number
      base_notional_per_trade?: number
    },
    context: ToolContext
  ): Promise<ToolResult<{ previous: FundConfig; updated: FundConfig; changes: string[] }>> {
    try {
      const { fundConfig } = context
      const previous: FundConfig = { ...fundConfig, universe: [...fundConfig.universe] }
      const changes: string[] = []

      if (input.universe !== undefined) {
        fundConfig.universe = input.universe.map(s => s.toUpperCase())
        changes.push(`universe → [${fundConfig.universe.join(", ")}]`)
      }
      if (input.max_position_pct !== undefined) {
        fundConfig.maxPositionPct = input.max_position_pct
        changes.push(`maxPositionPct → ${(fundConfig.maxPositionPct * 100).toFixed(0)}%`)
      }
      if (input.max_drawdown_pct !== undefined) {
        fundConfig.maxDrawdownPct = input.max_drawdown_pct
        changes.push(`maxDrawdownPct → ${(fundConfig.maxDrawdownPct * 100).toFixed(0)}%`)
      }
      if (input.base_notional_per_trade !== undefined) {
        fundConfig.baseNotionalPerTrade = input.base_notional_per_trade
        changes.push(`baseNotionalPerTrade → $${fundConfig.baseNotionalPerTrade}`)
      }

      if (changes.length === 0) {
        return { success: true, data: { previous, updated: { ...fundConfig }, changes: ["no changes"] } }
      }

      return {
        success: true,
        data: { previous, updated: { ...fundConfig, universe: [...fundConfig.universe] }, changes },
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  },
}
