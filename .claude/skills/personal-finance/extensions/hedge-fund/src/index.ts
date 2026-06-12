import type { OpenClawPluginApi } from "openclaw/plugin-sdk"
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk"

import { buildConfig } from "./config.js"
import type { FundConfig, HedgeFundConfig, ToolContext, ToolResult } from "./types.js"
import type { AlpacaEnv } from "./types.js"

import { hfResearchBriefTool } from "./tools/hf-research-brief.js"
import { hfGenerateSignalsTool } from "./tools/hf-generate-signals.js"
import { hfRiskReviewTool } from "./tools/hf-risk-review.js"
import { hfExecuteTradesTool } from "./tools/hf-execute-trades.js"
import { hfGetFundStatusTool } from "./tools/hf-get-fund-status.js"
import { hfRunDailyCycleTool } from "./tools/hf-run-daily-cycle.js"
import { hfSetFundConfigTool } from "./tools/hf-set-fund-config.js"

// ── Tool Adapter ──

interface ToolDef {
  readonly name: string
  readonly description: string
  readonly input_schema: Record<string, unknown>
  readonly handler: (input: any, context: ToolContext) => Promise<ToolResult<unknown>>
}

function jsonResult(payload: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload, null, 2) }],
    details: payload,
  }
}

const ALL_TOOLS: ReadonlyArray<ToolDef> = [
  hfResearchBriefTool,
  hfGenerateSignalsTool,
  hfRiskReviewTool,
  hfExecuteTradesTool,
  hfGetFundStatusTool,
  hfRunDailyCycleTool,
  hfSetFundConfigTool,
]

// ── Plugin Definition ──

const plugin: {
  id: string
  name: string
  description: string
  configSchema: any
  register: (api: OpenClawPluginApi) => void
} = {
  id: "hedge-fund",
  name: "Office Hedge Fund",
  description:
    "Multi-agent hedge fund with Research, Quant, Risk, and Execution teams for trading stocks, gold, and BTC",
  configSchema: emptyPluginConfigSchema(),

  register(api: OpenClawPluginApi) {
    const cfg = (api.pluginConfig ?? {}) as Record<string, unknown>

    // Lazy config — resolves env vars on first tool call
    let configCache: HedgeFundConfig | null = null
    const getConfig = (): HedgeFundConfig => {
      if (configCache) return configCache
      configCache = buildConfig({
        alpacaApiKeyEnv: (cfg.alpacaApiKeyEnv as string) ?? "ALPACA_API_KEY",
        alpacaApiSecretEnv: (cfg.alpacaApiSecretEnv as string) ?? "ALPACA_API_SECRET",
        alpacaEnv: ((cfg.alpacaEnv as string) ?? "paper") as AlpacaEnv,
        finnhubApiKeyEnv: (cfg.finnhubApiKeyEnv as string) ?? "FINNHUB_API_KEY",
        maxPositionPct: (cfg.maxPositionPct as number) ?? 0.15,
        maxDrawdownPct: (cfg.maxDrawdownPct as number) ?? 0.10,
        defaultUniverse: (cfg.defaultUniverse as string[]) ?? ["AAPL", "TSLA", "NVDA", "GLD", "BTCUSD"],
      })
      return configCache
    }

    // Shared mutable fund config — all tools see the same object reference,
    // so hf_set_fund_config mutations are immediately visible to subsequent calls.
    const fundConfig: FundConfig = {
      universe: ((cfg.defaultUniverse as string[]) ?? ["AAPL", "TSLA", "NVDA", "GLD", "BTCUSD"]).map(
        s => s.toUpperCase()
      ),
      maxPositionPct: (cfg.maxPositionPct as number) ?? 0.15,
      maxDrawdownPct: (cfg.maxDrawdownPct as number) ?? 0.10,
      baseNotionalPerTrade: 1000,
    }

    for (const tool of ALL_TOOLS) {
      api.registerTool(
        {
          name: tool.name,
          label: tool.name,
          description: tool.description,
          parameters: tool.input_schema,
          async execute(_toolCallId: string, params: unknown) {
            try {
              const config = getConfig()
              const context: ToolContext = { config, fundConfig }
              const result = await tool.handler(params as Record<string, unknown>, context)
              return jsonResult(result)
            } catch (error) {
              const message = error instanceof Error ? error.message : String(error)
              return jsonResult({ success: false, error: message })
            }
          },
        } as any
      )
    }
  },
}

export default plugin
export { buildConfig } from "./config.js"
export type { HedgeFundConfig, FundConfig, ToolContext } from "./types.js"
