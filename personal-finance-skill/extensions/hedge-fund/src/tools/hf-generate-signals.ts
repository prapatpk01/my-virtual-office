import type {
  SignalsReport,
  TradeSignal,
  SignalDirection,
  SignalStrength,
  MACDSignal,
  MarketRegime,
  ResearchBrief,
  ToolContext,
  ToolResult,
} from "../types.js"
import { getHistoricalBars } from "../client.js"
import { calcRSI, calcMACD } from "../indicators.js"

function toMACDSignal(macd: { macdLine: number; signalLine: number } | null): MACDSignal | null {
  if (!macd) return null
  return macd.macdLine > macd.signalLine ? "ABOVE_SIGNAL" : "BELOW_SIGNAL"
}

function inferMarketRegime(signals: TradeSignal[]): MarketRegime {
  if (signals.length === 0) return "RANGING"
  const holds = signals.filter(s => s.direction === "HOLD").length
  const highConf = signals.filter(s => s.confidence > 60).length
  if (holds > signals.length * 0.5) return "RANGING"
  if (highConf > signals.length * 0.5) return "TRENDING"
  return "VOLATILE"
}

export const hfGenerateSignalsTool = {
  name: "hf_generate_signals",
  description:
    "【Quant Team】Compute RSI-14 and MACD technical signals for each symbol, weighted with sentiment from the research brief, to produce BUY / SELL / HOLD recommendations with confidence scores.",
  input_schema: {
    type: "object",
    required: ["symbols"],
    properties: {
      symbols: {
        type: "array",
        items: { type: "string" },
        description: "Symbols to generate signals for",
      },
      research_brief: {
        type: "object",
        description: "Optional ResearchBrief from hf_research_brief — improves signal quality",
      },
      base_notional_per_trade: {
        type: "number",
        description: "Base dollar notional suggested per trade (default: fund config value)",
      },
    },
    additionalProperties: false,
  },
  async handler(
    input: {
      symbols: string[]
      research_brief?: ResearchBrief
      base_notional_per_trade?: number
    },
    context: ToolContext
  ): Promise<ToolResult<SignalsReport>> {
    try {
      const { config, fundConfig } = context
      const symbols = input.symbols.map(s => s.toUpperCase())
      const baseNotional = input.base_notional_per_trade ?? fundConfig.baseNotionalPerTrade

      const barSettled = await Promise.allSettled(symbols.map(s => getHistoricalBars(config, s, 60)))

      const signals: TradeSignal[] = symbols.flatMap((symbol, i) => {
        const settled = barSettled[i]!
        if (settled.status === "rejected") return []
        const closes = settled.value.map(b => b.c)
        const rsi = calcRSI(closes)
        const macd = calcMACD(closes)
        const sentimentScore =
          input.research_brief?.items.find(r => r.symbol === symbol)?.sentimentScore ?? 0

        const votes: number[] = []
        const reasons: string[] = []

        if (rsi !== null) {
          if (rsi < 30) {
            votes.push(1)
            reasons.push(`RSI ${rsi.toFixed(1)} — oversold`)
          } else if (rsi > 70) {
            votes.push(-1)
            reasons.push(`RSI ${rsi.toFixed(1)} — overbought`)
          } else {
            votes.push(0)
            reasons.push(`RSI ${rsi.toFixed(1)} — neutral`)
          }
        }

        if (macd !== null) {
          if (macd.macdLine > macd.signalLine) {
            votes.push(1)
            reasons.push("MACD above signal line — momentum bullish")
          } else {
            votes.push(-1)
            reasons.push("MACD below signal line — momentum bearish")
          }
        }

        if (sentimentScore > 0.2) {
          votes.push(1)
          reasons.push(`News sentiment bullish (score ${sentimentScore.toFixed(2)})`)
        } else if (sentimentScore < -0.2) {
          votes.push(-1)
          reasons.push(`News sentiment bearish (score ${sentimentScore.toFixed(2)})`)
        }

        const n = votes.length || 1
        const scoreSum = votes.reduce((a, b) => a + b, 0)
        const normalized = scoreSum / n // -1 to 1

        let direction: SignalDirection
        if (normalized > 0.2) direction = "BUY"
        else if (normalized < -0.2) direction = "SELL"
        else direction = "HOLD"

        const confidence = Math.round(Math.abs(normalized) * 100)
        const strength: SignalStrength =
          confidence > 66 ? "STRONG" : confidence > 33 ? "MODERATE" : "WEAK"

        return {
          symbol,
          direction,
          strength,
          confidence,
          reasons,
          suggestedNotional: direction !== "HOLD" ? baseNotional * (1 + confidence / 100) : 0,
          rsi14: rsi !== null ? Math.round(rsi * 10) / 10 : null,
          macdSignal: toMACDSignal(macd),
        }
      })

      return {
        success: true,
        data: {
          generatedAt: new Date().toISOString(),
          team: "Quant",
          signals,
          marketRegime: inferMarketRegime(signals),
        },
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  },
}
