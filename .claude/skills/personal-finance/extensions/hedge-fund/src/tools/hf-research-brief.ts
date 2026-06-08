import type {
  ResearchBrief,
  ResearchItem,
  SentimentLabel,
  MarketCondition,
  ToolContext,
  ToolResult,
  HedgeFundConfig,
} from "../types.js"
import { getSnapshots, finnhubRequest, assetClass, CRYPTO_SYMBOLS } from "../client.js"

interface FinnhubNewsItem {
  headline: string
}

interface FinnhubSentiment {
  sentiment?: { bearishPercent?: number; bullishPercent?: number }
}

async function fetchNewsSentiment(
  config: HedgeFundConfig,
  symbol: string
): Promise<{ headlines: string[]; sentimentScore: number }> {
  // Finnhub does not support crypto sentiment
  if (CRYPTO_SYMBOLS.has(symbol)) return { headlines: [], sentimentScore: 0 }

  try {
    const today = new Date()
    const weekAgo = new Date(today.getTime() - 7 * 24 * 3600 * 1000)
    const fmt = (d: Date) => d.toISOString().slice(0, 10)

    const [news, sentiment] = await Promise.all([
      finnhubRequest<FinnhubNewsItem[]>(config, "/company-news", {
        symbol,
        from: fmt(weekAgo),
        to: fmt(today),
      }),
      finnhubRequest<FinnhubSentiment>(config, "/news-sentiment", { symbol }),
    ])

    const headlines = (news ?? []).slice(0, 5).map(n => n.headline)
    const bull = sentiment?.sentiment?.bullishPercent ?? 0.5
    const bear = sentiment?.sentiment?.bearishPercent ?? 0.5
    return {
      headlines,
      sentimentScore: Math.max(-1, Math.min(1, bull - bear)),
    }
  } catch {
    return { headlines: [], sentimentScore: 0 }
  }
}

function toSentimentLabel(score: number): SentimentLabel {
  if (score > 0.15) return "BULLISH"
  if (score < -0.15) return "BEARISH"
  return "NEUTRAL"
}

function inferMarketCondition(items: ResearchItem[]): MarketCondition {
  if (items.length === 0) return "NEUTRAL"
  const avgChange = items.reduce((s, i) => s + i.dayChangePct, 0) / items.length
  const avgSent = items.reduce((s, i) => s + i.sentimentScore, 0) / items.length
  if (avgChange > 0.5 && avgSent > 0.1) return "RISK_ON"
  if (avgChange < -0.5 || avgSent < -0.1) return "RISK_OFF"
  return "NEUTRAL"
}

export const hfResearchBriefTool = {
  name: "hf_research_brief",
  description:
    "【Research Team】Gather market intelligence for a list of symbols — live prices, news headlines, and news sentiment. Returns a structured research brief used by the Quant team.",
  input_schema: {
    type: "object",
    required: ["symbols"],
    properties: {
      symbols: {
        type: "array",
        items: { type: "string" },
        description: "Symbols to research, e.g. [\"AAPL\", \"GLD\", \"BTCUSD\"]",
        minItems: 1,
        maxItems: 20,
      },
    },
    additionalProperties: false,
  },
  async handler(
    input: { symbols: string[] },
    context: ToolContext
  ): Promise<ToolResult<ResearchBrief>> {
    try {
      const { config } = context
      const symbols = input.symbols.map(s => s.toUpperCase())

      const [snapshots, ...sentimentResults] = await Promise.all([
        getSnapshots(config, symbols),
        ...symbols.map(s => fetchNewsSentiment(config, s)),
      ])

      const items: ResearchItem[] = symbols.map((symbol, i) => {
        const snap = snapshots.find(s => s.symbol === symbol)
        const sent = sentimentResults[i]!
        return {
          symbol,
          assetClass: assetClass(symbol),
          latestPrice: snap?.latestPrice ?? 0,
          dayChange: snap?.dayChange ?? 0,
          dayChangePct: snap?.dayChangePct ?? 0,
          newsHeadlines: sent.headlines,
          sentimentScore: sent.sentimentScore,
          sentimentLabel: toSentimentLabel(sent.sentimentScore),
        }
      })

      return {
        success: true,
        data: {
          generatedAt: new Date().toISOString(),
          team: "Research",
          symbols,
          items,
          marketCondition: inferMarketCondition(items),
        },
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  },
}
