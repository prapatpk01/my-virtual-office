import type { HedgeFundConfig, PriceBar, AssetSnapshot } from "./types.js"

// ── Alpaca Trading API ──

export async function alpacaTradeRequest<T>(
  config: HedgeFundConfig,
  path: string,
  options: { method?: string; body?: unknown } = {}
): Promise<T> {
  const res = await fetch(`${config.alpacaBaseUrl}${path}`, {
    method: options.method ?? "GET",
    headers: {
      "APCA-API-KEY-ID": config.alpacaApiKey,
      "APCA-API-SECRET-KEY": config.alpacaApiSecret,
      "Content-Type": "application/json",
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Alpaca trade API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ── Alpaca Market Data API ──

export async function alpacaDataRequest<T>(
  config: HedgeFundConfig,
  path: string,
  params: Record<string, string> = {}
): Promise<T> {
  const url = new URL(`${config.alpacaDataUrl}${path}`)
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v)
  const res = await fetch(url.toString(), {
    headers: {
      "APCA-API-KEY-ID": config.alpacaApiKey,
      "APCA-API-SECRET-KEY": config.alpacaApiSecret,
    },
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Alpaca data API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ── Finnhub API ──

export async function finnhubRequest<T>(
  config: HedgeFundConfig,
  path: string,
  params: Record<string, string> = {}
): Promise<T> {
  const url = new URL(`https://finnhub.io/api/v1${path}`)
  url.searchParams.set("token", config.finnhubApiKey)
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v)
  const res = await fetch(url.toString())
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Finnhub API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ── Asset Classification ──

export const CRYPTO_SYMBOLS = new Set([
  "BTCUSD", "ETHUSD", "LTCUSD", "SOLUSD", "AVAXUSD", "DOGEUSD", "LINKUSD",
])

const ETF_SYMBOLS = new Set([
  "GLD", "SLV", "USO", "SPY", "QQQ", "IWM", "TLT", "GDX", "GDXJ",
])

export function assetClass(symbol: string): "equity" | "crypto" | "etf" {
  if (CRYPTO_SYMBOLS.has(symbol)) return "crypto"
  if (ETF_SYMBOLS.has(symbol)) return "etf"
  return "equity"
}

// ── Price Snapshots ──

interface RawStockSnapshot {
  latestTrade?: { p: number }
  dailyBar?: { o: number; h: number; l: number; c: number; v: number }
  prevDailyBar?: { c: number }
}

export async function getSnapshots(
  config: HedgeFundConfig,
  symbols: readonly string[]
): Promise<AssetSnapshot[]> {
  const stocks = symbols.filter(s => !CRYPTO_SYMBOLS.has(s))
  const cryptos = symbols.filter(s => CRYPTO_SYMBOLS.has(s))
  const results: AssetSnapshot[] = []

  if (stocks.length > 0) {
    const data = await alpacaDataRequest<Record<string, RawStockSnapshot>>(
      config,
      "/v2/stocks/snapshots",
      { symbols: stocks.join(",") }
    )
    for (const [symbol, snap] of Object.entries(data)) {
      const price = snap.latestTrade?.p ?? snap.dailyBar?.c ?? 0
      const prev = snap.prevDailyBar?.c ?? price
      results.push({
        symbol,
        latestPrice: price,
        prevClose: prev,
        dayChange: price - prev,
        dayChangePct: prev > 0 ? ((price - prev) / prev) * 100 : 0,
        volume: snap.dailyBar?.v ?? 0,
      })
    }
  }

  if (cryptos.length > 0) {
    const data = await alpacaDataRequest<{
      snapshots: Record<string, RawStockSnapshot>
    }>(config, "/v1beta3/crypto/us/snapshots", { symbols: cryptos.join(",") })
    for (const [symbol, snap] of Object.entries(data.snapshots)) {
      const price = snap.latestTrade?.p ?? snap.dailyBar?.c ?? 0
      const prev = snap.prevDailyBar?.c ?? price
      results.push({
        symbol,
        latestPrice: price,
        prevClose: prev,
        dayChange: price - prev,
        dayChangePct: prev > 0 ? ((price - prev) / prev) * 100 : 0,
        volume: snap.dailyBar?.v ?? 0,
      })
    }
  }

  return results
}

// ── Historical Bars (60-day default for indicator calculation) ──

export async function getHistoricalBars(
  config: HedgeFundConfig,
  symbol: string,
  limit = 60
): Promise<PriceBar[]> {
  if (CRYPTO_SYMBOLS.has(symbol)) {
    const data = await alpacaDataRequest<{ bars: Record<string, PriceBar[]> }>(
      config,
      "/v1beta3/crypto/us/bars",
      { symbols: symbol, timeframe: "1Day", limit: String(limit) }
    )
    return data.bars[symbol] ?? []
  }
  const data = await alpacaDataRequest<{ bars: PriceBar[] }>(
    config,
    `/v2/stocks/${symbol}/bars`,
    { timeframe: "1Day", limit: String(limit), adjustment: "raw" }
  )
  return data.bars ?? []
}
