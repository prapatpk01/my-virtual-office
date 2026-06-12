import type { HedgeFundConfig, AlpacaEnv } from "./types.js"

const ALPACA_BASE_URLS: Record<AlpacaEnv, string> = {
  paper: "https://paper-api.alpaca.markets",
  live: "https://api.alpaca.markets",
}

const ALPACA_DATA_URL = "https://data.alpaca.markets"

export function buildConfig(pluginConfig: {
  readonly alpacaApiKeyEnv: string
  readonly alpacaApiSecretEnv: string
  readonly alpacaEnv: AlpacaEnv
  readonly finnhubApiKeyEnv: string
  readonly maxPositionPct: number
  readonly maxDrawdownPct: number
  readonly defaultUniverse: readonly string[]
}): HedgeFundConfig {
  const alpacaApiKey = process.env[pluginConfig.alpacaApiKeyEnv]
  if (!alpacaApiKey)
    throw new Error(`Alpaca API key not found in env var: ${pluginConfig.alpacaApiKeyEnv}`)

  const alpacaApiSecret = process.env[pluginConfig.alpacaApiSecretEnv]
  if (!alpacaApiSecret)
    throw new Error(`Alpaca API secret not found in env var: ${pluginConfig.alpacaApiSecretEnv}`)

  const finnhubApiKey = process.env[pluginConfig.finnhubApiKeyEnv]
  if (!finnhubApiKey)
    throw new Error(`Finnhub API key not found in env var: ${pluginConfig.finnhubApiKeyEnv}`)

  return {
    alpacaApiKey,
    alpacaApiSecret,
    alpacaEnv: pluginConfig.alpacaEnv,
    alpacaBaseUrl: ALPACA_BASE_URLS[pluginConfig.alpacaEnv],
    alpacaDataUrl: ALPACA_DATA_URL,
    finnhubApiKey,
    maxPositionPct: pluginConfig.maxPositionPct,
    maxDrawdownPct: pluginConfig.maxDrawdownPct,
    defaultUniverse: pluginConfig.defaultUniverse,
  }
}
