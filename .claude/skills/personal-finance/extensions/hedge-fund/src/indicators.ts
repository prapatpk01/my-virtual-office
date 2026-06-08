// Pure TypeScript technical indicators — deterministic, no floating-point LLM arithmetic

export interface MACDResult {
  readonly macdLine: number
  readonly signalLine: number
  readonly histogram: number
}

// Wilder's smoothed RSI
export function calcRSI(closes: number[], period = 14): number | null {
  if (closes.length < period + 1) return null

  let gains = 0
  let losses = 0
  for (let i = 1; i <= period; i++) {
    const diff = closes[i]! - closes[i - 1]!
    if (diff > 0) gains += diff
    else losses -= diff
  }

  let avgGain = gains / period
  let avgLoss = losses / period

  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i]! - closes[i - 1]!
    avgGain = (avgGain * (period - 1) + (diff > 0 ? diff : 0)) / period
    avgLoss = (avgLoss * (period - 1) + (diff < 0 ? -diff : 0)) / period
  }

  if (avgLoss === 0) return 100
  return 100 - 100 / (1 + avgGain / avgLoss)
}

function calcEMA(values: number[], period: number): number[] {
  if (values.length === 0) return []
  const k = 2 / (period + 1)
  const emas: number[] = [values[0]!]
  for (let i = 1; i < values.length; i++) {
    emas.push(values[i]! * k + emas[i - 1]! * (1 - k))
  }
  return emas
}

export function calcMACD(closes: number[]): MACDResult | null {
  if (closes.length < 35) return null // need 26 + 9 to stabilise both EMAs

  const ema12 = calcEMA(closes, 12)
  const ema26 = calcEMA(closes, 26)

  // Slice from index 25 so EMA-26 has had enough bars to stabilise
  const macdSeries = closes.map((_, i) => ema12[i]! - ema26[i]!).slice(25)
  const signalSeries = calcEMA(macdSeries, 9)

  const lastMACD = macdSeries[macdSeries.length - 1]!
  const lastSignal = signalSeries[signalSeries.length - 1]!

  return {
    macdLine: lastMACD,
    signalLine: lastSignal,
    histogram: lastMACD - lastSignal,
  }
}

// ATR — used for volatility-scaled position sizing
export function calcATR(
  highs: number[],
  lows: number[],
  closes: number[],
  period = 14
): number | null {
  if (closes.length < period + 1) return null
  const trs: number[] = []
  for (let i = 1; i < closes.length; i++) {
    trs.push(
      Math.max(
        highs[i]! - lows[i]!,
        Math.abs(highs[i]! - closes[i - 1]!),
        Math.abs(lows[i]! - closes[i - 1]!)
      )
    )
  }
  if (trs.length < period) return null
  return trs.slice(0, period).reduce((a, b) => a + b, 0) / period
}
