import { describe, it, expect } from "vitest"
import { calcRSI, calcMACD, calcATR } from "../src/indicators.js"
import { assetClass, CRYPTO_SYMBOLS } from "../src/client.js"

// ── Indicator Tests ──

describe("calcRSI", () => {
  it("returns null when data is insufficient", () => {
    expect(calcRSI([100, 101, 102], 14)).toBeNull()
  })

  it("returns 100 when all periods are gains", () => {
    const closes = Array.from({ length: 20 }, (_, i) => 100 + i)
    expect(calcRSI(closes)).toBe(100)
  })

  it("returns 0 when all periods are losses", () => {
    const closes = Array.from({ length: 20 }, (_, i) => 100 - i)
    expect(calcRSI(closes)).toBe(0)
  })

  it("returns a value in [0, 100] for mixed data", () => {
    const closes = [100, 102, 101, 105, 103, 107, 104, 108, 105, 109, 106, 107, 105, 108, 110, 109]
    const rsi = calcRSI(closes)
    expect(rsi).not.toBeNull()
    expect(rsi!).toBeGreaterThanOrEqual(0)
    expect(rsi!).toBeLessThanOrEqual(100)
  })

  it("gives higher RSI for consistently rising prices", () => {
    const rising = Array.from({ length: 20 }, (_, i) => 100 + i * 2)
    const falling = Array.from({ length: 20 }, (_, i) => 200 - i * 2)
    const rsiRising = calcRSI(rising)!
    const rsiFalling = calcRSI(falling)!
    expect(rsiRising).toBeGreaterThan(rsiFalling)
  })

  it("oversold threshold: RSI < 30 for sharp downtrend", () => {
    const closes = [200, 180, 160, 145, 135, 128, 122, 118, 115, 113, 111, 109, 108, 107, 106]
    const rsi = calcRSI(closes)
    expect(rsi).not.toBeNull()
    expect(rsi!).toBeLessThan(30)
  })
})

describe("calcMACD", () => {
  it("returns null with fewer than 35 data points", () => {
    expect(calcMACD(Array.from({ length: 30 }, (_, i) => 100 + i))).toBeNull()
  })

  it("returns correct shape for sufficient data", () => {
    const closes = Array.from({ length: 60 }, (_, i) => 100 + Math.sin(i / 5) * 10)
    const result = calcMACD(closes)
    expect(result).not.toBeNull()
    expect(typeof result!.macdLine).toBe("number")
    expect(typeof result!.signalLine).toBe("number")
    expect(result!.histogram).toBeCloseTo(result!.macdLine - result!.signalLine, 8)
  })

  it("histogram = macdLine - signalLine", () => {
    const closes = Array.from({ length: 60 }, (_, i) => 100 + i * 0.5 + Math.random() * 2)
    const result = calcMACD(closes)
    expect(result).not.toBeNull()
    expect(Math.abs(result!.histogram - (result!.macdLine - result!.signalLine))).toBeLessThan(1e-9)
  })
})

describe("calcATR", () => {
  it("returns null with insufficient data", () => {
    const h = [105, 106]
    const l = [95, 96]
    const c = [100, 101]
    expect(calcATR(h, l, c, 14)).toBeNull()
  })

  it("returns a positive number for valid data", () => {
    const n = 20
    const c = Array.from({ length: n }, (_, i) => 100 + i)
    const h = c.map(v => v + 2)
    const l = c.map(v => v - 2)
    const atr = calcATR(h, l, c)
    expect(atr).not.toBeNull()
    expect(atr!).toBeGreaterThan(0)
  })
})

// ── Asset Classification Tests ──

describe("assetClass", () => {
  it("classifies BTCUSD as crypto", () => {
    expect(assetClass("BTCUSD")).toBe("crypto")
  })

  it("classifies GLD as etf", () => {
    expect(assetClass("GLD")).toBe("etf")
  })

  it("classifies AAPL as equity", () => {
    expect(assetClass("AAPL")).toBe("equity")
  })

  it("classifies all CRYPTO_SYMBOLS as crypto", () => {
    for (const sym of CRYPTO_SYMBOLS) {
      expect(assetClass(sym)).toBe("crypto")
    }
  })
})
