# Quantum Capital Fund — Investment System Skill
**Fund Manager**: Victoria Chen | **Currency**: USD | **Target Return**: +24% / year
**Strategy**: Barbell Portfolio — Core 80% + Satellite Alpha 20%

---

## 1. ทีมงานและบทบาท

| Agent | ชื่อ | บทบาท | ความรับผิดชอบ |
|-------|------|-------|--------------|
| `hf-manager` | Victoria Chen | Fund Manager | ตัดสินใจซื้อขายขั้นสุดท้าย |
| `hf-research-1` | Dr. Emily Zhao | Research Analyst | Deep research + momentum scan |
| `hf-research-2` | Marcus Webb | Research Analyst | Equity research + financial model |
| `hf-research-3` | Nina Patel | Senior Analyst | Sector + theme analysis |
| `hf-quant-1` | Kenji Tanaka | Quant Analyst | Sentinel Signal v1.0 |
| `hf-quant-2` | Aisha Okonkwo | Quant Analyst | Technical + chart analysis |
| `hf-macro` | Sam Rivera | Macro Analyst | Fed, yields, fund flows |
| `hf-risk` | Chris Morgan | Risk Manager | Stop loss + position sizing |

---

## 2. โครงสร้าง Portfolio — Barbell Strategy

```
CORE (~80%)                    SATELLITE ALPHA (~20%)
─────────────────────          ──────────────────────────
ยึดความมั่นคง                   หา alpha เพิ่ม
ETF + Dividend Income          High-momentum growth stocks
Rebalance ทุก Quarter          Rotation 7-90 วัน
Target: +10%/year              Target: +50%/year (2x/year)
```

### Core Positions (ตัวอย่าง)
| Ticker | ประเภท | บทบาท |
|--------|-------|-------|
| SPMO | Momentum ETF | Large-cap momentum |
| VOO | S&P 500 ETF | Market base |
| GPIQ | Covered Call ETF | Income generation |
| SCHD | Dividend ETF | Dividend income |
| BALI | Buffer ETF | Downside protection (MSCI EAFE) |
| SGOV | T-Bill ETF | Cash buffer / yield |
| O, MAIN | REIT / BDC | Monthly income |

### Satellite Alpha — Phase Labels

| Phase | สัญลักษณ์ | ความหมาย | Action |
|-------|----------|---------|--------|
| IGNITE | 🔥 | กำลัง breakout | ซื้อทันที |
| BUILD | ⚡ | กำลังสะสม | ซื้อได้ |
| PEAK | ⚠️ | ใกล้ top | Trim 50% |
| RANGING | — | ไม่มี trend | ข้าม |

---

## 3. Sentinel Signal v1.0 — Quant Framework

ระบบ voting 3 ปัจจัย (RSI + MACD + Sentiment):

```
RSI-14:
  < 30    → +1 (oversold = buy opportunity)
  30-70   →  0 (neutral)
  > 70    → -1 (overbought = caution)

MACD:
  Line > Signal  → +1 (bullish momentum)
  Line < Signal  → -1 (bearish momentum)

Sentiment Score:
  > +0.2  → +1 (news bullish)
  < -0.2  → -1 (news bearish)
  else    →  0

Score = Sum / N  (-1.0 to +1.0)
  > +0.2  → BUY
  < -0.2  → SELL
  else    → HOLD

Confidence = |Score| × 100%
  > 66%   → STRONG
  33-66%  → MODERATE
  < 33%   → WEAK
```

### RSI Power Zone (Swing Trading)
```
60-75 = POWER ZONE ← Entry sweet spot
  ≥ 76 = Overbought — ไม่ไล่ราคา
  ≤ 59 = ยังสะสมอยู่ รอ momentum build
```

---

## 4. Momentum Scan Framework (Research Team Prompt)

ใช้ทุกครั้งที่หา Satellite candidates ใหม่:

```
Act as an institutional Quantitative Analyst and High-Beta Swing Trader.
Scan 5 high-momentum equity setups for 7-15 day swing timeframe.

SCORING MATRIX (100%):
1. MOMENTUM & RS (40%)     — RSI 60-75 Power Zone, MACD expansion, RS vs SPY
2. VOLUME ACCUMULATION (25%) — 5-day avg > 1.5x 20-day avg, Up/Down ratio > 1.5
3. STRUCTURAL BASE (20%)   — VCP / Flat Base / High-Tight Flag, above 10/20 EMA
4. CATALYST DRIFT (15%)    — PEAD / sector rotation / contract wins

CRITICAL FILTERS:
- Market Regime: VIX > 18 → focus on defensive momentum only
- Entry: within 3% of 10-day EMA or breakout pivot (no chasing)
- Target: 10-25% upside, Fibonacci 1.618 extension
- R:R minimum: 1:3 | Stop: below 20-day EMA

OUTPUT FORMAT per stock:
- Setup Type | Momentum Score | Expected Return | Win Probability
- Entry Range | Target Price | Stop Loss | Risk:Reward
- Momentum & RS analysis
- Volume Accumulation analysis
- Catalyst Drift analysis
- Swing Thesis & Execution plan
```

---

## 5. Rotation Rules — Entry & Exit

### ซื้อ (ENTRY)
```
✅ Sentinel Signal = BUY STRONG (≥ 67%)
✅ Phase = IGNITE 🔥 หรือ BUILD ⚡
✅ RSI < 65 (ยังไม่ overbought)
✅ มี Catalyst ชัดใน 1-3 เดือน
✅ VIX < 18 (ถ้า VIX > 18 = เลือกเฉพาะ extreme RS outlier)
```

### ขาย (EXIT)
```
+25%  → TRIM ครึ่ง (lock profit)
+35%  → ออกทั้งหมด → หา IGNITE ตัวใหม่
Phase = PEAK ⚠️ → ออก 50%
-12%  → STOP LOSS ออกทั้งหมด ← กฎเหล็ก
```

### กฎเหล็ก (ห้ามละเมิด)
```
❌ ห้าม Average Down เด็ดขาด
❌ ห้ามถือเกิน -12% โดยไม่มีเหตุผลพิเศษ
❌ ห้ามซื้อตัวที่ MACD ลงต่ำกว่า Signal line + ราคา < 50-day MA พร้อมกัน
❌ ห้ามซื้อทุกตัวพร้อมกันในวันเดียว (stagger entries)
```

---

## 6. Position Sizing Rules

| ประเภท | ขนาด | เหตุผล |
|--------|------|--------|
| Tier 1 (High Conviction) | ~$400-450 | แข็งแกร่งทุก metric |
| Tier 2 (Medium Conviction) | ~$300-350 | ดีแต่มีความเสี่ยงบางจุด |
| High Beta (β > 3) | **50% ของปกติ** (~$200) | ATR สูง ป้องกันผิดทิศ |
| Cash Buffer | 10-15% ของ pool | รอ FOMC / dip buying |

---

## 7. Macro Framework — ทีม Macro

### สัญญาณหลักที่ต้อง Monitor

| สัญญาณ | ความหมาย | Action |
|--------|---------|--------|
| **2Y Yield > Fed Funds Rate** | ตลาด price in rate hike | ลด satellite, เพิ่ม defensive |
| **VIX > 18** | Regime เปลี่ยน caution | เลือก extreme RS outliers เท่านั้น |
| **VIX > 25** | High Risk | ออก satellite 50% |
| **Core CPI > 3.5% YoY** | Fed hawkish risk | Trim growth stocks |
| **2Y > Fed Funds +50 bps** | Rate hike กำลังจะมา | เตรียม cash ก่อน FOMC |

### FOMC Blackout Rule
```
2 วันก่อน FOMC  → ไม่เพิ่ม position
วัน FOMC       → ถือ cash เท่านั้น
หลัง FOMC Dovish → ซื้อทันที
หลัง FOMC Hawkish → รอ dip อีก 2-3 วัน
```

---

## 8. "Buy Rumor — Sell News" Calendar

กลยุทธ์หลักสำหรับ Satellite rotation:

```
ซื้อตอน RUMOR (2-4 สัปดาห์ก่อน event)
ขายตอน NEWS (วันประกาศ หรือก่อน 1-2 วัน)

ตัวอย่าง Event Calendar ที่ต้อง Track:
  - Earnings (NVDA, PLTR, AVGO ฯลฯ)
  - FOMC meetings (8 ครั้ง/ปี)
  - CPI release (ทุกวันที่ 10-12 ของเดือน)
  - IPO/SpinOff ที่ re-rate sector
  - DoD contract announcements (Defense theme)
  - FDA approvals (Biotech theme)
```

---

## 9. Investment Themes Framework (2026)

### Active Themes
| ธีม | Catalyst | Tickers | Time Horizon |
|-----|---------|---------|-------------|
| **AI Infrastructure** | Hyperscaler capex $500B+ | NVDA, AVGO, AAOI | 2-3 ปี |
| **Space Economy** | SpaceX IPO sector re-rating | RKLB, ASTS | 1-2 ปี |
| **Defense/Drones** | Iran war, $1T DoD budget | KTOS, LMT, RTX | 2-3 ปี |
| **Uranium/Nuclear** | AI power demand + SMR | CCJ, NLR ETF | 3-5 ปี |
| **Biotech M&A** | Patent cliff + $250B deals | XBI ETF | 1-2 ปี |
| **AI Cloud** | GPU rental demand | CRWV (รอสัญญาณ) | 1-2 ปี |
| **AI Power** | Data center energy crisis | VRT, CEG | 2-3 ปี |

### Theme Rotation Rule
```
เข้าธีม: เมื่อ fund flow เริ่มเข้า + RSI อยู่ใน Power Zone
ออกธีม: เมื่อ fund flow ชะลอ + RSI > 75 + Volume ลด
ไม่ Overweight ธีมเดียวเกิน 40% ของ Satellite
```

---

## 10. Research Workflow — ลำดับการวิเคราะห์

```
Step 1: MACRO TEAM
  └─ ดู VIX, 2Y vs Fed Funds, CPI, FOMC calendar
  └─ กำหนด Market Regime (Risk-On / Risk-Off)

Step 2: RESEARCH TEAM
  └─ Run Momentum Scan (5 setups)
  └─ Deep research per candidate (web search + fetch)
  └─ Equity Research Workbook (financials + targets)

Step 3: QUANT TEAM
  └─ Sentinel Signal v1.0 (RSI + MACD + Sentiment)
  └─ Entry range precision (10-day EMA ± 3%)
  └─ R:R calculation (minimum 1:3)

Step 4: RISK TEAM
  └─ Position sizing per Beta
  └─ Portfolio concentration check
  └─ Stop loss placement

Step 5: VICTORIA CHEN (Fund Manager)
  └─ Final approval
  └─ Execution timing (limit orders, phased entry)
  └─ Post-trade monitoring
```

---

## 11. Portfolio Health Metrics

ตรวจสอบทุกสัปดาห์:

| Metric | Target | Alert |
|--------|--------|-------|
| Core / Satellite ratio | 80% / 20% | ถ้าเบี่ยง ±5% → rebalance |
| Max single position | < 15% NAV | ถ้าเกิน → trim |
| Satellite drawdown | < 12% per position | ถ้าถึง → stop loss |
| Portfolio YTD return | +6% per quarter | ถ้าต่ำกว่า → review strategy |
| Cash buffer | 10-15% | ถ้าต่ำกว่า → รอก่อน ไม่ซื้อใหม่ |
| Correlation ใน Satellite | < 0.7 | ถ้าสูง → diversify ธีม |

---

## 12. Execution Rules

```
ประเภท Order:
  ✅ ใช้ LIMIT orders เสมอ — ห้าม market order สำหรับ satellite
  ✅ Stagger entry: ซื้อ 50-70% ก่อน รอ pullback ซื้อที่เหลือ
  ✅ Set limit ที่ 10-day EMA หรือ -1% จากราคาตลาด

Timing:
  ✅ รอ 30-60 นาทีหลัง market open ก่อนซื้อ
  ✅ ไม่ซื้อในวัน FOMC หรือ 1 วันก่อน
  ✅ ไม่ซื้อวันที่ CPI ประกาศ (รอดูผลก่อน)
  ✅ เงินสดจากการขาย → พักใน SGOV ระหว่างรอ
```

---

## 13. Performance Review Template

ทำทุกสิ้นเดือน / สิ้นไตรมาส:

```
Portfolio NAV: $________
Target NAV:    $________ (+6% QoQ / +24% YoY)
Variance:      ________%

Core Performance:    _____%
Satellite Performance: _____%

Best Trade:   ________ (+___%)
Worst Trade:  ________ (-___%)

Lessons:
1. _________________________________
2. _________________________________

Next Quarter Focus:
- Theme: _________________________
- Watchlist: ____________________
- Risk factor: __________________
```

---

## 14. Emergency Protocols

```
IF VIX spikes > 30 suddenly:
  → ออก Satellite ทั้งหมดทันที
  → เข้า SGOV 100% ชั่วคราว
  → รอ VIX ต่ำกว่า 22 ก่อน re-enter

IF Fed surprise hike:
  → ลด Satellite 50%
  → เพิ่ม SCHD + BALI + SGOV
  → Review ทุกตำแหน่งใน 24 ชั่วโมง

IF position ถึง -12%:
  → ขายออกทันที ไม่มีข้อยกเว้น
  → บันทึก lesson learned
  → รอ 48 ชั่วโมงก่อนเปิด position ใหม่ในธีมเดียวกัน
```

---

*Last updated: June 11, 2026 | Victoria Chen Fund | Quantum Capital*
