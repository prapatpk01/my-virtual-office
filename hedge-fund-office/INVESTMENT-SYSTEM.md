---
name: investment-system
description: >
  Investment system skill for Quantum Capital Fund managed by Victoria Chen.
  Covers the full Barbell Portfolio strategy (Core 80% / Satellite Alpha 20%),
  Sentinel Signal v1.0 (RSI + MACD + Sentiment), momentum scan framework,
  rotation rules, macro signals (2Y vs Fed Funds, VIX, CPI), Buy-Rumor-Sell-News
  calendar, position sizing, risk management, and emergency protocols.
  Target return: +24% per year. Use when managing or reviewing the fund portfolio,
  finding new satellite positions, running momentum scans, or making buy/sell decisions.
  Portfolio holdings are uploaded separately by the fund manager each session.
---

# Quantum Capital Fund — Investment System Skill
**Fund Manager**: Victoria Chen | **Currency**: USD | **Target Return**: +24% / year
**Strategy**: Barbell Portfolio — Core 80% + Satellite Alpha 20%

> **หมายเหตุ**: ข้อมูล holdings และ positions จะถูก upload โดย fund manager ในแต่ละ session
> ไฟล์นี้เก็บเฉพาะ **ระบบ กฎ และกระบวนการทำงาน** ของกองทุน

---

## 1. ทีมงานและบทบาท

| Agent | ชื่อ | บทบาท | ความรับผิดชอบหลัก |
|-------|------|-------|-----------------|
| `hf-manager` | **Victoria Chen** | Fund Manager | ตัดสินใจซื้อขายขั้นสุดท้าย, อนุมัติ rebalance |
| `hf-research-1` | Dr. Emily Zhao | Research Analyst | Deep research, momentum scan, web research |
| `hf-research-2` | Marcus Webb | Research Analyst | Equity research workbook, financial model |
| `hf-research-3` | Nina Patel | Senior Analyst | Sector overview, theme analysis, fund flows |
| `hf-quant-1` | Kenji Tanaka | Quant Analyst | Sentinel Signal v1.0, RSI/MACD analysis |
| `hf-quant-2` | Aisha Okonkwo | Quant Analyst | Chart technicals, entry/exit levels |
| `hf-macro` | Sam Rivera | Macro Analyst | Fed policy, yields, CPI, FOMC calendar |
| `hf-risk` | Chris Morgan | Risk Manager | Stop loss, position sizing, drawdown monitoring |

### การเรียกทีมทำงาน
```
ทีม Research  → deep research, momentum scan, equity workbook
ทีม Quant     → Sentinel Signal, technical levels, chart setup
ทีม Macro     → yield curve, Fed, inflation, regime assessment
ทีม Risk      → sizing, stop loss, concentration check
Victoria Chen → final decision, execution approval
```

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

### ประเภทของ Core (หลักการ — ไม่ระบุ ticker)
| ประเภท | บทบาท | สัดส่วนแนะนำ |
|--------|-------|------------|
| Momentum ETF | Market base + growth | 15-20% NAV |
| Broad Market ETF (S&P/QQQ) | Beta exposure | 10-15% NAV |
| Covered Call / Income ETF | Yield generation | 10-15% NAV |
| Dividend ETF | Stable income | 8-12% NAV |
| Buffer ETF (International) | Downside protection | 8-12% NAV |
| T-Bill / Short-term | Cash buffer | 4-8% NAV |
| REIT / BDC | Monthly income | 5-8% NAV |

### ⬆️ Core Holdings ปัจจุบัน
> *[Fund manager จะ upload ข้อมูลจริงในแต่ละ session]*

### Satellite Alpha — Phase Labels

| Phase | สัญลักษณ์ | ความหมาย | Action |
|-------|----------|---------|--------|
| IGNITE | 🔥 | กำลัง breakout | ซื้อทันที |
| BUILD | ⚡ | กำลังสะสม | ซื้อได้ |
| PEAK | ⚠️ | ใกล้ top | Trim 50% |
| RANGING | — | ไม่มี trend | ข้าม |

### ⬆️ Satellite Holdings ปัจจุบัน
> *[Fund manager จะ upload ข้อมูลจริงในแต่ละ session]*

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

### สัญญาณ REJECT อัตโนมัติ
```
❌ MACD ลงต่ำกว่า Signal line AND ราคา < 50-day MA พร้อมกัน
❌ RSI > 75 (overbought) โดยไม่มี catalyst ใหม่
❌ Volume ลดลงขณะราคาขึ้น (divergence)
```

---

## 4. Momentum Scan Framework — Research Team Prompt

ใช้ทุกครั้งที่หา Satellite candidates ใหม่ (รัน deep-research skill):

```
Act as an institutional Quantitative Analyst and High-Beta Swing Trader.
Your objective is to scan, select, and analyze exactly 5 high-momentum equity
setups designed for a 7 to 15-day swing trading timeframe.

Execute the market scan using a highly rigorous Momentum-Centric Alpha Score
(Total 100%):

1. MOMENTUM & RELATIVE STRENGTH (40% — DOMINANT FACTOR)
   Stock must exhibit extreme RS against SPY/QQQ over last 30 days.
   RSI(14) must be in the 60-75 "Power Zone" without bearish divergence.
   MACD must show sustained zero-line separation and expansion.

2. VOLUME ACCUMULATION (25%)
   5-day average volume must be > 1.5x of 20-day average.
   Positive Up/Down Volume Ratio > 1.5 over 2 weeks.

3. STRUCTURAL BASE & TREND (20%)
   Stock emerging from multi-week consolidation (VCP, Flat Base, High-Tight Flag).
   Price firmly trending above 10 EMA and 20 EMA, MAs fanning upward.

4. CATALYST DRIFT (15%)
   PEAD / persistent sector rotation / major contract wins.

CRITICAL FILTERS:
- MARKET REGIME: VIX > 18 → defensive momentum or extreme RS outliers only
- ENTRY RANGE: tight to 10-day EMA or within 3% above breakout pivot. Reject extended.
- SWING TARGETS: 10-25% upside, Fibonacci 1.618 extensions or measured moves
- R:R minimum 1:3 | Stop: below 20-day EMA or base bottom

OUTPUT per stock:
### [TICKER] ([Company Name])
- Setup Type | Momentum Score /100 | Expected Return % | Win Probability %
| Timeframe | Risk:Reward | ENTRY RANGE | TARGET | STOP LOSS |
- MOMENTUM & RS: [RS vs SPY, RSI, MA slope]
- VOLUME ACCUMULATION: [5d vs 20d avg, institutional footprint]
- CATALYST DRIFT: [fundamental driver]
- SWING THESIS & EXECUTION: [entry trigger, edge, trade management]
```

---

## 5. Rotation Rules — Entry & Exit

### ซื้อ (ENTRY CONDITIONS)
```
✅ Sentinel Signal = BUY STRONG (≥ 67% confidence)
✅ Phase = IGNITE 🔥 หรือ BUILD ⚡
✅ RSI < 65 (ยังไม่ overbought)
✅ มี Catalyst ชัดใน 1-3 เดือน (earnings / contract / IPO halo)
✅ VIX < 18 | ถ้า VIX > 18 = เลือกเฉพาะ extreme RS outlier เท่านั้น
✅ ไม่ใช่วัน FOMC หรือ 1 วันก่อน
```

### ขาย (EXIT CONDITIONS)
```
+25%  → TRIM ครึ่ง (lock profit, ถือที่เหลือ)
+35%  → ออกทั้งหมด → หา IGNITE ตัวใหม่ทันที
Phase = PEAK ⚠️ → ออก 50%
-12%  → STOP LOSS ออกทั้งหมด ← กฎเหล็ก ห้ามยืดหยุ่น
```

### กฎเหล็ก (ห้ามละเมิดเด็ดขาด)
```
❌ ห้าม Average Down — ไม่มีข้อยกเว้น
❌ ห้ามถือเกิน -12% — ขายออกทันที
❌ ห้ามซื้อตัวที่ MACD + ราคา < 50MA พร้อมกัน
❌ ห้ามซื้อทุกตัวพร้อมกันในวันเดียว (stagger entries)
❌ ห้าม Market Order สำหรับ satellite — ใช้ Limit เท่านั้น
```

---

## 6. Position Sizing Rules

| ประเภท | ขนาด (% ของ Satellite Pool) | เงื่อนไข |
|--------|---------------------------|---------|
| Tier 1 — High Conviction | 18-20% | Sentinel STRONG + catalyst ชัด |
| Tier 2 — Medium Conviction | 13-15% | Sentinel MODERATE + setup ดี |
| High Beta (β > 3) | **50% ของ Tier ปกติ** | ATR สูง ลด size ป้องกันผิดทิศ |
| Cash Buffer | 10-15% ของ pool | รอ FOMC / dip / new catalyst |

### Stagger Entry Rule
```
Day 1: ซื้อ 60-70% ของ position ที่วางแผน
Day 2-3: ซื้อที่เหลือเมื่อ price confirm (ไม่ใช่ average down)
ห้ามซื้อครั้งเดียว 100% ยกเว้น catalyst เกิดขึ้น overnight
```

---

## 7. Macro Framework — ทีม Macro

### สัญญาณ Yield Curve ที่ต้อง Monitor

| สัญญาณ | ความหมาย | Action |
|--------|---------|--------|
| **2Y Yield > Fed Funds Rate** | ตลาด price in rate hike | ลด satellite exposure |
| **2Y > Fed Funds + 50 bps** | Rate hike ใกล้มา | เตรียม cash ก่อน FOMC |
| **Yield curve steepening** | Growth re-acceleration | เพิ่ม cyclicals |
| **Yield curve inverting** | Recession risk | เพิ่ม defensive core |

### VIX Regime Table

| VIX Level | Regime | Satellite Strategy |
|-----------|--------|--------------------|
| < 15 | Risk-On ✅ | Full deployment |
| 15-18 | Neutral 🟡 | Selective, high RS only |
| 18-25 | Caution ⚠️ | Extreme outliers only, reduce size |
| > 25 | Risk-Off 🔴 | ออก satellite 50%, cash is king |
| > 30 | Emergency 🚨 | ออกทั้งหมด → SGOV |

### FOMC Blackout Rule
```
T-2 วันก่อน FOMC  → ไม่เพิ่ม position ใหม่
T-0 วัน FOMC      → ถือสถานะเดิม ไม่ซื้อขาย
หลัง Dovish       → ซื้อ satellite เพิ่มทันที
หลัง Hawkish      → รอ 2-3 วัน ดู price action ก่อน
```

### Inflation Signals
```
Core CPI > 3.5% YoY  → Trim growth stocks, เพิ่ม dividend/value
Core CPI > 4.0% YoY  → ลด satellite 30%, เข้า short-duration
Energy-driven CPI     → ดู defense + nuclear theme (ไม่ใช่ structural inflation)
```

---

## 8. "Buy Rumor — Sell News" Calendar System

```
ซื้อตอน RUMOR (2-4 สัปดาห์ก่อน event)
ขายตอน NEWS   (วันประกาศ หรือก่อน 1-2 วัน)

Events ที่ต้อง Track ทุกเดือน:
  📊 Earnings Season  — ซื้อ 3-4 สัปดาห์ก่อน, ขายก่อน/หลัง 1 วัน
  🏦 FOMC Meeting     — ดู macro signal, ไม่ซื้อ 2 วันก่อน
  📈 CPI Release      — รอผลก่อนตัดสินใจซื้อ
  🚀 IPO / SpinOff    — ซื้อ sector peers ก่อน IPO (halo effect)
  🛡️ DoD Contract     — ซื้อ defense names ก่อนประกาศ
  💊 FDA Approval     — ซื้อ biotech ก่อน PDUFA date
```

---

## 9. Investment Themes Framework

### หลักการเลือกธีม
```
เข้าธีม: fund flow เริ่มเข้า (สังเกตจาก ETF inflows) + RSI Power Zone
ออกธีม: fund flow ชะลอ + RSI > 75 + Volume ลด + นักวิเคราะห์ consensus สูง
ไม่ Overweight ธีมเดียวเกิน 40% ของ Satellite pool
```

### ธีม Structural (2-5 ปี) — ใช้เป็น Core Satellite
```
AI Infrastructure    — Hyperscaler capex, custom chips, data center
Defense/Drones       — Global rearmament, drone warfare, budget surge
Nuclear/Uranium      — AI power demand, SMR commercialization
Critical Minerals    — US-China decoupling, EV + defense supply chain
```

### ธีม Cyclical (6-18 เดือน) — Rotation plays
```
Space Economy        — Launch cadence, satellite internet, sector IPOs
AI Cloud             — GPU rental, inference scaling
Biotech M&A          — Patent cliff, acquisition premium
AI Power Infra       — Cooling, power distribution, grid
Cybersecurity        — Zero-trust, AI-driven threats
```

### ⬆️ Active Themes ปัจจุบัน + Watchlist
> *[Fund manager จะ upload watchlist และ active positions ในแต่ละ session]*

---

## 10. Research Workflow — ลำดับการทำงาน

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: MACRO TEAM (Sam Rivera)                            │
│    → ดู VIX level, 2Y vs Fed Funds spread, CPI trend       │
│    → กำหนด Market Regime (Risk-On / Caution / Risk-Off)    │
│    → FOMC calendar + blackout dates                         │
├─────────────────────────────────────────────────────────────┤
│  Step 2: RESEARCH TEAM (Emily / Marcus / Nina)              │
│    → Run Momentum Scan prompt (5 setups)                    │
│    → Deep research per candidate (web + fetch)              │
│    → Equity Research Workbook (financials + targets)        │
│    → Fund flow verification per theme                       │
├─────────────────────────────────────────────────────────────┤
│  Step 3: QUANT TEAM (Kenji / Aisha)                        │
│    → Sentinel Signal v1.0 per candidate                     │
│    → Entry range: 10-day EMA ± 3%                          │
│    → R:R calculation (reject if < 1:3)                     │
│    → Fibonacci targets (1.618 extension)                    │
├─────────────────────────────────────────────────────────────┤
│  Step 4: RISK TEAM (Chris Morgan)                           │
│    → Position size per Beta tier                            │
│    → Portfolio concentration check (< 40% per theme)       │
│    → Stop loss placement (below 20-day EMA)                 │
│    → Cash buffer verification (≥ 10%)                       │
├─────────────────────────────────────────────────────────────┤
│  Step 5: VICTORIA CHEN — Final Decision                     │
│    → Review all team inputs                                 │
│    → Approve / reject / modify                              │
│    → Set limit orders + execution timing                    │
│    → Post-trade monitoring schedule                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Portfolio Health Metrics

ตรวจสอบทุกสัปดาห์ — Alert ถ้าเบี่ยงเกิน threshold:

| Metric | Target | Alert Level | Action |
|--------|--------|------------|--------|
| Core / Satellite ratio | 80% / 20% | ±5% drift | Rebalance |
| Max single position | < 15% NAV | > 18% | Trim immediately |
| Satellite stop loss | -12% per position | Hit -10% | Prepare exit |
| Portfolio quarterly return | +6% QoQ | < +2% | Strategy review |
| Cash buffer | 10-15% | < 8% | Pause new buys |
| Satellite theme correlation | < 0.7 | > 0.8 | Add new theme |
| VIX vs portfolio beta | Monitor | VIX > 25 | Reduce satellite |

---

## 12. Execution Rules

### Order Types
```
✅ LIMIT orders เสมอสำหรับ satellite — ห้าม market order
✅ Set limit ที่ 10-day EMA หรือ -1% จากราคาตลาด
✅ Stagger: 60-70% Day 1, ที่เหลือ Day 2-3
✅ เงินสดระหว่างรอ → SGOV (yield ระหว่างรอ)
```

### Timing Rules
```
✅ รอ 30-60 นาทีหลัง market open ก่อนซื้อ
✅ ไม่ซื้อในวัน FOMC หรือ 1 วันก่อน
✅ ไม่ซื้อวันที่ CPI ประกาศ (รอผลก่อน)
✅ ไม่ซื้อ 2 วันก่อน earnings ของ position ที่มีอยู่
✅ ซื้อ Pre-earnings: เริ่ม 3-4 สัปดาห์ก่อน (rumor phase)
```

---

## 13. Performance Review Template

ทำทุกสิ้นเดือน / สิ้นไตรมาส:

```
═══════════════════════════════════════
  QUANTUM CAPITAL — PERFORMANCE REVIEW
  Period: ____________  Date: _________
═══════════════════════════════════════

NAV Summary:
  Opening NAV:   $__________
  Closing NAV:   $__________
  Period Return: _____% (Target: +6% QoQ)
  YTD Return:    _____% (Target: +24%)

Layer Performance:
  Core (80%):        +_____%
  Satellite (20%):   +_____%

Trade Log:
  Best Trade:   ________ Entry:____ Exit:____ Return:+____%
  Worst Trade:  ________ Entry:____ Exit:____ Return: -____%
  Win Rate:     ___/___  (_____%)
  Avg Hold:     _____ days

Themes Active:
  1. _____________________________ P&L: _____
  2. _____________________________ P&L: _____
  3. _____________________________ P&L: _____

Lessons Learned:
  1. _______________________________________________
  2. _______________________________________________

Next Quarter Plan:
  Theme focus: _____________________________
  Watchlist:   _____________________________
  Key risk:    _____________________________
  FOMC dates:  _____________________________
═══════════════════════════════════════
```

---

## 14. Emergency Protocols

```
🚨 VIX spikes > 30 suddenly:
  → ออก Satellite ทั้งหมดทันที (market order ยอมรับได้ในกรณีนี้)
  → เข้า SGOV / T-Bill 100% ชั่วคราว
  → รอ VIX < 22 ก่อน re-enter ทีละตัว

🚨 Fed surprise rate hike:
  → ลด Satellite 50% ภายใน 24 ชั่วโมง
  → เพิ่ม defensive core (dividend + buffer ETF)
  → Review ทุก position ใน 24 ชั่วโมง

🚨 Position ถึง -12%:
  → ขายออกทันที — ไม่มีข้อยกเว้น ไม่มีการ hold ต่อ
  → บันทึก: เหตุผลที่เข้า / สิ่งที่ผิดพลาด / lesson
  → รอ 48 ชั่วโมงก่อนเปิด position ใหม่ในธีมเดียวกัน

🚨 Portfolio drawdown > 8%:
  → หยุดซื้อทุกอย่าง
  → ประชุมทีมทบทวน strategy ก่อน
  → ไม่ re-enter จนกว่า macro regime จะชัดเจน

🚨 Geopolitical shock (war / sanctions):
  → ดู Defense + Energy + Nuclear themes ทันที
  → ลด AI / Growth exposure ชั่วคราว
  → เพิ่ม BALI buffer + SGOV
```

---

## 15. Data Sources & Tools

| Tool | ทีมที่ใช้ | วัตถุประสงค์ |
|------|---------|------------|
| `deep-research` skill | Research | Multi-source web research + momentum scan |
| `equity-research:screen` | Research | Financial screening + target validation |
| `equity-research:thesis` | Research | Individual stock thesis |
| Sentinel Signal v1.0 | Quant | RSI + MACD + Sentiment signals |
| TradingView | Quant | Chart analysis, EMA, MACD visual |
| CME FedWatch | Macro | Fed rate probability |
| FRED (St. Louis Fed) | Macro | 2Y yield, Fed Funds, CPI data |
| Morningstar / iShares | Research | ETF fund flows |

---

---

## 16. Fundamental Analysis Framework — Equity Research

ใช้ทุกครั้งก่อนตัดสินใจเข้า Satellite position (ทีม Research + Marcus Webb)

---

### 16.1 Growth Scorecard (40 pts)

| Metric | คำถาม | เกณฑ์ผ่าน | คะแนน |
|--------|-------|---------|-------|
| Revenue Growth YoY | บริษัทโตเร็วแค่ไหน? | > 20% = 10pts / 10-20% = 5pts | /10 |
| EPS Growth YoY | กำไรต่อหุ้นโต? | > 25% = 10pts / 10-25% = 5pts | /10 |
| Revenue Guidance | management มองปีหน้าอย่างไร? | ขึ้น guidance = 10pts / คงที่ = 5pts | /10 |
| Estimate Revision | นักวิเคราะห์ปรับ estimate ขึ้นไหม? | ปรับขึ้น = 10pts / คงที่ = 0pts | /10 |

### 16.2 Quality Scorecard (30 pts)

| Metric | คำถาม | เกณฑ์ผ่าน | คะแนน |
|--------|-------|---------|-------|
| Gross Margin Trend | margin ขยายตัวหรือหด? | ขยาย = 10pts / คงที่ = 5pts / หด = 0pts | /10 |
| Free Cash Flow | FCF เป็นบวกไหม? | FCF > 0 และ FCF yield > 2% = 10pts | /10 |
| ROIC / ROE | ใช้ทุนได้มีประสิทธิภาพ? | ROIC > 15% = 10pts / 10-15% = 5pts | /10 |

### 16.3 Valuation Scorecard (20 pts)

| Metric | Tech/Growth | Value/Income | คะแนน |
|--------|------------|-------------|-------|
| Forward P/E vs Sector avg | < sector avg = 10pts | < 15x = 10pts | /10 |
| PEG Ratio (P/E ÷ Growth) | < 1.5 = 10pts / 1.5-2.5 = 5pts / > 2.5 = 0pts | — | /10 |

> สำหรับ High-Growth (revenue > 40% YoY) ใช้ EV/Revenue แทน P/E
> เกณฑ์: EV/Revenue < 10x = ถูก / 10-20x = Fair / > 20x = แพง

### 16.4 Catalyst Scorecard (10 pts)

| Catalyst Type | คะแนน |
|--------------|-------|
| Earnings ใน 4-8 สัปดาห์ + consensus beat expected | 10 |
| Contract win / product launch ชัดเจน | 8 |
| Sector re-rating event (IPO halo, M&A) | 7 |
| General sector momentum เท่านั้น | 3 |

---

### 16.5 Total Fundamental Score

```
Growth   (40 pts): ____
Quality  (30 pts): ____
Valuation(20 pts): ____
Catalyst (10 pts): ____
─────────────────────
TOTAL   (100 pts): ____

≥ 75 pts  → GREEN  ✅ เข้าได้ (ร่วมกับ Sentinel Signal)
50-74 pts → YELLOW 🟡 เข้าได้ถ้า technical แข็งมาก
< 50 pts  → RED    ❌ ข้าม — fundamental ไม่รองรับ
```

---

### 16.6 Red Flags — REJECT ทันที

```
🚩 Revenue growth ลดลงติดต่อกัน 2 ไตรมาส (deceleration)
🚩 Gross margin หดลงมากกว่า 300 bps YoY
🚩 Free Cash Flow ติดลบ และ burn rate > 6 เดือน
🚩 Debt/Equity > 3x โดยไม่มี asset backing
🚩 Guidance ถูกปรับลง (guidance cut) ใน 2 ไตรมาสล่าสุด
🚩 Insider selling > 5% ของหุ้นใน 30 วัน (ยกเว้น pre-scheduled)
🚩 Short interest > 20% ของ float
🚩 Customer concentration > 50% จาก 1 ราย
```

---

### 16.7 Key Metrics per Sector

| Sector | Metric หลัก | Metric รอง |
|--------|------------|-----------|
| **AI / Semiconductor** | AI revenue growth %, custom chip backlog | Gross margin, R&D/Revenue |
| **Cloud / SaaS** | ARR growth, NRR (Net Revenue Retention) | Rule of 40, CAC payback |
| **Defense** | Backlog ($), backlog-to-revenue ratio | EBITDA margin, contract type |
| **Energy / Nuclear** | Capacity additions (GW), PPA pricing | FCF, debt maturity |
| **Biotech** | Pipeline stage, FDA timeline | Cash runway, partnership deals |
| **Space / Satellite** | Launch cadence, backlog, gross margin % | R&D burn, government vs commercial mix |

---

### 16.8 Earnings Analysis Framework (Post-Earnings)

```
ขั้นตอนทันทีหลัง earnings ประกาศ:

1. BEAT OR MISS?
   Revenue: Actual vs Consensus est.
   EPS:     Actual vs Consensus est.
   → Beat both + Guidance up   = STRONG BUY signal
   → Beat revenue, miss EPS    = NEUTRAL
   → Miss revenue              = HOLD / EXIT depending on guidance

2. GUIDANCE CHECK
   Q+1 guidance vs current consensus?
   Full-year guidance raised / maintained / lowered?
   → Raised = PEAD setup (Post-Earnings Announcement Drift)
   → Lowered = EXIT หรือ ลด position ทันที

3. MANAGEMENT TONE
   ฟัง Earnings call: bullish / cautious / defensive?
   New products / contracts mentioned?
   Margin trajectory commentary?

4. PRICE ACTION POST-EARNINGS
   Gap up > 5% on volume = momentum entry (buy breakout)
   Gap down > 8% on beat = potential flush-reversal setup
   Flat on beat = แผ่ว — ดู next catalyst ก่อน
```

---

### 16.9 Valuation Quick Reference

```
GROWTH STOCKS (Revenue > 30% YoY):
  Cheap:  Forward P/E < 30x  |  EV/Revenue < 8x  |  PEG < 1.0
  Fair:   Forward P/E 30-50x |  EV/Revenue 8-15x |  PEG 1.0-1.5
  Pricey: Forward P/E > 50x  |  EV/Revenue > 15x |  PEG > 2.0

VALUE / DIVIDEND STOCKS:
  Cheap:  P/E < 12x  |  P/B < 1.5x  |  Yield > 4%
  Fair:   P/E 12-18x |  P/B 1.5-3x  |  Yield 2-4%
  Pricey: P/E > 20x  |  P/B > 4x    |  Yield < 2%

RULE OF 40 (SaaS):
  Score = Revenue Growth % + FCF Margin %
  > 40 = Excellent | 30-40 = Good | < 30 = Weak
```

---

### 16.10 Analyst Consensus Check

```
ดูก่อนเข้า position ทุกครั้ง:

Analyst Ratings:
  % Buy > 70%   = Strong consensus bullish ✅
  % Buy 50-70%  = Moderate consensus
  % Buy < 40%   = Contrarian setup (risky)

Price Target:
  Current price vs median target:
  Upside > 25%  = Significant upside ✅
  Upside 10-25% = Reasonable
  Upside < 10%  = Extended / priced in

Estimate Revisions (most important):
  Upward revisions in last 30 days = BULLISH signal ✅
  Downward revisions              = WARNING ⚠️
  No revisions                    = Neutral
```

---

---

## 17. Contrarian Signal Framework — "กล้าเมื่อคนกลัว กลัวเมื่อคนกล้า"

> *"Be fearful when others are greedy, and greedy when others are fearful."*
> — Warren Buffett

ตลาดมักเกินจริงทั้งสองทิศทาง — ระบบนี้ใช้ **ความสุดขีดของตลาด** เป็นสัญญาณเข้า/ออก
แทนที่จะวิ่งตามฝูงชน ให้มองในทิศตรงข้าม

---

### 17.1 VIX Contrarian Scale — ด้าน FEAR (สัญญาณซื้อ)

```
VIX Level     สภาวะ           Action
──────────────────────────────────────────────────────────
< 13          Extreme complacency  → ⚠️ อันตราย (ดู section 17.2)
13-18         Normal / Risk-On     → ปกติ follow momentum
18-20         Mild caution         → ระวัง เตรียม watchlist
20-25  🟡     Fear spike           → WATCH ดูทิศทาง อย่าเพิ่งซื้อ
              (อาจดีดสั้น หรืออาจเป็นจุดเริ่ม selloff)
25-30  🟠     Fear zone            → เริ่มมองหาจังหวะซื้อ
              Scale in 25-30% ของ position ที่วางแผน
              เน้น High Quality + Strong Balance Sheet ก่อน
> 30   🔴     PANIC zone           → ซื้ออย่างบ้าคลั่ง
              Scale in เต็ม position ทีละ tranche
              นี่คือจังหวะที่ดีที่สุดในรอบหลายเดือน
> 40   🚨     EXTREME PANIC        → ALL IN บน highest conviction
              เกิดไม่บ่อย — generational entry opportunity
              ประวัติศาสตร์: VIX 40+ ตามด้วย rally เสมอ
```

### 17.2 VIX Contrarian Scale — ด้าน GREED (สัญญาณขาย)

```
VIX Level     สภาวะ           Action
──────────────────────────────────────────────────────────
< 13  🚨      Extreme complacency  → DANGER — ลด satellite
              ตลาดลืมว่า risk มีอยู่ = setup สำหรับ correction
< 12  🔴      Euphoria volatility  → TRIM 50% satellite ทันที
              ตั้ง trailing stop บน ที่เหลือ
< 11          All-time-low VIX     → เตรียม cash เพื่อรอซื้อรอบหน้า
```

---

### 17.3 Fear & Greed Index Contrarian Scale — ด้าน FEAR (สัญญาณซื้อ)

```
Score         สภาวะ           Action
──────────────────────────────────────────────────────────
35-50         Neutral          → ปกติ follow momentum
25-35  🟡     Mild Fear        → WATCH — เตรียม watchlist ให้พร้อม
20-25  🟡     Fear             → เริ่มมองหา entry
15-20  🟠     Strong Fear      → เริ่ม scale in 25-30%
              ซื้อ Tier 1 ตัวแรกก่อน
< 15   🔴     Extreme Fear     → ซื้อจริงจัง scale in เต็ม
              นี่คือ majority ของคนกำลังตื่นตระหนก
              ราคาต่ำกว่า intrinsic value มากที่สุด
< 10   🚨     MAXIMUM FEAR     → ALL IN — เกิดน้อยมาก
              ตัวอย่าง: COVID crash, 2022 rate shock
```

### 17.4 Fear & Greed Index Contrarian Scale — ด้าน GREED (สัญญาณขาย)

```
Score         สภาวะ           Action
──────────────────────────────────────────────────────────
50-65         Neutral-Greed    → ปกติ hold momentum
65-75  🟡     Greed            → เริ่ม TRIM position ที่กำไรแล้ว
75-85  🟠     Extreme Greed    → TRIM 30-50% satellite
              lock profit — ตลาดกำลัง overheat
> 85   🔴     EUPHORIA         → DANGER สูง
              TRIM 50-70% ทันที ถือ cash รอ
> 90   🚨     MAXIMUM GREED    → เตรียม cash เต็มที่
              ประวัติศาสตร์: score > 90 ตามด้วย correction ทุกครั้ง
```

---

### 17.5 Combined Signal Matrix

ใช้ VIX + Fear & Greed ร่วมกัน — ต้องสอดคล้องกัน:

```
VIX > 30  AND  F&G < 20   → 🚀 STRONG BUY — สัญญาณซื้อที่แข็งแกร่งที่สุด
VIX > 25  AND  F&G < 25   → ✅ BUY — scale in อย่างมีวินัย
VIX 20-25 AND  F&G 20-35  → 🟡 WATCH — รอยืนยันทิศทาง
VIX 13-20 AND  F&G 40-65  → ➡️ HOLD — normal market, follow momentum
VIX < 15  AND  F&G > 75   → ⚠️ CAUTION — เริ่ม trim
VIX < 13  AND  F&G > 85   → 🔴 DANGER — trim aggressively
```

---

### 17.6 สัญญาณ Market Hype / Top — ขายก่อนคนอื่น

```
🚨 ขายเมื่อเห็นสัญญาณเหล่านี้:

Social / Media:
  → หุ้นตัวนั้นขึ้นข่าวทุกวัน ทุก outlet
  → คนรอบข้างที่ไม่เคยสนใจหุ้นเริ่มถามว่า "ซื้ออะไรดี?"
  → YouTube / TikTok เต็มไปด้วย "ซื้อ [ticker] ก่อนสาย!"
  → Cover story ใน mainstream magazine

Valuation:
  → P/E > 2x sector average โดยไม่มีเหตุผล
  → Analyst target ถูกปรับขึ้นทุกสัปดาห์ไม่หยุด
  → "This time is different" narrative แพร่หลาย

Technical:
  → RSI market index > 78 ต่อเนื่อง 2+ สัปดาห์
  → VIX ต่ำกว่า 12 ต่อเนื่อง
  → Volume ลดลงขณะราคาขึ้น (distribution เงียบ)
  → Breadth แคบ (แค่ 10 หุ้นดัน index ทั้งตลาด)
```

---

### 17.7 Contrarian Scale Overlay บน Portfolio

```
เมื่อ VIX หรือ F&G ส่งสัญญาณ — ปรับ Satellite allocation ทันที:

                    ปกติ    Fear Zone    Panic Zone
                   ────────────────────────────────
Satellite %         20%       25-30%       35-40%
Cash Buffer %      10-15%      5-8%         3-5%
Core %              80%       70-75%       60-65%

หมายเหตุ:
- เพิ่ม satellite ในช่วง panic โดยใช้ cash buffer ที่สำรองไว้
- กลับมา 80/20 เมื่อ VIX กลับต่ำกว่า 20 และ F&G กลับ > 40
- อย่าเพิ่ม satellite เกิน 40% แม้จะ panic มากแค่ไหน
```

---

### 17.8 ประวัติศาสตร์ Contrarian Signals ที่พิสูจน์แล้ว

| เหตุการณ์ | VIX สูงสุด | F&G ต่ำสุด | ผลลัพธ์ S&P 500 (12 เดือนถัดไป) |
|----------|-----------|-----------|-------------------------------|
| COVID Crash (Mar 2020) | 85 | 2 | +74% |
| Rate Shock (Oct 2022) | 34 | 5 | +39% |
| SVB Crisis (Mar 2023) | 30 | 22 | +28% |
| Middle East Shock (Oct 2023) | 23 | 25 | +31% |

> **บทเรียน**: ทุกครั้งที่ VIX > 30 และ F&G < 20 พร้อมกัน — ตลาดให้ผลตอบแทนบวกใน 12 เดือนเสมอ

---

*System Version: 1.1 | Last Updated: June 11, 2026 | Quantum Capital Fund*
*Holdings data: uploaded separately each session by Victoria Chen*
