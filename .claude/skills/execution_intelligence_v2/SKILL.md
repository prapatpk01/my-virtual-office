---
name: execution-intelligence-v2
description: >
  Pre-trade checklist automation for all 9 governance gates, slippage tracker, market impact estimator, optimal execution timing guide, order type selection, and post-trade review template for Sentinel Global Fund. Owned by Ryan Blackwood, Execution Trader.
---

# EXECUTION INTELLIGENCE v2.0
**Owner: Ryan Blackwood — Execution Trader**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
Execution คือขั้นตอนสุดท้ายก่อนเงินออก
ผิดพลาดตรงนี้ไม่มีทางแก้ได้ในภายหลัง
Slippage เล็กน้อยทุกครั้งรวมกันกัด return ทีละน้อย
ระบบนี้ทำให้ทุก execute เป็น deliberate ไม่ใช่ reactive

---

## MODULE 1 — PRE-TRADE CHECKLIST (9 GATES)

### 1A. Mandatory Gates ก่อน Execute ทุกครั้ง
```
PRE-TRADE CHECKLIST — [TICKER] [BUY/SELL] | [วันที่]
══════════════════════════════════════════════════════

GATE 1 — REGIME TIMESTAMP ✅/❌
  □ VIX verified (user feed สด)?
  □ Timestamp ≤ 24 ชม.?
  □ Regime: [Risk-On/Neutral/Risk-Off]

GATE 2 — SENTIMENT SCORE ✅/❌
  □ Regime score ≥ 40? (Neutral or better)
  □ Deploy permission: [Full/75%/1/3/Freeze]

GATE 3 — MOMENTUM SCORE ✅/❌
  □ Sentinel Score ≥ 58/100?
  □ Signal: [STRONG BUY/BUY/SOFT-BLOCK]
  □ Hard blocks: [X]/4

GATE 4 — SOFT-BLOCK CHECK ✅/❌
  □ ถ้า Soft-Block: score > 80 + only 1 block?

GATE 5 — POSITION SIZE ✅/❌
  □ Size ≤ 20% NAV? (Governance Rule #3)
  □ Size after trade: [X]% NAV
  □ Warning at 18%: [triggered/clear]

GATE 6 — ATR STOP ✅/❌
  □ Stop price calculated: $[X]
  □ Formula: $[entry] - 2 × ATR14($[X]) = $[X]
  □ Risk %: [X]% NAV (must be ≤ 1.5%)

GATE 7 — DATA INTEGRITY ✅/❌
  □ DQS ≥ 70%?
  □ All key data has [V/E/U] flag?
  □ No [U] data received any score?

GATE 8 — STAGGER RULE ✅/❌
  □ Before Tier-1 event? If yes: ≤ 1/3 of plan
  □ Next FOMC: [วันที่] — [X] days away
  □ Stagger status: [1/3 / Full / N/A]

GATE 9 — CIO SIGN-OFF ✅/❌
  □ James Hartwell approved?
  □ Approval timestamp: [วันที่+เวลา]

══════════════════════════════════════════════════════
GATES PASSED: [X]/9
STATUS: [CLEAR TO EXECUTE / HOLD — GATE [X] FAILED]
```

---

## MODULE 2 — ORDER TYPE GUIDE

### 2A. Limit vs Market Decision
```
USE LIMIT ORDER when:
  ✅ ATR% > 3% (high volatility — protect from gap)
  ✅ Dollar volume < $100M (less liquid)
  ✅ Pre/post market (wide spread)
  ✅ Size > 0.5% of average daily volume
  ✅ News just released (temporary volatility)
  Limit price: use last bid/ask midpoint − 0.1%

USE MARKET ORDER when:
  ✅ ATR% < 1.5% (low volatility, tight spread)
  ✅ Dollar volume > $500M (very liquid)
  ✅ Signal is time-sensitive (index inclusion)
  ✅ Small size < $500 (slippage minimal)
  ✅ During regular hours 10am-3:30pm ET
```

### 2B. Order Timing
```
OPTIMAL EXECUTION WINDOWS (ET):
  BEST:   10:00am - 11:30am (liquidity high, volatility settling)
  GOOD:   1:30pm - 3:00pm (stable, good spread)
  AVOID:  9:30-10:00am (opening volatility, wide spread)
  AVOID:  3:45-4:00pm (closing volatility, index effects)
  AVOID:  Pre/post market (wide spread, low liquidity)

NEWS/EVENT TIMING:
  Earnings just released: wait 15 min for dust to settle
  FOMC day: wait 30 min after announcement
  CPI release: wait 15 min after print
```

---

## MODULE 3 — MARKET IMPACT ESTIMATOR

### 3A. Impact Formula
```
MARKET IMPACT ESTIMATE
────────────────────────────────────────────────
Order Size ($):         $[X]
Avg Daily Dollar Vol:   $[X] (= avg volume × price)
Participation Rate:     $[X] / $[X] = [X]%

IMPACT ESTIMATE:
  < 1% participation   = negligible (<0.05%)
  1-5% participation   = low (0.05-0.1%)
  5-10% participation  = moderate (0.1-0.25%)
  > 10% participation  = significant (>0.25%)
  → ถ้า > 5%: ต้อง split into 2-3 tranches

SLIPPAGE ALLOWANCE:
  Normal market: budget 0.1-0.2% slippage
  Volatile day (ATR% > 4%): budget 0.3-0.5%
```

### 3B. Position Split Guide
```
ถ้า participation > 5%:
  Tranche 1: 50% of order, market open + 30 min
  Tranche 2: 30% of order, midday (1-2pm ET)
  Tranche 3: 20% of order, following day morning

ถ้า participation > 10%:
  ขยาย execution เป็น 3-5 วัน
  ใช้ TWAP/VWAP approach
```

---

## MODULE 4 — SLIPPAGE TRACKER

### 4A. Per-Trade Log
```
EXECUTION LOG — [TICKER] | [วันที่]
────────────────────────────────────────────────────
Planned entry:    $[X]   (from pre-trade plan)
Actual fill:      $[X]   (from broker confirm)
Slippage:         $[X]   ([X]% - positive=paid more)
Order type:       [LIMIT/MARKET]
Fill time:        [HH:MM ET]
Participation:    [X]% of ADDV
Market condition: [normal/volatile/news]
ATR% at fill:     [X]%
Notes:            [any issues]
```

### 4B. Slippage Dashboard
```
SLIPPAGE TRACKER — YTD [วันที่]
──────────────────────────────────────────────────────
Avg slippage:          [X]%   (target < 0.15%)
Best execution:        [TICKER] [X]%
Worst execution:       [TICKER] [X]%
Market orders:         [X] trades, avg slip [X]%
Limit orders:          [X] trades, avg slip [X]%
Total slippage cost:   $[X]

LESSONS:
  [Insight from worst executions]
```

---

## MODULE 5 — POST-TRADE REVIEW

### 5A. 24-Hour Review Template
```
POST-TRADE REVIEW — [TICKER] | ทำภายใน 24 ชม.
────────────────────────────────────────────────────
Trade:   [BUY/SELL] [X] shares @ $[X]
Date:    [วันที่]
ATR Stop confirmed: $[X] ✅

EXECUTION QUALITY:
  Planned: $[X]   Actual: $[X]   Slip: [X]%
  Timing:  [good/could improve]
  Order type: [appropriate?]

GATE REVIEW:
  All 9 gates passed? [ใช่/ไม่]
  Any issues encountered? [ระบุ]

1-DAY P&L:
  Price now: $[X]   Return: [X]%
  ATR Stop: $[X]   Buffer: [X]%

LESSON:
  [1 sentence improvement for next time]
```

---

## MODULE 6 — EXECUTION REPORT

```
EXECUTION REPORT — [วันที่] | Ryan Blackwood

TRADES THIS WEEK: [X]
  Buys: [X]  |  Sells: [X]

EXECUTION QUALITY:
  Avg slippage: [X]% (target < 0.15%) [✅/⚠️]
  Gates passed rate: [X]/[X] trades [✅/⚠️]
  Issues: [ระบุ]

ATR STOPS SET: [X]/[X] trades [✅/❌]
STAGGER RULE FOLLOWED: [✅/❌]

IMPROVEMENT THIS WEEK:
  [1-2 ประโยค]

SEND TO: Miriam (audit trail) → Priya (trade log)
```

---

## HARD RULES
```
❌ ห้าม execute ก่อนผ่าน 9 gates ครบ
❌ ห้าม execute โดยไม่มี ATR stop ระบุ (Governance Rule #4)
❌ ห้ามเทรดใน 9:30-10:00am ET (opening 30 min)
❌ ห้ามใช้ market order ถ้า ATR% > 3% หรือ participation > 5%
❌ Post-trade review ต้องทำภายใน 24 ชม.
❌ Slippage > 0.5% = escalate ให้ James พร้อม explanation
```
