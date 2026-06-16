---
name: risk-engine-v2
description: >
  Position sizing with Kelly Criterion, ATR-based stop loss calculator,
  portfolio correlation matrix, sleeve drift monitor, and drawdown projection for Sentinel Global Fund.
  Implements Governance Rules 3 (position cap 20%), 4 (ATR stop), and 7 (sleeve drift alert 5%).
  Use for all pre-trade risk checks, stop calculation, position sizing, and weekly risk reports.
  Owned by Kai Tanaka, Portfolio Risk Analyst.
---

# RISK ENGINE v2.0
**Owner: Kai Tanaka — Portfolio Risk Analyst**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
Risk management ไม่ใช่การป้องกันการขาดทุน
แต่คือการให้ขาดทุนในขนาดที่ยอมรับได้ และไม่ทำลายพอร์ต
ระบบนี้ implement Governance Rules #3 (Position Cap),
#4 (ATR Stop), #7 (Rebalance Alert) อย่างเป็นระบบ

---

## MODULE 1 — POSITION SIZING (Governance Rule #3)

### 1A. Hard Cap System
```
POSITION LIMITS
────────────────────────────────────────────────
Hard Cap:        20% of NAV / ตัว (REJECT ถ้าเกิน)
Warning Level:   18% of NAV / ตัว (auto-alert)
Soft Target:     10-15% / ตัว (optimal zone)
New Position:    Max 5% NAV ครั้งแรก (scale in)

SLEEVE LIMITS
────────────────────────────────────────────────
Growth/Momentum: สูงสุด 65% NAV
Income/Dividend: สูงสุด 45% NAV
Cash/Defensive:  ขั้นต่ำตาม regime (ดู Module 3)
```

### 1B. Kelly Criterion Position Sizing
```
FORMULA: f* = (WR × Avg_Win - (1-WR) × Avg_Loss) / Avg_Win
              × Kelly_Fraction

INPUT:
  WR           = Win Rate ล่าสุดจาก Priya (Quant v2.0)
  Avg_Win      = Average gain % ของ winning trades
  Avg_Loss     = Average loss % ของ losing trades
  Kelly_Fraction = 0.25 (quarter-Kelly สำหรับลด volatility)

EXAMPLE (WR=73%, Avg_Win=8%, Avg_Loss=5%):
  f* = (0.73×8 - 0.27×5) / 8 × 0.25
     = (5.84 - 1.35) / 8 × 0.25
     = 0.562 × 0.25 = 14.0% of NAV

CEILING: min(f*, 20%) = Position size
FLOOR:   max(f*, 3%)  = ขั้นต่ำที่มีความหมาย

หมายเหตุ: ถ้า WR < 60% → ลด Kelly_Fraction เป็น 0.1
```

### 1C. Signal-Based Size Adjustment
```
Signal          Kelly Output    Size Cap
──────────────────────────────────────────
STRONG BUY      100% Kelly      ≤ 20%
BUY             75% Kelly       ≤ 15%
SOFT-BLOCK      50% Kelly       ≤ 10%
WATCH           ห้าม deploy     0%
```

---

## MODULE 2 — ATR-BASED STOP LOSS (Governance Rule #4)

### 2A. Default Stop Calculation
```
FORMULA: Stop = Entry_Price − (ATR_Multiplier × ATR14)

DEFAULT: ATR_Multiplier = 2.0 (Governance v1.0 default)
AGGRESSIVE: 1.5× (trend ชัด, ต้องการ tight stop)
DEFENSIVE: 2.5× (volatile stock, ต้องการ room)

EXAMPLE (RKLB entry $110, ATR14=$8.5, mult=2.0):
  Stop = $110 - (2.0 × $8.5) = $93.00
  Risk per share = $17.00 (15.5%)
```

### 2B. Stop Management Rules
```
INITIAL STOP:   ระบุใน pre-trade record ก่อน execute เสมอ
TRAILING STOP:  ปรับขึ้นทุกสัปดาห์ ห้ามปรับลง
STOP TRIGGER:   ราคาปิดใต้ stop 2 วันติด = exit (ไม่ใช่ intraday)
EXCEPTION:      ถ้าเป็น news-driven spike = hold 1 วัน แล้วประเมินใหม่
```

### 2C. Risk Per Trade Limits
```
Max Risk / Trade    = 1.5% of NAV
Max Risk / Open     = 8% of NAV (ทุก open position รวม)
Max Daily Loss      = 3% NAV → pause new entries 24 ชม.
Max Weekly Loss     = 6% NAV → emergency review

CALCULATION:
  Shares × (Entry - Stop) = Dollar Risk
  Dollar Risk / NAV × 100 = Risk%
  ถ้า Risk% > 1.5% → ลด shares จนได้ ≤ 1.5%
```

---

## MODULE 3 — REGIME-BASED CASH BUFFER

### 3A. Cash Requirement by Regime
```
REGIME (Daniel Cho assessment)    VIX Range    Cash Min
───────────────────────────────────────────────────────
🟢 Risk-On  (Score 70-100)        < 18         10%
🟡 Neutral  (Score 40-69)         18-22        15%
🔴 Risk-Off (Score 0-39)          > 22         20-30%
⚫ Crisis   (VIX > 35)            > 35         40%+
```

### 3B. Cash Deployment Permission
```
Regime          Max Deploy / Week    Stagger Rule
────────────────────────────────────────────────
Risk-On         Full position        1 tranche ok
Neutral         75% of plan          2 tranches
Risk-Off        1/3 only (Rule #2)   3 tranches
Crisis          0% (freeze)          ไม่ deploy
Before FOMC     1/3 max (Rule #2)    2 tranches
```

---

## MODULE 4 — CORRELATION MATRIX

### 4A. Correlation Check (ทุก position ใหม่)
```
CURRENT PORTFOLIO CORRELATION (อัปเดตทุกสัปดาห์)
────────────────────────────────────────────────────
         GPIQ   SPMO   VOO    BALI   SCHD   O
GPIQ     1.00
SPMO     [X]    1.00
VOO      [X]    [X]    1.00
BALI     [X]    [X]    [X]    1.00
SCHD     [X]    [X]    [X]    [X]    1.00
O        [X]    [X]    [X]    [X]    [X]    1.00

FLAG ถ้า correlation > 0.7 (2 ตัวขึ้นไป)
FLAG ถ้า growth sleeve avg correlation > 0.65
```

### 4B. Diversification Score
```
SCORING:
  Avg correlation ≤ 0.40  = Excellent diversification ✅
  Avg correlation 0.41-0.60 = Good ✅
  Avg correlation 0.61-0.70 = Monitor ⚠️
  Avg correlation > 0.70  = Concentration risk ❌

CURRENT: Growth sleeve (GPIQ/SPMO/VOO) correlation
         ต้องเช็คทุกครั้งก่อนเพิ่ม growth position ใหม่
```

---

## MODULE 5 — SLEEVE DRIFT MONITOR (Governance Rule #7)

### 5A. Target vs Actual
```
SLEEVE MONITOR (อัปเดตทุกสัปดาห์ — Lena + Kai)
────────────────────────────────────────────────────
Sleeve              Target    Actual    Drift    Alert
Growth/Momentum     55%       [X]%      [X]%     [✅/⚠️]
Income/Dividend     30%       [X]%      [X]%     [✅/⚠️]
Cash/Defensive      13%       [X]%      [X]%     [✅/⚠️]
  → Alert threshold: |Drift| > 5% = auto-alert to James+Lena
```

### 5B. Rebalance Trigger Matrix
```
Drift      Action
────────────────────────────────────────────────
< 3%       No action needed
3-5%       Monitor weekly
> 5%       AUTO-ALERT → Lena runs rebalance plan
> 10%      URGENT → review ใน 48 ชม.
> 15%      EMERGENCY → same-day review + action
```

---

## MODULE 6 — DRAWDOWN PROJECTION

### 6A. Max Drawdown Estimate (Light Monte Carlo)
```
INPUT:
  Current NAV:     $[X]
  Avg volatility:  [X]% (30D)
  Worst case:      Avg_Vol × 3 (3-sigma event)

SCENARIOS:
  Normal pullback (-1σ):  NAV × (1 - Vol×1)
  Bear market (-2σ):      NAV × (1 - Vol×2)
  Crisis (-3σ):           NAV × (1 - Vol×3)

EXAMPLE (NAV=$11,960, 30D vol=15%):
  Normal:  $11,960 × 0.85 = $10,166 (-15%)
  Bear:    $11,960 × 0.70 = $8,372  (-30%)
  Crisis:  $11,960 × 0.55 = $6,578  (-45%)
```

### 6B. Recovery Time Estimate
```
ถ้า drawdown = -15%  → ต้องการ +17.6% เพื่อ recover
ถ้า drawdown = -25%  → ต้องการ +33.3%
ถ้า drawdown = -35%  → ต้องการ +53.8%
ถ้า drawdown = -50%  → ต้องการ +100%
```

---

## MODULE 7 — RISK REPORT FORMAT

### 7A. Weekly Risk Summary (ส่ง James ทุกศุกร์)
```
SENTINEL RISK REPORT — [วันที่] | Kai Tanaka

POSITION LIMITS:
  Largest position: [TICKER] [X]% NAV [✅/<18%/⚠️/>18%]
  ตัวที่ใกล้ hard cap: [TICKER] [X]%

ATR STOPS (active positions):
  [TICKER]: Entry $[X] | Stop $[X] | Risk $[X] ([X]% NAV)
  [TICKER]: Entry $[X] | Stop $[X] | Risk $[X] ([X]% NAV)
  Total open risk: [X]% NAV [target < 8%]

SLEEVE DRIFT:
  Growth:  [target 55%] → actual [X]% drift [X]% [✅/⚠️]
  Income:  [target 30%] → actual [X]% drift [X]% [✅/⚠️]
  Cash:    [target 13%] → actual [X]% drift [X]% [✅/⚠️]

CORRELATION:
  High-correlation pair: [TICKER A] / [TICKER B] = [X]
  Action needed: [ใช่/ไม่]

REGIME: [Risk-On/Neutral/Risk-Off]
Cash minimum required: [X]%
Current cash: [X]% [✅/⚠️]
```

---

## HARD RULES
```
❌ ห้าม execute ถ้าไม่มี ATR stop ระบุไว้ก่อน (Rule #4)
❌ ห้าม position เกิน 20% NAV (Rule #3)
❌ ห้าม deploy เกิน 1/3 ใน blackout period (Rule #2)
❌ ห้าม optimize Kelly ถ้า WR มาจาก < 30 trades
❌ Trailing stop ปรับขึ้นเท่านั้น ห้ามปรับลง
❌ Cash ต้องอยู่เหนือ minimum regime level เสมอ
```
