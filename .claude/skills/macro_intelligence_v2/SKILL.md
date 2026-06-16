---
name: macro-intelligence-v2
description: >
  Macro regime scoring 0-100, leading indicator dashboard, Fed policy tracker, sector rotation signals, and 30-day macro risk calendar for Sentinel Global Fund. Use when assessing market regime, reading CPI/FOMC/yield curve, determining risk-on/neutral/risk-off stance, and setting cash buffer requirements. Every data point must be timestamped. Owned by Daniel Cho, Head of Macro Strategy.
---

# MACRO INTELLIGENCE v2.0
**Owner: Daniel Cho — Head of Macro Strategy**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
Regime call ที่ผิดทำให้ทั้งทีมตัดสินใจผิด
ทุก regime assessment ต้องมี timestamp จาก feed สด
ห้าม lag — ถ้าข้อมูลเกิน 24 ชม. ต้องระบุ [STALE]
ข้อมูลจากผู้ใช้ (screenshot/feed สด) มีความสำคัญ
สูงสุด เหนือ search engine snippet เสมอ

---

## MODULE 1 — REGIME SCORING MATRIX (0-100)

### 1A. VIX Component (0-30 pts)
```
VIX < 15       = 30 pts  (extreme calm)
VIX 15-17      = 25 pts  (calm)
VIX 17-20      = 18 pts  (neutral)
VIX 20-25      = 10 pts  (elevated)
VIX 25-30      = 5 pts   (fear)
VIX > 30       = 0 pts   (panic)

SOURCE PRIORITY:
  1st: User TradingView screenshot [V]
  2nd: CBOE official close [V] with date
  3rd: Search result — must state lag [E/STALE]
```

### 1B. Yield/Rate Component (0-25 pts)
```
10Y Yield Direction (0-10):
  Falling + < 4.0%  = 10 pts
  Stable 3.5-4.5%   = 7 pts
  Rising + < 4.5%   = 5 pts
  Rising + > 4.5%   = 2 pts
  > 5.0% (crisis)   = 0 pts

Yield Curve Shape (0-10):
  Normal (10Y > 2Y + 50bps)  = 10 pts
  Flat (within 50bps)         = 6 pts
  Inverted (2Y > 10Y)        = 2 pts
  Deep invert (> 100bps)     = 0 pts

Fed Stance (0-5):
  Cutting / dovish pivot     = 5 pts
  On hold + dovish bias      = 4 pts
  On hold + neutral          = 2 pts
  On hold + hawkish bias     = 1 pt
  Hiking                     = 0 pts
```

### 1C. Inflation Component (0-20 pts)
```
CPI YoY (0-12):
  < 2.5%    = 12 pts
  2.5-3.0%  = 9 pts
  3.0-3.5%  = 6 pts
  3.5-4.0%  = 3 pts
  > 4.0%    = 0 pts  ← current: CPI 4.2% = 0 pts

CPI Trend (0-8):
  Declining 3 months consecutive   = 8 pts
  Declining 1-2 months             = 5 pts
  Stable (< 0.2% move)             = 3 pts
  Rising                           = 0 pts
```

### 1D. Market Breadth (0-15 pts)
```
% Stocks above 200 SMA (0-10):
  > 70%    = 10 pts
  60-70%   = 8 pts
  50-60%   = 5 pts
  40-50%   = 2 pts
  < 40%    = 0 pts

Advance/Decline Line (0-5):
  Rising (making new highs)  = 5 pts
  Flat                       = 3 pts
  Declining                  = 0 pts
```

### 1E. Credit Spread (0-10 pts)
```
HY Spread (0-10):
  < 300 bps   = 10 pts
  300-400 bps = 7 pts
  400-500 bps = 4 pts
  > 500 bps   = 0 pts
```

### 1F. Regime Classification
```
SCORE     REGIME        CASH MIN    DEPLOY RULE
────────────────────────────────────────────────
70-100    Risk-On  🟢   10%         Full position ok
40-69     Neutral  🟡   15%         75% of plan
20-39     Risk-Off 🔴   20-30%      1/3 staggered only
0-19      Crisis   ⚫   40%+        Freeze — no deploy

CURRENT TEMPLATE:
  VIX component:     [X]/30  (VIX=[X] [V/E] [วันที่])
  Yield component:   [X]/25  (10Y=[X]% [V/E] [วันที่])
  Inflation:         [X]/20  (CPI=[X]% [V/E] [วันที่])
  Breadth:           [X]/15  ([V/E] [วันที่])
  Credit:            [X]/10  ([V/E] [วันที่])
  ──────────────────────────
  TOTAL:             [X]/100
  REGIME:            [Risk-On/Neutral/Risk-Off/Crisis]
```

---

## MODULE 2 — LEADING INDICATOR DASHBOARD

### 2A. Economic Leading Indicators
```
INDICATOR         CURRENT    PREV    TREND    SIGNAL
──────────────────────────────────────────────────────
ISM Manufacturing [X]        [X]     [↑/↓/→]  [✅/⚠️/❌]
  (>50=expansion)
ISM Services      [X]        [X]     [↑/↓/→]  [✅/⚠️/❌]
Jobless Claims    [X]K       [X]K    [↑/↓/→]  [✅/⚠️/❌]
  (<250K=healthy)
PMI Composite     [X]        [X]     [↑/↓/→]  [✅/⚠️/❌]
Consumer Conf.    [X]        [X]     [↑/↓/→]  [✅/⚠️/❌]
──────────────────────────────────────────────────────
Leading score: [X]/5 green = economy [strong/ok/weak]
```

### 2B. Market Leading Indicators
```
INDICATOR            CURRENT    SIGNAL
────────────────────────────────────────────
Put/Call Ratio       [X]        >1.2=fear / <0.8=greed
Fear & Greed Index   [X]/100    <25=fear / >75=greed
High Yield Spread    [X]bps     rising=risk-off
AAII Bull-Bear       [X]%       contrarian signal
SPY above 200SMA     [✅/❌]     trend intact?
```

---

## MODULE 3 — FED POLICY TRACKER

### 3A. FOMC Calendar
```
NEXT FOMC: [วันที่]
  Current Rate: [X]%
  Market-implied next move: [cut/hold/hike] [X]% probability
  Dot plot median (last): [X]% end-[year]

BLACKOUT RULE: ห้าม deploy > 1/3 ใน 5 วันก่อน FOMC
REVIEW RULE:   ทบทวน regime ใน 24 ชม. หลัง FOMC
```

### 3B. Fed Stance History (6 meetings)
```
DATE        DECISION    RATE    SIGNAL
──────────────────────────────────────
[วันที่]    [hold/cut]  [X]%    [dovish/neutral/hawkish]
...
TREND: [easing / on-hold / tightening]
```

---

## MODULE 4 — SECTOR ROTATION SIGNAL

### 4A. Current Rotation Map
```
LEADING (top 3 — overweight):
  1. [Sector] — [reason] — [YTD return]
  2. [Sector] — [reason] — [YTD return]
  3. [Sector] — [reason] — [YTD return]

NEUTRAL (hold):
  [Sector list]

LAGGING (underweight/avoid):
  [Sector list]

SOURCE: [ระบุ + วันที่] [V/E/U]
```

### 4B. Rotation-Regime Matrix
```
REGIME          FAVOR                   AVOID
──────────────────────────────────────────────
Risk-On         Tech, Discretionary,    Utilities, Staples
                Industrials
Neutral         Healthcare, Energy,     High-beta growth
                Financials
Risk-Off        Staples, Utilities,     Tech, Discretionary
                Gold, Short-duration
Rate Fear       Energy, Financials,     REITs, Growth
                Value
Rate Relief     Growth, Tech, REITs     Value, Energy
```

---

## MODULE 5 — 30-DAY MACRO RISK CALENDAR

```
MACRO CALENDAR — อัปเดตทุกสัปดาห์
────────────────────────────────────────────────────
DATE        EVENT                   TIER    IMPACT
[วันที่]    FOMC Meeting            1       Rate decision
[วันที่]    CPI Release             1       Inflation read
[วันที่]    NFP                     1       Labor market
[วันที่]    ISM Manufacturing       2       PMI read
[วันที่]    GDP Revision            2       Growth read
[วันที่]    Earnings Season Start   2       Corporate health
────────────────────────────────────────────────────
TIER 1 = blackout 5 วันก่อน (max 1/3 staggered)
TIER 2 = monitor only
```

---

## OUTPUT FORMAT
```
MACRO REPORT — [วันที่+เวลา] | Daniel Cho

REGIME SCORE: [X]/100 → [Risk-On/Neutral/Risk-Off]
  VIX [X] [V/timestamp]: [X]/30
  10Y [X]% [V/timestamp]: [X]/25
  CPI [X]% [V/timestamp]: [X]/20
  Breadth [V/timestamp]: [X]/15
  Credit [V/timestamp]: [X]/10

CASH MINIMUM: [X]%
DEPLOY PERMISSION: [Full/75%/1/3 only/Freeze]

SECTOR LEADERSHIP: [1. X  2. Y  3. Z]
NEXT TIER-1 EVENT: [วันที่] — [event]
BLACKOUT STATUS: [ใช่/ไม่]

DATA QUALITY:
  [V] Verified: [list + source + timestamp]
  [E] Estimate: [list]
  [U] Unavailable: [list]
```

---

## HARD RULES
```
❌ ห้ามให้ regime score โดยไม่มี timestamp VIX + yield
❌ Search snippet > 24 ชม. ต้องระบุ [STALE]
❌ User feed สด override search result เสมอ
❌ ห้ามเปลี่ยน regime call โดยไม่มี data ใหม่
❌ FOMC blackout = max 1/3 deploy (Governance Rule #2)
```
