---
name: portfolio-management-pro-v2
description: >
  Dual-objective scorecard tracking 1.3x SPY and 5% yield, sleeve drift dashboard, rebalance optimizer, tax-loss harvesting flags, dividend yield tracker, and NAV attribution for Sentinel Global Fund. Implements Governance Rule 7 sleeve drift alert at 5%. Owned by Lena Muller, Portfolio Manager.
---

# PORTFOLIO MANAGEMENT PRO v2.0
**Owner: Lena Müller — Portfolio Manager**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
พอร์ตที่ดีไม่ใช่แค่ผลตอบแทนสูง
แต่คือผลตอบแทนที่ controlled, diversified, และ
สอดคล้องกับ dual objective ทุกไตรมาส
Lena ดูแล big picture — ให้แน่ใจว่าทุก trade ของทีม
บวกรวมกันเป็นพอร์ตที่มั่นคง ไม่ใช่แค่ trade ที่ดีแต่ละตัว

---

## MODULE 1 — DUAL-OBJECTIVE SCORECARD

### 1A. Performance vs Benchmark
```
DUAL OBJECTIVE TRACKER — [วันที่]
────────────────────────────────────────────────────────
OBJECTIVE 1: Total Return ≥ 1.3× SPY
  SPY YTD:          [X]% [V] (source: [X] วันที่ [X])
  Target (1.3×):    [X]%
  Portfolio YTD:    [X]% [V/E] (user-provided / audited)
  Status:           [✅ Beat / ⚠️ On Track / ❌ Behind]
  Gap to target:    [+/-X]%

OBJECTIVE 2: Dividend Yield ≥ 5%
  Blended yield:    [X]% (calculated from holdings)
  Target:           5.0%
  Status:           [✅ / ⚠️ / ❌]
  Gap:              [+/-X]%

SINCE INCEPTION:
  2025 return:  [X]% [E] user-provided
  2026 YTD:     [X]% [E] user-provided
  Note: [E] until audited statement available
```

### 1B. Remaining Return Required
```
เหลือถึงสิ้นปี: [X] เดือน
ทำได้แล้ว YTD: [X]%
ต้องการเพิ่ม:   [(1+target)/(1+ytd) - 1 × 100]%
เช่น: target +15%, ytd +10.94% → ต้องการ +3.66% อีก
```

---

## MODULE 2 — DIVIDEND YIELD TRACKER

### 2A. Per-Holding Yield
```
DIVIDEND YIELD BY HOLDING — [วันที่]
─────────────────────────────────────────────────────────
Ticker  Value($)  Weight  Yield%  Annual($)  Source   Date
GPIQ    2,882     24.1%   [X]%    $[X]       [src]    [d]
SPMO    2,156     18.0%   [X]%    $[X]       [src]    [d]
VOO     1,679     14.0%   [X]%    $[X]       [src]    [d]
BALI    1,673     14.0%   [X]%    $[X]       [src]    [d]
SCHD    1,233     10.3%   [X]%    $[X]       [src]    [d]
SGOV    1,214     10.2%   3.54%   $[X]       iShares  มิ.ย.26
O         433      3.6%   [X]%    $[X]       [src]    [d]
JAAA      374      3.1%   [X]%    $[X]       [src]    [d]
HSBC      316      2.6%   [X]%    $[X]       [src]    [d]
─────────────────────────────────────────────────────────
BLENDED:  [X]%    Annual income: $[X]
TARGET:   5.0%
STATUS:   [✅/⚠️/❌]
```

### 2B. Yield Gap Action
```
Yield gap > 1.5%  → เพิ่ม income sleeve (SCHD/O/BALI)
Yield gap 0.5-1.5% → DCA เข้า income ใน deploy ครั้งหน้า
Yield gap < 0.5%  → on track, ไม่ต้อง action
Yield > 5.5%      → ตรวจ credit quality (yield สูงอาจหมายถึง risk)
```

---

## MODULE 3 — SLEEVE DRIFT MONITOR (Governance Rule #7)

### 3A. Sleeve Dashboard
```
SLEEVE MONITOR — [วันที่]
──────────────────────────────────────────────────────────
Sleeve            Target  Actual  Drift   Alert    Action
Growth/Momentum   55%     [X]%    [X]%    [✅/⚠️]  [none/trim/add]
Income/Dividend   30%     [X]%    [X]%    [✅/⚠️]  [none/trim/add]
Cash/Defensive    13%     [X]%    [X]%    [✅/⚠️]  [none/trim/add]
  (+ 2% buffer for cash minimum regime)

DRIFT THRESHOLDS:
  < 3%   → No action
  3-5%   → Monitor weekly
  > 5%   → AUTO-ALERT → run rebalance plan (Rule #7)
  > 10%  → Urgent — 48hr review
```

### 3B. Holding-Level Drift
```
HOLDING DRIFT — [วันที่]
────────────────────────────────────────────────
Ticker   Target%   Actual%   Drift    Alert
GPIQ     15%       [X]%      [X]%     [✅/⚠️/❌]
  Note: Post-trim target ~15% (was 24% before)
SPMO     15%       [X]%      [X]%     [✅/⚠️]
VOO      14%       [X]%      [X]%     [✅/⚠️]
BALI     12%       [X]%      [X]%     [✅/⚠️]
SCHD     10%       [X]%      [X]%     [✅/⚠️]
SGOV     10%       [X]%      [X]%     [✅/⚠️]
O         4%       [X]%      [X]%     [✅/⚠️]
JAAA      4%       [X]%      [X]%     [✅/⚠️]
HSBC      3%       [X]%      [X]%     [✅/⚠️]
HARD CAP: ห้ามตัวใดตัวหนึ่งเกิน 20% (Rule #3)
```

---

## MODULE 4 — REBALANCE OPTIMIZER

### 4A. Rebalance Trigger Checklist
```
ตรวจก่อน rebalance:
  □ Sleeve drift > 5%? (Rule #7 trigger)
  □ Position > 18%? (Rule #3 warning level)
  □ Dividend yield ต่ำกว่า 4%? (yield emergency)
  □ Regime change? (Risk-On → Risk-Off = เพิ่ม cash)
  □ New capital available? (deploy plan)
```

### 4B. Rebalance Plan Template
```
REBALANCE PLAN — [วันที่] | Lena Müller

TRIGGER: [drift/cap/yield/regime/new capital]

CURRENT vs TARGET:
  [Ticker]: [X]% → [X]% = [TRIM/ADD] $[X]

EXECUTION ORDER (transaction cost minimize):
  1. TRIM [Ticker A] → sell $[X] (highest drift first)
  2. ADD  [Ticker B] → buy $[X] (lowest/most needed)
  3. [...]

ESTIMATED IMPACT:
  Sleeve drift after: [X]% (was [X]%)
  Blended yield after: [X]%
  Transaction cost est: $[X] (assume 0 for ETF)

TIMING:
  Stagger over [X] days
  Avoid [event] on [วันที่]

SEND TO: Kai (risk check) → Miriam (compliance) → James (approval)
```

### 4C. Tax-Loss Harvesting Flags
```
FLAG ถ้า:
  Position unrealized loss > 5%  → flag for TLH consideration
  Position held < 30 days        → wash sale risk (avoid same ticker 30 days)
  Year-end (Nov-Dec)             → scan all positions for TLH

CURRENT FLAGS:
  O: -1.34% → monitor (ยังไม่ถึง threshold)
  [others]: check per holding]
```

---

## MODULE 5 — NAV ATTRIBUTION

### 5A. Performance Attribution
```
NAV ATTRIBUTION — [period] | Lena Müller

TOTAL RETURN: [X]%

BY SLEEVE:
  Growth/Momentum ([X]% NAV): contributed [X]%
    GPIQ +[X]% × [X]% weight = [X]% attribution
    SPMO +[X]% × [X]% weight = [X]% attribution
    VOO  +[X]% × [X]% weight = [X]% attribution

  Income/Dividend ([X]% NAV): contributed [X]%
    BALI +[X]% × [X]% weight = [X]%
    SCHD +[X]% × [X]% weight = [X]%
    O    [X]%  × [X]% weight = [X]%
    HSBC +[X]% × [X]% weight = [X]%

  Cash/Defensive ([X]% NAV): contributed [X]%
    SGOV +[X]% × [X]% weight = [X]%
    JAAA [X]%  × [X]% weight = [X]%

TOP CONTRIBUTOR:  [Ticker] +[X]% attribution
DRAG:             [Ticker] [X]% attribution
```

---

## MODULE 6 — WEEKLY PM REPORT

```
PORTFOLIO MANAGEMENT REPORT — [วันที่] | Lena Müller

NAV: $[X] | Change: [+/-X]% week

DUAL OBJECTIVE:
  Return YTD: [X]% vs target [X]% [✅/⚠️/❌]
  Yield: [X]% vs target 5% [✅/⚠️/❌]

SLEEVE HEALTH:
  Growth:  [X]% (drift [X]%) [✅/⚠️]
  Income:  [X]% (drift [X]%) [✅/⚠️]
  Cash:    [X]% (drift [X]%) [✅/⚠️]

ACTIONS THIS WEEK:
  [รายการ buy/sell ที่เกิดขึ้น]

NEXT WEEK PLAN:
  [รายการที่วางแผน]

ALERTS:
  [drift/cap/yield/regime alerts ถ้ามี]
```

---

## HARD RULES
```
❌ ห้าม deploy ใหม่ถ้า sleeve drift > 10% ยังไม่แก้
❌ ห้ามตัวใดเกิน 20% NAV (Governance Rule #3)
❌ Cash ต้องอยู่เหนือ regime minimum เสมอ
❌ Rebalance plan ต้องผ่าน Kai → Miriam → James ก่อน execute
❌ TLH: ห้ามซื้อ same/similar ticker ใน 30 วัน (wash sale)
❌ Yield tracking ต้องอัปเดตทุกเดือน ห้ามใช้ข้อมูลเก่า > 60 วัน
```
