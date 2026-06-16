---
name: quant-backtest-engine-v2
description: >
  Trade log structure, rolling win rate tracker, factor attribution analysis,
  walk-forward testing protocol, and false positive post-mortem system for Sentinel Global Fund.
  Use when tracking live trade outcomes, calculating verified win rates, attributing performance
  to scoring phases, optimizing parameters, and enforcing WR disclosure rules.
  Owned by Priya Nair, Quantitative Strategist.
---

# QUANT BACKTEST ENGINE v2.0
**Owner: Priya Nair — Quantitative Strategist**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
WR claim ทุกตัวต้องมาจาก live trades จริง ไม่ใช่ component estimate
ระบบนี้ track trades จริงของกอง คำนวณ WR จริง และหา factor
ที่ contribute มากที่สุด เพื่อ optimize Phase 3 rule อย่างต่อเนื่อง
Target: ยืนยัน WR 76-82% ด้วย ≥ 100 trades จริง

---

## MODULE 1 — TRADE LOG STRUCTURE

### 1A. Required Fields ทุก trade
```
TRADE LOG ENTRY
───────────────────────────────────────────────
Ticker:           [SYMBOL]
Entry Date:       [YYYY-MM-DD] ✅ required
Entry Price:      [X.XX] [V/E/U]
Exit Date:        [YYYY-MM-DD] ✅ required
Exit Price:       [X.XX] [V/E/U]
Hold Days:        [คำนวณ auto]
Return %:         [(exit-entry)/entry × 100]
Win/Loss:         [W/L]

Phase Scores at Entry:
  3A Momentum:    [X]/35
  3B Volume:      [X]/25
  3C Structure:   [X]/15
  3D Beta:        [X]/10
  3E Maturity:    [X]/8
  3F Volatility:  [X]/7
  TOTAL:          [X]/100
  SIGNAL:         [STRONG BUY/BUY/SOFT-BLOCK/WATCH]

Catalyst:         [ระบุ catalyst หลัก]
Exit Reason:      [Stop Hit / Target / Signal Reversal / Manual]
Hard Blocks:      [จำนวน blocks ณ entry]
Regime:           [Risk-On/Neutral/Risk-Off] + VIX [X]
ATR Stop:         [entry − 2×ATR] ✅ required (Governance Rule #4)
```

### 1B. Data Quality Flag
```
[V] = verified มีแหล่ง+วันที่
[E] = estimate ต้องระบุ
[U] = unavailable = 0 pts เสมอ
```

---

## MODULE 2 — WIN RATE TRACKER

### 2A. Rolling WR Dashboard
```
METRIC                    CURRENT    TARGET    STATUS
──────────────────────────────────────────────────────
Overall WR                [X]%       76-82%    [✅/⚠️/❌]
WR (Signal=STRONG BUY)    [X]%       ≥ 80%     [✅/⚠️/❌]
WR (Signal=BUY)           [X]%       ≥ 72%     [✅/⚠️/❌]
WR (Signal=SOFT-BLOCK)    [X]%       ≥ 65%     [✅/⚠️/❌]
WR (Score ≥ 90)           [X]%       ≥ 85%     [✅/⚠️/❌]
WR (Score 75-89)          [X]%       ≥ 78%     [✅/⚠️/❌]
WR (Score 58-74)          [X]%       ≥ 68%     [✅/⚠️/❌]
Avg Win %                 [X]%       ≥ 8%      [✅/⚠️/❌]
Avg Loss %                [X]%       ≤ -5%     [✅/⚠️/❌]
Avg R:R                   [X]        ≥ 1.5     [✅/⚠️/❌]
Avg Hold Days             [X]        7-15      [✅/⚠️/❌]
Total Trades              [X]        ≥ 100     [✅/⚠️/❌]
```
*WR = "Component Estimate" จนกว่า Total Trades ≥ 100 (Governance Rule #6)*

### 2B. WR by Regime
```
Regime        Trades    WR%    Avg Return%
──────────────────────────────────────────
Risk-On       [X]       [X]%   [X]%
Neutral       [X]       [X]%   [X]%
Risk-Off      [X]       [X]%   [X]%
```

### 2C. WR by Sector
```
Sector        Trades    WR%    Note
──────────────────────────────────
Defense       [X]       [X]%
Technology    [X]       [X]%
Healthcare    [X]       [X]%
Industrials   [X]       [X]%
Energy        [X]       [X]%
```

---

## MODULE 3 — FACTOR ATTRIBUTION

### 3A. Phase Contribution Analysis
*สำหรับทุก trade: แต่ละ phase score correlate กับ outcome ไหม?*

```
PHASE ATTRIBUTION (อัปเดตทุก 10 trades)
──────────────────────────────────────────────────────────
Phase     Avg Score (W)   Avg Score (L)   Difference   Useful?
3A        [X]/35          [X]/35          [+/-X]        [✅/❌]
3B        [X]/25          [X]/25          [+/-X]        [✅/❌]
3C        [X]/15          [X]/15          [+/-X]        [✅/❌]
3D        [X]/10          [X]/10          [+/-X]        [✅/❌]
3E        [X]/8           [X]/8           [+/-X]        [✅/❌]
3F        [X]/7           [X]/7           [+/-X]        [✅/❌]
──────────────────────────────────────────────────────────
Phase ที่มี difference ≥ 3 pts = มี predictive power
Phase ที่มี difference < 1 pt  = พิจารณาปรับ weight
```

### 3B. Sub-indicator Attribution
```
INDICATOR     WR (high)   WR (low)   Threshold    Action
──────────────────────────────────────────────────────────
RSI Tier1     [X]%        [X]%       55-78        [keep/adjust]
RSI Tier2     [X]%        [X]%       45-54        [keep/adjust]
ADX>25        [X]%        [X]%       25           [keep/adjust]
ADX 20-25     [X]%        [X]%       20-25        [keep/adjust]
OBV rising    [X]%        [X]%       trend        [keep/adjust]
MFI>50        [X]%        [X]%       50           [keep/adjust]
BB in-range   [X]%        [X]%       mid-upper    [keep/adjust]
Squeeze       [X]%        [X]%       release      [keep/adjust]
```

---

## MODULE 4 — WALK-FORWARD TESTING

### 4A. Protocol
```
STEP 1: เก็บ trade log ≥ 30 trades (training set)
STEP 2: ทดสอบ parameter ใหม่กับ training set
STEP 3: Deploy parameter ใหม่ใน 10 trades (test set)
STEP 4: เทียบ WR training vs test
         ถ้า test WR < training - 10% = overfitting → revert
STEP 5: ทำซ้ำทุก 30 trades
```

### 4B. Parameter Optimization Boundaries
```
PARAMETER         CURRENT   MIN    MAX    Step
──────────────────────────────────────────────
RSI Tier1 Low     55        45     65     5
RSI Tier1 High    78        72     85     2
RSI Tier2 Low     45        35     54     5
ADX Threshold     20        15     30     5
MFI Threshold     50        40     60     5
BB Period         20        14     26     2
ATR Period        14        10     20     2
```

### 4C. Overfitting Guards
```
❌ ห้าม optimize > 2 parameters พร้อมกัน
❌ ห้าม optimize ถ้า trades < 30
❌ ห้าม adjust ถ้า test set < 10 trades
❌ Parameter ที่ optimize แล้ว ต้อง improve WR ≥ 3%
   ถึงจะ adopt — ต่ำกว่านี้ถือว่า noise
```

---

## MODULE 5 — FALSE POSITIVE LOG

### 5A. Trade Post-Mortem (ทุก loss trade)
```
FALSE POSITIVE ANALYSIS — [TICKER] | [วันที่]
───────────────────────────────────────────────
Score at entry:     [X]/100
Signal:             [STRONG BUY/BUY/etc]
Actual outcome:     [LOSS X%]
Hold days:          [X]

ROOT CAUSE (เลือก 1 หลัก):
□ Macro shock (เหตุการณ์ภายนอก ไม่ใช่ signal ผิด)
□ Score ผ่านแต่ Data quality ต่ำ ([U] มากเกิน)
□ Phase [X] ให้คะแนนสูงเกินจริง
□ Catalyst invalidated หลัง entry
□ Execution ผิด (เข้าราคาสูงกว่า plan)
□ Stop ไม่ได้ตั้ง หรือตั้งผิด

LESSON:
[ระบุ 1-2 ประโยค]

RULE CHANGE NEEDED? [ใช่/ไม่ใช่]
[ถ้าใช่ ระบุ rule ที่ควรปรับ]
```

---

## MODULE 6 — REPORTING

### 6A. Weekly Quant Report (ส่ง James + Lena ทุกศุกร์)
```
SENTINEL QUANT WEEKLY — [วันที่]

TRADE STATS (YTD):
  Total trades:  [X] | WR: [X]% [target 76-82%*]
  Avg return:    [X]% | Avg R:R: [X]
  Best trade:    [TICKER] +[X]%
  Worst trade:   [TICKER] -[X]%

FACTOR ATTRIBUTION:
  Strongest predictor this month: Phase [X]
  Weakest predictor: Phase [X] → พิจารณาปรับ

PARAMETER STATUS: [stable/optimization needed]
DATA QUALITY: [V%] / [E%] / [U%]

*WR = Component Estimate จนกว่า ≥ 100 trades
```

---

## HARD RULES
```
❌ ห้าม claim WR จริงจนกว่าจะมี ≥ 100 live trades
❌ ทุก optimization ต้องมี training + test set
❌ Loss trade ทุกตัวต้องมี post-mortem ภายใน 48 ชม.
❌ DATA [U] = 0 คะแนนเสมอ ห้ามประมาณเพื่อหลีกเลี่ยง
❌ ห้าม optimize > 2 parameters พร้อมกัน (overfitting)
```
