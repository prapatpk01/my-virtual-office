---
name: investment-system
description: >
  Sentinel Global Fund master investment system skill covering full team structure,
  roles, skills, governance rules, portfolio balance framework, momentum scoring,
  dual objectives (1.3x SPY + 5% yield), macro regime framework, pre-trade checklist,
  risk management, and operating procedures. Use when managing the fund, reviewing
  portfolio, making buy/sell decisions, deploying capital, running research scans,
  or assessing governance compliance. Holdings uploaded by fund manager each session.
  CIO: James Hartwell. Last updated: 16 June 2026.
---

# SENTINEL GLOBAL FUND — Investment System
**CIO**: James Hartwell | **CRO**: Miriam Osei | **Currency**: USD
**Benchmark**: SPY Total Return × 1.3 per year + Dividend Yield ≥ 5%
**Strategy**: Barbell Portfolio — Growth/Momentum + Income/Dividend + Cash/Defensive
**Last Updated**: 16 มิ.ย. 2026

> Holdings และ positions อัปโหลดโดยผู้ใช้ต่อ session
> ไฟล์นี้เก็บ **ระบบ กฎ ทีม และกระบวนการ** เท่านั้น

---

## SECTION 1 — TEAM STRUCTURE (14 คน)

### 1A. บริหาร

| ชื่อ | ตำแหน่ง | หน้าที่หลัก | Skill |
|------|---------|------------|-------|
| **James Hartwell** | CIO | ตัดสินใจขั้นสุดท้าย, อนุมัติ deploy ทุกรายการ, ประธานประชุม | investment-system |
| **Miriam Osei** | CRO | Gate ทุก trade ด้วย 4 ด่าน, enforce data integrity, bias detection, audit trail, WR disclosure | cro-intelligence-v2 |
| **Nina Okonkwo** | Data & Source Engineer | Log ทุก data point (source/date/URL), feed quality A/B/C, lineage tracker, conflict resolution | data-engineering-v2 |
| **Leo Tanaka** | Real-time Data Analyst | Parse screenshot จากผู้ใช้, timestamp standardization ICT→ET, pre-meeting data pack | live-feed-intelligence-v2 |

### 1B. Research

| ชื่อ | ตำแหน่ง | หน้าที่หลัก | Skill |
|------|---------|------------|-------|
| **Sofia Reyes** | Sr. Fundamental Analyst | Step 1-3: moat scoring, earnings quality, sector KPI templates, thesis construction | equity-research-pro-v2 + initiating-coverage + competitive-analysis + thesis-tracker |
| **Marcus Webb** | Sr. Financial Analyst | Step 2: 8Q earnings tracker, revision momentum, whisper vs consensus, PEAD linkage | financial-model-pro-v2 + earnings-analysis + earnings-preview |
| **Aisha Fontaine** | Momentum & Catalyst Analyst | Catalyst scoring matrix (strength×horizon×uniqueness), PEAD calculator, sector heatmap, event calendar, flow flags | catalyst-intelligence-v2 + idea-generation + catalyst-calendar |
| **Maya Chen** | Momentum & Catalyst Analyst | Institutional momentum scanner 7-phase, high-beta expansion filter, swing setup 7-15 วัน | institutional-high-beta-momentum-swing-scanner + idea-generation |

> Aisha และ Maya ดำรงตำแหน่งเดียวกัน ทำงานคู่ขนาน ใช้เทคนิคต่างกัน เสนอผลในที่ประชุมร่วมกัน

### 1C. Quant & Valuation

| ชื่อ | ตำแหน่ง | หน้าที่หลัก | Skill |
|------|---------|------------|-------|
| **Priya Nair** | Quantitative Strategist | Live trade log, rolling WR tracker, factor attribution, walk-forward testing, false positive log | quant-backtest-engine-v2 |
| **Thomas Eriksson** | Head of Valuation | DCF 3-scenario, reverse DCF, comps table, margin of safety, sector multiples | valuation-suite-v2 + dcf-model |

### 1D. Macro & Risk

| ชื่อ | ตำแหน่ง | หน้าที่หลัก | Skill |
|------|---------|------------|-------|
| **Daniel Cho** | Head of Macro Strategy | Regime score 0-100, leading indicators, Fed tracker, sector rotation, 30-day macro calendar, 1-2 quarter forward vision | macro-intelligence-v2 + sector-overview |
| **Kai Tanaka** | Portfolio Risk Analyst | Kelly sizing, ATR stop calculator, correlation matrix, sleeve drift monitor, drawdown projection | risk-engine-v2 |

### 1E. Portfolio & Execution

| ชื่อ | ตำแหน่ง | หน้าที่หลัก | Skill |
|------|---------|------------|-------|
| **Lena Müller** | Portfolio Manager | Dual-objective scorecard, yield tracker, sleeve drift dashboard, rebalance optimizer, NAV attribution | portfolio-management-pro-v2 + portfolio-rebalance + portfolio-monitoring |
| **Ryan Blackwood** | Execution Trader | 9-gate pre-trade checklist, slippage tracker, market impact estimator, order timing guide | execution-intelligence-v2 |

---

## SECTION 2 — WORKFLOW

```
Leo/Nina → feed สด → Daniel (regime) → Sofia/Marcus (fundamental)
                                       → Aisha/Maya (catalyst/momentum)
                                       → Thomas (valuation)
                                       ↓
                              Priya (scoring/backtest)
                                       ↓
                              Kai (risk/sizing/stop)
                                       ↓
                              Lena (portfolio fit)
                                       ↓
                              Miriam (4-gate compliance)
                                       ↓
                              James (CIO sign-off)
                                       ↓
                              Ryan (execution)
```

---

## SECTION 3 — DUAL OBJECTIVE

```
OBJECTIVE 1: Total Return ≥ 1.3× SPY per year
  SPY YTD ref:     +8.48% (ETF.com, 12 มิ.ย. 2026) [V]
  Target:          +11.02%
  Portfolio YTD:   +11.83% [E user-provided]
  Status:          ✅ Beat

OBJECTIVE 2: Dividend Yield ≥ 5%
  Blended yield:   5.08% [V calculated 16 มิ.ย. 2026]
  Status:          ✅ Pass

SINCE INCEPTION:
  Cost basis:      $10,056.64 [E user-provided]
  NAV (16 มิ.ย.):  $12,240.35 [V]
  Gain:            +$2,183.71 (+21.7%)
  2025 return:     +8.79% [E]
  2026 YTD:        +11.83% [E]
  Dividends rx:    $300.02 [E]

Note: [E] = user-provided unaudited จนกว่าจะ audit statement จริง
```

---

## SECTION 4 — PORTFOLIO STRUCTURE

### Sleeve Targets
```
Growth/Momentum:   ~55%  (GPIQ, SPMO, VOO)
Income/Dividend:   ~30%  (BALI, SCHD, O, HSBC)
Cash/Defensive:    ~13%  (SGOV, JAAA)
```

### Holdings (16 มิ.ย. 2026 00:05 [V])
```
GPIQ  $2,970.50  24.27%  $59.41  +2.34%
SPMO  $2,254.19  18.42%  $157.47 +3.27%
VOO   $1,717.56  14.03%  $693.88 +1.75%
BALI  $1,702.80  13.91%  $34.06  +1.30%
SCHD  $1,243.79  10.16%  $32.82  -0.02%
SGOV  $1,214.67   9.92%  $100.53 +0.02%
O       $438.06   3.57%  $62.58  -0.22%
JAAA    $374.44   3.05%  $50.62  +0.04%
HSBC    $324.34   2.65%  $93.02  +0.38%
─────────────────────────────────────────
NAV  $12,240.35  100%
```

### Blended Yield
```
GPIQ  24.27% × 10.81% = 2.62%
SPMO  18.42% ×  0.70% = 0.13%
VOO   14.03% ×  1.04% = 0.15%
BALI  13.91% ×  7.83% = 1.09%
SCHD  10.16% ×  3.22% = 0.33%
SGOV   9.92% ×  3.54% = 0.35%
O      3.57% ×  5.24% = 0.19%
JAAA   3.05% ×  5.50% = 0.17%
HSBC   2.65% ×  5.06% = 0.13%
─────────────────────────────────
Blended: 5.16% ✅
```

---

## SECTION 5 — GOVERNANCE RULES (v2.0 อนุมัติ 13 มิ.ย. 2026)

### Rule #1 — Soft-Block System *(Maya Chen)*
```
ถ้าติด Hard Block เพียง 1 ข้อ AND Score > 80/100
→ Signal = WATCH (ไม่ใช่ REJECT)
ติด Hard Block ≥ 2 ข้อ = REJECT เสมอ
```

### Rule #2 — Staggered Deploy *(Aisha Fontaine)*
```
ก่อน Tier-1 event (FOMC/CPI/NFP):
→ เข้าได้สูงสุด 1/3 ของแผน
→ เก็บ 2/3 ไว้หลัง event ผ่าน
```

### Rule #3 v2 — Position Balance Framework *(Kai Tanaka)*
```
ZONE          SIZE       ACTION
──────────────────────────────────────────────────
✅ BASE       ≤ 20%      Optimal
⚠️ WATCH      20–22%     ประชุม: trim หรือ watch
               (พิจารณาภาพรวม + macro)
🔴 TRIM       23–25%     Trim บังคับ → target 18-19%
               Research หาตัวทดแทนก่อน trim
🚨 EMERGENCY  > 25%      Trim ทันที

TRIM PROCEDURE:
1. Kai คำนวณ trim → target 18-19%
2. Research หาตัวทดแทน:
   - Income sleeve → yield ≥ ตัวที่ trim
   - Growth sleeve → return/momentum คล้ายกัน
   - ถ้าไม่มี → พัก SGOV/JAAA รอ
3. Lena approve → Miriam check → James sign-off
```

### Rule #4 — ATR-Based Stop *(Kai Tanaka)*
```
Default stop = entry − 2 × ATR(14)
ระบุ stop ก่อน execute ทุกครั้ง
Trailing stop: ปรับขึ้นได้ ห้ามปรับลง
```

### Rule #5 — Data Integrity *(Miriam Osei)*
```
DATA UNAVAILABLE = 0 คะแนน ห้ามเดา
ทุก data point ต้องมี [V/E/U] flag:
  [V] Verified:     แหล่งชัด + วันที่ ≤ 24 ชม.
  [E] Estimate:     ประมาณ ต้องระบุ basis
  [U] Unavailable:  = 0 pts เสมอ
```

### Rule #6 — WR Disclosure *(Priya Nair)*
```
ต้องระบุ "Component Estimate (ไม่ใช่ backtest จริง)"
ทุกครั้งที่อ้าง WR จนกว่าจะมี ≥ 100 live trades
```

### Rule #7 — Rebalance Alert *(Lena Müller)*
```
Sleeve drift > 5% จาก target = auto-alert ทันที
Lena ตรวจ NAV ทุกสัปดาห์
```

---

## SECTION 6 — MOMENTUM SCORING SYSTEM v3.0

### Phase 3A — Momentum (35 pts)
```
RSI Tier 1 (55-78):               +12 pts
RSI Tier 2 (45-54 + MA confirm):   +8 pts
RSI < 45 หรือ > 80:                 0 pts

MACD + Histogram:
  บวก + expanding:                 +13 pts
  บวก + ไม่ขยาย:                    +8 pts
  negative แต่ reversing:           +5 pts
  flat/hugging/divergence:           0 pts

ADX:
  > 25:                             +5 pts
  20-25:                            +3 pts
  < 20:                              0 pts [HARD BLOCK]

RS vs SPY (20D):                   +5 pts

Flow (OBV+MFI):
  OBV rising + MFI > 50:          +13-15 pts
  Flat:                            +6 pts
  Distribution:                     0 pts [HARD BLOCK]
```

### Phase 3B — Volume & Flow (25 pts)
```
OBV+MFI rising:         +15 pts
OBV declining ขณะราคาขึ้น: [HARD BLOCK]
Volume > 1.5× avg:      +10 pts
Volume 1-1.5×:           +6 pts
Volume < avg:             0 pts
```

### Phase 3C — Structure (15 pts)
```
MA bullish stack (ราคา > 10EMA > 20EMA > 50SMA): +10 pts
Tier 2 pullback (> 50SMA + > 200SMA):             +6 pts
ราคา > 200SMA เท่านั้น:                           +3 pts
ราคา < 200SMA:                                     0 pts [HARD BLOCK]

Pattern:
  VCP/HTF/Bull Flag:    +5 pts
  Base patterns:        +3 pts
  Unclear:              +1 pt
  Broken/Failed:         0 pts
```

### Phase 3D — High-Beta (10 pts)
```
ATR% > 3%:    +5 pts | 2-3%: +3 pts | < 2%: 0
Beta > 1.3 + DollarVol > $50M: +5 pts
Beta > 1.0 + DollarVol > $10M: +3 pts
DollarVol < $10M: [HARD BLOCK]
```

### Phase 3E — Trend Maturity (8 pts)
```
BB Position (ราคาระหว่าง Mid-Upper BB): +5 pts
ราคาเพิ่งข้าม 20 EMA ≤ 5 bars:          +3 pts
6-15 bars หลังข้าม:                      +2 pts
> 15 bars extended:                       0 pts
```

### Phase 3F — Volatility (7 pts)
```
ATR expanding + > avg:    +4 pts
ATR > avg เท่านั้น:       +2 pts
BB Squeeze release:        +3 pts
BB normal:                 +1 pt
```

### Phase 3E Sector + 3F Catalyst (15 pts รวม)
```
Sector Leadership (3E ext):
  Top 1-3 ใน leading sector:   +5 pts
  ใน leading แต่ไม่ top:       +3 pts
  Sector ไม่ได้นำ:             +1 pt

Catalyst (3F ext — Aisha score):
  Catalyst score ≥ 20/25:     +10 pts
  15-19/25:                    +8 pts
  10-14/25:                    +6 pts
  5-9/25:                      +3 pts
  < 5/25:                       0 pts
  Negative catalyst:           −3 pts
  Earnings ใน 5 วัน:           REJECT
```

### Signal Thresholds
```
≥ 75/100  🟢 STRONG BUY    → เข้า full size
58-74     🟡 BUY           → staggered entry
> 80 + 1 block ⚠️ SOFT-BLOCK → WATCH (Rule #1)
42-57     🟠 WATCH         → รอ confirm
< 42      🔴 REJECT
Hard Block ❌              → REJECT ทันที
```

### Hard Blocks (override ทุก score)
```
❌ ADX < 20
❌ ราคา < 200 SMA
❌ OBV distribution ขณะราคาขึ้น
❌ RSI < 45 ไม่มี MA confirm
❌ DollarVol < $10M
❌ Earnings ใน 5 วัน
```

---

## SECTION 7 — MACRO FRAMEWORK

### Regime Score (Daniel Cho)
```
SCORE     REGIME       CASH MIN    DEPLOY
70-100    Risk-On 🟢   10%         Full
40-69     Neutral 🟡   15%         75%
20-39     Risk-Off 🔴  20-30%      1/3 only
0-19      Crisis ⚫    40%+        Freeze

Components:
  VIX:       0-30 pts
  Yield:     0-25 pts
  Inflation: 0-20 pts
  Breadth:   0-15 pts
  Credit:    0-10 pts
```

### Macro Vision (1-2 Quarters)
```
Daniel ต้องมี forward view 1-2 quarters เสมอ:
- Fed cut timeline
- CPI trend
- Sector rotation prediction
- Q3/Q4 seasonality
ไม่ใช่แค่รายงาน regime ปัจจุบัน
```

### Tier-1 Event Blackout
```
FOMC / CPI / NFP ใน 5 วัน:
→ max 1/3 deploy (Rule #2)
→ review regime ใน 24 ชม. หลัง event
```

---

## SECTION 8 — PRE-TRADE CHECKLIST (9 Gates)

```
GATE 1  Regime timestamp [V] ≤ 24 ชม.
GATE 2  Regime score ≥ 40 (Neutral+)
GATE 3  Momentum score ≥ 58/100
GATE 4  Soft-block check (ถ้า apply)
GATE 5  Position ≤ 20% NAV (Rule #3)
GATE 6  ATR stop ระบุแล้ว (Rule #4)
GATE 7  DQS ≥ 70% + ทุก key data มี [V/E/U]
GATE 8  Stagger rule (ถ้าใกล้ Tier-1 event)
GATE 9  James Hartwell (CIO) sign-off

RESULT: ผ่าน 9/9 → Ryan execute
        ล้มเหลวข้อใดข้อหนึ่ง → HOLD
```

---

## SECTION 9 — RISK MANAGEMENT

### Position Sizing (Kelly Criterion)
```
f* = (WR × Avg_Win − (1−WR) × Avg_Loss) / Avg_Win × 0.25
Ceiling: min(f*, 20%)
Floor:   max(f*, 3%)
```

### Regime Cash Buffer
```
Risk-On:   10%+
Neutral:   15%
Risk-Off:  20-30%
Crisis:    40%+
Before FOMC: 1/3 max deploy
```

### Stop Loss
```
Default:  entry − 2 × ATR(14)
Trailing: ปรับขึ้นทุกสัปดาห์ ห้ามปรับลง
Max risk/trade: 1.5% NAV
Max risk/open:  8% NAV
```

### Correlation
```
Flag ถ้า correlation > 0.7 ระหว่าง 2 positions
Growth sleeve avg correlation > 0.65 = warning
```

---

## SECTION 10 — DEPLOY PLAN (อัปเดต 16 มิ.ย. 2026)

### SGOV $1,214.67 — รอ Execute 22 มิ.ย.
```
Timeline:
  17 มิ.ย.  FOMC → ดู tone dovish/hawkish
  19 มิ.ย.  ตลาดปิด (Juneteenth) + Iran deal sign Geneva
  22 มิ.ย.  ✅ วันแรกที่ execute ได้

SCENARIO A — Fed Hold (base):
  Tranche 1 ($500): AVDV
  Tranche 2 ($400): MAIN
  Tranche 3 ($200): DFIV
  Tranche 4 ($114): O เพิ่ม

SCENARIO B — Fed Hawkish:
  Hold ทั้งหมด → รอ 1 สัปดาห์ → เข้าราคาถูกลง

SCENARIO C — Fed Dovish/Cut:
  Tranche 1: AVDV/DFIV
  Tranche 2: UNH หรือ SPMO เพิ่ม
```

### GPIQ Trim Plan
```
Current: 50 หุ้น × $59.41 = $2,970.50 (24.27%) 🔴 TRIM ZONE
Target:  ~38-39 หุ้น (18-19% NAV)
Trim:    ~11-12 หุ้น (~$650-700)
Proceed: หลัง Research confirm replacement
Replacement: QDVO (yield ~10.5% + upside > GPIQ)
Timeline: หลัง FOMC + deal sign ผ่าน
```

---

## SECTION 11 — WATCHLIST (16 มิ.ย. 2026)

### Active Watchlist
```
SGOV DEPLOY TARGETS:
  AVDV  — International small-cap value, dollar อ่อน
  DFIV  — International large-cap value
  MAIN  — BDC, yield 8.41%, no cut since 2007
  O     — REIT, rate cut tailwind
  QDVO  — GPIQ replacement candidate, yield 10.52%

MOMENTUM RESEARCH:
  RKLB  — Nasdaq-100 inclusion 22 มิ.ย., 3 catalyst layers
  CRWV  — CoreWeave, Nasdaq-100 inclusion 22 มิ.ย.
  UAL   — Airlines, peace dividend
  BKNG  — Travel, Middle East tourism
  CAT   — Reconstruction + AI power
  UNH   — Healthcare re-rate
```

---

## SECTION 12 — QUARTERLY REVIEW

```
ทุกไตรมาส (รอบแรก: ก.ย. 2026):
  □ Performance vs 1.3× SPY
  □ Yield vs 5% target
  □ WR verification (≥ 100 trades?)
  □ Rule effectiveness review
  □ Data quality audit ([V]% ≥ 70%)
  □ Team skill update
  □ Governance rule amendment
```

---

## SECTION 13 — FUND RECORD

```
SENTINEL GLOBAL FUND
════════════════════════════════
Cost basis:       $10,056.64 [E]
NAV (16 มิ.ย.):   $12,240.35 [V]
Total gain:       +$2,183.71
Return:           +21.7% since inception [E]
2025 return:      +8.79% [E]
2026 YTD:         +11.83% [E]
Dividends 2026:   $300.02 [E]
Blended yield:    5.08% [V]
Holdings:         9 positions
FX rate:          1 USD = 32.45 THB (16 มิ.ย.) [V]
════════════════════════════════
[E] = user-provided unaudited
[V] = verified from feed
```

---

## HARD RULES (ห้ามละเมิดเด็ดขาด)

```
❌ ห้าม execute โดยไม่ผ่าน 9 gates ครบ
❌ ห้าม execute โดยไม่มี ATR stop ระบุ
❌ ห้าม position เกิน 20% NAV
❌ ห้าม DATA [U] ได้รับคะแนน
❌ ห้าม WR claim โดยไม่มี "Component Estimate" label
❌ ห้าม chase gap open > 3%
❌ ห้าม average down ในตัวที่ momentum พัง
❌ ห้าม deploy เกิน 1/3 ก่อน Tier-1 event
❌ ห้าม trim ก่อน Research หา replacement
❌ Trailing stop ห้ามปรับลง
```
