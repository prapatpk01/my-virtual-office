---
name: valuation-suite-v2
description: >
  DCF with 3-scenario analysis bull/base/bear, comparable company multiples, reverse DCF implied growth calculator, margin of safety scoring, and sector-specific valuation multiples for defense, space, industrial, tech, healthcare, and REIT. Use after Sofia completes Step 1-3 to produce Step 5 valuation output. Owned by Thomas Eriksson, Head of Valuation.
---

# VALUATION SUITE v2.0
**Owner: Thomas Eriksson — Head of Valuation**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
Valuation ไม่ใช่การหาราคา "จริง" ของหุ้น
แต่คือการเข้าใจว่าตลาดกำลัง assume อะไร
และ assumption นั้นสมเหตุสมผลไหม
DCF เดียวไม่พอ — ต้องเทียบหลาย methodology
และต้องรู้ว่า ณ ราคาปัจจุบัน ตลาด price in อะไรไปแล้ว

---

## MODULE 1 — DCF 3-SCENARIO MODEL

### 1A. Input Template
```
DCF INPUTS — [TICKER] | [วันที่]
────────────────────────────────────────────────────
BASE FINANCIAL DATA (จาก Sofia / Marcus):
  Current Revenue:      $[X]
  Current EBITDA:       $[X] ([X]% margin)
  Current FCF:          $[X] ([X]% margin)
  Net Debt:             $[X]
  Shares Outstanding:   [X]M
  Current Price:        $[X] [V] [วันที่]

WACC COMPONENTS:
  Risk-free rate:       [X]% (10Y Treasury [V] [วันที่])
  Equity risk premium:  5.5% (Damodaran US market)
  Beta:                 [X] [V/E]
  Cost of debt:         [X]%
  Tax rate:             [X]%
  D/E ratio:            [X]
  WACC:                 [X]% (calculated)
```

### 1B. 3-Scenario Assumptions
```
SCENARIO ASSUMPTIONS
────────────────────────────────────────────────────────────────
                    BEAR CASE    BASE CASE    BULL CASE
Revenue CAGR (5Y)   [X]%         [X]%         [X]%
Terminal Growth     1.5%         2.5%          3.5%
EBITDA Margin (Y5)  [X]%         [X]%          [X]%
FCF Conversion      [X]%         [X]%          [X]%
WACC                +100bps      base          -100bps
────────────────────────────────────────────────────────────────
BEAR requires: [trigger condition — e.g., margin compression]
BULL requires: [trigger condition — e.g., contract ramp-up]
```

### 1C. DCF Output
```
DCF VALUATION OUTPUT
────────────────────────────────────────────────────────
                    BEAR         BASE         BULL
Intrinsic Value     $[X]         $[X]         $[X]
Current Price       $[X]         $[X]         $[X]
Upside/(Downside)   [X]%         [X]%         [X]%
────────────────────────────────────────────────────────
PROBABILITY WEIGHT: Bear [X]% / Base [X]% / Bull [X]%
PROBABILITY-WEIGHTED VALUE: $[X]
MARGIN OF SAFETY (to base): [X]%
```

---

## MODULE 2 — COMPARABLE COMPANY MULTIPLES

### 2A. Comps Table
```
COMPS TABLE — [TICKER] vs Peers | [วันที่]
───────────────────────────────────────────────────────────────
Company     Mkt Cap   EV/Rev  EV/EBITDA  P/E(fwd)  P/FCF   Source
[TARGET]    $[X]B     [X]x    [X]x       [X]x      [X]x    [V][d]
[Peer 1]    $[X]B     [X]x    [X]x       [X]x      [X]x    [V][d]
[Peer 2]    $[X]B     [X]x    [X]x       [X]x      [X]x    [V][d]
[Peer 3]    $[X]B     [X]x    [X]x       [X]x      [X]x    [V][d]
───────────────────────────────────────────────────────────────
Sector Avg  $[X]B     [X]x    [X]x       [X]x      [X]x
PREMIUM/(DISCOUNT) vs avg: [X]%
JUSTIFIED? [ใช่/ไม่ เพราะ moat/growth/quality]
```

### 2B. Sector-Specific Primary Multiple
```
SECTOR             PRIMARY MULTIPLE    SECONDARY
────────────────────────────────────────────────
Defense/Aerospace  EV/Backlog          P/E fwd
Space/Pre-revenue  EV/Revenue          EV/Order
Technology/SaaS    EV/Revenue + ARR    Rule of 40
Industrial         EV/EBITDA           P/FCF
Healthcare/MC      P/E fwd             P/FCF
REIT               P/FFO               EV/EBITDA
Energy             EV/EBITDA           P/FCF yield
```

---

## MODULE 3 — REVERSE DCF

### 3A. Implied Growth Rate Calculator
```
REVERSE DCF — ตลาด price in อะไร?

FORMULA:
  ใช้ current price แทน intrinsic value
  solve for Revenue CAGR ที่ทำให้ DCF = current price

RESULT:
  Current Price:        $[X]
  Implied Revenue CAGR: [X]% ต่อปี (5Y)
  Implied FCF Margin:   [X]% (Y5)

REALITY CHECK:
  Historical CAGR:      [X]%
  Analyst consensus:    [X]%
  Implied vs consensus: [X]% premium
  VERDICT: [Reasonable / Aggressive / Very Aggressive / Cheap]
```

### 3B. Implied Growth Benchmarks
```
VERDICT GUIDE:
  Implied ≤ historical         = Cheap / Reasonable
  Implied 1-1.5× historical    = Fair value
  Implied 1.5-2× historical    = Expensive (needs catalyst)
  Implied > 2× historical      = Very aggressive
  Implied < 0 (negative)       = Distressed
```

---

## MODULE 4 — MARGIN OF SAFETY SCORE

### 4A. MoS Calculation
```
Margin of Safety = (Base DCF - Current Price) / Base DCF × 100

SCORE:
  MoS > 30%    = 5/5 (deeply undervalued)
  MoS 20-30%   = 4/5 (attractive)
  MoS 10-20%   = 3/5 (fair — requires catalyst)
  MoS 0-10%    = 2/5 (fully valued)
  MoS < 0      = 1/5 (premium to base case)
  MoS < -20%   = 0/5 (significantly overvalued)
```

### 4B. Valuation Conviction Score (0-10)
```
Components:
  MoS Score:              [X]/5 (as above)
  Model Agreement:        [X]/3
    (DCF, comps, reverse DCF all agree direction)
  Data Quality:           [X]/2
    ([V] ≥ 70% = 2, [V] 50-69% = 1, < 50% = 0)

Total: [X]/10
≥ 8 = High conviction ✅
5-7 = Moderate ⚠️
< 5 = Low conviction ❌
```

---

## MODULE 5 — ENTRY PRICE RECOMMENDATION

### 5A. Entry Zone Framework
```
ENTRY PRICE RECOMMENDATION — [TICKER]

IDEAL ENTRY ZONE: $[X] - $[X]
  (= Base DCF × 0.70-0.85 → 15-30% MoS)

AGGRESSIVE ENTRY: $[X]
  (= Base DCF × 0.90 → ~10% MoS)

AVOID ABOVE: $[X]
  (= Base DCF × 1.00 → no MoS)

CURRENT PRICE: $[X]
DISTANCE TO IDEAL: [X]% above/below entry zone
ACTION: [BUY NOW / WAIT FOR PULLBACK / AVOID]
```

---

## MODULE 6 — VALUATION OUTPUT FORMAT

```
VALUATION REPORT — [TICKER] | [วันที่] | Thomas Eriksson

DCF SUMMARY:
  Bear:  $[X] ([X]% upside)
  Base:  $[X] ([X]% upside) ← primary
  Bull:  $[X] ([X]% upside)
  Weighted: $[X]
  MoS to base: [X]%

COMPS:
  Target EV/EBITDA: [X]x vs peers avg [X]x ([premium/discount])
  Implied price: $[X]

REVERSE DCF:
  Implied CAGR: [X]% vs consensus [X]%
  Verdict: [Cheap/Fair/Expensive/Very Aggressive]

VALUATION CONVICTION: [X]/10

ENTRY RECOMMENDATION:
  Ideal: $[X]-$[X]
  Current: $[X]
  Action: [BUY NOW / WAIT / AVOID]

SEND TO: Aisha (catalyst check) → Maya (momentum) → James

DATA STATUS:
  [V] Verified: [list + source + date]
  [E] Estimate: [list]
  [U] Unavailable: [list → 0 pts]
```

---

## HARD RULES
```
❌ ห้ามใช้ DCF เดียวโดยไม่มี comps หรือ reverse DCF
❌ ห้าม assign probability weight โดยไม่มี thesis
❌ WACC ต้องอ้างอิง risk-free rate ปัจจุบัน [V] ไม่ใช่ fixed
❌ DATA [U] = 0 pts (Governance Rule #5)
❌ Valuation report ต้องผ่านหลัง Sofia ส่ง fundamental ก่อน
❌ ห้ามแนะนำ BUY ถ้า MoS Score < 2/5
```
