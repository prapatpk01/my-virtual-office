---
name: catalyst-intelligence-v2
description: >
  Catalyst scoring matrix, PEAD drift calculator, sector rotation heatmap,
  event calendar tracker, and institutional flow flags for Sentinel Global Fund.
  Use when analyzing catalyst strength, horizon, uniqueness scores, sector leadership,
  PEAD windows, event blackout periods, and flow signals for Phase 3F momentum scoring.
  Owned by Aisha Fontaine, Momentum and Catalyst Analyst.
---

# CATALYST INTELLIGENCE v2.0
**Owner: Aisha Fontaine — Momentum & Catalyst Analyst**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
Catalyst คือ trigger ที่ทำให้ราคาเคลื่อนไหวแบบ directional ในเวลาจำกัด
skill นี้วัด catalyst แบบ quantified ไม่ใช่แค่บอกว่า "มี catalyst"
ทุก catalyst ต้องมี: **Strength × Horizon × Uniqueness score**
เชื่อมตรงกับ Phase 3F ของ Sentinel Momentum Score v3.0

---

## MODULE 1 — CATALYST SCORING MATRIX

### 1A. Catalyst Strength (0-10)
```
10  Binary event ที่ตลาดยังไม่ price in (FDA approval, S&P inclusion)
 8  Earnings beat + guidance raise ครั้งแรกใน 2 quarters
 7  Record backlog / contract ใหม่ > 20% revenue
 6  Post-earnings drift (PEAD) ใน 0-20 วัน
 5  Management change (activist / new CEO ที่มีประวัติดี)
 4  Sector rotation tailwind (sector เข้า top 3)
 3  Analyst upgrade + target raise
 2  Insider buying > $1M (reported, มีวันที่)
 1  ข่าว positive ทั่วไป ไม่ใช่ event เฉพาะ
 0  Catalyst ไม่ชัด หรือ verify ไม่ได้
```

### 1B. Catalyst Horizon (0-10)
```
10  เกิดใน 0-5 วันทำการ (imminent)
 8  เกิดใน 6-10 วันทำการ
 6  เกิดใน 11-20 วันทำการ (prime PEAD window)
 4  เกิดใน 21-40 วันทำการ
 2  เกิดใน 41-60 วันทำการ
 0  > 60 วัน หรือไม่มีวันที่ชัดเจน
```

### 1C. Catalyst Uniqueness (0-5)
```
 5  Catalyst เฉพาะบริษัทนี้ (ไม่มีคู่แข่งได้เหมือนกัน)
 3  Catalyst ใน sector แต่บริษัทนี้ได้มากที่สุด
 1  Catalyst กระจาย sector-wide (ทุกตัวได้เหมือนกัน)
 0  Catalyst เป็น macro ล้วน (Fed/CPI/GDP)
```

### 1D. Catalyst Score = Strength + Horizon + Uniqueness (Max 25)
```
≥ 20  = Phase 3F Full (10/10 pts)
15-19 = Phase 3F Strong (8/10 pts)
10-14 = Phase 3F Pass (6/10 pts)
 5-9  = Phase 3F Weak (3/10 pts)
  < 5 = Phase 3F Fail (0/10 pts)
```

---

## MODULE 2 — SECTOR ROTATION HEATMAP

### 2A. Sector Classification (อัปเดตทุกสัปดาห์)
```
STATUS        DEFINITION                    ACTION
─────────────────────────────────────────────────────
LEADING    ── top 3 sector 4 สัปดาห์ติด    ADD weight
EMERGING   ── เพิ่งเข้า top 5             WATCH
NEUTRAL    ── mid-table                   HOLD
LAGGING    ── bottom 3 sector 2 สัปดาห์   REDUCE
AVOID      ── bottom 2 + breakdown        NO ENTRY
```

### 2B. Rotation Signal Matrix
```
Macro Regime    Leading Sector (ประวัติศาสตร์)
────────────────────────────────────────────
Risk-On         Tech, Discretionary, Industrials
Neutral         Healthcare, Financials, Energy
Risk-Off        Utilities, Staples, Gold, Bonds
Rate Fear       Energy, Financials, Value
Rate Relief     Growth, Tech, REITs
```

### 2C. Sector Score → Phase 3E Contribution
```
บริษัทอยู่ใน LEADING sector  = 3E bonus flag ✅
บริษัทอยู่ใน EMERGING sector = 3E neutral
บริษัทอยู่ใน LAGGING/AVOID  = 3E penalty flag ⚠️
```

---

## MODULE 3 — PEAD DRIFT CALCULATOR

### 3A. PEAD Setup Requirements
```
✅ Earnings รายงานแล้ว (มีวันที่)
✅ EPS surprise > 5% (beat consensus)
✅ Revenue surprise > 3%
✅ Guidance maintained or raised
✅ ราคาเคลื่อนไหว < 15% ใน 3 วันแรกหลัง earnings
   (ถ้าวิ่งแรงเกินแสดงว่า market priced in หมดแล้ว)
```

### 3B. PEAD Window Scoring
```
วันที่หลัง earnings    PEAD Score    Phase 3F pts
─────────────────────────────────────────────────
0-5 วัน             10/10          10 pts
6-10 วัน             8/10           8 pts
11-20 วัน (prime)    6/10           6 pts ← sweet spot
21-40 วัน            3/10           3 pts
> 40 วัน             0/10           0 pts
```

### 3C. PEAD Invalidators (ยกเลิก PEAD signal ทันที)
```
❌ CEO/CFO ขาย insider shares > $2M ใน 10 วันหลัง earnings
❌ Revenue miss แม้ EPS beat (EPS beat จาก buyback/cost cut)
❌ Guidance ลด forward quarters
❌ Short interest เพิ่ม > 20% หลัง earnings
```

---

## MODULE 4 — EVENT CALENDAR TRACKER

### 4A. Event Priority Tiers
```
TIER 1 (Market-moving, blackout เข้า 5 วันก่อน):
  - FOMC Meeting (ดอกเบี้ย)
  - CPI/PCE report
  - NFP (Non-Farm Payrolls)
  - Earnings of position held

TIER 2 (Sector-moving, reduce 1/3 ก่อน):
  - Sector earnings (peers ของ position)
  - FDA PDUFA dates (pharma/biotech)
  - Defense contract announcements
  - Index rebalancing dates

TIER 3 (Monitor, ไม่ต้อง action):
  - Fed speaker events
  - ISM/PMI reports
  - Analyst days / investor conferences
```

### 4B. Blackout Protocol
```
TIER 1 event ใน 5 วัน  → ห้าม deploy ใหม่ (ยกเว้น 1/3 staggered)
TIER 2 event ใน 3 วัน  → ลด size 50%
TIER 3 event           → monitor only, ไม่ต้อง action
หลัง TIER 1 ผ่าน       → review position ใน 24 ชม.
```

---

## MODULE 5 — DARK POOL & FLOW FLAGS

### 5A. Institutional Flow Signals (จาก public data)
```
BULLISH flags (เพิ่ม conviction):
  ✅ Options: unusual call buying > 3× avg (open interest spike)
  ✅ Short interest ลด > 10% ใน 2 สัปดาห์
  ✅ 13F filings: institutional นำเข้าใน quarter ล่าสุด
  ✅ Insider buying (reported Form 4, มีวันที่)

BEARISH flags (ลด conviction):
  ⚠️ Put/Call ratio > 2 (unusual put buying)
  ⚠️ Insider selling > $5M ใน 30 วัน
  ⚠️ Short interest เพิ่ม > 15% ใน 2 สัปดาห์
  ⚠️ Block trades ลง (dark pool selling)
```

### 5B. Flow Score → Phase 3B Supplement
```
3+ Bullish flags + 0 Bearish  = OBV bonus confirmation ✅
1-2 Bullish flags              = neutral (ใช้ OBV เดิม)
1+ Bearish flags               = flag ⚠️ ต้องระบุในรายงาน
2+ Bearish flags               = ลด confidence 1 tier
```

---

## OUTPUT FORMAT (ส่งทีมประชุม)

```
CATALYST REPORT — [TICKER] | [วันที่] | Aisha Fontaine

CATALYST SCORE: [X]/25
  Strength:   [X]/10 — [ระบุ catalyst]
  Horizon:    [X]/10 — [ระบุวันที่ event]
  Uniqueness: [X]/5  — [ระบุว่า unique แค่ไหน]
  → Phase 3F Score: [X]/10 pts

SECTOR: [ชื่อ sector] | Status: [LEADING/EMERGING/etc]
  → Sector contribution: [bonus/neutral/penalty]

PEAD: [ใช่/ไม่ใช่]
  Days since earnings: [X] วัน
  EPS surprise: [X]% | Rev surprise: [X]%
  PEAD window: [prime/active/expired]

EVENT CALENDAR (30 วัน):
  [วันที่] — [event] — [Tier 1/2/3]

FLOW FLAGS:
  Bullish: [รายการ]
  Bearish: [รายการ]

DATA STATUS:
  ✅ Verified: [รายการ + แหล่ง + วันที่]
  ⚠️ Estimate: [รายการ]
  ❌ Unavailable: [รายการ]
```

---

## HARD RULES
```
❌ ห้ามให้ Catalyst Score ถ้าไม่มีวันที่ event
❌ ห้าม entry ใน Tier 1 blackout window (ยกเว้น 1/3 staggered)
❌ PEAD ถูก invalidate ถ้ามี invalidator 1 ข้อขึ้นไป
❌ DATA UNAVAILABLE = 0 คะแนน (ตาม Governance Rule #5)
```
